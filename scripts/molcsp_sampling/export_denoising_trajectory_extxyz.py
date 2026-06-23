import gzip
import json
import os
from pathlib import Path

import ase.io
import hydra
import torch
from hydra.utils import instantiate
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.molecule_dataset import (
    _atom_feature_row,
    _bond_feature_row,
    _pbc_bond_distances,
)
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.data_utils import lattice_matrix_to_params_torch
from mattergen.common.utils.eval_utils import load_model_diffusion
from mattergen.common.utils.globals import get_device
from mattergen.generator import structure_from_model_output


ROOT = Path(os.environ.get("OMC25_ROOT", "."))
RUN = Path(os.environ["MOLCSP_RUN"])
OUT = Path(os.environ["MOLCSP_OUT"])
MATERIAL_ID = os.environ["MOLCSP_MATERIAL_ID"]
TARGET_CIF = Path(os.environ["MOLCSP_TARGET_CIF"])
LOAD_EPOCH = int(os.environ.get("MOLCSP_LOAD_EPOCH", "79"))
N_STEPS = int(os.environ.get("MOLCSP_N", "1000"))
MODES = [
    ("t0p8", 0.8),
    ("t0p6", 0.6),
    ("t0p5", 0.5),
    ("t0p3", 0.3),
    ("t0p1", 0.1),
]

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

MAPPING_JSONL = ROOT / "datasets/molecule_mapping/omc25_le300_val_molmap_hybrid_v3.jsonl.gz"
GRAPH_JSONL = ROOT / "scripts/oe62_hybrid_graphs_all_v3.jsonl.gz"


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def find_mapping(material_id: str) -> dict:
    for record in read_jsonl_gz(MAPPING_JSONL):
        if str(record.get("material_id")) == material_id and record.get("success", False):
            return record
    raise RuntimeError(f"Could not find mapping for {material_id}")


def find_graph(refcode: str) -> dict:
    for record in read_jsonl_gz(GRAPH_JSONL):
        if (
            str(record.get("refcode_csd")) == str(refcode)
            and record.get("transfer_mode") == "rdkit_explicit_h_full_match"
            and record.get("ok", True)
        ):
            return record
    raise RuntimeError(f"Could not find graph for {refcode}")


def graph_from_target_cif() -> tuple[list[int], torch.Tensor, torch.Tensor]:
    structure = Structure.from_file(str(TARGET_CIF))
    atomic_numbers = [int(site.specie.Z) for site in structure.sites]
    frac = torch.tensor(structure.frac_coords, dtype=torch.float32)
    cell = torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0)
    return atomic_numbers, frac, cell


def make_condition(mapping_record: dict, graph_record: dict) -> ChemGraph:
    atomic_numbers, frac, cell = graph_from_target_cif()
    mapping = mapping_record["mapping"]
    mol_atom_id = torch.tensor(mapping["mol_atom_idx"], dtype=torch.long)
    mol_copy_id = torch.tensor(mapping["mol_id"], dtype=torch.long)
    if len(atomic_numbers) != int(mol_atom_id.numel()):
        raise RuntimeError(f"Target CIF atom count mismatch: {len(atomic_numbers)} != {mol_atom_id.numel()}")

    atom_features = graph_record["atom_features"]
    mol_x = torch.tensor(
        [_atom_feature_row(atom_features[int(atom_idx)]) for atom_idx in mol_atom_id],
        dtype=torch.long,
    )

    directed_edges = []
    directed_attr = []
    for bond in mapping_record.get("crystal_bonds", []):
        begin = int(bond["begin"])
        end = int(bond["end"])
        attr = _bond_feature_row(int(bond.get("type", 0)))
        directed_edges.extend([[begin, end], [end, begin]])
        directed_attr.extend([attr, attr])
    if directed_edges:
        mol_bond_edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        mol_bond_attr = torch.tensor(directed_attr, dtype=torch.long)
        mol_bond_d0 = _pbc_bond_distances(frac_pos=frac, cell=cell, edge_index=mol_bond_edge_index)
    else:
        mol_bond_edge_index = torch.empty((2, 0), dtype=torch.long)
        mol_bond_attr = torch.empty((0, 2), dtype=torch.long)
        mol_bond_d0 = torch.empty((0,), dtype=frac.dtype)

    return ChemGraph(
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        pos=frac,
        cell=cell,
        num_atoms=torch.tensor([len(atomic_numbers)], dtype=torch.long),
        mol_x=mol_x,
        mol_bond_edge_index=mol_bond_edge_index,
        mol_bond_attr=mol_bond_attr,
        mol_bond_d0=mol_bond_d0,
        mol_atom_id=mol_atom_id,
        mol_copy_id=mol_copy_id,
        mol_num_molecules=torch.tensor([int(mapping["num_molecules"])], dtype=torch.long),
    )


def structures_from_batch(batch):
    batch = batch.to("cpu")
    lengths, angles = lattice_matrix_to_params_torch(batch.cell)
    return structure_from_model_output(
        batch.pos.reshape(-1, 3),
        batch.atomic_numbers.reshape(-1),
        lengths.reshape(-1, 3),
        angles.reshape(-1, 3),
        batch.num_atoms.reshape(-1),
    )


def one_structure(batch) -> Structure:
    return structures_from_batch(batch)[0]


def atoms_from_batch(batch, step: int, t_value: float, mode: str, frame_kind: str):
    structure = one_structure(batch)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.info["step"] = int(step)
    atoms.info["t"] = float(t_value)
    atoms.info["mode"] = mode
    atoms.info["frame_kind"] = frame_kind
    atoms.info["material_id"] = MATERIAL_ID
    return atoms


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traj_dir = OUT / "trajectories_extxyz"
    final_dir = OUT / "final_cifs"
    traj_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    print("loading mapping", flush=True)
    mapping_record = find_mapping(MATERIAL_ID)
    graph_record = find_graph(mapping_record["csd_refcode"])
    condition = make_condition(mapping_record, graph_record)

    print(f"loading model {RUN} epoch {LOAD_EPOCH}", flush=True)
    ckpt_info = MatterGenCheckpointInfo(model_path=RUN, load_epoch=LOAD_EPOCH, strict_checkpoint_loading=True)
    model = load_model_diffusion(ckpt_info).to(get_device()).eval()
    print(f"loaded {ckpt_info.checkpoint_path}", flush=True)

    with hydra.initialize_config_dir(str(ROOT / "sampling_conf")):
        sampling_cfg = hydra.compose(
            config_name="csp",
            overrides=[f"sampler_partial.N={N_STEPS}", f"sampler_partial.eps_t={1 / N_STEPS}"],
        )
    sampler = instantiate(sampling_cfg.sampler_partial)(pl_module=model)

    cond = collate([condition]).to(get_device())
    summary = {
        "material_id": MATERIAL_ID,
        "target_cif": str(TARGET_CIF),
        "checkpoint": ckpt_info.checkpoint_path,
        "N": N_STEPS,
        "trajectories": {},
    }
    for mode, t_start in MODES:
        print(f"[{mode}] corrupting target at t={t_start}", flush=True)
        sampler._max_t = t_start
        t = torch.full((cond.get_batch_size(),), t_start, device=get_device())
        noisy = model.diffusion_module.corruption.sample_marginal(cond, t)
        print(f"[{mode}] denoise N={N_STEPS} with record=True", flush=True)
        sample_batch, mean_batch, records = sampler._denoise(batch=noisy, mask={}, record=True)
        timesteps = torch.linspace(t_start, 1 / N_STEPS, N_STEPS).cpu().tolist()

        frames = []
        for i, rec_batch in enumerate(records):
            t_value = timesteps[min(i, len(timesteps) - 1)]
            frames.append(atoms_from_batch(rec_batch, i, t_value, mode, "recorded_state"))
        frames.append(atoms_from_batch(mean_batch.to("cpu"), len(frames), 1 / N_STEPS, mode, "final_mean"))

        extxyz_path = traj_dir / f"{mode}_N{N_STEPS}_trajectory.extxyz"
        ase.io.write(extxyz_path, frames, format="extxyz")
        final_cif = final_dir / f"{mode}_N{N_STEPS}_final_mean.cif"
        CifWriter(one_structure(mean_batch.to("cpu"))).write_file(str(final_cif))
        summary["trajectories"][mode] = {
            "t_start": t_start,
            "num_frames": len(frames),
            "extxyz": str(extxyz_path),
            "final_cif": str(final_cif),
        }
        print(f"[{mode}] wrote {extxyz_path} frames={len(frames)}", flush=True)

    with open(OUT / "trajectory_export_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
