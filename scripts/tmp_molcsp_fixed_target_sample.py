import json
import gzip
import math
import os
from pathlib import Path
from statistics import mean, median

import hydra
import torch
from hydra.utils import instantiate
from pymatgen.core import Structure
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
LOAD_EPOCH = int(os.environ.get("MOLCSP_LOAD_EPOCH", "54"))
N_STEPS = int(os.environ.get("MOLCSP_N", "200"))
BATCH_SIZE = int(os.environ.get("MOLCSP_BATCH_SIZE", "8"))
COUNT = int(os.environ["MOLCSP_COUNT"])
START_INDEX = int(os.environ["MOLCSP_START_INDEX"])
SHARD_ID = int(os.environ.get("MOLCSP_SHARD_ID", "0"))
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "2"))
torch.set_num_threads(TORCH_NUM_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

MAPPING_JSONL = ROOT / "datasets/molecule_mapping/omc25_le300_val_molmap_hybrid_v3.jsonl.gz"
GRAPH_JSONL = ROOT / "scripts/oe62_hybrid_graphs_all_v3.jsonl.gz"
DENSITY_RANGE = (0.04, 0.25)
ANGLE_RANGE = (45.0, 135.0)
MIN_DIST_CUTOFF = 0.7


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


def safe_mid(s: str) -> str:
    return str(s).replace("|", "_").replace("/", "_")


def pbc_min_dist(pos: torch.Tensor, cell: torch.Tensor) -> float:
    n = pos.shape[0]
    if n < 2:
        return float("inf")
    d = pos[:, None, :] - pos[None, :, :]
    d = d - torch.round(d)
    cart = d @ cell
    dist = torch.linalg.norm(cart, dim=-1)
    dist += torch.eye(n, dtype=dist.dtype) * 1e9
    return float(dist.min().item())


def cell_angles(cell: torch.Tensor) -> list[float]:
    a, b, c = cell[0], cell[1], cell[2]

    def angle(u: torch.Tensor, v: torch.Tensor) -> float:
        denom = (torch.linalg.norm(u) * torch.linalg.norm(v)).clamp_min(1e-12)
        cos = torch.dot(u, v) / denom
        cos = torch.clamp(cos, -1.0, 1.0)
        return float(torch.rad2deg(torch.acos(cos)).item())

    return [angle(b, c), angle(a, c), angle(a, b)]


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


def metrics_for_data(data, target_cell: torch.Tensor) -> dict:
    cell = data.cell.squeeze(0).detach().cpu().float()
    pos = data.pos.detach().cpu().float()
    n = int(data.atomic_numbers.numel())
    vol = abs(float(torch.det(cell).item()))
    dens = n / vol if vol > 1e-12 else float("inf")
    angles = cell_angles(cell)
    mind = pbc_min_dist(pos, cell) if vol > 1e-12 else 0.0
    target_vol = abs(float(torch.det(target_cell.squeeze(0).detach().cpu().float()).item()))
    target_dens = n / target_vol if target_vol > 1e-12 else float("inf")
    valid_cell = (
        DENSITY_RANGE[0] <= dens <= DENSITY_RANGE[1]
        and all(ANGLE_RANGE[0] <= x <= ANGLE_RANGE[1] for x in angles)
    )
    no_clash = mind >= MIN_DIST_CUTOFF
    return {
        "num_atoms": n,
        "volume": vol,
        "atom_density": dens,
        "min_dist": mind,
        "min_angle": min(angles),
        "max_angle": max(angles),
        "volume_ratio": vol / target_vol if target_vol > 1e-12 else None,
        "density_ratio": dens / target_dens if target_dens > 1e-12 else None,
        "valid_cell": bool(valid_cell),
        "no_clash": bool(no_clash),
        "pass_basic": bool(valid_cell and no_clash),
    }


def summarize(records: list[dict]) -> dict:
    out = {"count": len(records)}
    for key in ["valid_cell", "no_clash", "pass_basic"]:
        out[f"{key}_rate"] = sum(bool(r[key]) for r in records) / len(records)
    for key in ["volume_ratio", "density_ratio", "atom_density", "min_dist", "min_angle", "max_angle"]:
        vals = [float(r[key]) for r in records if r.get(key) is not None and math.isfinite(float(r[key]))]
        out[f"{key}_median"] = median(vals) if vals else None
        out[f"{key}_mean"] = mean(vals) if vals else None
    return out


def main() -> None:
    shard_dir = OUT / "shards" / f"shard_{SHARD_ID:02d}"
    cif_dir = OUT / "cifs" / "full_prior"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cif_dir.mkdir(parents=True, exist_ok=True)

    print(f"[shard {SHARD_ID}] loading mapping for {MATERIAL_ID}", flush=True)
    mapping_record = find_mapping(MATERIAL_ID)
    graph_record = find_graph(mapping_record["csd_refcode"])
    print(f"[shard {SHARD_ID}] building condition", flush=True)
    condition = make_condition(mapping_record, graph_record)
    target_cell = condition.cell.clone()

    print(f"[shard {SHARD_ID}] loading model from {RUN} epoch {LOAD_EPOCH}", flush=True)
    ckpt_info = MatterGenCheckpointInfo(model_path=RUN, load_epoch=LOAD_EPOCH, strict_checkpoint_loading=True)
    model = load_model_diffusion(ckpt_info).to(get_device()).eval()
    print(f"[shard {SHARD_ID}] loaded {ckpt_info.checkpoint_path}", flush=True)
    with hydra.initialize_config_dir(str(ROOT / "sampling_conf")):
        sampling_cfg = hydra.compose(
            config_name="csp",
            overrides=[f"sampler_partial.N={N_STEPS}", f"sampler_partial.eps_t={1 / N_STEPS}"],
        )
    sampler = instantiate(sampling_cfg.sampler_partial)(pl_module=model)
    print(f"[shard {SHARD_ID}] sampler ready N={N_STEPS} batch_size={BATCH_SIZE}", flush=True)

    records = []
    written = 0
    while written < COUNT:
        current = min(BATCH_SIZE, COUNT - written)
        print(f"[shard {SHARD_ID}] sampling {written}:{written + current}", flush=True)
        cond = collate([condition for _ in range(current)]).to(get_device())
        _, mean_batch = sampler.sample(cond)
        out_batch = mean_batch.to("cpu")
        structures = structures_from_batch(out_batch)
        data_list = out_batch.to_data_list()
        for j, (data, structure) in enumerate(zip(data_list, structures)):
            global_idx = START_INDEX + written + j
            metrics = metrics_for_data(data, target_cell)
            metrics.update(
                {
                    "sample_index": global_idx,
                    "local_index": written + j,
                    "shard_id": SHARD_ID,
                    "material_id": MATERIAL_ID,
                    "mode": "full_prior",
                }
            )
            records.append(metrics)
            CifWriter(structure).write_file(
                str(cif_dir / f"sample_{global_idx:04d}_{safe_mid(MATERIAL_ID)}.cif")
            )
        written += current

    with open(shard_dir / "records_full_prior.jsonl", "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    summary = {
        "checkpoint": ckpt_info.checkpoint_path,
        "material_id": MATERIAL_ID,
        "target_cif": str(TARGET_CIF),
        "N": N_STEPS,
        "batch_size": BATCH_SIZE,
        "count": COUNT,
        "start_index": START_INDEX,
        "shard_id": SHARD_ID,
        "summary": summarize(records),
    }
    with open(shard_dir / "summary_full_prior.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
