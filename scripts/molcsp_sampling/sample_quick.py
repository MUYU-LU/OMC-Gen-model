import json
import math
import os
from pathlib import Path
from statistics import mean, median

import hydra
import torch
from hydra.utils import instantiate
from pymatgen.io.cif import CifWriter

from mattergen.common.data.collate import collate
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.data_utils import lattice_matrix_to_params_torch
from mattergen.common.utils.eval_utils import load_model_diffusion
from mattergen.common.utils.globals import get_device
from mattergen.generator import structure_from_model_output


ROOT = Path(os.environ.get("OMC25_ROOT", "."))
RUN = Path(os.environ["MOLCSP_RUN"])
OUT = Path(os.environ["MOLCSP_OUT"])
N = int(os.environ.get("MOLCSP_N", "200"))
BATCH_SIZE = int(os.environ.get("MOLCSP_BATCH_SIZE", "4"))
NUM_VAL = int(os.environ.get("MOLCSP_NUM_VAL", "20"))
LOAD_EPOCH = int(os.environ.get("MOLCSP_LOAD_EPOCH", "54"))
DENSITY_RANGE = (0.04, 0.25)
ANGLE_RANGE = (45.0, 135.0)
MIN_DIST_CUTOFF = 0.7


def safe_mid(s: str) -> str:
    return str(s).replace("|", "_").replace("/", "_")


def batch_structures(batch):
    batch = batch.to("cpu")
    lengths, angles = lattice_matrix_to_params_torch(batch.cell)
    return structure_from_model_output(
        batch.pos.reshape(-1, 3),
        batch.atomic_numbers.reshape(-1),
        lengths.reshape(-1, 3),
        angles.reshape(-1, 3),
        batch.num_atoms.reshape(-1),
    )


def min_dist_pbc(pos: torch.Tensor, cell: torch.Tensor) -> float:
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


def metrics_for_data(data, ref) -> dict:
    cell = data.cell.squeeze(0).detach().cpu().float()
    pos = data.pos.detach().cpu().float()
    n = int(data.num_atoms.item()) if hasattr(data, "num_atoms") else int(data.atomic_numbers.numel())
    vol = abs(float(torch.det(cell).item()))
    dens = n / vol if vol > 1e-12 else float("inf")
    angles = cell_angles(cell)
    mind = min_dist_pbc(pos, cell) if vol > 1e-12 else 0.0

    ref_cell = ref.cell.squeeze(0).detach().cpu().float()
    ref_vol = abs(float(torch.det(ref_cell).item()))
    ref_dens = n / ref_vol if ref_vol > 1e-12 else float("inf")

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
        "volume_ratio": vol / ref_vol if ref_vol > 1e-12 else None,
        "density_ratio": dens / ref_dens if ref_dens > 1e-12 else None,
        "valid_cell": bool(valid_cell),
        "no_clash": bool(no_clash),
        "pass_basic": bool(valid_cell and no_clash),
    }


def summarize(records: list[dict]) -> dict:
    out = {"count": len(records)}
    if not records:
        return out
    for key in ["valid_cell", "no_clash", "pass_basic"]:
        out[f"{key}_rate"] = sum(bool(r[key]) for r in records) / len(records)
    for key in ["volume_ratio", "density_ratio", "atom_density", "min_dist", "min_angle", "max_angle"]:
        vals = [r[key] for r in records if r.get(key) is not None and math.isfinite(float(r[key]))]
        out[f"{key}_median"] = median(vals) if vals else None
        out[f"{key}_mean"] = mean(vals) if vals else None
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading model", RUN, "epoch", LOAD_EPOCH, flush=True)
    ckpt_info = MatterGenCheckpointInfo(
        model_path=RUN,
        load_epoch=LOAD_EPOCH,
        strict_checkpoint_loading=True,
    )
    model = load_model_diffusion(ckpt_info).to(get_device()).eval()
    print("loaded", ckpt_info.checkpoint_path, flush=True)

    cfg = ckpt_info.config
    print("instantiate data", flush=True)
    dm = instantiate(cfg.data_module)
    val = dm.val_dataset
    samples = [val[i] for i in range(NUM_VAL)]
    base = val.base_dataset
    material_ids = [str(base.structure_id[val.indices[i]]) for i in range(NUM_VAL)]
    print("val samples", len(samples), material_ids[:3], flush=True)

    with hydra.initialize_config_dir(str(ROOT / "sampling_conf")):
        scfg = hydra.compose(
            config_name="csp",
            overrides=[f"sampler_partial.N={N}", f"sampler_partial.eps_t={1 / N}"],
        )
    sampler_partial = instantiate(scfg.sampler_partial)
    sampler = sampler_partial(pl_module=model)

    all_summary = {}
    for mode in ["target", "full_prior", "t0p4_recovery"]:
        print("mode", mode, flush=True)
        records = []
        cdir = OUT / "cifs" / mode
        cdir.mkdir(parents=True, exist_ok=True)
        for start in range(0, NUM_VAL, BATCH_SIZE):
            chunk = samples[start : start + BATCH_SIZE]
            mids = material_ids[start : start + BATCH_SIZE]
            cond = collate(chunk).to(get_device())
            if mode == "target":
                out_batch = cond.to("cpu")
            elif mode == "full_prior":
                _, mean_batch = sampler.sample(cond)
                out_batch = mean_batch.to("cpu")
            elif mode == "t0p4_recovery":
                t0 = 0.4
                sampler._max_t = t0
                t = torch.full((cond.get_batch_size(),), t0, device=get_device())
                noisy = model.diffusion_module.corruption.sample_marginal(cond, t)
                _, mean_batch, _ = sampler._denoise(batch=noisy, mask={}, record=False)
                out_batch = mean_batch.to("cpu")
            else:
                raise ValueError(mode)

            refs = chunk
            outs = out_batch.to_data_list()
            structures = batch_structures(out_batch)
            for j, (data, ref, mid, struct) in enumerate(zip(outs, refs, mids, structures)):
                idx = start + j
                metrics = metrics_for_data(data, ref)
                metrics.update({"sample_index": idx, "material_id": mid, "mode": mode})
                records.append(metrics)
                CifWriter(struct).write_file(str(cdir / f"sample_{idx:03d}_{safe_mid(mid)}.cif"))

        with open(OUT / f"records_{mode}.jsonl", "w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        all_summary[mode] = summarize(records)
        with open(OUT / f"summary_{mode}.json", "w") as handle:
            json.dump(all_summary[mode], handle, indent=2)
        print(json.dumps({mode: all_summary[mode]}, indent=2), flush=True)

    summary = {
        "checkpoint": ckpt_info.checkpoint_path,
        "N": N,
        "batch_size": BATCH_SIZE,
        "num_val_samples": NUM_VAL,
        "modes": all_summary,
    }
    with open(OUT / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print("DONE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
