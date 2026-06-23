import csv
import gzip
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, median


ROOT = Path(os.environ.get("OMC25_ROOT", "."))
PYTHON = os.environ.get("MOLCSP_PYTHON", sys.executable)
SAMPLE_SCRIPT = ROOT / "scripts/tmp_molcsp_fixed_target_sample.py"
RDKIT_SCRIPT = ROOT / "scripts/tmp_rdkit_connectivity_check.py"
VAL_CSV = ROOT / "datasets/omc25_le50_mattergen/val.csv"
MAPPING_JSONL = ROOT / "datasets/molecule_mapping/omc25_le300_val_molmap_hybrid_v3.jsonl.gz"
GRAPH_JSONL = ROOT / "scripts/oe62_hybrid_graphs_all_v3.jsonl.gz"
DEFAULT_RUN = Path("/tmp/lmy_molcsp_epoch294_run")
SHARED_RUN = ROOT / "outputs/singlerun/2026-06-03/le50_molcsp_scratch_mean_cell025_bondhuber_lowt04_posonly_w1e-3_8gpu01234567"

DEFAULT_TARGETS = [
    "NUNLAJ|1|press|91b75842265b1f5",
    "KEZCUP|1|press|dcf4bb9f4bea973",
    "FUWRAQ|1|press|dcf4bb9f4bea973",
    "ANACEY|1|gener|91b75842265b1f5",
    "DCLETH02|2|gener|afbd67f619699cf",
    "BARBOL|2|press|21636368b529b4a",
    "BEYPUR|2|gener|cd613e3d8f16adf",
    "DCLETH02|4|press|a8c24d444ef7feb",
    "RHODIN01|4|gener|9b810e76ec9d286",
    "AKOVOL01|4|gener|1710cf527ac435a",
]


def safe_mid(material_id: str) -> str:
    return material_id.replace("|", "_").replace("/", "_")


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_val_rows() -> dict[str, dict]:
    rows = {}
    with open(VAL_CSV) as handle:
        for row in csv.DictReader(handle):
            rows[row["material_id"]] = row
    return rows


def read_mappings() -> dict[str, dict]:
    records = {}
    for rec in read_jsonl_gz(MAPPING_JSONL):
        if rec.get("success"):
            records[str(rec["material_id"])] = rec
    return records


def read_graphs() -> dict[str, dict]:
    records = {}
    for rec in read_jsonl_gz(GRAPH_JSONL):
        if rec.get("transfer_mode") == "rdkit_explicit_h_full_match" and rec.get("ok", True):
            records[str(rec["refcode_csd"])] = rec
    return records


def write_target_cif(material_id: str, val_rows: dict[str, dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if material_id not in val_rows:
        raise KeyError(f"{material_id} is not in {VAL_CSV}")
    cif_path = out_dir / f"target_{safe_mid(material_id)}.cif"
    cif_path.write_text(val_rows[material_id]["cif"])
    return cif_path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_records(records: list[dict]) -> dict:
    out = {"count": len(records)}
    if not records:
        return out
    bool_keys = [
        "valid_cell",
        "no_clash",
        "pass_basic",
        "rdkit_component_count_match",
        "rdkit_component_size_match",
        "rdkit_all_components_match_target",
        "rdkit_molecule_pass",
    ]
    for key in bool_keys:
        if key in records[0]:
            out[f"{key}_rate"] = sum(bool(r.get(key)) for r in records) / len(records)
    out["rdkit_basic_pass_rate"] = (
        sum(bool(r.get("rdkit_molecule_pass")) and bool(r.get("pass_basic")) for r in records) / len(records)
    )
    numeric_keys = [
        "atom_density",
        "volume_ratio",
        "density_ratio",
        "min_dist",
        "min_angle",
        "max_angle",
        "rdkit_num_components",
        "rdkit_num_target_like_components",
    ]
    for key in numeric_keys:
        vals = []
        for rec in records:
            val = rec.get(key)
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fval):
                vals.append(fval)
        if vals:
            out[f"{key}_median"] = median(vals)
            out[f"{key}_mean"] = mean(vals)
    passed = [r for r in records if r.get("rdkit_molecule_pass") and r.get("pass_basic")]
    out["rdkit_basic_pass_count"] = len(passed)
    if passed:
        vols = [float(r["volume"]) for r in passed if r.get("volume") is not None]
        dens = [float(r["atom_density"]) for r in passed if r.get("atom_density") is not None]
        if vols:
            out["pass_volume_median"] = median(vols)
            out["pass_volume_min"] = min(vols)
            out["pass_volume_max"] = max(vols)
            out["pass_volume_unique_0p01A3_bins"] = len({round(v, 2) for v in vols})
        if dens:
            out["pass_atom_density_median"] = median(dens)
    return out


def run_target(
    material_id: str,
    count: int,
    n_steps: int,
    batch_size: int,
    gpus: list[str],
    load_epoch: int,
    run_dir: Path,
    suite_dir: Path,
    force: bool,
    val_rows: dict[str, dict],
    mappings: dict[str, dict],
    graphs: dict[str, dict],
) -> dict:
    if material_id not in mappings:
        raise KeyError(f"{material_id} missing successful molecule mapping")
    mapping = mappings[material_id]
    refcode = str(mapping["csd_refcode"])
    graph = graphs.get(refcode, {})
    target_dir = suite_dir / "targets"
    target_cif = write_target_cif(material_id, val_rows, target_dir)
    out_dir = suite_dir / safe_mid(material_id)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "suite_target_summary.json"
    if summary_path.exists() and not force:
        with open(summary_path) as handle:
            return json.load(handle)

    per_gpu = count // len(gpus)
    remainder = count % len(gpus)
    procs = []
    start = 0
    for shard_id, gpu in enumerate(gpus):
        shard_count = per_gpu + (1 if shard_id < remainder else 0)
        if shard_count <= 0:
            continue
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMC25_ROOT": str(ROOT),
                "HYDRA_FULL_ERROR": "1",
                "MOLCSP_RUN": str(run_dir),
                "MOLCSP_OUT": str(out_dir),
                "MOLCSP_MATERIAL_ID": material_id,
                "MOLCSP_TARGET_CIF": str(target_cif),
                "MOLCSP_LOAD_EPOCH": str(load_epoch),
                "MOLCSP_N": str(n_steps),
                "MOLCSP_BATCH_SIZE": str(batch_size),
                "MOLCSP_COUNT": str(shard_count),
                "MOLCSP_START_INDEX": str(start),
                "MOLCSP_SHARD_ID": str(shard_id),
                "TORCH_NUM_THREADS": "2",
            }
        )
        log_path = logs_dir / f"sample_shard_{shard_id:02d}_gpu{gpu}.log"
        handle = open(log_path, "w")
        print(
            f"[target] {material_id} shard={shard_id} gpu={gpu} count={shard_count} start={start}",
            flush=True,
        )
        procs.append((subprocess.Popen([PYTHON, str(SAMPLE_SCRIPT)], env=env, cwd=str(ROOT / "mattergen"), stdout=handle, stderr=subprocess.STDOUT), handle, log_path))
        start += shard_count

    failures = []
    for proc, handle, log_path in procs:
        ret = proc.wait()
        handle.close()
        if ret != 0:
            failures.append((ret, str(log_path)))
    if failures:
        raise RuntimeError(f"{material_id} sampling failures: {failures}")

    env = os.environ.copy()
    env.update({"OMC25_ROOT": str(ROOT), "MOLCSP_SAMPLE_DIR": str(out_dir), "MOLCSP_MATERIAL_ID": material_id})
    rdkit_log = logs_dir / "rdkit_check.log"
    with open(rdkit_log, "w") as handle:
        ret = subprocess.call([PYTHON, str(RDKIT_SCRIPT)], env=env, cwd=str(ROOT / "mattergen"), stdout=handle, stderr=subprocess.STDOUT)
    if ret != 0:
        raise RuntimeError(f"{material_id} rdkit check failed, see {rdkit_log}")

    records = read_jsonl(out_dir / "rdkit_connectivity_records.jsonl")
    target_summary = {
        "material_id": material_id,
        "safe_id": safe_mid(material_id),
        "out_dir": str(out_dir),
        "target_cif": str(target_cif),
        "refcode": refcode,
        "smiles": graph.get("canonical_smiles_no_h") or graph.get("input_smiles"),
        "num_atoms": int(mapping["mapping"].get("num_atoms_all", sum(mapping["mapping"].get("component_sizes", [])))),
        "num_molecules": int(mapping["mapping"]["num_molecules"]),
        "component_sizes": mapping["mapping"]["component_sizes"],
        "count_requested": count,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "load_epoch": load_epoch,
        "summary": summarize_records(records),
    }
    with open(summary_path, "w") as handle:
        json.dump(target_summary, handle, indent=2)
    return target_summary


def main() -> None:
    suite_name = os.environ.get("MOLCSP_EVAL_NAME", "le50_molcsp_epoch294_fullprior_eval_v1")
    suite_dir = Path(os.environ.get("MOLCSP_EVAL_OUT", str(ROOT / "samples" / suite_name)))
    suite_dir.mkdir(parents=True, exist_ok=True)

    targets_env = os.environ.get("MOLCSP_EVAL_TARGETS")
    targets = [x.strip() for x in targets_env.split(",") if x.strip()] if targets_env else DEFAULT_TARGETS
    count = int(os.environ.get("MOLCSP_EVAL_COUNT", "512"))
    n_steps = int(os.environ.get("MOLCSP_EVAL_N", "1000"))
    batch_size = int(os.environ.get("MOLCSP_EVAL_BATCH_SIZE", "4"))
    gpus = [x.strip() for x in os.environ.get("MOLCSP_EVAL_GPUS", "0,1,2,3,4,5,6,7").split(",") if x.strip()]
    load_epoch = int(os.environ.get("MOLCSP_LOAD_EPOCH", "294"))
    run_dir = Path(os.environ.get("MOLCSP_RUN", str(DEFAULT_RUN if DEFAULT_RUN.exists() else SHARED_RUN)))
    force = os.environ.get("MOLCSP_EVAL_FORCE", "0") == "1"

    print(json.dumps({
        "suite_dir": str(suite_dir),
        "targets": targets,
        "count_per_target": count,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gpus": gpus,
        "load_epoch": load_epoch,
        "run_dir": str(run_dir),
        "force": force,
    }, indent=2), flush=True)

    val_rows = read_val_rows()
    mappings = read_mappings()
    graphs = read_graphs()

    summaries = []
    for idx, material_id in enumerate(targets, start=1):
        print(f"[suite] target {idx}/{len(targets)}: {material_id}", flush=True)
        summary = run_target(
            material_id=material_id,
            count=count,
            n_steps=n_steps,
            batch_size=batch_size,
            gpus=gpus,
            load_epoch=load_epoch,
            run_dir=run_dir,
            suite_dir=suite_dir,
            force=force,
            val_rows=val_rows,
            mappings=mappings,
            graphs=graphs,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    suite_summary = {
        "suite_dir": str(suite_dir),
        "count_per_target": count,
        "n_steps": n_steps,
        "load_epoch": load_epoch,
        "run_dir": str(run_dir),
        "targets": summaries,
    }
    rates = [s["summary"].get("rdkit_basic_pass_rate") for s in summaries if s["summary"].get("rdkit_basic_pass_rate") is not None]
    if rates:
        suite_summary["rdkit_basic_pass_rate_mean_over_targets"] = mean(rates)
        suite_summary["rdkit_basic_pass_rate_median_over_targets"] = median(rates)
    with open(suite_dir / "suite_summary.json", "w") as handle:
        json.dump(suite_summary, handle, indent=2)
    print("[suite] complete", flush=True)
    print(json.dumps(suite_summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[suite] failed: {exc}", file=sys.stderr, flush=True)
        raise
