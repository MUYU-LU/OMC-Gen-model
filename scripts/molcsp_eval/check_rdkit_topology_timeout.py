import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import pathlib
import time


M = None
TARGET_MOL = None
EXPECTED_N = None
EXPECTED_SIZES = None
SAMPLE_DIR = None
SAFE_MID = None


def _load_checker(root: str, sample_dir: str, material_id: str):
    os.environ["OMC25_ROOT"] = root
    os.environ["MOLCSP_SAMPLE_DIR"] = sample_dir
    os.environ["MOLCSP_MATERIAL_ID"] = material_id
    path = pathlib.Path(root) / "scripts/molcsp_eval/check_rdkit_topology.py"
    spec = importlib.util.spec_from_file_location("rdchk", str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _safe_mid(material_id: str) -> str:
    return material_id.replace("|", "_").replace("/", "_")


def _init_worker(root: str, sample_dir: str, material_id: str):
    global M, TARGET_MOL, EXPECTED_N, EXPECTED_SIZES, SAMPLE_DIR, SAFE_MID
    M = _load_checker(root, sample_dir, material_id)
    SAMPLE_DIR = pathlib.Path(sample_dir)
    SAFE_MID = _safe_mid(material_id)
    mapping = M.find_mapping(material_id)
    graph = M.find_graph(mapping["csd_refcode"])
    TARGET_MOL = M.target_mol_from_graph_record(graph)
    EXPECTED_N = int(mapping["mapping"]["num_molecules"])
    EXPECTED_SIZES = sorted(int(x) for x in mapping["mapping"]["component_sizes"])


def _eval_one(rec: dict) -> dict:
    assert M is not None
    sample_index = int(rec["sample_index"])
    cif_path = SAMPLE_DIR / "cifs/full_prior" / f"sample_{sample_index:04d}_{SAFE_MID}.cif"
    if not cif_path.exists():
        cif_path = M.find_cif(sample_index)
    atomic_numbers, frac, cell = M.structure_to_arrays(cif_path)
    bonds = M.infer_bonds(atomic_numbers, frac, cell)
    comps = M.components_from_bonds(len(atomic_numbers), bonds)
    comp_mols = [M.submol_for_component(atomic_numbers, bonds, comp) for comp in comps]
    comp_matches = [M.rdkit_isomorphic(mol, TARGET_MOL) for mol in comp_mols]
    component_sizes = sorted(len(c) for c in comps)
    component_count_match = len(comps) == EXPECTED_N
    component_size_match = component_sizes == EXPECTED_SIZES
    molecule_pass = component_count_match and component_size_match and all(comp_matches)
    return {
        "sample_index": sample_index,
        "pass_basic": bool(rec.get("pass_basic")),
        "valid_cell": bool(rec.get("valid_cell")),
        "no_clash": bool(rec.get("no_clash")),
        "rdkit_num_components": len(comps),
        "rdkit_num_target_like_components": sum(comp_matches),
        "rdkit_component_count_match": component_count_match,
        "rdkit_component_size_match": component_size_match,
        "rdkit_molecule_pass": molecule_pass,
        "rdkit_basic_pass": molecule_pass and bool(rec.get("pass_basic")),
    }


def _summarize(total_count: int, results: list[dict], timeout_count: int) -> dict:
    def rate(key: str, denom: int = total_count) -> float:
        if denom <= 0:
            return 0.0
        return sum(bool(r.get(key)) for r in results) / denom

    evaluated = len(results)
    return {
        "count": total_count,
        "evaluated_count": evaluated,
        "timeout_count": timeout_count,
        "valid_cell_rate": rate("valid_cell"),
        "no_clash_rate": rate("no_clash"),
        "pass_basic_rate": rate("pass_basic"),
        "rdkit_component_count_match_rate": rate("rdkit_component_count_match"),
        "rdkit_component_size_match_rate": rate("rdkit_component_size_match"),
        "rdkit_molecule_pass_rate": rate("rdkit_molecule_pass"),
        "rdkit_basic_pass_count": sum(bool(r.get("rdkit_basic_pass")) for r in results),
        "rdkit_basic_pass_rate": rate("rdkit_basic_pass"),
        "rdkit_basic_pass_rate_evaluated": rate("rdkit_basic_pass", evaluated),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("OMC25_ROOT", "."))
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--material-id", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--wall-timeout", type=float, default=900.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Parent import is only used for reading the lightweight records file.
    parent = _load_checker(args.root, args.sample_dir, args.material_id)
    records = parent.read_jsonl(pathlib.Path(args.sample_dir) / "records_full_prior.jsonl")
    total = len(records)

    ctx = mp.get_context("fork")
    pool = ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.root, args.sample_dir, args.material_id),
    )
    asyncs = [pool.apply_async(_eval_one, (rec,)) for rec in records]
    start = time.time()
    results = []
    seen = set()
    try:
        while len(seen) < len(asyncs) and time.time() - start < args.wall_timeout:
            for i, fut in enumerate(asyncs):
                if i in seen or not fut.ready():
                    continue
                seen.add(i)
                try:
                    results.append(fut.get(timeout=0))
                except Exception as exc:
                    results.append({"sample_index": int(records[i]["sample_index"]), "error": repr(exc)})
            time.sleep(0.2)
    finally:
        pool.terminate()
        pool.join()

    timeout_count = total - len(seen)
    summary = {
        "material_id": args.material_id,
        "sample_dir": args.sample_dir,
        "wall_timeout": args.wall_timeout,
        "workers": args.workers,
        "summary": _summarize(total, results, timeout_count),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
