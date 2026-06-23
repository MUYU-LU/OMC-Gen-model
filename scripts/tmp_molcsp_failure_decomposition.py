import csv
import gzip
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import torch
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from rdkit import Chem


ROOT = Path(os.environ.get("OMC25_ROOT", "."))
SUITE_DIR = Path(
    os.environ.get(
        "MOLCSP_SUITE_DIR",
        ROOT / "samples/le50_molcsp_epoch294_fullprior_eval_v1",
    )
)
MAPPING_JSONL = ROOT / "datasets/molecule_mapping/omc25_le300_val_molmap_hybrid_v3.jsonl.gz"
GRAPH_JSONL = ROOT / "scripts/oe62_hybrid_graphs_all_v3.jsonl.gz"

COVALENT_RADII = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "Br": 1.20,
    "I": 1.39,
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_mappings() -> dict[str, dict]:
    return {
        str(rec["material_id"]): rec
        for rec in read_jsonl_gz(MAPPING_JSONL)
        if rec.get("success")
    }


def load_graphs() -> dict[str, dict]:
    out = {}
    for rec in read_jsonl_gz(GRAPH_JSONL):
        if rec.get("transfer_mode") == "rdkit_explicit_h_full_match" and rec.get("ok", True):
            out[str(rec["refcode_csd"])] = rec
    return out


def radius_for_z(z: int) -> float:
    symbol = Element.from_Z(int(z)).symbol
    if symbol in COVALENT_RADII:
        return COVALENT_RADII[symbol]
    radius = Element(symbol).covalent_radius
    if radius is None:
        raise KeyError(f"No covalent radius for {symbol}")
    return float(radius)


def bond_cutoff(z1: int, z2: int) -> float:
    base = (radius_for_z(z1) + radius_for_z(z2)) * 1.15
    return base * (1.15 if (int(z1) == 1 or int(z2) == 1) else 1.10)


def structure_to_arrays(cif_path: Path) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    structure = Structure.from_file(str(cif_path))
    atomic_numbers = [int(site.specie.Z) for site in structure.sites]
    frac = torch.tensor(structure.frac_coords, dtype=torch.float32)
    cell = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
    return atomic_numbers, frac, cell


def pbc_distance(frac: torch.Tensor, cell: torch.Tensor, i: int, j: int) -> float:
    dfrac = frac[i] - frac[j]
    dfrac = dfrac - torch.round(dfrac)
    return float(torch.linalg.norm(dfrac @ cell).item())


def infer_bonds(atomic_numbers: list[int], frac: torch.Tensor, cell: torch.Tensor) -> set[tuple[int, int]]:
    bonds = set()
    for i in range(len(atomic_numbers)):
        for j in range(i + 1, len(atomic_numbers)):
            if pbc_distance(frac, cell, i, j) <= bond_cutoff(atomic_numbers[i], atomic_numbers[j]):
                bonds.add((i, j))
    return bonds


def mol_from_graph(atomic_numbers: list[int], bonds: set[tuple[int, int]]) -> Chem.Mol:
    rw = Chem.RWMol()
    for z in atomic_numbers:
        rw.AddAtom(Chem.Atom(int(z)))
    for i, j in sorted(bonds):
        rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol


def components_from_bonds(n: int, bonds: set[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in bonds:
        union(i, j)
    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    return sorted((sorted(v) for v in comps.values()), key=lambda x: (len(x), x))


def submol_for_component(
    atomic_numbers: list[int], bonds: set[tuple[int, int]], comp: list[int]
) -> Chem.Mol:
    comp_set = set(comp)
    old_to_new = {old: new for new, old in enumerate(comp)}
    comp_z = [atomic_numbers[i] for i in comp]
    comp_bonds = {
        tuple(sorted((old_to_new[i], old_to_new[j])))
        for i, j in bonds
        if i in comp_set and j in comp_set
    }
    return mol_from_graph(comp_z, comp_bonds)


def target_mol_from_graph_record(graph_record: dict) -> Chem.Mol:
    atomic_numbers = [int(z) for z in graph_record["atomic_numbers"]]
    bonds = {tuple(sorted((int(b["begin"]), int(b["end"])))) for b in graph_record["bonds"]}
    return mol_from_graph(atomic_numbers, bonds)


def rdkit_isomorphic(mol_a: Chem.Mol, mol_b: Chem.Mol) -> bool:
    if mol_a.GetNumAtoms() != mol_b.GetNumAtoms() or mol_a.GetNumBonds() != mol_b.GetNumBonds():
        return False
    return bool(mol_a.HasSubstructMatch(mol_b) and mol_b.HasSubstructMatch(mol_a))


def expected_edges(mapping_record: dict) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(b["begin"]), int(b["end"]))))
        for b in mapping_record.get("crystal_bonds", [])
    }


def primary_failure(record: dict) -> str:
    if record["rdkit_molecule_pass"]:
        return "rdkit_pass"
    if not record.get("pass_basic", True):
        return "basic_invalid"
    if record["num_components"] < record["expected_num_molecules"]:
        return "component_under_count_merge_like"
    if record["num_components"] > record["expected_num_molecules"]:
        return "component_over_count_split_like"
    if not record["component_size_match"]:
        return "component_size_mismatch"
    if record["inferred_bond_count"] < record["expected_bond_count"]:
        return "bond_count_low_missing_like"
    if record["inferred_bond_count"] > record["expected_bond_count"]:
        return "bond_count_high_extra_like"
    return "wrong_topology_same_count_size_bondcount"


def read_target_records(target_dir: Path) -> list[dict]:
    records = read_jsonl(target_dir / "rdkit_connectivity_records.jsonl")
    if not records:
        shard_records = []
        for shard_file in sorted((target_dir / "shards").glob("shard_*/records_full_prior.jsonl")):
            shard_records.extend(read_jsonl(shard_file))
        return shard_records
    return records


def find_cif(target_dir: Path, sample_index: int) -> Path:
    cdir = target_dir / "cifs/full_prior"
    matches = sorted(cdir.glob(f"sample_{sample_index:04d}_*.cif"))
    if not matches:
        matches = sorted(cdir.glob(f"sample_{sample_index:03d}_*.cif"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one CIF for sample {sample_index}, found {matches[:5]}")
    return matches[0]


def decompose_target(target_dir: Path, mappings: dict[str, dict], graphs: dict[str, dict]) -> dict:
    summary_path = target_dir / "suite_target_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    suite_summary = json.loads(summary_path.read_text())
    material_id = str(suite_summary["material_id"])
    mapping_record = mappings[material_id]
    mapping = mapping_record["mapping"]
    graph = graphs[str(mapping_record["csd_refcode"])]
    target_mol = target_mol_from_graph_record(graph)

    exp_edges = expected_edges(mapping_record)
    expected_num_molecules = int(mapping["num_molecules"])
    expected_component_sizes = sorted(int(x) for x in mapping["component_sizes"])
    mol_ids = [int(x) for x in mapping["mol_id"]]
    expected_bond_count = len(exp_edges)

    out_records = []
    for base in read_target_records(target_dir):
        sample_index = int(base["sample_index"])
        cif_path = find_cif(target_dir, sample_index)
        atomic_numbers, frac, cell = structure_to_arrays(cif_path)
        inf_edges = infer_bonds(atomic_numbers, frac, cell)
        comps = components_from_bonds(len(atomic_numbers), inf_edges)
        comp_sizes = sorted(len(c) for c in comps)
        comp_mols = [submol_for_component(atomic_numbers, inf_edges, c) for c in comps]
        comp_matches = [rdkit_isomorphic(mol, target_mol) for mol in comp_mols]

        fixed_missing = exp_edges - inf_edges
        fixed_extra = inf_edges - exp_edges
        inter_copy_edges = {
            (i, j) for i, j in inf_edges if mol_ids[i] != mol_ids[j]
        }
        same_copy_extra = {
            (i, j) for i, j in fixed_extra if mol_ids[i] == mol_ids[j]
        }
        mixed_components = [
            sorted({mol_ids[i] for i in comp})
            for comp in comps
            if len({mol_ids[i] for i in comp}) > 1
        ]
        copy_fragment_counts = []
        for copy_id in sorted(set(mol_ids)):
            copy_fragment_counts.append(
                sum(any(mol_ids[i] == copy_id for i in comp) for comp in comps)
            )
        inferred_bond_count = len(inf_edges)
        rec = {
            **base,
            "cif": str(cif_path),
            "expected_num_molecules": expected_num_molecules,
            "expected_component_sizes": expected_component_sizes,
            "expected_bond_count": expected_bond_count,
            "num_components": len(comps),
            "component_sizes": comp_sizes,
            "num_target_like_components": sum(comp_matches),
            "component_count_match": len(comps) == expected_num_molecules,
            "component_size_match": comp_sizes == expected_component_sizes,
            "rdkit_molecule_pass": (
                len(comps) == expected_num_molecules
                and comp_sizes == expected_component_sizes
                and all(comp_matches)
            ),
            "inferred_bond_count": inferred_bond_count,
            "bond_count_delta": inferred_bond_count - expected_bond_count,
            "fixed_missing_bonds": len(fixed_missing),
            "fixed_extra_bonds": len(fixed_extra),
            "same_copy_extra_bonds": len(same_copy_extra),
            "inter_copy_false_bonds": len(inter_copy_edges),
            "has_inter_copy_false_bond": bool(inter_copy_edges),
            "num_mixed_copy_components": len(mixed_components),
            "has_mixed_copy_component": bool(mixed_components),
            "copy_fragment_counts": copy_fragment_counts,
            "max_copy_fragments": max(copy_fragment_counts) if copy_fragment_counts else 0,
            "has_fixed_copy_split": any(x > 1 for x in copy_fragment_counts),
            "fixed_edge_match": len(fixed_missing) == 0 and len(fixed_extra) == 0,
        }
        rec["primary_failure"] = primary_failure(rec)
        rec["rdkit_pass_but_fixed_edge_mismatch"] = (
            rec["rdkit_molecule_pass"] and not rec["fixed_edge_match"]
        )
        rec["rdkit_pass_but_copy_mixed"] = (
            rec["rdkit_molecule_pass"] and rec["has_mixed_copy_component"]
        )
        out_records.append(rec)

    target_out = target_dir / "failure_decomposition_records.jsonl"
    with open(target_out, "w") as handle:
        for rec in out_records:
            handle.write(json.dumps(rec) + "\n")
    csv_keys = [
        "sample_index",
        "pass_basic",
        "rdkit_molecule_pass",
        "primary_failure",
        "num_components",
        "component_sizes",
        "num_target_like_components",
        "inferred_bond_count",
        "bond_count_delta",
        "fixed_missing_bonds",
        "fixed_extra_bonds",
        "same_copy_extra_bonds",
        "inter_copy_false_bonds",
        "num_mixed_copy_components",
        "max_copy_fragments",
        "fixed_edge_match",
        "rdkit_pass_but_fixed_edge_mismatch",
        "rdkit_pass_but_copy_mixed",
        "cif",
    ]
    with open(target_dir / "failure_decomposition_records.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_keys)
        writer.writeheader()
        for rec in out_records:
            writer.writerow({k: rec.get(k) for k in csv_keys})

    tgt_summary = summarize_target(suite_summary, out_records)
    with open(target_dir / "failure_decomposition_summary.json", "w") as handle:
        json.dump(tgt_summary, handle, indent=2)
    return tgt_summary


def rate(records: list[dict], key: str) -> float:
    return sum(bool(r.get(key)) for r in records) / len(records) if records else 0.0


def finite_values(records: list[dict], key: str) -> list[float]:
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
    return vals


def summarize_target(suite_summary: dict, records: list[dict]) -> dict:
    failures = [r for r in records if not r["rdkit_molecule_pass"]]
    rdkit_pass = [r for r in records if r["rdkit_molecule_pass"]]
    primary_counts = Counter(r["primary_failure"] for r in records)
    failure_primary_counts = Counter(r["primary_failure"] for r in failures)

    def med(key: str):
        vals = finite_values(records, key)
        return median(vals) if vals else None

    def meanv(key: str):
        vals = finite_values(records, key)
        return mean(vals) if vals else None

    return {
        "material_id": suite_summary["material_id"],
        "safe_id": suite_summary["safe_id"],
        "refcode": suite_summary["refcode"],
        "num_atoms": suite_summary["num_atoms"],
        "num_molecules": suite_summary["num_molecules"],
        "component_sizes": suite_summary["component_sizes"],
        "count": len(records),
        "rdkit_molecule_pass_rate": rate(records, "rdkit_molecule_pass"),
        "pass_basic_rate": rate(records, "pass_basic"),
        "fixed_edge_match_rate": rate(records, "fixed_edge_match"),
        "rdkit_pass_but_fixed_edge_mismatch_rate": (
            sum(bool(r["rdkit_pass_but_fixed_edge_mismatch"]) for r in rdkit_pass) / len(rdkit_pass)
            if rdkit_pass
            else None
        ),
        "rdkit_pass_but_copy_mixed_rate": (
            sum(bool(r["rdkit_pass_but_copy_mixed"]) for r in rdkit_pass) / len(rdkit_pass)
            if rdkit_pass
            else None
        ),
        "has_inter_copy_false_bond_rate": rate(records, "has_inter_copy_false_bond"),
        "has_mixed_copy_component_rate": rate(records, "has_mixed_copy_component"),
        "has_fixed_copy_split_rate": rate(records, "has_fixed_copy_split"),
        "fixed_missing_bonds_median": med("fixed_missing_bonds"),
        "fixed_extra_bonds_median": med("fixed_extra_bonds"),
        "inter_copy_false_bonds_median": med("inter_copy_false_bonds"),
        "bond_count_delta_median": med("bond_count_delta"),
        "num_components_median": med("num_components"),
        "num_target_like_components_mean": meanv("num_target_like_components"),
        "primary_counts": dict(primary_counts),
        "failure_primary_counts": dict(failure_primary_counts),
        "failure_primary_rates": {
            k: v / len(failures) if failures else 0.0
            for k, v in failure_primary_counts.items()
        },
    }


def write_suite_summary(target_summaries: list[dict]) -> None:
    out = {
        "suite_dir": str(SUITE_DIR),
        "num_targets": len(target_summaries),
        "targets": target_summaries,
    }
    by_z = {}
    for z in sorted({int(s["num_molecules"]) for s in target_summaries}):
        xs = [s for s in target_summaries if int(s["num_molecules"]) == z]
        by_z[str(z)] = {
            "num_targets": len(xs),
            "rdkit_molecule_pass_rate_mean": mean(s["rdkit_molecule_pass_rate"] for s in xs),
            "rdkit_molecule_pass_rate_median": median(s["rdkit_molecule_pass_rate"] for s in xs),
            "fixed_edge_match_rate_mean": mean(s["fixed_edge_match_rate"] for s in xs),
            "has_inter_copy_false_bond_rate_mean": mean(s["has_inter_copy_false_bond_rate"] for s in xs),
            "has_mixed_copy_component_rate_mean": mean(s["has_mixed_copy_component_rate"] for s in xs),
        }
    out["by_z"] = by_z
    with open(SUITE_DIR / "failure_decomposition_suite_summary.json", "w") as handle:
        json.dump(out, handle, indent=2)

    keys = [
        "safe_id",
        "num_molecules",
        "num_atoms",
        "rdkit_molecule_pass_rate",
        "fixed_edge_match_rate",
        "rdkit_pass_but_fixed_edge_mismatch_rate",
        "rdkit_pass_but_copy_mixed_rate",
        "has_inter_copy_false_bond_rate",
        "has_mixed_copy_component_rate",
        "has_fixed_copy_split_rate",
        "fixed_missing_bonds_median",
        "fixed_extra_bonds_median",
        "inter_copy_false_bonds_median",
        "bond_count_delta_median",
        "num_components_median",
        "num_target_like_components_mean",
    ]
    with open(SUITE_DIR / "failure_decomposition_suite_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for summary in target_summaries:
            writer.writerow({k: summary.get(k) for k in keys})


def main() -> None:
    mappings = load_mappings()
    graphs = load_graphs()
    target_summaries = []
    for summary_path in sorted(SUITE_DIR.glob("*/suite_target_summary.json")):
        target_dir = summary_path.parent
        print(f"[decompose] {target_dir.name}", flush=True)
        target_summaries.append(decompose_target(target_dir, mappings, graphs))
    write_suite_summary(target_summaries)
    print(json.dumps({"suite_dir": str(SUITE_DIR), "num_targets": len(target_summaries)}, indent=2))


if __name__ == "__main__":
    main()
