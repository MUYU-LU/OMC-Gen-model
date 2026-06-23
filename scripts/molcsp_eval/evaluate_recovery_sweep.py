import csv
import gzip
import json
import math
import os
from itertools import combinations
from pathlib import Path
from statistics import mean, median

import networkx as nx
import torch
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element


ROOT = Path(os.environ.get("OMC25_ROOT", "."))
SAMPLE_DIR = Path(os.environ["MOLCSP_SAMPLE_DIR"])
MATERIAL_ID = os.environ["MOLCSP_MATERIAL_ID"]
TARGET_CIF = Path(os.environ["MOLCSP_TARGET_CIF"])
MAPPING_JSONL = ROOT / "datasets/molecule_mapping/omc25_le300_val_molmap_hybrid_v3.jsonl.gz"
GRAPH_JSONL = ROOT / "scripts/oe62_hybrid_graphs_all_v3.jsonl.gz"
MODES = ["t0p8", "t0p6", "t0p5", "t0p3", "t0p1"]

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


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def find_mapping(material_id: str) -> dict:
    for record in read_jsonl_gz(MAPPING_JSONL):
        if str(record.get("material_id")) == material_id and record.get("success", False):
            return record
    raise RuntimeError(f"Missing mapping record for {material_id}")


def find_graph(refcode: str) -> dict:
    for record in read_jsonl_gz(GRAPH_JSONL):
        if (
            str(record.get("refcode_csd")) == str(refcode)
            and record.get("transfer_mode") == "rdkit_explicit_h_full_match"
            and record.get("ok", True)
        ):
            return record
    raise RuntimeError(f"Missing graph record for {refcode}")


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


def build_graph(atomic_numbers: list[int], bonds: set[tuple[int, int]]) -> nx.Graph:
    graph = nx.Graph()
    for idx, z in enumerate(atomic_numbers):
        graph.add_node(idx, Z=str(int(z)))
    for i, j in bonds:
        graph.add_edge(i, j)
    return graph


def target_smiles_hash(graph_record: dict) -> str:
    bonds = {tuple(sorted((int(b["begin"]), int(b["end"])))) for b in graph_record["bonds"]}
    graph = build_graph([int(z) for z in graph_record["atomic_numbers"]], bonds)
    return weisfeiler_lehman_graph_hash(graph, node_attr="Z")


def find_cif(mode: str, sample_index: int) -> Path:
    matches = sorted((SAMPLE_DIR / "cifs" / mode).glob(f"sample_{sample_index:04d}_*.cif"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one CIF for {mode} sample {sample_index}, found {matches[:5]}")
    return matches[0]


def summarize(records: list[dict]) -> dict:
    out = {"count": len(records)}
    bool_keys = [
        "valid_cell",
        "no_clash",
        "pass_basic",
        "expected_all_within_cutoff",
        "exact_bond_graph_match",
        "component_count_match",
        "component_size_match",
        "smiles_topology_hash_match_all",
        "molecule_connectivity_pass",
    ]
    for key in bool_keys:
        out[f"{key}_rate"] = sum(bool(r[key]) for r in records) / len(records)
    numeric_keys = [
        "bond_mae_A",
        "bond_rmse_A",
        "bond_rel_mae",
        "expected_bond_within_cutoff_rate",
        "missing_bonds",
        "extra_bonds",
        "num_components",
        "ref_pos_cart_rmsd_A",
        "ref_pos_frac_rmsd",
        "ref_cell_rmsd_A",
        "ref_cell_rel_rmsd",
    ]
    for key in numeric_keys:
        vals = [float(r[key]) for r in records if r.get(key) is not None and math.isfinite(float(r[key]))]
        out[f"{key}_median"] = median(vals) if vals else None
        out[f"{key}_mean"] = mean(vals) if vals else None
    return out


def pairwise_rmsd(frac_a: torch.Tensor, frac_b: torch.Tensor, ref_cell: torch.Tensor) -> float:
    dfrac = frac_a - frac_b
    dfrac = dfrac - torch.round(dfrac)
    cart = dfrac @ ref_cell
    return float(torch.sqrt(torch.mean(torch.sum(cart * cart, dim=-1))).item())


def diversity_summary(records: list[dict], arrays: dict[int, tuple[list[int], torch.Tensor, torch.Tensor]], ref_cell: torch.Tensor) -> dict:
    pair_pos = []
    pair_cell = []
    pair_cell_rel = []
    ids = [int(r["sample_index"]) for r in records]
    ref_norm = float(torch.sqrt(torch.mean(ref_cell**2)).item())
    for i, j in combinations(ids, 2):
        _, frac_i, cell_i = arrays[i]
        _, frac_j, cell_j = arrays[j]
        pos = pairwise_rmsd(frac_i, frac_j, ref_cell)
        cell = float(torch.sqrt(torch.mean((cell_i - cell_j) ** 2)).item())
        pair_pos.append(pos)
        pair_cell.append(cell)
        pair_cell_rel.append(cell / max(ref_norm, 1e-8))

    nearest = []
    for idx, i in enumerate(ids):
        vals = []
        for jdx, j in enumerate(ids):
            if i == j:
                continue
            _, frac_i, _ = arrays[i]
            _, frac_j, _ = arrays[j]
            vals.append(pairwise_rmsd(frac_i, frac_j, ref_cell))
        nearest.append(min(vals) if vals else 0.0)

    # Heuristic duplicate definition: almost same coordinates and cell under fixed atom order.
    duplicate_like = [
        pos < 0.15 and cell_rel < 0.03 for pos, cell_rel in zip(pair_pos, pair_cell_rel)
    ]
    return {
        "pair_count": len(pair_pos),
        "pair_pos_rmsd_A_median": median(pair_pos) if pair_pos else None,
        "pair_pos_rmsd_A_mean": mean(pair_pos) if pair_pos else None,
        "pair_cell_rmsd_A_median": median(pair_cell) if pair_cell else None,
        "pair_cell_rel_rmsd_median": median(pair_cell_rel) if pair_cell_rel else None,
        "nearest_neighbor_pos_rmsd_A_median": median(nearest) if nearest else None,
        "nearest_neighbor_pos_rmsd_A_mean": mean(nearest) if nearest else None,
        "duplicate_like_pair_rate": sum(duplicate_like) / len(duplicate_like) if duplicate_like else None,
    }


def main() -> None:
    mapping_record = find_mapping(MATERIAL_ID)
    graph_record = find_graph(mapping_record["csd_refcode"])
    target_atomic_numbers, target_frac, target_cell = structure_to_arrays(TARGET_CIF)
    expected_edge_set = {
        tuple(sorted((int(bond["begin"]), int(bond["end"]))))
        for bond in mapping_record.get("crystal_bonds", [])
    }
    target_hash = target_smiles_hash(graph_record)
    expected_component_sizes = sorted(int(x) for x in mapping_record["mapping"]["component_sizes"])
    expected_num_molecules = int(mapping_record["mapping"]["num_molecules"])

    summaries = {}
    all_rows = []
    for mode in MODES:
        rec_path = SAMPLE_DIR / f"records_{mode}.jsonl"
        if not rec_path.exists():
            continue
        records = []
        arrays = {}
        for rec in read_jsonl(rec_path):
            sample_index = int(rec["sample_index"])
            cif_path = find_cif(mode, sample_index)
            atomic_numbers, frac, cell = structure_to_arrays(cif_path)
            arrays[sample_index] = (atomic_numbers, frac, cell)
            inferred_bonds = infer_bonds(atomic_numbers, frac, cell)

            abs_errors = []
            rel_errors = []
            expected_within = []
            for i, j in expected_edge_set:
                d_gen = pbc_distance(frac, cell, i, j)
                d_ref = pbc_distance(target_frac, target_cell, i, j)
                abs_errors.append(abs(d_gen - d_ref))
                rel_errors.append(abs(d_gen - d_ref) / max(d_ref, 1e-8))
                expected_within.append(d_gen <= bond_cutoff(atomic_numbers[i], atomic_numbers[j]))

            missing = expected_edge_set - inferred_bonds
            extra = inferred_bonds - expected_edge_set
            inferred_graph = build_graph(atomic_numbers, inferred_bonds)
            components = [sorted(c) for c in nx.connected_components(inferred_graph)]
            component_sizes = sorted(len(c) for c in components)
            component_hashes = [
                weisfeiler_lehman_graph_hash(inferred_graph.subgraph(comp).copy(), node_attr="Z")
                for comp in components
            ]
            hash_match_all = len(components) == expected_num_molecules and all(h == target_hash for h in component_hashes)
            exact_graph_match = inferred_bonds == expected_edge_set

            out = {
                **rec,
                "cif": str(cif_path),
                "expected_num_bonds": len(expected_edge_set),
                "inferred_num_bonds": len(inferred_bonds),
                "missing_bonds": len(missing),
                "extra_bonds": len(extra),
                "num_components": len(components),
                "component_sizes": component_sizes,
                "expected_num_molecules": expected_num_molecules,
                "expected_component_sizes": expected_component_sizes,
                "atom_order_match": atomic_numbers == target_atomic_numbers,
                "bond_mae_A": mean(abs_errors),
                "bond_rmse_A": math.sqrt(mean([x * x for x in abs_errors])),
                "bond_rel_mae": mean(rel_errors),
                "expected_bond_within_cutoff_rate": sum(expected_within) / len(expected_within),
                "expected_all_within_cutoff": all(expected_within),
                "exact_bond_graph_match": exact_graph_match,
                "component_count_match": len(components) == expected_num_molecules,
                "component_size_match": component_sizes == expected_component_sizes,
                "smiles_topology_hash_match_all": hash_match_all,
                "molecule_connectivity_pass": (
                    atomic_numbers == target_atomic_numbers
                    and exact_graph_match
                    and len(components) == expected_num_molecules
                    and component_sizes == expected_component_sizes
                    and hash_match_all
                ),
                "missing_bond_examples": sorted(list(missing))[:12],
                "extra_bond_examples": sorted(list(extra))[:12],
            }
            records.append(out)
            all_rows.append(out)

        connectivity = summarize(records)
        diversity = diversity_summary(records, arrays, target_cell)
        summaries[mode] = {"connectivity": connectivity, "diversity": diversity}
        with open(SAMPLE_DIR / f"eval_records_{mode}.jsonl", "w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    with open(SAMPLE_DIR / "eval_summary.json", "w") as handle:
        json.dump(summaries, handle, indent=2)

    keys = [
        "mode",
        "sample_index",
        "t_start",
        "pass_basic",
        "molecule_connectivity_pass",
        "expected_bond_within_cutoff_rate",
        "missing_bonds",
        "extra_bonds",
        "bond_mae_A",
        "ref_pos_cart_rmsd_A",
        "ref_cell_rel_rmsd",
        "num_components",
        "smiles_topology_hash_match_all",
        "cif",
    ]
    with open(SAMPLE_DIR / "eval_records.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k) for k in keys})

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
