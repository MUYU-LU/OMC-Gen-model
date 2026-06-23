import csv
import gzip
import json
import math
import os
from pathlib import Path
from statistics import mean, median

import torch
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from rdkit import Chem


ROOT = Path(os.environ.get("OMC25_ROOT", "."))
SAMPLE_DIR = Path(os.environ["MOLCSP_SAMPLE_DIR"])
MATERIAL_ID = os.environ["MOLCSP_MATERIAL_ID"]
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
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def find_mapping(material_id: str) -> dict:
    for rec in read_jsonl_gz(MAPPING_JSONL):
        if str(rec.get("material_id")) == material_id and rec.get("success", False):
            return rec
    raise RuntimeError(f"Missing mapping for {material_id}")


def find_graph(refcode: str) -> dict:
    for rec in read_jsonl_gz(GRAPH_JSONL):
        if (
            str(rec.get("refcode_csd")) == str(refcode)
            and rec.get("transfer_mode") == "rdkit_explicit_h_full_match"
            and rec.get("ok", True)
        ):
            return rec
    raise RuntimeError(f"Missing graph for {refcode}")


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
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return sorted((sorted(v) for v in comps.values()), key=lambda x: (len(x), x))


def submol_for_component(atomic_numbers: list[int], bonds: set[tuple[int, int]], comp: list[int]) -> Chem.Mol:
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


def find_cif(sample_index: int) -> Path:
    cdir = SAMPLE_DIR / "cifs" / "full_prior"
    matches = sorted(cdir.glob(f"sample_{sample_index:04d}_*.cif"))
    if not matches:
        matches = sorted(cdir.glob(f"sample_{sample_index:03d}_*.cif"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one CIF for sample {sample_index}, found {matches[:5]}")
    return matches[0]


def summarize(records: list[dict]) -> dict:
    out = {"count": len(records)}
    bool_keys = [
        "rdkit_component_count_match",
        "rdkit_component_size_match",
        "rdkit_all_components_match_target",
        "rdkit_molecule_pass",
    ]
    for key in bool_keys:
        out[f"{key}_rate"] = sum(bool(r[key]) for r in records) / len(records)
    numeric_keys = ["rdkit_num_components", "rdkit_num_target_like_components"]
    for key in numeric_keys:
        vals = [float(r[key]) for r in records if r.get(key) is not None and math.isfinite(float(r[key]))]
        out[f"{key}_median"] = median(vals) if vals else None
        out[f"{key}_mean"] = mean(vals) if vals else None
    return out


def main() -> None:
    records_path = SAMPLE_DIR / "records_full_prior.jsonl"
    if not records_path.exists():
        with open(records_path, "w") as out:
            for shard_file in sorted((SAMPLE_DIR / "shards").glob("shard_*/records_full_prior.jsonl")):
                out.write(shard_file.read_text())

    mapping = find_mapping(MATERIAL_ID)
    graph = find_graph(mapping["csd_refcode"])
    target_mol = target_mol_from_graph_record(graph)
    expected_num_molecules = int(mapping["mapping"]["num_molecules"])
    expected_component_sizes = sorted(int(x) for x in mapping["mapping"]["component_sizes"])

    out_records = []
    for rec in read_jsonl(records_path):
        sample_index = int(rec["sample_index"])
        cif_path = find_cif(sample_index)
        atomic_numbers, frac, cell = structure_to_arrays(cif_path)
        bonds = infer_bonds(atomic_numbers, frac, cell)
        comps = components_from_bonds(len(atomic_numbers), bonds)
        comp_mols = [submol_for_component(atomic_numbers, bonds, comp) for comp in comps]
        comp_matches = [rdkit_isomorphic(mol, target_mol) for mol in comp_mols]
        component_sizes = sorted(len(c) for c in comps)
        out = {
            **rec,
            "cif": str(cif_path),
            "rdkit_target_smiles": graph.get("canonical_smiles_no_h") or graph.get("input_smiles"),
            "rdkit_num_components": len(comps),
            "rdkit_component_sizes": component_sizes,
            "rdkit_expected_num_molecules": expected_num_molecules,
            "rdkit_expected_component_sizes": expected_component_sizes,
            "rdkit_num_target_like_components": sum(comp_matches),
            "rdkit_component_count_match": len(comps) == expected_num_molecules,
            "rdkit_component_size_match": component_sizes == expected_component_sizes,
            "rdkit_all_components_match_target": (
                len(comps) == expected_num_molecules and all(comp_matches)
            ),
            "rdkit_molecule_pass": (
                len(comps) == expected_num_molecules
                and component_sizes == expected_component_sizes
                and all(comp_matches)
            ),
        }
        out_records.append(out)

    with open(SAMPLE_DIR / "rdkit_connectivity_records.jsonl", "w") as handle:
        for record in out_records:
            handle.write(json.dumps(record) + "\n")
    summary = summarize(out_records)
    with open(SAMPLE_DIR / "rdkit_connectivity_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    keys = [
        "sample_index",
        "material_id",
        "pass_basic",
        "rdkit_num_components",
        "rdkit_num_target_like_components",
        "rdkit_component_count_match",
        "rdkit_component_size_match",
        "rdkit_all_components_match_target",
        "rdkit_molecule_pass",
        "cif",
    ]
    with open(SAMPLE_DIR / "rdkit_connectivity_records.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in out_records:
            writer.writerow({k: record.get(k) for k in keys})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
