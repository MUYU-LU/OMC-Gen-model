#!/usr/bin/env python3
"""Build RDKit molecule graphs for OE62 refcodes.

The output is JSONL so the remote MatterGen environment can consume molecule
graphs without importing RDKit during training/data preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


BOND_TYPE_TO_INT = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}


def load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and {"columns", "data"}.issubset(data):
            columns = data["columns"]
            return [dict(zip(columns, row)) for row in data["data"]]
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported JSON shape in {path}")

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_graph(refcode: str, smiles: str, inchi: str | None) -> dict | None:
    smiles = (smiles or "").strip()
    if not refcode or not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "refcode_csd": refcode,
            "input_smiles": smiles,
            "inchi": inchi,
            "ok": False,
            "error": "rdkit_parse_failed",
        }

    mol = Chem.AddHs(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    order = [idx for idx, _ in sorted(enumerate(ranks), key=lambda item: item[1])]
    mol = Chem.RenumberAtoms(mol, order)
    Chem.SanitizeMol(mol)

    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(
            {
                "atomic_num": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "degree": atom.GetDegree(),
                "total_valence": atom.GetTotalValence(),
                "is_aromatic": bool(atom.GetIsAromatic()),
                "hybridization": str(atom.GetHybridization()),
            }
        )

    bonds = []
    for bond in mol.GetBonds():
        bonds.append(
            {
                "begin": bond.GetBeginAtomIdx(),
                "end": bond.GetEndAtomIdx(),
                "type": BOND_TYPE_TO_INT.get(bond.GetBondType(), 0),
                "type_name": str(bond.GetBondType()),
                "is_aromatic": bool(bond.GetIsAromatic()),
            }
        )

    return {
        "refcode_csd": refcode,
        "input_smiles": smiles,
        "canonical_smiles_h": Chem.MolToSmiles(mol, canonical=True),
        "canonical_smiles_no_h": Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True),
        "inchi": (inchi or "").strip(),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "num_atoms": mol.GetNumAtoms(),
        "atom_symbols": atom_symbols,
        "atomic_numbers": atomic_numbers,
        "atom_features": atom_features,
        "bonds": bonds,
        "ok": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_records(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    stats = {"rows": 0, "written": 0, "duplicates": 0, "failed": 0}
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            if args.limit is not None and stats["rows"] >= args.limit:
                break
            stats["rows"] += 1
            refcode = str(row.get("refcode_csd") or row.get("csd_refcode") or "").strip()
            if not refcode:
                continue
            if refcode in seen:
                stats["duplicates"] += 1
                continue
            seen.add(refcode)
            graph = build_graph(
                refcode=refcode,
                smiles=str(row.get("canonical_smiles") or row.get("smiles") or ""),
                inchi=row.get("inchi"),
            )
            if graph is None:
                continue
            if not graph.get("ok"):
                stats["failed"] += 1
            stats["written"] += 1
            f.write(json.dumps(graph, separators=(",", ":")) + "\n")

    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
