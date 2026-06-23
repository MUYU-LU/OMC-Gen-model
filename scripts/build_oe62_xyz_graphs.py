#!/usr/bin/env python3
"""Build OE62 molecule graphs from xyz coordinates.

This avoids using RDKit AddHs as the source of explicit hydrogens. OE62 already
contains the relaxed molecular geometry with hydrogens, so atom counts and H
placement should come from that coordinate block.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path


ELEMENT_COVALENT_RADII = {
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
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.61,
    "Fe": 1.52,
    "Co": 1.50,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Kr": 1.16,
    "Rb": 2.20,
    "Sr": 1.95,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Sb": 1.39,
    "Te": 1.38,
    "I": 1.39,
    "Xe": 1.40,
}

ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
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


def parse_xyz(xyz: str) -> list[tuple[str, tuple[float, float, float]]]:
    lines = [line.strip() for line in str(xyz or "").splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty_xyz")
    natoms = int(lines[0])
    atoms = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbol = parts[0]
        coords = (float(parts[1]), float(parts[2]), float(parts[3]))
        atoms.append((symbol, coords))
    if len(atoms) != natoms:
        raise ValueError(f"xyz_count_mismatch:{len(atoms)}!={natoms}")
    return atoms


def bond_threshold(symbol_a: str, symbol_b: str) -> float | None:
    r1 = ELEMENT_COVALENT_RADII.get(symbol_a)
    r2 = ELEMENT_COVALENT_RADII.get(symbol_b)
    if r1 is None or r2 is None:
        return None
    ref = (r1 + r2) * 1.15
    if symbol_a == "H" or symbol_b == "H":
        return ref * 1.15
    return ref * 1.10


def formula_from_symbols(symbols: list[str]) -> str:
    counts = Counter(symbols)
    ordered = []
    for symbol in ("C", "H"):
        if counts.get(symbol):
            ordered.append((symbol, counts.pop(symbol)))
    for symbol in sorted(counts):
        ordered.append((symbol, counts[symbol]))
    return "".join(symbol if count == 1 else f"{symbol}{count}" for symbol, count in ordered)


def build_graph(row: dict) -> dict:
    refcode = str(row.get("refcode_csd") or row.get("csd_refcode") or "").strip()
    smiles = str(row.get("canonical_smiles") or row.get("smiles") or "").strip()
    inchi = str(row.get("inchi") or "").strip()
    if not refcode:
        raise ValueError("missing_refcode")
    atoms = parse_xyz(str(row.get("xyz_pbe_relaxed") or ""))

    atom_symbols = [symbol for symbol, _ in atoms]
    atomic_numbers = []
    for symbol in atom_symbols:
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"unknown_element:{symbol}")
        atomic_numbers.append(ATOMIC_NUMBERS[symbol])

    bonds = []
    degrees = [0] * len(atoms)
    for i, (symbol_i, coords_i) in enumerate(atoms):
        for j in range(i):
            symbol_j, coords_j = atoms[j]
            threshold = bond_threshold(symbol_i, symbol_j)
            if threshold is None:
                continue
            distance = math.dist(coords_i, coords_j)
            if distance <= threshold:
                bonds.append(
                    {
                        "begin": j,
                        "end": i,
                        "type": 0,
                        "type_name": "UNKNOWN_XYZ",
                        "distance": distance,
                        "threshold": threshold,
                    }
                )
                degrees[i] += 1
                degrees[j] += 1

    atom_features = [
        {
            "atomic_num": z,
            "degree": degrees[idx],
        }
        for idx, z in enumerate(atomic_numbers)
    ]

    return {
        "refcode_csd": refcode,
        "input_smiles": smiles,
        "canonical_smiles_no_h": smiles,
        "inchi": inchi,
        "formula": formula_from_symbols(atom_symbols),
        "num_atoms": len(atoms),
        "number_of_atoms_oe62": row.get("number_of_atoms"),
        "atom_symbols": atom_symbols,
        "atomic_numbers": atomic_numbers,
        "atom_features": atom_features,
        "bonds": bonds,
        "graph_source": "oe62_xyz_covalent_radii",
        "ok": True,
    }


def open_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-refcodes", nargs="*", default=None)
    args = parser.parse_args()

    rows = load_records(args.input)
    only = set(args.only_refcodes or [])
    seen: set[str] = set()
    stats = Counter()

    with open_output(args.output) as f:
        for row in rows:
            if args.limit is not None and stats["rows"] >= args.limit:
                break
            refcode = str(row.get("refcode_csd") or row.get("csd_refcode") or "").strip()
            if only and refcode not in only:
                continue
            stats["rows"] += 1
            if not refcode:
                stats["missing_refcode"] += 1
                continue
            if refcode in seen:
                stats["duplicates"] += 1
                continue
            seen.add(refcode)
            try:
                graph = build_graph(row)
            except Exception as exc:  # noqa: BLE001
                graph = {
                    "refcode_csd": refcode,
                    "input_smiles": str(row.get("canonical_smiles") or row.get("smiles") or "").strip(),
                    "inchi": str(row.get("inchi") or "").strip(),
                    "ok": False,
                    "error": f"{type(exc).__name__}:{exc}",
                    "graph_source": "oe62_xyz_covalent_radii",
                }
                stats["failed"] += 1
            else:
                stats["written_ok"] += 1
            f.write(json.dumps(graph, separators=(",", ":")) + "\n")

    stats["written_total"] = stats["written_ok"] + stats["failed"]
    print(json.dumps(dict(stats), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
