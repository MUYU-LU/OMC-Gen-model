#!/usr/bin/env python3
"""Build canonical-ish OE62 molecule graphs without trusting RDKit AddHs blindly.

Policy:
- OE62 xyz is authoritative for explicit atoms, especially H count.
- RDKit/SMILES is used to transfer canonical ordering, bond order, aromaticity,
  charge, and hybridization when it can be matched to the xyz graph.
- If RDKit cannot be matched, keep the xyz covalent graph and use a deterministic
  graph-canonical fallback order.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


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


def open_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


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
        atoms.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))
    if len(atoms) != natoms:
        raise ValueError(f"xyz_count_mismatch:{len(atoms)}!={natoms}")
    return atoms


def bond_threshold(symbol_a: str, symbol_b: str) -> float | None:
    r1 = ELEMENT_COVALENT_RADII.get(symbol_a)
    r2 = ELEMENT_COVALENT_RADII.get(symbol_b)
    if r1 is None or r2 is None:
        return None
    ref = (r1 + r2) * 1.15
    return ref * (1.15 if symbol_a == "H" or symbol_b == "H" else 1.10)


def xyz_graph(atoms: list[tuple[str, tuple[float, float, float]]]) -> nx.Graph:
    graph = nx.Graph()
    for idx, (symbol, _) in enumerate(atoms):
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"unknown_element:{symbol}")
        graph.add_node(idx, atomic_num=ATOMIC_NUMBERS[symbol], symbol=symbol)
    for i, (symbol_i, coords_i) in enumerate(atoms):
        for j, (symbol_j, coords_j) in enumerate(atoms[:i]):
            threshold = bond_threshold(symbol_i, symbol_j)
            if threshold is None:
                continue
            distance = math.dist(coords_i, coords_j)
            if distance <= threshold:
                graph.add_edge(i, j, distance=distance)
    return graph


def rdkit_graph(mol: Chem.Mol) -> nx.Graph:
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return graph


def formula_from_symbols(symbols: list[str]) -> str:
    counts = Counter(symbols)
    ordered = []
    for symbol in ("C", "H"):
        if counts.get(symbol):
            ordered.append((symbol, counts.pop(symbol)))
    for symbol in sorted(counts):
        ordered.append((symbol, counts[symbol]))
    return "".join(symbol if count == 1 else f"{symbol}{count}" for symbol, count in ordered)


def graph_match(source: nx.Graph, target: nx.Graph) -> dict[int, int] | None:
    """Return mapping source_node -> target_node if graphs are isomorphic."""
    node_match = nx.algorithms.isomorphism.categorical_node_match("atomic_num", None)
    matcher = nx.algorithms.isomorphism.GraphMatcher(source, target, node_match=node_match)
    try:
        return next(matcher.isomorphisms_iter())
    except StopIteration:
        return None


def wl_fallback_order(graph: nx.Graph) -> list[int]:
    labels = {
        idx: f"{graph.nodes[idx]['atomic_num']}:{graph.degree(idx)}"
        for idx in graph.nodes
    }
    for _ in range(8):
        new_labels = {}
        for idx in graph.nodes:
            neigh = sorted(labels[n] for n in graph.neighbors(idx))
            payload = labels[idx] + "|" + "|".join(neigh)
            new_labels[idx] = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        labels = new_labels
    return sorted(graph.nodes, key=lambda idx: (labels[idx], graph.nodes[idx]["atomic_num"], graph.degree(idx), idx))


def heavy_first_order_from_rdkit(
    xyz: nx.Graph,
    mol: Chem.Mol,
    xyz_heavy_to_rdkit: dict[int, int],
) -> list[int]:
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    heavy_order = sorted(xyz_heavy_to_rdkit, key=lambda idx: (ranks[xyz_heavy_to_rdkit[idx]], idx))

    attached_h = defaultdict(list)
    for idx in xyz.nodes:
        if xyz.nodes[idx]["atomic_num"] != 1:
            continue
        heavy_neighbors = [n for n in xyz.neighbors(idx) if xyz.nodes[n]["atomic_num"] != 1]
        if heavy_neighbors:
            attached_h[heavy_neighbors[0]].append(idx)
    order = []
    used = set()
    for heavy_idx in heavy_order:
        order.append(heavy_idx)
        used.add(heavy_idx)
        for h_idx in sorted(attached_h.get(heavy_idx, [])):
            order.append(h_idx)
            used.add(h_idx)
    order.extend(idx for idx in wl_fallback_order(xyz) if idx not in used)
    return order


def full_order_from_rdkit(xyz_to_rdkit: dict[int, int], mol_h: Chem.Mol) -> list[int]:
    ranks = list(Chem.CanonicalRankAtoms(mol_h, breakTies=True))
    return sorted(xyz_to_rdkit, key=lambda idx: (ranks[xyz_to_rdkit[idx]], idx))


def transfer_rdkit_features(
    xyz: nx.Graph,
    mol: Chem.Mol | None,
    xyz_to_rdkit: dict[int, int] | None,
) -> tuple[dict[int, dict], dict[tuple[int, int], dict]]:
    atom_features = {}
    bond_features = {}
    if mol is None or xyz_to_rdkit is None:
        return atom_features, bond_features

    rdkit_to_xyz = {v: k for k, v in xyz_to_rdkit.items()}
    for xyz_idx, rd_idx in xyz_to_rdkit.items():
        atom = mol.GetAtomWithIdx(rd_idx)
        atom_features[xyz_idx] = {
            "atomic_num": atom.GetAtomicNum(),
            "formal_charge": atom.GetFormalCharge(),
            "degree": atom.GetDegree(),
            "total_valence": atom.GetTotalValence(),
            "is_aromatic": bool(atom.GetIsAromatic()),
            "hybridization": str(atom.GetHybridization()),
        }
    for bond in mol.GetBonds():
        a = rdkit_to_xyz.get(bond.GetBeginAtomIdx())
        b = rdkit_to_xyz.get(bond.GetEndAtomIdx())
        if a is None or b is None:
            continue
        key = tuple(sorted((a, b)))
        bond_features[key] = {
            "type": BOND_TYPE_TO_INT.get(bond.GetBondType(), 0),
            "type_name": str(bond.GetBondType()),
            "is_aromatic": bool(bond.GetIsAromatic()),
        }
    return atom_features, bond_features


def build_record(row: dict) -> dict:
    refcode = str(row.get("refcode_csd") or row.get("csd_refcode") or "").strip()
    smiles = str(row.get("canonical_smiles") or row.get("smiles") or "").strip()
    inchi = str(row.get("inchi") or "").strip()
    atoms = parse_xyz(str(row.get("xyz_pbe_relaxed") or ""))
    xyz = xyz_graph(atoms)

    if len(list(nx.connected_components(xyz))) != 1:
        graph_source_note = "xyz_disconnected"
    else:
        graph_source_note = "xyz_connected"

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    mol_h = None
    order = None
    atom_features = {}
    bond_features = {}
    transfer_mode = "xyz_binary_wl_order"

    if mol is not None:
        try:
            mol_h = Chem.AddHs(mol)
            full_mapping = graph_match(xyz, rdkit_graph(mol_h))
        except Exception:  # noqa: BLE001
            full_mapping = None
            mol_h = None
        if full_mapping is not None and mol_h is not None:
            order = full_order_from_rdkit(full_mapping, mol_h)
            atom_features, bond_features = transfer_rdkit_features(xyz, mol_h, full_mapping)
            transfer_mode = "rdkit_explicit_h_full_match"
        else:
            heavy_nodes = [idx for idx in xyz.nodes if xyz.nodes[idx]["atomic_num"] != 1]
            xyz_heavy = xyz.subgraph(heavy_nodes).copy()
            try:
                heavy_mapping = graph_match(xyz_heavy, rdkit_graph(mol))
            except Exception:  # noqa: BLE001
                heavy_mapping = None
            if heavy_mapping is not None:
                order = heavy_first_order_from_rdkit(xyz, mol, heavy_mapping)
                atom_features, bond_features = transfer_rdkit_features(xyz, mol, heavy_mapping)
                transfer_mode = "rdkit_heavy_match_xyz_h"

    if order is None:
        order = wl_fallback_order(xyz)

    old_to_new = {old: new for new, old in enumerate(order)}
    atom_symbols = [atoms[old][0] for old in order]
    atomic_numbers = [ATOMIC_NUMBERS[symbol] for symbol in atom_symbols]
    degrees = dict(xyz.degree())

    out_atom_features = []
    for old in order:
        out_atom_features.append(
            atom_features.get(
                old,
                {
                    "atomic_num": xyz.nodes[old]["atomic_num"],
                    "formal_charge": 0,
                    "degree": degrees[old],
                    "total_valence": 0,
                    "is_aromatic": False,
                    "hybridization": "UNKNOWN",
                },
            )
        )

    bonds = []
    for a, b, edge_data in xyz.edges(data=True):
        key = tuple(sorted((a, b)))
        transferred = bond_features.get(key, {})
        bonds.append(
            {
                "begin": old_to_new[a],
                "end": old_to_new[b],
                "type": int(transferred.get("type", 1 if xyz.nodes[a]["atomic_num"] == 1 or xyz.nodes[b]["atomic_num"] == 1 else 0)),
                "type_name": transferred.get("type_name", "SINGLE_XYZ_H" if xyz.nodes[a]["atomic_num"] == 1 or xyz.nodes[b]["atomic_num"] == 1 else "UNKNOWN_XYZ"),
                "is_aromatic": bool(transferred.get("is_aromatic", False)),
                "distance": float(edge_data.get("distance", 0.0)),
            }
        )
    bonds.sort(key=lambda bond: (bond["begin"], bond["end"]))

    return {
        "refcode_csd": refcode,
        "input_smiles": smiles,
        "canonical_smiles_no_h": Chem.MolToSmiles(mol, canonical=True) if mol is not None else smiles,
        "inchi": inchi,
        "formula": formula_from_symbols(atom_symbols),
        "num_atoms": len(atoms),
        "number_of_atoms_oe62": row.get("number_of_atoms"),
        "atom_symbols": atom_symbols,
        "atomic_numbers": atomic_numbers,
        "atom_features": out_atom_features,
        "bonds": bonds,
        "ok": True,
        "graph_source": "oe62_xyz_hybrid_v3",
        "graph_source_note": graph_source_note,
        "transfer_mode": transfer_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-refcodes", nargs="*", default=None)
    args = parser.parse_args()

    rows = load_records(args.input)
    only = set(args.only_refcodes or [])
    seen = set()
    stats = Counter()
    with open_output(args.output) as f:
        for row in rows:
            if args.limit is not None and stats["rows"] >= args.limit:
                break
            refcode = str(row.get("refcode_csd") or row.get("csd_refcode") or "").strip()
            if only and refcode not in only:
                continue
            stats["rows"] += 1
            if refcode in seen:
                stats["duplicates"] += 1
                continue
            seen.add(refcode)
            try:
                record = build_record(row)
            except Exception as exc:  # noqa: BLE001
                record = {
                    "refcode_csd": refcode,
                    "input_smiles": str(row.get("canonical_smiles") or row.get("smiles") or "").strip(),
                    "inchi": str(row.get("inchi") or "").strip(),
                    "ok": False,
                    "error": f"{type(exc).__name__}:{exc}",
                    "graph_source": "oe62_xyz_hybrid_v3",
                }
                stats["failed"] += 1
            else:
                stats["written_ok"] += 1
                stats[f"transfer_mode::{record['transfer_mode']}"] += 1
                stats[f"graph_source_note::{record['graph_source_note']}"] += 1
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    stats["written_total"] = stats["written_ok"] + stats["failed"]
    print(json.dumps(dict(stats), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
