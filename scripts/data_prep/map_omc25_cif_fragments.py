#!/usr/bin/env python3
"""Map OMC25 CIF atoms to OE62 SMILES molecule atoms.

This script intentionally does not depend on RDKit. It consumes pre-built
SMILES graphs and uses pymatgen + networkx to build covalent fragments from the
periodic crystal and graph-isomorphism match them to the target molecule.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import signal
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import networkx as nx
from pymatgen.core import Structure


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


def open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def load_smiles_graphs(path: Path) -> dict[str, dict]:
    graphs: dict[str, dict] = {}
    with open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("ok"):
                graphs[rec["refcode_csd"]] = rec
    return graphs


def graph_from_smiles_record(record: dict) -> nx.Graph:
    graph = nx.Graph()
    for idx, z in enumerate(record["atomic_numbers"]):
        graph.add_node(idx, atomic_num=int(z))
    for bond in record["bonds"]:
        graph.add_edge(
            int(bond["begin"]),
            int(bond["end"]),
            bond_type=int(bond.get("type", 0)),
        )
    add_node_signatures(graph)
    return graph


def add_node_signatures(graph: nx.Graph) -> None:
    for idx in graph.nodes:
        neigh_atomic_nums = sorted(int(graph.nodes[j]["atomic_num"]) for j in graph.neighbors(idx))
        graph.nodes[idx]["env_signature"] = ",".join(map(str, neigh_atomic_nums))


def bond_threshold(symbol_a: str, symbol_b: str) -> float | None:
    r1 = ELEMENT_COVALENT_RADII.get(symbol_a)
    r2 = ELEMENT_COVALENT_RADII.get(symbol_b)
    if r1 is None or r2 is None:
        return None
    ref = (r1 + r2) * 1.15
    if symbol_a == "H" or symbol_b == "H":
        return ref * 1.15
    return ref * 1.10


def structure_symbols(structure: Structure) -> list[str]:
    return [str(site.specie.symbol) for site in structure.sites]


def covalent_graph_from_structure(
    structure: Structure, keep_edge_records: bool = False
) -> tuple[nx.Graph, list[dict]]:
    graph = nx.Graph()
    symbols = structure_symbols(structure)
    for idx, site in enumerate(structure.sites):
        graph.add_node(idx, atomic_num=int(site.specie.Z), symbol=symbols[idx])

    max_cutoff = 0.0
    for a, r1 in ELEMENT_COVALENT_RADII.items():
        for b, r2 in ELEMENT_COVALENT_RADII.items():
            multiplier = 1.15 * (1.15 if (a == "H" or b == "H") else 1.10)
            max_cutoff = max(max_cutoff, (r1 + r2) * multiplier)
    max_cutoff += 0.05

    edge_records = []
    for i, site in enumerate(structure.sites):
        for neigh in structure.get_neighbors(site, max_cutoff):
            j = int(neigh.index)
            if j <= i:
                continue
            threshold = bond_threshold(symbols[i], symbols[j])
            if threshold is None:
                continue
            distance = float(neigh.nn_distance)
            if distance <= threshold:
                graph.add_edge(i, j, distance=distance)
                if keep_edge_records:
                    edge_records.append(
                        {
                            "begin": i,
                            "end": j,
                            "distance": distance,
                            "threshold": threshold,
                            "image": list(getattr(neigh, "image", (0, 0, 0))),
                        }
                    )
    add_node_signatures(graph)
    return graph, edge_records


def formula_from_atomic_numbers(atomic_numbers: Iterable[int]) -> dict[int, int]:
    return dict(Counter(int(z) for z in atomic_numbers))


def map_components(
    crystal_graph: nx.Graph,
    target_graph: nx.Graph,
    keep_component_maps: bool = False,
    match_timeout_sec: int = 20,
) -> tuple[dict | None, str]:
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        ["atomic_num", "env_signature"], [None, None]
    )
    components = [sorted(c) for c in nx.connected_components(crystal_graph)]
    components.sort(key=lambda c: (len(c), c[0]))

    if not components:
        return None, "no_components"

    target_formula = formula_from_atomic_numbers(
        target_graph.nodes[idx]["atomic_num"] for idx in target_graph.nodes
    )
    mol_atom_idx = [-1] * crystal_graph.number_of_nodes()
    mol_id = [-1] * crystal_graph.number_of_nodes()
    component_maps: list[dict] = []

    for comp_id, nodes in enumerate(components):
        sub = crystal_graph.subgraph(nodes).copy()
        add_node_signatures(sub)
        sub_formula = formula_from_atomic_numbers(sub.nodes[idx]["atomic_num"] for idx in sub.nodes)
        if sub_formula != target_formula:
            return None, f"component_formula_mismatch:{sub_formula}!={target_formula}"
        matcher = nx.algorithms.isomorphism.GraphMatcher(sub, target_graph, node_match=node_match)
        if match_timeout_sec > 0:
            def _raise_timeout(signum, frame):  # noqa: ARG001
                raise TimeoutError

            old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
            signal.alarm(match_timeout_sec)
        try:
            mapping = next(matcher.isomorphisms_iter())
        except StopIteration:
            return None, "component_not_isomorphic"
        except TimeoutError:
            return None, f"component_match_timeout_{match_timeout_sec}s"
        finally:
            if match_timeout_sec > 0:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        # mapping is crystal_atom_idx -> target_mol_atom_idx.
        for crystal_idx, target_idx in mapping.items():
            mol_id[crystal_idx] = comp_id
            mol_atom_idx[crystal_idx] = int(target_idx)
        component_maps.append({str(k): int(v) for k, v in sorted(mapping.items())})

    if any(idx < 0 for idx in mol_atom_idx):
        return None, "incomplete_mapping"

    result = {
        "num_molecules": len(components),
        "component_sizes": [len(c) for c in components],
        "mol_id": mol_id,
        "mol_atom_idx": mol_atom_idx,
    }

    if keep_component_maps:
        result["component_maps"] = component_maps

    return result, "ok"


def assign_smiles_bonds_to_crystal(record: dict, mapping_result: dict) -> list[dict]:
    target_bonds = record["bonds"]
    mol_id = mapping_result["mol_id"]
    mol_atom_idx = mapping_result["mol_atom_idx"]

    by_molecule: dict[int, dict[int, int]] = {}
    for crystal_idx, (m_id, m_atom) in enumerate(zip(mol_id, mol_atom_idx)):
        by_molecule.setdefault(int(m_id), {})[int(m_atom)] = int(crystal_idx)

    crystal_bonds = []
    for m_id, atom_to_crystal in sorted(by_molecule.items()):
        for bond in target_bonds:
            a = atom_to_crystal[int(bond["begin"])]
            b = atom_to_crystal[int(bond["end"])]
            crystal_bonds.append(
                {
                    "begin": a,
                    "end": b,
                    "mol_id": m_id,
                    "mol_atom_begin": int(bond["begin"]),
                    "mol_atom_end": int(bond["end"]),
                    "type": int(bond.get("type", 0)),
                }
            )
    return crystal_bonds


def process_row(
    row: dict,
    smiles_graphs: dict[str, dict],
    verbose_records: bool = False,
    match_timeout_sec: int = 20,
) -> dict:
    material_id = row.get("material_id", "")
    refcode = (row.get("csd_refcode") or material_id.split("|")[0]).strip()
    result = {"material_id": material_id, "csd_refcode": refcode, "success": False}

    target_record = smiles_graphs.get(refcode)
    if target_record is None:
        result["reason"] = "missing_smiles_graph"
        return result

    try:
        structure = Structure.from_str(row["cif"], fmt="cif")
    except Exception as exc:  # noqa: BLE001
        result["reason"] = f"cif_parse_failed:{type(exc).__name__}"
        return result

    target_graph = graph_from_smiles_record(target_record)
    crystal_graph, covalent_edges = covalent_graph_from_structure(
        structure, keep_edge_records=verbose_records
    )
    mapping_result, reason = map_components(
        crystal_graph,
        target_graph,
        keep_component_maps=verbose_records,
        match_timeout_sec=match_timeout_sec,
    )

    result.update(
        {
            "num_atoms": int(len(structure)),
            "target_num_atoms": int(target_record["num_atoms"]),
            "target_formula": target_record["formula"],
            "crystal_num_edges": int(crystal_graph.number_of_edges()),
            "crystal_component_sizes": [len(c) for c in nx.connected_components(crystal_graph)],
            "reason": reason,
        }
    )

    if mapping_result is None:
        return result

    result["success"] = True
    result["mapping"] = mapping_result
    result["crystal_bonds"] = assign_smiles_bonds_to_crystal(target_record, mapping_result)
    if verbose_records:
        result["covalent_edges_by_distance"] = covalent_edges
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--smiles-graphs", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument("--heartbeat-every", type=int, default=1000)
    parser.add_argument("--match-timeout-sec", type=int, default=20)
    parser.add_argument("--verbose-records", action="store_true")
    args = parser.parse_args()

    smiles_graphs = load_smiles_graphs(args.smiles_graphs)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    reason_stats = Counter()

    with args.input_csv.open(newline="", encoding="utf-8") as f_in, open_text(args.output_jsonl, "wt") as f_out:
        reader = csv.DictReader(f_in)
        for idx, row in enumerate(reader, start=1):
            if args.limit is not None and idx > args.limit:
                break
            result = process_row(
                row,
                smiles_graphs,
                verbose_records=args.verbose_records,
                match_timeout_sec=args.match_timeout_sec,
            )
            stats["rows"] += 1
            stats["success" if result.get("success") else "failed"] += 1
            reason_stats[result.get("reason", "unknown")] += 1
            f_out.write(json.dumps(result, separators=(",", ":")) + "\n")
            if args.heartbeat_every and idx % args.heartbeat_every == 0:
                print(
                    f"heartbeat row={idx} material_id={result.get('material_id')} "
                    f"success={result.get('success')} reason={result.get('reason')}",
                    file=sys.stderr,
                    flush=True,
                )
            if args.report_every and idx % args.report_every == 0:
                success = stats["success"]
                rows = stats["rows"]
                print(
                    json.dumps(
                        {
                            "rows": rows,
                            "success": success,
                            "success_rate": success / max(rows, 1),
                            "top_reasons": reason_stats.most_common(8),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    print(
        json.dumps(
            {
                "rows": stats["rows"],
                "success": stats["success"],
                "failed": stats["failed"],
                "success_rate": stats["success"] / max(stats["rows"], 1),
                "reasons": reason_stats.most_common(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
