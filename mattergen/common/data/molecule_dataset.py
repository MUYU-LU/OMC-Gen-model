from __future__ import annotations

import gzip
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.dataset import CrystalDataset, DatasetTransform
from mattergen.common.data.transform import Transform
from mattergen.common.data.types import PropertySourceId


HYBRIDIZATION_TO_INDEX = {
    "UNSPECIFIED": 0,
    "S": 1,
    "SP": 2,
    "SP2": 3,
    "SP3": 4,
    "SP3D": 5,
    "SP3D2": 6,
}


def _read_jsonl_gz(path: str | Path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _as_str(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _charge_index(charge: int) -> int:
    if -5 <= charge <= 5:
        return charge + 6
    return 12


def _atom_feature_row(atom_feature: dict[str, Any]) -> list[int]:
    return [
        int(atom_feature.get("atomic_num", 0)),
        _charge_index(int(atom_feature.get("formal_charge", 0))),
        min(max(int(atom_feature.get("degree", 0)), 0), 11),
        min(max(int(atom_feature.get("total_valence", 0)), 0), 13),
        int(bool(atom_feature.get("is_aromatic", False))),
        HYBRIDIZATION_TO_INDEX.get(str(atom_feature.get("hybridization", "UNSPECIFIED")), 7),
    ]


def _bond_feature_row(bond_type: int) -> list[int]:
    bond_type = min(max(int(bond_type), 0), 7)
    return [bond_type, int(bond_type == 4)]


def _pbc_bond_distances(
    frac_pos: torch.Tensor,
    cell: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((0,), dtype=frac_pos.dtype)

    src, dst = edge_index
    dfrac = frac_pos[src] - frac_pos[dst]
    dfrac = dfrac - torch.round(dfrac)
    cart = dfrac @ cell.squeeze(0)
    return torch.linalg.norm(cart, dim=-1)


class MoleculeMappedCrystalDataset(Dataset):
    """CrystalDataset wrapper that adds explicit-H molecular graph conditioning fields."""

    def __init__(
        self,
        base_dataset: CrystalDataset,
        mapping_jsonl: str | Path,
        smiles_graph_jsonl: str | Path,
        strict_transfer_mode: str | None = "rdkit_explicit_h_full_match",
        heavy_only: bool = False,
    ):
        self.base_dataset = base_dataset
        self.mapping_jsonl = str(mapping_jsonl)
        self.smiles_graph_jsonl = str(smiles_graph_jsonl)
        self.strict_transfer_mode = strict_transfer_mode
        self.heavy_only = heavy_only

        self.graphs_by_refcode = self._load_smiles_graphs(
            smiles_graph_jsonl=smiles_graph_jsonl,
            strict_transfer_mode=strict_transfer_mode,
        )
        self.mapping_by_material_id = self._load_mappings(mapping_jsonl=mapping_jsonl)
        self.indices = self._matched_base_indices()

        if len(self.indices) == 0:
            raise ValueError(
                "MoleculeMappedCrystalDataset found zero matched structures. "
                f"mapping_jsonl={mapping_jsonl}, smiles_graph_jsonl={smiles_graph_jsonl}, "
                f"strict_transfer_mode={strict_transfer_mode}"
            )

    @classmethod
    def from_cache_path(
        cls,
        cache_path: str,
        mapping_jsonl: str,
        smiles_graph_jsonl: str,
        transforms: list[Transform] | None = None,
        properties: list[PropertySourceId] | None = None,
        dataset_transforms: list[DatasetTransform] | None = None,
        strict_transfer_mode: str | None = "rdkit_explicit_h_full_match",
        heavy_only: bool = False,
    ) -> "MoleculeMappedCrystalDataset":
        base_dataset = CrystalDataset.from_cache_path(
            cache_path=cache_path,
            transforms=transforms,
            properties=properties,
            dataset_transforms=dataset_transforms,
        )
        return cls(
            base_dataset=base_dataset,
            mapping_jsonl=mapping_jsonl,
            smiles_graph_jsonl=smiles_graph_jsonl,
            strict_transfer_mode=strict_transfer_mode,
            heavy_only=heavy_only,
        )

    @staticmethod
    def _load_smiles_graphs(
        smiles_graph_jsonl: str | Path,
        strict_transfer_mode: str | None,
    ) -> dict[str, dict[str, Any]]:
        graphs = {}
        for record in _read_jsonl_gz(smiles_graph_jsonl):
            refcode = record.get("refcode_csd")
            if refcode is None:
                continue
            if strict_transfer_mode is not None:
                if record.get("transfer_mode") != strict_transfer_mode:
                    continue
            if not record.get("ok", True):
                continue
            graphs[str(refcode)] = record
        return graphs

    def _load_mappings(self, mapping_jsonl: str | Path) -> dict[str, dict[str, Any]]:
        mappings = {}
        for record in _read_jsonl_gz(mapping_jsonl):
            if not record.get("success", False):
                continue
            refcode = str(record.get("csd_refcode", ""))
            if refcode not in self.graphs_by_refcode:
                continue
            mappings[str(record["material_id"])] = record
        return mappings

    def _matched_base_indices(self) -> list[int]:
        indices = []
        for index, structure_id in enumerate(self.base_dataset.structure_id):
            material_id = _as_str(structure_id)
            if material_id in self.mapping_by_material_id:
                indices.append(index)
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> ChemGraph:
        base_index = self.indices[index]
        data = self.base_dataset[base_index]
        material_id = _as_str(self.base_dataset.structure_id[base_index])
        mapping_record = self.mapping_by_material_id[material_id]
        graph_record = self.graphs_by_refcode[mapping_record["csd_refcode"]]

        mapping = mapping_record["mapping"]
        mol_atom_id = torch.tensor(mapping["mol_atom_idx"], dtype=torch.long)
        mol_copy_id = torch.tensor(mapping["mol_id"], dtype=torch.long)
        if mol_atom_id.numel() != data.num_nodes:
            raise ValueError(
                f"Molecule mapping atom count mismatch for {material_id}: "
                f"{mol_atom_id.numel()} != {data.num_nodes}."
            )

        old_to_new = None
        keep_mask = None
        if self.heavy_only:
            keep_mask = data.atomic_numbers != 1
            num_heavy = int(keep_mask.sum().item())
            if num_heavy == 0:
                raise ValueError(f"Heavy-only sample has zero heavy atoms for {material_id}.")
            old_to_new = torch.full((int(data.num_nodes),), -1, dtype=torch.long)
            old_to_new[keep_mask] = torch.arange(num_heavy, dtype=torch.long)
            data = data.replace(
                pos=data.pos[keep_mask],
                atomic_numbers=data.atomic_numbers[keep_mask],
                num_atoms=torch.tensor(num_heavy, dtype=data.num_atoms.dtype),
                num_nodes=num_heavy,
            )
            mol_atom_id = mol_atom_id[keep_mask]
            mol_copy_id = mol_copy_id[keep_mask]

        atom_features = graph_record["atom_features"]
        mol_x = torch.tensor(
            [_atom_feature_row(atom_features[int(atom_idx)]) for atom_idx in mol_atom_id],
            dtype=torch.long,
        )

        directed_edges: list[list[int]] = []
        directed_attr: list[list[int]] = []
        for bond in mapping_record.get("crystal_bonds", []):
            begin = int(bond["begin"])
            end = int(bond["end"])
            if self.heavy_only:
                assert keep_mask is not None and old_to_new is not None
                if not (bool(keep_mask[begin]) and bool(keep_mask[end])):
                    continue
                begin = int(old_to_new[begin])
                end = int(old_to_new[end])
            attr = _bond_feature_row(int(bond.get("type", 0)))
            directed_edges.extend([[begin, end], [end, begin]])
            directed_attr.extend([attr, attr])

        if directed_edges:
            mol_bond_edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
            mol_bond_attr = torch.tensor(directed_attr, dtype=torch.long)
            mol_bond_d0 = _pbc_bond_distances(
                frac_pos=data.pos,
                cell=data.cell,
                edge_index=mol_bond_edge_index,
            )
        else:
            mol_bond_edge_index = torch.empty((2, 0), dtype=torch.long)
            mol_bond_attr = torch.empty((0, 2), dtype=torch.long)
            mol_bond_d0 = torch.empty((0,), dtype=data.pos.dtype)

        return data.replace(
            mol_x=mol_x,
            mol_bond_edge_index=mol_bond_edge_index,
            mol_bond_attr=mol_bond_attr,
            mol_bond_d0=mol_bond_d0,
            mol_atom_id=mol_atom_id,
            mol_copy_id=mol_copy_id,
            mol_num_molecules=torch.tensor([int(mapping["num_molecules"])], dtype=torch.long),
        )

    def subset(self, indices: Sequence[int]) -> "MoleculeMappedCrystalDataset":
        out = self.__class__.__new__(self.__class__)
        out.base_dataset = self.base_dataset
        out.mapping_jsonl = self.mapping_jsonl
        out.smiles_graph_jsonl = self.smiles_graph_jsonl
        out.strict_transfer_mode = self.strict_transfer_mode
        out.heavy_only = self.heavy_only
        out.graphs_by_refcode = self.graphs_by_refcode
        out.mapping_by_material_id = self.mapping_by_material_id
        out.indices = [self.indices[i] for i in indices]
        return out
