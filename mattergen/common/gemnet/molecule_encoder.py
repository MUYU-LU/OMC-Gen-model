from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.nn import GINEConv

from mattergen.common.data.chemgraph import ChemGraph


class MolecularAtomEncoder(nn.Module):
    """OGB-style categorical atom feature encoder for explicit-H molecular graphs."""

    def __init__(
        self,
        emb_dim: int,
        feature_sizes: tuple[int, ...] = (128, 13, 12, 14, 2, 8),
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(size, emb_dim) for size in feature_sizes)
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != len(self.embeddings):
            raise ValueError(f"Expected atom features [N, {len(self.embeddings)}], got {x.shape}.")
        x = x.long()
        out = torch.zeros(x.size(0), self.embeddings[0].embedding_dim, device=x.device)
        for col, emb in enumerate(self.embeddings):
            out = out + emb(x[:, col].clamp(min=0, max=emb.num_embeddings - 1))
        return out


class MolecularBondEncoder(nn.Module):
    """Categorical bond encoder: RDKit bond type plus aromatic flag."""

    def __init__(self, emb_dim: int, feature_sizes: tuple[int, ...] = (8, 2)):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(size, emb_dim) for size in feature_sizes)
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr.dim() != 2 or edge_attr.size(1) != len(self.embeddings):
            raise ValueError(
                f"Expected bond features [E, {len(self.embeddings)}], got {edge_attr.shape}."
            )
        edge_attr = edge_attr.long()
        out = torch.zeros(edge_attr.size(0), self.embeddings[0].embedding_dim, device=edge_attr.device)
        for col, emb in enumerate(self.embeddings):
            out = out + emb(edge_attr[:, col].clamp(min=0, max=emb.num_embeddings - 1))
        return out


class MolecularGINEEncoder(nn.Module):
    """Small GINE stack producing one molecule-condition embedding per crystal atom."""

    def __init__(
        self,
        emb_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.atom_encoder = MolecularAtomEncoder(emb_dim=emb_dim)
        self.bond_encoder = MolecularBondEncoder(emb_dim=emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 2),
                nn.SiLU(),
                nn.Linear(emb_dim * 2, emb_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=emb_dim))
            self.norms.append(nn.LayerNorm(emb_dim))

    def forward(
        self,
        mol_x: torch.Tensor,
        mol_bond_edge_index: torch.Tensor,
        mol_bond_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.atom_encoder(mol_x)
        if mol_bond_edge_index.numel() == 0:
            return h

        edge_attr = self.bond_encoder(mol_bond_attr)
        for conv, norm in zip(self.convs, self.norms):
            h_update = conv(h, mol_bond_edge_index.long(), edge_attr)
            h = h + self.dropout(self.act(norm(h_update)))
        return h


class MolecularGraphConditioner(nn.Module):
    """Project molecular graph node embeddings into GemNet's atom embedding space."""

    def __init__(
        self,
        hidden_dim: int = 512,
        emb_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0,
        zero_init: bool = True,
    ):
        super().__init__()
        self.encoder = MolecularGINEEncoder(
            emb_dim=emb_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.proj = nn.Linear(emb_dim, hidden_dim)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, batch: ChemGraph) -> torch.Tensor:
        required = ("mol_x", "mol_bond_edge_index", "mol_bond_attr")
        missing = [field for field in required if not hasattr(batch, field)]
        if missing:
            raise KeyError(f"Missing molecule graph fields for conditioning: {missing}.")
        return self.proj(
            self.encoder(
                mol_x=batch.mol_x,
                mol_bond_edge_index=batch.mol_bond_edge_index,
                mol_bond_attr=batch.mol_bond_attr,
            )
        )


class MolecularSetAttentionConditioner(nn.Module):
    """Unlabelled molecule conditioner using target graph roles as a set.

    The fixed conditioner maps crystal atom i to molecule node i. That is a poor
    match for full-prior generation where atom labels can switch. This module
    instead encodes the target molecular graph into a set of atom-role embeddings,
    then lets every current crystal atom attend to same-element roles in the same
    crystal. The output shape stays [N_atoms, hidden_dim], so GemNet integration is
    unchanged.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        emb_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0,
        zero_init: bool = True,
        same_element_only: bool = True,
        use_pos_query: bool = True,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = MolecularGINEEncoder(
            emb_dim=emb_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.same_element_only = same_element_only
        self.use_pos_query = use_pos_query
        self.atom_query_embedding = nn.Embedding(128, emb_dim)
        self.pos_query_mlp = nn.Sequential(
            nn.Linear(6, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(emb_dim, hidden_dim)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, batch: ChemGraph) -> torch.Tensor:
        required = ("mol_x", "mol_bond_edge_index", "mol_bond_attr")
        missing = [field for field in required if not hasattr(batch, field)]
        if missing:
            raise KeyError(f"Missing molecule graph fields for conditioning: {missing}.")
        if batch.mol_x.size(0) != batch.atomic_numbers.size(0):
            raise ValueError(
                "MolecularSetAttentionConditioner expects one molecule-role row per "
                f"crystal atom, got mol_x={batch.mol_x.shape}, "
                f"atomic_numbers={batch.atomic_numbers.shape}."
            )

        role_h = self.encoder(
            mol_x=batch.mol_x,
            mol_bond_edge_index=batch.mol_bond_edge_index,
            mol_bond_attr=batch.mol_bond_attr,
        )
        atom_z = batch.atomic_numbers.long().clamp(min=0, max=127)
        role_z = batch.mol_x[:, 0].long().clamp(min=0, max=127)
        query_h = self.atom_query_embedding(atom_z)
        if self.use_pos_query:
            frac_pos = torch.remainder(batch.pos, 1.0)
            pos_feat = torch.cat(
                [
                    torch.sin(2.0 * math.pi * frac_pos),
                    torch.cos(2.0 * math.pi * frac_pos),
                ],
                dim=-1,
            )
            query_h = query_h + self.pos_query_mlp(pos_feat)

        q = self.q_proj(query_h)
        k = self.k_proj(role_h)
        v = self.v_proj(role_h)

        batch_idx = batch.get_batch_idx("pos")
        out = torch.zeros_like(q)
        scale = float(q.size(-1)) ** -0.5
        for graph_idx in torch.unique(batch_idx):
            node_mask = batch_idx == graph_idx
            node_idx = torch.nonzero(node_mask, as_tuple=False).flatten()
            q_g = q[node_idx]
            k_g = k[node_idx]
            v_g = v[node_idx]
            scores = q_g @ k_g.transpose(0, 1) * scale
            if self.same_element_only:
                element_mask = atom_z[node_idx].unsqueeze(1) == role_z[node_idx].unsqueeze(0)
                if not bool(element_mask.any(dim=1).all()):
                    element_mask = torch.ones_like(element_mask, dtype=torch.bool)
                scores = scores.masked_fill(~element_mask, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            out[node_idx] = self.attn_dropout(attn) @ v_g

        return self.proj(out)
