# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_scatter import scatter

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.losses import SummedFieldLoss, denoising_score_matching
from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.training.field_loss import FieldLoss, d3pm_loss
from mattergen.diffusion.wrapped.wrapped_normal_loss import wrapped_normal_loss


_COVALENT_RADII = {
    1: 0.31,
    2: 0.28,
    3: 1.28,
    4: 0.96,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    10: 0.58,
    11: 1.66,
    12: 1.41,
    13: 1.21,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    18: 1.06,
    35: 1.20,
    53: 1.39,
}


def _covalent_radii(atomic_numbers: torch.Tensor) -> torch.Tensor:
    table = atomic_numbers.new_full((119,), 1.0, dtype=torch.float32)
    for atomic_number, radius in _COVALENT_RADII.items():
        table[atomic_number] = radius
    return table[atomic_numbers.long().clamp(min=0, max=118)]


def _covalent_cutoff(
    atomic_numbers_i: torch.Tensor,
    atomic_numbers_j: torch.Tensor,
    *,
    scale: float = 0.9,
) -> torch.Tensor:
    radii_i = _covalent_radii(atomic_numbers_i)
    radii_j = _covalent_radii(atomic_numbers_j)
    base = (radii_i + radii_j) * 1.15
    h_factor = torch.where((atomic_numbers_i == 1) | (atomic_numbers_j == 1), 1.15, 1.10)
    return scale * base * h_factor


def _estimate_clean_pos(
    *,
    multi_corruption: MultiCorruption,
    noisy_batch: BatchedData,
    score_model_output: BatchedData,
    t: torch.Tensor,
) -> torch.Tensor:
    pos_sde = multi_corruption.corruptions["pos"]
    pos_batch_idx = noisy_batch.get_batch_idx("pos")
    _, pos_std = pos_sde.marginal_prob(
        x=torch.ones_like(noisy_batch["pos"]),
        t=t,
        batch_idx=pos_batch_idx,
        batch=noisy_batch,
    )
    clean_pos = noisy_batch["pos"] + pos_std * score_model_output["pos"]
    return torch.remainder(clean_pos, 1.0)


def _estimate_clean_cell(
    *,
    multi_corruption: MultiCorruption,
    noisy_batch: BatchedData,
    score_model_output: BatchedData,
    t: torch.Tensor,
) -> torch.Tensor:
    cell_sde = multi_corruption.corruptions["cell"]
    mean_coeff, cell_std = cell_sde.mean_coeff_and_std(
        x=torch.ones_like(noisy_batch["cell"]),
        t=t,
        batch_idx=None,
        batch=noisy_batch,
    )
    if hasattr(cell_sde, "get_limit_mean"):
        limit_mean = cell_sde.get_limit_mean(x=noisy_batch["cell"], batch=noisy_batch)
    else:
        limit_mean = torch.zeros_like(noisy_batch["cell"])

    # score_model_output is score * std for the cell SDE.
    numerator = noisy_batch["cell"] + cell_std * score_model_output["cell"]
    numerator = numerator - (1.0 - mean_coeff) * limit_mean
    return numerator / mean_coeff.clamp_min(1e-6)


def _pbc_bond_distances(
    *,
    frac_pos: torch.Tensor,
    cell: torch.Tensor,
    edge_index: torch.Tensor,
    edge_batch_idx: torch.Tensor,
) -> torch.Tensor:
    src, dst = edge_index
    dfrac = frac_pos[src] - frac_pos[dst]
    dfrac = dfrac - torch.round(dfrac.detach())
    cart = torch.bmm(dfrac.unsqueeze(1), cell[edge_batch_idx]).squeeze(1)
    return torch.linalg.norm(cart, dim=-1)


def bond_length_loss_from_x0_hat(
    *,
    multi_corruption: MultiCorruption,
    batch: BatchedData,
    noisy_batch: BatchedData,
    score_model_output: BatchedData,
    t: torch.Tensor,
    loss_type: Literal["relative_mse", "relative_huber"] = "relative_mse",
    huber_beta: float = 0.2,
    time_gate_center: Optional[float] = None,
    time_gate_width: float = 0.05,
    detach_cell: bool = False,
) -> torch.Tensor:
    if "mol_bond_edge_index" not in batch or "mol_bond_d0" not in batch:
        return torch.zeros(batch.get_batch_size(), device=noisy_batch["pos"].device)

    edge_index = batch["mol_bond_edge_index"]
    if edge_index.numel() == 0:
        return torch.zeros(batch.get_batch_size(), device=noisy_batch["pos"].device)

    node_batch_idx = batch.get_batch_idx("pos")
    edge_batch_idx = node_batch_idx[edge_index[0]]
    clean_pos = _estimate_clean_pos(
        multi_corruption=multi_corruption,
        noisy_batch=noisy_batch,
        score_model_output=score_model_output,
        t=t,
    )
    clean_cell = _estimate_clean_cell(
        multi_corruption=multi_corruption,
        noisy_batch=noisy_batch,
        score_model_output=score_model_output,
        t=t,
    )
    if detach_cell:
        clean_cell = clean_cell.detach()
    pred_dist = _pbc_bond_distances(
        frac_pos=clean_pos,
        cell=clean_cell,
        edge_index=edge_index,
        edge_batch_idx=edge_batch_idx,
    )
    ref_dist = batch["mol_bond_d0"].to(pred_dist.device).clamp_min(1e-6)
    rel_error = (pred_dist - ref_dist) / ref_dist
    if loss_type == "relative_mse":
        bond_loss = rel_error.square()
    elif loss_type == "relative_huber":
        bond_loss = F.huber_loss(
            rel_error,
            torch.zeros_like(rel_error),
            reduction="none",
            delta=huber_beta,
        )
    else:
        raise ValueError(f"Unknown bond loss type {loss_type}.")

    if time_gate_center is not None:
        gate = torch.sigmoid((time_gate_center - t) / time_gate_width)
        bond_loss = bond_loss * gate[edge_batch_idx]

    return scatter(
        src=bond_loss,
        index=edge_batch_idx,
        dim=0,
        dim_size=batch.get_batch_size(),
        reduce="mean",
    )


def _time_window_gate(
    t: torch.Tensor,
    *,
    time_gate_min: Optional[float],
    time_gate_max: Optional[float],
    time_gate_width: float,
) -> torch.Tensor:
    gate = torch.ones_like(t)
    if time_gate_min is not None:
        gate = gate * torch.sigmoid((t - time_gate_min) / time_gate_width)
    if time_gate_max is not None:
        gate = gate * torch.sigmoid((time_gate_max - t) / time_gate_width)
    return gate


def _window_scaled_loss(
    loss_per_sample: torch.Tensor,
    t: torch.Tensor,
    *,
    scale: float,
    time_gate_min: Optional[float],
    time_gate_max: Optional[float],
    time_gate_width: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Down-weight a per-structure loss inside a time window without changing defaults."""
    if scale >= 1.0 or (time_gate_min is None and time_gate_max is None):
        return loss_per_sample, torch.ones_like(loss_per_sample)
    gate = _time_window_gate(
        t,
        time_gate_min=time_gate_min,
        time_gate_max=time_gate_max,
        time_gate_width=time_gate_width,
    )
    multiplier = 1.0 - (1.0 - float(scale)) * gate
    return loss_per_sample * multiplier, multiplier


def _default_valence_cap(atomic_numbers: torch.Tensor) -> torch.Tensor:
    caps = atomic_numbers.new_full((119,), 4, dtype=torch.long)
    for atomic_number, cap in {
        1: 1,
        6: 4,
        7: 4,
        8: 2,
        9: 1,
        15: 5,
        16: 6,
        17: 1,
        35: 1,
        53: 1,
    }.items():
        caps[atomic_number] = cap
    return caps[atomic_numbers.long().clamp(min=0, max=118)]


def _select_assignment_bond_pairs_for_sample(
    *,
    clean_pos: torch.Tensor,
    clean_cell: torch.Tensor,
    node_idx: torch.Tensor,
    node_batch_idx: torch.Tensor,
    atomic_numbers: torch.Tensor,
    bond_edge_index: torch.Tensor,
    bond_d0: torch.Tensor,
    bond_attr: Optional[torch.Tensor],
    batch_idx: int,
    distance_lower_factor: Optional[float],
    distance_upper_factor: Optional[float],
    use_valence_cap: bool,
) -> dict:
    """Greedily match target bond types to plausible unlabelled atom pairs."""
    n_atoms = int(node_idx.numel())
    empty = {
        "num_targets": 0,
        "selected_pair_indices": [],
        "selected_ref_distances": [],
        "selected_local_pairs": [],
        "usage": torch.zeros(n_atoms, device=clean_pos.device, dtype=torch.long),
        "pair_i": torch.empty((0,), device=clean_pos.device, dtype=torch.long),
        "pair_j": torch.empty((0,), device=clean_pos.device, dtype=torch.long),
        "pair_distances": torch.empty((0,), device=clean_pos.device, dtype=clean_pos.dtype),
        "fixed_edge_keys": set(),
    }
    if n_atoms < 2:
        return empty

    local_of_global = torch.full(
        (int(node_batch_idx.numel()),),
        -1,
        device=clean_pos.device,
        dtype=torch.long,
    )
    local_of_global[node_idx] = torch.arange(n_atoms, device=clean_pos.device)

    edge_mask = node_batch_idx[bond_edge_index[0]] == batch_idx
    edge_cols = torch.nonzero(edge_mask, as_tuple=False).flatten()
    if edge_cols.numel() == 0:
        return empty

    local_edges = local_of_global[bond_edge_index[:, edge_cols]]
    seen_edges = {}
    targets = []
    for edge_pos, (src_local, dst_local) in enumerate(local_edges.t().tolist()):
        if src_local < 0 or dst_local < 0:
            continue
        key = (src_local, dst_local) if src_local <= dst_local else (dst_local, src_local)
        if key not in seen_edges:
            seen_edges[key] = True
            edge_col = int(edge_cols[edge_pos])
            z_i = int(atomic_numbers[node_idx[key[0]]])
            z_j = int(atomic_numbers[node_idx[key[1]]])
            bond_type = int(bond_attr[edge_col, 0]) if bond_attr is not None and bond_attr.numel() else 0
            targets.append(
                {
                    "edge": key,
                    "d0_col": edge_col,
                    "group_key": (min(z_i, z_j), max(z_i, z_j), bond_type),
                }
            )
    if not targets:
        return empty

    pair_i, pair_j = torch.triu_indices(n_atoms, n_atoms, offset=1, device=clean_pos.device)
    global_i = node_idx[pair_i]
    global_j = node_idx[pair_j]
    pair_edge_index = torch.stack([global_i, global_j], dim=0)
    pair_batch_idx = torch.full_like(global_i, batch_idx)
    pair_distances = _pbc_bond_distances(
        frac_pos=clean_pos,
        cell=clean_cell,
        edge_index=pair_edge_index,
        edge_batch_idx=pair_batch_idx,
    )
    pair_z_i = atomic_numbers[global_i]
    pair_z_j = atomic_numbers[global_j]

    targets_by_group: Dict[Tuple[int, int, int], list] = {}
    for target_idx, target in enumerate(targets):
        targets_by_group.setdefault(target["group_key"], []).append((target_idx, target))

    selected_pair_indices = []
    selected_ref_distances = []
    selected_local_pairs = []
    selected_target_indices = set()
    selected_pair_keys = set()
    usage = torch.zeros(n_atoms, device=clean_pos.device, dtype=torch.long)
    valence_cap = _default_valence_cap(atomic_numbers[node_idx]) if use_valence_cap else None

    for group_key, group_targets in targets_by_group.items():
        z_a, z_b, _bond_type = group_key
        if z_a == z_b:
            candidate_mask = (pair_z_i == z_a) & (pair_z_j == z_b)
        else:
            candidate_mask = (
                ((pair_z_i == z_a) & (pair_z_j == z_b))
                | ((pair_z_i == z_b) & (pair_z_j == z_a))
            )
        if not torch.any(candidate_mask):
            continue

        candidate_pair_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        candidate_rows = []
        for target_idx, target in group_targets:
            ref_dist = bond_d0[target["d0_col"]].clamp_min(1e-6)
            distances = pair_distances[candidate_pair_indices]
            window_mask = torch.ones_like(distances, dtype=torch.bool)
            if distance_lower_factor is not None:
                window_mask = window_mask & (distances >= float(distance_lower_factor) * ref_dist)
            if distance_upper_factor is not None:
                window_mask = window_mask & (distances <= float(distance_upper_factor) * ref_dist)
            if not torch.any(window_mask):
                continue
            pair_subset = candidate_pair_indices[window_mask]
            costs = torch.abs(pair_distances[pair_subset].detach() - ref_dist.detach()) / ref_dist.detach()
            for cost, pair_idx in zip(costs.tolist(), pair_subset.tolist()):
                candidate_rows.append((float(cost), int(target_idx), int(pair_idx), ref_dist))

        candidate_rows.sort(key=lambda item: item[0])
        for _cost, target_idx, pair_idx, ref_dist in candidate_rows:
            if target_idx in selected_target_indices:
                continue
            local_i = int(pair_i[pair_idx])
            local_j = int(pair_j[pair_idx])
            pair_key = (local_i, local_j) if local_i <= local_j else (local_j, local_i)
            if pair_key in selected_pair_keys:
                continue
            if use_valence_cap and valence_cap is not None:
                if usage[local_i] >= valence_cap[local_i] or usage[local_j] >= valence_cap[local_j]:
                    continue
            selected_target_indices.add(target_idx)
            selected_pair_keys.add(pair_key)
            selected_pair_indices.append(pair_idx)
            selected_ref_distances.append(ref_dist)
            selected_local_pairs.append(pair_key)
            usage[local_i] += 1
            usage[local_j] += 1
            if len(selected_target_indices) == len(targets):
                break

    return {
        "num_targets": len(targets),
        "selected_pair_indices": selected_pair_indices,
        "selected_ref_distances": selected_ref_distances,
        "selected_local_pairs": selected_local_pairs,
        "usage": usage,
        "pair_i": pair_i,
        "pair_j": pair_j,
        "pair_distances": pair_distances,
        "fixed_edge_keys": {target["edge"] for target in targets},
    }


def assignment_bond_length_loss_from_x0_hat(
    *,
    multi_corruption: MultiCorruption,
    batch: BatchedData,
    noisy_batch: BatchedData,
    score_model_output: BatchedData,
    t: torch.Tensor,
    loss_type: Literal["relative_mse", "relative_huber"] = "relative_huber",
    huber_beta: float = 0.2,
    time_gate_min: Optional[float] = None,
    time_gate_max: Optional[float] = None,
    time_gate_width: float = 0.08,
    detach_cell: bool = False,
    distance_lower_factor: Optional[float] = 0.5,
    distance_upper_factor: Optional[float] = 1.6,
    use_valence_cap: bool = True,
    return_metrics: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Unlabelled bond loss with greedy element/bond-type assignment.

    This is a diagnostic topology loss for full-prior molecule CSP: it does not require the
    fixed crystal atom indices to form each target bond at medium noise.
    """
    if "mol_bond_edge_index" not in batch or "mol_bond_d0" not in batch:
        empty = torch.zeros(batch.get_batch_size(), device=noisy_batch["pos"].device)
        if return_metrics:
            return empty, {}
        return empty

    bond_edge_index = batch["mol_bond_edge_index"]
    if bond_edge_index.numel() == 0:
        empty = torch.zeros(batch.get_batch_size(), device=noisy_batch["pos"].device)
        if return_metrics:
            return empty, {}
        return empty

    clean_pos = _estimate_clean_pos(
        multi_corruption=multi_corruption,
        noisy_batch=noisy_batch,
        score_model_output=score_model_output,
        t=t,
    )
    clean_cell = _estimate_clean_cell(
        multi_corruption=multi_corruption,
        noisy_batch=noisy_batch,
        score_model_output=score_model_output,
        t=t,
    )
    if detach_cell:
        clean_cell = clean_cell.detach()

    node_batch_idx = batch.get_batch_idx("pos")
    atomic_numbers = batch["atomic_numbers"].long()
    bond_d0 = batch["mol_bond_d0"].to(clean_pos.device)
    bond_attr = batch["mol_bond_attr"] if "mol_bond_attr" in batch else None
    losses = []
    coverage_values = []
    unique_atom_ratio_values = []
    max_usage_values = []
    p95_usage_values = []
    nonfixed_pair_rate_values = []
    abs_error_values = []
    rel_error_values = []
    for batch_idx in range(batch.get_batch_size()):
        node_idx = torch.nonzero(node_batch_idx == batch_idx, as_tuple=False).flatten()
        n_atoms = int(node_idx.numel())
        if n_atoms < 2:
            losses.append(clean_pos.new_zeros(()))
            coverage_values.append(clean_pos.new_zeros(()))
            unique_atom_ratio_values.append(clean_pos.new_zeros(()))
            max_usage_values.append(clean_pos.new_zeros(()))
            p95_usage_values.append(clean_pos.new_zeros(()))
            nonfixed_pair_rate_values.append(clean_pos.new_zeros(()))
            abs_error_values.append(clean_pos.new_zeros(()))
            rel_error_values.append(clean_pos.new_zeros(()))
            continue

        local_of_global = torch.full(
            (int(node_batch_idx.numel()),),
            -1,
            device=clean_pos.device,
            dtype=torch.long,
        )
        local_of_global[node_idx] = torch.arange(n_atoms, device=clean_pos.device)

        edge_mask = node_batch_idx[bond_edge_index[0]] == batch_idx
        edge_cols = torch.nonzero(edge_mask, as_tuple=False).flatten()
        if edge_cols.numel() == 0:
            losses.append(clean_pos.new_zeros(()))
            coverage_values.append(clean_pos.new_zeros(()))
            unique_atom_ratio_values.append(clean_pos.new_zeros(()))
            max_usage_values.append(clean_pos.new_zeros(()))
            p95_usage_values.append(clean_pos.new_zeros(()))
            nonfixed_pair_rate_values.append(clean_pos.new_zeros(()))
            abs_error_values.append(clean_pos.new_zeros(()))
            rel_error_values.append(clean_pos.new_zeros(()))
            continue

        local_edges = local_of_global[bond_edge_index[:, edge_cols]]
        seen_edges = {}
        targets = []
        for edge_pos, (src_local, dst_local) in enumerate(local_edges.t().tolist()):
            if src_local < 0 or dst_local < 0:
                continue
            key = (src_local, dst_local) if src_local <= dst_local else (dst_local, src_local)
            if key not in seen_edges:
                seen_edges[key] = True
                edge_col = int(edge_cols[edge_pos])
                z_i = int(atomic_numbers[node_idx[key[0]]])
                z_j = int(atomic_numbers[node_idx[key[1]]])
                bond_type = int(bond_attr[edge_col, 0]) if bond_attr is not None and bond_attr.numel() else 0
                targets.append(
                    {
                        "edge": key,
                        "d0_col": edge_col,
                        "group_key": (min(z_i, z_j), max(z_i, z_j), bond_type),
                    }
                )
        if not targets:
            losses.append(clean_pos.new_zeros(()))
            coverage_values.append(clean_pos.new_zeros(()))
            unique_atom_ratio_values.append(clean_pos.new_zeros(()))
            max_usage_values.append(clean_pos.new_zeros(()))
            p95_usage_values.append(clean_pos.new_zeros(()))
            nonfixed_pair_rate_values.append(clean_pos.new_zeros(()))
            abs_error_values.append(clean_pos.new_zeros(()))
            rel_error_values.append(clean_pos.new_zeros(()))
            continue

        pair_i, pair_j = torch.triu_indices(n_atoms, n_atoms, offset=1, device=clean_pos.device)
        global_i = node_idx[pair_i]
        global_j = node_idx[pair_j]
        pair_edge_index = torch.stack([global_i, global_j], dim=0)
        pair_batch_idx = torch.full_like(global_i, batch_idx)
        pair_distances = _pbc_bond_distances(
            frac_pos=clean_pos,
            cell=clean_cell,
            edge_index=pair_edge_index,
            edge_batch_idx=pair_batch_idx,
        )
        pair_z_i = atomic_numbers[global_i]
        pair_z_j = atomic_numbers[global_j]

        targets_by_group: Dict[Tuple[int, int, int], list] = {}
        for target_idx, target in enumerate(targets):
            targets_by_group.setdefault(target["group_key"], []).append((target_idx, target))

        selected_pair_indices = []
        selected_ref_distances = []
        selected_local_pairs = []
        selected_target_indices = set()
        selected_pair_keys = set()
        usage = torch.zeros(n_atoms, device=clean_pos.device, dtype=torch.long)
        valence_cap = _default_valence_cap(atomic_numbers[node_idx]) if use_valence_cap else None
        fixed_edge_keys = {target["edge"] for target in targets}

        for group_key, group_targets in targets_by_group.items():
            z_a, z_b, _bond_type = group_key
            if z_a == z_b:
                candidate_mask = (pair_z_i == z_a) & (pair_z_j == z_b)
            else:
                candidate_mask = (
                    ((pair_z_i == z_a) & (pair_z_j == z_b))
                    | ((pair_z_i == z_b) & (pair_z_j == z_a))
                )
            if not torch.any(candidate_mask):
                continue

            candidate_pair_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
            candidate_rows = []
            for target_idx, target in group_targets:
                ref_dist = bond_d0[target["d0_col"]].clamp_min(1e-6)
                distances = pair_distances[candidate_pair_indices]
                window_mask = torch.ones_like(distances, dtype=torch.bool)
                if distance_lower_factor is not None:
                    window_mask = window_mask & (distances >= float(distance_lower_factor) * ref_dist)
                if distance_upper_factor is not None:
                    window_mask = window_mask & (distances <= float(distance_upper_factor) * ref_dist)
                if not torch.any(window_mask):
                    continue
                pair_subset = candidate_pair_indices[window_mask]
                costs = torch.abs(pair_distances[pair_subset].detach() - ref_dist.detach()) / ref_dist.detach()
                for cost, pair_idx in zip(costs.tolist(), pair_subset.tolist()):
                    candidate_rows.append((float(cost), int(target_idx), int(pair_idx), ref_dist))

            candidate_rows.sort(key=lambda item: item[0])
            for _cost, target_idx, pair_idx, ref_dist in candidate_rows:
                if target_idx in selected_target_indices:
                    continue
                local_i = int(pair_i[pair_idx])
                local_j = int(pair_j[pair_idx])
                pair_key = (local_i, local_j) if local_i <= local_j else (local_j, local_i)
                if pair_key in selected_pair_keys:
                    continue
                if use_valence_cap and valence_cap is not None:
                    if usage[local_i] >= valence_cap[local_i] or usage[local_j] >= valence_cap[local_j]:
                        continue
                selected_target_indices.add(target_idx)
                selected_pair_keys.add(pair_key)
                selected_pair_indices.append(pair_idx)
                selected_ref_distances.append(ref_dist)
                selected_local_pairs.append(pair_key)
                usage[local_i] += 1
                usage[local_j] += 1
                if len(selected_target_indices) == len(targets):
                    break

        gate = _time_window_gate(
            t[batch_idx : batch_idx + 1],
            time_gate_min=time_gate_min,
            time_gate_max=time_gate_max,
            time_gate_width=time_gate_width,
        ).squeeze(0)

        if not selected_pair_indices:
            losses.append(clean_pos.new_zeros(()))
            coverage_values.append(clean_pos.new_zeros(()))
            unique_atom_ratio_values.append(clean_pos.new_zeros(()))
            max_usage_values.append(usage.max().to(clean_pos.dtype) if usage.numel() else clean_pos.new_zeros(()))
            p95_usage_values.append(
                torch.quantile(usage.to(clean_pos.dtype), 0.95) if usage.numel() else clean_pos.new_zeros(())
            )
            nonfixed_pair_rate_values.append(clean_pos.new_zeros(()))
            abs_error_values.append(clean_pos.new_zeros(()))
            rel_error_values.append(clean_pos.new_zeros(()))
            continue

        selected_pair_idx_tensor = torch.tensor(selected_pair_indices, device=clean_pos.device, dtype=torch.long)
        ref_dist_tensor = torch.stack(selected_ref_distances).to(pair_distances.device).clamp_min(1e-6)
        selected_distances = pair_distances[selected_pair_idx_tensor]
        rel_error = (selected_distances - ref_dist_tensor) / ref_dist_tensor
        if loss_type == "relative_mse":
            pair_loss = rel_error.square()
        elif loss_type == "relative_huber":
            pair_loss = F.huber_loss(
                rel_error,
                torch.zeros_like(rel_error),
                reduction="none",
                delta=huber_beta,
            )
        else:
            raise ValueError(f"Unknown assignment bond loss type {loss_type}.")
        losses.append(pair_loss.mean() * gate)

        selected_count = len(selected_pair_indices)
        unique_atoms = int((usage > 0).sum().item())
        nonfixed_count = sum(1 for pair in selected_local_pairs if pair not in fixed_edge_keys)
        coverage_values.append(clean_pos.new_tensor(selected_count / max(len(targets), 1)) * gate)
        unique_atom_ratio_values.append(clean_pos.new_tensor(unique_atoms / max(n_atoms, 1)) * gate)
        max_usage_values.append(usage.max().to(clean_pos.dtype) * gate)
        p95_usage_values.append(torch.quantile(usage.to(clean_pos.dtype), 0.95) * gate)
        nonfixed_pair_rate_values.append(clean_pos.new_tensor(nonfixed_count / max(selected_count, 1)) * gate)
        abs_error_values.append(torch.abs(selected_distances - ref_dist_tensor).mean() * gate)
        rel_error_values.append(torch.abs(rel_error).mean() * gate)

    loss_tensor = torch.stack(losses)
    if not return_metrics:
        return loss_tensor
    metrics = {
        "coverage": torch.stack(coverage_values).mean(),
        "unique_atom_ratio": torch.stack(unique_atom_ratio_values).mean(),
        "max_endpoint_usage": torch.stack(max_usage_values).mean(),
        "p95_endpoint_usage": torch.stack(p95_usage_values).mean(),
        "nonfixed_pair_rate": torch.stack(nonfixed_pair_rate_values).mean(),
        "distance_mae": torch.stack(abs_error_values).mean(),
        "distance_rel_mae": torch.stack(rel_error_values).mean(),
    }
    return loss_tensor, metrics


def nonbond_repulsion_loss_from_x0_hat(
    *,
    multi_corruption: MultiCorruption,
    batch: BatchedData,
    noisy_batch: BatchedData,
    score_model_output: BatchedData,
    t: torch.Tensor,
    hard_negatives_per_bond: int = 4,
    cutoff_scale: float = 0.9,
    loss_type: Literal["hinge_squared", "relative_hinge_squared"] = "relative_hinge_squared",
    time_gate_center: Optional[float] = None,
    time_gate_width: float = 0.05,
    detach_cell: bool = False,
    mask_assignment_pairs_from_fixed_nonbond: bool = False,
    assignment_mask_time_gate_min: Optional[float] = None,
    assignment_mask_time_gate_max: Optional[float] = None,
    assignment_mask_time_gate_width: float = 0.08,
    assignment_mask_distance_lower_factor: Optional[float] = 0.5,
    assignment_mask_distance_upper_factor: Optional[float] = 1.6,
    assignment_mask_use_valence_cap: bool = True,
    return_metrics: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if "mol_bond_edge_index" not in batch:
        empty = torch.zeros(batch.get_batch_size(), device=noisy_batch["pos"].device)
        if return_metrics:
            return empty, {}
        return empty

    bond_edge_index = batch["mol_bond_edge_index"]
    if bond_edge_index.numel() == 0:
        empty = torch.zeros(batch.get_batch_size(), device=noisy_batch["pos"].device)
        if return_metrics:
            return empty, {}
        return empty

    clean_pos = _estimate_clean_pos(
        multi_corruption=multi_corruption,
        noisy_batch=noisy_batch,
        score_model_output=score_model_output,
        t=t,
    )
    clean_cell = _estimate_clean_cell(
        multi_corruption=multi_corruption,
        noisy_batch=noisy_batch,
        score_model_output=score_model_output,
        t=t,
    )
    if detach_cell:
        clean_cell = clean_cell.detach()

    node_batch_idx = batch.get_batch_idx("pos")
    atomic_numbers = batch["atomic_numbers"].long()
    bond_d0 = batch["mol_bond_d0"].to(clean_pos.device) if "mol_bond_d0" in batch else None
    bond_attr = batch["mol_bond_attr"] if "mol_bond_attr" in batch else None
    mol_copy_id = batch["mol_copy_id"].long() if "mol_copy_id" in batch else None
    losses = []
    conflict_rate_values = []
    different_component_rate_values = []
    selected_pair_count_values = []
    for batch_idx in range(batch.get_batch_size()):
        node_idx = torch.nonzero(node_batch_idx == batch_idx, as_tuple=False).flatten()
        n_atoms = int(node_idx.numel())
        if n_atoms < 2:
            losses.append(clean_pos.new_zeros(()))
            conflict_rate_values.append(clean_pos.new_zeros(()))
            different_component_rate_values.append(clean_pos.new_zeros(()))
            selected_pair_count_values.append(clean_pos.new_zeros(()))
            continue

        local_of_global = torch.full(
            (int(node_batch_idx.numel()),),
            -1,
            device=clean_pos.device,
            dtype=torch.long,
        )
        local_of_global[node_idx] = torch.arange(n_atoms, device=clean_pos.device)

        edge_mask = node_batch_idx[bond_edge_index[0]] == batch_idx
        local_edges = local_of_global[bond_edge_index[:, edge_mask]]
        bonded = torch.zeros((n_atoms, n_atoms), device=clean_pos.device, dtype=torch.bool)
        bonded[local_edges[0], local_edges[1]] = True
        bonded[local_edges[1], local_edges[0]] = True

        pair_i, pair_j = torch.triu_indices(n_atoms, n_atoms, offset=1, device=clean_pos.device)
        assignment_pairs = set()
        if mask_assignment_pairs_from_fixed_nonbond and bond_d0 is not None:
            assignment_gate = _time_window_gate(
                t[batch_idx : batch_idx + 1],
                time_gate_min=assignment_mask_time_gate_min,
                time_gate_max=assignment_mask_time_gate_max,
                time_gate_width=assignment_mask_time_gate_width,
            ).squeeze(0)
            if float(assignment_gate.detach().cpu()) > 0.5:
                selection = _select_assignment_bond_pairs_for_sample(
                    clean_pos=clean_pos,
                    clean_cell=clean_cell,
                    node_idx=node_idx,
                    node_batch_idx=node_batch_idx,
                    atomic_numbers=atomic_numbers,
                    bond_edge_index=bond_edge_index,
                    bond_d0=bond_d0,
                    bond_attr=bond_attr,
                    batch_idx=batch_idx,
                    distance_lower_factor=assignment_mask_distance_lower_factor,
                    distance_upper_factor=assignment_mask_distance_upper_factor,
                    use_valence_cap=assignment_mask_use_valence_cap,
                )
                assignment_pairs = set(selection["selected_local_pairs"])

        if assignment_pairs:
            conflict_count = sum(1 for pair in assignment_pairs if not bool(bonded[pair[0], pair[1]]))
            conflict_rate_values.append(clean_pos.new_tensor(conflict_count / max(len(assignment_pairs), 1)))
            if mol_copy_id is not None:
                local_copy_id = mol_copy_id[node_idx]
                diff_count = sum(
                    1
                    for pair in assignment_pairs
                    if int(local_copy_id[pair[0]]) != int(local_copy_id[pair[1]])
                )
                different_component_rate_values.append(
                    clean_pos.new_tensor(diff_count / max(len(assignment_pairs), 1))
                )
            else:
                different_component_rate_values.append(clean_pos.new_zeros(()))
            selected_pair_count_values.append(clean_pos.new_tensor(float(len(assignment_pairs))))
        else:
            conflict_rate_values.append(clean_pos.new_zeros(()))
            different_component_rate_values.append(clean_pos.new_zeros(()))
            selected_pair_count_values.append(clean_pos.new_zeros(()))

        nonbond_mask = ~bonded[pair_i, pair_j]
        if assignment_pairs:
            mask_values = [
                ((int(i), int(j)) if int(i) <= int(j) else (int(j), int(i))) in assignment_pairs
                for i, j in zip(pair_i.tolist(), pair_j.tolist())
            ]
            assignment_pair_mask = torch.tensor(mask_values, device=clean_pos.device, dtype=torch.bool)
            nonbond_mask = nonbond_mask & ~assignment_pair_mask
        if not torch.any(nonbond_mask):
            losses.append(clean_pos.new_zeros(()))
            continue

        pair_i = pair_i[nonbond_mask]
        pair_j = pair_j[nonbond_mask]
        global_i = node_idx[pair_i]
        global_j = node_idx[pair_j]
        pair_edge_index = torch.stack([global_i, global_j], dim=0)
        pair_batch_idx = torch.full_like(global_i, batch_idx)
        distances = _pbc_bond_distances(
            frac_pos=clean_pos,
            cell=clean_cell,
            edge_index=pair_edge_index,
            edge_batch_idx=pair_batch_idx,
        )
        cutoffs = _covalent_cutoff(
            atomic_numbers[global_i],
            atomic_numbers[global_j],
            scale=cutoff_scale,
        ).to(distances.device)
        violation = (cutoffs - distances).clamp_min(0.0)
        violating = violation > 0
        if not torch.any(violating):
            losses.append(clean_pos.new_zeros(()))
            continue

        violation = violation[violating]
        cutoffs = cutoffs[violating].clamp_min(1e-6)
        num_bonds = max(int(local_edges.shape[1] // 2), 1)
        k = min(int(hard_negatives_per_bond) * num_bonds, int(violation.numel()))
        top_violation = torch.topk(violation, k=k, largest=True).values
        if loss_type == "hinge_squared":
            pair_loss = top_violation.square()
        elif loss_type == "relative_hinge_squared":
            top_cutoffs = torch.topk(violation, k=k, largest=True).indices
            pair_loss = (top_violation / cutoffs[top_cutoffs]).square()
        else:
            raise ValueError(f"Unknown nonbond loss type {loss_type}.")

        if time_gate_center is not None:
            gate = torch.sigmoid((time_gate_center - t[batch_idx]) / time_gate_width)
            pair_loss = pair_loss * gate
        losses.append(pair_loss.mean())

    loss_tensor = torch.stack(losses)
    if not return_metrics:
        return loss_tensor
    metrics = {
        "assignment_pair_in_fixed_nonbond_rate": torch.stack(conflict_rate_values).mean(),
        "assignment_pair_in_different_fixed_component_rate": torch.stack(different_component_rate_values).mean(),
        "assignment_selected_pair_count": torch.stack(selected_pair_count_values).mean(),
    }
    return loss_tensor, metrics


def assignment_negative_loss_from_x0_hat(
    *,
    multi_corruption: MultiCorruption,
    batch: BatchedData,
    noisy_batch: BatchedData,
    score_model_output: BatchedData,
    t: torch.Tensor,
    hard_negatives_per_bond: int = 2,
    max_intra_negatives: int = 256,
    max_inter_negatives: int = 256,
    cutoff_scale: float = 0.9,
    loss_type: Literal["hinge_squared", "relative_hinge_squared"] = "relative_hinge_squared",
    time_gate_min: Optional[float] = None,
    time_gate_max: Optional[float] = None,
    time_gate_width: float = 0.08,
    detach_cell: bool = True,
    distance_lower_factor: Optional[float] = 0.5,
    distance_upper_factor: Optional[float] = 1.6,
    use_valence_cap: bool = True,
    min_coverage: float = 0.0,
    return_metrics: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Repel unlabelled near-covalent false bonds under the current assignment.

    Assignment bond loss pulls a greedy set of plausible unlabelled target bonds together.
    This complementary loss only repels pairs that were not selected as assignment-positive
    bonds, using components induced by the selected assignment pairs rather than fixed
    mol_copy_id labels. It is intended to reduce rewiring and multi-copy merging without
    reintroducing fixed-index topology constraints at medium noise.
    """
    if "mol_bond_edge_index" not in batch or "mol_bond_d0" not in batch:
        empty = torch.zeros(batch.get_batch_size(), device=noisy_batch["pos"].device)
        if return_metrics:
            return empty, {}
        return empty

    bond_edge_index = batch["mol_bond_edge_index"]
    if bond_edge_index.numel() == 0:
        empty = torch.zeros(batch.get_batch_size(), device=noisy_batch["pos"].device)
        if return_metrics:
            return empty, {}
        return empty

    clean_pos = _estimate_clean_pos(
        multi_corruption=multi_corruption,
        noisy_batch=noisy_batch,
        score_model_output=score_model_output,
        t=t,
    )
    clean_cell = _estimate_clean_cell(
        multi_corruption=multi_corruption,
        noisy_batch=noisy_batch,
        score_model_output=score_model_output,
        t=t,
    )
    if detach_cell:
        clean_cell = clean_cell.detach()

    node_batch_idx = batch.get_batch_idx("pos")
    atomic_numbers = batch["atomic_numbers"].long()
    bond_d0 = batch["mol_bond_d0"].to(clean_pos.device)
    bond_attr = batch["mol_bond_attr"] if "mol_bond_attr" in batch else None
    losses = []
    coverage_values = []
    unique_atom_ratio_values = []
    active_pair_count_values = []
    intra_count_values = []
    inter_count_values = []
    mean_violation_values = []

    def pair_loss_from_violation(
        violation: torch.Tensor,
        cutoffs: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if violation.numel() == 0 or k <= 0:
            empty = violation.new_zeros((0,))
            return empty, empty
        k = min(int(k), int(violation.numel()))
        top_values, top_indices = torch.topk(violation, k=k, largest=True)
        top_cutoffs = cutoffs[top_indices].clamp_min(1e-6)
        if loss_type == "hinge_squared":
            return top_values.square(), top_values
        if loss_type == "relative_hinge_squared":
            return (top_values / top_cutoffs).square(), top_values
        raise ValueError(f"Unknown assignment negative loss type {loss_type}.")

    for batch_idx in range(batch.get_batch_size()):
        node_idx = torch.nonzero(node_batch_idx == batch_idx, as_tuple=False).flatten()
        n_atoms = int(node_idx.numel())
        zeros = (clean_pos.new_zeros(()),) * 6
        if n_atoms < 2:
            losses.append(clean_pos.new_zeros(()))
            coverage_values.append(zeros[0])
            unique_atom_ratio_values.append(zeros[1])
            active_pair_count_values.append(zeros[2])
            intra_count_values.append(zeros[3])
            inter_count_values.append(zeros[4])
            mean_violation_values.append(zeros[5])
            continue

        selection = _select_assignment_bond_pairs_for_sample(
            clean_pos=clean_pos,
            clean_cell=clean_cell,
            node_idx=node_idx,
            node_batch_idx=node_batch_idx,
            atomic_numbers=atomic_numbers,
            bond_edge_index=bond_edge_index,
            bond_d0=bond_d0,
            bond_attr=bond_attr,
            batch_idx=batch_idx,
            distance_lower_factor=distance_lower_factor,
            distance_upper_factor=distance_upper_factor,
            use_valence_cap=use_valence_cap,
        )
        selected_pairs = set(selection["selected_local_pairs"])
        num_targets = int(selection["num_targets"])
        selected_count = len(selected_pairs)
        coverage = selected_count / max(num_targets, 1)
        usage = selection["usage"]
        unique_atom_ratio = float((usage > 0).sum().item()) / max(n_atoms, 1)
        coverage_values.append(clean_pos.new_tensor(coverage))
        unique_atom_ratio_values.append(clean_pos.new_tensor(unique_atom_ratio))

        gate = _time_window_gate(
            t[batch_idx : batch_idx + 1],
            time_gate_min=time_gate_min,
            time_gate_max=time_gate_max,
            time_gate_width=time_gate_width,
        ).squeeze(0)
        if selected_count == 0 or coverage < float(min_coverage):
            losses.append(clean_pos.new_zeros(()))
            active_pair_count_values.append(clean_pos.new_zeros(()))
            intra_count_values.append(clean_pos.new_zeros(()))
            inter_count_values.append(clean_pos.new_zeros(()))
            mean_violation_values.append(clean_pos.new_zeros(()))
            continue

        pair_i = selection["pair_i"]
        pair_j = selection["pair_j"]
        pair_distances = selection["pair_distances"]
        if pair_i.numel() == 0:
            losses.append(clean_pos.new_zeros(()))
            active_pair_count_values.append(clean_pos.new_zeros(()))
            intra_count_values.append(clean_pos.new_zeros(()))
            inter_count_values.append(clean_pos.new_zeros(()))
            mean_violation_values.append(clean_pos.new_zeros(()))
            continue

        parent = list(range(n_atoms))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for local_i, local_j in selected_pairs:
            union(int(local_i), int(local_j))

        used = usage > 0
        component_ids = torch.full((n_atoms,), -1, device=clean_pos.device, dtype=torch.long)
        root_to_component = {}
        for local_idx in range(n_atoms):
            if not bool(used[local_idx]):
                continue
            root = find(local_idx)
            if root not in root_to_component:
                root_to_component[root] = len(root_to_component)
            component_ids[local_idx] = root_to_component[root]

        selected_mask_values = [
            ((int(i), int(j)) if int(i) <= int(j) else (int(j), int(i))) in selected_pairs
            for i, j in zip(pair_i.tolist(), pair_j.tolist())
        ]
        selected_mask = torch.tensor(selected_mask_values, device=clean_pos.device, dtype=torch.bool)
        used_pair_mask = used[pair_i] & used[pair_j] & ~selected_mask
        comp_i = component_ids[pair_i]
        comp_j = component_ids[pair_j]
        assigned_pair_mask = used_pair_mask & (comp_i >= 0) & (comp_j >= 0)
        if not torch.any(assigned_pair_mask):
            losses.append(clean_pos.new_zeros(()))
            active_pair_count_values.append(clean_pos.new_zeros(()))
            intra_count_values.append(clean_pos.new_zeros(()))
            inter_count_values.append(clean_pos.new_zeros(()))
            mean_violation_values.append(clean_pos.new_zeros(()))
            continue

        global_i = node_idx[pair_i]
        global_j = node_idx[pair_j]
        cutoffs = _covalent_cutoff(
            atomic_numbers[global_i],
            atomic_numbers[global_j],
            scale=cutoff_scale,
        ).to(pair_distances.device)
        violation = (cutoffs - pair_distances).clamp_min(0.0)
        violating = assigned_pair_mask & (violation > 0)
        if not torch.any(violating):
            losses.append(clean_pos.new_zeros(()))
            active_pair_count_values.append(clean_pos.new_zeros(()))
            intra_count_values.append(clean_pos.new_zeros(()))
            inter_count_values.append(clean_pos.new_zeros(()))
            mean_violation_values.append(clean_pos.new_zeros(()))
            continue

        intra_mask = violating & (comp_i == comp_j)
        inter_mask = violating & (comp_i != comp_j)
        max_per_kind = max(int(hard_negatives_per_bond) * max(selected_count, 1), 1)
        k_intra = min(int(max_intra_negatives), max_per_kind)
        k_inter = min(int(max_inter_negatives), max_per_kind)
        intra_losses, intra_violations = pair_loss_from_violation(
            violation[intra_mask],
            cutoffs[intra_mask],
            k=k_intra,
        )
        inter_losses, inter_violations = pair_loss_from_violation(
            violation[inter_mask],
            cutoffs[inter_mask],
            k=k_inter,
        )
        selected_losses = torch.cat([intra_losses, inter_losses], dim=0)
        selected_violations = torch.cat([intra_violations, inter_violations], dim=0)
        if selected_losses.numel() == 0:
            losses.append(clean_pos.new_zeros(()))
            active_pair_count_values.append(clean_pos.new_zeros(()))
            intra_count_values.append(clean_pos.new_zeros(()))
            inter_count_values.append(clean_pos.new_zeros(()))
            mean_violation_values.append(clean_pos.new_zeros(()))
            continue

        losses.append(selected_losses.mean() * gate)
        active_pair_count_values.append(clean_pos.new_tensor(float(selected_losses.numel())))
        intra_count_values.append(clean_pos.new_tensor(float(intra_losses.numel())))
        inter_count_values.append(clean_pos.new_tensor(float(inter_losses.numel())))
        mean_violation_values.append(selected_violations.mean())

    loss_tensor = torch.stack(losses)
    if not return_metrics:
        return loss_tensor
    metrics = {
        "coverage": torch.stack(coverage_values).mean(),
        "unique_atom_ratio": torch.stack(unique_atom_ratio_values).mean(),
        "active_pair_count": torch.stack(active_pair_count_values).mean(),
        "intra_pair_count": torch.stack(intra_count_values).mean(),
        "inter_pair_count": torch.stack(inter_count_values).mean(),
        "mean_violation": torch.stack(mean_violation_values).mean(),
    }
    return loss_tensor, metrics


class MaterialsLoss(SummedFieldLoss):
    def __init__(
        self,
        reduce: Literal["sum", "mean"] = "mean",
        d3pm_hybrid_lambda: float = 0.0,
        include_pos: bool = True,
        include_cell: bool = True,
        include_atomic_numbers: bool = True,
        include_bond_lengths: bool = False,
        bond_weight: float = 0.0,
        bond_loss_type: Literal["relative_mse", "relative_huber"] = "relative_mse",
        bond_huber_beta: float = 0.2,
        bond_time_gate_center: Optional[float] = None,
        bond_time_gate_width: float = 0.05,
        bond_detach_cell: bool = False,
        include_assignment_bond_lengths: bool = False,
        assignment_bond_weight: float = 0.0,
        assignment_bond_loss_type: Literal["relative_mse", "relative_huber"] = "relative_huber",
        assignment_bond_huber_beta: float = 0.2,
        assignment_bond_time_gate_min: Optional[float] = None,
        assignment_bond_time_gate_max: Optional[float] = None,
        assignment_bond_time_gate_width: float = 0.08,
        assignment_bond_detach_cell: bool = False,
        assignment_bond_distance_lower_factor: Optional[float] = 0.5,
        assignment_bond_distance_upper_factor: Optional[float] = 1.6,
        assignment_bond_use_valence_cap: bool = True,
        include_assignment_negative: bool = False,
        assignment_negative_weight: float = 0.0,
        assignment_negative_loss_type: Literal["hinge_squared", "relative_hinge_squared"] = "relative_hinge_squared",
        assignment_negative_hard_negatives_per_bond: int = 2,
        assignment_negative_max_intra_negatives: int = 256,
        assignment_negative_max_inter_negatives: int = 256,
        assignment_negative_cutoff_scale: float = 0.9,
        assignment_negative_time_gate_min: Optional[float] = None,
        assignment_negative_time_gate_max: Optional[float] = None,
        assignment_negative_time_gate_width: float = 0.08,
        assignment_negative_detach_cell: bool = True,
        assignment_negative_distance_lower_factor: Optional[float] = 0.5,
        assignment_negative_distance_upper_factor: Optional[float] = 1.6,
        assignment_negative_use_valence_cap: bool = True,
        assignment_negative_min_coverage: float = 0.0,
        include_nonbond_repulsion: bool = False,
        nonbond_weight: float = 0.0,
        nonbond_loss_type: Literal["hinge_squared", "relative_hinge_squared"] = "relative_hinge_squared",
        nonbond_hard_negatives_per_bond: int = 4,
        nonbond_cutoff_scale: float = 0.9,
        nonbond_time_gate_center: Optional[float] = None,
        nonbond_time_gate_width: float = 0.05,
        nonbond_detach_cell: bool = False,
        mask_assignment_pairs_from_fixed_nonbond: bool = False,
        assignment_mask_time_gate_min: Optional[float] = None,
        assignment_mask_time_gate_max: Optional[float] = None,
        assignment_mask_time_gate_width: float = 0.08,
        assignment_mask_distance_lower_factor: Optional[float] = 0.5,
        assignment_mask_distance_upper_factor: Optional[float] = 1.6,
        assignment_mask_use_valence_cap: bool = True,
        fixed_topology_mid_t_scale: float = 1.0,
        fixed_topology_mid_t_gate_min: Optional[float] = None,
        fixed_topology_mid_t_gate_max: Optional[float] = None,
        fixed_topology_mid_t_gate_width: float = 0.08,
        weights: Optional[Dict[str, float]] = None,
    ):
        model_targets = {"pos": ModelTarget.score_times_std, "cell": ModelTarget.score_times_std}
        self.fields_to_score = []
        self.categorical_fields = []
        loss_fns: Dict[str, FieldLoss] = {}
        if include_pos:
            self.fields_to_score.append("pos")
            loss_fns["pos"] = partial(
                wrapped_normal_loss,
                reduce=reduce,
                model_target=model_targets["pos"],
            )
        if include_cell:
            self.fields_to_score.append("cell")
            loss_fns["cell"] = partial(
                denoising_score_matching,
                reduce=reduce,
                model_target=model_targets["cell"],
            )
        if include_atomic_numbers:
            model_targets["atomic_numbers"] = ModelTarget.logits
            self.fields_to_score.append("atomic_numbers")
            self.categorical_fields.append("atomic_numbers")
            loss_fns["atomic_numbers"] = partial(
                d3pm_loss,
                reduce=reduce,
                d3pm_hybrid_lambda=d3pm_hybrid_lambda,
            )
        self.reduce = reduce
        self.d3pm_hybrid_lambda = d3pm_hybrid_lambda
        self.include_bond_lengths = include_bond_lengths
        self.bond_weight = bond_weight
        self.bond_loss_type = bond_loss_type
        self.bond_huber_beta = bond_huber_beta
        self.bond_time_gate_center = bond_time_gate_center
        self.bond_time_gate_width = bond_time_gate_width
        self.bond_detach_cell = bond_detach_cell
        self.include_assignment_bond_lengths = include_assignment_bond_lengths
        self.assignment_bond_weight = assignment_bond_weight
        self.assignment_bond_loss_type = assignment_bond_loss_type
        self.assignment_bond_huber_beta = assignment_bond_huber_beta
        self.assignment_bond_time_gate_min = assignment_bond_time_gate_min
        self.assignment_bond_time_gate_max = assignment_bond_time_gate_max
        self.assignment_bond_time_gate_width = assignment_bond_time_gate_width
        self.assignment_bond_detach_cell = assignment_bond_detach_cell
        self.assignment_bond_distance_lower_factor = assignment_bond_distance_lower_factor
        self.assignment_bond_distance_upper_factor = assignment_bond_distance_upper_factor
        self.assignment_bond_use_valence_cap = assignment_bond_use_valence_cap
        self.include_assignment_negative = include_assignment_negative
        self.assignment_negative_weight = assignment_negative_weight
        self.assignment_negative_loss_type = assignment_negative_loss_type
        self.assignment_negative_hard_negatives_per_bond = assignment_negative_hard_negatives_per_bond
        self.assignment_negative_max_intra_negatives = assignment_negative_max_intra_negatives
        self.assignment_negative_max_inter_negatives = assignment_negative_max_inter_negatives
        self.assignment_negative_cutoff_scale = assignment_negative_cutoff_scale
        self.assignment_negative_time_gate_min = assignment_negative_time_gate_min
        self.assignment_negative_time_gate_max = assignment_negative_time_gate_max
        self.assignment_negative_time_gate_width = assignment_negative_time_gate_width
        self.assignment_negative_detach_cell = assignment_negative_detach_cell
        self.assignment_negative_distance_lower_factor = assignment_negative_distance_lower_factor
        self.assignment_negative_distance_upper_factor = assignment_negative_distance_upper_factor
        self.assignment_negative_use_valence_cap = assignment_negative_use_valence_cap
        self.assignment_negative_min_coverage = assignment_negative_min_coverage
        self.include_nonbond_repulsion = include_nonbond_repulsion
        self.nonbond_weight = nonbond_weight
        self.nonbond_loss_type = nonbond_loss_type
        self.nonbond_hard_negatives_per_bond = nonbond_hard_negatives_per_bond
        self.nonbond_cutoff_scale = nonbond_cutoff_scale
        self.nonbond_time_gate_center = nonbond_time_gate_center
        self.nonbond_time_gate_width = nonbond_time_gate_width
        self.nonbond_detach_cell = nonbond_detach_cell
        self.mask_assignment_pairs_from_fixed_nonbond = mask_assignment_pairs_from_fixed_nonbond
        self.assignment_mask_time_gate_min = assignment_mask_time_gate_min
        self.assignment_mask_time_gate_max = assignment_mask_time_gate_max
        self.assignment_mask_time_gate_width = assignment_mask_time_gate_width
        self.assignment_mask_distance_lower_factor = assignment_mask_distance_lower_factor
        self.assignment_mask_distance_upper_factor = assignment_mask_distance_upper_factor
        self.assignment_mask_use_valence_cap = assignment_mask_use_valence_cap
        self.fixed_topology_mid_t_scale = fixed_topology_mid_t_scale
        self.fixed_topology_mid_t_gate_min = fixed_topology_mid_t_gate_min
        self.fixed_topology_mid_t_gate_max = fixed_topology_mid_t_gate_max
        self.fixed_topology_mid_t_gate_width = fixed_topology_mid_t_gate_width
        super().__init__(
            loss_fns=loss_fns,
            weights=weights,
            model_targets=model_targets,
        )

    def __call__(
        self,
        *,
        multi_corruption: MultiCorruption,
        batch: BatchedData,
        noisy_batch: BatchedData,
        score_model_output: BatchedData,
        t: torch.Tensor,
        node_is_unmasked: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss, metrics_dict = super().__call__(
            multi_corruption=multi_corruption,
            batch=batch,
            noisy_batch=noisy_batch,
            score_model_output=score_model_output,
            t=t,
            node_is_unmasked=node_is_unmasked,
        )
        metrics_dict = dict(metrics_dict)
        if self.include_bond_lengths:
            bond_loss_per_sample = bond_length_loss_from_x0_hat(
                multi_corruption=multi_corruption,
                batch=batch,
                noisy_batch=noisy_batch,
                score_model_output=score_model_output,
                t=t,
                loss_type=self.bond_loss_type,
                huber_beta=self.bond_huber_beta,
                time_gate_center=self.bond_time_gate_center,
                time_gate_width=self.bond_time_gate_width,
                detach_cell=self.bond_detach_cell,
            )
            bond_loss_per_sample, fixed_topology_multiplier = _window_scaled_loss(
                bond_loss_per_sample,
                t,
                scale=self.fixed_topology_mid_t_scale,
                time_gate_min=self.fixed_topology_mid_t_gate_min,
                time_gate_max=self.fixed_topology_mid_t_gate_max,
                time_gate_width=self.fixed_topology_mid_t_gate_width,
            )
            bond_loss = bond_loss_per_sample.mean()
            metrics_dict["bond_lengths"] = bond_loss
            metrics_dict["fixed_topology_multiplier"] = fixed_topology_multiplier.mean()
            loss = loss + self.bond_weight * bond_loss

        if self.include_assignment_bond_lengths:
            assignment_bond_loss_per_sample, assignment_metrics = assignment_bond_length_loss_from_x0_hat(
                multi_corruption=multi_corruption,
                batch=batch,
                noisy_batch=noisy_batch,
                score_model_output=score_model_output,
                t=t,
                loss_type=self.assignment_bond_loss_type,
                huber_beta=self.assignment_bond_huber_beta,
                time_gate_min=self.assignment_bond_time_gate_min,
                time_gate_max=self.assignment_bond_time_gate_max,
                time_gate_width=self.assignment_bond_time_gate_width,
                detach_cell=self.assignment_bond_detach_cell,
                distance_lower_factor=self.assignment_bond_distance_lower_factor,
                distance_upper_factor=self.assignment_bond_distance_upper_factor,
                use_valence_cap=self.assignment_bond_use_valence_cap,
                return_metrics=True,
            )
            assignment_bond_loss = assignment_bond_loss_per_sample.mean()
            metrics_dict["assignment_bond_lengths"] = assignment_bond_loss
            for metric_name, metric_value in assignment_metrics.items():
                metrics_dict[f"assignment_bond_{metric_name}"] = metric_value
            loss = loss + self.assignment_bond_weight * assignment_bond_loss

        if self.include_assignment_negative:
            assignment_negative_loss_per_sample, assignment_negative_metrics = assignment_negative_loss_from_x0_hat(
                multi_corruption=multi_corruption,
                batch=batch,
                noisy_batch=noisy_batch,
                score_model_output=score_model_output,
                t=t,
                hard_negatives_per_bond=self.assignment_negative_hard_negatives_per_bond,
                max_intra_negatives=self.assignment_negative_max_intra_negatives,
                max_inter_negatives=self.assignment_negative_max_inter_negatives,
                cutoff_scale=self.assignment_negative_cutoff_scale,
                loss_type=self.assignment_negative_loss_type,
                time_gate_min=self.assignment_negative_time_gate_min,
                time_gate_max=self.assignment_negative_time_gate_max,
                time_gate_width=self.assignment_negative_time_gate_width,
                detach_cell=self.assignment_negative_detach_cell,
                distance_lower_factor=self.assignment_negative_distance_lower_factor,
                distance_upper_factor=self.assignment_negative_distance_upper_factor,
                use_valence_cap=self.assignment_negative_use_valence_cap,
                min_coverage=self.assignment_negative_min_coverage,
                return_metrics=True,
            )
            assignment_negative_loss = assignment_negative_loss_per_sample.mean()
            metrics_dict["assignment_negative"] = assignment_negative_loss
            for metric_name, metric_value in assignment_negative_metrics.items():
                metrics_dict[f"assignment_negative_{metric_name}"] = metric_value
            loss = loss + self.assignment_negative_weight * assignment_negative_loss

        if self.include_nonbond_repulsion:
            nonbond_loss_per_sample = nonbond_repulsion_loss_from_x0_hat(
                multi_corruption=multi_corruption,
                batch=batch,
                noisy_batch=noisy_batch,
                score_model_output=score_model_output,
                t=t,
                hard_negatives_per_bond=self.nonbond_hard_negatives_per_bond,
                cutoff_scale=self.nonbond_cutoff_scale,
                loss_type=self.nonbond_loss_type,
                time_gate_center=self.nonbond_time_gate_center,
                time_gate_width=self.nonbond_time_gate_width,
                detach_cell=self.nonbond_detach_cell,
                mask_assignment_pairs_from_fixed_nonbond=self.mask_assignment_pairs_from_fixed_nonbond,
                assignment_mask_time_gate_min=self.assignment_mask_time_gate_min,
                assignment_mask_time_gate_max=self.assignment_mask_time_gate_max,
                assignment_mask_time_gate_width=self.assignment_mask_time_gate_width,
                assignment_mask_distance_lower_factor=self.assignment_mask_distance_lower_factor,
                assignment_mask_distance_upper_factor=self.assignment_mask_distance_upper_factor,
                assignment_mask_use_valence_cap=self.assignment_mask_use_valence_cap,
                return_metrics=True,
            )
            nonbond_loss_per_sample, nonbond_metrics = nonbond_loss_per_sample
            nonbond_loss_per_sample, fixed_topology_multiplier = _window_scaled_loss(
                nonbond_loss_per_sample,
                t,
                scale=self.fixed_topology_mid_t_scale,
                time_gate_min=self.fixed_topology_mid_t_gate_min,
                time_gate_max=self.fixed_topology_mid_t_gate_max,
                time_gate_width=self.fixed_topology_mid_t_gate_width,
            )
            nonbond_loss = nonbond_loss_per_sample.mean()
            metrics_dict["nonbond_repulsion"] = nonbond_loss
            metrics_dict["nonbond_fixed_topology_multiplier"] = fixed_topology_multiplier.mean()
            for metric_name, metric_value in nonbond_metrics.items():
                metrics_dict[f"nonbond_{metric_name}"] = metric_value
            loss = loss + self.nonbond_weight * nonbond_loss

        return loss, metrics_dict
