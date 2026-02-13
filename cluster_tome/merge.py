from __future__ import annotations

from typing import Callable, Tuple

import torch


def do_nothing(x, mode=None):
    return x


def _normalize_cluster_map(
    cluster_map: torch.Tensor | None,
    *,
    batch_size: int,
    patch_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    if cluster_map is None:
        return torch.zeros(batch_size, patch_tokens, dtype=torch.long, device=device)

    cluster_map = cluster_map.to(device=device, dtype=torch.long)
    if cluster_map.ndim == 3:
        cluster_map = cluster_map.reshape(cluster_map.shape[0], -1)
    elif cluster_map.ndim == 2:
        if cluster_map.shape[-1] != patch_tokens:
            cluster_map = cluster_map.reshape(1, -1)
    elif cluster_map.ndim == 1:
        cluster_map = cluster_map.view(1, -1)
    else:
        raise ValueError("cluster_map must have shape [H, W], [B, H, W], [P], or [B, P].")

    if cluster_map.ndim != 2:
        raise ValueError("cluster_map normalization failed to produce [B, P].")
    if cluster_map.shape[-1] != patch_tokens:
        raise ValueError("cluster_map flattened size must match patch token count.")
    if cluster_map.shape[0] == 1 and batch_size > 1:
        cluster_map = cluster_map.expand(batch_size, -1)
    if cluster_map.shape[0] != batch_size:
        raise ValueError("cluster_map batch size must be 1 or equal to metric batch size.")
    return cluster_map


def _expand_idx(idx: torch.Tensor, target_shape: list[int], extra_leading: int) -> torch.Tensor:
    if extra_leading > 0:
        insert_pos = idx.ndim - 2
        for _ in range(extra_leading):
            idx = idx.unsqueeze(insert_pos)
    return idx.expand(*target_shape)


def cluster_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    *,
    cluster_map: torch.Tensor | None,
    num_special_tokens: int = 5,
) -> Tuple[Callable, Callable]:
    """
    Checkerboard ToMe matching over patch tokens, with independent matching per cluster.

    `r` is scaled per cluster by cluster size. Unused per-cluster budget is dropped.
    """
    if metric.ndim != 3:
        raise ValueError("cluster_bipartite_soft_matching expects metric shape [B, N, C].")

    batch_size, total_tokens, _ = metric.shape
    patch_tokens = total_tokens - num_special_tokens
    if patch_tokens <= 1:
        return do_nothing, do_nothing

    r = min(int(r), patch_tokens // 2)
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        cluster_flat = _normalize_cluster_map(
            cluster_map,
            batch_size=batch_size,
            patch_tokens=patch_tokens,
            device=metric.device,
        )

        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        metric_patch = metric[:, num_special_tokens:, :]
        a, b = metric_patch[..., ::2, :], metric_patch[..., 1::2, :]
        if b.shape[-2] == 0:
            return do_nothing, do_nothing

        scores = a @ b.transpose(-1, -2)

        a_cluster = cluster_flat[..., ::2]
        b_cluster = cluster_flat[..., 1::2]
        same_cluster = a_cluster[..., :, None] == b_cluster[..., None, :]
        scores = scores.masked_fill(~same_cluster, -torch.inf)

        node_max, node_idx = scores.max(dim=-1)

        cluster_count = int(cluster_flat.max().item()) + 1
        sizes = torch.stack(
            [torch.bincount(cluster_flat[b_idx], minlength=cluster_count) for b_idx in range(batch_size)],
            dim=0,
        )
        budgets = torch.floor((sizes.float() / float(patch_tokens)) * float(r)).to(torch.long)

        src_bank: list[torch.Tensor] = []
        dst_bank: list[torch.Tensor] = []
        score_bank: list[torch.Tensor] = []
        for b_idx in range(batch_size):
            src_parts = []
            dst_parts = []
            score_parts = []
            for cluster_id in range(cluster_count):
                budget = int(budgets[b_idx, cluster_id].item())
                if budget <= 0:
                    continue

                eligible = (a_cluster[b_idx] == cluster_id) & torch.isfinite(node_max[b_idx])
                candidates = torch.nonzero(eligible, as_tuple=False).squeeze(1)
                if candidates.numel() == 0:
                    continue

                take = min(budget, int(candidates.numel()))
                top_local = node_max[b_idx, candidates].argsort(descending=True)[:take]
                chosen_src = candidates[top_local]
                chosen_dst = node_idx[b_idx, chosen_src]

                src_parts.append(chosen_src)
                dst_parts.append(chosen_dst)
                score_parts.append(node_max[b_idx, chosen_src])

            if src_parts:
                src_bank.append(torch.cat(src_parts, dim=0))
                dst_bank.append(torch.cat(dst_parts, dim=0))
                score_bank.append(torch.cat(score_parts, dim=0))
            else:
                src_bank.append(torch.empty(0, dtype=torch.long, device=metric.device))
                dst_bank.append(torch.empty(0, dtype=torch.long, device=metric.device))
                score_bank.append(torch.empty(0, dtype=torch.float32, device=metric.device))

        r_effective = min((int(src.numel()) for src in src_bank), default=0)
        if r_effective <= 0:
            return do_nothing, do_nothing

        src_idx = torch.empty(batch_size, r_effective, 1, dtype=torch.long, device=metric.device)
        dst_idx = torch.empty(batch_size, r_effective, 1, dtype=torch.long, device=metric.device)
        for b_idx in range(batch_size):
            if src_bank[b_idx].numel() > r_effective:
                top = score_bank[b_idx].argsort(descending=True)[:r_effective]
                src_sel = src_bank[b_idx][top]
                dst_sel = dst_bank[b_idx][top]
            else:
                src_sel = src_bank[b_idx]
                dst_sel = dst_bank[b_idx]
            src_idx[b_idx, :, 0] = src_sel
            dst_idx[b_idx, :, 0] = dst_sel

        a_count = a.shape[-2]
        keep_mask = torch.ones(batch_size, a_count, dtype=torch.bool, device=metric.device)
        keep_mask.scatter_(1, src_idx[..., 0], False)

        unm_rows = []
        for b_idx in range(batch_size):
            unm_rows.append(torch.nonzero(keep_mask[b_idx], as_tuple=False).squeeze(1))
        unm_idx = torch.stack(unm_rows, dim=0)[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if x.shape[-2] != total_tokens:
            raise ValueError("merge input token count does not match metric token count.")

        special = x[..., :num_special_tokens, :]
        patch = x[..., num_special_tokens:, :]
        src, dst = patch[..., ::2, :], patch[..., 1::2, :]
        extra_leading = x.ndim - metric.ndim
        if extra_leading < 0:
            raise ValueError("merge expects input with >= metric dims.")

        unm_size = list(src.shape)
        unm_size[-2] = src.shape[-2] - r_effective
        unm = src.gather(dim=-2, index=_expand_idx(unm_idx, unm_size, extra_leading))

        if mode != "prune":
            dst_size = list(src.shape)
            dst_size[-2] = r_effective
            src_sel = src.gather(dim=-2, index=_expand_idx(src_idx, dst_size, extra_leading))
            dst = dst.scatter_reduce(
                -2,
                _expand_idx(dst_idx, dst_size, extra_leading),
                src_sel,
                reduce=mode,
            )

        return torch.cat([special, unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        extra_leading = x.ndim - metric.ndim
        if extra_leading < 0:
            raise ValueError("unmerge expects input with >= metric dims.")

        patch_len = patch_tokens - r_effective
        special, patch = torch.split(x, [num_special_tokens, patch_len], dim=-2)

        unm_len = unm_idx.shape[-2]
        unm, dst = patch[..., :unm_len, :], patch[..., unm_len:, :]

        rc_size = list(x.shape)
        rc_size[-2] = r_effective
        src_vals = dst.gather(dim=-2, index=_expand_idx(dst_idx, rc_size, extra_leading))

        out_patch_size = list(x.shape)
        out_patch_size[-2] = patch_tokens
        out_patch = torch.zeros(*out_patch_size, device=x.device, dtype=x.dtype)
        out_patch[..., 1::2, :] = dst

        unm_size = list(x.shape)
        unm_size[-2] = unm_len
        out_patch.scatter_(dim=-2, index=_expand_idx(2 * unm_idx, unm_size, extra_leading), src=unm)
        out_patch.scatter_(dim=-2, index=_expand_idx(2 * src_idx, rc_size, extra_leading), src=src_vals)
        return torch.cat([special, out_patch], dim=-2)

    return merge, unmerge
