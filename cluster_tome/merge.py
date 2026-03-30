from __future__ import annotations

from enum import Enum
from typing import Callable, Tuple

import torch


def do_nothing(x, mode=None):
    return x


class UnclusteredTokenMode(str, Enum):
    MERGE = "merge"
    NO_MERGE = "no_merge"

    @classmethod
    def coerce(cls, value: str | "UnclusteredTokenMode") -> "UnclusteredTokenMode":
        if isinstance(value, cls):
            return value
        if isinstance(value, Enum) and isinstance(value.value, str):
            normalized = value.value.strip().lower()
        else:
            normalized = str(value).strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        valid = ", ".join(member.value for member in cls)
        raise ValueError(
            f"Unsupported unclustered token mode '{value}'. Expected one of: {valid}."
        )


def _normalize_cluster_tokens(
    cluster_labels,
    *,
    batch_size: int,
    patch_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    if cluster_labels is None:
        raise ValueError("cluster_labels must be provided.")

    labels = torch.as_tensor(cluster_labels, device=device)
    if labels.ndim == 3:
        labels = labels.reshape(labels.shape[0], -1)
    elif labels.ndim == 2:
        if labels.shape == (16, 16):
            labels = labels.reshape(1, -1)
        elif labels.shape[-1] != patch_tokens:
            raise ValueError(
                f"cluster_labels with rank-2 input must have trailing dimension {patch_tokens}, "
                f"got shape {tuple(labels.shape)}."
            )
    elif labels.ndim == 1:
        labels = labels.reshape(1, -1)
    else:
        raise ValueError("cluster_labels must have rank 1, 2, or 3.")

    if labels.shape[-1] != patch_tokens:
        raise ValueError(
            f"cluster_labels token count ({labels.shape[-1]}) does not match patch tokens ({patch_tokens})."
        )
    if labels.shape[0] == 1 and batch_size > 1:
        labels = labels.expand(batch_size, -1)
    if labels.shape[0] != batch_size:
        raise ValueError(
            f"cluster_labels batch ({labels.shape[0]}) must be 1 or match metric batch ({batch_size})."
        )

    labels = labels.to(dtype=torch.long)
    if (labels < -1).any():
        raise ValueError("cluster_labels values must be -1 or non-negative cluster ids.")
    return labels


def _expand_idx(idx: torch.Tensor, target_shape: list[int], extra_leading: int) -> torch.Tensor:
    if extra_leading > 0:
        insert_pos = idx.ndim - 2
        for _ in range(extra_leading):
            idx = idx.unsqueeze(insert_pos)
    return idx.expand(*target_shape)


def cluster_bipartite_soft_matching(
    metric: torch.Tensor,
    alpha: float,
    top_k: int,
    *,
    cluster_labels,
    unclustered_token_mode: str | UnclusteredTokenMode = UnclusteredTokenMode.MERGE,
    num_special_tokens: int = 1,
) -> Tuple[Callable, Callable]:
    """
    Cluster-constrained ToMe matching with even/odd bipartitions.

    Only pairs within the same cluster can merge.
    Candidate merges are filtered by cosine distance <= alpha and then capped by top_k.
    """
    if metric.ndim != 3:
        raise ValueError("cluster_bipartite_soft_matching expects metric shape [B, N, C].")
    mode = UnclusteredTokenMode.coerce(unclustered_token_mode)

    alpha = float(alpha)
    if not (0.0 <= alpha <= 2.0):
        raise ValueError(f"alpha must be in [0, 2], got {alpha}.")

    top_k = int(top_k)
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}.")

    batch_size, total_tokens, _ = metric.shape
    if num_special_tokens < 0 or num_special_tokens >= total_tokens:
        raise ValueError(
            f"num_special_tokens must be in [0, {total_tokens - 1}], got {num_special_tokens}."
        )

    patch_tokens = total_tokens - num_special_tokens
    if patch_tokens <= 1 or top_k == 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        labels = _normalize_cluster_tokens(
            cluster_labels,
            batch_size=batch_size,
            patch_tokens=patch_tokens,
            device=metric.device,
        )

        metric_patch = metric[:, num_special_tokens:, :]
        metric_patch = metric_patch / metric_patch.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        src_metric = metric_patch[:, ::2, :]
        dst_metric = metric_patch[:, 1::2, :]
        if dst_metric.shape[1] == 0:
            return do_nothing, do_nothing

        src_labels = labels[:, ::2]
        dst_labels = labels[:, 1::2]

        sim_threshold = 1.0 - alpha
        scores = src_metric @ dst_metric.transpose(-1, -2)
        same_cluster = src_labels[:, :, None] == dst_labels[:, None, :]
        if mode is UnclusteredTokenMode.NO_MERGE:
            src_is_unclustered = src_labels == -1
            dst_is_unclustered = dst_labels == -1
            disallow_unclustered = src_is_unclustered[:, :, None] | dst_is_unclustered[:, None, :]
            allowed = same_cluster & ~disallow_unclustered
        else:
            allowed = same_cluster
        scores = scores.masked_fill(~allowed, -torch.inf)

        node_max, node_idx = scores.max(dim=-1)
        candidate_scores = node_max.masked_fill(~torch.isfinite(node_max) | (node_max < sim_threshold), -torch.inf)
        top_scores, src_idx = candidate_scores.topk(k=top_k, dim=-1, largest=True, sorted=True)
        shared_valid = torch.isfinite(top_scores).all(dim=0)
        src_idx = src_idx[:, shared_valid]
        if src_idx.shape[1] == 0:
            return do_nothing, do_nothing
        dst_idx = node_idx.gather(dim=-1, index=src_idx)
        k_effective = src_idx.shape[1]
        src_idx = src_idx[..., None]
        dst_idx = dst_idx[..., None]
        src_count = src_metric.shape[-2]
        src_positions = torch.arange(src_count, device=metric.device).expand(batch_size, -1)
        keep_mask = torch.ones(batch_size, src_count, dtype=torch.bool, device=metric.device)
        keep_mask.scatter_(1, src_idx.squeeze(-1), False)
        unm_idx = src_positions.masked_select(keep_mask).reshape(
            batch_size,
            src_count - k_effective,
        )[..., None]

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        if x.shape[-2] != total_tokens:
            raise ValueError(
                f"merge input token count ({x.shape[-2]}) does not match metric token count ({total_tokens})."
            )

        special = x[..., :num_special_tokens, :]
        patch = x[..., num_special_tokens:, :]
        src, dst = patch[..., ::2, :], patch[..., 1::2, :]

        extra_leading = x.ndim - metric.ndim
        if extra_leading < 0:
            raise ValueError("merge expects input with >= metric dims.")

        unm_size = list(src.shape)
        unm_size[-2] = src.shape[-2] - k_effective
        unm = src.gather(dim=-2, index=_expand_idx(unm_idx, unm_size, extra_leading))

        if mode != "prune":
            rc_size = list(src.shape)
            rc_size[-2] = k_effective
            src_sel = src.gather(dim=-2, index=_expand_idx(src_idx, rc_size, extra_leading))
            dst = dst.scatter_reduce(
                -2,
                _expand_idx(dst_idx, rc_size, extra_leading),
                src_sel,
                reduce=mode,
            )

        return torch.cat([special, unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        extra_leading = x.ndim - metric.ndim
        if extra_leading < 0:
            raise ValueError("unmerge expects input with >= metric dims.")

        patch_len = patch_tokens - k_effective
        special, patch = torch.split(x, [num_special_tokens, patch_len], dim=-2)
        unm_len = unm_idx.shape[-2]
        unm, dst = patch[..., :unm_len, :], patch[..., unm_len:, :]

        rc_size = list(x.shape)
        rc_size[-2] = k_effective
        src_vals = dst.gather(dim=-2, index=_expand_idx(dst_idx, rc_size, extra_leading))

        out_patch_size = list(x.shape)
        out_patch_size[-2] = patch_tokens
        out_patch = torch.zeros(*out_patch_size, device=x.device, dtype=x.dtype)
        out_patch[..., 1::2, :] = dst

        unm_size = list(x.shape)
        unm_size[-2] = unm_len
        out_patch.scatter_(
            dim=-2,
            index=_expand_idx(2 * unm_idx, unm_size, extra_leading),
            src=unm,
        )
        out_patch.scatter_(
            dim=-2,
            index=_expand_idx(2 * src_idx, rc_size, extra_leading),
            src=src_vals,
        )
        return torch.cat([special, out_patch], dim=-2)

    return merge, unmerge
