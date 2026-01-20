# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    num_special_tokens: int = 5,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[-2]
    r = min(r, (t - num_special_tokens) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        scores[..., :num_special_tokens, :] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens: ..., t/2 - r, 1
        src_idx = edge_idx[..., :r, :]  # Merged Tokens: ..., r, 1
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # ..., r, 1

        if num_special_tokens >= 1:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def _expand_idx(idx: torch.Tensor, target_shape: list[int], extra_leading: int) -> torch.Tensor:
        if extra_leading > 0:
            insert_pos = idx.ndim - 2
            for _ in range(extra_leading):
                idx = idx.unsqueeze(insert_pos)
        return idx.expand(*target_shape)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        extra_leading = x.ndim - metric.ndim
        if extra_leading < 0:
            raise ValueError("merge expects input with >= metric dims.")

        unm_size = list(src.shape)
        unm_size[-2] = src.shape[-2] - r  # Unmerged Token

        unm = src.gather(dim=-2, index=_expand_idx(unm_idx, unm_size, extra_leading))  # ..., t/2 - r, c
        if mode != "prune":
            dst_size = list(src.shape)
            dst_size[-2] = r  # Merged Token
            src = src.gather(dim=-2, index=_expand_idx(src_idx, dst_size, extra_leading))  # ..., r, c
            dst = dst.scatter_reduce(
                -2,
                _expand_idx(dst_idx, dst_size, extra_leading),
                src,
                reduce=mode,
            )  # ..., t/2, c

        return torch.cat([unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        extra_leading = x.ndim - metric.ndim
        if extra_leading < 0:
            raise ValueError("unmerge expects input with >= metric dims.")
        unm_len = unm_idx.shape[-2]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]

        rc_size = list(x.shape)
        rc_size[-2] = r
        src = dst.gather(dim=-2, index=_expand_idx(dst_idx, rc_size, extra_leading))

        out_size = list(x.shape)
        out_size[-2] = metric.shape[-2]
        out = torch.zeros(*out_size, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst

        unm_size = list(x.shape)
        unm_size[-2] = unm_len
        out.scatter_(dim=-2, index=_expand_idx(2 * unm_idx, unm_size, extra_leading), src=unm)
        out.scatter_(dim=-2, index=_expand_idx(2 * src_idx, rc_size, extra_leading), src=src)

        return out

    return merge, unmerge


def kth_bipartite_soft_matching(metric: torch.Tensor, k: int) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(metric: torch.Tensor, r: int) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


def merge_wavg(merge: Callable, x: torch.Tensor, size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(merge: Callable, x: torch.Tensor, source: torch.Tensor = None) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        t = x.shape[-2]
        source = torch.eye(t, device=x.device)[None, ...].expand(*x.shape[:-1], t)

    source = merge(source, mode="amax")
    return source
