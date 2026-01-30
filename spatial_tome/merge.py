from typing import Callable
from math import ceil
import torch
from torch.nn.functional import pad

from tome.merge import do_nothing

def get_checkerboard_mask(H, W, invert: bool = False):
    column = torch.tensor([True, False]).repeat(ceil(H / 2))[:H]
    row = torch.tensor([True, False]).repeat(ceil(W / 2))[:W]
    mask = torch.bitwise_xor(column[:, None], row[None, :])
    if invert:
        mask = ~mask
    return mask

def spatial_soft_matching(
    metric: torch.Tensor,
    H: int,
    W: int,
    r: int,
    num_special_tokens: int = 5,
    invert_mask: bool = False,
) -> tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    # metric: ..., H * W, c

    # We can only reduce by a maximum of 50% tokens
    assert H * W + num_special_tokens == metric.shape[-2]
    r = min(r, (H * W - num_special_tokens) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)

        metric_spatial = metric[..., num_special_tokens:, :].view(*metric.shape[:-2], H, W, metric.shape[-1]) # ..., H, W, c

        scores_y = torch.linalg.vecdot(metric_spatial[..., 1:, :, :], metric_spatial[..., :-1, :, :]) # ..., H-1, W
        scores_x = torch.linalg.vecdot(metric_spatial[..., :, 1:, :], metric_spatial[..., :, :-1, :]) # ..., H, W-1

        # pad scores to H, W
        scores_down = pad(scores_y, (0, 0, 0, 1), value=-torch.inf).flatten(-2)
        scores_right = pad(scores_x, (0, 1, 0, 0), value=-torch.inf).flatten(-2)
        scores_up = pad(scores_y, (0, 0, 1, 0), value=-torch.inf).flatten(-2)
        scores_left = pad(scores_x, (1, 0, 0, 0), value=-torch.inf).flatten(-2)

        # the source indices can either be merged with x or y, so we take the max
        max_scores, direction = torch.stack([scores_left, scores_right, scores_up, scores_down], dim=-1).max(dim=-1) # ..., H * W
        mask = get_checkerboard_mask(H, W, invert=invert_mask).flatten().to(metric.device)
        left_idx = torch.where(mask)[0] + num_special_tokens
        right_idx = torch.where(~mask)[0] + num_special_tokens

        max_scores = max_scores[..., left_idx - num_special_tokens]
        
        edge_idx = max_scores.argsort(dim=-1, descending=True)[..., None]

        unm_idx = left_idx[edge_idx[..., r:, :]]  # Unmerged Tokens: ..., t/2 - r, 1
        src_idx = left_idx[edge_idx[..., :r, :]]  # Merged Tokens: ..., r, 1
        direction = direction.gather(dim=-1, index=src_idx[..., 0] - num_special_tokens)  # ..., r, 1

        # set target indices as the neighboring index in the corresponding direction
        dst_idx = src_idx.clone()
        dst_idx[direction == 0] -= 1 # ..., r
        dst_idx[direction == 1] += 1 # ..., r
        dst_idx[direction == 2] -= W # ..., r
        dst_idx[direction == 3] += W # ..., r

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        assert x.shape[-2] == H * W + num_special_tokens

        sp_tok = x[..., :num_special_tokens, :]  # Special Tokens: ..., s, c

        unm = x.gather(dim=-2, index=unm_idx.expand(*x.shape[:-2], unm_idx.shape[-2], x.shape[-1])) # ..., t/2 - r, c

        if mode != 'prune':
            src = x.gather(dim=-2, index=src_idx.expand(*x.shape[:-2], r, x.shape[-1])) # ..., r, c
            x = x.scatter_reduce(-2, dst_idx.expand(*x.shape[:-2], r, x.shape[-1]), src, reduce=mode) # ..., t/2, c

        dst = x.gather(dim=-2, index=right_idx[:, None].expand(*x.shape[:-2], right_idx.shape[0], x.shape[-1])) # ..., t/2, c

        return torch.cat([sp_tok, unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:

        unm_len = unm_idx.shape[-2]
        dst_len = right_idx.shape[0]
        sp_tok, unm, dst = torch.split(x, [num_special_tokens, unm_len, dst_len], dim=-2)

        out = torch.empty(*x.shape[:-2], metric.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)

        out[..., :num_special_tokens, :] = sp_tok
        out.scatter_(dim=-2, index=unm_idx.expand(*x.shape[:-2], unm_len, x.shape[-1]), src=unm)
        out.scatter_(dim=-2, index=right_idx[:, None].expand(*x.shape[:-2], dst_len, x.shape[-1]), src=dst)
        src = out.gather(dim=-2, index=dst_idx.expand(*x.shape[:-2], r, x.shape[-1]))
        out.scatter_(dim=-2, index=src_idx.expand(*x.shape[:-2], r, x.shape[-1]), src=src)

        return out

    return merge, unmerge
