from __future__ import annotations

import torch
from torch import Tensor

from dinov3.layers.attention import SelfAttention
from dinov3.layers.block import SelfAttentionBlock

from cluster_tome.merge import (
    UnclusteredTokenMode,
    cluster_bipartite_soft_matching,
    do_nothing,
)
from tome.merge import merge_source, merge_wavg
from tome.utils import PatchedDinov3

SPECIAL_LABEL_SENTINEL = -2


def _prepare_initial_cluster_tokens(
    cluster_map,
    *,
    batch_size: int,
    patch_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    if cluster_map is None:
        raise ValueError("cluster_tome requires a cluster label map before forward.")

    labels = torch.as_tensor(cluster_map, device=device)
    if labels.ndim == 3:
        if labels.shape[-2:] != (16, 16):
            raise ValueError(
                f"cluster label map must have trailing shape (16, 16), got {tuple(labels.shape)}."
            )
        labels = labels.reshape(labels.shape[0], -1)
    elif labels.ndim == 2:
        if labels.shape == (16, 16):
            labels = labels.reshape(1, -1)
        elif labels.shape[-1] != 16 * 16:
            raise ValueError(
                "cluster label map rank-2 inputs must be [16, 16] or [B, 256]."
            )
    elif labels.ndim == 1:
        if labels.numel() != 16 * 16:
            raise ValueError(
                f"flattened cluster label map must have 256 entries, got {labels.numel()}."
            )
        labels = labels.reshape(1, -1)
    else:
        raise ValueError("cluster label map must have rank 1, 2, or 3.")

    if labels.shape[-1] != 16 * 16:
        raise ValueError(
            f"cluster label map must contain 256 patch labels, got {labels.shape[-1]}."
        )
    if patch_tokens != 16 * 16:
        raise ValueError(
            f"cluster_tome with a 16x16 map requires 256 patch tokens before merge, got {patch_tokens}."
        )
    if labels.shape[0] == 1 and batch_size > 1:
        labels = labels.expand(batch_size, -1)
    if labels.shape[0] != batch_size:
        raise ValueError(
            f"cluster label map batch ({labels.shape[0]}) must be 1 or match input batch ({batch_size})."
        )

    labels = labels.to(dtype=torch.long)
    if (labels < -1).any():
        raise ValueError("cluster label map values must be -1 or non-negative cluster ids.")
    return labels


class ToMeClusterBlock(SelfAttentionBlock):
    def _forward_list(self, x_list: list[Tensor], rope_list=None) -> list[Tensor]:
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        alpha = self._tome_info["alpha"]
        top_k = self._tome_info["top_k"]

        x_out: list[Tensor] = []
        for x, rope in zip(x_list, rope_list):
            x_norm1 = self.norm1(x)
            x_attn, metric = self.attn(x_norm1, size=attn_size, rope=rope)
            x = x + self.ls1(x_attn)

            if top_k > 0:
                num_special_tokens = self._tome_info["num_special_tokens"]
                cluster_tokens = self._tome_info.get("cluster_tokens")
                if cluster_tokens is None:
                    cluster_tokens = _prepare_initial_cluster_tokens(
                        self._tome_info.get("cluster_map"),
                        batch_size=x.shape[0],
                        patch_tokens=x.shape[1] - num_special_tokens,
                        device=x.device,
                    )
                else:
                    cluster_tokens = torch.as_tensor(cluster_tokens, device=x.device, dtype=torch.long)
                    if cluster_tokens.ndim == 1:
                        cluster_tokens = cluster_tokens.reshape(1, -1)
                    if cluster_tokens.shape[0] == 1 and x.shape[0] > 1:
                        cluster_tokens = cluster_tokens.expand(x.shape[0], -1)
                    expected_patches = x.shape[1] - num_special_tokens
                    if cluster_tokens.shape != (x.shape[0], expected_patches):
                        raise ValueError(
                            f"cluster token labels shape {tuple(cluster_tokens.shape)} does not match "
                            f"expected {(x.shape[0], expected_patches)}."
                        )
                    if (cluster_tokens < -1).any():
                        raise ValueError("cluster token labels must stay >= -1.")

                self._tome_info["cluster_tokens"] = cluster_tokens
                merge, _ = cluster_bipartite_soft_matching(
                    metric,
                    alpha,
                    top_k,
                    cluster_labels=cluster_tokens,
                    unclustered_token_mode=self._tome_info["unclustered_token_mode"],
                    num_special_tokens=num_special_tokens,
                )
                if merge is not do_nothing:
                    if self._tome_info["trace_source"]:
                        self._tome_info["source"] = merge_source(
                            merge, x, self._tome_info["source"]
                        )
                    x, size = merge_wavg(merge, x, self._tome_info["size"])

                    specials = torch.full(
                        (cluster_tokens.shape[0], num_special_tokens, 1),
                        fill_value=float(SPECIAL_LABEL_SENTINEL),
                        device=x.device,
                        dtype=torch.float32,
                    )
                    cluster_full = torch.cat([specials, cluster_tokens[..., None].float()], dim=1)
                    cluster_full = merge(cluster_full, mode="amax")
                    self._tome_info["cluster_tokens"] = cluster_full[:, num_special_tokens:, 0].to(
                        torch.long
                    )

                    sin, cos = rope
                    batch = x.shape[0]
                    if sin.ndim == 2:
                        sin = sin[None].expand(batch, -1, -1)
                    if cos.ndim == 2:
                        cos = cos[None].expand(batch, -1, -1)

                    zeros_special = torch.zeros(
                        batch,
                        num_special_tokens,
                        sin.shape[2],
                        device=sin.device,
                        dtype=sin.dtype,
                    )
                    sin_extended = torch.cat([zeros_special, sin], dim=1)
                    cos_extended = torch.cat([zeros_special, cos], dim=1)
                    sin, _ = merge_wavg(merge, sin_extended, self._tome_info["size"])
                    cos, _ = merge_wavg(merge, cos_extended, self._tome_info["size"])
                    sin = sin[:, num_special_tokens:, :]
                    cos = cos[:, num_special_tokens:, :]
                    rope[0] = sin
                    rope[1] = cos
                    self._tome_info["size"] = size

            x_norm2 = self.norm2(x)
            x_mlp = self.mlp(x_norm2)
            x = x + self.ls2(x_mlp)
            x_out.append(x)

        return x_out


class ToMeAttention(SelfAttention):
    def forward(self, x: Tensor, size=None, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v, metric = self.compute_attention(qkv=qkv, size=size, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x, metric

    def compute_attention(self, qkv: Tensor, size=None, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        batch, tokens, _ = qkv.shape
        channels = self.qkv.in_features

        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        if size is not None:
            attn_mask = size.log()[:, None, None, :, 0]
        else:
            attn_mask = None

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2)
        return x.reshape([batch, tokens, channels]), k.mean(1)


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            self._tome_info["cluster_tokens"] = None
            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model,
    trace_source: bool = False,
    prop_attn: bool = True,
    alpha: float = 0.5,
    top_k: int = 0,
    unclustered_token_mode: str | UnclusteredTokenMode = UnclusteredTokenMode.MERGE,
):
    alpha = float(alpha)
    if not (0.0 <= alpha <= 2.0):
        raise ValueError(f"alpha must be in [0, 2], got {alpha}.")
    top_k = int(top_k)
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}.")
    mode = UnclusteredTokenMode.coerce(unclustered_token_mode)

    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.backbone.__class__ = PatchedDinov3
    model.__class__ = ToMeVisionTransformer
    model.alpha = alpha
    model.top_k = top_k
    model.unclustered_token_mode = mode.value
    model._tome_info = {
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "num_special_tokens": model.backbone.n_storage_tokens + 1,
        "cluster_map": None,
        "cluster_tokens": None,
        "alpha": alpha,
        "top_k": top_k,
        "unclustered_token_mode": mode.value,
    }

    for module in model.modules():
        if isinstance(module, SelfAttentionBlock):
            module.__class__ = ToMeClusterBlock
            module._tome_info = model._tome_info
        elif isinstance(module, SelfAttention):
            module.__class__ = ToMeAttention
