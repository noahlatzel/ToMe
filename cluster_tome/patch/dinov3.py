import torch
from torch import Tensor

from dinov3.layers.attention import SelfAttention
from dinov3.layers.block import SelfAttentionBlock

from cluster_tome.merge import cluster_bipartite_soft_matching
from tome.merge import merge_source, merge_wavg
from tome.utils import PatchedDinov3, parse_r


class ToMeClusterBlock(SelfAttentionBlock):
    def _forward_list(self, x_list: list[Tensor], rope_list=None) -> list[Tensor]:
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        r = self._tome_info["r"].pop(0)

        x_out = []
        for x, rope in zip(x_list, rope_list):
            x_norm1 = self.norm1(x)
            x_attn, metric = self.attn(x_norm1, size=attn_size, rope=rope)
            x = x + self.ls1(x_attn)

            if r > 0:
                cluster_tokens = self._tome_info.get("cluster_tokens")
                if cluster_tokens is None:
                    base_cluster_map = self._tome_info.get("cluster_map")
                    if base_cluster_map is None:
                        raise ValueError("cluster_tome requires _tome_info['cluster_map'] to be set before forward.")
                    cluster_tokens = base_cluster_map
                cluster_tokens = cluster_tokens.to(device=x.device, dtype=torch.long)
                if cluster_tokens.ndim == 2:
                    cluster_tokens = cluster_tokens.unsqueeze(0)
                cluster_tokens = cluster_tokens.reshape(cluster_tokens.shape[0], -1)
                if cluster_tokens.shape[0] == 1 and x.shape[0] > 1:
                    cluster_tokens = cluster_tokens.expand(x.shape[0], -1)

                merge, _ = cluster_bipartite_soft_matching(
                    metric,
                    r,
                    cluster_map=cluster_tokens,
                    num_special_tokens=self._tome_info["num_special_tokens"],
                )
                if self._tome_info["trace_source"]:
                    self._tome_info["source"] = merge_source(merge, x, self._tome_info["source"])
                x, size = merge_wavg(merge, x, self._tome_info["size"])

                # Propagate cluster ids through the exact same merge operation.
                # We embed patch cluster ids in a full token tensor and mark specials as -1.
                specials = torch.full(
                    (cluster_tokens.shape[0], self._tome_info["num_special_tokens"], 1),
                    fill_value=-1.0,
                    device=cluster_tokens.device,
                    dtype=torch.float32,
                )
                cluster_full = torch.cat([specials, cluster_tokens[..., None].float()], dim=1)
                merged_full = merge(cluster_full, mode="amax").squeeze(-1)

                rows = []
                for b_idx in range(merged_full.shape[0]):
                    patch_only = merged_full[b_idx][merged_full[b_idx] >= 0].to(torch.long)
                    rows.append(patch_only)
                cluster_tokens = torch.stack(rows, dim=0)
                self._tome_info["cluster_tokens"] = cluster_tokens

                sin, cos = rope
                B = x.shape[0]
                if sin.ndim == 2:
                    sin = sin[None].expand(B, -1, -1)
                if cos.ndim == 2:
                    cos = cos[None].expand(B, -1, -1)
                zeros_special = torch.zeros(
                    B,
                    self._tome_info["num_special_tokens"],
                    sin.shape[2],
                    device=sin.device,
                    dtype=sin.dtype,
                )
                sin_extended = torch.cat([zeros_special, sin], dim=1)
                cos_extended = torch.cat([zeros_special, cos], dim=1)
                sin, _ = merge_wavg(merge, sin_extended, self._tome_info["size"])
                cos, _ = merge_wavg(merge, cos_extended, self._tome_info["size"])
                sin = sin[:, self._tome_info["num_special_tokens"] :, :]
                cos = cos[:, self._tome_info["num_special_tokens"] :, :]
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
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
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
        return x.reshape([B, N, C]), k.mean(1)


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.backbone.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            self._tome_info["cluster_tokens"] = None
            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(model, trace_source: bool = False, prop_attn: bool = True):
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.backbone.__class__ = PatchedDinov3
    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "num_special_tokens": model.backbone.n_storage_tokens + 1,
        "cluster_map": None,
        "cluster_tokens": None,
    }

    for module in model.modules():
        if isinstance(module, SelfAttentionBlock):
            module.__class__ = ToMeClusterBlock
            module._tome_info = model._tome_info
        elif isinstance(module, SelfAttention):
            module.__class__ = ToMeAttention
