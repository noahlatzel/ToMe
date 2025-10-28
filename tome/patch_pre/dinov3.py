import torch
from torch import Tensor

from dinov3.layers.block import SelfAttentionBlock
from dinov3.layers.attention import SelfAttention
from dinov3.models.vision_transformer import DinoVisionTransformer

from tome.merge import bipartite_soft_matching, merge_source
from tome.utils import parse_r, PatchedDinov3


class ToMeBlock(SelfAttentionBlock):
    def _forward_list(self, x_list: list[Tensor], rope_list=None) -> list[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """

        x_out = []
        for x, rope in zip(x_list, rope_list):
            
            x_norm1 = self.norm1(x)
            x_attn, merge, size, size_original = self.attn(x_norm1, rope=rope)
            if size is not None:
                x = merge(x * size_original, mode="sum")
                x = x / size
            x = x + self.ls1(x_attn)

            x_norm2 = self.norm2(x)
            x_mlp = self.mlp(x_norm2)
            x = x + self.ls2(x_mlp)
            x_out.append(x)
        x_ffn = x_out

        return x_ffn

class ToMeAttention(SelfAttention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v, metric, size, size_original = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x, metric, size, size_original

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        r = self._tome_info["r"].pop(0)
        metric = k.mean(1)

        size = self._tome_info["size"]

        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["num_special_tokens"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(merge, k, self._tome_info["source"])
            
            if size is None:
                size = torch.ones(B, N, 1, device=k.device, dtype=k.dtype) # B, N, 1
            
            size_original = size.clone()


            sin, cos = rope
            B = k.shape[0]
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
            
            size_extended = size[:, None, :, :] # B, 1, N, 1
            k = merge(k * size_extended, mode="sum")
            v = merge(v * size_extended, mode="sum")
            q = merge(q * size_extended, mode="sum")
            sin = merge(sin_extended * size, mode="sum")
            cos = merge(cos_extended * size, mode="sum")
            size_merged = merge(size, mode="sum")

            size_extended = size_merged[:, None, :, :] # B, 1, N, 1
            q, k, v = q / size_extended, k / size_extended, v / size_extended
            sin, cos = sin / size_merged, cos / size_merged

            sin = sin[:, self._tome_info["num_special_tokens"] :, :]
            cos = cos[:, self._tome_info["num_special_tokens"] :, :]
            rope[0] = sin
            rope[1] = cos
            size = self._tome_info["size"] = size_merged

        # attn_score: B, num_heads, N, N
        # size: B, N
        if size is not None:
            attn_mask = size.log()[:, None, None, :, 0] # B, 1, 1, N
        else:
            attn_mask = None
            
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2)
        return x.reshape([B, -1, C]), merge, size_merged, size_original


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.backbone.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(model: DinoVisionTransformer, trace_source: bool = False, prop_attn: bool = True):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For training and for evaluating MAE models off the self set this to be False.
    """
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
    }

    for module in model.modules():
        if isinstance(module, SelfAttentionBlock):
            module.__class__ = ToMeBlock
        elif isinstance(module, SelfAttention):
            module.__class__ = ToMeAttention
            module._tome_info = model._tome_info