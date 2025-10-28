import torch
from torch import Tensor

from dinov3.layers.block import SelfAttentionBlock
from dinov3.layers.attention import SelfAttention

from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r, PatchedDinov3


class ToMeBlock(SelfAttentionBlock):
    def _forward_list(self, x_list: list[Tensor], rope_list=None) -> list[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        r = self._tome_info["r"].pop(0)

        x_out = []
        for x, rope in zip(x_list, rope_list):
            x_norm1 = self.norm1(x)
            x_attn, metric = self.attn(x_norm1, size=attn_size, rope=rope)
            x = x + self.ls1(x_attn)

            if r > 0:
                # Apply ToMe here
                merge, _ = bipartite_soft_matching(
                    metric,
                    r,
                    self._tome_info["num_special_tokens"],
                )
                if self._tome_info["trace_source"]:
                    self._tome_info["source"] = merge_source(merge, x, self._tome_info["source"])
                x, size = merge_wavg(merge, x, self._tome_info["size"])

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
        x_ffn = x_out

        return x_ffn


class ToMeAttention(SelfAttention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

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


def apply_patch(model, trace_source: bool = False, prop_attn: bool = True):
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
            module._tome_info = model._tome_info
        elif isinstance(module, SelfAttention):
            module.__class__ = ToMeAttention
