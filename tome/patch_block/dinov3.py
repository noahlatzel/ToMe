import torch
from torch import Tensor

from dinov3.layers.block import SelfAttentionBlock
from dinov3.models.vision_transformer import DinoVisionTransformer

from tome.merge import bipartite_soft_matching
from tome.utils import parse_r, PatchedDinov3, init_source_if_needed


class ToMeBlock(SelfAttentionBlock):
    def _forward_list(self, x_list: list[Tensor], rope_list=None) -> list[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        r = self._tome_info["r"].pop(0)

        x_out: list[Tensor] = []
        for x, rope in zip(x_list, rope_list):
            if r > 0:
                # Apply ToMe here
                merge, unmerge = bipartite_soft_matching(
                    x,
                    r,
                    self._tome_info["num_special_tokens"],
                )

                if self._tome_info.get("trace_source", False):
                    source = init_source_if_needed(x, self._tome_info.get("source"))
                    source = merge(source, mode="mean")
                    self._tome_info["source"] = source

                x = merge(x, mode="mean")

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
                sin = merge(sin_extended, mode="mean")
                cos = merge(cos_extended, mode="mean")
                sin = sin[:, None, self._tome_info["num_special_tokens"] :, :]
                cos = cos[:, None, self._tome_info["num_special_tokens"] :, :]
                rope_local = [sin, cos]
            else:
                rope_local = rope

            x_norm1 = self.norm1(x)
            x_attn = self.attn(x_norm1, rope=rope_local)
            x = x + self.ls1(x_attn)

            x_norm2 = self.norm2(x)
            x_mlp = self.mlp(x_norm2)
            x = x + self.ls2(x_mlp)

            if r > 0:
                # Unmerge
                x = unmerge(x)

                if self._tome_info.get("trace_source", False):
                    source = self._tome_info.get("source")
                    if source is not None:
                        self._tome_info["source"] = unmerge(source)

            x_out.append(x)
        x_ffn = x_out

        return x_ffn


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.backbone.blocks), self.r)

            if self._tome_info.get("trace_source", False):
                self._tome_info["source"] = None

            return super().forward(x, *args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(model: DinoVisionTransformer, trace_source: bool = False):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If trace_source=True, the (unmerged) per-token source map will be available at:
        model._tome_info["source"]   # shape: [B, N, N]
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.backbone.__class__ = PatchedDinov3
    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "num_special_tokens": model.backbone.n_storage_tokens + 1,
        "trace_source": trace_source,
        "source": None,
    }

    for module in model.modules():
        if isinstance(module, SelfAttentionBlock):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
