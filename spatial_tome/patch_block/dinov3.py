import torch
from torch import Tensor

from dinov3.layers.block import SelfAttentionBlock
from dinov3.models.vision_transformer import DinoVisionTransformer

from spatial_tome.merge import spatial_soft_matching
from tome.merge import bipartite_soft_matching
from tome.utils import parse_r, PatchedDinov3, init_source_if_needed


class SpatialToMeBlock(SelfAttentionBlock):
    def _forward_list(self, x_list: list[Tensor], rope_list=None) -> list[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        r = self._tome_info["r"].pop(0)
        layer_idx = self._tome_info.get("layer_idx", 0)
        self._tome_info["layer_idx"] = layer_idx + 1

        merge_passes = max(int(self._tome_info.get("merge_passes", 1)), 1)
        x_out = []
        for x, rope in zip(x_list, rope_list):
            if r > 0:
                unmerge_stack = []
                rope_local = rope
                for pass_idx in range(merge_passes):
                    # First pass uses spatial matching; later passes fall back to generic matching.
                    if pass_idx == 0:
                        invert_mask = bool(self._tome_info.get("alternate_mask", False) and (layer_idx % 2 == 1))
                        merge, unmerge = spatial_soft_matching(
                            x,
                            self._tome_info["H"],
                            self._tome_info["W"],
                            r,
                            self._tome_info["num_special_tokens"],
                            invert_mask=invert_mask,
                        )
                    else:
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

                    sin, cos = rope_local
                    B = x.shape[0]
                    if sin.ndim == 4:
                        sin = sin.squeeze(1)
                    if cos.ndim == 4:
                        cos = cos.squeeze(1)
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

                    unmerge_stack.append(unmerge)
            else:
                rope_local = rope

            x_norm1 = self.norm1(x)
            x_attn = self.attn(x_norm1, rope=rope_local)
            x = x + self.ls1(x_attn)

            x_norm2 = self.norm2(x)
            x_mlp = self.mlp(x_norm2)
            x = x + self.ls2(x_mlp)

            if r > 0:
                for unmerge in reversed(unmerge_stack):
                    x = unmerge(x)

                    if self._tome_info.get("trace_source", False):
                        self._tome_info["source"] = unmerge(self._tome_info["source"])

            x_out.append(x)
        x_ffn = x_out

        return x_ffn


def make_spatial_tome_class(transformer_class):
    class SpatialToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.backbone.blocks), self.r)
            self._tome_info["size"] = None
            if self._tome_info.get("trace_source", False):
                self._tome_info["source"] = None
            self._tome_info["layer_idx"] = 0

            _, _, H, W = x.shape
            self._tome_info["H"] = H // self.backbone.patch_size
            self._tome_info["W"] = W // self.backbone.patch_size

            return super().forward(x, *args, **kwdargs)

    return SpatialToMeVisionTransformer


def apply_patch(
    model: DinoVisionTransformer,
    trace_source: bool = False,
    alternate_mask: bool = True,
    merge_passes: int = 1,
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    SpatialToMeVisionTransformer = make_spatial_tome_class(model.__class__)

    model.backbone.__class__ = PatchedDinov3
    model.__class__ = SpatialToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "num_special_tokens": model.backbone.n_storage_tokens + 1,
        "trace_source": trace_source,
        "source": None,
        "alternate_mask": alternate_mask,
        "merge_passes": merge_passes,
    }

    for module in model.modules():
        if isinstance(module, SelfAttentionBlock):
            module.__class__ = SpatialToMeBlock
            module._tome_info = model._tome_info
