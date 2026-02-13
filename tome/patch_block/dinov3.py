import torch
from torch import Tensor

from dinov3.layers.block import SelfAttentionBlock
from dinov3.models.vision_transformer import DinoVisionTransformer

from tome.merge import bipartite_soft_matching
from tome.utils import parse_r, PatchedDinov3, init_source_if_needed


def _build_patch_permutation(
    height: int | None,
    width: int | None,
    num_special_tokens: int,
    num_tokens: int,
    device: torch.device,
    *,
    use_column: bool,
):
    if not use_column:
        return None, None
    if height is None or width is None:
        return None, None
    patch_tokens = height * width
    if num_tokens != num_special_tokens + patch_tokens:
        return None, None

    idx = torch.arange(patch_tokens, device=device)
    idx = idx.view(height, width).t().reshape(-1)
    perm = torch.cat([torch.arange(num_special_tokens, device=device), num_special_tokens + idx])
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=device)
    return perm, inv_perm


class ToMeBlock(SelfAttentionBlock):
    def _forward_list(self, x_list: list[Tensor], rope_list=None) -> list[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        r = self._tome_info["r"].pop(0)
        layer_idx = self._tome_info.get("layer_idx", 0)
        self._tome_info["layer_idx"] = layer_idx + 1
        pairing = str(self._tome_info.get("pairing", "row")).lower()
        use_column = pairing == "column" or (pairing == "alternate" and (layer_idx % 2 == 1))

        merge_passes = max(int(self._tome_info.get("merge_passes", 1)), 1)
        x_out: list[Tensor] = []
        for x, rope in zip(x_list, rope_list):
            if r > 0:
                unmerge_stack: list[tuple] = []
                rope_local = rope
                for pass_idx in range(merge_passes):
                    perm = inv_perm = None
                    use_perm = use_column and pass_idx == 0
                    if use_perm:
                        perm, inv_perm = _build_patch_permutation(
                            self._tome_info.get("H"),
                            self._tome_info.get("W"),
                            self._tome_info["num_special_tokens"],
                            x.shape[-2],
                            x.device,
                            use_column=use_column,
                        )
                        use_perm = perm is not None

                    # Apply ToMe here (first pass uses pairing/perm, later passes are generic)
                    if self._tome_info.get("trace_source", False):
                        source = init_source_if_needed(x, self._tome_info.get("source"))
                        if use_perm and source is not None:
                            source = source.index_select(1, perm)
                    else:
                        source = None

                    if use_perm:
                        x = x.index_select(-2, perm)

                    merge, unmerge = bipartite_soft_matching(
                        x,
                        r,
                        self._tome_info["num_special_tokens"],
                    )

                    if self._tome_info.get("trace_source", False):
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
                    if use_perm:
                        sin_extended = sin_extended.index_select(1, perm)
                        cos_extended = cos_extended.index_select(1, perm)
                    sin = merge(sin_extended, mode="mean")
                    cos = merge(cos_extended, mode="mean")
                    sin = sin[:, None, self._tome_info["num_special_tokens"] :, :]
                    cos = cos[:, None, self._tome_info["num_special_tokens"] :, :]
                    rope_local = [sin, cos]

                    unmerge_stack.append((unmerge, use_perm, inv_perm))
            else:
                rope_local = rope

            x_norm1 = self.norm1(x)
            x_attn = self.attn(x_norm1, rope=rope_local)
            x = x + self.ls1(x_attn)

            x_norm2 = self.norm2(x)
            x_mlp = self.mlp(x_norm2)
            x = x + self.ls2(x_mlp)

            if r > 0:
                for unmerge, use_perm, inv_perm in reversed(unmerge_stack):
                    x = unmerge(x)

                    if self._tome_info.get("trace_source", False):
                        source = self._tome_info.get("source")
                        if source is not None:
                            source = unmerge(source)
                            if use_perm:
                                source = source.index_select(1, inv_perm)
                            self._tome_info["source"] = source

                    if use_perm:
                        x = x.index_select(-2, inv_perm)

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
            self._tome_info["layer_idx"] = 0
            _, _, height, width = x.shape
            patch_size = self.backbone.patch_size
            self._tome_info["H"] = height // patch_size
            self._tome_info["W"] = width // patch_size

            return super().forward(x, *args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model: DinoVisionTransformer,
    trace_source: bool = False,
    pairing: str = "alternate",
    merge_passes: int = 1,
):
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
        "pairing": pairing,
        "merge_passes": merge_passes,
    }

    for module in model.modules():
        if isinstance(module, SelfAttentionBlock):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
