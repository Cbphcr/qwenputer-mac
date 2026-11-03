"""Qwen's 32 layers of a merged checkpoint on MLX, through mlx-lm's own Qwen3.5 text model
(its Metal gated-delta kernel, fused attention and caches), on the residual the chip layers
leave (``input_embeddings``); text tokens only, so its caches and RoPE positions see the text
alone, as on the torch path.

Each linear map is its nonzero block (Qwen's layers read residual dims ``[0, mu_dim]`` and write
``[0, d0)``), the head the text rows over Qwen's dims plus the kept chip dims, as in
chat.load_qwen. Qwen's own text arithmetic is bf16, as in the checkpoint; it rounds as any
other kernel's does (the circuit's tokens do not depend on it).
"""

import copy

import torch

import mlx.core as mx
import mlx.nn as nn

from chipformer.q35.fast import SparseLinear
from macport.chat import Ckpt, _meta_model
from macport.mlx_chip import to_mx


class BlockLinear(nn.Module):
    """``x @ W.T`` for a W that is zero outside ``W[r0:r1, c0:c1]``."""

    def __init__(self, s: SparseLinear):
        super().__init__()
        self.weight = to_mx(s.weight.data)
        self.r0, self.r1, self.c0, self.c1 = s.r0, s.r1, s.c0, s.c1
        self.n = s.out_features

    def __call__(self, x):
        if self.r1 == self.r0 or self.c1 == self.c0:
            return mx.zeros((*x.shape[:-1], self.n), dtype=x.dtype)
        y = x[..., self.c0:self.c1] @ self.weight.T
        if self.r0 or self.r1 != self.n:
            y = mx.pad(y, [(0, 0)] * (y.ndim - 1) + [(self.r0, self.n - self.r1)])
        return y


class QBlockLinear(nn.Module):
    """BlockLinear with the block quantized (mx.quantize: affine, ``bits``, groups of 64 along
    the input; the block's columns padded with zeros to whole groups)."""

    def __init__(self, s: SparseLinear, bits: int, group: int = 64):
        super().__init__()
        w = s.weight.data
        n_in = -(-(s.c1 - s.c0) // group) * group
        c0 = min(s.c0, s.in_features - n_in)
        assert c0 >= 0
        full = torch.zeros(w.shape[0], n_in, dtype=w.dtype)
        full[:, s.c0 - c0:s.c1 - c0] = w
        self.wq, self.scales, self.biases = mx.quantize(to_mx(full), group_size=group, bits=bits)
        self.r0, self.r1, self.c0, self.c1 = s.r0, s.r1, c0, c0 + n_in
        self.n, self.bits, self.group = s.out_features, bits, group

    def __call__(self, x):
        y = mx.quantized_matmul(x[..., self.c0:self.c1], self.wq, self.scales, self.biases,
                                transpose=True, group_size=self.group, bits=self.bits)
        if self.r0 or self.r1 != self.n:
            y = mx.pad(y, [(0, 0)] * (y.ndim - 1) + [(self.r0, self.n - self.r1)])
        return y


class MlxQwen:
    def __init__(self, ck: Ckpt, cfg0, meta: dict, head: torch.Tensor, cols: torch.Tensor,
                 bits: int | None = None):
        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

        n_chip = meta["chip_layers"]
        types = cfg0.layer_types[n_chip:]
        fa = types.index("full_attention") + 1
        assert all((t == "full_attention") == ((i + 1) % fa == 0) for i, t in enumerate(types))
        args = TextModelArgs(
            model_type="qwen3_5_text", hidden_size=cfg0.hidden_size,
            intermediate_size=cfg0.intermediate_size, num_hidden_layers=len(types),
            num_attention_heads=cfg0.num_attention_heads, rms_norm_eps=cfg0.rms_norm_eps,
            vocab_size=1, num_key_value_heads=cfg0.num_key_value_heads,
            max_position_embeddings=cfg0.max_position_embeddings,
            linear_num_value_heads=cfg0.linear_num_value_heads,
            linear_num_key_heads=cfg0.linear_num_key_heads,
            linear_key_head_dim=cfg0.linear_key_head_dim,
            linear_value_head_dim=cfg0.linear_value_head_dim,
            linear_conv_kernel_dim=cfg0.linear_conv_kernel_dim, tie_word_embeddings=True,
            head_dim=cfg0.head_dim, full_attention_interval=fa,
            rope_parameters=dict(cfg0.rope_parameters))
        self.model = TextModel(args)
        self.bits = bits
        self.dense = self.compact = 0
        cfg = copy.deepcopy(cfg0)
        cfg.layer_types = types
        cfg.num_hidden_layers = len(types)
        layers = _meta_model(cfg, torch.bfloat16).model.layers
        for i, (tl, ml) in enumerate(zip(layers, self.model.model.layers)):
            self._load(ck, n_chip + i, tl, ml)
            layers[i] = torch.nn.Identity()  # free the torch copy
        self.model.model.norm.weight = to_mx(1.0 + ck.get("model.norm.weight").float()) \
            .astype(mx.bfloat16)
        self.model.model.embed_tokens = None
        # the head: Qwen's dims (quantized with ``bits``) and the kept chip dims (exact, bf16)
        d0 = meta["chip_dims_offset"]
        self.cols = to_mx(cols)
        self.head = to_mx(head)
        if bits is not None:
            self.hq = mx.quantize(to_mx(head[:, :d0].contiguous()), group_size=64, bits=bits)
            self.hkeep = to_mx(head[:, d0:].contiguous())
            self.head = None
        self.d0 = d0
        self.compact += head.numel()
        mx.eval(self.model.parameters(), *([self.head] if bits is None else
                                             [*self.hq, self.hkeep]))

    def _load(self, ck, src: int, tl, ml):
        """One torch decoder layer (dense on the CPU, then its nonzero blocks) into mlx-lm's."""
        tl.to_empty(device="cpu")
        pre = f"model.layers.{src}."
        tl.load_state_dict({k: ck.get(pre + k) for k in tl.state_dict()})
        mix_t = tl.linear_attn if hasattr(tl, "linear_attn") else tl.self_attn
        mix_m = ml.linear_attn if ml.is_linear else ml.self_attn
        for tmod, mmod in ((mix_t, mix_m), (tl.mlp, ml.mlp)):
            for name, child in list(tmod.named_children()):
                if isinstance(child, torch.nn.Linear):
                    s = SparseLinear(child.weight)
                    self.dense += child.weight.numel()
                    self.compact += s.weight.numel()
                    q = (BlockLinear(s) if self.bits is None or s.r1 == s.r0 or s.c1 == s.c0
                         else QBlockLinear(s, self.bits))
                    setattr(mmod, name, q)
        one = lambda w: to_mx(1.0 + w.float()).astype(mx.bfloat16)  # HF norms scale by 1 + w
        ml.input_layernorm.weight = one(tl.input_layernorm.weight)
        ml.post_attention_layernorm.weight = one(tl.post_attention_layernorm.weight)
        if ml.is_linear:
            la, ma = tl.linear_attn, ml.linear_attn
            ma.conv1d.weight = to_mx(la.conv1d.weight.data.permute(0, 2, 1).contiguous())
            ma.dt_bias = to_mx(la.dt_bias.data)
            ma.A_log = to_mx(la.A_log.data)
            ma.norm.weight = to_mx(la.norm.weight.data)
        else:
            sa, ms = tl.self_attn, ml.self_attn
            ms.q_norm.weight = one(sa.q_norm.weight)
            ms.k_norm.weight = one(sa.k_norm.weight)

    def new_cache(self):
        return self.model.make_cache()

    def logits(self, hc: mx.array, cache) -> mx.array:
        """Qwen's text logits after the residuals hc [T, hidden] (the last position)."""
        h = self.model.model(None, cache, input_embeddings=hc[None])[0, -1]
        x = h[self.cols]
        if self.bits is None:
            return x @ self.head.T
        wq, sc, bi = self.hq
        return (mx.quantized_matmul(x[None, : self.d0], wq, sc, bi, transpose=True,
                                    group_size=64, bits=self.bits)[0]
                + x[self.d0:] @ self.hkeep.T)

    def next_token(self, hc: mx.array, cache) -> int:
        return int(mx.argmax(self.logits(hc, cache)).item())
