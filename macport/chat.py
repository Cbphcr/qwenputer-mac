"""Qwenputer on Apple Silicon: the merged model's own layers on Metal, no CUDA.

    .venv/bin/python -m macport.chat /path/to/Qwen3.5-4B-Qwenputer \\
        --firmware upstream/experimental/riscv/tests/data/rv/calc-bridge.elf \\
        --ask "Use calc: what is 123456789 * 987654321?" --check

Decoding is upstream ``RvChat``'s (rv/chat.py), transparent to chip tokens: a chip token runs
through the 11 chip layers only, a text token through the chip layers and then Qwen's 32 layers
(with their own cache and RoPE positions, text tokens only). The chip layers run one of three
ways (``--chip``):

* ``neural`` (default): the checkpoint's chip layers, the firmware snapshot injected into their
  recurrent states (rv/hf.py): the model itself computing the RV32 machine, as upstream's fast
  path does on CUDA. ``--impl`` picks how (every one bit-exact against the circuit):
  ``hf`` the stock HF modules; ``sparse`` their linear maps from the nonzero weights only;
  ``lean`` plus a single-token decode step without the module plumbing; ``compiled`` that step
  through torch.compile; ``mlx`` the step in MLX (mx.compile); ``mlx-fused`` five fused Metal
  kernels per layer (mlx_kernels.py); ``mlx-inplace`` plus the delta rule in place on the state
  rows the step touches; ``mlx-freerun`` (default) plus runs of chip tokens decoded on the GPU,
  one host sync per chunk of tokens instead of per token.
* ``circuit``: the circuit those layers were compiled from, run by ``fastexec.Executor`` (the
  IR's arithmetic in fp32 on the CPU). The reference: after the chip layers a text token's
  residual is its embedding's first ``d0`` dims plus the kept dims (output code, tau, mu).
* ``verify``: both, every position's kept dims compared.

Qwen's layers run on mlx-lm's Qwen3.5 (``--backend mlx``, default; ``--qwen-bits 8|4``
quantizes them, never the chip layers) or as torch modules on MPS (``--backend torch``).

Every linear map is cut to its nonzero block (``fast.SparseLinear``; the chip layers' unused
value heads are pruned, ``fast.prune_heads``), the embedding to the column hulls of its text and
chip rows, and the tied output head to the text rows and the columns a text token's logits read.
The residual stays ``hidden_size`` wide, so every RMSNorm sees the same vector as in the merged
model.
"""

import argparse
import copy
import json
import resource
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from chipformer.q35 import fast as QF
from chipformer.q35.fast import SparseLinear, _hull, exact_matmuls
from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, fastexec, hf, ref
from chipformer.rv import spec as S
from chipformer.rv.chat import RvChat, replay


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def sync(dev) -> None:
    if str(dev).startswith("mps"):
        torch.mps.synchronize()


class Ckpt:
    """The checkpoint's tensors by name, read one at a time (or a slice of one)."""

    def __init__(self, path: Path):
        self.path = path
        self.where = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {}

    def _f(self, name):
        fn = self.where[name]
        if fn not in self.files:
            self.files[fn] = safe_open(self.path / fn, "pt")
        return self.files[fn]

    def get(self, name, sl=None) -> torch.Tensor:
        f = self._f(name)
        return f.get_slice(name)[sl] if sl is not None else f.get_tensor(name)

    def shape(self, name):
        return self._f(name).get_slice(name).get_shape()


@torch.no_grad()
def load_layers(layers, ck: Ckpt, first: int, device, dtype=torch.bfloat16, prune=False):
    """Fill meta decoder layers from checkpoint layers ``first..``, one at a time: dense on the
    CPU, unused value heads pruned (``prune``), every linear map cut to its nonzero block, then
    moved to ``device``. Returns (dense, compact) weight counts."""
    dense = compact = 0
    for i, layer in enumerate(layers):
        layer.to_empty(device="cpu")
        pre = f"model.layers.{first + i}."
        layer.load_state_dict({k: ck.get(pre + k) for k in layer.state_dict()})
        if prune and hasattr(layer, "linear_attn"):
            QF.prune_heads(layer.linear_attn)
        for parent in list(layer.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.Linear) and child.bias is None:
                    s = SparseLinear(child.weight)
                    s.weight = torch.nn.Parameter(s.weight.data.contiguous(), requires_grad=False)
                    dense += child.weight.numel()
                    compact += s.weight.numel()
                    setattr(parent, name, s)
        layer.to(device)
        layer.requires_grad_(False)
    if str(device).startswith("mps"):  # the allocator's cache from the loading
        torch.mps.empty_cache()
    return dense, compact


class TextHead(torch.nn.Module):
    """The tied output head on the text rows, reading only the columns ``cols``."""

    def __init__(self, w: torch.Tensor, cols: torch.Tensor):
        super().__init__()
        self.weight = w
        self.cols = cols

    def forward(self, x):
        return F.linear(x[..., self.cols], self.weight)


class Embedding:
    """The tied table as two column blocks: text rows ``[:v0]`` over columns ``[0, ct)`` and chip
    rows ``[v0:]`` over ``[cc0, cc1)`` (their column hulls; elsewhere every row is zero)."""

    def __init__(self, ck: Ckpt, v0: int, device, chip_rows: bool, dtype=torch.bfloat16,
                 chunk: int = 16384):
        name = "model.embed_tokens.weight"
        V, self.hidden = ck.shape(name)
        self.v0, self.dev = v0, device
        hull = lambda r0, r1, c0, c1: _hull(torch.cat([
            (ck.get(name, (slice(r, min(r + chunk, r1)), slice(c0, c1))) != 0).any(0)[None]
            for r in range(r0, r1, chunk)]).any(0))
        self.ct = hull(0, v0, 0, self.hidden)[1]
        self.text = self._read(ck, name, 0, v0, 0, self.ct, device, dtype, chunk)
        self.chip = None
        if chip_rows:
            a, b = hull(v0, V, 0, self.hidden)
            self.cc0, self.cc1 = a, b
            self.chip = self._read(ck, name, v0, V, a, b, device, dtype, chunk)

    @staticmethod
    def _read(ck, name, r0, r1, c0, c1, device, dtype, chunk):
        out = torch.empty(r1 - r0, c1 - c0, dtype=dtype, device=device)
        for r in range(r0, r1, chunk):
            rows = ck.get(name, (slice(r, min(r + chunk, r1)), slice(c0, c1)))
            out[r - r0:r - r0 + len(rows)] = rows.to(device, dtype)
        return out

    def numel(self) -> int:
        return self.text.numel() + (self.chip.numel() if self.chip is not None else 0)

    def __call__(self, ids: list[int]) -> torch.Tensor:
        """Full-width rows [T, hidden] for model ids."""
        out = torch.zeros(len(ids), self.hidden, dtype=self.text.dtype, device=self.dev)
        t = [x for x, i in enumerate(ids) if i < self.v0]
        c = [x for x, i in enumerate(ids) if i >= self.v0]
        lt = lambda a: torch.tensor(a, dtype=torch.long, device=self.dev)
        if t:
            out[lt(t), : self.ct] = self.text[lt([ids[x] for x in t])]
        if c:
            out[lt(c), self.cc0:self.cc1] = self.chip[lt([ids[x] - self.v0 for x in c])]
        return out


def _meta_model(cfg, dtype):
    from transformers import Qwen3_5ForCausalLM

    with torch.device("meta"):
        return Qwen3_5ForCausalLM(cfg).to(dtype)


@torch.no_grad()
def load_qwen(ck: Ckpt, cfg0, meta: dict, emb: Embedding, device, dtype=torch.bfloat16):
    """Qwen's layers as a stock HF text model (layers renumbered from 0); inputs_embeds only."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    n_chip = meta["chip_layers"]
    cfg = copy.deepcopy(cfg0)
    cfg.layer_types = cfg.layer_types[n_chip:]
    cfg.num_hidden_layers = len(cfg.layer_types)
    cfg.vocab_size = meta["chip_token_offset"]
    model = _meta_model(cfg, dtype)
    lm = model.model
    dense, compact = load_layers(lm.layers, ck, n_chip, device, dtype)
    lm.norm.weight = torch.nn.Parameter(ck.get("model.norm.weight").to(device, dtype),
                                        requires_grad=False)
    with torch.device(device):
        lm.rotary_emb = Qwen3_5TextRotaryEmbedding(config=cfg)
    # the head on the text rows: Qwen's dims, then the kept chip dims (a text token's logits read
    # the output code and tau there; the rest of the chip dims of the final residual are zero at
    # text positions)
    keep_cols = meta["code_dims"] + [meta["tau_dim"], meta["mu_dim"]]
    cols = torch.tensor(list(range(meta["chip_dims_offset"])) + keep_cols, device=device)
    assert int(cols.max()) < emb.ct
    model.lm_head = TextHead(emb.text[:, cols], cols)  # (a copy of those columns)
    lm.embed_tokens = torch.nn.Identity()
    left = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    assert not left, left[:10]
    return model.eval(), dense, compact + model.lm_head.weight.numel()


class NeuralChip:
    """The checkpoint's chip layers (stock HF decoder layers, compacted, heads pruned) with their
    own cache; the firmware snapshot is written into its recurrent states (hf.inject)."""

    def __init__(self, ck: Ckpt, cfg0, meta: dict, emb: Embedding, device,
                 dtype=torch.bfloat16):
        from transformers import DynamicCache

        self.n = meta["chip_layers"]
        cfg = copy.deepcopy(cfg0)
        cfg.layer_types = cfg.layer_types[: self.n]
        cfg.num_hidden_layers = self.n
        self.cfg, self.emb, self.dev = cfg, emb, device
        self.layers = _meta_model(cfg, dtype).model.layers
        self.dense, self.compact = load_layers(self.layers, ck, 0, device, dtype, prune=True)
        self._Cache = DynamicCache
        self.cache = None
        self.keep_cols = torch.tensor(meta["code_dims"] + [meta["tau_dim"], meta["mu_dim"]],
                                      device=device)

    def keep(self, hc) -> torch.Tensor:
        """A step's kept dims (output code, tau, mu) on the CPU, fp32."""
        return hc[self.keep_cols].float().cpu()

    def full(self, hc) -> torch.Tensor:
        """A step's whole residual on the model's device (for Qwen's layers)."""
        return hc

    def keep_rows(self, hs) -> torch.Tensor:
        """keep() of every row of start()'s residuals [T, hidden]."""
        return hs[:, self.keep_cols].float().cpu()

    def _run(self, h):
        for layer in self.layers:
            h = layer(h, position_embeddings=None, attention_mask=None, position_ids=None,
                      past_key_values=self.cache)
        return h

    @torch.no_grad()
    def start(self, c, ids: list[int]) -> torch.Tensor:
        """The prompt from zero states (one chunked prefill), then the snapshot now in the
        circuit ``c`` into the recurrent states. Returns the prompt's residuals [T, hidden]."""
        self.cache = self._Cache(config=self.cfg)
        h = self._run(self.emb(ids)[None])
        hf.inject(c, self.cache)
        return h[0]

    @torch.no_grad()
    def step(self, ids: int) -> torch.Tensor:
        """One token (the recurrent decode path); returns its residual [hidden]."""
        return self._run(self.emb([ids])[None])[0, -1]


class NZLinear(torch.nn.Module):
    """``F.linear(x, W)`` from W's nonzeros only: a gather, a product and a scatter-add in fp32,
    rounded to x's dtype once at the end. A bf16 product is exact in fp32 and the chip layers'
    sums are of exactly representable terms, so this equals the dense fp32-accumulated GEMM
    (any order of the sum is exact); it reads ~1M weights where the blocks hold ~185M."""

    def __init__(self, s: SparseLinear):
        super().__init__()
        self.out_features, self.in_features = s.out_features, s.in_features
        r, k = s.weight.nonzero(as_tuple=True)
        self.register_buffer("src", k + s.c0)
        self.register_buffer("dst", r + s.r0)
        self.register_buffer("w", s.weight[r, k].float())

    def forward(self, x):
        xf = x.reshape(-1, x.shape[-1]).float()
        y = torch.zeros(xf.shape[0], self.out_features, device=x.device)
        if len(self.w):
            y.index_add_(1, self.dst, xf[:, self.src] * self.w)
        return y.to(x.dtype).view(*x.shape[:-1], self.out_features)


def sparsify(layers) -> int:
    """Every SparseLinear in ``layers`` -> NZLinear, in place; returns the nonzero count."""
    n = 0
    for parent in list(layers.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, SparseLinear):
                nz = NZLinear(child)
                n += len(nz.w)
                setattr(parent, name, nz)
    return n


class SparseChip(NeuralChip):
    """NeuralChip whose linear maps read only their nonzero weights (NZLinear)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.compact = sparsify(self.layers)
        if str(self.dev).startswith("mps"):
            torch.mps.empty_cache()


def _cat_nz(mods) -> NZLinear:
    """One NZLinear for several maps of the same input, their outputs side by side."""
    out = NZLinear.__new__(NZLinear)
    torch.nn.Module.__init__(out)
    off, src, dst, w = 0, [], [], []
    for m in mods:
        src.append(m.src)
        dst.append(m.dst + off)
        w.append(m.w)
        off += m.out_features
    out.out_features, out.in_features = off, mods[0].in_features
    out.register_buffer("src", torch.cat(src))
    out.register_buffer("dst", torch.cat(dst))
    out.register_buffer("w", torch.cat(w))
    return out


class LeanChip(SparseChip):
    """SparseChip with its own single-token decode step: the same ops, in the same order and
    dtypes, as the HF decoder layer's cached decode path (causal_conv1d_update,
    torch_recurrent_gated_delta_rule, gated RMSNorm, SwiGLU), minus the module plumbing.

    * a layer's four input maps (qkv, z, b, a) are one gather / scatter-add, gate and up one;
    * repeat_interleave of q and k is an index into the key heads;
    * the conv and recurrent states are updated in place in the HF cache's own buffers;
    * a token mixer or MLP whose output map has no nonzero weight is skipped: it adds exact
      zeros to the residual (the state it would keep is never read).

    The chunked prefill stays the HF modules' (``start``)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.plan = []
        for layer in self.layers:
            la, mlp = layer.linear_attn, layer.mlp
            r = la.num_v_heads // la.num_k_heads
            self.plan.append({
                "ln1": layer.input_layernorm, "ln2": layer.post_attention_layernorm,
                "attn": len(la.out_proj.w) > 0, "mlp": len(mlp.down_proj.w) > 0,
                "inp": _cat_nz([la.in_proj_qkv, la.in_proj_z, la.in_proj_b, la.in_proj_a]),
                "split": [la.key_dim, la.key_dim, la.value_dim, la.value_dim,
                          la.num_v_heads, la.num_v_heads],
                "kidx": torch.arange(la.num_v_heads, device=self.dev) // r,
                "conv_w": la.conv1d.weight, "nv": la.num_v_heads, "dk": la.head_k_dim,
                "dv": la.head_v_dim, "A": -la.A_log.float().exp(), "dt_bias": la.dt_bias,
                "norm": la.norm, "out": la.out_proj,
                "gu": _cat_nz([mlp.gate_proj, mlp.up_proj]), "ni": mlp.gate_proj.out_features,
                "down": mlp.down_proj, "act": mlp.act_fn})

    @torch.no_grad()
    def start(self, c, ids: list[int]) -> torch.Tensor:
        """HF's chunked prefill, then the states copied into buffers kept across
        conversations (a compiled step keeps their addresses)."""
        h = super().start(c, ids)
        if not hasattr(self, "states"):
            self.states = [None] * len(self.plan)
        for L, p in enumerate(self.plan):
            lay = self.cache.layers[L]
            if not p["attn"]:
                continue
            cs, rs = lay.conv_states[0], lay.recurrent_states[0]
            if self.states[L] is None:
                self.states[L] = (torch.empty_like(cs), torch.empty_like(rs))
            self.states[L][0].copy_(cs)
            self.states[L][1].copy_(rs)
        return h

    def _mixer(self, L: int, p: dict, x):
        conv, S = self.states[L]
        y = p["inp"](x)  # [1, 1, ...] bf16
        qkv, z, b, a = torch.split(y, [p["split"][0] * 2 + p["split"][2], p["split"][3],
                                       p["nv"], p["nv"]], -1)
        # causal_conv1d_update
        win = torch.cat([conv, qkv.transpose(1, 2)], -1).to(p["conv_w"].dtype)
        conv.copy_(win[:, :, -conv.shape[-1]:])
        qkv = p["act"](F.conv1d(win, p["conv_w"], None, padding=0, groups=win.shape[1])
                       [:, :, -1:]).to(x.dtype).transpose(1, 2)
        dk, dv, nv = p["dk"], p["dv"], p["nv"]
        q, k, v = torch.split(qkv, p["split"][:3], -1)
        q = q.reshape(1, -1, dk)[:, p["kidx"]].float()
        k = k.reshape(1, -1, dk)[:, p["kidx"]].float()
        v = v.reshape(1, nv, dv).float()
        beta = b.sigmoid().reshape(1, nv).float()
        g = p["A"] * F.softplus(a.float() + p["dt_bias"])
        # torch_recurrent_gated_delta_rule, one step, in place on the cache's state
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = q / (dk ** 0.5)
        S.mul_(g.reshape(1, nv).exp()[..., None, None])
        kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        S.add_(k.unsqueeze(-1) * delta.unsqueeze(-2))
        o = (S * q.unsqueeze(-1)).sum(dim=-2).to(x.dtype)
        o = p["norm"](o.reshape(-1, dv), z.reshape(-1, dv)).reshape(1, 1, -1)
        return p["out"](o)

    @torch.no_grad()
    def step(self, ids: int) -> torch.Tensor:
        return self._body(self.emb([ids])[None])

    def _body(self, h):
        for L, p in enumerate(self.plan):
            if p["attn"]:
                h = h + self._mixer(L, p, p["ln1"](h))
            if p["mlp"]:
                gu = p["gu"](p["ln2"](h))
                h = h + p["down"](p["act"](gu[..., : p["ni"]]) * gu[..., p["ni"]:])
        return h[0, -1]


class CompiledChip(LeanChip):
    """LeanChip's decode step through torch.compile (inductor: Metal kernels on MPS, fused
    elementwise chains, one Python call per step); the cache states are mutated in place."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._fn = torch.compile(self._body, dynamic=False, fullgraph=False)

    @torch.no_grad()
    def step(self, ids: int) -> torch.Tensor:
        return self._fn(self.emb([ids])[None])


CHIP_IMPLS = {"hf": NeuralChip, "sparse": SparseChip, "lean": LeanChip,
              "compiled": CompiledChip, "mlx": None, "mlx-sparse": None,
              "mlx-fused": None, "mlx-inplace": None, "mlx-freerun": None}  # macport.mlx_chip


def make_chip(impl: str, ck: Ckpt, cfg, meta: dict, emb: Embedding, device):
    """The checkpoint's chip layers run by implementation ``impl`` (CHIP_IMPLS)."""
    if impl.startswith("mlx"):
        from macport import mlx_chip as X

        cls = {"mlx": X.MlxChip, "mlx-sparse": X.SparseMlxChip, "mlx-fused": X.FusedMlxChip,
               "mlx-inplace": X.InplaceMlxChip, "mlx-freerun": X.FreeRunMlxChip}
        return cls[impl](ck, cfg, meta, emb, device)
    return CHIP_IMPLS[impl](ck, cfg, meta, emb, device)


class CircuitChip:
    """The chip layers as the circuit itself: ``fastexec.Executor`` (batch 1, fp32)."""

    def __init__(self, c, v0: int, device: str = "cpu"):
        self.c, self.v0, self.dev = c, v0, device
        self.ex = fastexec.Executor(c, device)
        self.heads = [st for st in self.ex.steps if isinstance(st, dict)]
        self.keep = torch.tensor(c.keep, device=device)
        self.emb_cache: dict[int, torch.Tensor] = {}

    def emb(self, ids: list[int]) -> torch.Tensor:
        new = sorted({i for i in ids if i not in self.emb_cache})
        if new:
            rows = self.ex.embed([i - self.v0 if i >= self.v0 else BR.text_token(i) for i in new])
            for i, r in zip(new, rows):
                self.emb_cache[i] = r
        return torch.stack([self.emb_cache[i] for i in ids])

    @torch.no_grad()
    def start(self, c, ids: list[int]) -> torch.Tensor:
        """As the merged model's prefill: the prompt from zero states, then every head's state
        set to the snapshot now in the circuit, the lag windows kept. Returns the prompt's kept
        values [T, K] (on the CPU)."""
        for st in self.heads:
            st["S"].zero_()
        self.ex.reset()
        emb = self.emb(ids)
        out = torch.stack([self.ex.step(emb[t:t + 1])[0, self.keep] for t in range(len(ids))])
        self.ex.load_init()
        return out.cpu()

    @torch.no_grad()
    def step(self, ids: int) -> torch.Tensor:
        """One token; returns the kept values (fp32, on the CPU; the code first)."""
        return self.ex.step(self.emb([ids]))[0, self.keep].cpu()


class MacChat(RvChat):
    """RvChat for Apple Silicon. Reuses its prompt, output-code and replay helpers.

    ``backend="mlx"`` (default): Qwen's layers on mlx-lm's Qwen3.5 (macport.mlx_qwen) and the
    neural chip layers on MLX too (``impl`` mlx-*); torch only loads, on the CPU.
    ``backend="torch"``: Qwen's layers as torch modules on ``device`` (MPS)."""

    def __init__(self, path, device: str | None = None, chip: str = "neural",
                 exec_device: str = "cpu", impl: str = "mlx-freerun",
                 backend: str = "mlx", qwen_bits: int | None = None) -> None:
        from transformers import AutoTokenizer, Qwen3_5TextConfig

        from chipformer.q35.ir import Target

        exact_matmuls()
        assert chip in ("neural", "circuit", "verify") and backend in ("mlx", "torch")
        assert backend == "torch" or chip == "circuit" or impl.startswith("mlx"), \
            "the mlx backend runs the neural chip layers on MLX (--impl mlx-*)"
        self.mode, self.backend = chip, backend
        self.path = Path(path)
        self.meta = json.loads((self.path / "chipformer.json").read_text())
        self.dev = (device or default_device()) if backend == "torch" else "cpu"
        self.n_chip, self.v0 = self.meta["chip_layers"], self.meta["chip_token_offset"]
        self.d0 = self.meta["chip_dims_offset"]
        t0 = time.time()
        ck = Ckpt(self.path)
        cfg = Qwen3_5TextConfig.from_pretrained(self.path)
        self.hidden = cfg.hidden_size
        self.emb = Embedding(ck, self.v0, self.dev, chip_rows=chip != "circuit")
        keep = self.meta["code_dims"] + [self.meta["tau_dim"], self.meta["mu_dim"]]
        if backend == "torch":
            self.model, dq, cq = load_qwen(ck, cfg, self.meta, self.emb, self.dev)
        else:
            from macport.mlx_qwen import MlxQwen

            cols = torch.tensor(list(range(self.d0)) + keep)
            self.qwen = MlxQwen(ck, cfg, self.meta, self.emb.text[:, cols], cols, qwen_bits)
            dq, cq = self.qwen.dense, self.qwen.compact
        self.weights = {"qwen_dense": dq, "qwen_compact": cq, "embedding": self.emb.numel()}
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.T = BR.Text(self.path / "tokenizer.json")
        self.c = core.build(Target(cfg.linear_num_key_heads, cfg.linear_num_value_heads,
                                   cfg.intermediate_size), self.T)
        self.keep_cols = torch.tensor(keep, device=self.dev)
        assert len(self.keep_cols) == len(self.c.keep)
        self.n_code = len(self.meta["code_dims"])
        self.eos = {248046, 248044}
        self.neural = self.circuit = None
        if chip != "circuit":
            self.neural = make_chip(impl, ck, cfg, self.meta, self.emb, self.dev)
            self.weights.update(chip_dense=self.neural.dense, chip_compact=self.neural.compact)
            if impl.startswith("mlx"):  # the MLX chip has its own copies of the tables
                self.neural.out_torch = backend == "torch"
                self.emb.chip = None
                if chip == "neural":  # (the circuit's text residuals read the torch one)
                    self.emb.text = None
                if self.dev == "mps":
                    torch.mps.empty_cache()
        if chip != "neural":
            self.circuit = CircuitChip(self.c, self.v0, exec_device)
        del ck
        self._warm_up()
        sync(self.dev)
        self.load_seconds = time.time() - t0

    def _warm_up(self) -> None:
        """One short decode on an empty machine state: compiles the Metal kernels and the
        traced steps now, not in the first conversation."""
        ids = [198, 198]
        if self.circuit is not None:
            self.circuit.start(self.c, ids)
        if self.neural is not None:
            hs = self.neural.start(self.c, ids)
            for i in (198, self.v0):
                self.neural.keep(self.neural.step(i))
            if hasattr(self.neural, "free_run"):
                self.neural.free_run(self.v0, 64)
            self._qwen(hs)

    def _hc(self, ids: list[int], keep: torch.Tensor) -> torch.Tensor:
        """Text tokens' residuals after the chip layers from the circuit's kept values."""
        hc = torch.zeros(len(ids), self.hidden, dtype=torch.bfloat16, device=self.dev)
        hc[:, : self.d0] = self.emb.text[torch.tensor(ids, device=self.dev), : self.d0]
        hc[:, self.keep_cols] = keep.to(self.dev, torch.bfloat16)
        return hc

    @torch.no_grad()
    def _qwen(self, hc, cache=None):
        """Qwen's layers on text residuals hc [T, hidden]: (its argmax after the last, cache)."""
        if self.backend == "mlx":
            from macport.mlx_chip import to_mx

            if isinstance(hc, torch.Tensor):
                hc = to_mx(hc)
            cache = cache if cache is not None else self.qwen.new_cache()
            return self.qwen.next_token(hc, cache), cache
        out = self.model(inputs_embeds=hc[None], past_key_values=cache, use_cache=True,
                         logits_to_keep=1)
        return int(torch.argmax(out.logits[0, -1, : self.v0])), out.past_key_values

    @staticmethod
    def _lock_check(chat, lock, gen, nxt, expect):
        """The next token against the reference session: (lock_error or None, expect)."""
        st = nxt - chat.v0 if nxt >= chat.v0 else BR.text_token(nxt)
        if expect:
            e = expect.pop(0)
            if st != e:
                return {"at": len(gen) - 1,
                        "got": S.name(st) if st < S.VOCAB else chat.tok.decode([nxt]),
                        "want": S.name(e) if e < S.VOCAB else chat.tok.decode([e - S.TEXT_TOKEN]),
                        "recent": [S.name(t - chat.v0) if t >= chat.v0 else chat.tok.decode([t])
                                   for t in gen[-12:]]}, expect
            return None, expect
        if st < S.VOCAB:
            return {"at": len(gen) - 1, "got": S.name(st), "want": "text"}, expect
        k = len(lock.toks)
        try:
            lock.qwen([nxt])
        except RuntimeError as e:  # the reference run did not halt
            return {"at": len(gen) - 1, "got": str(e), "want": "a halt"}, expect
        return None, list(lock.toks[k + 1:])

    def _chip_step(self, nxt: int, stats: dict):
        """(the residual after the chip layers for a text token, else None; the next token the
        circuit owns, or None)."""
        keep = None
        if self.circuit is not None:
            keep = self.circuit.step(nxt)
        if self.neural is None:
            hc = self._hc([nxt], keep[None])[0] if nxt < self.v0 else None
            return hc, self.code_token(keep[: self.n_code])
        h = self.neural.step(nxt)
        k = self.neural.keep(h)
        if keep is not None:
            stats["verify_bad"] += not bool((k == keep.to(torch.bfloat16).float()).all())
        hc = self.neural.full(h) if nxt < self.v0 else None
        return hc, self.code_token(k[: self.n_code])

    @torch.no_grad()
    def generate(self, ids: list[int], machine: "ref.Machine", max_new: int = 4000,
                 max_chip: int = 60000, verbose: bool = False,
                 lock: "BR.Session | None" = None, on_token=None, **_) -> dict:
        """Greedy decoding after the prompt ``ids`` with the parked ``machine`` loaded (as
        RvChat.generate); ``on_token(id, owned)`` sees each token as it is generated, ``owned``
        whether the circuit (not Qwen) gave it."""
        tp = time.time()
        core.snapshot(self.c, machine)
        stats = {"verify_bad": 0}
        hcs = None
        if self.circuit is not None:
            keep = self.circuit.start(self.c, ids)
            hcs = self._hc(ids, keep)
        if self.neural is not None:
            nh = self.neural.start(self.c, ids)
            if hcs is not None:
                nk = self.neural.keep_rows(nh)
                stats["verify_bad"] += int((nk != keep.to(torch.bfloat16).float()).any(1).sum())
            hcs = nh
        nxt, cache = self._qwen(hcs)
        prefill = time.time() - tp
        gen, n_text, n_chip_tok = [], 0, 0
        t0 = time.time()
        t_chip = t_text = 0.0
        expect, lock_error = [], None
        # chip tokens decoded ahead by a free run on the GPU (FreeRunMlxChip; not in verify mode)
        free = self.circuit is None and hasattr(self.neural, "free_run")
        ahead = []
        owned = False  # whether the circuit (not Qwen) gave nxt
        while n_text < max_new and n_chip_tok < max_chip:
            gen.append(nxt)
            if on_token is not None:
                on_token(nxt, owned)
            if lock is not None:
                lock_error, expect = self._lock_check(self, lock, gen, nxt, expect)
                if lock_error is not None:
                    break
            if nxt in self.eos:
                break
            if verbose and len(gen) % 500 == 0:
                tail = [S.name(t - self.v0) if t >= self.v0 else repr(self.tok.decode([t]))
                        for t in gen[-8:]]
                print(f"  {len(gen)} tokens ({n_text} text, {n_chip_tok} chip) "
                      f"{time.time() - t0:.0f}s: {' '.join(tail)}", flush=True)
            ts = time.time()
            if free and nxt >= self.v0:
                if not ahead:
                    ahead = self.neural.free_run(nxt, max_chip - n_chip_tok)
                n_chip_tok += 1
                nxt, owned = ahead.pop(0), True
                if nxt < 0:
                    raise RuntimeError("the circuit gave up a chip position" if nxt == -2 else
                                       "the circuit predicts no token")
                t_chip += time.time() - ts
                continue
            hc, ct = self._chip_step(nxt, stats)
            if nxt >= self.v0:  # a chip token: the chip layers only
                n_chip_tok += 1
                if ct is None:
                    raise RuntimeError("the circuit gave up a chip position")
                nxt, owned = ct, True
                t_chip += time.time() - ts
            else:  # Qwen sees every text token, even where the circuit owns the next one
                n_text += 1
                q, _ = self._qwen(hc[None], cache)
                nxt, owned = (ct, True) if ct is not None else (q, False)
                t_text += time.time() - ts
        sync(self.dev)
        return {"tokens": gen, "text_tokens": n_text, "chip_tokens": n_chip_tok,
                "seconds": time.time() - t0, "prefill_seconds": round(prefill, 2),
                "prompt_tokens": len(ids), "lock_error": lock_error,
                "verify_mismatch": stats["verify_bad"] if self.mode == "verify" else None,
                "chip_seconds": round(t_chip, 2), "text_seconds": round(t_text, 2)}


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30  # bytes on macOS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--firmware", default="upstream/experimental/riscv/tests/data/rv/calc-bridge.elf")
    ap.add_argument("--ask", default="Use calc: what is 123456789 * 987654321?")
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--backend", default="mlx", choices=["mlx", "torch"])
    ap.add_argument("--device", default=None, help="torch backend: mps (default) or cpu")
    ap.add_argument("--qwen-bits", type=int, default=None, choices=[4, 8],
                    help="mlx backend: quantize Qwen's layers and head (not the chip layers)")
    ap.add_argument("--chip", default="neural", choices=["neural", "circuit", "verify"])
    ap.add_argument("--impl", default="mlx-freerun", choices=list(CHIP_IMPLS),
                    help="how the neural chip layers run (hf: the stock modules)")
    ap.add_argument("--exec-device", default="cpu", help="the circuit executor: cpu or mps")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    chat = MacChat(args.model, args.device, args.chip, args.exec_device, args.impl, args.backend,
                   args.qwen_bits)
    w = {k: f"{v / 1e9:.2f}B" for k, v in chat.weights.items()}
    print(f"[{args.backend} backend, loaded in {chat.load_seconds:.1f}s, chip layers: {args.chip}"
          f"{'' if args.chip == 'circuit' else ' (' + args.impl + ')'}; "
          f"weights {w}]", flush=True)
    m = ref.Machine.from_program(elf.program(args.firmware))
    BR.park(m)
    ids = chat.prompt_ids([{"role": "user", "content": args.ask}])
    m2 = ref.Machine.from_program(elf.program(args.firmware))
    BR.park(m2)
    lock = BR.Session(chat.T, m2)
    lock.history(ids)
    r = chat.generate(ids, m, args.max_new, verbose=True, lock=lock)
    print(chat.text_of(r["tokens"]))
    print(f"[prefill {r['prompt_tokens']} tokens in {r['prefill_seconds']:.1f}s; "
          f"{r['text_tokens']} text + {r['chip_tokens']} chip tokens in {r['seconds']:.1f}s "
          f"(text {r['text_tokens'] / max(r['text_seconds'], 1e-9):.1f} tok/s, "
          f"chip {r['chip_tokens'] / max(r['chip_seconds'], 1e-9):.0f} tok/s); "
          f"lock error {r['lock_error']}; verify mismatches {r['verify_mismatch']}; "
          f"peak RSS {peak_rss_gb():.1f} GB]")
    if args.check:
        print(replay(chat, ids, r["tokens"], args.firmware))


if __name__ == "__main__":
    main()
