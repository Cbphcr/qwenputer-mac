"""The checkpoint's chip layers on MLX: the HF decode step's arithmetic (as chat.LeanChip has
it) in MLX ops, one ``mx.compile``d function per token (MLX's stand-in for a CUDA graph: one
traced graph, fused elementwise kernels, no per-op Python dispatch).

The weights are the same tensors the torch path loads (compacted, heads pruned, nonzeros only);
the prompt runs token by token through the same step (the recurrent path), then the firmware
snapshot goes into the recurrent states (rv/hf.inject). Every float op follows HF's dtypes: bf16
where the module outputs bf16, fp32 inside the norms, the conv, the gates and the delta rule.
"""

import numpy as np
import torch

import mlx.core as mx

from chipformer.rv import hf
from macport.chat import LeanChip


def to_mx(t: torch.Tensor) -> mx.array:
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        return mx.array(t.view(torch.int16).numpy()).view(mx.bfloat16)
    return mx.array(t.numpy())


def to_torch(a: mx.array, device) -> torch.Tensor:
    if a.dtype == mx.bfloat16:
        return torch.from_numpy(np.array(a.view(mx.int16))).view(torch.bfloat16).to(device)
    return torch.from_numpy(np.array(a)).to(device)


class _Cache:
    """What hf.inject writes into: recurrent_states[0] per layer."""

    class _L:
        def __init__(self, S):
            self.recurrent_states = {0: S}

    def __init__(self, Ss):
        self.layers = [self._L(S) for S in Ss]


def _nz(m):
    return {"src": to_mx(m.src), "dst": to_mx(m.dst), "w": to_mx(m.w), "n": m.out_features}


def _lin(p, x):
    """NZLinear: gather, product, scatter-add in fp32; bf16 out. x [D] bf16."""
    y = mx.zeros((p["n"],), dtype=mx.float32)
    if p["w"].size:
        y = y.at[p["dst"]].add(x.astype(mx.float32)[p["src"]] * p["w"])
    return y.astype(mx.bfloat16)


def _rms(w1, x, eps):
    """Qwen3_5RMSNorm: fp32 norm times (1 + w), back to bf16."""
    xf = x.astype(mx.float32)
    out = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (out * w1).astype(mx.bfloat16)


def _silu_bf16(x):
    xf = x.astype(mx.float32)
    return (xf * mx.sigmoid(xf)).astype(mx.bfloat16)


def _softplus(x):  # torch's (beta 1, threshold 20)
    return mx.where(x > 20, x, mx.log1p(mx.exp(x)))


class MlxChip(LeanChip):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.eps = self.cfg.rms_norm_eps
        emb = self.emb
        self.v0, self.hidden = emb.v0, emb.hidden
        self.ct, self.cc0, self.cc1 = emb.ct, emb.cc0, emb.cc1
        self.mtext, self.mchip = to_mx(emb.text), to_mx(emb.chip)
        self.P = []
        for L, p in enumerate(self.plan):
            q = {"attn": p["attn"], "mlp": p["mlp"],
                 "ln1": to_mx(1.0 + p["ln1"].weight.float()),
                 "ln2": to_mx(1.0 + p["ln2"].weight.float()),
                 "gu": _nz(p["gu"]), "ni": p["ni"], "down": _nz(p["down"])}
            if p["attn"]:
                q.update(inp=_nz(p["inp"]), split=p["split"], kidx=to_mx(p["kidx"]),
                         conv_w=to_mx(p["conv_w"][:, 0].float()), nv=p["nv"], dk=p["dk"],
                         dv=p["dv"], A=to_mx(p["A"]), dt_bias=to_mx(p["dt_bias"].float()),
                         normw=to_mx(p["norm"].weight), neps=p["norm"].variance_epsilon,
                         out=_nz(p["out"]), K=p["conv_w"].shape[-1])
            self.P.append(q)
        # the torch chip layers are no longer needed
        del self.layers, self.plan
        self.cache = None
        if str(self.dev).startswith("mps"):
            torch.mps.empty_cache()
        self._fn = mx.compile(self._body)

    # --- one token ---------------------------------------------------------------------------
    def _mixer(self, q, x, conv, S):
        y = _lin(q["inp"], x)
        kd, vd, nv, dk, dv = q["split"][0], q["split"][2], q["nv"], q["dk"], q["dv"]
        n_qkv = 2 * kd + vd
        qkv, z = y[:n_qkv], y[n_qkv:n_qkv + vd]
        b, a = y[n_qkv + vd:n_qkv + vd + nv], y[n_qkv + vd + nv:]
        win = mx.concatenate([conv, qkv[:, None]], axis=1)  # [C, K + 1]
        conv = win[:, 1:]
        tap = (win[:, -q["K"]:].astype(mx.float32) * q["conv_w"]).sum(-1).astype(mx.bfloat16)
        qkv = _silu_bf16(tap)
        qq = qkv[:kd].reshape(-1, dk)[q["kidx"]].astype(mx.float32)
        kk = qkv[kd:2 * kd].reshape(-1, dk)[q["kidx"]].astype(mx.float32)
        vv = qkv[2 * kd:].reshape(nv, dv).astype(mx.float32)
        beta = mx.sigmoid(b.astype(mx.float32)).astype(mx.bfloat16).astype(mx.float32)
        g = q["A"] * _softplus(a.astype(mx.float32) + q["dt_bias"])
        qq = qq * mx.rsqrt((qq * qq).sum(-1, keepdims=True) + 1e-6)
        kk = kk * mx.rsqrt((kk * kk).sum(-1, keepdims=True) + 1e-6)
        qq = qq / (dk ** 0.5)
        S = S * mx.exp(g)[:, None, None]
        kv_mem = (S * kk[:, :, None]).sum(axis=-2)
        delta = (vv - kv_mem) * beta[:, None]
        S = S + kk[:, :, None] * delta[:, None, :]
        o = (S * qq[:, :, None]).sum(axis=-2).astype(mx.bfloat16)  # [nv, dv]
        # Qwen3_5RMSNormGated
        of = o.astype(mx.float32)
        of = of * mx.rsqrt(mx.mean(of * of, axis=-1, keepdims=True) + q["neps"])
        o = q["normw"] * of.astype(mx.bfloat16)
        zf = z.reshape(nv, dv).astype(mx.float32)
        o = (o * (zf * mx.sigmoid(zf))).astype(mx.bfloat16)
        return _lin(q["out"], o.reshape(-1)), conv, S

    def _body(self, h, *states):
        out = []
        i = 0
        for q in self.P:
            if q["attn"]:
                m, conv, S = self._mixer(q, _rms(q["ln1"], h, self.eps), states[i], states[i + 1])
                h = h + m
                out += [conv, S]
                i += 2
            if q["mlp"]:
                gu = _lin(q["gu"], _rms(q["ln2"], h, self.eps))
                h = h + _lin(q["down"], _silu_bf16(gu[: q["ni"]]) * gu[q["ni"]:])
        return (h, *out)

    def _emb(self, i: int) -> mx.array:
        if i < self.v0:
            return mx.pad(self.mtext[i], (0, self.hidden - self.ct))
        return mx.pad(self.mchip[i - self.v0], (self.cc0, self.hidden - self.cc1))

    def _step(self, i: int) -> mx.array:
        h, *st = self._fn(self._emb(i), *self.states)
        self.states = st
        return h

    # --- the chip interface -------------------------------------------------------------------
    def start(self, c, ids: list[int]) -> torch.Tensor:
        self.states = []
        for q in self.P:
            if q["attn"]:
                C = 2 * q["split"][0] + q["split"][2]
                self.states += [mx.zeros((C, q["K"]), dtype=mx.bfloat16),
                                mx.zeros((q["nv"], q["dk"], q["dv"]), dtype=mx.float32)]
        hs = []
        for i in ids:
            hs.append(self._step(i))
            mx.eval(hs[-1], *self.states)
        Ss = [torch.zeros(1, *self.states[j].shape) for j in range(1, len(self.states), 2)]
        attn = [L for L, q in enumerate(self.P) if q["attn"]]
        full = [torch.zeros(1, 32, 128, 128) for _ in self.P]  # (skipped mixers: not read)
        for L, S in zip(attn, Ss):
            full[L] = S
        hf.inject(c, _Cache(full))
        for n, S in enumerate(Ss):
            self.states[2 * n + 1] = to_mx(S[0])
        hs = mx.stack(hs)
        return to_torch(hs, self.dev) if self.out_torch else hs

    out_torch = True  # start / full give torch tensors on self.dev (else MLX arrays)

    def step(self, ids: int) -> mx.array:
        return self._step(ids)

    def keep(self, h) -> torch.Tensor:
        if not hasattr(self, "mkeep"):
            self.mkeep = to_mx(self.keep_cols)
        return torch.from_numpy(np.array(h[self.mkeep].astype(mx.float32)))

    def full(self, h):
        return to_torch(h, self.dev) if self.out_torch else h

    def keep_rows(self, hs) -> torch.Tensor:
        if isinstance(hs, torch.Tensor):
            return hs[:, self.keep_cols.to(hs.device)].float().cpu()
        return torch.from_numpy(np.array(hs[:, to_mx(self.keep_cols)].astype(mx.float32)))


class SparseMlxChip(MlxChip):
    """MlxChip whose delta rule reads and writes only the state rows the step can touch, so a
    32-head layer moves a few KB of its 2 MB state instead of sweeping it ~5 times.

    The circuit's keys are one-hot and its queries at most ``qhot``-hot (q35.ir.Head), and a
    head with no ``a`` weights has exp(g) == 1.0 exactly. Hence, per head:

    * ``S * exp(g)`` is S itself unless the head can decay (those are scaled in full);
    * ``kv_mem = sum_d S[d] k[d]`` is its one nonzero term (the rest are exact zeros);
    * the rank-1 write changes only the key's row;
    * the read ``sum_d S[d] q[d]`` has at most ``qhot`` nonzero terms.

    Each result is the same float as the dense sweep's. The step also returns whether any head
    had more nonzero key / query dims than that (then the shortcut would not hold): ``step``
    raises if so."""

    def start(self, c, ids):
        if not hasattr(self, "kq"):
            from chipformer.q35.ir import Head

            attn = [L for L, q in enumerate(self.P) if q["attn"]]
            for L, q in enumerate(self.P):
                if not q["attn"]:
                    continue
                heads = [n for n in c.nodes if isinstance(n, Head) and n.level == 2 * L + 1]
                q["Kq"] = max([getattr(n, "qhot", 2) for n in heads] or [2])
                n0 = 2 * q["split"][0] + 2 * q["split"][2] + q["nv"]
                dst = set(np.array(q["inp"]["dst"]).tolist())
                q["dec"] = mx.array([j for j in range(q["nv"]) if n0 + j in dst])
            self.kq = attn
            self._fn = mx.compile(self._body)
        return super().start(c, ids)

    def _mixer(self, q, x, conv, S):
        y = _lin(q["inp"], x)
        kd, vd, nv, dk, dv = q["split"][0], q["split"][2], q["nv"], q["dk"], q["dv"]
        n_qkv = 2 * kd + vd
        qkv, z = y[:n_qkv], y[n_qkv:n_qkv + vd]
        b, a = y[n_qkv + vd:n_qkv + vd + nv], y[n_qkv + vd + nv:]
        win = mx.concatenate([conv, qkv[:, None]], axis=1)
        conv = win[:, 1:]
        tap = (win[:, -q["K"]:].astype(mx.float32) * q["conv_w"]).sum(-1).astype(mx.bfloat16)
        qkv = _silu_bf16(tap)
        qq = qkv[:kd].reshape(-1, dk)[q["kidx"]].astype(mx.float32)
        kk = qkv[kd:2 * kd].reshape(-1, dk)[q["kidx"]].astype(mx.float32)
        vv = qkv[2 * kd:].reshape(nv, dv).astype(mx.float32)
        beta = mx.sigmoid(b.astype(mx.float32)).astype(mx.bfloat16).astype(mx.float32)
        qq = qq * mx.rsqrt((qq * qq).sum(-1, keepdims=True) + 1e-6)
        kk = kk * mx.rsqrt((kk * kk).sum(-1, keepdims=True) + 1e-6)
        qq = qq / (dk ** 0.5)
        Kq = q["Kq"]
        bad = ((kk != 0).sum(-1) > 1).any() | ((qq != 0).sum(-1) > Kq).any()
        if q["dec"].size:
            g = q["A"] * _softplus(a.astype(mx.float32) + q["dt_bias"])
            d = q["dec"]
            S[d] = S[d] * mx.exp(g[d])[:, None, None]
        ki = mx.argmax(mx.abs(kk), axis=-1)[:, None]  # [nv, 1]
        kv = mx.take_along_axis(kk, ki, axis=-1)  # [nv, 1]
        krow = mx.take_along_axis(S, ki[:, :, None], axis=1)  # [nv, 1, dv]
        kv_mem = krow[:, 0] * kv
        delta = (vv - kv_mem) * beta[:, None]
        krow = krow + kv[:, :, None] * delta[:, None, :]
        S = mx.put_along_axis(S, ki[:, :, None], krow, axis=1)
        qi = mx.argpartition(-mx.abs(qq), kth=Kq - 1, axis=-1)[:, :Kq] if Kq < dk else \
            mx.broadcast_to(mx.arange(dk), (nv, dk))
        qv = mx.take_along_axis(qq, qi, axis=-1)
        qrows = mx.take_along_axis(S, qi[:, :, None], axis=1)  # [nv, Kq, dv]
        o = (qrows * qv[:, :, None]).sum(axis=1).astype(mx.bfloat16)
        of = o.astype(mx.float32)
        of = of * mx.rsqrt(mx.mean(of * of, axis=-1, keepdims=True) + q["neps"])
        o = q["normw"] * of.astype(mx.bfloat16)
        zf = z.reshape(nv, dv).astype(mx.float32)
        o = (o * (zf * mx.sigmoid(zf))).astype(mx.bfloat16)
        return _lin(q["out"], o.reshape(-1)), conv, S, bad

    def _body(self, h, *states):
        out, bad = [], mx.array(False)
        i = 0
        for q in self.P:
            if q["attn"]:
                m, conv, S, b = self._mixer(q, _rms(q["ln1"], h, self.eps), states[i],
                                            states[i + 1])
                h = h + m
                bad = bad | b
                out += [conv, S]
                i += 2
            if q["mlp"]:
                gu = _lin(q["gu"], _rms(q["ln2"], h, self.eps))
                h = h + _lin(q["down"], _silu_bf16(gu[: q["ni"]]) * gu[q["ni"]:])
        return (h, bad, *out)

    def _step(self, i: int) -> mx.array:
        h, bad, *st = self._fn(self._emb(i), *self.states)
        self.states = st
        if bool(bad):
            raise RuntimeError("a key or query had more nonzero dims than the circuit allows")
        return h


class FusedMlxChip(MlxChip):
    """MlxChip with five fused Metal kernels per layer (macport.mlx_kernels): rmsnorm + the
    merged input map, the Gated DeltaNet step, the output map + residual, rmsnorm + gate / up +
    SwiGLU, the down map + residual."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        from macport import mlx_kernels as MK

        self.MK = MK

        def tocsr(p):
            return (*MK.csr(np.array(p["src"]), np.array(p["dst"]), np.array(p["w"]), p["n"]),
                    p["n"])

        longrow = lambda p: p["w"].size and np.bincount(np.array(p["dst"])).max() > 32
        for q in self.P:
            q["gu_csr"], q["down_csr"] = tocsr(q["gu"]), tocsr(q["down"])
            q["down_simd"] = bool(longrow(q["down"]))
            if q["attn"]:
                q["out_simd"] = bool(longrow(q["out"]))
            q["gu_lpr"] = MK.lanes(q["gu_csr"][0])
            if q["attn"]:
                q["inp_csr"], q["out_csr"] = tocsr(q["inp"]), tocsr(q["out"])
                q["inp_lpr"] = MK.lanes(q["inp_csr"][0])
                q["convw_k"] = q["conv_w"]  # [C, K] fp32
        self._fn = mx.compile(self._body)

    def _body(self, h, *states):
        MK, out, i = self.MK, [], 0
        for q in self.P:
            if q["attn"]:
                y = MK.norm_csr(h, q["ln1"], q["inp_csr"], self.eps, lpr=q["inp_lpr"])
                o, conv, S = MK.gdn(y, states[i], q["convw_k"], states[i + 1], q["A"],
                                    q["dt_bias"], q["normw"], kd=q["split"][0],
                                    vd=q["split"][2], nv=q["nv"], r=q["nv"] // (q["split"][0] // q["dk"]),
                                    dk=q["dk"], dv=q["dv"], K=q["K"], neps=q["neps"])
                h = MK.csr_res(o, h, q["out_csr"], q["out_simd"])
                out += [conv, S]
                i += 2
            if q["mlp"]:
                m = MK.norm_csr(h, q["ln2"], q["gu_csr"], self.eps, swiglu=True,
                              lpr=q["gu_lpr"])
                h = MK.csr_res(m, h, q["down_csr"], q["down_simd"])
        return (h, *out)


class InplaceMlxChip(FusedMlxChip):
    """FusedMlxChip whose Gated DeltaNet kernel updates the recurrent states in place and only
    on the rows the step reads or writes (mlx_kernels.gdn_inplace): a 32-head layer no longer
    copies its 2 MB state every token. The states are this object's own arrays."""

    def start(self, c, ids):
        h = super().start(c, ids)
        # materialized, contiguous copies (the kernel writes into them)
        self.states = [mx.array(np.array(s)) if s.dtype == mx.float32 else s
                       for s in self.states]
        return h

    live = mx.array([True])

    def _body(self, h, *states):
        return self._layers(h, self.live, states)

    def _layers(self, h, live, states):
        """The step; with ``live`` false it leaves the recurrent states as they are."""
        MK, out, i = self.MK, [], 0
        for q in self.P:
            if q["attn"]:
                y = MK.norm_csr(h, q["ln1"], q["inp_csr"], self.eps, lpr=q["inp_lpr"])
                o, conv = MK.gdn_inplace(y, states[i], q["convw_k"], states[i + 1], q["A"],
                                         q["dt_bias"], q["normw"], live, kd=q["split"][0],
                                         vd=q["split"][2], nv=q["nv"],
                                         r=q["nv"] // (q["split"][0] // q["dk"]), dk=q["dk"],
                                         dv=q["dv"], K=q["K"], neps=q["neps"])
                h = MK.csr_res(o, h, q["out_csr"], q["out_simd"])
                out += [conv, states[i + 1]]
                i += 2
            if q["mlp"]:
                m = MK.norm_csr(h, q["ln2"], q["gu_csr"], self.eps, swiglu=True,
                              lpr=q["gu_lpr"])
                h = MK.csr_res(m, h, q["down_csr"], q["down_simd"])
        return (h, *out)



def code_lut(T, v0: int) -> np.ndarray:
    """The model token of every output code (as RvChat.code_token / core.decode_code), indexed by
    the code's bits (family id bits, then payload bits): -1 where the code names no token."""
    from chipformer.rv import core
    from chipformer.rv import spec as S

    nf = core.NFAM
    lut = np.full(1 << (nf + core.NPAY), -1, np.int32)
    p = np.arange(1 << core.NPAY)
    for fid, nm in enumerate(core.OUT_NAMES):
        idx = fid + (p << nf)
        if nm == "Q":
            ok = p < len(T.printable)
            lut[idx[ok]] = np.array(T.printable, np.int32)[p[ok]]
        elif nm in S.CONTROLS:
            lut[idx] = v0 + S.CONTROLS[nm]
        else:
            base, size = S.FAMILIES[nm]
            ok = p < size
            lut[idx[ok]] = v0 + base + p[ok]
    return lut


class FreeRunMlxChip(InplaceMlxChip):
    """InplaceMlxChip that also decodes runs of chip tokens on the GPU (upstream's FreeRun, for
    the model's own layers): each step reads the next token off its output code dims (a lookup
    table of decode_code) and embeds it for the next step, so a run of chip tokens is queued as
    one lazy graph and the host syncs once per chunk, not once per token.

    A ``live`` flag goes down at the first token the chip layers do not own (text, or a code
    that names nothing); the steps after it leave the states untouched (the Gated DeltaNet
    kernel returns early; the conv states are passed through), so the run can be queued ahead
    and the host reads where it ended."""

    max_chunk = 32

    def start(self, c, ids):
        if not hasattr(self, "lut"):
            from chipformer.rv import core

            code = self.keep_cols[: core.NFAM + core.NPAY]
            self.lut = mx.array(code_lut(c.text, self.v0))
            self.mcode = to_mx(code.to(torch.int32))
            self.pow2 = mx.array([1 << i for i in range(len(code))], dtype=mx.int32)
            self._free = mx.compile(self._free_step)
        return super().start(c, ids)

    def _free_step(self, tok, live, *states):
        e = self.mchip[mx.maximum(tok - self.v0, 0)][0]
        h, *out = self._layers(mx.pad(e, (self.cc0, self.hidden - self.cc1)), live, states)
        code = h[self.mcode]
        idx = ((code > 0).astype(mx.int32) * self.pow2).sum(keepdims=True)
        nt = mx.where((code != 0).any(), self.lut[idx], -2)  # -2: an all-zero code (no token)
        nt = mx.where(live, nt, tok)
        return (nt, live & (nt >= self.v0), *out)

    def free_run(self, nxt: int, limit: int) -> list[int]:
        """The tokens after the chip token ``nxt`` and each chip token that follows it, up to and
        including the first one the chip layers do not own (or ``limit`` tokens); the states
        end after the last chip token's step. -1 / -2: a code naming no token / an all-zero
        code."""
        tok, live = mx.array([nxt], dtype=mx.int32), mx.array([True])
        out, pend, n = [], None, 2
        while True:
            toks = []
            for _ in range(n):
                tok, live, *self.states = self._free(tok, live, *self.states)
                toks.append(tok)
            blk = mx.concatenate(toks)
            mx.async_eval(blk, *self.states)
            if pend is not None:  # read the previous chunk while this one runs
                for t in np.array(pend).tolist():
                    out.append(t)
                    if t < self.v0 or len(out) >= limit:
                        return out
            pend, n = blk, min(2 * n, self.max_chunk)
