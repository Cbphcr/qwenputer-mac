"""Fused Metal kernels (``mx.fast.metal_kernel``) for the chip layers' decode step: five
kernels per layer instead of ~30 MLX ops, the role upstream's Triton executor v2 ("one fused
kernel per sublayer", docs/rv32.md) plays on CUDA, here for the model's own layers.

* ``norm_csr``: Qwen3_5RMSNorm fused into the input of a sparse linear map (CSR over its
  nonzero weights, one thread per output row, fp32 sums, no atomics), with an epilogue: plain
  bf16 out, or SwiGLU (``silu(gate) * up``) for a merged gate / up map;
* ``csr_res``: a sparse linear map plus the residual add (the token mixer's and the MLP's
  output maps);
* ``gdn``: a Gated DeltaNet decode step, one threadgroup per value head: causal conv + SiLU,
  q / k l2norm, beta and the decay gate, the delta rule on the head's state, gated RMSNorm.

Every float op keeps HF's order and dtypes (bf16 roundings where the modules output bf16, fp32
inside); IEEE ("safe") math. The sums in the linear maps are exact in any order; the norms'
reductions are inexact and only reordered, as any other kernel would (checked: rv32 traces,
every kept dim at every step, against the circuit).
"""

import numpy as np

import mlx.core as mx

TG = 256  # threads per group of the linear kernels

_HDR = """
inline float bf(float x) { return float(static_cast<bfloat16_t>(x)); }  // to bf16 and back
inline float silu(float x) { return x / (1.0f + metal::exp(-x)); }
inline float log1p_(float u) {  // accurate log(1 + u) (Metal has no log1p)
    float w = 1.0f + u;
    return w == 1.0f ? u : metal::log(w) * (u / (w - 1.0f));
}
inline float sigm(float x) { return 1.0f / (1.0f + metal::exp(-x)); }

// sum over the threadgroup (tg threads, a multiple of 32); every thread gets it
inline float tg_sum(float v, threadgroup float* buf, uint tid, uint tg) {
    v = simd_sum(v);
    if (tid % 32 == 0) buf[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint i = 0; i < tg / 32; ++i) s += buf[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}
"""

_NORM = """
    uint tid = thread_position_in_threadgroup.x;
    uint tg = threads_per_threadgroup.x;
    uint o = thread_position_in_grid.x;
    threadgroup float buf[32];
    float ss = 0.0f;
    for (uint i = tid; i < D; i += tg) { float x = float(h[i]); ss += x * x; }
    ss = tg_sum(ss, buf, tid, tg);
    float rinv = metal::rsqrt(ss / float(D) + {eps}f);
"""

_ROW = """
template <typename RP, typename CO, typename VA, typename HH, typename WW>
inline float row(uint r, RP rp, CO col, VA val, HH h, WW w1, float rinv) {
    float acc = 0.0f;
    for (int j = rp[r]; j < rp[r + 1]; ++j) {
        int c = col[j];
        acc += bf(float(h[c]) * rinv * w1[c]) * val[j];
    }
    return acc;
}
// row r split over LPR lanes of a simdgroup (lane l takes nonzeros l, l + LPR, ...; exact terms,
// so the split sums to the same float); every lane of the simdgroup must call it
template <uint LPR, typename RP, typename CO, typename VA, typename HH, typename WW>
inline float row_lanes(uint r, bool ok, uint l, RP rp, CO col, VA val, HH h, WW w1, float rinv) {
    float acc = 0.0f;
    if (ok)
        for (int j = rp[r] + l; j < rp[r + 1]; j += LPR) {
            int c = col[j];
            acc += bf(float(h[c]) * rinv * w1[c]) * val[j];
        }
    for (uint m = LPR / 2; m > 0; m /= 2) acc += simd_shuffle_xor(acc, m);
    return acc;
}
template <typename RP, typename CO, typename VA, typename XX>
inline float row_raw(uint r, RP rp, CO col, VA val, XX x) {
    float acc = 0.0f;
    for (int j = rp[r]; j < rp[r + 1]; ++j) acc += float(x[col[j]]) * val[j];
    return acc;
}
"""

_SRC = {}
_SRC["norm_csr"] = (["h", "w1", "rp", "col", "val"], ["y"], _HDR + _ROW, _NORM + """
    uint r = o / LPR, l = o % LPR;
    float acc = row_lanes<LPR>(r, r < N, l, rp, col, val, h, w1, rinv);
    if (r < N && l == 0) y[r] = static_cast<bfloat16_t>(acc);
""")
_SRC["norm_csr_swiglu"] = (["h", "w1", "rp", "col", "val"], ["y"], _HDR + _ROW, _NORM + """
    uint r = o / LPR, l = o % LPR;
    float g = bf(row_lanes<LPR>(r, r < N, l, rp, col, val, h, w1, rinv));
    float u = bf(row_lanes<LPR>(r + N, r < N, l, rp, col, val, h, w1, rinv));
    if (r < N && l == 0) y[r] = static_cast<bfloat16_t>(bf(silu(g)) * u);
""")
_SRC["csr_res"] = (["x", "res", "rp", "col", "val"], ["y"], _HDR + _ROW, """
    uint o = thread_position_in_grid.x;
    if (o >= N) return;
    y[o] = static_cast<bfloat16_t>(float(res[o]) + bf(row_raw(o, rp, col, val, x)));
""")
_SRC["gdn"] = (["y", "conv", "convw", "S", "A", "dtb", "normw"], ["o", "conv_out", "S_out"],
               _HDR, """
    // one threadgroup per value head h, thread t = one of the 128 key / value dims
    uint h = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    threadgroup float buf[32];
    threadgroup float qs[DK];
    threadgroup float ks[DK];
    uint kh = h / R;
    uint C = 2 * KD + VD;
    // causal_conv1d_update on one channel: the window is the state's last K-1 values then the
    // new one; the conv sums in fp32, its output is bf16, silu(bf16) is bf16
    float x3[3];
    uint chans[3] = {kh * DK + t, KD + kh * DK + t, 2 * KD + h * DV + t};
    for (uint w = 0; w < 3; ++w) {
        uint c = chans[w];
        float acc = 0.0f;
        for (uint j = 0; j < K; ++j) {
            float v = (j + 1 < K) ? float(conv[c * K + j + 1]) : float(y[c]);
            acc += v * convw[c * K + j];
            if (w == 2 || h % R == 0) conv_out[c * K + j] = static_cast<bfloat16_t>(v);
        }
        x3[w] = bf(silu(bf(acc)));
    }
    float q = x3[0], k = x3[1], v = x3[2];
    // l2norm (fp32), then the 1/sqrt(dk) query scale
    float sq = tg_sum(q * q, buf, t, DK);
    float sk = tg_sum(k * k, buf, t, DK);
    q = q * metal::rsqrt(sq + 1e-6f);
    k = k * metal::rsqrt(sk + 1e-6f);
    q = q / {qscale}f;
    qs[t] = q;
    ks[t] = k;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float beta = bf(sigm(float(y[C + VD + h])));
    float a = float(y[C + VD + NV + h]) + dtb[h];
    float sp = a > 20.0f ? a : log1p_(metal::exp(a));
    float decay = metal::exp(A[h] * sp);
    // the delta rule on column t of the head's state
    auto Sh = S + h * DK * DV;
    device float* So = S_out + h * DK * DV;
    float kv = 0.0f;
    for (uint d = 0; d < DK; ++d) kv += (Sh[d * DV + t] * decay) * ks[d];
    float delta = (v - kv) * beta;
    float acc = 0.0f;
    for (uint d = 0; d < DK; ++d) {
        float s = Sh[d * DV + t] * decay + ks[d] * delta;
        So[d * DV + t] = s;
        acc += s * qs[d];
    }
    // Qwen3_5RMSNormGated over the head's dv
    float of = bf(acc);
    float var = tg_sum(of * of, buf, t, DV) / float(DV);
    of = of * metal::rsqrt(var + {neps}f);
    float hn = bf(float(normw[t]) * bf(of));
    float z = float(y[C + h * DV + t]);
    o[h * DV + t] = static_cast<bfloat16_t>(hn * silu(z));
""")
_SRC["gdn_inplace"] = (["y", "conv", "convw", "S", "A", "dtb", "normw", "live"],
                       ["o", "conv_out"], _HDR, """
    // one threadgroup per value head h, thread t = one of the 128 key / value dims
    uint h = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    threadgroup float buf[32];
    threadgroup float qs[DK];
    threadgroup float ks[DK];
    threadgroup uint nact[DK / 32];
    threadgroup ushort act[DK];
    uint kh = h / R;
    uint C = 2 * KD + VD;
    // causal_conv1d_update on one channel: the window is the state's last K-1 values then the
    // new one; the conv sums in fp32, its output is bf16, silu(bf16) is bf16
    float x3[3];
    uint chans[3] = {kh * DK + t, KD + kh * DK + t, 2 * KD + h * DV + t};
    if (!live[0]) {  // a step after a free run ended: the states stay as they are
        for (uint w = 0; w < 3; ++w)
            if (w == 2 || h % R == 0)
                for (uint j = 0; j < K; ++j) conv_out[chans[w] * K + j] = conv[chans[w] * K + j];
        return;
    }
    for (uint w = 0; w < 3; ++w) {
        uint c = chans[w];
        float acc = 0.0f;
        for (uint j = 0; j < K; ++j) {
            float v = (j + 1 < K) ? float(conv[c * K + j + 1]) : float(y[c]);
            acc += v * convw[c * K + j];
            if (w == 2 || h % R == 0) conv_out[c * K + j] = static_cast<bfloat16_t>(v);
        }
        x3[w] = bf(silu(bf(acc)));
    }
    float q = x3[0], k = x3[1], v = x3[2];
    // l2norm (fp32), then the 1/sqrt(dk) query scale
    float sq = tg_sum(q * q, buf, t, DK);
    float sk = tg_sum(k * k, buf, t, DK);
    q = q * metal::rsqrt(sq + 1e-6f);
    k = k * metal::rsqrt(sk + 1e-6f);
    q = q / {qscale}f;
    qs[t] = q;
    ks[t] = k;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float beta = bf(sigm(float(y[C + VD + h])));
    float a = float(y[C + VD + NV + h]) + dtb[h];
    float sp = a > 20.0f ? a : log1p_(metal::exp(a));
    float decay = metal::exp(A[h] * sp);
    // the delta rule on column t of the head's state, in place on S. Where k[d] and q[d] are
    // both 0 and the head does not decay (exp(g) == 1 exactly), row d's terms are exact zeros:
    // the row is neither read nor written (the dense sweep computes the same floats)
    device float* Sw = (device float*)(S) + h * DK * DV;
    bool full = decay != 1.0f;
    float acc = 0.0f;
    if (full) {
        float kv = 0.0f;
        for (uint d = 0; d < DK; ++d) kv += (Sw[d * DV + t] * decay) * ks[d];
        float delta = (v - kv) * beta;
        for (uint d = 0; d < DK; ++d) {
            float s = Sw[d * DV + t] * decay + ks[d] * delta;
            Sw[d * DV + t] = s;
            acc += s * qs[d];
        }
    } else {
        // the rows with k[d] or q[d] nonzero, ascending (a ballot per simdgroup): the loops
        // below visit the same rows in the same order as the full sweep's nonzero terms
        bool on = ks[t] != 0.0f || qs[t] != 0.0f;
        uint lane = t % 32, sg = t / 32;
        simd_vote::vote_t m = static_cast<simd_vote::vote_t>(simd_ballot(on));
        if (lane == 0) nact[sg] = popcount(uint(m)) + popcount(uint(m >> 32));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint off = 0, tot = 0;
        for (uint i = 0; i < DK / 32; ++i) { off += i < sg ? nact[i] : 0; tot += nact[i]; }
        simd_vote::vote_t below = m & ((simd_vote::vote_t(1) << lane) - 1);
        if (on) act[off + popcount(uint(below)) + popcount(uint(below >> 32))] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sv[8];
        uint n8 = metal::min(tot, 8u);
        for (uint i = 0; i < n8; ++i) sv[i] = Sw[act[i] * DV + t];  // independent loads
        float kv = 0.0f;
        for (uint i = 0; i < tot; ++i) {
            uint d = act[i];
            float s = i < 8 ? sv[i] : Sw[d * DV + t];
            if (ks[d] != 0.0f) kv += (s * decay) * ks[d];
        }
        float delta = (v - kv) * beta;
        for (uint i = 0; i < tot; ++i) {
            uint d = act[i];
            float s = (i < 8 ? sv[i] : Sw[d * DV + t]) * decay + ks[d] * delta;
            if (ks[d] != 0.0f) Sw[d * DV + t] = s;
            acc += s * qs[d];
        }
    }
    // Qwen3_5RMSNormGated over the head's dv
    float of = bf(acc);
    float var = tg_sum(of * of, buf, t, DV) / float(DV);
    of = of * metal::rsqrt(var + {neps}f);
    float hn = bf(float(normw[t]) * bf(of));
    float z = float(y[C + h * DV + t]);
    o[h * DV + t] = static_cast<bfloat16_t>(hn * silu(z));
""")
_SRC["csr_res_simd"] = (["x", "res", "rp", "col", "val"], ["y"], _HDR, """
    // one simdgroup per output row: lanes stride over the row's nonzeros (exact terms, so the
    // simd_sum's order does not matter)
    uint g = thread_position_in_grid.x;
    uint o = g / 32, lane = g % 32;
    float acc = 0.0f;
    if (o < N)
        for (int j = rp[o] + lane; j < rp[o + 1]; j += 32) acc += float(x[col[j]]) * val[j];
    acc = simd_sum(acc);
    if (o < N && lane == 0) y[o] = static_cast<bfloat16_t>(float(res[o]) + bf(acc));
""")



_K = {}


def _kernel(name: str, **consts):
    """The kernel ``name`` with its float constants written into the source (cached)."""
    key = (name, tuple(sorted(consts.items())))
    if key not in _K:
        ins, outs, hdr, src = _SRC[name]
        for c, v in consts.items():
            src = src.replace("{" + c + "}", repr(float(np.float32(v))))
        _K[key] = mx.fast.metal_kernel(name=f"{name}_{len(_K)}", input_names=ins,
                                       output_names=outs, header=hdr, source=src,
                                       ensure_row_contiguous=name != "gdn_inplace")
    return _K[key]


def csr(src: np.ndarray, dst: np.ndarray, w: np.ndarray, n_rows: int):
    """CSR arrays (row pointers, columns, values) of the nonzeros (dst row, src col, w)."""
    order = np.argsort(dst, kind="stable")
    rp = np.zeros(n_rows + 1, np.int32)
    np.add.at(rp, dst + 1, 1)
    return (mx.array(np.cumsum(rp).astype(np.int32)), mx.array(src[order].astype(np.int32)),
            mx.array(w[order].astype(np.float32)))


def _grid(n):
    return (-(-n // TG) * TG, 1, 1)


def lanes(rp) -> int:
    """Lanes per row for norm_csr: about 8 nonzeros per lane on the longest row (1 to 8)."""
    longest = int(np.diff(np.array(rp)).max()) if rp.size > 1 else 0
    return min(8, 1 << max(0, (longest - 1) // 8).bit_length())


def norm_csr(h, w1, m, eps: float, swiglu: bool = False, lpr: int = 1):
    """y = linear(rmsnorm(h)) (or its SwiGLU over the gate / up halves); m = (rp, col, val, n);
    ``lpr`` lanes per output row."""
    rp, col, val, n = m
    n_out = n // 2 if swiglu else n
    k = _kernel("norm_csr_swiglu" if swiglu else "norm_csr", eps=eps)
    return k(inputs=[h, w1, rp, col, val], grid=_grid(n_out * lpr), threadgroup=(TG, 1, 1),
             template=[("D", h.size), ("N", n_out), ("LPR", lpr)],
             output_shapes=[(n_out,)], output_dtypes=[mx.bfloat16])[0]


def csr_res(x, res, m, simd: bool = False):
    """res + linear(x) (bf16, as the decoder layer's residual add); ``simd``: a simdgroup per
    row (for maps with long rows)."""
    rp, col, val, n = m
    if simd:
        return _kernel("csr_res_simd")(inputs=[x, res, rp, col, val], grid=_grid(32 * n),
                                       threadgroup=(TG, 1, 1), template=[("N", n)],
                                       output_shapes=[(n,)], output_dtypes=[mx.bfloat16])[0]
    return _kernel("csr_res")(inputs=[x, res, rp, col, val], grid=_grid(n), threadgroup=(TG, 1, 1),
                    template=[("N", n)], output_shapes=[(n,)], output_dtypes=[mx.bfloat16])[0]


def gdn(y, conv, convw, S, A, dtb, normw, *, kd, vd, nv, r, dk, dv, K, neps):
    """One Gated DeltaNet decode step from the merged input map's output y."""
    assert dk == dv == 128
    k = _kernel("gdn", neps=neps, qscale=dk ** 0.5)
    return k(inputs=[y, conv, convw, S, A, dtb, normw], grid=(nv * dk, 1, 1),
                threadgroup=(dk, 1, 1),
                template=[("KD", kd), ("VD", vd), ("NV", nv), ("R", r), ("DK", dk),
                          ("DV", dv), ("K", K)],
                output_shapes=[(nv * dv,), conv.shape, S.shape],
                output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32])


def gdn_inplace(y, conv, convw, S, A, dtb, normw, live, *, kd, vd, nv, r, dk, dv, K, neps):
    """gdn with the recurrent state S updated in place (S must be a materialized, contiguous
    array owned by the caller and used by nothing else); returns (o, conv_out). With ``live``
    ([1] bool) false it changes nothing (conv_out = conv)."""
    assert dk == dv == 128
    k = _kernel("gdn_inplace", neps=neps, qscale=dk ** 0.5)
    return k(inputs=[y, conv, convw, S, A, dtb, normw, live], grid=(nv * dk, 1, 1),
             threadgroup=(dk, 1, 1),
             template=[("KD", kd), ("VD", vd), ("NV", nv), ("R", r), ("DK", dk), ("DV", dv),
                       ("K", K)],
             output_shapes=[(nv * dv,), conv.shape], output_dtypes=[mx.bfloat16, mx.bfloat16])

