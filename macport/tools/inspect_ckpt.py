"""Inspect a merged Qwenputer checkpoint without loading it whole.

Reports, streaming one tensor at a time (peak memory ~ one tensor):
  * which residual dims >= d0 (the chip dims) Qwen's own 32 layers read or write,
  * the final norm / embedding usage of the chip dims,
  * value-range statistics of every weight (fp16 / ANE feasibility),
and writes them to a JSON report.

    python -m macport.tools.inspect_ckpt /path/to/Qwen3.5-4B-Qwenputer --out report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

FP16_MAX = 65504.0


def fp16_stats(t: torch.Tensor) -> dict:
    f = t.float()
    nz = f[f != 0]
    if nz.numel() == 0:
        return {"nnz": 0}
    a = nz.abs()
    back = nz.half().float()
    return {
        "nnz": int(nz.numel()),
        "absmax": float(a.max()),
        "absmin_nz": float(a.min()),
        "n_over_fp16_max": int((a > FP16_MAX).sum()),
        "n_fp16_inexact": int((back != nz).sum()),
        "n_fp16_subnormal_or_zero": int(((back.abs() < 6.104e-5)).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", default="ckpt_report.json")
    args = ap.parse_args()
    root = Path(args.model)
    cfg = json.loads((root / "config.json").read_text())
    meta = json.loads((root / "chipformer.json").read_text())
    idx = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    H = cfg["hidden_size"]
    n_chip, d0, v0 = meta["chip_layers"], meta["chip_dims_offset"], meta["chip_token_offset"]
    print(f"hidden {H} chip_layers {n_chip} d0 {d0} v0 {v0}")

    by_file: dict[str, list[str]] = {}
    for k, f in idx.items():
        by_file.setdefault(f, []).append(k)

    qwen_in_cols = torch.zeros(H, dtype=torch.bool)   # chip dims Qwen's matrices read
    qwen_out_rows = torch.zeros(H, dtype=torch.bool)  # residual dims Qwen's matrices write
    per_tensor_in: dict[str, list[int]] = {}
    chip_stats, qwen_stats, other = {}, {}, {}
    chip_nnz_total = 0
    for f, keys in sorted(by_file.items()):
        with safe_open(root / f, framework="pt") as sf:
            for k in sorted(keys):
                t = sf.get_tensor(k)
                layer = int(k.split(".")[2]) if k.startswith("model.layers.") else None
                if layer is not None and layer < n_chip:
                    s = fp16_stats(t)
                    chip_stats[k] = s
                    chip_nnz_total += s["nnz"]
                    continue
                if layer is not None:
                    if t.ndim == 2:
                        if t.shape[1] == H:  # reads the residual
                            nzc = (t != 0).any(0)
                            ext = torch.nonzero(nzc[d0:]).flatten() + d0
                            if len(ext):
                                per_tensor_in[k] = ext.tolist()[:16]
                            qwen_in_cols |= nzc
                        if t.shape[0] == H:  # writes the residual
                            qwen_out_rows |= (t != 0).any(1)
                    elif t.ndim == 1 and t.shape[0] == H:  # a full-residual norm
                        other.setdefault("qwen_norm_chipdim_values", {})[k] = sorted(
                            {round(float(x), 4) for x in t[d0:].float().unique()})[:8]
                    if layer == n_chip:
                        qwen_stats[k] = fp16_stats(t)
                    continue
                if k == "model.embed_tokens.weight":
                    text = t[:v0]
                    other["embed_text_rows_nonzero_chip_cols"] = int((text[:, d0:] != 0).any(0).sum())
                    other["embed_text_rows_nonzero_cols_list"] = (
                        torch.nonzero((text[:, d0:] != 0).any(0)).flatten() + d0).tolist()
                    other["embed_chip_rows_nonzero_text_cols"] = int((t[v0:, :d0] != 0).any(0).sum())
                    other["embed_stats_text_part"] = fp16_stats(text[:, :d0])
                    other["embed_stats_chip_part"] = fp16_stats(t[:, d0:])
                elif k == "model.norm.weight":
                    nw = t.float()
                    other["final_norm_chip_dims"] = {
                        str(i + d0): float(nw[d0 + i]) for i in range(H - d0) if float(nw[d0 + i]) != -1.0}
                print("  ", k, tuple(t.shape))
        print("done", f, flush=True)

    rep = {
        "hidden": H, "d0": d0, "v0": v0, "n_chip": n_chip,
        "qwen_reads_chip_dims": (torch.nonzero(qwen_in_cols[d0:]).flatten() + d0).tolist(),
        "qwen_writes_chip_dims": (torch.nonzero(qwen_out_rows[d0:]).flatten() + d0).tolist(),
        "qwen_reads_text_dims": int(qwen_in_cols[:d0].sum()),
        "per_tensor_chip_inputs": dict(list(per_tensor_in.items())[:12]),
        "keep_cols": meta["code_dims"] + [meta["tau_dim"], meta["mu_dim"]],
        "chip_nnz_total": chip_nnz_total,
        "chip_absmax": max((s.get("absmax", 0) for s in chip_stats.values()), default=0),
        "chip_n_over_fp16_max": sum(s.get("n_over_fp16_max", 0) for s in chip_stats.values()),
        "chip_n_fp16_inexact": sum(s.get("n_fp16_inexact", 0) for s in chip_stats.values()),
        "chip_worst": sorted(((k, s) for k, s in chip_stats.items() if s.get("nnz")),
                             key=lambda kv: -kv[1]["absmax"])[:10],
        "qwen_layer_stats_first": qwen_stats,
        **other,
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("chip_worst", "qwen_layer_stats_first")},
                     indent=1)[:6000])


if __name__ == "__main__":
    main()
