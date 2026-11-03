"""The chip layers alone, teacher-forced over a real RV32 trace: speed per step, and every
step's kept dims (output code, tau, mu) checked against the circuit executor.

    .venv/bin/python -m macport.bench_chip /path/to/Qwen3.5-4B-Qwenputer \\
        --program upstream/experimental/riscv/tests/data/rv/c-crc32.elf --steps 600

The trace is the program's own (``ref.Machine.run``) from its first instruction; the chip
layers start as in a conversation (a one-token text prompt, then the snapshot injected).
"""

import argparse
import json
import time
from pathlib import Path

import torch

from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, ref
from chipformer.rv import spec as S
from macport import chat as MC


def load(path: Path, device: str, impl: str = "hf"):
    from transformers import Qwen3_5TextConfig

    from chipformer.q35.ir import Target

    meta = json.loads((path / "chipformer.json").read_text())
    cfg = Qwen3_5TextConfig.from_pretrained(path)
    ck = MC.Ckpt(path)
    emb = MC.Embedding(ck, meta["chip_token_offset"], device, chip_rows=True)
    T = BR.Text(path / "tokenizer.json")
    c = core.build(Target(cfg.linear_num_key_heads, cfg.linear_num_value_heads,
                          cfg.intermediate_size), T)
    chip = MC.make_chip(impl, ck, cfg, meta, emb, device)
    return meta, c, chip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--program", default="upstream/experimental/riscv/tests/data/rv/c-crc32.elf")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--device", default=MC.default_device())
    ap.add_argument("--impl", default="hf", help=", ".join(MC.CHIP_IMPLS))
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--free", action="store_true",
                    help="free-run (mlx-freerun): the chip layers decode every token themselves "
                         "from the first one; the tokens are checked against the trace")
    args = ap.parse_args()
    path = Path(args.model)
    t0 = time.time()
    meta, c, chip = load(path, args.device, args.impl)
    v0 = meta["chip_token_offset"]
    keep_cols = torch.tensor(meta["code_dims"] + [meta["tau_dim"], meta["mu_dim"]],
                             device=args.device)
    print(f"[{args.impl} chip layers on {args.device}: {chip.compact / 1e6:.0f}M weights, "
          f"loaded in {time.time() - t0:.1f}s]", flush=True)
    prog = elf.program(args.program)
    core.snapshot(c, ref.Machine.from_program(prog))
    toks = ref.Machine.from_program(prog).run(10 ** 6)[: args.steps + 1]
    ids = [t - S.TEXT_TOKEN if S.is_text(t) else v0 + t for t in toks]
    prompt = [198]  # "\n"
    if args.free:
        return free(chip, c, ids, v0, prompt, args)
    circ = None if args.no_check else MC.CircuitChip(c, v0, "cpu")
    if circ is not None:
        circ.start(c, prompt)
    chip.start(c, prompt)
    bad, t_n, t_c, warm = 0, 0.0, 0.0, 20
    for i, x in enumerate(ids):
        MC.sync(args.device)
        ts = time.time()
        h = chip.keep(chip.step(x))  # (the decode reads the code every step anyway)
        dt = time.time() - ts
        if i >= warm:
            t_n += dt
        if circ is not None:
            ts = time.time()
            k = circ.step(x)
            if i >= warm:
                t_c += time.time() - ts
            bad += not bool((h == k.to(torch.bfloat16).float()).all())
    n = len(ids) - warm
    print(f"{args.impl}: {1e3 * t_n / n:.2f} ms/step ({n / t_n:.0f} tok/s) over {n} steps of "
          f"{Path(args.program).name}" + ("" if circ is None else
          f"; circuit {1e3 * t_c / n:.2f} ms/step; mismatched steps {bad}/{len(ids)}"))


def free(chip, c, ids, v0, prompt, args) -> None:
    """The chip layers' own decoding from the trace's first token; text tokens (putchar
    echoes) go through one ordinary step, the rest through free runs."""
    from macport.mlx_chip import code_lut

    lut = code_lut(c.text, v0)
    pow2 = 1 << torch.arange(len(meta_code := chip.keep_cols[: core.NFAM + core.NPAY]))
    chip.start(c, prompt)
    chip.free_run(ids[0], 64)  # compile the traced steps
    chip.step(198)
    chip.start(c, prompt)
    got, pos, runs = [], 0, 0
    MC.sync(args.device)
    t0 = time.time()
    while pos < len(ids) - 1:
        if ids[pos] >= v0:
            out = chip.free_run(ids[pos], len(ids) - 1 - pos)
            runs += 1
        else:
            k = chip.keep(chip.step(ids[pos]))[: len(meta_code)]
            out = [int(lut[int(((k > 0).long() * pow2).sum())]) if (k != 0).any() else -2]
        got += out
        pos += len(out)
        if got[-1] != ids[pos]:
            break
    dt = time.time() - t0
    n = min(len(got), len(ids) - 1)
    bad = next((i for i in range(n) if got[i] != ids[i + 1]), None)
    print(f"{args.impl} free run: {len(got)} tokens in {dt:.2f}s ({len(got) / dt:.0f} tok/s, "
          f"{1e3 * dt / len(got):.3f} ms/token, {runs} runs) of {Path(args.program).name}; "
          + ("every token as the trace" if bad is None and len(got) == len(ids) - 1 else
             f"first difference at {bad}: got {got[bad] if bad is not None else None}"))


if __name__ == "__main__":
    main()
