#!/usr/bin/env python3
"""Train one model and save its trajectory.

Usage:
    python scripts/run_train.py --tag main --op add --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grokking.data import make_dataset                    # noqa: E402
from grokking.model import ModelConfig, OneLayerTransformer  # noqa: E402
from grokking.train import TrainConfig, Trainer           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--op", default="add")
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--train-frac", type=float, default=0.3)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-mlp", type=int, default=512)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--n-log-checkpoints", type=int, default=120)
    ap.add_argument("--dense-from", type=int, default=5_000)
    ap.add_argument("--dense-to", type=int, default=20_000)
    ap.add_argument("--dense-every", type=int, default=250)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--loss-dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--d-vocab-out", type=int, default=None,
                    help="output width; default p.  The shipped runs at p != 113 used 113 "
                         "(unused columns, see the README), and main_add_s0 used 114")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing checkpoints for this tag")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    data = make_dataset(p=args.p, op=args.op, train_frac=args.train_frac,
                        seed=args.data_seed)
    mcfg = ModelConfig(
        d_vocab=data.vocab_size, d_vocab_out=args.d_vocab_out or data.p,
        d_model=args.d_model, n_heads=args.n_heads,
        d_head=args.d_model // args.n_heads, d_mlp=args.d_mlp, seed=args.model_seed,
    )
    model = OneLayerTransformer(mcfg)
    tcfg = TrainConfig(
        steps=args.steps, lr=args.lr, weight_decay=args.weight_decay,
        log_every=args.log_every,
        n_log_checkpoints=args.n_log_checkpoints, dense_from=args.dense_from,
        dense_to=args.dense_to, dense_every=args.dense_every,
        warmup_steps=args.warmup_steps, loss_dtype=args.loss_dtype,
    )

    print(json.dumps({"tag": args.tag, "op": args.op, "p": args.p,
                      "n_train": data.n_train, "n_test": data.n_test,
                      "n_params": model.n_params(),
                      "model": mcfg.to_dict(), "train": tcfg.to_dict()}, indent=2),
          flush=True)

    if model.cfg.d_vocab_out < data.p:
        raise SystemExit(f"--d-vocab-out {model.cfg.d_vocab_out} is smaller than p = {data.p}")
    trainer = Trainer(model, data, tcfg, Path(args.out), tag=args.tag, overwrite=args.overwrite)
    hist = trainer.run()

    final = hist[-1]
    print("DONE", json.dumps({k: round(v, 5) for k, v in final.items()}), flush=True)


if __name__ == "__main__":
    main()
