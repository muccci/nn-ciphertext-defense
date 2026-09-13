#!/usr/bin/env python3
"""Train recon: trace → image. tonly=h (T), full=x* (T+I+GAN invert).

Usage:
  python3 attack/recon/run.py --case MC01 --mode off --variant tonly
  python3 attack/recon/run.py --case MC04 --mode on --variant full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ART = Path(__file__).resolve().parents[2]
if str(ART) not in sys.path:
    sys.path.insert(0, str(ART))

from attack.common.cases import get_spec
from attack.common.dataset import TraceImageDataset, discover_samples, make_loader, split_samples
from attack.common.glow import GlowBatch
from attack.common.util import ART_ROOT, load_cipher_model, resolve_device, set_seed, write_json
from attack.recon.train import train_ti, train_tonly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("MC01", "MC04", "MC10", "MC15"))
    parser.add_argument("--mode", required=True, choices=("off", "on"))
    parser.add_argument("--variant", required=True, choices=("tonly", "full"))
    parser.add_argument("--site", default="")
    parser.add_argument("--traces", type=Path, default=None)
    parser.add_argument("--input-bin", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--gan-ckpt", type=Path, default=None)
    parser.add_argument("--gan-src", type=Path, default=None)
    parser.add_argument("--action", choices=("train",), default="train")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = get_spec(args.case, args.site or None, args.mode)
    set_seed(args.seed)
    device = resolve_device(args.device)
    out = args.out or (ART_ROOT / "attack_out" / "recon" / spec.case / args.mode / args.variant)
    out.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(spec, args.mode, args.traces, input_bin=args.input_bin)
    splits = split_samples(samples, args.val_ratio, args.seed)
    print(
        f"case={spec.case} mode={args.mode} site={spec.site} variant={args.variant} "
        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} "
        f"loss={spec.loss_kind} stride={spec.trace_stride} latent={spec.latent_space}",
        flush=True,
    )
    write_json(
        out / "args.json",
        {
            "case": spec.case,
            "mode": args.mode,
            "site": spec.site,
            "variant": args.variant,
            "bits": spec.bits,
            "fold": spec.fold,
            "loss_kind": spec.loss_kind,
            "trace_stride": spec.trace_stride,
            "latent_space": spec.latent_space,
            "nz": spec.nz,
        },
    )
    train_bs = args.batch_size or (spec.batch_size if args.variant == "tonly" else spec.ti_batch_size)
    loaders = {
        name: make_loader(
            TraceImageDataset(spec, splits[name]),
            batch_size=train_bs,
            shuffle=name == "train",
            num_workers=args.num_workers,
            device=device,
        )
        for name in ("train", "val", "test")
        if splits[name]
    }
    if "val" not in loaders:
        loaders["val"] = loaders["train"]
    cipher = load_cipher_model()
    with GlowBatch(ART_ROOT, spec, args.mode, out / "glow_work") as glow:
        if args.variant == "tonly":
            train_tonly(
                spec,
                cipher,
                loaders,
                out,
                device=device,
                epochs=args.epochs or spec.t_epochs,
                lr=spec.t_lr,
                beta1=0.5,
                glow=glow,
            )
        else:
            if args.gan_ckpt is not None:
                write_json(out / "gan.json", {"gan_ckpt": str(args.gan_ckpt), "gan_src": str(args.gan_src or "")})
            train_ti(
                spec,
                cipher,
                loaders,
                out,
                device=device,
                epochs=args.epochs or spec.ti_epochs,
                lr=spec.ti_lr,
                beta1=0.5,
                glow=glow,
            )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
