#!/usr/bin/env python3
"""Train label: trace → teacher hard label. MC01/MC04/MC10/MC15.

Usage:
  python3 attack/label/run.py --case MC01 --mode off
  python3 attack/label/run.py --case MC15 --mode off --site transition2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ART = Path(__file__).resolve().parents[2]
if str(ART) not in sys.path:
    sys.path.insert(0, str(ART))

from attack.common.cases import SITES, get_spec
from attack.common.dataset import TraceOnlyDataset, discover_samples, make_loader, split_samples
from attack.common.glow import GlowBatch
from attack.common.util import ART_ROOT, load_cipher_model, resolve_device, set_seed, write_json
from attack.label.train import train_hard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("MC01", "MC04", "MC10", "MC15"))
    parser.add_argument("--mode", required=True, choices=("off", "on"))
    parser.add_argument("--site", default="")
    parser.add_argument("--traces", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def _teacher_logits(spec, mode, splits, out: Path, device: torch.device) -> dict[str, torch.Tensor]:
    cache = out / "teacher_logits.pt"
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu")
        return {k: v for k, v in payload.items()}
    result: dict[str, torch.Tensor] = {}
    with GlowBatch(ART_ROOT, spec, mode, out / "glow_work") as glow:
        for name, samples in splits.items():
            rows = []
            ids = []
            for sample in samples:
                if sample.image_path is None or not sample.image_path.is_file():
                    raise FileNotFoundError(f"teacher needs image for {sample.sample_id}")
                from attack.common.images import load_nchw_f32, nchw_to_unit01

                image01 = nchw_to_unit01(load_nchw_f32(sample.image_path, spec), spec)
                rows.append(glow.logits(image01))
                ids.append(sample.sample_id)
            result[name] = {
                "ids": ids,
                "logits": torch.from_numpy(np.stack(rows, axis=0))
                if rows
                else torch.zeros((0, spec.n_out)),
            }
    torch.save(result, cache)
    return result


def main() -> int:
    args = parse_args()
    if args.case == "MC15" and not args.site:
        raise SystemExit("MC15 needs --site " + "|".join(SITES["MC15"]))
    spec = get_spec(args.case, args.site or None, args.mode, task="label")
    set_seed(args.seed)
    device = resolve_device(args.device)
    out = args.out or (ART_ROOT / "attack_out" / "label" / spec.case / args.mode / spec.site)
    out.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(spec, args.mode, args.traces)
    splits = split_samples(samples, args.val_ratio, args.seed)
    print(
        f"case={spec.case} mode={args.mode} site={spec.site} "
        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}",
        flush=True,
    )
    write_json(
        out / "args.json",
        {
            "case": spec.case,
            "mode": args.mode,
            "site": spec.site,
            "bits": spec.bits,
            "fold": spec.fold,
            "nz": spec.nz,
            "head": [spec.head_hidden1, spec.head_hidden2, spec.head_dropout],
        },
    )
    loaders = {
        name: make_loader(
            TraceOnlyDataset(spec, splits[name]),
            batch_size=args.batch_size or spec.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            device=device,
        )
        for name in ("train", "val", "test")
        if splits[name]
    }
    if "val" not in loaders:
        loaders["val"] = loaders["train"]
        splits["val"] = splits["train"]
    teacher = _teacher_logits(spec, args.mode, splits, out, device)
    train_hard(
        spec,
        load_cipher_model(),
        loaders,
        teacher,
        out,
        device=device,
        epochs=args.epochs or spec.label_epochs,
        lr=spec.label_lr,
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
