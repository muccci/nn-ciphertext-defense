#!/usr/bin/env python3
"""Glow-query reconstructed images.

Usage:
  python3 attack/recon/compute.py --case MC01
  python3 attack/recon/compute.py --case MC04 --mode on --variant full
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ART = Path(__file__).resolve().parents[2]
if str(ART) not in sys.path:
    sys.path.insert(0, str(ART))

from attack.common.cases import get_spec
from attack.common.dataset import TraceImageDataset, discover_samples, make_loader, split_samples
from attack.common.glow import GlowBatch, pred_from_logits
from attack.common.images import model_to_victim, save_unit01_png
from attack.common.util import ART_ROOT, load_cipher_model, resolve_device, write_json
from attack.recon.full import invert_latent
from attack.recon.nets import ImageAE, TraceTransform
from attack.recon.prior import load_prior

if str(ART / "scripts") not in sys.path:
    sys.path.insert(0, str(ART / "scripts"))
import compute_victim_acc as acc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("MC01", "MC04", "MC10", "MC15"))
    parser.add_argument("--mode", default="", choices=("", "off", "on"))
    parser.add_argument("--variant", default="", choices=("", "tonly", "full"))
    parser.add_argument("--site", default="")
    parser.add_argument("--traces", type=Path, default=None)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gan-ckpt", type=Path, default=None)
    parser.add_argument("--gan-src", type=Path, default=None)
    parser.add_argument("--max-items", type=int, default=0)
    return parser.parse_args()


def _jobs(args: argparse.Namespace) -> list[tuple[str, str]]:
    modes = [args.mode] if args.mode else ["off", "on"]
    variants = [args.variant] if args.variant else ["tonly", "full"]
    return [(m, v) for m in modes for v in variants]


def _load_models(spec, cipher, ckpt_path: Path, variant: str, device: torch.device):
    payload = torch.load(ckpt_path, map_location="cpu")
    t_model = TraceTransform(cipher, spec).to(device)
    t_model.load_state_dict(payload["t_model"])
    t_model.eval()
    i_model = None
    if variant == "full":
        i_model = ImageAE(cipher, spec).to(device)
        i_model.load_state_dict(payload["i_model"])
        i_model.eval()
    return t_model, i_model


def compute_one(args: argparse.Namespace, mode: str, variant: str) -> Path:
    spec = get_spec(args.case, args.site or None, mode)
    device = resolve_device(args.device)
    out = args.out or (ART_ROOT / "attack_out" / "recon" / spec.case / mode / variant)
    ckpt = args.ckpt or (out / ("tonly" if variant == "tonly" else "full") / "ckpt" / "best.pth")
    if not ckpt.is_file():
        alt = ckpt.with_name("final.pth")
        ckpt = alt if alt.is_file() else ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing recon ckpt {ckpt}")
    samples = discover_samples(spec, mode, args.traces)
    splits = split_samples(samples, 0.1, 20260413)
    chosen = splits.get(args.split) or splits["test"] or samples
    if args.max_items > 0:
        chosen = chosen[: args.max_items]
    loader = make_loader(
        TraceImageDataset(spec, chosen),
        batch_size=args.batch_size or spec.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )
    cipher = load_cipher_model()
    t_model, i_model = _load_models(spec, cipher, ckpt, variant, device)
    g_model = None
    if variant == "full":
        g_model = load_prior(spec, args.gan_ckpt, args.gan_src).to(device)

    recon_dir = out / "recovered" / args.split
    recon_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    work = out / "glow_work"
    with GlowBatch(ART_ROOT, spec, mode, work) as glow:
        for trace, image01, label, sids in loader:
            trace = trace.to(device)
            image01 = image01.to(device)
            with torch.no_grad():
                h = t_model(trace)
            if variant == "full":
                assert i_model is not None and g_model is not None
                recon_model = invert_latent(h.detach(), i_model, g_model, spec, device, glow=glow)
            else:
                recon_model = h
            recon01 = model_to_victim(recon_model)
            for i, sid in enumerate(sids):
                r = recon01[i].detach().cpu().numpy()
                t = image01[i].detach().cpu().numpy()
                save_unit01_png(recon_dir / f"{sid}.png", r)
                r.tofile(recon_dir / f"{sid}.nchw_f32.bin")
                recon_logits = glow.logits(r)
                true_logits = glow.logits(t)
                recon_pred = pred_from_logits(recon_logits, spec)
                true_pred = pred_from_logits(true_logits, spec)
                pixel_l1 = float(np.mean(np.abs(r - t)))
                pixel_mse = float(np.mean((r - t) ** 2))
                row = {
                    "sample_id": sid,
                    "pixel_l1": pixel_l1,
                    "pixel_mse": pixel_mse,
                }
                if spec.task == "single_label":
                    gt = int(label[i].item())
                    row["gt"] = gt
                    row["recon_pred"] = int(recon_pred)
                    row["orig_pred"] = int(true_pred)
                    row["victim_consistency"] = int(int(recon_pred) == int(true_pred))
                    row["label_acc"] = int(int(recon_pred) == gt)
                else:
                    gt = label[i].detach().cpu().numpy()
                    row["macro_auroc"] = float(
                        acc.macro_auroc(gt[None, :], np.asarray(recon_pred, dtype=np.float32)[None, :])
                    )
                rows.append(row)

    tsv = out / f"{args.split}_metrics.tsv"
    if rows:
        with tsv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    summary: dict[str, float | int | str] = {
        "case": spec.case,
        "mode": mode,
        "variant": variant,
        "n": len(rows),
        "pixel_l1": float(np.mean([r["pixel_l1"] for r in rows])) if rows else 0.0,
        "pixel_mse": float(np.mean([r["pixel_mse"] for r in rows])) if rows else 0.0,
    }
    if spec.task == "single_label" and rows:
        summary["victim_consistency"] = float(np.mean([r["victim_consistency"] for r in rows]))
        summary["label_acc"] = float(np.mean([r["label_acc"] for r in rows]))
    elif rows:
        summary["macro_auroc"] = float(np.mean([r["macro_auroc"] for r in rows]))
    write_json(out / f"{args.split}_summary.json", summary)
    print(summary, flush=True)
    return tsv


def main() -> int:
    args = parse_args()
    for mode, variant in _jobs(args):
        try:
            compute_one(args, mode, variant)
        except FileNotFoundError as exc:
            print(f"missing {args.case}/{mode}/{variant}: {exc}", flush=True)
            if args.mode and args.variant:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
