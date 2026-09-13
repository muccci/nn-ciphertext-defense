from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from attack.common.cases import CaseSpec
from attack.common.util import AverageMeter, write_json
from attack.label.nets import TraceLabel, hard_teacher_loss, teacher_hard_mask


def train_hard(
    spec: CaseSpec,
    cipher,
    loaders: dict[str, DataLoader],
    teacher: dict[str, torch.Tensor],
    out_dir: Path,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Path:
    stage = out_dir / "hard"
    ckpt_dir = stage / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = TraceLabel(cipher, spec).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    best_path = ckpt_dir / "best.pth"
    history: list[dict] = []
    best = None

    def run_epoch(name: str, loader: DataLoader, training: bool) -> dict[str, float]:
        model.train(training)
        meters = {k: AverageMeter() for k in ("total", "label_acc", "teacher_consistency")}
        teacher_logits = teacher[name]
        sid_to_row = {sid: i for i, sid in enumerate(teacher_logits["ids"])}
        logits = teacher_logits["logits"].to(device)
        for trace, _image01, labels, sids in loader:
            trace = trace.to(device)
            labels = labels.to(device)
            rows = [sid_to_row[sid] for sid in sids]
            batch = logits[rows]
            with torch.set_grad_enabled(training):
                student = model(trace)
                loss = hard_teacher_loss(student, batch, spec).mean()
                if training:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    optim.step()
            n = int(trace.shape[0])
            meters["total"].update(float(loss.item()), n)
            if spec.task == "multi_label":
                student_mask = teacher_hard_mask(student)
                teacher_mask = teacher_hard_mask(batch)
                agree = (student_mask == teacher_mask).float().mean()
                meters["label_acc"].update(float(agree.item()), n)
                meters["teacher_consistency"].update(float(agree.item()), n)
            else:
                pred = student.argmax(dim=1)
                teacher_pred = batch.argmax(dim=1)
                meters["label_acc"].update(float((pred == labels).float().mean().item()), n)
                meters["teacher_consistency"].update(
                    float((pred == teacher_pred).float().mean().item()), n
                )
        return {k: m.avg for k, m in meters.items()}

    for epoch in range(1, epochs + 1):
        train_m = run_epoch("train", loaders["train"], True)
        val_m = run_epoch("val", loaders["val"], False)
        history.append({"epoch": epoch, "train": train_m, "val": val_m})
        print(
            f"[label] epoch={epoch:03d} train={train_m['total']:.6f} "
            f"val_acc={val_m['label_acc']:.4f} val_teacher={val_m['teacher_consistency']:.4f}",
            flush=True,
        )
        key = (val_m["teacher_consistency"], val_m["label_acc"], -val_m["total"])
        if best is None or key > best:
            best = key
            torch.save({"epoch": epoch, "model": model.state_dict(), "val": val_m}, best_path)

    torch.save({"epoch": epochs, "model": model.state_dict()}, ckpt_dir / "final.pth")
    write_json(
        stage / "train_summary.json",
        {"best_key": list(best) if best else None, "best_ckpt": str(best_path)},
    )
    write_json(stage / "epoch_history.json", {"history": history})
    return best_path
