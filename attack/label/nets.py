from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from attack.common.cases import CaseSpec, effective_trace_len


class TraceLabel(nn.Module):
    def __init__(self, cipher, spec: CaseSpec):
        super().__init__()
        if spec.fold is None:
            self.enc = cipher.dense_trace_encoder(effective_trace_len(spec), spec.nz, spec.nz)
            self.dense = True
        else:
            ctor = cipher.__dict__[f"trace_encoder_{spec.fold[2]}"]
            self.enc = ctor(dim=spec.nz, nc=spec.fold[0])
            self.dense = False
        self.head = nn.Sequential(
            nn.Linear(spec.nz, spec.head_hidden1),
            nn.ReLU(inplace=False),
            nn.Dropout(spec.head_dropout),
            nn.Linear(spec.head_hidden1, spec.head_hidden2),
            nn.ReLU(inplace=False),
            nn.Dropout(spec.head_dropout),
            nn.Linear(spec.head_hidden2, spec.n_out),
        )

    def forward(self, trace: torch.Tensor) -> torch.Tensor:
        if self.dense:
            trace = trace.view(trace.shape[0], -1)
        return self.head(self.enc(trace))


SEMANTIC_THRESHOLD = 0.5


def hard_teacher_ce(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(student, teacher.argmax(dim=1), reduction="none")


def teacher_hard_mask(teacher_logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(teacher_logits) >= SEMANTIC_THRESHOLD).float()


def hard_teacher_loss(student: torch.Tensor, teacher_logits: torch.Tensor, spec: CaseSpec) -> torch.Tensor:
    if spec.task == "multi_label":
        target = teacher_hard_mask(teacher_logits)
        return F.binary_cross_entropy_with_logits(student, target, reduction="none").mean(dim=1)
    return hard_teacher_ce(student, teacher_logits)
