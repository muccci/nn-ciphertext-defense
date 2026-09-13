#!/usr/bin/env python3
"""Compute ρ_trans and R from Pin traces. Artifact-local.

Backends (--backend):
  mc  sites + float_stores_only + exclude_pointer;
      exclude_defense except MC01/MC09.
  ic  addresses + float memory writes only:
      FP store mnemonics (vmovss/...) ∪ mem* (memmove/memcpy/memset/...);
      exclude_defense except IC01/IC09.
  tv  sites + tvm_so_mem (*_tvm.so ∪ memmove/memset). Pools {train,test}/idx*.
  ta  same filter as tv, on *_tvm_aot_kernels.so traces.

Usage:
  python3 scripts/compute_victim_rho_trans.py --backend mc
  python3 scripts/compute_victim_rho_trans.py --backend ic
  python3 scripts/compute_victim_rho_trans.py --backend tv
  python3 scripts/compute_victim_rho_trans.py --backend ta --case TA01
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFENSE_NEEDLES = (
    "patchedrelu", "reluresultbitsfrominputbits", "selectpatchedseedbits",
    "applyinputzerocheckerboardditherorexit", "inputzerodither",
    "patchpositiveoutputs", "relulow12",
)
EXCLUDE_DEFENSE_BY_ID = {
    **{f"MC{i:02d}": (i not in (1, 9)) for i in range(1, 18)},
    **{f"IC{i:02d}": (i not in (1, 9)) for i in range(1, 18)},
}
POINTER_FRAME_OPERANDS = frozenset({
    "rbp", "rsp", "r12", "r13", "r14", "r15", "rbx", "r11", "r10", "r9", "r8",
    "rdi", "rsi", "rdx", "rcx",
})
FLOAT_STORE_MNEMS = (
    "vmovss ", "vmovsd ", "vmovd ", "movss ", "movsd ",
    "vmovups ", "vmovaps ", "vmovupd ", "vmovapd ",
)
MEM_STAR_NEEDLES = ("memmove", "memcpy", "memset", "mempcpy", "bcopy")
TVM_USER_MEM_TOKENS = ("memmove", "memset")
TVM_KERNEL_SO_SUFFIXES = ("_tvm.so", "_tvm_aot_kernels.so")
_SITE_FIELD = re.compile(
    rb'"(ip_hex|module|symbol|routine|disasm|compare_count|unchanged_count)":'
    rb'\s*(?:"([^"]*)"|(\d+))'
)


@dataclass
class Acc:
    compare: int = 0
    unchanged: int = 0

    def add(self, compare: int, unchanged: int) -> None:
        self.compare += compare
        self.unchanged += unchanged

    @property
    def ratio(self) -> float:
        return self.unchanged / self.compare if self.compare else float("nan")


def _store_attrs(item: dict) -> dict:
    owner = item.get("owner")
    return owner if isinstance(owner, dict) else item


def _owner_text(item: dict) -> str:
    attrs = _store_attrs(item)
    if isinstance(attrs, dict):
        return " ".join(str(attrs.get(k, "")) for k in ("symbol", "routine", "module"))
    return str(attrs)


def _disasm(item: dict) -> str:
    attrs = _store_attrs(item)
    return ((attrs.get("disasm") if isinstance(attrs, dict) else "") or "").lower().strip()


def _is_defense(item: dict) -> bool:
    return any(n in _owner_text(item).lower() for n in DEFENSE_NEEDLES)


def _is_float_store_mnem(item: dict) -> bool:
    dis = _disasm(item)
    return any(dis.startswith(m) for m in FLOAT_STORE_MNEMS)


def _is_tvm_kernel_module(module: str | None) -> bool:
    if not module:
        return False
    name = Path(module).name.lower()
    return name.endswith(TVM_KERNEL_SO_SUFFIXES)


def _is_tvm_user_mem(item: dict) -> bool:
    attrs = _store_attrs(item)
    text = " ".join(
        str(attrs.get(k, "") if isinstance(attrs, dict) else "")
        for k in ("symbol", "routine")
    ).lower()
    return any(tok in text for tok in TVM_USER_MEM_TOKENS)


def keep_tvm_so_mem(item: dict) -> bool:
    """Keep kernel .so sites + libc memmove/memset (any module)."""
    attrs = _store_attrs(item)
    if _is_tvm_kernel_module(attrs.get("module") if isinstance(attrs, dict) else None):
        return True
    return _is_tvm_user_mem(item)


def _is_mem_star(item: dict) -> bool:
    attrs = _store_attrs(item)
    text = " ".join(
        str(attrs.get(k, "") if isinstance(attrs, dict) else "")
        for k in ("symbol", "routine")
    ).lower()
    return any(n in text for n in MEM_STAR_NEEDLES)


def _is_float_mem_write(item: dict) -> bool:
    """IC ρ_trans pool: float SIMD/scalar stores ∪ mem* copies of float tensors."""
    return _is_float_store_mnem(item) or _is_mem_star(item)


def _is_pointer(item: dict) -> bool:
    attrs = _store_attrs(item)
    if not isinstance(attrs, dict):
        attrs = item
    sym = (attrs.get("symbol") or attrs.get("routine") or "").lower()
    dis = _disasm(item)
    if any(x in sym for x in ("getxyzw", "getxy", "getxyz")):
        return True
    if dis.startswith(("push ", "pop ", "lea ")):
        return True
    if "mov qword ptr" in dis and "rsp" in dis:
        return True
    if "qword" in dis and "xmm" not in dis and "movss" not in dis and "vmov" not in dis:
        parts = dis.split(",")
        if parts:
            src = parts[-1].strip()
            if src in POINTER_FRAME_OPERANDS or "rsp" in src or "rbp" in src:
                return True
    return False


def _find_sites_offset(path: Path) -> int | None:
    size = path.stat().st_size
    step = 64 << 20
    pos, overlap = size, b""
    with path.open("rb") as handle:
        while pos > 0:
            take = min(step, pos)
            pos -= take
            handle.seek(pos)
            chunk = handle.read(take)
            found = (chunk + overlap).find(b'"sites": [')
            if found >= 0:
                return pos + found
            overlap = chunk[:16]
    return None


def _iter_sites(path: Path) -> Iterator[dict]:
    offset = _find_sites_offset(path)
    if offset is None:
        return
    current: dict = {}
    with path.open("rb") as handle:
        handle.seek(offset)
        for line in handle:
            if len(line) > 512:
                continue
            match = _SITE_FIELD.search(line)
            if not match:
                continue
            key = match.group(1).decode()
            current[key] = match.group(2).decode() if match.group(2) else int(match.group(3))
            if key == "unchanged_count":
                yield current
                current = {}


def _iter_addresses(path: Path) -> Iterator[dict]:
    """Stream addresses[] objects without loading the full JSON."""
    marker = b'"addresses"'
    with path.open("rb") as handle:
        chunk = handle.read(1 << 20)
        idx = chunk.find(marker)
        if idx < 0:
            return
        handle.seek(idx)
        # Find the '[' after "addresses"
        while True:
            b = handle.read(1)
            if not b:
                return
            if b == b"[":
                break
        decoder = json.JSONDecoder()
        buf = ""
        while True:
            while True:
                s = buf.lstrip()
                if not s:
                    more = handle.read(1 << 20)
                    if not more:
                        return
                    buf = more.decode("utf-8", errors="replace")
                    continue
                if s[0] == "]":
                    return
                if s[0] == ",":
                    buf = s[1:]
                    continue
                break
            try:
                obj, end = decoder.raw_decode(buf.lstrip())
            except json.JSONDecodeError:
                more = handle.read(1 << 20)
                if not more:
                    return
                buf = buf + more.decode("utf-8", errors="replace")
                continue
            # consume decoded prefix from stripped view
            stripped = buf.lstrip()
            lead = len(buf) - len(stripped)
            buf = buf[lead + end :]
            if isinstance(obj, dict):
                yield obj


def accumulate_mc(path: Path, case_id: str) -> Acc:
    acc = Acc()
    ex_def = EXCLUDE_DEFENSE_BY_ID.get(case_id, True)
    for item in _iter_sites(path):
        if ex_def and _is_defense(item):
            continue
        if _is_pointer(item):
            continue
        if not _is_float_store_mnem(item):
            continue
        compare = int(item.get("compare_count") or 0)
        unchanged = int(item.get("unchanged_count") or 0)
        if compare > 0:
            acc.add(compare, unchanged)
    return acc


def accumulate_tvm(path: Path, case_id: str = "") -> Acc:
    acc = Acc()
    for item in _iter_sites(path):
        if not keep_tvm_so_mem(item):
            continue
        compare = int(item.get("compare_count") or 0)
        unchanged = int(item.get("unchanged_count") or 0)
        if compare > 0:
            acc.add(compare, unchanged)
    return acc


def accumulate_ic(path: Path, case_id: str) -> Acc:
    acc = Acc()
    ex_def = EXCLUDE_DEFENSE_BY_ID.get(case_id, True)
    for item in _iter_addresses(path):
        if ex_def and _is_defense(item):
            continue
        if not _is_float_mem_write(item):
            continue
        compare = int(item.get("compare_count") or 0)
        unchanged = int(item.get("unchanged_count") or 0)
        if compare > 0:
            acc.add(compare, unchanged)
    return acc


def r_adj(off: Acc, on: Acc) -> float:
    if off.ratio <= 0 or off.ratio != off.ratio:
        return float("nan")
    return (off.ratio - on.ratio) / off.ratio


def load_manifest(manifest: Path, backend: str) -> dict[str, dict]:
    prefix = {"mc": "MC", "ic": "IC", "tv": "TV", "ta": "TA"}[backend]
    rows: dict[str, dict] = {}
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5 or not parts[0].startswith("MC"):
            continue
        mc, task, _, onnx_rel, sample = parts[0], parts[1], parts[2], parts[3], parts[4]
        case_id = f"{prefix}{mc[2:]}"
        rows[case_id] = {
            "task": task,
            "model": onnx_rel.split("/")[0],
            "dataset": onnx_rel.split("/")[1],
            "input_sample": sample,
            "onnx_rel": onnx_rel,
        }
    return rows


def default_trace_root(art_root: Path, backend: str) -> Path:
    return {
        "mc": art_root / "trace_out/rho_trans/mc",
        "ic": art_root / "trace_out/rho_trans/ic",
        "tv": art_root / "trace_out/rho_trans/tv",
        "ta": art_root / "trace_out/rho_trans/ta",
    }[backend]


def mode_jsons(case_dir: Path, mode: str) -> list[Path]:
    """<case>/<mode>/{train,test}/idx*/taint_block16_bits.json"""
    found: list[Path] = []
    for split in ("train", "val", "test"):
        found.extend(sorted((case_dir / mode).glob(f"{split}/idx*/taint_block16_bits.json")))
    return found


def accumulate_paths(paths: list[Path], fn, case_id: str) -> Acc:
    acc = Acc()
    for path in paths:
        part = fn(path, case_id)
        acc.add(part.compare, part.unchanged)
    return acc


def main() -> int:
    scripts_dir = Path(__file__).resolve().parent
    art_root = scripts_dir.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("mc", "ic", "tv", "ta"), required=True)
    p.add_argument("--trace-root", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=art_root / "victims/models/MANIFEST.txt")
    p.add_argument("--case", action="append", default=[], dest="cases",
                   help="MC01 / IC01 / TV01 / TA01 (repeatable). Default: all from manifest.")
    p.add_argument("--mc", action="append", default=[], help=argparse.SUPPRESS)  # compat
    p.add_argument("--out-tsv", type=Path, default=None)
    args = p.parse_args()

    backend = args.backend
    trace_root = (args.trace_root or default_trace_root(art_root, backend)).resolve()
    meta = load_manifest(args.manifest, backend)
    cases = args.cases or args.mc or sorted(meta.keys())
    if backend == "ic":
        cases = [c if c.startswith("IC") else f"IC{c[2:]}" if c.startswith("MC") else c for c in cases]
    elif backend == "tv":
        cases = [c if c.startswith("TV") else f"TV{c[2:]}" if c[:2] in ("MC", "IC", "TA") else c for c in cases]
    elif backend == "ta":
        cases = [c if c.startswith("TA") else f"TA{c[2:]}" if c[:2] in ("MC", "IC", "TV") else c for c in cases]
    out_rows = []
    missing = 0
    id_key = backend

    metrics = {
        "mc": "sites + float_stores_only + exclude_pointer",
        "ic": "addresses + float_mem_writes (FP mnem ∪ mem*)",
        "tv": "sites + tvm_so_mem",
        "ta": "sites + tvm_so_mem",
    }
    print(f"backend={backend}")
    print(f"metric={metrics[backend]}")
    print(f"{id_key}\trho_trans_off\trho_trans_on\tR%")

    accumulate = {
        "mc": accumulate_mc,
        "ic": accumulate_ic,
        "tv": accumulate_tvm,
        "ta": accumulate_tvm,
    }[backend]

    for case_id in cases:
        case_dir = trace_root / case_id
        off_ps = mode_jsons(case_dir, "off")
        on_ps = mode_jsons(case_dir, "on")
        if not off_ps or not on_ps:
            missing += 1
            print(f"{case_id}\tmissing")
            out_rows.append({id_key: case_id, "status": "missing"})
            continue
        off = accumulate_paths(off_ps, accumulate, case_id)
        on = accumulate_paths(on_ps, accumulate, case_id)
        r_pct = r_adj(off, on) * 100.0
        info = meta.get(case_id, {})
        out_rows.append({
            id_key: case_id,
            "task": info.get("task", ""),
            "model": info.get("model", ""),
            "dataset": info.get("dataset", ""),
            "rho_trans_off": off.ratio,
            "rho_trans_on": on.ratio,
            "R_pct": r_pct,
            "cmp_off": off.compare,
            "cmp_on": on.compare,
        })
        print(f"{case_id}\t{off.ratio:.6f}\t{on.ratio:.6f}\t{r_pct:.2f}")

    out_tsv = args.out_tsv or (trace_root / "rho_trans.tsv")
    if out_rows:
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        fields = list(out_rows[0].keys())
        with out_tsv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            w.writerows(out_rows)
        print(f"wrote {out_tsv}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
