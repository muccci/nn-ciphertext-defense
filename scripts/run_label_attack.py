#!/usr/bin/env python3
"""Trace-to-label for MC01 / MC04 / MC10 / MC15.

Schedules existing doors only:
  scripts/trace/collect_attack.sh
  attack/label/run.py
  attack/label/compute.py

Default sample counts (train / test):
  MC01 2000 / 500
  MC04 20000 / 10000
  MC10 30000 / 1000
  MC15 63747 / 2500

Does not write acc_out/ or trace_out/rho_trans/.

  python3 scripts/run_label_attack.py
  python3 scripts/run_label_attack.py --case MC01 --stage pin
  python3 scripts/run_label_attack.py --case MC01 MC04 MC10 MC15 --detach
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ART = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ART) not in sys.path:
    sys.path.insert(0, str(ART))

import collect_inputs as inputs  # noqa: E402
from attack.common.cases import SITES  # noqa: E402

COLLECT = SCRIPTS / "trace" / "collect_attack.sh"
LABEL_RUN = ART / "attack" / "label" / "run.py"
LABEL_COMPUTE = ART / "attack" / "label" / "compute.py"
MANIFEST = ART / "victims" / "models" / "MANIFEST.txt"
LOG_ROOT = ART / "attack_out" / "label" / "logs"

CASES = ("MC01", "MC04", "MC10", "MC15")
DEFAULT_N = {
    "MC01": (2000, 500),
    "MC04": (20000, 10000),
    "MC10": (30000, 1000),
    "MC15": (63747, 2500),
}
CASE_GPU = {
    "MC01": "3",
    "MC04": "4",
    "MC10": "1",
    "MC15": "2",
}
NM_NEEDLE = {
    "MC01": "libjit_fc_f",
    "MC04": "libjit_convDKKC8_convolve_channel",
    "MC10": "libjit_convDKKC8_convolve_channel",
    "MC15": "libjit_convDKKC8_convolve_channel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs="+", default=list(CASES), choices=CASES)
    parser.add_argument("--modes", default="off,on")
    parser.add_argument("--stage", default="all", choices=("pin", "train", "compute", "all"))
    parser.add_argument("--train-n", type=int, default=0, help="0 = default train count")
    parser.add_argument("--test-n", type=int, default=0, help="0 = default test count")
    parser.add_argument("--pin-par", type=int, default=int(os.environ.get("PAR", "4")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detach", action="store_true", help="one background process per case")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def modes_of(raw: str) -> list[str]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if item not in {"off", "on"}:
            raise SystemExit(f"mode must be off|on, got {item}")
        if item not in out:
            out.append(item)
    return out


def family_variant(case: str) -> tuple[str, str]:
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] != case or len(parts) < 4:
            continue
        family, variant = parts[3].split("/")[:2]
        return family, variant
    raise SystemExit(f"MANIFEST missing {case}")


def collect_site(case: str) -> str:
    return SITES[case][0]


def sample_jobs(case: str, train_n: int, test_n: int) -> list[tuple[str, str]]:
    family, variant = family_variant(case)
    default_train, default_test = DEFAULT_N[case]
    n_train = train_n or default_train
    n_test = test_n or default_test
    jobs: list[tuple[str, str]] = []
    for split, limit in (("train", n_train), ("test", n_test)):
        for index in inputs.list_indices(family, variant, split, limit):
            jobs.append((split, inputs.sample_name(index)))
    return jobs


def live_case_pids(case: str) -> list[int]:
    me = os.getpid()
    try:
        raw = subprocess.check_output(
            ["pgrep", "-af", f"run_label_attack.py --case {case}"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    found: list[int] = []
    for line in raw.splitlines():
        if "pgrep" in line or "extglob" in line:
            continue
        if "python" not in line:
            continue
        pid_s = line.split(None, 1)[0]
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid == me:
            continue
        found.append(pid)
    return found


def preflight(case: str, modes: list[str]) -> None:
    check = subprocess.run(["bash", "-n", str(COLLECT)], check=False)
    if check.returncode != 0:
        raise SystemExit(f"collect_attack.sh is not valid bash rc={check.returncode}")
    needle = NM_NEEDLE[case]
    for mode in modes:
        runner = ART / "victims" / "glow" / "libs" / case / mode / f"{case}_runner"
        if not runner.exists():
            raise SystemExit(f"missing runner {runner}")
        proc = subprocess.run(
            ["nm", "-n", str(runner)],
            check=False,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise SystemExit(f"nm failed {runner}: {proc.stderr.decode(errors='replace')}")
        if needle.encode() not in proc.stdout:
            raise SystemExit(f"nm missing {needle} in {runner}")
        print(f"preflight {case}/{mode} runner ok needle={needle}", flush=True)


def collect_one(case: str, mode: str, site: str, split: str, sample: str) -> None:
    cmd = [
        "bash",
        str(COLLECT),
        "--case",
        case,
        "--mode",
        mode,
        "--site",
        site,
        "--split",
        split,
        "--sample",
        sample,
    ]
    print(f"pin {case}/{mode}/{site}/{split}/{sample}", flush=True)
    result = subprocess.run(cmd, cwd=str(ART), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"collect_attack.sh failed {case}/{mode}/{site}/{split}/{sample} "
            f"rc={result.returncode}"
        )


def run_python(script: Path, extra: list[str], python: str) -> None:
    cmd = [python, str(script), *extra]
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(ART), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{script.name} failed rc={result.returncode}")


def _pin_batch(
    case: str,
    site: str,
    items: list[tuple[str, str, str]],
    pin_par: int,
) -> list[tuple[str, str, str]]:
    failed: list[tuple[str, str, str]] = []

    def _one(mode: str, split: str, sample: str) -> tuple[str, str, str] | None:
        try:
            collect_one(case, mode, site, split, sample)
        except Exception as exc:
            print(f"collect fail {case}/{mode}/{site}/{split}/{sample}: {exc}", flush=True)
            return (mode, split, sample)
        return None

    if pin_par <= 1:
        for mode, split, sample in items:
            item = _one(mode, split, sample)
            if item is not None:
                failed.append(item)
        return failed
    with ThreadPoolExecutor(max_workers=pin_par) as pool:
        futs = {
            pool.submit(_one, mode, split, sample): (mode, split, sample)
            for mode, split, sample in items
        }
        for fut in as_completed(futs):
            item = fut.result()
            if item is not None:
                failed.append(item)
    return failed


def stage_pin(case: str, modes: list[str], args: argparse.Namespace) -> None:
    site = collect_site(case)
    jobs = sample_jobs(case, args.train_n, args.test_n)
    work = [(mode, split, sample) for mode in modes for split, sample in jobs]
    print(
        f"pin {case} site={site} modes={','.join(modes)} "
        f"samples={len(jobs)} jobs={len(work)} par={args.pin_par}",
        flush=True,
    )
    failed = _pin_batch(case, site, work, args.pin_par)
    if failed:
        print(f"retry {len(failed)} failed collects", flush=True)
        failed = _pin_batch(case, site, failed, args.pin_par)
    if failed:
        preview = ", ".join(f"{mode}/{split}/{sample}" for mode, split, sample in failed[:8])
        raise RuntimeError(
            f"{len(failed)} collect failures after retry: {preview}"
        )


def stage_train(case: str, modes: list[str], args: argparse.Namespace) -> None:
    for mode in modes:
        for site in SITES[case]:
            extra = ["--case", case, "--mode", mode, "--device", args.device]
            if case == "MC15":
                extra.extend(["--site", site])
            run_python(LABEL_RUN, extra, args.python)


def stage_compute(case: str, modes: list[str], args: argparse.Namespace) -> None:
    for mode in modes:
        extra = ["--case", case, "--mode", mode, "--device", args.device]
        if case == "MC15":
            for site in SITES[case]:
                run_python(LABEL_COMPUTE, extra + ["--site", site], args.python)
        else:
            run_python(LABEL_COMPUTE, extra, args.python)


def run_case(case: str, args: argparse.Namespace) -> None:
    modes = modes_of(args.modes)
    live = live_case_pids(case)
    if live:
        raise SystemExit(f"{case} already running pids={live}; not starting a second copy")
    if args.stage in {"pin", "all"}:
        preflight(case, modes)
    print(f"==> {case} stage={args.stage} modes={','.join(modes)}", flush=True)
    if args.stage in {"pin", "all"}:
        stage_pin(case, modes, args)
    if args.stage in {"train", "all"}:
        stage_train(case, modes, args)
    if args.stage in {"compute", "all"}:
        stage_compute(case, modes, args)
    print(f"done {case}", flush=True)


def detach_cases(args: argparse.Namespace) -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    modes = modes_of(args.modes)
    lines: list[str] = []
    existing = LOG_ROOT / "pids.txt"
    kept = ""
    if existing.is_file():
        kept = existing.read_text(encoding="utf-8")
    for case in args.case:
        live = live_case_pids(case)
        log = LOG_ROOT / f"{case}.log"
        if live:
            line = f"{case} pid={live[0]} gpu={CASE_GPU.get(case, '?')} log={log} (already running)\n"
            lines.append(line)
            print(line, end="", flush=True)
            continue
        if args.stage in {"pin", "all"}:
            preflight(case, modes)
        cmd = [
            args.python,
            str(Path(__file__).resolve()),
            "--case",
            case,
            "--modes",
            args.modes,
            "--stage",
            args.stage,
            "--train-n",
            str(args.train_n),
            "--test-n",
            str(args.test_n),
            "--pin-par",
            str(args.pin_par),
            "--device",
            args.device,
            "--python",
            args.python,
        ]
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", CASE_GPU.get(case, "0"))
        handle = log.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ART),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handle.close()
        line = f"{case} pid={proc.pid} gpu={env['CUDA_VISIBLE_DEVICES']} log={log}\n"
        lines.append(line)
        print(line, end="", flush=True)
    text = kept
    for line in lines:
        case = line.split()[0]
        text = "".join(
            row for row in text.splitlines(True) if not row.startswith(f"{case} ")
        )
        text += line
    existing.write_text(text, encoding="utf-8")
    return 0


def main() -> int:
    args = parse_args()
    if args.pin_par < 1:
        raise SystemExit("--pin-par must be >= 1")
    if args.detach:
        if len(args.case) == 1:
            run_case(args.case[0], args)
            return 0
        return detach_cases(args)
    for case in args.case:
        run_case(case, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
