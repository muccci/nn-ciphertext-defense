#!/usr/bin/env python3
"""Compile shared victims/models ONNX into TVM TV (VM) / TA (AOT) libraries.

Writes under this directory:
  libs/TVxx/{off,on}/TVxx_tvm.so
  libs/TAxx/{off,on}/TAxx_tvm_aot_kernels.so + constants/ + generated_runner
  bin/generic_tvm_native_vm_runner
  libs/TAxx/TAxx_tvm_aot_runner   (linked from on generated_runner)

Uses only artifact MixIR TVM + victims/models.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import onnx

from tvm_backend import TVM_BUILD, TVM_DIR, TVM_HOME, load_tvm_modules, tvm_include_flags
from aot_codegen import (
    TensorSpec,
    collect_runner_plan,
    export_constant_blobs,
    extract_kernel_module,
    generate_runner_source,
    module_to_text,
    tensor_spec_from_sinfo,
)

TARGET = "llvm"
OPT_LEVEL = 3
MODELS = TVM_DIR.parent / "models"
MANIFEST = MODELS / "MANIFEST.txt"
LIBS = TVM_DIR / "libs"
BIN = TVM_DIR / "bin"
VM_RUNNER_SRC = TVM_DIR / "runner_src" / "generic_vm_runner.cc"

DEFENSE_ON = {
    "TVM_WRITEBACK_PROTECT": "1",
    "GLOW_WRITEBACK_PROTECT": "1",
    "TVM_RELU_LOW12_PATCH": "1",
    "TVM_RELU_PATCH_BITS": "30",
    "TVM_RELU_PATCH_FIXED_BITS": "7",
    "TVM_RELU_PATCH_FIXED_VALUE": "0x75",
    "TVM_RELU_PATCH_INC": "3",
    "TVM_RELU_PATCH_POSITIVE": "0",
    "TVM_INPUT_ZERO_DITHER": "random",
    "TVM_INPUT_ZERO_DITHER_LAYOUT": "NCHW",
    "TVM_INPUT_ZERO_DITHER_THRESH": "1",
    "TVM_INPUT_ZERO_DITHER_EPS_MIN": "1e-5",
    "TVM_INPUT_ZERO_DITHER_EPS_MAX": "2e-5",
    "TVM_INPUT_ZERO_DITHER_SILENT": "0",
    "GLOW_INPUT_ZERO_DITHER": "random",
    "GLOW_INPUT_ZERO_DITHER_LAYOUT": "NCHW",
    "GLOW_INPUT_ZERO_DITHER_THRESH": "1",
    "GLOW_INPUT_ZERO_DITHER_EPS_MIN": "1e-5",
    "GLOW_INPUT_ZERO_DITHER_EPS_MAX": "2e-5",
    "GLOW_INPUT_ZERO_DITHER_SILENT": "0",
}
DEFENSE_KEYS = tuple(DEFENSE_ON.keys())


def read_manifest() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5 or not parts[0].startswith("MC"):
            continue
        mc, _tg, _sha, onnx_rel, sample = parts[0], parts[1], parts[2], parts[3], parts[4]
        rows.append(
            {
                "n": mc[2:],
                "onnx_rel": onnx_rel,
                "sample": sample,
                "model": onnx_rel.split("/")[0],
                "dataset": onnx_rel.split("/")[1],
            }
        )
    return rows


def onnx_main_output_spec(model: onnx.ModelProto) -> TensorSpec | None:
    if len(model.graph.output) != 1:
        return None
    output = model.graph.output[0]
    dims: list[int] = []
    for dim in output.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            return None
    dtype = output.type.tensor_type.elem_type
    dtype_name = onnx.TensorProto.DataType.Name(dtype).lower()
    if dtype_name == "float":
        dtype_name = "float32"
    return TensorSpec(shape=tuple(dims), dtype=dtype_name)


@contextmanager
def scoped_defense(mode: str) -> Iterator[None]:
    saved = {k: os.environ.get(k) for k in DEFENSE_KEYS}
    try:
        for k in DEFENSE_KEYS:
            os.environ.pop(k, None)
        if mode == "on":
            os.environ.update(DEFENSE_ON)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def verify_library(library_path: Path, mode: str) -> dict[str, bool]:
    nm = subprocess.run(
        ["nm", "-D", str(library_path)], capture_output=True, text=True, check=True
    ).stdout
    has_helper = ("tvm_relu_low12_f32_scalar" in nm) or ("tvm_relu6_low12_f32_scalar" in nm)
    has_wb = "tvm_wb_patch_" in nm
    if mode == "on":
        if has_helper:
            raise RuntimeError(f"{library_path} still has paper helper refs")
        if not has_wb:
            raise RuntimeError(f"{library_path} missing tvm_wb_patch_* (MixIR not applied)")
    else:
        if has_helper or has_wb:
            raise RuntimeError(f"{library_path} unexpectedly has defense refs")
    return {"has_paper_helper_ref": has_helper, "has_tvm_wb_patch_ref": has_wb}


def compile_vm(onnx_path: Path, out_dir: Path, library_filename: str) -> dict[str, Any]:
    tvm, relax, from_onnx, *_ = load_tvm_modules()
    build_dir = out_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = out_dir / library_filename

    model = onnx.load(str(onnx_path))
    imported_mod = from_onnx(model)
    inference_mod = relax.transform.DecomposeOpsForInference()(imported_mod)
    lowered_mod = relax.transform.LegalizeOps()(inference_mod)
    main = lowered_mod["main"]
    if len(main.params) != 1:
        raise ValueError(f"expected one input, got {len(main.params)}")
    input_spec = tensor_spec_from_sinfo(main.params[0].struct_info)
    output_spec = onnx_main_output_spec(model)

    write_text(build_dir / "lowered_relax.py", module_to_text(lowered_mod))
    t0 = time.perf_counter()
    with tvm.transform.PassContext(opt_level=OPT_LEVEL):
        executable = tvm.compile(lowered_mod, target=TARGET)
    executable.export_library(str(library_path))
    meta = {
        "tvm_home": str(TVM_HOME),
        "tvm_build_dir": str(TVM_BUILD),
        "library_path": str(library_path),
        "onnx_model": str(onnx_path),
        "target": TARGET,
        "opt_level": OPT_LEVEL,
        "compile_seconds": time.perf_counter() - t0,
        "input_shape": list(input_spec.shape),
        "output_shape": list(output_spec.shape) if output_spec else None,
        "defense_env": {k: os.environ[k] for k in DEFENSE_KEYS if k in os.environ},
    }
    write_json(build_dir / "build_meta.json", meta)
    if meta.get("input_shape"):
        write_text(out_dir / "input_shape.txt", ",".join(str(x) for x in meta["input_shape"]))
    return meta


def compile_aot(onnx_path: Path, out_dir: Path, library_filename: str) -> dict[str, Any]:
    tvm, relax, from_onnx, *_ = load_tvm_modules()
    build_dir = out_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    constants_dir = out_dir / "constants"
    library_path = out_dir / library_filename
    runner_source_path = out_dir / "generated_runner.cc"

    model = onnx.load(str(onnx_path))
    imported_mod = from_onnx(model)
    inference_mod = relax.transform.DecomposeOpsForInference()(imported_mod)
    lowered_mod = relax.transform.LegalizeOps()(inference_mod)
    kernel_mod, kernel_names = extract_kernel_module(lowered_mod, tvm)
    input_name, input_spec, output_name, output_spec, constants, calls = collect_runner_plan(
        lowered_mod
    )
    write_text(build_dir / "tir_kernels.py", module_to_text(kernel_mod))
    export_constant_blobs(lowered_mod=lowered_mod, constants_dir=constants_dir)
    write_text(
        runner_source_path,
        generate_runner_source(
            input_name=input_name,
            input_spec=input_spec,
            output_name=output_name,
            output_spec=output_spec,
            constants=constants,
            calls=calls,
            library_filename=library_filename,
        ),
    )
    t0 = time.perf_counter()
    with tvm.transform.PassContext(opt_level=OPT_LEVEL):
        library = tvm.tirx.build(kernel_mod, target=TARGET)
    library.export_library(str(library_path))
    meta = {
        "tvm_home": str(TVM_HOME),
        "tvm_build_dir": str(TVM_BUILD),
        "library_path": str(library_path),
        "constants_dir": str(constants_dir),
        "generated_runner_source": str(runner_source_path),
        "onnx_model": str(onnx_path),
        "target": TARGET,
        "opt_level": OPT_LEVEL,
        "compile_seconds": time.perf_counter() - t0,
        "kernel_names": kernel_names,
        "input_shape": list(input_spec.shape),
        "output_shape": list(output_spec.shape),
        "defense_env": {k: os.environ[k] for k in DEFENSE_KEYS if k in os.environ},
    }
    write_json(build_dir / "build_meta.json", meta)
    write_text(out_dir / "input_shape.txt", ",".join(str(x) for x in meta["input_shape"]))
    return meta


def link_vm_runner() -> Path:
    BIN.mkdir(parents=True, exist_ok=True)
    out = BIN / "generic_tvm_native_vm_runner"
    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        str(VM_RUNNER_SRC),
        *tvm_include_flags(),
        "-o",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def link_aot_runner(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        str(src),
        *tvm_include_flags(),
        "-o",
        str(dest),
    ]
    subprocess.run(cmd, check=True)
    return dest


def build_one(runtime: str, n: str, onnx_rel: str, mode: str) -> None:
    spec_id = f"{'TV' if runtime == 'tv' else 'TA'}{n}"
    onnx_path = MODELS / onnx_rel
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    out_dir = LIBS / spec_id / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    if runtime == "tv":
        lib_name = f"{spec_id}_tvm.so"
        with scoped_defense(mode):
            meta = compile_vm(onnx_path, out_dir, lib_name)
    else:
        lib_name = f"{spec_id}_tvm_aot_kernels.so"
        with scoped_defense(mode):
            meta = compile_aot(onnx_path, out_dir, lib_name)
    verify = verify_library(Path(meta["library_path"]), mode)
    print(json.dumps({"built": spec_id, "mode": mode, "verify": verify, "lib": meta["library_path"]}))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", choices=("tv", "ta", "both"), default="both")
    p.add_argument("--case", action="append", default=[], help="TV01 / TA01 / MC01 / 01")
    p.add_argument("--modes", default="off,on")
    args = p.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows = read_manifest()
    wanted: set[str] | None = None
    if args.case:
        wanted = set()
        for c in args.case:
            if c.startswith(("TV", "TA", "MC", "IC")):
                wanted.add(c[2:])
            else:
                wanted.add(c.zfill(2))
        rows = [r for r in rows if r["n"] in wanted]
        if not rows:
            print("no MANIFEST rows matched", file=sys.stderr)
            return 2

    runtimes = ["tv", "ta"] if args.runtime == "both" else [args.runtime]
    if "tv" in runtimes:
        link_vm_runner()
        print(f"runner {BIN / 'generic_tvm_native_vm_runner'}")
    for row in rows:
        for runtime in runtimes:
            for mode in modes:
                print(f"=== {runtime.upper()}{row['n']} {mode} {row['onnx_rel']} ===", flush=True)
                build_one(runtime, row["n"], row["onnx_rel"], mode)
            if runtime == "ta" and "on" in modes:
                spec_id = f"TA{row['n']}"
                src = LIBS / spec_id / "on" / "generated_runner.cc"
                dest = LIBS / spec_id / f"{spec_id}_tvm_aot_runner"
                if src.is_file():
                    link_aot_runner(src, dest)
                    print(f"runner {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
