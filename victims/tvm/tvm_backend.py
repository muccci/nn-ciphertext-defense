"""Locate the artifact MixIR TVM (LLVM14) and import it."""
from __future__ import annotations

import os
import sys
from pathlib import Path

TVM_DIR = Path(__file__).resolve().parent
ART_ROOT = TVM_DIR.parents[1]
TVM_HOME = ART_ROOT / "writeback-protect" / "tvm"
TVM_BUILD = TVM_HOME / "build-llvm14-glow"


def _ffi_core_so() -> Path | None:
    ffi_pkg = TVM_HOME / "3rdparty" / "tvm-ffi" / "python" / "tvm_ffi"
    for name in ("core.abi3.so", "core.so"):
        path = ffi_pkg / name
        if path.is_file():
            return path
    matches = sorted(ffi_pkg.glob("core.cpython-*.so"))
    return matches[0] if matches else None


def prepare_tvm_imports() -> None:
    if not TVM_BUILD.joinpath("libtvm.so").is_file():
        raise FileNotFoundError(f"missing artifact libtvm: {TVM_BUILD / 'libtvm.so'}")
    if _ffi_core_so() is None:
        raise FileNotFoundError(
            "missing tvm_ffi core extension under "
            f"{TVM_HOME / '3rdparty' / 'tvm-ffi' / 'python' / 'tvm_ffi'} "
            "(need core.abi3.so)"
        )
    python_dir = TVM_HOME / "python"
    ffi_python_dir = TVM_HOME / "3rdparty" / "tvm-ffi" / "python"
    os.environ["TVM_LIBRARY_PATH"] = str(TVM_BUILD)
    lib = f"{TVM_BUILD}:{TVM_BUILD / 'lib'}"
    os.environ["LD_LIBRARY_PATH"] = lib + (
        (":" + os.environ["LD_LIBRARY_PATH"]) if os.environ.get("LD_LIBRARY_PATH") else ""
    )
    for path in (python_dir, ffi_python_dir, TVM_DIR):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    # Do not inherit a caller PYTHONPATH: a half-present tvm_ffi (missing
    # libinfo.py) used to show up as a circular-import error.
    os.environ["PYTHONPATH"] = f"{python_dir}:{ffi_python_dir}:{TVM_DIR}"


def load_tvm_modules():
    prepare_tvm_imports()
    import tvm
    from tvm import relax
    from tvm.relax.expr import (
        Call,
        Constant,
        DataflowVar,
        GlobalVar,
        Tuple,
        TupleGetItem,
        Var,
        VarBinding,
    )
    from tvm.relax.frontend.onnx import from_onnx

    return (
        tvm,
        relax,
        from_onnx,
        Call,
        Constant,
        DataflowVar,
        GlobalVar,
        Tuple,
        TupleGetItem,
        Var,
        VarBinding,
    )


def tvm_include_flags() -> list[str]:
    return [
        f"-I{TVM_HOME / 'include'}",
        f"-I{TVM_HOME / '3rdparty' / 'tvm-ffi' / 'include'}",
        f"-I{TVM_HOME / '3rdparty' / 'tvm-ffi' / '3rdparty' / 'dlpack' / 'include'}",
        f"-L{TVM_BUILD}",
        f"-L{TVM_BUILD / 'lib'}",
        f"-Wl,-rpath,{TVM_BUILD}",
        f"-Wl,-rpath,{TVM_BUILD / 'lib'}",
        "-ltvm_ffi",
        "-ltvm_runtime",
        "-ldl",
        "-lpthread",
    ]
