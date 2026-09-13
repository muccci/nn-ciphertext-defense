#!/usr/bin/env bash
# Compile victims/models ONNX → TV (VM) / TA (AOT) MixIR libraries.
# Uses artifact writeback-protect/tvm (LLVM14).
#
#   bash victims/tvm/compile.sh
#   bash victims/tvm/compile.sh --runtime tv --case TV01
#   bash victims/tvm/compile.sh --runtime ta --case TA01 --modes off,on
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Do not inherit caller PYTHON / PYTHONPATH. Conda python and a leftover
# site-packages tvm_ffi produce:
#   ImportError: cannot import name 'libinfo' from partially initialized
#   module 'tvm_ffi'
# System python3.12 loads the in-tree tvm_ffi core.abi3.so.
# Override only with ARTIFACT_TVM_PYTHON if you must.
if [[ -n "${ARTIFACT_TVM_PYTHON:-}" && -x "${ARTIFACT_TVM_PYTHON}" ]]; then
  PYTHON="$ARTIFACT_TVM_PYTHON"
elif [[ -x /usr/bin/python3 ]]; then
  PYTHON=/usr/bin/python3
else
  PYTHON=python3
fi
export PYTHON
export PYTHONPATH="$HERE"
exec "$PYTHON" "$HERE/compile.py" "$@"
