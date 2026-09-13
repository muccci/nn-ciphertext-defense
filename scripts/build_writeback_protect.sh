#!/usr/bin/env bash
# Build the compilers/runtimes under ../writeback-protect (not victim models).
#
#   LLVM 14  →  Glow model-compiler + image-classifier  →  MixIR TVM
#   (shared llvm-14.0.6/build_x86_pass_rtti)
#
# Host tools: see ../environment
#
# Usage:
#   bash scripts/build_writeback_protect.sh              # llvm + glow + tvm
#   bash scripts/build_writeback_protect.sh llvm|mc|ic|glow|tvm
#
set -euo pipefail

SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Fixed relative layout from scripts/ -> writeback-protect/
WB_ROOT="$(cd -- "$SCRIPTS_DIR/../writeback-protect" && pwd)"

LLVM_SRC="$WB_ROOT/llvm-14.0.6"
LLVM_BUILD="$LLVM_SRC/build_x86_pass_rtti"

GLOW_SRC="$WB_ROOT/glow"
GLOW_MC_BUILD="$GLOW_SRC/build_mc"
GLOW_IC_BUILD="$GLOW_SRC/build_ic"

TVM_SRC="$WB_ROOT/tvm"
TVM_BUILD="$TVM_SRC/build-llvm14-glow"
TVM_LLVM14_COMPAT="$TVM_SRC/compat/llvm14_headers"
LLVM_CONFIG="$LLVM_BUILD/bin/llvm-config"

# Tool names: clang-14; resolve from PATH.
CC_BIN="$(command -v clang-14 || true)"
CXX_BIN="$(command -v clang++-14 || true)"

# Glow links against this tree's LLVM build product.
LLVM_DIR="$LLVM_BUILD/lib/cmake/llvm"
LIBJIT_LLVM_LINK_BIN="$LLVM_BUILD/bin/llvm-link"

STAGE="${1:-all}"

die() { echo "error: $*" >&2; exit 1; }

require_tools() {
  [[ -d "$WB_ROOT" ]] || die "missing ../writeback-protect relative to scripts/"
  command -v cmake >/dev/null || die "cmake not found"
  command -v ninja >/dev/null || die "ninja not found"
  [[ -n "$CC_BIN" && -x "$CC_BIN" ]] || die "clang-14 not found in PATH"
  [[ -n "$CXX_BIN" && -x "$CXX_BIN" ]] || die "clang++-14 not found in PATH"
  [[ -d "$LLVM_SRC/llvm" ]] || die "missing LLVM source: llvm-14.0.6/llvm"
  [[ -f "$GLOW_SRC/CMakeLists.txt" ]] || die "missing Glow source: glow/CMakeLists.txt"
}

require_tvm() {
  require_tools
  [[ -f "$TVM_SRC/CMakeLists.txt" ]] || die "missing TVM source: tvm/CMakeLists.txt"
  [[ -d "$TVM_LLVM14_COMPAT" ]] || die "missing TVM LLVM14 shim: tvm/compat/llvm14_headers"
}

# Conda include paths break Folly feature detection (memrchr, etc.).
cmake_glow_env() {
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CONDA_PREFIX "$@"
}

# Conda libstdc++ breaks NodeGen/InstrGen at link/run time.
export_ninja_lib_path() {
  local path="${LD_LIBRARY_PATH:-/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu}"
  path="$(printf '%s' "$path" | tr ':' '\n' | grep -v miniconda | paste -sd: -)"
  export LD_LIBRARY_PATH="$path"
}

configure_llvm() {
  mkdir -p "$LLVM_BUILD"
  # Source dir is llvm/ ; do not enable extra LLVM projects.
  cmake -G Ninja -S "$LLVM_SRC/llvm" -B "$LLVM_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC_BIN" \
    -DCMAKE_CXX_COMPILER="$CXX_BIN" \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_TARGETS_TO_BUILD=X86 \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_EH=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF \
    -DLLVM_BUILD_TOOLS=ON \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=ON \
    -DLLVM_INSTALL_UTILS=OFF \
    -DLLVM_OPTIMIZED_TABLEGEN=OFF
  echo "Configured LLVM under writeback-protect/llvm-14.0.6/build_x86_pass_rtti"
}

llvm_cache_matches_tree() {
  [[ -f "$LLVM_BUILD/CMakeCache.txt" ]] || return 1
  # Reject stale caches that still point outside this delivery tree.
  local home
  home="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$LLVM_BUILD/CMakeCache.txt" | head -1)"
  [[ -n "$home" && "$home" == "$LLVM_SRC/llvm" ]]
}

build_llvm() {
  require_tools
  if [[ ! -f "$LLVM_BUILD/build.ninja" ]] || ! llvm_cache_matches_tree; then
    if [[ -d "$LLVM_BUILD" ]] && ! llvm_cache_matches_tree; then
      echo "LLVM build cache is stale or points outside writeback-protect; reconfiguring"
      rm -rf "$LLVM_BUILD"
    fi
    configure_llvm
  fi
  export_ninja_lib_path
  ninja -C "$LLVM_BUILD"
  [[ -f "$LLVM_DIR/LLVMConfig.cmake" ]] || die "LLVMConfig.cmake missing"
  [[ -x "$LIBJIT_LLVM_LINK_BIN" ]] || die "llvm-link missing"
  llvm_cache_matches_tree || die "LLVM cache does not point at writeback-protect/llvm-14.0.6/llvm"
  echo "Built LLVM under writeback-protect/llvm-14.0.6/build_x86_pass_rtti"
}

configure_glow_build() {
  local build_dir="$1"
  local label="$2"
  [[ -f "$LLVM_DIR/LLVMConfig.cmake" ]] || die "configure Glow after LLVM"
  [[ -x "$LIBJIT_LLVM_LINK_BIN" ]] || die "configure Glow after LLVM"
  mkdir -p "$build_dir"
  (
    cd "$build_dir"
    cmake_glow_env cmake -G Ninja "$GLOW_SRC" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_C_COMPILER="$CC_BIN" \
      -DCMAKE_CXX_COMPILER="$CXX_BIN" \
      -DLLVM_DIR="$LLVM_DIR" \
      -DLIBJIT_LLVM_LINK_BIN="$LIBJIT_LLVM_LINK_BIN" \
      -DGLOW_WITH_CPU=ON \
      -DGLOW_WITH_LLVMIRCODEGEN=ON \
      -DGLOW_WITH_OPENCL=OFF \
      -DGLOW_WITH_BUNDLES=OFF \
      -DGLOW_BUILD_TESTS=OFF \
      -DGLOW_BUILD_EXAMPLES=OFF \
      -DGLOW_USE_MARCH_NATIVE=ON
  )
  echo "Configured Glow under writeback-protect/glow/$label"
}

build_glow_mc() {
  require_tools
  if [[ ! -f "$GLOW_MC_BUILD/build.ninja" ]]; then
    configure_glow_build "$GLOW_MC_BUILD" "build_mc"
  fi
  export_ninja_lib_path
  ninja -C "$GLOW_MC_BUILD" model-compiler
  [[ -x "$GLOW_MC_BUILD/bin/model-compiler" ]] || die "model-compiler not produced"
  echo "Built MC: writeback-protect/glow/build_mc/bin/model-compiler"
}

build_glow_ic() {
  require_tools
  if [[ ! -f "$GLOW_IC_BUILD/build.ninja" ]]; then
    configure_glow_build "$GLOW_IC_BUILD" "build_ic"
  fi
  export_ninja_lib_path
  ninja -C "$GLOW_IC_BUILD" image-classifier
  [[ -x "$GLOW_IC_BUILD/bin/image-classifier" ]] || die "image-classifier not produced"
  echo "Built IC: writeback-protect/glow/build_ic/bin/image-classifier"
}

tvm_cache_matches_tree() {
  [[ -f "$TVM_BUILD/CMakeCache.txt" ]] || return 1
  local home llvm_cfg cxxflags
  home="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$TVM_BUILD/CMakeCache.txt" | head -1)"
  llvm_cfg="$(sed -n 's/^USE_LLVM:UNINITIALIZED=//p' "$TVM_BUILD/CMakeCache.txt" | head -1)"
  cxxflags="$(sed -n 's/^CMAKE_CXX_FLAGS:STRING=//p' "$TVM_BUILD/CMakeCache.txt" | head -1)"
  [[ "$home" == "$TVM_SRC" ]] || return 1
  [[ "$llvm_cfg" == "$LLVM_CONFIG" ]] || return 1
  [[ "$cxxflags" == *"$TVM_LLVM14_COMPAT"* ]] || return 1
}

configure_tvm() {
  [[ -x "$LLVM_CONFIG" ]] || die "configure TVM after LLVM (missing $LLVM_CONFIG)"
  mkdir -p "$TVM_BUILD"
  cmake -G Ninja -S "$TVM_SRC" -B "$TVM_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC_BIN" \
    -DCMAKE_CXX_COMPILER="$CXX_BIN" \
    -DUSE_LLVM="$LLVM_CONFIG" \
    -DUSE_CUDA=OFF \
    -DUSE_RPC=OFF \
    -DUSE_GRAPH_EXECUTOR=ON \
    -DCMAKE_CXX_FLAGS="-I$TVM_LLVM14_COMPAT"
  echo "Configured TVM under writeback-protect/tvm/build-llvm14-glow"
}

build_tvm() {
  require_tvm
  [[ -x "$LLVM_CONFIG" ]] || die "build TVM after LLVM"
  if [[ ! -f "$TVM_BUILD/build.ninja" ]] || ! tvm_cache_matches_tree; then
    if [[ -d "$TVM_BUILD" ]] && ! tvm_cache_matches_tree; then
      echo "TVM build cache is stale or points outside writeback-protect; reconfiguring"
      rm -rf "$TVM_BUILD"
    fi
    configure_tvm
  fi
  export_ninja_lib_path
  ninja -C "$TVM_BUILD"
  [[ -f "$TVM_BUILD/libtvm.so" ]] || die "libtvm.so not produced"
  echo "Built TVM: writeback-protect/tvm/build-llvm14-glow/libtvm.so"
}

case "$STAGE" in
  all)
    build_llvm
    build_glow_mc
    build_glow_ic
    build_tvm
    ;;
  llvm) build_llvm ;;
  mc) build_glow_mc ;;
  ic) build_glow_ic ;;
  glow)
    build_glow_mc
    build_glow_ic
    ;;
  tvm) build_tvm ;;
  configure-llvm) require_tools; configure_llvm ;;
  configure-mc) require_tools; configure_glow_build "$GLOW_MC_BUILD" "build_mc" ;;
  configure-ic) require_tools; configure_glow_build "$GLOW_IC_BUILD" "build_ic" ;;
  configure-tvm) require_tvm; configure_tvm ;;
  *)
    die "usage: $0 [all|llvm|mc|ic|glow|tvm|configure-llvm|configure-mc|configure-ic|configure-tvm]"
    ;;
esac
