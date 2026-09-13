#!/usr/bin/env bash
# Compile shared victims/models ONNX into Glow MC / IC products.
#
#   bash victims/glow/compile.sh
#   bash victims/glow/compile.sh --runtime mc --case MC01
#   bash victims/glow/compile.sh --runtime ic --case IC01
#
# MC  → libs/MCxx/{off,on}/  (model-compiler bundle + runner)
# IC  → libs/ICxx/meta.json  (runtime image-classifier; no precompiled bundle)
#
# Uses artifact writeback-protect/glow. Collect stays in scripts/.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ART_ROOT="$(cd -- "$HERE/../.." && pwd)"
WB_ROOT="$ART_ROOT/writeback-protect"
GLOW_SRC="${GLOW_SRC:-$WB_ROOT/glow}"
MODEL_COMPILER="${MODEL_COMPILER:-$GLOW_SRC/build_mc/bin/model-compiler}"
MODEL_COMPILER_LD_LIBRARY_PATH="${MODEL_COMPILER_LD_LIBRARY_PATH:-/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu}"

MODELS_DIR="${MODELS_DIR:-$ART_ROOT/victims/models}"
MANIFEST="${MANIFEST:-$MODELS_DIR/MANIFEST.txt}"
COMMON_DIR="$HERE/common"
RUNNER_SRC_ROOT="$HERE/runner_src"
LIBS_ROOT="${GLOW_LIBS:-$HERE/libs}"

BACKEND="${BACKEND:-CPU}"
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--std=c++17 -O2 -fno-pie}"
LDFLAGS="${LDFLAGS:--no-pie}"
REBUILD_RUNNERS=1
RUNTIME="both"
MODES="off,on"
CASES=()

die() { echo "error: $*" >&2; exit 1; }

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

cases_for_mode() {
  case "$1" in
    off) echo 'off:relu_off_maxpool_off:0:0:BUNDLE_CASE_RELU_OFF_MAXPOOL_OFF' ;;
    on)  echo 'on:relu_on_maxpool_on:1:1:BUNDLE_CASE_RELU_ON_MAXPOOL_ON' ;;
    *) die "unknown mode: $1 (off|on)" ;;
  esac
}

variant_macro() {
  local v
  v="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
  printf 'BUNDLE_VARIANT_%s' "$v"
}

load_manifest() {
  [[ -f "$MANIFEST" ]] || die "missing MANIFEST: $MANIFEST"
  while read -r mc _tg _sha onnx_rel _input_sample || [[ -n "${mc:-}" ]]; do
    [[ -z "${mc:-}" || "$mc" == \#* ]] && continue
    [[ "$mc" == MC* ]] || continue
    local family variant net
    family="${onnx_rel%%/*}"
    variant="$(printf '%s' "$onnx_rel" | cut -d/ -f2)"
    net="$(basename "$onnx_rel" .onnx)"
    printf '%s|%s|%s|%s|%s\n' "$mc" "$family" "$variant" "$net" "$onnx_rel"
  done <"$MANIFEST"
}

case_key() {
  local c="$1"
  if [[ "$c" =~ ^(MC|IC|TV|TA|MC) ]]; then
    printf '%s' "${c:2}"
  else
    printf '%s' "$c"
  fi
}

select_rows() {
  local row mc fam var net onnx_rel
  while IFS='|' read -r mc fam var net onnx_rel; do
    if [[ ${#CASES[@]} -eq 0 ]]; then
      printf '%s|%s|%s|%s|%s\n' "$mc" "$fam" "$var" "$net" "$onnx_rel"
      continue
    fi
    local f ok=0 n
    n="$(case_key "$mc")"
    for f in "${CASES[@]}"; do
      if [[ "$f" == "$mc" || "$f" == "IC${n}" || "$f" == "$n" || "$f" == "$fam/$var" || "$f" == "$fam" || "$f" == "$var" || "$f" == "$net" ]]; then
        ok=1
        break
      fi
    done
    [[ "$ok" -eq 1 ]] && printf '%s|%s|%s|%s|%s\n' "$mc" "$fam" "$var" "$net" "$onnx_rel"
  done
}

compile_mc_one() {
  local onnx_path="$1"
  local out_dir="$2"
  local net_base="$3"
  local case_name="$4"
  local wb_protect="$5"
  local maxpool_patch="$6"

  local net_name="${net_base}_${case_name}"
  local -a extra_args=()
  mkdir -p "$out_dir"

  if [[ "$maxpool_patch" == "0" ]]; then
    extra_args+=(--x86-maxpool-spill-patch=false)
  fi

  {
    printf 'timestamp=%s\n' "$(date -Iseconds)"
    printf 'writeback_protect=%s\n' "$wb_protect"
    printf 'maxpool_patch=%s\n' "$maxpool_patch"
    printf 'backend=%s\n' "$BACKEND"
    printf 'onnx=%s\n' "$onnx_path"
    printf 'out_dir=%s\n' "$out_dir"

    if [[ "$wb_protect" == "1" ]]; then
      # shellcheck disable=SC1091
      source "$COMMON_DIR/wb_protect_compile_env.sh"
    fi

    env \
      LD_LIBRARY_PATH="$MODEL_COMPILER_LD_LIBRARY_PATH" \
      GLOW_WRITEBACK_PROTECT="$wb_protect" \
      GLOW_RELU_LOW12_PATCH=0 \
      GLOW_MAXPOOL_LOWBIT_PATCH="$maxpool_patch" \
      "$MODEL_COMPILER" \
      "${extra_args[@]}" \
      "-model=$onnx_path" \
      "-backend=$BACKEND" \
      "-emit-bundle=$out_dir" \
      "-network-name=$net_name" \
      "-main-entry-name=$net_name"
  } >"$out_dir/model_compiler.stdout.txt" 2>"$out_dir/model_compiler.stderr.txt"

  [[ -f "$out_dir/${net_name}.o" ]] || die "missing object: $out_dir/${net_name}.o (see model_compiler.stderr.txt)"
  [[ -f "$out_dir/${net_name}.weights.bin" ]] || die "missing weights: $out_dir/${net_name}.weights.bin"
}

link_mc_runner() {
  local family="$1"
  local out_dir="$2"
  local net_base="$3"
  local variant="$4"
  local case_name="$5"
  local case_macro="$6"

  local runner_cpp="$RUNNER_SRC_ROOT/$family/bundle_runner.cpp"
  local net="${net_base}_${case_name}"
  local vmacro
  vmacro="$(variant_macro "$variant")"

  [[ -f "$runner_cpp" ]] || die "missing runner source: $runner_cpp"
  [[ -f "$COMMON_DIR/FixedAddressArena.h" ]] || die "missing FixedAddressArena.h"
  [[ -f "$GLOW_SRC/lib/Support/WritebackPatchRuntime.cpp" ]] || die "missing WritebackPatchRuntime.cpp"
  [[ -f "$GLOW_SRC/lib/Support/WritebackPatchSeedRuntime.cpp" ]] || die "missing WritebackPatchSeedRuntime.cpp"

  "$CXX" $CXXFLAGS \
    "$runner_cpp" \
    "$GLOW_SRC/lib/Support/WritebackPatchRuntime.cpp" \
    "$GLOW_SRC/lib/Support/WritebackPatchSeedRuntime.cpp" \
    "$out_dir/${net}.o" \
    -I"$out_dir" \
    -I"$COMMON_DIR" \
    -I"$GLOW_SRC/include" \
    -D"$case_macro" \
    -D"$vmacro" \
    $LDFLAGS \
    -o "$out_dir/${net}_runner"

  [[ -x "$out_dir/${net}_runner" ]] || die "failed to link runner: $out_dir/${net}_runner"
}

# Public names match TVM: libs/MCxx/{off,on}/MCxx_{runner,weights.bin,h}.
# model-compiler / runner_src still emit the long family_case names; alias them.
install_stable_names() {
  local out_dir="$1" mc="$2" net_name="$3"
  ln -sfn "${net_name}_runner" "$out_dir/${mc}_runner"
  ln -sfn "${net_name}.weights.bin" "$out_dir/${mc}.weights.bin"
  ln -sfn "${net_name}.h" "$out_dir/${mc}.h"
}

write_ic_meta() {
  local ic="$1" family="$2" variant="$3" net="$4" onnx_rel="$5"
  local out_dir="$LIBS_ROOT/$ic"
  mkdir -p "$out_dir"
  cat >"$out_dir/meta.json" <<EOF
{
  "case": "$ic",
  "runtime": "image-classifier",
  "onnx_rel": "$onnx_rel",
  "onnx": "$MODELS_DIR/$onnx_rel",
  "input_root": "$ART_ROOT/data/inputs/$family/$variant",
  "family": "$family",
  "dataset": "$variant",
  "net": "$net",
  "note": "off/on are GLOW_WRITEBACK_PROTECT runtime flags; IC has no compiled bundle"
}
EOF
  echo "  wrote $out_dir/meta.json"
}

build_mc_row() {
  local mc="$1" family="$2" variant="$3" net_base="$4" onnx_rel="$5"
  local onnx_path="$MODELS_DIR/$onnx_rel"
  local mode line mode_name case_name wb mp case_macro out_dir
  [[ -f "$onnx_path" ]] || die "missing ONNX: $onnx_path"
  echo "===== $mc $family/$variant ($net_base) ====="
  IFS=',' read -r -a mode_list <<<"$MODES"
  for mode in "${mode_list[@]}"; do
    mode="$(echo "$mode" | tr -d ' ')"
    [[ -n "$mode" ]] || continue
    IFS=':' read -r mode_name case_name wb mp case_macro <<<"$(cases_for_mode "$mode")"
    out_dir="$LIBS_ROOT/$mc/$mode_name"
    compile_mc_one "$onnx_path" "$out_dir" "$net_base" "$case_name" "$wb" "$mp"
    if [[ "$REBUILD_RUNNERS" -eq 1 ]]; then
      link_mc_runner "$family" "$out_dir" "$net_base" "$variant" "$case_name" "$case_macro"
    fi
    install_stable_names "$out_dir" "$mc" "${net_base}_${case_name}"
    echo "  ok $mc/$mode_name"
  done
}

# ---- args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --runtime)
      RUNTIME="$2"
      shift 2
      ;;
    --case)
      CASES+=("$2")
      shift 2
      ;;
    --modes)
      MODES="$2"
      shift 2
      ;;
    --no-runners) REBUILD_RUNNERS=0; shift ;;
    --)
      shift
      CASES+=("$@")
      break
      ;;
    -*)
      die "unknown option: $1 (try --help)"
      ;;
    *)
      CASES+=("$1")
      shift
      ;;
  esac
done

[[ "$RUNTIME" == "mc" || "$RUNTIME" == "ic" || "$RUNTIME" == "both" ]] || die "--runtime must be mc|ic|both"

mapfile -t ROWS < <(load_manifest | select_rows)
[[ ${#ROWS[@]} -gt 0 ]] || die "no MANIFEST rows matched: ${CASES[*]:-<all>}"

mkdir -p "$LIBS_ROOT"
LOG="$LIBS_ROOT/build_all.log"
{
  echo "start $(date -Iseconds)"
  echo "RUNTIME=$RUNTIME"
  echo "MODEL_COMPILER=$MODEL_COMPILER"
  echo "LIBS_ROOT=$LIBS_ROOT"
} | tee "$LOG"

if [[ "$RUNTIME" == "mc" || "$RUNTIME" == "both" ]]; then
  [[ -x "$MODEL_COMPILER" ]] || die "missing model-compiler: $MODEL_COMPILER (scripts/build_writeback_protect.sh mc)"
  [[ -d "$RUNNER_SRC_ROOT" ]] || die "missing runner_src: $RUNNER_SRC_ROOT"
  command -v "$CXX" >/dev/null || die "CXX not found: $CXX"
fi

built=0
for row in "${ROWS[@]}"; do
  IFS='|' read -r mc fam var net onnx_rel <<<"$row"
  if [[ "$RUNTIME" == "mc" || "$RUNTIME" == "both" ]]; then
    build_mc_row "$mc" "$fam" "$var" "$net" "$onnx_rel" | tee -a "$LOG"
  fi
  if [[ "$RUNTIME" == "ic" || "$RUNTIME" == "both" ]]; then
    write_ic_meta "IC${mc#MC}" "$fam" "$var" "$net" "$onnx_rel" | tee -a "$LOG"
  fi
  built=$((built + 1))
done

echo "DONE $(date -Iseconds) models=$built libs=$LIBS_ROOT" | tee -a "$LOG"
