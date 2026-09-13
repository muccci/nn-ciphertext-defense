#!/usr/bin/env bash
# Trace-to-label: collect → train → metrics. MC01 / MC04 / MC10 / MC15.
#
#   ./scripts/run_label_attack.sh
#   ./scripts/run_label_attack.sh --foreground --case MC01
#
# Default: four cases in the background (one process each).
# Prefers the statistical conda env when it has CUDA.
set -euo pipefail

SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ART="$(cd -- "$SCRIPTS/.." && pwd)"
cd "$ART"

pick_python() {
  if [[ -n "${LABEL_PYTHON:-}" && -x "${LABEL_PYTHON}" ]]; then
    printf '%s' "$LABEL_PYTHON"
    return 0
  fi
  local cand py=""
  for cand in \
    "$HOME/.conda/envs/statistical/bin/python"
  do
    if [[ -x "$cand" ]] && "$cand" -c "import torch; assert torch.cuda.is_available()" \
        >/dev/null 2>&1; then
      printf '%s' "$cand"
      return 0
    fi
    [[ -x "$cand" && -z "$py" ]] && py="$cand"
  done
  printf '%s' "${py:-$(command -v python3)}"
}

PY="$(pick_python)"
export PATH="$(dirname "$PY"):$PATH"

DETACH=1
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fg|--foreground) DETACH=0; shift ;;
    -h|--help) DETACH=0; ARGS+=("$1"); shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ "$DETACH" == 1 ]]; then
  exec "$PY" "$SCRIPTS/run_label_attack.py" --python "$PY" --detach "${ARGS[@]}"
fi
exec "$PY" "$SCRIPTS/run_label_attack.py" --python "$PY" "${ARGS[@]}"
