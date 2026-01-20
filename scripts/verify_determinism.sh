#!/usr/bin/env bash
set -euo pipefail

ADDRESS=${1:-0x06147e073b854521c7b778280e7d7dbafb2d4898}
CHAIN=${2:-mainnet}
OUT_DIR=${3:-/tmp}
BASELINE_FILE=${4:-}
OUT1="$OUT_DIR/determinism_out1.html"
OUT2="$OUT_DIR/determinism_out2.html"

if [[ ! -f "venv/bin/activate" ]]; then
  echo "venv/bin/activate not found; create/activate the venv first." >&2
  exit 1
fi

source venv/bin/activate

python3 main.py "$ADDRESS" "$CHAIN"
cp "src/${CHAIN}_${ADDRESS}.html" "$OUT1"

python3 main.py "$ADDRESS" "$CHAIN"
cp "src/${CHAIN}_${ADDRESS}.html" "$OUT2"

if ! diff -u "$OUT1" "$OUT2" > /dev/null; then
  echo "Determinism check FAILED: outputs differ." >&2
  echo "Diff (first 200 lines):" >&2
  diff -u "$OUT1" "$OUT2" | head -n 200 >&2
  exit 2
fi

HASH=$(sha256sum "$OUT1" | awk '{print $1}')
echo "Determinism check passed: outputs are identical."
echo "Hash: $HASH"

if [[ -n "$BASELINE_FILE" ]]; then
  if [[ ! -f "$BASELINE_FILE" ]]; then
    echo "$HASH" > "$BASELINE_FILE"
    echo "Baseline saved to $BASELINE_FILE"
    exit 0
  fi
  BASELINE_HASH=$(cat "$BASELINE_FILE" | tr -d '[:space:]')
  if [[ "$HASH" != "$BASELINE_HASH" ]]; then
    echo "Baseline mismatch." >&2
    echo "Expected: $BASELINE_HASH" >&2
    echo "Actual:   $HASH" >&2
    exit 3
  fi
  echo "Baseline match."
fi
