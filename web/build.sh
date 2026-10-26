#!/usr/bin/env bash
# Build the standalone app into web/dist/: index.html, app.js, worker.js, timestral.js, timestral.wasm
# and model/ (the Timestral-5M weights, config and license). All URLs in it are relative, so the folder
# can be served from the root or any sub-path of a static host, with no other server involved.
# Needs: rustup target add wasm32-unknown-unknown, curl, sha256sum. wasm-opt (binaryen) is used when present.
set -euo pipefail
cd "$(dirname "$0")"

REPO=mistralai/Timestral-5M
REVISION=bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f  # pinned so builds are reproducible
WEIGHTS_SHA256=7ace108c4987e7f2a99a75a445240019d1d31c2117139d1bd05ffdfd25aca738

# Download the checkpoint once into .model-cache/ (gitignored); later builds reuse it.
fetch() {  # fetch <file> [sha256]
  local f=.model-cache/$1
  if [ ! -f "$f" ] || { [ -n "${2:-}" ] && ! echo "$2  $f" | sha256sum -c --status; }; then
    mkdir -p .model-cache
    echo "downloading $1 from huggingface.co/$REPO @ ${REVISION:0:7}"
    curl -fL --progress-bar -o "$f" "https://huggingface.co/$REPO/resolve/$REVISION/$1"
  fi
  if [ -n "${2:-}" ] && ! echo "$2  $f" | sha256sum -c --status; then
    echo "checksum mismatch for $f" >&2; exit 1
  fi
}
fetch config.json
fetch LICENSE
fetch model.safetensors "$WEIGHTS_SHA256"

# Run from the crate dir: cargo only reads .cargo/config.toml (which turns on +simd128) from the cwd upwards.
(cd timestral-wasm && cargo build --release --target wasm32-unknown-unknown)
src=timestral-wasm/target/wasm32-unknown-unknown/release/timestral_wasm.wasm

rm -rf dist
mkdir -p dist/model
cp index.html app.js worker.js timestral.js dist/
cp .model-cache/config.json .model-cache/LICENSE .model-cache/model.safetensors dist/model/
if command -v wasm-opt >/dev/null; then
  wasm-opt -O3 --all-features "$src" -o dist/timestral.wasm
else
  cp "$src" dist/timestral.wasm
fi
ls -lR dist
