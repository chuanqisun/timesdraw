#!/usr/bin/env bash
# Build the standalone app into web/dist/: index.html, app.js, worker.js, timestral.js and timestral.wasm.
# All URLs in it are relative, so the folder can be served from the root or any sub-path of a static host.
# Needs: rustup target add wasm32-unknown-unknown. wasm-opt (binaryen) is used when present.
set -euo pipefail
cd "$(dirname "$0")"

# Run from the crate dir: cargo only reads .cargo/config.toml (which turns on +simd128) from the cwd upwards.
(cd timestral-wasm && cargo build --release --target wasm32-unknown-unknown)
src=timestral-wasm/target/wasm32-unknown-unknown/release/timestral_wasm.wasm

rm -rf dist
mkdir dist
cp index.html app.js worker.js timestral.js dist/
if command -v wasm-opt >/dev/null; then
  wasm-opt -O3 --all-features "$src" -o dist/timestral.wasm
else
  cp "$src" dist/timestral.wasm
fi
ls -l dist
