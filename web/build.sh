#!/usr/bin/env bash
# Build the wasm (SIMD128) into web/pkg/timestral.wasm.
# Needs: rustup target add wasm32-unknown-unknown. wasm-opt (binaryen) is used when present.
set -euo pipefail
cd "$(dirname "$0")"
# Run from the crate dir: cargo only reads .cargo/config.toml (which turns on +simd128) from the cwd upwards.
(cd timestral-wasm && cargo build --release --target wasm32-unknown-unknown)
mkdir -p pkg
src=timestral-wasm/target/wasm32-unknown-unknown/release/timestral_wasm.wasm
if command -v wasm-opt >/dev/null; then
  wasm-opt -O3 --all-features "$src" -o pkg/timestral.wasm
else
  cp "$src" pkg/timestral.wasm
fi
ls -l pkg/timestral.wasm
