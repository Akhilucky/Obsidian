#!/usr/bin/env bash
# Build the C++ performance kernels for Obsidian Terminal.
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-clang++}"
FLAGS="-O3 -std=c++17 -shared -fPIC"

echo ">> Building libobsidian_core..."
"$CXX" $FLAGS obsidian_core.cpp -o libobsidian_core.dylib || {
    echo ">> clang++ failed, trying g++..."
    CXX=g++
    "$CXX" $FLAGS obsidian_core.cpp -o libobsidian_core.dylib
}
echo ">> OK: $(pwd)/libobsidian_core.dylib"
