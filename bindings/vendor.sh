#!/usr/bin/env bash
# vendor.sh - copies vecindex's C sources (src/, include/, LICENSE) into each binding's csrc/, which their builds use
# when they are not inside the repository: run it before `python -m build --sdist`, `cargo package` or `npm pack`.
set -euo pipefail
cd "$(dirname "$0")"
for b in python rust js; do
    [[ -d $b ]] || continue
    rm -rf "$b/csrc"
    mkdir -p "$b/csrc"
    cp -r ../src ../include ../LICENSE "$b/csrc/"
    echo "vendor: $b/csrc"
done
