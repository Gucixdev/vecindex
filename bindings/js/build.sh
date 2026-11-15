#!/usr/bin/env bash
# build.sh - vecindex.wasm from vecindex's C (src/vecindex.c, one unity file) with the WASI SDK: a wasm32-wasip1
# reactor that imports only wasi_snapshot_preview1 (index.mjs implements those calls itself) and exports the C API.
#   WASI_SDK=/path/to/wasi-sdk-29.0-x86_64-linux [WASM_OPT=wasm-opt] ./build.sh        (default: /opt/wasi-sdk)
# https://github.com/WebAssembly/wasi-sdk/releases: unpack the x86_64-linux (or arm64-linux, macos) tarball anywhere.
set -euo pipefail
# the SHA-256 of every C source and header the module is built from, in a fixed order
srchash() { (cd "$1" && LC_ALL=C sha256sum src/*.c src/*.h include/*.h | sha256sum | cut -d' ' -f1); }
[[ ${1:-} == --srchash ]] && { srchash "${2:?}"; exit 0; }
here=$(cd "$(dirname "$0")" && pwd)
root=$here/../..
[[ -f $root/src/vecindex.c ]] || root=$here/csrc
W=${WASI_SDK:-/opt/wasi-sdk}
[[ -x $W/bin/clang ]] || { echo "build.sh: no WASI SDK at $W (set WASI_SDK)" >&2; exit 2; }
exports=$(grep -o 'VI_API [^;]*' "$root/include/vecindex.h" | grep -o '\bvi_[a-z_0-9]*(' | tr -d '(' | sort -u \
          | sed 's/^/-Wl,--export=/')
# shellcheck disable=SC2086
"$W/bin/clang" --target=wasm32-wasip1 --sysroot="$W/share/wasi-sysroot" -O2 -std=gnu99 -mexec-model=reactor -mcpu=mvp -msign-ext -mmutable-globals -mbulk-memory -mnontrapping-fptoint \
    -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS -D_WASI_EMULATED_GETPID \
    -I"$root/include" -I"$root/src" "$root/src/vecindex.c" -o "$here/vecindex.wasm" \
    $exports -Wl,--export=malloc -Wl,--export=free -Wl,--strip-all \
    -lwasi-emulated-mman -lwasi-emulated-signal -lwasi-emulated-process-clocks -lwasi-emulated-getpid
# wasi-libc comes compiled with reference types, whose encoding of call_indirect engines before V8 9.6 (Node 16,
# Safari 14) refuse; wasm-opt (binaryen: npm i binaryen) writes it back the old way and optimizes. Without it the
# module still runs on Node 18+ and every current browser.
opt=${WASM_OPT:-$(command -v wasm-opt || true)}
if [[ -n $opt ]]; then
    "$opt" --mvp-features --enable-bulk-memory --enable-sign-ext --enable-mutable-globals \
        --enable-nontrapping-float-to-int -O2 "$here/vecindex.wasm" -o "$here/vecindex.wasm"
else
    echo "build.sh: no wasm-opt (set WASM_OPT): the module needs Node 18+ or a current browser" >&2
fi
srchash "$root" > "$here/vecindex.wasm.src"                  # which C it was built from: ./run says when it is older
echo "build.sh: $here/vecindex.wasm ($(wc -c < "$here/vecindex.wasm") bytes)"
