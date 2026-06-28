#!/usr/bin/env bash
# build.sh — compiles hybrid_sigma_strategy.cpp into libhybridsigma.(so|dylib)
# for the platform you run this on. The .so shipped alongside this script
# was built in a Linux x86_64 sandbox and will NOT work on macOS/Windows —
# always rebuild locally with this script.
set -euo pipefail

SRC="cpp/hybrid_sigma_strategy.cpp"
OUT="libhybridsigma.so"
EXTRA_FLAGS=""

case "$(uname -s)" in
    Darwin*)
        OUT="libhybridsigma.dylib"
        EXTRA_FLAGS="-undefined dynamic_lookup"
        ;;
    Linux*)
        OUT="libhybridsigma.so"
        ;;
    *)
        echo "Unrecognized platform '$(uname -s)'. On Windows, use MSYS2/MinGW or WSL and adapt this script." >&2
        exit 1
        ;;
esac

if ! command -v g++ >/dev/null 2>&1; then
    echo "g++ not found. Install a C++23-capable compiler (GCC 13+/Clang 16+) first." >&2
    exit 1
fi

echo "Compiling ${SRC} -> ${OUT} ..."
g++ -std=c++23 -O3 -march=native -Wall -Wextra -shared -fPIC \
    -DHYBRID_SIGMA_C_ABI -x c++ "${SRC}" -o "${OUT}" ${EXTRA_FLAGS}

echo "Build OK -> ${OUT}"
echo "If this is macOS/Linux ARM or you saw illegal-instruction errors, drop -march=native and rebuild."
