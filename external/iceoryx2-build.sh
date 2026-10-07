#!/bin/sh
# Builds and installs the pinned, patched iceoryx2. The single source of the version and the patch,
# used by the host build (README) and docker/Dockerfile.iceoryx.
#   iceoryx2-build.sh <src-dir> <build-dir> <install-prefix>
# The patch (iceoryx2-file-backend.patch) makes the ipc service file-backed: data segments, dynamic
# configs and connections become files under global.root-path instead of POSIX shm in /dev/shm,
# so each client's iceoryx2 state lives in its own directory. Every process that talks to another
# one must use this build: an unpatched process looks in /dev/shm and never finds the services.
# IOX2_NO_PATCH=1 builds plain upstream v0.8.1 (POSIX shm backend) instead, for comparisons such as
# benchmarks/iox2-pingpong; use separate src/build/install dirs for it.
set -eu
VERSION=v0.8.1
PATCH="$(cd "$(dirname "$0")" && pwd)/iceoryx2-file-backend.patch"
SRC=$1 BUILD=$2 PREFIX=$3

if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "$VERSION" https://github.com/eclipse-iceoryx/iceoryx2.git "$SRC"
fi
if [ "$(git -C "$SRC" describe --tags --exact-match HEAD)" != "$VERSION" ]; then
  echo "$SRC is not at $VERSION" >&2; exit 1
fi
# Apply once; a re-run on an already patched tree is fine.
if [ -n "${IOX2_NO_PATCH:-}" ]; then
  if git -C "$SRC" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "IOX2_NO_PATCH set but $SRC is patched; use a separate source dir" >&2; exit 1
  fi
elif ! git -C "$SRC" apply --reverse --check "$PATCH" 2>/dev/null; then
  git -C "$SRC" apply "$PATCH"
fi
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DBUILD_CXX=ON \
  -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build "$BUILD" -j"$(nproc)"
cmake --install "$BUILD"
