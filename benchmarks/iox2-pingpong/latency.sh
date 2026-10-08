#!/bin/bash

# iceoryx2 ping-pong RTT: file backend (patched iceoryx2) vs POSIX shm (unpatched), wait vs busy,
# host<->host and host ping <-> container pong. Roots are a private dir on tmpfs, removed at the end.
#   latency.sh FILE_BUILD_DIR [POSIX_BUILD_DIR]     (dirs that hold the pingpong binary)
# Env: N=20000 round trips, REPS=3, MODES="wait busy", PEERS="host container", IMAGE=ubuntu:24.04,
#      PING_CPU=6, PONG_CPU=4 (taskset/cpuset), SHM=/dev/shm (tmpfs for the roots).
# The container runs the host binary, so IMAGE needs a glibc/libstdc++ compatible with the host's.
#
set -u
FILE_BIN=$(readlink -f "${1:?usage: latency.sh FILE_BUILD_DIR [POSIX_BUILD_DIR]}")/pingpong
POSIX_BIN=${2:+$(readlink -f "$2")/pingpong}
N=${N:-20000} REPS=${REPS:-3} MODES=${MODES:-wait busy} PEERS=${PEERS:-host container}
IMAGE=${IMAGE:-ubuntu:24.04} PING_CPU=${PING_CPU:-6} PONG_CPU=${PONG_CPU:-4}
for b in "$FILE_BIN" $POSIX_BIN; do [ -x "$b" ] || {
  echo "no pingpong binary: $b" >&2
  exit 1
}; done

BASE=$(mktemp -d "${SHM:-/dev/shm}/iox2-pp-XXXXXX") # 0700
LABEL=mignificient.test=iox2-pingpong-$$
export IOX2_PREFIX=pp$$_ # private iceoryx2 names (POSIX shm files in /dev/shm)
PONG=
cleanup() {
  [ -n "$PONG" ] && kill "$PONG" 2>/dev/null
  ids=$(docker ps -aq --filter "label=$LABEL" 2>/dev/null)
  [ -n "$ids" ] && docker rm -f $ids >/dev/null
  rm -rf "$BASE"
  rm -f /dev/shm/"$IOX2_PREFIX"*
}
trap cleanup EXIT

run_host() { # label bin root busy
  echo -n "$1: "
  IOX2_ROOT=$3 BUSY=$4 taskset -c "$PONG_CPU" timeout 150 "$2" pong &
  PONG=$!
  sleep 0.3
  IOX2_ROOT=$3 BUSY=$4 taskset -c "$PING_CPU" timeout 120 "$2" ping "$N"
  wait "$PONG"
  PONG=
}

run_container() { # label root busy  (file backend only: the container sees nothing but the root dir)
  echo -n "$1: "
  local dir
  dir=$(dirname "$FILE_BIN")
  docker run --rm -d --label "$LABEL" --user "$(id -u):$(id -g)" --cpuset-cpus "$PONG_CPU" \
    --mount "type=bind,source=${2%/},target=${2%/}" -v "$dir:/pp:ro" -e IOX2_ROOT="$2" -e IOX2_PREFIX ${3:+-e BUSY=1} \
    "$IMAGE" timeout 150 /pp/pingpong pong >/dev/null || return
  sleep 1.5
  IOX2_ROOT=$2 BUSY=$3 taskset -c "$PING_CPU" timeout 120 "$FILE_BIN" ping "$N"
  docker wait $(docker ps -q --filter "label=$LABEL") >/dev/null 2>&1 # pong gone before the next rep reuses the root
}

mkdir -p "$BASE/file" "$BASE/posix"
for rep in $(seq "$REPS"); do
  for mode in $MODES; do
    busy=
    [ "$mode" = busy ] && busy=1
    for peer in $PEERS; do
      if [ "$peer" = host ]; then
        [ -n "$POSIX_BIN" ] && run_host "posix host<->host $mode rep$rep" "$POSIX_BIN" "$BASE/posix/" "$busy"
        run_host "file  host<->host $mode rep$rep" "$FILE_BIN" "$BASE/file/" "$busy"
      else
        run_container "file  host<->ct($IMAGE) $mode rep$rep" "$BASE/file/" "$busy"
      fi
    done
  done
done
