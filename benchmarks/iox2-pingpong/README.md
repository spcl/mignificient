# iceoryx2 ping-pong RTT

Round-trip time of one 64-byte publish/subscribe message plus an event notification per direction, the pattern of
the executor ↔ gpuless hot path. `ping` sends, `pong` echoes; 1000 warm-up round trips, then N timed ones.
Variants:
- wait (listener `blocking_wait_one`) vs busy (`BUSY=1`, spin on `receive()`)
- file-backed (our patch) vs POSIX shm (unpatched) iceoryx2
- host↔host vs host ping ↔ container pong

## Build

Standalone CMake project (not part of the main build), linked statically against an iceoryx2 install.

```bash
# patched, file-backed iceoryx2 (the default MIGnificient build, see the top-level README)
external/iceoryx2-build.sh <build-dir>/iox2-src <build-dir>/iox2-build $PWD/<build_dir>/iox2-install
cmake -S benchmarks/iox2-pingpong -B <build-dir>/pingpong-file -DCMAKE_PREFIX_PATH=$PWD/<build-dir>/iox2-install
cmake --build <build-dir>/pingpong-file

# unpatched upstream iceoryx2 v0.8.1 (POSIX shm backend), separate dirs
IOX2_NO_PATCH=1 external/iceoryx2-build.sh <build-dir>/iox2-src-posix <build-dir>/iox2-build-posix $PWD/<build-dir>/iox2-install-posix
cmake -S benchmarks/iox2-pingpong -B <build-dir>/pingpong-posix -DCMAKE_PREFIX_PATH=$PWD/<build-dir>/iox2-install-posix
cmake --build <build-dir>/pingpong-posix
```

## Run

```bash
# full matrix: 3 reps x {wait, busy} x {POSIX, file} host<->host, plus file host<->container (ubuntu:24.04)
benchmarks/iox2-pingpong/latency.sh <build-dir>/pingpong-file <build-dir>/pingpong-posix
# one short variant: file backend, host<->host, wait
N=2000 REPS=1 MODES=wait PEERS=host benchmarks/iox2-pingpong/latency.sh <build-dir>/pingpong-file
```

Configurations:
- `N` (20000)
- `REPS` (3)
- `MODES` (`wait busy`)
- `PEERS` (`host container`)
- `IMAGE` (`ubuntu:24.04`),
- `PING_CPU`/`PONG_CPU` (6/4, pinned with taskset / `--cpuset-cpus`)
- `SHM` (`/dev/shm`).

Without the POSIX shmem build dir, only the file variants run; the container variant is file backend only.
The script uses a private root dir under `/dev/shm` and a private iceoryx2 prefix, labels its containers, and removes
all of it on exit. The container runs the host binary, so `IMAGE` needs a compatible glibc/libstdc++.

By hand: `IOX2_ROOT=<dir>/ [IOX2_PREFIX=<p>] [BUSY=1] pingpong pong &` then the same env with `pingpong ping [N]`.
`pingpong multi <rootA>/ <rootB>/` checks that two nodes with different roots in one process are isolated.

## Reference results

2026-10-07, RTX 4070 Ti machine (Intel i7-12700K, Linux 6.8), iceoryx2 v0.8.1, roots on tmpfs, N = 20000,
3 reps (host↔host) / 1 rep (container), ping on CPU 6, pong on CPU 4. RTT in µs, p50 / p99.

| Variant | POSIX shm (unpatched) | file backend (patched) |
|---|---|---|
| host↔host, wait | 5.80 (5.79–5.80) / 6.82–6.94 | 5.85 (5.84–5.86) / 6.78–7.06 |
| host↔host, busy | 2.57 (2.55–2.59) / 3.84–3.93 | 2.60 (2.58–2.61) / 3.88–3.93 |
| host ping ↔ container pong, wait | – | 6.19 / 7.38 |
| host ping ↔ container pong, busy | – | 2.67 / 3.99 |

The file backend costs nothing measurable over POSIX shm; busy-polling halves the RTT (≈ 3.2 µs saved per round trip); a container peer adds ≈ 0.1–0.4 µs.
