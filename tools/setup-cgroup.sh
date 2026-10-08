#!/bin/bash
# Prepares the cgroup v2 parent for the functions' CPU caps (executor.cgroup-root) and optionally writes it into an
# orchestrator config (with executor.cpu-cap = true). Rerun after a reboot (cgroupfs is not persistent).
#
#   tools/setup-cgroup.sh [config.json]
#       No root: creates <user@UID.service>/mignificient in the user's own systemd tree. Works when the orchestrator
#       is started from a shell inside that tree (desktop terminals; check with `cat /proc/self/cgroup`).
#   sudo tools/setup-cgroup.sh --pid $$ [config.json]
#       Root: creates /sys/fs/cgroup/mignificient-<user>, delegates it to the user and moves the shell with pid $$
#       into its "shell" leaf; start the orchestrator from that shell (e.g. over ssh).
#
# The orchestrator moves executors (and gpuless servers) from its own cgroup into per-client children, which needs
# write access to the common ancestor: hence the orchestrator must run inside the tree prepared here.
set -euo pipefail

CG=/sys/fs/cgroup
pid=""
if [ "${1:-}" = "--pid" ]; then
  pid=$2
  shift 2
fi
config=${1:-}

if [ "$(id -u)" = 0 ]; then
  [ -n "$pid" ] || { echo "as root, pass the shell to move: sudo $0 --pid \$\$ [config]" >&2; exit 1; }
  user=${SUDO_USER:?run through sudo}
  uid=$(id -u "$user"); gid=$(id -g "$user")
  root=$CG/mignificient-$user
  grep -qw cpu $CG/cgroup.subtree_control || echo +cpu > $CG/cgroup.subtree_control
  mkdir -p "$root/shell"
  # Delegation (cgroup v2): the directories and their procs/threads/subtree_control files.
  for d in "$root" "$root/shell"; do
    chown "$uid:$gid" "$d" "$d/cgroup.procs" "$d/cgroup.threads" "$d/cgroup.subtree_control"
  done
  echo "$pid" > "$root/shell/cgroup.procs"
  echo +cpu > "$root/cgroup.subtree_control"
  echo "moved pid $pid into $root/shell"
else
  [ -z "$pid" ] || { echo "--pid needs root" >&2; exit 1; }
  uid=$(id -u)
  service=$CG/user.slice/user-$uid.slice/user@$uid.service
  case "$(cut -d: -f3 /proc/self/cgroup)" in
    /user.slice/user-$uid.slice/user@$uid.service/*) ;;
    *) echo "this shell is not in $service (ssh session?): use sudo $0 --pid \$\$" >&2; exit 1 ;;
  esac
  grep -qw cpu "$service/cgroup.subtree_control" || { echo "cpu controller not delegated to $service" >&2; exit 1; }
  root=$service/mignificient
  mkdir -p "$root"
  echo +cpu > "$root/cgroup.subtree_control"
fi

echo "cgroup-root: $root"
if [ -n "$config" ]; then
  python3 - "$config" "$root" <<'EOF'
import json, sys
path, root = sys.argv[1:]
cfg = json.load(open(path))
cfg["executor"]["cgroup-root"] = root
cfg["executor"]["cpu-cap"] = True
json.dump(cfg, open(path, "w"), indent=2)
print(f"set executor.cgroup-root and cpu-cap in {path}")
EOF
fi
