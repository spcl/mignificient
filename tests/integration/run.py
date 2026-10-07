#!/usr/bin/env python3
"""Integration harness: orchestrator -> N ok requests -> handler switch + bad config -> error -> hang -> recovery -> teardown/leak check.
Exit 0 = pass, 1 = fail. Python stdlib only."""
import argparse, json, os, shutil, signal, socket, subprocess, sys, tempfile, threading, time
import urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
LEAK_NAMES = ("manager_device", "executor_cpp", "executor_python")
IOX_HINT = ("hint: stale iceoryx2 state from another iceoryx2 version? Remove /tmp/iceoryx2 "
            "manually if no other iceoryx2 apps run.")


class Fail(Exception):
    pass


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def device_db(build):
    d = os.path.join(build, "test-cache")
    if not os.path.exists(os.path.join(d, "devices.json")):
        subprocess.run([os.path.join(REPO, "tools", "list-gpus.sh"), d], check=True,
                       stdout=subprocess.DEVNULL, timeout=60)
    return os.path.join(d, "devices.json")


def start_orchestrator(args, tmp, port, state):
    with open(os.path.join(args.build, "config", "orchestrator.json")) as f:
        cfg = json.load(f)
    cfg["http"]["port"] = port
    cfg["executor"]["type"] = args.executor
    cfg["ipc"]["backend"] = args.signal  # ponytail: futex will need its own key once it exists
    cfg["timeout-check-interval-ms"] = 100
    cfg_path = os.path.join(tmp, "orchestrator.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    log = open(os.path.join(tmp, "orchestrator.log"), "w")
    proc = subprocess.Popen([os.path.join(args.build, "orchestrator", "orchestrator"), cfg_path,
                             device_db(args.build)],
                            cwd=tmp, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state["proc"] = proc  # register at once so teardown/watchdog see it during startup
    t0 = time.time()
    while time.time() - t0 < 30:
        if proc.poll() is not None:
            raise Fail(f"orchestrator exited early with {proc.returncode}\n{IOX_HINT}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return proc
        except OSError:
            time.sleep(0.1)
    raise Fail("orchestrator not ready after 30 s\n" + IOX_HINT)


def request(port, case, block, user, payload=None):
    spec = case[block]
    # client key = user+function and the executor binds the symbol once, so each block is its own client
    body = {
        "function": spec["function"], "function-handler": spec.get("function-handler", spec["function"]),
        "function-language": case["language"], "function-path": spec.get("function-path", case["function-path"]),
        "user": user, "uuid": f"it-{block}-{time.time_ns()}", "modules": [],
        "mig-instance": "7g", "gpu-memory": case["gpu-memory"],
        "timeout": spec.get("timeout", case["timeout"]),
        "input-payload": json.dumps(spec.get("payload", {}) if payload is None else payload),
    }
    req = urllib.request.Request(f"http://127.0.0.1:{port}/invoke", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=case["timeout"] + 10) as r:
            status, text = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        status, text = e.code, e.read().decode()
    except (urllib.error.URLError, OSError) as e:
        raise Fail(f"{block}: no HTTP response: {e}")
    return status, text, time.time() - t0


def step(port, case, block, user, label, want, max_s=None, payload=None):
    status, text, dt = request(port, case, block, user, payload)
    print(f"{label:<24} {block:<5} status={status} ms={dt*1000:.0f}", flush=True)
    if status != want:
        raise Fail(f"{label}: {block} expected HTTP {want}, got {status}: {text[:300]}")
    if max_s is not None and dt > max_s:
        raise Fail(f"{label}: {block} took {dt:.1f}s, limit {max_s:.1f}s")
    expect = case[block].get("expect")
    if expect:
        got = json.loads(text)
        for k, v in expect.items():
            if got.get(k) != v:
                raise Fail(f"{label}: expected {k}={v!r}, got {got.get(k)!r}")


def group_leaks(pgid):
    out = subprocess.run(["ps", "-eo", "pid,pgid,args"], capture_output=True, text=True).stdout
    leaks = []
    for line in out.splitlines()[1:]:
        pid, pg, cmd = line.split(None, 2)
        if int(pg) == pgid and any(n in cmd for n in LEAK_NAMES):
            leaks.append(f"{pid} {cmd}")
    return leaks


def teardown(proc):
    """Returns a list of leak descriptions. Add the docker label check here later."""
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        pass
    time.sleep(0.5)
    leaks = group_leaks(pgid)
    if leaks:  # give children a moment, then force
        time.sleep(2)
        leaks = group_leaks(pgid)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    return leaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--executor", choices=["bare-metal"], required=True)
    ap.add_argument("--signal", choices=["iceoryx2"], required=True)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--deadline", type=int, default=600)
    ap.add_argument("--keep-logs", action="store_true")
    args = ap.parse_args()
    args.build = os.path.abspath(args.build)

    with open(args.case) as f:
        case = json.loads(f.read().replace("@BUILD@", args.build))
    name = os.path.splitext(os.path.basename(args.case))[0]
    tmp = tempfile.mkdtemp(prefix=f"it-{name}-")
    port = free_port()
    state = {"proc": None}
    failed = False

    def watchdog():
        print(f"FAIL: global deadline {args.deadline}s exceeded", file=sys.stderr, flush=True)
        os.kill(os.getpid(), signal.SIGTERM)  # -> SystemExit -> finally (teardown, logs, cleanup)
        time.sleep(30)  # last resort if teardown itself is stuck
        if state["proc"]:
            try:
                os.killpg(state["proc"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        os._exit(1)
    wd = threading.Timer(args.deadline, watchdog)
    wd.daemon = True
    wd.start()

    def on_term(signum, frame):
        raise SystemExit(1)
    signal.signal(signal.SIGTERM, on_term)

    try:
        user = "it-user"
        start_orchestrator(args, tmp, port, state)
        for i in range(args.iterations):
            step(port, case, "ok", user, f"ok-{i}" + (" cold" if i == 0 else ""), 200)
        # check that if we switch to a failing function handler for the same client,
        # then we truly execute that handler.
        # if we execute the old handler, we would get 200, but we expect 500.
        step(port, case, "switch", user, "handler-switch-new-client", 500, max_s=case["timeout"] / 2)
        step(port, case, "ok", user, "ok-after-switch", 200)
        before = group_leaks(state["proc"].pid)
        # verify that if we send wrong but different config, system will not spawn
        # new executor/manager process, but will reject the request
        step(port, case, "badpath", user, "inconsistent-config-rejected", 400, max_s=5)
        if group_leaks(state["proc"].pid) != before:
            raise Fail("inconsistent-config-rejected: a new executor/manager process was spawned")
        step(port, case, "fail", user, "error", 500)
        step(port, case, "fail", user, "fail-repeat-same-client", 500, max_s=case["timeout"] / 2)
        step(port, case, "ok", user, "ok-after-error", 200)
        hang_to = case["hang"].get("timeout", 3)
        step(port, case, "hang", user, "hang", 504, max_s=hang_to + 0.1 + 10)
        # after a hang, the next invocation without the sleep (set to 0)
        # this one should succeed
        step(port, case, "hang", user, "hang-restart-same-client", 200,
             payload=case["hang"].get("restart-payload", {"sleep-s": 0}))
        step(port, case, "ok", user, "ok-after-hang", 200)
    except Fail as e:
        failed = True
        print(f"FAIL: {e}", file=sys.stderr)
    except BaseException as e:  # incl. SystemExit/KeyboardInterrupt: still tear down, then fail
        failed = True
        print(f"FAIL: interrupted/unexpected {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if state["proc"]:
                leaks = teardown(state["proc"])
                if leaks:
                    failed = True
                    print("FAIL: leaked processes:\n  " + "\n  ".join(leaks), file=sys.stderr)
        finally:
            wd.cancel()
            if failed or args.keep_logs:
                dst = os.path.join(args.build, "test-logs", name)
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(tmp, dst)
                print(f"logs: {dst}")
            shutil.rmtree(tmp, ignore_errors=True)
    print("FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
