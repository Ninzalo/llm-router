#!/usr/bin/env python3
"""Live test: Ctrl+C (SIGINT) must stop a foreground `llm-router dashboard`
immediately, even with the Logs SSE stream open and 2s polling active.

Isolated ports only (18xxx). Never touches the real :8080/:8081 servers.
"""
import os, sys, time, signal, subprocess, threading, json
import httpx

ROOT = os.path.dirname(os.path.abspath(__file__))          # tests/ (this script's dir)
PY = sys.executable
LLMROUTER = os.path.join(os.path.dirname(sys.executable), "llm-router")
CFG = f"{ROOT}/config.dash.yaml"
DASH_PORT = 18910
ROUTER_PORT = 18590
MOCK_PORTS = {18501: "mock-a", 18502: "mock-b", 18503: "mock-c"}
STATE = "/tmp/dash_ctrlc_state"
MAX_SHUTDOWN_S = 5.0
# Ensure the config's `python3 mock_llm.py` model commands resolve to THIS python.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

procs = []
def spawn(argv, **kw):
    p = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
    procs.append(p)
    return p

def wait_http(url, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False

def cleanup():
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass

def main():
    # clean state
    subprocess.run(["rm", "-rf", STATE])
    os.makedirs(STATE, exist_ok=True)
    env = dict(os.environ, LLM_ROUTER_STATE_DIR=STATE, LLM_ROUTER_CONFIG=CFG)

    ok = True
    try:
        # 1) mocks
        for port, model in MOCK_PORTS.items():
            spawn([PY, f"{ROOT}/mock_llm.py", "--host", "127.0.0.1", "--port", str(port), "--model", model], env=env)
        for port in MOCK_PORTS:
            if not wait_http(f"http://127.0.0.1:{port}/v1/models", timeout=20):
                print(f"FAIL: mock :{port} not ready"); return 1
        print("mocks ready")

        # 2) router instance on :18590 (daemonized)
        spawn([LLMROUTER, "run", "-c", CFG], env=env)
        if not wait_http(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=30):
            print("FAIL: router :18590 not ready"); return 1
        print(f"router :{ROUTER_PORT} ready")

        # 3) foreground dashboard
        dash = spawn([LLMROUTER, "dashboard", "-c", CFG, "--port", str(DASH_PORT), "--no-browser"], env=env)
        if not wait_http(f"http://127.0.0.1:{DASH_PORT}/health", timeout=20):
            print("FAIL: dashboard :18910 not ready"); return 1
        print(f"dashboard :{DASH_PORT} ready (pid {dash.pid})")

        # 4) find a router log to stream
        logs = httpx.get(f"http://127.0.0.1:{DASH_PORT}/api/logs", timeout=5).json()
        router_log = next((l for l in logs["logs"] if l["kind"] == "router" and l["exists"]), None)
        if not router_log:
            print("FAIL: no router log to stream"); return 1
        print(f"streaming log id={router_log['id']}")

        # 5) open an SSE stream (the Logs tab) + a 2s poller thread
        client = httpx.Client()
        req = client.build_request("GET", f"http://127.0.0.1:{DASH_PORT}/api/logs/stream?id={router_log['id']}&lines=50")
        stream_resp = client.send(req, stream=True)
        print(f"SSE stream open (HTTP {stream_resp.status_code})")

        stop_poll = threading.Event()
        poll_count = [0]
        def poller():
            pc = httpx.Client()
            while not stop_poll.is_set():
                try:
                    pc.get(f"http://127.0.0.1:{DASH_PORT}/api/instances", timeout=2)
                    poll_count[0] += 1
                except Exception:
                    pass
                stop_poll.wait(0.8)
        th = threading.Thread(target=poller, daemon=True)
        th.start()
        time.sleep(2.5)  # let a couple of polls + SSE bytes flow
        print(f"in-flight: SSE open, polls done so far={poll_count[0]}")

        # 6) SIGINT and time the shutdown
        t0 = time.time()
        dash.send_signal(signal.SIGINT)
        try:
            rc = dash.wait(timeout=MAX_SHUTDOWN_S + 2)
            dur = time.time() - t0
        except subprocess.TimeoutExpired:
            dur = time.time() - t0
            dash.kill()
            print(f"FAIL: dashboard did not exit within {MAX_SHUTDOWN_S+2}s after SIGINT")
            ok = False

        stop_poll.set()
        th.join(timeout=3)
        try:
            stream_resp.close(); client.close()
        except Exception:
            pass

        if ok:
            verdict = "PASS" if dur < MAX_SHUTDOWN_S else "FAIL"
            print(f"{verdict}: SIGINT -> exit in {dur:.2f}s (rc={rc}, limit {MAX_SHUTDOWN_S}s)")
            if dur >= MAX_SHUTDOWN_S:
                ok = False
    finally:
        cleanup()
        time.sleep(1)
        # stop the router instance too (daemonized, may survive)
        subprocess.run([LLMROUTER, "stop", "-c", CFG], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
