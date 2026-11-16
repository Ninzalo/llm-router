#!/usr/bin/env python3
"""
Validate the shared-port external model-detection fix (mocks only).

Bug
---
When two configured models share one backend host:port and a backend is already
serving that port EXTERNALLY (not started by the router), the router used to mark
whichever configured model it was trying to start as "external active" — without
checking which model was actually loaded. So with qwen-q4km loaded externally on
a shared port and default_model=qwen-q4kxl, the dashboard showed qwen-q4kxl as the
active external model instead of qwen-q4km.

Fix
---
On adoption the router now queries the backend's OpenAI-compatible
``GET /v1/models`` and attributes external/running/active state to the model that
is really loaded there (see ``ModelManager._identify_external``).

This test reproduces the scenario with ``mock_llm.py`` on isolated 18xxx ports
(no real models; the live :8080 / :8081 servers are never touched):

  * an external mock serves 'qwen-q4km' on :18095 (started by the test, BEFORE
    the router -> genuinely external);
  * the router config (config.adopt.yaml) lists qwen-q4km AND qwen-q4kxl, BOTH
    on :18095, with default_model = qwen-q4kxl.

Two scenarios are run (one router instance each, sequentially):
  A) default = qwen-q4kxl  (the model that is NOT loaded)   -> the reported bug
  B) default = qwen-q4km   (the model that IS loaded)       -> no regression

Expected after preload in BOTH scenarios:
  * qwen-q4km  : running=True,  external=True,  active=True,  managed=False
  * qwen-q4kxl : running=False, external=False, active=False, managed=False
  * the router spawned NO model process (it adopted the external one)
  * the external mock is still the one serving :18095 (not killed, not replaced)
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import httpx
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))          # tests/ (this script's dir)
# Ensure the config's `python3 mock_llm.py` model commands resolve to THIS python
# (the venv/runner that has the deps), not a bare system python.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
PY = sys.executable
MOCK = os.path.join(ROOT, "mock_llm.py")
CFG = os.path.join(ROOT, "config.adopt.yaml")
CFG_B = os.path.join(ROOT, "config.adopt_b.yaml")   # variant: default = the loaded model

EXT_PORT = 18095          # shared external backend port (mock serves here)
ROUTER_A_PORT = 18795     # scenario A router port
ROUTER_B_PORT = 18796     # scenario B router port

PASSED = 0
FAILED = 0


def check(cond, label, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"PASS: {label}")
    else:
        FAILED += 1
        print(f"FAIL: {label}" + (f"  [{detail}]" if detail else ""))


def get_json(url, timeout=4.0):
    r = httpx.get(url, timeout=timeout)
    return r.status_code, (r.json() if r.status_code == 200 else None)


def wait_http(url, tries=80, delay=0.25):
    for _ in range(tries):
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def free_port(port: int) -> None:
    """Kill anything currently bound to `port` (leftover mock / router)."""
    subprocess.run(
        ["pkill", "-f", f"mock_llm.py.*--port {port}\\b"],
        capture_output=True,
    )
    subprocess.run(
        [PY, "-m", "llm_router", "stop", "--port", str(port)],
        capture_output=True,
        timeout=30,
    )
    time.sleep(0.5)


def start_external_mock(model: str):
    log = open(os.path.join(ROOT, "ext_detect_mock.log"), "ab")
    p = subprocess.Popen(
        [PY, MOCK, "--host", "127.0.0.1", "--port", str(EXT_PORT), "--model", model],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd=ROOT,
    )
    return p, log


def start_router(port: int, cfg_path: str):
    cmd = [PY, "-m", "llm_router", "serve", "--config", cfg_path, "--port", str(port)]
    log = open(os.path.join(ROOT, f"ext_detect_router_{port}.log"), "ab")
    p = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd=ROOT,
    )
    return p, log


def kill_proc(p) -> None:
    if p is None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass
    try:
        p.wait(timeout=8)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass


def model_status(status: dict, name: str) -> dict:
    for m in status.get("models", []):
        if m.get("name") == name:
            return m
    return {}


def run_scenario(label: str, router_port: int, cfg_path: str, default_label: str) -> None:
    print(f"\n=== {label}  (router :{router_port}, default={default_label}) ===")
    router, rlog = start_router(router_port, cfg_path)
    try:
        if not wait_http(f"http://127.0.0.1:{router_port}/health"):
            check(False, f"{label}: router /health came up", "never healthy")
            return

        # Wait for the background preload task to finish adoption (active is set
        # once ensure_active() completes).
        status = {}
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            code, status = get_json(f"http://127.0.0.1:{router_port}/admin/status")
            if code == 200 and status and status.get("active"):
                break
            time.sleep(0.3)
        if not status:
            check(False, f"{label}: /admin/status reachable")
            return

        active = status.get("active")
        km = model_status(status, "qwen-q4km")
        kxl = model_status(status, "qwen-q4kxl")

        # --- the core detection assertions (identical in both scenarios) ---
        check(active == "qwen-q4km",
              f"{label}: active model is the externally-loaded one",
              f"active={active!r}")
        check(km.get("running") is True, f"{label}: qwen-q4km is running", f"{km}")
        check(km.get("external") is True,
              f"{label}: qwen-q4km is external (adopted)", f"{km}")
        check(km.get("active") is True,
              f"{label}: qwen-q4km is marked active", f"{km}")
        check(kxl.get("running") is False,
              f"{label}: qwen-q4kxl is NOT running", f"{kxl}")
        check(kxl.get("external") is False,
              f"{label}: qwen-q4kxl is NOT external", f"{kxl}")
        check(kxl.get("active") is False,
              f"{label}: qwen-q4kxl is NOT active", f"{kxl}")

        # --- the router must NOT have spawned any model process ---
        check(km.get("managed") is False,
              f"{label}: qwen-q4km is not router-managed (adopted, not spawned)",
              f"{km}")
        check(kxl.get("managed") is False,
              f"{label}: qwen-q4kxl was never spawned", f"{kxl}")

        # --- the external mock is still the one serving :18095 ---
        code, models = get_json(f"http://127.0.0.1:{EXT_PORT}/v1/models")
        ids = [m.get("id") for m in (models or {}).get("data", [])] if code == 200 else []
        check(code == 200 and "qwen-q4km" in ids,
              f"{label}: external mock on :{EXT_PORT} still serves qwen-q4km",
              f"ids={ids}")
    finally:
        kill_proc(router)
        rlog.close()
        time.sleep(0.5)


def main() -> None:
    # Pre-clean: nothing may be squatting on the ports we use.
    free_port(EXT_PORT)
    free_port(ROUTER_A_PORT)
    free_port(ROUTER_B_PORT)
    time.sleep(0.5)

    mock, mlog = start_external_mock("qwen-q4km")
    try:
        if not wait_http(f"http://127.0.0.1:{EXT_PORT}/health"):
            print(f"FATAL: external mock on :{EXT_PORT} did not come up")
            sys.exit(2)
        print(f"external mock on :{EXT_PORT} serving qwen-q4km (pid {mock.pid})")

        # Scenario B uses a variant config whose default_model IS the loaded one.
        with open(CFG) as f:
            base = yaml.safe_load(f)
        base["router"]["default_model"] = "qwen-q4km"
        with open(CFG_B, "w") as f:
            yaml.safe_dump(base, f, sort_keys=False)

        run_scenario("A: default = qwen-q4kxl (NOT the loaded model)",
                     ROUTER_A_PORT, CFG, "config:qwen-q4kxl")
        run_scenario("B: default = qwen-q4km (the loaded model)",
                     ROUTER_B_PORT, CFG_B, "config:qwen-q4km")
    finally:
        kill_proc(mock)
        mlog.close()

    print(f"\n=== {PASSED} passed, {FAILED} failed ===")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
