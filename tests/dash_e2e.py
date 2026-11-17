#!/usr/bin/env python
"""End-to-end test for the llm-router web dashboard. Isolated ports (18xxx),
isolated state dir, mock backends. Never touches the real servers."""
import json, os, subprocess, sys, time, tempfile, urllib.request, urllib.error, shutil
import yaml

# Ensure the config's `python3 mock_llm.py` model commands resolve to THIS python
# (the venv/runner that has the deps), not a bare system python. Must run before
# ENV is copied below.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

REPO = os.path.dirname(os.path.abspath(__file__))   # tests/ (this script's dir)
PY = sys.executable
MOCK = os.path.join(REPO, "mock_llm.py")
# Render config.dash.yaml with an ABSOLUTE mock path: the dashboard's run-instance
# flow copies the model list into a per-instance config in the state dir, where a
# relative `mock_llm.py` would not resolve. Real user configs use absolute model
# commands, so mirror that here.
_cfg_src = os.path.join(REPO, "config.dash.yaml")
_cfg_dir = tempfile.mkdtemp(prefix="llmrouter_dash_cfg_")
CFG = os.path.realpath(os.path.join(_cfg_dir, "config.dash.yaml"))
with open(_cfg_src, encoding="utf-8") as _f:
    _cfg_text = _f.read()
with open(CFG, "w", encoding="utf-8") as _f:
    _f.write(_cfg_text.replace("python3 mock_llm.py", "python3 " + MOCK))
STATE = "/tmp/dash_test_state"
LOGDIR = "/tmp/dash_test_logs"
ROUTER1 = 18590
DASH = 18900
DASHCFG = 18930   # the dashboard port stored in config.dash.yaml's `dashboard:` section
INST2 = 18591

ENV = dict(os.environ)
ENV["LLM_ROUTER_STATE_DIR"] = STATE
ENV["LLM_ROUTER_CONFIG"] = CFG
ENV.pop("PYTHONPATH", None)

FAILS = []
def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)

def cli(*args, timeout=60):
    p = subprocess.run([PY, "-m", "llm_router", *args], env=ENV, cwd=REPO,
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()

def http(url, method="GET", body=None, timeout=15, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json"}
    if headers: h.update(headers)
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace"), dict(r.headers)

def http_err(url, method, body, timeout=15):
    """Like http() but returns (status, text) even on 4xx/5xx (no raise)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")

def up(url, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False

def wait_up(url, tries=40, delay=0.5):
    for _ in range(tries):
        if up(url):
            return True
        time.sleep(delay)
    return False

# ------------------------------------------------------------------- setup
for d in (STATE, LOGDIR):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
if os.path.exists(CFG + ".bak"):
    os.unlink(CFG + ".bak")
if os.path.exists(CFG + ".tmp"):
    os.unlink(CFG + ".tmp")
orig_raw = open(CFG, encoding="utf-8").read()

def read_cfg_json():
    """Read CFG as YAML, tolerating a transient empty/invalid state (returns None)."""
    for _ in range(6):
        try:
            raw = open(CFG, encoding="utf-8").read()
            if raw.strip():
                loaded = yaml.safe_load(raw)
                if isinstance(loaded, dict):
                    return loaded
        except Exception:
            pass
        time.sleep(0.2)
    return None

try:
    # ---- 1. start router instance :18590 (preload=true) ----
    print("== start router :18590 (default, preload) ==")
    rc, out = cli("run", "--config", CFG, "--host", "127.0.0.1", "--port", str(ROUTER1))
    check("router run rc==0", rc == 0, out)
    check("router :18590 /health up", wait_up(f"http://127.0.0.1:{ROUTER1}/health", tries=40))

    # ---- 2. make a few chat requests -> generate pp/tg timing lines ----
    print("== send chat requests through :18590 (populate rates) ==")
    body = {"model": "qwen-local", "messages": [{"role": "user", "content": "say hi"}]}
    for _ in range(4):
        http(f"http://127.0.0.1:{ROUTER1}/v1/chat/completions", "POST", body, timeout=30)
        time.sleep(0.15)
    time.sleep(1.0)
    mock_log = os.path.join(LOGDIR, "qwen-local.log")
    _ml = open(mock_log, encoding="utf-8", errors="replace").read() if os.path.exists(mock_log) else ""
    ml_ok = "Done" in _ml and "tok/s" in _ml
    check("mock log has splash 'Done ... tok/s' timing lines", ml_ok, mock_log)

    # ---- 3. start the dashboard (detached) ----
    print(f"== start dashboard (detached) :{DASH} ==")
    rc, out = cli("dashboard", "-d", "--host", "127.0.0.1", "--port", str(DASH), "--config", CFG)
    check("dashboard -d rc==0", rc == 0, out)
    check("dashboard /health up", wait_up(f"http://127.0.0.1:{DASH}/health", tries=30))

    # ---- 4. GET / (HTML) ----
    print("== GET / (UI) ==")
    st, html, hdrs = http(f"http://127.0.0.1:{DASH}/")
    check("/ returns 200", st == 200)
    check("/ is text/html", "html" in (hdrs.get("content-type") or hdrs.get("Content-Type") or ""), str(hdrs))
    for marker in ["llm-router", "Overview", "Run Instance", "Config", "Logs",
                   "data-tip", "data-theme", "Run default instance", "pp", "tg",
                   "Raw YAML", "Watch", "structured", "rate-hover", "router_recent"]:
        check(f"UI contains {marker!r}", marker in html)

    # ---- 5. GET /health ----
    print("== GET /health ==")
    st, body, _ = http(f"http://127.0.0.1:{DASH}/health")
    j = json.loads(body)
    check("health status ok", j.get("status") == "ok", body)
    check("health service name", j.get("service") == "llm-router dashboard", body)

    # ---- 6. GET /api/instances ----
    print("== GET /api/instances ==")
    st, body, _ = http(f"http://127.0.0.1:{DASH}/api/instances")
    inst = json.loads(body)
    check("default port==18590", inst["default"]["port"] == ROUTER1, body[:300])
    check("default running==True", inst["default"]["running"] is True, str(inst["default"]))
    by_port = {i["port"]: i for i in inst["instances"]}
    check("instance :18590 present", ROUTER1 in by_port, str(list(by_port)))
    i1 = by_port.get(ROUTER1, {})
    check("instance reached", i1.get("reached") is True, str(i1)[:200])
    check("instance active==qwen-local", i1.get("active") == "qwen-local", str(i1.get("active")))
    check("instance has router_log", bool(i1.get("router_log")), str(i1.get("router_log")))
    models1 = {m["name"]: m for m in i1.get("models", [])}
    check("models include qwen-local", "qwen-local" in models1, str(list(models1)))
    check("models include embed-local (config)", "embed-local" in models1, str(list(models1)))
    q1 = models1.get("qwen-local", {})
    check("qwen-local active label", q1.get("active") is True, str(q1)[:200])
    check("qwen-local running", q1.get("running") is True, str(q1.get("running")))
    llm = q1.get("llm") or {}
    check("qwen-local llm has pp_current", llm.get("pp_current") is not None, str(llm))
    check("qwen-local llm has tg_current", llm.get("tg_current") is not None, str(llm))
    check("qwen-local llm has pp_avg", llm.get("pp_avg") is not None, str(llm))
    check("qwen-local llm has tg_avg", llm.get("tg_avg") is not None, str(llm))
    check("qwen-local llm exposes the recent-rate window for the hover table", isinstance(i1.get("router_recent"), list), str(i1.get("router_recent")))
    r1 = i1.get("router") or {}
    check("instance router has pp_current (active model)", r1.get("pp_current") is not None, str(r1))
    check("instance router has tg_avg", r1.get("tg_avg") is not None, str(r1))
    # current tg/pp come from the active (running) model...
    check("router pp_current == active model pp_current", r1.get("pp_current") == llm.get("pp_current"), f"router={r1} active={llm}")
    check("router tg_current == active model tg_current", r1.get("tg_current") == llm.get("tg_current"), f"router={r1} active={llm}")
    # ...and with only ONE running model on :18590, the pooled instance average
    # equals that model's own average (the multi-model pooling is unit-tested).
    check("router pp_avg == active model pp_avg (1 running model)", r1.get("pp_avg") == llm.get("pp_avg"), f"router={r1} active={llm}")
    check("router tg_avg == active model tg_avg (1 running model)", r1.get("tg_avg") == llm.get("tg_avg"), f"router={r1} active={llm}")
    # persistence: the live dashboard must have written the running-average state
    # to the (fresh) isolated state dir so the averages survive a dashboard/router
    # restart. The unit tests prove the resume; this proves the running dashboard
    # actually persists. (STATE is wiped at test start, so this is this run's file.)
    check("rate state persisted to the state dir (rates.json)", os.path.exists(os.path.join(STATE, "rates.json")), STATE)
    emb = models1.get("embed-local", {})
    check("embed-local NOT active", emb.get("active") is not True, str(emb)[:150])
    check("suggested_port present", isinstance(inst.get("suggested_port"), int), str(inst.get("suggested_port")))

    # ---- 6b. model switch: switch away from qwen-local; its AVERAGE must survive ----
    # The core v1.2.7 behaviour, exercised against the LIVE dashboard+router: a
    # switched-AWAY (stopped) model keeps its average pp/tg, its current goes
    # empty, and the router (instance) average stays pooled across all models.
    print("== switch qwen-local -> qwen-second; the switched-away average must survive ==")
    st, body = http_err(f"http://127.0.0.1:{ROUTER1}/admin/switch", "POST", {"model": "qwen-second"}, timeout=60)
    check("admin/switch -> 200", st == 200, f"st={st} {body[:120]}")
    time.sleep(1.5)
    st, body, _ = http(f"http://127.0.0.1:{DASH}/api/instances")
    i1s = {i["port"]: i for i in json.loads(body)["instances"]}.get(ROUTER1, {})
    check("after switch active==qwen-second", i1s.get("active") == "qwen-second", str(i1s.get("active")))
    models1s = {m["name"]: m for m in i1s.get("models", [])}
    llms = (models1s.get("qwen-local", {}) or {}).get("llm") or {}
    check("switched-AWAY qwen-local keeps its pp_avg (NOT null)", llms.get("pp_avg") is not None, str(llms))
    check("switched-AWAY qwen-local keeps its tg_avg (NOT null)", llms.get("tg_avg") is not None, str(llms))
    check("switched-AWAY qwen-local current is empty (it is stopped)",
          llms.get("pp_current") is None and llms.get("tg_current") is None, str(llms))
    rs = i1s.get("router") or {}
    check("router pp_avg/tg_avg still present after the switch (pooled across models)",
          rs.get("pp_avg") is not None and rs.get("tg_avg") is not None, str(rs))
    # switch back so the later run-instance step (which also uses qwen-second /
    # mock-c :18503) does not collide with a still-running copy on :18590.
    http_err(f"http://127.0.0.1:{ROUTER1}/admin/switch", "POST", {"model": "qwen-local"}, timeout=60)
    time.sleep(1.5)

    # ---- 7. GET /api/config ----
    print("== GET /api/config ==")
    st, body, _ = http(f"http://127.0.0.1:{DASH}/api/config")
    cfgj = json.loads(body)
    check("config path", cfgj.get("path") == CFG, str(cfgj.get("path")))
    check("config valid", cfgj.get("valid") is True, str(cfgj.get("errors")))
    check("config router.port", cfgj["router"].get("port") == ROUTER1, str(cfgj["router"].get("port")))
    check("config has 3 models", len(cfgj.get("models", [])) == 3, str(len(cfgj.get("models", []))))
    check("config raw non-empty", len(cfgj.get("raw", "")) > 50, str(len(cfgj.get("raw", ""))))
    dashj = cfgj.get("dashboard", {})
    check("config dashboard.port", dashj.get("port") == DASHCFG, str(dashj))
    check("config dashboard.host", dashj.get("host") == "127.0.0.1", str(dashj))
    check("config dashboard.browser false", dashj.get("browser") is False, str(dashj))

    # ---- 8. PUT /api/config (structured) — change a field + add a model ----
    print("== PUT /api/config (structured) ==")
    mod = json.loads(body)
    mod["router"]["port"] = ROUTER1  # keep same port
    mod["router"]["startup_timeout"] = 42
    mod["models"].append({"name": "tmp-dash-model", "upstream_model": "mock-z",
                          "context_length": 8192,
                          "command": f"{PY} {os.path.join(REPO,'mock_llm.py')} --host 127.0.0.1 --port 18504 --model mock-z"})
    st, body, _ = http(f"http://127.0.0.1:{DASH}/api/config", "PUT", {"router": mod["router"], "models": mod["models"]})
    savej = json.loads(body)
    check("structured save ok", savej.get("ok") is True, body)
    check("bak backup created", os.path.exists(CFG + ".bak"), CFG + ".bak")
    newcfg = read_cfg_json()
    check("config file re-read (non-empty)", newcfg is not None, "empty/invalid after save")
    if newcfg:
        check("startup_timeout persisted", newcfg.get("router", {}).get("startup_timeout") == 42, str(newcfg.get("router", {}).get("startup_timeout")))
        check("new model persisted", any(m.get("name") == "tmp-dash-model" for m in newcfg.get("models", [])), str(len(newcfg.get("models", []))))

    # ---- 8b. structured save REJECTS an invalid config ----
    print("== PUT /api/config (invalid) rejected ==")
    base = read_cfg_json() or yaml.safe_load(orig_raw)
    if not isinstance(base, dict):
        base = {"router": {"host": "127.0.0.1", "port": ROUTER1, "api_key": "",
                           "default_model": "qwen-local", "log_dir": LOGDIR, "preload": True},
                "models": []}
    bad = json.loads(json.dumps(base))
    bad["router"]["default_model"] = "no-such-model"
    st, body, _ = http(f"http://127.0.0.1:{DASH}/api/config", "PUT", {"router": bad["router"], "models": bad["models"]}, timeout=20)
    badj = json.loads(body)
    check("invalid save ok==False", badj.get("ok") is False, body)
    after = read_cfg_json()
    check("invalid save did not clobber", after is not None and after.get("router", {}).get("default_model") != "no-such-model", str(after))

    # ---- 9. PUT /api/config/raw ----
    print("== PUT /api/config/raw ==")
    rawtext = orig_raw.replace("startup_timeout: 30", "startup_timeout: 77")
    st, body, _ = http(f"http://127.0.0.1:{DASH}/api/config/raw", "PUT", {"raw": rawtext})
    rawj = json.loads(body)
    check("raw save ok", rawj.get("ok") is True, body)
    time.sleep(0.3)
    check("raw change persisted", "startup_timeout: 77" in open(CFG, encoding="utf-8", errors="replace").read())

    # restore original raw so later steps use the known-good config
    st, body, _ = http(f"http://127.0.0.1:{DASH}/api/config/raw", "PUT", {"raw": orig_raw})
    check("restore original raw ok", json.loads(body).get("ok") is True, body)

    # ---- 9b. PUT /api/config (dashboard section) round-trip ----
    print("== PUT /api/config (dashboard section) ==")
    base = read_cfg_json() or {}
    router_now = base.get("router", {}) or {}
    models_now = base.get("models", []) or []
    st, body, _ = http(f"http://127.0.0.1:{DASH}/api/config", "PUT",
                       {"router": router_now, "models": models_now,
                        "dashboard": {"host": "127.0.0.1", "port": DASHCFG + 1, "browser": True}}, timeout=20)
    check("dashboard save ok", json.loads(body).get("ok") is True, body)
    time.sleep(0.3)
    after = read_cfg_json()
    if after:
        ad = after.get("dashboard", {})
        check("dashboard.port persisted", ad.get("port") == DASHCFG + 1, str(ad))
        check("dashboard.browser persisted", ad.get("browser") is True, str(ad))
    # restore the dashboard section to its original value
    st, body, _ = http(f"http://127.0.0.1:{DASH}/api/config", "PUT",
                       {"router": router_now, "models": models_now,
                        "dashboard": {"host": "127.0.0.1", "port": DASHCFG, "browser": False}}, timeout=20)
    check("dashboard restore ok", json.loads(body).get("ok") is True, body)
    # a PUT that omits 'dashboard' must NOT wipe the existing section
    st, body, _ = http(f"http://127.0.0.1:{DASH}/api/config", "PUT",
                       {"router": router_now, "models": models_now}, timeout=20)
    check("omitted-dashboard save ok", json.loads(body).get("ok") is True, body)
    after2 = read_cfg_json()
    check("dashboard preserved when omitted", bool(after2 and after2.get("dashboard")),
          str(after2 and after2.get("dashboard")))

    # ---- 10. GET /api/logs ----
    print("== GET /api/logs ==")
    st, body, _ = http(f"http://127.0.0.1:{DASH}/api/logs")
    logsj = json.loads(body)
    loglabels = [l["label"] for l in logsj.get("logs", [])]
    check("logs list non-empty", len(logsj.get("logs", [])) > 0, str(loglabels))
    check("router :18590 log listed", any("router :18590" in l for l in loglabels), str(loglabels))
    check("llm qwen-local log listed", any("qwen-local" in l for l in loglabels), str(loglabels))
    rlog = next((l for l in logsj["logs"] if l["kind"] == "router" and l["port"] == ROUTER1), None)
    check("router log exists flag", rlog is not None and rlog.get("exists") is True, str(rlog))

    # ---- 11. GET /api/logs/stream (follow=0 -> tail then end) ----
    print("== GET /api/logs/stream ==")
    rlog_id = rlog["id"] if rlog else None
    st, sbody, _ = http(f"http://127.0.0.1:{DASH}/api/logs/stream?id={rlog_id}&lines=40&follow=0", timeout=20)
    check("stream returns text", len(sbody) > 0, f"len={len(sbody)}")
    check("stream has router content", "router" in sbody.lower() or "18590" in sbody, sbody[:120])

    # ---- 12. POST /api/run-instance (new instance :18591 with qwen-second) ----
    print("== POST /api/run-instance :18591 ==")
    st, body, _ = http(f"http://127.0.0.1:{DASH}/api/run-instance", "POST",
                       {"port": INST2, "models": ["qwen-second"], "default_model": "qwen-second", "preload": True}, timeout=30)
    rj = json.loads(body)
    check("run-instance ok", rj.get("ok") is True, body)
    check("run-instance launch_pid present", isinstance(rj.get("launch_pid"), int), str(rj))
    check("instance :18591 /health up", wait_up(f"http://127.0.0.1:{INST2}/health", tries=40))
    time.sleep(1.5)
    st, body, _ = http(f"http://127.0.0.1:{DASH}/api/instances")
    inst2 = json.loads(body)
    by_port2 = {i["port"]: i for i in inst2["instances"]}
    check("instance :18591 present", INST2 in by_port2, str(list(by_port2)))
    i2 = by_port2.get(INST2, {})
    check("instance :18591 active==qwen-second", i2.get("active") == "qwen-second", str(i2.get("active")))
    m2 = {m["name"]: m for m in i2.get("models", [])}
    check("instance :18591 has qwen-second only", "qwen-second" in m2 and "qwen-local" not in m2, str(list(m2)))

    # ---- 12b. POST /api/instances/stop (scoped: stop :18591 only) ----
    print("== POST /api/instances/stop :18591 (scoped) ==")
    st, body = http_err(f"http://127.0.0.1:{DASH}/api/instances/stop", "POST", {"port": INST2}, timeout=40)
    stopj = json.loads(body)
    check("stop :18591 -> 200 ok", st == 200 and stopj.get("ok") is True, f"st={st} {body}")
    check("stop :18591 reports pids", isinstance(stopj.get("stopped"), list) and len(stopj.get("stopped")) > 0, str(stopj))
    time.sleep(1.5)
    check("instance :18591 DOWN after stop", not up(f"http://127.0.0.1:{INST2}/health", timeout=2))
    check("instance :18590 STILL UP (port-scoped)", up(f"http://127.0.0.1:{ROUTER1}/health", timeout=2))
    # error paths
    st, body = http_err(f"http://127.0.0.1:{DASH}/api/instances/stop", "POST", {"port": "abc"}, timeout=10)
    check("stop non-int port -> 400", st == 400, f"st={st} {body}")
    st, body = http_err(f"http://127.0.0.1:{DASH}/api/instances/stop", "POST", {}, timeout=10)
    check("stop missing port -> 400", st == 400, f"st={st} {body}")
    st, body = http_err(f"http://127.0.0.1:{DASH}/api/instances/stop", "POST", {"port": 18777}, timeout=10)
    check("stop no instance on port -> 404", st == 404, f"st={st} {body}")

    # ---- 13. POST /api/run-default (should report already running) ----
    print("== POST /api/run-default (already running) ==")
    st, body, _ = http(f"http://127.0.0.1:{DASH}/api/run-default", "POST", {}, timeout=30)
    rdj = json.loads(body)
    check("run-default ok", rdj.get("ok") is True, body)
    check("run-default already_running", rdj.get("already_running") is True, body)

    # ---- 13b. POST /api/rates/clear (wipe all stored avg pp/tg rates) ----
    print("== POST /api/rates/clear (wipe stored rates) ==")
    st, body = http_err(f"http://127.0.0.1:{DASH}/api/rates/clear", "POST", {}, timeout=15)
    clearj = json.loads(body)
    check("clear -> 200 ok", st == 200 and clearj.get("ok") is True, f"st={st} {body}")
    check("clear reports an int cleared >= 1 (models had rates)",
          isinstance(clearj.get("cleared"), int) and clearj.get("cleared") >= 1, str(clearj))
    # After a clear, a re-fetch with NO new request shows EMPTY averages: each
    # model's tail was re-aimed at its log's current end, so nothing is re-seeded
    # from history, and the pooled instance (router) average is wiped too.
    time.sleep(1.0)
    st, body, _ = http(f"http://127.0.0.1:{DASH}/api/instances", timeout=15)
    i1c = {i["port"]: i for i in json.loads(body)["instances"]}.get(ROUTER1, {})
    m1c = {m["name"]: m for m in i1c.get("models", [])}.get("qwen-local", {}) or {}
    empty = lambda x: x is None or (isinstance(x, dict) and x.get("pp_avg") is None and x.get("tg_avg") is None)
    check("after clear, qwen-local average is empty (no re-seed from history)",
          empty(m1c.get("llm")), str(m1c.get("llm")))
    check("after clear, the pooled router average is empty too",
          empty(i1c.get("router")), str(i1c.get("router")))

    # ---- 14. llm-router dashboard stop ----
    print("== llm-router dashboard stop ==")
    rc, out = cli("dashboard", "stop", timeout=30)
    check("dashboard stop rc==0", rc == 0, out)
    check("dashboard stop message", "stopped" in out.lower(), out)
    time.sleep(1.0)
    check("dashboard :18900 down", not up(f"http://127.0.0.1:{DASH}/health", timeout=2))

    # ---- 14b. `llm-router dashboard` with NO --port binds to config dashboard.port ----
    print(f"== dashboard (no --port) binds to config port :{DASHCFG} ==")
    rc, out = cli("dashboard", "-d", "--config", CFG, timeout=40)
    check("dashboard (no --port) rc==0", rc == 0, out)
    check(f"dashboard up on config port :{DASHCFG}", wait_up(f"http://127.0.0.1:{DASHCFG}/health", tries=30))
    rc, out = cli("dashboard", "stop", timeout=30)
    check("dashboard (config port) stop ok", rc == 0, out)
    time.sleep(1.0)
    check(f"dashboard :{DASHCFG} down", not up(f"http://127.0.0.1:{DASHCFG}/health", timeout=2))

finally:
    # ----------------------------------------------------------- cleanup
    print("== cleanup ==")
    try:
        cli("stop", timeout=30)
    except Exception as e:
        print("  (stop failed:", e, ")")
    time.sleep(1.0)
    for port in (ROUTER1, INST2, DASH, DASHCFG):
        still = up(f"http://127.0.0.1:{port}/health", timeout=1)
        print(f"  port {port} still up: {still}")
    # restore the config to its original form
    try:
        open(CFG, "w", encoding="utf-8").write(orig_raw)
        for ext in (".bak",):
            if os.path.exists(CFG + ext):
                os.unlink(CFG + ext)
    except Exception as e:
        print("  (config restore failed:", e, ")")

print()
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("RESULT: ALL DASHBOARD E2E CHECKS PASSED")
sys.exit(0)
