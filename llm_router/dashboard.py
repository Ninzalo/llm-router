"""Web dashboard for the llm-router.

A *separate* FastAPI app (its own process) that observes and controls running
router instances. It:

* discovers running router instances (pidfiles in the state dir + a `ps` scan);
* reads each instance's ``/admin/status`` (models + which one is active);
* tails each model's log file for llama.cpp-style ``prompt eval time`` /
  ``eval time`` lines and reports **current** + **average** pg (prompt) / tg
  (generation) token rates, both per instance and per running LLM;
* offers a structured config editor (with a raw-YAML mode) for the default
  config, a "run a new instance" form, and a live log viewer (router logs +
  each instance's LLM log).

It NEVER starts a router on its own — you click a button to do that.

Run it with ``llm-router dashboard`` (foreground), ``llm-router dashboard -d``
(detached), or stop it with ``llm-router dashboard stop``.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import __version__
from .config import load_config, safe_filename
from .__main__ import (
    _is_router_server, _log_file, _pid_alive, _pid_file, _read_pid,
    _running_instances, _state_dir, resolve_config,
)

logger = logging.getLogger("llm_router.dashboard")

DEFAULT_DASH_PORT = 8799
_SEED_CAP = 5 * 1024 * 1024   # at most this many bytes are read when first seeding a log


# --------------------------------------------------------------- rate parsing
_TOK_RE = re.compile(r"([\d.]+)\s*(?:tokens?/sec|tok/s)", re.IGNORECASE)


def parse_rate_line(line: str):
    """Return ``(kind, tokens_per_sec)`` for a llama.cpp-style timing line.

    ``kind`` is ``"prompt"`` (pg — prompt-processing speed) or ``"gen"``
    (tg — generation speed). A line is a *prompt* line if it mentions "prompt";
    a *generation* line if it mentions "eval"/"generation" but not "prompt".
    Returns ``None`` for lines that carry no rate.
    """
    low = line.lower()
    if "tokens/sec" not in low and "tok/s" not in low:
        return None
    if "prompt" in low:
        kind = "prompt"
    elif "eval" in low or "generation" in low:
        kind = "gen"
    else:
        return None
    matches = _TOK_RE.findall(line)
    if not matches:
        return None
    try:
        return kind, float(matches[-1])
    except ValueError:
        return None


class RateTracker:
    """Tracks current + average pg/tg rates per log file by tailing new bytes.

    Keys are arbitrary strings (``"<port>:<model>"``). On first sight of a key we
    seed from the whole log (capped to the last ``_SEED_CAP`` bytes) so the
    average reflects history; after that we only read newly-appended bytes and
    accumulate running sums. ``forget_stale`` drops keys that are no longer
    running so the table does not grow unbounded.
    """

    def __init__(self):
        self._st: dict[str, dict] = {}

    def _fresh(self, key: str, pos: int) -> dict:
        return {
            "pos": pos, "pg_sum": 0.0, "pg_n": 0,
            "tg_sum": 0.0, "tg_n": 0,
            "cur_pg": None, "cur_tg": None,
        }

    def sample(self, key: str, path: str):
        """Read newly-appended lines from ``path`` and return a rate snapshot."""
        if not path or not os.path.exists(path):
            return None
        try:
            size = os.path.getsize(path)
        except OSError:
            return None
        st = self._st.get(key)
        if st is None:
            st = self._fresh(key, max(0, size - _SEED_CAP))
            self._st[key] = st
        if size < st["pos"]:  # rotated / truncated -> reseed
            st = self._fresh(key, max(0, size - _SEED_CAP))
            self._st[key] = st
        if size > st["pos"]:
            try:
                with open(path, "rb") as f:
                    f.seek(st["pos"])
                    data = f.read(size - st["pos"])
                st["pos"] = size
                self._absorb(key, data.decode("utf-8", "replace"))
            except OSError:
                pass
        if st["pg_n"] == 0 and st["tg_n"] == 0:
            return None
        return {
            "pg_current": round(st["cur_pg"], 2) if st["cur_pg"] is not None else None,
            "tg_current": round(st["cur_tg"], 2) if st["cur_tg"] is not None else None,
            "pg_avg": round(st["pg_sum"] / st["pg_n"], 2) if st["pg_n"] else None,
            "tg_avg": round(st["tg_sum"] / st["tg_n"], 2) if st["tg_n"] else None,
            "pg_n": st["pg_n"], "tg_n": st["tg_n"],
        }

    def _absorb(self, key: str, text: str):
        st = self._st[key]
        for line in text.splitlines():
            r = parse_rate_line(line)
            if not r:
                continue
            kind, val = r
            if kind == "prompt":
                st["pg_sum"] += val; st["pg_n"] += 1; st["cur_pg"] = val
            else:
                st["tg_sum"] += val; st["tg_n"] += 1; st["cur_tg"] = val

    def forget_stale(self, live_keys):
        live = set(live_keys)
        for k in list(self._st):
            if k not in live:
                self._st.pop(k, None)


# ----------------------------------------------------------------- small utils
def _port_in_use(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except Exception:
        return False


def _suggest_port(inst_ports, cfg_port: int) -> int:
    for off in range(1, 200):
        cand = cfg_port + off
        if cand not in inst_ports and not _port_in_use(cand):
            return cand
    for cand in range(8800, 9000):
        if cand not in inst_ports and not _port_in_use(cand):
            return cand
    return cfg_port + 1


def _log_id(path: str) -> str:
    return hashlib.sha1(os.path.realpath(path).encode()).hexdigest()[:16]


def _routers_for_port(port: int) -> list:
    """PIDs of the router instance bound to ``port``, scoped to that port only.

    Sources: the pidfile ``run`` writes for this port, plus any stray
    ``serve``/``run`` server whose command line carries ``--port <port>`` (or a
    pidfile/log name ``<port>``). The port is matched with a word boundary so
    port 1859 never matches 18591 — this never signals other-port instances.
    """
    pids: set = set()
    me = os.getpid()
    pid = _read_pid(_pid_file(port))
    if pid and pid != me and _pid_alive(pid):
        pids.add(pid)
    want = str(port)
    try:
        out = subprocess.check_output(
            ["ps", "-A", "-o", "pid=", "-o", "command="], text=True
        )
        for ln in out.splitlines():
            parts = ln.strip().split(None, 1)
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            p = int(parts[0])
            if p in pids or p == me:
                continue
            cmd = parts[1]
            if not _is_router_server(cmd):
                continue
            toks = cmd.split()
            for i, t in enumerate(toks):
                if (t == "--port" and i + 1 < len(toks) and toks[i + 1] == want) or t == f"--port={want}":
                    pids.add(p)
                    break
    except Exception:
        pass
    return sorted(pids)


# -------------------------------------------------------------------- the app
def build_dashboard_app() -> FastAPI:
    app = FastAPI(title="llm-router dashboard", docs_url=None, openapi_url=None, redoc_url=None)
    tracker = RateTracker()
    app.state.tracker = tracker

    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "dashboard.html")
    try:
        with open(static_path, "r", encoding="utf-8") as f:
            html = f.read()
    except OSError:
        html = "<h1>llm-router dashboard</h1><p>static/dashboard.html missing</p>"

    # -------------------------------------------------------------- helpers
    def _default_cfg():
        try:
            return load_config(resolve_config())
        except Exception:
            return None

    def _gather_instances():
        """Synchronous: discover instances, query each /admin/status, sample rates.

        Returns the full payload for ``GET /api/instances`` (and the allowlists
        the log endpoints reuse).
        """
        cfg = _default_cfg()
        cfg_port = cfg.port if cfg else 8787
        insts = _running_instances()
        inst_ports = {i["port"] for i in insts}
        out = []
        live_keys = set()
        for inst in insts:
            port = inst["port"]
            host = inst["host"]
            base = f"http://{host}:{port}"
            base_url = f"{base}/v1"
            reached = False
            status = None
            try:
                r = httpx.get(base + "/admin/status", timeout=3.0)
                if r.status_code == 200:
                    status = r.json()
                    reached = True
            except Exception:
                pass
            entry = {
                "port": port, "host": host, "base_url": base_url,
                "pid": inst["pid"], "sources": inst["sources"],
                "reached": reached, "active": None, "ready": False,
                "router": None, "models": [], "router_log": os.fspath(_log_file(port)),
            }
            if status:
                active = status.get("active")
                entry["active"] = active
                entry["ready"] = bool(active)
                for m in status.get("models", []):
                    if m.get("running"):
                        key = f"{port}:{m['name']}"
                        live_keys.add(key)
                        llm_rate = tracker.sample(key, m.get("log_path"))
                    else:
                        llm_rate = None
                    if m.get("active"):
                        # The instance's pg/tg == its active model's pg/tg (a
                        # router only ever talks to its one active model).
                        entry["router"] = llm_rate
                    entry["models"].append({
                        "name": m["name"], "kind": m.get("kind"),
                        "running": m.get("running"), "active": m.get("active"),
                        "managed": m.get("managed"), "external": m.get("external"),
                        "pid": m.get("pid"), "port": m.get("port"),
                        "base_url": m.get("base_url"), "log_path": m.get("log_path"),
                        "llm": llm_rate,
                    })
            out.append(entry)
        tracker.forget_stale(live_keys)
        default_running = any(i["port"] == cfg_port and i["reached"] for i in out)
        return {
            "default": {
                "path": str((cfg.source if cfg else None) or resolve_config()),
                "port": cfg_port,
                "running": default_running,
            },
            "instances": out,
            "suggested_port": _suggest_port(inst_ports, cfg_port),
            "version": __version__,
        }

    def _idmap(data):
        m = {}
        for inst in data["instances"]:
            m[_log_id(inst["router_log"])] = os.path.realpath(inst["router_log"])
            for mo in inst["models"]:
                if mo.get("running") and mo.get("log_path"):
                    m[_log_id(mo["log_path"])] = os.path.realpath(mo["log_path"])
        return m

    def _spawn_router(config_path: str, port: int, preload, model_names=None, default_model=None) -> int:
        """Spawn ``llm-router run ...`` in the background; returns the launch pid."""
        config_path = os.fspath(config_path)
        cfg = load_config(config_path)
        run_cfg = config_path
        if model_names is not None:
            run_cfg = _write_instance_config(cfg, port, model_names, default_model)
        argv = [sys.executable, "-m", "llm_router", "run", "--config", run_cfg, "--port", str(port)]
        if preload is True:
            argv.append("--preload")
        elif preload is False:
            argv.append("--no-preload")
        logger.info("dashboard: starting router on :%d (config %s): %s", port, run_cfg, " ".join(argv))
        # `llm-router run` daemonizes itself (double-fork); Popen just triggers it.
        proc = subprocess.Popen(
            argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, env=os.environ.copy(),
        )
        return proc.pid

    def _write_instance_config(cfg, port: int, model_names, default_model) -> str:
        """Write a per-instance config (a subset of the default's models)."""
        inst_dir = os.path.join(_state_dir(), "instances")
        os.makedirs(inst_dir, exist_ok=True)
        want = set(model_names)
        raw_models = (cfg.raw or {}).get("models", []) or []
        sel = [m for m in raw_models if isinstance(m, dict) and m.get("name") in want]
        if not sel:
            sel = raw_models
        names = {m.get("name") for m in sel if isinstance(m, dict)}
        if default_model and default_model in names:
            dm = default_model
        elif cfg.default_model in names:
            dm = cfg.default_model
        else:
            dm = sel[0]["name"] if sel else None
        doc = {
            "router": {
                "host": cfg.host, "port": port, "api_key": cfg.api_key,
                "default_model": dm, "startup_timeout": cfg.startup_timeout,
                "health_interval": cfg.health_interval, "health_path": cfg.health_path,
                "log_dir": cfg.log_dir, "keep_inactive_stopped": cfg.keep_inactive_stopped,
                "preload": cfg.preload, "shutdown_grace": cfg.shutdown_grace,
            },
            "models": sel,
        }
        path = os.path.join(inst_dir, f"instance-{port}.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
        return path

    def _save_structured(router: dict, models: list, dashboard: dict | None = None) -> dict:
        path = os.fspath(resolve_config())
        existing = {}
        if os.path.exists(path):
            try:
                loaded = yaml.safe_load(open(path, encoding="utf-8").read())
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                existing = {}
        existing["router"] = router
        existing["models"] = models
        if dashboard is not None:
            existing["dashboard"] = dashboard
        text = yaml.safe_dump(existing, sort_keys=False, allow_unicode=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        try:
            load_config(tmp)  # validate before committing
        except Exception as e:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return {"ok": False, "error": str(e)}
        if os.path.exists(path):
            try:
                shutil.copy2(path, path + ".bak")
            except OSError:
                pass
        os.replace(tmp, path)
        return {"ok": True, "path": path}

    def _save_raw(text: str) -> dict:
        path = os.fspath(resolve_config())
        try:
            yaml.safe_load(text)
        except Exception as e:
            return {"ok": False, "error": f"YAML parse error: {e}"}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        try:
            load_config(tmp)
        except Exception as e:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return {"ok": False, "error": str(e)}
        if os.path.exists(path):
            try:
                shutil.copy2(path, path + ".bak")
            except OSError:
                pass
        os.replace(tmp, path)
        return {"ok": True, "path": path}

    # ------------------------------------------------------------ endpoints
    @app.get("/", response_class=HTMLResponse)
    async def index():
        return html

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "llm-router dashboard", "version": __version__}

    @app.get("/api/instances")
    async def api_instances():
        return await asyncio.to_thread(_gather_instances)

    @app.get("/api/config")
    async def api_get_config():
        def _load():
            path = os.fspath(resolve_config())
            exists = os.path.exists(path)
            raw_text = ""
            if exists:
                with open(path, encoding="utf-8") as f:
                    raw_text = f.read()
            errors = None
            structured = None
            try:
                structured = yaml.safe_load(raw_text) or {}
                if not isinstance(structured, dict):
                    errors = "config root must be a mapping"
                    structured = {}
            except Exception as e:
                errors = f"YAML parse error: {e}"
                structured = {}
            if errors is None:
                try:
                    load_config(path)
                except Exception as e:
                    errors = str(e)
            return {
                "path": path, "exists": exists, "valid": errors is None, "errors": errors,
                "router": (structured or {}).get("router", {}) or {},
                "models": (structured or {}).get("models", []) or [],
                "dashboard": (structured or {}).get("dashboard", {}) or {},
                "raw": raw_text,
            }
        return await asyncio.to_thread(_load)

    @app.put("/api/config")
    async def api_save_config(request: Request):
        body = await request.json()
        router = body.get("router", {}) or {}
        models = body.get("models", []) or []
        dashboard = body.get("dashboard")  # None => keep the existing section
        return await asyncio.to_thread(_save_structured, router, models, dashboard)

    @app.put("/api/config/raw")
    async def api_save_config_raw(request: Request):
        body = await request.json()
        text = body.get("raw", "")
        return await asyncio.to_thread(_save_raw, text)

    @app.post("/api/run-default")
    async def api_run_default():
        def _run():
            path = os.fspath(resolve_config())
            try:
                cfg = load_config(path)
            except Exception as e:
                return {"ok": False, "error": str(e)}
            port = cfg.port
            if _port_in_use(port):
                return {"ok": True, "already_running": True, "port": port}
            _spawn_router(path, port, preload=None)
            return {"ok": True, "already_running": False, "port": port}
        return await asyncio.to_thread(_run)

    @app.post("/api/run-instance")
    async def api_run_instance(request: Request):
        body = await request.json()
        try:
            port = int(body.get("port") or 0)
        except (TypeError, ValueError):
            port = 0
        model_names = body.get("models") or None
        default_model = body.get("default_model")
        preload = body.get("preload")
        if not port:
            data = await asyncio.to_thread(_gather_instances)
            port = data["suggested_port"]
        if _port_in_use(port):
            return {"ok": False, "error": f"port {port} is already in use"}
        try:
            pid = await asyncio.to_thread(
                _spawn_router, resolve_config(), port, preload, model_names, default_model
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "port": port, "launch_pid": pid}

    @app.post("/api/instances/stop")
    def stop_instance(req: dict):
        """Stop ONE running router instance by port.

        Sends SIGTERM (the router's graceful shutdown stops its active model),
        waits up to the grace for it to exit, then SIGKILLs any straggler.
        Scoped to the port: it only signals the instance whose pidfile or
        command line carries this port, so other instances are untouched.
        (A sync ``def`` so the blocking wait runs in a worker thread, not on
        the event loop.)"""
        port = req.get("port") if isinstance(req, dict) else None
        if not isinstance(port, int) or isinstance(port, bool) or port <= 0:
            return JSONResponse({"error": "port (int) required"}, 400)
        pids = _routers_for_port(port)
        if not pids:
            return JSONResponse({"error": f"no running instance found on port {port}"}, 404)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                return JSONResponse({"error": f"cannot signal pid {pid} (permission)"}, 403)
        grace = 15.0
        remaining = set(pids)
        deadline = time.time() + grace
        while remaining and time.time() < deadline:
            for pid in list(remaining):
                if not _pid_alive(pid):
                    remaining.discard(pid)
            if remaining:
                time.sleep(0.25)
        sigkill = False
        for pid in remaining:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
            sigkill = True
        try:
            _pid_file(port).unlink()
        except Exception:
            pass
        return JSONResponse({"ok": True, "stopped": pids, "sigkill": sigkill})

    @app.get("/api/logs")
    async def api_logs():
        def _list():
            data = _gather_instances()
            logs = []
            for inst in data["instances"]:
                logs.append({
                    "id": _log_id(inst["router_log"]), "kind": "router",
                    "port": inst["port"], "label": f"router :{inst['port']}",
                    "path": inst["router_log"],
                    "exists": os.path.exists(inst["router_log"]),
                })
                for m in inst["models"]:
                    if m.get("running") and m.get("log_path"):
                        logs.append({
                            "id": _log_id(m["log_path"]), "kind": "llm",
                            "port": inst["port"], "model": m["name"],
                            "label": f"llm {m['name']}  (instance :{inst['port']})",
                            "path": m["log_path"],
                            "exists": os.path.exists(m["log_path"]),
                        })
            return {"logs": logs}
        return await asyncio.to_thread(_list)

    @app.get("/api/logs/stream")
    async def api_log_stream(request: Request):
        q = request.query_params
        log_id = q.get("id")
        try:
            lines = int(q.get("lines", "200"))
        except ValueError:
            lines = 200
        follow = q.get("follow", "1") != "0"
        lines = max(1, min(lines, 5000))
        data = await asyncio.to_thread(_gather_instances)
        path = _idmap(data).get(log_id)
        if not path or not os.path.exists(path):
            return JSONResponse({"error": "log not found or no longer running"}, status_code=404)

        async def gen():
            try:
                with open(path, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    back = min(size, 64 * 1024)
                    f.seek(size - back)
                    tail = f.read().decode("utf-8", "replace")
                    for ln in tail.splitlines()[-lines:]:
                        yield ln + "\n"
                    last_pos = size
            except Exception as e:
                yield f"[error reading log] {e}\n"
                return
            if not follow:
                return
            while True:
                if not os.path.exists(path):
                    await asyncio.sleep(0.5)
                    continue
                try:
                    size = os.path.getsize(path)
                    if size < last_pos:
                        last_pos = 0
                    if size > last_pos:
                        with open(path, "rb") as f:
                            f.seek(last_pos)
                            new = f.read(size - last_pos)
                        last_pos = size
                        for ln in new.decode("utf-8", "replace").splitlines():
                            yield ln + "\n"
                except Exception:
                    pass
                await asyncio.sleep(0.4)

        return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "llm-router-dashboard", "version": __version__}

    _STATIC_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "dashboard.html")

    @app.get("/")
    async def index():
        """Serve the single-page dashboard UI."""
        try:
            with open(_STATIC_HTML, "r", encoding="utf-8") as f:
                return HTMLResponse(f.read())
        except Exception as e:  # pragma: no cover
            return JSONResponse({"error": f"dashboard UI missing: {e}"}, status_code=500)

    return app


# Module-level app instance so `uvicorn llm_router.dashboard:dashboard_app` can
# serve the detached (`-d`) dashboard, and `cmd_dashboard` can run it in the
# foreground. Building the app is pure object construction (no I/O at import).
dashboard_app = build_dashboard_app()
