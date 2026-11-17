"""Web dashboard for the llm-router.

A *separate* FastAPI app (its own process) that observes and controls running
router instances. It:

* discovers running router instances (pidfiles in the state dir + a `ps` scan);
* reads each instance's ``/admin/status`` (models + which one is active);
* tails each model's log for timing lines — llama.cpp ``prompt eval time`` /
  ``eval time`` and splash-serve ``Done ... tok/s`` — and reports **current**
  + **average** pp (prompt) / tg (generation) token rates, both per instance
  and per running LLM;
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
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import deque

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
_TOK_RE = re.compile(r"([\d.]+)\s*(?:tokens?/sec|t(?:o(?:k)?)?/s)", re.IGNORECASE)
# A field number that may carry thousands separators, e.g. "32,226" or "1.5".
_NUM_FIELD = r"[\d][\d,]*(?:\.\d+)?"


def parse_rate_line(line: str):
    """Return a list of ``(kind, tokens_per_sec)`` for a timing line, else None.

    ``kind`` is ``"prompt"`` (pp — prompt-processing speed) or ``"gen"``
    (tg — generation speed). Two log formats are recognised:

    * **splash serve** — the per-request ``Done`` line this project's models
      emit::

          13:24:11 Done · request 1 · input 32,226 · cached 0 · output 118 ·
          tools 15·5217171e · TTFT 54.1s · 46.2 tok/s · 6.2 tok/cycle

      yields ``gen = 46.2`` (the logged decode rate) and
      ``prompt = (32,226 - 0) / 54.1`` (new prompt tokens / time-to-first-token).
    * **llama.cpp** — per-iteration lines::

          prompt eval time = X ms / N tokens ( ... T tokens/sec)   -> prompt
          eval time = X ms / N tokens ( ... T tokens/sec)          -> gen

      (both the modern ``tokens/sec`` and legacy ``t/s`` rate spellings).

    Returns ``None`` for a line that carries no rate.
    """
    low = line.lower()

    # splash serve: a per-request "Done" line carrying a tok/s rate plus the
    # (input / cached / TTFT) fields needed to derive the prompt rate.
    if "tok/s" in low and "done" in low:
        return _parse_splash_done(low)

    # llama.cpp-style per-iteration line: a "prompt" / "eval" / "generation"
    # line carrying a rate the regex recognises (tokens/sec, token/sec,
    # tok/s, t/s). No rate present -> None.
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
        return [(kind, float(matches[-1]))]
    except ValueError:
        return None


def _parse_splash_done(low: str):
    """Parse a splash ``Done`` line into ``[(kind, tok/s), ...]`` (or None).

    The line is not cleanly splittable (the "·" separator is reused inside
    fields like ``tools 15·<hash>``), so each value is pulled out by its field
    name. The prompt rate is derived rather than logged: it is the number of
    *new* (uncached) prompt tokens divided by the time-to-first-token.
    """
    rates = []
    m_tg = re.search(r"([\d.]+)\s*tok/s", low)
    if m_tg:
        try:
            rates.append(("gen", float(m_tg.group(1))))
        except ValueError:
            pass
    m_in = re.search(r"\binput\s+(" + _NUM_FIELD + ")", low)
    m_ca = re.search(r"\bcached\s+(" + _NUM_FIELD + ")", low)
    m_tt = re.search(r"\bttft\s+([\d.]+)\s*s\b", low)
    if m_in and m_ca and m_tt:
        try:
            new_tokens = float(m_in.group(1).replace(",", "")) - float(m_ca.group(1).replace(",", ""))
            ttft = float(m_tt.group(1))
            if ttft > 0 and new_tokens > 0:
                rates.append(("prompt", new_tokens / ttft))
        except ValueError:
            pass
    return rates or None


class RateTracker:
    """Tracks current + average pp/tg rates per model, persisting across restarts.

    pp = prompt-processing speed (new prompt tokens / time-to-first-token);
    tg = token-generation speed (decode tokens/sec). Keys are arbitrary strings
    (``"<port>:<model>"``). For each model we tail newly-appended bytes from its
    log and keep the most recent ``_SAMPLE_CAP`` rate pairs (one per request); a
    model's log is *appended* (never truncated) by the router.

    The pp/tg *averages* must never disappear on a model/router/dashboard restart
    (and must survive a model switch — a stopped model keeps its average even
    though it shows no current rate). They reset only for a model that was never
    started via the router before. To guarantee that, each model's rate window is
    persisted to ``<state_dir>/rates.json`` (keyed by the log file's real path);
    the live *current* rates are deliberately not persisted. On first sight of a
    model in a process we adopt its persisted window (so the average resumes
    exactly where it left off). ``sample`` (running models) returns a live current
    rate plus the average over the window; ``avg_snapshot`` (stopped models)
    returns the average with an empty current.
    """

    _SAMPLE_CAP = 20   # most recent rate pairs kept per model (feeds the average)
    _RECENT_CAP = 10   # most recent rates kept per instance (the hover table)
    _PERSIST = ("pos", "samples")   # the only fields written to disk (no current)

    def __init__(self, state_dir: str | None = None):
        self._st: dict[str, dict] = {}
        self._state_dir = state_dir
        self._dirty = False
        self._recent: dict[str, deque] = {}   # port -> most recent (model, pp, tg)

    def _fresh(self, key: str, pos: int) -> dict:
        return {
            "pos": pos, "samples": [],
            "cur_pp": None, "cur_tg": None,
        }

    # --------------------------------------------------------- persistence
    def _rates_path(self) -> str:
        assert self._state_dir is not None
        return os.path.join(self._state_dir, "rates.json")

    def _load_state(self, path_real: str):
        """Load a model's persisted rate window (or None if it was never tracked).

        Returns a fresh state whose window is the persisted one; the live current
        rates start empty (they are re-observed, not persisted). A pre-1.2.7 file
        stored running sums instead of a window; that format is detected and
        re-seeded from the log (pos reset to 0 so the whole appended history is
        read once, rebuilding the last ``_SAMPLE_CAP`` pairs).
        """
        if not self._state_dir:
            return None
        try:
            with open(self._rates_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        st = data.get(path_real) if isinstance(data, dict) else None
        if not isinstance(st, dict):
            return None
        samples = st.get("samples")
        if not isinstance(samples, list):
            # Old format (sums only) -> re-seed the window from the log.
            return {"pos": 0, "samples": [], "cur_pp": None, "cur_tg": None}
        pairs = [s for s in samples if isinstance(s, (list, tuple)) and len(s) == 2]
        pairs = [[pp, tg] for pp, tg in pairs][-self._SAMPLE_CAP:]
        try:
            pos = max(0, int(st.get("pos", 0)))
        except (TypeError, ValueError):
            pos = 0
        return {"pos": pos, "samples": pairs, "cur_pp": None, "cur_tg": None}

    def flush(self):
        """Persist every tracked model's rate window (pos + recent samples).

        Merges into the existing file (never deletes) so a model's averages
        survive even after it stops. Only ``_PERSIST`` fields are written — the
        live current rates are deliberately not stored.
        """
        if not self._state_dir or not self._dirty:
            return
        updated: dict[str, dict] = {}
        for st in self._st.values():
            p = st.get("_path")
            if p:
                updated[p] = {k: st[k] for k in self._PERSIST}
        if not updated:
            self._dirty = False
            return
        try:
            with open(self._rates_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        data.update(updated)
        try:
            os.makedirs(self._state_dir, exist_ok=True)
            tmp = self._rates_path() + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self._rates_path())
            self._dirty = False
        except OSError:
            pass

    # ------------------------------------------------------------- sampling
    def _tail(self, key: str, path: str):
        """Adopt the model's state (or create it) and tail newly-appended bytes.

        On first sight the persisted window is adopted (so the average resumes
        across a restart); a pre-1.2.7 file (sums only) is re-seeded from the log
        because its tail offset starts at 0. Returns the state, or None if the
        log path is missing.
        """
        if not path or not os.path.exists(path):
            return None
        try:
            size = os.path.getsize(path)
        except OSError:
            return None
        st = self._st.get(key)
        if st is None:
            path_real = os.path.realpath(path)
            persisted = self._load_state(path_real)
            if persisted is not None:
                st = persisted
                st["pos"] = min(st["pos"], size)   # never read ahead of the file
            else:
                st = self._fresh(key, max(0, size - _SEED_CAP))
            st["_path"] = path_real
            self._st[key] = st
        if size < st["pos"]:
            # Log rotated/truncated: keep the window; re-aim at the (new) tail so
            # we don't double-count new content.
            st["pos"] = max(0, size - _SEED_CAP)
        if size > st["pos"]:
            try:
                with open(path, "rb") as f:
                    f.seek(st["pos"])
                    data = f.read(size - st["pos"])
                st["pos"] = size
                self._absorb(key, data.decode("utf-8", "replace"))
                self._dirty = True
            except OSError:
                pass
        return st

    def sample(self, key: str, path: str):
        """Live rate snapshot for a running model: average over the window plus
        the current rates (latest request observed in this process)."""
        st = self._tail(key, path)
        return self._snapshot(st) if st else None

    def _absorb(self, key: str, text: str):
        st = self._st[key]
        port, _, model_name = key.partition(":")
        recent = self._recent.setdefault(port, deque(maxlen=self._RECENT_CAP))
        for line in text.splitlines():
            rates = parse_rate_line(line)
            if not rates:
                continue
            pp = tg = None
            for kind, val in rates:
                if kind == "prompt":
                    pp = val
                else:
                    tg = val
            if pp is None and tg is None:
                continue
            st["samples"].append([pp, tg])
            if len(st["samples"]) > self._SAMPLE_CAP:
                del st["samples"][: len(st["samples"]) - self._SAMPLE_CAP]
            if pp is not None:
                st["cur_pp"] = pp
            if tg is not None:
                st["cur_tg"] = tg
            recent.append({"model": model_name, "pp": pp, "tg": tg})

    def _snapshot(self, st: dict) -> dict | None:
        """Rate snapshot for a state: average over the recent window + the live
        current rates (empty until a request is observed in this process)."""
        pp_vals = [s[0] for s in st["samples"] if s[0] is not None]
        tg_vals = [s[1] for s in st["samples"] if s[1] is not None]
        if not pp_vals and not tg_vals:
            return None
        return {
            "pp_current": round(st["cur_pp"], 2) if st["cur_pp"] is not None else None,
            "tg_current": round(st["cur_tg"], 2) if st["cur_tg"] is not None else None,
            "pp_avg": round(sum(pp_vals) / len(pp_vals), 2) if pp_vals else None,
            "tg_avg": round(sum(tg_vals) / len(tg_vals), 2) if tg_vals else None,
            "pp_n": len(pp_vals), "tg_n": len(tg_vals),
        }

    def avg_snapshot(self, key: str, path: str):
        """Average-rate snapshot for a model that may be stopped: the average over
        its persisted window, with an *empty* current. Re-seeds from the log if
        the model was never tracked in this process. Returns None only if the
        model was never started via the router (no window exists anywhere)."""
        st = self._tail(key, path)
        if st is None:
            return None
        snap = self._snapshot(st)
        if snap:
            snap["pp_current"] = None
            snap["tg_current"] = None
        return snap

    def recent(self, port: str, limit: int | None = None):
        """Most recent (model, pp, tg) rate entries for an instance (hover table)."""
        d = self._recent.get(port)
        if not d:
            return []
        items = list(d)
        return items[-limit:] if limit else items

    def aggregate_avg(self, keys):
        """Instance-level average pooled across the recent windows of ``keys``
        (every model of the instance that has produced rates, running or stopped).
        Returns ``{"pp_avg","tg_avg","pp_n","tg_n"}`` or None if no samples yet."""
        pp_vals: list[float] = []
        tg_vals: list[float] = []
        for k in keys:
            st = self._st.get(k)
            if not st:
                continue
            for s in st["samples"]:
                if s[0] is not None:
                    pp_vals.append(s[0])
                if s[1] is not None:
                    tg_vals.append(s[1])
        if not pp_vals and not tg_vals:
            return None
        return {
            "pp_avg": round(sum(pp_vals) / len(pp_vals), 2) if pp_vals else None,
            "tg_avg": round(sum(tg_vals) / len(tg_vals), 2) if tg_vals else None,
            "pp_n": len(pp_vals), "tg_n": len(tg_vals),
        }


def _router_rate(active_rate, tracker, inst_keys):
    """Build the instance (router) rate object.

    The *current* pp/tg come from the active (currently-running) model (empty
    until a request is observed for it); the *average* pp/tg are pooled across
    ALL models of the instance that have produced rates (running or stopped —
    their windows persist). The average is therefore independent of the active
    model and survives a model switch even when the newly-active model has no
    rate yet. Returns None only when there is neither an active-model rate nor
    any pooled window.
    """
    agg = tracker.aggregate_avg(inst_keys) or {}
    if not active_rate and agg.get("pp_avg") is None and agg.get("tg_avg") is None:
        return None
    return {
        "pp_current": (active_rate or {}).get("pp_current"),
        "tg_current": (active_rate or {}).get("tg_current"),
        "pp_avg": agg.get("pp_avg"),
        "tg_avg": agg.get("tg_avg"),
        "pp_n": agg.get("pp_n") or (active_rate or {}).get("pp_n"),
        "tg_n": agg.get("tg_n") or (active_rate or {}).get("tg_n"),
    }


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
    tracker = RateTracker(state_dir=str(_state_dir()))
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
                active_rate = None
                inst_keys = []
                for m in status.get("models", []):
                    key = f"{port}:{m['name']}"
                    if m.get("running"):
                        llm_rate = tracker.sample(key, m.get("log_path"))
                    else:
                        llm_rate = tracker.avg_snapshot(key, m.get("log_path"))
                    if m.get("active"):
                        active_rate = llm_rate
                    inst_keys.append(key)
                    entry["models"].append({
                        "name": m["name"], "kind": m.get("kind"),
                        "running": m.get("running"), "active": m.get("active"),
                        "managed": m.get("managed"), "external": m.get("external"),
                        "pid": m.get("pid"), "port": m.get("port"),
                        "base_url": m.get("base_url"), "log_path": m.get("log_path"),
                        "llm": llm_rate,
                    })
                # Instance (router) rate: current pp/tg from the active (running)
                # model; average pp/tg pooled across ALL models of the instance
                # that have produced rates (their windows persist after stop).
                entry["router"] = _router_rate(active_rate, tracker, inst_keys)
                entry["router_recent"] = tracker.recent(str(port))
            out.append(entry)
        tracker.flush()  # persist rate windows so averages survive restarts
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
