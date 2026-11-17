"""CLI entry point for llm-router.

Commands (usable from any directory; the router's own venv is used automatically):

  llm-router run     [--config P] [--host H] [--port N]
      Start the router as a *background service*. Returns immediately; the log
      goes to a file (follow it with `llm-router watch`). Preloads the default
      model in the background.
  llm-router serve   [--config P] [--host H] [--port N]
      Run the router in the foreground (used by `run` to daemonize).
  llm-router watch   [--port N] [--lines N] [--llm]
      Follow the router log (default) or the active model's own log (--llm,
      which auto-switches to the new model's log when the active model changes).
      Ctrl+C stops watching only — the router keeps running.
  llm-router stop    [--port N]
      Stop all running router instances (SIGTERM -> their shutdown stops the
      active model), and any `watch` / `watch --llm` tailers running in other
      terminals. `--port N` stops just the instance on that port.
  llm-router stop-model <model>
      Stop one model via the running router (the router itself keeps running).
  llm-router config  [--path P] [--print]
      Open the config file in $EDITOR (or --print to just show its path).
  llm-router check   [--config P]
      Validate the config and print the resolved specs (no server).
  llm-router models  [--config P]
      List preconfigured models (name, kind, target, context_length, reasoning).
  llm-router status  [--config P] [--host H] [--port N]
      Show live state of a *running* router (admin API).
  llm-router switch  <model> [--config P] ...
      Make a preconfigured model the active one (running router).

Every command also accepts --help / -h, which prints its full markdown-styled
description + arguments. `llm-router --help` (or no arguments) shows everything.

Config resolution (so it works from any cwd):
  --config/--path  >  $LLM_ROUTER_CONFIG  >  ~/.config/llm-router/config.yaml
  (if present)  >  <repo>/config.yaml (if present)  >  ~/.config/llm-router/config.yaml

State (pids + the router's own log) lives in a global dir so `stop`/`watch` work
regardless of cwd/config: ~/.config/llm-router/  (override: $LLM_ROUTER_STATE_DIR).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn

from . import __version__
from .app import build_app
from .config import MIN_CONTEXT_LEN, load_config, safe_filename


# ------------------------------------------------------------------ config path
def _installed_config() -> Path:
    # <repo>/config.yaml  (parent of the llm_router package dir)
    return Path(__file__).resolve().parent.parent / "config.yaml"


def _user_config() -> Path:
    return Path.home() / ".config" / "llm-router" / "config.yaml"


def resolve_config(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("LLM_ROUTER_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    user = _user_config()
    if user.exists():
        return user
    installed = _installed_config()
    if installed.exists():
        return installed
    return user


def _load(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"no config file at {path}\n"
            f"       run `llm-router config` to create one, or pass --config PATH"
        )
    return load_config(str(path))


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _print_resolved(cfg, cfg_path: Path) -> None:
    print(f"router: {cfg.public_base_url}   (api_key={'set' if cfg.api_key else 'off'})")
    print(f"config: {cfg_path}")
    print(f"default model: {cfg.default_model}   preload: {cfg.preload}")
    print(f"keep_inactive_stopped: {cfg.keep_inactive_stopped}   log_dir: {cfg.log_dir}")
    print("models:")
    for m in cfg.models:
        ctx = f"  context_length={m.context_length}" if m.context_length else ""
        re_ = f"  reasoning_effort={m.reasoning_effort}" if m.reasoning_effort else ""
        in_picker = "" if m.kind == "chat" else "  (not in the /model picker)"
        print(f"  - {m.name!r} [{m.kind}] -> {m.base_url}  upstream_model={m.upstream_model!r}"
              f"  default={m.name == cfg.default_model}{ctx}{re_}{in_picker}")
        if m.kind == "chat" and m.context_length is not None and m.context_length < MIN_CONTEXT_LEN:
            print(f"      WARNING: context_length below the client's {MIN_CONTEXT_LEN} minimum — "
                  f"a client will refuse to select it (mark kind: embedding, or raise the window)")
        print(f"      cmd: {m.command}")


# ----------------------------------------------------------------------- editor
def _open_editor(path: Path) -> int:
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    cmd = shlex.split(editor) + [str(path)]
    _err(f"opening {path} in: {editor}")
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        _err(f"editor '{editor}' not found; set $EDITOR or $VISUAL")
        return 127


_TEMPLATE = """\\\
# llm-router configuration
# Docs: README.md  (this file is edited with `llm-router config`)
router:
  host: 127.0.0.1
  port: 8787
  api_key: ""            # if set, require "Authorization: Bearer ***"; mirror in your client
  default_model: my-model
  startup_timeout: 600
  log_dir: ./logs
  keep_inactive_stopped: true   # only the active model occupies RAM
  preload: true                 # preload the default model on `start` (override: --preload / --no-preload)

models:
  - name: my-model
    # command: splash serve --model <repo:tag> --host 0.0.0.0 --port 8080 ...
    #   (port/host are parsed from --port/--host unless set explicitly)
    command: "splash serve --model <repo:tag> --host 127.0.0.1 --port 8080"
    upstream_model: "<repo:tag>"   # id sent to the backend (defaults to name)
    context_length: 131072         # advertised via /v1/models (>= 64000)
    reasoning_effort: high         # model's default reasoning effort (optional)
    # kind: chat                   # chat (default) or embedding (not a chat model)
"""


def cmd_config(args) -> int:
    path = resolve_config(args.path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_TEMPLATE)
        _err(f"created starter config at {path}")
    if args.print:
        print(str(path))
        return 0
    return _open_editor(path)


# -------------------------------------------------------------------- state dir
def _state_dir() -> Path:
    env = os.environ.get("LLM_ROUTER_STATE_DIR")
    d = Path(env).expanduser() if env else (Path.home() / ".config" / "llm-router")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log_file(port: int) -> Path:
    return _state_dir() / f"router-{port}.log"


def _pid_file(port: int) -> Path:
    return _state_dir() / f"router-{port}.pid"


def _read_pid(path: Path):
    try:
        v = path.read_text().strip()
        return int(v) if v.isdigit() else None
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, not ours
    except OSError:
        return False


def _health_up(host: str, port: int, timeout: float = 2.0) -> bool:
    """True if a router is answering /health on host:port within `timeout` seconds.
    If the router process is up, /health answers 200; if nothing is listening the
    connection is refused (and this returns False)."""
    deadline = time.time() + timeout
    url = f"http://{host}:{port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def _port_has_router(cfg, timeout: float = 2.0) -> bool:
    """True if a router is already answering /health on this host:port."""
    return _health_up(cfg.host, cfg.port, timeout)


def _wait_for_health(cfg, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://{cfg.host}:{cfg.port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


# --------------------------------------------------------------------------- logging
# Paths whose per-request access-log lines are dropped from the router log:
#   * unknown "probe" paths that some clients (e.g. model-capability discovery)
#     send to the router even though it implements none of them -- they 404,
#     and their lines are pure noise;
#   * /admin/active -- the lightweight endpoint `llm-router watch --llm` polls
#     to track the active model. It's internal monitoring traffic (not a user
#     request), so leaving it logged would flood the log with status pings.
# Only the paths below are affected; every other request is still logged.
_SUPPRESSED_ACCESS_PATHS = frozenset({
    "/api/v1/models",
    "/api/tags",
    "/v1/props",
    "/props",
    "/version",
    "/admin/active",
})
_ACCESS_REQ_LINE = re.compile(r'"[A-Za-z]+ (\S+) HTTP')


class _AccessLogFilter(logging.Filter):
    """Drop uvicorn access-log records for the suppressed paths above."""

    def filter(self, record):
        match = _ACCESS_REQ_LINE.search(record.getMessage())
        if match:
            path = match.group(1).split("?", 1)[0]
            return path not in _SUPPRESSED_ACCESS_PATHS
        return True


def _install_access_log_filter() -> None:
    """Suppress access-log lines for the paths above on the uvicorn.access logger.

    Added *before* ``uvicorn.run`` so the filter is already on the logger when
    uvicorn's ``dictConfig`` configures it. ``dictConfig`` only rewrites
    ``logger.filters`` if the config names a ``filters`` key (uvicorn's does not),
    so this filter survives uvicorn's logging setup.
    """
    uv_access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _AccessLogFilter) for f in uv_access.filters):
        uv_access.addFilter(_AccessLogFilter())


# --------------------------------------------------------------------------- start
def _resolve_preload(cfg, args) -> bool:
    """Effective preload-on-start decision.

    The --preload / --no-preload flags override the config's `router.preload`
    (default True). With neither flag present the config value wins, so
    `start`'s behaviour matches its config default unless overridden.
    """
    if getattr(args, "preload", False):
        return True
    if getattr(args, "no_preload", False):
        return False
    return cfg.preload


def cmd_serve(args) -> int:
    """Foreground server. `run` launches this detached in the background."""
    cfg_path = resolve_config(args.config)
    try:
        cfg = _load(cfg_path)
    except Exception as e:
        _err(f"config error: {e}")
        return 2
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    cfg.preload = _resolve_preload(cfg, args)
    _err(
        f"llm-router: serving {cfg.public_base_url}  "
        f"(default model: {cfg.default_model}, preload: {'on' if cfg.preload else 'off'}, config: {cfg_path})"
    )
    app = build_app(cfg)
    _install_access_log_filter()
    try:
        uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")
    except KeyboardInterrupt:
        _err("interrupted; shutting down")
    return 0


def cmd_run(args) -> int:
    """Start the router as a background service; log to a file; return right away."""
    cfg_path = resolve_config(args.config)
    try:
        cfg = _load(cfg_path)
    except Exception as e:
        _err(f"config error: {e}")
        return 2
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port

    if _port_has_router(cfg):
        pid = _read_pid(_pid_file(cfg.port))
        print(f"llm-router is already running on {cfg.public_base_url}"
              + (f" (pid {pid})" if pid else ""))
        print(f"  logs: {_log_file(cfg.port)}   stop it with: llm-router stop")
        return 0

    log_path = _log_file(cfg.port)
    pid_path = _pid_file(cfg.port)
    # Rotate the previous run's log so every boot starts a fresh, readable log
    # (keeps one backup as <log>.1 for reference).
    if log_path.exists() and log_path.stat().st_size > 0:
        try:
            log_path.replace(log_path.with_name(log_path.name + ".1"))
        except Exception:
            pass
    logf = open(log_path, "ab")
    logf.flush()
    # Forward the effective preload decision to the `run` child explicitly so its
    # behaviour is deterministic regardless of what it re-reads from the config.
    preload = _resolve_preload(cfg, args)
    cfg.preload = preload
    preload_flag = ["--preload"] if preload else ["--no-preload"]
    child = subprocess.Popen(
        [sys.executable, "-m", "llm_router", "serve",
         "--config", str(cfg_path), "--host", cfg.host, "--port", str(cfg.port)]
        + preload_flag,
        stdout=logf,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,   # own session: survives this shell; killable as a group
        cwd=(cfg.base_dir or None),
        close_fds=True,
    )
    pid_path.write_text(str(child.pid))
    up = _wait_for_health(cfg, timeout=15.0)
    print(f"llm-router started in the background (pid {child.pid}) on {cfg.public_base_url}")
    print(f"  default model: {cfg.default_model}   preload: {'on' if preload else 'off'}   config: {cfg_path}")
    print(f"  logs: {log_path}")
    print(f"  follow logs:   llm-router watch")
    print(f"  stop:          llm-router stop")
    if not up:
        _err("  warning: router not answering /health yet — check the log")
    return 0


# --------------------------------------------------------------------------- watch
def _tail_files(files: list, lines: int = 30) -> None:
    """Show the last N lines of each file, then follow new output (blocks)."""
    multi = len(files) > 1
    state = []  # list of [Path, offset]
    for f in files:
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        if size:
            with open(f, "rb") as fh:
                if size > 1 << 20:
                    fh.seek(-1 << 20, 2)
                    data = fh.read()
                else:
                    data = fh.read()
            for ln in data.decode("utf-8", "replace").splitlines()[-lines:]:
                print(f"[{f.name}] {ln}" if multi else ln)
            sys.stdout.flush()
        state.append([f, size])
    while True:
        for st in state:
            f, offset = st
            try:
                size = f.stat().st_size
            except OSError:
                continue
            if size < offset:
                offset = 0  # file rotated/truncated
            if size > offset:
                with open(f, "rb") as fh:
                    fh.seek(offset)
                    chunk = fh.read(size - offset)
                st[1] = size
                for ln in chunk.decode("utf-8", "replace").splitlines():
                    print(f"[{f.name}] {ln}" if multi else ln)
                sys.stdout.flush()
        time.sleep(0.4)


def cmd_watch(args) -> int:
    if args.llm:
        return cmd_watch_model_log(args)
    state = _state_dir()
    try:
        cfg = _load(resolve_config(args.config))
    except Exception:
        cfg = None
    host = args.host or (cfg.host if cfg else "127.0.0.1")
    port = args.port or (cfg.port if cfg else None)

    if args.port:
        files = [_log_file(args.port)]
    elif cfg:
        files = [_log_file(cfg.port)]
    else:
        files = sorted(state.glob("router-*.log"))
    files = [f for f in files if f.exists()]
    if not files:
        guess = port or 8787
        _err("no router logs found yet — has `llm-router run` been run?")
        _err(f"(expected log: {_log_file(guess)})")
        return 1

    # A leftover .log file is not enough: only watch a router that is ACTUALLY
    # running. Otherwise `watch` would tail a stale log from a previous, now-stopped
    # run -- exactly the case the `--llm` path already guards against.
    if port is not None:
        if not _health_up(host, port):
            _err(f"cannot reach a router at http://{host}:{port} — is `llm-router run` running?")
            return 3
    else:
        live = []
        for f in files:
            m = re.search(r"router-(\d+)\.log$", f.name)
            if m and _health_up(host, int(m.group(1))):
                live.append(f)
        if not live:
            _err("no running router found (log files exist but nothing is listening) — "
                 "is `llm-router run` running?")
            return 3
        files = live

    print(f"watching {', '.join(str(f) for f in files)} — Ctrl+C stops watching only, "
          f"the router keeps running", file=sys.stderr)
    try:
        _tail_files(files, args.lines)
    except KeyboardInterrupt:
        print("\nstopped watching (router keeps running)", file=sys.stderr)
    return 0


# ------------------------------------------------------ watch --llm (model log)
def _active_model(base: str) -> str | None:
    """The name of the active model on a running router, or None."""
    code, data = _admin(base, "/admin/active")
    if code == 200 and isinstance(data, dict):
        return data.get("active")
    return None


def _print_tail_lines(path: Path, lines: int) -> int:
    """Print the last `lines` lines of `path`; return the file's current size."""
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    if not size:
        return 0
    with open(path, "rb") as fh:
        if size > (1 << 20):
            fh.seek(-(1 << 20), 2)
            data = fh.read()
        else:
            data = fh.read()
    for ln in data.decode("utf-8", "replace").splitlines()[-lines:]:
        print(ln)
    sys.stdout.flush()
    return size


def _tail_new(path: Path, offset: int) -> int:
    """Print bytes appended to `path` after `offset`; return the new offset."""
    try:
        size = path.stat().st_size
    except OSError:
        return 0  # file vanished (model stopped) -> caller resets next tick
    if size < offset:
        return 0  # truncated/rotated -> caller resets
    if size > offset:
        with open(path, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read(size - offset)
        for ln in chunk.decode("utf-8", "replace").splitlines():
            print(ln)
        sys.stdout.flush()
        return size
    return offset


def cmd_watch_model_log(args) -> int:
    """Follow the ACTIVE model's own log, auto-switching when it changes."""
    cfg_path = resolve_config(args.config)
    try:
        cfg = _load(cfg_path)
    except Exception as e:
        _err(f"config error: {e}")
        return 2
    host = args.host or cfg.host
    port = args.port or cfg.port
    base = f"http://{host}:{port}"
    log_dir = cfg.log_dir  # absolute (resolved against the config dir on load)

    code, _ = _admin(base, "/admin/active")
    if code is None:
        _err(f"cannot reach a router at {base} — is `llm-router run` running?")
        return 3

    print("watching the ACTIVE model's own log  (not the router log)", file=sys.stderr)
    print(f"  log_dir: {log_dir}", file=sys.stderr)
    print("  auto-switches when the active model changes; "
          "Ctrl+C stops watching only (the router keeps running)", file=sys.stderr)

    current: Path | None = None
    offset = 0
    try:
        while True:
            active = _active_model(base)
            if not active:
                if current is not None:
                    current = None
                    print("\n--- no active model right now (all stopped) ---",
                          file=sys.stderr)
                time.sleep(1.0)
                continue
            log_path = Path(log_dir) / (safe_filename(active) + ".log")
            if log_path != current:
                current = log_path
                print(f"\n--- following model log: {active}  ({log_path}) ---",
                      file=sys.stderr)
                offset = _print_tail_lines(log_path, args.lines)
            else:
                offset = _tail_new(log_path, offset)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopped watching (router keeps running)", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- stop
def _is_router_server(cmd: str) -> bool:
    """True if `cmd` is a real llm_router server process (`serve`/`run`) — a
    python interpreter or the `llm-router` entry point running that subcommand.
    A shell whose command TEXT merely contains that string (e.g. the script that
    launched `run`) is NOT a server; without this guard `stop` would signal the
    very shell that invoked it and `status` would report a phantom instance."""
    argv = cmd.split()
    if not argv:
        return False
    exe = os.path.basename(argv[0])
    if not (exe.lower().startswith("python") or "llm-router" in argv[0]):
        return False
    return bool(re.search(r"(llm_router|llm-router)\s+(serve|run)\b", cmd))


def _collect_router_pids(port: int | None) -> list:
    """PIDs of running router servers: from pidfiles + any stray `serve`/`run`
    processes (only real python/llm-router server processes; see _is_router_server)."""
    pids, seen = [], set()
    me = os.getpid()
    # 1) pidfiles written by `run`
    for pf in sorted(_state_dir().glob("router-*.pid")):
        m = re.search(r"router-(\d+)\.pid$", pf.name)
        if port is not None and m and int(m.group(1)) != port:
            continue
        pid = _read_pid(pf)
        if pid and pid != me and _pid_alive(pid) and pid not in seen:
            seen.add(pid)
            pids.append(pid)
    # 2) stray foreground `serve`/`run` processes with no pidfile
    try:
        out = subprocess.check_output(["ps", "-A", "-o", "pid=", "-o", "command="], text=True)
        for ln in out.splitlines():
            parts = ln.strip().split(None, 1)
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            pid = int(parts[0])
            if pid in seen or pid == me:
                continue
            if _is_router_server(parts[1]):
                seen.add(pid)
                pids.append(pid)
    except Exception:
        pass
    return pids


def _is_watch_process(cmd: str) -> bool:
    """True if `cmd` is a real llm_router `watch` process (with or without --llm),
    i.e. a python interpreter or the `llm-router` entry point running that
    subcommand. A shell/terminal whose command TEXT merely contains that string is
    NOT a watcher, so `stop` never signals the terminal that launched `watch`."""
    argv = cmd.split()
    if not argv:
        return False
    exe = os.path.basename(argv[0])
    if not (exe.lower().startswith("python") or "llm-router" in argv[0]):
        return False
    return bool(re.search(r"(llm_router|llm-router)\s+watch\b", cmd))


def _cmdline_port(cmd: str):
    """The `--port` value in a process command line (int), or None if absent."""
    m = re.search(r"--port[= ](\d+)", cmd)
    return int(m.group(1)) if m else None


def _collect_watch_pids(port: int | None) -> list:
    """PIDs of running `watch` / `watch --llm` tailers (other terminal tabs).
    Only real python/llm-router `watch` processes count (see _is_watch_process).
    With `port` set, only watchers that explicitly target that port (`--port N`)
    are returned; a watcher with no `--port` (config default) is only matched by
    an unscoped `stop`."""
    pids, seen = [], set()
    me = os.getpid()
    try:
        out = subprocess.check_output(["ps", "-A", "-o", "pid=", "-o", "command="], text=True)
    except Exception:
        return pids
    for ln in out.splitlines():
        parts = ln.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid in seen or pid == me:
            continue
        cmd = parts[1]
        if not _is_watch_process(cmd):
            continue
        if port is not None and _cmdline_port(cmd) != port:
            continue
        seen.add(pid)
        pids.append(pid)
    return pids


def cmd_stop(args) -> int:
    """Stop all running router instances (and, via graceful shutdown, their active
    model), plus any `watch` / `watch --llm` tailers running in other terminals."""
    port = args.port  # None => all instances
    pids = _collect_router_pids(port)
    watch_pids = _collect_watch_pids(port)
    if not pids and not watch_pids:
        scope = f" on port {port}" if port else ""
        print(f"no running llm-router instance{scope} or watcher found (nothing to stop)")
        return 0

    grace = 15.0
    # Signal the routers first (graceful: uvicorn shutdown stops the active model).
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"sent SIGTERM to router pid {pid}")
        except ProcessLookupError:
            print(f"pid {pid} already gone")
        except PermissionError:
            _err(f"cannot signal pid {pid} (permission)")
    # Signal the watchers too, so they don't keep tailing a log that no longer
    # grows. They exit promptly on SIGTERM (no special handling needed).
    for pid in watch_pids:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"sent SIGTERM to watcher pid {pid}")
        except ProcessLookupError:
            print(f"watcher pid {pid} already gone")
        except PermissionError:
            _err(f"cannot signal watcher pid {pid} (permission)")

    # wait for graceful router exit (this is what stops the active model)
    remaining = set(pids)
    deadline = time.time() + grace
    while remaining and time.time() < deadline:
        for pid in list(remaining):
            if not _pid_alive(pid):
                remaining.discard(pid)
        if remaining:
            time.sleep(0.25)
    for pid in remaining:
        print(f"pid {pid} still running after {grace:.0f}s; sending SIGKILL")
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass

    for pf in _state_dir().glob("router-*.pid"):
        try:
            pf.unlink()
        except Exception:
            pass

    print(f"stopped {len(pids)} llm-router instance(s)")
    if pids:
        if not remaining:
            print("  active model(s) stopped on graceful shutdown")
        else:
            _err("  note: SIGKILL was needed; check for orphaned model processes")

    # Give the watchers a short grace, then SIGKILL any stragglers.
    if watch_pids:
        wremaining = set(watch_pids)
        wdeadline = time.time() + 5.0
        while wremaining and time.time() < wdeadline:
            for pid in list(wremaining):
                if not _pid_alive(pid):
                    wremaining.discard(pid)
            if wremaining:
                time.sleep(0.2)
        for pid in wremaining:
            print(f"watcher pid {pid} still running after 5s; sending SIGKILL")
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        print(f"stopped {len(watch_pids)} watcher(s)")
    return 0


# --------------------------------------------------------------- live admin clients
def _admin(base: str, path: str, method: str = "GET", payload=None):
    url = base + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode()
            return r.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body}
    except Exception as e:  # connection refused, timeout, ...
        return None, str(e)


def _router_base(args) -> str:
    if getattr(args, "base_url", None):
        return args.base_url.rstrip("/")
    cfg = _load(resolve_config(args.config))
    host = args.host or cfg.host
    port = args.port or cfg.port
    return f"http://{host}:{port}"


def _running_instances() -> list:
    """Discover running router instances -> [{port, pid, host, sources}].

    `sources` tags each instance 'daemon' (pidfile written by `run`) and/or
    'foreground' (a stray `serve`/`run` process). A daemonized instance is both
    (its pidfile plus the `serve` child it spawned), so it is reported once.
    """
    found = {}
    me = os.getpid()
    for pf in sorted(_state_dir().glob("router-*.pid")):
        m = re.search(r"router-(\d+)\.pid$", pf.name)
        if not m:
            continue
        port = int(m.group(1))
        pid = _read_pid(pf)
        if pid and pid != me and _pid_alive(pid):
            rec = found.setdefault(port, {"port": port, "pid": pid, "host": "127.0.0.1", "sources": set()})
            rec["sources"].add("daemon")
    try:
        out = subprocess.check_output(["ps", "-A", "-o", "pid=", "-o", "command="], text=True)
        for ln in out.splitlines():
            parts = ln.strip().split(None, 1)
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            pid = int(parts[0])
            if pid == me:
                continue
            if not _is_router_server(parts[1]):
                continue
            pm = re.search(r"--port[= ](\d+)", parts[1])
            if not pm:
                continue
            port = int(pm.group(1))
            hm = re.search(r"--host[= ](\S+)", parts[1])
            rec = found.setdefault(port, {"port": port, "pid": pid, "host": "127.0.0.1", "sources": set()})
            rec["sources"].add("foreground")
            if hm:
                rec["host"] = hm.group(1)
    except Exception:
        pass
    return [found[k] for k in sorted(found)]


def _split_base(base: str):
    """Split 'http://host:port' into (host, port:int)."""
    hp = base.split("://", 1)[1]
    if ":" in hp:
        h, _, p = hp.rpartition(":")
        if p.isdigit():
            return (h or "127.0.0.1"), int(p)
    return (hp or "127.0.0.1"), 8787


def _status_rec(rec) -> dict:
    """JSON-friendly form of one instance."""
    st = rec.get("status") or {}
    inst = rec["instance"]
    return {
        "port": inst["port"],
        "pid": inst.get("pid"),
        "base_url": f"http://{inst['host']}:{inst['port']}/v1",
        "sources": sorted(inst["sources"]),
        "reached": bool(rec.get("reached")),
        "active": st.get("active"),
        "ready": (rec.get("health") or {}).get("ready"),
        "models": [
            {"name": m.get("name"), "kind": m.get("kind"), "default": m.get("default"),
             "running": m.get("running"), "managed": m.get("managed"),
             "external": m.get("external"), "pid": m.get("pid")}
            for m in st.get("models", [])
        ],
        "error": rec.get("error"),
    }


def _status_block(rec) -> list:
    """Render one instance's status as a list of lines."""
    inst = rec["instance"]
    host, port = inst["host"], inst["port"]
    pid = inst.get("pid")
    src = "/".join(sorted(inst["sources"])) or "unknown"
    pid_s = f"pid {pid}" if pid else "pid ?"
    lines = [f"port {port}   http://{host}:{port}/v1   {pid_s}   [{src}]"]
    if not rec.get("reached"):
        lines.append(f"    not responding   ({rec.get('error') or 'no /admin/status'})")
        return lines
    health = rec.get("health") or {}
    data = rec.get("status") or {}
    active = data.get("active")
    ready = health.get("ready")
    if active:
        state = "ready" if ready is True else ("loading" if ready is False else "state unknown")
        lines.append(f"    active   {active}   ({state})")
    else:
        lines.append("    active   (none)   - starts on first request or `llm-router switch`")
    models = data.get("models", [])
    n_run = sum(1 for m in models if m.get("running"))
    n_mgr = sum(1 for m in models if m.get("managed"))
    n_ext = sum(1 for m in models if m.get("external"))
    lines.append(f"    models   {n_run} running / {len(models)} total   ({n_mgr} managed, {n_ext} external)")
    for m in models:
        name = m.get("name", "?")
        kind = m.get("kind") or "chat"
        st = "running" if m.get("running") else "stopped"
        pid = m.get("pid")
        pid_s = f"  pid {pid}" if pid else ""
        ctx = m.get("context_length")
        ctx_s = f"  ctx {ctx}" if ctx else ""
        dflt = "  [default]" if m.get("default") else ""
        ext = "  (external)" if m.get("external") else ""
        lines.append(f"      {name:<24} [{kind}]  {st:<7}{pid_s}{ctx_s}{dflt}{ext}")
    lf = _log_file(port)
    if lf.exists():
        lines.append(f"    log      {lf}")
    return lines


def _status_single(args) -> int:
    """Status of one router (selected via --port / --base-url / --config)."""
    try:
        base = _router_base(args)
    except Exception as e:
        _err(f"config error: {e}")
        return 2
    host, port = _split_base(base)
    code, data = _admin(base, "/admin/status", "GET")
    if code is None:
        _err(f"cannot reach router at {base}: {data}")
        return 3
    _, health = _admin(base, "/health", "GET")
    rec = {
        "instance": {"port": port, "pid": _read_pid(_pid_file(port)), "host": host, "sources": {"selected"}},
        "reached": code == 200,
        "status": data if isinstance(data, dict) else None,
        "health": health if isinstance(health, dict) else None,
        "error": None if code == 200 else str(data),
    }
    if getattr(args, "json", False):
        print(json.dumps([_status_rec(rec)], indent=2))
    else:
        print("\n".join(_status_block(rec)))
    return 0 if code == 200 else 1


def cmd_status(args) -> int:
    """Show the status of running router instance(s).

    No selector -> enumerate every running instance. `--port`/`--base-url` ->
    just that one. `--json` -> machine-readable output.
    """
    if getattr(args, "base_url", None) or getattr(args, "port", None):
        return _status_single(args)
    instances = _running_instances()
    if not instances:
        print("no running llm-router instances found.")
        print(f"  state dir: {_state_dir()}   - start one with: llm-router run")
        return 0
    recs = []
    for inst in instances:
        base = f"http://{inst['host']}:{inst['port']}"
        code, data = _admin(base, "/admin/status", "GET")
        _, health = _admin(base, "/health", "GET")
        recs.append({
            "instance": inst,
            "reached": code == 200,
            "status": data if isinstance(data, dict) else None,
            "health": health if isinstance(health, dict) else None,
            "error": None if code == 200 else str(data),
        })
    if getattr(args, "json", False):
        print(json.dumps([_status_rec(r) for r in recs], indent=2))
        return 0
    if len(recs) == 1:
        print("\n".join(_status_block(recs[0])))
    else:
        print(f"llm-router - {len(recs)} running instance(s)\n")
        for r in recs:
            print("\n".join(_status_block(r)))
            print()
    return 0


def cmd_switch(args) -> int:
    try:
        base = _router_base(args)
    except Exception as e:
        _err(f"config error: {e}")
        return 2
    code, data = _admin(base, "/admin/switch", "POST", {"model": args.model})
    if code is None:
        _err(f"cannot reach router at {base}: {data}")
        return 3
    if code != 200:
        _err(f"switch to '{args.model}' failed ({code}): {data}")
        return 1
    _err(f"active model is now '{args.model}'")
    return 0


def cmd_stop_model(args) -> int:
    try:
        base = _router_base(args)
    except Exception as e:
        _err(f"config error: {e}")
        return 2
    code, data = _admin(base, f"/admin/models/{args.model}/stop", "POST")
    if code is None:
        _err(f"cannot reach router at {base}: {data}")
        return 3
    if code != 200:
        _err(f"stop '{args.model}' failed ({code}): {data}")
        return 1
    _err(f"stopped model '{args.model}' (router itself keeps running)")
    return 0


# ---------------------------------------------------------------------- read-only
def cmd_check(args) -> int:
    cfg_path = resolve_config(args.config)
    try:
        cfg = _load(cfg_path)
    except Exception as e:
        _err(f"config error: {e}")
        return 2
    _print_resolved(cfg, cfg_path)
    return 0


def cmd_models(args) -> int:
    cfg_path = resolve_config(args.config)
    try:
        cfg = _load(cfg_path)
    except Exception as e:
        _err(f"config error: {e}")
        return 2
    print(f"config: {cfg_path}")
    for m in cfg.models:
        tag = "  (default)" if m.name == cfg.default_model else ""
        ctx = f"  context_length={m.context_length}" if m.context_length else ""
        re_ = f"  reasoning_effort={m.reasoning_effort}" if m.reasoning_effort else ""
        picker = "" if m.kind == "chat" else "  (not in the /model picker)"
        print(f"  {m.name} [{m.kind}]{tag}{ctx}{re_}{picker}")
        print(f"      -> {m.base_url}   upstream={m.upstream_model}")
    return 0


# -------------------------------------------------------------------- help text
MAIN_HELP = """# llm-router

OpenAI-compatible **router + lifecycle manager** for local LLM backends — the
`custom` provider. It advertises one preconfigured model id at a
time, keeps **only one large model in RAM** (memory-safe switching), and
transparently proxies the OpenAI API (`/v1/chat/completions`, `/v1/embeddings`,
…) to the active backend.

## Commands

| command              | what it does |
|---|---|
| `run`                | start the router as a background service (log to a file) |
| `serve`              | run in the foreground (blocking; used by `run`; also for launchd) |
| `watch`              | follow the router log — add `--llm` for the active model's own log |
| `stop`               | stop all router instances; graceful shutdown stops the active model |
| `stop-model <model>` | stop one model via the running router (the router keeps running) |
| `switch <model>`     | make a preconfigured model the active one (memory-safe) |
| `status`             | show status of ALL running instances (`--port N` for one; `--json`) |
| `models`             | list preconfigured models (name, kind, target, context, reasoning) |
| `check`              | validate the config and print the resolved specs (no server) |
| `config`             | open the config in `$EDITOR` (`--print` to just show its path) |
| `dashboard`          | web dashboard: instances, LLMs, pg/tg rates, config editor, live logs |
| `dashboard stop`     | stop a running dashboard (the `-d` background one, or a foreground one) |

Run `llm-router <command> --help` for the full markdown description + arguments
of any command.

## Config resolution (works from any directory)

    --config/--path  >  $LLM_ROUTER_CONFIG  >  ~/.config/llm-router/config.yaml
                     >  <repo>/config.yaml  >  ~/.config/llm-router/config.yaml

Relative paths in the config (`log_dir`, model `command`) resolve against the
config file's own directory.

## State (pids + the router log)

Lives in `~/.config/llm-router/` (override: `$LLM_ROUTER_STATE_DIR`) so
`stop` / `watch` work regardless of cwd or config. On each `run` the previous
run's log is rotated to `router-<port>.log.1`.
"""

HELP = {
    "run": """# llm-router run

Start the router as a **background service** and return immediately. The log is
written to `~/.config/llm-router/router-<port>.log` (follow it with
`llm-router watch`). By default the default model is preloaded in the background,
so the router answers `/health` right away even while a large model is still
loading — control this with `--preload` / `--no-preload` or the config's
`router.preload`.

## Usage

    llm-router run [--config P] [--host H] [--port N] [--preload | --no-preload]

## Arguments

| flag             | description |
|---|---|
| `-c, --config P` | config file path (default: auto-resolved) |
| `--host H`       | override the router bind host |
| `--port N`       | override the router port |
| `--preload`      | force-preload the default model (overrides `router.preload`) |
| `--no-preload`   | do NOT preload the default model (overrides `router.preload: true`) |

## Notes

- Preload precedence: `--preload` / `--no-preload` override the config; with
  neither flag the config's `router.preload` (default `true`) decides.
- With preload off, no model is loaded on boot; the first request to
  `/v1/chat/completions` (or `llm-router switch`) starts the model lazily.
- If a router is already running on the port, it reports that and does nothing.
- The previous run's log is rotated to `router-<port>.log.1` before starting.
- Stop it later with `llm-router stop`.
""",
    "serve": """# llm-router serve

Run the router in the **foreground** (it blocks). This is what `run` launches
detached in the background, and what the launchd plist runs under supervision.
Ctrl+C stops the router (graceful shutdown stops the active model). Preload is
controlled the same way as `run` (`--preload` / `--no-preload` / `router.preload`).

## Usage

    llm-router serve [--config P] [--host H] [--port N] [--preload | --no-preload]

## Arguments

| flag             | description |
|---|---|
| `-c, --config P` | config file path (default: auto-resolved) |
| `--host H`       | override the router bind host |
| `--port N`       | override the router port |
| `--preload`      | force-preload the default model (overrides `router.preload`) |
| `--no-preload`   | do NOT preload the default model (overrides `router.preload: true`) |
""",
    "watch": """# llm-router watch

Follow logs live. By default it tails the **router's** log; with `--llm` it tails
the **active model's own** log (its per-model file under `log_dir`) and
**auto-switches** to the new model's log whenever the active model changes (via
`llm-router switch` or the client changing models). Ctrl+C stops watching only — the
router keeps running. Both modes require a router to be **running** on the target
port: with none reachable, `watch` prints `cannot reach a router …` and exits — a
leftover log file from a stopped run is not tailed.

## Usage

    llm-router watch [--port N] [--lines N] [--config P]
    llm-router watch --llm [--host H] [--port N] [--lines N] [--config P]

## Arguments

| flag             | description |
|---|---|
| `--llm`          | watch the active model's own log instead of the router log |
| `--port N`       | instance port (default: from config) |
| `--host H`       | router host (default: from config) |
| `--lines N`      | initial lines to show (default: 30) |
| `-c, --config P` | config file path (default: auto-resolved) |

## Notes

- The active model's log lives at `<log_dir>/<name>.log`, where `<name>` is the
  model name flattened to a safe filename — Hugging Face `org/repo:tag` names are
  flattened, e.g. `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` becomes
  `unsloth-Qwen3.8-27B-GGUF-UD-Q4_K_M.log`.
- Both modes need a **running** router on the target port: the plain form probes
  `/health`, and `--llm` additionally reads the active model from `/admin/active`.
  With no router running, `watch` reports `cannot reach a router …` and exits.
""",
    "stop": """# llm-router stop

Stop all running router instances. Each shuts down **gracefully**: on shutdown it
stops the model it started (SIGTERM to the whole process group, then SIGKILL after
`shutdown_grace`). So the local LLM is stopped too — *including when it was still
loading* when `stop` was called. `stop` also stops any `watch` / `watch --llm`
tailers running in other terminals, so they don't keep tailing a log that no longer
grows. `--port N` stops just one instance (and, scoped to that port, only the
watchers that explicitly target it).

## Usage

    llm-router stop [--port N] [--config P]

## Arguments

| flag             | description |
|---|---|
| `--port N`       | only stop the instance on this port (default: all instances) |
| `-c, --config P` | config file path (default: auto-resolved) |

## Notes

- The router's log records `stopping model '<name>'` on the way out.
- A model that was already running *before* the router started (external/adopted)
  is left running — the router only stops what it started itself.
- If `stop` had to escalate to SIGKILL, it warns you to check for orphans.
- `stop` also signals any `watch` / `watch --llm` processes it finds (in other
  terminals): SIGTERM, then SIGKILL after 5s if they don't exit.
""",
    "stop-model": """# llm-router stop-model

Stop one specific model through the **running** router (its admin API). The
router itself keeps running — only that model's process is stopped.

## Usage

    llm-router stop-model <model> [--config P] [--host H] [--port N] [--base-url U]

## Arguments

| argument / flag   | description |
|---|---|
| `model`           | the name of the model to stop (required) |
| `-c, --config P`  | config file path (default: auto-resolved) |
| `--host H`        | router host (default: from config) |
| `--port N`        | router port (default: from config) |
| `--base-url U`    | full router base URL (overrides host/port) |
""",
    "switch": """# llm-router switch

Make a preconfigured model the active one on the **running** router. The
currently active model is stopped **completely first**, and only then is the new
model started — memory-safe, so the two never occupy RAM at the same time.

## Usage

    llm-router switch <model> [--config P] [--host H] [--port N] [--base-url U]

## Arguments

| argument / flag   | description |
|---|---|
| `model`           | the name of the model to activate (required) |
| `-c, --config P`  | config file path (default: auto-resolved) |
| `--host H`        | router host (default: from config) |
| `--port N`        | router port (default: from config) |
| `--base-url U`    | full router base URL (overrides host/port) |
""",
    "status": """# llm-router status

Show the live status of the **running** router instance(s). With no selector it
enumerates **every** running instance (discovered from the `router-<port>.pid`
pidfiles in `~/.config/llm-router/` plus any stray `serve`/`run` processes) and,
for each, reports its port + base URL + pid, how it was started, the active model
(and whether it is ready), and each preconfigured model's running / managed /
external state with its pid.

Add `--port N` (or `--base-url U`) to inspect just one instance, and `--json` for
machine-readable output.

## Usage

    llm-router status
    llm-router status --port 8787
    llm-router status --json

## Arguments

| flag             | description |
|---|---|
| `-c, --config P` | config file path (default: auto-resolved) |
| `--host H`       | router host (single-instance mode; default: from config) |
| `--port N`       | show just this port (default: enumerate all running instances) |
| `--base-url U`   | full router base URL (single instance; overrides host/port) |
| `--json`         | print machine-readable JSON |
""",
    "models": """# llm-router models

List the preconfigured models from the config (no server needed): name, kind,
target, context_length, and reasoning_effort.

## Usage

    llm-router models [--config P]

## Arguments

| flag             | description |
|---|---|
| `-c, --config P` | config file path (default: auto-resolved) |
""",
    "check": """# llm-router check

Validate the config file and print the fully resolved router + model specs
(nothing is started). Good for checking your config before running the router.

## Usage

    llm-router check [--config P]

## Arguments

| flag             | description |
|---|---|
| `-c, --config P` | config file path (default: auto-resolved) |

## Notes

- Warns if a `kind: chat` model advertises `context_length` below the client's
  64000 minimum (a client would refuse to select it as the active model).
""",
    "config": """# llm-router config

Open the resolved config file in your `$EDITOR`. If it does not exist yet, a
starter template is written first. Use `--print` to just show the path.

## Usage

    llm-router config [--path P] [--print]

## Arguments

| flag             | description |
|---|---|
| `--path P`       | config file path (default: auto-resolved) |
| `--print`        | print the resolved path instead of opening an editor |
""",
}


HELP["start"] = HELP["run"]   # hidden backward-compat alias for `run`

HELP["dashboard"] = """# llm-router dashboard

Serve the **web dashboard** for the router: running instances, their configured
LLMs (with the active one labelled), current + average prompt/generation (pg/tg)
rates per instance and per LLM, a full config editor, and live log watching.

The dashboard is its **own process** and **never auto-starts a router** — use the
"Run default instance" button (or the Run Instance tab) to start one.

## Usage

    llm-router dashboard [--host H] [--port N] [-c CONFIG] [-d|--detached] [--no-browser]
    llm-router dashboard stop

## Arguments

| flag                | description |
|---|---|
| `--host H`          | dashboard bind host (overrides `dashboard.host` in config; else `127.0.0.1`) |
| `--port N`          | dashboard port (overrides `dashboard.port` in config; else `8799`) |
| `-c, --config P`    | config file the dashboard edits/starts from (default: auto-resolved) |
| `-d, --detached`    | run in the background (log to `dashboard.log`); default is foreground |
| `--no-browser`      | don't auto-open the browser this run (overrides `dashboard.browser` in config) |
| `stop`              | stop a running dashboard (the `-d` one and any foreground one) |

By default `llm-router dashboard` runs in the **foreground** and **Ctrl+C** stops
it, and it **opens the dashboard in your default browser** (new window). With `-d`
it starts in the background; `llm-router dashboard stop` stops it. Use
`--no-browser` to skip the automatic browser open (handy over SSH).

## What's in the UI

- **Overview** — running instances, their LLMs (active labelled), instance and
  per-LLM current + average pg/tg rates. A "Run default instance" button starts
  the default config's instance if it isn't already running, and each instance
  card has a "Stop" button to stop just that instance.
- **Run Instance** — start a new instance on a fresh port, reusing LLMs from the
  default config (pick which ones and which is the default/active one).
- **Config** — a structured form for every router + model field (add/remove LLMs),
  plus a Raw YAML mode. Validated before save; a `.bak` backup is kept.
- **Logs** — watch a router's log, or any running LLM's own log, live. Click a
  "Watch" button to start; it turns into a "Stop watching" button to stop it.
- Dark/light theme. Most controls have a `?` tooltip explaining them on hover.
"""


# ------------------------------------------------------------------------ dashboard
def _dash_pidfile() -> Path:
    return _state_dir() / "dashboard.pid"


def _dash_log_file() -> Path:
    return _state_dir() / "dashboard.log"


def _port_free(host: str, port: int) -> bool:
    """True if nothing is bound to host:port yet (best-effort bind test)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _is_dashboard_process(cmd: str) -> bool:
    """True if `cmd` is a real running dashboard server: the `llm-router
    dashboard` CLI in the foreground (uvicorn in-process), or the
    `uvicorn llm_router.dashboard:dashboard_app` child spawned by `-d`. A
    shell/grep whose text merely mentions it is NOT counted, and the transient
    `dashboard stop` invocation itself is excluded."""
    argv = cmd.split()
    if not argv:
        return False
    exe = os.path.basename(argv[0])
    if not (exe.lower().startswith("python") or exe == "uvicorn" or "llm-router" in argv[0]):
        return False
    if re.search(r"uvicorn\s+llm_router\.dashboard\b", cmd):
        return True
    if re.search(r"(llm_router|llm-router)\s+dashboard\s+stop\b", cmd):
        return False
    if re.search(r"(llm_router|llm-router)\s+dashboard\b", cmd):
        return True
    return False


def _collect_dashboard_pids(port: int | None) -> list:
    """PIDs of running dashboard servers (the `-d` pidfile + any foreground/stray)."""
    pids, seen = [], set()
    me = os.getpid()
    pid = _read_pid(_dash_pidfile())
    if pid and pid != me and _pid_alive(pid) and pid not in seen:
        seen.add(pid)
        pids.append(pid)
    try:
        out = subprocess.check_output(["ps", "-A", "-o", "pid=", "-o", "command="], text=True)
    except Exception:
        return pids
    for ln in out.splitlines():
        parts = ln.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid in seen or pid == me:
            continue
        cmd = parts[1]
        if not _is_dashboard_process(cmd):
            continue
        if port is not None and _cmdline_port(cmd) != port:
            continue
        seen.add(pid)
        pids.append(pid)
    return pids


def cmd_dashboard_stop(args) -> int:
    """Stop a running dashboard (the `-d` background one and any foreground one)."""
    pids = _collect_dashboard_pids(None)
    if not pids:
        print("no running dashboard found (nothing to stop)")
        return 0
    grace = 8.0
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"sent SIGTERM to dashboard pid {pid}")
        except ProcessLookupError:
            print(f"dashboard pid {pid} already gone")
        except PermissionError:
            _err(f"cannot signal dashboard pid {pid} (permission)")
    deadline = time.time() + grace
    alive = set(pids)
    while alive and time.time() < deadline:
        for pid in list(alive):
            if not _pid_alive(pid):
                alive.discard(pid)
        if alive:
            time.sleep(0.2)
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"sent SIGKILL to dashboard pid {pid}")
        except ProcessLookupError:
            pass
    try:
        _dash_pidfile().unlink()
    except Exception:
        pass
    print("dashboard stopped")
    return 0


def _open_browser(url: str) -> None:
    """Best-effort open of ``url`` in the default browser (new window). Fails
    quietly on headless hosts (no browser / no display) — we just note it and
    continue serving."""
    import webbrowser
    try:
        opened = webbrowser.open(url, new=1)
    except Exception:
        opened = False
    if not opened:
        _err(f"note: no browser available to open {url} automatically")


def cmd_dashboard(args) -> int:
    """Serve the web dashboard. Foreground by default (Ctrl+C stops it);
    `-d`/`--detached` runs it in the background. `dashboard stop` stops it."""
    if getattr(args, "dash_stop", None) == "stop":
        return cmd_dashboard_stop(args)
    from .dashboard import dashboard_app, DEFAULT_DASH_PORT
    cfg_path = resolve_config(args.config)
    # Dashboard defaults come from the config's `dashboard:` section (host/port/
    # browser); CLI flags (--host/--port/--no-browser) override for this run.
    try:
        _dash = load_config(os.fspath(cfg_path)).dashboard
    except Exception:
        _dash = None
    host = args.host or (_dash.host if _dash else "127.0.0.1")
    port = args.port or (_dash.port if _dash else DEFAULT_DASH_PORT)
    browser = (_dash.browser if _dash else True) and not args.no_browser
    url = f"http://{host}:{port}"

    if not _port_free(host, port):
        existing = _collect_dashboard_pids(port)
        if existing:
            print(f"dashboard is already running on {url} (pid {existing[0]})")
            print(f"  stop it with: llm-router dashboard stop")
            return 0
        _err(f"port {port} is already in use by something else; pick another with --port")
        return 1

    if args.detached:
        log_path = _dash_log_file()
        if log_path.exists() and log_path.stat().st_size > 0:
            try:
                log_path.replace(log_path.with_name(log_path.name + ".1"))
            except Exception:
                pass
        logf = open(log_path, "ab")
        env = dict(os.environ, LLM_ROUTER_CONFIG=str(cfg_path))
        child = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "llm_router.dashboard:dashboard_app",
             "--host", host, "--port", str(port)],
            stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, env=env, close_fds=True,
        )
        _dash_pidfile().write_text(str(child.pid))
        up = _health_up(host, port, timeout=8.0)
        print(f"dashboard started in the background (pid {child.pid}) on {url}")
        print(f"  config: {cfg_path}")
        print(f"  logs:   {log_path}")
        print(f"  stop:   llm-router dashboard stop")
        if browser:
            _open_browser(url)
        if not up:
            _err("  warning: dashboard not answering /health yet — check the log")
        return 0

    _err(f"llm-router dashboard: serving {url}  (config: {cfg_path})")
    print(f"open {url} in your browser   (Ctrl+C to stop)")
    if browser:
        _open_browser(url)
    try:
        # timeout_graceful_shutdown=0 => on Ctrl+C, drop in-flight requests
        # (the dashboard's 2s poll and the Logs SSE stream, which never ends on
        # its own) instead of waiting for the open browser page to close.
        uvicorn.run(dashboard_app, host=host, port=port, log_level="warning",
                    timeout_graceful_shutdown=0)
    except KeyboardInterrupt:
        _err("interrupted; dashboard stopped")
    return 0


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # `--help` / `-h`  =>  full markdown-styled help (intercepted before argparse).
    for i, tok in enumerate(argv):
        if tok in ("-h", "--help"):
            sub = next((t for t in argv[:i] if not t.startswith("-")), None)
            print(HELP.get(sub, MAIN_HELP))
            return 0

    parser = argparse.ArgumentParser(
        prog="llm-router",
        description="OpenAI-compatible router + lifecycle manager for local LLM backends.",
    )
    parser.add_argument("--version", action="version", version=f"llm-router {__version__}")
    sub = parser.add_subparsers(dest="command")

    def _add_boot_args(p):
        p.add_argument("--config", "-c", default=None, help="config file path (default: auto-resolved)")
        p.add_argument("--host", default=None, help="override router bind host")
        p.add_argument("--port", type=int, default=None, help="override router port")
        pg = p.add_mutually_exclusive_group()
        pg.add_argument("--preload", action="store_true",
                        help="force-preload the default model (overrides router.preload)")
        pg.add_argument("--no-preload", dest="no_preload", action="store_true",
                        help="do NOT preload the default model (overrides router.preload: true)")

    p = sub.add_parser("run", help="start the router as a background service (logs to a file)")
    _add_boot_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("serve", help="run in the foreground (blocking; used by launchd, and by `run` to daemonize)")
    _add_boot_args(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("start", help=argparse.SUPPRESS)   # hidden backward-compat alias for `run`
    _add_boot_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("watch", help="follow the router log (--llm: the active model's own log)")
    p.add_argument("--port", type=int, default=None, help="instance port (default: from config)")
    p.add_argument("--lines", type=int, default=30, help="initial lines to show")
    p.add_argument("--config", "-c", default=None, help="config file path (default: auto-resolved)")
    p.add_argument("--host", default=None, help="router host (default: from config)")
    p.add_argument("--llm", action="store_true",
                   help="watch the active model's own log (not the router log); auto-switches on model change")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("stop", help="stop all running router instances + their active model, and any watch tailers")
    p.add_argument("--port", type=int, default=None, help="only stop the instance on this port (default: all)")
    p.add_argument("--config", "-c", default=None)
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("dashboard", help="web dashboard (Ctrl+C stops; -d for background; `dashboard stop` to stop it)")
    p.add_argument("--host", default=None,
                   help="dashboard bind host (default: dashboard.host in config, else 127.0.0.1)")
    p.add_argument("--port", type=int, default=None,
                   help="dashboard port (default: dashboard.port in config, else 8799)")
    p.add_argument("--config", "-c", default=None, help="config file (default: auto-resolved)")
    p.add_argument("-d", "--detached", dest="detached", action="store_true",
                   help="run in the background (log to dashboard.log)")
    p.add_argument("--no-browser", dest="no_browser", action="store_true",
                   help="do NOT auto-open the dashboard in the default browser")
    p.add_argument("dash_stop", nargs="?", default=None,
                   help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("stop-model", help="stop one model via the running router")
    p.add_argument("model", help="model name")
    p.add_argument("--config", "-c", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--base-url", default=None)
    p.set_defaults(func=cmd_stop_model)

    p = sub.add_parser("config", help="open the config file in $EDITOR")
    p.add_argument("--path", default=None, help="config file path (default: auto-resolved)")
    p.add_argument("--print", action="store_true", help="print the resolved path instead of opening an editor")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("check", help="validate config and print resolved specs (no server)")
    p.add_argument("--config", "-c", default=None)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("models", help="list preconfigured models")
    p.add_argument("--config", "-c", default=None)
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("status", help="show status of running router instance(s)")
    p.add_argument("--config", "-c", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None, help="show just this port (single instance)")
    p.add_argument("--base-url", default=None, help="full router base URL (single instance)")
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("switch", help="make a preconfigured model the active one")
    p.add_argument("model", help="model name")
    p.add_argument("--config", "-c", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--base-url", default=None)
    p.set_defaults(func=cmd_switch)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        print(MAIN_HELP)
        return 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # Keep the log clean: silence httpx/httpcore per-request INFO lines (verbose,
    # and they include noisy upstream 404s for capability probes like /v1/version).
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
