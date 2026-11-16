"""Configuration loading and validation for the local LLM router.

A config file is a YAML document with two top-level keys:

    router:    router-level settings (port, default model, timeouts, auth, ...)
    models:    a list of preconfigured local LLM backends

Each model entry needs at minimum:

    name:    the id exposed to hermes-agent (what you type in /model,
             and what is returned by GET /v1/models).
    command: the full startup command *with arguments*, e.g.
             "splash serve --model X --host 127.0.0.1 --port 8080 ..."

The host/port the backend listens on is taken from an explicit ``host``/``port``
field if present, otherwise it is parsed out of ``--host`` / ``--port`` in the
command string. ``upstream_model`` (optional) is the model id that is sent to the
local server in the request body; it defaults to ``name``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

# Match "--port 8080", "--port=8080".
_PORT_RE = re.compile(r"--port[= ]([0-9]+)")
# Match "--host 127.0.0.1", "--host=0.0.0.0".
_HOST_RE = re.compile(r"--host[= ]([A-Za-z0-9.\-]+)")

# hermes-agent refuses to make a model the active one unless its advertised
# context window is at least this (agent/model_metadata.py: MINIMUM_CONTEXT_LENGTH).
# A *chat* model below this cannot be selected as the active model in hermes.
HERMES_CONTEXT_MIN = 64_000

# Reasoning-effort levels hermes-agent's `custom` provider may send/accept.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# Model kinds. "chat" models are hermes active-model candidates (listed in
# GET /v1/models). "embedding" models are managed by the router (start/stop,
# /v1/embeddings) but are NOT offered in the hermes chat picker, because they
# can never serve /v1/chat/completions.
KINDS = ("chat", "embedding")


@dataclass
class ModelSpec:
    name: str
    command: str
    upstream_model: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    startup_timeout: Optional[float] = None
    health_path: Optional[str] = None
    context_length: Optional[int] = None   # advertised to hermes via /v1/models
    reasoning_effort: Optional[str] = None  # model's default reasoning effort (forwarded)
    kind: str = "chat"                      # "chat" or "embedding"
    env: dict[str, str] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        host = self.host or "127.0.0.1"
        return f"http://{host}:{self.port}"


def safe_filename(name: str) -> str:
    """Sanitize a model name for use as a flat log filename.

    A Hugging Face model id like ``unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`` contains
    ``/`` and ``:``; used verbatim in ``<log_dir>/<name>.log`` it becomes a
    nested or invalid path. Map any character outside ``[A-Za-z0-9._-]`` to ``-``
    so the result is a single, safe filename component (the name is only a log
    filename here — the model id used in /v1/models and routing is left as-is).
    """
    return re.sub(r"[^A-Za-z0-9._-]", "-", name) or "_"


def _parse_command(command: str):
    """Extract (host, port) from --host/--port flags in a command string."""
    host_m = _HOST_RE.search(command)
    port_m = _PORT_RE.search(command)
    return (
        host_m.group(1) if host_m else None,
        int(port_m.group(1)) if port_m else None,
    )


def _resolve_host_port(spec: ModelSpec) -> ModelSpec:
    cmd_host, cmd_port = _parse_command(spec.command)
    if spec.host is None:
        spec.host = cmd_host or "127.0.0.1"
    if spec.port is None:
        spec.port = cmd_port
    if spec.port is None:
        raise ValueError(
            f"model '{spec.name}': could not determine the port it listens on. "
            f"Set 'port' explicitly or include '--port <n>' in the command."
        )
    if not spec.upstream_model:
        spec.upstream_model = spec.name
    return spec


@dataclass
class DashboardConfig:
    """Defaults for the web dashboard (`llm-router dashboard`).

    Optional ``dashboard:`` section in the config file. CLI flags on
    ``llm-router dashboard`` (--host/--port/--no-browser) override these for
    a single run; the web dashboard's Config tab edits them back here.
    """
    host: str = "127.0.0.1"
    port: int = 8799
    browser: bool = True


@dataclass
class RouterConfig:
    host: str = "127.0.0.1"
    port: int = 8787
    api_key: str = ""
    default_model: Optional[str] = None
    startup_timeout: float = 600.0
    health_interval: float = 1.0
    health_path: str = "/health"
    log_dir: str = "./logs"
    keep_inactive_stopped: bool = True
    preload: bool = True              # preload the default model on start (override: --preload/--no-preload)
    shutdown_grace: float = 10.0
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)  # web dashboard defaults
    models: list[ModelSpec] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    source: str = ""    # absolute path of the config file
    base_dir: str = ""  # directory of the config file (cwd for spawned models)

    @property
    def public_base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def _parse_kind(name: str, raw) -> str:
    kind = str(raw or "chat").strip().lower()
    if kind in ("embeddings", "embed"):
        kind = "embedding"
    if kind not in KINDS:
        raise ValueError(f"model '{name}': kind must be one of {KINDS} (got {raw!r})")
    return kind


def _parse_reasoning_effort(name: str, raw):
    if raw is None or str(raw).strip() == "":
        return None
    eff = str(raw).strip().lower()
    if eff not in REASONING_EFFORTS:
        raise ValueError(
            f"model '{name}': reasoning_effort must be one of {REASONING_EFFORTS} (got {raw!r})"
        )
    return eff


def load_config(path: str) -> RouterConfig:
    path = os.path.abspath(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping with 'router' and 'models'")

    r = data.get("router", {}) or {}
    models_raw = data.get("models", []) or []
    if not models_raw:
        raise ValueError("config must define a non-empty 'models' list")

    models: list[ModelSpec] = []
    seen: set[str] = set()
    for i, m in enumerate(models_raw):
        if not isinstance(m, dict) or "name" not in m or "command" not in m:
            raise ValueError(f"models[{i}] must be a mapping with 'name' and 'command'")
        name = str(m["name"])
        if name in seen:
            raise ValueError(f"duplicate model name: '{name}'")
        seen.add(name)
        models.append(
            ModelSpec(
                name=name,
                command=str(m["command"]),
                upstream_model=m.get("upstream_model") or m.get("upstream"),
                host=m.get("host"),
                port=int(m["port"]) if m.get("port") is not None else None,
                startup_timeout=(
                    float(m["startup_timeout"]) if m.get("startup_timeout") is not None else None
                ),
                health_path=m.get("health_path"),
                context_length=(
                    int(m["context_length"]) if m.get("context_length") is not None else None
                ),
                reasoning_effort=_parse_reasoning_effort(name, m.get("reasoning_effort")),
                kind=_parse_kind(name, m.get("kind", "chat")),
                env=dict(m.get("env", {}) or {}),
            )
        )

    d = data.get("dashboard")
    if d is None:
        d = {}
    if not isinstance(d, dict):
        raise ValueError("config 'dashboard' section must be a mapping")
    try:
        dash_port = int(d.get("port", 8799))
    except (TypeError, ValueError):
        raise ValueError(f"config dashboard.port must be an integer (got {d.get('port')!r})")
    if dash_port <= 0:
        raise ValueError("config dashboard.port must be a positive integer")
    dash_cfg = DashboardConfig(
        host=str(d.get("host", "127.0.0.1")),
        port=dash_port,
        browser=bool(d.get("browser", True)),
    )

    cfg = RouterConfig(
        host=str(r.get("host", "127.0.0.1")),
        port=int(r.get("port", 8787)),
        api_key=str(r.get("api_key", "") or ""),
        default_model=r.get("default_model"),
        startup_timeout=float(r.get("startup_timeout", 600.0)),
        health_interval=float(r.get("health_interval", 1.0)),
        health_path=str(r.get("health_path", "/health")),
        log_dir=str(r.get("log_dir", "./logs")),
        keep_inactive_stopped=bool(r.get("keep_inactive_stopped", True)),
        preload=bool(r.get("preload", True)),
        shutdown_grace=float(r.get("shutdown_grace", 10.0)),
        dashboard=dash_cfg,
        models=models,
        raw=data,
    )

    for spec in cfg.models:
        _resolve_host_port(spec)
        if spec.context_length is not None and spec.context_length <= 0:
            raise ValueError(f"model '{spec.name}': context_length must be a positive integer")

    if cfg.default_model is None:
        cfg.default_model = models[0].name
    if cfg.default_model not in seen:
        raise ValueError(f"default_model '{cfg.default_model}' is not a configured model")

    cfg.source = path
    cfg.base_dir = os.path.dirname(path)
    if not os.path.isabs(cfg.log_dir):
        cfg.log_dir = os.path.join(cfg.base_dir, cfg.log_dir)
    cfg.log_dir = os.path.abspath(cfg.log_dir)
    os.makedirs(cfg.log_dir, exist_ok=True)
    return cfg
