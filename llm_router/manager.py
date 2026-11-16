"""Process lifecycle management for local LLM backends.

The manager owns the processes it launches. At any point in time the router
keeps a single "active" model: switching to a model (re)starts it if needed and
stops every other *managed* model, so only one large model occupies RAM at once.

A model that is already listening on its port when the router starts (e.g. you
boots the router while an LLM is already up) is *adopted* as external: it is
routed to, but never killed by the router (it was not started by the router).
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import signal
import time
from typing import Optional

import httpx

from .config import ModelSpec, RouterConfig, safe_filename

logger = logging.getLogger("llm_router.manager")


class ModelNotReady(Exception):
    """Raised when a model fails to become healthy within its startup timeout."""


class ModelManager:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.specs: dict[str, ModelSpec] = {m.name: m for m in cfg.models}
        self.proc: dict[str, asyncio.subprocess.Process] = {}
        self.owned: set[str] = set()      # started (and killable) by the router
        self.external: set[str] = set()   # pre-existing, adopted, not killable
        self.log_fds: dict[str, object] = {}
        self.active: Optional[str] = None
        self.lock = asyncio.Lock()
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=10.0, read=600.0, write=60.0, pool=30.0)
        )

    # ------------------------------------------------------------- health
    async def _probe(self, spec: ModelSpec) -> bool:
        """Best-effort readiness probe against the backend's own endpoint."""
        paths = []
        if spec.health_path:
            paths.append(spec.health_path)
        paths.append(self.cfg.health_path)
        paths.append("/v1/models")
        seen = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            url = spec.base_url + path
            try:
                r = await self.http.get(url, timeout=5.0)
                # 2xx-4xx means the server is up and answering; 5xx or no answer = not ready
                if r.status_code < 500:
                    return True
            except Exception:
                continue
        return False

    async def is_up(self, name: str) -> bool:
        spec = self.specs[name]
        if name in self.proc and self.proc[name].returncode is not None:
            return False
        return await self._probe(spec)

    # ------------------------------------------------- external model identity
    async def _identify_external(self, spec: ModelSpec) -> Optional[str]:
        """Best-effort: which *configured* model does the backend at ``spec``'s
        host:port actually serve?

        A port probe alone can't tell which model an already-listening backend is
        loading — several configured models may share one host:port (e.g. two GGUF
        variants pointed at the same server port). The backend advertises its
        loaded model via the OpenAI-compatible ``GET /v1/models`` endpoint, so we
        match the returned id(s) against the configured models to attribute the
        external/running state to the model that is really loaded there.

        Returns the matched configured model name, or ``None`` if the backend can't
        be reached or its model matches no configured model (the caller then falls
        back to assuming the port serves ``spec.name``).
        """
        for url in (f"{spec.base_url}/v1/models", f"{spec.base_url}/models"):
            try:
                r = await self.http.get(url, timeout=5.0)
            except Exception:
                continue
            if r.status_code != 200:
                continue
            try:
                data = r.json().get("data", [])
            except Exception:
                continue
            ids = [str(m.get("id") or m.get("name")) for m in data]
            ids = [i for i in ids if i]
            if not ids:
                continue
            matched = self._match_loaded_model(ids)
            if matched is not None:
                return matched
        return None

    def _match_loaded_model(self, ids: list[str]) -> Optional[str]:
        """Match backend-reported model id(s) to a configured model name.

        Tries, in order: exact (name / upstream_model), case-insensitive exact,
        then basename equality (a backend may report a full path, e.g.
        ``/models/qwen-q4km.gguf``). Returns the first matching configured model
        name (in config order) or ``None``.
        """
        if not ids:
            return None
        # 1) exact match on the hermes-facing name or the upstream id
        for name, spec in self.specs.items():
            for lid in ids:
                if lid == name or lid == spec.upstream_model:
                    return name
        # 2) case-insensitive exact
        low = {lid.lower(): lid for lid in ids}
        for name, spec in self.specs.items():
            if name.lower() in low or (spec.upstream_model and spec.upstream_model.lower() in low):
                return name
        # 3) basename match (full path vs bare model id)
        for name, spec in self.specs.items():
            for cand in (name, spec.upstream_model):
                if not cand:
                    continue
                for lid in ids:
                    if os.path.basename(lid).lower() == os.path.basename(cand).lower():
                        return name
        return None

    # ---------------------------------------------------------- lifecycle
    async def start(self, name: str, wait: bool = True) -> str:
        """Start (or adopt) a model. Returns when it is healthy if ``wait``.

        Returns the name of the model that ended up running: normally ``name``,
        but if ``name``'s host:port is already serving an *external* backend that
        is loading a **different** configured model, that model is adopted instead
        (``name`` cannot start on a busy port) and is returned.
        """
        spec = self.specs[name]
        alive = name in self.proc and self.proc[name].returncode is None

        if not alive:
            # Already listening externally? Adopt instead of spawning a conflict.
            if await self._probe(spec):
                # The port is up, but a port probe alone cannot say WHICH model is
                # actually loaded there -- several configured models may share one
                # host:port. Ask the backend so the external/running state is
                # attributed to the model that is really loaded, not whichever
                # configured model happens to share the port.
                loaded = await self._identify_external(spec)
                if loaded is not None and loaded != name and loaded in self.specs:
                    if loaded not in self.external:
                        self.external.add(loaded)
                    logger.warning(
                        "model '%s' cannot start: port %s is already serving external "
                        "model '%s' -> adopting '%s' as the running external model and "
                        "leaving '%s' stopped",
                        name, spec.base_url, loaded, loaded, name,
                    )
                    return loaded
                self.external.add(name)
                logger.info(
                    "model '%s' already listening at %s -> adopted as external "
                    "(router will route to it but will not kill it)",
                    name, spec.base_url,
                )
                return name
            await self._spawn(name)

        if wait:
            await self._wait_ready(name)
        return name

    async def _spawn(self, name: str) -> None:
        spec = self.specs[name]
        argv = shlex.split(spec.command)
        env = {**os.environ, **{k: str(v) for k, v in spec.env.items()}}
        log_path = os.path.join(self.cfg.log_dir, safe_filename(name) + ".log")
        logf = open(log_path, "ab")
        logger.info("starting model '%s': %s", name, spec.command)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            env=env,
            stdout=logf,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True,  # own process group so we can kill the whole tree
            cwd=(self.cfg.base_dir or None),  # resolve relative commands next to the config
        )
        self.proc[name] = proc
        self.owned.add(name)
        self.external.discard(name)
        self.log_fds[name] = logf
        logger.info("model '%s' spawned (pid %s)", name, proc.pid)

    async def _wait_ready(self, name: str) -> None:
        spec = self.specs[name]
        timeout = spec.startup_timeout or self.cfg.startup_timeout
        deadline = time.monotonic() + timeout
        log_path = os.path.join(self.cfg.log_dir, safe_filename(name) + ".log")
        while time.monotonic() < deadline:
            if name in self.proc and self.proc[name].returncode is not None:
                raise ModelNotReady(
                    f"model '{name}' exited early (code {self.proc[name].returncode}); "
                    f"see {log_path}"
                )
            if await self._probe(spec):
                logger.debug("model '%s' ready at %s", name, spec.base_url)
                return
            await asyncio.sleep(self.cfg.health_interval)
        raise ModelNotReady(f"model '{name}' not ready within {timeout:.0f}s")

    async def stop(self, name: str) -> bool:
        """Stop a model. Only kills processes the router started itself."""
        if name in self.proc:
            proc = self.proc[name]
            if proc.returncode is None:
                pid = proc.pid
                logger.info("stopping model '%s' (pid %s)", name, pid)
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=self.cfg.shutdown_grace)
                except asyncio.TimeoutError:
                    logger.warning("model '%s' did not exit; sending SIGKILL", name)
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except Exception:
                        pass
                    await proc.wait()
            self.proc.pop(name, None)
            self.owned.discard(name)
            fd = self.log_fds.pop(name, None)
            if fd is not None:
                try:
                    fd.close()
                except Exception:
                    pass
            return True
        if name in self.external:
            logger.warning(
                "model '%s' is external (not started by the router); leaving it running",
                name,
            )
        return False

    async def ensure_active(self, name: str) -> str:
        """Make ``name`` the active model.

        Any other router-*owned* model is stopped FIRST — and completely — before
        the target is started, so the old and new models never occupy RAM at the
        same time (memory-safe switching on low-memory machines). Adopted
        (external) models are never killed; if one is still up it is left running.

        Returns the model that ended up active. Normally ``name``; but if
        ``name``'s port is already occupied by an external backend serving a
        different configured model, that model becomes active instead (the
        requested model cannot start on a busy port).
        """
        if name not in self.specs:
            raise KeyError(name)
        async with self.lock:
            if self.cfg.keep_inactive_stopped:
                # Stop the old model(s) completely before bringing the new one up.
                for other in list(self.specs):
                    if other == name:
                        continue
                    if other in self.proc and self.proc[other].returncode is None:
                        await self.stop(other)
                    elif other in self.external:
                        logger.info(
                            "external model '%s' still running; not managed by router", other
                        )
            active = await self.start(name, wait=True)
            self.active = active
            if active != name:
                logger.warning(
                    "requested model '%s' could not be activated (its port is already "
                    "serving external model '%s'); the active model is now '%s'",
                    name, active, active,
                )
            logger.debug("active model is now '%s'", active)
            return active

    # ------------------------------------------------------------ reporting
    def status(self) -> dict:
        out = []
        for name, spec in self.specs.items():
            running = (
                (name in self.proc and self.proc[name].returncode is None)
                or (name in self.external)
            )
            out.append(
                {
                    "name": name,
                    "upstream_model": spec.upstream_model,
                    "base_url": spec.base_url,
                    "port": spec.port,
                    "default": name == self.cfg.default_model,
                    "context_length": spec.context_length,
                    "reasoning_effort": spec.reasoning_effort,
                    "kind": spec.kind,
                    "running": running,
                    "managed": name in self.owned,
                    "external": name in self.external,
                    "active": self.active == name,
                    "pid": self.proc[name].pid if name in self.proc else None,
                    "log_path": os.path.join(self.cfg.log_dir, safe_filename(name) + ".log"),
                }
            )
        return {"active": self.active, "models": out, "log_dir": self.cfg.log_dir}

    async def shutdown(self) -> None:
        owned = list(self.owned)
        logger.info(
            "shutting down: stopping %d owned model(s): %s",
            len(owned), ", ".join(owned) if owned else "(none)",
        )
        async with self.lock:
            for name in owned:
                try:
                    await self.stop(name)
                except Exception as e:
                    # One model's stop failing must not prevent the others from
                    # being cleaned up on shutdown.
                    logger.warning(
                        "error while stopping model '%s' on shutdown: %s", name, e
                    )
        try:
            await self.http.aclose()
        except Exception:
            pass
