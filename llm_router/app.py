"""FastAPI app for the local LLM router.

Exposes an OpenAI-compatible surface so a client can use it as a ``custom``
provider endpoint:

    GET  /v1/models           -> the list of *preconfigured* models (req #2)
    POST /v1/chat/completions -> forwarded to the active model   (req #7/#8)
    POST /v1/embeddings       -> forwarded to the active model
    GET  /v1/health           -> router health

Any other path under the app is transparently forwarded to the active model at
the same logical path, so the router is a faithful proxy, not a chat-only stub.

Admin endpoints (for lifecycle control, req #3/#4/#5):

    GET  /admin/status
    GET  /admin/models
    POST /admin/switch                {"model": "<name>"}
    POST /admin/models/{name}/start
    POST /admin/models/{name}/stop
"""
from __future__ import annotations

import json
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import MIN_CONTEXT_LEN, RouterConfig
from .manager import ModelManager, ModelNotReady

logger = logging.getLogger("llm_router.app")


def _mgr(request: Request) -> ModelManager:
    return request.app.state.mgr


def _logical_upstream_path(incoming: str) -> str:
    """Map an incoming path onto the upstream OpenAI path.

    Clients may use base_url with or without a trailing ``/v1``; both must land
    on the upstream's ``/v1/...`` route.
    """
    p = incoming.lstrip("/")
    if p.startswith("v1/"):
        return "/" + p
    return "/v1/" + p


def _model_from_body(body: bytes):
    """Extract the ``model`` field from a JSON body, if any."""
    if not body:
        return None
    try:
        data = json.loads(body)
    except Exception:
        return None
    if isinstance(data, dict) and "model" in data:
        return data["model"]
    return None


def _prepare_body(body: bytes, spec) -> bytes:
    """Prepare the outbound body for a configured model.

    * rewrites ``model`` to what the local server expects (req #7);
    * injects the model's *default* ``reasoning_effort`` (from config) when the
      client did not set one, so the model's default applies while
      the client can still override it per-request (req #3).
    """
    if not body:
        return body
    try:
        data = json.loads(body)
    except Exception:
        return body
    if not isinstance(data, dict):
        return body
    if "model" in data:
        data["model"] = spec.upstream_model
    if (
        "messages" in data
        and data.get("reasoning_effort") in (None, "")
        and spec.reasoning_effort
    ):
        data["reasoning_effort"] = spec.reasoning_effort
    return json.dumps(data).encode()


async def _forward(mgr: ModelManager, request: Request, model_name: str) -> Response:
    spec = mgr.specs[model_name]
    url = spec.base_url + _logical_upstream_path(request.url.path)
    if request.url.query:
        url += "?" + request.url.query

    body = await request.body()
    fwd_body = _prepare_body(body, spec)

    fwd_headers = {}
    for k, v in request.headers.items():
        low = k.lower()
        if low in ("host", "content-length", "accept-encoding", "authorization"):
            continue
        fwd_headers[k] = v
    if fwd_body:
        fwd_headers["content-length"] = str(len(fwd_body))

    req = mgr.http.build_request(request.method, url, headers=fwd_headers, content=fwd_body)
    resp = await mgr.http.send(req, stream=True)

    out_headers = {}
    for k, v in resp.headers.items():
        if k.lower() in (
            "content-length", "transfer-encoding", "connection",
            "keep-alive", "content-encoding", "date",
        ):
            continue
        out_headers[k] = v

    async def gen():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        except Exception:
            logger.exception("upstream stream error for model '%s'", model_name)
        finally:
            await resp.aclose()

    return StreamingResponse(gen(), status_code=resp.status_code, headers=out_headers)


def build_app(cfg: RouterConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mgr = ModelManager(cfg)
        app.state.mgr = mgr
        logger.info(
            "router starting; default model '%s' (public base %s)",
            cfg.default_model, cfg.public_base_url,
        )
        for m in cfg.models:
            if m.kind != "chat":
                logger.info(
                    "model '%s' is kind=%s (managed by the router, but not listed "
                    "in the /v1/models picker)", m.name, m.kind,
                )
            elif m.context_length is not None and m.context_length < MIN_CONTEXT_LEN:
                logger.warning(
                    "model '%s' advertises context_length=%d, below the client's "
                    "%d minimum — a client will refuse to select it as the active model "
                    "(mark it kind: embedding, or raise its real context window)",
                    m.name, m.context_length, MIN_CONTEXT_LEN,
                )
        # Start the default model in the background so the router is reachable
        # immediately: a daemonised `llm-router run` returns right away and
        # /health answers while the (possibly large) default model is loading.
        async def _preload() -> None:
            try:
                active = await mgr.ensure_active(cfg.default_model)   # req #3
                if active != cfg.default_model:
                    logger.warning(
                        "default model '%s' was not activated (its port is already "
                        "serving external model '%s'); active model is '%s'",
                        cfg.default_model, active, active,
                    )
                else:
                    logger.info("default model '%s' is active and ready", cfg.default_model)
            except Exception as e:
                logger.error("could not start default model '%s': %s", cfg.default_model, e)
        # Preload is optional: `router.preload` (config) sets the default, and the
        # --preload / --no-preload flags on `start`/`run` override it. When off,
        # nothing loads on boot — the first request lazily starts the model (the
        # /{path} forward handler calls ensure_active before proxying).
        app.state.preload_task = None
        if cfg.preload:
            app.state.preload_task = asyncio.create_task(_preload())
        else:
            logger.info(
                "preload disabled — no model loaded on boot; default model '%s' "
                "starts on the first request (or 'llm-router switch')",
                cfg.default_model,
            )
        yield
        task = getattr(app.state, "preload_task", None)
        if task is not None and not task.done():
            # The default model may still be loading (its process is already
            # spawned). Cancel the in-flight load so it can't block shutdown…
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                # A cancelled load raises CancelledError — a BaseException that
                # `except Exception` does NOT catch. If it escaped, `mgr.shutdown()`
                # below would be skipped and a model that was mid-startup would be
                # left running (orphaned) after `llm-router stop`.
                pass
        # ALWAYS stop every model the router started, even if the load above was
        # cancelled mid-way: a model in startup state is still a real, running
        # process and must not survive the router's shutdown.
        await mgr.shutdown()

    app = FastAPI(title="llm-router", version="1.2.5", lifespan=lifespan,
                  docs_url=None, openapi_url=None, redoc_url=None)

    def _authed(request: Request) -> bool:
        if not cfg.api_key:
            return True
        return request.headers.get("authorization", "") == f"Bearer {cfg.api_key}"

    # ------------------------------------------------------------ OpenAI surface
    @app.get("/v1/models")
    @app.get("/models")
    async def list_models(request: Request):
        if not _authed(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        mgr = _mgr(request)
        data = []
        for m in cfg.models:
            # Only "chat" models are offered to the /model picker. An
            # "embedding"-kind model can never be a valid active (chat) model,
            # so it stays out of the list — it is still managed (start/stop) and
            # served for /v1/embeddings, just not selectable in the picker.
            if m.kind != "chat":
                continue
            obj = {
                "id": m.name,
                "object": "model",
                "created": 0,
                "owned_by": "llm-router",
                "active": mgr.active == m.name,
            }
            if m.context_length:
                # clients read `context_length` (then context_window/max_context_length)
                # from the /v1/models object for custom endpoints.
                obj["context_length"] = m.context_length
                obj["max_context_length"] = m.context_length
            if m.reasoning_effort:
                # the model's default reasoning effort, advertised so clients can
                # see it (and override it per-request).
                obj["reasoning_effort"] = m.reasoning_effort
            data.append(obj)
        return {"object": "list", "data": data}

    @app.get("/v1/health")
    @app.get("/health")
    async def health(request: Request):
        mgr = _mgr(request)
        task = getattr(request.app.state, "preload_task", None)
        return {
            "status": "ok",
            "active": mgr.active,
            "ready": bool(mgr.active) and (task is None or task.done()),
            "managed": len(mgr.owned),
            "external": len(mgr.external),
        }

    @app.get("/")
    async def index():
        return {
            "service": "llm-router",
            "hint": "OpenAI-compatible. Use base_url %s ; see /admin/status"
            % cfg.public_base_url,
        }

    # ------------------------------------------------------------------- admin
    @app.get("/admin/status")
    async def admin_status(request: Request):
        return _mgr(request).status()

    @app.get("/admin/active")
    async def admin_active(request: Request):
        # Lightweight "who's active" probe for `llm-router watch --llm`. Kept out
        # of the access log (see __main__._SUPPRESSED_ACCESS_PATHS) so the
        # watcher's polling never floods the router log.
        mgr = _mgr(request)
        task = getattr(request.app.state, "preload_task", None)
        return {
            "active": mgr.active,
            "ready": bool(mgr.active) and (task is None or task.done()),
        }

    @app.get("/admin/models")
    async def admin_models(request: Request):
        st = _mgr(request).status()
        return {
            "active": st["active"],
            "models": [
                {
                    "name": m["name"],
                    "upstream_model": m["upstream_model"],
                    "base_url": m["base_url"],
                    "port": m["port"],
                    "default": m["default"],
                    "context_length": m.get("context_length"),
                    "active": m["active"],
                    "running": m["running"],
                }
                for m in st["models"]
            ],
        }

    @app.post("/admin/switch")
    async def admin_switch(request: Request):
        mgr = _mgr(request)
        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        name = data.get("model") or data.get("name")
        if name not in mgr.specs:
            return JSONResponse(
                {"error": f"unknown model '{name}'", "available": list(mgr.specs)},
                status_code=400,
            )
        try:
            await mgr.ensure_active(name)
        except ModelNotReady as e:
            return JSONResponse({"error": str(e), "model": name}, status_code=503)
        return mgr.status()

    @app.post("/admin/models/{name}/start")
    async def admin_start(name: str, request: Request):
        mgr = _mgr(request)
        if name not in mgr.specs:
            return JSONResponse(
                {"error": f"unknown model '{name}'", "available": list(mgr.specs)},
                status_code=400,
            )
        try:
            await mgr.ensure_active(name)
        except ModelNotReady as e:
            return JSONResponse({"error": str(e), "model": name}, status_code=503)
        return mgr.status()

    @app.post("/admin/models/{name}/stop")
    async def admin_stop(name: str, request: Request):
        mgr = _mgr(request)
        if name not in mgr.specs:
            return JSONResponse(
                {"error": f"unknown model '{name}'", "available": list(mgr.specs)},
                status_code=400,
            )
        await mgr.stop(name)
        if mgr.active == name:
            mgr.active = None
        return mgr.status()

    # ------------------------------------------------------- transparent proxy
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def forward(request: Request, path: str):
        if not _authed(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        mgr = _mgr(request)
        body = await request.body()
        requested = _model_from_body(body)
        if requested is not None:
            if requested in mgr.specs:
                target = requested
            else:
                # Req #4: only preconfigured models are allowed.
                return JSONResponse(
                    {"error": f"unknown model '{requested}'",
                     "available": list(mgr.specs)},
                    status_code=400,
                )
        elif mgr.active:
            target = mgr.active
        else:
            target = cfg.default_model
        try:
            await mgr.ensure_active(target)
        except ModelNotReady as e:
            return JSONResponse({"error": str(e), "model": target}, status_code=503)
        return await _forward(mgr, request, target)

    return app
