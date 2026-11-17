#!/usr/bin/env python3
"""A tiny OpenAI-compatible fake local LLM, for validating the router in an
isolated environment (separate ports) without loading real GGUF models.

Mimics the surface the real backends expose:
    GET  /health              -> {"status":"ok"}
    GET  /v1/models           -> {"object":"list","data":[{"id":...}]}
    POST /v1/chat/completions -> echoes a canned reply; honors "stream"

It echoes back the ``model`` field it received, so the router's model-rewrite
can be verified end to end.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

STATE = {"model": "mock"}

app = FastAPI(docs_url=None, openapi_url=None)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{"id": STATE["model"], "object": "model", "created": 0, "owned_by": "mock"}],
    }


_REQ_SEQ = {"n": 0}


def _log_timing() -> None:
    """Emit a splash-serve-style per-request ``Done`` line to stdout.

    The router captures a model's stdout into its per-model log file
    (<log_dir>/<name>.log). A real local backend (splash serve) writes one
    ``Done ...`` line per finished request carrying the generation rate
    (``tok/s``) plus the input / cached / TTFT fields the dashboard uses to
    derive the prompt rate. We emit the same shape (with jitter, so "current"
    and "average" are distinguishable) so the parser and the dashboard are
    validated end to end against the REAL log format.
    """
    _REQ_SEQ["n"] += 1
    req = _REQ_SEQ["n"]
    inp = random.randint(4000, 40000)                       # prompt tokens
    # first request is cold (nothing cached); later ones reuse most of it
    cached = 0 if req == 1 else min(inp, int(inp * random.uniform(0.0, 0.9)))
    ttft = random.uniform(0.5, 60.0)                        # time to first token, s
    tg = round(random.uniform(20.0, 120.0), 1)              # generation tok/s (jitter)
    out = random.randint(50, 400)                           # generated tokens
    tok_cycle = round(tg / random.uniform(4.0, 9.0), 1)     # speculative-decode rate
    print(
        f"{time.strftime('%H:%M:%S')} Done · request {req} · input {inp:,} · "
        f"cached {cached:,} · output {out} · tools 0 · TTFT {ttft:.1f}s · "
        f"{tg} tok/s · {tok_cycle} tok/cycle",
        flush=True,
    )


@app.post("/v1/chat/completions")
async def chat(request: Request):
    data = await request.json()
    model = data.get("model")
    msgs = data.get("messages") or []
    stream = bool(data.get("stream"))
    last = (msgs[-1].get("content") if msgs else "") or ""
    reply = f"[mock:{model}] you said: {last} (reasoning_effort={data.get('reasoning_effort')!r})"
    created = int(time.time())
    cid = f"chatcmpl-{created}"
    _log_timing()

    if stream:
        async def gen():
            for tok in reply.split(" "):
                chunk = {
                    "id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": tok + " "}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            done = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return {
        "id": cid, "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--model", default="mock")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="sleep this long before serving (simulates a slow model load)")
    a = ap.parse_args()
    STATE["model"] = a.model
    if a.delay > 0:
        time.sleep(a.delay)
    # log at INFO so the per-model log file (what `llm-router watch --llm` tails)
    # has the same kind of content a real local model (splash/llama.cpp) produces:
    # startup lines + per-request access logs.
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
