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


def _log_timing() -> None:
    """Emit llama.cpp-style per-request timing lines to stdout.

    The router captures a model's stdout into its per-model log file
    (<log_dir>/<name>.log). A real local backend (splash / llama.cpp) writes
    ``prompt eval time`` and ``eval time`` lines that report prompt-processing
    (pg) and generation (tg) speed in tokens/sec; the dashboard parses those
    lines for current + average rates. We emit the same shape (with a little
    jitter, so "current" and "average" are distinguishable) so the parser and
    the dashboard are validated end to end.
    """
    pg = round(random.uniform(18.0, 26.0), 2)     # prompt tokens/sec
    tg = round(random.uniform(90.0, 120.0), 2)    # generation tokens/sec
    pt = random.randint(150, 320)                 # prompt tokens
    gt = random.randint(80, 180)                  # generated tokens
    prompt_ms = pt * (1000.0 / pg)
    gen_ms = gt * (1000.0 / tg)
    print(
        f"prompt eval time =  {prompt_ms:9.2f} ms /  {pt:4d} tokens "
        f"({1000.0 / pg:8.2f} ms per token, {pg:8.2f} tokens/sec)",
        flush=True,
    )
    print(
        f"        eval time =  {gen_ms:9.2f} ms /  {gt:4d} tokens "
        f"({1000.0 / tg:8.2f} ms per token, {tg:8.2f} tokens/sec)",
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
