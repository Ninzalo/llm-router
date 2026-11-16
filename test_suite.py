#!/usr/bin/env python
"""End-to-end validation of the llm-router against the mock (test) config.
Asserts each of the 8 requested features. Run with the venv python."""
import json, os, sys, time, socket
import httpx

BASE = "http://127.0.0.1:18787"
V1 = BASE + "/v1"
HF = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"   # HF-style model id (with '/' and ':')

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

def port_up(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except Exception:
        return False

# ---- wait for router + default model ----
for _ in range(30):
    try:
        h = httpx.get(BASE + "/health", timeout=2).json()
        if h.get("active"):
            break
    except Exception:
        pass
    time.sleep(0.5)

# Req 2: /v1/models lists the preconfigured CHAT models (the hermes picker source).
# Embedding-only models (kind: embedding) are still router-managed but are NOT
# offered here, so hermes can't pick one as its active chat model.
r = httpx.get(V1 + "/models", timeout=5)
ids = [m["id"] for m in r.json().get("data", [])]
check("Req2  GET /v1/models returns chat models only (embedding model hidden)",
      set(ids) == {"qwen-local", "qwen-local-2", HF} and "embed-local" not in ids,
      f"ids={ids} status={r.status_code}")

# root alias also works (and also hides the embedding model)
r2 = httpx.get(BASE + "/models", timeout=5)
check("Req2  GET /models (root alias) works + hides embedding model",
      set(m["id"] for m in r2.json().get("data", [])) == {"qwen-local", "qwen-local-2", HF})

# context_length + reasoning_effort are advertised per-model on /v1/models (hermes
# reads them from here for custom endpoints); /admin/status still lists all models.
rctx = httpx.get(V1 + "/models", timeout=5).json()
ctx = {m["id"]: m.get("context_length") for m in rctx.get("data", [])}
check("Ctx  /v1/models advertises per-model context_length (chat models only)",
      ctx.get("qwen-local") == 131072 and "embed-local" not in ctx, f"ctx={ctx}")
stctx = httpx.get(BASE + "/admin/status", timeout=5).json()
ctxs = {x["name"]: x.get("context_length") for x in stctx["models"]}
check("Ctx  /admin/status advertises per-model context_length (all models)",
      ctxs.get("qwen-local") == 131072 and ctxs.get("embed-local") == 4096, f"ctx={ctxs}")

# reasoning_effort: the router advertises the model's default, injects it when the
# client omits it, and passes the client's value through when provided.
qm = next(m for m in rctx["data"] if m["id"] == "qwen-local")
check("Reason /v1/models advertises the chat model's default reasoning_effort",
      qm.get("reasoning_effort") == "high", f"qm={qm}")
r = httpx.post(V1 + "/chat/completions",
               json={"model": "qwen-local", "messages": [{"role": "user", "content": "re-a"}],
                     "stream": False}, timeout=10)
ca = r.json()["choices"][0]["message"]["content"]
check("Reason default reasoning_effort injected when client omits it",
      "reasoning_effort='high'" in ca, f"content={ca!r}")
r = httpx.post(V1 + "/chat/completions",
               json={"model": "qwen-local", "messages": [{"role": "user", "content": "re-b"}],
                     "stream": False, "reasoning_effort": "low"}, timeout=10)
cb = r.json()["choices"][0]["message"]["content"]
check("Reason client reasoning_effort passes through (wins over default)",
      "reasoning_effort='low'" in cb, f"content={cb!r}")

# Req 3: default model auto-started on boot
h = httpx.get(BASE + "/health", timeout=5).json()
check("Req3  default model started on startup",
      h.get("active") == "qwen-local" and port_up(18080),
      f"health={h} mock-a(18080)={port_up(18080)}")

# Req 7+8 (hop 1): route to qwen-local -> mock-a; answer comes back
body = {"model": "qwen-local", "messages": [{"role": "user", "content": "hello-router"}], "stream": False}
r = httpx.post(V1 + "/chat/completions", json=body, timeout=10)
content = r.json()["choices"][0]["message"]["content"]
check("Req7  request routed to correct local LLM (mock-a)",
      content.startswith("[mock:mock-a]"), f"status={r.status_code} content={content!r}")
check("Req8  answer from local LLM passed back to caller",
      "you said: hello-router" in content, f"content={content!r}")
check("Req6  model field rewritten to upstream id (qwen-local -> mock-a)",
      "mock-a" in content and "qwen-local" not in content, f"content={content!r}")

# base_url WITHOUT /v1 must also work (path normalization)
r = httpx.post(BASE + "/chat/completions", json=body, timeout=10)
check("Req1  OpenAI-compat endpoint (base_url without /v1) works",
      r.status_code == 200 and "[mock:mock-a]" in r.json()["choices"][0]["message"]["content"],
      f"status={r.status_code}")

# Req 4 (switch): route to embed-local -> router must start mock-b, stop mock-a
body2 = {"model": "embed-local", "messages": [{"role": "user", "content": "switching"}], "stream": False}
r = httpx.post(V1 + "/chat/completions", json=body2, timeout=20)
content2 = r.json()["choices"][0]["message"]["content"]
check("Req4  can change to another preconfigured model (embed-local)",
      content2.startswith("[mock:mock-b]"), f"status={r.status_code} content={content2!r}")

# Req 5: only the active model runs; the other one was stopped
st = httpx.get(BASE + "/admin/status", timeout=5).json()
m = {x["name"]: x for x in st["models"]}
check("Req5  switching started the new model (mock-b up)",
      m["embed-local"]["running"] and port_up(18081), f"{m['embed-local']}")
check("Req5  switching stopped the inactive model (mock-a down)",
      (not port_up(18080)) and (not m["qwen-local"]["running"]),
      f"18080={port_up(18080)} status={m['qwen-local']}")
check("Req5  exactly one model running",
      sum(1 for x in st["models"] if x["running"]) == 1,
      f"running={[x['name'] for x in st['models'] if x['running']]}")

# Req 4 (enforce): a non-preconfigured model is rejected
bad = {"model": "totally-bogus", "messages": [{"role": "user", "content": "x"}]}
r = httpx.post(V1 + "/chat/completions", json=bad, timeout=10)
check("Req4  non-preconfigured model rejected (400)",
      r.status_code == 400, f"status={r.status_code} body={r.text[:120]}")

# Req 5 (explicit admin stop): stop the active model, confirm it goes down
r = httpx.post(BASE + "/admin/models/embed-local/stop", timeout=10).json()
time.sleep(0.3)
check("Req5  explicit admin stop works",
      (not port_up(18081)) and not r["models"][1]["running"], f"{r['models'][1]}")

# Req 4 (explicit admin switch back): switch to qwen-local via admin
r = httpx.post(BASE + "/admin/switch", json={"model": "qwen-local"}, timeout=20).json()
check("Req4  admin switch works",
      r["active"] == "qwen-local" and port_up(18080) and not port_up(18081),
      f"active={r['active']} 18080={port_up(18080)} 18081={port_up(18081)}")

# --- memory-safe switch: old model FULLY stopped BEFORE the new one starts ---
# Switch to the 2nd chat model (qwen-local-2 -> mock-c :18082). The old active
# model (qwen-local -> mock-a :18080) must be completely stopped (process exited,
# RAM freed) before mock-c is spawned, so the two never share RAM. Proven via the
# daemon log: "stopped model 'qwen-local'" must PRECEDE "starting model
# 'qwen-local-2'".
body4 = {"model": "qwen-local-2", "messages": [{"role": "user", "content": "mem-switch"}], "stream": False}
r = httpx.post(V1 + "/chat/completions", json=body4, timeout=30)
content4 = r.json()["choices"][0]["message"]["content"]
check("Mem   switch to 2nd chat model works (routed to mock-c)",
      content4.startswith("[mock:mock-c]"), f"status={r.status_code} content={content4!r}")

st2 = httpx.get(BASE + "/admin/status", timeout=5).json()
m2 = {x["name"]: x for x in st2["models"]}
check("Mem   exactly one model running after switch (mock-c up, mock-a down)",
      m2["qwen-local-2"]["running"] and port_up(18082)
      and not port_up(18080) and not m2["qwen-local"]["running"],
      f"18082={port_up(18082)} 18080={port_up(18080)} {m2['qwen-local']}")

_state = os.environ.get("LLM_ROUTER_STATE_DIR") or os.path.expanduser("~/.config/llm-router")
DLOG = os.path.join(_state, "router-18787.log")
logtext = open(DLOG, encoding="utf-8", errors="replace").read() if os.path.exists(DLOG) else ""
# Ordering proof: the daemon must log "stopping model 'qwen-local'" (stop() is
# invoked, then awaits the process to exit) BEFORE "starting model
# 'qwen-local-2'" (the new one is spawned). stop() waits for the old process to
# fully exit (killpg + proc.wait()) before returning, so by the time start()
# logs "starting model", the old model's RAM is already freed.
idx_start = logtext.find("starting model 'qwen-local-2'")
before_start = logtext[:idx_start] if idx_start != -1 else ""
idx_stop = before_start.rfind("stopping model 'qwen-local'")
check("Mem   old model fully stopped BEFORE new model started (log ordering)",
      idx_start != -1 and idx_stop != -1,
      f"idx_start={idx_start} idx_stop={idx_stop} (stop precedes start in log)")

# --- model names with '/' and ':' (Hugging Face ids) must be usable ---
# The user must be able to name a model exactly as its HF id, e.g.
# 'unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M'. That name is used VERBATIM as the
# /v1/models id and for routing; only the per-model LOG FILE is sanitized to a
# flat filename (no '/'), so switching to it must NOT create a nested
# <log_dir>/unsloth/ directory (that is exactly the bug that broke startup).
body_hf = {"model": HF, "messages": [{"role": "user", "content": "hf-name"}], "stream": False}
r = httpx.post(V1 + "/chat/completions", json=body_hf, timeout=30)
content_hf = r.json()["choices"][0]["message"]["content"]
check("Name  HF-style model name (with '/' and ':') is routable",
      r.status_code == 200 and content_hf.startswith("[mock:mock-d]"),
      f"status={r.status_code} content={content_hf!r}")
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
safe_log = os.path.join(LOGDIR, "unsloth-Qwen3.8-27B-GGUF-UD-Q4_K_M.log")
nested_dir = os.path.join(LOGDIR, "unsloth")
check("Name  per-model log written to a FLAT filename (slash/colon sanitized)",
      os.path.exists(safe_log), f"exists={os.path.exists(safe_log)} path={safe_log}")
check("Name  no nested 'unsloth/' subdirectory created under log_dir",
      not os.path.isdir(nested_dir), f"nested={nested_dir} isdir={os.path.isdir(nested_dir)}")

# Streaming (Req 7/8 with stream:true) -> SSE passthrough
body3 = {"model": "qwen-local", "messages": [{"role": "user", "content": "stream-me"}], "stream": True}
chunks, got_done = [], False
with httpx.Client(timeout=20) as c:
    with c.stream("POST", V1 + "/chat/completions", json=body3) as resp:
        ct = resp.headers.get("content-type", "")
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    got_done = True
                else:
                    chunks.append(json.loads(payload))
check("Req7  streaming request routed (SSE passthrough)",
      chunks and chunks[0].get("model") == "mock-a", f"n_chunks={len(chunks)} ct={ct}")
check("Req8  streaming answer reassembled + [DONE]",
      got_done and "".join(c["choices"][0]["delta"].get("content", "") for c in chunks).strip().startswith("[mock:mock-a]"),
      f"resembled={''.join(c['choices'][0]['delta'].get('content','') for c in chunks)!r}")

print("\n==== SUMMARY ====")
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
for name, ok, detail in results:
    print(f"  {'OK ' if ok else 'XX '} {name}")
print(f"\n{passed}/{total} checks passed")
sys.exit(0 if passed == total else 1)
