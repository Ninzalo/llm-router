#!/usr/bin/env python
"""Unit tests for the dashboard pg/tg rate parser. Pure — no server, no ports,
no real models, so it runs anywhere (local + CI).

Covers BOTH real log formats the router's models emit:
  * splash serve  "Done · ... input X · cached Y · ... TTFT Ss · R tok/s ..."
                  (the format this project's models actually log — the one the
                  bug was breaking)
  * llama.cpp     "prompt eval time = ... ( T tokens/sec)" / "eval time = ..."

Run:  python tests/test_rate_parse.py    (exit 0 == all pass)
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from llm_router.dashboard import parse_rate_line, RateTracker, _router_rate  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


def as_dict(rates):
    return dict(rates) if rates else {}


# --- Real log lines, verbatim, from a live splash-serve model log ---
SPLASH_COLD = ("13:24:11 Done \u00b7 request 1 \u00b7 input 32,226 \u00b7 cached 0 \u00b7 output 118 \u00b7 "
               "tools 15\u00b75217171e \u00b7 TTFT 54.1s \u00b7 46.2 tok/s \u00b7 6.2 tok/cycle")
SPLASH_WARM = ("14:39:44 Done \u00b7 request 2 \u00b7 input 36,099 \u00b7 cached 35,712 \u00b7 output 199 \u00b7 "
               "tools 15\u00b75217171e \u00b7 TTFT 1.1s \u00b7 73.2 tok/s \u00b7 6.4 tok/cycle")
SPLASH_ALL_CACHED = ("15:00:00 Done \u00b7 request 3 \u00b7 input 1,000 \u00b7 cached 1,000 \u00b7 output 10 \u00b7 "
                     "TTFT 0.2s \u00b7 50.0 tok/s \u00b7 5.0 tok/cycle")

# llama.cpp per-iteration lines (legacy / other backends)
LLAMA_PROMPT = "prompt eval time =    7743.71 ms /   197 tokens (   39.31 ms per token,    25.44 tokens/sec)"
LLAMA_GEN = "        eval time =    1611.04 ms /   153 tokens (   10.53 ms per token,    94.97 tokens/sec)"
LLAMA_LEGACY_TS = "prompt eval time =  2274.99 ms /  871 tokens (   2.61 ms per token,  382.87 t/s)"

# Non-timing lines that must yield None
NOT_TIMING = [
    "14:38:40 Started \u00b7 request 1 \u00b7 input 35,738 \u00b7 cached 0 \u00b7 effort medium \u00b7 thinking on",
    "14:39:42 TTFT of request 1: 61960.9 ms: own prefill 61928.4, claim wait 0, decode 0, other prefill 0, rest 32.5095",
    "14:38:39 Ready \u00b7 unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M \u00b7 context 128K \u00b7 http://127.0.0.1:8080",
    "14:40:42 Stopping \u00b7 releasing engine resources",
    "some ordinary log line with no rate",
]

print("== splash serve 'Done' lines (the real format) ==")
d = as_dict(parse_rate_line(SPLASH_COLD))
check("cold Done -> gen + prompt rates", set(d) == {"gen", "prompt"}, str(d))
check("cold gen (tg) == 46.2", d.get("gen") == 46.2, str(d))
check("cold prompt (pg) == (32,226-0)/54.1", abs(d.get("prompt", -1) - 32226 / 54.1) < 1e-6, str(d))

d = as_dict(parse_rate_line(SPLASH_WARM))
check("warm gen (tg) == 73.2", d.get("gen") == 73.2, str(d))
check("warm prompt (pg) == (36,099-35,712)/1.1", abs(d.get("prompt", -1) - (36099 - 35712) / 1.1) < 1e-6, str(d))

d = as_dict(parse_rate_line(SPLASH_ALL_CACHED))
check("all-cached Done -> gen only (no prompt rate)", set(d) == {"gen"} and d.get("gen") == 50.0, str(d))

print("== non-timing lines yield None ==")
for ln in NOT_TIMING:
    check(f"-> None: {ln[:42]!r}...", parse_rate_line(ln) is None, str(parse_rate_line(ln)))

print("== llama.cpp lines (legacy / other backends) still parse ==")
d = as_dict(parse_rate_line(LLAMA_PROMPT))
check("llama prompt (pg) == 25.44", set(d) == {"prompt"} and d.get("prompt") == 25.44, str(d))
d = as_dict(parse_rate_line(LLAMA_GEN))
check("llama gen (tg) == 94.97", set(d) == {"gen"} and d.get("gen") == 94.97, str(d))
d = as_dict(parse_rate_line(LLAMA_LEGACY_TS))
check("llama legacy 't/s' prompt == 382.87", set(d) == {"prompt"} and d.get("prompt") == 382.87, str(d))

print("== RateTracker: current + average accumulate across requests ==")
rt = RateTracker()
key = "/tmp/test_rate_parse.log"
rt._st[key] = rt._fresh(key, 0)  # seed the entry (as sample() does on first sight)
for ln in (SPLASH_COLD, SPLASH_WARM):
    rt._absorb(key, ln)
st = rt._st[key]
check("pg_n == 2 requests", st["pg_n"] == 2, str(st))
check("tg_n == 2 requests", st["tg_n"] == 2, str(st))
check("cur_tg == 73.2 (latest request)", abs(st["cur_tg"] - 73.2) < 1e-9, str(st["cur_tg"]))
check("cur_pg == warm pg (latest request)", abs(st["cur_pg"] - (36099 - 35712) / 1.1) < 1e-6, str(st["cur_pg"]))
check("tg_avg == mean(46.2, 73.2) == 59.7", abs(st["tg_sum"] / st["tg_n"] - (46.2 + 73.2) / 2) < 1e-9, str(st))
exp_pg_avg = (32226 / 54.1 + (36099 - 35712) / 1.1) / 2
check("pg_avg == mean of the two prompt rates", abs(st["pg_sum"] / st["pg_n"] - exp_pg_avg) < 1e-9, str(st))
check("current != average (jitter preserved)", st["cur_tg"] != st["tg_sum"] / st["tg_n"], str(st))

print("== instance (router) rate: current from active model, average pooled across all models ==")
# Two models with distinct, known rates. The instance "current" tg/pg must come
# from the ACTIVE model, while the instance AVERAGE tg/pg is pooled across ALL
# running models of the instance (not just the active one).
#   alpha: input 1,000 cached 0, TTFT 2.0s, 100.0 tok/s  -> pg=(1000-0)/2.0=500, tg=100
#   beta : input 1,000 cached 0, TTFT 1.0s, 200.0 tok/s  -> pg=(1000-0)/1.0=1000, tg=200
import tempfile as _tf
_tmp = _tf.mkdtemp(prefix="llmrouter_router_rate_")
a_path = os.path.join(_tmp, "alpha.log")
b_path = os.path.join(_tmp, "beta.log")
open(a_path, "w", encoding="utf-8").write("13:00:00 Done \u00b7 request 1 \u00b7 input 1,000 \u00b7 cached 0 \u00b7 output 10 \u00b7 tools 0 \u00b7 TTFT 2.0s \u00b7 100.0 tok/s \u00b7 10.0 tok/cycle\n")
open(b_path, "w", encoding="utf-8").write("13:00:05 Done \u00b7 request 1 \u00b7 input 1,000 \u00b7 cached 0 \u00b7 output 10 \u00b7 tools 0 \u00b7 TTFT 1.0s \u00b7 200.0 tok/s \u00b7 20.0 tok/cycle\n")
rt2 = RateTracker()
ka, kb = "18590:alpha", "18590:beta"
a_rate = rt2.sample(ka, a_path)   # seeds + reads alpha.log (the active model)
b_rate = rt2.sample(kb, b_path)   # seeds + reads beta.log (a second running model)
check("alpha parsed (active): tg_avg 100, pg_avg 500", a_rate is not None and abs(a_rate["tg_avg"] - 100.0) < 1e-9 and abs(a_rate["pg_avg"] - 500.0) < 1e-9, str(a_rate))
check("beta parsed (2nd model): tg_avg 200, pg_avg 1000", b_rate is not None and abs(b_rate["tg_avg"] - 200.0) < 1e-9 and abs(b_rate["pg_avg"] - 1000.0) < 1e-9, str(b_rate))
router_raw = _router_rate(a_rate, rt2, [ka, kb])
check("router object built", router_raw is not None, str(router_raw))
router = router_raw or {}
check("router CURRENT tg == active model tg (100.0)", router.get("tg_current") == 100.0, str(router))
check("router CURRENT pg == active model pg (500.0)", router.get("pg_current") == 500.0, str(router))
check("router AVG tg pooled == mean(100,200)=150.0 (NOT active's 100)", router.get("tg_avg") == 150.0, str(router))
check("router AVG pg pooled == mean(500,1000)=750.0 (NOT active's 500)", router.get("pg_avg") == 750.0, str(router))
check("router AVG pulls in the other model (differs from active-only AVG)", router.get("tg_avg") != (a_rate or {}).get("tg_avg") and router.get("pg_avg") != (a_rate or {}).get("pg_avg"), f"router={router} active={a_rate}")
single = _router_rate(b_rate, rt2, [kb]) or {}
check("single running model -> average equals that model's own (no other model)", single.get("tg_avg") == 200.0 and single.get("pg_avg") == 1000.0, str(single))
check("no active-model rate -> None", _router_rate(None, rt2, [ka, kb]) is None)

print("== persistence: averages survive a router/dashboard restart (never disappear) ==")
_pdir = _tf.mkdtemp(prefix="llmrouter_rate_persist_")
_plog = os.path.join(_pdir, "qwen.log")
_pstate = os.path.join(_pdir, "state")
REQ1 = "13:00:00 Done \u00b7 request 1 \u00b7 input 1,000 \u00b7 cached 0 \u00b7 output 10 \u00b7 tools 0 \u00b7 TTFT 2.0s \u00b7 100.0 tok/s \u00b7 10.0 tok/cycle\n"
REQ2 = "13:05:00 Done \u00b7 request 2 \u00b7 input 1,000 \u00b7 cached 0 \u00b7 output 10 \u00b7 tools 0 \u00b7 TTFT 1.0s \u00b7 200.0 tok/s \u00b7 20.0 tok/cycle\n"
open(_plog, "w", encoding="utf-8").write(REQ1)
rt1 = RateTracker(state_dir=_pstate)
r1 = rt1.sample("18590:qwen", _plog)
rt1.flush()
check("p1: first request captured (tg 100, pg 500)",
      r1 is not None and abs(r1["tg_avg"] - 100.0) < 1e-9 and abs(r1["pg_avg"] - 500.0) < 1e-9, str(r1))
check("p1: rates.json written to the state dir",
      os.path.exists(os.path.join(_pstate, "rates.json")))
with open(_plog, "a", encoding="utf-8") as f:
    f.write(REQ2)
rt2 = RateTracker(state_dir=_pstate)   # a fresh process (router/dashboard restart)
r2 = rt2.sample("18590:qwen", _plog)
check("p2: restart resumes from persisted state (tg avg mean(100,200)=150, not 200)",
      r2 is not None and abs(r2["tg_avg"] - 150.0) < 1e-9, str(r2))
check("p2: restart resumes (pg avg mean(500,1000)=750, not 1000)",
      r2 is not None and abs(r2["pg_avg"] - 750.0) < 1e-9, str(r2))
check("p2: request counts carried across the restart (pg_n 2, tg_n 2)",
      r2 is not None and r2["pg_n"] == 2 and r2["tg_n"] == 2, str(r2))

print("== persistence: log truncation keeps the accumulated averages ==")
_tdir = _tf.mkdtemp(prefix="llmrouter_rate_trunc_")
_tlog = os.path.join(_tdir, "qwen.log")
_tstate = os.path.join(_tdir, "state")
open(_tlog, "w", encoding="utf-8").write(REQ1)
rtA = RateTracker(state_dir=_tstate)
rA = rtA.sample("18590:qwen", _tlog)
rtA.flush()
open(_tlog, "w", encoding="utf-8").close()   # log truncated to empty
rtB = RateTracker(state_dir=_tstate)
rB = rtB.sample("18590:qwen", _tlog)
check("t: truncation does not reset the averages (still tg 100, pg 500)",
      rB is not None and abs(rB["tg_avg"] - 100.0) < 1e-9 and abs(rB["pg_avg"] - 500.0) < 1e-9, str(rB))

print("== persistence: a stopped model keeps its averages (merge, never delete) ==")
_kdir = _tf.mkdtemp(prefix="llmrouter_rate_keep_")
_klog = os.path.join(_kdir, "qwen.log")
_kstate = os.path.join(_kdir, "state")
open(_klog, "w", encoding="utf-8").write(REQ1)
rtK = RateTracker(state_dir=_kstate)
rtK.sample("18590:qwen", _klog)   # running -> captured
rtK.flush()
rtK.forget_stale([])              # the model stops -> dropped from the in-memory table
rtK.flush()                       # merge: nothing new, prior state stays on disk
rtK2 = RateTracker(state_dir=_kstate)   # restart
rK = rtK2.sample("18590:qwen", _klog)
check("k: stopped-then-restarted model resumes its averages (tg 100, pg 500)",
      rK is not None and abs(rK["tg_avg"] - 100.0) < 1e-9 and abs(rK["pg_avg"] - 500.0) < 1e-9, str(rK))

print("== persistence: a model never started via the router has no rates (the one allowed gap) ==")
_ndir = _tf.mkdtemp(prefix="llmrouter_rate_never_")
_nlog = os.path.join(_ndir, "newmodel.log")
_nstate = os.path.join(_ndir, "state")
open(_nlog, "w", encoding="utf-8").close()   # empty, never tracked
rtN = RateTracker(state_dir=_nstate)
rN = rtN.sample("18590:newmodel", _nlog)
check("n: never-started model -> None (no fabricated rates)", rN is None, str(rN))

print()
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("RESULT: ALL RATE-PARSER CHECKS PASSED")
sys.exit(0)
