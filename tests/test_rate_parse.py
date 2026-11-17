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
from llm_router.dashboard import parse_rate_line, RateTracker  # noqa: E402

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

print()
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("RESULT: ALL RATE-PARSER CHECKS PASSED")
sys.exit(0)
