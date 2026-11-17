#!/usr/bin/env python
"""Unit tests for the dashboard pp/tg rate parser + RateTracker. Pure — no server,
no ports, no real models, so it runs anywhere (local + CI).

Covers BOTH real log formats the router's models emit:
  * splash serve  "Done · ... input X · cached Y · ... TTFT Ss · R tok/s ..."
                  (the format this project's models actually log)
  * llama.cpp     "prompt eval time = ... ( T tokens/sec)" / "eval time = ..."

And the RateTracker behaviour that must hold:
  * pp = prompt-processing speed, tg = token-generation speed
  * the AVERAGES survive model/router/dashboard restarts and a model switch
    (a stopped model keeps its average even though it shows no current rate);
    only a model never started via the router has empty rates
  * the per-model window is capped at 20 samples and the current rates are
    never persisted to rates.json

Run:  python tests/test_rate_parse.py    (exit 0 == all pass)
"""
import os
import sys
import json
import tempfile as _tf

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


def near(a, b, tol=0.0051):
    """Approx-equal with a tolerance that covers the tracker's 2-dp rounding."""
    return a is not None and abs(a - b) <= tol


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


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

COLD_PP = 32226 / 54.1
WARM_PP = (36099 - 35712) / 1.1

print("== splash serve 'Done' lines (the real format) ==")
d = as_dict(parse_rate_line(SPLASH_COLD))
check("cold Done -> gen + prompt rates", set(d) == {"gen", "prompt"}, str(d))
check("cold gen (tg) == 46.2", d.get("gen") == 46.2, str(d))
check("cold prompt (pp) == (32,226-0)/54.1", near(d.get("prompt"), COLD_PP, 1e-6), str(d))

d = as_dict(parse_rate_line(SPLASH_WARM))
check("warm gen (tg) == 73.2", d.get("gen") == 73.2, str(d))
check("warm prompt (pp) == (36,099-35,712)/1.1", near(d.get("prompt"), WARM_PP, 1e-6), str(d))

d = as_dict(parse_rate_line(SPLASH_ALL_CACHED))
check("all-cached Done -> gen only (no prompt rate)", set(d) == {"gen"} and d.get("gen") == 50.0, str(d))

print("== non-timing lines yield None ==")
for ln in NOT_TIMING:
    check(f"-> None: {ln[:42]!r}...", parse_rate_line(ln) is None, str(parse_rate_line(ln)))

print("== llama.cpp lines (legacy / other backends) still parse ==")
d = as_dict(parse_rate_line(LLAMA_PROMPT))
check("llama prompt (pp) == 25.44", set(d) == {"prompt"} and d.get("prompt") == 25.44, str(d))
d = as_dict(parse_rate_line(LLAMA_GEN))
check("llama gen (tg) == 94.97", set(d) == {"gen"} and d.get("gen") == 94.97, str(d))
d = as_dict(parse_rate_line(LLAMA_LEGACY_TS))
check("llama legacy 't/s' prompt == 382.87", set(d) == {"prompt"} and d.get("prompt") == 382.87, str(d))

print("== RateTracker: window + current accumulate across requests ==")
rt = RateTracker()
key = "/tmp/test_rate_parse.log"
rt._st[key] = rt._fresh(key, 0)  # seed the entry (as _tail() does on first sight)
for ln in (SPLASH_COLD, SPLASH_WARM):
    rt._absorb(key, ln)
st = rt._st[key]
snap = rt._snapshot(st)
check("2 requests kept as [pp,tg] samples", len(st["samples"]) == 2, str(st))
check("cur_tg == 73.2 (latest request)", near(st["cur_tg"], 73.2, 1e-9), str(st["cur_tg"]))
check("cur_pp == warm pp (latest request)", near(st["cur_pp"], WARM_PP, 1e-6), str(st["cur_pp"]))
check("tg_avg == mean(46.2, 73.2) == 59.7", near(snap["tg_avg"], (46.2 + 73.2) / 2), str(snap))
check("pp_avg == mean of the two prompt rates", near(snap["pp_avg"], (COLD_PP + WARM_PP) / 2), str(snap))
check("current != average (jitter preserved)", st["cur_tg"] != snap["tg_avg"], str(st))

print("== instance (router) rate: current from active model, average pooled across all models ==")
# Two models with distinct, known rates. The instance "current" pp/tg must come
# from the ACTIVE model, while the instance AVERAGE pp/tg is pooled across ALL
# models of the instance that have produced rates (not just the active one).
#   alpha: input 1,000 cached 0, TTFT 2.0s, 100.0 tok/s  -> pp=(1000-0)/2.0=500, tg=100
#   beta : input 1,000 cached 0, TTFT 1.0s, 200.0 tok/s  -> pp=(1000-0)/1.0=1000, tg=200
_tmp = _tf.mkdtemp(prefix="llmrouter_router_rate_")
a_path = os.path.join(_tmp, "alpha.log")
b_path = os.path.join(_tmp, "beta.log")
open(a_path, "w", encoding="utf-8").write("13:00:00 Done \u00b7 request 1 \u00b7 input 1,000 \u00b7 cached 0 \u00b7 output 10 \u00b7 tools 0 \u00b7 TTFT 2.0s \u00b7 100.0 tok/s \u00b7 10.0 tok/cycle\n")
open(b_path, "w", encoding="utf-8").write("13:00:05 Done \u00b7 request 1 \u00b7 input 1,000 \u00b7 cached 0 \u00b7 output 10 \u00b7 tools 0 \u00b7 TTFT 1.0s \u00b7 200.0 tok/s \u00b7 20.0 tok/cycle\n")
rt2 = RateTracker()
ka, kb = "18590:alpha", "18590:beta"
a_rate = rt2.sample(ka, a_path)   # seeds + reads alpha.log (the active model)
b_rate = rt2.sample(kb, b_path)   # seeds + reads beta.log (a second running model)
check("alpha parsed (active): tg_avg 100, pp_avg 500", a_rate is not None and near(a_rate["tg_avg"], 100.0) and near(a_rate["pp_avg"], 500.0), str(a_rate))
check("beta parsed (2nd model): tg_avg 200, pp_avg 1000", b_rate is not None and near(b_rate["tg_avg"], 200.0) and near(b_rate["pp_avg"], 1000.0), str(b_rate))
router_raw = _router_rate(a_rate, rt2, [ka, kb])
check("router object built", router_raw is not None, str(router_raw))
router = router_raw or {}
check("router CURRENT tg == active model tg (100.0)", router.get("tg_current") == 100.0, str(router))
check("router CURRENT pp == active model pp (500.0)", router.get("pp_current") == 500.0, str(router))
check("router AVG tg pooled == mean(100,200)=150.0 (NOT active's 100)", near(router.get("tg_avg"), 150.0), str(router))
check("router AVG pp pooled == mean(500,1000)=750.0 (NOT active's 500)", near(router.get("pp_avg"), 750.0), str(router))
check("router AVG pulls in the other model (differs from active-only AVG)",
      router.get("tg_avg") != (a_rate or {}).get("tg_avg") and router.get("pp_avg") != (a_rate or {}).get("pp_avg"),
      f"router={router} active={a_rate}")
single = _router_rate(b_rate, rt2, [kb]) or {}
check("single model -> average equals that model's own (no other model)",
      near(single.get("tg_avg"), 200.0) and near(single.get("pp_avg"), 1000.0), str(single))
# no active-model rate, but pooled windows exist -> the AVERAGE still surfaces
# (current is empty) so a model switch / newly-active model with no request yet
# never erases the instance average.
noact = _router_rate(None, rt2, [ka, kb]) or {}
check("no active-model rate but pooled window -> avg survives (current empty)",
      noact.get("tg_current") is None and noact.get("pp_current") is None
      and near(noact.get("tg_avg"), 150.0) and near(noact.get("pp_avg"), 750.0),
      str(noact))
# no active-model rate AND no window anywhere -> None
_rt_empty = RateTracker()
check("no active-model rate and no window -> None",
      _router_rate(None, _rt_empty, ["18590:ghost"]) is None)

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
check("p1: first request captured (tg 100, pp 500)", r1 is not None and near(r1["tg_avg"], 100.0) and near(r1["pp_avg"], 500.0), str(r1))
check("p1: rates.json written to the state dir", os.path.exists(os.path.join(_pstate, "rates.json")))
check("p1: the stored entry holds only the window + tail offset (no current, no sums)",
      set(read_json(os.path.join(_pstate, "rates.json"))[os.path.realpath(_plog)].keys()) == {"pos", "samples"},
      str(read_json(os.path.join(_pstate, "rates.json"))))
with open(_plog, "a", encoding="utf-8") as f:
    f.write(REQ2)
rt2 = RateTracker(state_dir=_pstate)   # a fresh process (router/dashboard restart)
r2 = rt2.sample("18590:qwen", _plog)
check("p2: restart resumes from persisted window (tg avg mean(100,200)=150, not 200)", near(r2 and r2["tg_avg"], 150.0), str(r2))
check("p2: restart resumes (pp avg mean(500,1000)=750, not 1000)", near(r2 and r2["pp_avg"], 750.0), str(r2))
check("p2: request counts carried across the restart (pp_n 2, tg_n 2)", r2 is not None and r2["pp_n"] == 2 and r2["tg_n"] == 2, str(r2))

print("== persistence: a STOPPED model keeps its averages (the core v1.2.7 fix) ==")
_kdir = _tf.mkdtemp(prefix="llmrouter_rate_keep_")
_klog = os.path.join(_kdir, "qwen.log")
_kstate = os.path.join(_kdir, "state")
open(_klog, "w", encoding="utf-8").write(REQ1)
rtK = RateTracker(state_dir=_kstate)
rtK.sample("18590:qwen", _klog)   # running -> captured
rtK.flush()
# The model STOPS (switched away). avg_snapshot must keep the average, empty current.
rKstop = rtK.avg_snapshot("18590:qwen", _klog)
check("k1: stopped model keeps its average with EMPTY current",
      rKstop is not None and near(rKstop["tg_avg"], 100.0) and near(rKstop["pp_avg"], 500.0)
      and rKstop["tg_current"] is None and rKstop["pp_current"] is None, str(rKstop))
# Full restart: the stopped model's averages must STILL survive (item 1).
rtK2 = RateTracker(state_dir=_kstate)
rK = rtK2.avg_snapshot("18590:qwen", _klog)
check("k2: after a restart, the stopped model's averages survive (tg 100, pp 500, empty current)",
      rK is not None and near(rK["tg_avg"], 100.0) and near(rK["pp_avg"], 500.0)
      and rK["tg_current"] is None and rK["pp_current"] is None, str(rK))
# A model that is running again gets a live current, and its average is extended
# (not reset) by the new request.
with open(_klog, "a", encoding="utf-8") as f:
    f.write(REQ2)   # a fresh request now that the model is back (pp 1000, tg 200)
rKrun = rtK2.sample("18590:qwen", _klog)
check("k3: started again -> live current + surviving/extended average (no reset)",
      rKrun is not None and rKrun["tg_current"] == 200.0 and near(rKrun["tg_avg"], 150.0) and near(rKrun["pp_avg"], 750.0), str(rKrun))

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
check("t: truncation does not reset the averages (still tg 100, pp 500)",
      rB is not None and near(rB["tg_avg"], 100.0) and near(rB["pp_avg"], 500.0), str(rB))

print("== cap: window is limited to 20 samples; current is never persisted ==")
_cdir = _tf.mkdtemp(prefix="llmrouter_rate_cap_")
_clog = os.path.join(_cdir, "qwen.log")
_cstate = os.path.join(_cdir, "state")
with open(_clog, "w", encoding="utf-8") as f:
    for i in range(1, 26):  # 25 requests: pp = i*100, tg = i*2
        f.write(f"13:00:{i:02d} Done \u00b7 request {i} \u00b7 input {i*1000:,} \u00b7 cached 0 \u00b7 output 10 \u00b7 tools 0 \u00b7 TTFT 10.0s \u00b7 {i*2} tok/s \u00b7 1.0 tok/cycle\n")
rtC = RateTracker(state_dir=_cstate)
c = rtC.sample("18590:qwen", _clog)
rtC.flush()
raw = read_json(os.path.join(_cstate, "rates.json"))
gk = [k for k in raw if os.path.basename(k) == "qwen.log"][0]
stored = raw[gk]
check("c1: only the last 20 samples are stored (not all 25)", len(stored.get("samples", [])) == 20, str(len(stored.get("samples", []))))
check("c2: current pp/tg are NOT persisted", "cur_pp" not in stored and "cur_tg" not in stored, str(sorted(stored.keys())))
check("c3: pp_avg == mean of the LAST 20 (i=6..25 -> mean 1550)", c is not None and near(c["pp_avg"], 1550.0), str(c))
check("c4: tg_avg == mean of the LAST 20 (i=6..25 -> mean 31)", c is not None and near(c["tg_avg"], 31.0), str(c))

print("== migration: a pre-1.2.7 sums-based rates.json is re-seeded from the log ==")
_mdir = _tf.mkdtemp(prefix="llmrouter_rate_mig_")
_mlog = os.path.join(_mdir, "qwen.log")
_mstate = os.path.join(_mdir, "state")
os.makedirs(_mstate, exist_ok=True)
with open(_mlog, "w", encoding="utf-8") as f:
    for i in range(1, 11):  # 10 requests: pp = i*100, tg = i*2
        f.write(f"13:00:{i:02d} Done \u00b7 request {i} \u00b7 input {i*1000:,} \u00b7 cached 0 \u00b7 output 10 \u00b7 tools 0 \u00b7 TTFT 10.0s \u00b7 {i*2} tok/s \u00b7 1.0 tok/cycle\n")
realM = os.path.realpath(_mlog)
# old schema: running sums (no "samples")
with open(os.path.join(_mstate, "rates.json"), "w", encoding="utf-8") as f:
    json.dump({realM: {"pos": 0, "pg_sum": 999.0, "pg_n": 99, "tg_sum": 888.0, "tg_n": 88, "cur_pg": 1.0, "cur_tg": 2.0}}, f)
rtM = RateTracker(state_dir=_mstate)
d = rtM.avg_snapshot("18590:qwen", _mlog)   # stopped model on the upgrade run
check("m1: old-format stopped model re-seeded from the log (pp avg mean(100..1000)=550)",
      d is not None and near(d["pp_avg"], 550.0) and near(d["tg_avg"], 11.0) and d["pp_current"] is None, str(d))
rtM.flush()
migrated = read_json(os.path.join(_mstate, "rates.json"))[realM]
check("m2: entry is rewritten to the new window schema (samples, no sums/current)",
      "samples" in migrated and "pg_sum" not in migrated and "cur_pg" not in migrated and len(migrated["samples"]) == 10,
      str(sorted(migrated.keys())))

print("== a model never started via the router has no rates (the one allowed gap) ==")
_ndir = _tf.mkdtemp(prefix="llmrouter_rate_never_")
_nlog = os.path.join(_ndir, "newmodel.log")
_nstate = os.path.join(_ndir, "state")
open(_nlog, "w", encoding="utf-8").close()   # empty, never tracked
rtN = RateTracker(state_dir=_nstate)
check("n1: never-started model (sample) -> None (no fabricated rates)", rtN.sample("18590:newmodel", _nlog) is None)
check("n2: never-started model (avg_snapshot) -> None (no fabricated rates)", rtN.avg_snapshot("18590:newmodel", _nlog) is None)

print()
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("RESULT: ALL RATE-PARSER CHECKS PASSED")
sys.exit(0)
