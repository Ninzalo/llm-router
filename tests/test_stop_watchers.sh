#!/usr/bin/env bash
# Regression: `llm-router stop` must stop running `watch` / `watch --llm`
# tailers (e.g. the ones you start in other terminal tabs), not just the router.
#
# Before the fix, `stop` only scanned for router `serve`/`run` processes, so a
# foreground `watch` kept tailing the log after the router was stopped. This
# test boots a router, starts a plain `watch` and a `watch --llm` in the
# background (other-terminal style, both scoped to the test port), calls
# `llm-router stop`, and asserts BOTH watcher processes are gone.
# Exit 0 = pass, 1 = a watcher (or the router) survived, 2 = env problem.
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"      # repo root (tests/..)
cd "$SCRIPT_DIR"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
export PATH="$(dirname "$PY"):$PATH"      # so bare `llm-router` + `python3` resolve

WORK="$(mktemp -d /tmp/llm-stop-watchers.XXXXXX)"
CFG="$WORK/config.yaml"
LOGDIR="$WORK/logs"
ROUTER_PORT=18789
MOCK_PORT=18091

WATCH_PID=""
WATCH_LLM_PID=""

health() {
  "$PY" -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$ROUTER_PORT/health', timeout=2); sys.exit(0)" 2>/dev/null
}

cleanup() {
  # Kill any watcher we started, in case stop failed to do so (no stray tails).
  [ -n "$WATCH_PID" ] && kill -9 "$WATCH_PID" 2>/dev/null || true
  [ -n "$WATCH_LLM_PID" ] && kill -9 "$WATCH_LLM_PID" 2>/dev/null || true
  llm-router stop --port "$ROUTER_PORT" >/dev/null 2>&1 || true
  pkill -f "mock_llm.py --host 127.0.0.1 --port $MOCK_PORT" >/dev/null 2>&1 || true
  rm -rf "$WORK" 2>/dev/null || true
}
trap cleanup EXIT

if lsof -i :"$ROUTER_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "FATAL: router port $ROUTER_PORT busy"; exit 2
fi
if lsof -i :"$MOCK_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "FATAL: mock port $MOCK_PORT busy"; exit 2
fi

MOCK_CMD="$PY $PWD/mock_llm.py --host 127.0.0.1 --port $MOCK_PORT --model m1"
cat > "$CFG" <<EOF
router:
  host: 127.0.0.1
  port: $ROUTER_PORT
  api_key: ""
  default_model: m1
  startup_timeout: 30
  health_interval: 0.2
  log_dir: $LOGDIR
models:
  - name: m1
    upstream_model: m1
    context_length: 131072
    command: $MOCK_CMD
EOF

# Boot WITHOUT preload so the test stays fast and model-independent: the watchers
# only need the router's /health + /admin/active endpoints up.
llm-router run --config "$CFG" --no-preload >/dev/null || { echo "FATAL: router failed to start"; exit 2; }

for i in $(seq 1 100); do
  if health; then break; fi
  sleep 0.3
done
if ! health; then
  echo "FATAL: router did not come up on $ROUTER_PORT"; exit 2
fi

# Start a plain `watch` and a `watch --llm` in the background (other-terminal style).
# Each is scoped to the test port so `stop --port` targets exactly these watchers.
"$PY" -m llm_router watch --port "$ROUTER_PORT" >/dev/null 2>&1 &
WATCH_PID=$!
"$PY" -m llm_router watch --llm --port "$ROUTER_PORT" >/dev/null 2>&1 &
WATCH_LLM_PID=$!
sleep 1.5

if ! kill -0 "$WATCH_PID" 2>/dev/null; then
  echo "FATAL: plain watch (pid $WATCH_PID) did not stay running before stop"; exit 2
fi
if ! kill -0 "$WATCH_LLM_PID" 2>/dev/null; then
  echo "FATAL: watch --llm (pid $WATCH_LLM_PID) did not stay running before stop"; exit 2
fi
echo "  watchers running before stop: plain=$WATCH_PID  llm=$WATCH_LLM_PID"

echo "  -> llm-router stop --port $ROUTER_PORT"
llm-router stop --port "$ROUTER_PORT"
# Give the stop time to propagate (a fixed 1.5s sleep is fragile on slow/loaded
# CI runners): poll until both watchers are gone, up to ~10s, then judge below.
for _ in $(seq 1 20); do
  if ! kill -0 "$WATCH_PID" 2>/dev/null && ! kill -0 "$WATCH_LLM_PID" 2>/dev/null; then break; fi
  sleep 0.5
done

FAIL=0
if kill -0 "$WATCH_PID" 2>/dev/null; then
  echo "FAIL: plain watch (pid $WATCH_PID) still alive after stop -> NOT STOPPED (regression!)"; FAIL=1
else
  echo "  plain watch stopped by llm-router stop"
fi
if kill -0 "$WATCH_LLM_PID" 2>/dev/null; then
  echo "FAIL: watch --llm (pid $WATCH_LLM_PID) still alive after stop -> NOT STOPPED (regression!)"; FAIL=1
else
  echo "  watch --llm stopped by llm-router stop"
fi
if health; then
  echo "FAIL: router still reachable on $ROUTER_PORT after stop"; FAIL=1
else
  echo "  router down after stop"
fi

if [ "$FAIL" -ne 0 ]; then
  echo "FAIL: stop did not stop the watchers"; exit 1
fi
echo "PASS: stop stopped the router + both the watch and watch --llm tailers"
exit 0
