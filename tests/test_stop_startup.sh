#!/usr/bin/env bash
# Regression: `llm-router stop` must stop the model even when it is mid-startup.
#
# Before the fix (app.py caught only `Exception`, not asyncio.CancelledError),
# cancelling the in-flight preload task during shutdown let the CancelledError
# escape the try/except (it's a BaseException), so `mgr.shutdown()` was skipped
# and an already-spawned model process was left running (orphaned). This test
# reproduces that: it starts a router whose default model has a long load
# (--delay), calls `stop` while the model is still loading, and asserts the
# model process is gone. Exit 0 = pass, 1 = orphaned, 2 = env problem.
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"      # repo root (tests/..)
cd "$SCRIPT_DIR"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
export PATH="$(dirname "$PY"):$PATH"      # so bare `llm-router` + `python3` resolve

WORK="$(mktemp -d /tmp/llm-stop-startup.XXXXXX)"
CFG="$WORK/config.yaml"
LOGDIR="$WORK/logs"
ROUTER_PORT=18788
MOCK_PORT=18090
DELAY=8
MOCK_CMD="$PY $PWD/mock_llm.py --host 127.0.0.1 --port $MOCK_PORT --model slow"

cleanup() {
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

cat > "$CFG" <<EOF
router:
  host: 127.0.0.1
  port: $ROUTER_PORT
  api_key: ""
  default_model: slow
  startup_timeout: 30
  health_interval: 0.2
  log_dir: $LOGDIR
models:
  - name: slow
    upstream_model: slow
    context_length: 131072
    default: true
    command: $MOCK_CMD --delay $DELAY
EOF

llm-router run --config "$CFG" >/dev/null || { echo "FATAL: router failed to start"; exit 2; }
# Poll for the model process instead of a fixed sleep: preload spawns the model
# shortly after start, but a fixed 3s is fragile on slow/loaded CI runners.
MPID=""
for _ in $(seq 1 30); do
  MPID="$(pgrep -f "mock_llm.py --host 127.0.0.1 --port $MOCK_PORT" | head -1 || true)"
  if [ -n "$MPID" ] && kill -0 "$MPID" 2>/dev/null; then break; fi
  sleep 0.5
done
if [ -z "$MPID" ] || ! kill -0 "$MPID" 2>/dev/null; then
  echo "FAIL: could not confirm the model was in startup state (pid=${MPID:-none})"
  exit 1
fi
echo "  model in startup state (pid $MPID, still loading)"

llm-router stop --port "$ROUTER_PORT" >/dev/null
sleep 1.5

if kill -0 "$MPID" 2>/dev/null; then
  echo "FAIL: model (pid $MPID) still alive after stop -> ORPHANED (regression!)"
  exit 1
fi
echo "PASS: model stopped on graceful shutdown (no orphan during startup)"
exit 0
