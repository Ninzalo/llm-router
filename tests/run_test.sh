#!/usr/bin/env bash
# Boot the ISOLATED test router (mocks on 18xxx; never touches real :8080/:8081)
# using the daemon `llm-router run`, run the full 27-check suite, then stop it.
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"      # repo root (tests/..)
cd "$SCRIPT_DIR"
# Interpreter: prefer the repo venv, else a system python3. Prepend its bin dir
# to PATH so the config's `python3 mock_llm.py` model commands use the same
# interpreter (the one that has fastapi/uvicorn).
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
export PATH="$(dirname "$PY"):$PATH"
PORT=18787
DLOG="$HOME/.config/llm-router/router-$PORT.log"

kill_port() {
  local pids
  pids="$(lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | tr '\n' ' ')"
  [ -n "${pids// /}" ] && kill -TERM $pids 2>/dev/null
}

echo "[1/4] cleaning 18xxx ports + stale daemon log..."
for p in $PORT 18080 18081 18082 18083; do kill_port "$p"; done
rm -f "$HOME/.config/llm-router/router-$PORT.log"
rm -rf logs/unsloth "logs/unsloth-Qwen3.8-27B-GGUF-UD-Q4_K_M.log"   # stale HF-name test artifacts
sleep 1

echo "[2/4] starting test router in the background (daemon)..."
"$PY" -m llm_router run --config config.test.yaml --host 127.0.0.1 --port "$PORT"
if [ $? -ne 0 ]; then echo "start failed"; exit 1; fi

echo "[3/4] waiting for router..."
for i in $(seq 1 60); do
  if curl -sf --max-time 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
    curl -s --max-time 2 http://127.0.0.1:$PORT/health | sed 's/^/    /'; echo; break
  fi
  sleep 0.5
  if [ "$i" -eq 60 ]; then
    echo "router did not come up; tail of daemon log:"; tail -20 "$DLOG" 2>/dev/null
    "$PY" -m llm_router stop --port "$PORT"; exit 1
  fi
done

echo "[4/4] running test_suite.py ..."
"$PY" test_suite.py
RC=$?

echo
echo "stopping test router (daemon stop)..."
"$PY" -m llm_router stop --port "$PORT"
sleep 1
for p in $PORT 18080 18081 18082 18083; do kill_port "$p"; done
sleep 1

echo
if [ "$RC" -eq 0 ]; then
  echo "RESULT: PASS (27/27). Real :8080/:8081 were not touched."
else
  echo "RESULT: FAIL (rc=$RC). See $DLOG"
fi
exit "$RC"
