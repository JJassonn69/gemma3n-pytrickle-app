#!/usr/bin/env bash
set -euo pipefail

# This script starts the vLLM OpenAI server in the background
# and then runs the app that connects to it via the OpenAI-compatible API.
# Any arguments to this script are passed through to vLLM's entrypoint.

# Resolve Python binary (prefer python3, fallback to python)
PYBIN=$(command -v python3 || true)
if [ -z "${PYBIN}" ]; then
  PYBIN=$(command -v python || true)
fi
if [ -z "${PYBIN}" ]; then
  echo "No python interpreter found in PATH." >&2
  exit 1
fi

# Respect an optional VLLM_CMD override (for advanced/custom runs)
VLLM_CMD=${VLLM_CMD:-"${PYBIN} -m vllm.entrypoints.openai.api_server"}

# Default vLLM args unless provided by the user/compose CMD.
# You can override these by providing arguments to the container run command.
DEFAULT_ARGS=(
  --model "${MODEL_ID:-google/gemma-3n-E4B-it}"
  --dtype bfloat16
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}"
  --max-model-len "${MAX_MODEL_LEN:-32768}"
  --port "${VLLM_PORT:-9000}"
)

# Prefer user-supplied args if provided, else use defaults
if [ "$#" -gt 0 ]; then
  VLLM_ARGS=("$@")
else
  VLLM_ARGS=("${DEFAULT_ARGS[@]}")
fi

# Set cache dirs
export HF_HOME=${HF_HOME:-/data/hf}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export VLLM_ALLOW_HTTP=${VLLM_ALLOW_HTTP:-1}

# Start vLLM server in background
set -x
$VLLM_CMD "${VLLM_ARGS[@]}" &
VLLM_PID=$!
set +x

# Wait until the API is reachable (default up to ~15 min, configurable)
BASE_URL=${VLLM_BASE_URL:-http://127.0.0.1:9000/v1}
WAIT_INTERVAL=${WAIT_INTERVAL:-5}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-900} # seconds
echo "Waiting for vLLM to be ready at ${BASE_URL} (timeout ${WAIT_TIMEOUT}s, interval ${WAIT_INTERVAL}s) ..."
start_ts=$(date +%s)
while true; do
  if curl -sf "${BASE_URL}/models" > /dev/null; then
    echo "vLLM is up."
    break
  fi
  sleep "${WAIT_INTERVAL}"
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM process exited unexpectedly." >&2
    exit 1
  fi
  now=$(date +%s)
  if [ $((now - start_ts)) -ge ${WAIT_TIMEOUT} ]; then
    echo "Still waiting for vLLM after ${WAIT_TIMEOUT}s; extending wait..." >&2
    # Extend timeout once more by the same amount to avoid hard failure on slow warm-up
    start_ts=$(date +%s)
  fi
done

# Run the app (PyTrickle server) connecting to local vLLM
# Ensure the app uses the local base URL by default
export USE_VLLM=${USE_VLLM:-1}
export MODEL_ID=${MODEL_ID:-google/gemma-3n-E4B-it}
export VLLM_BASE_URL=${BASE_URL}

# Optional: pass-through a different port for the app
APP_PORT=${APP_PORT:-8000}
export PYTHONUNBUFFERED=1

echo "Starting app on port ${APP_PORT} ..."
exec "${PYBIN}" /app/app.py
