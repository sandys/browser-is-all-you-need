#!/bin/bash
# test.sh - Harbor verifier for browser tasks
#
# Contract:
#   - ALWAYS writes /logs/verifier/reward.json (flat numeric dict)
#   - ALWAYS writes /logs/verifier/diagnostics.json (strings for debugging)
#   - NEVER writes reward.txt (avoids Harbor precedence issues)
#   - Exit 0 on all evaluated outcomes (even failures)
#   - Exit non-zero only on truly unrecoverable infra errors
#
# Terminal states:
#   ready_for_chromiumrl   - build/serve succeeded; worker should call ChromiumRL
#   build_failed           - App build failed
#   serve_failed           - App serve failed to start
#   preview_url_missing    - environment failed to provide a preview URL

set +e  # Don't exit on error - we catch everything and write reward.json

TESTBED_ROOT="${TESTBED_ROOT:-/testbed}"
SERVE_PORT="${SERVE_PORT:-6006}"
SERVE_PID_PATH="${SERVE_PID_PATH:-/tmp/serve.pid}"
SERVE_LOG_PATH="${SERVE_LOG_PATH:-/tmp/serve.log}"
VERIFIER_DIR="${VERIFIER_DIR:-/logs/verifier}"
PREVIEW_URL="${PREVIEW_URL:-}"
PREVIEW_URL_FILE="${PREVIEW_URL_FILE:-/trial/preview_url}"
PREVIEW_PATH="${PREVIEW_PATH:-/iframe.html?id=typography-heading--basic&viewMode=story}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-60}"
SLOT_SAFE_SERVE_CMD="${SLOT_SAFE_SERVE_CMD:-}"
READINESS_MODE="${READINESS_MODE:-http}"
READY_LOG_PATTERN="${READY_LOG_PATTERN:-}"

mkdir -p "$VERIFIER_DIR"

REWARD_FILE="$VERIFIER_DIR/reward.json"
DIAG_FILE="$VERIFIER_DIR/diagnostics.json"
RESULT_WRITTEN=0

# Backstop: ensure artifact emission even on unexpected shell failure path.
_finalize_if_missing() {
    if [ "$RESULT_WRITTEN" -eq 1 ]; then
        return
    fi
    echo '{"reward": 0.0, "chromiumrl_quality": 0.0, "domdiff_total": 0.0, "domdiff_structural": 0.0, "domdiff_text": 0.0, "domdiff_layout": 0.0, "domdiff_style": 0.0, "rubric_passed": 0, "build_ok": 0, "serve_ok": 0, "loss_mask": 0}' > "$REWARD_FILE"
    echo '{"terminal_state": "verifier_internal_error", "error_type": "missing_write_result", "error_message": "", "preview_url": ""}' > "$DIAG_FILE"
}
trap _finalize_if_missing EXIT

# Helper: write reward and diagnostics, then exit 0
write_result() {
    local reward="$1"
    local build_ok="$2"
    local serve_ok="$3"
    local loss_mask="$4"
    local terminal_state="$5"
    local error_type="$6"
    local error_message="$7"
    local preview_url="${8:-}"

    cat > "$REWARD_FILE" <<REWARD_EOF
{"reward": $reward, "chromiumrl_quality": 0.0, "domdiff_total": 0.0, "domdiff_structural": 0.0, "domdiff_text": 0.0, "domdiff_layout": 0.0, "domdiff_style": 0.0, "rubric_passed": 0, "build_ok": $build_ok, "serve_ok": $serve_ok, "loss_mask": $loss_mask}
REWARD_EOF

    # diagnostics.json - string values, use python3 for safe JSON encoding
    python3 -c "
import json, sys
d = {
    'terminal_state': sys.argv[1],
    'error_type': sys.argv[2],
    'error_message': sys.argv[3],
    'preview_url': sys.argv[4],
}
with open(sys.argv[5], 'w') as f:
    json.dump(d, f)
" "$terminal_state" "$error_type" "$error_message" "$preview_url" "$DIAG_FILE" 2>/dev/null     || echo '{"terminal_state": "unknown", "error_type": "diagnostics_write_failed", "error_message": "", "preview_url": ""}' > "$DIAG_FILE"

    RESULT_WRITTEN=1
    echo "=== Verifier result: reward=$reward terminal_state=$terminal_state ==="
    exit 0
}

cd "$TESTBED_ROOT"

# -----------------------------------------------------------------------
# Step 1: Build the app
# -----------------------------------------------------------------------
echo "=== Step 1: Building app ==="
pnpm clean:storybook

BUILD_EXIT=$?
if [ $BUILD_EXIT -ne 0 ]; then
    echo "Build failed with exit code $BUILD_EXIT"
    write_result 0.0 0 0 1 "build_failed" "build_error" "Build exited with code $BUILD_EXIT"
fi

# -----------------------------------------------------------------------
# Step 2: Start serving
# -----------------------------------------------------------------------
echo "=== Step 2: Starting serve on port $SERVE_PORT ==="
if [ -n "$SLOT_SAFE_SERVE_CMD" ]; then
  nohup bash -lc "$SLOT_SAFE_SERVE_CMD" >"$SERVE_LOG_PATH" 2>&1 &
else
  nohup bash -lc 'PORT=$SERVE_PORT CI=true pnpm storybook -- --host 0.0.0.0 --no-open --disable-telemetry' >"$SERVE_LOG_PATH" 2>&1 &
fi
SERVE_PID=$!
echo "$SERVE_PID" > "$SERVE_PID_PATH" 

# Wait for serve to be ready
echo "=== Step 3: Waiting for serve readiness ==="
SERVE_READY=0
LOG_READY_SEEN=0
for i in $(seq 1 "$READY_TIMEOUT_SEC"); do
    if [ "$READINESS_MODE" = "log_ready" ]; then
        if [ -n "$READY_LOG_PATTERN" ] && [ -f "$SERVE_LOG_PATH" ] && grep -q "$READY_LOG_PATTERN" "$SERVE_LOG_PATH"; then
            LOG_READY_SEEN=1
            if [ -z "$PREVIEW_PATH" ]; then
                SERVE_READY=1
                echo "Serve is ready after ${i}s (log_ready)"
                break
            fi
        fi
        if [ "$LOG_READY_SEEN" -eq 1 ] && curl -sf "http://localhost:${SERVE_PORT}${PREVIEW_PATH}" >/dev/null 2>&1; then
            SERVE_READY=1
            echo "Serve is ready after ${i}s (log_ready+preview_path)"
            break
        fi
    elif curl -sf "http://localhost:${SERVE_PORT}/" >/dev/null 2>&1; then
        SERVE_READY=1
        echo "Serve is ready after ${i}s"
        break
    fi
    sleep 1
done

if [ $SERVE_READY -eq 0 ]; then
    echo "Serve failed to start within $READY_TIMEOUT_SEC s"
    write_result 0.0 1 0 1 "serve_failed" "serve_timeout" "Serve not ready after $READY_TIMEOUT_SEC s"
fi

# -----------------------------------------------------------------------
# Step 4: Read preview URL
# -----------------------------------------------------------------------
echo "=== Step 4: Reading preview URL ==="
if [ -z "$PREVIEW_URL" ]; then
    if [ ! -f "$PREVIEW_URL_FILE" ]; then
        echo "No preview URL file at $PREVIEW_URL_FILE"
        write_result 0.0 1 1 0 "preview_url_missing" "no_preview_url" "Preview URL file not found"
    fi
    PREVIEW_URL=$(cat "$PREVIEW_URL_FILE")
fi
echo "Preview URL: $PREVIEW_URL"

# -----------------------------------------------------------------------
# Step 5: Emit verifier context for external ChromiumRL evaluation
# -----------------------------------------------------------------------
echo "=== Step 5: Recording preview URL for external ChromiumRL evaluation ==="
write_result 0.0 1 1 1 "ready_for_chromiumrl" "" "" "$PREVIEW_URL"
