#!/bin/sh
# Supervisor: app server is NOT PID1. Container survives app crashes.
# restart_app.sh is called by the agent after patch+rebuild.

APP_PID=""

start_app() {
    pnpm run dev &
    APP_PID=$!
    echo "$APP_PID" > /tmp/app.pid
}

stop_app() {
    if [ -n "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; then
        kill "$APP_PID"
        wait "$APP_PID" 2>/dev/null
    fi
}

# Trap SIGTERM for clean shutdown
trap 'stop_app; exit 0' TERM INT

start_app

# Wait forever (supervisor keeps container alive)
while true; do
    wait "$APP_PID" 2>/dev/null
    # If app died, keep container alive for diagnostics
    sleep 1
done
