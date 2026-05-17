#!/bin/sh
# Kill existing app process, supervisor keeps container alive
if [ -f /tmp/app.pid ]; then
    kill $(cat /tmp/app.pid) 2>/dev/null
    sleep 1
fi
# Start fresh app
pnpm storybook -- --host 0.0.0.0 --no-open --disable-telemetry &
echo $! > /tmp/app.pid
