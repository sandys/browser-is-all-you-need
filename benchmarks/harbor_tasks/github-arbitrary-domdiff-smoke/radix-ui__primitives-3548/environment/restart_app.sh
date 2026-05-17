#!/bin/sh
# Kill existing app process, supervisor keeps container alive
if [ -f /tmp/app.pid ]; then
    kill $(cat /tmp/app.pid) 2>/dev/null
    sleep 1
fi
# Start fresh app
pnpm run dev &
echo $! > /tmp/app.pid
