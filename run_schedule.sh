#!/bin/sh
# Runs the autopilot at fixed UTC hours (default 06,12,18,23 = 4x/day), forever.
HOURS="${RUN_HOURS:-06 12 18 23}"
echo "[scheduler] Will run at UTC hours: $HOURS"
# Run once at boot so you can verify the setup immediately.
python /app/autopilot.py || true
while true; do
  NOW=$(date -u +%H)
  for H in $HOURS; do
    if [ "$NOW" = "$H" ]; then
      python /app/autopilot.py || echo "[scheduler] run failed, continuing"
      sleep 3600  # don't double-fire within the same hour
    fi
  done
  sleep 300
done
