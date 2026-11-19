#!/usr/bin/env bash
# Run a command only when no other process has used the GPU for 5 minutes in a row, so the GPU is not shared
# with another job (checked once a minute). Usage: src/run_when_gpu_free.sh <command> [args...]
set -u
idle=0
while [ "$idle" -lt 5 ]; do
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then idle=$((idle + 1)); else idle=0; fi
  [ "$idle" -lt 5 ] && sleep 60
done
echo "$(date '+%F %T') GPU idle for 5 min, starting: $*"
exec "$@"
