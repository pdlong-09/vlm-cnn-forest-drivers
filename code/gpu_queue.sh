#!/bin/bash
# GPU queue on server1 for the GPU batches of this project, shared infrastructure, no research code.
#
#   gpu_queue.sh GPU WORKDIR CMD [ARGS...]
#
# Our jobs for one GPU run one at a time, in the order they were queued (a ticket per job in $Q/gpu$GPU, oldest first;
# tickets of dead processes are removed). At its turn a job waits until the GPU has < $FREE_MIB MiB in use for
# $STABLE_S seconds in a row (so it does not slip into the gap between two stages of someone else's job), checking every
# $POLL_S seconds, then runs CMD in WORKDIR with the GPU still visible as itself (pipelines take --gpus GPU).
# 24/10/2026: two of our pipelines took GPU 0 in the same gap between stages and one of them died.
set -u
GPU=$1; WORKDIR=$2; shift 2
Q=/data/tmp/llm4deforest_gpuq/gpu$GPU
FREE_MIB=${FREE_MIB:-2000}; STABLE_S=${STABLE_S:-30}; POLL_S=${POLL_S:-10}
LOG=/data/tmp/llm4deforest_gpuq/log.txt
mkdir -p "$Q"
ticket="$(date +%s%N).$$"
touch "$Q/$ticket"
trap 'rm -f "$Q/$ticket"' EXIT
say() { echo "$(date '+%F %T') gpu$GPU [$ticket] $*" >> "$LOG"; echo "[gpu_queue] $*"; }
say "queued: $WORKDIR $*"
while true; do
  [ -e "$Q/$ticket" ] || { mkdir -p "$Q"; touch "$Q/$ticket"; }  # queue directory removed under us: queue again
  head=$(ls "$Q" | sort | head -1)
  [ "$head" = "$ticket" ] && break
  pid=${head##*.}
  if ! kill -0 "$pid" 2>/dev/null; then rm -f "$Q/$head"; say "removed stale ticket $head"; continue; fi
  sleep "$POLL_S"
done
say "my turn; waiting for GPU $GPU below $FREE_MIB MiB for $STABLE_S s"
free_since=""
while true; do
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  now=$(date +%s)
  if [ -n "$used" ] && [ "$used" -lt "$FREE_MIB" ]; then  # no reading (nvidia-smi error) counts as busy
    [ -z "$free_since" ] && free_since=$now
    [ $((now - free_since)) -ge "$STABLE_S" ] && break
  else
    free_since=""
  fi
  sleep "$POLL_S"
done
say "GPU $GPU free ($used MiB); starting"
cd "$WORKDIR" && "$@"
code=$?
say "finished with exit $code"
exit $code
