#!/bin/bash
# Auto power-off gpu2 after IDLE_MINUTES without activity.
#
# Activity signals:
#   0) a fresh LEASE file (explicit "hold the box awake" — deployments, benchmarks,
#      and the Laya service refreshing it on every /predict),
#   1) NEW POST lines in the Ollama log (real LLM requests; our own GET /api/ps
#      probes never count),
#   2) SSH sessions — interactive (utmp) OR any established inbound connection on
#      port 22 (utmp misses non-interactive `ssh gpu2 cmd`, which is how tmux-based
#      deployments stay invisible),
#   3) a loaded model (/api/ps),
#   4) a GPU compute process, with a *recognised* model server (llama-server, the
#      Laya service) treated as idle while it merely holds VRAM; it counts only
#      while actually busy (util>0) or with a fresh log,
#   5) active model download (fresh *.partial-* blobs),
#   6) a long install/download running outside any SSH session (pip/hf in tmux) —
#      exactly the case that got the Laya deployment powered off mid-install.
#
# Env: IDLE_MINUTES (default 30), DRY_RUN=1 logs instead of powering off,
#      IGNORE_SSH=1 disables the SSH check (testing).
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
IDLE_MINUTES="${IDLE_MINUTES:-30}"
DRY_RUN="${DRY_RUN:-0}"
IGNORE_SSH="${IGNORE_SSH:-0}"
OLLAMA_LOG=/home/dry/logs/ollama.log
LOGFILE=/home/dry/logs/idle-shutdown.log
OLLAMA_API=http://127.0.0.1:11434/api/ps
LEASEFILE="${LEASEFILE:-/home/dry/logs/activity.lease}"
LAYAY_LOG="${LAYAY_LOG:-/home/dry/logs/laya-service.log}"
SPARK_LOG="${SPARK_LOG:-/home/dry/spark-models/server.log}"
SPARK_PIDFILE="${SPARK_PIDFILE:-/home/dry/spark-models/server.pid}"
LAYAY_PIDFILE="${LAYAY_PIDFILE:-/home/dry/logs/laya-service.pid}"

log() { echo "[$(date "+%F %T")] $*" >> "$LOGFILE"; }

# A model server that legitimately holds VRAM while idle must not keep the box
# alive forever, or gpu2 would never power off. Such processes are "sleep-ok":
# they count as activity only by their own busy signals (see check 4).
is_sleepok() {
  local pid="$1" pname="$2" cmd
  case "$pname" in
    *llama-server*) return 0 ;;
  esac
  for f in "$SPARK_PIDFILE" "$LAYAY_PIDFILE"; do
    [ -f "$f" ] || continue
    [ "$(tr -dc '0-9' < "$f" 2>/dev/null)" = "$pid" ] && return 0
  done
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
  case "$cmd" in
    *gpu2_laya_service*|*llama-server*) return 0 ;;
  esac
  return 1
}

# Single-instance guard: if another watchdog is running, exit.
exec 9>/home/dry/logs/idle-shutdown.lock
if ! flock -n 9; then
  echo "[$(date "+%F %T")] another watchdog instance already running — exiting" >> "$LOGFILE"
  exit 0
fi

# Start counting from the current log position (ignore pre-existing history).
last_size=$(stat -c %s "$OLLAMA_LOG" 2>/dev/null || echo 0)
last_activity=$(date +%s)

log "idle-shutdown watchdog started (idle=${IDLE_MINUTES} min, dry_run=${DRY_RUN}, ignore_ssh=${IGNORE_SSH}, lease=${LEASEFILE})"

while true; do
  sleep 60
  now=$(date +%s)
  activity=0

  # 0) explicit lease: any deploy/benchmark (and the Laya service, on each
  #    request) refreshes this file to hold the machine awake.
  if [ -f "$LEASEFILE" ]; then
    lease_mtime=$(stat -c %Y "$LEASEFILE" 2>/dev/null || echo 0)
    if [ $(( now - lease_mtime )) -lt $(( IDLE_MINUTES * 60 )) ]; then
      activity=1
    fi
  fi

  # 1) New POST lines = real LLM requests since last check (GET probes ignored)
  if [ -f "$OLLAMA_LOG" ]; then
    size=$(stat -c %s "$OLLAMA_LOG" 2>/dev/null || echo 0)
    if [ "$size" -gt "$last_size" ]; then
      chunk=$(tail -c +$((last_size + 1)) "$OLLAMA_LOG" 2>/dev/null)
      if printf '%s' "$chunk" | grep -q "POST"; then
        activity=1
      fi
    fi
    last_size=$size
  fi

  # 2) SSH session present — utmp (interactive) or any established inbound
  #    connection on port 22 (non-interactive `ssh gpu2 cmd`, tmux work).
  if [ "$IGNORE_SSH" != "1" ]; then
    if who 2>/dev/null | grep -q .; then
      activity=1
    elif [ -n "$(ss -Htn state established '( sport = :22 )' 2>/dev/null)" ]; then
      activity=1
    fi
  fi

  # 3) a model is loaded (covers long-running generations)
  ps_out=$(curl -s -m 3 "$OLLAMA_API" 2>/dev/null)
  if [ -n "$ps_out" ] && [ "$ps_out" != '{"models":[]}' ]; then
    activity=1
  fi

  # 4) GPU compute processes. Any process that is NOT a recognised model server
  #    keeps the box awake. Recognised servers hold VRAM while idle by design, so
  #    they count only while actually working: GPU busy, or a fresh log write from
  #    the service that just answered a request.
  compute=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep "[0-9]")
  if [ -n "$compute" ]; then
    busy_other=0
    while IFS=, read -r cpid cpname; do
      cpid=$(printf '%s' "$cpid" | tr -dc '0-9')
      [ -z "$cpid" ] && continue
      if ! is_sleepok "$cpid" "$cpname"; then
        busy_other=1
        break
      fi
    done <<< "$compute"
    if [ "$busy_other" = "1" ]; then
      activity=1
    else
      util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')
      if [ "${util:-0}" -gt 0 ] \
        || [ -n "$(find "$SPARK_LOG" -mmin -10 2>/dev/null)" ] \
        || [ -n "$(find "$LAYAY_LOG" -mmin -10 2>/dev/null)" ]; then
        activity=1
      fi
    fi
  fi

  # 5) active model download (pull): fresh *.partial-* files in blobs dir
  if find "$HOME/.ollama_models/blobs" -maxdepth 1 -name "*-partial*" -mmin -15 2>/dev/null | grep -q .; then
    activity=1
  fi

  # 6) long install/download outside any SSH session (tmux, @reboot jobs).
  if pgrep -f "[p]ip install|[p]ip download|[s]napshot_download|[h]f download|[u]v pip" >/dev/null 2>&1; then
    activity=1
  fi

  if [ "$activity" = "1" ]; then
    last_activity=$now
    continue
  fi

  idle=$(( now - last_activity ))
  if [ "$idle" -ge $(( IDLE_MINUTES * 60 )) ]; then
    log "idle ${idle}s >= ${IDLE_MINUTES}m -> power off (dry_run=${DRY_RUN})"
    if [ "$DRY_RUN" != "1" ]; then
      sudo -n /usr/sbin/poweroff 2>/dev/null || sudo -n poweroff 2>/dev/null || log "poweroff failed (sudoers rule missing?)"
    fi
    exit 0
  fi
done
