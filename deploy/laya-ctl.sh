#!/usr/bin/env bash
# laya-ctl.sh — управление Laya Tier-1 сервисом на GPU2 (:8031).
#
#   laya-ctl.sh start            поднять сервис (перед этим берёт lease)
#   laya-ctl.sh stop             остановить
#   laya-ctl.sh restart          перезапустить
#   laya-ctl.sh status           процесс + /health + /stats + VRAM + lease
#   laya-ctl.sh wait             ждать, пока модель загрузится (health=ok)
#   laya-ctl.sh warmup           прогреть CUDA-ядра до бенчмарка
#   laya-ctl.sh hold             продлить lease (не дать watchdog погасить машину)
#   laya-ctl.sh predict "запрос" одиночный прогноз
set -uo pipefail

BASE="${LAYA_HOME:-$HOME/jev-laya}"
VENV="$BASE/.venv"
LOG="$HOME/logs/laya-service.log"
PIDFILE="$HOME/logs/laya-service.pid"
LEASEFILE="$HOME/logs/activity.lease"
PORT="${LAYA_PORT:-8031}"
PAT="[g]pu2_laya_service:app"   # bracket-трюк: паттерн не матчит сам себя

mkdir -p "$HOME/logs"

running() { pgrep -f "$PAT" >/dev/null 2>&1; }
lease() { touch "$LEASEFILE"; }

start() {
  if running; then
    echo "laya: уже запущен (pid $(pgrep -f "$PAT" | head -1))"
    return 0
  fi
  lease   # не дать watchdog погасить GPU2 во время скачивания/загрузки модели
  cd "$BASE" || { echo "laya: нет каталога $BASE"; return 1; }
  nohup "$VENV/bin/uvicorn" gpu2_laya_service:app \
    --host 0.0.0.0 --port "$PORT" --workers 1 >> "$LOG" 2>&1 &
  sleep 2
  pgrep -f "$PAT" | head -1 > "$PIDFILE"
  echo "laya: запуск pid $(cat "$PIDFILE") (порт $PORT, лог $LOG)"
}

stop() {
  if ! running; then
    echo "laya: не запущен"
    rm -f "$PIDFILE"
    return 0
  fi
  pkill -f "$PAT"
  for _ in $(seq 1 30); do running || break; sleep 0.5; done
  rm -f "$PIDFILE"
  echo "laya: остановлен"
}

wait_ready() {
  for i in $(seq 1 "${1:-180}"); do
    if curl -sf -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
      echo "laya: ГОТОВ за ${i}s"
      return 0
    fi
    lease
    sleep 1
  done
  echo "laya: не поднялся за ${1:-180}s — смотри $LOG"
  curl -s -m 3 "http://127.0.0.1:$PORT/health" || true
  return 1
}

status() {
  if running; then
    echo "процесс : pid $(pgrep -f "$PAT" | head -1) — запущен"
  else
    echo "процесс : не запущен"
  fi
  echo -n "health  : "
  curl -s -m 5 "http://127.0.0.1:$PORT/health" || echo "нет ответа"
  echo
  echo -n "stats   : "
  curl -s -m 5 "http://127.0.0.1:$PORT/stats" || echo "нет ответа"
  echo
  echo -n "VRAM    : "
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
  echo -n "lease   : "
  if [ -f "$LEASEFILE" ]; then stat -c '%y' "$LEASEFILE"; else echo "нет"; fi
  echo -n "watchdog: "
  if pgrep -f "[i]dle-shutdown.sh" >/dev/null 2>&1; then echo "работает"; else echo "НЕ работает"; fi
}

predict() {
  local q="${1:-status check}"
  curl -s -m 30 -X POST "http://127.0.0.1:$PORT/predict" \
    -H 'Content-Type: application/json' \
    --data "$(printf '{"query":%s}' "$(printf '%s' "$q" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")"
  echo
}

case "${1:-status}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  status)  status ;;
  wait)    wait_ready "${2:-180}" ;;
  warmup)  curl -s -m 120 -X POST "http://127.0.0.1:$PORT/warmup"; echo ;;
  hold)    lease; echo "lease обновлён: $(stat -c '%y' "$LEASEFILE")" ;;
  predict) shift; predict "$@" ;;
  *)       sed -n '2,14p' "$0" ;;
esac
