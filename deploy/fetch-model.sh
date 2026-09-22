#!/usr/bin/env bash
# fetch-model.sh — устойчивая загрузка чекпойнтов Laya на GPU2.
#
# Зачем отдельный скрипт: `snapshot_download` внутри сервиса умеет резюмить, но
# когда xet-чанк зависает, процесс висит бесконечно и сервис никогда не стартует.
# Наблюдаемое поведение xet: первая попытка переносит почти все байты и виснет на
# последнем файле, повторный вызов коммитит их за секунды. Поэтому — цикл с timeout.
#
#   SUBFOLDER=multilingual ./fetch-model.sh          # mmBERT-base (100+ языков)
#   ./fetch-model.sh                                 # английский чекпойнт (корень)
set -uo pipefail

export PATH="$HOME/jev-laya/.venv/bin:/usr/local/bin:/usr/bin:/bin"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

REPO="${REPO:-convaiinnovations/laya}"
SUBFOLDER="${SUBFOLDER:-}"
CACHE="$HOME/.cache/huggingface/hub/models--${REPO//\//--}"
ATTEMPTS="${ATTEMPTS:-6}"
# ВАЖНО: на этом канале HF отдаёт ~0.75 МБ/с, а чекпойнт весит ~650-850 МБ, то есть
# одна попытка честно идёт 10-20 минут. Слишком короткий PER_TRY_TIMEOUT убивает
# попытку раньше, чем она успевает закоммитить файл, и выглядит это как «качается 0
# байт, хотя сеть загружена». Ставьте с запасом.
PER_TRY_TIMEOUT="${PER_TRY_TIMEOUT:-1800}"
MIN_WEIGHTS="${MIN_WEIGHTS:-600000000}"
PREFIX=""
[ -n "$SUBFOLDER" ] && PREFIX="$SUBFOLDER/"
FILES=("${PREFIX}rl_agent_config.json" "${PREFIX}model.safetensors" "${PREFIX}tokenizer/tokenizer.json" "${PREFIX}tokenizer/tokenizer_config.json" "${PREFIX}encoder/config.json")

snap_dir() { find "$CACHE/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1; }
weights_path() { echo "$(snap_dir)/${PREFIX}model.safetensors"; }

weights_ready() {
  local path size
  path=$(weights_path)
  [ -f "$path" ] || return 1
  size=$(stat -Lc %s "$path" 2>/dev/null || echo 0)
  [ "$size" -gt "$MIN_WEIGHTS" ]
}

echo "репозиторий: $REPO ${SUBFOLDER:+(подпапка $SUBFOLDER)}; цель: >$MIN_WEIGHTS байт"

for attempt in $(seq 1 "$ATTEMPTS"); do
  touch "$HOME/logs/activity.lease"   # не дать watchdog погасить машину
  if weights_ready; then
    echo "ГОТОВО: веса на месте ($(stat -Lc %s "$(weights_path)") байт), попыток: $((attempt - 1))"
    exit 0
  fi
  echo "[$(date +%T)] попытка $attempt/$ATTEMPTS, веса: ${MIN_WEIGHTS:-?} порог, сейчас $(stat -Lc %s "$(weights_path)" 2>/dev/null || echo 0) байт"
  timeout "$PER_TRY_TIMEOUT" hf download "$REPO" "${FILES[@]}" >/dev/null 2>&1
  echo "[$(date +%T)] попытка $attempt завершилась кодом $?"
done

if weights_ready; then
  echo "ГОТОВО: веса на месте"
  exit 0
fi
echo "НЕУДАЧА: веса не скачаны за $ATTEMPTS попыток"
exit 1
