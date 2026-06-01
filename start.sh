#!/usr/bin/env bash
# Speech_Voiceprint 一键启动 / 停止 / 状态脚本
# 用法:
#   ./start.sh                # 启动 api(后台),等待 /api/ready
#   ./start.sh start          # 同上
#   ./start.sh start -fg      # 前台启动(日志直接打到当前终端,Ctrl-C 退出)
#   ./start.sh stop           # 停止
#   ./start.sh restart        # 重启
#   ./start.sh status         # 看进程 / 端口 / readiness
#   ./start.sh logs [-f]      # 查看 / 跟随日志
#
# 环境变量优先级:shell export > .env > 本脚本默认。
# 端口/注册表路径可通过环境覆盖:SV_PORT、SV_HOST、SV_REGISTRY_PATH。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
PID_FILE="${ROOT}/.run/api.pid"
LOG_FILE="${ROOT}/.run/api.log"
HOST="${SV_HOST:-127.0.0.1}"
PORT="${SV_PORT:-8080}"
READY_URL="http://${HOST}:${PORT}/api/ready"

mkdir -p "${ROOT}/.run"

_load_env() {
  if [[ -f "${ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${ROOT}/.env"
    set +a
  fi
}

_check_venv() {
  if [[ ! -x "$PY" ]]; then
    echo "❌ 找不到 .venv。先建好虚拟环境并安装依赖:" >&2
    echo "   uv venv --python 3.13 .venv" >&2
    echo "   .venv/bin/pip install -e \".[pyannote,speechbrain,dev]\"" >&2
    exit 1
  fi
}

_running_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    return 1
  fi
  echo "$pid"
}

_wait_ready() {
  local tries="${1:-60}"
  for ((i=0; i<tries; i++)); do
    if curl -fsS "$READY_URL" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

cmd_start() {
  local foreground=0
  [[ "${1:-}" == "-fg" || "${1:-}" == "--foreground" ]] && foreground=1

  _check_venv
  _load_env

  if pid="$(_running_pid)"; then
    echo "ℹ️  已在运行 (pid=$pid),URL: http://${HOST}:${PORT}/"
    return 0
  fi

  local args=(--host "$HOST" --port "$PORT")
  [[ -n "${SV_REGISTRY_PATH:-}" ]] && args+=(--registry "$SV_REGISTRY_PATH")

  if [[ $foreground -eq 1 ]]; then
    echo "▶  前台启动 pipeline.api (Ctrl-C 退出)…"
    exec "$PY" -m pipeline.api "${args[@]}"
  fi

  echo "▶  启动 pipeline.api → $LOG_FILE"
  nohup "$PY" -m pipeline.api "${args[@]}" >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"

  if _wait_ready 60; then
    echo "✅ 就绪:http://${HOST}:${PORT}/  (pid=$(cat "$PID_FILE"))"
    echo "   /api/ready -> $(curl -fsS "$READY_URL")"
  else
    echo "❌ 60s 内未就绪,看日志:tail -f $LOG_FILE" >&2
    return 1
  fi
}

cmd_stop() {
  if ! pid="$(_running_pid)"; then
    # PID 文件残留也清掉
    rm -f "$PID_FILE"
    # 兜底:杀掉本机上同名进程(只在没有 PID 文件时才扫)
    if pgrep -f "pipeline\.api" >/dev/null 2>&1; then
      echo "⚠️  PID 文件丢失,但发现 pipeline.api 进程,kill 之"
      pkill -TERM -f "pipeline\.api" || true
      sleep 1
      pkill -KILL -f "pipeline\.api" 2>/dev/null || true
    else
      echo "ℹ️  未运行"
    fi
    return 0
  fi

  echo "⏹  发送 SIGTERM (pid=$pid)…"
  kill -TERM "$pid"

  # SV_SHUTDOWN_TIMEOUT 默认 60s,这里给 75s 余量
  for ((i=0; i<75; i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "✅ 已停止"
      return 0
    fi
    sleep 1
  done

  echo "⚠️  优雅停机超时,SIGKILL"
  kill -KILL "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
}

cmd_status() {
  if pid="$(_running_pid)"; then
    echo "✅ running   pid=$pid"
    echo "   URL      http://${HOST}:${PORT}/"
    if ready=$(curl -fsS "$READY_URL" 2>/dev/null); then
      echo "   ready    $ready"
    else
      echo "   ready    (无响应)"
    fi
    echo "   log      $LOG_FILE"
  else
    echo "❌ not running"
    return 1
  fi
}

cmd_logs() {
  if [[ ! -f "$LOG_FILE" ]]; then
    echo "(尚无日志文件 $LOG_FILE)"
    return 0
  fi
  if [[ "${1:-}" == "-f" || "${1:-}" == "--follow" ]]; then
    tail -f "$LOG_FILE"
  else
    tail -n 200 "$LOG_FILE"
  fi
}

case "${1:-start}" in
  start|"")   shift || true; cmd_start "$@" ;;
  stop)       cmd_stop ;;
  restart)    cmd_stop; cmd_start ;;
  status)     cmd_status ;;
  logs)       shift; cmd_logs "$@" ;;
  -h|--help|help)
    sed -n '2,15p' "${BASH_SOURCE[0]}"
    ;;
  *)
    echo "未知命令: $1" >&2
    echo "用法: $0 {start [-fg] | stop | restart | status | logs [-f]}" >&2
    exit 2
    ;;
esac
