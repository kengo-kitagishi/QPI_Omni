#!/bin/zsh
# daily_log_cron.sh
# launchd から呼ばれる日次ログ自動生成スクリプト
#
# launchd plist: ~/Library/LaunchAgents/com.kitak.daily_log.plist
#   StartCalendarInterval: Hour=23, Minute=0  # 毎日 23:00 JST
#
# --recover フラグ: 翌朝の recovery ジョブから呼ばれた場合（同じロジック）
#
# 改善 (2026-04-08):
#   - 各ステージが独立実行（1段階の失敗で全体が止まらない）
#   - 日次ログ生成を最大3回リトライ（rate limit 対策）
#   - health.json に毎日の成否を記録
#   - 30日より古いログを自動削除

export HOME=/Users/kitak
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin

# Anthropic API キー（launchd は shell env を継承しないため）
if [ -f "$HOME/.anthropic_api_key" ]; then
    export ANTHROPIC_API_KEY=$(cat "$HOME/.anthropic_api_key")
elif [ -f "$HOME/.env" ]; then
    export $(grep ANTHROPIC_API_KEY "$HOME/.env" | xargs)
fi

SCRIPT_DIR="/Users/kitak/QPI_Omni/scripts"
LOG_DIR="/Users/kitak/QPI_Omni/logs"
TODAY=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/daily_log_${TODAY}.log"
HEALTH_FILE="$LOG_DIR/daily_log_health.json"
mkdir -p "$LOG_DIR"

# ── ログローテーション: 30日より古いログを削除 ──
find "$LOG_DIR" -name "daily_log_*.log" -mtime +30 -delete 2>/dev/null
find "$LOG_DIR" -name "daily_log_recovery_*.log" -mtime +30 -delete 2>/dev/null

{
    echo "=== daily_log_cron.sh 開始: $(date '+%Y-%m-%d %H:%M:%S') ==="
    if [ "$1" = "--recover" ]; then
        echo "  (recovery モード)"
    fi

    cd /Users/kitak/QPI_Omni

    # ── Stage 1: 索引を最新化（各段階で失敗しても次に進む） ──
    echo "--- Stage 1a: jsonl_to_obsidian ---"
    python3 "$SCRIPT_DIR/jsonl_to_obsidian.py" || echo "WARN: jsonl_to_obsidian failed (rc=$?, continuing)"

    echo "--- Stage 1b: figure_inbox_to_obsidian ---"
    python3 "$SCRIPT_DIR/figure_inbox_to_obsidian.py" || echo "WARN: figure_inbox_to_obsidian failed (rc=$?, continuing)"

    # ── Stage 2: weekly index と daily_index を生成 ──
    echo "--- Stage 2: weekly_report_hub ---"
    WEEK_LABEL=$(python3 -c "from datetime import date; print(date.today().strftime('%Y-W%V'))")
    python3 "$SCRIPT_DIR/weekly_report_hub.py" --week "$WEEK_LABEL" --no-preprocess || echo "WARN: weekly_report_hub failed (rc=$?, continuing)"

    # ── Stage 3: 日次ログ生成（最大3回リトライ） ──
    echo "--- Stage 3: generate_daily_log ---"
    final_rc=1
    final_attempt=0
    for attempt in 1 2 3; do
        final_attempt=$attempt
        echo "  attempt $attempt/3 ..."
        python3 "$SCRIPT_DIR/generate_daily_log.py" --overwrite --backend cli
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "  SUCCESS on attempt $attempt"
            final_rc=0
            break
        fi
        if [ $rc -eq 2 ]; then
            echo "  ERROR: 索引が空 — リトライしても無駄なので中止"
            final_rc=2
            break
        fi
        final_rc=$rc
        if [ $attempt -lt 3 ]; then
            echo "  FAILED on attempt $attempt (rc=$rc), waiting 30min..."
            sleep 1800
        else
            echo "  FAILED on attempt $attempt (rc=$rc), 最大リトライ回数到達"
        fi
    done

    echo "=== 完了: $(date '+%Y-%m-%d %H:%M:%S') (rc=$final_rc, attempts=$final_attempt) ==="

    # ── health.json に結果を記録 ──
    DAILY_LOG_PATH="/Users/kitak/Documents/Obsidian Vault/01_Daily/${TODAY}.md"
    python3 -c "
import json, os
from datetime import datetime, timezone, timedelta
JST = timezone(timedelta(hours=9))
today = '${TODAY}'
log_path = '${DAILY_LOG_PATH}'
health = {
    'date': today,
    'updated_at': datetime.now(JST).isoformat(),
    'success': os.path.exists(log_path) and os.path.getsize(log_path) > 100,
    'log_exists': os.path.exists(log_path),
    'attempts': ${final_attempt},
    'final_rc': ${final_rc},
}
with open('${HEALTH_FILE}', 'w') as f:
    json.dump(health, f, indent=2, ensure_ascii=False)
print(f'health.json: {json.dumps(health, ensure_ascii=False)}')
"

} 2>&1 | tee -a "$LOG_FILE"
