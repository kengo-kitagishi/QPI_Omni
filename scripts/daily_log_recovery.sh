#!/bin/zsh
# daily_log_recovery.sh
# 翌朝 06:00 に launchd から呼ばれ、前日の日次ログが欠落していれば再生成を試みる。
#
# launchd plist: ~/Library/LaunchAgents/com.kitak.daily_log_recovery.plist

export HOME=/Users/kitak
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin

SCRIPT_DIR="/Users/kitak/QPI_Omni/scripts"
LOG_DIR="/Users/kitak/QPI_Omni/logs"
HEALTH_FILE="$LOG_DIR/daily_log_health.json"

# 前日の日付を計算
YESTERDAY=$(python3 -c "from datetime import date, timedelta; print((date.today() - timedelta(days=1)).strftime('%Y-%m-%d'))")
RECOVERY_LOG="$LOG_DIR/daily_log_recovery_${YESTERDAY}.log"
DAILY_LOG_PATH="/Users/kitak/Documents/Obsidian Vault/01_Daily/${YESTERDAY}.md"

{
    echo "=== daily_log_recovery 開始: $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "  対象日: $YESTERDAY"

    # health.json を読んで判定
    needs_recovery=false

    if [ ! -f "$DAILY_LOG_PATH" ]; then
        echo "  日次ログが存在しない → recovery 実行"
        needs_recovery=true
    elif [ ! -s "$DAILY_LOG_PATH" ]; then
        echo "  日次ログが空 → recovery 実行"
        needs_recovery=true
    elif [ -f "$HEALTH_FILE" ]; then
        # health.json の success を読む
        success=$(python3 -c "import json; h=json.load(open('$HEALTH_FILE')); print(h.get('success', False))")
        if [ "$success" = "False" ]; then
            echo "  health.json: success=False → recovery 実行"
            needs_recovery=true
        else
            echo "  health.json: success=True, ログも存在 → recovery 不要"
        fi
    else
        echo "  health.json が存在しない, ログは存在するが念のため確認"
        # ログが 100 bytes 以下ならスケルトンの可能性
        log_size=$(wc -c < "$DAILY_LOG_PATH" 2>/dev/null || echo 0)
        if [ "$log_size" -lt 100 ]; then
            echo "  日次ログが小さすぎる (${log_size} bytes) → recovery 実行"
            needs_recovery=true
        fi
    fi

    if [ "$needs_recovery" = "false" ]; then
        echo "=== recovery 不要 ==="
        exit 0
    fi

    # recovery 実行（daily_log_cron.sh を --recover で呼ぶ）
    echo "--- recovery 開始 ---"
    # 対象日を指定して generate_daily_log を直接呼ぶ
    cd /Users/kitak/QPI_Omni

    python3 "$SCRIPT_DIR/jsonl_to_obsidian.py" || echo "WARN: jsonl_to_obsidian failed (continuing)"
    python3 "$SCRIPT_DIR/figure_inbox_to_obsidian.py" || echo "WARN: figure_inbox failed (continuing)"

    WEEK_LABEL=$(python3 -c "from datetime import date; d=date.fromisoformat('$YESTERDAY'); print(d.strftime('%Y-W%V'))")
    python3 "$SCRIPT_DIR/weekly_report_hub.py" --week "$WEEK_LABEL" --no-preprocess || echo "WARN: weekly_report_hub failed (continuing)"

    python3 "$SCRIPT_DIR/generate_daily_log.py" --date "$YESTERDAY" --overwrite --backend cli
    rc=$?

    if [ $rc -eq 0 ]; then
        echo "  recovery SUCCESS"
        # health.json を更新
        python3 -c "
import json, os
from datetime import datetime, timezone, timedelta
JST = timezone(timedelta(hours=9))
health = {
    'date': '$YESTERDAY',
    'updated_at': datetime.now(JST).isoformat(),
    'success': True,
    'log_exists': True,
    'attempts': 1,
    'final_rc': 0,
    'recovered': True,
}
with open('$HEALTH_FILE', 'w') as f:
    json.dump(health, f, indent=2, ensure_ascii=False)
"
    else
        echo "  recovery FAILED (rc=$rc)"
        # macOS 通知
        osascript -e 'display notification "前日の日次ログ recovery に失敗しました (rc='$rc')" with title "Daily Log Recovery"' 2>/dev/null || true
    fi

    echo "=== recovery 完了: $(date '+%Y-%m-%d %H:%M:%S') ==="

} 2>&1 | tee "$RECOVERY_LOG"
