#!/bin/zsh
# daily_log_cron.sh
# launchd から呼ばれる日次ログ自動生成スクリプト
#
# launchd plist: ~/Library/LaunchAgents/com.kitak.daily_log.plist
#   StartCalendarInterval: Hour=23, Minute=0  # 毎日 23:00 JST

export HOME=/Users/kitak
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin

# Anthropic API キー（launchd は shell env を継承しないため）
if [ -f "$HOME/.anthropic_api_key" ]; then
    export ANTHROPIC_API_KEY=$(cat "$HOME/.anthropic_api_key")
elif [ -f "$HOME/.env" ]; then
    export $(grep ANTHROPIC_API_KEY "$HOME/.env" | xargs)
fi

SCRIPT_DIR="/Users/kitak/QPI_Omni/scripts"
LOG_FILE="/Users/kitak/QPI_Omni/logs/daily_log_$(date +%Y-%m-%d).log"
mkdir -p "$(dirname "$LOG_FILE")"

{
    echo "=== daily_log_cron.sh 開始: $(date '+%Y-%m-%d %H:%M:%S') ==="

    cd /Users/kitak/QPI_Omni

    # 1. 索引を最新化
    python3 "$SCRIPT_DIR/jsonl_to_obsidian.py"
    python3 "$SCRIPT_DIR/figure_inbox_to_obsidian.py"

    # 2. 今日の weekly index と daily_index を生成
    WEEK_LABEL=$(python3 -c "from datetime import date; print(date.today().strftime('%Y-W%V'))")
    python3 "$SCRIPT_DIR/weekly_report_hub.py" --week "$WEEK_LABEL" --no-preprocess

    # 3. 今日の日次ログを生成（Anthropic API 経由、1コールで完結）
    python3 "$SCRIPT_DIR/generate_daily_log.py" --overwrite

    echo "=== 完了: $(date '+%Y-%m-%d %H:%M:%S') ==="
} 2>&1 | tee "$LOG_FILE"
