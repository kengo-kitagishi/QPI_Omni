#!/bin/zsh
# daily_log_cron.sh - launchd 経由で毎朝 05:00 JST に前日の日次ログを生成
#
# 方式: claude CLI (OAuth) + Agent ツール並列読み
#   - generate_daily_log.py (map-reduce via claude --print) は使わない
#   - claude -p を一発起動し、Claude Code 自身が Agent ツールで
#     daily_index を並列読みしてログを書く
#   - API キーは使わない（OAuth のみ）
#
# launchd plist: ~/Library/LaunchAgents/com.kitak.daily_log.plist
#   StartCalendarInterval: Hour=5, Minute=0
#
# 手動実行: bash daily_log_cron.sh [YYYY-MM-DD]

set -u

export HOME=/Users/kitak
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin

# API キーを明示的に unset（OAuth 経路を強制）
unset ANTHROPIC_API_KEY

SCRIPT_DIR="/Users/kitak/QPI_Omni/scripts"
LOG_DIR="/Users/kitak/QPI_Omni/logs"
OBSIDIAN="/Users/kitak/Documents/Obsidian Vault"

# 引数: 第1引数 = 対象日（省略時は前日）、--no-catchup で過去日の再生成を抑制
NO_CATCHUP=false
TARGET_DATE=""
for arg in "$@"; do
    case "$arg" in
        --no-catchup) NO_CATCHUP=true ;;
        --recover) ;;  # legacy flag, ignored
        --*) ;;
        *) [ -z "$TARGET_DATE" ] && TARGET_DATE="$arg" ;;
    esac
done
if [ -z "$TARGET_DATE" ]; then
    TARGET_DATE=$(python3 -c "from datetime import date, timedelta; print((date.today() - timedelta(days=1)).strftime('%Y-%m-%d'))")
fi

LOG_FILE="$LOG_DIR/daily_log_${TARGET_DATE}.log"
HEALTH_FILE="$LOG_DIR/daily_log_health.json"
mkdir -p "$LOG_DIR"

# 30日より古いログを自動削除
find "$LOG_DIR" -name "daily_log_*.log" -mtime +30 -delete 2>/dev/null
find "$LOG_DIR" -name "daily_log_recovery_*.log" -mtime +30 -delete 2>/dev/null

{
    echo "=== daily_log_cron.sh 開始: $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "  対象日: $TARGET_DATE"
    echo "  方式: claude CLI + Agent 並列読み (OAuth)"

    cd /Users/kitak/QPI_Omni

    # ── Stage 1: 索引を最新化（各段階で失敗しても次に進む） ──
    echo "--- Stage 1a: jsonl_to_obsidian ---"
    python3 "$SCRIPT_DIR/jsonl_to_obsidian.py" || echo "WARN: jsonl_to_obsidian failed (rc=$?, continuing)"

    echo "--- Stage 1b: figure_inbox_to_obsidian ---"
    python3 "$SCRIPT_DIR/figure_inbox_to_obsidian.py" || echo "WARN: figure_inbox_to_obsidian failed (rc=$?, continuing)"

    # ── Stage 2: weekly index と daily_index を生成 ──
    echo "--- Stage 2: weekly_report_hub ---"
    WEEK_LABEL=$(python3 -c "from datetime import date; d=date.fromisoformat('$TARGET_DATE'); print(d.strftime('%Y-W%V'))")
    python3 "$SCRIPT_DIR/weekly_report_hub.py" --week "$WEEK_LABEL" --no-preprocess || echo "WARN: weekly_report_hub failed (rc=$?, continuing)"

    # 索引ファイル存在チェック（空なら中断）
    INDEX_FILE="$OBSIDIAN/00_Inbox/daily_index_${TARGET_DATE}.md"
    if [ ! -f "$INDEX_FILE" ]; then
        echo "ERROR: daily_index が存在しません: $INDEX_FILE"
        final_rc=2
        echo "=== 完了: $(date '+%Y-%m-%d %H:%M:%S') (rc=$final_rc, 索引なし) ==="
    else
        INDEX_LINES=$(wc -l < "$INDEX_FILE" | tr -d ' ')
        echo "  索引行数: $INDEX_LINES"

        # ── Stage 3: claude -p で Claude Code を起動 → Agent ツールで並列読み → ログ生成 ──
        echo "--- Stage 3: claude -p (Agent 並列読み) ---"

        PROMPT="daily-log スキルを使って ${TARGET_DATE} の日次ログを生成してください。

重要な実行手順:
1. ${INDEX_FILE} を読む（${INDEX_LINES} 行）
2. 3000 行超の場合は Agent ツール（subagent_type: Explore）で 2500 行ずつ並列読み
   - 必要な数のエージェントを 1 つのメッセージで同時起動
3. スタイルガイド:
   - /Users/kitak/.claude/skills/gm-log-compiler/references/gm_style.md
   - /Users/kitak/.claude/skills/daily-log/references/daily_log_style.md
   両方を読んで従うこと
4. トピック（内容）単位で複数ファイルに分割保存:
   - ハブ: ${OBSIDIAN}/01_Daily/${TARGET_DATE}.md（トピック一覧のみ・短め）
   - 各トピック: ${OBSIDIAN}/01_Daily/${TARGET_DATE}_<トピック名>.md（5000 日本語文字以上）
5. 既存ファイルは上書き
6. 図は figure inbox JSON から全件掲載。選別禁止。各図に data_source / params / data_file / data_keys のパラメータ表を付ける
7. generate_daily_log.py は呼ばない。Agent 並列読み → 自分のコンテキストで執筆
8. Notion / Google Drive / Google Calendar MCP は使わない
9. git commit / push は行わない
10. 完了後、保存したファイル名を列挙して 1 行で報告

最重要: api 呼び出し（anthropic SDK / ANTHROPIC_API_KEY）は一切しないこと。claude CLI のツール経由のみで完結させること。"

        final_rc=1
        for attempt in 1 2 3; do
            echo "  attempt $attempt/3 ..."
            claude -p "$PROMPT" \
                --model claude-opus-4-6 \
                --allowedTools "Bash,Read,Write,Edit,Glob,Grep,Agent,TaskCreate,TaskUpdate,TaskList" \
                --dangerously-skip-permissions \
                --no-session-persistence
            rc=$?
            if [ $rc -eq 0 ]; then
                echo "  SUCCESS on attempt $attempt"
                final_rc=0
                break
            fi
            final_rc=$rc
            if [ $attempt -lt 3 ]; then
                echo "  FAILED (rc=$rc), waiting 30min..."
                sleep 1800
            fi
        done

        echo "=== 完了: $(date '+%Y-%m-%d %H:%M:%S') (rc=$final_rc) ==="
    fi

    # ── Catch-up: 過去2日の日次ログがスタブ（no-activity）かつ daily_index に中身があれば再生成 ──
    # 理由: Drive 同期遅延で当日処理に間に合わなかった日のリカバリ。
    # --no-catchup 指定時 or catch-up 内の再帰呼び出しでは実行しない。
    if [ "$NO_CATCHUP" = "false" ]; then
        for offset in 1 2; do
            CATCHUP_DATE=$(python3 -c "from datetime import date, timedelta; print((date.fromisoformat('$TARGET_DATE')-timedelta(days=$offset)).strftime('%Y-%m-%d'))")
            CATCHUP_LOG="$OBSIDIAN/01_Daily/${CATCHUP_DATE}.md"
            CATCHUP_INDEX="$OBSIDIAN/00_Inbox/daily_index_${CATCHUP_DATE}.md"
            # daily_index に中身があり (>50行)、本文がスタブ or 不在のときだけ再生成
            if [ -f "$CATCHUP_INDEX" ] && [ "$(wc -l < "$CATCHUP_INDEX" | tr -d ' ')" -gt 50 ]; then
                if [ ! -f "$CATCHUP_LOG" ] || grep -q "status: no-activity" "$CATCHUP_LOG" 2>/dev/null; then
                    echo "--- Catch-up: $CATCHUP_DATE （スタブのため再生成） ---"
                    bash "$0" "$CATCHUP_DATE" --no-catchup
                fi
            fi
        done
    fi

    # ── health.json に結果を記録 ──
    DAILY_LOG_PATH="$OBSIDIAN/01_Daily/${TARGET_DATE}.md"
    python3 -c "
import json, os
from datetime import datetime, timezone, timedelta
JST = timezone(timedelta(hours=9))
log_path = '${DAILY_LOG_PATH}'
health = {
    'date': '${TARGET_DATE}',
    'updated_at': datetime.now(JST).isoformat(),
    'success': os.path.exists(log_path) and os.path.getsize(log_path) > 100,
    'log_exists': os.path.exists(log_path),
    'final_rc': ${final_rc},
    'method': 'claude_cli_agent_parallel',
}
with open('${HEALTH_FILE}', 'w') as f:
    json.dump(health, f, indent=2, ensure_ascii=False)
print(f'health.json: {json.dumps(health, ensure_ascii=False)}')
"

} 2>&1 | tee -a "$LOG_FILE"
