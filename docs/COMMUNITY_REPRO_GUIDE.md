# Claude Code + Claude.ai + Codex + ClickUp + Notion + Obsidian + Google Drive
# 研究自動化システム 完全再現手順（macOS基準）

最終更新: 2026-02-27 (JST)

---

## 0. この手順で再現できるもの

- Cursor / Claude Code から ClickUp・Notion を呼ぶ MCP 構成
- QPI_Omni の作業ログ標準（`WORKLOG_SPEC.md`）
- Notion -> Obsidian の定期同期（launchd / 24時間）
- Obsidian・プレゼン資料・thesis を Google Drive に定期ミラー（launchd / 24時間）
- 作業ログの自動収集（任意: `session_activity_logger.py`）
- 論文フォルダ/解析結果フォルダのシンボリックリンク導線

この手順は、**上から順に実行すれば再現できる**ように書いている。

---

## 0.1 配布者側: 自動化バンドルを作る（コミュニティ公開用）

> 受け手が「コマンドだけ」で再現できるように、先に配布者が素材を固める手順。

```bash
export USER_HOME="$HOME"
export REPO_DIR="$USER_HOME/QPI_Omni"
export VAULT_DIR="$USER_HOME/Documents/Obsidian Vault"
export AUTOMATION_DIR="$VAULT_DIR/98_Automation"
export GDRIVE_RUNTIME_DIR="$USER_HOME/Library/Application Support/gdrive-mirror"

mkdir -p "$REPO_DIR/dist/automation_bundle"
mkdir -p "$REPO_DIR/dist/automation_bundle/98_Automation"
mkdir -p "$REPO_DIR/dist/automation_bundle/gdrive-mirror"

cp -a "$AUTOMATION_DIR/." "$REPO_DIR/dist/automation_bundle/98_Automation/"
cp -a "$GDRIVE_RUNTIME_DIR/run_gdrive_mirror.sh" "$REPO_DIR/dist/automation_bundle/gdrive-mirror/"
cp -a "$GDRIVE_RUNTIME_DIR/install_gdrive_mirror_launchd.sh" "$REPO_DIR/dist/automation_bundle/gdrive-mirror/"

# secretsは含めない（公開禁止）
rm -f "$REPO_DIR/dist/automation_bundle/98_Automation/.notion_sync.env" || true
rm -f "$REPO_DIR/dist/automation_bundle/gdrive-mirror/.gdrive_mirror.env" || true
rm -rf "$REPO_DIR/dist/automation_bundle/98_Automation/logs" || true

tar -czf "$REPO_DIR/dist/community_automation_bundle.tgz" -C "$REPO_DIR/dist" automation_bundle
ls -la "$REPO_DIR/dist/community_automation_bundle.tgz"
```

受け手には以下を配布:

- `community_automation_bundle.tgz`
- この手順書（`docs/COMMUNITY_REPRO_GUIDE.md`）

---

## 1. 前提（必須）

- OS: macOS（Ventura/Sonoma 系を想定）
- シェル: zsh
- 既存アカウント: `kitak`（他ユーザーの場合は `~/` を置換）
- 必須アプリ: Cursor, Obsidian, Google Drive for Desktop, Terminal
- APIトークン:
  - Notion Internal Integration Token
  - ClickUp API Token

> 注意: 本書では機密値を `<...>` として記載。実値は貼らないこと。

---

## 2. 変数定義（最初に実行）

```bash
# 実行ユーザーごとに必要なら書き換え
export USER_HOME="$HOME"
export REPO_DIR="$USER_HOME/QPI_Omni"
export VAULT_DIR="$USER_HOME/Documents/Obsidian Vault"
export AUTOMATION_DIR="$VAULT_DIR/98_Automation"
export NOTION_RUNTIME_DIR="$USER_HOME/Library/Application Support/notion-sync"
export GDRIVE_RUNTIME_DIR="$USER_HOME/Library/Application Support/gdrive-mirror"
```

確認:

```bash
echo "$REPO_DIR"
echo "$VAULT_DIR"
```

---

## 3. macOS ベース環境構築

### Step 3-1. Xcode CLI / Homebrew

```bash
xcode-select --install || true
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || true
```

### Step 3-2. 必須ツール

```bash
brew update
brew install git gh node python@3.11 ripgrep
```

確認:

```bash
git --version
gh --version
node --version
npm --version
python3 --version
rg --version
```

---

## 4. リポジトリ取得

```bash
cd "$USER_HOME"
git clone https://github.com/kengo-kitagishi/QPI_Omni.git "$REPO_DIR" || true
cd "$REPO_DIR"
git pull
```

Python依存:

```bash
cd "$REPO_DIR"
pip3 install -r requirements.txt
```

GitHub CLI 認証（Issue自動化を使うなら）:

```bash
gh auth login
# GitHub.com -> HTTPS -> Login with a web browser
```

---

## 5. Cursor MCP 設定（Notion + ClickUp）

### Step 5-1. `~/.cursor/mcp.json` を作成

```bash
mkdir -p "$USER_HOME/.cursor"
cat > "$USER_HOME/.cursor/mcp.json" <<'JSON'
{
  "mcpServers": {
    "notionApi": {
      "command": "npx",
      "args": ["-y", "@mieubrisse/notion-mcp-server"],
      "env": {
        "OPENAPI_MCP_HEADERS": "{\"Authorization\": \"Bearer <NOTION_TOKEN>\", \"Notion-Version\": \"2022-06-28\"}"
      }
    },
    "clickup": {
      "env": {
        "CLICKUP_API_TOKEN": "<CLICKUP_TOKEN>"
      }
    }
  }
}
JSON
```

### Step 5-2. 反映

- Cursorを完全終了 -> 再起動

### Step 5-3. 安全確認（秘密値は表示しない）

```bash
python3 - <<'PY'
import json, pathlib
p = pathlib.Path.home()/'.cursor'/'mcp.json'
obj = json.loads(p.read_text())
print('mcp exists:', p.exists())
print('servers:', ', '.join(sorted(obj.get('mcpServers',{}).keys())))
PY
```

---

## 6. QPI_Omni ルール統一（Claude/Cursor/Codex）

この構成は「仕様正本 = `docs/WORKLOG_SPEC.md`」。

確認コマンド:

```bash
cd "$REPO_DIR"
ls -la AGENTS.md CLAUDE.md docs/WORKLOG_SPEC.md .cursor/rules/task-routing.mdc
rg -n "WORKLOG_SPEC|作業ログ" AGENTS.md CLAUDE.md .cursor/rules/task-routing.mdc
```

運用ルール:

- 作業ログ形式は `WORKLOG_SPEC.md` を変更すればよい
- Cursor / Claude / Codex の各ルールは「正本参照」に留める
- 形式変更はユーザー明示承認がある時だけ

---

## 7. Notion DB 初期化（Typeプロパティ）

```bash
cd "$REPO_DIR"
python3 scripts/notion_setup_type.py
```

期待結果:

- `Type` select に `思考 / 作業ログ / 図 / プラン` が利用可能になる

---

## 8. Notion -> Obsidian 同期（24時間自動）

### Step 8-1. Obsidian Vault 準備

```bash
mkdir -p "$VAULT_DIR"
mkdir -p "$VAULT_DIR/00_Inbox/notion_sync/api"
mkdir -p "$VAULT_DIR/01_Daily"
mkdir -p "$AUTOMATION_DIR"
```

### Step 8-2. 自動化バンドル展開（受け手側）

```bash
cd "$USER_HOME"
tar -xzf /path/to/community_automation_bundle.tgz
cp -a "$USER_HOME/automation_bundle/98_Automation/." "$AUTOMATION_DIR/"
ls -la "$AUTOMATION_DIR"
```

### Step 8-3. env作成

```bash
cp -f "$AUTOMATION_DIR/.notion_sync.env.template" "$AUTOMATION_DIR/.notion_sync.env"
/usr/bin/sed -i '' "s|^OBSIDIAN_VAULT_DIR=.*|OBSIDIAN_VAULT_DIR=\"$VAULT_DIR\"|" "$AUTOMATION_DIR/.notion_sync.env"
/usr/bin/sed -i '' "s|^QPI_ROOT_DIR=.*|QPI_ROOT_DIR=\"$REPO_DIR\"|" "$AUTOMATION_DIR/.notion_sync.env"
cat "$AUTOMATION_DIR/.notion_sync.env"
```

### Step 8-4. launchd登録（24時間）

```bash
"$AUTOMATION_DIR/install_notion_sync_launchd.sh" 86400
launchctl print "gui/$(id -u)/com.kitak.obsidian.notion-sync" | rg "state =|run interval|last exit code|program"
```

### Step 8-5. 手動実行テスト

```bash
"$NOTION_RUNTIME_DIR/run_notion_sync.sh"
tail -n 50 "$NOTION_RUNTIME_DIR/logs/notion_sync.out.log"
tail -n 50 "$NOTION_RUNTIME_DIR/logs/notion_sync.err.log"
```

期待結果:

- `Done notion sync` が出る
- `00_Inbox/notion_sync/api/YYYY-MM-DD/*.md` が増える

---

## 9. Google Drive ミラー（24時間自動）

### Step 9-1. runtime配置

```bash
mkdir -p "$GDRIVE_RUNTIME_DIR/logs"
cp -a "$USER_HOME/automation_bundle/gdrive-mirror/run_gdrive_mirror.sh" "$GDRIVE_RUNTIME_DIR/"
cp -a "$USER_HOME/automation_bundle/gdrive-mirror/install_gdrive_mirror_launchd.sh" "$GDRIVE_RUNTIME_DIR/"
chmod 755 "$GDRIVE_RUNTIME_DIR/run_gdrive_mirror.sh" "$GDRIVE_RUNTIME_DIR/install_gdrive_mirror_launchd.sh"
ls -la "$GDRIVE_RUNTIME_DIR"
```

### Step 9-2. env設定

```bash
cat > "$GDRIVE_RUNTIME_DIR/.gdrive_mirror.env" <<'EOF_ENV'
GDRIVE_BASE="/Users/kitak/Library/CloudStorage/GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/共有ドライブ/wakamotolab_meeting/kitagishi"

SRC_OBSIDIAN_DAILY="/Users/kitak/Documents/Obsidian Vault/01_Daily"
SRC_LAB_MEETINGS="/Users/kitak/Desktop/presentations/lab_meetings"
SRC_CONFERENCES="/Users/kitak/Desktop/presentations/conferences"
SRC_FIGURE="/Users/kitak/Desktop/presentations/figure"
SRC_THESIS="/Users/kitak/Desktop/thesis"
EOF_ENV
```

### Step 9-3. launchd登録（terminal_openモード）

```bash
export GDRIVE_MIRROR_RUN_MODE=terminal_open
"$GDRIVE_RUNTIME_DIR/install_gdrive_mirror_launchd.sh" 86400
launchctl print "gui/$(id -u)/com.kitak.gdrive.mirror" | rg "state =|run interval|last exit code|program"
```

### Step 9-4. 手動実行テスト

```bash
"$GDRIVE_RUNTIME_DIR/run_gdrive_mirror.sh"
tail -n 80 "$GDRIVE_RUNTIME_DIR/logs/gdrive_mirror.out.log"
tail -n 80 "$GDRIVE_RUNTIME_DIR/logs/gdrive_mirror.err.log"
```

期待結果:

- `Synced: <src> -> <dst>` が5本出る
- `Done gdrive mirror` が出る

---

## 10. 作業ログ自動収集（任意）

### Step 10-1. dry-run

```bash
cd "$REPO_DIR"
python3 -m py_compile scripts/session_activity_logger.py
python3 scripts/session_activity_logger.py --dry-run --inactivity-minutes 1
```

### Step 10-2. launchd登録（例: 15分）

```bash
cd "$REPO_DIR"
bash scripts/install_activity_logger_launchd.sh 900
launchctl print "gui/$(id -u)/com.kitak.qpi.activity-logger" | rg "state =|run interval|last exit code|program"
```

ログ:

```bash
tail -n 80 "$REPO_DIR/.figure_history/activity_logger.out.log"
tail -n 80 "$REPO_DIR/.figure_history/activity_logger.err.log"
```

---

## 11. 論文・会議導線リンク（任意）

### Step 11-1. master thesis から overleaf 参照

```bash
mkdir -p "/Users/kitak/Desktop/thesis/master thesis"
cd "/Users/kitak/Desktop/thesis/master thesis"
ln -sfn "/Users/kitak/History-dependent-survival-and-adaptation-to-glucose-starvation-in-fission-yeast" overleaf
ln -sfn "/Users/kitak/History-dependent-survival-and-adaptation-to-glucose-starvation-in-fission-yeast/thesis_log.md" thesis_log.md
mkdir -p pdf_log
ls -la
```

### Step 11-2. group_meeting から results 参照

```bash
mkdir -p "/Users/kitak/Desktop/group_meeting"
ln -sfn "/Users/kitak/QPI_Omni/results" "/Users/kitak/Desktop/group_meeting/results"
ls -la "/Users/kitak/Desktop/group_meeting"
```

---

## 12. 動作検証（最終チェック）

```bash
# MCP
python3 - <<'PY'
import json, pathlib
p = pathlib.Path.home()/'.cursor'/'mcp.json'
obj = json.loads(p.read_text())
print('servers=', sorted(obj['mcpServers'].keys()))
PY

# launchd jobs
launchctl print "gui/$(id -u)/com.kitak.obsidian.notion-sync" | rg "state =|run interval|last exit code|runs ="
launchctl print "gui/$(id -u)/com.kitak.gdrive.mirror" | rg "state =|run interval|last exit code|runs ="
launchctl print "gui/$(id -u)/com.kitak.qpi.activity-logger" | rg "state =|run interval|last exit code|runs =" || true

# logs timestamp
stat -f '%Sm %N' -t '%Y-%m-%d %H:%M:%S %z' \
  "$NOTION_RUNTIME_DIR/logs/notion_sync.out.log" \
  "$GDRIVE_RUNTIME_DIR/logs/gdrive_mirror.out.log"
```

---

## 13. Windows への持ち込み（pull時の注意）

結論:

- `git pull` 自体は問題ない
- macOS専用launchd・`/Users/...` パスは Windows ではそのまま動かない
- ただし **読み込まれないだけ** で、通常は致命エラーにならない

必要対応:

1. `~/.cursor/mcp.json` を Windows 側で別途作成
2. Obsidian Vault パスを Windows 用に置換
3. launchd の代わりに Task Scheduler へ置換（必要なら）

---

## 14. トラブルシュート

### 14-1. Notion sync が `Skip notion sync` で終わる

```bash
cat "$NOTION_RUNTIME_DIR/.notion_sync.env"
python3 - <<'PY'
import json, pathlib
p=pathlib.Path('/Users/kitak/QPI_Omni/.cursor/mcp.json')
obj=json.loads(p.read_text())
print('has notionApi=', 'notionApi' in obj.get('mcpServers',{}))
PY
```

確認ポイント:

- `OBSIDIAN_VAULT_DIR` が実在するか
- mcp.json からトークンが読めるか
- DB ID が scripts 側から抽出できるか

### 14-2. gdrive mirror が Permission denied

- `GDRIVE_MIRROR_RUN_MODE=terminal_open` を使用
- 必要に応じてフルディスクアクセス付与

確認:

```bash
plutil -p "$HOME/Library/LaunchAgents/com.kitak.gdrive.mirror.plist" | rg "ProgramArguments|open|Terminal"
```

### 14-3. launchd 再登録

```bash
launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.kitak.obsidian.notion-sync.plist" || true
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.kitak.obsidian.notion-sync.plist"
launchctl kickstart -k "gui/$(id -u)/com.kitak.obsidian.notion-sync"

launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.kitak.gdrive.mirror.plist" || true
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.kitak.gdrive.mirror.plist"
launchctl kickstart -k "gui/$(id -u)/com.kitak.gdrive.mirror"
```

---

## 15. コミュニティ配布時の推奨構成

- `QPI_Omni/docs/COMMUNITY_REPRO_GUIDE.md`（この手順書）
- `QPI_Omni/docs/WORKLOG_SPEC.md`（作業ログ正本）
- `QPI_Omni/.cursor/rules/task-routing.mdc`
- `QPI_Omni/AGENTS.md`
- `QPI_Omni/CLAUDE.md`
- `Obsidian Vault/98_Automation/*`（Notion同期一式）
- `~/Library/Application Support/gdrive-mirror/*`（GDrive同期一式）

配布時は機密情報を必ず除外:

- `~/.cursor/mcp.json` のトークン
- `.notion_sync.env` のトークン
- ログファイル中の個人情報

---

## 16. 1コマンド検証（健康診断）

```bash
bash -lc '
set -e
launchctl print "gui/$(id -u)/com.kitak.obsidian.notion-sync" >/dev/null
launchctl print "gui/$(id -u)/com.kitak.gdrive.mirror" >/dev/null
python3 -m py_compile "$REPO_DIR/scripts/session_activity_logger.py"
echo "OK: core automations are installed and script is valid"
'
```

---

以上。
