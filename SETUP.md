# QPI_omni セットアップ手順

このリポジトリをクローンした後、以下の手順で環境を構築してください。

---

## 1. リポジトリのクローン

```bash
git clone https://github.com/kengo-kitagishi/QPI_Omni.git
cd QPI_Omni
```

---

## 2. Python 環境

```bash
pip install -r requirements.txt
```

必要なパッケージ: `matplotlib`, `numpy`, `pandas`, `requests` など（requirements.txtを参照）

---

## 3. Node.js のインストール（Notion MCP に必要）

[https://nodejs.org](https://nodejs.org) から LTS 版をダウンロードしてインストール。

確認:
```bash
node --version
npm --version
```

---

## 4. gh CLI のインストール（GitHub Issues 自動登録に必要）

**Windows:**
```powershell
winget install --id GitHub.cli
```

**Mac:**
```bash
brew install gh
```

インストール後、GitHubアカウントで認証:
```bash
gh auth login
# → GitHub.com → HTTPS → Login with a web browser の順に選択
```

---

## 5. Cursor の MCP 設定

MCP設定ファイルは**ホームディレクトリ**に作成する必要があります（プロジェクトフォルダ内ではありません）。

- **Windows**: `C:\Users\<ユーザー名>\.cursor\mcp.json`
- **Mac**: `~/.cursor/mcp.json`

フォルダが存在しない場合は先に作成してください:

```powershell
# Windows（PowerShell）
mkdir "$env:USERPROFILE\.cursor"
```

以下の内容で `mcp.json` を作成:

```json
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
```

> **注意**: `@notionhq/notion-mcp-server`（公式パッケージ）には既知のバグがあります。必ず `@mieubrisse/notion-mcp-server`（修正済みフォーク）を使用してください。

`<NOTION_TOKEN>` と `<CLICKUP_TOKEN>` は管理者（研究室の先輩）から受け取ってください。

---

## 6. Cursor の再起動

MCP設定を反映させるため、Cursor を完全に再起動（終了→再起動）してください。

---

## 7. 動作確認

- Cursorのチャットで「明日10時にアラームを入れて」→ Google Calendar MCP が動作
- 「PDMSのカット実験の予定を入れて」→ ClickUp にタスク作成
- 「〇〇はいずれ実装したい」→ GitHub Issues に自動登録
- 「notionにまとめて」→ Notion に研究ノートを保存

---

## ファイル構成（主要なもの）

| ファイル | 役割 |
|---|---|
| `scripts/figure_logger.py` | 図の保存 + EXPERIMENT_LOG.md への自動追記 |
| `scripts/clickup_helper.py` | ClickUp タスク作成ユーティリティ |
| `.cursor/rules/task-routing.mdc` | Cursor の AIルーティングルール |
| `docs/EXPERIMENT_LOG.md` | 実験ログ（figure_logger.py が自動更新） |

---

## トークンの取得方法（引き継ぎ用メモ）

- **Notion トークン**: [https://www.notion.so/my-integrations](https://www.notion.so/my-integrations) → 「QPI Research」インテグレーション → Internal Integration Key
- **ClickUp トークン**: ClickUp 設定 → Apps → API Token
- **GitHub**: `gh auth login` でブラウザ認証（トークン不要）

---

## 顕微鏡PC（Windows）でのセットアップ手順

### 1. リポジトリの更新（初回はclone、2回目以降はpull）

```powershell
# 初回
git clone https://github.com/kengo-kitagishi/QPI_Omni.git
cd QPI_Omni

# 2回目以降（既にclone済みの場合）
git pull
```

### 2. Node.js のインストール確認

[https://nodejs.org](https://nodejs.org) から LTS版をダウンロードしてインストール。

```powershell
node --version
npm --version
```

### 3. MCP設定ファイルの作成

```powershell
# フォルダ作成
mkdir "$env:USERPROFILE\.cursor"

# ファイル作成（メモ帳で開く）
notepad "$env:USERPROFILE\.cursor\mcp.json"
```

以下の内容を貼り付けて保存（トークンは先輩から受け取る）:

```json
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
```

### 4. Cursor を再起動

Cursor を完全に終了してから再起動してください。

### 5. 動作確認

- 「PDMSのカット実験の予定を入れて」→ ClickUp にタスク作成
- 「notionにまとめて」→ Notion に研究ノートを保存

VSCode + Claude 拡張を使う場合は、下記「Claude Code CLI のセットアップ」を参照してください。

---

## Mac でのセットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/kengo-kitagishi/QPI_Omni.git
cd QPI_Omni
```

### 2. Python 環境

Anaconda または Miniforge をインストール後：

```bash
pip install -r requirements.txt
```

### 3. Node.js のインストール

```bash
brew install node
node --version  # 確認
```

### 4. gh CLI のインストールと認証

```bash
brew install gh
gh auth login
# → GitHub.com → HTTPS → Login with a web browser の順に選択
```

### 5. Cursor の MCP 設定（Cursor を使う場合）

`~/.cursor/mcp.json`（ホームディレクトリ）を以下の内容で作成（トークンは先輩から受け取る）：

```bash
mkdir -p ~/.cursor
```

```json
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
```

Cursor を再起動して完了。

---

## Claude Code CLI のセットアップ（Windows / Mac 共通）

Claude Code CLI はターミナルから使える Anthropic 公式の AI ツールです。Cursor を使わない環境（顕微鏡PC の VSCode など）で利用できます。

### 1. インストール

```bash
npm install -g @anthropic-ai/claude-code
```

### 2. MCP 設定ファイルの作成

**Windows:** `C:\Users\<あなたのユーザー名>\.claude.json`
**Mac:** `~/.claude.json`

以下の内容で作成（`.cursor/mcp.json` と同じトークンを使う）：

```json
{
  "mcpServers": {
    "notionApi": {
      "command": "npx",
      "args": ["-y", "@mieubrisse/notion-mcp-server"],
      "env": {
        "OPENAPI_MCP_HEADERS": "{\"Authorization\": \"Bearer <NOTION_TOKEN>\", \"Notion-Version\": \"2022-06-28\"}"
      }
    }
  }
}
```

### 3. ルールの確認

`CLAUDE.md`（プロジェクトルートに存在）は `git clone` で自動取得されます。追加作業不要です。

### 4. 起動

プロジェクトフォルダ内でターミナルを開き：

```bash
claude
```

### 5. 動作確認

- 「PDMSのカット実験の予定を入れて」→ ClickUp にタスク作成
- 「notionにまとめて」→ Notion に研究ノートを保存
- 「〇〇はいずれ実装したい」→ GitHub Issues に自動登録
