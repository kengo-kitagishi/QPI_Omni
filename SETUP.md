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

`.cursor/mcp.json` は `.gitignore` により除外されているため、**手動で作成**する必要があります。

`.cursor/mcp.json` を以下の内容で作成:

```json
{
  "mcpServers": {
    "notionApi": {
      "command": "npx",
      "args": ["-y", "@notionhq/notion-mcp-server"],
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
