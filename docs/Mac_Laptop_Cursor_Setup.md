# Mac Laptop - Cursor環境セットアップ完了

**日付**: 2026年2月26日  
**スクリプト**: clickup_helper.py, .cursor/mcp.json

## 何をしたかった

Mac laptopのCursorで、ClickUpとNotionのMCP統合を設定し、研究タスク管理と実験ノート記録を自動化できる環境を構築したかった。

## 何ができた

- ✅ `~/.cursor/mcp.json`の作成と設定完了（ホームディレクトリに配置）
- ✅ Node.jsのインストール完了（v25.6.1）
- ✅ ClickUp API統合の動作確認（QPI実験パイプライン11タスク + 培地交換6タスク = 計17タスク登録成功）
- ✅ Notion API統合の設定・動作確認完了

## 作成・変更したファイル

- `~/.cursor/mcp.json`: ClickUpとNotionのAPIトークンを含む設定ファイル。ホームディレクトリに配置してCursorから自動読み込み。
- `scripts/clickup_helper.py`: ClickUpタスク作成ユーティリティ。.cursor/mcp.jsonから設定を読み込む。

## 考えたこと・判断したこと

当初はプロジェクトフォルダ内の`.cursor/mcp.json`を使おうとしたが、CursorのMCP機能はホームディレクトリ（`~/.cursor/mcp.json`）から設定を読み込む仕様だった。プロジェクトごとではなくユーザーごとの設定という位置づけ。

Notion MCP統合にはNode.jsが必須。Homebrewでインストールした（v25.6.1）。

これでMac laptopと顕微鏡PC（Windows）両方で同じ研究自動化ワークフローが使えるようになった。ClickUpでタスク管理、Notionで実験ノート記録が自動化される。

## セットアップ内容

**ClickUp統合:**
- リストID: 901813997604 (QPI > 3_EXPERIMENT)
- 機能: タスクの自動作成、日時設定、説明文付与

**Notion統合:**
- データベースID: 1645ff49cac9809d82d2dd87b4ea8e80 (QPI Research Notes)
- 機能: 研究ノート自動保存、思考メモ記録

## 動作確認済み

1. **QPI実験パイプライン（11タスク）**:
   - パラメータ確認、バッチ位相再構成、領域Crop、背景補正、アライメント計算・適用、差分画像、セグメンテーション準備・実行、ROI作成、体積・密度解析

2. **培地交換スケジュール（6タスク）**:
   - 細胞導入完了（2/27 15:00基準）
   - T0+48h: 2%培地交換（3/1 15:00）
   - T0+96h: 0.0055%培地交換（3/3 15:00）
   - T0+120h: 0%培地交換（3/4 15:00）
   - T0+168h: 2%培地交換（3/6 15:00）
   - T0+216h: 2%培地交換（3/8 15:00）

## 今後の利用方法

- **実験予定登録**: `python scripts/clickup_helper.py add --name "タスク名" --list experiment --date 2026-03-01 --start 10 --duration 2`
- **研究ノート保存**: Cursorで「Notionにまとめて」と指示
- **思考メモ**: 研究中の気づきを「メモ」と言うだけで自動保存（Notion）

## フィードバック

（空欄 - 後で自分で書く）
