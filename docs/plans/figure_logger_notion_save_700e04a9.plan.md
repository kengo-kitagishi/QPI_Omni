---
name: figure_logger Notion save
overview: figure_logger.py に Notion 自動保存を追加する。呼び出し回数が閾値（5回）を超えたらバッチとみなしスキップ
todos:
  - id: add-notion-save
    content: figure_logger.py に _save_to_notion() とバッチ判定カウンタを追加
    status: completed
  - id: commit-push
    content: git commit & push
    status: in_progress
isProject: false
---

# figure_logger に Notion 保存を追加

## 変更ファイル

[`scripts/figure_logger.py`](c:\Users\QPI\Documents\QPI_omni\scripts\figure_logger.py)

## 追加する仕組み

`save_figure()` の末尾に Notion 保存を追加。モジュールレベルのカウンタで呼び出し回数を追跡し、閾値を超えたらスキップ。

```python
_call_count = 0          # セッション内の呼び出し回数
BATCH_THRESHOLD = 5      # これを超えたらバッチとみなしNotionスキップ

def save_figure(...):
    global _call_count
    _call_count += 1
    # ...既存の処理...
    
    if _call_count <= BATCH_THRESHOLD:
        _save_to_notion(meta, fig_path)
    else:
        if _call_count == BATCH_THRESHOLD + 1:
            print("[figure_logger] バッチ処理とみなし、以降Notion保存をスキップします")
```

## Notion保存の内容

`QPI Research Notes` データベース（ID: `312eda96228e81659726cd75b221357a`）に：

- タイトル: スクリプト名（ファイル名はDateプロパティから参照）
- Date: 今日の日付
- Script: スクリプト名
- Description: `description` 引数の内容
- 本文: パラメータ・前回差分・図ファイルパス

トークンは `.cursor/mcp.json` から動的に読み込む（既存の `_load_notion_token()` パターンを流用）。

## 注意点

- Notion API 失敗してもメイン保存（PNG・ログ）は必ず成功させる（try/except でくるむ）
- ネットワークがないPC（顕微鏡用PC等）でも動作するように