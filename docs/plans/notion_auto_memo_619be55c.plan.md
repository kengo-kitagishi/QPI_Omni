---
name: Notion Auto Memo
overview: task-routing.mdc の思考メモトリガーを「キーワード検出」から「研究内容の文脈判断 + 短縮トリガー」に変更する
todos:
  - id: update-trigger
    content: task-routing.mdc の思考メモトリガーを文脈判断方式に書き換える
    status: completed
  - id: commit-push
    content: git commit & push
    status: completed
isProject: false
---

# 思考メモ自動保存の強化

## 変更箇所

[`.cursor/rules/task-routing.mdc`](c:\Users\QPI\Documents\QPI_omni\.cursor\rules\task-routing.mdc) の115〜118行目（トリガー部分）を書き換える。

## 変更前（現在）

```
以下のキーワードが含まれる場合は研究の思考メモとして保存する：
- 「気がする」「かもしれない」「気づいた」「仮説」
- 「メモして」「思ったこと」「アイデア」「ひらめいた」
- 「次はこれを試す」「こうすべきかも」「この図を見ると」
```

## 変更後

**2つのトリガーを追加：**

**A. 研究内容の文脈判断（自動）**
以下の内容を含む発言は、キーワード不要で自動保存する：
- 実験・測定・サンプル・試料・装置・光学系に関する観察や考察
- データ・図・解析結果についての気づき
- アライメント・チャネル・波長・RI・位相など研究固有の語が含まれる

**B. 短縮トリガー（明示的）**
- 「メモ」「メモして」だけでも保存する（キーワードの前後に研究内容がなくてもOK）

**保存しない例外（変更なし）：**
- Cursorの使い方・ツール設定の話題
- コードのデバッグや文法的な質問
- 完全に日常会話