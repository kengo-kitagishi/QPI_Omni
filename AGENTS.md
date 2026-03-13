<!-- AUTO-GENERATED: edit AGENTS.local.md, not AGENTS.md -->
<!-- source: /Users/kitak/dotfiles/claude/CLAUDE.md -->

# 固定スケジュール（APIで絶対に変更・削除しないこと）

## 部活（陸上）

- **火曜・木曜**: 17:00〜19:30
- **土曜**: 10:00〜12:30
- **例外**: 春休みなどの長期休暇中は時間が変則になる場合がある

### ルール
- この時間帯に他のタスクを入れない
- ClickUp・Google Calendar APIを使って部活タスクを**移動・変更・削除しない**
- 手動での変更はユーザー自身がClickUp上で行う
- 「昨日の予定を今日に移す」などの一括操作でも、部活タスクは対象外とする

---

# タスク・スケジュール管理ルール

## 大原則：予定・タスク管理はすべてClickUpを使う

**Google Calendar MCPは使用しない。**
タスクの追加・変更・削除・確認・スケジュール整理はすべて **ClickUp MCP** を使う。

## ClickUp（タスク・予定管理）

以下のキーワードが含まれる場合は **ClickUp MCP** を使ってタスクを作成する：
- 「予定を入れて」「予定を追加して」
- 「タスクを入れて」「タスクを追加して」
- 「スケジュールして」「ToDo」
- 「やること」「やるべきこと」
- 「実験の計画を入れて」「実験予定」

## ClickUpのリスト振り分け

タスクの内容に応じて以下のリストに入れる：

| 内容 | フォルダ | リスト | リストID |
|------|----------|--------|----------|
| 実験・測定・試料作製（PDMS, Bonding, 光学系など） | QPI | 3_EXPERIMENT | 901813997604 |
| コード・解析・スクリプト作業 | QPI | 4_CODE | 901813997608 |
| 計画・方針検討 | QPI | 1_PLAN | 901813997590 |
| 論文・原稿作業 | QPI | 5_MANUSCRIPT | 901813997612 |
| 勉強・論文読み | STUDY | （内容に応じて選ぶ） | — |
| ミーティング・発表 | MEETING | （内容に応じて選ぶ） | — |
| その他私生活 | OTHERs | （内容に応じて選ぶ） | — |

判断が難しい場合は `3_EXPERIMENT`（ID: 901813997604）をデフォルトとして使う。

## ClickUpタスクの時間設定

- 時間の指定がない場合: **12:00開始・2時間（14:00まで）** をデフォルトとして設定する
- 「午前」と言われた場合: 9:00開始・3時間（12:00まで）
- 「午後」と言われた場合: 13:00開始・3時間（16:00まで）
- 時間が明示された場合（例: 「10時から」）: その時間を使う

## 判断が曖昧な場合

- 「明日〇〇して」→ ClickUp（タスク）
- 「明日〇時に〇〇の通知」→ Google Calendar（アラーム）
- 両方が求められている場合は両方に作成する

## GitHub Issues（「いずれやること」の自動登録）

以下のような発言が出たとき、**確認せずに自動で GitHub Issue を作成**し、作成後にリンクだけ報告する。

### トリガーとなる言葉のパターン
- 「いずれ〜したい/すべき/しないといけない」
- 「そのうち〜する必要がある/やらないと」
- 「あとで〜実装/追加/修正/対応したい」
- 「いつか〜やらないと/やりたい」
- 「〜が気になるが今はやらない」「今はスキップするが〜」
- 「〜を忘れないようにしたい」「メモしておきたい（コードに関すること）」
- 「〜はTODO」「〜をissueに挙げて」（明示的な指示）

### トリガーにしない（Issueを作らない）
- 今すぐやること → ClickUp または直接実装
- 研究の思考・気づき → Notionの思考メモ
- 過去形の発言

### Issue作成コマンド
```bash
gh issue create \
  --title "タイトル（端的に）" \
  --body "背景と内容（会話の文脈から書く）" \
  --label "someday" \
  --repo kengo-kitagishi/QPI_Omni
```

バグ・不具合の場合は `--label "bug"` を使う。両方に該当する場合は `--label "someday,bug"`。

### 報告形式
Issue作成後は以下の1行だけ報告する（長い説明不要）：
```
Issue作成: #番号「タイトル」→ URL
```

---

## 図の保存ルール

スクリプトで図を生成・保存する際は `scripts/figure_logger.py` の `save_figure()` を使うこと。
`plt.savefig()` を直接使うのは避ける。

```python
from figure_logger import save_figure
save_figure(fig, params={"key": value, ...}, description="この図が何を示しているか")
```

保存先は `results/figures/`、ログは `docs/EXPERIMENT_LOG.md` に自動追記される。

---

## 図管理ワークフロー（figure-hub）

図のバージョン管理・反映・配布は `figure_hub.py` / `fig_register.py` / `fig_project.py` を使う。

```bash
# 修正した図を新バージョンとして登録（SVG/PDFペア対応）
python3 ~/Desktop/figure-hub/scripts/fig_register.py --id fig_xxx --src /path/to/fig.svg --note "修正内容"

# lock更新して project に反映（lock → sync）
python3 ~/Desktop/figure-hub/scripts/figure_hub.py use --project thesis_overleaf --id fig_xxx --version latest --dest "figure/xxx.pdf"
python3 ~/Desktop/figure-hub/scripts/figure_hub.py sync --project thesis_overleaf --project-root "/Users/kitak/History-dependent-survival-and-adaptation-to-glucose-starvation-in-fission-yeast"

# 図ごとに複数フォルダへ配布（lockは変更しない）
python3 ~/Desktop/figure-hub/scripts/fig_project.py deliver --id-item "fig_xxx::figure/xxx.pdf" --project-root "/path/to/dst1" --project-root "/path/to/dst2"

# 必要なら Drive に mirror
python3 ~/Desktop/figure-hub/scripts/figure_hub.py push-drive
```

### 図修正依頼のルール（重要）

**修正依頼の起点はユーザーの発言のみ。**
AIが図を見て気になる点を発見しても、Obsidianへの記録・ClickUpへのタスク化は一切行わない。

**ユーザーが「この図を直したい」と言ったとき：**
1. ユーザーの言葉をもとに以下フォーマットで整形し、Obsidianの修正依頼ファイルに追記する
2. 「記録しました」と報告するだけ。ClickUpタスク化はしない

```markdown
- fig_id: fig_xxx
  issue: 修正内容を一言で
  status: open
  targets: thesis / presentation / poster（該当するもの）
  note: 詳細・背景
```

記録先: `~/Documents/Obsidian Vault/00_Inbox/figures/figure_fix_inbox.md`

**ClickUpタスク化の自動トリガー（確認不要、自動で実行）：**

会話の冒頭で `figure_fix_inbox.md` を読み、以下の条件に該当する項目があれば自動で `5_MANUSCRIPT` にClickUpタスクを作成し、ユーザーに報告する：

| 条件 | タイミング |
|---|---|
| `status: open` の項目が記録されてから **7日以上**経過 | 週1回相当で自然に発火 |
| その図が使われている学会・提出締め切りが **2週間以内** | 締め切りベースで優先化 |

ClickUpタスク作成後、`figure_fix_inbox.md` の該当項目に `clickup_task_id: xxxxx` を追記する。

**図を修正・registerしたとき：**
ユーザーが `register` コマンドを実行したとき、またはCursorが代わりに実行したとき、
`figure_fix_inbox.md` の対応する `fig_id` のエントリを以下に更新する：

```markdown
- fig_id: fig_xxx
  issue: 〇〇を修正
  status: fixed
  note: vXXX としてregister済み（YYYY-MM-DD）
```

これをしないと、修正済みの図に対して再びClickUpタスクが作られてしまう。

**`recommend` / `clickup-sync` コマンドの自動実行は禁止。** ユーザーが明示的に依頼した時のみ実行する。

### プロジェクトとバージョンの考え方

図はlibrary内でバージョン管理され、プロジェクトごとのロックは完全に独立している。

```
fig_xxx
  ├── v001（最初の荒削り版）
  ├── v002（グループミーティング用に整えた）
  └── v003（学会用にさらに修正）

groupmeeting_2026-03-05.json  → v002 を指す（変わらない）
conference_XXX_2026.json      → v003 を指す
thesis_overleaf.json          → v003 を指す（syncで修論にコピー済み）
```

v003を登録・使用しても、groupmeetingのロックはv002のまま。プロジェクトをまたいで自動上書きは起きない。

### プロジェクト命名規則

figure-hub のプロジェクト名は以下の規則で統一する。

| 用途 | 命名規則 | 例 |
|------|----------|----|
| 修論 | `thesis_overleaf`（固定） | `thesis_overleaf` |
| グループミーティング | `groupmeeting_YYYY-MM-DD` | `groupmeeting_2026-03-05` |
| 学会発表 | `conference_<略称>_YYYY` | `conference_BSJ_2026` |
| Progress meeting | `progress_YYYY-MM-DD` | `progress_2026-03-10` |
| Journal club | `journalclub_YYYY-MM-DD` | `journalclub_2026-03-12` |
| Academic application | `academic_<機関略称>_YYYY` | `academic_EMBL_2026` |
| ポスター | `poster_<略称>_YYYY` | `poster_BSJ_2026` |

### 用途別ロックポリシー

**全用途でuse + syncを使う。**直コピーは追跡できないので使わない。

| 用途 | project-root の場所 | 終了後 |
|------|---------------------|--------|
| 修論 | `/Users/kitak/History-dependent-survival-and-adaptation-to-glucose-starvation-in-fission-yeast` | git push したらそのまま |
| グループミーティング | 都度ユーザーに確認 | freeze で固定 |
| 学会・poster | 都度ユーザーに確認 | freeze で固定 |
| progress / journal club | 都度ユーザーに確認 | freeze で固定 |
| academic application | 都度ユーザーに確認 | freeze で固定 |

**use・sync のプロジェクト指定は毎回ユーザーに確認する。**`thesis_overleaf` 以外は project-root が毎回異なるため、自動で決め打ちしない。

**終了後の freeze：**
```bash
python3 ~/Desktop/figure-hub/scripts/figure_hub.py freeze \
  --project <project_name> \
  --snapshot <project_name>_done
```

### 図修正の手順（register → use → sync）

**自動検出トリガー：**
- ファイルパス（`.pdf` / `.png` / `.svg` / `.afdesign`）が会話に登場した
- 「直した」「修正した」「できた」「書き出した」「export」などの言葉 + 図に関する文脈

**Step 1. register（確認不要・自動）**

`fig_register.py` を使う（`figure_hub.py register` の直接呼び出しは禁止）。

```bash
python3 ~/Desktop/figure-hub/scripts/fig_register.py \
  --id <fig_id> \
  --src <書き出したファイルのパス> \
  --note "<修正内容>"
```

- SVG を渡した場合: SVG→`{fig_id}_svg` + PDF自動書き出し→`{fig_id}` を同時 register、staging から削除
- 非 SVG の場合: `{fig_id}` として register、staging から削除

**Step 2. use と sync（ユーザーに確認してから実行）**

```
どのプロジェクトに反映しますか？（例: thesis_overleaf / groupmeeting_2026-03-05 / ...）
```

と聞いてから実行する。`thesis_overleaf` の場合は project-root が固定なので確認不要。

```bash
# use
python3 ~/Desktop/figure-hub/scripts/figure_hub.py use \
  --project <project_name> \
  --id <fig_id> \
  --version latest \
  --dest "<figure/内のファイル名.pdf>"

# sync
python3 ~/Desktop/figure-hub/scripts/figure_hub.py sync \
  --project <project_name> \
  --project-root "<発表資料などのルートディレクトリ>"
```

**Step 3. thesis_overleaf の場合のみ git push**
```bash
cd "/Users/kitak/History-dependent-survival-and-adaptation-to-glucose-starvation-in-fission-yeast"
git add figure/
git commit -m "Update <fig_id> <バージョン>: <修正内容>"
git push origin master
```

**Step 4. Obsidian 更新（figure_fix_inbox.md の status を fixed に）**
該当 `fig_id` のエントリを `status: fixed`・`note: vXXX register済み（日付）` に更新する。

**Step 5. figure-hub 側の自動記録を使う（確認不要・自動）**

`figure_hub.py sync` 実行後に `figure-hub` 側で以下を自動実行する（手動の Notion MCP 投稿はしない）:

- `reports/sync_post/events/*.json` に変更前後の差分を保存
- `reports/sync_post/summaries/*.md` に人間向けサマリを保存
- `sync_post_hook` が設定されていれば Notion 投稿を自動実行

**Step 6. 完了報告**
「fig_xxx vXXX を登録・<project_name> に反映しました」と1行で報告する。

---

## Notion研究ノートのまとめ方

**⚠️ 絶対ルール: Notion ページは必ず QPI Research Notes データベースに作成する。**

```
parent:
  type: data_source_id
  data_source_id: "312eda96-228e-8143-bc09-000b7c78ab26"
```

- standalone（親なし）での作成は**禁止**。private に入ってしまうため。
- `page_id` を parent に使うのも禁止（データベース外ページになる）。
- 上記 `data_source_id` を**毎回必ず**指定する。

「Notionにまとめて」「研究ノートに保存して」などと言われたら、以下のテンプレートで Notion MCP を使って `QPI Research Notes` データベースにページを作成する。

## 研究の気づき・仮説・アイデアをNotionにメモする

以下の **いずれか** に該当する発言は、**研究の思考メモ** として Notion MCP を使って保存する。キーワードを意識して使う必要はない。

**A. 研究内容を含む発言（自動判断）**
次のような内容が含まれていれば自動で保存する：
- 実験・測定・サンプル・試料・装置・光学系に関する観察や考察
- データ・図・解析結果についての気づきや解釈
- アライメント・チャネル・波長・RI・位相・カバーガラス・PDMS・Bonding など研究固有の語を含む発言
- 「〇〇したほうがいいかも」「〇〇が原因じゃないか」「〇〇を変えたら改善するかも」のような研究上の推測・改善案

**B. 短縮トリガー（明示的）**
- 「メモ」「メモして」だけで保存する（前後に研究内容がなくてもOK）
- 「気がする」「かもしれない」「気づいた」「仮説」「思ったこと」「アイデア」「ひらめいた」

**保存しない例外：**
- Cursorの使い方・ツール設定・MCP設定の話題
- コードの文法・デバッグに関する純粋な技術的質問
- 完全に日常会話（研究と無関係なもの）

### 1日1ページルール（思考メモ・研究ノート共通）

**保存前に必ず当日の既存ページを検索する。**

#### 思考メモの場合
1. Notion MCPで `[思考] YYYY-MM-DD`（今日の日付）のページを検索する
2. **見つかった場合** → そのページに新しいセクション（区切り線 + 時刻 + 内容）を追記する
3. **見つからない場合** → 新規ページ `[思考] YYYY-MM-DD` を作成する

#### 研究ノート（作業ログ）の場合
1. Notion MCPで今日の日付のページ（`[思考]`や`[会話]`プレフィックスのないもの）を検索する
2. **同じテーマの作業が続いている場合** → 既存ページに追記する
3. **新しいテーマ・別セッションの場合** → 新規ページを作成する

### 思考メモのページ構成

```
タイトル: [思考] YYYY-MM-DD

--- プロパティ ---
Date: 今日の日付
Script: 関連スクリプト or "なし"
Description: その日の思考メモのまとめ（後から更新）

--- 本文（追記していく形式）---

## HH:MM | [気づき一言]
[内容]
**文脈:** 何をやっていたときか
**次に試すかもしれないこと:** 〜〜かもしれない

---（次の思考が来たら区切り線の後に追記）---
```

**重要ルール:**
- ユーザーの言葉をそのまま使う。勝手に結論を書かない
- 「次にやること」を断定しない。あくまで「〜かもしれない」という形で書く
- 1日に何度追記しても1ページに収める

### 必ず行う: 保存と同時に過去メモを検索して提示する

`[思考]` メモを保存した後、**必ず** Notion MCP の検索機能を使って関連する過去メモを探し、以下の形式でユーザーに提示する：

```
保存しました → [Notionページへのリンク]

─── 関連する過去のメモ ───
（見つかった場合）
・[いつ] [タイトル] → [リンク]
  → 当時の気づき: 〜〜〜

（見つからなかった場合）
関連する過去メモは見つかりませんでした。これが最初の記録です。
```

### 過去メモの検索方法（キーワード一致では不十分なので以下の手順を守る）

**Step 1: 概念の展開**
ユーザーの発言から中心的な概念を理解し、その言い換え・関連語・上位概念を自分で考える。
例: 「チャネルのずれ」→「アライメント」「ドリフト」「位置合わせ」「channel shift」「ずれ補正」

**Step 2: 複数クエリで検索**
展開した語句で2〜3回 Notion を検索する（Notion MCPのsearch機能を使う）。

**Step 3: 内容で類似度を判断**
検索結果のタイトルだけでなく、本文の内容をClaudeが読んで「本当に関連しているか」を自分で判断する。
キーワードが一致しなくても概念的に近ければ関連ありとみなす。

**Step 4: 提示**
関連度が高いものだけを絞って提示する（無関係なものは省く）。
「完全に同じ議論」「一部重なる」「遠いが参考になるかも」の3段階で分けて提示する。

---

## 解析パイプラインの記録

「Notionにまとめて」「研究ノートに保存して」などと言われたとき、**当日のセッションログが存在すれば**、Notionページの本文に「解析パイプライン」セクションを追加する。

セッションログのパス: `.figure_history/session_YYYY-MM-DD.json`（今日の日付で読む）

セッションログが存在する場合、以下の形式でNotionページに追加する：

```
## 解析パイプライン（実行順）
| 時刻 | スクリプト | 内容 |
|------|-----------|------|
| 13:00 | align_and_subtract | アライメント補正・背景引き算 |
| 13:30 | 32_simple_ellipse_ri | RI計算・図出力 |
```

セッションログが存在しない場合はこのセクションを省略する。

---

## 「まとめて」トリガー — WORKLOG_SPEC形式でNotionに投稿

会話中にユーザーが **「まとめて」** と言ったとき、以下を実行する。

### 基本方針

- 作業ログ形式の正本は `~/dotfiles/docs/WORKLOG_SPEC.md` とする。
- **Claudeが会話の文脈を読み取り、Notion MCPで直接ページを作成する**（スクリプトによる自動抽出ではない）。
- 目標: コミュニティの誰でも上から順に実行して完全再現できる粒度。
- 形式を変更する場合は、ユーザーの明示許可を必須とする。

### 自動実行ステップ

1. **会話を振り返り、以下のセクションを埋める**（WORKLOG_SPEC準拠）：

   | セクション | 内容 |
   |-----------|------|
   | 前提 | OS・ユーザー・トークン取得元・バックアップ方針 |
   | 背景 | この作業を始めた理由と直前の状態 |
   | 要件定義 | 目的 / スコープ / 期待成果物 / 受け入れ条件 / 制約 / 未確定事項 |
   | 実装方針 | 採用方針 / 採用理由 / 実行順序 / 代替案と不採用理由 / リスクと緩和策 |
   | 実装手順 | Step 0, 1, 2... 各Stepに **目的・コマンド・期待結果・実結果** をセットで |
   | 検証手順と結果 | コマンド + 観測結果 + pass/fail判定 + evidence path |
   | 変更ファイル一覧 | 絶対パス + 何を変えたか + なぜ変えたか |
   | 他PCでの再現手順 | パス・環境変数・OS差分の明記 |

2. **Notion MCP でページを作成する**：
   ```
   parent:
     data_source_id: "312eda96-228e-8143-bc09-000b7c78ab26"
   properties:
     Name: [作業タイトル（日付なし・テーマを一言で）]
     date:Date:start: YYYY-MM-DD
     Type: 作業ログ
     Description: 一言要約（ユーザーの言葉で）
     Document: 修論 / 学振 / group meeting / progress meeting / conference / その他 （会話の文脈から判断）
     WorkType: figure / 執筆 / input / 実験 / コード （作業内容から判断・複数該当する場合は最も主要なもの）
   ```

3. **Notion URL を1行で報告する**：
   ```
   Notion 作業ログ投稿: <URL>
   ```

### 各Stepの書き方（必須）

```markdown
#### Step N: [作業名]
- 目的: このStepが存在する理由
```bash
# コマンド（コピペで動く形式・絶対パス使用）
```
- 期待結果: 成功したときに見えるもの
- 実結果: 実際に起きたこと
```

### 守るべきルール

- **「次回やること」「残タスク」のセクションは作らない**（ユーザーが自分で決める）
- フィードバック欄は**空欄のまま**作成する（ユーザーが後から書く）
- コマンドは**コピペで動く**形式で書く（相対パス不可・環境依存部分は明記）
- ユーザーの言葉をそのまま使う（AIが勝手に結論を書かない）
- 不確かなことは「〜と思われる」と明記する

### Quality Gate（投稿前に確認）

- [ ] 前提が埋まっている
- [ ] 要件定義に 目的 / スコープ / 期待成果物 / 受け入れ条件 / 制約 がある
- [ ] 実装手順の各 Step に 目的 / コマンド / 期待結果 / 実結果 がある
- [ ] 変更ファイル一覧が絶対パスで記録されている
- [ ] 他PCでの再現に必要な差分（パス・環境変数）が明記されている

---

## 「週次レポート」トリガー — Obsidian に保存

会話中にユーザーが **「週次レポート（作って/生成して）」** と言ったとき、以下を実行する。

### 基本方針

- レポート形式の正本は `QPI_Omni/docs/WEEKLY_LOG_SPEC.md` とする。
- **トピック（内容）単位**でまとめ、**因果の流れ**（設計→実行→図→次にこう変更→...）を可視化する。
- セッション単位で区切らず、各まとまりを **1セッション ≒ 1 Qiita 記事**程度の厚みで書く。
- 保存先: `~/Documents/Obsidian Vault/04_WeeklyReports/YYYY-Www.md`
- 特に指定がなければ **直近の月〜日（今週）** を対象とする。

### 自動実行ステップ

**Step 1: 対象週の特定**
```python
# 今日の日付から今週（月〜日）を計算
# 例: 2026-W09 → 2026-02-23 (月) 〜 2026-03-01 (日)
from datetime import date, timedelta
today = date.today()
monday = today - timedelta(days=today.weekday())
sunday = monday + timedelta(days=6)
week_label = today.strftime("%Y-W%V")
```

**Step 2: 情報収集（以下の順で読む）**

1. **週次索引**（`python3 scripts/weekly_report_hub.py` 出力）を読む。索引に列挙されたファイルを Read する。
2. **Obsidian notion_sync** を読む:
   ```bash
   ls ~/Documents/Obsidian\ Vault/00_Inbox/notion_sync/api/YYYY-MM-DD/
   ```
3. **figure-hub 新規図** を確認:
   ```bash
   python3 ~/Desktop/figure-hub/scripts/figure_hub.py list
   ```
4. **thesis git log** を確認
5. **figure_fix_inbox** を確認
6. **JSONL** から notion_sync にない作業を補足（`~/.claude/projects` 内の対象週の .jsonl）

claude_sessions .md のタイムライン（Edit/Bash の順）と figure inbox .md を突き合わせ、編集・実行・図の時刻から因果を推論する。

**Step 3: レポート作成**

収集した情報を読み、**トピックを抽出**して `docs/WEEKLY_LOG_SPEC.md` 形式で書く。

- **トピック（内容）単位**でまとめる。セッション単位で区切らない
- 各トピックで「設計 → 実行 → 図 → 次にこう変更」の**因果の流れ**を記述
- 1まとまりあたり **Qiita 記事相当の厚み**（背景・手順・コード抜粋・結果・学び）
- 編集・Bash・図の時刻から因果を推論し、その順で記述
- 「まとめ」「結論」などの形式張った見出しは使わない
- **タスクリスト形式にしない**
- 定量的に書く（可能なら「〇〇の結果、XX% / N個 が得られた」）
- 図を参照（`![[filename]]` または `[fig_id vXXX]`）
- 情報が少ないトピックは「解析中」と明記（でっち上げ禁止）

**Step 4: Obsidian に保存**

```bash
# ファイルが既に存在する場合は上書き確認をユーザーに求める
~/Documents/Obsidian\ Vault/04_WeeklyReports/YYYY-Www.md
```

**Step 5: 完了報告**
```
週次レポート作成: YYYY-Www（MM/DD–MM/DD）
保存先: ~/Documents/Obsidian Vault/04_WeeklyReports/YYYY-Www.md
トピック数: N
```

### Quality Gate（保存前に確認）

- [ ] 各トピックに因果の流れ（設計→実行→結果→次の一手）がある
- [ ] コード・コマンドの抜粋がある（再現できる程度）
- [ ] 図を適切に参照している（ある場合）
- [ ] タスクリスト形式になっていない
- [ ] 定量的な記述または「解析中」の明記がある
