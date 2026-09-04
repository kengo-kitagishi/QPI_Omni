<!-- AUTO-GENERATED: edit AGENTS.local.md, not AGENTS.md -->
<!-- source: /Users/kitak/dotfiles/claude/CLAUDE.md -->

# 固定スケジュール（APIで絶対に変更・削除しないこと）

## 部活（陸上）

- **火曜・木曜**: 17:00〜19:30
- **土曜**: 10:00〜12:30
- **例外**: 春休みなどの長期休暇中は時間が変則になる場合がある

### ルール
- この時間帯に他のタスクを入れない
- Notion API（Notion MCP）を使って部活タスクを**移動・変更・削除しない**
- 手動での変更はユーザー自身が Notion（Tasks DB）上で行う
- 「昨日の予定を今日に移す」などの一括操作（リスケ・期限の一括変更を含む）でも、部活タスクおよび繰り返しテンプレート由来の固定タスク（部活・jog 等）は常に対象外とする

---

# タスク・スケジュール管理ルール

## 大原則：予定・タスク管理はすべて Notion を使う

**Google Calendar MCP は使用しない。**
予定・タスクの追加・変更・確認・整理はすべて **Notion MCP** で、GTD の **Tasks DB** に対して行う（ClickUp はもう使わない）。**予定はすべてこの Tasks DB に入れる。**

- Tasks DB: https://www.notion.so/82b434bf5057464e888a6b3be2bc9e87
- **data_source_id: `7fb3f2db-e277-4883-b686-b364b2be9df7`**（ページ作成時の parent に必ず指定）
- 毎日の入口は「☀️ Today」ページ → https://www.notion.so/Today-36feda96228e8191be78efdb26b825a4

## タスク作成のトリガー

以下のキーワードが含まれる場合は **Notion MCP（notion-create-pages）** で Tasks DB にタスク（ページ）を作成する：
- 「予定を入れて」「予定を追加して」
- 「タスクを入れて」「タスクを追加して」
- 「スケジュールして」「ToDo」
- 「やること」「やるべきこと」
- 「実験の計画を入れて」「実験予定」

## 分類（Tag プロパティ）

ClickUp のリストの代わりに Tasks DB の **Tag**（複数選択）で振り分ける：

| 内容 | Tag |
|------|-----|
| 実験・測定・試料作製（PDMS, Bonding, 光学系 等） | `実験` |
| 論文・原稿作業 | `原稿` |
| コード・解析・計画などその他の研究 | `QPI` |
| 勉強・論文読み | `STUDY` |
| ミーティング・発表 | `MEETING` |
| 就活・応募 | `JOB APPLICATION` |
| 事務・雑務 | `ADMIN` |
| 陸上 | `部活` / `T&F` / `jog` / `weight` |
| その他私生活 | `プライベート` / `OTHERs` |

判断が難しい場合は `QPI` をデフォルトにする。

## Status（GTD の状態）

- 日時が決まった予定 → `Remind` ＋ `期限` をセット
- 今日やる → `NextAction` ／ とりあえず放り込む → `Inbox`
- 待ち → `Waiting` ／ いつか → `WishList` ／ 完了 → `Done`

迷ったら「日時あり = `Remind`」「日時なし = `Inbox`」。

## 時間設定（期限プロパティ）

`期限` は展開プロパティで指定する（`date:期限:start` に ISO-8601、時刻ありは `date:期限:is_datetime` = 1）。

- 時間の指定がない場合: **12:00開始**
- 「午前」: 9:00 開始 ／ 「午後」: 13:00 開始
- 時間が明示された場合（例: 「10時から」）: その時刻
- 終日でよい場合は is_datetime=0（日付のみ）

## 空き時間を答えるとき

「いつ空いてる？」「来週いける日は？」と聞かれたら、**Notion Tasks だけを読む。Google Calendar は見ない。**
予定はすべて Tasks DB に入っている前提で答え、Tasks DB に無いものは「無い」として扱う。
読むだけで、この確認のためにタスクは作らない。

## 判断が曖昧な場合

- 「明日〇〇して」→ Notion（Tasks DB にタスク。日時なしは `Inbox`）
- 「明日〇時に〇〇の通知」「アラーム」「リマインド」→ Siri に設定するよう案内する（Google Calendar は使わない）
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
- 今すぐやること → Notion（Tasks DB）または直接実装
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

## figure_logger JSONサイドカーへの自動コンテキスト注記

`figure_logger.py` を使うスクリプトを実行し、結果の解釈を会話で提示した後、**確認なしに自動で**対応するJSONサイドカーファイルに `context` フィールドをpatchする。

### patchする内容

```json
{
  "context": {
    "objective": "ユーザーが何を見たかったか（会話から読み取る）",
    "method": "どのスクリプトをどのデータで実行したか",
    "result": "主要な数値結果",
    "interpretation": "その数値が何を意味するか（会話での解釈をそのまま）"
  }
}
```

### JSONサイドカーの場所

```
G:\共有ドライブ\wakamotolab_meeting\kitagishi\figure-hub\inbox\YYYY-MM-DD\<script>\<run_id>\<basename>.json
```

実行ログに `inbox saved:` として表示されたパスの `.json` ファイル（`.png` と同じbasenamで拡張子が `.json`）。

### patchの方法

```python
import json
from pathlib import Path

p = Path(r"<JSONサイドカーのパス>")
meta = json.loads(p.read_text(encoding="utf-8"))
meta["context"] = {
    "objective": "...",
    "method": "...",
    "result": "...",
    "interpretation": "..."
}
p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
```

### タイミング

結果を会話でまとめた直後に実行する。同じ実行セッションで複数の図が保存された場合は、まとめてpatchする。報告は1行でよい：

```
JSONコンテキスト注記: <script> <run_id>（N件）
```

---

## 図への後付けメモ（figure inbox notes）

図を生成した後に気づいた解釈・仮説を figure inbox JSON の `notes` フィールドに追記する。

### トリガー
会話中に「〇〇スクリプトの〇〇時の図」＋ 解釈・気づきが含まれる発言。

例：
- 「14時ごろのshift_visualizeの図、飢餓期のシフト増加は細胞変形かも」
- 「昨日の36の図、背景差し引きが不完全な可能性がある」

### Claudeが行う手順

1. **特定**: `~/Documents/Obsidian Vault/00_Inbox/figure_inbox/` 内のJSONを検索
   - スクリプト名（部分一致）でフィルタ
   - `created_at_utc` を JST（UTC+9）に変換して時刻照合
   - 複数候補 → ファイル名リストを提示してユーザーに確認

2. **追記**: JSON の `notes` フィールドに追加
```json
"notes": [
  {
    "added_at": "YYYY-MM-DDTHH:MM:SS+09:00",
    "text": "ユーザーの言葉をそのまま記載"
  }
]
```
   - `notes` フィールドが存在しない場合は新規追加
   - 既存の場合は配列に append

3. **報告**: 1行のみ
```
メモ追記: 2026-03-11_shift_visualize_...f005.json → 「飢餓期のシフト...」
```

4. **Notion 思考メモにも保存**（研究内容を含む場合）: 通常の思考メモルールに従う

---

## 図を作る/保存する時（必ず）— FIGURE_SPEC 準拠

QPI_Omni の `docs/FIGURE_SPEC.md` に従う。**最低限ここで担保**（詳細・型別実例・標準対応は FIGURE_SPEC.md を読む）:
1. 図は必ず **source-data（実数値 csv/npz）と caption を一緒に保存**する（任意にしない）
2. caption に各プロット量の **「操作的定義（生データからの計算法。軸ラベルではない）」を必ず**書く
3. caption に **誤差バーの定義・n（単位付き）・統計検定名・条件（株/培地/温度/phase）** を書く

満たす標準: FAIR / journal Source Data / Ten Simple Rules / Cumming-Vaux 誤差報告 / MDAR。

---

## 図の品質ルール（最重要）

**図を生成するときは常に論文品質（publication-ready）にすること。**
品質は「努力目標」ではなく**3層のインフラで担保する**。手書きで rcParams を散発的に設定しない。

### 第1層: 見た目の地（自動・忘れ防止の本体）

- スクリプト冒頭で **`import figure_logger`（または `from figure_logger import save_figure`）すれば、`paper.mplstyle` が import 時に自動適用される**（フォント7pt・Arial/Helvetica・上右spine除去・内向きティック・Okabe-Ito・pdf.fonttype=42・figsize 89mm）。
- 実体: `~/QPI_Omni/scripts/paper.mplstyle`。手動なら `plt.style.use("paper")` でも可。
- **`subplots()` より前に import すること**（rcParams は図の生成時に焼き付くため）。

### 第2層: 色の意味（生の #xxxxxx を書かない）

- 色は必ず **`qpi_colors`** から取る。同じ条件はどの図でも同じ色にする。
  - `fate_color("divided")` / `fate_palette(...)`（survivor=青 / non_survivor=朱 / no_data=灰、同義語を自動吸収）
  - `PHASE_COLORS`（growth/starvation/recovery）, `SEQUENTIAL="cividis"`（連続量 dry mass/RI）
- jet/rainbow は禁止。

### 第3層: 種類別の作法（素の ax.scatter / ax.bar を書かない）

散布図・バー・lineage は **`qpi_plots` の関数を使う**（設計判断が内蔵済み）:
- `qp.scatter_fate(ax, x, y, fate)` … alpha・小マーカー・rasterize・白フチ・2群凡例
- `qp.bar_with_points(ax, {群:値})` … 0始まり・SEM/95%CI明示・個別点重ね
- `qp.lineage_traces(ax, t, traces, groups=)` … 薄い個別線＋濃い平均±CI帯
- `qp.add_phase_spans(ax, MEDIA_SWITCHES)`, `qp.panel_label(ax,"a")`, `qp.two_legends(...)`, `qp.new_figure("single"|"double")`

### 共通の必須事項

- **サイズ**: 単幅 89mm / 両幅 183mm。指定なければ単幅（`qp.new_figure` が実寸固定）。
- **保存**: PDF/SVG 優先（`save_figure(..., fmt="pdf")`）。PNG は 300 DPI 以上（save_figure 既定=300）。
- **エラー表示**: データがあれば SEM か 95%CI を必ず入れ、どちらか明記。
- **軸ラベル**: 単位を必ず含める（例: 時間 [h]、RI、dry mass [pg]）。
- 異なる凡例のデータは**まずパネルを分ける**。1軸に重ねるなら色=変数A・形/線種=変数Bにして `two_legends`。

### 自己検証（報告前に必ず実行・ループを閉じる）

図を生成したら、**完了報告の前に出力画像を Read して**目視チェックする。1つでも✗なら直してから報告する：
1. 上・右の spine が消えているか
2. フォント・線幅が揃い matplotlib デフォルト感がないか
3. 散布: 重なり処理（alpha/小マーカー/rasterize）されているか
4. バー: 0 から始まり、誤差バーと個別点があるか
5. 凡例が適切（枠なし・過不足なし、複数なら整理）か
6. 色が `qpi_colors` の意味どおりか（条件と色が一致）

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
# → fig_register.py 実行時に route-sync が自動実行され、登録済みの配布先に最新版が届く

# 配布先を新規登録（初回のみ）
python3 ~/Desktop/figure-hub/scripts/figure_hub.py route-add \
  --id fig_xxx \
  --root /path/to/destination/folder \
  --dest figure/fig_xxx.pdf

# 手動で全図を配布先に同期
python3 ~/Desktop/figure-hub/scripts/figure_hub.py route-sync

# 特定の版を固定したい場合のみ use + sync（thesis提出など）
python3 ~/Desktop/figure-hub/scripts/figure_hub.py use --project thesis_overleaf --id fig_xxx --version latest --dest "figure/xxx.pdf"
python3 ~/Desktop/figure-hub/scripts/figure_hub.py sync --project thesis_overleaf --project-root "/Users/kitak/History-dependent-survival-and-adaptation-to-glucose-starvation-in-fission-yeast"

# 必要なら Drive に mirror
python3 ~/Desktop/figure-hub/scripts/figure_hub.py push-drive
```

### 図修正依頼のルール（重要）

**修正依頼の起点はユーザーの発言のみ。**
AIが図を見て気になる点を発見しても、Obsidianへの記録・Notion へのタスク化は一切行わない。

**ユーザーが「この図を直したい」と言ったとき：**
1. ユーザーの言葉をもとに以下フォーマットで整形し、Obsidianの修正依頼ファイルに追記する
2. 「記録しました」と報告するだけ。Notion タスク化はしない

```markdown
- fig_id: fig_xxx
  issue: 修正内容を一言で
  status: open
  targets: thesis / presentation / poster（該当するもの）
  note: 詳細・背景
```

記録先: `~/Documents/Obsidian Vault/00_Inbox/figures/figure_fix_inbox.md`

**Notion タスク化の自動トリガー（確認不要、自動で実行）：**

会話の冒頭で `figure_fix_inbox.md` を読み、以下の条件に該当する項目があれば自動で Tasks DB（Tag=`原稿`、Status=`NextAction`、`期限`=締め切り）にタスクを作成し、ユーザーに報告する：

| 条件 | タイミング |
|---|---|
| `status: open` の項目が記録されてから **7日以上**経過 | 週1回相当で自然に発火 |
| その図が使われている学会・提出締め切りが **2週間以内** | 締め切りベースで優先化 |

Notion タスク作成後、`figure_fix_inbox.md` の該当項目に `notion_task_url: <作成したNotionページURL>` を追記する。

**図を修正・registerしたとき：**
ユーザーが `register` コマンドを実行したとき、またはCursorが代わりに実行したとき、
`figure_fix_inbox.md` の対応する `fig_id` のエントリを以下に更新する：

```markdown
- fig_id: fig_xxx
  issue: 〇〇を修正
  status: fixed
  note: vXXX としてregister済み（YYYY-MM-DD）
```

これをしないと、修正済みの図に対して再び Notion タスクが作られてしまう。

**`recommend` コマンドの自動実行は禁止。** ユーザーが明示的に依頼した時のみ実行する。

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

**日常運用は route-sync が主軸。**`use + sync` は特定バージョンを固定したいときだけ使う。直コピーは追跡できないので使わない。

| 用途 | 配布方法 | 終了後 |
|------|----------|--------|
| グループミーティング | `route-add` で配布先登録 → `route-sync`（自動）| `freeze-root` で固定 |
| 学会・poster | `route-add` で配布先登録 → `route-sync`（自動）| `freeze-root` で固定 |
| progress / journal club | `route-add` で配布先登録 → `route-sync`（自動）| `freeze-root` で固定 |
| academic application | `route-add` で配布先登録 → `route-sync`（自動）| `freeze-root` で固定 |
| 修論（バージョン固定） | `use + sync`（thesis_overleaf、特定版を固定） | git push したらそのまま |

**配布先の登録は初回だけ。以降は `fig_register.py` 実行時に自動で `route-sync` が走る。**

**発表終了後の freeze：**
```bash
python3 ~/Desktop/figure-hub/scripts/figure_hub.py freeze-root \
  --root /path/to/presentation/folder
# → FIGURES.md と sources/ を生成し、以後 route-sync の対象外になる
```

### 図修正の手順（register → route-sync）

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
  # オプション: --data /path/to/data.npz  --code /path/to/generate.py
```

- SVG を渡した場合: SVG→`{fig_id}_svg` + PDF自動書き出し→`{fig_id}` を同時 register、staging から削除
- 非 SVG の場合: `{fig_id}` として register、staging から削除
- **register 完了時に `route-sync` が自動実行**され、登録済みの全配布先に最新版が届く

**Step 2. 配布先が未登録の場合のみ route-add（ユーザーに確認してから）**

新しい発表フォルダへの配布が必要な場合は、route-addで配布先を登録する。

```bash
python3 ~/Desktop/figure-hub/scripts/figure_hub.py route-add \
  --id <fig_id> \
  --root <発表フォルダのルートパス> \
  --dest "figure/<ファイル名.pdf>"
```

登録後は次回以降 register するたびに自動で届く。

**thesis_overleaf にバージョン固定で反映する場合のみ use + sync：**
```bash
python3 ~/Desktop/figure-hub/scripts/figure_hub.py use \
  --project thesis_overleaf --id <fig_id> --version latest --dest "figure/<ファイル名.pdf>"
python3 ~/Desktop/figure-hub/scripts/figure_hub.py sync \
  --project thesis_overleaf \
  --project-root "/Users/kitak/History-dependent-survival-and-adaptation-to-glucose-starvation-in-fission-yeast"
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

`route-sync` 実行後に `figure-hub` 側で以下を自動実行する（手動の Notion MCP 投稿はしない）:

- `reports/sync_post/events/*.json` に変更前後の差分を保存
- `reports/sync_post/summaries/*.md` に人間向けサマリを保存
- `sync_post_hook` が設定されていれば Notion 投稿を自動実行

**Step 6. 完了報告**
「fig_xxx vXXX を登録・全配布先に route-sync しました」と1行で報告する。

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

- レポート形式の正本は `docs/WEEKLY_LOG_SPEC.md` とする。
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

【重大ルール】索引ファイルは 10,000〜20,000 行になることがある。**絶対に冒頭だけ読んで省略してはならない。**
行数を `wc -l` で確認し、3000 行超なら 2500 行ずつ Task agents を並列実行して全行読む。

1. **週次索引**（`python3 scripts/weekly_report_hub.py --week YYYY-Www` 出力）を全行読む。
   - 行数確認 → 3000行超の場合は 2500行ずつ並列 Task agents で分割読み
   - 索引に列挙されたセッション .md は**全文**読む（先頭 N 行だけではない）
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
- 1まとまりあたり **最低 2000 日本語文字以上、目安 3000〜5000 字**（Qiita 記事相当）
  - 「2000字」= 日本語文字で 2000 文字（ASCII 換算ではない）。もっと長くていい
  - 情報を落とさない。短くまとめようとしない
- 編集・Bash・図の時刻から因果を推論し、その順で記述
- 「まとめ」「結論」などの形式張った見出しは使わない
- **タスクリスト形式にしない**
- 定量的に書く（可能なら「〇〇の結果、XX% / N個 が得られた」）
- 図を参照（`![[filename]]` または `[fig_id vXXX]`）
- 情報が少ないトピックは「解析中」と明記（でっち上げ禁止）

**Step 4: Obsidian に保存**

```bash
# ファイルが既に存在する場合も確認なしで上書きする
~/Documents/Obsidian\ Vault/04_WeeklyReports/YYYY-Www.md
```

**Step 5: 完了報告**
```
週次レポート作成: YYYY-Www（MM/DD–MM/DD）
保存先: ~/Documents/Obsidian Vault/04_WeeklyReports/YYYY-Www.md
トピック数: N
総文字数（概算）: N 字
```

### Quality Gate（保存前に確認）

- [ ] 索引を全行読んだ（冒頭だけで省略していない）
- [ ] セッションファイルを全文読んだ（先頭 N 行だけではない）
- [ ] 各トピックに因果の流れ（設計→実行→結果→次の一手）がある
- [ ] 各トピックが最低 2000 日本語文字以上ある
- [ ] コード・コマンドの抜粋がある（再現できる程度）
- [ ] 図を適切に参照している（ある場合）
- [ ] タスクリスト形式になっていない
- [ ] 定量的な記述または「解析中」の明記がある
- [ ] Obsidian 保存はユーザー確認なしで自動実行した

---

# 研究背景・実験の種類・解析フロー

## 研究の目的

**ラベルフリーQPI（定量位相イメージング）で酵母細胞の乾燥質量（dry mass）を測定し、栄養飢餓からの回復における細胞の運命決定を調べる。**

具体的な問い：
- 栄養回復後に**分裂を再開できる細胞**と**再開できない細胞**は、飢餓時点で乾燥質量・細胞サイズがどう違うのか
- どのような細胞が生き残り、どのような細胞が生き残らないのか
- 乾燥質量の違いが生存・非生存の群の違いをどの程度説明できるか

**使用生物**: 分裂酵母（*Schizosaccharomyces pombe*）

## 測定原理

QPI（オフアクシス干渉計）で位相シフトを測定 → 屈折率（RI）を算出 → dry mass に換算

```
dry mass ∝ ∫∫ Δn(x,y) dA
Δn = n_cell - n_medium
```

- 位相シフト（ラジアン）から積分すると乾燥質量に比例する量が得られる
- 細胞ごとにセグメンテーション → ROI内で積分

## 実験の種類

### タイプA: 2% → 0% → 2%（標準飢餓・回復実験）

```
[増殖期]    → [飢餓]      → [回復期]
2% glucose    0% glucose    2% glucose
（wo_2）      （wo_0）      （wo_2）
```

- 最もよく使う主力実験
- MEDIA_SWITCHES の典型例: `(0,"wo_2"), (288,"wo_0"), (576,"wo_2")`

### タイプB: 2% → Low% → 0% → 2%（段階的飢餓）

```
[増殖期] → [中間濃度] → [飢餓] → [回復期]
2%          0.0055/0.01/0.04%    0%    2%
```

- `wo_0.0055`, `wo_0.01`, `wo_0.04` を追加使用
- 段階的にグルコースを下げてから0%にする

### 焦点確認・光学調整（単発撮影）

- 目的: 焦点が合っているか・アライメントが正しいかを確認
- スクリプト: `01_realtime_visibility_monitor.py`, `34_align_and_subtract_simple.py`
- 解析パイプラインは走らせない

### スナップショット

- 目的: 光学系の安定性確認・セットアップ確認
- 解析パイプラインは基本走らせない

## 解析パイプライン（タイムラプス）

```
生データ (img_*.tif)
    ↓ 10_batch_reconstruction_new.py
位相再構成 (output_phase/*.tif, float32, radian)
    ↓ 19_gaussian_backsub.py
背景補正 (bg_corr/*.tif)
    ↓ 36_align_and_subtract_timelapse.py
アライメント + 空チャンネル差し引き (subtracted/*.tif)
    ↓ 07_segmentation.py (Omnisegger)
セグメンテーション (inference_out/*_masks.tif)
    ↓ 32_simple_ellipse_ri.py
細胞ごとのRI・サイズ時系列 (Results.csv → 楕円近似)
    ↓ qpi_fig_*.py / Omnisegger
図生成・キモグラフ・統計解析
```

## 最終アウトプット

1. **細胞ごとのRI（屈折率）時系列** → dry mass の代理指標
2. **細胞サイズ（Major/Minor軸）時系列**
3. **分裂再開群 vs 非再開群の比較**:
   - 飢餓前・飢餓中・回復期ごとの統計的違い
   - キモグラフ（Omnisegger経由）
   - 生存・非生存を分ける予測因子としての乾燥質量・サイズ

## Omnisegger との連携

- マスク（`*_masks.tif`）と位相差し引き画像を渡す
- キモグラフ生成・細胞追跡・可視化に使用

## 用語整理

| 用語 | 意味 |
|------|------|
| `ph_1` | 細胞のタイムラプス本体フォルダ |
| `wo_*` | 培地（without cells）の空チャンネル |
| `Pos0` | 常に背景参照ポジション（細胞なし） |
| dry mass | 乾燥質量。位相シフトの積分から算出 |
| RI | 屈折率（refractive index）。dry massと線形関係 |
| アライメント | フレーム間のずれ補正（ECC法） |
| 背景差し引き | `wo_*` を引いて培地由来の位相を除去 |
| 分裂再開群 | 栄養回復後に分裂を再開した細胞 |
| 非分裂群 | 栄養回復後も分裂しなかった細胞 |

---

# データフォルダ構造・命名規則

## 生データ（Micromanager出力）

```
E:\Acquisition\kitagishi\YYMMDD\{experiment_name}\
├── Pos0/          ← 必ず空チャンネル（細胞なし・背景参照用）
├── Pos1/          ← 測定ポジション（細胞あり）
├── Pos2/
└── PosN/
```

各 Pos フォルダ内のファイル命名：
```
img_000000004_ph_000.tif
img_000000004_Default_001.tif
```
- 中間の長いゼロ列はMicromanager固有のID
- 末尾3〜5桁がフレーム番号（0-indexed）
- モード: `ph`（位相） or `Default`

## タイムラプス実験のフォルダ構造

```
YYMMDD\{experiment_name}\
├── Pos0/          ← 空チャンネル
├── ph_1/          ← メイン計測（細胞あり・タイムラプス）
│   ├── Pos1/
│   ├── Pos2/
│   └── ...
├── wo_0/          ← 空チャンネル（0% グルコース培地）
│   ├── Pos1/
│   └── ...
├── wo_2/          ← 空チャンネル（2% グルコース培地）
│   ├── Pos1/
│   └── ...
├── wo_0.0055/     ← 中間濃度（実験タイプBのみ）
├── wo_0.01/
└── wo_0.04/
```

- `ph_1` = 細胞のタイムラプス
- `wo_*` = 培地ごとの空チャンネル（背景差し引き・RI補正に使用）
- 空チャンネルと細胞チャンネルは同一ポジションで対応

## パイプライン出力フォルダ（自動生成）

```
Pos{N}/
├── output_phase/          ← 位相再構成画像（float32, ラジアン）
│   └── img_*_phase.tif
├── output_colormap/       ← カラーマップ可視化（オプション）
├── bg_corr/               ← ガウス背景補正後
│   └── *_bg_corr.tif
└── {timelapse_dir}/
    ├── aligned/           ← アライメント後
    ├── subtracted/        ← 背景差し引き後
    │   └── *_subtracted.tif
    └── subtracted_colored/ ← 可視化
```

アライメントメタデータ：`alignment_transforms.json`（shift_x, shift_y, correlation含む）

## セグメンテーション出力

```
{timelapse_dir}/
└── inference_out/
    ├── *_masks.tif         (uint16, ラベルID)
    ├── *_binary.tif        (uint8)
    └── *_overlay.tif       (RGB)
```

## グルコース濃度と wo_* の対応

| フォルダ名 | グルコース濃度 | 使用場面 |
|-----------|--------------|---------|
| `wo_2`    | 2%           | 増殖期・回復期 |
| `wo_0`    | 0%           | 飢餓期 |
| `wo_0.0055` | 0.0055%   | 実験タイプB中間 |
| `wo_0.01` | 0.01%        | 実験タイプB中間 |
| `wo_0.04` | 0.04%        | 実験タイプB中間 |

## MEDIA_SWITCHES（タイムライン定義）

```python
MEDIA_SWITCHES = [
    (0,   "wo_2"),   # 0フレーム〜: 2%グルコース
    (288, "wo_0"),   # 288フレーム〜: 0%（飢餓）
    (576, "wo_2"),   # 576フレーム〜: 2%（回復）
]
# フレーム数 = 時間(h) × 12（5分間隔 = 12枚/h）
```

## 光学定数（optical_config.py）

```python
WAVELENGTH = 658e-9          # 658nm レーザー
NA = 0.95                    # 対物レンズ NA
PIXELSIZE = 3.45e-6 / 40    # m/px（センサ3.45µm, 40x対物）
CROP_REGION = (0, 2048, 208, 2256)  # (y_start, y_end, x_start, x_end)
OFFAXIS_CENTER = (1710, 644) # オフアクシス干渉縞の中心（定期更新）
```

## ImageJ ROI解析 CSV（Results.csv）

```
Label, Major, Minor, X, Y, Angle, Slice, Area, ...
```
- `Major`, `Minor`: 楕円近似の長径・短径（ピクセル）
- `X`, `Y`: 重心座標
- `Slice`: フレーム番号（1-indexed）
