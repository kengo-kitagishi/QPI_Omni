# 系譜 gallery HTML を CSV から作る（顕微鏡 PC 用）

母細胞ごとの 3 段 plot（RI または濃度 / 位相積分 dry mass / 体積）を共通 time 軸で並べた HTML を、
**手元の CSV だけから**作る手順。publish 済み master も GPU 環境も要らない。

対象は 260517 データセット（img_2 = 0 h、5 min/frame、窓 img_2〜img_2017）。

---

## 1. 前提（1 回だけ）

- **リポジトリを最新にする**（`--source csv` は 2026-09-16 に入った）:
  ```powershell
  cd $env:USERPROFILE\Documents\QPI_omni    # clone 済みの場所
  git pull
  ```
- **Python 環境**: `pandas` / `numpy` / `matplotlib` があればよい。omnipose 環境（`environment\bootstrap_windows.ps1` で作るもの）に入っている。
  無ければ任意の Python に `pip install pandas numpy matplotlib` でも動く。以下 `<py>` はその python.exe。
  - 例（omnipose 環境がある場合）: `<py>` = `%USERPROFILE%\miniconda3\envs\omnipose\python.exe`
    （解析 PC は `C:\Users\QPI\anaconda3\envs\omnipose\python.exe`）。

## 2. 使う CSV

必要なのは **consolidated の all-cells CSV** 1 本（tracker の全列が入っているもの）:

```
all_cells_lineage_data3D.csv.gz      ← 必須。pos, ch, cell_id, frame, volume_um3_efd, mean_ri, total_phase,
                                        density_pg_um3_efd, is_outlier, touches_border ... を含む
all_cells_lineage_bad_frames.csv.gz  ← 任意。drift 除外点（紫の菱形）を出したいとき
channels.csv                          ← 任意。分類フラグで解析対象を絞りたいとき（無ければ端 ch 以外を全部使う）
```

これらは master の `consolidated/`（と `derived/phase1_img0002-2017/channels.csv`）にある。
共有ドライブのミラーからそのまま読める:

```
G:\共有ドライブ\wakamotolab_meeting\kitagishi\data_master\260517\<tag>\consolidated\all_cells_lineage_data3D.csv.gz
```

`<tag>` は `...\data_master\260517\LATEST.txt` に入っている現行版（例 `v20260916_noshrink`）。
別の場所にコピーして使ってもよい。

## 3. 作る

```powershell
$py  = "$env:USERPROFILE\miniconda3\envs\omnipose\python.exe"   # 自分の python に置き換える
$cons = "G:\共有ドライブ\wakamotolab_meeting\kitagishi\data_master\260517\v20260916_noshrink\consolidated"

& $py scripts\lineage_html_gallery_260517.py `
    --source csv `
    --csv     "$cons\all_cells_lineage_data3D.csv.gz" `
    --bad-csv "$cons\all_cells_lineage_bad_frames.csv.gz" `
    --panel1 conc `
    --pos-min 1 --pos-max 104 `
    --min-mother-frames 500 `
    --max-lineages 400 `
    --out results\figures\lineage_html
```

`results\figures\lineage_html\lineage_gallery_csv_<日時>.html` ができる。ブラウザで開く:

```powershell
start results\figures\lineage_html\lineage_gallery_csv_<日時>.html
```

## 4. オプション

| 指定 | 意味 | 既定 |
|---|---|---|
| `--panel1 ri` / `conc` | 1 段目を mean RI か dry-mass 濃度 [mg/mL] にする | `ri` |
| `--pos-min N` / `--pos-max N` | Pos 範囲を絞る | 1 / 52 |
| `--min-mother-frames N` | 母細胞が窓内 N frame 未満の ch を除く（窓は 2016 frame）| 1000 |
| `--max-lineages N` | 載せる系列の上限 | 30 |
| `--ylim-ri` / `--ylim-conc` / `--ylim-mass` / `--ylim-vol` | 各軸の範囲 `lo,hi`（`auto` で自動）| `1.37,1.40` / `150,400` / `0,50` / `0,120` |
| `--bad-csv <path>` | drift 除外点（紫菱形）を出す | なし |
| `--channels-csv <path>` | 分類フラグで解析対象を絞る（cells かつ OOB 除外でない ch のみ）| なし＝端 ch 以外を全部 |
| `--out <dir>` | 出力先 | `results\figures\lineage_html` |

- **端 trap ch00 / ch11 は常に除外**される（解析対象外）。
- 全系列を載せたいときは `--max-lineages` を大きく（例 400）、`--min-mother-frames` を下げる（例 50）。
  ただし母系列が途中で切れている ch は短い切れ端しか出ない（既知の課題）。
- 体積の縮め版（0.5px）と縮めなし版（0px）は別 master。使った CSV の版がそのまま出る。

## 5. plot の見方

各段: **有効 = 濃い青の点と線 / tracker outlier = 赤い × / border = オレンジ三角 /
drift 除外 rank-1 = 紫の菱形 / 検証済み母細胞分裂 = 灰色の縦線**。
mass 段には解析対象 cycle の ln(mass) 直線 fit を赤線で描く。y 軸は全系列で共通。

- mass_pg = total_phase × 0.658 × pixel² / (2π × 0.00018) × 1e-3（pixel = 0.34567514677103717 µm）
- 濃度 [mg/mL] = density_pg_um3_efd × 1000 = (n_cell − n_medium) / α。培地切替をまたいで比較できる。

## 6. 困ったとき

- `--csv file lacks the required column 'pos'`: per-channel の `lineage_data3D.csv`（pos/ch 列なし）を渡している。
  **consolidated の all-cells CSV**（pos, ch 列あり）を使う。
- `no lineages selected`: `--min-mother-frames` が高すぎる。下げる（例 50）。
- 図が出ない / backend エラー: `pip install matplotlib` 済みか確認。表示不要（PNG を埋め込むだけ）なので画面が無くても動く。
- HTML が重い: `--pos-min` / `--pos-max` で範囲を分ける。1 系列あたり約 0.4 MB。
