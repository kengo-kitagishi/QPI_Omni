# MicroManager対応: realtime_visibility_monitor.py 移行手順

対象ファイル: `scripts/realtime_visibility_monitor.py`

---

## Step 1: MicroManagerの保存形式を確認する

MicroManager 1.4.23 で画像保存前に確認する。

1. MicroManager を起動
2. **Tools > Options** を開く
3. "Saving format" が以下のどちらか確認する：
   - `Separate image files`（推奨・デフォルト） → **このまま Step 2 へ**
   - `Image5D / OME-TIFF`（多ページTIFF） → **Step 5-B へ**

> **確認方法（簡易）**: Snap を1枚撮って保存し、保存先に `img_000000000_*.tif` が1ファイルできていれば `Separate image files` 形式。

---

## Step 2: 保存先フォルダを確認する

MicroManager で保存先に指定しているルートフォルダのパスを確認する。

例:
```
D:\AquisitionData\Kitagishi\micromanager_seq\
```

フォルダ構造はこうなる:
```
micromanager_seq\
  ├── Pos0\
  │     ├── img_000000000_Default_000.tif
  │     ├── img_000000001_Default_000.tif
  │     └── metadata.txt
  ├── Pos1\
  │     └── ...
  └── metadata.txt
```

---

## Step 3: スクリプトの設定を変更する

`scripts/realtime_visibility_monitor.py` の先頭の設定セクションを編集する。

### 3-1: WATCH_FOLDER を変更

```python
# Before（Basler）
WATCH_FOLDER = r"d:\AquisitionData\Kitagishi\basler_image_seq"

# After（MicroManager）
WATCH_FOLDER = r"D:\AquisitionData\Kitagishi\micromanager_seq"
```

### 3-2: recursive を True に変更

`main()` 関数内:

```python
# Before
observer.schedule(event_hander, WATCH_FOLDER, recursive=False)

# After（Pos サブフォルダを監視するため）
observer.schedule(event_hander, WATCH_FOLDER, recursive=True)
```

### 3-3: metadata.txt を除外するフィルタを追加

`ImageHandlar.on_created()` 内、既存の `.tif` チェックの直後に追加:

```python
# 既存
if not (event.src_path.lower().endswith('.tif') or
        event.src_path.lower().endswith('.tiff')):
    return

# 追加（MicroManager の metadata.txt などを除外）
filename = Path(event.src_path).name
if not filename.startswith('img_'):
    return
```

---

## Step 4: 動作確認

1. スクリプトを実行する:
   ```
   python scripts/realtime_visibility_monitor.py
   ```
   → `file monitoring start` と表示されれば監視開始

2. MicroManager で **Snap** を1枚撮って保存先に保存する

3. モニター画面が更新されることを確認する
   - Visibility マップが表示される
   - コンソールに `mean visibility: 0.XXXX` が出る

4. 問題なければ連続保存（Live + Save）でリアルタイム監視を確認する

---

## Step 5-B: OME-TIFF 形式の場合（Step 1 で該当した場合のみ）

MicroManager が多ページTIFFで保存している場合、PIL では最終フレームを取り出せないため、
`ImageHandlar.on_created()` の画像読み込み部分を以下に差し替える:

```python
# Before（PIL）
img = np.array(Image.open(event.src_path))

# After（tifffile で最終ページを読む）
import tifffile
with tifffile.TiffFile(event.src_path) as tif:
    img = tif.pages[-1].asarray()
```

> **注意**: OME-TIFF は書き込み完了前に読まれる可能性があるため、
> `time.sleep(0.1)` を `time.sleep(0.3)` に延ばすと安定しやすい。

---

## 変更箇所まとめ

| 箇所 | 変更内容 |
|------|---------|
| `WATCH_FOLDER` | MicroManager の保存先ルートに変更 |
| `observer.schedule(...)` | `recursive=False` → `recursive=True` |
| `on_created()` 内フィルタ | `img_` で始まるファイルのみ処理 |
| （OME-TIFFの場合のみ）画像読み込み | PIL → tifffile に差し替え |

---

## 元に戻す方法

Basler Pylon での使用に戻すときは `WATCH_FOLDER` と `recursive` を元の値に戻すだけでよい。
変更箇所が少ないので、設定セクションに両方をコメントで残しておくのが楽:

```python
# Basler Pylon
# WATCH_FOLDER = r"d:\AquisitionData\Kitagishi\basler_image_seq"

# MicroManager
WATCH_FOLDER = r"D:\AquisitionData\Kitagishi\micromanager_seq"
```
