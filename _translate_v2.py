"""
Translate Japanese text segments to English in Python files.
Strategy: Match full Japanese segments (including particles/grammar)
and replace them, preserving surrounding code/English text.
"""
import re
import os

JP_RE = re.compile(r'[\u3041-\u3096\u30A1-\u30F6\u4E00-\u9FFF\u3000-\u303F]')

# Match contiguous Japanese text (including particles, punctuation, spaces between JP chars)
# This captures full phrases like "画像を読み込んでfloatに変換"
JP_SEGMENT = re.compile(
    r'[\u3041-\u3096\u30A1-\u30F6\u4E00-\u9FFF\u3000-\u303F]'
    r'[\u3041-\u3096\u30A1-\u30F6\u4E00-\u9FFF\u3000-\u303F\u3001\u3002'
    r'\uFF01-\uFF5E\u2018-\u201F\s]*'
    r'[\u3041-\u3096\u30A1-\u30F6\u4E00-\u9FFF\u3000-\u303F]?'
)

# ============================================================
# Full phrase translations (longest match first)
# These handle complete Japanese segments with particles
# ============================================================
PHRASE_TRANSLATIONS = [
    # ---- Common section headers ----
    ("設定パラメータ", "Configuration parameters"),
    ("パラメータ設定", "Parameter settings"),
    ("ベースディレクトリ設定", "Base directory settings"),
    ("出力ディレクトリ作成", "Create output directory"),
    ("出力ディレクトリの作成", "Create output directory"),
    ("日本語フォント設定", "Japanese font settings"),
    ("カラーマップの範囲", "Colormap range"),
    ("カラーマップ設定", "Colormap settings"),
    ("背景引き算と平均0化", "Background subtraction and zero-mean normalization"),
    ("背景画像読み込み", "Load background image"),
    ("画像読み込みとソート", "Load and sort images"),

    # ---- File I/O patterns ----
    ("ファイル読み込みとソート", "Load and sort files"),
    ("読み込みとソート", "Load and sort"),
    ("ファイルの読み込み", "Load file"),
    ("ファイル読み込み", "Load file"),
    ("ファイル一覧を取得", "Get file list"),
    ("ファイル対応", "File mapping"),
    ("画像を読み込んで", "Load image and"),
    ("画像を読み込み", "Load image"),
    ("画像の読み込み", "Load image"),
    ("画像読み込み", "Load image"),
    ("背景画像を読み込み", "Load background image"),
    ("に変換", " conversion"),
    ("として保存", " save as"),
    ("として出力", " output as"),
    ("を保存", " save"),
    ("を読み込む", " load"),
    ("を読み込み", " load"),
    ("を取得", " get"),
    ("を計算", " calculate"),
    ("を作成", " create"),
    ("を生成", " generate"),
    ("を実行", " execute"),
    ("を確認", " check"),
    ("を設定", " set"),
    ("を更新", " update"),
    ("を削除", " delete"),
    ("を追加", " add"),
    ("を表示", " display"),
    ("を描画", " draw"),
    ("を出力", " output"),
    ("を適用", " apply"),
    ("を抽出", " extract"),
    ("を解析", " analyze"),
    ("を検索", " search"),

    # ---- Docstrings ----
    ("画像を読み込んでフロートに変換", "Load TIF image and convert to float"),
    ("フロートに変換", "convert to float"),
    ("に変換する", " conversion"),
    ("メイン処理", "Main processing"),
    ("メイン実行", "Main execution"),
    ("メイン関数", "Main function"),
    ("使い方", "Usage"),
    ("戻り値", "Return value"),

    # ---- Common comment phrases ----
    ("不要な隠しファイルをスキップ", "Skip unnecessary hidden files"),
    ("の不要な隠しファイルをスキップ", "skip unnecessary hidden files"),
    ("対応するフォルダのペアを探す", "Find corresponding folder pairs"),
    ("対応するフォルダを探しています", "Searching for corresponding folders"),
    ("差分画像を保存", "Save difference images"),
    ("差分画像の保存", "Save difference images"),
    ("差分画像", "Difference image"),
    ("整列としてピング保存", "Save as aligned PNG"),
    ("カラーマップ付き保存", "Save with colormap"),
    ("固定範囲で", "With fixed range "),
    ("アライメント用", "for alignment"),
    ("指定範囲でクロップして正規化", "Crop to specified range and normalize"),
    ("番号を取得", "Get number"),
    ("全部完了", "All done"),
    ("処理完了", "Processing complete"),
    ("処理開始", "Start processing"),
    ("アライメント計算", "Alignment calculation"),
    ("アライメント実行", "Alignment execution"),
    ("アライメント適用", "Apply alignment"),
    ("アライメント処理", "Alignment processing"),
    ("アライメント結果", "Alignment result"),
    ("アライメント情報", "Alignment info"),
    ("位相再構成", "Phase reconstruction"),
    ("背景引き算", "Background subtraction"),
    ("背景差分", "Background subtraction"),
    ("サイズチェック", "Size check"),
    ("範囲チェック", "Range check"),
    ("エラーチェック", "Error check"),
    ("存在確認", "Check existence"),
    ("ディレクトリ存在確認", "Check directory existence"),

    # ---- Pos/grid processing ----
    ("番号（", "number ("),
    ("（全て）", "(all)"),
    ("全てのポジション", "all positions"),
    ("開始ポジション", "start position"),
    ("終了ポジション", "end position"),
    ("ポジション範囲", "Position range"),
    ("対応する", "corresponding "),
    ("フォルダのペア", "folder pairs"),
    ("フォルダを探す", "find folders"),
    ("フォルダ名", "folder name"),
    ("フォルダが見つかりません", "Folder not found"),
    ("が見つかりません", " not found"),
    ("が見つかりませんでした", " not found"),
    ("が存在しません", " does not exist"),
    ("が既に存在します", " already exists"),
    ("が見つかりました", " found"),
    ("スキップします", "skipping"),
    ("をスキップ", " skip"),
    ("をスキップします", " skipping"),
    ("が空です", " is empty"),
    ("読み込みました", "loaded"),
    ("保存しました", "saved"),
    ("作成しました", "created"),
    ("完了しました", "completed"),
    ("失敗しました", "failed"),
    ("成功しました", "succeeded"),
    ("開始しました", "started"),
    ("終了しました", "ended"),
    ("中断されました", "interrupted"),
    ("停止しました", "stopped"),
    ("画像を1枚も取得できませんでした。", "No images could be loaded."),
    ("取得できませんでした", "could not get"),
    ("を取得できません", " could not get"),

    # ---- Common particles/connectors ----
    ("の場合", " case"),
    ("の場合は", " case: "),
    ("の場合のみ", " only"),
    ("する場合", " when doing"),
    ("した場合", " when done"),
    ("がない場合", " if not present"),
    ("がある場合", " if present"),
    ("の方が", " is more "),
    ("ため", " because"),
    ("ために", " for"),
    ("について", " about"),
    ("に対して", " for"),
    ("として", " as"),
    ("による", " by"),
    ("によって", " by"),
    ("における", " in"),
    ("のための", " for"),
    ("に基づいて", " based on"),
    ("に基づく", " based on"),

    # ---- Print message patterns ----
    ("処理中", "Processing"),
    ("処理中：", "Processing: "),
    ("処理対象：", "Target: "),
    ("出力先：", "Output: "),
    ("入力：", "Input: "),
    ("エラー：", "Error: "),
    ("警告：", "Warning: "),
    ("完了：", "Done: "),
    ("結果：", "Result: "),
    ("合計：", "Total: "),
    ("合計", "Total"),
    ("全体", "Overall"),
    ("枚処理済み", " images processed"),
    ("枚スキップ", " images skipped"),
    ("件", " items"),
    ("枚", " images"),
    ("個", " items"),

    # ---- Technical terms (QPI domain) ----
    ("位相差", "Phase difference"),
    ("位相画像", "Phase image"),
    ("位相", "Phase"),
    ("再構成", "Reconstruction"),
    ("再構成画像", "Reconstructed image"),
    ("干渉", "Interference"),
    ("干渉縞", "Interference fringes"),
    ("波長", "Wavelength"),
    ("屈折率", "Refractive index"),
    ("偏光", "Polarization"),
    ("蛍光", "Fluorescence"),
    ("培地", "Medium"),
    ("培地差分", "Medium subtraction"),
    ("試料", "Sample"),
    ("細胞", "Cell"),
    ("楕円", "Ellipse"),
    ("楕円フィット", "Ellipse fit"),
    ("楕円体", "Ellipsoid"),
    ("体積", "Volume"),
    ("体積推定", "Volume estimation"),
    ("面積", "Area"),
    ("厚み", "Thickness"),
    ("半径", "Radius"),
    ("直径", "Diameter"),
    ("輪郭", "Contour"),
    ("輪郭抽出", "Contour extraction"),
    ("領域", "Region"),
    ("関心領域", "Region of interest"),
    ("重心", "Centroid"),
    ("重心の座標を取得", "Get centroid coordinates"),
    ("回転対称", "Rotational symmetry"),
    ("球形度", "Sphericity"),
    ("球形率", "Sphericity"),
    ("アンラップ", "Unwrap"),
    ("フーリエ変換", "Fourier transform"),
    ("ガウシアン", "Gaussian"),
    ("ヒストグラム", "Histogram"),
    ("参照フレーム", "Reference frame"),
    ("参照フレーム（最初のスライス）の座標", "Reference frame (first slice) coordinates"),
    ("正規化", "Normalization"),

    # ---- Grid/stage/drift terms ----
    ("ステージ座標", "Stage coordinates"),
    ("ステージ位置", "Stage position"),
    ("ドリフト補正", "Drift correction"),
    ("ドリフト計算", "Drift calculation"),
    ("ドリフト量", "Drift amount"),
    ("シフト量", "Shift amount"),
    ("シフト計算", "Shift calculation"),
    ("グリッド走査", "Grid scan"),
    ("グリッド位置", "Grid position"),
    ("グリッドデータ", "Grid data"),
    ("タイムラプス", "Timelapse"),
    ("タイムポイント", "Timepoint"),
    ("フレーム", "Frame"),
    ("チャネル", "Channel"),
    ("チャンネル", "Channel"),
    ("クロップ", "Crop"),
    ("マスク", "Mask"),
    ("フィルタ", "Filter"),
    ("ノイズ", "Noise"),
    ("セグメンテーション", "Segmentation"),
    ("ラベリング", "Labeling"),
    ("二値化", "Binarization"),
    ("閾値", "Threshold"),
    ("外れ値", "Outlier"),
    ("補間", "Interpolation"),
    ("平滑化", "Smoothing"),
    ("傾き補正", "Tilt correction"),
    ("傾き", "Slope/tilt"),
    ("勾配", "Gradient"),
    ("勾配除去", "Gradient removal"),

    # ---- Alignment specific ----
    ("相互相関", "Cross-correlation"),
    ("相関係数", "Correlation coefficient"),
    ("相関", "Correlation"),
    ("位置合わせ", "Alignment"),
    ("平行移動", "Translation"),
    ("回転", "Rotation"),
    ("反転", "Flip"),
    ("水平反転", "Horizontal flip"),
    ("垂直反転", "Vertical flip"),
    ("拡大", "Magnification"),
    ("縮小", "Reduction"),
    ("変換", "Transform"),
    ("逆変換", "Inverse transform"),
    ("残差", "Residual"),

    # ---- Statistics ----
    ("平均値", "Mean value"),
    ("平均", "Mean"),
    ("標準偏差", "Standard deviation"),
    ("分散", "Variance"),
    ("中央値", "Median"),
    ("最大値", "Maximum value"),
    ("最小値", "Minimum value"),

    # ---- Image processing ----
    ("ピクセル", "Pixel"),
    ("スライス", "Slice"),
    ("解像度", "Resolution"),
    ("カラーマップ", "Colormap"),
    ("グレースケール", "Grayscale"),
    ("バンドパスフィルタ", "Bandpass filter"),
    ("パディング", "Padding"),
    ("リサイズ", "Resize"),
    ("トリミング", "Trimming"),

    # ---- Common verbs/endings ----
    ("します", ""),
    ("しました", ""),
    ("ください", ""),
    ("できます", ""),
    ("ありません", "not available"),
    ("あります", "available"),
    ("です。", "."),
    ("です", ""),
    ("ます。", "."),
    ("ます", ""),

    # ---- Single kanji/common short words ----
    ("設定", "Settings"),
    ("保存", "Save"),
    ("読み込み", "Load"),
    ("書き込み", "Write"),
    ("出力", "Output"),
    ("入力", "Input"),
    ("結果", "Results"),
    ("計算", "Calculation"),
    ("実行", "Execute"),
    ("処理", "Processing"),
    ("確認", "Check"),
    ("検証", "Verification"),
    ("描画", "Draw"),
    ("可視化", "Visualization"),
    ("プロット", "Plot"),
    ("グラフ", "Graph"),
    ("表示", "Display"),
    ("比較", "Comparison"),
    ("分析", "Analysis"),
    ("解析", "Analysis"),
    ("統計", "Statistics"),
    ("データ", "Data"),
    ("ファイル", "File"),
    ("ディレクトリ", "Directory"),
    ("フォルダ", "Folder"),
    ("パス", "Path"),
    ("サイズ", "Size"),
    ("範囲", "Range"),
    ("座標", "Coordinates"),
    ("配列", "Array"),
    ("リスト", "List"),
    ("パラメータ", "Parameters"),
    ("サマリー", "Summary"),
    ("メイン", "Main"),
    ("テスト", "Test"),
    ("デバッグ", "Debug"),
    ("ログ", "Log"),
    ("エラー", "Error"),
    ("警告", "Warning"),
    ("スキップ", "Skip"),
    ("完了", "Done"),
    ("開始", "Start"),
    ("終了", "End"),
    ("成功", "Success"),
    ("失敗", "Failure"),
    ("初期化", "Initialize"),
    ("名前", "Name"),
    ("番号", "Number"),
    ("数値", "Numerical value"),
    ("行列", "Matrix"),
    ("係数", "Coefficient"),
    ("定数", "Constant"),
    ("変数", "Variable"),
    ("関数", "Function"),
    ("クラス", "Class"),
    ("モジュール", "Module"),
    ("ライブラリ", "Library"),
    ("オプション", "Option"),
    ("コマンド", "Command"),
    ("スクリプト", "Script"),
    ("バッチ", "Batch"),
    ("条件", "Condition"),
    ("状態", "State"),
    ("進捗", "Progress"),
    ("結合", "Merge"),
    ("分割", "Split"),
    ("追加", "Add"),
    ("削除", "Delete"),
    ("更新", "Update"),
    ("変更", "Change"),
    ("修正", "Fix"),
    ("取得", "Get"),
    ("検出", "Detection"),
    ("判定", "Judgment"),
    ("推定", "Estimation"),
    ("近似", "Approximation"),
    ("最適化", "Optimization"),
    ("収束", "Convergence"),
    ("反復", "Iteration"),
    ("学習", "Training"),
    ("モデル", "Model"),
    ("重み", "Weight"),
    ("ステップ", "Step"),
    ("レイヤー", "Layer"),
    ("インデックス", "Index"),
    ("キー", "Key"),
    ("値", "Value"),
    ("型", "Type"),
    ("幅", "Width"),
    ("高さ", "Height"),
    ("数", "Count"),
    ("左", "Left"),
    ("右", "Right"),
    ("上", "Top"),
    ("下", "Bottom"),
    ("前", "Before"),
    ("後", "After"),
    ("全て", "All"),
    ("一部", "Partial"),
    ("有効", "Valid"),
    ("無効", "Invalid"),
    ("自動", "Auto"),
    ("手動", "Manual"),
    ("共通", "Common"),
    ("個別", "Individual"),
    ("固定", "Fixed"),
    ("可変", "Variable"),
    ("通常", "Normal"),
    ("特殊", "Special"),
    ("標準", "Standard"),
    ("名目値", "Nominal value"),
    ("実測値", "Measured value"),
    ("基準", "Reference"),
    ("参照", "Reference"),
    ("累積", "Cumulative"),
    ("補正", "Correction"),
    ("適用", "Apply"),
    ("抽出", "Extract"),
    ("生成", "Generate"),
    ("作成", "Create"),
    ("画像", "Image"),
    ("光学", "Optical"),
    ("空間", "Spatial"),
    ("時間", "Time"),
    ("周波数", "Frequency"),
    ("振幅", "Amplitude"),
]

# Sort by length descending for longest-match-first
PHRASE_TRANSLATIONS.sort(key=lambda x: -len(x[0]))


def translate_jp_segment(segment):
    """Translate a Japanese text segment to English using phrase dictionary."""
    result = segment
    for jp, en in PHRASE_TRANSLATIONS:
        result = result.replace(jp, en)
    return result


def translate_line(line):
    """Translate Japanese segments in a line while preserving code structure."""
    if not JP_RE.search(line):
        return line

    # Find all Japanese segments and replace them
    def replace_segment(match):
        jp_text = match.group()
        translated = translate_jp_segment(jp_text)
        return translated

    return JP_SEGMENT.sub(replace_segment, line)


def translate_file(filepath):
    """Translate all Japanese text in a file."""
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    changed = 0
    new_lines = []
    for line in lines:
        new_line = translate_line(line)
        if new_line != line:
            changed += 1
        new_lines.append(new_line)

    if changed > 0:
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            f.writelines(new_lines)

    return changed


def count_remaining_jp(filepath):
    """Count lines still containing Japanese."""
    with open(filepath, encoding='utf-8') as f:
        return sum(1 for line in f if JP_RE.search(line))


def main():
    total_changed = 0
    total_remaining = 0
    files_processed = 0

    for root, dirs, files in os.walk('scripts'):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in sorted(files):
            if not f.endswith('.py'):
                continue
            filepath = os.path.join(root, f)

            with open(filepath, encoding='utf-8') as fh:
                content = fh.read()
            if not JP_RE.search(content):
                continue

            changed = translate_file(filepath)
            remaining = count_remaining_jp(filepath)
            total_changed += changed
            total_remaining += remaining
            files_processed += 1

            if remaining > 0:
                print(f"  {remaining:4d} remaining  {filepath}")

    print(f"\n{'='*60}")
    print(f"Files processed: {files_processed}")
    print(f"Lines changed: {total_changed}")
    print(f"Lines still containing Japanese: {total_remaining}")


if __name__ == "__main__":
    main()
