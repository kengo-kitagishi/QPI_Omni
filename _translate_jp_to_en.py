"""
Bulk translate Japanese comments/strings/docstrings to English in all Python files.
Uses a dictionary of common JP->EN patterns plus context-aware translation for
domain-specific QPI/microscopy terms.
"""
import re
import os
import sys

# Japanese character detection regex
JP_RE = re.compile(r'[\u3041-\u3096\u30A1-\u30F6\u4E00-\u9FFF\u3000-\u303F]')

# ============================================================
# Translation dictionary: Japanese phrase -> English
# Organized by category for maintainability
# ============================================================

TRANSLATIONS = {
    # --- Section headers / separators ---
    "設定": "Settings",
    "設定パラメータ": "Configuration parameters",
    "パラメータ設定": "Parameter settings",
    "パラメータ": "Parameters",
    "定数設定": "Constant settings",
    "初期設定": "Initial settings",
    "基本設定": "Basic settings",
    "光学パラメータ": "Optical parameters",
    "ベースディレクトリ設定": "Base directory settings",
    "ベースディレクトリ": "Base directory",
    "出力ディレクトリ": "Output directory",
    "出力ディレクトリ作成": "Create output directory",
    "ディレクトリ存在確認": "Check directory existence",
    "サマリー": "Summary",
    "全体サマリー": "Overall summary",
    "メイン処理": "Main processing",
    "メイン": "Main",
    "メインループ": "Main loop",
    "メイン実行": "Main execution",
    "メイン関数": "Main function",
    "結果を保存": "Save results",
    "結果": "Results",
    "計算": "Calculation",
    "保存": "Save",
    "読み込み": "Load",
    "プロット": "Plot",
    "図": "Figure",
    "解像度": "Resolution",
    "再構成": "Reconstruction",
    "位相再構成": "Phase reconstruction",
    "位相": "Phase",
    "差分位相": "Differential phase",
    "アライメント": "Alignment",
    "アライメント計算": "Alignment calculation",
    "アライメント適用": "Apply alignment",
    "アライメント実行": "Alignment execution",
    "カラーマップ": "Colormap",
    "カラーマップ設定": "Colormap settings",
    "カラーマップの範囲": "Colormap range",
    "範囲チェック": "Range check",
    "サイズチェック": "Size check",
    "背景引き算": "Background subtraction",
    "背景差分": "Background subtraction",
    "背景画像読み込み": "Load background image",
    "背景引き算と平均0化": "Background subtraction and zero-mean",
    "画像読み込み": "Load image",
    "画像の読み込み": "Load image",
    "画像を読み込み": "Load image",
    "ファイル読み込み": "Load file",
    "ファイル読み込みとソート": "Load and sort files",
    "日本語フォント設定": "Japanese font settings",
    "正規化OFF": "Normalization OFF",
    "ポンペの球形率": "Pompe sphericity",
    "マイナスキャンの符号を反転": "Flip sign for minus scan",
    "固定範囲でuint8に変換（アライメント用）": "Convert to uint8 with fixed range (for alignment)",
    "指定範囲でクロップして正規化（0-255）": "Crop to specified range and normalize (0-255)",
    "番号を取得": "Get number",
    "全部完了": "All done",
    "処理完了": "Processing complete",
    "完了": "Done",
    "開始": "Start",
    "実行": "Execute",
    "処理中": "Processing",
    "スキップ": "Skip",
    "エラー": "Error",
    "警告": "Warning",
    "注意": "Caution",
    "ファイルパス": "File path",
    "保存先": "Save destination",
    "保存（オプション）": "Save (optional)",
    "保存（カラーマップ付き）": "Save (with colormap)",
    "出力": "Output",
    "入力": "Input",
    "変換": "Conversion",
    "補正": "Correction",
    "検証": "Verification",
    "可視化": "Visualization",
    "描画": "Drawing",
    "表示": "Display",
    "比較": "Comparison",
    "分析": "Analysis",
    "解析": "Analysis",
    "統計": "Statistics",
    "平均": "Mean",
    "標準偏差": "Standard deviation",
    "分散": "Variance",
    "中央値": "Median",
    "最大値": "Maximum",
    "最小値": "Minimum",
    "閾値": "Threshold",
    "係数": "Coefficient",
    "座標": "Coordinates",
    "軸": "Axis",
    "行列": "Matrix",
    "配列": "Array",
    "データ": "Data",
    "ファイル": "File",
    "フォルダ": "Folder",
    "ディレクトリ": "Directory",
    "パス": "Path",
    "名前": "Name",
    "番号": "Number",
    "数": "Count",
    "幅": "Width",
    "高さ": "Height",
    "サイズ": "Size",
    "範囲": "Range",
    "最大": "Maximum",
    "最小": "Minimum",
    "全体": "Overall",
    "一部": "Partial",
    "各": "Each",
    "全て": "All",
    "全部": "All",
    "なし": "None",
    "あり": "Present",
    "有効": "Valid",
    "無効": "Invalid",
    "成功": "Success",
    "失敗": "Failure",
    "不明": "Unknown",
    "未定": "Undecided",
    "対象": "Target",
    "除外": "Exclude",
    "含む": "Include",
    "対応": "Corresponding",
    "基準": "Reference",
    "参照": "Reference",
    "現在": "Current",
    "前回": "Previous",
    "次回": "Next",
    "初回": "First time",
    "最終": "Final",
    "既存": "Existing",
    "新規": "New",
    "追加": "Add",
    "削除": "Delete",
    "更新": "Update",
    "変更": "Change",
    "修正": "Fix",
    "確認": "Confirm",
    "取得": "Get",
    "送信": "Send",
    "受信": "Receive",
    "接続": "Connect",
    "切断": "Disconnect",
    "開始Pos番号（None=全て）": "Start Pos number (None=all)",
    "終了Pos番号（None=全て）": "End Pos number (None=all)",
    "フォルダ名（例: \"Pos1\"）": "Folder name (e.g., \"Pos1\")",
    "存在しません": "does not exist",
    "見つかりません": "not found",
    "見つかりませんでした": "not found",
    "スキップします": "skipping",
    "読み込みました": "loaded",
    "保存しました": "saved",
    "作成しました": "created",
    "画像を1枚も取得できませんでした。": "No images could be loaded.",
    "不要な隠しファイルをスキップ": "Skip unnecessary hidden files",
    "対応するPosフォルダのペアを探す": "Find corresponding Pos folder pairs",
    "対応するPosフォルダを探しています...": "Searching for corresponding Pos folders...",
    "差分画像を保存": "Save difference images",
    "整列としてPNG保存": "Save as aligned PNG",
    "整列としてPNG保存（1=全部、10=10枚に1枚）": "Save as aligned PNG (1=all, 10=every 10th)",
    "カラーマップ付きPNG保存": "Save with colormap as PNG",
    "最初": "first",
    "最後": "last",
    "画像": "Image",
    "位相画像": "Phase image",
    "使い方": "Usage",
    "戻り値": "Return value",
    "引数": "Arguments",
    "必須": "Required",
    "任意": "Optional",
    "デフォルト": "Default",
    "以上": "or more",
    "以下": "or less",
    "未満": "less than",
    "超過": "exceeding",
    "間": "between",
    "ステージ": "Stage",
    "チャネル": "Channel",
    "ドリフト": "Drift",
    "シフト": "Shift",
    "グリッド": "Grid",
    "タイムラプス": "Timelapse",
    "クロップ": "Crop",
    "セグメンテーション": "Segmentation",
    "マスク": "Mask",
    "フィルタ": "Filter",
    "ノイズ": "Noise",
    "信号": "Signal",
    "干渉": "Interference",
    "波長": "Wavelength",
    "屈折率": "Refractive index",
    "位相差": "Phase difference",
    "楕円": "Ellipse",
    "体積": "Volume",
    "面積": "Area",
    "厚み": "Thickness",
    "半径": "Radius",
    "直径": "Diameter",
    "輪郭": "Contour",
    "境界": "Boundary",
    "領域": "Region",
    "関心領域": "Region of interest",
    "細胞": "Cell",
    "試料": "Sample",
    "培地": "Medium",
    "蛍光": "Fluorescence",
    "位相コントラスト": "Phase contrast",
    "回転対称": "Rotational symmetry",
    "球形度": "Sphericity",
    "推定": "Estimation",
    "近似": "Approximation",
    "フィッティング": "Fitting",
    "最適化": "Optimization",
    "収束": "Convergence",
    "反復": "Iteration",
    "相関": "Correlation",
    "相互相関": "Cross-correlation",
    "アンラップ": "Unwrapping",
    "フーリエ変換": "Fourier transform",
    "バンドパスフィルタ": "Bandpass filter",
    "ガウシアン": "Gaussian",
    "平滑化": "Smoothing",
    "微分": "Differentiation",
    "勾配": "Gradient",
    "傾き": "Slope",
    "切片": "Intercept",
    "残差": "Residual",
    "外れ値": "Outlier",
    "ヒストグラム": "Histogram",
    "スペクトル": "Spectrum",
    "ピクセル": "Pixel",
    "フレーム": "Frame",
    "スライス": "Slice",
    "タイムポイント": "Time point",
    "レイヤー": "Layer",
    "チャンネル": "Channel",
    "ファイル名": "Filename",
    "拡張子": "Extension",
    "圧縮": "Compression",
    "展開": "Expand",
    "回転": "Rotation",
    "反転": "Flip",
    "平行移動": "Translation",
    "拡大": "Magnification",
    "縮小": "Reduction",
    "クリップ": "Clip",
    "パディング": "Padding",
    "正規化": "Normalization",
    "標準化": "Standardization",
    "二値化": "Binarization",
    "ラベリング": "Labeling",
    "重心": "Centroid",
    "中心": "Center",
    "上": "top",
    "下": "bottom",
    "左": "left",
    "右": "right",
    "水平": "horizontal",
    "垂直": "vertical",
    "前": "before",
    "後": "after",
    "時間": "Time",
    "空間": "Spatial",
    "周波数": "Frequency",
    "振幅": "Amplitude",
    "位相（rad）": "Phase (rad)",
    "名目値": "Nominal values",
    "実測値": "Measured values",
    "使用": "Use",
    "利用": "Use",
    "実行中": "Running",
    "処理開始": "Start processing",
    "処理終了": "End processing",
    "初期化": "Initialize",
    "終了": "End",
    "中断": "Interrupt",
    "継続": "Continue",
    "再試行": "Retry",
    "復帰": "Recover",
    "例外": "Exception",
    "例外処理": "Exception handling",
    "ログ": "Log",
    "デバッグ": "Debug",
    "テスト": "Test",
    "検出": "Detection",
    "判定": "Judgment",
    "分類": "Classification",
    "予測": "Prediction",
    "学習": "Training",
    "モデル": "Model",
    "重み": "Weight",
    "パッチ": "Patch",
    "バッチ": "Batch",
    "エポック": "Epoch",
    "ステップ": "Step",
    "イテレーション": "Iteration",
    "最適": "Optimal",
    "最良": "Best",
    "最悪": "Worst",
    "条件": "Condition",
    "状態": "State",
    "進捗": "Progress",
    "結合": "Merge",
    "分割": "Split",
    "連結": "Concatenation",
    "要素": "Element",
    "インデックス": "Index",
    "辞書": "Dictionary",
    "リスト": "List",
    "タプル": "Tuple",
    "キー": "Key",
    "値": "Value",
    "型": "Type",
    "整数": "Integer",
    "浮動小数点": "Floating point",
    "文字列": "String",
    "真": "True",
    "偽": "False",
    "空": "Empty",
    "存在": "Exists",
    "有": "Present",
    "無": "Absent",
    "可能": "Possible",
    "不可能": "Impossible",
    "必要": "Necessary",
    "不要": "Unnecessary",
    "自動": "Automatic",
    "手動": "Manual",
    "共通": "Common",
    "個別": "Individual",
    "固定": "Fixed",
    "可変": "Variable",
    "動的": "Dynamic",
    "静的": "Static",
    "一時的": "Temporary",
    "恒久的": "Permanent",
    "通常": "Normal",
    "特殊": "Special",
    "標準": "Standard",
    "カスタム": "Custom",
    "オプション": "Option",
    "コマンド": "Command",
    "スクリプト": "Script",
    "ライブラリ": "Library",
    "モジュール": "Module",
    "パッケージ": "Package",
    "フレームワーク": "Framework",
    "インターフェース": "Interface",
    "クラス": "Class",
    "メソッド": "Method",
    "関数": "Function",
    "変数": "Variable",
    "プロパティ": "Property",
    "属性": "Attribute",
    "グラフ": "Graph",
    "チャート": "Chart",
    "テーブル": "Table",
    "行": "Row",
    "列": "Column",
}

def translate_line(line):
    """Translate Japanese text in a single line to English."""
    if not JP_RE.search(line):
        return line

    original = line

    # Sort translations by length (longest first) to avoid partial matches
    sorted_translations = sorted(TRANSLATIONS.items(), key=lambda x: -len(x[0]))

    for jp, en in sorted_translations:
        if jp in line:
            line = line.replace(jp, en)

    return line


def translate_file(filepath):
    """Translate all Japanese text in a file, return count of changed lines."""
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

            # Check if file has Japanese
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
