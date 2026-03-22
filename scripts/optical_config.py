"""
optical_config.py — QPI光学系共通パラメータ

使い方:
    from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE, CROP_REGION

実験前に更新するもの:
    - OFFAXIS_CENTER : CursorVisualizer (get_offaxis_center.py) で取得した値
    - CROP_REGION    : 使用するクロップ領域（変わった場合のみ）

WAVELENGTH / NA / PIXELSIZE はハード変更のない限り触らない。
"""

# ============================================================
# ★ 実験前に更新するパラメータ
# ============================================================

OFFAXIS_CENTER = (1634, 532)   # (row, col) — 2026-03-21 更新

# クロップ領域 (row_start, row_end, col_start, col_end)
# カメラ位置を変えた場合は要更新
# Basler aca2440 (最大 2048 行)  → (0, 2048, 208, 2256)  = 2048×2048
# MicroManager 1.4 (余白あり)    → (8, 2056, 208, 2256)  = 2048×2048
CROP_REGION = (0, 2048, 208, 2256)

# ============================================================
# 変わらない光学パラメータ
# ============================================================

WAVELENGTH   = 658e-9           # m  (658 nm レーザー)
NA           = 0.95             # 対物レンズ開口数
PIXELSIZE    = 3.45e-6 / 40     # m/px  (センサー 3.45 µm, 40x 対物)

# ============================================================
# OFFAXIS_CENTER 履歴
# 「いつの実験のoffaxisはなんだっけ？」→ ここを見る
# ============================================================
# フォーマット: {"date": "YYYY-MM-DD", "center": (row, col), "note": "メモ"}
# 新しい実験を行うたびに先頭に追加する

OFFAXIS_HISTORY = [
    {"date": "2026-03-21", "center": (1634,  532), "note": "回折格子変更後"},
    {"date": "2026-02-28", "center": (1712,  532), "note": ""},
    {"date": "2025-12-12", "center": (1664,  485), "note": "ph_1 / Pos20"},
    {"date": "unknown",    "center": (1642,  443), "note": "realtime monitor時の値"},
    {"date": "unknown",    "center": (1623, 1621), "note": "qpi_03 batch時の値"},
    {"date": "unknown",    "center": (1504, 1708), "note": "focus setup時の値"},
    {"date": "unknown",    "center": ( 858,  759), "note": "250522 test timelapse"},
    {"date": "unknown",    "center": ( 857,  759), "note": "250528 visibility test"},
]


def get_offaxis_for_date(date_str: str):
    """
    日付文字列（'YYYY-MM-DD'）に対応する offaxis_center を返す。
    見つからない場合は現在の OFFAXIS_CENTER を返す。

    例:
        center = get_offaxis_for_date("2025-12-12")
        # → (1664, 485)
    """
    for entry in OFFAXIS_HISTORY:
        if entry["date"] == date_str:
            return entry["center"]
    return OFFAXIS_CENTER


def print_history():
    """OFFAXIS_HISTORY を一覧表示する。"""
    print(f"{'Date':<14} {'Center':<16} Note")
    print("-" * 50)
    for e in OFFAXIS_HISTORY:
        print(f"{e['date']:<14} {str(e['center']):<16} {e['note']}")


if __name__ == "__main__":
    print_history()
