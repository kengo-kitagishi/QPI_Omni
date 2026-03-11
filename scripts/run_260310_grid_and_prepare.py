"""
run_260310_grid_and_prepare.py
------------------------------
タイムラプス開始前のセットアップを一括実行する。

実行順:
  Step 1. Grid reconstruction   : GRID_DIR の全 Pos を再構成
  Step 2. Channel crop          : GRID_ORIGIN_DIR/output_phase/ に対して検出＆適用
  Step 3. prepare_drift_session : drift_config.json / positions.csv / grid_ref_crops.tif 生成

使い方:
  python run_260310_grid_and_prepare.py

次のステップ:
  MM1.4 の Script Panel で realtime_drift_mda.bsh を開いて Run する。
"""

import sys
from pathlib import Path

_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))

# ============================================================
# ★★★ 実験ごとにここを変更 ★★★
# ============================================================

# グリッド撮影ディレクトリ
GRID_DIR        = r"D:\AquisitionData\Kitagishi\260310\grid_0p5_0p5_0p1_exp200ms_1pos_EMM2_1"
GRID_BASE_LABEL = "Pos1"   # ドリフト推定・channel_crop に使う Pos ラベル
GRID_Z_INDEX    = 5        # grid_ref_crops に使う z スライス番号

# タイムラプス撮影設定
POSITIONS_FILE   = r"D:\AquisitionData\Kitagishi\260310\movetest_test_Pos1.pos"
SAVE_DIR         = r"D:\AquisitionData\Kitagishi\260310\timelapse_11day_exp200ms_1pos_EMM2"
REF_POS_INDEX    = 1    # ドリフト推定用 Pos（サンプルがいる Pos）
BG_POS_INDEX     = 0    # BG Pos（細胞なし、位相補正用）
N_TIMEPOINTS     = 3168  # 11日間 × 5分間隔 (11*24*60/5)
INTERVAL_SEC     = 300   # タイムポイント間隔 [秒]
EXPOSURE_MS      = 200.0
SETTLE_MS        = 1500
PFS_SETTLE_MS    = 2000

# セッションファイルの出力先
SESSION_DIR      = r"C:\Users\QPI\Documents\QPI_Omni\drift_session"

# ============================================================


GRID_ORIGIN_DIR = f"{GRID_BASE_LABEL}_x+0_y+0"


# ============================================================
# Step 1: Grid reconstruction
# ============================================================
print("=" * 60)
print("Step 1: Grid reconstruction")
print("=" * 60)

import pipeline_full as pf

pf.GRID_DIR                      = GRID_DIR
pf.STEP_GRID_RECONSTRUCTION      = True
pf.STEP_TIMELAPSE_RECONSTRUCTION = False
pf.STEP_CHANNEL_CROP             = False
pf.STEP_GAUSSIAN_BACKSUB         = False
pf.STEP_ALIGN_SIMPLE             = False
pf.STEP_COMPUTE_SHIFTS           = False
pf.STEP_GRID_SUBTRACT            = False
pf.GRID_SKIP_IF_EXISTS           = True   # 再構成済みならスキップ

pf.step_grid_reconstruction()
print("\nStep 1 done\n")


# ============================================================
# Step 2: Channel crop (origin only)
# ============================================================
print("=" * 60)
print(f"Step 2: Channel crop ({GRID_ORIGIN_DIR})")
print("=" * 60)

phase_dir = Path(GRID_DIR) / GRID_ORIGIN_DIR / "output_phase"
if not phase_dir.exists():
    print(f"ERROR: output_phase not found: {phase_dir}")
    sys.exit(1)

pf.CROP_W              = 40
pf.CROP_H              = 120
pf.CROP_FORCE_RECOMPUTE = True   # channels/ を削除して再検出
pf.CROP_DETECT         = True
pf.CROP_APPLY          = True

ok = pf.step_channel_crop(phase_dir)
if not ok:
    print("ERROR: channel_crop failed")
    sys.exit(1)

channel_rois_json = phase_dir / "channels" / "channel_rois.json"
if not channel_rois_json.exists():
    print(f"ERROR: channel_rois.json not generated: {channel_rois_json}")
    sys.exit(1)

print(f"channel_rois.json: {channel_rois_json}")
print("\nStep 2 done\n")


# ============================================================
# Step 3: prepare_drift_session
# ============================================================
print("=" * 60)
print("Step 3: prepare_drift_session")
print("=" * 60)

import prepare_drift_session as pds

pds.POSITIONS_FILE    = POSITIONS_FILE
pds.GRID_DIR          = GRID_DIR
pds.GRID_BASE_LABEL   = GRID_BASE_LABEL
pds.GRID_Z_INDEX      = GRID_Z_INDEX
pds.CHANNEL_ROIS_JSON = str(channel_rois_json)
pds.SESSION_DIR       = SESSION_DIR
pds.SAVE_DIR          = SAVE_DIR
pds.REF_POS_INDEX     = REF_POS_INDEX
pds.BG_POS_INDEX      = BG_POS_INDEX
pds.N_TIMEPOINTS      = N_TIMEPOINTS
pds.INTERVAL_SEC      = INTERVAL_SEC
pds.EXPOSURE_MS       = EXPOSURE_MS
pds.SETTLE_MS         = SETTLE_MS
pds.PFS_SETTLE_MS     = PFS_SETTLE_MS

pds.main()
print("\nStep 3 done\n")


print("=" * 60)
print("Setup complete.")
print(f"  drift_config.json: {SESSION_DIR}/drift_config.json")
print(f"  Save dir:          {SAVE_DIR}")
print(f"  N timepoints:      {N_TIMEPOINTS}  interval: {INTERVAL_SEC}s")
print("")
print("Next: open realtime_drift_mda.bsh in MM Script Panel and click Run.")
print("=" * 60)
