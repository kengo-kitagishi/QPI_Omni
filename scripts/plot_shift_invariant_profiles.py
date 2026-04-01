"""
plot_shift_invariant_profiles.py
--------------------------------
シフト量が大きく異なる2枚のフレームを選び、その中心横線プロファイルを重ねて表示する。
「プロットの形がシフト量に依存せず一定である」ことを示す。

使い方:
  python plot_shift_invariant_profiles.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image

_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))

from figure_logger import save_figure, _resolve_inbox_root


def _find_latest_data_npz(desc_pattern: str) -> Path | None:
    """figure_logger inbox から desc_pattern を含む最新の _data.npz を返す。"""
    inbox_root = _resolve_inbox_root()
    if inbox_root is None:
        return None
    script = "plot_shift_invariant_profiles"
    # glob: inbox_root/<date>/script/<run_id>/*_data.npz  (date降順で最新を優先)
    script_dirs = sorted(inbox_root.glob(f"*/{script}"), reverse=True)
    for sd in script_dirs:
        for npz in sorted(sd.glob("**/*_data.npz"), key=lambda p: p.stat().st_mtime, reverse=True):
            if desc_pattern in npz.name:
                return npz
    return None


# ベースパスと対象 Pos
BASE_DIR = Path(r"C:\ph_260327")
POS_LABELS = ["Pos1"]

# Pos0 に output_phase/channels がない場合、この Pos のメタデータをフォールバックに使う
FALLBACK_POS = "Pos1"

CHANNEL_INDEX = 1  # 2つめのchannel（0-indexed: 0=1番目, 1=2番目）
TL_Z_INDEX = 0  # img_*_ph_000_phase.tif
FINAL_CROP_W = 40  # crop_subtracted の行数（中心行 = FINAL_CROP_W // 2）

Y_SCALE = None  # None で自動
N_FRAMES = 2
# "diverse": y が似て x が遠い2枚  /  "similar": シフト量が似た2枚
SHIFT_MODE = "diverse"   # x シフトが最も異なる2フレームを選ぶ（強いテスト）
# similar モードで、フレーム番号の差がこれ以上離れたペアのみ候補にする
MIN_FRAME_GAP = 20

# ---- X-tilt 補正版 timelapse プロファイル設定 ----
TL_PHASE_DIR_XTILT = Path(r"D:\AquisitionData\Kitagishi\basler_image_seq\ph_3\Pos0\output_phase")
TL_ROIS_JSON_XTILT = Path(
    r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1"
    r"\Pos1_x+0_y+0\output_phase\channels\channel_rois.json"
)
TL_CH_INDICES_XTILT = [0, 3, 6, 9]   # 可視化する代表チャネル（4ch）
TL_PICK_TPS         = [0, 9, 19]      # 先頭・中間・末尾の TP
TL_CROP_H_TILT      = 270             # X 方向拡大サイズ [px]
TL_FIT_FRAC_XTILT   = 1 / 3          # 左 1/3 (90pt) = BG 領域

# NPZ差分プロット: 以下の desc_pattern で最新の _data.npz を自動検索して差し引く
# (desc_pattern1, label1, desc_pattern2, label2)
NPZ_DIFF_PAIRS = [
    ("Pos1_grid_subtr", "Pos1 grid_sub", "Pos0_raw_raw", "Pos0 raw_raw"),
    ("Pos1_grid_subtr", "Pos1 grid_sub", "Pos1_raw_raw", "Pos1 raw_raw"),
    # aligned_raw 比較 (plot 1, 2) — ファイル名省略に合わせ "aligned_ra" で検索
    ("Pos1_raw_raw", "Pos1 raw_raw", "Pos1_aligned_ra", "Pos1 aligned_raw"),
    ("Pos0_raw_raw", "Pos0 raw_raw", "Pos0_aligned_ra", "Pos0 aligned_raw"),
]

# 4-way diff: (pat1 - pat2) - (pat3 - pat4)
# → (Pos1 raw_raw - Pos0 raw_raw) - (Pos1 aligned_raw - Pos0 aligned_raw)
NPZ_QUAD_DIFF_PAIRS = [
    # profiles_ プレフィックスで diff NPZ と区別する
    ("profiles_Pos1_raw_raw", "Pos1 raw", "profiles_Pos0_raw_raw", "Pos0 raw",
     "profiles_Pos1_aligned_ra", "Pos1 aligned", "profiles_Pos0_aligned_ra", "Pos0 aligned"),
]


def _xtilt_correct_big_crop(
    phase_img: np.ndarray,
    cy: int, cx: int,
    crop_w: int,
    crop_h_tilt: int = 270,
    fit_frac: float = 1 / 3,
) -> tuple[np.ndarray, float]:
    """
    横長 270px crop の左 1/3 で線形フィット → slope * x を引いて返す。
    戻り値: (corrected_big_crop [crop_w × crop_h_tilt], slope)
    """
    big = extract_rect_roi(phase_img, cy, cx, crop_w, crop_h_tilt).astype(np.float64)
    x_prof = big.mean(axis=0)
    n_fit = max(3, int(len(x_prof) * fit_frac))
    slope, _ = np.polyfit(np.arange(n_fit, dtype=float), x_prof[:n_fit], 1)
    x_local = np.arange(crop_h_tilt, dtype=float)
    return big - slope * x_local[np.newaxis, :], float(slope)


def load_tiff_stack(path: str) -> np.ndarray:
    img = Image.open(path)
    frames = []
    for i in range(1000):
        try:
            img.seek(i)
            frames.append(np.array(img))
        except EOFError:
            break
    return np.stack(frames, axis=0)


def extract_rect_roi(img: np.ndarray, cy: int, cx: int,
                     crop_w: int, crop_h: int) -> np.ndarray:
    """(cx, cy) を中心とした crop_w × crop_h の矩形を切り出す。"""
    h, w = img.shape
    y1 = max(0, cy - crop_w // 2)
    y2 = min(h, y1 + crop_w)
    x1 = max(0, cx - crop_h // 2)
    x2 = min(w, x1 + crop_h)
    crop = img[y1:y2, x1:x2]
    if crop.shape[0] < crop_w or crop.shape[1] < crop_h:
        crop = np.pad(crop, ((0, crop_w - crop.shape[0]), (0, crop_h - crop.shape[1])),
                     mode="constant", constant_values=0)
    return crop


def _load_shift_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    シフトデータを読み込み (sx, sy, frame_indices) を返す。
    .json: pos_shifts.json の frame_results から shift_x_avg, shift_y_avg
    .npz: pos_shifts_corr_data.npz の pass2_shift_x/y（channel_index でフィルタ可）
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        frame_results = data.get("frame_results") or data.get("alignment_results")
        if not frame_results:
            raise ValueError(f"frame_results not found in {path}")
        sx, sy, frame_indices = [], [], []
        for i, r in enumerate(frame_results):
            vx = r.get("shift_x_avg") or r.get("shift_x")
            vy = r.get("shift_y_avg") or r.get("shift_y")
            if vx is None and vy is None:
                continue
            sx.append(float(vx or 0))
            sy.append(float(vy or 0))
            frame_indices.append(r.get("frame_index", i))
        return np.array(sx), np.array(sy), np.array(frame_indices)

    # NPZ
    data = np.load(str(path))
    sx = np.array(data["pass2_shift_x"])
    sy = np.array(data["pass2_shift_y"])
    if "ch" in data.files and CHANNEL_INDEX is not None:
        ch = data["ch"]
        t = data["t"]
        mask = ch == CHANNEL_INDEX
        if not np.any(mask):
            raise ValueError(f"No records for channel {CHANNEL_INDEX} in {path}")
        sx = sx[mask]
        sy = sy[mask]
        frame_indices = t[mask]
    else:
        frame_indices = np.arange(len(sx))
    return sx, sy, frame_indices


def _pick_frames(path: Path, n_pick: int, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    mode="diverse": y が似て x が遠い 2 枚
    mode="similar": シフト量が似た 2 枚（|Δx|+|Δy| が最小のペア）
    戻り値: (indices, shifts_x, shifts_y)
    """
    sx, sy, frame_indices = _load_shift_data(path)
    if len(sx) < n_pick:
        raise ValueError(f"Not enough frames with shift data: {len(sx)} < {n_pick}")

    if mode == "similar":
        # シフト量が似た2枚: |Δx| + |Δy| を最小化（フレーム差 >= MIN_FRAME_GAP のペアのみ）
        best_dist = np.inf
        best_i, best_j = 0, 0
        for i in range(len(sx)):
            for j in range(i + 1, len(sx)):
                frame_gap = abs(frame_indices[i] - frame_indices[j])
                if frame_gap < MIN_FRAME_GAP:
                    continue
                dx = abs(sx[i] - sx[j])
                dy = abs(sy[i] - sy[j])
                dist = dx + dy
                if dist < best_dist:
                    best_dist = dist
                    best_i, best_j = i, j
    else:
        # diverse: |Δx| 大・|Δy| 小 を優先
        best_score = -np.inf
        best_i, best_j = 0, 0
        for i in range(len(sx)):
            for j in range(i + 1, len(sx)):
                dx = abs(sx[i] - sx[j])
                dy = abs(sy[i] - sy[j])
                score = dx - 2.0 * dy
                if score > best_score:
                    best_score = score
                    best_i, best_j = i, j

    indices = np.array([frame_indices[best_i], frame_indices[best_j]])
    shifts_x = np.array([sx[best_i], sx[best_j]])
    shifts_y = np.array([sy[best_i], sy[best_j]])
    return indices, shifts_x, shifts_y


def main():
    out_dir = _script_dir.parent / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pos_label in POS_LABELS:
        pos_dir = BASE_DIR / pos_label
        output_phase_raw = pos_dir / "output_phase_raw"
        channels_dir = pos_dir / "output_phase" / "channels"
        channel_rois_json = channels_dir / "channel_rois.json"
        shift_source = channels_dir / "pos_shifts.json"

        # フォールバック: Pos0 等で output_phase/channels がない場合、FALLBACK_POS のメタデータを使用
        fallback_dir = BASE_DIR / FALLBACK_POS / "output_phase" / "channels"
        _shift_source = shift_source if shift_source.exists() else fallback_dir / "pos_shifts.json"
        _channel_rois_json = channel_rois_json if channel_rois_json.exists() else fallback_dir / "channel_rois.json"

        print(f"\n{'='*60}\n{pos_label}")
        print("=" * 60)

        if not _shift_source.exists():
            print(f"SKIP: pos_shifts.json not found (self & {FALLBACK_POS})")
            continue
        if _shift_source != shift_source:
            print(f"  [fallback] shift data from {FALLBACK_POS}")

        indices, shifts_x, shifts_y = _pick_frames(_shift_source, N_FRAMES, SHIFT_MODE)
        print(f"Selected frames ({SHIFT_MODE}): {indices.tolist()}")
        for i, (sx_val, sy_val) in enumerate(zip(shifts_x, shifts_y)):
            print(f"  frame {indices[i]}: shift_x={sx_val:.4f}, shift_y={sy_val:.2f}")

        if not _channel_rois_json.exists():
            print(f"SKIP: channel_rois.json not found (self & {FALLBACK_POS})")
            continue
        if _channel_rois_json != channel_rois_json:
            print(f"  [fallback] channel_rois from {FALLBACK_POS}")
        with open(_channel_rois_json, encoding="utf-8") as f:
            rois = json.load(f)
        if CHANNEL_INDEX >= len(rois):
            print(f"SKIP: channel {CHANNEL_INDEX} not in channel_rois ({len(rois)} channels)")
            continue
        roi = rois[CHANNEL_INDEX]
        cy, cx = roi["cy"], roi["cx"]
        crop_w, crop_h = roi["crop_w"], roi["crop_h"]
        center_y = crop_w // 2

        # crop_sub_ecc*/ を自動検出して条件リストを構築
        _cs_conds = sorted(
            [(d.name, d.name) for d in channels_dir.glob("crop_sub_ecc*") if d.is_dir()]
        )
        for source_type, label in _cs_conds + [("raw_crop", "raw_cropped"), ("raw_raw_force", "raw_raw"), ("aligned_raw", "aligned_raw"), ("grid_sub", "grid_subtracted")]:
            if source_type.startswith("crop_sub_ecc"):
                path = None   # 後で _cs_dir として処理
            elif source_type == "raw_crop":
                path = output_phase_raw
            elif source_type == "raw_raw_force":
                path = None
            elif source_type == "aligned_raw":
                path = channels_dir / "grid_subtracted" / f"channel_{CHANNEL_INDEX:02d}_aligned_raw.tif"
            else:
                path = channels_dir / "grid_subtracted" / f"channel_{CHANNEL_INDEX:02d}_grid_sub.tif"

            # Pos0 等で grid_subtracted がない場合 or raw_raw_force: raw-raw フォールバック
            # output_phase_raw を crop → stack → stack[t] - stack[0]（raw の1枚目を引く）
            use_raw_raw_fallback = False
            fallback_label = label
            if source_type == "raw_raw_force":
                _raw_dir = output_phase_raw
                if _raw_dir.exists():
                    use_raw_raw_fallback = True
                else:
                    print(f"SKIP (no output_phase_raw): {_raw_dir}")
                    continue
            elif source_type == "aligned_raw" and not path.exists():
                print(f"SKIP (aligned_raw not found, run generate_aligned_raw.py first): {path}")
                continue
            elif source_type == "grid_sub" and not path.exists():
                _raw_dir = output_phase_raw
                if _raw_dir.exists():
                    use_raw_raw_fallback = True
                    fallback_label = "raw_raw"  # raw crop - raw crop 1枚目
                    path = None

            if source_type.startswith("crop_sub_ecc"):
                # crop_sub_ecc*/ch{CHANNEL_INDEX:02d}/ を参照
                _cs_dir = channels_dir / source_type / f"ch{CHANNEL_INDEX:02d}"
                if not _cs_dir.exists():
                    print(f"SKIP ({source_type} not found): {_cs_dir}")
                    continue
                path = _cs_dir
            elif not use_raw_raw_fallback and (path is None or not path.exists()):
                print(f"SKIP (not found): {path}")
                continue

            profiles = []
            if source_type.startswith("crop_sub_ecc"):
                files = sorted(path.glob("img_*_phase.tif"))
                if not files:
                    print(f"SKIP: no img_*_phase.tif in {path}")
                    continue
                frame_to_file = {i: f for i, f in enumerate(files)}
                center_row = FINAL_CROP_W // 2
                print(f"  Loading {source_type}: {len(files)} frames, center_row={center_row}")
                for idx in indices:
                    if idx not in frame_to_file:
                        continue
                    frame = tifffile.imread(str(frame_to_file[idx])).astype(np.float64)
                    profiles.append(frame[center_row, :])
            elif source_type == "raw_crop":
                pattern = f"img_*_ph_{TL_Z_INDEX:03d}_phase.tif"
                files = sorted(path.glob(pattern))
                if not files:
                    print(f"SKIP: no files {path / pattern}")
                    continue
                frame_to_file = {i: f for i, f in enumerate(files)}
                print(f"  Loading raw crop: ch{CHANNEL_INDEX}")
                for idx in indices:
                    if idx not in frame_to_file:
                        continue
                    img = tifffile.imread(str(frame_to_file[idx])).astype(np.float64)
                    crop = extract_rect_roi(img, cy, cx, crop_w, crop_h)
                    line = crop[center_y, :]
                    profiles.append(line)
            elif use_raw_raw_fallback:
                pattern = f"img_*_ph_{TL_Z_INDEX:03d}_phase.tif"
                files = sorted(_raw_dir.glob(pattern))
                if not files:
                    print(f"SKIP: no files {_raw_dir / pattern}")
                    continue
                frame_to_file = {i: f for i, f in enumerate(files)}
                need_frames = sorted(set([0] + indices.tolist()))
                crops = []
                for idx in need_frames:
                    if idx not in frame_to_file:
                        continue
                    img = tifffile.imread(str(frame_to_file[idx])).astype(np.float64)
                    crop = extract_rect_roi(img, cy, cx, crop_w, crop_h)
                    crops.append((idx, crop))
                if not crops:
                    print(f"SKIP: no crops for indices")
                    continue
                idx_to_crop = {idx: c for idx, c in crops}
                ref_crop = idx_to_crop.get(0)
                if ref_crop is None:
                    print(f"SKIP: frame 0 not in output_phase_raw (raw-raw に必要)")
                    continue
                print(f"  Loading raw-raw (output_phase_raw crop - 1枚目): ch{CHANNEL_INDEX}")
                for idx in indices:
                    c = idx_to_crop.get(idx)
                    if c is None:
                        continue
                    sub = c.astype(np.float64) - ref_crop.astype(np.float64)
                    line = sub[center_y, :]
                    line = line - line.mean()  # 各プロファイルの平均を0に
                    profiles.append(line)
            elif source_type == "aligned_raw":
                # raw_raw 形式: stack[t] - stack[0]、mean 引き算
                print(f"  Loading aligned_raw (stack - frame0): ch{CHANNEL_INDEX}")
                stack = load_tiff_stack(str(path))
                n_frames_st, h, w = stack.shape
                cy_stack = h // 2
                if 0 >= n_frames_st:
                    print(f"SKIP: aligned_raw stack empty")
                    continue
                ref_line = stack[0, cy_stack, :].astype(np.float64)
                for idx in indices:
                    if idx >= n_frames_st:
                        continue
                    line = stack[idx, cy_stack, :].astype(np.float64) - ref_line
                    line = line - line.mean()
                    profiles.append(line)
            else:
                print(f"  Loading: {path.name}")
                stack = load_tiff_stack(str(path))
                n_frames, h, w = stack.shape
                cy_stack = h // 2
                for idx in indices:
                    if idx >= n_frames:
                        continue
                    line = stack[idx, cy_stack, :].astype(np.float64)
                    profiles.append(line)

            if len(profiles) < N_FRAMES:
                print(f"  WARNING: only {len(profiles)} frames (expected {N_FRAMES})")
                continue

            n_pts = len(profiles[0])
            x = np.arange(n_pts)
            fig, ax = plt.subplots(figsize=(10, 5))
            for idx, prof, sx_val, sy_val in zip(indices, profiles, shifts_x, shifts_y):
                ax.plot(x, prof, label=f"f{idx} (Δx={sx_val:.2f}, Δy={sy_val:.2f})", alpha=0.7)

            ax.set_xlabel("x (pixel)")
            ax.set_ylabel("Phase (rad)")
            if source_type.startswith("crop_sub_ecc"):
                ax.set_ylim(-1, 3)
            elif fallback_label == "raw_raw":
                ax.set_ylim(-1, 1)
            elif Y_SCALE is not None:
                ax.set_ylim(Y_SCALE)
            ax.set_title(f"Shift-invariant: center line (y={center_y})\n{pos_label} {fallback_label} | ch{CHANNEL_INDEX} | {N_FRAMES} frames")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
            ax.grid(True, alpha=0.3)
            fig.subplots_adjust(left=0.08, right=0.72, top=0.88, bottom=0.13)

            base = f"shift_invariant_profiles_{pos_label}_{fallback_label}"
            try:
                save_figure(
                    fig,
                    params={"pos": pos_label, "source": str(path) if path else "output_phase_raw crop - frame0 (raw-raw)", "frame_indices": indices.tolist(), "center_y": center_y},
                    data={"x": x, "frame_indices": indices, "profiles": np.array(profiles)},
                    description=f"Shift-invariant profiles {pos_label} {fallback_label}",
                )
            except Exception as e:
                print(f"[warn] save_figure: {e}")
            fig_path = out_dir / f"{base}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {fig_path}")

    # NPZ差分プロット: profiles1 - profiles0 をフレームごとに重ねてプロット
    for pat1, label1, pat2, label2 in (NPZ_DIFF_PAIRS or []):
        print(f"\n{'='*60}")
        print(f"NPZ diff: ({label1}) - ({label2})")
        npz_path1 = _find_latest_data_npz(pat1)
        npz_path2 = _find_latest_data_npz(pat2)
        if npz_path1 is None:
            print(f"SKIP: no _data.npz matching '{pat1}' in figure_logger inbox")
            continue
        if npz_path2 is None:
            print(f"SKIP: no _data.npz matching '{pat2}' in figure_logger inbox")
            continue
        print(f"  npz1: {npz_path1}")
        print(f"  npz2: {npz_path2}")
        d1 = np.load(str(npz_path1))
        d2 = np.load(str(npz_path2))
        if "profiles" not in d1.files or "profiles" not in d2.files:
            print(f"SKIP: npz missing 'profiles' key (got {list(d1.files)} / {list(d2.files)})")
            continue
        x1, fi1, p1 = d1["x"], d1["frame_indices"], d1["profiles"]
        x2, fi2, p2 = d2["x"], d2["frame_indices"], d2["profiles"]
        if not np.array_equal(x1, x2):
            print("WARNING: x axes differ, skipping")
            continue
        if not np.array_equal(fi1, fi2):
            print(f"WARNING: frame_indices differ ({fi1} vs {fi2}), skipping")
            continue
        diffs = p1 - p2  # shape (N_FRAMES, n_pts)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (fi, diff) in enumerate(zip(fi1, diffs)):
            ax.plot(x1, diff, label=f"f{fi}", alpha=0.7)
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("pixel value diff")
        ax.set_title(f"Shift-invariant diff: ({label1}) − ({label2})\nch{CHANNEL_INDEX} | {len(fi1)} frames")
        ax.set_ylim(-1, 1)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.subplots_adjust(left=0.08, right=0.72, top=0.88, bottom=0.13)
        tag = f"npz_diff_{label1.replace(' ', '_')}__minus__{label2.replace(' ', '_')}"
        try:
            save_figure(
                fig,
                params={"label1": label1, "label2": label2, "frame_indices": fi1.tolist()},
                data={"x": x1, "frame_indices": fi1, "diffs": diffs},
                description=f"NPZ profile diff: ({label1}) - ({label2})",
            )
        except Exception as e:
            print(f"[warn] save_figure: {e}")
        fig_path = out_dir / f"{tag}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")

    # 4-way diff: (p1 - p2) - (p3 - p4)
    for pat1, lbl1, pat2, lbl2, pat3, lbl3, pat4, lbl4 in (NPZ_QUAD_DIFF_PAIRS or []):
        print(f"\n{'='*60}")
        print(f"NPZ quad diff: ({lbl1} - {lbl2}) - ({lbl3} - {lbl4})")
        n1 = _find_latest_data_npz(pat1)
        n2 = _find_latest_data_npz(pat2)
        n3 = _find_latest_data_npz(pat3)
        n4 = _find_latest_data_npz(pat4)
        missing = [p for p, n in [(pat1, n1), (pat2, n2), (pat3, n3), (pat4, n4)] if n is None]
        if missing:
            print(f"SKIP: no npz for {missing}")
            continue
        d1, d2, d3, d4 = (np.load(str(n)) for n in [n1, n2, n3, n4])
        x1, fi1, p1 = d1["x"], d1["frame_indices"], d1["profiles"]
        x2, fi2, p2 = d2["x"], d2["frame_indices"], d2["profiles"]
        x3, fi3, p3 = d3["x"], d3["frame_indices"], d3["profiles"]
        x4, fi4, p4 = d4["x"], d4["frame_indices"], d4["profiles"]
        if not (np.array_equal(x1, x2) and np.array_equal(x1, x3) and np.array_equal(x1, x4)):
            print("WARNING: x axes differ, skipping")
            continue
        if not (np.array_equal(fi1, fi2) and np.array_equal(fi1, fi3) and np.array_equal(fi1, fi4)):
            print(f"WARNING: frame_indices differ, skipping")
            continue
        diffs = (p1 - p2) - (p3 - p4)
        fig, ax = plt.subplots(figsize=(10, 5))
        for fi, diff in zip(fi1, diffs):
            ax.plot(x1, diff, label=f"f{fi}", alpha=0.7)
        title_top = f"({lbl1} − {lbl2}) − ({lbl3} − {lbl4})"
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("pixel value diff")
        ax.set_title(f"Shift-invariant quad diff:\n{title_top}\nch{CHANNEL_INDEX} | {len(fi1)} frames")
        ax.set_ylim(-1, 1)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.subplots_adjust(left=0.08, right=0.72, top=0.85, bottom=0.13)
        tag = f"npz_quaddiff_({lbl1}-{lbl2})_minus_({lbl3}-{lbl4})".replace(" ", "_")
        try:
            save_figure(
                fig,
                params={"labels": [lbl1, lbl2, lbl3, lbl4], "frame_indices": fi1.tolist()},
                data={"x": x1, "frame_indices": fi1, "diffs": diffs},
                description=f"NPZ quad diff: ({lbl1}-{lbl2}) - ({lbl3}-{lbl4})",
            )
        except Exception as e:
            print(f"[warn] save_figure: {e}")
        fig_path = out_dir / f"{tag}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")

    # =================================================================
    # X-tilt 補正版 timelapse プロファイル（補正前後を比較）
    # =================================================================
    if TL_PHASE_DIR_XTILT.exists() and TL_ROIS_JSON_XTILT.exists():
        print(f"\n{'='*60}")
        print("X-tilt corrected timelapse profiles")
        print("="*60)
        tl_files = sorted(TL_PHASE_DIR_XTILT.glob("img_*_phase.tif"))
        if not tl_files:
            print(f"SKIP: no phase files in {TL_PHASE_DIR_XTILT}")
        else:
            with open(TL_ROIS_JSON_XTILT, encoding="utf-8") as _f:
                tl_rois = json.load(_f)
            pick_tps = [min(t, len(tl_files) - 1) for t in TL_PICK_TPS]
            n_fit_v  = max(3, int(TL_CROP_H_TILT * TL_FIT_FRAC_XTILT))  # 90
            x_axis   = np.arange(TL_CROP_H_TILT)

            for c_idx in TL_CH_INDICES_XTILT:
                if c_idx >= len(tl_rois):
                    print(f"SKIP: ch{c_idx} not in rois ({len(tl_rois)} channels)")
                    continue
                roi    = tl_rois[c_idx]
                cy_r, cx_r   = roi["cy"], roi["cx"]
                crop_w_r, crop_h_r = roi["crop_w"], roi["crop_h"]
                center_y_r   = crop_w_r // 2
                ecc_start    = (TL_CROP_H_TILT - crop_h_r) // 2

                raw_profiles  = []
                corr_profiles = []
                for tp in pick_tps:
                    img = tifffile.imread(str(tl_files[tp])).astype(np.float64)
                    big_raw = extract_rect_roi(img, cy_r, cx_r, crop_w_r, TL_CROP_H_TILT).astype(np.float64)
                    big_corr, sl = _xtilt_correct_big_crop(
                        img, cy_r, cx_r, crop_w_r, TL_CROP_H_TILT, TL_FIT_FRAC_XTILT
                    )
                    raw_profiles.append(big_raw[center_y_r, :])
                    corr_profiles.append(big_corr[center_y_r, :])
                    print(f"  ch{c_idx} TP={tp}: slope={sl:.5f} rad/px")

                # 補正前後を上下に並べて1枚の図に
                fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
                fig.subplots_adjust(hspace=0.35, left=0.08, right=0.78, top=0.90, bottom=0.09)

                for tp, prof in zip(pick_tps, raw_profiles):
                    ax_top.plot(x_axis, prof, label=f"TP={tp}", alpha=0.8)
                ax_top.axvline(n_fit_v, color="#999", lw=0.7, ls=":", label="BG fit boundary")
                ax_top.axvspan(ecc_start, ecc_start + crop_h_r, alpha=0.10,
                               color="#1E88E5", label="ECC crop region")
                ax_top.set_ylabel("Phase (rad)")
                ax_top.set_title(f"Raw (270px center line)  ch{c_idx}", fontsize=10)
                ax_top.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
                ax_top.spines["top"].set_visible(False)
                ax_top.spines["right"].set_visible(False)

                for tp, prof in zip(pick_tps, corr_profiles):
                    ax_bot.plot(x_axis, prof, label=f"TP={tp}", alpha=0.8)
                ax_bot.axvline(n_fit_v, color="#999", lw=0.7, ls=":")
                ax_bot.axvspan(ecc_start, ecc_start + crop_h_r, alpha=0.10, color="#1E88E5")
                ax_bot.set_xlabel("X [px]")
                ax_bot.set_ylabel("Phase (rad)")
                ax_bot.set_title(f"X-tilt corr (left-1/3 subtracted)  ch{c_idx}", fontsize=10)
                ax_bot.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
                ax_bot.spines["top"].set_visible(False)
                ax_bot.spines["right"].set_visible(False)

                fig.suptitle(
                    f"Shift-invariant profile: raw vs X-tilt corr  |  ch{c_idx}  |  timelapse ph_3\n"
                    f"yellow dotted: BG fit boundary  |  blue shaded: ECC crop region",
                    fontsize=9,
                )
                try:
                    save_figure(
                        fig,
                        params={
                            "ch": c_idx, "pick_tps": pick_tps,
                            "crop_h_tilt": TL_CROP_H_TILT, "fit_frac": TL_FIT_FRAC_XTILT,
                        },
                        data={
                            "x": x_axis,
                            "raw_profiles":  np.array(raw_profiles),
                            "corr_profiles": np.array(corr_profiles),
                            "pick_tps":      np.array(pick_tps),
                        },
                        description=(
                            f"Shift-invariant profile raw vs X-tilt corr: "
                            f"ch{c_idx} TPs={pick_tps}"
                        ),
                    )
                except Exception as e:
                    print(f"[warn] save_figure: {e}")
                plt.close(fig)
                print(f"  ch{c_idx}: saved")

    # ============================================================
    # crop_sub_ecc* 全条件 × 全チャネル時系列重ねプロット
    # ============================================================
    print(f"\n{'='*60}")
    print("crop_sub_ecc*: all-channel timeseries overlay")
    for pos_label in POS_LABELS:
        pos_dir = BASE_DIR / pos_label
        channels_dir = pos_dir / "output_phase" / "channels"
        for cs_root in sorted(channels_dir.glob("crop_sub_ecc*")):
          if not cs_root.is_dir():
            continue
          cond_name = cs_root.name

          ch_dirs = sorted(cs_root.glob("ch*/"))
          if not ch_dirs:
            print(f"SKIP: no ch*/ subdirs in {cs_root}")
            continue
          n_ch = len(ch_dirs)
          n_cols = 4
          n_rows = (n_ch + n_cols - 1) // n_cols
          fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.5), squeeze=False)

          for ch_idx, ch_dir in enumerate(ch_dirs):
            ax = axes[ch_idx // n_cols][ch_idx % n_cols]
            files = sorted(ch_dir.glob("img_*_phase.tif"))
            if not files:
                ax.set_visible(False)
                continue
            n_frames = len(files)
            cmap = plt.cm.viridis
            center_row = FINAL_CROP_W // 2
            for fi, fp in enumerate(files):
                frame = tifffile.imread(str(fp)).astype(np.float64)
                profile = frame[center_row, :]
                ax.plot(np.arange(len(profile)), profile,
                        color=cmap(fi / max(n_frames - 1, 1)), lw=0.6, alpha=0.5)
            ax.set_title(f"ch{ch_idx:02d} (n={n_frames})", fontsize=8)
            ax.set_xlabel("x (pixel)", fontsize=7)
            ax.set_ylabel("Phase (rad)", fontsize=7)
            ax.set_ylim(-1, 3)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)

          # 使わない軸を非表示
          for ch_idx in range(n_ch, n_rows * n_cols):
            axes[ch_idx // n_cols][ch_idx % n_cols].set_visible(False)

          # カラーバー（時間軸）
          sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_frames - 1))
          sm.set_array([])
          fig.colorbar(sm, ax=axes.ravel().tolist(), label="Frame index", shrink=0.6)

          fig.suptitle(f"{cond_name} center-row profiles: {pos_label}\n"
                       f"(all {n_frames} frames, center_row={center_row}, width={frame.shape[1]}px)",
                       fontsize=10)
          fig.tight_layout(rect=[0, 0, 0.92, 0.95])

          try:
            save_figure(
                fig,
                params={"pos": pos_label, "cond": cond_name, "n_channels": n_ch, "n_frames": n_frames, "center_row": center_row},
                data={"n_frames": n_frames, "n_channels": n_ch},
                description=f"{cond_name} all-channel timeseries overlay {pos_label}",
            )
          except Exception as e:
            print(f"[warn] save_figure: {e}")
          fig_path = out_dir / f"{cond_name}_all_channels_{pos_label}.png"
          fig.savefig(fig_path, dpi=150, bbox_inches="tight")
          plt.close(fig)
          print(f"  Saved: {fig_path}")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
