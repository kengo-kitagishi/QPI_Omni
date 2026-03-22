"""
quick_compare_v1v2.py
---------------------
drift_log.json (v1) と 2-pass ECC (v2ロジック) を 20 タイムポイントで比較する。
再構成済み位相（output_phase/）を直接読むので再構成不要・高速。
"""
import sys, json, re, threading, numpy as np, tifffile, cv2
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure
import matplotlib.pyplot as plt

# ---- 設定 ----
CONFIG       = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config_test.json")
LOG_JSON     = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log.json")
PHASE_DIR    = Path(r"C:\ph_1\Pos1\output_phase")
N_PICK       = None  # None = 全フレーム

cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
log = json.loads(LOG_JSON.read_text(encoding="utf-8"))

vmin           = cfg.get("ecc_vmin", -5.0)
vmax           = cfg.get("ecc_vmax",  2.0)
pixel_scale_um = cfg["pixel_scale_um"]
sx_sign        = cfg["shift_sign_x"]
sy_sign        = cfg["shift_sign_y"]
x_step         = cfg.get("x_step_um", 0.1)
y_step         = cfg.get("y_step_um", 0.1)
second_pass_half = cfg.get("second_pass_half", "right")
grid_dir       = cfg["grid_dir"]
grid_base_label = cfg["grid_base_label"]
grid_z_index   = cfg["grid_z_index"]

rois = json.loads(Path(cfg["channel_rois_json"]).read_text(encoding="utf-8"))
n_channels = len(rois)

ref_crops = tifffile.imread(cfg["grid_ref_crops_tif"]).astype(np.float64)
if ref_crops.ndim == 2:
    ref_crops = ref_crops[np.newaxis, ...]

# ---- 共通関数 ----

def extract_rect_roi(img, cy, cx, crop_w, crop_h):
    h, w = img.shape
    y1 = cy - crop_w // 2; y2 = y1 + crop_w
    x1 = cx - crop_h // 2; x2 = x1 + crop_h
    pad_y0 = max(0, -y1); y1 = max(0, y1)
    pad_y1 = max(0, y2 - h); y2 = min(h, y2)
    pad_x0 = max(0, -x1); x1 = max(0, x1)
    pad_x1 = max(0, x2 - w); x2 = min(w, x2)
    crop = img[y1:y2, x1:x2]
    if any([pad_y0, pad_y1, pad_x0, pad_x1]):
        crop = np.pad(crop, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant")
    return crop

def to_u8(img):
    c = np.clip(img, vmin, vmax)
    return ((c - vmin) / (vmax - vmin) * 255).astype(np.uint8)

def compute_backsub(img):
    min_phase = cfg.get("backsub_min_phase", -1.1)
    hist_min  = cfg.get("backsub_hist_min", -1.1)
    hist_max  = cfg.get("backsub_hist_max",  1.5)
    n_bins    = cfg.get("backsub_n_bins", 512)
    sw        = cfg.get("backsub_smooth_window", 20)
    edges = np.linspace(hist_min, hist_max, n_bins + 1)
    counts, _ = np.histogram(img.flatten(), bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    bw = centers[1] - centers[0]
    sm = uniform_filter1d(counts.astype(float), size=sw, mode='nearest')
    sm = uniform_filter1d(sm, size=sw, mode='nearest')
    vidx = np.where(centers >= min_phase)[0]
    sidx = vidx[vidx < int(len(centers) * 0.95)]
    if not len(sidx): return 0.0
    pi = sidx[np.argmax(sm[sidx])]
    pv = centers[pi]
    s, e = max(0, pi-300), min(len(centers), pi+300)
    try:
        popt, _ = curve_fit(lambda x,a,m,s_: a*np.exp(-((x-m)**2)/(2*s_**2)),
                            centers[s:e], sm[s:e],
                            p0=[float(sm[s:e].max()), pv, bw*20], maxfev=5000)
        return float(-popt[1])
    except:
        return float(-pv)

def ecc_v1(ref_u8, cur_u8):
    wm = np.eye(2,3,dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1e-6)
    try:
        corr, wm = cv2.findTransformECC(ref_u8, cur_u8, wm, cv2.MOTION_TRANSLATION, crit)
        return float(wm[0,2]), float(wm[1,2]), float(corr)
    except: return None

def ecc_v2(ref_u8, cur_u8):
    wm = np.eye(2,3,dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-7)
    try:
        corr, wm = cv2.findTransformECC(ref_u8, cur_u8, wm, cv2.MOTION_TRANSLATION, crit)
        return float(wm[0,2]), float(wm[1,2]), float(corr)
    except: return None

def mad_outliers(vals, thresh=2.5):
    arr = np.array(vals)
    md = float(np.median(np.abs(arr - np.median(arr))))
    if md == 0: return np.zeros(len(arr), dtype=bool)
    return np.abs(arr - np.median(arr)) > thresh * md

def average_with_mad(tx_list, ty_list, corr_list):
    n = len(tx_list)
    if n >= 3:
        ox = mad_outliers(tx_list); oy = mad_outliers(ty_list)
        mask = ~(ox | oy)
        if not mask.any(): mask = np.ones(n, dtype=bool)
    else:
        mask = np.ones(n, dtype=bool)
    idx = np.where(mask)[0]
    return (float(np.mean(np.array(tx_list)[idx])),
            float(np.mean(np.array(ty_list)[idx])),
            float(np.mean(np.array(corr_list)[idx])),
            int(mask.sum()))

# ---- グリッドポジション ----
pattern = re.compile(rf"^{re.escape(grid_base_label)}_x([+-]?\d+)_y([+-]?\d+)$")
pos_map = {}
for d in Path(grid_dir).iterdir():
    if d.is_dir():
        m = pattern.match(d.name)
        if m: pos_map[(int(m.group(1)), int(m.group(2)))] = d
print(f"Grid positions: {len(pos_map)}")

def find_nearest_grid(shift_x, shift_y):
    dx_um = sx_sign * shift_y * pixel_scale_um
    dy_um = sy_sign * shift_x * pixel_scale_um
    best, bd = None, float('inf')
    for (xi,yi) in pos_map:
        dist = ((xi*x_step - dx_um)**2 + (yi*y_step - dy_um)**2)**0.5
        if dist < bd: bd = dist; best = (xi,yi)
    return best

def get_offset_px(xi, yi):
    ox = sy_sign * yi * y_step / pixel_scale_um
    oy = sx_sign * xi * x_step / pixel_scale_um
    return ox, oy

grid_half_cache = {}
cache_lock = threading.Lock()

def load_half(xi, yi):
    with cache_lock:
        if (xi,yi) in grid_half_cache:
            return grid_half_cache[(xi,yi)]
    pos_dir = pos_map[(xi,yi)]
    fname = f"img_000000000_ph_{grid_z_index:03d}_phase.tif"
    path = pos_dir / "output_phase" / fname
    if not path.exists():
        return None
    img = tifffile.imread(str(path)).astype(np.float64)
    crops = []
    for ch in range(n_channels):
        roi = rois[ch] if ch < len(rois) else rois[-1]
        crop = extract_rect_roi(img, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"])
        crop = crop + compute_backsub(crop)
        crops.append(crop)
    with cache_lock:
        grid_half_cache[(xi,yi)] = crops
    return crops

# ---- タイムポイント選択 ----
valid_ts = sorted(set(e["timepoint"] for e in log))
if N_PICK is None:
    pick_ts = valid_ts
else:
    step = max(1, len(valid_ts) // N_PICK)
    pick_ts = valid_ts[::step][:N_PICK]
log_dict = {e["timepoint"]: e for e in log}
print(f"Comparing {len(pick_ts)} timepoints (T={pick_ts[0]}..{pick_ts[-1]})")

# ---- フレーム処理関数（並列化用）----
import concurrent.futures as cf

ref_u8_list = [to_u8(ref_crops[ch] if ch < len(ref_crops) else ref_crops[-1])
               for ch in range(n_channels)]

def process_frame(t):
    phase_path = PHASE_DIR / f"img_{t:09d}_ph_000_phase.tif"
    if not phase_path.exists():
        return None

    phase = tifffile.imread(str(phase_path)).astype(np.float64)
    h_p, w_p = phase.shape
    region = phase[1:h_p-1, 1:w_p//2]
    if region.size > 0:
        phase -= np.mean(region)

    crops = []
    for roi in rois:
        crop = extract_rect_roi(phase, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"])
        crop = crop + compute_backsub(crop)
        crops.append(crop)

    # v1 ECC (10000/1e-6)
    tx1_list, ty1_list, c1_list = [], [], []
    for ch in range(n_channels):
        r = ecc_v1(ref_u8_list[ch], to_u8(crops[ch]))
        if r: tx1_list.append(r[0]); ty1_list.append(r[1]); c1_list.append(r[2])
    if not tx1_list:
        return None
    tx1, ty1, c1, _ = average_with_mad(tx1_list, ty1_list, c1_list)

    # v2 pass1 (100000/1e-7)
    tx_p1_list, ty_p1_list, cp1_list = [], [], []
    for ch in range(n_channels):
        r = ecc_v2(ref_u8_list[ch], to_u8(crops[ch]))
        if r: tx_p1_list.append(r[0]); ty_p1_list.append(r[1]); cp1_list.append(r[2])
    if not tx_p1_list:
        return None

    # v2 pass2
    tx2_list, ty2_list, c2_list = [], [], []
    for ch in range(n_channels):
        if ch >= len(tx_p1_list): continue
        s1x, s1y = tx_p1_list[ch], ty_p1_list[ch]
        xi2, yi2 = find_nearest_grid(s1x, s1y)
        ox2, oy2 = get_offset_px(xi2, yi2)
        halves = load_half(xi2, yi2)
        if halves is None:
            tx2_list.append(s1x); ty2_list.append(s1y); c2_list.append(cp1_list[ch])
            continue
        r2 = ecc_v2(to_u8(halves[ch]), to_u8(crops[ch]))
        if r2:
            tx2_list.append(r2[0] + ox2); ty2_list.append(r2[1] + oy2); c2_list.append(r2[2])
        else:
            tx2_list.append(s1x); ty2_list.append(s1y); c2_list.append(cp1_list[ch])

    tx2, ty2, c2, n2 = average_with_mad(tx2_list, ty2_list, c2_list)
    v1e = log_dict.get(t, {})
    return {
        "t": t,
        "v1_tx": v1e.get("tx_avg_px"), "v1_ty": v1e.get("ty_avg_px"),
        "v1_corr": v1e.get("ecc_correlation"),
        "v1_cx": v1e.get("correction_stage_x_um"),
        "v1_cy": v1e.get("correction_stage_y_um"),
        "v2_tx": tx2, "v2_ty": ty2, "v2_corr": c2,
        "v2_cx": sy_sign * ty2 * pixel_scale_um,
        "v2_cy": sx_sign * tx2 * pixel_scale_um,
        "v2_n_used": n2,
    }

results = []
done = 0
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(process_frame, t): t for t in pick_ts}
    for fut in cf.as_completed(futs):
        r = fut.result()
        done += 1
        if r is not None:
            results.append(r)
        if done % 50 == 0 or done == len(pick_ts):
            print(f"  {done}/{len(pick_ts)} done, {len(results)} valid")

results.sort(key=lambda r: r["t"])
print(f"\nDone: {len(results)} timepoints")

# ---- 可視化 ----
ts  = np.array([r["t"] for r in results])
# Stage correction
v1cx = np.array([r["v1_cx"] for r in results])
v1cy = np.array([r["v1_cy"] for r in results])
v2cx = np.array([r["v2_cx"] for r in results])
v2cy = np.array([r["v2_cy"] for r in results])
v1cr = np.array([r["v1_corr"] for r in results])
v2cr = np.array([r["v2_corr"] for r in results])
diff_cx = v2cx - v1cx
diff_cy = v2cy - v1cy

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle(f"v1 vs v2: {len(results)} timepoints", fontsize=12)

kw = dict(marker='o', markersize=4, lw=1.0)

# stage X correction (画像縦)
ax = axes[0, 0]
ax.plot(ts, v1cx, label='v1', color='tab:blue', **kw)
ax.plot(ts, v2cx, label='v2 (2-pass)', color='tab:orange', **kw)
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_title('Stage X correction [um]  (画像 Y / 縦方向由来)')
ax.set_ylabel('[um]'); ax.legend(fontsize=8)

# stage Y correction (画像横)
ax = axes[0, 1]
ax.plot(ts, v1cy, label='v1', color='tab:blue', **kw)
ax.plot(ts, v2cy, label='v2 (2-pass)', color='tab:orange', **kw)
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_title('Stage Y correction [um]  (画像 X / 横方向由来)')
ax.set_ylabel('[um]'); ax.legend(fontsize=8)

# diff stage X
ax = axes[1, 0]
ax.bar(ts, diff_cx, color='tab:purple', alpha=0.7)
ax.axhline(0, color='k', lw=0.5)
ax.set_title(f'diff stage X (v2-v1)  std={diff_cx.std():.4f} um')
ax.set_ylabel('[um]')

# diff stage Y
ax = axes[1, 1]
ax.bar(ts, diff_cy, color='tab:purple', alpha=0.7)
ax.axhline(0, color='k', lw=0.5)
ax.set_title(f'diff stage Y (v2-v1)  std={diff_cy.std():.4f} um')
ax.set_ylabel('[um]')

# ECC correlation
ax = axes[2, 0]
ax.plot(ts, v1cr, label='v1', color='tab:blue', **kw)
ax.plot(ts, v2cr, label='v2', color='tab:orange', **kw)
ax.set_title(f'ECC corr  v1 mean={v1cr.mean():.4f}  v2 mean={v2cr.mean():.4f}')
ax.set_ylabel('corr'); ax.legend(fontsize=8); ax.set_xlabel('timepoint')

# scatter v1 vs v2 stage_y (主要方向)
ax = axes[2, 1]
lim = max(np.abs(v1cy).max(), np.abs(v2cy).max()) * 1.1
ax.scatter(v1cy, v2cy, s=20, color='tab:purple', alpha=0.8)
ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, label='y=x')
ax.set_xlabel('v1 stage Y [um]'); ax.set_ylabel('v2 stage Y [um]')
ax.set_title('v1 vs v2 stage Y (横方向)')
ax.legend(fontsize=8); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

plt.tight_layout()
save_figure(fig,
    params={"n_compare": len(results), "log": "drift_log.json"},
    description=f"v1 vs v2 ECC comparison: {len(results)} timepoints, stage correction and correlation")
plt.close(fig)
print("Figure saved.")
