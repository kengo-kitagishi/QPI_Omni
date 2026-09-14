import glob
import json

base = (r"G:\共有ドライブ\wakamotolab_meeting\kitagishi\figure-hub\inbox"
        r"\2026-06-22\run_binwidth_sweep\20260622T033838Z_cb8032")
ctx = {
    "objective": (
        "Try several bin widths for the 3 figures (media-switch exact-frame, "
        "RI-drop single 2884, RI-drop window 2864-2874) and lay them out as "
        "easy-to-compare grids to pick a readable bin width."),
    "method": (
        "scripts/_run_binwidth_sweep.py, QPI_USE_CORRECTED=1 "
        "QPI_VOLUME_VARIANT=efd. Widths = [0.0010, 0.0015, 0.0020, 0.0030]. "
        "f001 = media-switch (3 frames x 4 widths), f002 = RI drop (2 defs x 4 "
        "widths). Style as current figures: count y-axis, gray revived vs red "
        "dead, white edges, mean lines."),
    "result": (
        "KS p is invariant to bin width (computed on raw values): media-switch "
        "2884 p~0.003-0.01 range per the no-window run; RI-drop single p=0.0012, "
        "window p=0.0197 across all columns. Only the visual smoothness changes."),
    "interpretation": (
        "w=0.0020 is the most readable: 0.0010 is noisy/jagged, 0.0030 over-"
        "smooths and hides the overlap. At 0.0020 the revived(right, smaller "
        "drop) vs dead(left, larger drop) shift is clearest. Statistics unchanged "
        "by binning."),
}
for p in glob.glob(base + r"\*.json"):
    if p.endswith("_manifest.jsonl"):
        continue
    meta = json.loads(open(p, encoding="utf-8").read())
    meta["context"] = ctx
    open(p, "w", encoding="utf-8").write(
        json.dumps(meta, ensure_ascii=False, indent=2))
    print("patched", p.split("\\")[-1])
