import glob
import json

base = (r"G:\共有ドライブ\wakamotolab_meeting\kitagishi\figure-hub\inbox"
        r"\2026-06-22\run_ri_drop_matched\20260622T033407Z_5531a8")
ctx = {
    "objective": (
        "Re-output the two RI-drop histograms (RI(2884)-RI(2018) and "
        "meanRI(2864-2874)-RI(2018)) with bins and y-axis matched to the "
        "media-switch no-window figure, for visual consistency."),
    "method": (
        "scripts/_run_ri_drop_matched.py, QPI_USE_CORRECTED=1 "
        "QPI_VOLUME_VARIANT=efd. Drops from ri_drop_starvation.drops (nearest "
        "good frame within +/-8 for the single 2884 def; mean over 2864-2874 for "
        "the window def). Style matched to media-switch figure: y-axis=count "
        "(was density), bin width=0.04/28~=0.00143 (was data/28), dark-gray "
        "revived vs red dead, white edges, mean vertical lines. Both figures "
        "share common bins (18 bins) and y-limit (19) for direct comparison."),
    "result": (
        "single 2884: revived mean -0.0041/med -0.0042 (n=73), dead -0.0063/"
        "-0.0066 (n=52), KS p=0.0012. window 2864-2874: revived -0.0045 (med), "
        "dead -0.0059 (med), KS p=0.0197. Values identical to the previous "
        "RI-drop run; only the histogram styling changed."),
    "interpretation": (
        "Dead cells drop more in RI across starvation than revived; separation "
        "is significant for the single-frame def (p=0.0012) and weaker but still "
        "significant for the window def (p=0.0197). Restyling does not change the "
        "statistics, only makes the bins/axis consistent with the media-switch "
        "figure."),
}
for p in glob.glob(base + r"\*.json"):
    if p.endswith("_manifest.jsonl"):
        continue
    meta = json.loads(open(p, encoding="utf-8").read())
    meta["context"] = ctx
    open(p, "w", encoding="utf-8").write(
        json.dumps(meta, ensure_ascii=False, indent=2))
    print("patched", p.split("\\")[-1])
