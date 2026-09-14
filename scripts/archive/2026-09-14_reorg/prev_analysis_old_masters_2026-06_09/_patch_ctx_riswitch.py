import glob
import json

pat = (r"G:\共有ドライブ\wakamotolab_meeting\kitagishi\figure-hub\inbox"
       r"\2026-06-22\run_ri_at_switch_efd\20260622T030307Z_b1be9d\*.json")
for p in glob.glob(pat):
    if p.endswith("_manifest.jsonl"):
        continue
    meta = json.loads(open(p, encoding="utf-8").read())
    meta["context"] = {
        "objective": (
            "Regenerate the previously-shown 'mean RI at media-switch' dead/alive "
            "histogram using the NEW EFD volume estimation (EFD K=6 contour + one "
            "midpoint update; short axis = contour-intersection distance, centerline "
            "= intersection midpoint), to see whether mean RI at the three media "
            "switches separates revived (alive) from never_revived (dead)."),
        "method": (
            "scripts/_run_ri_at_switch_efd.py -> "
            "analyze_starvation_entry_cell_cycle.plot_ri_at_media_switches, run with "
            "QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd so find_lineage_csv resolves "
            "to corrected_lineage_efd/ (mean_ri = phase / EFD volume). Each mother's "
            "value is the mean of good-frame mean_ri within +/-RI_WINDOW_FRAMES of the "
            "switch frame (is_outlier / touches_border / mass<10pg dropped). "
            "revived(alive) n=73, never_revived(dead) n=52."),
        "result": (
            "frame 2018 (2%->0.0055%, starvation entry): alive 1.36885/med 1.36863, "
            "dead 1.36916/med 1.36904, KS p=0.27. "
            "frame 2306 (0.0055%->0%): alive 1.37057/1.37061, dead 1.36867/1.36884, "
            "KS p=0.079. "
            "frame 2884 (0%->2% recovery, starvation exit): alive 1.36783/1.36812, "
            "dead 1.36557/1.36528, KS p=0.0099."),
        "interpretation": (
            "The alive/dead RI difference is NOT present at starvation entry (2018, "
            "p=0.27; dead even slightly higher) but grows during the 0% period and is "
            "significant by starvation exit (2884, p=0.0099), where dead cells have "
            "dropped further so alive (~1.368) > dead (~1.366). Direction matches the "
            "RI-drop figure (dead lose more RI during starvation). The discriminating "
            "signal is built during starvation, not set at its onset."),
    }
    open(p, "w", encoding="utf-8").write(
        json.dumps(meta, ensure_ascii=False, indent=2))
    print("patched", p)
