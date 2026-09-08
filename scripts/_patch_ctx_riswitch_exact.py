import glob
import json

pat = (r"G:\共有ドライブ\wakamotolab_meeting\kitagishi\figure-hub\inbox"
       r"\2026-06-22\run_ri_at_switch_exact_efd\20260622T031257Z_e254b6\*.json")
for p in glob.glob(pat):
    if p.endswith("_manifest.jsonl"):
        continue
    meta = json.loads(open(p, encoding="utf-8").read())
    meta["context"] = {
        "objective": (
            "Re-make the mean-RI-at-media-switch dead/alive histogram WITHOUT the "
            "+/-RI_WINDOW_FRAMES averaging: use the mean_ri value exactly AT each "
            "switch frame (2018/2306/2884). Motivated by the concern that the +/-6 "
            "window around 2884 bleeds into recovery frames (>=2885), biasing RI up."),
        "method": (
            "scripts/_run_ri_at_switch_exact_efd.py, QPI_USE_CORRECTED=1 "
            "QPI_VOLUME_VARIANT=efd (EFD contour-section volume -> mean_ri = "
            "phase/volume). Per mother (rank=1): mean_ri at EXACTLY the switch frame; "
            "dropped if that frame is absent or bad (is_outlier/touches_border/"
            "mass<10pg). No window averaging. revived(alive)/never_revived(dead) "
            "cohorts of 73/52; valid-frame n shown per panel."),
        "result": (
            "frame 2018: alive 1.37087/med 1.37051 (n=71), dead 1.37109/med 1.37094 "
            "(n=52), KS p=0.83. "
            "frame 2306: alive 1.37031/1.37027 (n=70), dead 1.36835/1.36803 (n=50), "
            "KS p=0.0091. "
            "frame 2884: alive 1.36694/1.36721 (n=69), dead 1.36465/1.36466 (n=49), "
            "KS p=0.0032."),
        "interpretation": (
            "Removing the +/-6 window sharpens the separation: frame 2306 goes from "
            "marginal (windowed p=0.079) to significant (0.0091), and 2884 strengthens "
            "(0.0099 -> 0.0032). At 2884 both cohorts' RI drop vs the windowed version "
            "(alive 1.36783->1.36694, dead 1.36557->1.36465), confirming the windowed "
            "2884 was upward-biased by recovery frames (>=2885). Entry (2018) still "
            "shows no separation (p=0.83): the alive>dead RI signal is absent at "
            "starvation onset and emerges during the 0% period."),
    }
    open(p, "w", encoding="utf-8").write(
        json.dumps(meta, ensure_ascii=False, indent=2))
    print("patched", p)
