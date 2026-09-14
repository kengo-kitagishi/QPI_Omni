"""Extract well-tracked mother cells from per_channel_figures inbox JSONs.

Cross-references with docs/channel_classification_260517.yaml and produces an
Obsidian markdown summary of mother cells that are:
  - well tracked (n_frames close to 3748, reasonable n_mother_divisions)
  - classified as having a live mother in the YAML (status: cells, phase1 alive,
    and mother either revived or mixed-with-mother-revived)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import yaml


INBOX_DIRS = [
    Path("/Users/kitak/Library/CloudStorage/GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/共有ドライブ/wakamotolab_meeting/kitagishi/figure-hub/inbox/2026-06-08/per_channel_figures"),
    Path("/Users/kitak/Library/CloudStorage/GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/共有ドライブ/wakamotolab_meeting/kitagishi/figure-hub/inbox/2026-06-09/per_channel_figures"),
]

YAML_PATH = Path("/Users/kitak/QPI_Omni/docs/channel_classification_260517.yaml")

OUT_MD = Path(
    "/Users/kitak/Documents/Obsidian Vault/01_Daily/260517_well_tracked_mother_cells_2026-06-09.md"
)
OUT_MD_FOCUSED = Path(
    "/Users/kitak/Documents/Obsidian Vault/01_Daily/260517_mother_revived_vs_dead_2026-06-09.md"
)

MAX_FRAMES = 3748
N_FRAMES_MIN = 3500
N_DIV_MIN, N_DIV_MAX = 20, 100


def parse_inbox(inbox_dirs):
    """Walk inbox per_channel_figures dirs, return latest entry per (Pos, ch)."""
    latest: dict[tuple[str, str], dict] = {}
    for inbox in inbox_dirs:
        if not inbox.exists():
            continue
        for run_dir in inbox.iterdir():
            if not run_dir.is_dir():
                continue
            # use f001 (mother_volume) as the canonical entry
            for json_path in run_dir.glob("*f001.json"):
                try:
                    meta = json.loads(json_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                params = meta.get("params", {})
                channel_label = params.get("channel_label", "")
                m = re.match(r"(Pos\d+)_(ch\d+)", channel_label)
                if not m:
                    continue
                pos, ch = m.group(1), m.group(2)
                entry = {
                    "pos": pos,
                    "ch": ch,
                    "n_frames": params.get("n_frames", 0),
                    "n_mother_divisions": params.get("n_mother_divisions", 0),
                    "run_id": meta.get("run_id", ""),
                    "created_at_utc": meta.get("created_at_utc", ""),
                    "channel_dir": params.get("channel_dir", ""),
                    "calibration_id": params.get("calibration_id", ""),
                    "json_path": str(json_path),
                    "pdf_volume": json_path.with_suffix(".pdf").name,
                    "pdf_ri": str(json_path.parent
                                  / json_path.name.replace("_mother_volume_vs_time_", "_mother_mean_RI_vs_time_")
                                  .replace("f001.json", "f002.pdf")),
                    "pdf_tree": str(json_path.parent
                                  / json_path.name.replace("_mother_volume_vs_time_", "_lineage_tree_")
                                  .replace("f001.json", "f003.pdf")),
                }
                key = (pos, ch)
                # keep the latest by created_at_utc
                existing = latest.get(key)
                if existing is None or entry["created_at_utc"] > existing["created_at_utc"]:
                    latest[key] = entry
    return latest


def load_yaml_classification(yaml_path):
    """Load channel_classification YAML, return {(Pos, ch): {status, phase1, phase2}}."""
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    out = {}
    for pos_name, channels in (data.get("positions") or {}).items():
        if not isinstance(channels, dict):
            continue
        for ch_name, fields in channels.items():
            if not isinstance(fields, dict):
                continue
            phase1 = fields.get("phase1") or {}
            phase2 = fields.get("phase2") or {}
            out[(pos_name, ch_name)] = {
                "status": fields.get("status"),
                "phase1_outcome": phase1.get("outcome"),
                "phase1_notes": phase1.get("notes"),
                "phase2_outcome": phase2.get("outcome"),
                "phase2_notes": phase2.get("notes"),
            }
    return out


def is_well_tracked(entry):
    return (entry["n_frames"] >= N_FRAMES_MIN
            and N_DIV_MIN <= entry["n_mother_divisions"] <= N_DIV_MAX)


def classify_yaml_compatibility(yaml_info):
    """Return one of: 'mother_revived', 'mother_dead', 'mixed_partial',
    'phase1_dead', 'no_classification', 'unclassified'."""
    if yaml_info is None:
        return "no_classification"
    status = yaml_info.get("status")
    if status is None:
        return "unclassified"
    if status in ("empty", "unused", "dead_at_start"):
        return "no_living_mother"
    if status != "cells":
        return "unknown"
    phase1 = yaml_info.get("phase1_outcome")
    phase2 = yaml_info.get("phase2_outcome")
    if phase1 == "dead":
        return "phase1_dead"
    if phase1 in (None, "alive", "mixed"):
        # check phase2 outcome
        if phase2 == "revived":
            return "mother_revived"
        if phase2 == "died_starvation":
            return "mother_dead"
        if phase2 == "mixed":
            notes = (yaml_info.get("phase2_notes") or "")
            if "mother: revived" in notes or "mother + " in notes and "revived" in notes:
                return "mother_revived_partial"
            if "mother: never_revived" in notes or "mother: dead" in notes:
                return "mother_dead_partial"
            return "mixed_unclear"
        if phase2 == "never_revived":
            return "mother_dead"
    return "unclassified"


def main():
    print("Scanning per_channel_figures inboxes...")
    inbox = parse_inbox(INBOX_DIRS)
    print(f"  Found {len(inbox)} unique (Pos, ch) pairs across both dates")

    print("Loading YAML classification...")
    yaml_cls = load_yaml_classification(YAML_PATH)
    print(f"  YAML has {len(yaml_cls)} classified channels")

    # build groups
    well_tracked_revived = []
    well_tracked_other = []
    well_tracked_no_yaml = []
    well_tracked_dead = []
    not_well_tracked_but_revived = []
    suspicious_high_div = []
    diagnostic_rows = []

    for key, entry in sorted(inbox.items(), key=lambda kv: (int(kv[0][0][3:]), kv[0][1])):
        yaml_info = yaml_cls.get(key)
        category = classify_yaml_compatibility(yaml_info)
        well = is_well_tracked(entry)

        row = {
            **entry,
            "yaml_category": category,
            "yaml_status": (yaml_info or {}).get("status"),
            "yaml_phase1": (yaml_info or {}).get("phase1_outcome"),
            "yaml_phase2": (yaml_info or {}).get("phase2_outcome"),
            "yaml_phase2_notes": (yaml_info or {}).get("phase2_notes"),
        }
        diagnostic_rows.append(row)

        if entry["n_mother_divisions"] > N_DIV_MAX:
            suspicious_high_div.append(row)
            continue
        if well:
            if category in ("mother_revived", "mother_revived_partial"):
                well_tracked_revived.append(row)
            elif category in ("mother_dead", "mother_dead_partial", "phase1_dead"):
                well_tracked_dead.append(row)
            elif category == "no_living_mother":
                well_tracked_no_yaml.append(row)
            else:
                well_tracked_other.append(row)
        else:
            if category in ("mother_revived", "mother_revived_partial"):
                not_well_tracked_but_revived.append(row)

    # write markdown
    md_lines = []
    md_lines.append("---")
    md_lines.append("title: 260517 well-tracked mother cells（YAML分類との突合）")
    md_lines.append("date: 2026-06-09")
    md_lines.append("type: analysis-summary")
    md_lines.append("exp: [\"260517\"]")
    md_lines.append("---\n")

    md_lines.append("## 抽出基準\n")
    md_lines.append(f"- well-tracked = `n_frames >= {N_FRAMES_MIN}` （最大 {MAX_FRAMES}） AND "
                    f"`{N_DIV_MIN} <= n_mother_divisions <= {N_DIV_MAX}`")
    md_lines.append("- YAML 分類は `docs/channel_classification_260517.yaml` から読み込み")
    md_lines.append("- `n_mother_divisions > 100` は誤分裂を拾っているチャネル（後述）として別扱い")
    md_lines.append("- 同じ (Pos, ch) で複数 run がある場合は最新 `created_at_utc` を採用\n")

    md_lines.append("## サマリ\n")
    md_lines.append(f"| カテゴリ | 件数 |")
    md_lines.append(f"|---|---|")
    md_lines.append(f"| ✅ well-tracked + YAML で mother 復活 | **{len(well_tracked_revived)}** |")
    md_lines.append(f"| well-tracked + YAML で mother 死亡（phase1/phase2）| {len(well_tracked_dead)} |")
    md_lines.append(f"| well-tracked + YAML で empty/unused/dead_at_start | {len(well_tracked_no_yaml)} |")
    md_lines.append(f"| well-tracked + YAML 未分類 or 不明 | {len(well_tracked_other)} |")
    md_lines.append(f"| well-tracked できなかったが YAML 上 mother 復活 | {len(not_well_tracked_but_revived)} |")
    md_lines.append(f"| 誤分裂疑い（n_mother_divisions > {N_DIV_MAX}）| {len(suspicious_high_div)} |")
    md_lines.append("")

    # main table: well-tracked + mother revived
    md_lines.append("## ✅ Well-tracked mother cells（YAML 上 mother 復活確認済み）\n")
    md_lines.append("これが最も信頼できる解析対象群。lineage tracking が完走し、YAML での目視分類でも mother が「revived（全細胞 or mother のみ）」と一致している。\n")
    md_lines.append("| Pos | ch | n_frames | n_div | YAML status | YAML phase2 | mother fate（notes より） | run_id |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for r in well_tracked_revived:
        notes = (r["yaml_phase2_notes"] or "").replace("\n", " ").strip()
        fate = "revived" if r["yaml_phase2"] == "revived" else _extract_mother_fate_from_notes(notes)
        notes_short = notes[:60] + "..." if len(notes) > 60 else notes
        md_lines.append(
            f"| {r['pos']} | {r['ch']} | {r['n_frames']} | {r['n_mother_divisions']} | "
            f"{r['yaml_status']} | {r['yaml_phase2']} | {fate} | `{r['run_id'][:24]}` |"
        )
    md_lines.append("")

    # figures section: embed pdfs for revived group
    md_lines.append("### 図（各 mother の volume / mean RI / lineage tree）\n")
    md_lines.append("各 well-tracked mother について 3 セット（volume vs time、mean RI vs time、lineage tree）を埋め込み。\n")
    for r in well_tracked_revived:
        md_lines.append(f"#### {r['pos']}_{r['ch']}（n_div={r['n_mother_divisions']}）")
        md_lines.append(f"")
        md_lines.append(f"![[{r['pdf_volume']}]]")
        md_lines.append(f"![[{Path(r['pdf_ri']).name}]]")
        md_lines.append(f"![[{Path(r['pdf_tree']).name}]]")
        md_lines.append("")
        md_lines.append(f"- run_id: `{r['run_id']}`")
        md_lines.append(f"- channel_dir: `{r['channel_dir']}`")
        md_lines.append(f"- YAML phase2 notes:\n  ```\n  {(r['yaml_phase2_notes'] or '').rstrip()}\n  ```")
        md_lines.append("")

    # secondary: well-tracked + mother dead
    md_lines.append("## ⚠️ Well-tracked だが YAML 上 mother 死亡\n")
    md_lines.append("Tracking 自体は完走しているが、YAML 目視分類で mother が phase1 か phase2 で死亡した case。「死亡前の vol/RI 軌跡を見たい」「死亡時刻と tracking 上の急変を照合したい」など、解析的に別の用途がある。\n")
    md_lines.append("| Pos | ch | n_frames | n_div | YAML phase1 | YAML phase2 | run_id |")
    md_lines.append("|---|---|---|---|---|---|---|")
    for r in well_tracked_dead:
        md_lines.append(
            f"| {r['pos']} | {r['ch']} | {r['n_frames']} | {r['n_mother_divisions']} | "
            f"{r['yaml_phase1']} | {r['yaml_phase2']} | `{r['run_id'][:24]}` |"
        )
    md_lines.append("")

    # discrepancy: YAML says empty/unused/dead_at_start but tracking succeeded
    if well_tracked_no_yaml:
        md_lines.append("## 🚨 Well-tracked だが YAML が empty/unused/dead_at_start\n")
        md_lines.append("YAML の目視分類と lineage tracking で結論が食い違う case。YAML の分類修正、または segmentation が空チャネルでも何かを拾った疑い。\n")
        md_lines.append("| Pos | ch | n_frames | n_div | YAML status | run_id |")
        md_lines.append("|---|---|---|---|---|---|")
        for r in well_tracked_no_yaml:
            md_lines.append(
                f"| {r['pos']} | {r['ch']} | {r['n_frames']} | {r['n_mother_divisions']} | "
                f"{r['yaml_status']} | `{r['run_id'][:24]}` |"
            )
        md_lines.append("")

    # well-tracked but YAML unclassified (Pos46+ など未 dictation)
    if well_tracked_other:
        md_lines.append("## ❓ Well-tracked だが YAML 未分類\n")
        md_lines.append("Pos46+ の未 dictation 範囲か、判定不明な case。\n")
        md_lines.append("| Pos | ch | n_frames | n_div | YAML status | YAML phase2 |")
        md_lines.append("|---|---|---|---|---|---|")
        for r in well_tracked_other:
            md_lines.append(
                f"| {r['pos']} | {r['ch']} | {r['n_frames']} | {r['n_mother_divisions']} | "
                f"{r['yaml_status']} | {r['yaml_phase2']} |"
            )
        md_lines.append("")

    # suspicious high divisions
    md_lines.append("## ⚠️ 誤分裂疑い（n_mother_divisions > 100）\n")
    md_lines.append(f"3748 frame × 5 min = 312 h で >{N_DIV_MAX} 分裂 = 平均 3.1h 以下 1 回 → 分裂酵母の生理から外れる。lineage が誤分裂を拾っている可能性。\n")
    md_lines.append("| Pos | ch | n_frames | n_div | YAML status | YAML phase2 |")
    md_lines.append("|---|---|---|---|---|---|")
    for r in sorted(suspicious_high_div, key=lambda x: -x["n_mother_divisions"]):
        md_lines.append(
            f"| {r['pos']} | {r['ch']} | {r['n_frames']} | {r['n_mother_divisions']} | "
            f"{r['yaml_status']} | {r['yaml_phase2']} |"
        )
    md_lines.append("")

    # not well-tracked but YAML revived (good candidates for re-tracking)
    if not_well_tracked_but_revived:
        md_lines.append("## 🔧 YAML 上 mother 復活だが tracking が取れていない\n")
        md_lines.append("YAML 目視では mother が復活していると判定したのに、lineage tracking が n_frames<3500 または n_mother_divisions<20 で取りこぼしている case。segmentation/lineage の再走行候補。\n")
        md_lines.append("| Pos | ch | n_frames | n_div | YAML phase2 | mother fate（notes より） |")
        md_lines.append("|---|---|---|---|---|---|")
        for r in not_well_tracked_but_revived:
            notes = (r["yaml_phase2_notes"] or "").replace("\n", " ").strip()
            fate = _extract_mother_fate_from_notes(notes)
            md_lines.append(
                f"| {r['pos']} | {r['ch']} | {r['n_frames']} | {r['n_mother_divisions']} | "
                f"{r['yaml_phase2']} | {fate} |"
            )
        md_lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nwrote: {OUT_MD}")

    # focused MD: phase1-alive + phase2-never_revived vs phase1-alive + phase2-revived
    # both groups must be well-tracked
    revived_full = [r for r in well_tracked_revived
                    if r["yaml_phase1"] == "alive"]
    never_revived_phase2 = []
    for r in diagnostic_rows:
        # well-tracked filter
        if not is_well_tracked(r):
            continue
        # phase1 must be alive (mother survived phase1)
        if r["yaml_phase1"] != "alive":
            continue
        notes = (r["yaml_phase2_notes"] or "")
        phase2 = r["yaml_phase2"]
        # mother never revived in phase2
        if phase2 == "never_revived":
            never_revived_phase2.append(r)
        elif phase2 == "died_starvation":
            never_revived_phase2.append(r)
        elif phase2 == "mixed" and (
                "mother: never_revived" in notes
                or re.search(r"mother[+ ]\d?\s*[番目]*[:：]?\s*never_revived", notes)
                or notes.lstrip().startswith("mother + 2番目: never_revived")
                or notes.lstrip().startswith("mother + 2番目 + 3番目: never_revived")):
            never_revived_phase2.append(r)

    focused = []
    focused.append("---")
    focused.append("title: 260517 mother cell revived vs never_revived 比較（well-tracked のみ）")
    focused.append("date: 2026-06-09")
    focused.append("type: analysis-comparison")
    focused.append("exp: [\"260517\"]")
    focused.append("---\n")
    focused.append("## 目的\n")
    focused.append("飢餓回復時に mother cell が **生き返らなかった群** と **最後まで生き返った群** を、lineage tracking が完走している channel に絞って横並びで見るためのまとめ。\n")
    focused.append("- 抽出条件（両群共通）: `n_frames >= 3500`、`20 <= n_mother_divisions <= 100`、YAML 上 `phase1: alive`")
    focused.append("- 群A（never_revived）: YAML 上 mother が phase2 で never_revived / died_starvation / mixed-with-mother-never_revived")
    focused.append("- 群B（revived）: YAML 上 mother が phase2 で revived（全細胞 revived or mixed で mother 復活）")
    focused.append("")
    focused.append(f"| 群 | 件数 |")
    focused.append(f"|---|---|")
    focused.append(f"| 🟥 mother never_revived（飢餓回復で死亡）| **{len(never_revived_phase2)}** |")
    focused.append(f"| 🟩 mother revived（最後まで生存）| **{len(revived_full)}** |")
    focused.append("")

    # GROUP A
    focused.append("## 🟥 群A: 飢餓回復時に生き返らなかった mother cells\n")
    focused.append("Phase1 を生き延びたが、phase2（0.0055% → 0% → 2%回復）で mother が復活せず死亡した case。「死ぬ前の体積・密度の動き」「死亡時刻の特定」「死亡 precursor の定量」など、死亡条件を探る素材になる。\n")
    focused.append("| Pos | ch | n_frames | n_div | YAML phase2 | YAML notes（mother fate） |")
    focused.append("|---|---|---|---|---|---|")
    for r in never_revived_phase2:
        notes = (r["yaml_phase2_notes"] or "").replace("\n", " ").strip()
        notes_short = notes[:70] + "..." if len(notes) > 70 else notes
        focused.append(f"| {r['pos']} | {r['ch']} | {r['n_frames']} | {r['n_mother_divisions']} | "
                       f"{r['yaml_phase2']} | {notes_short} |")
    focused.append("")

    focused.append("### 図（群A: never_revived mother）\n")
    for r in never_revived_phase2:
        focused.append(f"#### {r['pos']}_{r['ch']}（n_div={r['n_mother_divisions']}, phase2={r['yaml_phase2']}）\n")
        focused.append(f"![[{r['pdf_volume']}]]")
        focused.append(f"![[{Path(r['pdf_ri']).name}]]")
        focused.append(f"![[{Path(r['pdf_tree']).name}]]\n")
        focused.append(f"- run_id: `{r['run_id']}`")
        focused.append(f"- channel_dir: `{r['channel_dir']}`")
        focused.append(f"- YAML phase2 notes:")
        focused.append(f"  ```")
        focused.append(f"  {(r['yaml_phase2_notes'] or '').rstrip()}")
        focused.append(f"  ```\n")

    # GROUP B
    focused.append("## 🟩 群B: 最後まで生き返った mother cells\n")
    focused.append("Phase1 を生き延び、phase2 で復活して最後まで分裂を継続した mother。「生存細胞の代謝・体積回復軌跡」「分裂タイミング再開」「dry mass 回復速度」を見るのに使える reference 群。\n")
    focused.append("| Pos | ch | n_frames | n_div | YAML phase2 | YAML notes |")
    focused.append("|---|---|---|---|---|---|")
    for r in revived_full:
        notes = (r["yaml_phase2_notes"] or "").replace("\n", " ").strip()
        notes_short = notes[:70] + "..." if len(notes) > 70 else notes
        focused.append(f"| {r['pos']} | {r['ch']} | {r['n_frames']} | {r['n_mother_divisions']} | "
                       f"{r['yaml_phase2']} | {notes_short} |")
    focused.append("")

    focused.append("### 図（群B: revived mother）\n")
    for r in revived_full:
        focused.append(f"#### {r['pos']}_{r['ch']}（n_div={r['n_mother_divisions']}, phase2={r['yaml_phase2']}）\n")
        focused.append(f"![[{r['pdf_volume']}]]")
        focused.append(f"![[{Path(r['pdf_ri']).name}]]")
        focused.append(f"![[{Path(r['pdf_tree']).name}]]\n")
        focused.append(f"- run_id: `{r['run_id']}`")
        focused.append(f"- channel_dir: `{r['channel_dir']}`\n")

    OUT_MD_FOCUSED.write_text("\n".join(focused), encoding="utf-8")
    print(f"wrote: {OUT_MD_FOCUSED}")
    print(f"  🟥 群A never_revived mothers: {len(never_revived_phase2)}")
    print(f"  🟩 群B revived mothers:       {len(revived_full)}")
    print(f"  ✅ well-tracked + mother revived: {len(well_tracked_revived)}")
    print(f"  ⚠️  well-tracked + mother dead:    {len(well_tracked_dead)}")
    print(f"  🚨 well-tracked + YAML empty:     {len(well_tracked_no_yaml)}")
    print(f"  ❓ well-tracked + unclassified:   {len(well_tracked_other)}")
    print(f"  ⚠️  suspicious (n_div > {N_DIV_MAX}):    {len(suspicious_high_div)}")
    print(f"  🔧 not-tracked + YAML revived:    {len(not_well_tracked_but_revived)}")


def _extract_mother_fate_from_notes(notes: str) -> str:
    if "mother: revived" in notes:
        return "mother revived"
    if "mother + 2番目" in notes and "revived" in notes:
        return "mother+daughter revived"
    if "mother: never_revived" in notes:
        return "mother never_revived"
    if "mother: dead" in notes:
        return "mother dead"
    if "全細胞 revived" in notes:
        return "全細胞 revived"
    return notes[:40] + ("..." if len(notes) > 40 else "")


if __name__ == "__main__":
    main()
