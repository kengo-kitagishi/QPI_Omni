"""parse_bsh_log.py — BeanShell Script Panel ログから per-channel ECC データを抽出し
drift_log.json に channel_details として遡及 merge する。

Usage:
    python scripts/parse_bsh_log.py
"""

import json
import re
import sys
from pathlib import Path

BSH_LOG   = Path(r"C:\Users\QPI\Documents\QPI_Omni\scripts\260326_timelapse_drift.txt")
DRIFT_LOG = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log.json")

# ---- 正規表現 ----
RE_TP   = re.compile(r"-- Timepoint (\d+) /")
RE_CH   = re.compile(
    r"ch(\d+): pass1=\(([+-]?\d+\.\d+),([+-]?\d+\.\d+)\)px corr1=(\d+\.\d+)"
    r"\s+grid=\(([+-]?\d+),([+-]?\d+)\)"
    r"\s+pass2=\(([+-]?\d+\.\d+),([+-]?\d+\.\d+)\)px corr2=(\d+\.\d+)"
)
RE_EXCL = re.compile(r"idx=(\[[^\]]+\])")   # idx=[0, 1, 3] など

# ---- ログ解析 ----
txt = BSH_LOG.read_text(encoding="utf-8", errors="replace")
lines = txt.splitlines()

parsed = {}   # tp -> list of channel dicts
current_tp = None
outlier_idx = set()

for line in lines:
    m_tp = RE_TP.search(line)
    if m_tp:
        current_tp = int(m_tp.group(1))
        outlier_idx = set()
        if current_tp not in parsed:
            parsed[current_tp] = []
        continue

    if current_tp is None:
        continue

    m_ch = RE_CH.search(line)
    if m_ch:
        ch     = int(m_ch.group(1))
        tx1    = float(m_ch.group(2))
        ty1    = float(m_ch.group(3))
        corr1  = float(m_ch.group(4))
        xi     = int(m_ch.group(5))
        yi     = int(m_ch.group(6))
        tx2    = float(m_ch.group(7))
        ty2    = float(m_ch.group(8))
        corr2  = float(m_ch.group(9))
        parsed[current_tp].append({
            "ch": ch,
            "tx1": tx1, "ty1": ty1, "corr1": corr1,
            "xi": xi, "yi": yi,
            "tx2": tx2, "ty2": ty2, "corr2": corr2,
            "outlier": False,   # 後で更新
            "status": "pass2_ok",
        })
        continue

    m_excl = RE_EXCL.search(line)
    if m_excl:
        try:
            excl_list = json.loads(m_excl.group(1))
            outlier_idx = set(excl_list)
            # 除外インデックスはチャネルリスト内の順番（parsed 末尾から）
            chs = parsed.get(current_tp, [])
            n = len(chs)
            for idx in outlier_idx:
                if idx < n:
                    chs[idx]["outlier"] = True
        except Exception:
            pass

print(f"Parsed TPs: {len(parsed)}, range {min(parsed)}..{max(parsed)}")
ch_counts = [len(v) for v in parsed.values()]
print(f"Channels per TP: min={min(ch_counts)}, max={max(ch_counts)}, "
      f"median={sorted(ch_counts)[len(ch_counts)//2]}")

# ---- drift_log.json に merge ----
records = json.loads(DRIFT_LOG.read_text(encoding="utf-8"))

updated = 0
skipped = 0
for rec in records:
    tp = rec.get("timepoint")
    if tp is None:
        continue
    if tp not in parsed:
        skipped += 1
        continue
    # 既存の channel_details は上書き（ログの方が正確）
    rec["channel_details"] = parsed[tp]
    updated += 1

print(f"\nMerge result: updated={updated}, skipped(no log data)={skipped}")

# ---- バックアップ → 書き込み ----
backup = DRIFT_LOG.with_suffix(".json.bak")
backup.write_text(DRIFT_LOG.read_text(encoding="utf-8"), encoding="utf-8")
print(f"Backup: {backup}")

DRIFT_LOG.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Written: {DRIFT_LOG}")

# ---- 検証 ----
verify = json.loads(DRIFT_LOG.read_text(encoding="utf-8"))
with_cd = [r for r in verify if r.get("channel_details")]
print(f"\nVerify: {len(with_cd)} / {len(verify)} TPs have channel_details")
if with_cd:
    sample = with_cd[0]["channel_details"][0]
    print(f"Sample: {json.dumps(sample)}")
