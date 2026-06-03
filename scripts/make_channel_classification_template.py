"""Generate a blank channel-classification YAML template.

Usage:
    python3 scripts/make_channel_classification_template.py \
        --dataset 260517 \
        --positions 20 \
        --channels-per-position 12 \
        --out docs/channel_classification_260517.yaml

The template is meant to be filled by hand (or by dictation) one position at a time.
After filling, it can be read by pandas/Python for downstream analysis:

    import yaml, pandas as pd
    with open("docs/channel_classification_260517.yaml") as f:
        data = yaml.safe_load(f)
    rows = []
    for pos, channels in data["positions"].items():
        for ch, fields in channels.items():
            rows.append({"position": pos, "channel": ch, **fields})
    df = pd.json_normalize(rows)
"""
from __future__ import annotations

import argparse
from pathlib import Path

HEADER_TEMPLATE = """\
# Channel classification for dataset {dataset}
# Media schedule (experiment type B, single intermediate step):
#   2%  ->  0.0055%  ->  0%  ->  2%
#
# Phase boundary used for classification:
#   phase1 = first 7 days at 2%  (frames 0 - 2015, inclusive)
#   phase2 = after 7 days        (0.0055% -> 0% -> 2% recovery)
#
# MEDIA_SWITCHES (frames) -- VERIFY against acquisition metadata once HDD is connected.
# 5 min interval => 12 frames/h => 288 frames/day.
media_switches:
  - {{frame: 0,     medium: "wo_2",      note: "growth, 7 days"}}
  - {{frame: 2016,  medium: "wo_0.0055", note: "intermediate starvation -- verify frame"}}
  - {{frame: null,  medium: "wo_0",      note: "full starvation -- fill frame once known"}}
  - {{frame: null,  medium: "wo_2",      note: "recovery -- fill frame once known"}}

dataset: "{dataset}"
classified_at: null      # ISO date, fill when classification is finalized
classifier: "kitak"

# Schema reminder (per channel):
#   status:         empty | cells | unused
#   phase1.outcome: alive | dead       (during first 7 days at 2%)
#   phase2.outcome: revived | died_starvation | never_revived | n/a
#                   (n/a means cell already died in phase1)
#   phase*.notes:   free-text, can mention sub-phase (0.0055%, 0%, recovery)
#
# Leave fields as null while unfilled; that way unfilled vs n/a stay distinguishable.

positions:
"""

POSITION_BLOCK = """\
  Pos{n}:
{channels}
"""

CHANNEL_BLOCK = """\
    ch{c:02d}:
      status: null            # empty | cells | unused
      phase1:
        outcome: null         # alive | dead
        notes: null
      phase2:
        outcome: null         # revived | died_starvation | never_revived | n/a
        notes: null
"""


def build_template(dataset: str, n_positions: int, n_channels: int,
                   channel_start: int = 0) -> str:
    out = [HEADER_TEMPLATE.format(dataset=dataset)]
    for n in range(1, n_positions + 1):
        channels = "".join(
            CHANNEL_BLOCK.format(c=c)
            for c in range(channel_start, channel_start + n_channels)
        )
        out.append(POSITION_BLOCK.format(n=n, channels=channels))
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset folder name, e.g. 260517")
    parser.add_argument("--positions", type=int, default=20)
    parser.add_argument("--channels-per-position", type=int, default=12)
    parser.add_argument("--channel-start-index", type=int, default=0,
                        help="First channel index (0 for ch00..ch11, 1 for ch01..ch12)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    text = build_template(args.dataset, args.positions, args.channels_per_position,
                          args.channel_start_index)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote: {args.out}  ({args.positions} positions x {args.channels_per_position} channels)")


if __name__ == "__main__":
    main()
