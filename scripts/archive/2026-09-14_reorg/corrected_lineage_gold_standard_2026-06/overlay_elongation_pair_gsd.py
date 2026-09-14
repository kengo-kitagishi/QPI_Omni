"""Plot the two elongation-cascade death lineages using the
overlay_gold_standard_and_phase1_dead plotting code directly.

i.e. feed the elongation pair (Pos20 ch06, Pos30 ch04) as the "dead" group to
plot_overlay_two_groups, so they are drawn in the exact gold-standard-overlay
style: gold-standard survivors as thin gray spaghetti + the elongation lineages
in color, one figure per metric (mean RI / volume / dry mass).

Corrected-volume aware via QPI_USE_CORRECTED=1 (inherited from the imported code).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from overlay_gold_standard_and_phase1_dead import (  # noqa: E402
    select_gold_standard, plot_overlay_two_groups,
)

ELONGATION = [("Pos20", "ch06"), ("Pos30", "ch04")]


def main():
    gold = select_gold_standard()
    print(f"gold={len(gold)} elongation={ELONGATION}")
    plot_overlay_two_groups(gold, ELONGATION, "mean_RI", "mother mean RI",
                            (1.36, 1.41), "elongation_pair_mean_RI")
    plot_overlay_two_groups(gold, ELONGATION, "volume",
                            r"mother volume [$\mu m^3$]", (0, 300),
                            "elongation_pair_volume")
    plot_overlay_two_groups(gold, ELONGATION, "mass",
                            "mother dry mass [pg]", (0, 80),
                            "elongation_pair_mass")


if __name__ == "__main__":
    main()
