#!/usr/bin/env python
"""
19a_aggregate_scoreC.py — Aggregate CellOracle delta_flux into score_C

读取 06_celloracle_cellrank_fate.py 产生的每个 ko 的 delta_flux_<ko>.csv，
计算每个 (ko, from_subset) 的 score_C = max(|delta_flux|) across all to_subsets。

若 CellOracle 尚未跑，输出空表 + warning（不阻塞 20_master_table.py）。

输出:
  results/AIHA/downstream/scoreC_fate_transition.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _config import RESULTS_DIR, DOWNSTREAM_DIR

# CellOracle 产出目录（按修改后脚本约定）
FATE_BASE = RESULTS_DIR / "fate"

EMPTY_COLS = [
    "ko", "from_subset", "to_subset",
    "baseline_flux", "post_KO_flux", "delta_flux",
    "n_cells_shifted", "score_C",
]

# 生物学关注的转移方向（AIHA 相关）
INTERESTING_TRANSITIONS = [
    ("CD4_Th17",       "CD4_Treg"),       # 促炎→抑制（治疗方向）
    ("CD4_Treg",       "CD4_Th17"),       # 抑制→促炎（致病方向）
    ("CD4_Treg",       "CD4_Th1"),
    ("CD8_cytotoxic",  "CD8_exhausted"),  # 失能
    ("CD8_exhausted",  "CD8_cytotoxic"),  # 重激活
    ("B_naive",        "B_plasmablast"),  # AIHA 核心
    ("B_memory",       "B_plasmablast"),
    ("CD4_unpol",      "CD4_Th17"),
    ("CD4_unpol",      "CD4_Treg"),
]


def load_lineage(lineage: str) -> list[pd.DataFrame]:
    """Load all delta_flux CSVs for a given lineage (T or B)."""
    fate_dir = FATE_BASE / lineage
    if not fate_dir.exists():
        return []
    rows = []
    for csv_path in sorted(fate_dir.glob("delta_flux_*.csv")):
        ko = csv_path.stem.removeprefix("delta_flux_")
        try:
            df = pd.read_csv(csv_path, index_col=0)
            # df is a subset × subset matrix (rows=from, cols=to)
            for from_s in df.index:
                for to_s in df.columns:
                    delta = float(df.loc[from_s, to_s])
                    rows.append({
                        "ko":             ko,
                        "from_subset":    from_s,
                        "to_subset":      to_s,
                        "delta_flux":     delta,
                        "lineage":        lineage,
                    })
        except Exception as e:
            print(f"  [warn] {csv_path.name}: {e}", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Aggregate CellOracle delta_flux -> score_C")
    ap.add_argument("--lineages", nargs="+", default=["T", "B"],
                    help="Lineages to aggregate (T and/or B)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_csv = DOWNSTREAM_DIR / "scoreC_fate_transition.csv"
    if out_csv.exists() and not args.force:
        print(f"Already exists: {out_csv}  (use --force to recompute)"); return

    all_rows = []
    for lineage in args.lineages:
        batch = load_lineage(lineage)
        if batch:
            print(f"  Loaded {len(batch)} rows from lineage={lineage}")
        all_rows.extend(batch)

    if not all_rows:
        print("\n[WARNING] No CellOracle delta_flux files found under "
              f"{FATE_BASE}/{{T,B}}/delta_flux_*.csv")
        print("         score_C will be NaN in master table (expected before CellOracle runs)")
        empty = pd.DataFrame(columns=EMPTY_COLS)
        empty.to_csv(out_csv, index=False)
        print(f"Wrote empty scoreC -> {out_csv}")
        return

    long = pd.DataFrame(all_rows)

    # Attach baseline flux if available
    for lineage in args.lineages:
        bl_path = FATE_BASE / lineage / "baseline_flux.csv"
        if bl_path.exists():
            try:
                bl = pd.read_csv(bl_path, index_col=0)
                bl_long = (bl.stack()
                             .reset_index()
                             .rename(columns={"level_0": "from_subset",
                                              "level_1": "to_subset",
                                              0: "baseline_flux"}))
                bl_long["lineage"] = lineage
                long = long.merge(bl_long, on=["from_subset", "to_subset", "lineage"],
                                  how="left")
            except Exception as e:
                print(f"  [warn] baseline_flux load ({lineage}): {e}", file=sys.stderr)

    if "baseline_flux" not in long.columns:
        long["baseline_flux"] = np.nan
    if "post_KO_flux" not in long.columns:
        long["post_KO_flux"] = np.nan
    if "n_cells_shifted" not in long.columns:
        long["n_cells_shifted"] = np.nan

    # score_C per (ko, from_subset) = max |delta_flux| across all to_subsets
    score_c = (long.groupby(["ko", "from_subset"])["delta_flux"]
               .apply(lambda x: x.abs().max())
               .reset_index()
               .rename(columns={"delta_flux": "score_C"}))
    long = long.merge(score_c, on=["ko", "from_subset"], how="left")

    # Flag interesting transitions
    interesting_set = {(a, b) for a, b in INTERESTING_TRANSITIONS}
    long["interesting"] = long.apply(
        lambda r: (r["from_subset"], r["to_subset"]) in interesting_set, axis=1
    )

    # Keep full table
    out_cols = [c for c in EMPTY_COLS if c in long.columns] + \
               [c for c in ["lineage", "interesting"] if c in long.columns]
    long[out_cols].to_csv(out_csv, index=False)
    print(f"\nWrote {len(long)} rows -> {out_csv}")

    # Summary: top transitions per ko
    top = (long[long["interesting"]]
           .nlargest(20, "delta_flux")[["ko", "from_subset", "to_subset", "delta_flux"]])
    if not top.empty:
        print("\nTop interesting fate transitions:")
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
