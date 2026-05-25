#!/usr/bin/env python
"""
20_master_table.py — Integrate 4 dimensions into composite ranking

整合 A(高影响) / B(亚群特异) / C(命运决定) / D(疾病相关) 四维分数，
生成主排名表并分配 Tier（Tier1 = top 5%, Tier2 = top 20%）。

当 scoreC 为空（CellOracle 未跑）时，C 权重自动分摊到 A/B（各 0.45）。

输出:
  results/AIHA/downstream/E3_master_ranking.csv
  results/AIHA/downstream/E3_master_ranking_Tier1.csv
  results/AIHA/downstream/whitelist_coverage.txt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _config import DOWNSTREAM_DIR, load_e3_family

WHITELIST_AIHA: set[str] = {
    "ITCH", "CBLB", "CUL5", "VHL", "STUB1", "RNF128", "TRAF6", "CUL3", "RBX1",
    "TRIM21", "PELI1", "UBE2L3", "RNF31", "RBCK1", "TRAF3", "GRAIL", "HOIP", "HOIL1",
}


def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def parse_weights(s: str) -> tuple[float, float, float, float]:
    parts = s.split(":")
    w = [float(x) for x in parts]
    if len(w) != 4:
        raise ValueError("weights must be 4 colon-separated values, e.g. 30:30:30:10")
    total = sum(w)
    return tuple(x / total for x in w)


def main():
    ap = argparse.ArgumentParser(description="Master ranking table (4 dimensions)")
    ap.add_argument("--weights", default="30:30:30:10",
                    help="A:B:C:D weights (will be normalized to sum=1)")
    ap.add_argument("--tier1-pct", type=float, default=5.0,
                    help="Top %% of composite -> Tier1")
    ap.add_argument("--tier2-pct", type=float, default=20.0,
                    help="Top %% of composite -> Tier2")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_csv   = DOWNSTREAM_DIR / "E3_master_ranking.csv"
    out_tier1 = DOWNSTREAM_DIR / "E3_master_ranking_Tier1.csv"
    if out_csv.exists() and not args.force:
        print(f"Already exists: {out_csv}  (use --force to recompute)"); return

    wA, wB, wC, wD = parse_weights(args.weights)

    # ---- Load scoreA ----
    a_path = DOWNSTREAM_DIR / "scoreA_high_impact.csv"
    if not a_path.exists():
        print("ERROR: scoreA_high_impact.csv missing. Run 17_score_A_high_impact.py first.")
        sys.exit(1)
    score_a = pd.read_csv(a_path)

    # ---- Load scoreB ----
    b_path = DOWNSTREAM_DIR / "scoreB_subset_specific.csv"
    if not b_path.exists():
        print("ERROR: scoreB_subset_specific.csv missing. Run 18_score_B_specificity.py first.")
        sys.exit(1)
    score_b = pd.read_csv(b_path)[["ko", "subset", "specificity_z",
                                    "boosted_spec_z", "top_contrast", "score_B"]]

    # ---- Load scoreC (optional) ----
    c_path = DOWNSTREAM_DIR / "scoreC_fate_transition.csv"
    has_C = False
    score_c = pd.DataFrame()
    if c_path.exists():
        tmp = pd.read_csv(c_path)
        if not tmp.empty and "score_C" in tmp.columns and tmp["score_C"].notna().any():
            # Per (ko, from_subset) best score_C; use from_subset as "subset" to merge
            score_c = (tmp.groupby(["ko", "from_subset"])["score_C"]
                         .max()
                         .reset_index()
                         .rename(columns={"from_subset": "subset"}))
            has_C = True
            # Top transition label
            top_trans = (tmp.sort_values("delta_flux", ascending=False)
                            .drop_duplicates(["ko"])
                            [["ko", "from_subset", "to_subset", "delta_flux"]]
                            .assign(top_transition=lambda d:
                                d["from_subset"] + "→" + d["to_subset"] +
                                ": Δ=" + d["delta_flux"].round(3).astype(str)))
            score_c = score_c.merge(top_trans[["ko", "top_transition"]], on="ko", how="left")

    if not has_C:
        print("[INFO] scoreC is empty (CellOracle not yet run). "
              "Redistributing C weight to A/B.")
        wA = (wA + wC / 2)
        wB = (wB + wC / 2)
        wC = 0.0

    # ---- Load scoreD ----
    d_path = DOWNSTREAM_DIR / "scoreD_disease_relevance.csv"
    if not d_path.exists():
        print("ERROR: scoreD_disease_relevance.csv missing. Run 19_score_D_disease.py first.")
        sys.exit(1)
    score_d = pd.read_csv(d_path)[["ko", "aiha_prior", "gwas_autoimmune",
                                    "is_E2", "substrate_overlap_sig",
                                    "score_D", "score_D_norm"]]

    # ---- Load QC ----
    qc_path = DOWNSTREAM_DIR / "qc_runs.csv"
    qc = pd.read_csv(qc_path) if qc_path.exists() else pd.DataFrame()

    # ---- Merge: start from scoreA (subset, ko) as base ----
    main_tbl = score_a[["subset", "ko", "n_sig", "out_deg",
                         "residual_z", "gini", "score_A", "pass_qc"]].copy()

    main_tbl = main_tbl.merge(
        score_b[["ko", "subset", "specificity_z", "top_contrast", "score_B"]],
        on=["ko", "subset"], how="left"
    )

    if has_C and not score_c.empty:
        main_tbl = main_tbl.merge(
            score_c[["ko", "subset"] + (["top_transition"] if "top_transition" in score_c else [])],
            on=["ko", "subset"], how="left"
        )
        # Find max delta per ko (for score_C)
        c_max = (pd.read_csv(c_path)
                   .groupby("ko")["delta_flux"].apply(lambda x: x.abs().max())
                   .reset_index().rename(columns={"delta_flux": "max_delta_flux"}))
        main_tbl = main_tbl.merge(c_max, on="ko", how="left")
    else:
        main_tbl["max_delta_flux"] = np.nan
        main_tbl["top_transition"] = np.nan

    main_tbl = main_tbl.merge(score_d, on="ko", how="left")

    # E3 family
    fam = load_e3_family()[["gene", "family"]]
    main_tbl = main_tbl.merge(fam, left_on="ko", right_on="gene", how="left")
    main_tbl["family"] = main_tbl["family"].fillna("Other")
    main_tbl.drop(columns=["gene"], errors="ignore", inplace=True)

    # Pass QC (from QC table or scoreA flag)
    if not qc.empty:
        qc_pass = qc.set_index(["subset", "ko"])["is_outlier"].to_dict()
        main_tbl["pass_qc"] = main_tbl.apply(
            lambda r: not qc_pass.get((r["subset"], r["ko"]), False), axis=1
        )

    # ---- Normalize and composite ----
    main_tbl["A_norm"] = minmax(main_tbl["score_A"].fillna(0))
    main_tbl["B_norm"] = minmax(main_tbl["score_B"].fillna(0))
    main_tbl["C_norm"] = minmax(main_tbl["max_delta_flux"].fillna(0)) if wC > 0 else 0.0
    main_tbl["D_norm"] = minmax(main_tbl["score_D_norm"].fillna(0))

    main_tbl["composite"] = (
        wA * main_tbl["A_norm"] +
        wB * main_tbl["B_norm"] +
        wC * main_tbl["C_norm"] +
        wD * main_tbl["D_norm"]
    ).round(4)

    # Tier assignment (per-subset ranking)
    def assign_tier(group: pd.DataFrame) -> pd.Series:
        q1 = group["composite"].quantile(1 - args.tier1_pct / 100)
        q2 = group["composite"].quantile(1 - args.tier2_pct / 100)
        return pd.Series(
            ["Tier1" if v >= q1 else "Tier2" if v >= q2 else "Other"
             for v in group["composite"]],
            index=group.index,
        )

    main_tbl["tier"] = main_tbl.groupby("subset", group_keys=False).apply(assign_tier)

    # Sort
    main_tbl = main_tbl.sort_values(["subset", "composite"], ascending=[True, False])

    # ---- Output columns (matching §5.2 spec) ----
    col_order = [
        "subset", "ko", "family", "n_sig", "out_deg",
        "residual_z", "gini", "score_A",
        "specificity_z", "top_contrast", "score_B",
        "max_delta_flux", "top_transition", "score_C",
        "aiha_prior", "gwas_autoimmune", "score_D", "score_D_norm",
        "A_norm", "B_norm", "C_norm", "D_norm", "composite",
        "tier", "pass_qc",
    ]
    col_order = [c for c in col_order if c in main_tbl.columns]
    main_tbl = main_tbl[col_order]

    main_tbl.to_csv(out_csv, index=False)
    tier1 = main_tbl[main_tbl["tier"] == "Tier1"]
    tier1.to_csv(out_tier1, index=False)
    print(f"Wrote {len(main_tbl)} rows -> {out_csv}")
    print(f"Tier1: {len(tier1)} rows (unique E3: {tier1['ko'].nunique()}) -> {out_tier1}")

    # ---- Whitelist coverage check ----
    tier1_kos = set(tier1["ko"])
    covered   = WHITELIST_AIHA & tier1_kos
    coverage  = len(covered) / len(WHITELIST_AIHA)
    report = [
        f"Whitelist coverage in Tier1: {len(covered)}/{len(WHITELIST_AIHA)} = {coverage:.1%}",
        f"Covered: {sorted(covered)}",
        f"Missing: {sorted(WHITELIST_AIHA - tier1_kos)}",
        "",
    ]
    if coverage < 0.5:
        report.append("⚠ WARNING: coverage < 50% — consider adjusting weights or QC thresholds")
    elif coverage > 0.9:
        report.append("NOTE: coverage > 90% — verify D weight is not dominating (score_D_norm)")
    else:
        report.append("✓ Coverage in expected range (50–90%)")

    cov_txt = DOWNSTREAM_DIR / "whitelist_coverage.txt"
    cov_txt.write_text("\n".join(report))
    print("\n" + "\n".join(report))


if __name__ == "__main__":
    main()
