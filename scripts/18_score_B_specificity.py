#!/usr/bin/env python
"""
18_score_B_specificity.py — Dimension B: cross-subset specificity z-score

核心思想：同一个 E3 在不同 cell type 里 KO 后扰动谱差异越大，越值得关注
  (1) 行 z-score：在该 E3 自身的 subset 分布中，某亚群是否显著偏高
  (2) Spearman ρ：扰动谱向量相似性（低 ρ 表示亚群特异功能）
  (3) 优先级亚群对加权（Th17/Treg、B_naive/Plasmablast 等）

score_B = specificity_z（对应特定 subset），附带 top_contrast 字段

输出:
  results/AIHA/downstream/scoreB_subset_specific.csv
  results/AIHA/downstream/figs/scoreB/<contrast_pair>.pdf
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from _config import DOWNSTREAM_DIR, FIGS_DIR, set_mpl_style

set_mpl_style()
OUT_DIR = FIGS_DIR / "scoreB"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 生物学优先级对（权重 > 1 放大特异性分数）
PRIORITY_PAIRS: dict[tuple[str, str], float] = {
    ("CD4_Th17", "CD4_Treg"):       1.5,   # AIHA 核心轴
    ("B_naive",  "B_plasmablast"):   1.5,   # AIHA 核心：B→PC 分化
    ("CD8_cytotoxic", "CD8_exhausted"): 1.3,
    ("CD4_Th1",  "CD4_Th2"):        1.2,
    ("CD4_Th1",  "CD4_Th17"):       1.2,
    ("B_memory", "B_plasmablast"):   1.2,
    ("CD4_Th2",  "CD4_Th2_Th17"):   1.1,
}


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def priority_weight(subset_a: str, subset_b: str) -> float:
    return PRIORITY_PAIRS.get(_pair_key(subset_a, subset_b), 1.0)


def align_vectors(cols: dict[str, pd.Series]) -> pd.DataFrame:
    """Align Z-vectors from different subsets to common gene index (fill 0 for missing)."""
    all_genes = sorted(set.union(*[set(s.index) for s in cols.values()]))
    return pd.DataFrame(
        {k: v.reindex(all_genes, fill_value=0.0) for k, v in cols.items()}
    )


def compute_pairwise_rho(z_by_subset: dict[str, pd.Series],
                         ko: str, subsets: list[str]) -> dict[tuple, float]:
    """Return pairwise Spearman ρ for all subset pairs (only where ko exists)."""
    available = {s: z_by_subset[s][ko] for s in subsets
                 if s in z_by_subset and ko in z_by_subset[s].columns}
    if len(available) < 2:
        return {}
    aligned = align_vectors(available)
    result = {}
    for a, b in combinations(available.keys(), 2):
        rho, _ = stats.spearmanr(aligned[a], aligned[b])
        result[_pair_key(a, b)] = float(rho) if np.isfinite(rho) else 0.0
    return result


def main():
    ap = argparse.ArgumentParser(description="Score B: cross-subset specificity")
    ap.add_argument("--subsets", nargs="+", default=None,
                    help="Subsets to include (default: all with scoreA)")
    ap.add_argument("--top-pairs", choices=["all", "priority"], default="priority",
                    help="Which subset pairs to plot")
    ap.add_argument("--min-subsets", type=int, default=3,
                    help="Min number of subsets an E3 must appear in")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_csv = DOWNSTREAM_DIR / "scoreB_subset_specific.csv"
    if out_csv.exists() and not args.force:
        print(f"Already exists: {out_csv}  (use --force to recompute)"); return

    # Load scoreA for residual_z (preferred input)
    score_a_path = DOWNSTREAM_DIR / "scoreA_high_impact.csv"
    if not score_a_path.exists():
        print("Run 17_score_A_high_impact.py first."); sys.exit(1)
    score_a = pd.read_csv(score_a_path)

    subsets = args.subsets or sorted(score_a["subset"].unique())
    print(f"Subsets: {subsets}")

    # Build E3 × subset matrix of residual_z
    M = (score_a[score_a["subset"].isin(subsets)]
         .pivot(index="ko", columns="subset", values="residual_z"))
    # Only E3s present in >= min_subsets
    n_present = M.notna().sum(axis=1)
    M = M[n_present >= args.min_subsets]
    print(f"E3s with >= {args.min_subsets} subsets: {len(M)}")

    if M.empty:
        print("No E3 meets min_subsets threshold."); sys.exit(0)

    # Row z-score: for each E3, how far above its own mean is each subset
    mu = M.mean(axis=1)
    sd = M.std(axis=1).replace(0, np.nan)
    Z_subset = M.sub(mu, axis=0).div(sd, axis=0)   # E3 × subset

    # Load Z-mats for Spearman ρ
    z_by_subset: dict[str, pd.DataFrame] = {}
    for s in subsets:
        p = DOWNSTREAM_DIR / f"dr_zmat_{s}.parquet"
        if p.exists():
            z_by_subset[s] = pd.read_parquet(p)

    # Per (subset, ko) result
    rows = []
    all_contrasts: dict[tuple, list[float]] = {}  # for plotting
    for ko in M.index:
        # Pairwise ρ
        rho_dict = compute_pairwise_rho(z_by_subset, ko, subsets)
        min_rho   = min(rho_dict.values()) if rho_dict else np.nan
        # Top contrast = pair with largest |Δ specificity_z|
        best_pair    = None
        best_delta   = 0.0
        for s in subsets:
            spec_z = Z_subset.loc[ko, s] if s in Z_subset.columns else np.nan
            if not np.isfinite(spec_z):
                continue
            other_mean = Z_subset.loc[ko].drop(s, errors="ignore").mean()
            delta = abs(spec_z - other_mean) * priority_weight(s, s)  # single-subset
            # prefer high-priority pairs
            for s2 in subsets:
                if s2 == s:
                    continue
                z2 = Z_subset.loc[ko, s2] if s2 in Z_subset.columns else np.nan
                if not np.isfinite(z2):
                    continue
                pair_delta = abs(spec_z - z2) * priority_weight(s, s2)
                if pair_delta > best_delta:
                    best_delta = pair_delta
                    best_pair  = _pair_key(s, s2)

        for s in subsets:
            spec_z = Z_subset.loc[ko, s] if s in Z_subset.columns else np.nan
            if not np.isfinite(spec_z):
                continue
            other_z = Z_subset.loc[ko].drop(s, errors="ignore")
            mean_other = other_z.mean() if not other_z.empty else np.nan

            # Boost specificity_z for priority pairs
            boosted = spec_z
            if best_pair and s in best_pair:
                boosted = spec_z * priority_weight(*best_pair)

            # Collect delta for plotting
            if best_pair:
                all_contrasts.setdefault(best_pair, []).append(boosted)

            top_contrast_str = (
                f"{best_pair[0]}_vs_{best_pair[1]}: Δ={best_delta:.2f}"
                if best_pair else ""
            )
            rows.append({
                "ko":                ko,
                "subset":            s,
                "specificity_z":     round(float(spec_z), 4),
                "boosted_spec_z":    round(float(boosted), 4),
                "mean_other_z":      round(float(mean_other), 4) if np.isfinite(mean_other) else np.nan,
                "n_subsets_compared": int(n_present[ko]),
                "min_rho":           round(float(min_rho), 4) if np.isfinite(min_rho) else np.nan,
                "top_contrast":      top_contrast_str,
                "score_B":           round(float(boosted), 4),
            })

    result = pd.DataFrame(rows)
    result.to_csv(out_csv, index=False)
    print(f"\nWrote {len(result)} rows -> {out_csv}")

    # -- Plots: top 20 E3 per priority contrast pair --
    pairs_to_plot = (
        list(PRIORITY_PAIRS.keys()) if args.top_pairs == "priority"
        else list(all_contrasts.keys())
    )
    for pair in pairs_to_plot:
        s_a, s_b = pair
        if s_a not in subsets or s_b not in subsets:
            continue
        if s_a not in Z_subset.columns or s_b not in Z_subset.columns:
            continue
        delta = (Z_subset[s_a] - Z_subset[s_b]).dropna().sort_values()
        top20 = pd.concat([delta.head(10), delta.tail(10)]).drop_duplicates()
        fig, ax = plt.subplots(figsize=(6, max(4, 0.3 * len(top20))))
        colors = ["#E03131" if v < 0 else "#2f9e44" for v in top20.values]
        ax.barh(top20.index, top20.values, color=colors)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel(f"ΔspecificityZ  ({s_a} - {s_b})")
        ax.set_title(f"E3 specificity: {s_a} vs {s_b}", fontsize=9)
        fig.tight_layout()
        fname = f"{s_a}_vs_{s_b}.pdf".replace("/", "-")
        fig.savefig(OUT_DIR / fname)
        plt.close(fig)
        print(f"  Saved {fname}")

    # Summary
    high_B = result[result["specificity_z"].abs() > 1.5]
    print(f"Subset-specific E3 (|spec_z|>1.5): {high_B['ko'].nunique()} unique E3s")


if __name__ == "__main__":
    main()
