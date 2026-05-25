#!/usr/bin/env python
"""
17_score_A_high_impact.py — Dimension A: degree-corrected vulnerability + GINI

核心思想：把 vulnerability 拆成两部分
  expected（由 GRN degree 解释）+ residual（真正的功能信号）
后者才是值得关注的 E3。同时计算扰动集中度（GINI），功能性 E3 应把扰动集中在几条通路。

综合分数: score_A = 0.6 * residual_z + 0.4 * gini_z

输出:
  results/AIHA/downstream/scoreA_high_impact.csv
  results/AIHA/downstream/figs/scoreA/<subset>_residual_vs_degree.pdf
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent))
from _config import (
    DOWNSTREAM_DIR, FIGS_DIR, FDR_STRICT, list_completed_subsets,
    load_wt, set_mpl_style,
)

set_mpl_style()
OUT_DIR = FIGS_DIR / "scoreA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# QC 硬过滤阈值
QC_MIN_GENES  = 1000
QC_MAX_SKEW   = 5.0
EPS           = 0.01   # GRN 边权阈值


def gini(x: np.ndarray) -> float:
    """Gini coefficient of absolute values — measures concentration of perturbation."""
    x = np.abs(x[np.isfinite(x)])
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def process_subset(subset: str, qc: pd.DataFrame, eps: float) -> pd.DataFrame | None:
    padj_path = DOWNSTREAM_DIR / f"dr_padj_{subset}.parquet"
    zmat_path = DOWNSTREAM_DIR / f"dr_zmat_{subset}.parquet"
    if not padj_path.exists() or not zmat_path.exists():
        print(f"  [{subset}] missing parquet, skip"); return None

    padj = pd.read_parquet(padj_path)
    zmat = pd.read_parquet(zmat_path)

    n_sig = (padj < FDR_STRICT).sum(axis=0)   # Series: ko -> count

    # QC mask from qc_runs.csv
    if qc is not None and not qc.empty:
        sub_qc = qc[qc["subset"] == subset].set_index("ko")
        pass_qc = {}
        for ko in n_sig.index:
            if ko not in sub_qc.index:
                pass_qc[ko] = True
                continue
            row = sub_qc.loc[ko]
            bad = (row.get("is_outlier", False) is True or
                   row.get("n_genes", 9999) < QC_MIN_GENES or
                   abs(row.get("z_skew", 0)) > QC_MAX_SKEW)
            pass_qc[ko] = not bad
        pass_series = pd.Series(pass_qc)
    else:
        pass_series = pd.Series(True, index=n_sig.index)

    # WT degree
    wt = load_wt(subset)
    if wt is None:
        print(f"  [{subset}] no WT scGRN, skip"); return None

    wt_abs = wt.abs() > eps
    out_deg = wt_abs.sum(axis=1).reindex(n_sig.index).fillna(0)

    # Linear model: log1p(n_sig) ~ log1p(out_deg)
    X = np.log1p(out_deg.values).reshape(-1, 1)
    y = np.log1p(n_sig.values.astype(float))
    valid = np.isfinite(X.ravel()) & np.isfinite(y)
    if valid.sum() < 5:
        print(f"  [{subset}] too few valid points, skip"); return None

    model = LinearRegression().fit(X[valid], y[valid])
    expected = model.predict(X)
    residual = y - expected

    residual_s = pd.Series(residual, index=n_sig.index)
    residual_z = zscore(residual_s)

    # GINI per KO (on Z vector)
    gini_vals = {}
    for ko in zmat.columns:
        gini_vals[ko] = gini(zmat[ko].values)
    gini_s = pd.Series(gini_vals).reindex(n_sig.index).fillna(0)
    gini_z = zscore(gini_s)

    score_A = 0.6 * residual_z + 0.4 * gini_z

    rows = []
    for ko in n_sig.index:
        rows.append({
            "subset":         subset,
            "ko":             ko,
            "n_sig":          int(n_sig[ko]),
            "out_deg":        float(out_deg[ko]),
            "expected_n_sig": float(np.expm1(expected[list(n_sig.index).index(ko)])),
            "residual_z":     float(residual_z[ko]),
            "gini":           float(gini_s.get(ko, np.nan)),
            "gini_z":         float(gini_z[ko]),
            "score_A":        float(score_A[ko]),
            "pass_qc":        bool(pass_series.get(ko, True)),
        })

    df = pd.DataFrame(rows)

    # -- plot --
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#E03131" if not p else "#26215C" for p in df["pass_qc"]]
    ax.scatter(df["out_deg"], df["n_sig"], s=15, c=colors, alpha=0.6)
    # regression line
    x_line = np.linspace(df["out_deg"].min(), df["out_deg"].max(), 100)
    y_line = np.expm1(model.predict(np.log1p(x_line).reshape(-1, 1)))
    ax.plot(x_line, y_line, "k--", lw=1, label="expected")
    # label top/bottom residuals
    for _, row in df.nlargest(5, "residual_z").iterrows():
        ax.annotate(row["ko"], (row["out_deg"], row["n_sig"]),
                    fontsize=5, xytext=(2, 2), textcoords="offset points", color="#2f9e44")
    for _, row in df.nsmallest(5, "residual_z").iterrows():
        ax.annotate(row["ko"], (row["out_deg"], row["n_sig"]),
                    fontsize=5, xytext=(2, -8), textcoords="offset points", color="#e03131")
    ax.set_xlabel("E3 out-degree in WT scGRN")
    ax.set_ylabel("# significant DR genes (p_adj<0.05)")
    ax.set_title(f"{subset}: degree-corrected vulnerability", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{subset}_residual_vs_degree.pdf")
    plt.close(fig)

    return df


def main():
    ap = argparse.ArgumentParser(description="Score A: degree-corrected vulnerability + GINI")
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_csv = DOWNSTREAM_DIR / "scoreA_high_impact.csv"
    if out_csv.exists() and not args.force:
        print(f"Already exists: {out_csv}  (use --force to recompute)"); return

    subsets = args.subsets or list_completed_subsets()
    if not subsets:
        print("No completed subsets found. Run scTenifoldKnk first."); sys.exit(1)

    # QC table
    qc_path = DOWNSTREAM_DIR / "qc_runs.csv"
    qc = pd.read_csv(qc_path) if qc_path.exists() else pd.DataFrame()

    all_dfs = []
    for subset in subsets:
        print(f"[{subset}] computing score_A ...")
        df = process_subset(subset, qc, args.eps)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("No results generated."); sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(out_csv, index=False)
    print(f"\nWrote {len(combined)} rows -> {out_csv}")
    print(f"Figures saved in {OUT_DIR}")

    # Summary
    tier_A = combined[combined["residual_z"] > 1.5]
    print(f"High-impact E3 (residual_z > 1.5): {tier_A['ko'].nunique()} unique E3s "
          f"across {tier_A['subset'].nunique()} subsets")


if __name__ == "__main__":
    main()
