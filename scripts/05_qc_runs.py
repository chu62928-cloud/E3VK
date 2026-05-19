#!/usr/bin/env python
"""
05_qc_runs.py — KO 运行质量控制

目的: 检查 scTenifoldKnk 输出是否健康，过滤掉异常 run，避免污染下游富集。

检查项:
  1. 每个 KO 的 gene 数（应 ≈ 该亚群 GRN 节点数，~4-5k）
  2. Z 值分布的偏度/峰度（异常 KO 通常 Z 全部塌缩或全部巨大）
  3. p_adj < 0.05 的基因数（"vulnerability"）的分布，识别 outlier KO
  4. 与论文 Figure 2C 一致的「Q-Q plot of observed vs expected -log10(p)」抽查图

输出:
  results/AIHA/downstream/qc_runs.csv         # 每行一个 (subset, ko)
  results/AIHA/downstream/figs/qc/<subset>_*.pdf
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from _config import (
    DOWNSTREAM_DIR, FIGS_DIR,
    DR_COL_GENE, DR_COL_Z, DR_COL_PADJ, DR_COL_KO,
    FDR_STRICT, FDR_LOOSE, set_mpl_style,
)

set_mpl_style()
QC_FIG_DIR = FIGS_DIR / "qc"
QC_FIG_DIR.mkdir(parents=True, exist_ok=True)


def qc_one_subset(subset: str) -> pd.DataFrame:
    zmat_path = DOWNSTREAM_DIR / f"dr_zmat_{subset}.parquet"
    padj_path = DOWNSTREAM_DIR / f"dr_padj_{subset}.parquet"
    if not zmat_path.exists():
        print(f"[{subset}] missing {zmat_path.name}, skip")
        return pd.DataFrame()
    zmat = pd.read_parquet(zmat_path)
    padj = pd.read_parquet(padj_path)

    n_gene = zmat.shape[0]
    rows = []
    for ko in zmat.columns:
        z = zmat[ko].dropna().values
        p = padj[ko].dropna().values
        if len(z) == 0:
            continue
        n_strict = int((p < FDR_STRICT).sum())
        n_loose  = int((p < FDR_LOOSE).sum())
        rows.append({
            "subset":   subset,
            "ko":       ko,
            "n_genes":  len(z),
            "z_median": float(np.median(z)),
            "z_iqr":    float(np.percentile(z, 75) - np.percentile(z, 25)),
            "z_max":    float(np.max(z)),
            "z_skew":   float(stats.skew(z, nan_policy="omit")),
            "z_kurt":   float(stats.kurtosis(z, nan_policy="omit")),
            "n_sig_05": n_strict,
            "n_sig_10": n_loose,
            "frac_sig_05": n_strict / len(z),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 标记 outlier: 显著基因数 > Q3+3*IQR 或 < Q1-3*IQR
    q1, q3 = df["n_sig_05"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["is_outlier"] = (
        (df["n_sig_05"] > q3 + 3 * iqr) | (df["n_sig_05"] < q1 - 3 * iqr)
    )

    # --- 图 1: Z value distribution (前 6 个 KO + 6 个随机 KO 的 ECDF) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    kos_to_plot = list(df.sort_values("n_sig_05", ascending=False)["ko"].head(6)) + \
                  list(df.sample(min(6, len(df)), random_state=0)["ko"])
    for ko in kos_to_plot:
        z = np.sort(np.abs(zmat[ko].dropna().values))
        ax.plot(z, np.linspace(0, 1, len(z)), lw=0.7, alpha=0.6, label=ko)
    ax.set_xlabel("|Z|")
    ax.set_ylabel("ECDF")
    ax.set_title(f"{subset} — |Z| ECDF (top hits + random)")
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    fig.savefig(QC_FIG_DIR / f"{subset}_z_ecdf.pdf")
    plt.close(fig)

    # --- 图 2: # significant per KO 的分布 ---
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(df["n_sig_05"], bins=40, color="#888780", edgecolor="white")
    ax.axvline(q3 + 3 * iqr, ls="--", color="#A32D2D", lw=0.8, label="Q3+3·IQR")
    ax.set_xlabel(f"# significant DR genes (p_adj < {FDR_STRICT})")
    ax.set_ylabel("# E3 KOs")
    ax.set_title(f"{subset} — vulnerability distribution across {len(df)} KOs")
    ax.legend(fontsize=7)
    fig.savefig(QC_FIG_DIR / f"{subset}_nsig_hist.pdf")
    plt.close(fig)

    # --- 图 3: Q-Q plot for the median-vulnerability KO（与论文 Fig 2C 对照） ---
    median_ko = df.iloc[(df["n_sig_05"] - df["n_sig_05"].median()).abs().argsort().iloc[0]]["ko"]
    z = zmat[median_ko].dropna().values
    chi2 = z ** 2
    # 1. 计算观测 P 值：按 Z^2 降序排，此时 P 值从小到大 (最显著的在前面)
    p_obs = np.clip(stats.chi2.sf(np.sort(chi2)[::-1], df=1), 1e-300, 1)
    obs = -np.log10(p_obs)
    # 2. 生成理论 P 值：也必须从小到大 (1/N, 2/N, ..., 1)
    p_exp = np.arange(1, len(obs) + 1) / len(obs)
    exp = -np.log10(p_exp)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(exp, obs, s=4, color="#444441", alpha=0.6)
    lim = max(exp.max(), obs.max())
    ax.plot([0, lim], [0, lim], "--", color="#5F5E5A", lw=0.8)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    ax.set_title(f"{subset} — Q-Q for {median_ko} (median KO)")
    fig.savefig(QC_FIG_DIR / f"{subset}_qq_median.pdf")
    plt.close(fig)

    print(f"[{subset}] {len(df)} KOs, median n_sig={df['n_sig_05'].median():.0f}, "
          f"{df['is_outlier'].sum()} outliers")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=None)
    args = ap.parse_args()

    # 从 dr_zmat_*.parquet 自动发现
    if args.subsets is None:
        subsets = sorted(p.stem.removeprefix("dr_zmat_")
                         for p in DOWNSTREAM_DIR.glob("dr_zmat_*.parquet"))
    else:
        subsets = args.subsets

    if not subsets:
        print("No dr_zmat_*.parquet found. Run 04_aggregate_dr.py first.")
        sys.exit(1)

    all_qc = []
    for s in subsets:
        all_qc.append(qc_one_subset(s))
    qc = pd.concat([d for d in all_qc if not d.empty], ignore_index=True)
    out = DOWNSTREAM_DIR / "qc_runs.csv"
    qc.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(qc)} KO runs)")
    print(f"  outliers: {qc['is_outlier'].sum()}")
    print(f"  figures in {QC_FIG_DIR}")


if __name__ == "__main__":
    main()