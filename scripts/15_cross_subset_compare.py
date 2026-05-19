#!/usr/bin/env python
"""
15_cross_subset_compare.py — Same E3 across cell subsets [paper Fig 7]

完全对应 Osorio et al. 2022, Result 7 + Fig 7:
  "We first examined the perturbation profile of endothelial cells, in which
   the function of MYDFG has been studied. ... We also concatenated
   perturbation profiles of 45 different cell types. ... When using the
   Spearman's ρ to measure the similarity between the perturbation profiles
   observed in endothelial cells and that observed in other cell types, ...
   these are the most similar cell types: T-cell (ρ=0.59), hepatocyte
   (ρ=0.45), ... In contrast, two cell types, proximal tubule cell (ρ=0) and
   Leydig cell (ρ=0.08), differed the most..."

为什么对 E3VK 项目至关重要:
  这是把我们的「亚群 × E3 vulnerability heatmap」做的更深入: 对同一个 E3,
  比较它在 Th1 / Th17 / Treg / B / CD8 等亚群中 perturbation profile 的相关性,
  能识别出:
    - 哪些 E3 在 T 细胞通用 (高亚群-亚群相关性)
    - 哪些 E3 有亚群特异功能 (低相关性, 例如 Treg-specific)
  另外用每个 E3 的「显著 DR 基因 IoU」和「DR profile Spearman ρ」两种度量,
  互相印证.

输出:
  - figs/cross_subset/<E3>_corr.pdf          (论文 Fig 7B 风格的对称热图)
  - cross_subset_similarity.csv              长表: E3 × pair(subset_a, subset_b)
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
import seaborn as sns
from scipy import stats
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent))
from _config import DOWNSTREAM_DIR, FIGS_DIR, FDR_STRICT, set_mpl_style

set_mpl_style()
XS_DIR = FIGS_DIR / "cross_subset"
XS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-e3", type=int, default=30,
                    help="只对 vulnerability 综合排名前 N 的 E3 做")
    ap.add_argument("--min-subsets", type=int, default=3,
                    help="该 E3 至少在 N 个亚群里都被跑过")
    ap.add_argument("--fdr", type=float, default=FDR_STRICT)
    args = ap.parse_args()

    vuln = pd.read_csv(DOWNSTREAM_DIR / "vulnerability.csv")
    subsets = sorted(vuln["subset"].unique())
    print(f"Subsets: {subsets}")

    # 选 top E3: 按 frac_sig 综合 (各亚群 mean) 排
    e3_score = vuln.groupby("ko")["frac_sig_05"].mean().sort_values(ascending=False)
    e3_n_subset = vuln.groupby("ko")["subset"].nunique()
    candidates = e3_score[e3_n_subset >= args.min_subsets].head(args.top_e3).index.tolist()
    print(f"Top E3 candidates ({len(candidates)}): first few = {candidates[:5]}")

    # 把每个 E3 在所有亚群的 z 向量收成 dict-of-DataFrame: e3 -> (gene × subset)
    z_by_subset = {s: pd.read_parquet(DOWNSTREAM_DIR / f"dr_zmat_{s}.parquet")
                   for s in subsets}
    padj_by_subset = {s: pd.read_parquet(DOWNSTREAM_DIR / f"dr_padj_{s}.parquet")
                      for s in subsets}

    summary = []
    for ko in candidates:
        # 收集每个亚群中该 KO 的 Z 向量
        cols = {}
        sig_sets = {}
        for s in subsets:
            zm = z_by_subset[s]
            if ko not in zm.columns:
                continue
            cols[s] = zm[ko]
            pm = padj_by_subset[s]
            sig_sets[s] = set(pm.index[pm[ko] < args.fdr].astype(str))
        if len(cols) < args.min_subsets:
            continue

        # 对齐所有亚群的 gene index (取并集, 缺失补 0)
        all_genes = sorted(set().union(*[set(v.index) for v in cols.values()]))
        mat = pd.DataFrame({s: v.reindex(all_genes).fillna(0.0) for s, v in cols.items()})

        # 相关性矩阵 (两种)
        spear = mat.corr(method="spearman")

        # Sig-gene Jaccard 矩阵
        ss = list(sig_sets.keys())
        jac = pd.DataFrame(np.zeros((len(ss), len(ss))), index=ss, columns=ss)
        for a, b in combinations(ss, 2):
            A, B = sig_sets[a], sig_sets[b]
            j = len(A & B) / max(1, len(A | B))
            jac.loc[a, b] = jac.loc[b, a] = j
        for s in ss:
            jac.loc[s, s] = 1.0

        # 画图 (两图并排)
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
        sns.heatmap(spear, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    vmin=-1, vmax=1, ax=axs[0], cbar_kws={"label": "Spearman ρ"})
        axs[0].set_title(f"{ko} — perturbation profile correlation")
        sns.heatmap(jac, annot=True, fmt=".2f", cmap="Purples",
                    vmin=0, vmax=1, ax=axs[1], cbar_kws={"label": "Jaccard"})
        axs[1].set_title(f"{ko} — significant DR-gene overlap")
        fig.suptitle(f"E3 = {ko}", y=1.02)
        fig.savefig(XS_DIR / f"{ko}_corr.pdf")
        plt.close(fig)

        # 长表记录
        for a, b in combinations(ss, 2):
            summary.append({
                "ko": ko, "subset_a": a, "subset_b": b,
                "spearman_rho": float(spear.loc[a, b]),
                "sig_jaccard":  float(jac.loc[a, b]),
                "n_sig_a": len(sig_sets[a]),
                "n_sig_b": len(sig_sets[b]),
                "n_sig_overlap": len(sig_sets[a] & sig_sets[b]),
            })

    if summary:
        sdf = pd.DataFrame(summary)
        sdf.to_csv(DOWNSTREAM_DIR / "cross_subset_similarity.csv", index=False)
        print(f"\nWrote cross_subset_similarity.csv ({len(sdf)} pair rows)")
        # 简要总结表: 每个 E3 的「亚群间一致性」
        agg = (sdf.groupby("ko")
               .agg(median_rho=("spearman_rho", "median"),
                    min_rho=("spearman_rho", "min"),
                    median_jaccard=("sig_jaccard", "median"))
               .sort_values("median_rho"))
        agg.to_csv(DOWNSTREAM_DIR / "cross_subset_consistency.csv")
        print("\nLeast cross-subset-consistent E3 (potential subset-specific):")
        print(agg.head(10))


if __name__ == "__main__":
    main()