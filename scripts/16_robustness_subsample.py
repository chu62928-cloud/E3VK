#!/usr/bin/env python
"""
16_robustness_subsample.py — Cell-subsampling robustness  [paper Fig S5]

完全对应 Osorio et al. 2022, Result 5 末段:
  "We randomly selected 500 cells each time and repeated the process ten
   times. For each subsampled data, we applied scTenifoldKnk to knock out
   Trem2 and obtain a perturbation profile. ... We found that the
   perturbation profiles of 10 subsamples are significantly positively
   correlated (average Spearman's ρ = 0.55)."

为什么必须做（针对你的项目尤其重要）:
  你跑 KO 用的是 n_samp_cells=200 (5 个子采样 × 200 cells), 比论文的 500
  cells × 10 次少. 应该至少对**少数 top E3 hits** 在 1-2 个亚群上做这个鲁棒性
  检验, 提供「我们的结果在 n_samp_cells=200 下也稳定」的证据.

⚠️ 重要: 这个脚本与其他下游脚本不同, 它**需要调用 scTenifoldKnk 重跑 KO**.
   因此要在 sctenifoldpy conda 环境里跑, 而不是普通的 pandas 环境.
   一次只对几个 E3 + 一个亚群 + n_repeats 次重采样, 耗时较长.

输出:
  results/AIHA/downstream/robustness/<subset>_<E3>_repeat<i>.csv  (每次的 DR)
  results/AIHA/downstream/robustness/<subset>_<E3>_summary.csv     (重复间相关性)
  results/AIHA/downstream/figs/robustness/<subset>_<E3>.pdf        (相关性热图)
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
from _config import (
    DOWNSTREAM_DIR, FIGS_DIR, SUBSETS_DIR, set_mpl_style,
)

set_mpl_style()
ROB_DIR = DOWNSTREAM_DIR / "robustness"
ROB_FIG = FIGS_DIR / "robustness"
ROB_DIR.mkdir(exist_ok=True); ROB_FIG.mkdir(parents=True, exist_ok=True)


def run_one_subsample(adata, ko: str, n_cells: int, seed: int,
                      n_nets: int, n_samp_cells: int, td_rank: int) -> pd.DataFrame:
    """
    Run scTenifoldKnk on a single random subsample of cells.
    Returns DR DataFrame with cols: gene, Z, p_adj (already from build()).
    """
    import scanpy as sc
    from scTenifold import scTenifoldKnk as sKnk
    from scipy import stats as scstats
    try:
        from scipy.stats import false_discovery_control
    except Exception:
        false_discovery_control = None

    rng = np.random.default_rng(seed)
    n_total = adata.n_obs
    if n_cells > n_total:
        n_cells = n_total
    idx = rng.choice(n_total, n_cells, replace=False)
    sub = adata[idx].copy()
    # 转 gene × cell DataFrame
    X = sub.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    df = pd.DataFrame(X.T.astype("float32"),
                      index=sub.var_names, columns=sub.obs_names)

    knk = sKnk(
        data=df, ko_method="default", ko_genes=[ko],
        qc_kws={"min_lib_size": 200, "min_percent": 0.001},
        nc_kws={"n_nets": n_nets, "n_samp_cells": n_samp_cells, "n_cpus": 1},
        td_kws={"K": td_rank},
    )
    knk.run_step("qc")
    knk.run_step("nc")
    knk.run_step("td")

    # check ko gene survived
    wt_tensor = knk.tensor_dict["WT"]
    if ko not in wt_tensor.index:
        return pd.DataFrame()

    knk_ko = sKnk(
        data=df, ko_method="default", ko_genes=[ko],
        qc_kws={"min_lib_size": 200, "min_percent": 0.001},
        nc_kws={"n_nets": n_nets, "n_samp_cells": n_samp_cells, "n_cpus": 1},
        td_kws={"K": td_rank},
    )
    knk_ko.QC_dict      = knk.QC_dict
    knk_ko.network_dict = knk.network_dict
    knk_ko.tensor_dict  = {"WT": wt_tensor.copy(), "KO": wt_tensor.copy()}
    knk_ko.manifold     = None
    knk_ko.run_step("ko")
    knk_ko.run_step("ma")
    knk_ko.run_step("dr")
    dr = knk_ko.d_regulation.copy()
    # 列名: Gene, Distance, FC, T, Z
    p = 2 * scstats.norm.sf(np.abs(dr["Z"].fillna(0).values))
    if false_discovery_control is not None:
        dr["p_adj"] = false_discovery_control(np.clip(p, 1e-300, 1))
    else:
        dr["p_adj"] = p   # fallback: 不校正
    dr = dr.rename(columns={"Gene": "gene"})
    return dr[["gene", "Z", "Distance", "FC", "p_adj"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True,
                    help="例如 B_naive")
    ap.add_argument("--ko-genes", nargs="+", required=True,
                    help="例如 --ko-genes TRIM21 BRCA1")
    ap.add_argument("--n-cells", type=int, default=500,
                    help="每次重采样的细胞数 (论文用 500)")
    ap.add_argument("--n-repeats", type=int, default=10,
                    help="重复次数 (论文用 10)")
    ap.add_argument("--n-nets", type=int, default=5)
    ap.add_argument("--n-samp-cells", type=int, default=200)
    ap.add_argument("--td-rank", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import scanpy as sc
    h5 = SUBSETS_DIR / f"{args.subset}.h5ad"
    if not h5.exists():
        print(f"Missing {h5}", file=sys.stderr); sys.exit(1)
    print(f"Loading {h5}...")
    adata = sc.read_h5ad(h5)
    print(f"  → {adata.n_obs} cells × {adata.n_vars} genes")

    for ko in args.ko_genes:
        print(f"\n=== {args.subset} / {ko}: {args.n_repeats} repeats × {args.n_cells} cells ===")
        all_z = {}
        for i in range(args.n_repeats):
            out_csv = ROB_DIR / f"{args.subset}_{ko}_repeat{i:02d}.csv"
            if out_csv.exists() and not args.force:
                dr = pd.read_csv(out_csv)
            else:
                try:
                    dr = run_one_subsample(adata, ko, args.n_cells, seed=i,
                                           n_nets=args.n_nets,
                                           n_samp_cells=args.n_samp_cells,
                                           td_rank=args.td_rank)
                except Exception as e:
                    print(f"  repeat {i}: failed: {e}")
                    continue
                if dr.empty:
                    print(f"  repeat {i}: {ko} not in tensor genes")
                    continue
                dr.to_csv(out_csv, index=False)
            all_z[f"r{i:02d}"] = dr.set_index("gene")["Z"]
            print(f"  repeat {i}: {len(dr)} genes")

        if len(all_z) < 2:
            continue
        Z = pd.DataFrame(all_z).fillna(0)
        corr = Z.corr(method="spearman")
        # 平均 r (off-diagonal)
        n = corr.shape[0]
        offd = corr.values[np.triu_indices(n, k=1)]
        mean_rho = float(np.mean(offd))

        # 显著基因 Jaccard
        sig = {c: set(Z.index[(Z[c].abs() > 0).values & (Z[c].notna().values)]) for c in Z.columns}
        jacs = []
        for a, b in combinations(Z.columns, 2):
            A, B = sig[a], sig[b]
            jacs.append(len(A & B) / max(1, len(A | B)))
        # 也写一份显著基因 Jaccard 矩阵
        jac_mat = pd.DataFrame(1.0, index=Z.columns, columns=Z.columns)
        for a, b in combinations(Z.columns, 2):
            A, B = sig[a], sig[b]
            jac_mat.loc[a, b] = jac_mat.loc[b, a] = len(A & B) / max(1, len(A | B))

        # 图
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    vmin=-1, vmax=1, ax=axs[0])
        axs[0].set_title(f"{args.subset} / {ko}\nZ-vector Spearman (mean off-diag = {mean_rho:.2f})")
        sns.heatmap(jac_mat, annot=True, fmt=".2f", cmap="Purples", vmin=0, vmax=1, ax=axs[1])
        axs[1].set_title("Significant-gene Jaccard")
        out_pdf = ROB_FIG / f"{args.subset}_{ko}.pdf"
        fig.savefig(out_pdf)
        plt.close(fig)

        # 汇总
        pd.Series({
            "subset": args.subset, "ko": ko,
            "n_repeats": len(Z.columns),
            "n_cells": args.n_cells,
            "mean_spearman": mean_rho,
            "mean_jaccard": float(np.mean(jacs)),
        }).to_frame().T.to_csv(
            ROB_DIR / f"{args.subset}_{ko}_summary.csv", index=False)
        print(f"  mean Spearman = {mean_rho:.3f}  | mean Jaccard = {np.mean(jacs):.3f}")


if __name__ == "__main__":
    main()