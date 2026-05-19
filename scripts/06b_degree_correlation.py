#!/usr/bin/env python
"""
06b_degree_correlation.py — n_sig DR genes vs E3 node degree in scGRN

完全对应 Osorio et al. 2022, Result 3 末段 + Fig S1:
  "We analyzed the relationship between the number of significantly
   perturbed genes and the degree (i.e., the number of connections) of the
   KO gene in the network. We found a positive correlation between the two
   metrics (Pearson's r = 0.75, P = 0.03). Hnf4a and Hnf4g exhibited a lower
   degree (i.e., fewer connections) than Nkx2-1 and Trem2, which may
   explain why KO of Hnf4a and Hnf4g produced fewer perturbed genes."

对 E3VK 项目的意义:
  有些 E3 KO 出来显著基因数很少 (e.g. <10), 这未必说明该 E3 不重要 ——
  可能只是它在 GRN 中度数低. 把这层混杂报告出来, 给读者一个 caveat.

依赖:
  - 需要从 <subset>_knk.pkl 里读 WT scGRN 的邻接矩阵
  - 或者从 results/AIHA/knk/<subset>/Wd.parquet 兜底
  - 度数 = |W_d[gene, :]| > eps 的列数 (out-degree)
         + |W_d[:, gene]| > eps 的行数 (in-degree)

输出:
  results/AIHA/downstream/figs/degree_correlation/<subset>.pdf
  results/AIHA/downstream/e3_degree.csv  (subset, ko, out_deg, in_deg, n_sig_05)
"""
from __future__ import annotations
import argparse
import pickle
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
    DOWNSTREAM_DIR, FIGS_DIR, KNK_DIR, set_mpl_style,
)

set_mpl_style()
DEG_DIR = FIGS_DIR / "degree_correlation"
DEG_DIR.mkdir(parents=True, exist_ok=True)


def load_wt(subset: str) -> pd.DataFrame | None:
    """Try multiple sources to get WT scGRN adjacency matrix."""
    # Fallback parquet
    wd = KNK_DIR / subset / "Wd.parquet"
    if wd.exists():
        return pd.read_parquet(wd)
    # pkl
    pkl = KNK_DIR / subset / f"{subset}_knk.pkl"
    if pkl.exists():
        try:
            with open(pkl, "rb") as f:
                obj = pickle.load(f)
            td = getattr(obj, "tensor_dict", None) or getattr(obj, "_tensor_dict", None)
            if td and "WT" in td and isinstance(td["WT"], pd.DataFrame):
                return td["WT"]
        except Exception as e:
            print(f"  [{subset}] pkl load failed: {e}", file=sys.stderr)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.01,
                    help="边权阈值 (绝对值 > eps 算作一条边)")
    ap.add_argument("--subsets", nargs="+", default=None)
    args = ap.parse_args()

    vuln = pd.read_csv(DOWNSTREAM_DIR / "vulnerability.csv")
    subsets = args.subsets or sorted(vuln["subset"].unique())

    all_rows = []
    for subset in subsets:
        wt = load_wt(subset)
        if wt is None:
            print(f"[{subset}] no WT scGRN found, skip "
                  f"(expected Wd.parquet or *_knk.pkl)"); continue
        wt_abs = wt.abs() > args.eps
        out_deg = wt_abs.sum(axis=1)   # row sum: edges out of gene
        in_deg  = wt_abs.sum(axis=0)   # col sum

        sub_vuln = vuln[vuln["subset"] == subset].copy()
        sub_vuln["out_deg"] = sub_vuln["ko"].map(out_deg.to_dict()).astype(float)
        sub_vuln["in_deg"]  = sub_vuln["ko"].map(in_deg.to_dict()).astype(float)
        sub_vuln["total_deg"] = sub_vuln["out_deg"] + sub_vuln["in_deg"]
        # 论文用 out_deg (since KO sets row to 0 -> out edges)
        df = sub_vuln.dropna(subset=["out_deg", "n_sig_05"])
        if df.empty:
            continue
        if df["out_deg"].nunique() <= 1 or df["n_sig_05"].nunique() <= 1:
            print(f"[{subset}] Constant input detected (zero variance). Skipping correlation.")
            r, p = np.nan, np.nan
            rs, ps = np.nan, np.nan
            title_str = f"{subset}: Cannot compute correlation (constant data)"
        else:
            r, p = stats.pearsonr(df["out_deg"], df["n_sig_05"])
            rs, ps = stats.spearmanr(df["out_deg"], df["n_sig_05"])
            title_str = f"{subset}: Pearson r={r:.2f} (p={p:.1e}) | Spearman ρ={rs:.2f}"

        # 图
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(df["out_deg"], df["n_sig_05"], s=15, c="#26215C", alpha=0.6)
        # 标记 top 5 / bottom 5
        n_labels = min(5, len(df))
        top = df.nlargest(5, "n_sig_05")
        bot = df.nsmallest(5, "n_sig_05")
        for _, r_ in pd.concat([top, bot]).drop_duplicates(subset=["ko"]).iterrows():
            ax.annotate(r_["ko"], (r_["out_deg"], r_["n_sig_05"]),
                        fontsize=6, xytext=(2, 2), textcoords="offset points")
                        
        ax.set_xlabel("E3 out-degree in WT scGRN")
        ax.set_ylabel("# significant DR genes (p_adj < 0.05)")
        ax.set_title(title_str, fontsize=10) # 自动应用上面的 title_str
        fig.tight_layout()
        
        fig.savefig(DEG_DIR / f"{subset}.pdf")
        plt.close(fig)
        all_rows.append(df.assign(pearson_r=r, pearson_p=p))
        
        # 打印日志时同样处理 NaN
        if pd.isna(r):
            print(f"[{subset}] Correlation is NaN, n_E3={len(df)}")
        else:
            print(f"[{subset}] Pearson r={r:.3f}, p={p:.2e}, n_E3={len(df)}")

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(DOWNSTREAM_DIR / "e3_degree.csv", index=False)
        print(f"\nWrote e3_degree.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()