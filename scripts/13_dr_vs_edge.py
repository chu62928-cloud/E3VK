#!/usr/bin/env python
"""
13_dr_vs_edge.py — DR distance vs edge weight in WT scGRN  [Fig 5 of paper]

完全对应 Osorio et al. 2022, Result 5 & Fig 5:
  "we examined the correlation between each gene's DR distance (estimated via
   manifold alignment after virtual KO of Trem2) and the edge weight of the
   link between the gene and Trem2. We found that the correlation was
   positive and significant (Spearman's r = 0.69, p value < 0.001), ...
   However, the relationship is not linear—absolute values of the edge
   weights of most significant DR genes are variable, ranging from 0.075 to
   0.40 ... significant DR genes identified using scTenifoldKnk are not
   necessarily always those strongly linked with the KO gene. Thus,
   scTenifoldKnk is not simply analyzing the adjacency matrix..."

为什么必须做（针对你的项目特别关键）:
  审稿人会问: "你不就是把 GRN 里和 E3 直接相连的基因列出来吗?"
  这张图证明: scTenifoldKnk 通过 manifold alignment 能捕到那些与 E3 弱相连
  但功能相关的基因 — 这是 virtual KO 区别于 co-expression 分析的核心.
  对每个 top E3 hit 必须出一张, 才能说服读者.

输入要求:
  - 每个 (subset, E3) 都需要拿到 WT scGRN 邻接矩阵的「KO 行」(即 E3 -> 其它基因的边权).
  - 这些信息保存在 <subset>_knk.pkl 里 (scTenifold 的 network_dict / tensor_dict["WT"]).
  - 我们的脚本会尝试 pickle.load, 失败则提示用户把 W_d 单独导出.

输出:
  results/AIHA/downstream/figs/dr_vs_edge/<subset>/<E3>.pdf
  results/AIHA/downstream/dr_vs_edge_summary.csv   (Spearman r, p, n_sig)
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
    DOWNSTREAM_DIR, FIGS_DIR, KNK_DIR, FDR_STRICT, set_mpl_style,
)

set_mpl_style()
DRE_DIR = FIGS_DIR / "dr_vs_edge"
DRE_DIR.mkdir(parents=True, exist_ok=True)


def load_wt_network(subset: str) -> pd.DataFrame | None:
    """
    Try to recover W_d (denoised WT adjacency, gene × gene) from the saved
    <subset>_knk.pkl. scTenifold stores it as a DataFrame in
    knk.tensor_dict['WT'].

    Returns DataFrame index/cols = gene names, values = signed edge weights.
    """
    pkl = KNK_DIR / subset / f"{subset}_knk.pkl"
    if not pkl.exists():
        return None
    try:
        with open(pkl, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"  [{subset}] cannot load pkl: {e}", file=sys.stderr)
        return None
    # scTenifold 不同版本 API 略有差异; 多个 fallback
    for key in ("tensor_dict", "_tensor_dict"):
        td = getattr(obj, key, None)
        if td and "WT" in td:
            wt = td["WT"]
            if isinstance(wt, pd.DataFrame):
                return wt
            # 若是 ndarray, 找 gene 名
            return None
    # 老版本可能存在 obj.network_dict
    nd = getattr(obj, "network_dict", None)
    if isinstance(nd, dict) and nd:
        first = next(iter(nd.values()))
        if isinstance(first, pd.DataFrame):
            return first
    return None


def plot_one(z_abs: pd.Series, edge_w: pd.Series, sig_mask: pd.Series,
             ko: str, out: Path):
    """Density scatter; significant genes in red, with top-5 labeled."""
    # 对齐
    common = z_abs.index.intersection(edge_w.index)
    x = z_abs.reindex(common).values
    y = np.abs(edge_w.reindex(common).values)
    sig = sig_mask.reindex(common).values.astype(bool)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, sig = x[valid], y[valid], sig[valid]
    if len(x) < 20:
        return None

    rho, pval = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(5, 4))
    # 密度（非显著）
    ax.scatter(x[~sig], y[~sig], s=4, c="#5F5E5A", alpha=0.25, rasterized=True)
    # 显著基因
    ax.scatter(x[sig], y[sig], s=15, marker="*", c="#A32D2D",
               label=f"significant ({sig.sum()})", zorder=3)
    # 标注 top-5 |Z|
    sig_idx = np.where(sig)[0]
    if len(sig_idx) > 0:
        top5 = sig_idx[np.argsort(-x[sig_idx])[:5]]
        common_arr = common[valid]
        for k in top5:
            ax.annotate(common_arr[k], (x[k], y[k]), fontsize=6,
                        xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Z-score (|DR distance|)")
    ax.set_ylabel(f"|Edge weight|  (gene ↔ {ko})")
    ax.set_title(f"{ko}: Spearman ρ = {rho:.2f}, p = {pval:.2e}")
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return rho, pval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-per-subset", type=int, default=10)
    ap.add_argument("--fdr", type=float, default=FDR_STRICT)
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    vuln = pd.read_csv(DOWNSTREAM_DIR / "vulnerability.csv")
    subsets = args.subsets or sorted(vuln["subset"].unique())

    summary = []
    for subset in subsets:
        wt = load_wt_network(subset)
        if wt is None:
            print(f"[{subset}] skip: cannot recover WT scGRN from pkl. "
                  f"Consider exporting W_d to results/AIHA/knk/{subset}/Wd.parquet")
            # 兜底: 看 Wd.parquet
            wd_fallback = KNK_DIR / subset / "Wd.parquet"
            if wd_fallback.exists():
                wt = pd.read_parquet(wd_fallback)
                print(f"  loaded fallback {wd_fallback.name} shape={wt.shape}")
            else:
                continue
        zmat = pd.read_parquet(DOWNSTREAM_DIR / f"dr_zmat_{subset}.parquet")
        padj = pd.read_parquet(DOWNSTREAM_DIR / f"dr_padj_{subset}.parquet")

        sub_vuln = vuln[vuln["subset"] == subset].sort_values("n_sig_05", ascending=False)
        sub_vuln = sub_vuln.head(args.top_per_subset)

        outdir = DRE_DIR / subset
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"[{subset}] plotting top {len(sub_vuln)} E3 KOs (WT scGRN {wt.shape})")
        for _, row in sub_vuln.iterrows():
            ko = row["ko"]
            if ko not in wt.index:
                continue
            out = outdir / f"{ko}.pdf"
            if out.exists() and not args.force:
                continue
            z_abs = zmat[ko].abs() if ko in zmat.columns else None
            if z_abs is None:
                continue
            edge_w = wt.loc[ko]  # KO -> 其他基因的边权
            sig_mask = padj[ko] < args.fdr
            res = plot_one(z_abs, edge_w, sig_mask, ko, out)
            if res:
                rho, pval = res
                summary.append({"subset": subset, "ko": ko,
                                "spearman_rho": rho, "p_value": pval,
                                "n_sig": int(sig_mask.sum())})
    if summary:
        sdf = pd.DataFrame(summary)
        sdf.to_csv(DOWNSTREAM_DIR / "dr_vs_edge_summary.csv", index=False)
        print(f"\nWrote dr_vs_edge_summary.csv ({len(sdf)} rows)")
        print(f"  median ρ across all E3: {sdf['spearman_rho'].median():.2f}")


if __name__ == "__main__":
    main()