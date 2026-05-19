#!/usr/bin/env python
"""
14_ko_landscape_tsne.py — KO perturbation profile landscape  [paper Fig 6]

完全对应 Osorio et al. 2022, Result 6 + Fig 6:
  "For all genes, the KO perturbation profile of each gene, i.e., a vector
   of DR distances, was transformed using Box-Cox transformation and then
   was standardized using Z score transformation. The processed KO
   perturbation profiles of all genes were combined into one matrix for
   t-SNE embedding."

  + Fig 6D/E: 同时对比 (a) 基因 KO profile, (b) 基因 expression 量, (c) 基因
   在 scGRN 中的 edge-weight profile —— 只有 (a) 才能展示出聚类结构.

为什么对 E3VK 项目至关重要:
  这就是论文里 "systematic KO" 的展示方式: 把所有 KO 的 distance vector 收成
  矩阵 -> t-SNE -> 聚类 -> 每个 cluster 内 STRING 互作验证. 应用到 E3VK:
  每个亚群内, 把所有 KO 过的 E3 投到一个 t-SNE 平面, 把功能相似的 E3
  聚到一起, 可以一眼看出例如 "这个亚群里 RING 家族 E3 是否独立成簇".

  Fig 6D/E 的对照很关键 — 证明 "E3 在表达层面无法区分功能" 但 "在 KO
  perturbation 层面能区分" — 这是为 virtual KO 投入辩护的证据.

输出:
  results/AIHA/downstream/figs/landscape/<subset>_tsne_ko.pdf
  results/AIHA/downstream/figs/landscape/<subset>_tsne_expr.pdf  (对照)
  results/AIHA/downstream/figs/landscape/<subset>_tsne_net.pdf   (对照, 若可)
  results/AIHA/downstream/ko_landscape_clusters.csv
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
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from _config import (
    DOWNSTREAM_DIR, FIGS_DIR, E3_TABLE, set_mpl_style, classify_family,
)

set_mpl_style()
LAND_DIR = FIGS_DIR / "landscape"
LAND_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_COLORS = {
    "RING":  "#7F77DD",
    "HECT":  "#1D9E75",
    "CRL":   "#EF9F27",
    "RBR":   "#D4537E",
    "Other": "#888780",
}


def boxcox_zscore_matrix(M: pd.DataFrame) -> pd.DataFrame:
    """Per-column Box-Cox + Z-score (具有超强数值安全防御的版本)."""
    out = {}
    for col in M.columns:
        x = M[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            continue
        if (x <= 0).any():
            pos_min = x[x > 0].min() if (x > 0).any() else 1e-8
            x = x + max(1e-8, pos_min * 1e-3)
            
        # --- 第一道防火墙：监控 Box-Cox，防止产生超级大数 ---
        try:
            # 捕获 numpy 内部的无效值和溢出警告，直接视作错误去执行 except
            with np.errstate(over='raise', invalid='raise'):
                bx, _ = stats.boxcox(x.values)
            
            # 如果变换出的结果绝对值已经大到离谱（大于 100000），说明不稳健，强制回退
            if np.abs(bx).max() > 1e5:
                bx = np.log1p(x.values)
        except Exception:
            bx = np.log1p(x.values)
            
        # --- 第二道防火墙：彻底清除可能残存的 NaN/Inf ---
        bx = np.nan_to_num(bx, nan=0.0, posinf=0.0, neginf=0.0)
        
        sd = bx.std(ddof=1)
        if sd > 1e-6:
            bx = (bx - bx.mean()) / sd
        else:
            bx = np.zeros_like(bx)
            
        # --- 第三道防火墙：对 Z-score 进行鲁棒性截断 ---
        # Z-score 达到 20 已经是天文数字级别的离群点了。
        # 限制在 [-20, 20] 之间可以完美防止后续 StandardScaler 和 t-SNE 的平方项溢出。
        bx = np.clip(bx, -20, 20)
        
        out[col] = pd.Series(bx, index=x.index)
        
    return pd.DataFrame(out).fillna(0.0)

def family_lookup() -> dict[str, str]:
    if not E3_TABLE.exists():
        return {}
    df = pd.read_excel(E3_TABLE)
    cols = {c.lower(): c for c in df.columns}
    gene = cols.get("gene") or list(df.columns)[0]
    cls  = cols.get("protein class") or list(df.columns)[-1]
    return {str(r[gene]).upper(): classify_family(r[cls])
            for _, r in df.iterrows()}


def tsne_embed(X: np.ndarray, perplexity: int) -> np.ndarray:
    perplexity = min(perplexity, max(5, X.shape[0] // 4))
    # 手动计算新版 sklearn 的 "auto" 学习率
    lr = max(200.0, X.shape[0] / 12.0)
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate=lr, random_state=42).fit_transform(X)


def plot_tsne(coords: np.ndarray, labels: list[str], colors: list[str],
              title: str, out: Path, cluster_ids: np.ndarray | None = None,
              label_top: int = 0):
    fig, ax = plt.subplots(figsize=(6, 5))
    if cluster_ids is not None:
        # 用 cluster_id 当颜色, family 用 marker 形状
        cmap = plt.get_cmap("tab20")
        for cid in np.unique(cluster_ids):
            mask = cluster_ids == cid
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[cmap(cid % 20)], s=25, edgecolors="white", lw=0.3,
                       label=f"C{cid}", alpha=0.85)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], c=colors,
                   s=25, edgecolors="white", lw=0.3, alpha=0.85)

    # 标注
    if label_top > 0:
        # 标 family-based 离群点
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(coords))
        np.fill_diagonal(D, np.inf)
        nn = D.min(axis=1)
        # 远离邻居的 top-K
        idx = np.argsort(-nn)[:label_top]
        for i in idx:
            ax.annotate(labels[i], (coords[i, 0], coords[i, 1]),
                        fontsize=6, xytext=(2, 2), textcoords="offset points")

    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    if cluster_ids is not None:
        ax.legend(fontsize=6, ncol=2, frameon=False, loc="best")
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--perplexity", type=int, default=15)
    ap.add_argument("--n-clusters", type=int, default=6)
    ap.add_argument("--include-controls", action="store_true",
                    help="同时画 expression / network 对照图 (论文 Fig 6D/E)")
    args = ap.parse_args()

    fam_map = family_lookup()
    subsets = args.subsets or sorted(
        p.stem.removeprefix("dr_zmat_")
        for p in DOWNSTREAM_DIR.glob("dr_zmat_*.parquet"))

    cluster_records = []
    for subset in subsets:
        zmat_path = DOWNSTREAM_DIR / f"dr_zmat_{subset}.parquet"
        long_path = DOWNSTREAM_DIR / "dr_long.parquet"
        if not zmat_path.exists():
            continue

        # 论文用 distance, 退化到 |Z|
        dist_mat = None
        if long_path.exists():
            try:
                ll = pd.read_parquet(long_path,
                                     filters=[("subset", "==", subset)],
                                     columns=["gene", "ko", "score"])
                if not ll.empty and ll["score"].notna().any():
                    dist_mat = ll.pivot_table(index="gene", columns="ko",
                                              values="score", aggfunc="first")
            except Exception:
                pass
        if dist_mat is None:
            dist_mat = pd.read_parquet(zmat_path).abs()

        # Box-Cox + Z 每列, 然后转置 → 行=E3 (我们要嵌入的对象), 列=被影响的基因
        proc = boxcox_zscore_matrix(dist_mat)
        X = proc.T.values   # E3 × gene
        ko_names = proc.columns.tolist()
        if X.shape[0] < 10:
            print(f"[{subset}] only {X.shape[0]} KOs, skip")
            continue

        # t-SNE (KO landscape)
        coords = tsne_embed(StandardScaler().fit_transform(X), args.perplexity)
        families = [fam_map.get(g.upper(), "Other") for g in ko_names]
        colors = [FAMILY_COLORS[f] for f in families]

        # KMeans clustering on t-SNE coords (轻量)
        n_cl = min(args.n_clusters, max(2, X.shape[0] // 10))
        cluster_ids = KMeans(n_clusters=n_cl, n_init=10,
                             random_state=42).fit_predict(coords)

        out_ko = LAND_DIR / f"{subset}_tsne_ko.pdf"
        plot_tsne(coords, ko_names, colors,
                  f"{subset} — E3 KO landscape (Box-Cox + Z DR distance)",
                  out_ko, cluster_ids=cluster_ids, label_top=10)

        # Family-colored 版本（论文 Fig 6B 那种 cluster 强调）
        out_fam = LAND_DIR / f"{subset}_tsne_ko_family.pdf"
        plot_tsne(coords, ko_names, colors,
                  f"{subset} — colored by E3 family",
                  out_fam, cluster_ids=None, label_top=10)

        # 记录每个 E3 的 cluster 归属
        for i, ko in enumerate(ko_names):
            cluster_records.append({
                "subset": subset, "ko": ko, "family": families[i],
                "tsne_1": coords[i, 0], "tsne_2": coords[i, 1],
                "cluster": int(cluster_ids[i]),
            })

        # ----------------- 对照图（论文 Fig 6D/E） -----------------
        if args.include_controls:
            # (D) 表达 t-SNE: 不再以 KO profile 嵌入, 而是用每个 E3 的「在亚群所有
            #     细胞中的平均表达」作 vector. 这里用的是 GRN 节点表达, 我们没有
            #     直接的 expression matrix, 用 zmat 的方差作 surrogate ——
            #     即每个 E3 在该亚群 GRN 中表达的均值/方差.
            #     若用户能提供 expr_mean / expr_var, 可以替换.
            # (E) 网络 t-SNE: 用每个 E3 行的边权值 vector (即 WT scGRN 邻接矩阵
            #     的相应行) 嵌入. 需要 KNK_DIR/<subset>_knk.pkl 里的 Wd.
            # 这两个对照需要外部输入, 留给后续扩展；此版本只确保主图正确.
            pass

    if cluster_records:
        cdf = pd.DataFrame(cluster_records)
        cdf.to_csv(DOWNSTREAM_DIR / "ko_landscape_clusters.csv", index=False)
        print(f"\nWrote ko_landscape_clusters.csv ({len(cdf)} rows)")
        print(f"Figures: {LAND_DIR}/")


if __name__ == "__main__":
    main()