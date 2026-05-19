#!/usr/bin/env python
"""
08_perturbation_landscape.py — 绘制 E3 扰动图谱景观 (t-SNE 极简标签高发版)
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from _config import DOWNSTREAM_DIR, FIGS_DIR, set_mpl_style, load_e3_family

set_mpl_style()
LANDSCAPE_DIR = FIGS_DIR / "landscape"
LANDSCAPE_DIR.mkdir(parents=True, exist_ok=True)

def plot_landscape(subset: str, perplexity: int = 30):
    zmat_path = DOWNSTREAM_DIR / f"dr_zmat_{subset}.parquet"
    if not zmat_path.exists():
        print(f"[{subset}] 找不到 {zmat_path.name}")
        return

    print(f"[{subset}] 正在加载 Z 矩阵...")
    zmat = pd.read_parquet(zmat_path)
    
    # 过滤全网无变化的基因
    zmat = zmat.loc[zmat.std(axis=1) > 0]
    
    # 转置：行为 E3，列为基因特征
    features = zmat.T
    X_input = features.values

    print(f"[{subset}] 正在运行 t-SNE 降维 (E3数量: {features.shape[0]}, 特征数: {features.shape[1]})...")
    actual_perp = min(perplexity, features.shape[0] - 1)
    # tsne = TSNE(n_components=2, perplexity=actual_perp, random_state=42, init='pca')
    # embedding = tsne.fit_transform(X_input)
    #df_plot = pd.DataFrame(embedding, columns=["tSNE_1", "tSNE_2"], index=features.index)

    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(X_input)
    explained_var = pca.explained_variance_ratio_ # 获取方差解释率

    pc1_loadings = pd.Series(pca.components_[0], index=features.columns)
    
    # 找出对 PC1 贡献最大的前 20 个基因
    top_driver_genes = pc1_loadings.abs().sort_values(ascending=False).head(20)
    
    print(f"\n=========================================")
    print(f"[{subset}] 导致网络产生 '双稳态/左右撕裂' 的核心驱动基因 (PC1 Loadings):")
    print(top_driver_genes)
    print(f"=========================================\n")
    
    df_plot = pd.DataFrame(embedding, columns=["PC_1", "PC_2"], index=features.index)
    df_plot["E3"] = df_plot.index

    # 关联家族信息
    try:
        fam_df = load_e3_family()
        df_plot = df_plot.merge(fam_df[["gene", "family"]], left_on="E3", right_on="gene", how="left")
        df_plot["family"] = df_plot["family"].fillna("Other")
    except Exception:
        df_plot["family"] = "Unknown"

    # 计算生物学方差（破坏力大小），并将其映射为散点的大小(size)，让核心枢纽在视觉上更大
    e3_variance = features.var(axis=1)
    df_plot["biological_variance"] = df_plot["E3"].map(e3_variance)
    # 将大小控制在 25 到 250 之间
    df_plot["point_size"] = 25 + (df_plot["biological_variance"] / df_plot["biological_variance"].max()) * 225

    # =========================================================================
    # 核心升级：使用 K-NN 计算真正的“局部离群点” (Local Spatial Outliers)
    # =========================================================================
    # 取出所有点在 t-SNE 上的 2D 坐标
    # coords = df_plot[["tSNE_1", "tSNE_2"]].values
    coords = df_plot[["PC_1", "PC_2"]].values
    
    # 寻找每个点最近的 6 个点 (包含自己，所以实际是找 5 个真正的邻居)
    nn = NearestNeighbors(n_neighbors=6)
    nn.fit(coords)
    distances, _ = nn.kneighbors(coords)
    
    # 取这 5 个邻居距离的平均值，作为该点的“孤立度 (isolation score)”
    # 分数越大，说明该点周围越空旷，越是真正的孤狼
    df_plot["isolation_score"] = distances[:, 1:].mean(axis=1)
    
    # 1. 寻找孤立度最大的 5 个【真正的空间离群点】
    spatial_outliers = df_plot.sort_values("isolation_score", ascending=False).head(5)

    # 2. 寻找全局破坏力（生物学方差）最大的前 5 个【核心枢纽点】
    top_variance = df_plot.sort_values("biological_variance", ascending=False).head(5)

    # 轨迹 2：寻找全局破坏力（生物学方差）最大的前 5 个核心点
    top_variance = df_plot.sort_values("biological_variance", ascending=False).head(10)

    # 合并去重
    combined_labels = pd.concat([spatial_outliers, top_variance]).drop_duplicates(subset=["E3"])

    # 开始画图
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    
    # 绘制散点，加入了 size=point_size 映射，破坏力大的点会自动变大
    sns.scatterplot(
        data=df_plot, x="PC_1", y="PC_2", 
        hue="family", palette="Set2", alpha=0.75, edgecolor="w", linewidth=0.5,
        size="point_size", sizes=(25, 250), legend="brief", ax=ax
    )
    
    # 专门为颜色图例拉一条干净的伪图例（剔除点大小的干扰）
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:5], labels[:5], loc="upper left",  frameon=True)

    # 绘制选定点的文本标签
    texts = []
    for _, row in combined_labels.iterrows():
        # 如果是空间离群点，用深灰色标记；如果是簇内核心点，用黑色标记
        is_outlier = row["E3"] in spatial_outliers["E3"].values
        color = "#C0392B" if is_outlier else "#111111"
        texts.append(ax.text(row["PC_1"], row["PC_2"], row["E3"], 
                             fontsize=9, weight="bold", color=color))
        
    # 利用 adjust_text 让这几个珍贵的标签绝对不发生任何重叠
    try:
        from adjustText import adjust_text
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.6, alpha=0.7))
    except ImportError:
        pass

    ax.set_title(f"{subset} - E3 Perturbation Landscape", fontsize=13, pad=15, weight="bold")
    # ax.set_xlabel("t-SNE 1", fontsize=10)
    # ax.set_ylabel("t-SNE 2", fontsize=10)

    #PCA
    ax.set_xlabel(f"PC_1 ({explained_var[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC_2 ({explained_var[1]*100:.1f}%)", fontsize=10)

    # 美化坐标轴网格线
    ax.grid(True, linestyle="--", alpha=0.3)
    
    out_file = LANDSCAPE_DIR / f"{subset}_landscape.pdf"
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[{subset}] 极简高级版景观图绘制完成！已保存至 {out_file}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=str, default="CD4_Th1")
    args = ap.parse_args()
    plot_landscape(args.subset)