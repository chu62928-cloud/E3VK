#!/usr/bin/env python
"""
09_plot_enrichment.py — 基于 NES 方差的特异性可视化 (Publication Quality)
特点：抛弃单纯的差异基因数量，改用通路 NES 的方差 (极化程度) 来挑选最具特异性生物学意义的 E3 KOs。

DEPRECATED: NES variance KO selection logic has been merged into viz_enrichment.py.
Use `viz_enrichment.py --ko-select nes_var` instead.
This script is kept as a reference/backup only.
"""
from __future__ import annotations
import argparse
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.linewidth"] = 1.0     
plt.rcParams["axes.labelsize"] = 12      
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

sys.path.insert(0, str(Path(__file__).parent))
BASE_DIR = Path("results/AIHA/downstream")
DOWNSTREAM_DIR = BASE_DIR
ENRICH_DIR = BASE_DIR / "enrichment_v1"

PLOT_DIR = DOWNSTREAM_DIR / "plots"

# 黑名单：过滤基础代谢和疾病噪音
BLACKLIST_TERMS = [
    "ribosom", "translation", "rna processing", "spliceosome", "proteasome",
    "alzheimer", "parkinson", "huntington", "coronavirus", "infection", "disease",
    "chemical carcinogenesis", "diabetic", "amyotrophic", "prion"
]

def is_premium_database(term: str) -> bool:
    """只允许 Hallmark 和 KEGG"""
    term_lower = term.lower()
    return "hallmark" in term_lower or "kegg" in term_lower

def clean_term_name_premium(term: str, max_len: int = 50) -> str:
    """清理通路名称为高级格式"""
    if "__" in term:
        term = term.split("__", 1)[1]
    if term.upper().startswith("HALLMARK_"):
        term = term[9:]
    term = term.replace("Homo sapiens", "").replace("WP", "")
    term = re.sub(r'\s*\([^)]+\)', '', term) 
    term = term.replace('_', ' ').strip()
    term = term.title()
    
    replacements = {"Nf-Kappab": "NF-kB", "Jak-Stat": "JAK-STAT", "Pi3K": "PI3K", 
                    "Mtor": "mTOR", "Tgf": "TGF", "Tnf": "TNF", "Il6": "IL-6", "Vegf": "VEGF",
                    "ErbB": "ErbB", "Fc Epsilon Ri": "Fc Epsilon RI"}
    for k, v in replacements.items():
        term = term.replace(k, v)
        
    if len(term) > max_len:
        return term[:max_len-3] + "..."
    return term

def is_interesting_term(term_clean: str) -> bool:
    term_lower = term_clean.lower()
    for black_word in BLACKLIST_TERMS:
        if black_word in term_lower:
            return False
    return True

def get_ko_specificity_score(file_path: Path) -> float:
    """
    【核心创新】：计算一个 KO 的特异性得分。
    使用过滤后的 NES 绝对值之和与方差的综合指标，找出“极化最严重”的靶点。
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty or "FDR q-val" not in df.columns: return 0.0
        
        # 宽容过滤以计算整体方差趋势
        df = df[df["Term"].apply(is_premium_database)]
        if "NOM p-val" in df.columns:
            df_sig = df[df["NOM p-val"] < 0.05].copy()
        else:
            df_sig = df[df["FDR q-val"] < 0.25].copy()
            
        if len(df_sig) < 3: return 0.0 # 通路太少，没有方差意义
        
        df_sig["Term_clean"] = df_sig["Term"].apply(clean_term_name_premium)
        df_sig = df_sig[df_sig["Term_clean"].apply(is_interesting_term)]
        
        if len(df_sig) < 3: return 0.0
        
        # NES的方差越大，说明上调和下调两极分化越严重，特异性越强
        return df_sig["NES"].var()
    except Exception:
        return 0.0

def plot_gsea_dotplot(file_path: Path, out_path: Path, top_n: int = 10, ko_name: str = "", score: float = 0.0):
    """根据分数挑选后，绘制高级气泡图"""
    if not file_path.exists(): return
    df = pd.read_csv(file_path)
    
    df = df[df["Term"].apply(is_premium_database)].copy()
    df_sig = df[df["FDR q-val"] < 0.25].copy()  
    if df_sig.empty and "NOM p-val" in df.columns:
        df_sig = df[df["NOM p-val"] < 0.05].copy()
            
    if df_sig.empty: return
            
    df_sig["Term_clean"] = df_sig["Term"].apply(clean_term_name_premium)
    df_sig = df_sig[df_sig["Term_clean"].apply(is_interesting_term)]
    if df_sig.empty: return
    
    df_up = df_sig[df_sig["NES"] > 0].sort_values("NES", ascending=False).head(top_n)
    df_dn = df_sig[df_sig["NES"] < 0].sort_values("NES", ascending=True).head(top_n)
    plot_df = pd.concat([df_up, df_dn]).sort_values("NES", ascending=True)
    if plot_df.empty: return
        
    fdr_col = "FDR q-val" if "FDR q-val" in plot_df.columns and plot_df["FDR q-val"].min() < 0.25 else "NOM p-val"
    plot_df["-log10(Sig)"] = -np.log10(plot_df[fdr_col].replace(0, 1e-5))
    
    fig, ax = plt.subplots(figsize=(6, max(4, len(plot_df)*0.4)))
    max_abs_nes = max(abs(plot_df["NES"].min()), abs(plot_df["NES"].max()))
    max_abs_nes = max(max_abs_nes, 0.5)
    
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='gray', zorder=0)
    scatter = ax.scatter(
        x=plot_df["NES"], y=plot_df["Term_clean"], 
        s=plot_df["-log10(Sig)"] * 50, c=plot_df["NES"],
        cmap="RdBu_r", vmin=-max_abs_nes, vmax=max_abs_nes, 
        edgecolors="black", linewidth=0.5, alpha=0.85, zorder=3
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, zorder=1)
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Normalized Enrichment Score (NES)", fontsize=10, weight='bold')
    for size in [1, 2, 3]:
        ax.scatter([], [], s=size*50, c='gray', alpha=0.6, edgecolors='black', label=f'{10**(-size)}')
    ax.legend(title=f"{fdr_col}", bbox_to_anchor=(1.25, 0.95), loc='upper left', frameon=False)

    ax.set_xlabel("NES", weight='bold')
    ax.set_title(f"Core Biological Signatures\n(KO: {ko_name} | VarScore: {score:.2f})", weight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_gsea_heatmap(subset: str, gsea_dir: Path, out_path: Path, top_n: int = 30):
    """挑选特异性得分最高的 KO 参与热图绘制，减少横向冗余"""
    files = list(gsea_dir.glob("*_gsea.csv"))
    if not files: return
        
    all_res = []
    # 记录每个KO的方差得分
    ko_scores = {}
    
    for f in files:
        ko = f.stem.replace("_gsea", "")
        score = get_ko_specificity_score(f)
        if score > 0:
            ko_scores[ko] = score
            
        df = pd.read_csv(f)
        if not df.empty and "FDR q-val" in df.columns:
            df = df[df["Term"].apply(is_premium_database)].copy()
            df = df[df["FDR q-val"] < 0.25].copy()
            if not df.empty:
                df["KO"] = ko
                all_res.append(df[["KO", "Term", "NES"]])
                
    if not all_res: return
    merged = pd.concat(all_res)
    
    # 【优化】：只保留排名前 25 的高方差 KO，让热图更精炼、特征更明显
    top_kos = sorted(ko_scores, key=ko_scores.get, reverse=True)[:25]
    merged = merged[merged["KO"].isin(top_kos)]
    
    merged["Term_clean"] = merged["Term"].apply(clean_term_name_premium)
    merged = merged[merged["Term_clean"].apply(is_interesting_term)]
    
    if merged.empty: return
    
    nes_matrix_raw = merged.pivot_table(index="Term_clean", columns="KO", values="NES").fillna(0)
    if nes_matrix_raw.shape[0] < 2 or nes_matrix_raw.shape[1] < 2: return
        
    # 提取变化最剧烈的 Top 通路
    term_variance = nes_matrix_raw.var(axis=1).sort_values(ascending=False)
    top_terms = term_variance.head(top_n).index
    nes_matrix = nes_matrix_raw.loc[top_terms]
    max_abs = max(abs(nes_matrix.min().min()), abs(nes_matrix.max().max()))
    
    sns.set_context("paper", font_scale=0.9)
    g = sns.clustermap(
        nes_matrix, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, center=0, 
        figsize=(min(14, 4 + 0.4*nes_matrix.shape[1]), min(10, 2 + 0.35*nes_matrix.shape[0])),
        linewidths=0.5, linecolor='white', cbar_kws={"label": "NES"}
    )
    g.fig.suptitle(f"Specific Signature Divergence - {subset}", y=1.02, fontsize=14, weight='bold')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-gsea-ko", type=int, default=4, help="按方差排名挑选几个最特异的靶点作图")
    ap.add_argument("--top-gsea-terms", type=int, default=10)
    ap.add_argument("--top-heatmap-terms", type=int, default=30)
    args = ap.parse_args()

    target_plot_dir = DOWNSTREAM_DIR / "plots_specificity_ranked"
    dot_dir = target_plot_dir / "enrichment_dotplots"
    heat_dir = target_plot_dir / "enrichment_heatmaps"
    dot_dir.mkdir(parents=True, exist_ok=True)
    heat_dir.mkdir(parents=True, exist_ok=True)

    if ENRICH_DIR.exists():
        subsets = [p.name for p in ENRICH_DIR.iterdir() if p.is_dir()]
        for subset in subsets:
            print(f"\n--- Analyzing Subset: {subset} ---")
            gsea_sub_dir = ENRICH_DIR / subset
            
            # 第一步：计算该亚群下所有 KOs 的特异性方差得分
            ko_scores = {}
            for f in gsea_sub_dir.glob("*_gsea.csv"):
                ko = f.stem.replace("_gsea", "")
                score = get_ko_specificity_score(f)
                if score > 0:
                    ko_scores[ko] = score
            
            if not ko_scores:
                print("  [!] No valid KOs found for specificity ranking.")
                continue
                
            # 第二步：按方差从大到小排序，选取 Top KOs
            ranked_kos = sorted(ko_scores.items(), key=lambda x: x[1], reverse=True)
            print(f"  Top Specific KOs (by NES Variance):")
            for ko, score in ranked_kos[:args.top_gsea_ko]:
                print(f"    - {ko}: {score:.3f}")
            
            # 绘制热图 (自动选取 Top KOs)
            heat_out = heat_dir / f"{subset}_specificity_heatmap.pdf"
            plot_gsea_heatmap(subset, gsea_sub_dir, heat_out, top_n=args.top_heatmap_terms)
            
            # 绘制气泡图
            for ko, score in ranked_kos[:args.top_gsea_ko]:
                f = gsea_sub_dir / f"{ko}_gsea.csv"
                dot_out = dot_dir / f"{subset}_{ko}_specific_dotplot.pdf"
                plot_gsea_dotplot(f, dot_out, top_n=args.top_gsea_terms, ko_name=ko, score=score)

    print(f"\nAll done! New plots saved to: {target_plot_dir}")

if __name__ == "__main__":
    main()