#!/usr/bin/env python
"""
viz_enrichment_v2.py — 重写版

主要改动 vs v1:
  1. 黑名单过滤看家通路 (ribosome/translation/NMD/rRNA/splicing 等)
  2. 图1 改为按「亚群差异度」排序 term (高方差优先), 突出亚群特异信号
  3. 图2 改为「亚群特异 E3」: 每个亚群选在该亚群 vulnerability 显著高于其他
     亚群的 E3 (z-score across subsets), 气泡图才有差异意义
  4. 图3 只保留 human TF, 过滤 mouse
  5. 横轴 E3 加 family 颜色条 (RING/HECT/CRL/RBR/Other)
  6. Pathway 分类着色: 免疫/代谢/细胞周期/信号/其他

用法:
    conda activate sctenifoldpy
    cd /root/autodl-tmp/E3VK
    python viz_enrichment_v2.py
    python viz_enrichment_v2.py --fdr-gsea 0.25 --top-terms 25 --top-e3 12
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import seaborn as sns

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ─────────────────────────────────────────────────────────────────────────────
# 常量与黑名单
# ─────────────────────────────────────────────────────────────────────────────

# 看家/背景通路关键词 — 命中任意一个即过滤
BLACKLIST_KEYWORDS = [
    # 翻译/核糖体
    "ribosom", "translation", "trna", "rrna", "mrna", "nonsense.mediated",
    "nmd", "selenoamino", "peptide biosynthetic", "cotranslational",
    "start codon", "40s subunit", "60s subunit", "eukaryotic translation",
    "translation initiation", "srp-dependent", "l13a",
    # 转录通用
    "rna polymerase", "rna processing", "rna degradation", "rna splicing",
    "spliceosom",
    # 代谢看家
    "oxidative phosphorylation",   # 若不想看代谢可保留; 视情况
    # 病毒/感染（非免疫）
    "influenza", "viral rna transcription", "viral replication",
    # 极通用
    "cell cycle", "dna repair", "dna replication", "macromolecule biosynthetic",
    "gene expression",
]

def is_blacklisted(term: str) -> bool:
    t = term.lower()
    return any(kw in t for kw in BLACKLIST_KEYWORDS)

# Pathway 大类关键词 → 类别标签
PATHWAY_CATS = [
    ("Immune",    ["immune", "interferon", "cytokine", "nf-kb", "jak-stat",
                   "t cell", "b cell", "nk cell", "toll", "interleukin",
                   "chemokine", "complement", "antigen", "mhc", "innate",
                   "adaptive", "lymphocyte", "tcr", "bcr", "pd-1", "pd-l",
                   "cd28", "foxp3", "treg", "th1", "th17", "exhaustion",
                   "inflammatory", "inflammation", "ifn", "tnf", "nfat",
                   "pi3k", "akt signaling"]),
    ("Metabolism",["metabol", "lipid", "fatty acid", "glycolysis", "tca",
                   "oxidation", "cholesterol", "glucose", "amino acid",
                   "nucleotide", "purine", "pyrimidine", "bile", "steroid",
                   "sphingolipid", "phospholipid", "carbohydrate",
                   "mitochondria", "electron transport", "nadh", "atp"]),
    ("Ubiquitin", ["ubiquitin", "proteasom", "sumo", "nedd8", "cullin",
                   "e3 ligase", "deubiquitin", "protein degradation"]),
    ("Apoptosis", ["apoptosis", "apoptotic", "caspase", "bcl-2", "p53",
                   "cell death", "necroptosis", "ferroptosis"]),
    ("Signaling", ["mapk", "erk", "ras", "wnt", "notch", "hedgehog",
                   "tgf-beta", "vegf", "egfr", "hippo", "mtor", "ampk",
                   "autophagy", "calcium", "camp", "pkc", "g protein",
                   "receptor tyrosine kinase", "kinase"]),
]

PATHWAY_COLORS = {
    "Immune":     "#D85A30",
    "Metabolism": "#1D9E75",
    "Ubiquitin":  "#7F77DD",
    "Apoptosis":  "#D4537E",
    "Signaling":  "#EF9F27",
    "Other":      "#888780",
}

def pathway_category(term: str) -> str:
    t = term.lower()
    for cat, kws in PATHWAY_CATS:
        if any(kw in t for kw in kws):
            return cat
    return "Other"

# E3 family 颜色
FAMILY_COLORS = {
    "RING":  "#534AB7",
    "HECT":  "#0F6E56",
    "CRL":   "#BA7517",
    "RBR":   "#993556",
    "Other": "#5F5E5A",
}

LIBRARY_SHORT = {
    "KEGG_2021_Human.gmt":                       "KEGG",
    "KEGG_2019_Human.gmt":                       "KEGG",
    "GO_Biological_Process_2023.gmt":            "GO:BP",
    "GO_Biological_Process_2018.gmt":            "GO:BP",
    "GO_Molecular_Function_2023.gmt":            "GO:MF",
    "GO_Cellular_Component_2023.gmt":            "GO:CC",
    "Reactome_Pathways_2024.gmt":                "Reactome",
    "Reactome_2016.gmt":                         "Reactome",
    "WikiPathway_2021_Human.gmt":                "Wiki",
    "WikiPathways_2024_Human.gmt":               "Wiki",
    "WikiPathways_2019_Human.gmt":               "Wiki",
    "BioPlanet_2019.gmt":                        "BioPlanet",
    "MSigDB_Hallmark_2020.gmt":                  "Hallmark",
    "PanglaoDB_Augmented_2021.gmt":              "PanglaoDB",
    "MGI_Mammalian_Phenotype_Level_4_2021.gmt":  "MGI:MP",
}

def parse_term(raw: str) -> tuple[str, str]:
    parts = raw.split("__", 1)
    lib_raw, term = (parts[0], parts[1]) if len(parts) == 2 else ("", raw)
    lib = LIBRARY_SHORT.get(lib_raw, lib_raw.replace(".gmt", "").split("_")[0])
    term = re.sub(r"\s*\(GO:\d+\)", "", term).strip()
    if len(term) > 60:
        term = term[:57] + "…"
    return lib, term

# ─────────────────────────────────────────────────────────────────────────────
# 数据读取
# ─────────────────────────────────────────────────────────────────────────────

def load_all_gsea(enrich_dir: Path) -> pd.DataFrame:
    rows = []
    for sub_dir in sorted(enrich_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        subset = sub_dir.name
        for f in sorted(sub_dir.glob("*_gsea.csv")):
            # 跳过去重后文件
            if "dedup" in f.name:
                continue
            ko = f.stem.replace("_gsea", "")
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if df.empty:
                continue
            fdr_col  = next((c for c in ["FDR q-val","fdr","FDR"] if c in df.columns), None)
            nes_col  = next((c for c in ["NES","nes"] if c in df.columns), None)
            term_col = next((c for c in ["Term","term"] if c in df.columns), None)
            if not all([fdr_col, nes_col, term_col]):
                continue
            sub = df[[term_col, nes_col, fdr_col]].copy()
            sub.columns = ["raw_term","NES","FDR"]
            sub["NES"] = pd.to_numeric(sub["NES"], errors="coerce")
            sub["FDR"] = pd.to_numeric(sub["FDR"], errors="coerce")
            sub = sub.dropna(subset=["NES","FDR"])
            sub[["lib","term"]] = sub["raw_term"].apply(
                lambda x: pd.Series(parse_term(str(x))))
            sub["subset"] = subset
            sub["ko"]     = ko
            rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_e3_family(e3_table: Path) -> dict[str, str]:
    """读 E3-ome.xlsx → {gene_upper: family}"""
    if not e3_table.exists():
        return {}
    df = pd.read_excel(e3_table)
    cols = {c.lower(): c for c in df.columns}
    gene_col = cols.get("gene") or list(df.columns)[0]
    cls_col  = cols.get("protein class") or list(df.columns)[-1]

    def classify(s):
        if not isinstance(s, str):
            return "Other"
        s = s.lower()
        if "crl" in s or "cullin" in s or "f-box" in s or "btb" in s:
            return "CRL"
        if "rbr" in s:
            return "RBR"
        if "hect" in s:
            return "HECT"
        if "ring" in s or "u-box" in s:
            return "RING"
        return "Other"

    return {str(r[gene_col]).upper(): classify(r[cls_col])
            for _, r in df.iterrows()}


def ko_family_colors(ko_list: list[str], fam_map: dict[str, str]) -> list[str]:
    return [FAMILY_COLORS[fam_map.get(k.upper(), "Other")] for k in ko_list]


# ─────────────────────────────────────────────────────────────────────────────
# 图 1: 亚群差异通路热图 (过滤看家通路, 按跨亚群方差排序)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pathway_heatmap(gsea: pd.DataFrame, out: Path,
                         fdr_cut: float, top_n: int):
    """
    选 term 逻辑 (三步):
      1. 黑名单过滤
      2. 只保留 FDR < fdr_cut 的显著 (subset, E3, term)
      3. 按「跨亚群方差」排序: 方差越高 → 越是亚群特异通路
         不再按命中频率排, 避免普遍通路占据顶部
    """
    sig = gsea[gsea["FDR"] < fdr_cut].copy()
    sig = sig[~sig["term"].apply(is_blacklisted)]
    if sig.empty:
        print("  [Fig1] 过滤后无显著 term")
        return

    # 每 (subset, term) 取最佳 FDR
    agg = (sig.groupby(["term","lib","subset"])
              .agg(best_fdr=("FDR","min"), n_e3=("ko","nunique"))
              .reset_index())
    agg["neg_log"] = -np.log10(agg["best_fdr"].clip(1e-10))

    # pivot: term × subset
    pivot = agg.pivot_table(index="term", columns="subset",
                            values="neg_log", aggfunc="max").fillna(0)

    # 排序: 跨亚群方差 (std) 高 → 亚群特异
    # 同时要求至少在 1 个亚群里达到 -log10 > 3 (FDR < 0.001)
    pivot = pivot[pivot.max(axis=1) > 3]
    pivot["_std"] = pivot.std(axis=1)
    pivot = pivot.sort_values("_std", ascending=False).drop(columns="_std")

    # 从通路类别里每类取若干, 保证多样性
    term_lib = agg.groupby("term")["lib"].first().to_dict()
    pivot["_cat"] = [pathway_category(t) for t in pivot.index]
    pivot["_lib"] = [term_lib.get(t, "") for t in pivot.index]

    # 每个类别最多取 top_n//5 个 term (保证多样性), 总共 top_n
    selected = []
    per_cat = max(3, top_n // 5)
    for cat in ["Immune","Metabolism","Signaling","Apoptosis","Ubiquitin","Other"]:
        candidates = pivot[pivot["_cat"] == cat].drop(columns=["_cat","_lib"])
        if candidates.empty:
            continue
        selected += candidates.head(per_cat).index.tolist()
    # 补足到 top_n (用其余高方差 term)
    remaining = [t for t in pivot.index if t not in selected]
    selected = (selected + remaining)[:top_n]
    if not selected:
        print("  [Fig1] 筛选后无 term 可画"); return
    plot_df = pivot.loc[selected].drop(columns=["_cat","_lib"], errors="ignore")

    # 行标签: 加 [lib] 和 pathway 类别颜色
    row_labels   = [f"[{term_lib.get(t,'?')}] {t}" for t in plot_df.index]
    row_cats     = [pathway_category(t) for t in plot_df.index]
    row_colors   = [PATHWAY_COLORS[c] for c in row_cats]

    n_r, n_c = plot_df.shape
    fig_w = max(6, n_c * 1.6 + 4.0)
    fig_h = max(5, n_r * 0.35 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        plot_df, ax=ax,
        cmap="YlOrRd", linewidths=0.3, linecolor="#e8e8e8",
        cbar_kws={"label": "-log₁₀(FDR)", "shrink": 0.5},
        annot=plot_df.applymap(lambda v: f"{v:.1f}" if v > 1 else ""),
        fmt="", annot_kws={"size": 5.5},
        yticklabels=row_labels,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Pathway enrichment — subset-specific signal\n"
                 f"(black-listed housekeeping terms removed, sorted by cross-subset variance, FDR<{fdr_cut})",
                 pad=10)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    # 行彩色标记条 (左侧)
    for i, (lbl, color) in enumerate(zip(row_labels, row_colors)):
        ax.get_yticklabels()[i].set_color(color)

    # 图例: pathway category
    handles = [mpatches.Patch(color=v, label=k)
               for k, v in PATHWAY_COLORS.items()
               if k in set(row_cats)]
    ax.legend(handles=handles, title="Pathway category",
              loc="upper left", bbox_to_anchor=(1.18, 1.0),
              fontsize=6, title_fontsize=6.5, frameon=True)

    fig.savefig(out / "fig1_pathway_heatmap_v2.pdf")
    fig.savefig(out / "fig1_pathway_heatmap_v2.png", dpi=150)
    plt.close(fig)
    print(f"  [Fig1] → fig1_pathway_heatmap_v2.pdf  ({n_r} terms × {n_c} subsets)")


# ─────────────────────────────────────────────────────────────────────────────
# 图 2: 亚群特异 E3 × 免疫/代谢通路气泡图
# ─────────────────────────────────────────────────────────────────────────────

def get_subset_specific_e3(vuln: pd.DataFrame, top_n: int = 12) -> dict[str, list[str]]:
    """
    对每个亚群，选在该亚群里 frac_sig_05 z-score 显著高于其他亚群的 E3。
    方法: 对每个 E3 计算其 frac_sig_05 在亚群间的 z-score, 取各亚群 z-score 最高的 top_n。
    """
    if vuln.empty:
        return {}
    # E3 × subset 矩阵
    wide = vuln.pivot_table(index="ko", columns="subset",
                            values="frac_sig_05", aggfunc="mean").fillna(0)
    # z-score across subsets (行)
    mean = wide.mean(axis=1)
    std  = wide.std(axis=1).replace(0, np.nan)
    z = wide.sub(mean, axis=0).div(std, axis=0).fillna(0)

    result = {}
    for subset in wide.columns:
        # 该亚群 z-score 最高的 E3
        top_e3 = z[subset].sort_values(ascending=False).head(top_n).index.tolist()
        # 同时要求该 E3 在该亚群里有至少 5 个显著基因
        min_sig = vuln[(vuln["subset"] == subset) & (vuln["ko"].isin(top_e3))]
        top_e3 = min_sig[min_sig["n_sig_05"] >= 5]["ko"].tolist()
        # 按 z-score 重排
        top_e3 = sorted(top_e3, key=lambda k: z.loc[k, subset], reverse=True)[:top_n]
        result[subset] = top_e3
    return result


def plot_bubble_v2(gsea: pd.DataFrame, vuln: pd.DataFrame,
                   fam_map: dict[str, str], out: Path,
                   fdr_cut: float, top_terms: int, top_e3: int):
    """
    对每个亚群:
      - 横轴: 该亚群特异性 E3 (z-score 最高)
      - 纵轴: 过滤黑名单后, 按 max(|NES|) 排序的免疫/代谢/信号通路
      - 气泡: 大小=|NES|, 颜色=NES方向
    每个亚群一张独立 PDF (避免拼在一起太密)
    """
    sig = gsea[(gsea["FDR"] < fdr_cut)].copy()
    sig = sig[~sig["term"].apply(is_blacklisted)]
    # 只保留有生物学意义的类别
    sig["cat"] = sig["term"].apply(pathway_category)
    sig = sig[sig["cat"].isin(["Immune","Metabolism","Signaling","Apoptosis","Ubiquitin"])]
    if sig.empty:
        print("  [Fig2] 过滤后无可用 term")
        return

    specific_e3 = get_subset_specific_e3(vuln, top_n=top_e3)
    subsets = sorted(sig["subset"].unique())

    for subset in subsets:
        ko_order = specific_e3.get(subset, [])
        if not ko_order:
            print(f"  [Fig2] {subset}: 无特异性 E3，跳过")
            continue
        sub = sig[sig["subset"] == subset]
        sub = sub[sub["ko"].isin(ko_order)]
        if sub.empty:
            continue

        # 选 term: 该 subset 内 |NES| 最大的 top_terms
        term_max_nes = (sub.groupby("term")["NES"]
                          .apply(lambda x: x.abs().max())
                          .sort_values(ascending=False)
                          .head(top_terms))
        term_order = term_max_nes.index.tolist()
        sub = sub[sub["term"].isin(term_order)]

        # pivot
        pivot_nes = sub.pivot_table(index="term", columns="ko",
                                     values="NES",  aggfunc="first")
        pivot_fdr = sub.pivot_table(index="term", columns="ko",
                                     values="FDR",  aggfunc="min")
        pivot_nes = pivot_nes.reindex(index=term_order, columns=ko_order)
        pivot_fdr = pivot_fdr.reindex(index=term_order, columns=ko_order)

        # term 类别颜色
        term_cats   = [pathway_category(t) for t in term_order]
        term_colors = [PATHWAY_COLORS[c] for c in term_cats]
        # E3 family 颜色
        e3_fam_colors = ko_family_colors(ko_order, fam_map)

        n_t = len(term_order)
        n_k = len(ko_order)
        fig_w = max(5, n_k * 0.75 + 3.5)
        fig_h = max(5, n_t * 0.50 + 2.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        cmap_b = plt.cm.RdBu_r
        norm_b = Normalize(vmin=-3, vmax=3)

        for ti, term in enumerate(term_order):
            for ki, ko in enumerate(ko_order):
                nes = pivot_nes.loc[term, ko] if (
                    term in pivot_nes.index and ko in pivot_nes.columns) else np.nan
                fdr = pivot_fdr.loc[term, ko] if (
                    term in pivot_fdr.index and ko in pivot_fdr.columns) else np.nan
                if pd.isna(nes) or pd.isna(fdr):
                    continue
                size  = max(15, min(350, abs(nes) ** 2.2 * 55))
                alpha = max(0.4, 1 - fdr / fdr_cut * 0.6)
                color = cmap_b(norm_b(nes))
                ax.scatter(ki, ti, s=size, c=[color], alpha=alpha,
                           edgecolors="#333" if abs(nes) > 1.8 else "none",
                           linewidth=0.4, zorder=3)

        # y 轴: term 文字, 类别颜色
        ax.set_yticks(range(n_t))
        ax.set_yticklabels(term_order, fontsize=6.5)
        for i, (lbl, color) in enumerate(zip(ax.get_yticklabels(), term_colors)):
            lbl.set_color(color)

        # x 轴: E3 文字
        ax.set_xticks(range(n_k))
        ax.set_xticklabels(ko_order, rotation=45, ha="right", fontsize=7)

        # E3 family 颜色条（在 x 轴下方加色块）
        for ki, (ko, fc) in enumerate(zip(ko_order, e3_fam_colors)):
            ax.add_patch(plt.Rectangle(
                (ki - 0.4, -1.1), 0.8, 0.55,
                transform=ax.transData, color=fc, clip_on=False, zorder=4))

        ax.set_xlim(-0.6, n_k - 0.4)
        ax.set_ylim(-1.8, n_t - 0.4)
        ax.set_title(f"{subset}  —  Subset-specific E3 KO × Pathway\n"
                     f"(E3 selected by cross-subset z-score; FDR<{fdr_cut})",
                     pad=8, fontsize=9)
        ax.grid(True, color="#f0f0f0", linewidth=0.5, zorder=0)
        ax.set_facecolor("#fafafa")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # 图例 — NES colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_b, norm=norm_b)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.02)
        cbar.set_label("NES", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        # 图例 — pathway category
        cat_handles = [mpatches.Patch(color=PATHWAY_COLORS[c], label=c)
                       for c in sorted(set(term_cats))]
        # 图例 — E3 family
        used_fams = sorted(set(fam_map.get(k.upper(), "Other") for k in ko_order))
        fam_handles = [mpatches.Patch(color=FAMILY_COLORS[f], label=f)
                       for f in used_fams]

        leg1 = ax.legend(handles=cat_handles, title="Pathway",
                         loc="upper left", bbox_to_anchor=(1.12, 1.0),
                         fontsize=6, title_fontsize=6.5, frameon=True)
        ax.add_artist(leg1)
        ax.legend(handles=fam_handles, title="E3 family",
                  loc="upper left", bbox_to_anchor=(1.12, 0.55),
                  fontsize=6, title_fontsize=6.5, frameon=True)

        # 气泡大小图例 (角落)
        for nes_ex, label in [(1.0,"|NES|=1"), (1.8,"1.8"), (2.5,"2.5")]:
            ax.scatter([], [], s=nes_ex**2.2*55, c="gray", alpha=0.6, label=label)
        ax.legend(title="Bubble size", loc="lower left",
                  fontsize=5.5, title_fontsize=6, frameon=True,
                  bbox_to_anchor=(1.12, 0.0))
        ax.add_artist(leg1)

        fname = f"fig2_bubble_{subset}_v2"
        fig.savefig(out / f"{fname}.pdf")
        fig.savefig(out / f"{fname}.png", dpi=150)
        plt.close(fig)
        print(f"  [Fig2] → {fname}.pdf  ({n_k} E3 × {n_t} terms)")


# ─────────────────────────────────────────────────────────────────────────────
# 图 3: TF upstream 热图 (只保留 human)
# ─────────────────────────────────────────────────────────────────────────────

def plot_tf_heatmap_v2(tf_sum: pd.DataFrame, vuln: pd.DataFrame,
                       fam_map: dict[str, str], out: Path,
                       p_cut: float, top_e3: int, min_e3: int = 2):
    if tf_sum.empty:
        print("  [Fig3] TF summary 为空"); return

    # 只保留 human TF
    tf_sum = tf_sum[~tf_sum["term"].str.lower().str.contains("mouse")].copy()
    tf_sum = tf_sum[tf_sum["p_adj"] < p_cut]
    if tf_sum.empty:
        print("  [Fig3] 过滤 mouse + p_adj 后为空"); return

    subsets = sorted(tf_sum["subset"].unique())
    specific_e3 = get_subset_specific_e3(vuln, top_n=top_e3)

    for subset in subsets:
        sub = tf_sum[tf_sum["subset"] == subset]
        ko_order = specific_e3.get(subset, [])
        # fallback: 用 vulnerability 排
        if not ko_order:
            vv = vuln[vuln["subset"] == subset].sort_values("n_sig_05", ascending=False)
            ko_order = [k for k in vv["ko"].head(top_e3).tolist()
                        if k in sub["ko"].values]
        ko_order = [k for k in ko_order if k in sub["ko"].values]
        if not ko_order:
            continue

        sub = sub[sub["ko"].isin(ko_order)]
        tf_count = sub.groupby("term")["ko"].nunique()
        tf_keep  = tf_count[tf_count >= min_e3].sort_values(ascending=False).index.tolist()
        if not tf_keep:
            tf_keep = tf_count.sort_values(ascending=False).head(15).index.tolist()
        sub = sub[sub["term"].isin(tf_keep)]
        if sub.empty:
            continue

        pivot = sub.pivot_table(index="term", columns="ko",
                                values="p_adj", aggfunc="min")
        pivot = pivot.reindex(columns=ko_order)
        nlp   = -np.log10(pivot.clip(lower=1e-10)).fillna(0)
        nlp   = nlp.loc[nlp.sum(axis=1).sort_values(ascending=False).index]

        # E3 family 颜色
        e3_fam_colors = ko_family_colors(ko_order, fam_map)

        n_tf = nlp.shape[0]; n_e3 = nlp.shape[1]
        fig_w = max(5, n_e3 * 0.60 + 3.0)
        fig_h = max(4, n_tf * 0.42 + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        sns.heatmap(nlp, ax=ax, cmap="Blues",
                    linewidths=0.3, linecolor="#e8e8e8",
                    cbar_kws={"label": "-log₁₀(p_adj)", "shrink": 0.5},
                    annot=True, fmt=".1f", annot_kws={"size": 5.5})

        ax.set_title(f"{subset} — Upstream TF enrichment (human TFs only)\n"
                     f"(TF in ≥{min_e3} E3 KOs, p_adj<{p_cut})", pad=8)
        ax.set_xlabel("E3 KO")
        ax.set_ylabel("Upstream TF")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

        # E3 family 颜色条
        for ki, fc in enumerate(e3_fam_colors):
            ax.add_patch(plt.Rectangle(
                (ki, n_tf + 0.05), 1.0, 0.55,
                transform=ax.transData, color=fc, clip_on=False, zorder=4))

        # family 图例
        used_fams = sorted(set(fam_map.get(k.upper(), "Other") for k in ko_order))
        fam_handles = [mpatches.Patch(color=FAMILY_COLORS[f], label=f)
                       for f in used_fams]
        ax.legend(handles=fam_handles, title="E3 family",
                  loc="upper right", bbox_to_anchor=(1.0, -0.22),
                  ncol=3, fontsize=6, title_fontsize=6.5, frameon=True)

        fname = f"fig3_TF_{subset}_v2"
        fig.savefig(out / f"{fname}.pdf")
        fig.savefig(out / f"{fname}.png", dpi=150)
        plt.close(fig)
        print(f"  [Fig3] → {fname}.pdf  ({n_tf} TFs × {n_e3} E3)")


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",      default=".", help="E3VK 项目根目录")
    ap.add_argument("--out",       default="results/AIHA/downstream/figs/enrichment_v2")
    ap.add_argument("--fdr-gsea",  type=float, default=0.25)
    ap.add_argument("--fdr-tf",    type=float, default=0.3)
    ap.add_argument("--top-terms", type=int,   default=25)
    ap.add_argument("--top-e3",    type=int,   default=12)
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    out  = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    enrich_dir = root / "results/AIHA/downstream/enrichment"
    tf_path    = root / "results/AIHA/downstream/tf_enrichment/summary.csv"
    vuln_path  = root / "results/AIHA/downstream/vulnerability.csv"
    e3_table   = root / "data/E3-ome.xlsx"

    print("Loading GSEA results...")
    gsea = load_all_gsea(enrich_dir)
    if gsea.empty:
        print("ERROR: 没找到 *_gsea.csv"); sys.exit(1)
    n_total = len(gsea)
    n_sig   = (gsea["FDR"] < args.fdr_gsea).sum()
    n_after = (gsea[gsea["FDR"] < args.fdr_gsea]["term"]
               .apply(lambda t: not is_blacklisted(t)).sum())
    print(f"  总 term: {n_total:,} | FDR<{args.fdr_gsea}: {n_sig:,} | "
          f"过滤黑名单后: {n_after:,}")
    print(f"  亚群: {sorted(gsea['subset'].unique())}")
    print(f"  E3: {gsea['ko'].nunique()}")

    print("\nLoading vulnerability & E3 family...")
    vuln    = pd.read_csv(vuln_path) if vuln_path.exists() else pd.DataFrame()
    fam_map = load_e3_family(e3_table)
    print(f"  E3 family 覆盖: {len(fam_map)} 个基因")

    print("\nLoading TF enrichment summary...")
    tf_sum = pd.read_csv(tf_path) if tf_path.exists() else pd.DataFrame()
    if not tf_sum.empty:
        n_human = (~tf_sum["term"].str.lower().str.contains("mouse")).sum()
        print(f"  {len(tf_sum):,} 行，其中 human TF: {n_human:,}")

    print(f"\n输出目录: {out}")

    print("\n── Fig 1: Pathway × subset heatmap (subset-specific, no housekeeping) ──")
    plot_pathway_heatmap(gsea, out, fdr_cut=args.fdr_gsea, top_n=args.top_terms)

    print("\n── Fig 2: Subset-specific E3 × Pathway bubble (per subset) ──")
    plot_bubble_v2(gsea, vuln, fam_map, out,
                   fdr_cut=args.fdr_gsea,
                   top_terms=args.top_terms,
                   top_e3=args.top_e3)

    print("\n── Fig 3: TF heatmap (human only, per subset) ──")
    plot_tf_heatmap_v2(tf_sum, vuln, fam_map, out,
                       p_cut=args.fdr_tf,
                       top_e3=args.top_e3)

    print(f"\n完成。图表列表:")
    for f in sorted(out.glob("fig*.p*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()