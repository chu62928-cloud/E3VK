import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from adjustText import adjust_text
    _HAS_ADJUST_TEXT = True
except ImportError:
    _HAS_ADJUST_TEXT = False

# 路径设置
RES = Path("results/AIHA")
FIG = Path("figs/AIHA")
FIG.mkdir(parents=True, exist_ok=True)

# ============ 1. 加载 E3 白名单 ============
print(">>> Loading E3 genes list...")
e3_csv = RES / "E3_genes_present.csv"
if not e3_csv.exists():
    raise FileNotFoundError(f"找不到 {e3_csv}，请确认前面的步骤已成功生成此文件。")
e3_list = pd.read_csv(e3_csv)['gene'].tolist()

# ============ 2. 核心画图函数 ============
def plot_e3_volcano(adata, groupby, group_test, group_ref, e3_genes, title, filename):
    print(f"  Calculating DE: {group_test} vs {group_ref} ...")
    
    sub_adata = adata[adata.obs[groupby].isin([group_test, group_ref])].copy()
    if sub_adata.n_obs < 50:
        print(f"  [Skip] 细胞数过少 ({sub_adata.n_obs})")
        return

    # 【防错机制：强制转为 Category，解决 Scanpy 内部索引越界问题】
    sub_adata.obs[groupby] = sub_adata.obs[groupby].astype(str).astype('category')

    # 【防通胀机制 1：降采样】
    MAX_CELLS = 2000
    if sub_adata.n_obs > MAX_CELLS:
        sc.pp.subsample(sub_adata, n_obs=MAX_CELLS, random_state=42)
        print(f"    Subsampled to {MAX_CELLS} cells to prevent P-value inflation.")

    # 1. 运行差异分析
    sc.tl.rank_genes_groups(sub_adata, groupby=groupby, groups=[group_test], 
                            reference=group_ref, method='wilcoxon')
    
    res = sc.get.rank_genes_groups_df(sub_adata, group=group_test)
    
    # 2. 表达比例计算
    is_test = (sub_adata.obs[groupby] == group_test).values
    is_ref = (sub_adata.obs[groupby] == group_ref).values
    X = sub_adata.X
    
    if hasattr(X, "tocsc"):
        pct_test = np.array((X[is_test, :] > 0).sum(axis=0)).flatten() / is_test.sum()
        pct_ref = np.array((X[is_ref, :] > 0).sum(axis=0)).flatten() / is_ref.sum()
    else:
        pct_test = np.array((X[is_test] > 0).sum(axis=0)).flatten() / is_test.sum()
        pct_ref = np.array((X[is_ref] > 0).sum(axis=0)).flatten() / is_ref.sum()

    pct_dict_test = dict(zip(sub_adata.var_names, pct_test))
    pct_dict_ref = dict(zip(sub_adata.var_names, pct_ref))
    res['pct_test'] = res['names'].map(pct_dict_test)
    res['pct_ref'] = res['names'].map(pct_dict_ref)
    
    # 低表达基因过滤
    MIN_EXPR_FRAC = 0.10
    res = res[(res['pct_test'] >= MIN_EXPR_FRAC) | (res['pct_ref'] >= MIN_EXPR_FRAC)].copy()
    res['names'] = res['names'].astype(str)
    
    # 截断 P 值天花板
    res['nlog10_padj'] = -np.log10(res['pvals_adj'].clip(lower=1e-100))
    
    FC_CUTOFF = 0.5
    PADJ_CUTOFF = 0.05
    
    res['sig'] = 'NS'
    res.loc[(res['logfoldchanges'] > FC_CUTOFF) & (res['pvals_adj'] < PADJ_CUTOFF), 'sig'] = 'Up'
    res.loc[(res['logfoldchanges'] < -FC_CUTOFF) & (res['pvals_adj'] < PADJ_CUTOFF), 'sig'] = 'Down'
    
    res['is_E3'] = res['names'].isin(e3_genes)
    e3_sig = res[res['is_E3'] & (res['sig'] != 'NS')].copy()
    
    top_e3 = e3_sig.copy() 
    top_global = res[(res['sig'] != 'NS') & (~res['is_E3'])].sort_values('pvals_adj').head(5)
    
    # ---- 开始绘图 ----
    fig, ax = plt.subplots(figsize=(9, 7))
    
    ns_mask = res['sig'] == 'NS'
    ax.scatter(res.loc[ns_mask, 'logfoldchanges'], res.loc[ns_mask, 'nlog10_padj'], 
               color='#e0e0e0', alpha=0.5, s=10, label='Not Sig', zorder=1)
    
    up_mask = (res['sig'] == 'Up') & (~res['is_E3'])
    down_mask = (res['sig'] == 'Down') & (~res['is_E3'])
    ax.scatter(res.loc[up_mask, 'logfoldchanges'], res.loc[up_mask, 'nlog10_padj'], 
               color='#ff9999', alpha=0.6, s=20, label=f'Up in {group_test}', zorder=2)
    ax.scatter(res.loc[down_mask, 'logfoldchanges'], res.loc[down_mask, 'nlog10_padj'], 
               color='#99ccff', alpha=0.6, s=20, label=f'Up in {group_ref}', zorder=2)
    
    e3_up = e3_sig[e3_sig['sig'] == 'Up']
    e3_down = e3_sig[e3_sig['sig'] == 'Down']
    ax.scatter(e3_up['logfoldchanges'], e3_up['nlog10_padj'], 
               color='#cc0000', edgecolors='black', linewidth=0.8, s=70, label='Sig E3 (Up)', zorder=5)
    ax.scatter(e3_down['logfoldchanges'], e3_down['nlog10_padj'], 
               color='#0000cc', edgecolors='black', linewidth=0.8, s=70, label='Sig E3 (Down)', zorder=5)
    
    ax.axvline(FC_CUTOFF, color='k', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
    ax.axvline(-FC_CUTOFF, color='k', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
    ax.axhline(-np.log10(PADJ_CUTOFF), color='k', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
    
    texts = []
    for _, row in top_e3.iterrows():
        texts.append(ax.text(row['logfoldchanges'], row['nlog10_padj'], row['names'], 
                             color='black', fontsize=11, fontweight='bold', zorder=10))
    for _, row in top_global.iterrows():
        texts.append(ax.text(row['logfoldchanges'], row['nlog10_padj'], row['names'], 
                             color='#444444', fontsize=10, style='italic', zorder=9))
    
    if _HAS_ADJUST_TEXT and texts:
        adjust_text(texts, 
                    ax=ax,
                    force_points=(0.4, 0.5),
                    force_text=(0.4, 0.5),
                    arrowprops=dict(arrowstyle='-', color='#555555', lw=0.8, zorder=8))
    
    ax.set_title(title, pad=15, fontsize=14, fontweight='bold')
    ax.set_xlabel(f"Log2 Fold Change\n(Positive = Up in {group_test})", fontsize=12)
    ax.set_ylabel("-Log10(FDR)", fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    if not e3_sig.empty:
        e3_sig.to_csv(RES / f"Sig_E3_{filename.replace('.pdf', '.csv')}", index=False)

# ============ 3. 组装数据并画图 ============
print("\n>>> Building comparisons from saved subsets...")

def load_and_label(subset_name, label_col, label_val):
    p = RES / "subsets" / f"{subset_name}.h5ad"
    if not p.exists():
        print(f"  [WARN] 找不到文件 {p.name}，跳过此组。")
        return None
    a = sc.read_h5ad(p)
    a.obs[label_col] = label_val
    return a

subsets = {
    "Th1": load_and_label("CD4_Th1", "compare_label", "Th1"),
    "Th2": load_and_label("CD4_Th2", "compare_label", "Th2"),
    "Treg": load_and_label("CD4_Treg", "compare_label", "Treg"),
    "Th17": load_and_label("CD4_Th17", "compare_label", "Th17"),
    # 【修复处 1】统一字典的 Key 为真实的 label 名字
    "Th2_Th17_DP": load_and_label("CD4_Th2_Th17", "compare_label", "Th2_Th17_DP"),
    "Cytotoxic": load_and_label("CD8_cytotoxic", "compare_label", "Cytotoxic"),
    "Exhausted": load_and_label("CD8_exhausted", "compare_label", "Exhausted")
}

def safe_plot(group1, group2, title, filename, test_is_group1=True):
    a1, a2 = subsets.get(group1), subsets.get(group2)
    if a1 is not None and a2 is not None:
        adata_cmp = ad.concat([a1, a2])
        test_grp = group1 if test_is_group1 else group2
        ref_grp = group2 if test_is_group1 else group1
        plot_e3_volcano(adata_cmp, 'compare_label', test_grp, ref_grp, e3_list, title, filename)

# # ---- 执行所有火山图比较 ----
# safe_plot("Th1", "Th2", 'CD4 Helper: Th1 vs Th2', 'Volcano_CD4_Th1_vs_Th2.pdf')
# safe_plot("Treg", "Th17", 'CD4: Treg vs Th17', 'Volcano_CD4_Treg_vs_Th17.pdf')
# safe_plot("Th1", "Th17", 'CD4 Helper: Th1 vs Th17', 'Volcano_CD4_Th1_vs_Th17.pdf')
# safe_plot("Th2", "Th17", 'CD4 Helper: Th2 vs Th17', 'Volcano_CD4_Th2_vs_Th17.pdf')
# safe_plot("Th2", "Treg", 'CD4 Helper: Th2 vs Treg', 'Volcano_CD4_Th2_vs_Treg.pdf')

# # 【修复处 2】调用时使用一致的 Key ("Th2_Th17_DP" 而不是 "DP")
# safe_plot("Th2_Th17_DP", "Th2", 'CD4 Helper: Th2_Th17_DP vs Th2 Baseline', 'Volcano_CD4_DP_vs_Th2.pdf')
# safe_plot("Th2_Th17_DP", "Th17", 'CD4 Helper: Th2_Th17_DP vs Th17 Baseline', 'Volcano_CD4_DP_vs_Th17.pdf')
# safe_plot("Cytotoxic", "Exhausted", 'CD8: Cytotoxic vs Exhausted', 'Volcano_CD8_Cytotoxic_vs_Exhausted.pdf')

# # CD4 vs CD8
# cd4_keys = ["Th1", "Th2", "Treg", "Th17"]
# cd8_keys = ["Cytotoxic", "Exhausted"]
# cd4_parts = [subsets[k] for k in cd4_keys if subsets[k] is not None]
# cd8_parts = [subsets[k] for k in cd8_keys if subsets[k] is not None]

# if cd4_parts and cd8_parts:
#     macro_cd4 = ad.concat(cd4_parts)
#     macro_cd4.obs['macro_label'] = 'CD4_T'
#     macro_cd8 = ad.concat(cd8_parts)
#     macro_cd8.obs['macro_label'] = 'CD8_T'
#     macro_adata = ad.concat([macro_cd4, macro_cd8])
#     plot_e3_volcano(macro_adata, 'macro_label', 'CD4_T', 'CD8_T', e3_list, 'CD4 vs CD8 T cells Baseline', 'Volcano_CD4_vs_CD8.pdf')

# print(">>> Volcano plots generated successfully!")

# ============ 4. 生成多亚型 E3 基因 Heatmap ============
print("\n>>> Generating multi-subtype Heatmap for E3 genes...")
valid_subsets = [v for k, v in subsets.items() if v is not None]

if valid_subsets:
    adata_all = ad.concat(valid_subsets)
    
    # 强制将表达矩阵转换为 float32 浮点型
    if adata_all.X.dtype not in ['float32', 'float64']:
        adata_all.X = adata_all.X.astype('float32')

    valid_e3_genes = [g for g in e3_list if g in adata_all.var_names]
    
    if len(valid_e3_genes) > 0:
        
        # 【终极优化：基于“组间方差”筛选，只保留最具生物学信息量的特异性基因】
        print("  [Info] Calculating variance across subtypes to find informative genes...")
        group_means = []
        groups = adata_all.obs['compare_label'].unique()
        
        # 1. 遍历每个细胞亚群，计算该亚群内各个 E3 基因的平均表达量
        for g in groups:
            sub_matrix = adata_all[adata_all.obs['compare_label'] == g, valid_e3_genes].X
            group_means.append(np.array(sub_matrix.mean(axis=0)).flatten())
        
        # 转为矩阵 (形状: 亚群数 x 基因数)
        group_means = np.array(group_means)
        
        # 2. 计算每个基因在不同亚群均值之间的方差
        e3_variances = np.var(group_means, axis=0)
        var_df = pd.DataFrame({'gene': valid_e3_genes, 'variance': e3_variances})
        
        MAX_GENES_TO_SHOW = 40 
        
        if len(valid_e3_genes) > MAX_GENES_TO_SHOW:
            print(f"  [Info] 过滤掉恒定表达的管家基因，选取组间差异最显著的前 {MAX_GENES_TO_SHOW} 个...")
            # 根据方差降序排列，取 Top N
            filtered_e3_genes = var_df.sort_values('variance', ascending=False).head(MAX_GENES_TO_SHOW)['gene'].tolist()
        else:
            filtered_e3_genes = valid_e3_genes

        # 绘图设置
        sc.settings.figdir = FIG
        
        sc.pl.matrixplot(
            adata_all, 
            var_names=filtered_e3_genes, 
            groupby='compare_label', 
            dendrogram=True, 
            standard_scale='var', 
            cmap='RdBu_r', 
            title=f'Top {len(filtered_e3_genes)} Variable E3 Ligases Across Subtypes',
            save='_E3_Subtypes_Heatmap.pdf',
            show=False
        )
        print(f">>> Heatmap saved to {FIG / 'matrixplot__E3_Subtypes_Heatmap.pdf'}")
    else:
        print("  [WARN] 未能在合并的数据中找到对应的 E3 基因进行 Heatmap 绘制。")
else:
    print("  [WARN] 没有有效的亚群数据可用于绘制 Heatmap。")

print("\n>>> All tasks completed successfully!")