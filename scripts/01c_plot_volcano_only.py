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
FIG.mkdir(exist_ok=True)

# ============ 1. 加载 E3 白名单 ============
print(">>> Loading E3 genes list...")
e3_csv = RES / "E3_genes_present.csv"
if not e3_csv.exists():
    raise FileNotFoundError(f"找不到 {e3_csv}，请确认前面的步骤已成功生成此文件。")
e3_list = pd.read_csv(e3_csv)['gene'].tolist()

# ============ 2. 核心画图函数 (与之前一致) ============
def plot_e3_volcano(adata, groupby, group_test, group_ref, e3_genes, title, filename):
    print(f"  Calculating DE: {group_test} vs {group_ref} ...")
    
    sub_adata = adata[adata.obs[groupby].isin([group_test, group_ref])].copy()
    if sub_adata.n_obs < 50:
        print(f"  [Skip] 细胞数过少 ({sub_adata.n_obs})")
        return

    # 【防通胀机制 1：降采样】
    MAX_CELLS = 2000
    if sub_adata.n_obs > MAX_CELLS:
        sc.pp.subsample(sub_adata, n_obs=MAX_CELLS, random_state=42)
        print(f"    Subsampled to {MAX_CELLS} cells to prevent P-value inflation.")

    # 1. 运行差异分析 (不依赖 Scanpy 脆弱的 pts 参数)
    sc.tl.rank_genes_groups(sub_adata, groupby=groupby, groups=[group_test], 
                            reference=group_ref, method='wilcoxon')
    
    res = sc.get.rank_genes_groups_df(sub_adata, group=group_test)
    
    # 2. 【核心修复】：手工计算表达比例，100% 免疫版本冲突
    is_test = (sub_adata.obs[groupby] == group_test).values
    is_ref = (sub_adata.obs[groupby] == group_ref).values
    X = sub_adata.X
    
    # 获取两组各自的表达比例 (支持稀疏矩阵和密集矩阵)
    if hasattr(X, "tocsc"):
        pct_test = np.array((X[is_test, :] > 0).sum(axis=0)).flatten() / is_test.sum()
        pct_ref = np.array((X[is_ref, :] > 0).sum(axis=0)).flatten() / is_ref.sum()
    else:
        pct_test = np.array((X[is_test] > 0).sum(axis=0)).flatten() / is_test.sum()
        pct_ref = np.array((X[is_ref] > 0).sum(axis=0)).flatten() / is_ref.sum()

    # 映射回差异分析结果表
    pct_dict_test = dict(zip(sub_adata.var_names, pct_test))
    pct_dict_ref = dict(zip(sub_adata.var_names, pct_ref))
    res['pct_test'] = res['names'].map(pct_dict_test)
    res['pct_ref'] = res['names'].map(pct_dict_ref)
    
    # 【防通胀机制 2：低表达基因过滤】
    MIN_EXPR_FRAC = 0.10
    before_filter = len(res)
    res = res[(res['pct_test'] >= MIN_EXPR_FRAC) | (res['pct_ref'] >= MIN_EXPR_FRAC)].copy()
    print(f"    Filtered low-expression genes: {before_filter} -> {len(res)}")
    
    res['names'] = res['names'].astype(str)
    
    # 截断 P 值天花板，拉开散点间距
    res['nlog10_padj'] = -np.log10(res['pvals_adj'].clip(lower=1e-100))
    
    FC_CUTOFF = 0.5
    PADJ_CUTOFF = 0.05
    
    res['sig'] = 'NS'
    res.loc[(res['logfoldchanges'] > FC_CUTOFF) & (res['pvals_adj'] < PADJ_CUTOFF), 'sig'] = 'Up'
    res.loc[(res['logfoldchanges'] < -FC_CUTOFF) & (res['pvals_adj'] < PADJ_CUTOFF), 'sig'] = 'Down'
    
    res['is_E3'] = res['names'].isin(e3_genes)
    e3_sig = res[res['is_E3'] & (res['sig'] != 'NS')].copy()
    
    top_global = res[(res['sig'] != 'NS') & (~res['is_E3'])].sort_values('pvals_adj').head(5)
    if not e3_sig.empty:
        e3_sig['abs_logfc'] = e3_sig['logfoldchanges'].abs()
        top_e3 = e3_sig.sort_values('abs_logfc', ascending=False).head(5)
    else:
        top_e3 = pd.DataFrame()
    
    # ---- 开始绘图 ----
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ns_mask = res['sig'] == 'NS'
    ax.scatter(res.loc[ns_mask, 'logfoldchanges'], res.loc[ns_mask, 'nlog10_padj'], color='#d3d3d3', alpha=0.5, s=10, label='Not Sig')
    
    up_mask = (res['sig'] == 'Up') & (~res['is_E3'])
    down_mask = (res['sig'] == 'Down') & (~res['is_E3'])
    ax.scatter(res.loc[up_mask, 'logfoldchanges'], res.loc[up_mask, 'nlog10_padj'], color='#ff9999', alpha=0.6, s=15, label=f'Up in {group_test}')
    ax.scatter(res.loc[down_mask, 'logfoldchanges'], res.loc[down_mask, 'nlog10_padj'], color='#99ccff', alpha=0.6, s=15, label=f'Up in {group_ref}')
    
    e3_up = e3_sig[e3_sig['sig'] == 'Up']
    e3_down = e3_sig[e3_sig['sig'] == 'Down']
    ax.scatter(e3_up['logfoldchanges'], e3_up['nlog10_padj'], color='#cc0000', edgecolors='black', linewidth=0.5, s=60, label='Sig E3 (Up)', zorder=5)
    ax.scatter(e3_down['logfoldchanges'], e3_down['nlog10_padj'], color='#0000cc', edgecolors='black', linewidth=0.5, s=60, label='Sig E3 (Down)', zorder=5)
    
    ax.axvline(FC_CUTOFF, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(-FC_CUTOFF, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(-np.log10(PADJ_CUTOFF), color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    
    texts = []
    for _, row in top_e3.iterrows():
        texts.append(ax.text(row['logfoldchanges'], row['nlog10_padj'], row['names'], color='black', fontsize=10, fontweight='bold'))
    for _, row in top_global.iterrows():
        texts.append(ax.text(row['logfoldchanges'], row['nlog10_padj'], row['names'], color='gray', fontsize=9, style='italic'))
    
    if _HAS_ADJUST_TEXT and texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    ax.set_title(title, pad=15)
    ax.set_xlabel(f"Log2 Fold Change\n(Positive = Up in {group_test})")
    ax.set_ylabel("-Log10(FDR)")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    if not e3_sig.empty:
        e3_sig.to_csv(RES / f"Sig_E3_{filename.replace('.pdf', '.csv')}", index=False)

# ============ 3. 组装数据并画图 ============
print("\n>>> Building comparisons from saved subsets...")

def load_and_label(subset_name, label_col, label_val):
    """安全加载亚群并打上统一标签"""
    p = RES / "subsets" / f"{subset_name}.h5ad"
    if not p.exists():
        print(f"  [WARN] 找不到文件 {p.name}，跳过此组。")
        return None
    a = sc.read_h5ad(p)
    a.obs[label_col] = label_val
    return a

# ---- 比较 A: CD4 Th1 vs Th2 ----
a_th1 = load_and_label("CD4_Th1", "compare_label", "Th1")
a_th2 = load_and_label("CD4_Th2", "compare_label", "Th2")
if a_th1 is not None and a_th2 is not None:
    adata_cmp = ad.concat([a_th1, a_th2])
    plot_e3_volcano(adata_cmp, 'compare_label', 'Th1', 'Th2', e3_list, 'CD4 Helper: Th1 vs Th2', 'Volcano_CD4_Th1_vs_Th2.pdf')

# ---- 比较 B: CD4 Treg vs Th17 ----
a_treg = load_and_label("CD4_Treg", "compare_label", "Treg")
a_th17 = load_and_label("CD4_Th17", "compare_label", "Th17")
if a_treg is not None and a_th17 is not None:
    adata_cmp = ad.concat([a_treg, a_th17])
    plot_e3_volcano(adata_cmp, 'compare_label', 'Treg', 'Th17', e3_list, 'CD4: Treg vs Th17', 'Volcano_CD4_Treg_vs_Th17.pdf')

# ---- 比较 C: CD4 Th1 vs Th17 ----
if a_th1 is not None and a_th17 is not None:
    adata_cmp = ad.concat([a_th1, a_th17])
    plot_e3_volcano(adata_cmp, 'compare_label', 'Th1', 'Th17', e3_list, 'CD4 Helper: Th1 vs Th17', 'Volcano_CD4_Th1_vs_Th17.pdf')

# ---- 比较 D: CD4 Th2 vs Th17 ----
if a_th2 is not None and a_th17 is not None:
    adata_cmp = ad.concat([a_th2, a_th17])
    plot_e3_volcano(adata_cmp, 'compare_label', 'Th2', 'Th17', e3_list, 'CD4 Helper: Th2 vs Th17', 'Volcano_CD4_Th2_vs_Th17.pdf')

# ---- 比较 E: CD4 Th2 vs Treg ----
if a_th2 is not None and a_treg is not None:
    adata_cmp = ad.concat([a_th2, a_treg])
    plot_e3_volcano(adata_cmp, 'compare_label', 'Th2', 'Treg', e3_list, 'CD4 Helper: Th2 vs Treg', 'Volcano_CD4_Th2_vs_Treg.pdf')

# ---- 比较 F: CD4 Th2 vs DP ----
a_dp  = load_and_label("CD4_Th2_Th17", "compare_label", "Th2_Th17_DP")
if a_th2 is not None and a_dp is not None:
    adata_cmp = ad.concat([a_th2, a_dp])
    # 将 DP 作为 test(实验组)，Th2 作为 ref(对照组)
    # 这样图的右边 (LogFC > 0) 就是 DP 中特异高表达的基因 (包括 E3)
    plot_e3_volcano(adata_cmp, 'compare_label', 
                    group_test='Th2_Th17_DP', 
                    group_ref='Th2', 
                    e3_genes=e3_list, 
                    title='CD4 Helper: Th2_Th17_DP vs Th2 Baseline', 
                    filename='Volcano_CD4_DP_vs_Th2.pdf')

# ---- 比较 F2: CD4 Th17 vs DP ----
if a_th17 is not None and a_dp is not None:
    adata_cmp = ad.concat([a_th17, a_dp])
    # 将 DP 作为 test(实验组)，Th2 作为 ref(对照组)
    # 这样图的右边 (LogFC > 0) 就是 DP 中特异高表达的基因 (包括 E3)
    plot_e3_volcano(adata_cmp, 'compare_label', 
                    group_test='Th2_Th17_DP', 
                    group_ref='Th17', 
                    e3_genes=e3_list, 
                    title='CD4 Helper: Th2_Th17_DP vs Th17 Baseline', 
                    filename='Volcano_CD4_DP_vs_Th17.pdf')
    

# ---- 比较 G: CD8 Cytotoxic vs Exhausted ----
a_cyto = load_and_label("CD8_cytotoxic", "compare_label", "Cytotoxic")
a_exh = load_and_label("CD8_exhausted", "compare_label", "Exhausted")
if a_cyto is not None and a_exh is not None:
    adata_cmp = ad.concat([a_cyto, a_exh])
    plot_e3_volcano(adata_cmp, 'compare_label', 'Cytotoxic', 'Exhausted', e3_list, 'CD8: Cytotoxic vs Exhausted', 'Volcano_CD8_Cytotoxic_vs_Exhausted.pdf')

# ---- 比较 D: 宏观 CD4 vs CD8 (抽取代表性亚群进行比较，防内存溢出) ----
print("  Preparing macro CD4 vs CD8 comparison...")
cd4_parts = [a for a in [a_th1, a_th2, a_treg, a_th17] if a is not None]
cd8_parts = [a for a in [a_cyto, a_exh] if a is not None]

if cd4_parts and cd8_parts:
    macro_cd4 = ad.concat(cd4_parts)
    macro_cd4.obs['macro_label'] = 'CD4_T'
    
    macro_cd8 = ad.concat(cd8_parts)
    macro_cd8.obs['macro_label'] = 'CD8_T'
    
    macro_adata = ad.concat([macro_cd4, macro_cd8])
    plot_e3_volcano(macro_adata, 'macro_label', 'CD4_T', 'CD8_T', e3_list, 'CD4 vs CD8 T cells Baseline', 'Volcano_CD4_vs_CD8.pdf')

print("\n>>> All volcano plots generated successfully! Check the 'figs/AIHA' folder.")