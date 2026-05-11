"""
01b_aiha_functional.py
AIHA 三个 h5ad → 功能亚型注释 → scTenifoldKnk-ready 亚群

Strategy:
  1. 保留 AIHA 已有 hard-call 标签:Treg, Tfh, MAIT, gdT, Plasma, Naive B, Memory B
  2. 对剩下的 CD4 helper memory/effector → signature score → Th1/Th2/Th17 分配 (引入可塑性与双重校验)
  3. 对剩下的 CD8 effector → exhaustion score → Tex / cytotoxic 分配
  4. 每个亚型做信号质量 QC,信号弱的标 [LOW_CONF]
"""
import scanpy as sc, anndata as ad, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt, seaborn as sns
import h5py

DATA = Path("/root/autodl-tmp/E3VK/data/AIHA")
RES = Path("results/AIHA")
FIG = Path("figs/AIHA")
(RES/"subsets").mkdir(parents=True, exist_ok=True)
FIG.mkdir(exist_ok=True)

FILES = {
    "cd4": "human_immune_health_atlas_cd4t-treg-dnt.h5ad",
    "cd8": "human_immune_health_atlas_cd8t-gdt-mait.h5ad",
    "b":   "human_immune_health_atlas_b-plasma.h5ad",
}

# ============ Signature 基因集 ============
SIGNATURES = {
    # CD4 helper polarization
    "Th1":   ["TBX21","IFNG","CXCR3","CCL5","STAT4","IL12RB2","CCR5"],
    "Th2":   ["GATA3","IL4","IL5","IL13","CCR4","PTGDR2","IL1RL1","HAVCR1"],
    "Th17":  ["RORC","IL17A","IL17F","CCR6","KLRB1","IL23R","IL22","CCL20"],
    "Tfh":   ["CXCR5","BCL6","PDCD1","ICOS","IL21","CXCL13","BTLA"],
    "Treg_extra": ["FOXP3","IL2RA","IKZF2","CTLA4","TNFRSF18","TNFRSF4"],

    # CD8 functional states
    "CD8_cytotoxic":  ["GZMB","GZMA","GZMH","PRF1","GNLY","NKG7","FGFBP2","FCGR3A"],
    "CD8_exhausted":  ["PDCD1","HAVCR2","LAG3","TIGIT","TOX","TOX2","ENTPD1","CXCL13","LAYN"],
    "CD8_memory":     ["IL7R","CCR7","SELL","TCF7","LEF1"],
    "CD8_effector_mem": ["GZMK","CXCR3","EOMES","CCL5"],
}

CORE_GENES = {
    "Th1":  ["TBX21", "CXCR3", "IFNG"],
    "Th2":  ["GATA3", "PTGDR2", "CCR4"],
    "Th17": ["RORC", "CCR6", "IL17A"]
}

# ============ 1. 加载 ============
adatas = {}
for k, fn in FILES.items():
    file_path = DATA / fn
    
    # 【兼容性魔法】：强行抹除高版本 Scanpy 留下的不兼容 metadata
    try:
        with h5py.File(file_path, 'r+') as f:
            if 'uns' in f and 'log1p' in f['uns']:
                del f['uns']['log1p']
                print(f"  [MAGIC] Successfully removed incompatible 'uns/log1p' from {fn}")
    except Exception as e:
        # 如果文件没有这个字段，或者已经被清理过，会自动跳过
        pass

    # 现在可以安全地用老版本 anndata 读取了
    a = sc.read_h5ad(file_path)
    print(f"\n=== {k}: {a.shape} ===")
    print(f"  obs cols: {[c for c in a.obs.columns if 'AIFI' in c or 'cell_type' in c.lower() or 'label' in c.lower()][:10]}")

    # raw counts 恢复
    if a.raw is not None:
        common = a.var_names.intersection(a.raw.var_names)
        if len(common) < a.n_vars:
            a = a[:, common].copy()
        a.layers["counts"] = a.raw[:, a.var_names].X.copy()
    elif "counts" in a.layers:
        pass
    elif "raw_counts" in a.layers:
        a.layers["counts"] = a.layers["raw_counts"].copy()
    else:
        a.layers["counts"] = a.X.copy()

    # var_names → symbol
    if "feature_name" in a.var.columns:
        a.var["ensembl_id"] = a.var_names
        a.var_names = a.var["feature_name"].astype(str)
        a.var_names_make_unique()

    # 确保有 lognorm 给 score_genes 用 (因为我们刚才可能删了旧的 log1p，这里会触发重新计算，非常安全)
    if "log1p" not in a.uns:
        a.X = a.layers["counts"].copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)

    a.obs["dataset"] = k
    adatas[k] = a

# ============ 2. 探查 AIHA 注释列 ============
for k, a in adatas.items():
    cand = [c for c in a.obs.columns if "AIFI" in c.upper() or c == "cell_type"]
    print(f"\n{k} candidate label columns:")
    for c in cand:
        n = a.obs[c].nunique()
        print(f"  {c}: {n} levels, top={a.obs[c].value_counts().head(3).to_dict()}")

LABEL_COL = "AIFI_L3"   # ← 如果实际不是这个名字改这里

# ============ 3. 计算所有 signature scores ============
for k, a in adatas.items():
    print(f"\nProcessing dataset: {k}")
    for sig_name, genes in SIGNATURES.items():
        present = [g for g in genes if g in a.var_names]
        
        if len(present) < 3:
            print(f"  [{k}] {sig_name}: only {len(present)} genes present, skip")
            continue
            
        # 1. 基础 Signature 打分
        sc.tl.score_genes(a, present, score_name=f"score_{sig_name}",
                          ctrl_size=max(50, len(present)*5), random_state=0)
        
        # 2. 单独提取核心基因的表达特征，用于后续校验
        if sig_name in CORE_GENES:
            core_present = [g for g in CORE_GENES[sig_name] if g in a.var_names]
            if len(core_present) == 0:
                print(f"  [警告] {sig_name} 的核心基因 (TBX21/GATA3/RORC等) 全部未检测到！")
                a.obs[f"core_expr_{sig_name}"] = 0.0
            else:
                core_X = a[:, core_present].X
                if hasattr(core_X, "toarray"):
                    core_X = core_X.toarray()
                a.obs[f"core_expr_{sig_name}"] = core_X.mean(axis=1)

# ============ 4. 功能亚型分配 ============
def assign_cd4_helper(a, label_col, base_percentile=80, strict_percentile=95):
    """
    对 CD4 helper细胞进行分配, 结合核心基因和数据驱动阈值，允许 Hybrid states
    新增参数:
    - base_percentile: 进入该极化亚型考核的基础分数线
    - strict_percentile: 在核心基因 Drop-out 时的替补极高分线
    """
    is_helper = (
        a.obs[label_col].astype(str).str.contains("CD4", case=False) &
        ~a.obs[label_col].astype(str).str.contains("Naive|Treg|Tfh", case=False)
    )
    sub_idx = a.obs.index[is_helper]
    if len(sub_idx) == 0: return pd.Series(dtype=str)

    scores = a.obs.loc[sub_idx, ["score_Th1", "score_Th2", "score_Th17"]]
    
    # 动态阈值：仅针对 Helper 细胞池计算，而不是全局 PBMC
    thresholds_base = {sig: np.percentile(scores[f"score_{sig}"], base_percentile) for sig in ["Th1", "Th2", "Th17"]}
    thresholds_strict = {sig: np.percentile(scores[f"score_{sig}"], strict_percentile) for sig in ["Th1", "Th2", "Th17"]}
    
    assigned_labels = []
    
    for idx in sub_idx:
        active = []
        for sig in ["Th1", "Th2", "Th17"]:
            score_val = a.obs.loc[idx, f"score_{sig}"]
            core_val = a.obs.loc[idx, f"core_expr_{sig}"] if f"core_expr_{sig}" in a.obs.columns else 0.0
            
            # 基础线及格
            if score_val > thresholds_base[sig]:
                # 核心基因有表达，或者总分极高对冲 Drop-out
                if (core_val > 0) or (score_val > thresholds_strict[sig]):
                    active.append(sig)
                    
        # 判断状态
        if len(active) == 0:
            assigned_labels.append("CD4_helper_unpolarized")
        elif len(active) == 1:
            assigned_labels.append(active[0])
        else:
            assigned_labels.append("_".join(active)) # 产生如 Th1_Th17 标签
            
    return pd.Series(assigned_labels, index=sub_idx)

def assign_cd8_state(a, label_col):
    """CD8 effector 中按 cytotoxic vs exhausted 比值分配"""
    is_cd8_eff = (
        a.obs[label_col].astype(str).str.contains("CD8", case=False) &
        ~a.obs[label_col].astype(str).str.contains("Naive|MAIT|gdT", case=False)
    )
    sub_idx = a.obs.index[is_cd8_eff]
    if len(sub_idx) == 0: return pd.Series(dtype=str)

    cyt = a.obs.loc[sub_idx, "score_CD8_cytotoxic"]
    exh = a.obs.loc[sub_idx, "score_CD8_exhausted"]
    # 用 z-score 后比较
    cyt_z = (cyt - cyt.mean()) / cyt.std()
    exh_z = (exh - exh.mean()) / exh.std()

    out = pd.Series("CD8_effector_general", index=sub_idx)
    out[(exh_z > 0.5) & (exh_z > cyt_z)] = "CD8_exhausted"
    out[(cyt_z > 0.5) & (cyt_z > exh_z)] = "CD8_cytotoxic"
    return out

# CD4 数据集
a4 = adatas["cd4"]
a4.obs["functional_label"] = a4.obs[LABEL_COL].astype(str)   # 默认沿用原始
helper_labels = assign_cd4_helper(a4, LABEL_COL)
a4.obs.loc[helper_labels.index, "functional_label"] = helper_labels.values

# CD8 数据集
a8 = adatas["cd8"]
a8.obs["functional_label"] = a8.obs[LABEL_COL].astype(str)
cd8_labels = assign_cd8_state(a8, LABEL_COL)
a8.obs.loc[cd8_labels.index, "functional_label"] = cd8_labels.values

# B 数据集:沿用 AIHA 原标签
ab = adatas["b"]
ab.obs["functional_label"] = ab.obs[LABEL_COL].astype(str)

# ============ 5. 信号质量 QC:每个新亚型 marker 表达分布 ============
qc_report = []
for k, a in adatas.items():
    for label, count in a.obs["functional_label"].value_counts().items():
        if count < 50: continue
        cells = a.obs["functional_label"] == label
        for sig_name in ["Th1","Th2","Th17","Tfh","CD8_exhausted","CD8_cytotoxic"]:
            score_col = f"score_{sig_name}"
            if score_col not in a.obs.columns: continue
            in_score = a.obs.loc[cells, score_col].mean()
            out_score = a.obs.loc[~cells, score_col].mean()
            qc_report.append({
                "dataset": k, "label": label, "n": count,
                "signature": sig_name,
                "mean_score_in": in_score, "mean_score_out": out_score,
                "delta": in_score - out_score,
            })
qc_df = pd.DataFrame(qc_report)
qc_df.to_csv(RES/"functional_label_qc.csv", index=False)

# 可视化:每个亚型自家 signature 应该最高
key_pairs = [("Th1","Th1"),("Th2","Th2"),("Th17","Th17"),("Tfh","Tfh"),
             ("CD8_exhausted","CD8_exhausted"),("CD8_cytotoxic","CD8_cytotoxic")]
self_check = []
for label, sig in key_pairs:
    rows = qc_df.query("label == @label & signature == @sig")
    if not rows.empty:
        self_check.append({"label": label, "self_signature_delta": rows["delta"].iloc[0]})
print("\n=== Self-signature delta (should be > 0.1, ideally > 0.3) ===")
print(pd.DataFrame(self_check).to_string(index=False))

# ============ 6. 拆分输出 ============
target_subsets = {
    # CD4 纯净单阳性与主要双阳性态
    "CD4_naive":   ("cd4", lambda s: s.str.contains("CD4 Naive|CD4_Naive", regex=True)),
    "CD4_Treg":    ("cd4", lambda s: s.str.contains("Treg", case=False)),
    "CD4_Tfh":     ("cd4", lambda s: s.str.contains("Tfh", case=False)),
    "CD4_Th1":     ("cd4", lambda s: s == "Th1"),
    "CD4_Th2":     ("cd4", lambda s: s == "Th2"),
    "CD4_Th17":    ("cd4", lambda s: s == "Th17"),
    # 捕获高潜力的混合态 (如果数量足够将被保存)
    "CD4_Th1_Th17":("cd4", lambda s: s == "Th1_Th17"),
    "CD4_Th2_Th17":("cd4", lambda s: s == "Th2_Th17"),
    "CD4_unpol":   ("cd4", lambda s: s == "CD4_helper_unpolarized"),
    
    # CD8
    "CD8_naive":      ("cd8", lambda s: s.str.contains("CD8 Naive|CD8_Naive", regex=True)),
    "CD8_cytotoxic":  ("cd8", lambda s: s == "CD8_cytotoxic"),
    "CD8_exhausted":  ("cd8", lambda s: s == "CD8_exhausted"),
    "CD8_eff_general":("cd8", lambda s: s == "CD8_effector_general"),
    "MAIT":           ("cd8", lambda s: s.str.contains("MAIT", case=False)),
    "gdT":            ("cd8", lambda s: s.str.contains("gdT|γδ|gamma", case=False)),
    
    # B
    "B_naive":        ("b",   lambda s: s.str.contains("Naive B", case=False)),
    "B_memory":       ("b",   lambda s: s.str.contains("Memory B", case=False)),
    "B_plasmablast":  ("b",   lambda s: s.str.contains("Plasmablast|Plasma", case=False)),
}

summary = []
for name, (src, sel) in target_subsets.items():
    a = adatas[src]
    mask = sel(a.obs["functional_label"].astype(str))
    sub = a[mask].copy()
    n = sub.n_obs
    summary.append({"subset": name, "source": src, "n_cells": n})
    if n < 300:
        print(f"[skip] {name}: {n} cells (need ≥300)")
        continue
    sub.X = sub.layers["counts"]
    for ok in list(sub.obsm.keys()):
        if ok not in ["X_umap","X_scVI"]: del sub.obsm[ok]
    sub.uns = {}
    sub.obsp = {}
    sub.write_h5ad(RES/"subsets"/f"{name}.h5ad", compression="gzip")
    print(f"✓ {name}: {n} cells")

pd.DataFrame(summary).to_csv(RES/"subset_summary.csv", index=False)

# ============ 7. E3 列表 + 数据基因取交集 ============
e3 = pd.read_excel("data/E3-ome.xlsx").rename(columns={"Gene":"gene","Protein class":"family"})
all_var = set()
for f in (RES/"subsets").glob("*.h5ad"):
    a = sc.read_h5ad(f, backed="r"); all_var |= set(a.var_names); a.file.close()

e3["present"] = e3["gene"].isin(all_var)
e3.to_csv(RES/"E3_full_with_presence.csv", index=False)
e3_present = e3.query("present").copy()
e3_present.to_csv(RES/"E3_genes_present.csv", index=False)
print(f"\nE3 present: {len(e3_present)}/{len(e3)}")

# ============ 8. E3 Ligase 表达差异分析与火山图可视化 ============
print("\n>>> STEP 8: Differential Expression & Volcano Plots for E3 Ligases")

# 尝试导入 adjustText 以防止火山图标签重叠
try:
    from adjustText import adjust_text
    _HAS_ADJUST_TEXT = True
except ImportError:
    print("  [WARN] 推荐安装 'pip install adjustText' 以防止火山图上的基因名重叠。")
    _HAS_ADJUST_TEXT = False

import matplotlib.pyplot as plt
import seaborn as sns

def plot_e3_volcano(adata, groupby, group_test, group_ref, e3_genes, title, filename):
    """
    计算组间差异表达并绘制火山图，高亮显示 E3 基因及 Top10 显著基因。
    """
    print(f"  Calculating DE: {group_test} vs {group_ref} ...")
    
    # 提取需要的亚群
    sub_adata = adata[adata.obs[groupby].isin([group_test, group_ref])].copy()
    if sub_adata.n_obs < 50:
        print(f"  [Skip] 细胞数过少 ({sub_adata.n_obs})")
        return

    # Wilcoxon 秩和检验
    sc.tl.rank_genes_groups(sub_adata, groupby=groupby, groups=[group_test], 
                            reference=group_ref, method='wilcoxon')
    
    # 提取结果到 DataFrame
    res = sc.get.rank_genes_groups_df(sub_adata, group=group_test)
    res['names'] = res['names'].astype(str)
    
    # 计算 -log10(padj)，处理 p=0 的情况
    res['nlog10_padj'] = -np.log10(res['pvals_adj'].clip(lower=1e-300))
    
    # 设定显著性阈值
    FC_CUTOFF = 0.5   # Log2FC > 0.5 (即 fold change > 1.414)
    PADJ_CUTOFF = 0.05
    
    # 标记不同类别的基因
    res['sig'] = 'NS'
    res.loc[(res['logfoldchanges'] > FC_CUTOFF) & (res['pvals_adj'] < PADJ_CUTOFF), 'sig'] = 'Up'
    res.loc[(res['logfoldchanges'] < -FC_CUTOFF) & (res['pvals_adj'] < PADJ_CUTOFF), 'sig'] = 'Down'
    
    # 分离出 E3 基因
    res['is_E3'] = res['names'].isin(e3_genes)
    e3_sig = res[res['is_E3'] & (res['sig'] != 'NS')]
    
    # 获取全局 Top 10 显著差异基因 (基于绝对值 logFC 和 padj)
    top_genes = res[res['sig'] != 'NS'].sort_values('pvals_adj').head(10)
    
    # ---- 开始绘图 ----
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. 画非显著背景基因 (灰色)
    ns_mask = res['sig'] == 'NS'
    ax.scatter(res.loc[ns_mask, 'logfoldchanges'], res.loc[ns_mask, 'nlog10_padj'], 
               color='#d3d3d3', alpha=0.5, s=10, label='Not Sig')
    
    # 2. 画显著的非 E3 基因 (浅蓝/浅红)
    up_mask = (res['sig'] == 'Up') & (~res['is_E3'])
    down_mask = (res['sig'] == 'Down') & (~res['is_E3'])
    ax.scatter(res.loc[up_mask, 'logfoldchanges'], res.loc[up_mask, 'nlog10_padj'], 
               color='#ff9999', alpha=0.6, s=15, label=f'Up in {group_test}')
    ax.scatter(res.loc[down_mask, 'logfoldchanges'], res.loc[down_mask, 'nlog10_padj'], 
               color='#99ccff', alpha=0.6, s=15, label=f'Up in {group_ref}')
    
    # 3. 画显著的 E3 基因 (深红/深蓝，大圆点)
    e3_up = e3_sig[e3_sig['sig'] == 'Up']
    e3_down = e3_sig[e3_sig['sig'] == 'Down']
    ax.scatter(e3_up['logfoldchanges'], e3_up['nlog10_padj'], 
               color='#cc0000', edgecolors='black', linewidth=0.5, s=60, label='Sig E3 (Up)', zorder=5)
    ax.scatter(e3_down['logfoldchanges'], e3_down['nlog10_padj'], 
               color='#0000cc', edgecolors='black', linewidth=0.5, s=60, label='Sig E3 (Down)', zorder=5)
    
    # 添加阈值虚线
    ax.axvline(FC_CUTOFF, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(-FC_CUTOFF, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(-np.log10(PADJ_CUTOFF), color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # 添加文本标签 (Top10 基因 + 所有显著的 E3 基因)
    texts = []
    # 标记 E3
    for _, row in e3_sig.iterrows():
        texts.append(ax.text(row['logfoldchanges'], row['nlog10_padj'], row['names'], 
                             color='black', fontsize=9, fontweight='bold'))
    # 标记 Top10 非 E3
    for _, row in top_genes.iterrows():
        if not row['is_E3']:
            texts.append(ax.text(row['logfoldchanges'], row['nlog10_padj'], row['names'], 
                                 color='gray', fontsize=8, style='italic'))
    
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
    
    # 顺便保存显著 E3 的表格供后续参考
    if not e3_sig.empty:
        e3_sig.to_csv(RES / f"Sig_E3_{filename.replace('.pdf', '.csv')}", index=False)

# ---- 提取 E3 基因列表 ----
e3_list = e3_present['gene'].tolist()

# ---- 1. 宏观比较：CD4 vs CD8 (需要临时合并数据) ----
# ---- 1. 宏观比较：CD4 vs CD8 (极度省内存版) ----
print("\n  Preparing data for CD4 vs CD8 comparison (Memory-Safe Mode)...")
import anndata as ad
import gc  # 引入 Python 垃圾回收模块

# 绝不使用 .copy() 复制全表，而是只提取核心骨架 (X 矩阵和作为标签的 obs)
def get_lean_adata(adata, lineage_name):
    lean = ad.AnnData(
        X=adata.X, 
        obs=pd.DataFrame({'broad_lineage': [lineage_name] * adata.n_obs}, index=adata.obs_names),
        var=adata.var[[]] # 清空不必要的基因注释列
    )
    return lean

lean_4 = get_lean_adata(adatas["cd4"], 'CD4_T')
lean_8 = get_lean_adata(adatas["cd8"], 'CD8_T')

# 合并精简版数据
adata_all_t = ad.concat([lean_4, lean_8])

# 【关键】合并后立刻删掉临时变量，并强制清理系统内存
del lean_4
del lean_8
gc.collect()

plot_e3_volcano(adata_all_t, groupby='broad_lineage', 
                group_test='CD4_T', group_ref='CD8_T', 
                e3_genes=e3_list, 
                title='CD4 vs CD8 T cells Baseline', 
                filename='Volcano_CD4_vs_CD8.pdf')
del adata_all_t
gc.collect()
# ---- 2. CD4 内部关键比较 ----
# 比如 Th1 (细胞免疫) vs Th2 (体液免疫辅助)
plot_e3_volcano(adatas["cd4"], groupby='functional_label', 
                group_test='Th1', group_ref='Th2', 
                e3_genes=e3_list, 
                title='CD4 Helper: Th1 vs Th2', 
                filename='Volcano_CD4_Th1_vs_Th2.pdf')

# 比如 Th2 vs Th17
plot_e3_volcano(adatas["cd4"], groupby='functional_label', 
                group_test='Th2', group_ref='Th17', 
                e3_genes=e3_list, 
                title='CD4 Helper: Th2 vs Th17', 
                filename='Volcano_CD4_Th2_vs_Th17.pdf')

# 比如 Th1 vs Th17
plot_e3_volcano(adatas["cd4"], groupby='functional_label', 
                group_test='Th1', group_ref='Th17', 
                e3_genes=e3_list, 
                title='CD4 Helper: Th1 vs Th17', 
                filename='Volcano_CD4_Th1_vs_Th17.pdf')

# 比如 Treg (免疫抑制) vs Th17 (促炎)
plot_e3_volcano(adatas["cd4"], groupby='functional_label', 
                group_test='CD4_Treg', group_ref='Th17', # 注意对应你在 step 6 生成的标签
                e3_genes=e3_list, 
                title='CD4: Treg vs Th17', 
                filename='Volcano_CD4_Treg_vs_Th17.pdf')

# 比如 Th2 vs DP
plot_e3_volcano(adatas["cd4"], groupby='functional_label', 
                group_test='Th2', group_ref='Th2_Th17', 
                e3_genes=e3_list, 
                title='CD4 Helper: Th2 vs Th2_Th17', 
                filename='Volcano_CD4_Th2_vs_DP.pdf')

# 比如 Th17 vs DP
plot_e3_volcano(adatas["cd4"], groupby='functional_label', 
                group_test='Th17', group_ref='Th2_Th17', 
                e3_genes=e3_list, 
                title='CD4 Helper: Th17 vs Th2_Th17', 
                filename='Volcano_CD4_Th17_vs_DP.pdf')

# ---- 3. CD8 内部关键比较 ----
# 比如 Cytotoxic (杀伤活跃) vs Exhausted (耗竭)
plot_e3_volcano(adatas["cd8"], groupby='functional_label', 
                group_test='CD8_cytotoxic', group_ref='CD8_exhausted', 
                e3_genes=e3_list, 
                title='CD8: Cytotoxic vs Exhausted', 
                filename='Volcano_CD8_Cytotoxic_vs_Exhausted.pdf')

print(">>> STEP 8 Complete. Check the 'figs/AIHA' and 'results/AIHA' directories for plots and tables.")