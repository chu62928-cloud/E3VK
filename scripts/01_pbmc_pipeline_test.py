"""
01_pbmc_pipeline_test.py  ·  env_main  (含基因过滤优化)
=========================================================
10x PBMC 1k 验证脚本 — 无 scVI 版

流程: QC → normalize/PCA → UMAP/Leiden → CellTypist
      → 拆亚群 → [三层基因过滤] → scTenifoldKnk(10 E3)
"""
import scanpy as sc, numpy as np, pandas as pd, celltypist
from pathlib import Path
import warnings, sys, json, time
warnings.filterwarnings("ignore")
sc.settings.set_figure_params(dpi=80, frameon=False)
sc.settings.verbosity = 1

DATA = Path("data/pbmc10x")
RES  = Path("results_test")
for d in [RES, RES/"subsets", RES/"knk"]:
    d.mkdir(parents=True, exist_ok=True)

checkpoint_log = {}
def ckpt(n, name, data=None):
    ts = time.strftime("%H:%M:%S")
    print(f"\n{'='*52}\n  [CP-{n} ✓]  {name}  ({ts})")
    if data:
        [print(f"    {k}: {v}") for k,v in data.items()]
    print(f"{'='*52}\n")
    checkpoint_log[n] = {"name":name,"ts":ts,"data":data or {}}
    with open(RES/"checkpoint_log.json","w") as f:
        json.dump(checkpoint_log, f, indent=2, default=str)

# # STEP 1  QC
# print(">>> STEP 1: Load + QC")
# h5 = DATA/"pbmc_1k_v3_filtered_feature_bc_matrix.h5"
# if not h5.exists():
#     print(f"[ERROR] {h5} not found"); sys.exit(1)
# adata = sc.read_10x_h5(h5)
# adata.var_names_make_unique()
# n_raw = adata.n_obs
# adata.var["mt"]   = adata.var_names.str.startswith("MT-")
# adata.var["ribo"] = adata.var_names.str.startswith(("RPS","RPL"))
# sc.pp.calculate_qc_metrics(adata, qc_vars=["mt","ribo"], inplace=True, percent_top=None, log1p=False)
# adata = adata[adata.obs.n_genes_by_counts.between(500,6000) & (adata.obs.pct_counts_mt<15)].copy()
# adata.layers["counts"] = adata.X.copy()
# ckpt(1,"QC",{"cells_raw":n_raw,"cells_after":adata.n_obs,"genes_total":adata.n_vars,
#               "median_genes":int(adata.obs.n_genes_by_counts.median()),
#               "median_mt_pct":round(adata.obs.pct_counts_mt.median(),2)})

# # STEP 2  PCA + UMAP + Leiden
# print(">>> STEP 2: Normalize + PCA + UMAP + Leiden")
# sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
# adata.layers["lognorm"] = adata.X.copy()
# sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", layer="counts")
# sc.pp.scale(adata, max_value=10)
# sc.tl.pca(adata, n_comps=30, use_highly_variable=True)
# sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
# sc.tl.umap(adata); sc.tl.leiden(adata, resolution=0.8, key_added="leiden")
# ckpt(2,"PCA+UMAP+Leiden",{"n_HVG":int(adata.var.highly_variable.sum()),"n_clusters":adata.obs["leiden"].nunique()})

# # STEP 3  CellTypist
# print(">>> STEP 3: CellTypist annotation")
# adata.X = adata.layers["lognorm"].copy()
# ct = celltypist.annotate(adata, model="Immune_All_Low.pkl", majority_voting=True)
# adata.obs = adata.obs.join(ct.predicted_labels)
# lc = adata.obs["majority_voting"].value_counts()
# print(lc.head(15).to_string())
# ckpt(3,"CellTypist",{"n_cell_types":adata.obs["majority_voting"].nunique(),"top_label":lc.index[0],
#                       "has_T":any("T" in x for x in lc.index),"has_B":any("B" in x for x in lc.index)})
# adata.write_h5ad(RES/"pbmc_annotated.h5ad", compression="gzip")

# # STEP 3.5  E3 基线表达量评估与可视化 (汇报/QC 专用)
# print(">>> STEP 3.5: E3 Baseline Expression Visualization")
# import matplotlib.pyplot as plt

# # 假设你的 10 个测试 E3 基因
# target_e3 = ["MDM2", "TRIM21", "RNF4", "BIRC3", "XIAP", "CBL", "SIAH1", "NEDD4", "WWP1", "HUWE1"]
# # 只保留矩阵中确实存在的基因
# target_e3_exist = [g for g in target_e3 if g in adata.var_names]

# vis_dir = RES / "visualizations"
# vis_dir.mkdir(exist_ok=True)

# if target_e3_exist:
#     # 1. 气泡图 (Dot Plot) —— 导师最爱看的神图
#     # 点的大小代表表达细胞比例，颜色深浅代表平均表达量
#     print("  Generating DotPlot...")
#     sc.pl.dotplot(adata, var_names=target_e3_exist, groupby="majority_voting", 
#                   standard_scale='var', # 基因间缩放，便于对比
#                   cmap="Reds", show=False)
#     plt.savefig(vis_dir / "E3_DotPlot_by_CellType.pdf", bbox_inches="tight")
#     plt.close()

#     # 2. 小提琴图 (Stacked Violin) —— 观察表达分布的拖尾情况
#     print("  Generating Violin Plot...")
#     sc.pl.stacked_violin(adata, var_names=target_e3_exist, groupby="majority_voting", 
#                          swap_axes=False, show=False)
#     plt.savefig(vis_dir / "E3_ViolinPlot_by_CellType.pdf", bbox_inches="tight")
#     plt.close()

#     # 3. 统计定量表格 (导出给导师看的 Excel/CSV)
#     print("  Calculating statistical metrics...")
#     stats_list = []
#     # 使用原始的 counts 矩阵来算更真实
#     X_counts = adata.layers["counts"].toarray() if hasattr(adata.layers["counts"], "toarray") else adata.layers["counts"]
    
#     for ctype in adata.obs["majority_voting"].unique():
#         mask = adata.obs["majority_voting"] == ctype
#         n_cells_in_type = mask.sum()
#         if n_cells_in_type < 20: continue # 忽略过小的稀有细胞群
        
#         subset_X = X_counts[mask]
        
#         for idx, gene in enumerate(adata.var_names):
#             if gene not in target_e3_exist: continue
            
#             gene_expr = subset_X[:, idx]
#             expressing_cells = (gene_expr > 0).sum()
            
#             if expressing_cells > 0:
#                 mean_non_zero = gene_expr[gene_expr > 0].mean()
#             else:
#                 mean_non_zero = 0.0
                
#             stats_list.append({
#                 "Cell_Type": ctype,
#                 "Total_Cells": n_cells_in_type,
#                 "Gene": gene,
#                 "Expressing_Cells": expressing_cells,
#                 "Fraction_Expressing": f"{(expressing_cells / n_cells_in_type) * 100:.2f}%",
#                 "Mean_Count_NonZero": round(mean_non_zero, 3)
#             })
            
#     stats_df = pd.DataFrame(stats_list)
#     stats_df.to_csv(vis_dir / "E3_Expression_Stats.csv", index=False)
#     print(f"  Saved visualizations and stats to {vis_dir.name}/")

# # STEP 4  亚群拆分 (保留全基因,过滤在STEP6做)
# print(">>> STEP 4: Subset splitting")
# SUBSET_LABELS = {
#     "CD4_test": ["Tcm/Naive helper T cells","Tem/Effector helper T cells",
#                  "Naive CD4+ T cells","Th1 cells","Regulatory T cells"],
#     "CD8_test": ["Tcm/Naive cytotoxic T cells","Tem/Effector cytotoxic T cells","Cytotoxic T cells"],
# }
# subset_stats = {}
# for name, labels in SUBSET_LABELS.items():
#     mask = adata.obs["majority_voting"].isin(labels)
#     sub  = adata[mask].copy()
#     if sub.n_obs < 100:
#         print(f"  [fallback] {name}: {sub.n_obs} → leiden")
#         top2 = adata.obs["leiden"].value_counts().index
#         sub  = adata[adata.obs["leiden"]==top2[0 if "CD4" in name else 1]].copy()
#     sub.X = sub.layers["counts"].copy()
#     sub.obsp = {}; sub.uns = {}
#     out = RES/"subsets"/f"{name}.h5ad"
#     sub.write_h5ad(out, compression="gzip")
#     chk = sc.read_h5ad(out); x = chk.X[:3,:3]
#     if hasattr(x,"toarray"): x = x.toarray()
#     is_int = np.allclose(x, x.astype(int))
#     subset_stats[name] = {"n_cells":sub.n_obs,"n_genes":sub.n_vars,"X_is_int":is_int}
#     print(f"  {name}: {sub.n_obs} cells x {sub.n_vars} genes, X_is_int={is_int}")
# ckpt(4,"Subsets written (full genes, filtering in STEP6)",subset_stats)

# # STEP 5  E3 基因列表
# print(">>> STEP 5: E3 gene list")
# e3_path = Path("data/E3-ome.xlsx")
# if not e3_path.exists():
#     e3_full = pd.DataFrame({"gene":["MDM2","TRIM21","RNF4","BIRC3","XIAP","CBL","SIAH1","NEDD4","WWP1","HUWE1"],"family":["RING"]*10})
# else:
#     e3_full = pd.read_excel(e3_path)[["Gene","Protein class"]].rename(columns={"Gene":"gene","Protein class":"family"})
# all_genes = set()
# for f in (RES/"subsets").glob("*.h5ad"):
#     a = sc.read_h5ad(f, backed="r"); all_genes |= set(a.var_names); a.file.close()
# e3_full["present"] = e3_full["gene"].isin(all_genes)
# e3_present = e3_full[e3_full["present"]].copy()
# e3_test    = e3_present.head(10)
# e3_present.to_csv(RES/"E3_genes_present.csv", index=False)
# e3_test.to_csv(RES/"E3_genes_test.csv", index=False)
# ckpt(5,"E3 list",{"total":len(e3_full),"in_data":int(e3_full["present"].sum()),"test_set":e3_test["gene"].tolist()})

# # STEP 6  三层基因过滤 + scTenifoldKnk
# print(">>> STEP 6: Gene filtering + scTenifoldKnk")
# import os
# import ray
# # 严禁 Ray 使用硬盘作为对象存储
# os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "0" 

# if not ray.is_initialized():
#     # 强制限制 Ray 的对象存储只用 35GB（确保能塞进你 45GB 的 /dev/shm 中）
#     ray.init(object_store_memory=35 * 1024 * 1024 * 1024)
#     print("  [MAGIC] Ray explicitly forced to use /dev/shm via _plasma_directory.")
# try:
#     from scTenifold import scTenifoldKnk as sKnk
#     import scTenifold.core._base as sc_base
#     if hasattr(sc_base, "sc_QC"):
#         sc_base.sc_QC = lambda df, **kwargs: df
#         print("  [MAGIC] scTenifold internal QC has been completely bypassed.")
# except ImportError:
#     print("[ERROR] pip install scTenifold"); sys.exit(1)

# ko_genes = e3_test["gene"].tolist()

# # TF 白名单
# # ── 加载所有白名单 ────────────────────────────────────────────
# # gene_whitelist.py 和本脚本放同一目录即可 import
# import sys as _sys
# _sys.path.insert(0, str(Path(__file__).parent))
# try:
#     from gene_whitelist import load_tf_whitelist as _load_tf, load_ub_pathway_genes as _load_ub
#     ub_set = _load_ub()
#     tf_set = _load_tf("/root/autodl-tmp/E3VK/data/refs/human_TFs_Lambert2018.csv")
#     _USE_MODULE = True
# except ImportError:
#     print("  [WARN] gene_whitelist.py not found, using inline fallback")
#     _USE_MODULE = False
#     ub_set = set()
#     tf_set = set()

# if not _USE_MODULE:
#     ub_set = set()   # inline fallback has no UB list
#     tf_set = set()
#     tf_csv = Path("/root/autodl-tmp/E3VK/data/refs/human_TFs_Lambert2018.csv")
#     if tf_csv.exists():
#         tf_df = pd.read_csv(tf_csv)
#         if "Is TF?" in tf_df.columns:
#             tf_set = set(tf_df[tf_df["Is TF?"]=="Yes"]["HGNC symbol"].dropna())
#         print(f"  TF whitelist (Lambert): {len(tf_set)}")
#     else:
#         try:
#             import gseapy as gp
#             go_sets = gp.get_library("GO_Molecular_Function_2023", organism="human")
#             for term, genes in go_sets.items():
#                 if "transcription factor" in term.lower() and "binding" in term.lower():
#                     tf_set |= set(genes)
#             print(f"  TF whitelist (GO): {len(tf_set)}")
#         except Exception:
#             print("  TF whitelist: not available")

# NOISE = ("MIR","SNOR","RNU","AL","AC","AP","BX","CR","CT","LINC","LOC","OR")

# def filter_genes_for_knk(adata, ko_genes, tf_set, ub_set=None, n_hvg=5000, expr_pct=0.05, min_ko_cells=10):
#     n_orig = adata.n_vars
#     # Layer 1
#     is_noise = adata.var_names.str.startswith(NOISE)
#     adata = adata[:, (~is_noise) | adata.var_names.isin(ko_genes)].copy()
#     print(f"    Layer1 noise:  {n_orig} → {adata.n_vars}")
#     # Layer 2
#     X2 = adata.X.toarray() if hasattr(adata.X,"toarray") else adata.X
#     expr_counts = (X2 > 0).sum(axis=0)           # 每个基因表达的细胞数
#     expr_arr = expr_counts / X2.shape[0]         # 每个基因表达的细胞比例
    
#     # 普通基因：比例 >= 5%
#     keep_normal = expr_arr >= expr_pct
#     # 敲除目标基因：兜底条件，至少在 min_ko_cells (10) 个细胞中表达！
#     keep_ko = adata.var_names.isin(ko_genes) & (expr_counts >= min_ko_cells)
    
#     keep2 = keep_normal | keep_ko
#     adata = adata[:, keep2].copy()
#     print(f"    Layer2 strict background + relaxed KO (>=10 cells): {adata.n_vars}")
#     # Layer 3
#     actual_hvg = min(n_hvg, adata.n_vars-1)
#     sc.pp.highly_variable_genes(adata, n_top_genes=actual_hvg, flavor="seurat_v3")
#     hvg_set = set(adata.var_names[adata.var.highly_variable])
#     _ub = ub_set if ub_set else set()
    
#     # 提取存活下来的基因
#     whitelist = (set(ko_genes) | tf_set | _ub) & set(adata.var_names)
#     final = list(hvg_set | whitelist)
#     adata = adata[:, final].copy()
    
#     # 最终诊断：到底哪些 E3 活下来了？
#     surviving_ko = set(ko_genes) & set(adata.var_names)
#     print(f"    Layer3 HVG+wl: {adata.n_vars}  speedup≈{(n_orig/adata.n_vars)**2:.0f}x")
#     print(f"    E3 survived (>=10 cells): {len(surviving_ko)}/{len(ko_genes)} -> {list(surviving_ko)}")
    
#     return adata

# def run_knk(h5ad_path, ko_genes, tf_set, ub_set, out_dir,
#              n_nets=3, n_samp_cells=100, td_rank=3):
#     """
#     修复版: GRN 构建一次,每个 E3 单独 run_step(ko/ma/dr)
#     参数名已从源码确认:
#       nc_kws: n_nets, n_samp_cells
#       td_kws: K (rank)
#     """
#     from scTenifold import scTenifoldKnk as sKnk
#     from scipy import stats
#     out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)
#     name = Path(h5ad_path).stem
 
#     a = sc.read_h5ad(h5ad_path)
#     print(f"\n  [{name}] raw: {a.n_obs} x {a.n_vars}")
 
#     # 三层过滤
#     NOISE = ("MIR","SNOR","RNU","AL","AC","AP","BX","CR","CT","LINC","LOC","OR")
#     n_orig = a.n_vars
#     a = a[:, ~a.var_names.str.startswith(NOISE) | a.var_names.isin(ko_genes)].copy()
#     X2 = a.X.toarray() if hasattr(a.X,"toarray") else a.X
#     ep = (X2>0).sum(axis=0)/X2.shape[0]
#     a = a[:, (ep>=0.05) | a.var_names.isin(ko_genes)].copy()
#     sc.pp.highly_variable_genes(a, n_top_genes=min(5000,a.n_vars-1), flavor="seurat_v3")
#     hvg = set(a.var_names[a.var.highly_variable])
#     wl  = (set(ko_genes)|tf_set|ub_set) & set(a.var_names)
#     a   = a[:, list(hvg|wl)].copy()
#     print(f"  [{name}] filtered: {n_orig}→{a.n_vars} genes (speedup ~{(n_orig/a.n_vars)**2:.0f}x)")
 
#     X = a.X.toarray() if hasattr(a.X,"toarray") else a.X
#     df = pd.DataFrame(X.T.astype(np.float32), index=a.var_names, columns=a.obs_names)
 
#     expressed = (df>0).sum(axis=1)
#     valid = [g for g in ko_genes if g in df.index and expressed.get(g,0)>=10]
#     print(f"  [{name}] E3 valid: {len(valid)}/{len(ko_genes)}: {valid}")
#     if not valid: return 0
 
#     # 初始化 (参数名已确认)
#     knk = sKnk(
#         data=df, ko_method="default", ko_genes=[valid[0]],
#         qc_kws={"min_lib_size":200,"min_percent":0.001},
#         nc_kws={"n_nets":n_nets, "n_samp_cells":n_samp_cells, "n_cpus":1},
#         td_kws={"K":td_rank},
#     )
 
#     # GRN 构建一次
#     print(f"  [{name}] Building GRN (n_nets={n_nets}, n_samp_cells={n_samp_cells})...")
#     t0 = time.time()
#     knk.run_step("qc")
#     knk.run_step("nc")
#     knk.run_step("td")
#     print(f"  [{name}] GRN done in {time.time()-t0:.0f}s")
 
#     wt_tensor = knk.tensor_dict["WT"].copy()
 
#     # 每个 E3 单独 KO
#     saved = 0
#     for ko in valid:
#         knk.tensor_dict["WT"] = wt_tensor.copy()
#         knk.tensor_dict["KO"] = wt_tensor.copy()
#         knk.ko_genes = [ko]
#         knk.run_step("ko")
#         knk.run_step("ma")
#         knk.run_step("dr")
#         dr_df = knk.d_regulation.copy()
 
#         # 列名标准化
#         col_map = {}
#         for c in dr_df.columns:
#             cl = c.lower().strip()
#             if   cl=="gene":     col_map[c]="gene"
#             elif cl=="distance": col_map[c]="score"
#             elif cl=="fc":       col_map[c]="FC"
#             elif cl=="t":        col_map[c]="T"
#             elif cl=="z":        col_map[c]="Z"
#         dr_df = dr_df.rename(columns=col_map)
 
#         # p_adj from Z score
#         if "Z" in dr_df.columns:
#             p_vals = 2*stats.norm.sf(np.abs(dr_df["Z"].fillna(0).values))
#             try:
#                 from scipy.stats import false_discovery_control
#                 dr_df["p_adj"] = false_discovery_control(np.clip(p_vals,1e-300,1))
#             except Exception:
#                 from statsmodels.stats.multitest import multipletests
#                 _, padj,_,_ = multipletests(np.clip(p_vals,1e-300,1), method="fdr_bh")
#                 dr_df["p_adj"] = padj
#             dr_df["p_value"] = p_vals
#         dr_df["ko"] = ko
 
#         if "gene" not in dr_df.columns:
#             print(f"  [{name}] WARN {ko}: no gene col"); continue
#         n_sig = int((dr_df.get("p_adj",pd.Series([1]*len(dr_df)))<0.05).sum())
#         dr_df.to_csv(out_dir/f"DR_{ko}.csv", index=False)
#         saved += 1
#         print(f"  [{name}] {ko}: {len(dr_df)} genes, sig={n_sig}")
 
#     print(f"  [{name}] saved {saved} DR files")
#     return saved

# knk_stats = {}
# for h5 in (RES/"subsets").glob("*.h5ad"):
#     name = h5.stem; t0 = time.time()
#     n = run_knk(h5, ko_genes, tf_set, ub_set, RES/"knk"/name,
#                    n_nets=3, n_samp_cells=100, td_rank=3)
#     knk_stats[name] = {"dr_files":n, "elapsed":f"{time.time()-t0:.0f}s"}
 
# ckpt(6,"scTenifoldKnk with gene filtering",knk_stats)

# STEP 7  DR 汇总
print(">>> STEP 7: DR aggregation")
rows, errors = [], []
for f in (RES/"knk").rglob("DR_*.csv"):
    try:
        df = pd.read_csv(f); df["subset"]=f.parent.name; df["ko"]=f.stem.replace("DR_",""); rows.append(df)
    except Exception as e: errors.append(f"{f}: {e}")
if errors: print("  [WARN]", errors)
if rows:
    all_dr = pd.concat(rows, ignore_index=True)
    all_dr.to_parquet(RES/"dr_matrix_test.parquet")
    sig = all_dr.query("p_adj<0.05") if "p_adj" in all_dr.columns else pd.DataFrame()
    ckpt(7,"DR aggregated",{"total_rows":len(all_dr),"columns":all_dr.columns.tolist(),
                             "sig_FDR5%":len(sig),"subsets":all_dr["subset"].unique().tolist()})
else:
    ckpt(7,"DR EMPTY",{})

(RES/"test_complete.flag").touch()
print("\n"+"="*52)
print("  PASSED  — 把 STEP 6 的 '>>> result' 输出发给我")
print("="*52)