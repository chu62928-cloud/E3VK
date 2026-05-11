"""
Inputs:  results/dr_matrix.parquet, results/vulnerability.csv,
         results/integrated_annotated.h5ad, data/UbiBrowser_substrates.csv
Outputs: figs/* and results/enrichment/*.csv
"""
import scanpy as sc, pandas as pd, numpy as np
import gseapy as gp, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from itertools import combinations

RES = Path("results"); FIG = Path("figs"); FIG.mkdir(exist_ok=True)
(RES/"enrichment").mkdir(exist_ok=True)

e3_meta = pd.read_csv(RES/"E3_genes_present.csv")
dr   = pd.read_parquet(RES/"dr_matrix.parquet")
vuln = pd.read_csv(RES/"vulnerability.csv", index_col=0)

# ============ A. 亚群间差异反应:vulnerability heatmap ============
top_e3 = vuln.sum(0).sort_values(ascending=False).head(50).index
plt.figure(figsize=(10,8))
sns.heatmap(vuln[top_e3], cmap="rocket_r", linewidths=0.2,
            cbar_kws={"label":"# DR genes (FDR<0.05)"})
plt.title("Subset × E3 vulnerability (top 50 E3)")
plt.tight_layout(); plt.savefig(FIG/"vulnerability_heatmap.pdf"); plt.close()

# ============ B. KO 命中的亚群特异性:Jaccard 距离 ============
def jaccard(a, b): 
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))

specificity = []
for ko, g in dr.query("`p.adj`<0.05").groupby("ko"):
    sets = g.groupby("subset")["gene"].apply(list).to_dict()
    if len(sets) < 2: continue
    js = [jaccard(sets[a], sets[b]) for a,b in combinations(sets, 2)]
    specificity.append({"ko": ko, "mean_jaccard": np.mean(js),
                        "n_subsets_hit": len(sets)})
spec_df = pd.DataFrame(specificity).sort_values("mean_jaccard")
spec_df.to_csv(RES/"e3_subset_specificity.csv", index=False)
# 低 Jaccard = 亚群特异;高 = 通用守门员

# ============ C. 通路富集:对每个 (subset, KO) 跑 GSEApy ============
gene_sets = ["KEGG_2021_Human","Reactome_2022","GO_Biological_Process_2023",
             "MSigDB_Hallmark_2020"]

# 只对 vulnerability 排名前 30 的 (subset, KO) 跑富集,避免 672*7 次
focus = (vuln.stack().reset_index()
              .rename(columns={"level_1":"ko",0:"n_dr"})
              .nlargest(30,"n_dr"))

for _, row in focus.iterrows():
    sub, ko, n = row["subset"], row["ko"], row["n_dr"]
    genes = (dr.query("subset==@sub and ko==@ko and `p.adj`<0.05")
               .sort_values("p.adj")["gene"].head(300).tolist())
    if len(genes) < 15: continue
    enr = gp.enrichr(gene_list=genes, gene_sets=gene_sets,
                     organism="human", outdir=None, no_plot=True)
    enr.results.to_csv(RES/"enrichment"/f"{sub}__{ko}.csv", index=False)

# ============ D. T 细胞功能模块评分 KO 前后变化 ============
# 用 scTenifoldKnk 的 KO 后表达谱(res$tensorNetworks$Y) 估算
# 简化做法:基于 DR 基因方向重新打分原始 adata
adata = sc.read_h5ad(RES/"integrated_annotated.h5ad")
modules = {
    "cytotoxicity": ["GZMB","GZMA","PRF1","GNLY","NKG7","IFNG"],
    "exhaustion":   ["PDCD1","HAVCR2","LAG3","TIGIT","TOX","CTLA4"],
    "treg":         ["FOXP3","IL2RA","IKZF2","CTLA4","TNFRSF18"],
    "ifn_response": ["ISG15","IFIT1","IFIT3","MX1","OAS1","STAT1"],
    "naive":        ["CCR7","SELL","TCF7","LEF1","IL7R"],
}
for name, genes in modules.items():
    sc.tl.score_genes(adata, [g for g in genes if g in adata.var_names],
                      score_name=f"score_{name}")

# ============ E. KO 后释放底物验证(UbiBrowser 交叉) ============
sub_db = pd.read_csv("data/UbiBrowser_substrates.csv")  # cols: E3, substrate
# 期望:KO 一个 E3 后,其底物在 DR 上调基因里出现
val = []
for ko, g in dr.query("`p.adj`<0.05").groupby("ko"):
    known = sub_db.query("E3==@ko")["substrate"].tolist()
    if not known: continue
    hit_rate = g["gene"].isin(known).mean()
    val.append({"ko": ko, "n_known_substrates": len(known),
                "dr_overlap_rate": hit_rate,
                "n_overlap": g["gene"].isin(known).sum()})
pd.DataFrame(val).sort_values("n_overlap", ascending=False)\
   .to_csv(RES/"substrate_validation.csv", index=False)

# ============ F. 总览图:特异性 vs 影响力散点 ============
summary = (vuln.stack().reset_index()
              .rename(columns={"level_1":"ko",0:"n_dr"})
              .merge(spec_df, on="ko"))
plt.figure(figsize=(8,6))
sns.scatterplot(data=summary, x="mean_jaccard", y="n_dr",
                hue="subset", alpha=0.6)
plt.xlabel("Cross-subset Jaccard (low = specific)")
plt.ylabel("# DR genes per subset")
plt.title("E3 KO landscape: specificity vs magnitude")
plt.tight_layout(); plt.savefig(FIG/"specificity_vs_magnitude.pdf"); plt.close()

# ============ G. Family-level analysis ============
def simplify(f):
    if pd.isna(f): return "Other"
    f = str(f)
    if "RING-between-RING" in f: return "RBR"
    if f.startswith("RING") or "Degenerate RING" in f: return "RING"
    if "HECT" in f: return "HECT"
    if f.startswith("CRL"): return "CRL"
    if "SUMO" in f: return "SUMO_E3"
    return "Atypical"
e3_meta["fam_simple"] = e3_meta["family"].apply(simplify)

# Family x subset: 平均 vulnerability
fam_map = e3_meta.set_index("gene")["fam_simple"].to_dict()
vuln = pd.read_csv(RES/"vulnerability.csv", index_col=0)   # subset x KO
long = vuln.stack().reset_index()
long.columns = ["subset","ko","n_dr"]
long["family"] = long["ko"].map(fam_map)
fam_subset = long.groupby(["subset","family"])["n_dr"].mean().unstack()

plt.figure(figsize=(8,5))
sns.heatmap(fam_subset, cmap="rocket_r", annot=True, fmt=".1f",
            cbar_kws={"label":"Mean DR genes per E3 in family"})
plt.title("E3 family vs subset vulnerability (mean)")
plt.tight_layout(); plt.savefig(FIG/"family_vulnerability.pdf"); plt.close()

# Family-level GSEA: 每个亚群里,DR 上的家族富集
# 用 Fisher exact 看每个 family 是否在该亚群 top-DR-hit E3 里富集
from scipy.stats import fisher_exact
fam_enrich = []
for sub in vuln.index:
    top_n = max(50, int(0.1 * vuln.shape[1]))
    top_e3 = vuln.loc[sub].nlargest(top_n).index.tolist()
    for fam in e3_meta["fam_simple"].unique():
        fam_genes = e3_meta.query("fam_simple==@fam")["gene"].tolist()
        a = len(set(top_e3) & set(fam_genes))
        b = len(top_e3) - a
        c = len(fam_genes) - a
        d = vuln.shape[1] - a - b - c
        odds, p = fisher_exact([[a,b],[c,d]], alternative="greater")
        fam_enrich.append({"subset":sub, "family":fam,
                           "n_top": a, "n_family": len(fam_genes),
                           "odds_ratio": odds, "p": p})
pd.DataFrame(fam_enrich).to_csv(RES/"family_subset_enrichment.csv", index=False)