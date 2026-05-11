"""
01b_zheng_load.py
处理 Zheng 2021 pan-cancer T cell atlas + Schafflick 2020 MS

策略:
  - 使用作者已有的 meta.cluster 注释(质量极高,直接信任)
  - 保留癌种和组织来源信息用于后续对比
  - B 细胞从 Schafflick 数据集补充
"""
import scanpy as sc, anndata as ad, pandas as pd, numpy as np
from pathlib import Path

DATA = Path("data"); RES = Path("results")
(RES/"subsets").mkdir(parents=True, exist_ok=True)

# ============ 1. 加载 Zheng CD4 + CD8 ============
adatas = {}
for which in ["CD4", "CD8"]:
    a = sc.read_h5ad(DATA/"zheng2021"/f"{which}.h5ad")
    print(f"\n=== Zheng {which}: {a.shape} ===")
    print(f"  meta.cluster levels: {a.obs['meta.cluster'].nunique()}")
    print(a.obs['meta.cluster'].value_counts().head(20))

    # raw counts
    if a.raw is not None:
        a.layers["counts"] = a.raw[:, a.var_names].X.copy()
    elif "counts" not in a.layers:
        # 如果 X 是 normalized,需要其他来源
        x_sample = a.X[:3,:3].toarray() if hasattr(a.X,"toarray") else a.X[:3,:3]
        if not np.allclose(x_sample, x_sample.astype(int)):
            print(f"  WARNING: no raw counts found, using X (may be normalized)")
        a.layers["counts"] = a.X.copy()

    a.obs["dataset_origin"] = f"zheng_{which}"
    adatas[which] = a

# ============ 2. Zheng 的标签映射到我们的目标亚群 ============
# meta.cluster 命名规则:CD4.c01.Tn.TCF7 = cluster 编号.功能.marker
# 论文 Table S2 给出 c00-c20 的功能标注

TARGET_SUBSETS = {
    # CD4 helper polarization
    "CD4_naive":      ("CD4", lambda s: s.str.contains("Tn|naive", case=False)),
    "CD4_Tcm":        ("CD4", lambda s: s.str.contains("Tcm", case=False)),
    "CD4_Tem":        ("CD4", lambda s: s.str.contains("Tem", case=False) & ~s.str.contains("Th17|Th1", case=False)),
    "CD4_Th1":        ("CD4", lambda s: s.str.contains(r"\.Th1\.", regex=True) & ~s.str.contains("Tfh", case=False)),
    "CD4_Th17":       ("CD4", lambda s: s.str.contains("Th17", case=False)),
    "CD4_Tfh":        ("CD4", lambda s: s.str.contains("Tfh", case=False) & ~s.str.contains("Th1", case=False)),
    "CD4_TfhTh1":     ("CD4", lambda s: s.str.contains("TfhTh1|Tfh.*Th1", case=False, regex=True)),
    "CD4_Treg":       ("CD4", lambda s: s.str.contains("Treg", case=False)),

    # CD8 functional states
    "CD8_naive":          ("CD8", lambda s: s.str.contains("Tn", case=False) & ~s.str.contains("Trm|Tex", case=False)),
    "CD8_Tcm":            ("CD8", lambda s: s.str.contains("Tcm", case=False)),
    "CD8_Tem_GZMK":       ("CD8", lambda s: s.str.contains("Tem.*GZMK|GZMK.*Tem", case=False, regex=True)),
    "CD8_Tem_general":    ("CD8", lambda s: s.str.contains("Tem", case=False) & ~s.str.contains("GZMK", case=False)),
    "CD8_Trm":            ("CD8", lambda s: s.str.contains("Trm", case=False) & ~s.str.contains("Tex", case=False)),
    "CD8_Tex_terminal":   ("CD8", lambda s: s.str.contains("Tex.*PDCD1|Tex.*HAVCR2", case=False, regex=True)),
    "CD8_Tex_proliferating": ("CD8", lambda s: s.str.contains("Tex.*MKI67|Tex.*proliferat", case=False, regex=True)),
    "CD8_MAIT":           ("CD8", lambda s: s.str.contains("MAIT", case=False)),
}

summary = []
for name, (src, sel) in TARGET_SUBSETS.items():
    a = adatas[src]
    mask = sel(a.obs["meta.cluster"].astype(str))
    sub = a[mask].copy()
    n = sub.n_obs
    summary.append({"subset": name, "source": f"zheng_{src}", "n_cells": n})
    if n < 300:
        print(f"[skip] {name}: only {n} cells")
        continue

    sub.X = sub.layers["counts"]
    # 保留 cancerType 和 loc 用于后续 batch correction
    keep_cols = ["meta.cluster","cancerType","loc","dataset","patient"]
    keep_cols = [c for c in keep_cols if c in sub.obs.columns]
    sub.obs = sub.obs[keep_cols].copy()
    for ok in list(sub.obsm.keys()):
        if ok not in ["X_umap","X_scVI"]: del sub.obsm[ok]
    sub.uns = {}; sub.obsp = {}
    sub.write_h5ad(RES/"subsets"/f"{name}.h5ad", compression="gzip")
    print(f"✓ {name}: {n} cells")

# ============ 3. 加载 Schafflick MS 数据 + B 细胞 ============
# Schafflick 是 10x 标准格式,每个样本一个三元组
# 我们用作者提供的元数据合并所有样本
# (这里假设你已经预处理为单个 h5ad,细节略)
# ms_path = DATA/"schafflick2020"/"merged.h5ad"
# if ms_path.exists():
#     ms = sc.read_h5ad(ms_path)
#     # Schafflick 在论文里给出了 cluster 注释,大致是 CD4 Th1, CD4 Th17, B cells 等
#     # 取 B 细胞和 Th17 增强样本
#     if "cell_type" in ms.obs.columns:
#         for label, name in [("B cells","B_naive_MS"),
#                             ("memory B cells","B_memory_MS"),
#                             ("CD4 Th17","CD4_Th17_MS"),
#                             ("CD4 Th1","CD4_Th1_MS")]:
#             sub = ms[ms.obs["cell_type"].str.contains(label, case=False, na=False)].copy()
#             if sub.n_obs >= 200:
#                 sub.X = sub.layers.get("counts", sub.X)
#                 sub.write_h5ad(RES/"subsets"/f"{name}.h5ad", compression="gzip")
#                 summary.append({"subset": name, "source": "schafflick_MS", "n_cells": sub.n_obs})
#                 print(f"✓ {name} (MS): {sub.n_obs}")

# pd.DataFrame(summary).to_csv(RES/"subset_summary.csv", index=False)

# ============ 4. E3 列表与基因取交集 ============
e3 = pd.read_excel("data/E3-ome.xlsx").rename(columns={"Gene":"gene","Protein class":"family"})
all_var = set()
for f in (RES/"subsets").glob("*.h5ad"):
    a = sc.read_h5ad(f, backed="r"); all_var |= set(a.var_names); a.file.close()

e3["present"] = e3["gene"].isin(all_var)
e3.to_csv(RES/"E3_full_with_presence.csv", index=False)
e3_present = e3.query("present").copy()
e3_present.to_csv(RES/"E3_genes_present.csv", index=False)
print(f"\nE3 present: {len(e3_present)}/{len(e3)}")