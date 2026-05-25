"""
_config.py — 下游分析共享配置

所有路径都相对 PROJECT_ROOT 解析。若项目根目录不同，改这一处即可。
"""
from __future__ import annotations
from pathlib import Path
import os
import pickle
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
# 默认假设脚本被放在 E3VK/scripts/，PROJECT_ROOT 自动定位到 E3VK/。
# 可通过环境变量 E3VK_ROOT 覆盖。
PROJECT_ROOT = Path(os.environ.get("E3VK_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR        = PROJECT_ROOT / "data"
RESULTS_DIR     = PROJECT_ROOT / "results" / "AIHA"
KNK_DIR         = RESULTS_DIR / "knk"
SUBSETS_DIR     = RESULTS_DIR / "AIHA" / "subsets"

E3_TABLE        = DATA_DIR / "E3-ome.xlsx"
E3_PRESENT_CSV  = RESULTS_DIR / "E3_genes_present.csv"
TF_LIST_CSV     = DATA_DIR / "refs" / "human_TFs_Lambert2018.csv"

DOWNSTREAM_DIR  = RESULTS_DIR / "downstream"
FIGS_DIR        = DOWNSTREAM_DIR / "figs"
ENRICH_DIR      = DOWNSTREAM_DIR / "enrichment"
TF_ENRICH_DIR   = DOWNSTREAM_DIR / "tf_enrichment"
REFS_DIR        = DATA_DIR / "refs"

for d in (DOWNSTREAM_DIR, FIGS_DIR, ENRICH_DIR, TF_ENRICH_DIR, REFS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 显著性阈值
# ---------------------------------------------------------------------------
FDR_STRICT  = 0.05    # 默认显著阈值
FDR_LOOSE   = 0.10    # 论文 Figure 2C 用的阈值

# ---------------------------------------------------------------------------
# scTenifoldKnk DR CSV 的列名（按 README 中约定）
# ---------------------------------------------------------------------------
DR_COL_GENE  = "gene"      # 被影响的基因
DR_COL_SCORE = "score"     # manifold distance
DR_COL_FC    = "FC"        # fold change of distance (论文里同名)
DR_COL_T     = "T"
DR_COL_Z     = "Z"
DR_COL_P     = "p_value"
DR_COL_PADJ  = "p_adj"
DR_COL_KO    = "ko"        # 被敲除的 E3

# ---------------------------------------------------------------------------
# E3 家族归类
# E3-ome.xlsx 的 Protein class 列里有 RING / HECT / RBR / CRL 等多种类别字符串
# 这里把所有可能的细分映射成 5 个一级类，方便家族层面统计。
#
# 注意优先级: CRL/RBR 要在 RING 之前匹配, 因为「Cullin RING Ligase」「RBR」
# 都包含 "RING" 子串, 否则会被误判为普通 RING。
# ---------------------------------------------------------------------------
FAMILY_RULES = [
    # (family_name, keyword_list)  按此顺序匹配, 先匹配先归类
    ("CRL",  ["CRL", "Cullin", "F-box", "SOCS-box", "BC-box", "BTB", "DCAF", "Elongin"]),
    ("RBR",  ["RBR"]),
    ("HECT", ["HECT"]),
    ("RING", ["RING", "U-box", "Ubox"]),
]

def classify_family(protein_class: str) -> str:
    if not isinstance(protein_class, str):
        return "Other"
    pc = protein_class.strip().lower()
    if not pc:
        return "Other"
    for fam, keys in FAMILY_RULES:
        for k in keys:
            if k.lower() in pc:
                return fam
    return "Other"

# ---------------------------------------------------------------------------
# matplotlib 全局风格
# ---------------------------------------------------------------------------
def set_mpl_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
        "savefig.bbox": "tight",
        "savefig.dpi":  300,
        "font.size":    9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def list_completed_subsets() -> list[str]:
    """以 <subset>_knk.pkl 是否存在为完成标志（与 README 一致）。"""
    if not KNK_DIR.exists():
        return []
    out = []
    for d in sorted(KNK_DIR.iterdir()):
        if d.is_dir() and (d / f"{d.name}_knk.pkl").exists():
            out.append(d.name)
    return out

def load_wt(subset: str) -> "pd.DataFrame | None":
    """Load WT scGRN adjacency matrix from Wd.parquet or <subset>_knk.pkl."""
    wd = KNK_DIR / subset / "Wd.parquet"
    if wd.exists():
        return pd.read_parquet(wd)
    pkl = KNK_DIR / subset / f"{subset}_knk.pkl"
    if pkl.exists():
        try:
            with open(pkl, "rb") as f:
                obj = pickle.load(f)
            td = getattr(obj, "tensor_dict", None) or getattr(obj, "_tensor_dict", None)
            if td and "WT" in td and isinstance(td["WT"], pd.DataFrame):
                return td["WT"]
        except Exception as e:
            print(f"  [{subset}] pkl load failed: {e}", file=sys.stderr)
    return None


def load_e3_family() -> pd.DataFrame:
    """读 E3-ome.xlsx, 输出 Gene -> family 映射。"""
    if not E3_TABLE.exists():
        # 如果文件不存在，返回一个空的 DataFrame 防止报错
        return pd.DataFrame(columns=["gene", "protein_class", "family"])
        
    df = pd.read_excel(E3_TABLE)
    # 容忍列名大小写差异
    cols_lower = {c.lower(): c for c in df.columns}
    gene_col  = cols_lower.get("gene")  or list(df.columns)[0]
    class_col = cols_lower.get("protein class") or cols_lower.get("class") or list(df.columns)[-1]
    
    out = df[[gene_col, class_col]].rename(columns={gene_col: "gene", class_col: "protein_class"})
    out["family"] = out["protein_class"].apply(classify_family)
    return out