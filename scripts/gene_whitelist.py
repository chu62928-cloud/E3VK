"""
gene_whitelist.py  ·  被 01 和 03 脚本 import 的共享模块
=========================================================
提供两个函数:
  load_tf_whitelist(path)      → set of TF gene symbols
  load_ub_pathway_genes()      → set of ubiquitin pathway gene symbols

使用方式 (在 01/03 脚本顶部):
  from gene_whitelist import load_tf_whitelist, load_ub_pathway_genes
"""

import pandas as pd
from pathlib import Path


# ══════════════════════════════════════════════════
# 硬编码的泛素通路核心基因
# 来源: KEGG hsa04120 + UniProt keyword "Ubiquitin" + 文献
# 这些基因保证始终在白名单中,不依赖任何外部下载
# ══════════════════════════════════════════════════
_UB_CORE_GENES = set([

    # ── 泛素蛋白本身 ──────────────────────────────
    "UBB", "UBC", "UBA52", "RPS27A",

    # ── E1 泛素激活酶 (2个) ──────────────────────
    "UBA1", "UBA6",

    # ── E2 泛素结合酶 (主要亚家族) ───────────────
    "UBE2A", "UBE2B", "UBE2C", "UBE2D1", "UBE2D2", "UBE2D3", "UBE2D4",
    "UBE2E1", "UBE2E2", "UBE2E3", "UBE2F", "UBE2G1", "UBE2G2",
    "UBE2H", "UBE2I", "UBE2J1", "UBE2J2", "UBE2K", "UBE2L3", "UBE2L6",
    "UBE2M", "UBE2N", "UBE2NL", "UBE2O", "UBE2Q1", "UBE2Q2",
    "UBE2R1", "UBE2R2", "UBE2S", "UBE2T", "UBE2U", "UBE2V1", "UBE2V2",
    "UBE2W", "UBE2Z", "CDC34",

    # ── 主要 DUB 去泛素化酶 ──────────────────────
    # USP 家族
    "USP1",  "USP2",  "USP3",  "USP4",  "USP5",  "USP6",  "USP7",
    "USP8",  "USP9X", "USP9Y", "USP10", "USP11", "USP12", "USP13",
    "USP14", "USP15", "USP16", "USP17L2", "USP18", "USP19", "USP20",
    "USP21", "USP22", "USP24", "USP25", "USP26", "USP27X", "USP28",
    "USP29", "USP30", "USP31", "USP32", "USP33", "USP34", "USP35",
    "USP36", "USP37", "USP38", "USP39", "USP40", "USP41", "USP42",
    "USP43", "USP44", "USP45", "USP46", "USP47", "USP48", "USP49",
    "USP50", "USP51", "USP52", "USP53", "USP54",
    # OTU 家族
    "OTUB1", "OTUB2", "OTUD1", "OTUD3", "OTUD4", "OTUD5", "OTUD6A",
    "OTUD6B", "OTUD7A", "OTUD7B", "OTULIN", "VCPIP1", "TNFAIP3",
    "ZRANB1",
    # UCH 家族
    "UCHL1", "UCHL3", "UCHL5", "BAP1",
    # JAMM/MPN 家族
    "PSMD7", "PSMD14", "BRCC3", "COPS5", "COPS6", "EIF3H", "MYSM1",
    # MJD 家族
    "ATXN3", "ATXN3L", "JOSD1", "JOSD2",

    # ── 26S 蛋白酶体亚基 ─────────────────────────
    # 19S 调节颗粒 (lid)
    "PSMD1",  "PSMD2",  "PSMD3",  "PSMD4",  "PSMD5",  "PSMD6",
    "PSMD7",  "PSMD8",  "PSMD9",  "PSMD10", "PSMD11", "PSMD12",
    "PSMD13", "PSMD14",
    # 19S 调节颗粒 (base, ATPase)
    "PSMC1", "PSMC2", "PSMC3", "PSMC4", "PSMC5", "PSMC6",
    # 20S 核心颗粒 (alpha)
    "PSMA1", "PSMA2", "PSMA3", "PSMA4", "PSMA5", "PSMA6", "PSMA7",
    "PSMA8",
    # 20S 核心颗粒 (beta,含催化亚基)
    "PSMB1", "PSMB2", "PSMB3", "PSMB4", "PSMB5", "PSMB6", "PSMB7",
    "PSMB8", "PSMB9", "PSMB10", "PSMB11",
    # 11S 调节颗粒
    "PSME1", "PSME2", "PSME3", "PSME4",
    # 其他蛋白酶体相关
    "PSMF1", "POMP", "UBL5",

    # ── CRL/SCF 复合体核心组分 ───────────────────
    # Cullin 家族
    "CUL1", "CUL2", "CUL3", "CUL4A", "CUL4B", "CUL5", "CUL7", "CUL9",
    # RING-box/ROC
    "RBX1", "RBX2",
    # SKP1 (SCF 衔接)
    "SKP1",
    # CSN/COP9 信号小体 (调控 CRL neddylation)
    "COPS1", "COPS2", "COPS3", "COPS4", "COPS5", "COPS6", "COPS7A",
    "COPS7B", "COPS8",

    # ── NEDD8 修饰通路 ───────────────────────────
    "NEDD8", "NAE1", "UBA3", "UBE2F", "UBE2M",
    "SENP8",   # NEDD8 去修饰
    "NEDP1",

    # ── SUMO 修饰通路 ────────────────────────────
    # (SUMO 与泛素化有拮抗/协同关系,影响 E3 底物选择)
    "SUMO1", "SUMO2", "SUMO3", "SUMO4",
    "SAE1", "UBA2",       # E1
    "UBE2I",              # E2 (UBC9)
    "SENP1", "SENP2", "SENP3", "SENP5", "SENP6", "SENP7",  # SUMO 蛋白酶

    # ── p97/VCP 相关 (ERAD 和蛋白质质控) ─────────
    "VCP", "UBXN1", "UBXN2A", "UBXN2B", "UBXN4", "UBXN6", "UBXN7",
    "NSFL1C", "UFD1", "NPLOC4", "FAF1", "FAF2",

    # ── 泛素受体和衔接蛋白 ───────────────────────
    "UBQLN1", "UBQLN2", "UBQLN4",
    "RAD23A", "RAD23B",
    "SQSTM1",   # p62, 自噬-蛋白酶体交叉
    "HDAC6",    # 泛素受体,调控错误折叠蛋白
    "RACK1",

    # ── 泛素结合结构域蛋白 (UBA/UBL/UIM 等) ─────
    "UBAC1", "UBAC2",
    "UBL4A", "UBL5",
    "HGS",     # UIM domain, ESCRT
    "STAM",    # UIM domain, ESCRT
    "EPS15",   # UIM domain

    # ── 免疫相关特殊泛素通路 (T细胞重要) ─────────
    "LUBAC",   # alias for HOIL-1L complex
    "RBCK1",   # HOIL-1L (LUBAC 组分, RBR E3)
    "RNF31",   # HOIP (LUBAC 核心, 线性泛素)
    "SHARPIN", # LUBAC 组分
    "CYLD",    # DUB,去除线性/K63 泛素链
    "A20",     # TNFAIP3, 免疫负调控 DUB/E3
    "LUBAC",
    "TRAF6",   # K63 E3,NF-κB 通路
    "TRAF2",   # K63 E3,TNF 通路
    "TRAF3",   # E3,I型干扰素通路

])


def load_ub_pathway_genes(extra_csv: str = None) -> set:
    """
    返回泛素通路白名单基因集合。

    优先使用硬编码列表(~300个核心基因),
    可选通过 extra_csv 追加自定义基因。

    参数:
        extra_csv: 额外基因列表 CSV 路径(第一列为 gene symbol),可选
    Returns:
        set of gene symbols
    """
    genes = set(_UB_CORE_GENES)

    # 可选:追加自定义基因
    if extra_csv:
        p = Path(extra_csv)
        if p.exists():
            try:
                extra = pd.read_csv(p)
                added = set(extra.iloc[:, 0].dropna().astype(str))
                genes |= added
                print(f"[UB] Extra genes from {p.name}: +{len(added)} → total {len(genes)}")
            except Exception as e:
                print(f"[UB] extra_csv read error: {e}")

    # 可选:从 gseapy 追加 KEGG hsa04120 (网络可用时)
    try:
        import gseapy as gp
        kegg = gp.get_library("KEGG_2021_Human", organism="human")
        ub_kegg = kegg.get("Ubiquitin mediated proteolysis", [])
        if ub_kegg:
            before = len(genes)
            genes |= set(ub_kegg)
            print(f"[UB] KEGG hsa04120: +{len(genes)-before} genes → total {len(genes)}")
    except Exception:
        pass   # 网络不可用时静默跳过,用硬编码列表

    print(f"[UB] Ubiquitin pathway whitelist: {len(genes)} genes")
    return genes


def load_tf_whitelist(tf_csv_path: str) -> set:
    """
    加载 TF 白名单。
    优先 Lambert 2018 CSV;不可用时从 gseapy GO 拉;都失败返回空集。
    """
    p = Path(tf_csv_path)
    if p.exists():
        try:
            df = pd.read_csv(p)
            if "Is TF?" in df.columns and "HGNC symbol" in df.columns:
                tfs = set(df[df["Is TF?"] == "Yes"]["HGNC symbol"].dropna())
                print(f"[TF] Lambert2018 CSV: {len(tfs)} TFs")
                return tfs
            tfs = set(df.iloc[:, 0].dropna().astype(str))
            print(f"[TF] fallback first col: {len(tfs)}")
            return tfs
        except Exception as e:
            print(f"[TF] CSV read error: {e}")

    try:
        import gseapy as gp
        print("[TF] CSV not found, fetching from GO_MF via gseapy ...")
        go_sets = gp.get_library("GO_Molecular_Function_2023", organism="human")
        tfs = set()
        for term, genes in go_sets.items():
            if "transcription factor" in term.lower() and "binding" in term.lower():
                tfs |= set(genes)
        print(f"[TF] GO-based: {len(tfs)} TF genes")
        return tfs
    except Exception as e:
        print(f"[TF] gseapy fallback failed: {e}")

    print("[TF] No TF list — using HVG + E3 + UB pathway only")
    return set()