#!/usr/bin/env python
"""
09_redundancy_collapse.py — Collapse redundant gene sets by leading-edge Jaccard

完全对应 Osorio et al. 2022, Methods:
  "To remove redundant results produced by the GSEA analysis, identified
   significant functional terms and gene sets were grouped according to the
   overlap of leading-edge genes. More specifically, for a given pair of
   identified gene sets, if Jaccard index, J = |A ∩ B|/|A ∪ B|, is greater
   than 0.8, where A and B are leading-edge genes of the two gene sets,
   then this pair of gene sets were grouped together."

为什么必须做:
  672 个 E3 × 9 个库 = 海量 term, 同义/嵌套通路（如 "Lysosome" / "Lysosomal
  transport" / "Lysosome assembly"）会被重复列出, 表格变得无法看. 论文用
  Jaccard ≥ 0.8 合并, 报告 "function group" 而非单个 term, 这是其结果可读
  的关键.

实现:
  - 对每个 <subset>/<E3>_gsea.csv, 取 FDR < 0.25 的显著 term
  - 解析 gseapy 的 'Lead_genes' 列, 计算两两 Jaccard
  - 用 networkx 的 connected_components 合并 (Jaccard > threshold 连边)
  - 每组保留 NES 最大 / FDR 最小的代表 term, 加列 group_size
  - 输出 dedup 后的 CSV, 同时保留原始 CSV (不覆盖)

输出:
  results/AIHA/downstream/enrichment/<subset>/<E3>_gsea_dedup.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _config import ENRICH_DIR


# gseapy res2d 的 leading-edge 列名（不同版本叫法）
LEAD_CANDIDATES = ["Lead_genes", "Leading_edge", "Genes", "LeadingEdge"]
TERM_CANDIDATES = ["Term", "term", "name"]
FDR_CANDIDATES  = ["FDR q-val", "fdr", "FDR", "NOM p-val", "p-val"]
NES_CANDIDATES  = ["NES", "nes", "ES"]


def pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def parse_lead(s) -> set[str]:
    if pd.isna(s):
        return set()
    s = str(s)
    # gseapy 用 ';' 或 ',' 分隔
    for sep in (";", ",", "|"):
        if sep in s:
            return {x.strip() for x in s.split(sep) if x.strip()}
    return {s.strip()} if s.strip() else set()


def collapse_one(df: pd.DataFrame, threshold: float, fdr_cut: float) -> pd.DataFrame:
    """Take a single GSEA result df, return de-duplicated rows + group_id col."""
    term_col = pick_col(df.columns, TERM_CANDIDATES)
    lead_col = pick_col(df.columns, LEAD_CANDIDATES)
    fdr_col  = pick_col(df.columns, FDR_CANDIDATES)
    nes_col  = pick_col(df.columns, NES_CANDIDATES)
    if not all([term_col, lead_col, fdr_col]):
        return df.assign(group_id=np.nan, group_size=1, group_rep=True)

    work = df.copy()
    work[fdr_col] = pd.to_numeric(work[fdr_col], errors="coerce")
    sig = work[work[fdr_col] < fdr_cut].copy()
    if sig.empty:
        return work.assign(group_id=np.nan, group_size=1, group_rep=False)

    sig["_lead"] = sig[lead_col].map(parse_lead)
    # 构图: 节点 = term 索引, 边 = Jaccard > threshold
    G = nx.Graph()
    idx = sig.index.tolist()
    for i in idx:
        G.add_node(i)
    leads = {i: sig.loc[i, "_lead"] for i in idx}
    for ii, i in enumerate(idx):
        for j in idx[ii + 1:]:
            a, b = leads[i], leads[j]
            if not a or not b:
                continue
            jac = len(a & b) / max(1, len(a | b))
            if jac >= threshold:
                G.add_edge(i, j, w=jac)
    # 每个 connected component 选一个代表（NES 最大；缺则 FDR 最小）
    groups = list(nx.connected_components(G))
    group_id_map = {}
    rep_set = set()
    for gid, members in enumerate(groups):
        mem_df = sig.loc[list(members)]
        if nes_col and mem_df[nes_col].notna().any():
            rep = mem_df.sort_values(nes_col, ascending=False).index[0]
        else:
            rep = mem_df.sort_values(fdr_col, ascending=True).index[0]
        for m in members:
            group_id_map[m] = gid
        rep_set.add(rep)

    sig["group_id"]   = sig.index.map(group_id_map)
    sig["group_size"] = sig["group_id"].map(
        lambda g: sum(1 for v in group_id_map.values() if v == g)
    )
    sig["group_rep"]  = sig.index.isin(rep_set)
    sig = sig.drop(columns=["_lead"])

    # 非显著的 term 也保留, group_id NaN
    nonsig = work.loc[~work.index.isin(sig.index)].copy()
    nonsig["group_id"] = np.nan
    nonsig["group_size"] = 1
    nonsig["group_rep"] = False

    out = pd.concat([sig, nonsig]).sort_index()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="Jaccard threshold for merging (paper: 0.8)")
    ap.add_argument("--fdr", type=float, default=0.25,
                    help="FDR cutoff for terms considered for merging")
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not ENRICH_DIR.exists():
        print("Run 07_enrichment_perKO.py first."); sys.exit(1)
    subsets = args.subsets or sorted([d.name for d in ENRICH_DIR.iterdir() if d.is_dir()])

    n_total = 0
    for subset in subsets:
        sdir = ENRICH_DIR / subset
        files = sorted(sdir.glob("*_gsea.csv"))
        if not files:
            continue
        print(f"[{subset}] {len(files)} GSEA files")
        for f in tqdm(files, desc=subset):
            out = f.with_name(f.name.replace("_gsea.csv", "_gsea_dedup.csv"))
            if out.exists() and not args.force:
                continue
            try:
                df = pd.read_csv(f)
                if df.empty:
                    continue
                dedup = collapse_one(df, args.threshold, args.fdr)
                dedup.to_csv(out, index=False)
                n_total += 1
            except Exception as e:
                print(f"  [{f.name}] failed: {e}", file=sys.stderr)
    print(f"\nDedup CSVs written for {n_total} (subset, E3) pairs.")


if __name__ == "__main__":
    main()