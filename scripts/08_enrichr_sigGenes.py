#!/usr/bin/env python
"""
08_enrichr_sigGenes_local.py — Enrichr on significant DR genes (Local GMT version)
"""
from __future__ import annotations
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _config import DOWNSTREAM_DIR, FDR_STRICT

try:
    import gseapy as gp
except ImportError:
    print("Need: pip install gseapy"); sys.exit(1)

ENRICHR_DIR = DOWNSTREAM_DIR / "enrichr"
ENRICHR_DIR.mkdir(exist_ok=True)

# 1. 新增：定义本地 GMT 文件夹路径
GMT_DIR = Path("/root/autodl-tmp/E3VK/data/genesets")

# ---------------------------------------------------------------------------
# 论文用到的所有库 + 类别标签 (供 summary 表分组)
# ---------------------------------------------------------------------------
LIB_GROUPS: dict[str, list[str]] = {
    "pathway":   ["KEGG_2021_Human", "Reactome_Pathways_2024", "BioPlanet_2019",
                  "WikiPathways_2024_Human", "MSigDB_Hallmark_2020"],
    "GO":        ["GO_Biological_Process_2023", "GO_Cellular_Component_2023",
                  "GO_Molecular_Function_2023"],
    "TF":        ["ChEA_2022", "ENCODE_TF_ChIP-seq_2015",
                  "TRRUST_Transcription_Factors_2019"],
    "phenotype": ["Human_Phenotype_Ontology",
                  "Disease_Perturbations_from_GEO_up",
                  "Disease_Perturbations_from_GEO_down"],
    "marker":    ["PanglaoDB_Augmented_2021", "CellMarker_Augmented_2021"],
}


# 2. 修改：让 run_enrichr 支持读取本地 GMT 文件，并合并结果
def run_enrichr(gene_list: list[str], libs: list[str], out: Path) -> pd.DataFrame:
    if len(gene_list) < 5:
        return pd.DataFrame()
    
    all_results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for lib in libs:
            gmt_path = GMT_DIR / f"{lib}.gmt"
            
            if not gmt_path.exists():
                print(f"\n  [Warning] Local GMT missing: {gmt_path}, skipping...", file=sys.stderr)
                continue
            
            try:
                # ==========================================
                # 核心修改：手动安全读取 GMT 文件，跳过/替换无法解码的字符
                # ==========================================
                custom_gene_sets = {}
                # 使用 errors="replace" 会把无法识别的 0xfc 替换成 ，保证程序不崩溃
                with open(gmt_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            term = parts[0]
                            genes = parts[2:] # 第一列是term，第二列是desc(通常可忽略)，后面全是genes
                            custom_gene_sets[term] = genes
                
                # 传入字典 custom_gene_sets 而不是 gmt_path 字符串
                e = gp.enrichr(
                    gene_list=gene_list, 
                    gene_sets=custom_gene_sets, 
                    outdir=None, 
                    no_plot=True, 
                    background=None,
                )
                
                if e is not None and e.results is not None and not e.results.empty:
                    df = e.results.copy()
                    df["Gene_set"] = lib 
                    all_results.append(df)
                    
            except Exception as ex:
                print(f"\n  [Error] Failed on local GMT {lib}: {ex}", file=sys.stderr)

    if not all_results:
        return pd.DataFrame()
    
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(out, index=False)
    return final_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-per-subset", type=int, default=30)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-sig", type=int, default=10)
    ap.add_argument("--fdr", type=float, default=FDR_STRICT)
    ap.add_argument("--groups", nargs="+",
                    default=list(LIB_GROUPS.keys()),
                    help="哪些库分组要跑 (默认全跑)")
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tier1", default=None, metavar="CSV",
                    help="Tier1 master table CSV with subset,ko columns; overrides --top-per-subset")
    # 因为变成了纯本地读取，去掉了 sleep 参数，不再需要限制 API 请求频率
    args = ap.parse_args()
    # Tier1 filter: (subset, ko) pairs if --tier1 provided
    tier1_pairs: set = set()
    if args.tier1:
        import pandas as _pd
        _t = _pd.read_csv(args.tier1)
        tier1_pairs = set(zip(_t["subset"], _t["ko"]))
        print(f"[08_enrichr_sigGenes.py] Tier1 filter: {len(tier1_pairs)} (subset, ko) pairs")


    libs = sum((LIB_GROUPS[g] for g in args.groups), [])
    print(f"Running {len(libs)} libraries across groups {args.groups}")

    vuln = pd.read_csv(DOWNSTREAM_DIR / "vulnerability.csv")
    subsets = args.subsets or sorted(vuln["subset"].unique())

    summary = []
    for subset in subsets:
        padj_path = DOWNSTREAM_DIR / f"dr_padj_{subset}.parquet"
        if not padj_path.exists():
            continue
        padj = pd.read_parquet(padj_path)

        sub_vuln = vuln[vuln["subset"] == subset].sort_values("n_sig_05", ascending=False)
        sub_vuln = sub_vuln[sub_vuln["n_sig_05"] >= args.min_sig]
        if tier1_pairs:
            sub_vuln = sub_vuln[sub_vuln["ko"].apply(lambda k: (subset, k) in tier1_pairs)]
        elif not args.all and args.top_per_subset > 0:
            sub_vuln = sub_vuln.head(args.top_per_subset)
        if sub_vuln.empty:
            continue

        outdir = ENRICHR_DIR / subset
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"[{subset}] {len(sub_vuln)} KOs")

        for _, row in tqdm(sub_vuln.iterrows(), total=len(sub_vuln), desc=subset):
            ko = row["ko"]
            out = outdir / f"{ko}_enrichr.csv"
            if out.exists() and not args.force:
                res = pd.read_csv(out)
            else:
                sig_genes = padj.index[padj[ko] < args.fdr].astype(str).tolist()
                try:
                    res = run_enrichr(sig_genes, libs, out)
                except Exception as e:
                    print(f"  [{subset}/{ko}] enrichr failed: {e}", file=sys.stderr)
                    res = pd.DataFrame()
                
                # 3. 去掉了 time.sleep(args.sleep)

            if res.empty:
                continue
            lib_col = "Gene_set" if "Gene_set" in res.columns else None
            for lib in libs:
                # 找该 lib 所属 group
                group = next((g for g, lst in LIB_GROUPS.items() if lib in lst), "other")
                if lib_col is None:
                    continue
                lib_res = res[res[lib_col] == lib]
                if lib_res.empty:
                    continue
                top = lib_res.sort_values("Adjusted P-value").head(3)
                for _, r in top.iterrows():
                    summary.append({
                        "subset": subset, "ko": ko,
                        "group": group, "library": lib,
                        "term": r.get("Term"),
                        "p_adj": r.get("Adjusted P-value"),
                        "odds": r.get("Odds Ratio"),
                        "overlap": r.get("Overlap"),
                    })

    if summary:
        sdf = pd.DataFrame(summary)
        sdf.to_csv(ENRICHR_DIR / "summary_top3_per_lib.csv", index=False)
        print(f"\nWrote summary_top3_per_lib.csv ({len(sdf)} rows)")


if __name__ == "__main__":
    main()