#!/usr/bin/env python
"""
07_enrichment_perKO_fixed.py — Per-(subset, E3) 通路富集
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from _config import (
    DOWNSTREAM_DIR, ENRICH_DIR, FDR_STRICT,
)

try:
    import gseapy as gp
except ImportError:
    print("Need: pip install gseapy", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# 默认 gene set libraries 路径
# ---------------------------------------------------------------------------
GMT_DIR = Path("/root/autodl-tmp/E3VK/data/genesets")

DEFAULT_LIBS = [
    str(GMT_DIR / "Reactome_Pathways_2024.gmt"),
    str(GMT_DIR / "KEGG_2021_Human.gmt"),
    str(GMT_DIR / "GO_Biological_Process_2023.gmt"),
    str(GMT_DIR / "GO_Molecular_Function_2023.gmt"),
    str(GMT_DIR / "GO_Cellular_Component_2023.gmt"),
    str(GMT_DIR / "WikiPathways_2024_Human.gmt"),
    str(GMT_DIR / "MSigDB_Hallmark_2020.gmt"),
    str(GMT_DIR / "BioPlanet_2019.gmt"),
    str(GMT_DIR / "PanglaoDB_Augmented_2021.gmt"),
    str(GMT_DIR / "Human_Phenotype_Ontology.gmt"),
]


def load_gmt_robust(gmt_paths: list[str]) -> dict:
    """
    自建 GMT 读取器，解决 utf-8 解码失败问题。
    将所有库合并为一个字典，并在 Term 前缀加上库的名称以便区分。
    """
    combined_gene_sets = {}
    for path in gmt_paths:
        p = Path(path)
        if not p.exists():
            print(f"  [Warning] Missing library: {p.name}", file=sys.stderr)
            continue
        
        lib_name = p.stem
        # 使用 errors='replace' 忽略并替换无法识别的字符 (如 0xfc)
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    # 格式化 term 名称：库名__原本的Term，例如 KEGG_2021__Glycolysis
                    term = f"{lib_name}__{parts[0]}"
                    genes = parts[2:]
                    combined_gene_sets[term] = genes
                    
    return combined_gene_sets


def boxcox_zscore(distance: pd.Series) -> pd.Series:
    # 1. 强制转换为 float64，防止 float32 计算平方时溢出
    d = distance.astype(np.float64).copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d.groupby(d.index).mean()
    
    # 2. 截断极端的离群值 (Winsorization) 
    # 把大于 99.9% 分位数的异常高值拉回到 99.9% 的水平，防止 Box-Cox 爆炸
    upper_limit = d.quantile(0.999)
    if pd.notna(upper_limit):
        d = d.clip(upper=upper_limit)
    
    if (d <= 0).any():
        pos_min = d[d > 0].min() if (d > 0).any() else 1e-8
        d = d + max(1e-8, pos_min * 1e-3)
        
    # 3. 执行 Box-Cox 转换
    bx, _ = stats.boxcox(d.values)
    
    # 4. 拦截处理后可能产生的 inf 或 -inf
    # 将它们替换为当前数组中的最大/最小有效值
    if np.isinf(bx).any():
        valid_max = np.nanmax(bx[np.isfinite(bx)]) if np.any(np.isfinite(bx)) else 1.0
        valid_min = np.nanmin(bx[np.isfinite(bx)]) if np.any(np.isfinite(bx)) else -1.0
        bx = np.clip(bx, valid_min, valid_max)
        
    sd = bx.std(ddof=1)
    
    # 5. 安全计算 Z-score
    if sd == 0 or np.isnan(sd):
        z = np.zeros_like(bx)
    else:
        z = (bx - bx.mean()) / sd
        
    # 添加极微小的随机噪声，打破 tie
    np.random.seed(42)
    noise = np.random.normal(0, 1e-8, size=len(z))
    z = z + noise
        
    return pd.Series(z, index=d.index, name=distance.name)


# 删除了原脚本中重复的 run_one 函数，保留参数最全的版本
def run_one(rank: pd.Series, out_path: Path, gene_sets_dict: dict,
            min_size: int = 10, max_size: int = 500,
            permutation_num: int = 1000, threads: int = 1) -> pd.DataFrame:
    rnk = rank.reset_index()
    rnk.columns = ["gene", "metric"]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 直接传入处理好的字典，避开 gseapy 读取文件时的编码崩溃
        res = gp.prerank(
            rnk=rnk, 
            gene_sets=gene_sets_dict, 
            outdir=None,
            min_size=min_size, 
            max_size=max_size,
            permutation_num=permutation_num, 
            seed=42,
            threads=threads, 
            verbose=False,
        )
    
    df = res.res2d
    df.to_csv(out_path, index=False)
    return df
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-per-subset", type=int, default=30)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--use-latest", action="store_true")
    ap.add_argument("--libs", nargs="+", default=None)
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--min-sig", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tier1", default=None, metavar="CSV",
                    help="Tier1 master table CSV with subset,ko columns; overrides --top-per-subset")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    # Tier1 filter: (subset, ko) pairs if --tier1 provided
    tier1_pairs: set = set()
    if args.tier1:
        import pandas as _pd
        _t = _pd.read_csv(args.tier1)
        tier1_pairs = set(zip(_t["subset"], _t["ko"]))
        print(f"[07_enrichment_perKO.py] Tier1 filter: {len(tier1_pairs)} (subset, ko) pairs")

 
    libs = args.libs or DEFAULT_LIBS
    print(f"Loading {len(libs)} libraries safely into memory...")
    
    # 提前一次性安全加载所有 GMT 到内存字典
    custom_gene_sets = load_gmt_robust(libs)
    print(f"Total functional terms loaded: {len(custom_gene_sets)}")
 
    vuln = pd.read_csv(DOWNSTREAM_DIR / "vulnerability.csv")
    subsets = args.subsets or sorted(vuln["subset"].unique())
 
    for subset in subsets:
        zmat_path = DOWNSTREAM_DIR / f"dr_zmat_{subset}.parquet"
        if not zmat_path.exists():
            continue
        zmat = pd.read_parquet(zmat_path)
 
        long_path = DOWNSTREAM_DIR / "dr_long.parquet"
        dist_mat = None
        if long_path.exists():
            try:
                ll = pd.read_parquet(long_path,
                                     filters=[("subset", "==", subset)],
                                     columns=["gene", "ko", "score"])
                if not ll.empty and ll["score"].notna().any():
                    dist_mat = ll.pivot_table(index="gene", columns="ko",
                                              values="score", aggfunc="first")
            except Exception:
                pass
                
        if dist_mat is None:
            print(f"  [{subset}] using |Z| as distance proxy")
            dist_mat = zmat.abs()
 
        sub_vuln = vuln[vuln["subset"] == subset].sort_values("n_sig_05", ascending=False)
        sub_vuln = sub_vuln[sub_vuln["n_sig_05"] >= args.min_sig]
        if tier1_pairs:
            sub_vuln = sub_vuln[sub_vuln["ko"].apply(lambda k: (subset, k) in tier1_pairs)]
        elif not args.all and args.top_per_subset > 0:
            sub_vuln = sub_vuln.head(args.top_per_subset)
        if sub_vuln.empty:
            continue
 
        outdir = ENRICH_DIR / subset
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"[{subset}] {len(sub_vuln)} KOs (Box-Cox + Z ranking)")
 
        for _, row in tqdm(sub_vuln.iterrows(), total=len(sub_vuln), desc=subset):
            ko = row["ko"]
            out = outdir / f"{ko}_gsea.csv"
            
            if out.exists() and not args.force:
                continue
            if ko not in dist_mat.columns:
                continue
                
            try:
                rank = boxcox_zscore(dist_mat[ko].dropna()).sort_values(ascending=False)
                run_one(rank, out, custom_gene_sets, threads=args.threads)
            except Exception as e:
                print(f"\n  [{subset}/{ko}] failed: {e}", file=sys.stderr)
 
    print("Done. Results in", ENRICH_DIR)
 
if __name__ == "__main__":
    main()