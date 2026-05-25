#!/usr/bin/env python
"""
10_string_validation.py — STRING PPI interaction enrichment

完全对应 Osorio et al. 2022, Result 3 反复出现:
  "The network of virtual KO perturbed genes had significantly more
   interactions than expected (p value < 0.01, STRING interaction enrichment
   test), which indicates that gene products exhibited more interactions
   among themselves than what would be expected for a random set of proteins
   of similar size, drawn from the genome."

为什么必须做:
  论文每个 KO 都用 STRING 的 "PPI enrichment p-value" 作 sanity check —
  若显著 DR 基因在 PPI 上不显著聚集, 该 KO 的结果可信度低. 这是过滤假阳性
  E3 hit 的关键步骤.

实现:
  使用 STRING 的 REST API (无需登录, 有限速):
    https://string-db.org/api/json/ppi_enrichment?identifiers=...&species=9606
  返回 JSON 含 'p_value' 字段.

输出:
  results/AIHA/downstream/string/<subset>_string_enrichment.csv
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _config import DOWNSTREAM_DIR, FDR_STRICT


STRING_URL  = "https://string-db.org/api/json/ppi_enrichment"
STRING_SPECIES_HUMAN = 9606
STRING_DIR  = DOWNSTREAM_DIR / "string"
STRING_DIR.mkdir(exist_ok=True)


def ppi_enrichment(genes: list[str], species: int = STRING_SPECIES_HUMAN,
                   timeout: int = 30) -> dict | None:
    """Call STRING API; return parsed dict or None on failure."""
    if len(genes) < 5:
        return None
    params = {
        "identifiers": "%0d".join(genes),
        "species": species,
        "caller_identity": "E3VK_pipeline",
    }
    try:
        r = requests.post(STRING_URL, data=params, timeout=timeout)
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        return {"error": str(e)}
    if isinstance(js, list) and js:
        return js[0]
    return js if isinstance(js, dict) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-per-subset", type=int, default=30)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-sig", type=int, default=10)
    ap.add_argument("--fdr", type=float, default=FDR_STRICT)
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="STRING 限速保护: 每次 API 调用间隔秒")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tier1", default=None, metavar="CSV",
                    help="Tier1 master table CSV with subset,ko columns; overrides --top-per-subset")
    args = ap.parse_args()
    # Tier1 filter: (subset, ko) pairs if --tier1 provided
    tier1_pairs: set = set()
    if args.tier1:
        import pandas as _pd
        _t = _pd.read_csv(args.tier1)
        tier1_pairs = set(zip(_t["subset"], _t["ko"]))
        print(f"[10_string_validation.py] Tier1 filter: {len(tier1_pairs)} (subset, ko) pairs")


    vuln = pd.read_csv(DOWNSTREAM_DIR / "vulnerability.csv")
    subsets = args.subsets or sorted(vuln["subset"].unique())

    for subset in subsets:
        padj_path = DOWNSTREAM_DIR / f"dr_padj_{subset}.parquet"
        if not padj_path.exists():
            continue
        out = STRING_DIR / f"{subset}_string_enrichment.csv"
        if out.exists() and not args.force:
            print(f"[{subset}] cached {out.name}, skip (use --force to redo)")
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

        rows = []
        for _, row in tqdm(sub_vuln.iterrows(), total=len(sub_vuln), desc=subset):
            ko = row["ko"]
            sig_genes = padj.index[padj[ko] < args.fdr].astype(str).tolist()
            r = ppi_enrichment(sig_genes)
            time.sleep(args.sleep)
            entry = {"subset": subset, "ko": ko, "n_sig": len(sig_genes)}
            if r and "error" not in r:
                entry.update({
                    "ppi_pvalue":      r.get("p_value"),
                    "n_nodes":         r.get("number_of_nodes"),
                    "n_edges":         r.get("number_of_edges"),
                    "n_edges_expected": r.get("expected_number_of_edges"),
                    "avg_node_degree":  r.get("average_node_degree"),
                    "avg_local_cc":     r.get("average_local_clustering_coefficient"),
                })
            else:
                entry["error"] = (r or {}).get("error", "unknown")
            rows.append(entry)

        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        n_sig = (pd.to_numeric(df.get("ppi_pvalue"), errors="coerce") < 0.01).sum()
        print(f"[{subset}] wrote {out.name} | PPI-enriched (p<0.01): {n_sig}/{len(df)}")


if __name__ == "__main__":
    main()