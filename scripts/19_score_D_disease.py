#!/usr/bin/env python
"""
19_score_D_disease.py — Dimension D: disease relevance score

三个证据来源:
  D1. 文献白名单 (AIHA/自免病强先验 E3, 硬编码)
  D2. GWAS 自免病风险基因 (data/refs/gwas_autoimmune.csv, 可用 --fetch-gwas 拉取)
  D3. UbiBrowser 已知底物显著重叠 (12_substrate_validation.py 产出)

评分: score_D = (D1*3 + D2*2 + E2_family*1 + substrate_sig*1) / 7

输出:
  results/AIHA/downstream/scoreD_disease_relevance.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _config import DOWNSTREAM_DIR, REFS_DIR

# ---- 先验白名单 (来自文献综述 §1.1–1.3) ----
WHITELIST_AIHA: set[str] = {
    # CD4 Th 分化
    "ITCH",   "CBLB",  "CUL5",   "VHL",    "STUB1",
    "RNF128", "TRAF6", "CUL3",   "RBX1",
    # CD8
    "TRIM21", "PELI1",
    # B细胞 / 浆细胞 / AIHA
    "UBE2L3", "RNF31", "RBCK1",  "TRAF3",
    # aliases / synonyms
    "GRAIL",  "CHIP",  "HOIP",   "HOIL1",  "HOIL-1L",
}

# E2 家族（UBE2L3 本身已在白名单，这里做额外加分）
UBE2_E2_FAMILY: set[str] = {
    "UBE2L3", "UBE2N", "UBE2D3", "UBE2K", "UBE2I",
    "UBE2G1", "UBE2G2", "UBE2H", "UBE2J1", "UBE2J2",
    "UBE2R1", "UBE2R2", "UBE2E1", "UBE2E2", "UBE2E3",
}

# 关键转录因子底物（UbiBrowser overlap 的目标）
KEY_TF_SUBSTRATES: set[str] = {
    "FOXP3", "BCL6", "HIF1A", "STAT3", "GATA3", "TBX21",
    "RORC", "IRF4", "NFKB1", "RELA", "TP53", "MYC",
}

MAX_SCORE = 7.0


def fetch_gwas_genes(efo_ids: list[str]) -> set[str]:
    """Attempt OpenTargets GraphQL API; returns empty set on failure."""
    try:
        import requests
        QUERY = """
        query ($efoId: String!) {
          disease(efoId: $efoId) {
            associatedTargets(page: {index: 0, size: 500}) {
              rows { target { approvedSymbol } score }
            }
          }
        }
        """
        url = "https://api.platform.opentargets.org/api/v4/graphql"
        genes: set[str] = set()
        for eid in efo_ids:
            try:
                r = requests.post(url, json={"query": QUERY, "variables": {"efoId": eid}},
                                  timeout=15)
                data = r.json()
                rows = (data.get("data", {})
                            .get("disease", {}) or {})
                rows = (rows.get("associatedTargets", {}) or {}).get("rows", [])
                for row in rows:
                    sym = (row.get("target") or {}).get("approvedSymbol")
                    if sym:
                        genes.add(sym)
            except Exception as e:
                print(f"  [GWAS] {eid}: {e}", file=sys.stderr)
        return genes
    except ImportError:
        print("  [GWAS] requests not available", file=sys.stderr)
        return set()


EFO_IDS = [
    "EFO_0004245",   # AIHA
    "MONDO_0007915", # SLE
    "EFO_0001071",   # ITP
    "EFO_0000685",   # RA
    "MONDO_0005301", # MS
]


def main():
    ap = argparse.ArgumentParser(description="Score D: disease relevance")
    ap.add_argument("--fetch-gwas", action="store_true",
                    help="Attempt to fetch GWAS genes from OpenTargets API")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_csv = DOWNSTREAM_DIR / "scoreD_disease_relevance.csv"
    if out_csv.exists() and not args.force:
        print(f"Already exists: {out_csv}  (use --force to recompute)"); return

    # Load all known KOs from scoreA (or vulnerability.csv as fallback)
    score_a_path = DOWNSTREAM_DIR / "scoreA_high_impact.csv"
    vuln_path    = DOWNSTREAM_DIR / "vulnerability.csv"
    if score_a_path.exists():
        all_kos = sorted(pd.read_csv(score_a_path)["ko"].unique())
    elif vuln_path.exists():
        all_kos = sorted(pd.read_csv(vuln_path)["ko"].unique())
    else:
        print("Run 17_score_A_high_impact.py (or 06_vulnerability_map.py) first.")
        sys.exit(1)
    print(f"Scoring {len(all_kos)} unique E3s")

    # D2: GWAS genes
    gwas_csv = REFS_DIR / "gwas_autoimmune.csv"
    gwas_genes: set[str] = set()
    if gwas_csv.exists():
        gwas_df = pd.read_csv(gwas_csv)
        col = next((c for c in gwas_df.columns
                    if "gene" in c.lower() or "symbol" in c.lower()), gwas_df.columns[0])
        gwas_genes = set(gwas_df[col].dropna().astype(str))
        print(f"GWAS genes loaded: {len(gwas_genes)} from {gwas_csv.name}")
    elif args.fetch_gwas:
        print("Fetching GWAS genes from OpenTargets ...")
        gwas_genes = fetch_gwas_genes(EFO_IDS)
        if gwas_genes:
            df_g = pd.DataFrame({"gene_symbol": sorted(gwas_genes)})
            gwas_csv.parent.mkdir(parents=True, exist_ok=True)
            df_g.to_csv(gwas_csv, index=False)
            print(f"  Saved {len(gwas_genes)} genes -> {gwas_csv}")
        else:
            print("  No GWAS genes fetched; D2 will be 0")
    else:
        print(f"  {gwas_csv} not found. Run with --fetch-gwas or create manually.")
        print("  Writing placeholder gwas_autoimmune.csv ...")
        gwas_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["gene_symbol"]).to_csv(gwas_csv, index=False)

    # D3: UbiBrowser substrate overlap from 12_substrate_validation.py
    sub_val_path = DOWNSTREAM_DIR / "substrate_validation.csv"
    sub_sig: set[str] = set()
    if sub_val_path.exists():
        sv = pd.read_csv(sub_val_path)
        # hyper_p_adj < 0.05 counts as significant substrate overlap
        col_p = next((c for c in sv.columns if "hyper_p_adj" in c.lower()), None)
        col_ko = next((c for c in sv.columns if c.lower() in ("ko", "e3", "gene")), None)
        if col_p and col_ko:
            sub_sig = set(sv[sv[col_p] < 0.05][col_ko].dropna())
            print(f"Significant substrate overlap E3s: {len(sub_sig)}")
    else:
        print("  substrate_validation.csv not found; D3 will be 0 (run 12 first)")

    # Score
    rows = []
    for ko in all_kos:
        d1 = int(ko in WHITELIST_AIHA)
        d2 = int(ko in gwas_genes)
        d3 = int(ko in UBE2_E2_FAMILY)
        d4 = int(ko in sub_sig)
        raw = d1 * 3 + d2 * 2 + d3 * 1 + d4 * 1
        rows.append({
            "ko":                ko,
            "aiha_prior":        bool(d1),
            "gwas_autoimmune":   bool(d2),
            "is_E2":             bool(d3),
            "substrate_overlap_sig": bool(d4),
            "score_D_raw":       raw,
            "score_D":           raw,                     # unnormalized, for master_table
            "score_D_norm":      round(raw / MAX_SCORE, 4),
        })

    result = pd.DataFrame(rows)
    result.to_csv(out_csv, index=False)
    print(f"\nWrote {len(result)} rows -> {out_csv}")
    n_wl = result["aiha_prior"].sum()
    n_gw = result["gwas_autoimmune"].sum()
    print(f"  Whitelist hits: {n_wl} | GWAS hits: {n_gw} | E2 family: {result['is_E2'].sum()}")


if __name__ == "__main__":
    main()
