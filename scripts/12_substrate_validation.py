#!/usr/bin/env python
"""
09_substrate_validation.py — E3 底物释放验证 (UbiBrowser 2.0 + UbiNet 2.0)

逻辑:
  KO 一个 E3 应当导致其底物在转录后稳定性变化。虽然 scRNA-seq 看不到蛋白水平，
  但底物 mRNA 的网络邻居关系 (cofactor / 反馈) 在 KO 后应被显著扰动。
  → 检验「该 E3 的已知/预测底物」是否在 DR 显著基因 + 高 Z 排名中富集。

两种检验:
  1. Hypergeometric: 显著 DR 基因 (p_adj<0.05) 与 UbiBrowser 底物集的 overlap
  2. GSEA-style: |Z| 排名是否在底物集上富集 (Mann-Whitney U)

参考数据库:
  - UbiBrowser 2.0:  http://ubibrowser.bio-it.cn/ubibrowser_v3/  (download page)
                     known + predicted E3-substrate interactions, human only
  - UbiNet 2.0:      http://awi.cuhk.edu.cn/~ubinet/index.php   (verified ESIs)

数据下载方式:
  - 推荐手动从两个网站下载到 data/refs/ubibrowser_known.tsv / predicted.tsv
  - 若提供 --download-refs，脚本会尝试自动下载（URL 可能随时间变化）

输入:
  data/refs/ubibrowser_known.tsv         (列: E3, Substrate, ...)
  data/refs/ubibrowser_predicted.tsv     (可选, 量大)
  data/refs/ubinet_known.tsv             (可选)
输出:
  results/AIHA/downstream/substrate_validation.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _config import (
    DOWNSTREAM_DIR, REFS_DIR, FDR_STRICT,
)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
UBIBROWSER_KNOWN = REFS_DIR / "ubibrowser_known.tsv"
UBIBROWSER_PRED  = REFS_DIR / "ubibrowser_predicted.tsv"
UBINET_KNOWN     = REFS_DIR / "ubinet_known.tsv"


def maybe_download():
    """打印下载指南；自动抓取 URL 容易失效，推荐手动。"""
    msg = f"""
请手动下载下列文件到 {REFS_DIR}/ 后再跑本脚本:

1. UbiBrowser 2.0 known E3-substrate interactions (human):
     http://ubibrowser.bio-it.cn/ubibrowser_v3/home/download
     选 "Known" + species=Homo sapiens → 保存为 ubibrowser_known.tsv
     (TSV，列含 SwissProt AC / Gene name 等)

2. (可选) UbiBrowser 2.0 predicted (high confidence):
     同一页面 "Predicted (high confidence)" → ubibrowser_predicted.tsv

3. (可选) UbiNet 2.0 verified ESIs:
     http://awi.cuhk.edu.cn/~ubinet/index.php → Download → ubinet_known.tsv

如果列名与脚本默认不一致, 用 --e3-col / --sub-col 指定。
"""
    print(msg)


def load_substrate_table(path: Path,
                         e3_col_candidates=("E3", "E3_Gene", "Gene_Symbol_E3"),
                         sub_col_candidates=("Substrate", "Sub_Gene", "Gene_Symbol_Sub"),
                         ) -> dict[str, set[str]]:
    """读 (E3, Substrate) 关系表，返回 {E3: {sub1, sub2, ...}}。"""
    if not path.exists():
        return {}
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)
    cols_lower = {c.lower(): c for c in df.columns}
    e3_col = next((cols_lower[c.lower()] for c in e3_col_candidates if c.lower() in cols_lower), None)
    sub_col = next((cols_lower[c.lower()] for c in sub_col_candidates if c.lower() in cols_lower), None)
    if e3_col is None or sub_col is None:
        print(f"  [warn] {path.name}: cannot find E3/Substrate cols among {list(df.columns)}")
        return {}
    df = df[[e3_col, sub_col]].dropna()
    df[e3_col] = df[e3_col].astype(str).str.upper()
    df[sub_col] = df[sub_col].astype(str).str.upper()
    out = df.groupby(e3_col)[sub_col].apply(lambda s: set(s.unique())).to_dict()
    print(f"  {path.name}: {len(df)} interactions, {len(out)} E3 with ≥1 substrate")
    return out


# ---------------------------------------------------------------------------
# 检验
# ---------------------------------------------------------------------------
def hypergeom_test(sig_genes: set[str], substrates: set[str], universe: set[str]):
    """单尾上侧 hypergeometric: P(X >= overlap)."""
    universe = universe & (sig_genes | substrates | universe)
    M = len(universe)
    n = len(substrates & universe)     # 底物在 universe 中的总数
    N = len(sig_genes & universe)      # sig 基因数
    k = len(sig_genes & substrates & universe)
    if M == 0 or n == 0 or N == 0:
        return np.nan, k, n, N
    pval = stats.hypergeom.sf(k - 1, M, n, N)
    return float(pval), int(k), int(n), int(N)


def ranksum_test(zvec: pd.Series, substrates: set[str]):
    """|Z| 在底物 vs 非底物的 Mann-Whitney."""
    universe = set(zvec.index.astype(str).str.upper())
    sub_in = substrates & universe
    if len(sub_in) < 5:
        return np.nan, len(sub_in)
    zvec_up = zvec.copy()
    zvec_up.index = zvec_up.index.astype(str).str.upper()
    a = np.abs(zvec_up.loc[list(sub_in)].dropna().values)
    b = np.abs(zvec_up.drop(list(sub_in), errors="ignore").dropna().values)
    if len(a) < 5 or len(b) < 5:
        return np.nan, len(sub_in)
    _, p = stats.mannwhitneyu(a, b, alternative="greater")
    return float(p), len(sub_in)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-refs", action="store_true")
    ap.add_argument("--fdr", type=float, default=FDR_STRICT)
    ap.add_argument("--use-predicted", action="store_true",
                    help="也用 UbiBrowser predicted 集（敏感度高，特异性低）")
    args = ap.parse_args()

    if args.download_refs:
        maybe_download(); sys.exit(0)

    # 加载底物表
    print("Loading substrate references...")
    known = load_substrate_table(UBIBROWSER_KNOWN)
    pred  = load_substrate_table(UBIBROWSER_PRED) if args.use_predicted else {}
    ubinet = load_substrate_table(UBINET_KNOWN)

    # 合并：known 优先，UbiNet 作为补充
    all_known: dict[str, set[str]] = {}
    for table in (known, ubinet):
        for k, v in table.items():
            all_known.setdefault(k, set()).update(v)

    if not all_known and not pred:
        print("No substrate reference loaded. Run --download-refs for instructions.")
        sys.exit(1)
    print(f"Total known-substrate E3: {len(all_known)}")
    if pred:
        print(f"Predicted-substrate E3: {len(pred)}")

    # 跑每个亚群
    subsets = sorted(p.stem.removeprefix("dr_padj_")
                     for p in DOWNSTREAM_DIR.glob("dr_padj_*.parquet"))
    results = []
    for subset in subsets:
        padj = pd.read_parquet(DOWNSTREAM_DIR / f"dr_padj_{subset}.parquet")
        zmat = pd.read_parquet(DOWNSTREAM_DIR / f"dr_zmat_{subset}.parquet")
        universe = set(padj.index.astype(str).str.upper())

        for ko in tqdm(zmat.columns, desc=subset):
            ko_u = ko.upper()
            ko_subs_known = all_known.get(ko_u, set())
            ko_subs_pred  = pred.get(ko_u, set()) if pred else set()
            if not (ko_subs_known or ko_subs_pred):
                continue

            sig = set(padj.index[padj[ko] < args.fdr].astype(str).str.upper())

            for label, subs in [("known", ko_subs_known), ("predicted", ko_subs_pred)]:
                if not subs:
                    continue
                hp, k_obs, n_sub, n_sig = hypergeom_test(sig, subs, universe)
                rp, n_sub_in = ranksum_test(zmat[ko], subs)
                results.append({
                    "subset": subset, "ko": ko, "ref": label,
                    "n_substrates":      len(subs),
                    "n_substrates_in_universe": n_sub,
                    "overlap_with_sig":  k_obs,
                    "n_sig":             n_sig,
                    "hyper_p":           hp,
                    "ranksum_p":         rp,
                })

    if not results:
        print("No (subset, E3) had any known/predicted substrates in the universe.")
        sys.exit(0)

    df = pd.DataFrame(results)
    # BH 校正
    from scipy.stats import false_discovery_control
    for col in ("hyper_p", "ranksum_p"):
        ok = df[col].notna()
        df[f"{col}_adj"] = np.nan
        if ok.sum():
            df.loc[ok, f"{col}_adj"] = false_discovery_control(
                np.clip(df.loc[ok, col].values, 1e-300, 1)
            )
    out = DOWNSTREAM_DIR / "substrate_validation.csv"
    df.sort_values(["subset", "ko", "ref"]).to_csv(out, index=False)
    print(f"\nWrote {out} ({len(df)} rows)")
    sig_rows = df[(df["hyper_p_adj"] < 0.1) | (df["ranksum_p_adj"] < 0.1)]
    print(f"  {len(sig_rows)} (subset, E3, ref) with FDR<0.1 in either test")


if __name__ == "__main__":
    main()