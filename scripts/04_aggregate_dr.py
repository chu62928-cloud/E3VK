#!/usr/bin/env python
"""
04_aggregate_dr.py — 汇总所有 scTenifoldKnk DR 结果

输入: results/AIHA/knk/<subset>/DR_<E3>.csv
输出:
  - results/AIHA/downstream/dr_long.parquet
        长表，列: subset, ko, gene, score, FC, T, Z, p_value, p_adj
        ~ N_subset × N_E3 × N_gene_per_KO 行，~ 千万级
  - results/AIHA/downstream/dr_zmat_<subset>.parquet
        每个亚群一份 gene × E3 的 Z 值矩阵，加载快，给 06/07 用
  - results/AIHA/downstream/dr_padj_<subset>.parquet
        同上但是 p_adj 矩阵

设计要点:
  - 只读取以 <subset>_knk.pkl 标记完成的亚群（与 README 一致）
  - 增量友好: 已存在的 dr_zmat_<subset>.parquet 默认跳过（除非 --force）
  - 用 float32 + zstd 压缩，能省一半空间
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _config import (
    KNK_DIR, DOWNSTREAM_DIR, list_completed_subsets,
    DR_COL_GENE, DR_COL_SCORE, DR_COL_FC, DR_COL_T, DR_COL_Z,
    DR_COL_P, DR_COL_PADJ, DR_COL_KO,
)


REQUIRED_COLS = [DR_COL_GENE, DR_COL_Z, DR_COL_PADJ]
NUMERIC_COLS  = [DR_COL_SCORE, DR_COL_FC, DR_COL_T, DR_COL_Z, DR_COL_P, DR_COL_PADJ]


def load_one_dr(csv_path: Path) -> pd.DataFrame | None:
    """Load a single DR_<E3>.csv, return None if malformed."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [skip] {csv_path.name}: {e}", file=sys.stderr)
        return None
    if not all(c in df.columns for c in REQUIRED_COLS):
        print(f"  [skip] {csv_path.name}: missing required cols", file=sys.stderr)
        return None
    # ko 列从文件名兜底（不依赖 CSV 内部的 ko 列，因为某些版本可能缺失）
    ko = csv_path.stem.removeprefix("DR_")
    if DR_COL_KO not in df.columns:
        df[DR_COL_KO] = ko
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df[DR_COL_GENE] = df[DR_COL_GENE].astype(str)
    df[DR_COL_KO]   = df[DR_COL_KO].astype(str)
    return df


def aggregate_subset(subset: str, force: bool = False) -> pd.DataFrame:
    """
    Load all DR_*.csv under one subset directory, return concatenated long-format df.
    Also writes dr_zmat_<subset>.parquet and dr_padj_<subset>.parquet.
    """
    sub_dir = KNK_DIR / subset
    csvs = sorted(sub_dir.glob("DR_*.csv"))
    if not csvs:
        print(f"[{subset}] no DR_*.csv found")
        return pd.DataFrame()

    zmat_path    = DOWNSTREAM_DIR / f"dr_zmat_{subset}.parquet"
    padj_path    = DOWNSTREAM_DIR / f"dr_padj_{subset}.parquet"
    long_chunks  = []
    z_cols       = {}
    padj_cols    = {}

    for csv in tqdm(csvs, desc=f"[{subset}]", leave=False):
        df = load_one_dr(csv)
        if df is None:
            continue
        df["subset"] = subset
        long_chunks.append(df)
        ko = csv.stem.removeprefix("DR_")
        z_cols[ko]    = df.set_index(DR_COL_GENE)[DR_COL_Z]
        padj_cols[ko] = df.set_index(DR_COL_GENE)[DR_COL_PADJ]

    if not long_chunks:
        return pd.DataFrame()

    long = pd.concat(long_chunks, ignore_index=True)
    print(f"[{subset}] {len(csvs)} KOs × ~{long[DR_COL_GENE].nunique():,} genes = {len(long):,} rows")

    # gene × E3 Z 矩阵
    zmat = pd.DataFrame(z_cols).astype("float32")
    padj = pd.DataFrame(padj_cols).astype("float32")
    # 排序列名让矩阵稳定
    zmat = zmat.reindex(columns=sorted(zmat.columns))
    padj = padj.reindex(columns=sorted(padj.columns))
    zmat.to_parquet(zmat_path, compression="zstd")
    padj.to_parquet(padj_path, compression="zstd")
    print(f"        wrote {zmat_path.name} {zmat.shape} and {padj_path.name}")
    return long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing per-subset parquet")
    ap.add_argument("--subsets", nargs="+", default=None,
                    help="restrict to these subsets (default: all completed)")
    args = ap.parse_args()
    print(f"DEBUG: KNK_DIR 实际指向的路径是 -> {KNK_DIR}")
    print(f"DEBUG: 该路径是否存在 -> {KNK_DIR.exists()}")
    subsets = args.subsets or list_completed_subsets()
    if not subsets:
        print("No completed subsets found. Check that <subset>_knk.pkl exists.")
        sys.exit(1)
    print(f"Found {len(subsets)} completed subsets: {subsets}")

    all_long = []
    for s in subsets:
        zmat_path = DOWNSTREAM_DIR / f"dr_zmat_{s}.parquet"
        if zmat_path.exists() and not args.force:
            # 若需要 long 表，仍要 load 一次；但跳过 zmat 重写
            pass
        long = aggregate_subset(s, force=args.force)
        if not long.empty:
            all_long.append(long)

    if not all_long:
        print("Nothing aggregated.")
        sys.exit(1)

    long = pd.concat(all_long, ignore_index=True)
    out = DOWNSTREAM_DIR / "dr_long.parquet"
    long.to_parquet(out, compression="zstd")
    print(f"\nWrote {out} ({len(long):,} rows, {long['subset'].nunique()} subsets, "
          f"{long[DR_COL_KO].nunique()} unique E3 KOs)")


if __name__ == "__main__":
    main()