#!/usr/bin/env python
"""
11_egocentric_plot.py — Egocentric plots: KO gene at center, sig DR genes around

完全对应 Osorio et al. 2022 Fig 3A/B/C, Fig 4A/B:
  "The egocentric plot (right) shows the connections between the KO gene
   and significant virtual KO perturbed genes (FDR <0.05). Nodes are
   color-coded by each gene's membership association with enriched
   functional groups, as reported in the Enrichr analysis."

实现:
  - 以 E3 KO 为中心, 显著 DR 基因 (p_adj<FDR) 为外环
  - 边宽编码 |Z|; 边颜色 sign(FC-1) (上调红/下调蓝)
  - 节点颜色按 Enrichr 富集到的功能群 (取 top-K 互不重叠功能群)
  - 一张图一个 (subset, E3); 只对 top hits 跑（默认每亚群前 10）

输出:
  results/AIHA/downstream/figs/egocentric/<subset>/<E3>.pdf
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from _config import DOWNSTREAM_DIR, FIGS_DIR, FDR_STRICT, set_mpl_style

set_mpl_style()
EGO_DIR = FIGS_DIR / "egocentric"
EGO_DIR.mkdir(parents=True, exist_ok=True)

# 7 个高对比度颜色（避开红蓝 — 留给上/下调边）
PALETTE = ["#7F77DD", "#1D9E75", "#EF9F27", "#D85A30", "#D4537E", "#634CB2", "#73726c"]


def get_function_groups(enrichr_csv: Path, top_k: int = 6,
                        fdr_cut: float = 0.05) -> dict[str, str]:
    """Return {gene -> function_group_label}. Pick top-K terms by FDR."""
    if not enrichr_csv.exists():
        return {}
    try:
        df = pd.read_csv(enrichr_csv)
    except Exception:
        return {}
    if df.empty or "Adjusted P-value" not in df.columns:
        return {}
    df = df[df["Adjusted P-value"] < fdr_cut].copy()
    if df.empty:
        return {}
    # 取 top-K terms, 不区分 library (论文一致做法)
    df = df.sort_values("Adjusted P-value").head(top_k)
    gene2grp = {}
    for i, (_, row) in enumerate(df.iterrows()):
        term = str(row.get("Term", ""))[:40]
        # gseapy Enrichr 用 ';' 分隔
        for sep in (";", ",", "|"):
            if sep in str(row.get("Genes", "")):
                gs = [g.strip().upper() for g in str(row["Genes"]).split(sep) if g.strip()]
                break
        else:
            gs = [str(row.get("Genes", "")).strip().upper()]
        for g in gs:
            # 一个基因可能属于多功能, 先到先得
            gene2grp.setdefault(g, (i, term))
    return gene2grp


def plot_one(ko: str, sig: pd.DataFrame, gene2grp: dict, out: Path,
             max_nodes: int = 80):
    """Polar layout: KO center, sig DR genes on a circle, edges to center."""
    if sig.empty:
        return
    sig = sig.sort_values("z_abs", ascending=False).head(max_nodes).reset_index(drop=True)
    n = len(sig)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r_outer = 1.0
    xs = r_outer * np.cos(angles)
    ys = r_outer * np.sin(angles)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.axis("off")

    # 边
    z_max = sig["z_abs"].max() if sig["z_abs"].max() > 0 else 1
    for i, row in sig.iterrows():
        w = 0.3 + 1.7 * (row["z_abs"] / z_max)
        # 边颜色: 上调红 / 下调蓝（用 sign of FC-1; FC 缺则灰）
        fc = row.get("FC", np.nan)
        if pd.notna(fc):
            color = "#A32D2D" if fc > 1 else "#185FA5"
        else:
            color = "#888780"
        ax.plot([0, xs[i]], [0, ys[i]], color=color, lw=w, alpha=0.55, zorder=1)

    # 节点
    legend_labels: dict[int, str] = {}
    for i, row in sig.iterrows():
        g = str(row["gene"]).upper()
        if g in gene2grp:
            gid, term = gene2grp[g]
            color = PALETTE[gid % len(PALETTE)]
            legend_labels[gid] = term
        else:
            color = "#D3D1C7"
        ax.add_patch(Circle((xs[i], ys[i]), 0.05, color=color,
                            ec="white", lw=0.5, zorder=2))
        # 文本（远离中心）
        rot = np.degrees(angles[i])
        ha = "left" if -90 < rot < 90 else "right"
        if ha == "right":
            rot += 180
        ax.text(xs[i] * 1.10, ys[i] * 1.10, row["gene"],
                fontsize=7, rotation=rot, ha=ha, va="center")
    # 中心
    ax.add_patch(Circle((0, 0), 0.10, color="#26215C",
                        ec="white", lw=1, zorder=3))
    ax.text(0, 0, ko, color="white", fontsize=9, weight="bold",
            ha="center", va="center", zorder=4)

    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_title(f"{ko} KO — {n} significant DR genes (top {max_nodes})", pad=8)

    # 图例: 功能群颜色
    if legend_labels:
        handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=PALETTE[gid % len(PALETTE)],
                          markersize=7, label=term)
                   for gid, term in sorted(legend_labels.items())]
        # 边方向图例
        handles += [Line2D([0], [0], color="#A32D2D", lw=2, label="up (FC>1)"),
                    Line2D([0], [0], color="#185FA5", lw=2, label="down (FC<1)")]
        ax.legend(handles=handles, loc="lower center",
                  bbox_to_anchor=(0.5, -0.05), ncol=2,
                  fontsize=6, frameon=False)
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-per-subset", type=int, default=10)
    ap.add_argument("--max-nodes", type=int, default=80,
                    help="超过此数只画 top-|Z|")
    ap.add_argument("--fdr", type=float, default=FDR_STRICT)
    ap.add_argument("--subsets", nargs="+", default=None)
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
        print(f"[11_egocentric_plot.py] Tier1 filter: {len(tier1_pairs)} (subset, ko) pairs")


    vuln = pd.read_csv(DOWNSTREAM_DIR / "vulnerability.csv")
    enrichr_dir = DOWNSTREAM_DIR / "enrichr"
    subsets = args.subsets or sorted(vuln["subset"].unique())

    for subset in subsets:
        padj_path = DOWNSTREAM_DIR / f"dr_padj_{subset}.parquet"
        zmat_path = DOWNSTREAM_DIR / f"dr_zmat_{subset}.parquet"
        if not (padj_path.exists() and zmat_path.exists()):
            continue
        padj = pd.read_parquet(padj_path)
        zmat = pd.read_parquet(zmat_path)

        # 获取 FC 矩阵（可选）
        fcmat = None
        long_path = DOWNSTREAM_DIR / "dr_long.parquet"
        if long_path.exists():
            try:
                ll = pd.read_parquet(long_path,
                                     filters=[("subset", "==", subset)],
                                     columns=["gene", "ko", "FC"])
                if not ll.empty:
                    fcmat = ll.pivot_table(index="gene", columns="ko",
                                           values="FC", aggfunc="first")
            except Exception:
                pass

        sub_vuln = vuln[vuln["subset"] == subset].sort_values("n_sig_05", ascending=False)
        if tier1_pairs:
            sub_vuln = sub_vuln[sub_vuln["ko"].apply(lambda k: (subset, k) in tier1_pairs)]
        else:
            sub_vuln = sub_vuln.head(args.top_per_subset)
        outdir = EGO_DIR / subset
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"[{subset}] plotting top {len(sub_vuln)} E3 KOs")

        for _, row in sub_vuln.iterrows():
            ko = row["ko"]
            out = outdir / f"{ko}.pdf"
            if out.exists() and not args.force:
                continue
            mask = padj[ko] < args.fdr
            if mask.sum() == 0:
                continue
            sig = pd.DataFrame({
                "gene":  padj.index[mask].astype(str),
                "z_abs": np.abs(zmat.loc[mask, ko].values),
            })
            if fcmat is not None and ko in fcmat.columns:
                sig["FC"] = fcmat.loc[mask, ko].values
            else:
                sig["FC"] = np.nan
            enrichr_csv = enrichr_dir / subset / f"{ko}_enrichr.csv"
            gene2grp = get_function_groups(enrichr_csv)
            plot_one(ko, sig, gene2grp, out, max_nodes=args.max_nodes)
        print(f"  → {outdir}/")


if __name__ == "__main__":
    main()