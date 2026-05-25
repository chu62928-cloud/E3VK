#!/usr/bin/env python
"""
21_master_visualize.py — Master ranking visualizations

出三张核心图:
  1. 主表概览   overview.pdf   — dot plot: x=subset, y=Tier1 E3, size=n_sig, color=composite
  2. A vs B 散点 A_vs_B_scatter.pdf — x=A_norm, y=B_norm, 白名单五角星, top10 标签
  3. 命运转移 Sankey  fate_sankey.html + fate_sankey.pdf（需要 plotly/kaleido）

输出:
  results/AIHA/downstream/figs/main/overview.pdf
  results/AIHA/downstream/figs/main/A_vs_B_scatter.pdf
  results/AIHA/downstream/figs/main/fate_sankey.html
  results/AIHA/downstream/figs/main/fate_sankey.pdf  (若 kaleido 可用)
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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from _config import DOWNSTREAM_DIR, FIGS_DIR, set_mpl_style

set_mpl_style()
OUT_DIR = FIGS_DIR / "main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_COLORS = {
    "RING":  "#4263eb",
    "HECT":  "#e67700",
    "RBR":   "#862e9c",
    "CRL":   "#2f9e44",
    "Other": "#868e96",
}
FAMILY_MARKERS = {
    "RING":  "o",
    "HECT":  "s",
    "RBR":   "^",
    "CRL":   "D",
    "Other": "x",
}

WHITELIST_AIHA: set[str] = {
    "ITCH", "CBLB", "CUL5", "VHL", "STUB1", "RNF128", "TRAF6", "CUL3", "RBX1",
    "TRIM21", "PELI1", "UBE2L3", "RNF31", "RBCK1", "TRAF3", "GRAIL", "HOIP", "HOIL1",
}


# ------------------------------------------------------------------ #
#  Figure 1 — Overview dot plot                                        #
# ------------------------------------------------------------------ #
def fig_overview(tier1: pd.DataFrame, out: Path) -> None:
    subsets = sorted(tier1["subset"].unique())
    # Select E3s: those in Tier1 in >= 1 subset, sorted by mean composite
    e3_order = (tier1.groupby("ko")["composite"].mean()
                .sort_values(ascending=False).index.tolist())

    if not e3_order:
        print("  [overview] No Tier1 E3 to plot"); return

    # Build grid
    n_e3, n_sub = len(e3_order), len(subsets)
    sub_idx  = {s: i for i, s in enumerate(subsets)}
    e3_idx   = {e: i for i, e in enumerate(e3_order)}

    # Size scale: n_sig -> dot radius
    n_sig_max = tier1["n_sig"].max() if "n_sig" in tier1.columns else 1
    size_scale = 800 / max(n_sig_max, 1)

    fig, ax = plt.subplots(figsize=(max(6, n_sub * 0.8 + 2),
                                    max(5, n_e3 * 0.35 + 2)))
    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=tier1["composite"].min(),
                             vmax=tier1["composite"].max())

    for _, row in tier1.iterrows():
        xi = sub_idx.get(row["subset"])
        yi = e3_idx.get(row["ko"])
        if xi is None or yi is None:
            continue
        fam   = row.get("family", "Other")
        color = cmap(norm(row["composite"]))
        size  = max(20, row.get("n_sig", 1) * size_scale)
        marker = FAMILY_MARKERS.get(fam, "o")
        ax.scatter(xi, yi, s=size, c=[color], marker=marker,
                   edgecolors="white", linewidths=0.3, zorder=2)
        if row["ko"] in WHITELIST_AIHA:
            ax.scatter(xi, yi, s=size * 2, facecolors="none",
                       edgecolors="#e03131", linewidths=1.5, zorder=3)

    ax.set_xticks(range(n_sub)); ax.set_xticklabels(subsets, rotation=45, ha="right")
    ax.set_yticks(range(n_e3));  ax.set_yticklabels(e3_order, fontsize=6)
    ax.set_xlabel("Cell subset"); ax.set_ylabel("E3 (Tier1)")
    ax.set_title("E3 KO Master Ranking — Tier 1", fontsize=11, fontweight="bold")

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.02)
    cb.set_label("Composite score")

    # Family legend
    handles = [Line2D([0], [0], marker=FAMILY_MARKERS.get(f, "o"),
                      color=FAMILY_COLORS.get(f, "grey"),
                      linestyle="None", markersize=6, label=f)
               for f in FAMILY_COLORS]
    handles.append(Line2D([0], [0], marker="o", color="none",
                          markeredgecolor="#e03131", markeredgewidth=1.5,
                          markersize=8, label="AIHA whitelist"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.12, 1),
              fontsize=7, frameon=True)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ------------------------------------------------------------------ #
#  Figure 2 — A vs B scatter                                           #
# ------------------------------------------------------------------ #
def fig_ab_scatter(df: pd.DataFrame, out: Path) -> None:
    # Per (ko), aggregate across subsets: mean A_norm, mean B_norm
    agg = df.groupby("ko").agg(
        A_norm=("A_norm", "mean"),
        B_norm=("B_norm", "mean"),
        composite=("composite", "mean"),
        family=("family", "first"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter non-whitelist
    for fam, grp in agg.groupby("family"):
        mask = ~grp["ko"].isin(WHITELIST_AIHA)
        ax.scatter(grp.loc[mask, "A_norm"], grp.loc[mask, "B_norm"],
                   s=grp.loc[mask, "composite"] * 60 + 10,
                   c=FAMILY_COLORS.get(fam, "#868e96"),
                   marker=FAMILY_MARKERS.get(fam, "o"),
                   alpha=0.6, label=fam, edgecolors="white", linewidths=0.3)
    # Whitelist on top (star)
    wl = agg[agg["ko"].isin(WHITELIST_AIHA)]
    ax.scatter(wl["A_norm"], wl["B_norm"],
               s=wl["composite"] * 80 + 30,
               c="#e03131", marker="*", zorder=5,
               edgecolors="white", linewidths=0.4, label="AIHA whitelist")

    # Top 10 composite labels
    top10 = agg.nlargest(10, "composite")
    for _, row in top10.iterrows():
        ax.annotate(row["ko"], (row["A_norm"], row["B_norm"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Score A (high impact, degree-corrected)")
    ax.set_ylabel("Score B (subset specificity)")
    ax.set_title("E3 KO: Impact vs Specificity", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, markerscale=0.8, bbox_to_anchor=(1, 1), loc="upper left")
    ax.axhline(0.5, color="grey", lw=0.5, linestyle=":")
    ax.axvline(0.5, color="grey", lw=0.5, linestyle=":")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ------------------------------------------------------------------ #
#  Figure 3 — Fate transition Sankey                                   #
# ------------------------------------------------------------------ #
def fig_sankey(score_c: pd.DataFrame, out_html: Path, out_pdf: Path,
               top_n: int = 20) -> None:
    if score_c.empty or "delta_flux" not in score_c.columns:
        print("  [sankey] No scoreC data; skipping Sankey"); return

    # Top N KOs by max |delta_flux|
    top_kos = (score_c.groupby("ko")["delta_flux"]
               .apply(lambda x: x.abs().max())
               .nlargest(top_n).index.tolist())
    sub = score_c[score_c["ko"].isin(top_kos)].copy()
    sub = sub[sub["delta_flux"].abs() > 0.001]

    if sub.empty:
        print("  [sankey] No significant delta_flux; skipping Sankey"); return

    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [sankey] plotly not available; skipping Sankey"); return

    # Build node list: (KO) → from_subset → to_subset
    kos_list = sorted(sub["ko"].unique())
    froms    = sorted(sub["from_subset"].unique())
    tos      = sorted(sub["to_subset"].unique())

    nodes = kos_list + froms + [t for t in tos if t not in froms]
    n_idx = {n: i for i, n in enumerate(nodes)}

    # Colors
    node_colors = []
    for n in nodes:
        if n in kos_list:
            node_colors.append("rgba(66,99,235,0.7)")
        elif n in froms:
            node_colors.append("rgba(224,103,49,0.7)")
        else:
            node_colors.append("rgba(47,158,68,0.7)")

    sources, targets, values, link_colors = [], [], [], []
    for _, row in sub.iterrows():
        ko_n  = row["ko"]
        from_n = row["from_subset"]
        to_n   = row["to_subset"]
        dflux  = abs(row["delta_flux"])
        # KO → from_subset
        sources.append(n_idx[ko_n]); targets.append(n_idx[from_n])
        values.append(dflux)
        link_colors.append("rgba(150,150,150,0.3)")
        # from_subset → to_subset
        sources.append(n_idx[from_n]); targets.append(n_idx[to_n])
        values.append(dflux)
        color = "rgba(224,49,49,0.5)" if row["delta_flux"] > 0 else "rgba(47,158,68,0.5)"
        link_colors.append(color)

    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, color=node_colors, pad=15, thickness=15),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
    ))
    fig.update_layout(
        title_text=f"CellOracle Fate Transition (Top {top_n} E3 KOs)",
        font_size=9, height=600,
    )
    fig.write_html(str(out_html))
    print(f"  Saved {out_html.name}")

    try:
        fig.write_image(str(out_pdf))
        print(f"  Saved {out_pdf.name}")
    except Exception as e:
        print(f"  [sankey] PDF export failed (kaleido?): {e}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="Master ranking visualizations (3 figures)")
    ap.add_argument("--top-sankey", type=int, default=20,
                    help="Top N KOs to include in Sankey diagram")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    master_path = DOWNSTREAM_DIR / "E3_master_ranking.csv"
    if not master_path.exists():
        print("Run 20_master_table.py first."); sys.exit(1)
    df = pd.read_csv(master_path)
    tier1 = df[df["tier"] == "Tier1"]

    # Fig 1
    fig_overview(tier1, OUT_DIR / "overview.pdf")

    # Fig 2
    fig_ab_scatter(df, OUT_DIR / "A_vs_B_scatter.pdf")

    # Fig 3
    c_path = DOWNSTREAM_DIR / "scoreC_fate_transition.csv"
    if c_path.exists():
        score_c = pd.read_csv(c_path)
    else:
        score_c = pd.DataFrame()
    fig_sankey(score_c,
               OUT_DIR / "fate_sankey.html",
               OUT_DIR / "fate_sankey.pdf",
               top_n=args.top_sankey)

    print(f"\nAll figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
