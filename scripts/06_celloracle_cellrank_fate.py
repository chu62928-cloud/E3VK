"""
06_celloracle_cellrank_fate.py
对 scTenifoldKnk top E3 hits，用 CellOracle 模拟 KO，
然后用 CellRank 2 量化 KO 前后细胞从 (subset_i → subset_j) 的转移概率变化。

修改说明（vs 初版）：
  1. 补 baseline（no perturbation）run — 缺失 baseline 时所有 post-KO flux 无参照
  2. 用 --lineage {T,B} 分开跑 T 细胞和 B 细胞（两系 UMAP 结构不同，不可合并）
  3. KO 候选只跑 top-only（scoreA residual_z>1.5 ∪ scoreB |spec_z|>2 ∪ AIHA 白名单）
  4. delta_flux = post_KO_flux - baseline_flux，保存至 results/AIHA/fate/<lineage>/

环境要求:
  conda activate env_fate
  (celloracle, cellrank, scvi-tools, scanpy, anndata)

运行示例:
  python scripts/06_celloracle_cellrank_fate.py --lineage T
  python scripts/06_celloracle_cellrank_fate.py --lineage B
  # 跑完两个 lineage 后，用 19a_aggregate_scoreC.py 汇总
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
import scanpy as sc
import anndata as ad

sys.path.insert(0, str(Path(__file__).parent))
from _config import RESULTS_DIR, DOWNSTREAM_DIR, FIGS_DIR

# ---- paths ----
SUBSETS_DIR = RESULTS_DIR / "subsets"
FATE_BASE   = RESULTS_DIR / "fate"
FIG_BASE    = FIGS_DIR / "fate"

WHITELIST_AIHA: set[str] = {
    "ITCH", "CBLB", "CUL5", "VHL", "STUB1", "RNF128", "TRAF6", "CUL3", "RBX1",
    "TRIM21", "PELI1", "UBE2L3", "RNF31", "RBCK1", "TRAF3", "GRAIL", "HOIP", "HOIL1",
}

LINEAGE_GLOB = {
    "T": "CD*.h5ad",
    "B": "B*.h5ad",
}

# 关注的转移（AIHA 相关）
INTERESTING_TRANSITIONS = [
    ("CD4_Th17",      "CD4_Treg"),
    ("CD4_Treg",      "CD4_Th17"),
    ("CD4_Treg",      "CD4_Th1"),
    ("CD8_cytotoxic", "CD8_exhausted"),
    ("CD8_exhausted", "CD8_cytotoxic"),
    ("B_naive",       "B_plasmablast"),
    ("B_memory",      "B_plasmablast"),
]


def select_ko_candidates(lineage: str) -> list[str]:
    """Union of high-impact, subset-specific, and whitelist E3s."""
    candidates: set[str] = set(WHITELIST_AIHA)

    a_path = DOWNSTREAM_DIR / "scoreA_high_impact.csv"
    if a_path.exists():
        a = pd.read_csv(a_path)
        high = a[a["residual_z"] > 1.5]["ko"].unique()
        candidates.update(high)
        print(f"  scoreA high-impact E3: {len(high)}")

    b_path = DOWNSTREAM_DIR / "scoreB_subset_specific.csv"
    if b_path.exists():
        b = pd.read_csv(b_path)
        specific = b[b["specificity_z"].abs() > 2.0]["ko"].unique()
        candidates.update(specific)
        print(f"  scoreB specific E3: {len(specific)}")

    return sorted(candidates)


def compute_subset_flux(adata_sim, oracle, subset_col: str = "subset") -> pd.DataFrame:
    """
    Build subset → subset transition flux from CellOracle embedding shift.
    Returns a normalized DataFrame (rows=from, cols=to).
    """
    from scipy.sparse import csr_matrix

    n     = adata_sim.n_obs
    umap  = adata_sim.obsm["X_umap"]
    shift = oracle.delta_embedding          # (n_cells, 2)
    knn   = adata_sim.obsp["distances"].toarray() > 0

    T = np.zeros((n, n))
    for i in range(n):
        nbrs = np.where(knn[i])[0]
        if len(nbrs) == 0:
            continue
        dirs   = umap[nbrs] - umap[i]
        norms  = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
        dirs_n = dirs / norms
        s      = shift[i] / (np.linalg.norm(shift[i]) + 1e-8)
        sims   = np.clip(dirs_n @ s, 0, None)
        if sims.sum() > 0:
            T[i, nbrs] = sims / sims.sum()

    subsets = adata_sim.obs[subset_col].astype(str).values
    uniq    = sorted(set(subsets))
    flux    = pd.DataFrame(0.0, index=uniq, columns=uniq)
    for i in range(n):
        si = subsets[i]
        for j in csr_matrix(T)[i].nonzero()[1]:
            flux.loc[si, subsets[j]] += T[i, j]

    row_sums = flux.sum(axis=1).replace(0, np.nan)
    return flux.div(row_sums, axis=0).fillna(0)


def build_anndata(lineage: str) -> ad.AnnData:
    """Concatenate h5ad files for the given lineage."""
    glob = LINEAGE_GLOB[lineage]
    files = sorted(SUBSETS_DIR.glob(glob))
    if not files:
        print(f"No h5ad files matching {SUBSETS_DIR}/{glob}")
        sys.exit(1)
    parts = []
    for f in files:
        a = sc.read_h5ad(f)
        a.obs["subset"] = f.stem
        parts.append(a)
    adata = ad.concat(parts, join="outer", index_unique="-")
    print(f"[{lineage}] Combined {len(files)} subsets: {adata.shape}")
    return adata


def run_celloracle(adata: ad.AnnData, lineage: str,
                   ko_genes: list[str], fate_dir: Path, fig_dir: Path) -> None:
    """Main CellOracle simulation loop with baseline."""
    import celloracle as co
    try:
        import scvi
    except ImportError:
        print("[warn] scvi not available; using PCA for embedding")
        scvi = None

    fate_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Preprocessing ----
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2500, flavor="seurat_v3", layer="counts")

    if scvi is not None:
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="subset")
        m = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
        m.train(max_epochs=300, accelerator="gpu", devices=1, early_stopping=True)
        adata.obsm["X_scVI"] = m.get_latent_representation()
        sc.pp.neighbors(adata, use_rep="X_scVI")
    else:
        sc.pp.pca(adata, n_comps=30)
        sc.pp.neighbors(adata, use_rep="X_pca")

    sc.tl.umap(adata)

    # ---- CellOracle init ----
    oracle = co.Oracle()
    oracle.import_anndata_as_normalized_count(
        adata=adata, cluster_column_name="subset", embedding_name="X_umap"
    )
    base_grn = co.data.load_human_promoter_base_GRN()
    oracle.import_TF_data(TF_info_matrix=base_grn)
    oracle.perform_PCA()
    oracle.knn_imputation(n_pca_dims=30, k=50, balanced=True, b_sight=200, b_maxl=100)

    links = oracle.get_links(cluster_name_for_GRN_unit="subset", alpha=10,
                             model_method="bagging_ridge", n_jobs=-1)
    links.filter_links(p=0.001, weight="coef_abs", threshold_number=2000)
    oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
    oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True)

    # ---- BASELINE (no perturbation) ----
    print("\n>>> Running baseline (no KO) ...")
    oracle.simulate_shift(perturb_condition={}, n_propagation=3)
    oracle.estimate_transition_prob(n_neighbors=200, knn_random=True, sampled_fraction=1)
    oracle.calculate_embedding_shift(sigma_corr=0.05)
    flux_baseline = compute_subset_flux(oracle.adata, oracle)
    flux_baseline.to_csv(fate_dir / "baseline_flux.csv")
    print(f"  Baseline saved -> {fate_dir}/baseline_flux.csv")

    # ---- Per-KO simulation ----
    for ko in ko_genes:
        if ko not in oracle.adata.var_names:
            print(f"  [skip] {ko}: not in adata"); continue
        print(f"\n>>> Simulating KO: {ko}")

        oracle.simulate_shift(perturb_condition={ko: 0.0}, n_propagation=3)
        oracle.estimate_transition_prob(n_neighbors=200, knn_random=True, sampled_fraction=1)
        oracle.calculate_embedding_shift(sigma_corr=0.05)

        flux_ko   = compute_subset_flux(oracle.adata, oracle)
        delta_flux = flux_ko - flux_baseline
        delta_flux.to_csv(fate_dir / f"delta_flux_{ko}.csv")

        # quiver plot
        umap  = oracle.adata.obsm["X_umap"]
        shift = oracle.delta_embedding
        fig, ax = plt.subplots(figsize=(7, 5))
        sc.pl.umap(oracle.adata, color="subset", ax=ax, show=False, size=3, alpha=0.5)
        ax.quiver(umap[:, 0], umap[:, 1], shift[:, 0], shift[:, 1],
                  angles="xy", scale_units="xy", scale=20, alpha=0.35, width=0.002)
        ax.set_title(f"KO {ko} — {lineage} cells")
        plt.tight_layout()
        plt.savefig(fig_dir / f"quiver_{ko}.pdf")
        plt.close(fig)

    print(f"\n[{lineage}] Done. delta_flux files in {fate_dir}")


def main():
    ap = argparse.ArgumentParser(description="CellOracle fate simulation with baseline")
    ap.add_argument("--lineage", choices=["T", "B"], required=True,
                    help="T = CD4+CD8, B = B+Plasmablast")
    ap.add_argument("--ko-genes", nargs="*", default=None,
                    help="Override KO list (default: auto-select from scores + whitelist)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    lineage  = args.lineage
    fate_dir = FATE_BASE / lineage
    fig_dir  = FIG_BASE / lineage

    ko_genes = args.ko_genes or select_ko_candidates(lineage)
    print(f"KO candidates ({len(ko_genes)}): {ko_genes[:10]} ...")

    if not SUBSETS_DIR.exists():
        print(f"Subsets dir not found: {SUBSETS_DIR}"); sys.exit(1)

    adata = build_anndata(lineage)
    run_celloracle(adata, lineage, ko_genes, fate_dir, fig_dir)


if __name__ == "__main__":
    main()
