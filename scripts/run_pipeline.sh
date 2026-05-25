#!/usr/bin/env bash
# run_pipeline.sh — E3VK 下游分析 pipeline (4-dimension scoring framework)
#
# 用法：
#   cd /path/to/E3VK
#   bash scripts/run_pipeline.sh                 # 跑全部 stage 0-7 (默认)
#   bash scripts/run_pipeline.sh --stage 1       # 仅 stage 1
#   bash scripts/run_pipeline.sh --top 20        # 富集只跑每亚群 top20 (无 Tier1 时)
#   bash scripts/run_pipeline.sh --all           # 富集全跑（耗时显著）
#   bash scripts/run_pipeline.sh --no-tier1      # 强制使用 top-N 而非 Tier1 名单
#
# 维度说明：
#   A 高影响 (degree-corrected vulnerability)   ← 17
#   B 亚群特异 (cross-subset specificity)        ← 18
#   C 命运决定 (CellOracle delta_flux)           ← 06_celloracle + 19a (env_fate 单独跑)
#   D 疾病相关 (whitelist + GWAS + UbiBrowser)   ← 19
#
# stage 6 (CellOracle) 因环境冲突默认跳过，需手动在 env_fate 下执行。
# stage 8 (robustness) 因需 sctenifoldpy 环境，默认跳过。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

STAGE="all"
TOP_N=30
ALL_FLAG=""
USE_TIER1=1
TIER1_CSV="results/AIHA/downstream/E3_master_ranking_Tier1.csv"

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)        ALL_FLAG="--all"; shift ;;
        --top)        TOP_N="$2"; shift 2 ;;
        --stage)      STAGE="$2"; shift 2 ;;
        --no-tier1)   USE_TIER1=0; shift ;;
        *)            echo "Unknown arg $1"; exit 1 ;;
    esac
done

log() { echo "[$(date +%H:%M:%S)] $*"; }
run() { log "▶ $*"; "$@"; }

# Helper: emit --tier1 arg only if Tier1 CSV exists and not disabled
tier1_arg() {
    if [[ "$USE_TIER1" -eq 1 && -f "$TIER1_CSV" ]]; then
        echo "--tier1 $TIER1_CSV"
    else
        echo "--top-per-subset $1"
    fi
}

# ──────────────────────────────────────────────────────────────
# Stage 0: aggregation + QC
# ──────────────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "0" ]]; then
    log "=== Stage 0: aggregation + QC ==="
    run python scripts/04_aggregate_dr.py
    run python scripts/05_qc_runs.py
fi

# ──────────────────────────────────────────────────────────────
# Stage 1: legacy vulnerability + degree (kept for back-compat /
#          downstream scripts that still read vulnerability.csv)
# ──────────────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "1" ]]; then
    log "=== Stage 1: legacy vulnerability + degree ==="
    run python scripts/06_vulnerability_map.py
    run python scripts/06b_degree_correlation.py || log "  (degree skipped — no WT scGRN)"
fi

# ──────────────────────────────────────────────────────────────
# Stage 2: 4-dimension scoring (A + B + D)
#          C is filled NaN here (CellOracle runs separately in stage 6)
# ──────────────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "2" ]]; then
    log "=== Stage 2: 4-dimension scoring ==="
    run python scripts/17_score_A_high_impact.py
    run python scripts/18_score_B_specificity.py

    # D needs substrate_validation.csv (run 12 first if UbiBrowser refs present)
    if [[ -f data/refs/ubibrowser_known.tsv ]]; then
        run python scripts/12_substrate_validation.py
    else
        log "  data/refs/ubibrowser_known.tsv 不存在; D3 (substrate overlap) 将为 0."
        log "  执行 python scripts/12_substrate_validation.py --download-refs 查看指引."
    fi
    run python scripts/19_score_D_disease.py
    run python scripts/19a_aggregate_scoreC.py    # 空表; CellOracle 跑完后 --force 重跑
fi

# ──────────────────────────────────────────────────────────────
# Stage 3: master table + Tier 分级 + 主表可视化
# ──────────────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "3" ]]; then
    log "=== Stage 3: master table + visualizations ==="
    run python scripts/20_master_table.py
    run python scripts/21_master_visualize.py
fi

# ──────────────────────────────────────────────────────────────
# Stage 4: explanatory layer — enrichment (GSEA + Enrichr + dedup)
#          会自动使用 Tier1 名单 (若存在) 过滤 KO
# ──────────────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "4" ]]; then
    log "=== Stage 4: enrichment ==="
    if [[ -n "$ALL_FLAG" ]]; then
        run python scripts/07_enrichment_perKO.py --all
        run python scripts/08_enrichr_sigGenes.py --all
    else
        # shellcheck disable=SC2046
        run python scripts/07_enrichment_perKO.py $(tier1_arg "$TOP_N")
        # shellcheck disable=SC2046
        run python scripts/08_enrichr_sigGenes.py $(tier1_arg "$TOP_N")
    fi
    run python scripts/09_redundancy_collapse.py
fi

# ──────────────────────────────────────────────────────────────
# Stage 5: network validation + egocentric + DR-vs-edge
# ──────────────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "5" ]]; then
    log "=== Stage 5: validation + egocentric ==="
    # shellcheck disable=SC2046
    run python scripts/10_string_validation.py  $(tier1_arg 30)
    # shellcheck disable=SC2046
    run python scripts/11_egocentric_plot.py    $(tier1_arg 10)
    # shellcheck disable=SC2046
    run python scripts/13_dr_vs_edge.py         $(tier1_arg 10) || log "  (DR-vs-edge skipped — no WT scGRN)"
fi

# ──────────────────────────────────────────────────────────────
# Stage 6: CellOracle fate analysis (env_fate required, manual)
# ──────────────────────────────────────────────────────────────
# 不会自动触发。使用:
#   conda activate env_fate
#   python scripts/06_celloracle_cellrank_fate.py --lineage T
#   python scripts/06_celloracle_cellrank_fate.py --lineage B
#   conda deactivate
#   python scripts/19a_aggregate_scoreC.py --force
#   python scripts/20_master_table.py --force      # 重算 composite 含 C 维度
#   python scripts/21_master_visualize.py --force

# ──────────────────────────────────────────────────────────────
# Stage 7: landscape + cross-subset + enrichment plots
# ──────────────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "7" ]]; then
    log "=== Stage 7: landscape + enrichment plots ==="
    run python scripts/14_ko_landscape_tsne.py
    run python scripts/15_cross_subset_compare.py --top-e3 30
    run python scripts/08_perturbation_landscape.py || log "  (landscape skipped)"
    run python scripts/viz_enrichment.py --root . || log "  (viz_enrichment skipped)"
fi

# ──────────────────────────────────────────────────────────────
# Stage 8: robustness (sctenifoldpy env required, manual)
# ──────────────────────────────────────────────────────────────
# 不会自动触发。使用:
#   conda activate sctenifoldpy
#   python scripts/16_robustness_subsample.py --subset CD4_Treg \
#       --ko-genes ITCH CBLB STUB1 VHL TRAF6 --n-cells 500 --n-repeats 10

log "=== Pipeline finished ==="
log "Outputs under results/AIHA/downstream/"
if [[ -f "$TIER1_CSV" ]]; then
    n_tier1=$(($(wc -l < "$TIER1_CSV") - 1))
    log "Tier1 candidates: $n_tier1 rows in $TIER1_CSV"
    log "Whitelist coverage: $(cat results/AIHA/downstream/whitelist_coverage.txt 2>/dev/null | head -1)"
fi
