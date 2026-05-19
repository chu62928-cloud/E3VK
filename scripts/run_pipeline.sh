#!/usr/bin/env bash
# run_pipeline.sh — E3VK 下游分析 pipeline (paper-faithful, Osorio 2022)
#
# 用法：
#   cd /path/to/E3VK
#   bash scripts/run_pipeline.sh                 # 跑全部 stage 1-4 (默认)
#   bash scripts/run_pipeline.sh --stage 1       # 仅 stage 1
#   bash scripts/run_pipeline.sh --top 20        # 富集只跑每亚群 top20
#   bash scripts/run_pipeline.sh --all           # 富集全跑（耗时显著）
#
# stage 5 (鲁棒性 + landscape t-SNE) 默认不开，因为
#   - landscape 至少需要 1 个亚群 30+ E3
#   - robustness 需要 sctenifoldpy 环境单独跑

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

STAGE="all"
TOP_N=30
ALL_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)   ALL_FLAG="--all"; shift ;;
        --top)   TOP_N="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        *)       echo "Unknown arg $1"; exit 1 ;;
    esac
done

log() { echo "[$(date +%H:%M:%S)] $*"; }
run() { log "▶ $*"; "$@"; }

# Stage 1: aggregation + QC + degree
if [[ "$STAGE" == "all" || "$STAGE" == "1" ]]; then
    log "=== Stage 1 ==="
    run python scripts/04_aggregate_dr.py
    run python scripts/05_qc_runs.py
fi

# Stage 2: vulnerability + degree + DR-vs-edge
if [[ "$STAGE" == "all" || "$STAGE" == "2" ]]; then
    log "=== Stage 2 ==="
    run python scripts/06_vulnerability_map.py
    run python scripts/06b_degree_correlation.py || log "  (degree skipped — no WT scGRN)"
    run python scripts/13_dr_vs_edge.py --top-per-subset 10 || log "  (DR-vs-edge skipped — no WT scGRN)"
fi

# Stage 3: functional enrichment (GSEA + Enrichr + redundancy collapse)
if [[ "$STAGE" == "all" || "$STAGE" == "3" ]]; then
    log "=== Stage 3 ==="
    if [[ -n "$ALL_FLAG" ]]; then
        run python scripts/07_enrichment_perKO.py --all
        run python scripts/08_enrichr_sigGenes.py --all
    else
        run python scripts/07_enrichment_perKO.py --top-per-subset "$TOP_N"
        run python scripts/08_enrichr_sigGenes.py --top-per-subset "$TOP_N"
    fi
    run python scripts/09_redundancy_collapse.py
fi

# Stage 4: network validation + egocentric plots + substrate validation
if [[ "$STAGE" == "all" || "$STAGE" == "4" ]]; then
    log "=== Stage 4 ==="
    run python scripts/10_string_validation.py --top-per-subset 30
    run python scripts/11_egocentric_plot.py --top-per-subset 10
    if [[ -f data/refs/ubibrowser_known.tsv ]]; then
        run python scripts/12_substrate_validation.py
    else
        log "  data/refs/ubibrowser_known.tsv 不存在; 跳过底物验证."
        log "  执行 python scripts/12_substrate_validation.py --download-refs 查看下载指引."
    fi
fi

# Stage 5: systematic-KO landscape + cross-subset
if [[ "$STAGE" == "all" || "$STAGE" == "5" ]]; then
    log "=== Stage 5 ==="
    run python scripts/14_ko_landscape_tsne.py
    run python scripts/15_cross_subset_compare.py --top-e3 30
fi

# Stage 6: robustness (单独跑，因为需要 sctenifoldpy 环境)
# 不会自动触发，使用如:
#   conda activate sctenifoldpy
#   python scripts/16_robustness_subsample.py --subset B_naive --ko-genes TRIM21 BRCA1

log "=== Pipeline finished ==="
log "Outputs under results/AIHA/downstream/"