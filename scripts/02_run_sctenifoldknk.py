"""
03_run_sctenifoldknk_full.py  ·  env_main  (完整修复版)
========================================================
修复内容:
  1. nc_kws 参数名确认: n_nets=5, n_samp_cells=200
  2. td_kws 参数名确认: K=3 (rank,加速 tensor decomp)
  3. GRN 只构建一次,每个 E3 复用 WT tensor
  4. result 是 DataFrame,每个 E3 单独调用 run_step(ko/ma/dr)
  5. Z score → p_adj (BH校正)
  6. 三层基因过滤

用法:
  python scripts/03_run_sctenifoldknk_full.py --quick        # 前5个E3,n_nets=3
  python scripts/03_run_sctenifoldknk_full.py --subset CD4_test
  python scripts/03_run_sctenifoldknk_full.py                # 全量
"""

import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle, argparse, time, json, warnings, traceback, sys

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
try:
    from gene_whitelist import load_tf_whitelist, load_ub_pathway_genes
    _WHITELIST_MODULE = True
except ImportError:
    _WHITELIST_MODULE = False
    print("[WARN] gene_whitelist.py not found")


# ══════════════════════════════════════════════════
# 参数
# ══════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--subsets-dir",  default="results/AIHA/subsets")
parser.add_argument("--e3-csv",       default="results/AIHA/E3_genes_present.csv")
parser.add_argument("--out-dir",      default="results/AIHA/knk")
parser.add_argument("--tf-csv",       default="data/refs/human_TFs_Lambert2018.csv")
parser.add_argument("--n-workers",    type=int, default=4)
# ── 已确认的 make_networks 参数 ──
parser.add_argument("--n-nets",       type=int, default=5,
                    help="GRN构建次数 nc_kws[n_nets] (测试用3,正式用5-10)")
parser.add_argument("--n-samp-cells", type=int, default=200,
                    help="每次GRN采样细胞数 nc_kws[n_samp_cells]")
# ── 已确认的 tensor_decomp 参数 ──
parser.add_argument("--td-rank",      type=int, default=3,
                    help="Tensor decomp rank td_kws[K] (默认3,原始5)")
# ── 基因过滤参数 ──
parser.add_argument("--n-hvg",        type=int, default=5000)
parser.add_argument("--expr-pct",     type=float, default=0.05)
parser.add_argument("--min-cells",    type=int, default=10,
                    help="E3基因至少在多少个细胞表达才KO")
parser.add_argument("--subset",       default=None)
parser.add_argument("--quick",        action="store_true",
                    help="前5个E3,n_nets=3,n_samp_cells=100")
args = parser.parse_args()

if args.quick:
    args.n_nets       = 3
    args.n_samp_cells = 100
    args.n_workers    = 2
    print("[QUICK MODE] first 5 E3 | n_nets=3 | n_samp_cells=100")

Path(args.out_dir).mkdir(parents=True, exist_ok=True)

NOISE_PREFIXES = ("MIR","SNOR","RNU","AL","AC","AP","BX","CR","CT","LINC","LOC","OR")


# ══════════════════════════════════════════════════
# 子进程函数
# ══════════════════════════════════════════════════
def run_one_subset(h5ad_path, ko_genes, tf_set, ub_set,
                   out_dir, n_nets, n_samp_cells, td_rank,
                   n_hvg, expr_pct, min_cells):
    import scanpy as sc, numpy as np, pandas as pd
    from pathlib import Path
    from scTenifold import scTenifoldKnk as sKnk
    from scipy import stats
    import pickle, time, warnings, traceback
    warnings.filterwarnings("ignore")

    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name     = Path(h5ad_path).stem
    log_path = out_dir / f"{name}.log"

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        print(f"  [{name}] {msg}", flush=True)
        with open(log_path, "a") as f:
            f.write(f"[{ts}] {msg}\n")

    summary = {"subset": name, "status": "failed",
                "n_cells": 0, "n_genes_raw": 0, "n_genes_filtered": 0,
                "n_e3_valid": 0, "n_dr_files": 0,
                "grn_time_s": 0, "total_time_s": 0, "error": ""}
    t0 = time.time()

    try:
        # ── 1. 加载 ──────────────────────────────────────
        adata = sc.read_h5ad(h5ad_path)
        log(f"Loaded: {adata.shape}")
        summary["n_cells"]     = adata.n_obs
        summary["n_genes_raw"] = adata.n_vars
        n_orig = adata.n_vars

        # ── 2. 三层基因过滤 ──────────────────────────────
        NOISE = ("MIR","SNOR","RNU","AL","AC","AP","BX","CR","CT","LINC","LOC","OR")
        is_noise = adata.var_names.str.startswith(NOISE)
        adata = adata[:, (~is_noise) | adata.var_names.isin(ko_genes)].copy()
        log(f"Layer1 noise: {n_orig} → {adata.n_vars}")

        X2 = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        ep = (X2 > 0).sum(axis=0) / X2.shape[0]
        adata = adata[:, (ep >= expr_pct) | adata.var_names.isin(ko_genes)].copy()
        log(f"Layer2 expr>={expr_pct*100:.0f}%: {adata.n_vars}")

        actual_hvg = min(n_hvg, adata.n_vars - 1)
        sc.pp.highly_variable_genes(adata, n_top_genes=actual_hvg, flavor="seurat_v3")
        hvg_set   = set(adata.var_names[adata.var.highly_variable])
        whitelist = (set(ko_genes) | tf_set | ub_set) & set(adata.var_names)
        adata     = adata[:, list(hvg_set | whitelist)].copy()
        log(f"Layer3 HVG+whitelist: {adata.n_vars} genes | "
            f"speedup ~{(n_orig/adata.n_vars)**2:.0f}x")
        summary["n_genes_filtered"] = adata.n_vars

        # ── 3. 转 DataFrame (gene × cell) ────────────────
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        df = pd.DataFrame(X.T.astype(np.float32),
                           index=adata.var_names, columns=adata.obs_names)

        # ── 4. 筛选有效 E3 ────────────────────────────────
        expressed = (df > 0).sum(axis=1)
        ko_valid  = [g for g in ko_genes
                     if g in df.index and expressed.get(g, 0) >= min_cells]
        log(f"E3 valid: {len(ko_valid)}/{len(ko_genes)} → {ko_valid}")
        summary["n_e3_valid"] = len(ko_valid)

        if not ko_valid:
            log("SKIP: no valid KO genes")
            summary["status"] = "skipped"
            return summary

        # ── 5. 初始化 scTenifoldKnk ──────────────────────
        # 用已确认的参数名
        knk = sKnk(
            data=df,
            ko_method="default",
            ko_genes=[ko_valid[0]],   # 先给一个占位,后面逐个替换
            qc_kws={
                "min_lib_size": 200,
                "min_percent":  0.001,
            },
            nc_kws={
                "n_nets":       n_nets,       # 确认参数名
                "n_samp_cells": n_samp_cells,
                "n_cpus":       1,            # 禁用 Ray，单线程更快 # 确认参数名
            },
            td_kws={
                "K": td_rank,   # tensor rank,降低加速 decomp
            },
        )

        # ── 6. 构建 GRN 一次 (最慢步骤) ──────────────────
        log(f"Building GRN: n_nets={n_nets}, n_samp_cells={n_samp_cells}, "
            f"td_rank={td_rank} ...")
        t_grn = time.time()

        knk.run_step("qc")
        log(f"  QC done ({time.time()-t_grn:.0f}s)")

        knk.run_step("nc")
        log(f"  Network construction done ({time.time()-t_grn:.0f}s)")

        knk.run_step("td")
        log(f"  Tensor decomposition done ({time.time()-t_grn:.0f}s)")

        grn_time = time.time() - t_grn
        summary["grn_time_s"] = round(grn_time, 1)
        log(f"GRN ready in {grn_time:.0f}s — reusing for {len(ko_valid)} KOs")

        # 保存 WT tensor (GRN 结果),每个 KO 前恢复
        wt_tensor = knk.tensor_dict["WT"].copy()

        # ── 关键修复: 重新筛选 ko_valid ──────────────────
        # scTenifold 内部 QC 会剔除低质量基因
        # wt_tensor 的 index 才是 GRN 里实际存在的基因集
        # 必须用它来二次过滤,否则 run_step("ko") 报 KeyError
        tensor_genes   = set(wt_tensor.index)
        ko_valid_final = [g for g in ko_valid if g in tensor_genes]
        skipped_by_qc  = sorted(set(ko_valid) - set(ko_valid_final))
        if skipped_by_qc:
            log(f"Skipped by internal QC: {skipped_by_qc}")
        log(f"E3 after GRN QC: {len(ko_valid_final)}/{len(ko_valid)}")

        if not ko_valid_final:
            log("SKIP: no valid KO genes after GRN QC")
            summary["status"] = "skipped"
            return summary

        # ── 7. 每个 E3 单独 KO → MA → DR ─────────────────
        saved = 0
        for i, ko_gene in enumerate(ko_valid_final):
            t_ko = time.time()

            # 恢复 WT tensor,避免上一个 KO 污染
            knk.tensor_dict["WT"] = wt_tensor.copy()
            knk.tensor_dict["KO"] = wt_tensor.copy()

            # 更新 ko_genes
            knk.ko_genes = [ko_gene]

            # KO step: 把该基因行设为 0
            knk.run_step("ko")

            # Manifold alignment
            knk.run_step("ma")

            # Differential regulation
            knk.run_step("dr")

            # 获取结果
            dr_df = knk.d_regulation.copy()

            # ── 列名标准化 ──
            # 已知列: Gene, Distance, FC, T, Z
            col_map = {}
            for c in dr_df.columns:
                cl = c.lower().strip()
                if   cl == "gene":     col_map[c] = "gene"
                elif cl == "distance": col_map[c] = "score"
                elif cl == "fc":       col_map[c] = "FC"
                elif cl == "t":        col_map[c] = "T"
                elif cl == "z":        col_map[c] = "Z"
            dr_df = dr_df.rename(columns=col_map)

            # ── 计算 p_adj ──
            # 优先用 Z score (更标准),其次用 T
            if "Z" in dr_df.columns:
                p_vals = 2 * stats.norm.sf(np.abs(dr_df["Z"].fillna(0).values))
                try:
                    from scipy.stats import false_discovery_control
                    dr_df["p_adj"] = false_discovery_control(
                        np.clip(p_vals, 1e-300, 1))
                except Exception:
                    # scipy < 1.11 没有 false_discovery_control
                    from statsmodels.stats.multitest import multipletests
                    _, padj, _, _ = multipletests(
                        np.clip(p_vals, 1e-300, 1), method="fdr_bh")
                    dr_df["p_adj"] = padj
                dr_df["p_value"] = p_vals

            elif "T" in dr_df.columns:
                df_dof = max(df.shape[1] - 2, 1)
                p_vals = 2 * stats.t.sf(np.abs(dr_df["T"].fillna(0).values),
                                         df=df_dof)
                try:
                    from scipy.stats import false_discovery_control
                    dr_df["p_adj"] = false_discovery_control(
                        np.clip(p_vals, 1e-300, 1))
                except Exception:
                    from statsmodels.stats.multitest import multipletests
                    _, padj, _, _ = multipletests(
                        np.clip(p_vals, 1e-300, 1), method="fdr_bh")
                    dr_df["p_adj"] = padj
                dr_df["p_value"] = p_vals

            # 标记 KO 来源
            dr_df["ko"] = ko_gene

            # 检查必要列
            if "gene" not in dr_df.columns:
                log(f"  [WARN] {ko_gene}: no 'gene' col, got {dr_df.columns.tolist()}")
                continue

            n_sig = int((dr_df.get("p_adj", pd.Series([1]*len(dr_df))) < 0.05).sum())
            dr_df.to_csv(out_dir / f"DR_{ko_gene}.csv", index=False)
            saved += 1
            log(f"  [{i+1}/{len(ko_valid)}] {ko_gene}: "
                f"{len(dr_df)} genes, sig={n_sig}, {time.time()-t_ko:.1f}s")

        summary["n_dr_files"] = saved
        summary["status"]     = "ok"
        log(f"Done: {saved}/{len(ko_valid)} DR files")

        with open(out_dir / f"{name}_knk.pkl", "wb") as f:
            pickle.dump(knk, f)

    except Exception as e:
        summary["error"] = str(e)
        log(f"ERROR: {e}\n{traceback.format_exc()}")

    summary["total_time_s"] = round(time.time() - t0, 1)
    return summary


# ══════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════
def main():
    e3_df    = pd.read_csv(args.e3_csv)
    ko_genes = e3_df["gene"].tolist()
    if args.quick:
        ko_genes = ko_genes[:5]
    print(f"E3 to KO: {len(ko_genes)}")

    if _WHITELIST_MODULE:
        tf_set = load_tf_whitelist(args.tf_csv)
        ub_set = load_ub_pathway_genes()
    else:
        tf_set, ub_set = set(), set()

    all_h5 = sorted(Path(args.subsets_dir).glob("*.h5ad"))
    if args.subset:
        all_h5 = [f for f in all_h5 if f.stem == args.subset]
        if not all_h5:
            print(f"[ERROR] '{args.subset}' not found"); sys.exit(1)
    pending_h5 = []
    for h in all_h5:
        # 用最终生成的 pkl 文件作为该亚群已彻底跑完的标志
        done_marker = Path(args.out_dir) / h.stem / f"{h.stem}_knk.pkl"
        if done_marker.exists():
            print(f"  [SKIP] 亚群 {h.stem} 已检测到完整结果，自动跳过。")
        else:
            pending_h5.append(h)
    
    all_h5 = pending_h5
    if not all_h5:
        print("\n🎉 所有亚群均已处理完毕！无需执行任何计算。")
        sys.exit(0)
        
    print(f"Subsets: {[f.stem for f in all_h5]}")
    print(f"Config: n_nets={args.n_nets} | n_samp_cells={args.n_samp_cells} | "
          f"td_rank={args.td_rank} | n_hvg={args.n_hvg} | "
          f"expr_pct={args.expr_pct} | n_workers={args.n_workers}")

    # 预估时间
    # GRN: n_nets × n_samp_cells × genes² 相关
    # 原来: n_nets=10, n_samp_cells=500, genes=2523 → 9569s
    # 现在: n_nets=5,  n_samp_cells=200, genes=2523
    # 理论加速: (10/5) × (500/200) = 5x → 约 1900s → ~32min
    est_grn = 9569 * (args.n_nets/10) * (args.n_samp_cells/500)
    est_ko  = len(ko_genes) * 15   # 每个 KO 约 15s (ma+dr)
    print(f"Estimated: GRN ~{est_grn/60:.0f}min + "
          f"KO {len(ko_genes)}×~15s = ~{(est_grn+est_ko)/60:.0f}min per subset")

    all_summaries = []
    t_start       = time.time()

    with ProcessPoolExecutor(max_workers=args.n_workers) as exe:
        futures = {
            exe.submit(
                run_one_subset,
                str(h), ko_genes, tf_set, ub_set,
                f"{args.out_dir}/{h.stem}",
                args.n_nets, args.n_samp_cells, args.td_rank,
                args.n_hvg, args.expr_pct, args.min_cells,
            ): h.stem
            for h in all_h5
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                s = fut.result()
                all_summaries.append(s)
                raw = s.get("n_genes_raw", 0)
                fil = s.get("n_genes_filtered", 1)
                print(f"\n[{'OK' if s['status']=='ok' else 'FAIL'}] {name}: "
                      f"genes {raw}→{fil} | "
                      f"GRN {s.get('grn_time_s',0):.0f}s | "
                      f"DR files {s['n_dr_files']} | "
                      f"total {s.get('total_time_s',0):.0f}s")
            except Exception as e:
                print(f"\n[ERROR] {name}: {e}")
                all_summaries.append({"subset": name, "status": "exception",
                                       "n_genes_raw": 0, "n_genes_filtered": 0,
                                       "n_dr_files": 0, "grn_time_s": 0,
                                       "total_time_s": 0, "error": str(e)})

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(f"{args.out_dir}/run_summary.csv", index=False)

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*55}")
    print(f"  COMPLETE  ({elapsed:.1f} min total)")
    print(f"{'='*55}")
    cols = [c for c in ["subset","status","n_genes_raw","n_genes_filtered",
                         "n_e3_valid","n_dr_files","grn_time_s","total_time_s"]
            if c in summary_df.columns]
    print(summary_df[cols].to_string(index=False))

    ok = summary_df[summary_df["status"] == "ok"]
    print(f"\n  OK: {len(ok)}/{len(summary_df)}")

    if len(ok) > 0:
        print(f"""
  检查点:
    1. GRN 时间 (grn_time_s): 目标 <1800s (~30min)
       原来 9569s → 现在应约 1900s (5x 加速)
    2. 每个 KO 时间: 目标 <30s
    3. DR_*.csv 有列: gene, score, Z, p_adj, ko
    4. sig genes (p_adj<0.05): 每个 KO 有几个到几百个正常
  """)
    print("  Next: python scripts/04_aggregate_dr.py")


if __name__ == "__main__":
    main()