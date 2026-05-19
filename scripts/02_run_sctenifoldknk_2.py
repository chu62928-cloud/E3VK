"""
03_run_sctenifoldknk_full.py  ·  env_main  (完整修复版 v4)
===========================================================
修复内容:
  1. 修复 arrays must all be same length: 强制对齐 df_tensor 和 wt_tensor 的基因维度。
  2. 锁死底层多线程，防止 Numpy/OpenBLAS 的 CPU 踩踏卡死。
  3. 子进程内彻底 Shutdown Ray。
  4. 支持断点续跑，跳过已有 pkl 的亚群。
"""

# ==================== 👇 终极防 CPU 踩踏锁 (必须在最前面) 👇 ====================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["RAY_OBJECT_STORE_MEMORY"] = "32212254720"
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
# os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
# os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"
# ==================== 👆 ====================

import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
import gc
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
parser.add_argument("--n-workers",    type=int, default=1) # 跑单亚群，外层 worker 设为1即可
parser.add_argument("--n-cpus",       type=int, default=16) # 新增：内层 scTenifold 调用的核心数
parser.add_argument("--n-nets",       type=int, default=5)
parser.add_argument("--n-samp-cells", type=int, default=200)
parser.add_argument("--td-rank",      type=int, default=3)
parser.add_argument("--n-hvg",        type=int, default=5000)
parser.add_argument("--expr-pct",     type=float, default=0.05)
parser.add_argument("--min-cells",    type=int, default=10)
parser.add_argument("--subset",       default=None)
parser.add_argument("--quick",        action="store_true")
args = parser.parse_args()

if args.quick:
    args.n_nets = 3; args.n_samp_cells = 100; args.n_workers = 1
    print("[QUICK MODE] first 5 E3 | n_nets=3 | n_samp_cells=100 | n_workers=1")

Path(args.out_dir).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════
# 子进程函数
# ══════════════════════════════════════════════════
def run_one_subset(h5ad_path, ko_genes, tf_set, ub_set,
                   out_dir, n_nets, n_samp_cells, td_rank,
                   n_hvg, expr_pct, min_cells, quick, n_cpus):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # 强制确保子进程内部不会意外启动 Ray 并行
    # try:
    #     import ray
    #     if ray.is_initialized():
    #         ray.shutdown()
    # except Exception:
    #     pass

    import scanpy as sc, numpy as np, pandas as pd
    from pathlib import Path
    from scTenifold import scTenifoldKnk as sKnk
    from scipy import stats
    import pickle, time, warnings, traceback
    warnings.filterwarnings("ignore")

    # 瘫痪 scTenifold 内部多余的过滤，防止它误删重要的 E3 靶点
    import scTenifold.core._base as sc_base
    if hasattr(sc_base, "sc_QC"):
        sc_base.sc_QC = lambda df, **kwargs: df

    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name     = Path(h5ad_path).stem
    log_path = out_dir / f"{name}.log"

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        print(f"  [{name}] {msg}", flush=True)
        with open(log_path, "a") as f:
            f.write(f"[{ts}] {msg}\n")

    summary = {"subset": name, "status": "failed", "n_cells": 0, "n_genes_raw": 0, "n_genes_filtered": 0, "n_e3_valid": 0, "n_dr_files": 0, "grn_time_s": 0, "total_time_s": 0, "error": ""}
    t0 = time.time()

    try:
        adata = sc.read_h5ad(h5ad_path)
        log(f"Loaded: {adata.shape}")
        summary["n_cells"] = adata.n_obs; summary["n_genes_raw"] = adata.n_vars
        n_orig = adata.n_vars

        NOISE = ("MIR","SNOR","RNU","AL","AC","AP","BX","CR","CT","LINC","LOC","OR")
        is_noise = adata.var_names.str.startswith(NOISE)
        adata = adata[:, (~is_noise) | adata.var_names.isin(ko_genes)].copy()

        X2 = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        ep = (X2 > 0).sum(axis=0) / X2.shape[0]
        adata = adata[:, (ep >= expr_pct) | adata.var_names.isin(ko_genes)].copy()

        actual_hvg = min(n_hvg, adata.n_vars - 1)
        sc.pp.highly_variable_genes(adata, n_top_genes=actual_hvg, flavor="seurat_v3")
        hvg_set   = set(adata.var_names[adata.var.highly_variable])
        whitelist = (set(ko_genes) | tf_set | ub_set) & set(adata.var_names)
        adata     = adata[:, list(hvg_set | whitelist)].copy()
        
        summary["n_genes_filtered"] = adata.n_vars

        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        df = pd.DataFrame(X.T.astype(np.float32), index=adata.var_names, columns=adata.obs_names)

        expressed = (df > 0).sum(axis=1)
        ko_genes_use = ko_genes[:5] if quick else ko_genes
        ko_valid = [g for g in ko_genes_use if g in df.index and expressed.get(g, 0) >= min_cells]
        summary["n_e3_valid"] = len(ko_valid)

        if not ko_valid:
            log("SKIP: no valid KO genes")
            summary["status"] = "skipped"
            return summary

        import ray
        if not ray.is_initialized():
            # 强制限定 Ray 的共享内存池为 35 GB (35 * 1024^3 字节)，确保小于 AutoDL 的 45G 上限
            ray.init(
                num_cpus=n_cpus, 
                object_store_memory=35 * 1024 * 1024 * 1024,
                ignore_reinit_error=True
            )
            log(f"Ray initialized manually with {n_cpus} CPUs and 35GB Object Store.")

        knk = sKnk(
            data=df,
            ko_method="default",
            ko_genes=[ko_valid[0]],
            qc_kws={"min_lib_size": 200, "min_percent": 0.001},
            nc_kws={"n_nets": n_nets, "n_samp_cells": n_samp_cells, "n_cpus": n_cpus},
            td_kws={"K": td_rank},
        )

        log(f"Building GRN: n_nets={n_nets}, n_samp_cells={n_samp_cells}, n_cpus=1, td_rank={td_rank}")
        t_grn = time.time()
        knk.run_step("qc");  log(f"  QC done ({time.time()-t_grn:.0f}s)")
        knk.run_step("nc");  log(f"  NC done ({time.time()-t_grn:.0f}s)")
        knk.run_step("td");  log(f"  TD done ({time.time()-t_grn:.0f}s)")
        summary["grn_time_s"] = round(time.time() - t_grn, 1)

        # 保存 WT tensor (GRN 结果)
        wt_tensor = knk.tensor_dict["WT"].copy()

        # ======================================================================
        # 🔥 核心修复区域：解决 arrays must all be same length
        # 把原始表达矩阵 df 裁切到跟网络(wt_tensor)一模一样的大小！
        tensor_genes = list(wt_tensor.index)
        df_tensor = df.loc[tensor_genes, :].copy() 
        
        ko_valid_final = [g for g in ko_valid if g in tensor_genes]
        skipped_qc = sorted(set(ko_valid) - set(ko_valid_final))
        if skipped_qc:
            log(f"Skipped by network filtering ({len(skipped_qc)}): {skipped_qc[:5]}...")
        # ======================================================================

        saved = 0
        for i, ko_gene in enumerate(ko_valid_final):
            # ======== 👇 新增：基因级别的断点续跑 👇 ========
            out_csv_path = out_dir / f"DR_{ko_gene}.csv"
            if out_csv_path.exists():
                saved += 1
                log(f"  [SKIP] {ko_gene} 已存在，跳过。 ({saved}/{len(ko_valid_final)})")
                continue

            t_ko = time.time()

            # 🔥 创建新对象时，使用裁切好的 df_tensor，确保维度100%匹配
            knk_ko = sKnk(
                data=df_tensor, 
                ko_method="default",
                ko_genes=[ko_gene],
                qc_kws={"min_lib_size": 0, "min_percent": 0}, # 不要再过滤了
            )
            
            knk_ko.QC_dict      = knk.QC_dict
            knk_ko.network_dict = knk.network_dict
            knk_ko.tensor_dict  = {"WT": wt_tensor.copy(), "KO": wt_tensor.copy()}
            knk_ko.manifold     = None 

            knk_ko.run_step("ko")
            knk_ko.run_step("ma")
            # ======== 【终极神操作：接管 DR 计算，彻底抛弃官方有 Bug 的函数】 ========
            manifold = knk_ko.manifold.values
            n_g = len(manifold) // 2
            wt_coords = manifold[:n_g, :]
            ko_coords = manifold[n_g:, :]
            
            # 1. 计算欧氏距离
            dist = np.linalg.norm(wt_coords - ko_coords, axis=1)
            
            # 2. 核心补丁：当距离为 0 时赋予极小常数，完美避开官方“删数据导致长度不匹配”的 Bug
            dist[dist <= 0] = 1e-12
            
            # 3. Box-Cox 变换
            boxcox_dist, _ = stats.boxcox(dist)
            
            # 4. Z-score 和 FC
            z_scores = (boxcox_dist - np.mean(boxcox_dist)) / np.std(boxcox_dist, ddof=1)
            fc = dist / np.mean(dist)
            
            # 5. P-value 和 FDR
            p_vals = 2 * stats.norm.sf(np.abs(z_scores))
            try:
                from scipy.stats import false_discovery_control
                p_adj = false_discovery_control(np.clip(p_vals, 1e-300, 1))
            except Exception:
                from statsmodels.stats.multitest import multipletests
                _, p_adj, _, _ = multipletests(np.clip(p_vals, 1e-300, 1), method="fdr_bh")
            
            # 6. 直接生成干净的 DataFrame（使用我们之前完美对齐的 tensor_genes）
            dr_df = pd.DataFrame({
                "gene": tensor_genes,
                "score": dist,
                "FC": fc,
                "Z": z_scores,
                "p_value": p_vals,
                "p_adj": p_adj,
                "ko": ko_gene
            })
            # =====================================================================

            n_sig = int((dr_df.get("p_adj", pd.Series([1]*len(dr_df))) < 0.05).sum())
            dr_df.to_csv(out_dir / f"DR_{ko_gene}.csv", index=False)
            saved += 1
            log(f"  [{i+1}/{len(ko_valid_final)}] {ko_gene}: {len(dr_df)} genes, sig={n_sig}, {time.time()-t_ko:.1f}s")
            # ======== 👇 新增：每跑完一个基因，强制清空内存垃圾 👇 ========
            del knk_ko
            del dr_df
            gc.collect()

        summary["n_dr_files"] = saved
        summary["status"]     = "ok"
        log(f"Done: {saved}/{len(ko_valid_final)} DR files")

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

    pending = []
    for h in all_h5:
        marker = Path(args.out_dir) / h.stem / f"{h.stem}_knk.pkl"
        if marker.exists():
            print(f"  [SKIP] {h.stem} 已完成，自动跳过。")
        else:
            pending.append(h)

    if not pending:
        print("\n🎉 所有亚群已完成"); sys.exit(0)

    print(f"Pending Subsets: {[f.stem for f in pending]}")
    
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
                args.quick,
                args.n_cpus
            ): h.stem
            for h in pending
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                s = fut.result()
                all_summaries.append(s)
                status = s["status"]
                print(f"\n[{'OK' if status=='ok' else 'FAIL'}] {name}: "
                      f"GRN {s.get('grn_time_s', 0):.0f}s | "
                      f"DR {s.get('n_dr_files', 0)} | total {s.get('total_time_s', 0):.0f}s")
                if status != "ok":
                    print(f"  error: {s.get('error','')[:200]}")
            except Exception as e:
                print(f"\n[ERROR] {name}: {e}")

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*55}\n  COMPLETE  ({elapsed:.1f} min)\n{'='*55}")

if __name__ == "__main__":
    main()