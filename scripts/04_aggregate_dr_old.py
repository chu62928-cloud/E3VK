"""
04_aggregate_dr.py
========================================================
功能: 
  1. 扫描 results/AIHA/knk/ 下所有亚群的虚拟敲除结果 (DR_*.csv)
  2. 自动兼容/标准化列名 (统一为 gene, score, Z, p_adj, ko, subset)
  3. 合并为 long-format 极速 Parquet 大表
  4. 统计每个亚群中每个 E3 敲除后的显著 (p_adj < 0.05) 靶基因数量，生成脆弱性矩阵 (Vulnerability Matrix)
"""

import pandas as pd
import re
from pathlib import Path

# 1. 统一根路径
RES = Path("results/AIHA")
KNK = RES / "knk"
rows = []

def normalize_dr_columns(df):
    """
    针对性升级版：
    1. 优先检测并保留带下划线（精确计算版）的 'p_value' 和 'p_adj'。
    2. 如果存在精确版，则主动丢弃带连字符/空格的旧版 'p-value' / 'adjusted p-value'，防止去重时劣币驱逐良币。
    3. 自动将 boxcox 距离映射为 score。
    """
    # 强制复制一份，避免 DataFrame 切片警告
    df = df.copy()
    
    # 【核心防御机制】：如果新旧两套列同时存在，直接物理超度带连字符的旧列
    cols_to_drop = []
    if "p_value" in df.columns and "p-value" in df.columns:
        cols_to_drop.append("p-value")
    if "p_adj" in df.columns and "adjusted p-value" in df.columns:
        cols_to_drop.append("adjusted p-value")
        
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        
    # 建立标准的列名映射
    rename_map = {}
    for c in df.columns:
        # 清理空格、点、连字符，统一转为下划线
        cl = c.lower().strip().replace(".", "_").replace(" ", "_").replace("-", "_")
        
        if cl in ("p_value", "pvalue", "p_val"): 
            rename_map[c] = "p_value"
        elif cl in ("p_adj", "padj", "fdr", "adjusted_p_value", "adjusted_pvalue"): 
            rename_map[c] = "p_adj"
        elif cl in ("gene", "gene_name"): 
            rename_map[c] = "gene"
        elif cl in ("distance", "score", "boxcox_transformed_distance"): 
            # 统一将 boxcox 变换后的距离作为最终的 score
            rename_map[c] = "score"
        elif cl in ("z", "z_score"):
            rename_map[c] = "Z"
        elif cl in ("fc", "fold_change"):
            rename_map[c] = "FC"
            
    # 重命名
    df = df.rename(columns=rename_map)
    
    # 再次兜底去重：万一还有同名列，保留后出现的（因为精确版通常在右侧/后面）
    df = df.loc[:, ~df.columns.duplicated(keep='last')].copy()
    
    # 补齐可能缺失的核心列，确保输出格式绝对规整
    import numpy as np
    for col in ["gene", "score", "Z", "FC", "p_value", "p_adj"]:
        if col not in df.columns:
            df[col] = np.nan
            
    # 只返回最核心的 6 列，扔掉其他所有杂质
    return df[["gene", "score", "Z", "FC", "p_value", "p_adj"]]

# 确保输入目录存在
if not KNK.exists():
    raise FileNotFoundError(f"[ERROR] 找不到 KNK 结果目录: {KNK}，请确认 03 脚本已成功运行并生成了数据。")

print(">>> 正在扫描并合并所有 DR_*.csv 文件...")
csv_files = list(KNK.rglob("DR_*.csv"))
print(f"    共检索到 {len(csv_files)} 个结果文件。")

for csv in csv_files:
    # 自动获取父文件夹名作为亚群名称 (例如 CD4_Th1, CD8_exhausted)
    subset = csv.parent.name
    
    # 提取被敲除的 E3 基因名
    match = re.match(r"DR_(.+)\.csv", csv.name)
    if not match:
        continue
    ko_gene = match.group(1)
    
    # 读取并标准化列名
    df = pd.read_csv(csv)
    df = normalize_dr_columns(df)
    
    # 验证关键列，防止空文件或损坏文件混入
    if "p_adj" not in df.columns:
        print(f"    [WARN] 文件 {csv.name} 缺少 p_adj 列，已跳过。")
        continue
        
    # 注入元数据列
    df = df.assign(subset=subset, ko=ko_gene)
    rows.append(df)

if not rows:
    print("[ERROR] 未能合并任何有效的 DR 数据！请检查 KNK 文件夹下的文件内容。")
    exit(1)

# 合并大表并保存
all_dr = pd.concat(rows, ignore_index=True)
parquet_path = RES / "dr_matrix.parquet"
all_dr.to_parquet(parquet_path)
print(f"✓ 成功生成 long-format 大表: {parquet_path} (共 {len(all_dr)} 行)")

# 2. 计算 Vulnerability Matrix (脆弱性矩阵)
# 筛选显著差异基因 (FDR < 0.05)
sig = all_dr[all_dr["p_adj"] < 0.05]

if sig.empty:
    print("[WARN] 未能在所有结果中找到任何 p_adj < 0.05 的显著基因！将生成空的脆弱性矩阵。")
    vuln = pd.DataFrame()
else:
    # 聚合计数：计算每个亚群下，敲除每个 E3 影响了多少个下游基因
    vuln = sig.groupby(["subset", "ko"]).size().unstack(fill_value=0)

vuln_path = RES / "vulnerability.csv"
vuln.to_csv(vuln_path)
print(f"✓ 成功生成脆弱性矩阵: {vuln_path} (尺寸: {vuln.shape[0]} subsets x {vuln.shape[1]} E3s)")