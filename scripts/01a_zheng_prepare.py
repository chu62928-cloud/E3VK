import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import gc

# 1. 定义你的路径 (请根据你的实际情况修改)
EXPR_DIR = Path("/root/autodl-tmp/E3VK/data/zheng2021/data/expression")
META_DIR = Path("/root/autodl-tmp/E3VK/data/zheng2021/data/metaInfo")
OUT_DIR = Path("/root/autodl-tmp/E3VK/data/zheng2021")

for lineage in ["CD4", "CD8"]:
    print(f"\n🚀 开始处理 {lineage} T细胞...")
    
    expr_file = EXPR_DIR / lineage / "integration" / f"{lineage}.expr.txt.gz"
    meta_file = META_DIR / f"{lineage}.miniInfo.txt.gz"
    
    if not expr_file.exists():
        print(f"❌ 找不到表达矩阵: {expr_file}")
        continue

    # === 1. 读取细胞注释 (Metadata) ===
    print("  -> 正在读取细胞注释 (miniInfo)...")
    # index_col=0 刚好将 'BC.Elham2018.10X.C0000' 设置为行索引
    meta_df = pd.read_csv(meta_file, sep='\t', index_col=0) 
    
    # === 2. 读取表达矩阵 (Expression Matrix) ===
    print(f"  -> 正在读取巨大的表达矩阵 {expr_file.name} (请耐心等待)...")
    # 这个矩阵本身已经是 行=细胞, 列=基因 了
    expr_df = pd.read_csv(expr_file, sep='\t', index_col=0)
    
    # === 3. 组装 AnnData 对象 (修复：去掉了 .T) ===
    print("  -> 正在组装 AnnData 对象...")
    adata = sc.AnnData(X=expr_df.values)
    adata.obs_names = expr_df.index    # 现在 index 才是细胞名
    adata.var_names = expr_df.columns  # columns 才是基因名
    
    # 释放原始矩阵内存防 OOM
    del expr_df
    gc.collect()
    
    # === 4. 将注释信息对齐并合并到 AnnData ===
    # 现在这里是 细胞名 与 细胞名 的交集，必然能匹配上！
    common_cells = adata.obs_names.intersection(meta_df.index)
    print(f"  -> 成功匹配到 {len(common_cells)} 个细胞的注释信息！")
    
    adata = adata[common_cells].copy()
    adata.obs = meta_df.loc[common_cells]
    
    if "log1p" in adata.uns:
        del adata.uns["log1p"]

    # === 5. 保存为 h5ad ===
    out_file = OUT_DIR / f"{lineage}.h5ad"
    print(f"  -> 正在保存至 {out_file} ...")
    adata.write_h5ad(out_file, compression="gzip")
    print(f"✅ {lineage} 处理完成！")

print("\n🎉 全部组装完毕！你现在可以再次运行 01b_zheng_load.py 了。")