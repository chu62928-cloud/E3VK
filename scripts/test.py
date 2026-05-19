import pandas as pd
import gzip
from pathlib import Path

# 路径设置
expr_file = "/root/autodl-tmp/E3VK/data/zheng2021/data/expression/CD4/integration/CD4.expr.txt.gz"
meta_file = "/root/autodl-tmp/E3VK/data/zheng2021/data/metaInfo/CD4.miniInfo.txt.gz"

print("🔍 正在检查 CD4 相关文件大小...")

# 1. 检查 miniInfo 里的细胞量
meta_df = pd.read_csv(meta_file, sep='\t', index_col=0)
print(f"👉 [注释文件] CD4.miniInfo.txt.gz 包含细胞数: {len(meta_df)}")

# 2. 极速统计表达矩阵的行数 (细胞数)
def count_cells_in_expr(filepath):
    count = 0
    with gzip.open(filepath, 'rt') as f:
        for _ in f:
            count += 1
    # 减去 1 行表头（基因名）
    return count - 1

expr_cells = count_cells_in_expr(expr_file)
print(f"👉 [表达矩阵] CD4.expr.txt.gz 包含细胞数: {expr_cells}")

# 3. 诊断结论
if expr_cells == len(meta_df):
    print("\n⚠️ 诊断结论：表达矩阵和注释文件一样大。")
    print("这说明 integration 文件夹里的这个 .expr.txt.gz 本身就是一个被作者下采样过的精简版矩阵！")
elif expr_cells > len(meta_df):
    print("\n⚠️ 诊断结论：表达矩阵很大，但是 miniInfo 注释文件太小了！")
    print("这说明我们漏掉了大量的注释，需要去提取那个完整的 .rds 注释文件。")
else:
    print("\n⚠️ 诊断结论：数据维度异常，需进一步排查。")