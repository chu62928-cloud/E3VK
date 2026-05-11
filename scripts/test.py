import scanpy as sc
import numpy as np

# 读任意一个已经拆好的亚群
a = sc.read_h5ad("/root/autodl-tmp/E3VK/results_test/subsets/CD4_test.h5ad")

print("=== 基本维度 ===")
print(f"细胞数: {a.n_obs}")
print(f"基因数: {a.n_vars}")

print("\n=== X 矩阵实际情况 ===")
X = a.X
if hasattr(X, "toarray"):
    X = X.toarray()
print(f"X shape: {X.shape}")
print(f"X 是否稀疏: {hasattr(a.X, 'toarray')}")
print(f"X 值域: [{X.min():.1f}, {X.max():.1f}]")
print(f"X 是否整数: {np.allclose(X[:50,:50], X[:50,:50].astype(int))}")

print("\n=== 表达基因过滤后剩多少 ===")
import pandas as pd
df = pd.DataFrame(X.T, index=a.var_names, columns=a.obs_names)

expr_pct = (df > 0).sum(axis=1) / df.shape[1]
for thr in [0.01, 0.05, 0.10]:
    n = (expr_pct >= thr).sum()
    print(f"  表达 >= {thr*100:.0f}% 细胞的基因: {n}")

print("\n=== var_names 格式 ===")
print(f"前 5 个基因名: {a.var_names[:5].tolist()}")
print(f"var 列: {a.var.columns.tolist()}")