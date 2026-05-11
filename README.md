# E3 Ligase Virtual Knockout in Immune Cells



---

## 一、项目概述

### 科学目标

对 672 个 E3 泛素连接酶（E3 ligase）基因在免疫细胞各亚群中进行**虚拟敲除（Virtual KO）**，分析：

1. 不同细胞亚群对 E3 缺失的反应差异（vulnerability map）
2. 敲除后显著差异调控基因（DR genes）的通路富集
3. KO 后的 T 细胞功能亚型转换（如 Th1→Treg，需 CellOracle + CellRank）
4. E3 底物释放验证（与 UbiBrowser 数据库交叉）
5. E3 family（RING/HECT/CRL/RBR）层面的亚群特异性富集

### 数据集

**主数据集：Allen Immune Health Atlas (AIHA)**

- 来源：Allen Institute，三个 h5ad 文件
- 路径：`data/aiha/`
  - `human_immune_health_atlas_cd4t-treg-dnt.h5ad`
  - `human_immune_health_atlas_cd8t-gdt-mait.h5ad`
  - `human_immune_health_atlas_b-plasma.h5ad`
- 已用 signature score 方法划分功能亚型（Th1/Th17/Treg/Tex 等）

**辅助参考：Inflammation Landscape (Jiménez-Gracia 2026)**

- 已下载：`data/inflammation_atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad`（17.2GB）
- 包含 SLE/PSO/RA/IBD/MS 等 19 种炎症/自免疾病，Level1/Level2 注释直接可用
- 可作为疾病背景验证集，后续重跑时可加入

### E3 基因列表

- 文件：`data/E3-ome.xlsx`（672 个基因，含 Gene/GeneID/Protein class 三列）
- Family 分布：RING 为主（294个），另有 HECT/RBR/CRL 等
- 与数据取交集后约 **550-560 个基因**可实际 KO（因亚群表达差异有所不同）

---

## 二、工具选择与理由

### 虚拟敲除主工具：`scTenifoldKnk`（Python 版）

- **包名**：`scTenifold`（`pip install scTenifold`）
- **环境**：`conda activate sctenifoldpy`
- **选择理由**：
  - E3 多数不是转录因子，不适合 CellOracle/SCENIC 等依赖 motif 的工具
  - 基于数据驱动的共表达 GRN，对任何基因（TF、E3、激酶）一视同仁
  - 不需要配对扰动训练数据（无 Perturb-seq 数据）
  - 672 个 KO 复用同一 GRN，批量计算天然高效

### 命运转换辅助工具：`CellOracle` + `CellRank 2`（单独环境）

- **环境**：`conda activate env_fate`（与 scTenifold 分开，存在依赖冲突）
- **用途**：对 scTenifoldKnk 找到的 top hits（前 30 个 E3）验证 KO 后的 T 细胞命运偏移
- **尚未开始**，待 scTenifoldKnk 全量跑完后进行

---

## 三、目录结构

```
E3VK/
├── data/
│   ├── E3-ome.xlsx                    # 672个E3基因列表(Gene/GeneID/Protein class)
│   ├── aiha/                          # AIHA原始h5ad文件
│   ├── inflammation_atlas/            # Inflammation Landscape数据
│   └── refs/
│       └── human_TFs_Lambert2018.csv  # Lambert 2018 TF列表(1639个)
├── results/
│   └── AIHA/
│       ├── subsets/                   # 拆分好的亚群h5ad(scTenifoldKnk输入)
│       │   ├── B_memory.h5ad          # 502MB
│       │   ├── B_naive.h5ad           # 731MB
│       │   ├── B_plasmablast.h5ad     # 47MB
│       │   ├── CD4_Th1.h5ad           # 306MB
│       │   ├── CD4_Th17.h5ad          # 258MB
│       │   ├── CD4_Th1_Th17.h5ad      # 27MB
│       │   ├── CD4_Th2.h5ad           # 557MB
│       │   ├── CD4_Th2_Th17.h5ad      # 63MB
│       │   ├── CD4_Treg.h5ad          # 378MB
│       │   ├── CD4_unpol.h5ad         # 2.1GB ← 超大，单独跑
│       │   ├── CD8_cytotoxic.h5ad     # 484MB
│       │   ├── CD8_eff_general.h5ad   # 986MB
│       │   ├── CD8_exhausted.h5ad     # 391MB
│       │   ├── MAIT.h5ad              # 482MB
│       │   └── gdT.h5ad               # 508MB
│       ├── E3_genes_present.csv       # 与数据取交集后的E3列表
│       └── knk/                       # scTenifoldKnk输出
│           ├── B_naive/               # ✅ 已完成(271 DR files + _knk.pkl)
│           ├── B_plasmablast/         # ✅ 已完成(425 DR files + _knk.pkl)
│           ├── CD4_Th1_Th17/          # ✅ 已完成(332 DR files + _knk.pkl)
│           ├── CD4_Th2/               # ✅ 已完成(317 DR files + _knk.pkl)
│           └── run_summary.csv
└── scripts/
    ├── gene_whitelist.py              # TF + 泛素通路白名单(共享模块)
    ├── 01_pbmc_pipeline_test.py       # 10x PBMC pipeline验证脚本
    ├── 02_aiha_load.py                # AIHA数据加载+亚群拆分
    ├── 03_run_sctenifoldknk_full.py   # 主KO脚本(当前核心)
    ├── 04_aggregate_dr.py             # DR结果汇总(待跑)
    └── 05_downstream.py               # 富集+可视化(待跑)
```

---

## 四、当前进度（接手时状态）

### ✅ 已完成

| 步骤 | 状态 |
|---|---|
| 环境配置（sctenifoldpy / env_fate） | 完成 |
| AIHA 数据加载 + 15个亚群拆分 | 完成 |
| E3 列表与数据基因取交集 | 完成（约550个基因/亚群） |
| Pipeline 验证（10x PBMC 1k） | 完成 |
| scTenifoldKnk API 参数名确认 | 完成（见下方） |
| 4个亚群 KO 完成 | B_naive / B_plasmablast / CD4_Th1_Th17 / CD4_Th2 |

### ⏳ 进行中 / 待跑

- **11个亚群的 KO 计算**（核心任务，见第五节）
- 下游分析脚本（04 / 05）尚未运行

---

## 五、立即可以执行的操作（接手后第一件事）

### Step 1：确认环境

```bash
conda activate sctenifoldpy
python -c "from scTenifold import scTenifoldKnk; print('OK')"
```

### Step 2：清理 failed 的残留目录

断点续跑以 `_knk.pkl` 文件存在为完成标志，没有 pkl 的目录需要清理重跑：

```bash
# 查看哪些有 pkl（真正完成）
for d in results/AIHA/knk/*/; do
    name=$(basename $d)
    pkl="${d}${name}_knk.pkl"
    dr_count=$(ls ${d}DR_*.csv 2>/dev/null | wc -l)
    if [ -f "$pkl" ]; then
        echo "DONE: $name ($dr_count DR files)"
    else
        echo "PENDING: $name ($dr_count DR files, no pkl) — will remove"
        rm -rf "$d"
    fi
done
```

### Step 3：隐藏超大亚群，先跑其余 10 个

`CD4_unpol`（2.1GB h5ad）细胞数极多，GRN 构建时间远超其他亚群，先单独处理：

```bash
# 临时隐藏超大亚群
mv results/AIHA/subsets/CD4_unpol.h5ad \
   results/AIHA/subsets/CD4_unpol.h5ad.bak

# 跑其余所有亚群（断点续跑，已完成的自动跳过）
tmux new -s main_run
conda activate sctenifoldpy
python scripts/03_run_sctenifoldknk_full.py \
    --n-workers 8 \
    --n-nets 5 \
    --n-samp-cells 200 \
    --td-rank 3
```

### Step 4：等第一批完成后跑超大亚群

```bash
# 恢复并单独跑
mv results/AIHA/subsets/CD4_unpol.h5ad.bak \
   results/AIHA/subsets/CD4_unpol.h5ad

tmux new -s unpol_run
conda activate sctenifoldpy
python scripts/03_run_sctenifoldknk_full.py \
    --n-workers 1 \
    --n-nets 5 \
    --n-samp-cells 200 \
    --td-rank 3 \
    --subset CD4_unpol
```

### Step 5：监控 GRN 速度（关键检查点）

```bash
# 开跑后约5分钟查看
watch -n 30 'grep -h "NC done\|GRN ready\|QC done" \
    results/AIHA/knk/*/*.log 2>/dev/null | tail -20'
```

**判断标准**：
- NC done < 500s → Ray 成功禁用，速度正常 ✅
- NC done ≈ 7000s → Ray 未被禁用，需要进一步排查 ❌

---

## 六、scTenifoldKnk 关键技术细节

### 已确认的 API 参数名

```python
from scTenifold import scTenifoldKnk as sKnk

knk = sKnk(
    data=df,                          # gene × cell DataFrame，float32，raw counts
    ko_method="default",
    ko_genes=["TRIM21"],              # 占位，后面逐个替换
    qc_kws={"min_lib_size": 200, "min_percent": 0.001},
    nc_kws={
        "n_nets":       5,            # GRN构建次数（测试用3，正式用5）
        "n_samp_cells": 200,          # 每次子采样细胞数
        "n_cpus":       1,            # 禁用Ray内部并行
    },
    td_kws={"K": 3},                  # Tensor decomp rank（原默认5，降到3加速）
)
```

### GRN 只构建一次，每个 E3 用新对象复用

```python
# 构建 GRN
knk.run_step("qc")
knk.run_step("nc")   # 最慢，约2000s（禁Ray后）
knk.run_step("td")

wt_tensor = knk.tensor_dict["WT"].copy()   # 保存 WT tensor

# 二次过滤: GRN 内部 QC 会再次剔除部分基因
tensor_genes   = set(wt_tensor.index)
ko_valid_final = [g for g in ko_valid if g in tensor_genes]

# 每个 E3 创建新对象（不复用 knk，避免 manifold 残留报错）
for ko_gene in ko_valid_final:
    knk_ko = sKnk(data=df, ko_method="default", ko_genes=[ko_gene], ...)
    knk_ko.QC_dict      = knk.QC_dict       # 复用 QC 结果
    knk_ko.network_dict = knk.network_dict  # 复用网络
    knk_ko.tensor_dict  = {"WT": wt_tensor.copy(), "KO": wt_tensor.copy()}
    knk_ko.manifold     = None              # 必须清空，否则报 arrays length 错误

    knk_ko.run_step("ko")   # 把该基因行设为0
    knk_ko.run_step("ma")   # Manifold alignment，约11s
    knk_ko.run_step("dr")   # Differential regulation，约1.7s

    dr_df = knk_ko.d_regulation.copy()
    # 列名: Gene, Distance, FC, T, Z
    # 需要自己从 Z score 计算 p_adj（BH校正）
```

### build() 返回值格式

`build()` 返回单个 `pd.DataFrame`，列为 `Gene, Distance, FC, T, Z`，**不是 dict**，**没有 p_adj**，需要手动计算：

```python
from scipy import stats
from scipy.stats import false_discovery_control
p_vals = 2 * stats.norm.sf(np.abs(dr_df["Z"].fillna(0).values))
dr_df["p_adj"] = false_discovery_control(np.clip(p_vals, 1e-300, 1))
```

### 三层基因过滤（33538 → ~5000 基因，加速 ~35x）

```
Layer 1: 去噪音前缀（MIR/SNOR/RNU/AL/AC/LINC/LOC/OR 等）
Layer 2: 表达过滤（在 ≥5% 细胞中表达，E3 基因豁免）
Layer 3: HVG 5000 + 白名单（E3 + TF + 泛素通路基因）取并集
```

白名单来自 `scripts/gene_whitelist.py`：
- **TF**：Lambert 2018（1639个），文件 `data/refs/human_TFs_Lambert2018.csv`
- **泛素通路**：硬编码 239 个核心基因（E1/E2/DUB/蛋白酶体/CRL/NEDD8/SUMO/免疫特异性）

---

## 七、输出格式说明

### DR_\<gene\>.csv 格式

每个 KO 基因对应一个 CSV，约 4000-5000 行（过滤后的基因数）：

| gene | score | FC | T | Z | p_value | p_adj | ko |
|---|---|---|---|---|---|---|---|
| TP53 | 0.823 | 1.24 | 3.21 | 4.15 | 0.0001 | 0.003 | TRIM21 |

- `score`：manifold alignment 后的 Distance
- `Z`：差异调控的 Z 统计量
- `p_adj`：BH 校正后的 FDR（显著阈值：< 0.05）
- `ko`：被敲除的 E3 基因名

### 完成标志

每个亚群目录下有 `<subset>_knk.pkl` 文件 = 该亚群完整完成。

---

## 八、已知 Bug 和修复

| Bug | 原因 | 修复 |
|---|---|---|
| `KeyError: None of [Index(['BRCA1'])]` | E3 基因通过自定义过滤但被 scTenifold 内部 QC 剔除 | GRN 构建后用 `wt_tensor.index` 二次筛选 `ko_valid` |
| `arrays must all be same length` | 复用同一 `knk` 对象，`manifold` 残留污染下一个 KO | 每个 KO 创建新 `knk_ko`，`manifold=None` |
| GRN 耗时 7000+s（应 <2000s） | Ray 在子进程里未被禁用 | 子进程开头 `ray.shutdown()` + `nc_kws["n_cpus"]=1` |
| saved 0 DR files | `build()` 返回 DataFrame 非 dict，解析逻辑错误 | 每个 E3 单独 `run_step(ko/ma/dr)`，不用 `build()` |

---

## 九、下游分析（KO 全部跑完后执行）

### Step 1：汇总 DR 矩阵

```bash
python scripts/04_aggregate_dr.py
# 输出: results/AIHA/dr_matrix.parquet, results/AIHA/vulnerability.csv
```

`vulnerability.csv` 是 **亚群 × E3 的矩阵**，值为每个 KO 的显著 DR 基因数（p_adj < 0.05）。

### Step 2：下游分析与可视化

```bash
python scripts/05_downstream.py
# 输出:
#   figs/vulnerability_heatmap.pdf   — 主热图
#   figs/family_vulnerability.pdf    — RING vs HECT vs CRL
#   results/enrichment/*.csv         — top hits 通路富集
#   results/substrate_validation.csv — UbiBrowser 底物验证
```

### Step 3：命运转换分析（env_fate 环境）

```bash
conda activate env_fate
python scripts/06_celloracle_cellrank.py
# 对 top 30 E3 hits 验证 KO 后 T 细胞命运偏移
# 输出: figs/fate/quiver_<E3>.pdf, results/fate_transition_changes.csv
```

---

## 十、服务器配置参考

| 资源 | 规格 |
|---|---|
| 内存 | 754GB total，可用约 570GB（有上传任务时） |
| CPU | 25核 |
| 数据盘 | 950GB，剩余约 370GB |
| 推荐 `--n-workers` | **8**（中小亚群批次）/ **1**（CD4_unpol 超大亚群） |

### 预估剩余计算时间

| 批次 | 亚群数 | 预计总时间 |
|---|---|---|
| 中小亚群（10个，8 workers） | 10 | ~5-6小时 |
| CD4_unpol（单独） | 1 | ~8-15小时 |

---

## 十一、联系已知问题备忘

1. **AnimalTFDB 链接挂了**：改用 Lambert 2018（`humantfs.ccbr.utoronto.ca`），或 gseapy GO 备选。
2. **`/dev/shm` Ray 警告**：Ray 已被禁用（`ray.shutdown()` + `n_cpus=1`），此警告可忽略。
3. **CD4_Th2 的亚群注释**：AIHA 健康 PBMC 数据中 Th2 信号弱（IL4 极稀疏），结果解读时需注意信号质量，查看 `functional_label_qc.csv` 里的 `self_signature_delta`。
4. **Tex（耗竭T细胞）信号**：在健康 PBMC 中弱，`CD8_exhausted` 亚群的 KO 结果置信度低于肿瘤数据集，解读时加 caveat。
