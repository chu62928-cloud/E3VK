import gseapy as gp
from pathlib import Path

# 1. 定义你要下载的库
libs = [
    "Reactome_Pathways_2024", 
    "KEGG_2021_Human", 
    "GO_Biological_Process_2023",
    "WikiPathways_2024_Human", 
    "MSigDB_Hallmark_2020", 
    "BioPlanet_2019"
]

# 2. 创建保存目录
out_dir = Path("data/genesets")
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Starting downloads to {out_dir.absolute()}...")

for lib in libs:
    try:
        print(f"Downloading {lib}...", end=" ", flush=True)
        # gp.get_library 会直接从服务器获取字典格式的基因集
        # organism='Human' 确保下载人类的
        gmt_data = gp.get_library(name=lib, organism='Human')
        
        # 3. 构造本地文件名
        out_file = out_dir / f"{lib}.gmt"
        
        # 4. 写入文件 (GMT格式: 路径名\t描述\t基因1\t基因2...)
        with open(out_file, "w", encoding="utf-8") as f:
            for term, genes in gmt_data.items():
                f.write(f"{term}\t\t" + "\t".join(genes) + "\n")
        
        print("Done!")
    except Exception as e:
        print(f"Failed! Error: {e}")
        print(f"提示: 如果报错 'ConnectionError'，说明服务器连不上 Enrichr。请在本地电脑下载后上传到该目录。")

print("\n全部尝试完成。")