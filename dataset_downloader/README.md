# scRNA-seq Dataset Downloader for E3 Ligase Immune-Cell Project

This downloader focuses on easy-to-download immune-cell datasets with T cells.

Included by default:
1. 10x Genomics PBMC 10k v3
2. GEO GSE147424, atopic dermatitis skin scRNA-seq
3. GEO GSE150728, COVID-19 PBMC scRNA-seq
4. Allen Human Immune Health Atlas h5ad files, downloaded by scraping official download links

## Usage

```bash
conda create -n e3_download python=3.10 -y
conda activate e3_download
pip install -r requirements.txt

python scripts/download_datasets.py --outdir data --all
```

Download only selected datasets:

```bash
python scripts/download_datasets.py --outdir data --datasets 10x_pbmc_10k_v3 GSE147424
```

Download selected Allen Human Immune Health Atlas files:

```bash
python scripts/download_datasets.py --outdir data --allen-files human_immune_health_atlas_cd4t-treg-dnt.h5ad human_immune_health_atlas_cd8t-gdt-mait.h5ad human_immune_health_atlas_dc.h5ad
```

## Notes

- GEO downloads use the official NCBI GEO supplementary-file endpoint.
- 10x download uses the public 10x cloud file URL.
- Allen HIA links are parsed from the official h5ad download page, so if the site changes, rerun the script or manually paste the h5ad URL into `manual_urls.tsv`.
- Large files can take a long time. This script skips files that already exist unless `--overwrite` is used.
