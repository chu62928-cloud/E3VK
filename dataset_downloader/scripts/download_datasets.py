#!/usr/bin/env python3
"""
Download public immune-cell scRNA-seq datasets for E3 ligase virtual knockout analysis.

Default datasets:
- 10x PBMC 10k v3
- GEO GSE147424
- GEO GSE150728
- Allen Human Immune Health Atlas h5ad files

The script is intentionally conservative:
- skips existing files by default
- streams large files
- supports manual URLs for CELLxGENE/HCA h5ad files
"""

import argparse
import os
import re
import sys
import tarfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


DEFAULT_DATASETS = {
    "10x_pbmc_10k_v3": {
        "type": "direct",
        "url": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.tar.gz",
        "subdir": "10x/pbmc_10k_v3",
        "filename": "pbmc_10k_v3_filtered_feature_bc_matrix.tar.gz",
        "unpack": True,
    }
}

ALLEN_SCRNA_PAGE = "https://apps.allenimmunology.org/aifi/resources/imm-health-atlas/downloads/scrna/"

DEFAULT_ALLEN_FILES = [
    "human_immune_health_atlas_cd4t-treg-dnt.h5ad",
    "human_immune_health_atlas_cd8t-gdt-mait.h5ad",
    "human_immune_health_atlas_b-plasma.h5ad",
    "human_immune_health_atlas_dc.h5ad",
]


def log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def download_file(url: str, outpath: Path, overwrite: bool = False, chunk_size: int = 1024 * 1024) -> Path:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if outpath.exists() and outpath.stat().st_size > 0 and not overwrite:
        log(f"Skip existing file: {outpath}")
        return outpath

    tmp = outpath.with_suffix(outpath.suffix + ".part")
    headers = {}
    if tmp.exists() and tmp.stat().st_size > 0:
        headers["Range"] = f"bytes={tmp.stat().st_size}-"

    log(f"Downloading: {url}")
    log(f"Output: {outpath}")

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()

        mode = "ab" if "Range" in headers and r.status_code == 206 else "wb"
        total = int(r.headers.get("content-length", 0))
        if mode == "ab":
            total += tmp.stat().st_size

        with open(tmp, mode) as f, tqdm(
            total=total if total > 0 else None,
            initial=tmp.stat().st_size if mode == "ab" else 0,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    tmp.replace(outpath)
    return outpath


def unpack_archive(path: Path, outdir: Path) -> None:
    if not path.exists():
        return
    if path.name.endswith(".tar.gz") or path.name.endswith(".tgz") or path.name.endswith(".tar"):
        extract_dir = outdir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        log(f"Unpacking {path.name} -> {extract_dir}")
        try:
            with tarfile.open(path) as tar:
                tar.extractall(extract_dir)
        except tarfile.ReadError:
            log(f"WARNING: Could not unpack as tar: {path}")


def geo_supplementary_url(acc: str) -> str:
    # Official GEO supplementary-file endpoint.
    return f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={acc}&format=file"


def download_default_dataset(name: str, outdir: Path, overwrite: bool) -> None:
    if name not in DEFAULT_DATASETS:
        raise ValueError(f"Unknown default dataset: {name}")

    meta = DEFAULT_DATASETS[name]
    target_dir = outdir / meta["subdir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    if meta["type"] == "direct":
        url = meta["url"]
    elif meta["type"] == "geo":
        url = geo_supplementary_url(meta["acc"])
    else:
        raise ValueError(meta["type"])

    outpath = target_dir / meta["filename"]
    download_file(url, outpath, overwrite=overwrite)

    if meta.get("unpack", False):
        unpack_archive(outpath, target_dir)


def get_allen_h5ad_links() -> dict:
    log(f"Fetching Allen HIA scRNA-seq download page: {ALLEN_SCRNA_PAGE}")
    html = requests.get(ALLEN_SCRNA_PAGE, timeout=60).text
    soup = BeautifulSoup(html, "html.parser")

    links = {}
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        text = a.get_text(" ", strip=True)
        joined = urljoin(ALLEN_SCRNA_PAGE, href)

        # Skip login / OAuth / Google IAP links.
        bad_patterns = [
            "accounts.google.com",
            "oauth",
            "iap.googleapis.com",
            "login",
            "signin",
        ]
        if any(x in joined.lower() for x in bad_patterns):
            continue

        # Only keep links that look like direct h5ad downloads.
        if ".h5ad" not in (href + " " + text):
            continue

        candidates = re.findall(
            r"human_immune_health_atlas_[A-Za-z0-9_.\\-]+\\.h5ad",
            href + " " + text
        )

        for fname in candidates:
            if joined and href and joined.endswith(".h5ad"):
                links[fname] = joined

    return links


def download_allen_files(outdir: Path, filenames: list[str], overwrite: bool) -> None:
    links = get_allen_h5ad_links()
    if not links:
        raise RuntimeError(
            "Could not parse Allen HIA h5ad links. Open the Allen download page manually and paste URLs into manual_urls.tsv."
        )

    target_dir = outdir / "allen_human_immune_health_atlas"
    target_dir.mkdir(parents=True, exist_ok=True)

    for fname in filenames:
        if fname not in links:
            log(f"WARNING: Allen file not found on page: {fname}")
            continue
        download_file(links[fname], target_dir / fname, overwrite=overwrite)


def download_manual_urls(manual_tsv: Path, outdir: Path, overwrite: bool) -> None:
    if not manual_tsv.exists():
        return

    target_dir = outdir / "manual_urls"
    target_dir.mkdir(parents=True, exist_ok=True)

    with open(manual_tsv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                name, url = line.split("\t", 1)
            except ValueError:
                log(f"WARNING: skip malformed manual_urls.tsv line: {line}")
                continue
            parsed_name = Path(urlparse(url).path).name
            filename = parsed_name if parsed_name else f"{name}.h5ad"
            download_file(url, target_dir / filename, overwrite=overwrite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data", help="Output directory")
    ap.add_argument("--all", action="store_true", help="Download default datasets and default Allen files")
    ap.add_argument("--datasets", nargs="*", default=[], help=f"Default datasets: {', '.join(DEFAULT_DATASETS)}")
    ap.add_argument("--allen-files", nargs="*", default=[], help="Allen Human Immune Health Atlas .h5ad filenames")
    ap.add_argument("--manual-urls", default="manual_urls.tsv", help="TSV with columns: name<TAB>url")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = list(args.datasets)
    allen_files = list(args.allen_files)

    if args.all:
        datasets = list(DEFAULT_DATASETS.keys())
        allen_files = DEFAULT_ALLEN_FILES

    if not datasets and not allen_files:
        log("Nothing selected. Use --all or --datasets / --allen-files.")
        sys.exit(1)

    for name in datasets:
        download_default_dataset(name, outdir, overwrite=args.overwrite)

    if allen_files:
        download_allen_files(outdir, allen_files, overwrite=args.overwrite)

    # Optional user-provided CELLxGENE/HCA permanent h5ad links.
    download_manual_urls(Path(args.manual_urls), outdir, overwrite=args.overwrite)

    log("Finished.")


if __name__ == "__main__":
    main()
