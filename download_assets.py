#!/usr/bin/env python3
"""Download curated HDRIs and CC textures for domain randomization.

Usage:
    python download_assets.py [--output-dir /workspace/blenderproc-1x4090/assets]

Downloads:
  - ~20 indoor HDRIs from Poly Haven (1k resolution, ~1-3 MB each)
  - ~30 PBR textures from ambientCG (2K-JPG, ~5-15 MB each)
"""

import argparse
import os
import shutil
import zipfile
import tempfile
import concurrent.futures
from pathlib import Path

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "assets")

# Indoor-relevant HDRI categories for beer pong scenes
HDRI_IDS = [
    "abandoned_parking",
    "artist_workshop",
    "autoshop_01",
    "brown_photostudio_02",
    "cafe_wall",
    "canteen",
    "carpentry_shop_01",
    "cinema_hall",
    "empty_warehouse_01",
    "gear_store",
    "hotel_room",
    "industrial_sunset",
    "just_a_day",
    "kloppenheim_02",
    "lebombo",
    "loft_hall",
    "neon_photostudio",
    "old_bus_depot",
    "paul_lobe_haus",
    "photo_studio_01",
    "pump_station",
    "small_harbor_sunset",
    "studio_small_06",
    "sunset_jhbcentral",
    "urban_alley_01",
    "venetian_crossroads",
    "wasteland_clouds_puresky",
    "wrestling_gym",
]

# Texture asset IDs from ambientCG - relevant for table/floor surfaces
CC_TEXTURE_IDS = [
    "Wood051",
    "Wood054",
    "Wood067",
    "Wood078",
    "WoodFloor041",
    "WoodFloor043",
    "WoodFloor050",
    "WoodFloor052",
    "PaintedWood004",
    "PaintedWood007",
    "Concrete034",
    "Concrete040",
    "Concrete048",
    "Tiles074",
    "Tiles087",
    "Tiles101",
    "Carpet004",
    "Carpet008",
    "Fabric045",
    "Fabric062",
    "Leather026",
    "Leather037",
    "Metal032",
    "Metal038",
    "PaintedPlaster017",
    "Plaster003",
    "Marble006",
    "Marble013",
    "Ground054",
    "Terrazzo007",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (BlenderProc asset downloader)"}
HDRI_RESOLUTION = "1k"
TEXTURE_RESOLUTION = "2K-JPG"


def download_file(url: str, output_path: str, timeout: int = 60):
    """Download a file from URL to output_path."""
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)


def download_hdri(item_id: str, output_dir: Path):
    """Download a single HDRI from Poly Haven."""
    item_dir = output_dir / item_id
    # Check if already downloaded
    if item_dir.exists() and any(item_dir.glob("*.hdr")):
        return f"  [skip] {item_id} (already exists)"

    item_dir.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(
            f"https://api.polyhaven.com/files/{item_id}",
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if "hdri" not in data or HDRI_RESOLUTION not in data["hdri"]:
            return f"  [warn] {item_id}: resolution {HDRI_RESOLUTION} not available"

        url = data["hdri"][HDRI_RESOLUTION]["hdr"]["url"]
        filename = url.split("/")[-1]
        download_file(url, str(item_dir / filename))
        return f"  [ok]   {item_id}"
    except Exception as e:
        return f"  [err]  {item_id}: {e}"


def download_cc_texture(asset_id: str, output_dir: Path):
    """Download a single CC texture from ambientCG."""
    asset_dir = output_dir / asset_id

    # Check for existing texture files
    if asset_dir.exists():
        color_files = list(asset_dir.glob("*_Color.*")) + list(
            asset_dir.glob("*_Color.*")
        )
        if color_files:
            return f"  [skip] {asset_id} (already exists)"

    asset_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Get download info from ambientCG API
        url = (
            f"https://ambientcg.com/api/v2/full_json"
            f"?include=downloadData&id={asset_id}"
        )
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        json_data = resp.json()

        if "foundAssets" not in json_data or len(json_data["foundAssets"]) == 0:
            return f"  [warn] {asset_id}: not found on ambientCG"

        asset = json_data["foundAssets"][0]
        dl_cats = asset["downloadFolders"]["default"]["downloadFiletypeCategories"]
        downloads = dl_cats["zip"]["downloads"]

        # Find the 2K-JPG download
        download_link = None
        for dl in downloads:
            if dl.get("attribute") == TEXTURE_RESOLUTION:
                download_link = dl["downloadLink"]
                break

        if download_link is None:
            return f"  [warn] {asset_id}: {TEXTURE_RESOLUTION} not available"

        # Download and extract zip
        resp = requests.get(download_link, headers=HEADERS, timeout=120)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(str(asset_dir))

        os.unlink(tmp_path)
        return f"  [ok]   {asset_id}"
    except Exception as e:
        return f"  [err]  {asset_id}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Download HDRIs and CC textures")
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT, help="Output directory"
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="Download threads"
    )
    parser.add_argument(
        "--hdri-only", action="store_true", help="Only download HDRIs"
    )
    parser.add_argument(
        "--textures-only", action="store_true", help="Only download textures"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Download HDRIs
    if not args.textures_only:
        hdri_dir = output_dir / "hdris"
        hdri_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {len(HDRI_IDS)} HDRIs to {hdri_dir}/")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = {
                pool.submit(download_hdri, hid, hdri_dir): hid for hid in HDRI_IDS
            }
            for future in concurrent.futures.as_completed(futures):
                print(future.result())

        # Count successful downloads
        n = sum(1 for d in hdri_dir.iterdir() if d.is_dir() and any(d.glob("*.hdr")))
        print(f"HDRIs ready: {n}/{len(HDRI_IDS)}\n")

    # Download CC textures
    if not args.hdri_only:
        tex_dir = output_dir / "cctextures"
        tex_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {len(CC_TEXTURE_IDS)} CC textures to {tex_dir}/")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = {
                pool.submit(download_cc_texture, tid, tex_dir): tid
                for tid in CC_TEXTURE_IDS
            }
            for future in concurrent.futures.as_completed(futures):
                print(future.result())

        n = sum(1 for d in tex_dir.iterdir() if d.is_dir() and any(d.iterdir()))
        print(f"Textures ready: {n}/{len(CC_TEXTURE_IDS)}\n")

    print("Done!")


if __name__ == "__main__":
    main()
