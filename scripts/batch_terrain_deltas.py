"""Batch-run the dG/dC pipeline over many independent pre/post DEM chunk pairs
(e.g. per-analysis-area tiles), matched by a shared numeric ID embedded in
each chunk's filename.

Generic -- not tied to any one site. Point --config at a yaml with:
    pre_dir:      directory of pre-event DEM chunks
    post_dir:     directory of post-event DEM chunks
    output_root:  directory to write <id>/ subfolders into
    alpha:        G = a * S^alpha  (default 1.167)
    id_pattern:   {pre: regex, post: regex}, each with one capture group = the ID
                  (default matches "<id>_Pre.tif" / "<id>_Post.tif")

Each ID is treated independently: chunks are not spatially related (they are
arbitrary buffered analysis areas, not a tile grid), so nothing is merged
across IDs. IDs present on only one side are skipped and logged, never guessed.

Reuses compute_terrain_deltas_1m.py's functions directly -- no duplicated
G/curvature logic, and that script is not modified by this one.

Run:
    conda run -n ml_debris python scripts/batch_terrain_deltas.py --config config/eagle_creek.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import whitebox
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import compute_terrain_deltas_1m as ctd  # noqa: E402

DEFAULT_PRE_PATTERN = r"^(\d+)_Pre\.tif$"
DEFAULT_POST_PATTERN = r"^(\d+)_Post\.tif$"


def find_ids(directory: Path, pattern: str) -> dict[str, Path]:
    regex = re.compile(pattern)
    found = {}
    for f in directory.iterdir():
        m = regex.match(f.name)
        if m:
            found[m.group(1)] = f
    return found


def run_pair(wbt, pre_dem: Path, post_dem: Path, out_dir: Path, alpha: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    G_pre, profile = ctd.compute_g_dinf(wbt, pre_dem, out_dir, "pre", alpha)
    G_post, _ = ctd.compute_g_dinf(wbt, post_dem, out_dir, "post", alpha)
    DeltaG = G_post - G_pre
    ctd.write_raster(out_dir / "DeltaG_1m_dinf.tif", DeltaG, profile)

    curv_pre, cprofile = ctd.compute_curvature(wbt, pre_dem, out_dir, "pre")
    curv_post, _ = ctd.compute_curvature(wbt, post_dem, out_dir, "post")
    DeltaC = curv_post - curv_pre
    ctd.write_raster(out_dir / "DeltaC_1m.tif", DeltaC, cprofile)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated list of IDs to run (e.g. 19 or 19,20,21). "
             "Default: run every matched ID.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pre_dir = Path(cfg["pre_dir"])
    post_dir = Path(cfg["post_dir"])
    out_root = Path(cfg["output_root"])
    alpha = float(cfg.get("alpha", 1.167))
    patterns = cfg.get("id_pattern", {})
    pre_pattern = patterns.get("pre", DEFAULT_PRE_PATTERN)
    post_pattern = patterns.get("post", DEFAULT_POST_PATTERN)

    pre_ids = find_ids(pre_dir, pre_pattern)
    post_ids = find_ids(post_dir, post_pattern)

    common = sorted(set(pre_ids) & set(post_ids), key=int)
    pre_only = sorted(set(pre_ids) - set(post_ids), key=int)
    post_only = sorted(set(post_ids) - set(pre_ids), key=int)

    print(f"Pre chunks : {len(pre_ids)}  ({pre_dir})")
    print(f"Post chunks: {len(post_ids)}  ({post_dir})")
    print(f"Matched IDs: {len(common)} -> {common}")
    if pre_only:
        print(f"SKIP (pre only, no post match): {pre_only}")
    if post_only:
        print(f"SKIP (post only, no pre match): {post_only}")

    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        missing = wanted - set(common)
        if missing:
            raise SystemExit(f"--only requested IDs not in matched set: {sorted(missing)}")
        common = [c for c in common if c in wanted]
        print(f"--only filter applied -> running: {common}")

    print()
    out_root.mkdir(parents=True, exist_ok=True)

    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False

    for i, chunk_id in enumerate(common, 1):
        pre_dem = pre_ids[chunk_id]
        post_dem = post_ids[chunk_id]
        out_dir = out_root / chunk_id

        print(f"[{i}/{len(common)}] id={chunk_id}  pre={pre_dem.name}  post={post_dem.name}")
        run_pair(wbt, pre_dem, post_dem, out_dir, alpha)
        print(f"    done -> {out_dir}")

    print(f"\nAll done. {len(common)} chunk(s) written under {out_root}")


if __name__ == "__main__":
    main()
