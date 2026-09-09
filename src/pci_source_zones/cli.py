#!/usr/bin/env python
"""Single entry point for the source-zone evidence pipeline.

One command for one site or many analysis areas; `--ids` is the only difference.

Examples
--------
Whole pipeline for a single site:

    pci-zones run --config config/sites/montecito_2016_2018.yaml

Selected stages, selected analysis areas, forcing recomputation:

    pci-zones run --config config/sites/eagle_creek_2018_2021.yaml \
        --stages evidence,p_obs --ids 1-5,12 --overwrite

Alignment is a stage, not a separate command. A site config with an `align:`
section aligns its predictor rasters first, then runs the rest:

    pci-zones run --config config/sites/montecito_2016_2018.yaml --stages align

Validate a config without running anything:

    pci-zones validate --config config/sites/eaton_2016_2025.yaml

List the analysis areas a config resolves to:

    pci-zones list --config config/sites/eagle_creek_2018_2021.yaml
"""
from __future__ import annotations

import argparse
import sys

import yaml

from . import runner
from .align import AlignError
from .geometry import parse_ids
from .site_config import ConfigError, SiteConfig


def load_cfg(args) -> SiteConfig:
    overrides = None
    if args.experiment:
        with open(args.experiment) as f:
            overrides = yaml.safe_load(f) or {}
        print(f"experiment override: {args.experiment}")
    return SiteConfig.load(args.config, overrides)


def cmd_run(args) -> int:
    cfg = load_cfg(args)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    results = runner.run(cfg, stages, parse_ids(args.ids),
                         overwrite=args.overwrite, fail_fast=args.fail_fast)
    runner.write_summary(cfg, results, stages)
    return 1 if any(r.status == "failed" for r in results) else 0


def cmd_validate(args) -> int:
    cfg = load_cfg(args)
    print(f"config OK: {cfg.site}")
    print(f"  mode        : {cfg.inputs.mode}")
    print(f"  grid        : {cfg.grid.fine_res_m} m -> {cfg.grid.coarse_res_m} m "
          f"(block {cfg.block_size})")
    print(f"  domain      : G in [{cfg.evidence.domain.g_min}, "
          f"{cfg.evidence.domain.g_channel})")
    c3 = cfg.evidence.c3
    how = "auto (measured on stable ground)" if c3.is_auto else (
        f"raster {c3.raster_path} x {c3.scale}" if c3.raster_path
        else f"constant {c3.lod} m")
    print(f"  c3 lod      : {how}")
    print(f"  terrain     : dinf, alpha={cfg.terrain.alpha}")
    if cfg.stable:
        flag = "  [CIRCULAR]" if cfg.stable.method == "threshold" else ""
        print(f"  stable      : {cfg.stable.method}, {len(cfg.stable.layers)} "
              f"layer(s), combine={cfg.stable.combine}{flag}")
    else:
        print("  stable      : not configured (inputs.stable_cells used directly)")
    print(f"  lod         : {cfg.lod.method} x {cfg.lod.scale}")
    acfg = cfg.align_config()
    print(f"  align       : {len(acfg.layers)} layer(s)" if acfg else
          "  align       : not configured")
    print(f"  output root : {cfg.output_root()}")
    return 0


def cmd_list(args) -> int:
    cfg = load_cfg(args)
    ids = runner.discover_areas(cfg, parse_ids(args.ids))
    if not cfg.is_batch:
        print(f"single site: {cfg.site}")
    else:
        print(f"{len(ids)} analysis area(s):")
        for i in ids:
            print(f"  {i}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pci-zones", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--config", required=True, help="site config YAML")
        sp.add_argument("--experiment", default=None,
                        help="override YAML, deep-merged over the site config")
        sp.add_argument("--ids", default=None,
                        help="analysis areas, e.g. '1,2,5' or '1-4,9'. "
                             "Omit to process all.")

    r = sub.add_parser("run", help="execute pipeline stages")
    add_common(r)
    r.add_argument("--stages", default=",".join(runner.STAGES),
                   help=f"comma-separated, from {list(runner.STAGES)}")
    r.add_argument("--overwrite", action="store_true",
                   help="recompute even when outputs exist")
    r.add_argument("--fail-fast", action="store_true",
                   help="stop at the first failure instead of isolating it")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("validate", help="parse and report a config, run nothing")
    add_common(v)
    v.set_defaults(func=cmd_validate)

    l = sub.add_parser("list", help="list the analysis areas a config resolves to")
    add_common(l)
    l.set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, AlignError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
