"""Hyperparameter search for the UNet.

Deliberately simpler than tuning.py's sklearn machinery. The tabular models
search a large space cheaply; a UNet trial costs minutes on a GPU and there
are only a handful of hyperparameters that matter at this data size, so this
runs an explicit grid (or a random sample of it) rather than a Bayesian
optimiser whose extra sophistication would be wasted.

Selection uses the same leave-one-polygon-out validation the training
workflow uses — a random patch split would rank configs by how well they
exploit spatial autocorrelation. The test polygons are never touched here.

Cost control: raster I/O and the pixel-level patch mask are computed once
and reused across every trial and fold. Only the patch datasets (which
depend on patch_size/overlap) and the model itself are rebuilt per trial.
"""
from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .dataset import build_ml_dataset
from .features import build_feature_stack, configured_feature_names
from .models.unet import build_unet
from .outputs import ml_output_dir, write_json
from .patch_dataset import (
    PatchDataset,
    assign_patch_groups,
    compute_norm_stats,
    leave_one_polygon_out_indices,
    normalize_features,
    pad_like_patches,
    stack_feature_arrays,
)
from .preflight import run_preflight
from .provenance import new_run_id, write_run_manifest
from .splits import polygon_group_raster
from .targets import build_target
from .unet_workflow import select_patch_pixels

# Which config section each searchable hyperparameter belongs to.
MODEL_KEYS = {"base_filters", "depth", "dropout", "attention"}


@dataclass
class TrialResult:
    params: dict[str, Any]
    val_loss: float
    val_r2: float
    fold_losses: list[float] = field(default_factory=list)
    fold_r2s: list[float] = field(default_factory=list)
    n_train_patches: int = 0
    n_val_patches: int = 0
    epochs_run: int = 0
    error: str | None = None


def build_trials(search_space: dict[str, list[Any]], tuning_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the search space into concrete trials."""
    if not search_space:
        raise ValueError("Set ml.tuning.search_space with at least one hyperparameter.")

    names = sorted(search_space)
    grid = [dict(zip(names, values)) for values in itertools.product(*(search_space[n] for n in names))]

    method = str(tuning_cfg.get("search_method", "grid_search")).lower()
    if method in {"grid", "grid_search"}:
        return grid

    n_iter = int(tuning_cfg.get("n_iter", 10))
    if n_iter >= len(grid):
        return grid
    rng = np.random.default_rng(int(tuning_cfg.get("random_state", 42)))
    picked = rng.choice(len(grid), size=n_iter, replace=False)
    return [grid[i] for i in sorted(picked)]


def apply_trial(cfg: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of cfg with one trial's hyperparameters applied."""
    ml = {**cfg.get("ml", {})}
    model_cfg = {**ml.get("model", {})}
    unet_cfg = {**(ml.get("unet") or {})}

    for key, value in params.items():
        if key in MODEL_KEYS:
            model_cfg[key] = value
        else:
            unet_cfg[key] = value

    ml["model"] = model_cfg
    ml["unet"] = unet_cfg
    return {**cfg, "ml": ml}


def _fold_labels(tuning_cfg: dict[str, Any], ids: list[Any]) -> list[int]:
    """1-based polygon labels to use as validation folds."""
    cv_cfg = tuning_cfg.get("cv", {}) or {}
    requested = cv_cfg.get("val_polygon_ids")
    if requested:
        missing = [r for r in requested if r not in ids]
        if missing:
            raise ValueError(f"cv.val_polygon_ids {missing} are not train polygon ids {ids}.")
        return [ids.index(r) + 1 for r in requested]

    n_splits = int(cv_cfg.get("n_splits", 1))
    return list(range(1, min(n_splits, len(ids)) + 1))


def _patch_val_metrics(model: Any, dataset: Any, device: Any, batch_size: int) -> float:
    """Pixel-level R² over the held-out patches (valid pixels only)."""
    import torch
    from torch.utils.data import DataLoader

    if len(dataset) == 0:
        return float("nan")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, truths = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y, valid = batch[0], batch[1], batch[2]
            out = model(x.to(device))
            mu = out[:, 0] if out.shape[1] > 1 else out.squeeze(1)
            mu = mu.cpu().numpy()
            y_np, v_np = y.numpy(), valid.numpy().astype(bool)
            preds.append(mu[v_np])
            truths.append(y_np[v_np])

    y_pred = np.concatenate(preds) if preds else np.array([])
    y_true = np.concatenate(truths) if truths else np.array([])
    if y_true.size == 0:
        return float("nan")

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def run_unet_tuning(cfg: dict[str, Any]) -> dict[str, Any]:
    """Grid/random search over UNet hyperparameters with polygon-grouped CV."""
    from .unet_train import _get_device, train_unet

    cfg, data_report = run_preflight(cfg)
    run_id = new_run_id()
    out_dir = ml_output_dir(cfg, "unet_tuned", run_id=run_id)
    write_run_manifest(cfg, out_dir, run_id, data_report)

    tuning_cfg = cfg.get("ml", {}).get("tuning", {}) or {}
    base_unet_cfg = cfg.get("ml", {}).get("unet", {}) or {}
    target_cfg = cfg.get("ml", {}).get("target", {})
    is_regression = str(target_cfg.get("type", "")).lower() == "raster_continuous"
    tobit_mode = bool(cfg.get("ml", {}).get("model", {}).get("tobit", False))
    nodata_val = -9999 if is_regression else int(target_cfg.get("nodata", 255))

    if tobit_mode:
        raise ValueError(
            "UNet tuning does not support tobit mode (the approach was retired). "
            "Set ml.model.tobit: false."
        )

    # ---- Load everything that does not vary across trials, once ----
    target_data = build_target(cfg)
    shape = target_data.target.shape
    feature_stack = build_feature_stack(cfg, reference_shape=shape)
    numeric, categorical = configured_feature_names(cfg)
    feature_names = numeric + categorical
    features_2d = stack_feature_arrays(feature_stack.arrays, feature_names)
    data = build_ml_dataset(cfg)

    train_mask_2d, patch_indices = select_patch_pixels(
        cfg, target_data, data, shape, tobit_mode=False, censored_arr=None
    )

    grouped = polygon_group_raster(
        cfg, target_data.profile, shape, cfg.get("ml", {}).get("split", {})
    )
    if grouped is None:
        raise ValueError(
            "UNet tuning requires ml.split.method: polygons with polygons.path and train_ids "
            "so validation folds are spatially separated."
        )
    group_raster, ids = grouped
    folds = _fold_labels(tuning_cfg, ids)

    trials = build_trials(tuning_cfg.get("search_space", {}), tuning_cfg)
    max_epochs = tuning_cfg.get("max_epochs")
    device = _get_device(base_unet_cfg)

    print(f"[unet-tune] {len(trials)} trials x {len(folds)} fold(s) on {device}")
    print(f"[unet-tune] folds hold out polygon id(s): {[ids[f - 1] for f in folds]}")
    print(f"[unet-tune] output: {out_dir}")

    results: list[TrialResult] = []
    for t_i, params in enumerate(trials, start=1):
        print(f"\n[unet-tune] trial {t_i}/{len(trials)}: {params}")
        trial_cfg = apply_trial(cfg, params)
        if max_epochs is not None:
            trial_cfg["ml"]["unet"]["epochs"] = int(max_epochs)

        fold_losses, fold_r2s = [], []
        n_train = n_val = epochs_run = 0
        error: str | None = None

        for fold in folds:
            try:
                loss, r2, nt, nv, ep = _run_one_fold(
                    trial_cfg,
                    features_2d,
                    target_data,
                    shape,
                    patch_indices,
                    train_mask_2d,
                    group_raster,
                    fold,
                    nodata_val,
                    is_regression,
                    out_dir,
                    device,
                    train_unet,
                )
            except Exception as exc:  # a bad HP combo must not kill the sweep
                error = f"{type(exc).__name__}: {exc}"
                print(f"  fold {ids[fold - 1]}: FAILED — {error}")
                break
            fold_losses.append(loss)
            fold_r2s.append(r2)
            n_train, n_val, epochs_run = nt, nv, ep
            print(f"  fold (held out id={ids[fold - 1]}): val_loss={loss:.4f} val_R2={r2:.4f} "
                  f"(train={nt}, val={nv}, epochs={ep})")

        results.append(
            TrialResult(
                params=params,
                val_loss=float(np.mean(fold_losses)) if fold_losses else float("inf"),
                val_r2=float(np.mean(fold_r2s)) if fold_r2s else float("nan"),
                fold_losses=fold_losses,
                fold_r2s=fold_r2s,
                n_train_patches=n_train,
                n_val_patches=n_val,
                epochs_run=epochs_run,
                error=error,
            )
        )

    ranked = sorted(results, key=lambda r: (r.error is not None, r.val_loss))
    best = ranked[0] if ranked else None

    results_path = _write_results_csv(out_dir / "unet_tuning_results.csv", ranked, ids, folds)
    summary_path = write_json(
        out_dir / "unet_tuning_summary.json",
        {
            "n_trials": len(trials),
            "folds_held_out": [ids[f - 1] for f in folds],
            "selection_metric": "mean val_loss over folds (lower is better)",
            "best_params": best.params if best else None,
            "best_val_loss": best.val_loss if best else None,
            "best_val_r2": best.val_r2 if best else None,
        },
    )
    best_cfg_path = None
    if best is not None and best.error is None:
        best_cfg_path = _write_best_config(cfg, best.params, out_dir / "unet_best.yaml")

    if best is not None:
        print(f"\n[unet-tune] best: {best.params}")
        print(f"[unet-tune]   mean val_loss={best.val_loss:.4f}  mean val_R2={best.val_r2:.4f}")

    return {
        "run_id": run_id,
        "output_dir": out_dir,
        "n_trials": len(trials),
        "folds_held_out": [ids[f - 1] for f in folds],
        "best_params": best.params if best else None,
        "best_val_loss": best.val_loss if best else None,
        "best_val_r2": best.val_r2 if best else None,
        "results": results_path,
        "summary": summary_path,
        "best_config": best_cfg_path,
    }


def _run_one_fold(
    trial_cfg: dict[str, Any],
    features_2d: np.ndarray,
    target_data: Any,
    shape: tuple[int, int],
    patch_indices: np.ndarray,
    train_mask_2d: np.ndarray,
    group_raster: np.ndarray,
    fold: int,
    nodata_val: int,
    is_regression: bool,
    out_dir: Path,
    device: Any,
    train_unet: Any,
) -> tuple[float, float, int, int, int]:
    from torch.utils.data import Subset

    unet_cfg = trial_cfg["ml"]["unet"]
    patch_size = int(unet_cfg.get("patch_size", 128))
    overlap = float(unet_cfg.get("overlap", 0.5))
    stride = max(1, int(patch_size * (1.0 - overlap)))

    # Locations first, so the fold's train pixels are known before normalising.
    probe = PatchDataset.from_data(
        np.zeros((1, *shape), dtype="float32"), target_data.target, patch_indices, shape,
        patch_size=patch_size, stride=stride, nodata=nodata_val, augment=False,
    )
    padded_groups = pad_like_patches(group_raster, shape, patch_size, fill=0)
    dominant, touches = assign_patch_groups(probe.locations, patch_size, padded_groups)
    train_idx, val_idx = leave_one_polygon_out_indices(dominant, touches, fold)
    if not train_idx or not val_idx:
        raise ValueError(
            f"fold produced train={len(train_idx)}, val={len(val_idx)} patches at "
            f"patch_size={patch_size}, overlap={overlap}"
        )

    # Normalise on this fold's training pixels only — using the whole train
    # region would fold the held-out polygon's statistics into the inputs.
    fold_train_mask = train_mask_2d & (group_raster != fold)
    norm_stats = compute_norm_stats(features_2d, fold_train_mask)
    features_norm = normalize_features(features_2d, norm_stats)

    train_full = PatchDataset.from_data(
        features_norm, target_data.target, patch_indices, shape,
        patch_size=patch_size, stride=stride, nodata=nodata_val, augment=True,
    )
    val_full = PatchDataset.from_data(
        features_norm, target_data.target, patch_indices, shape,
        patch_size=patch_size, stride=stride, nodata=nodata_val, augment=False,
    )
    train_ds, val_ds = Subset(train_full, train_idx), Subset(val_full, val_idx)

    model = build_unet(
        trial_cfg["ml"]["model"], in_channels=features_2d.shape[0], regression=is_regression
    )
    model, history = train_unet(
        model, train_ds, val_ds, trial_cfg, out_dir, is_regression=is_regression, tobit_mode=False
    )

    best_loss = min((h["val_loss"] for h in history), default=float("inf"))
    val_r2 = _patch_val_metrics(model, val_ds, device, int(unet_cfg.get("batch_size", 16)))
    return best_loss, val_r2, len(train_idx), len(val_idx), len(history)


def _write_results_csv(
    path: Path, ranked: list[TrialResult], ids: list[Any], folds: list[int]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    param_names = sorted({k for r in ranked for k in r.params})
    fold_names = [f"fold_{ids[f - 1]}_val_loss" for f in folds]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["rank", *param_names, "mean_val_loss", "mean_val_r2",
             *fold_names, "n_train_patches", "n_val_patches", "epochs", "error"]
        )
        for rank, r in enumerate(ranked, start=1):
            fold_cols = [f"{v:.6f}" for v in r.fold_losses] + [""] * (len(folds) - len(r.fold_losses))
            writer.writerow(
                [rank, *[r.params.get(n, "") for n in param_names],
                 f"{r.val_loss:.6f}", f"{r.val_r2:.6f}", *fold_cols,
                 r.n_train_patches, r.n_val_patches, r.epochs_run, r.error or ""]
            )
    return path


def _write_best_config(cfg: dict[str, Any], best_params: dict[str, Any], path: Path) -> Path:
    """Write a ready-to-run training config with the winning HPs applied."""
    out = apply_trial(cfg, best_params)
    out["ml"].pop("tuning", None)
    out["ml"]["output_subdir"] = "source_area_workflow/ml/unet_best"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"[unet-tune] best config written → {path}")
    return path
