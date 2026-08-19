import pytest

from pci_source_zones.ml.unet_tuning import apply_trial, build_trials


def test_build_trials_grid_is_full_product():
    space = {"patch_size": [64, 128], "overlap": [0.5, 0.75]}
    trials = build_trials(space, {"search_method": "grid_search"})

    assert len(trials) == 4
    assert {"patch_size": 64, "overlap": 0.5} in trials
    assert {"patch_size": 128, "overlap": 0.75} in trials


def test_build_trials_random_subsamples_and_is_seeded():
    space = {"patch_size": [32, 64, 128], "overlap": [0.25, 0.5, 0.75]}
    cfg = {"search_method": "random_search", "n_iter": 4, "random_state": 7}

    first = build_trials(space, cfg)
    second = build_trials(space, cfg)

    assert len(first) == 4
    assert first == second  # same seed -> same trials


def test_build_trials_random_caps_at_grid_size():
    space = {"patch_size": [64, 128]}
    trials = build_trials(space, {"search_method": "random_search", "n_iter": 99})
    assert len(trials) == 2


def test_build_trials_rejects_empty_space():
    with pytest.raises(ValueError, match="search_space"):
        build_trials({}, {})


def test_apply_trial_routes_params_to_correct_section():
    cfg = {
        "ml": {
            "model": {"type": "unet", "attention": True},
            "unet": {"epochs": 100, "patch_size": 128},
        }
    }
    out = apply_trial(cfg, {"base_filters": 32, "patch_size": 64, "overlap": 0.75})

    # architecture params -> ml.model
    assert out["ml"]["model"]["base_filters"] == 32
    assert out["ml"]["model"]["attention"] is True  # untouched keys survive
    # training/geometry params -> ml.unet
    assert out["ml"]["unet"]["patch_size"] == 64
    assert out["ml"]["unet"]["overlap"] == 0.75
    assert out["ml"]["unet"]["epochs"] == 100


def test_apply_trial_does_not_mutate_input():
    cfg = {"ml": {"model": {"type": "unet"}, "unet": {"patch_size": 128}}}
    apply_trial(cfg, {"patch_size": 64, "base_filters": 32})

    assert cfg["ml"]["unet"]["patch_size"] == 128
    assert "base_filters" not in cfg["ml"]["model"]
