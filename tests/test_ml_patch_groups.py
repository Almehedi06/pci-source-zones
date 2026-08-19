import numpy as np

from pci_source_zones.ml.patch_dataset import (
    assign_patch_groups,
    leave_one_polygon_out_indices,
    pad_like_patches,
)


def _two_polygon_raster() -> np.ndarray:
    """20x20 grid: group 1 = top half, group 2 = bottom half."""
    g = np.zeros((20, 20), dtype="int32")
    g[:10, :] = 1
    g[10:, :] = 2
    return g


def test_assign_patch_groups_dominant_and_touches():
    group_raster = _two_polygon_raster()
    # patch_size 10: patch at (0,0) is all group 1, (10,0) all group 2,
    # (5,0) straddles both.
    locations = [(0, 0), (10, 0), (5, 0)]

    dominant, touches = assign_patch_groups(locations, 10, group_raster)

    assert dominant[0] == 1
    assert dominant[1] == 2
    assert touches[0].tolist() == [False, True, False]  # only group 1
    assert touches[1].tolist() == [False, False, True]  # only group 2
    assert touches[2, 1] and touches[2, 2]  # straddling patch touches both


def test_leave_one_polygon_out_excludes_straddlers_from_train():
    group_raster = _two_polygon_raster()
    locations = [(0, 0), (10, 0), (5, 0)]  # group1, group2, straddler
    dominant, touches = assign_patch_groups(locations, 10, group_raster)

    train_idx, val_idx = leave_one_polygon_out_indices(dominant, touches, held_out_group=2)

    assert val_idx == [1]  # the group-2 patch validates
    assert train_idx == [0]  # only the pure group-1 patch trains
    # The straddler (index 2) is in neither — it contains held-out pixels, so
    # training on it would leak the validation polygon.
    assert 2 not in train_idx and 2 not in val_idx


def test_no_train_patch_touches_the_held_out_polygon():
    group_raster = _two_polygon_raster()
    locations = [(r, c) for r in range(0, 11, 5) for c in range(0, 11, 5)]
    dominant, touches = assign_patch_groups(locations, 10, group_raster)

    train_idx, val_idx = leave_one_polygon_out_indices(dominant, touches, held_out_group=1)

    assert val_idx, "expected some validation patches"
    for i in train_idx:
        assert not touches[i, 1], "a training patch overlaps the held-out polygon"


def test_pad_like_patches_matches_patchdataset_padding():
    arr = np.ones((15, 15), dtype="int32")
    padded = pad_like_patches(arr, (15, 15), patch_size=10, fill=0)

    # 15 -> next multiple of 10 is 20
    assert padded.shape == (20, 20)
    assert padded[:15, :15].sum() == 225
    assert padded[15:, :].sum() == 0  # padding is "outside any polygon"


def test_pad_like_patches_noop_when_already_multiple():
    arr = np.ones((20, 20), dtype="int32")
    assert pad_like_patches(arr, (20, 20), patch_size=10).shape == (20, 20)
