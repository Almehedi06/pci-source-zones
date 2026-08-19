import pytest
from rasterio.transform import from_origin

from pci_source_zones.ml.features import _check_transform_alignment


def _profile(origin_x: float = 0.0) -> dict:
    return {"transform": from_origin(origin_x, 100, 10, 10)}


def test_check_transform_alignment_passes_when_grids_match():
    reference = _profile()
    feature = _profile()
    _check_transform_alignment("slope", feature, reference)  # must not raise


def test_check_transform_alignment_raises_on_misaligned_grid():
    reference = _profile(origin_x=0.0)
    feature = _profile(origin_x=5.0)  # same shape, shifted half a pixel

    with pytest.raises(ValueError, match="transform mismatch"):
        _check_transform_alignment("slope", feature, reference)
