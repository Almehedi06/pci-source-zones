import json
from pathlib import Path

from pci_source_zones.ml.data_contract import ContractReport, RasterFingerprint
from pci_source_zones.ml.provenance import new_run_id, write_run_manifest


def test_new_run_id_shape():
    run_id = new_run_id()
    parts = run_id.split("_")
    assert len(parts) == 3  # YYYYMMDD, HHMMSS, git-hash-or-"nogit"
    assert len(parts[0]) == 8
    assert len(parts[1]) == 6
    assert parts[2]  # non-empty


def test_write_run_manifest_with_data_report(tmp_path: Path):
    cfg = {"ml": {"model": {"type": "random_forest"}}}
    report = ContractReport(
        reference="slope",
        rasters=[RasterFingerprint(name="slope", path="/tmp/slope.tif", size_bytes=100, mtime=123.0)],
    )
    run_id = "20260101_000000_abcdef"

    path = write_run_manifest(cfg, tmp_path, run_id, report)

    assert path.exists()
    manifest = json.loads(path.read_text())
    assert manifest["run_id"] == run_id
    assert manifest["config"] == cfg
    assert "git" in manifest
    assert "package_versions" in manifest
    assert manifest["data_contract"]["reference"] == "slope"
    assert manifest["data_contract"]["rasters"][0]["name"] == "slope"


def test_write_run_manifest_without_data_report(tmp_path: Path):
    path = write_run_manifest({"ml": {}}, tmp_path, "run1", None)
    manifest = json.loads(path.read_text())
    assert manifest["data_contract"] is None
