"""Checks that attribution files are present and named in the package metadata."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DERIVED = [
    "src/tababax/_model/_cgroup_memory.py",
    "src/tababax/_model/attention.py",
    "src/tababax/_model/encoders.py",
    "src/tababax/_model/inference_config.py",
    "src/tababax/_model/layers.py",
    "src/tababax/_model/learning.py",
    "src/tababax/_model/quantile_dist.py",
    "src/tababax/_model/rope.py",
    "src/tababax/_model/ssmax.py",
    "src/tababax/_sklearn/sklearn_utils.py",
    "src/tababax/_torch_devices.py",
    "src/tababax/_model/embedding.py",
    "src/tababax/_model/inference.py",
    "src/tababax/_model/interaction.py",
    "src/tababax/_model/kv_cache.py",
    "src/tababax/_model/model.py",
    "src/tababax/_sklearn/base.py",
    "src/tababax/_sklearn/classifier.py",
    "src/tababax/_sklearn/preprocessing.py",
    "src/tababax/_sklearn/regressor.py",
]


def test_derived_files_name_tabicl():
    for rel in DERIVED:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "TabICL" in text, rel
        assert "BSD" in text, rel
        assert "soda-inria/tabicl" in text, rel


def test_notice_lists_third_parties():
    text = (ROOT / "NOTICE").read_text(encoding="utf-8")
    for phrase in (
        "Soda team @ Inria",
        "Phil Wang",
        "scikit-learn",
        "TabR",
        "PyTabKit",
        "1eb422d6c5",
        "75105013189c76bc4f247633c2fb856bc948e579",
    ):
        assert phrase in text


def test_sdist_keeps_license_files():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for name in ("NOTICE", "LICENSES", "PROVENANCE.md", "LICENSE"):
        assert name in text
    assert "License :: OSI Approved :: BSD License" in text
    assert "License :: OSI Approved :: MIT License" in text


def test_license_index_is_not_apache_only():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "multiple licenses" in text
    assert "not a statement that" in text
    for name in ("Apache-2.0.txt", "BSD-3-Clause.txt", "MIT.txt"):
        assert (ROOT / "LICENSES" / name).is_file()
