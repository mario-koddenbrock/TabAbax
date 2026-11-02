"""Checks that attribution files are present, named in the metadata, and shipped.

The last group of tests builds a wheel and an sdist and opens the archives, so a
regression in the packaging configuration fails here instead of only in a release.
The archives are built once per session and reused. Set ``TABABAX_SKIP_BUILD=1``
to skip the build checks in environments without a network or build backend.
"""

import os
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

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

LICENSE_FILES = [
    "LICENSE",
    "NOTICE",
    "LICENSES/Apache-2.0.txt",
    "LICENSES/BSD-3-Clause.txt",
    "LICENSES/MIT.txt",
    "PROVENANCE.md",
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


def test_license_files_are_declared_in_project_metadata():
    # ``license-files`` is a PEP 639 field and only takes effect under ``[project]``.
    # Declaring it under ``[tool.hatch.build]`` reads as configured but is silently
    # ignored, which is how the wheel shipped without the per-license texts.
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = {name.replace("\\", "/") for name in project["license-files"]}
    for name in LICENSE_FILES:
        assert name in declared, name
    assert project["license"] == {"file": "LICENSE"}
    for classifier in (
        "License :: OSI Approved :: Apache Software License",
        "License :: OSI Approved :: BSD License",
        "License :: OSI Approved :: MIT License",
    ):
        assert classifier in project["classifiers"]


def test_license_files_exist_on_disk():
    for name in LICENSE_FILES:
        path = ROOT / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name


def test_license_index_is_not_apache_only():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "multiple licenses" in text
    assert "not a statement that" in text


def test_provenance_lists_inference_as_modified():
    """``inference.py`` carries a functional patch, so it is not a header-only match.

    A provenance sweep once used a local tree that was labelled as unmodified
    upstream as the baseline. That tree already contained the patch, so the sweep
    concluded the file matched. Keep this assertion so the claim cannot silently
    regress to "matches the reference version" again.
    """
    text = (ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
    modified = text.split("### Files that are modified derivatives")[1]
    assert "`inference.py`" in modified
    unmodified = text.split("### Files that are modified derivatives")[0]
    assert "`inference.py`" not in unmodified.split("### Files that match the reference version")[1]
    source = (ROOT / "src/tababax/_model/inference.py").read_text(encoding="utf-8")
    assert "PATCH BEGIN: auto downgrade helpers" in source


def test_reference_commit_is_not_claimed_as_the_copy_source():
    for name in ("PROVENANCE.md", "NOTICE"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3" in text
        # Wrapped prose, so compare on a whitespace-normalised copy.
        flat = " ".join(text.split()).lower()
        assert "not a record of" in flat, name


@pytest.fixture(scope="session")
def built_archives(tmp_path_factory):
    if os.environ.get("TABABAX_SKIP_BUILD"):
        pytest.skip("TABABAX_SKIP_BUILD is set")
    outdir = tmp_path_factory.mktemp("dist")
    try:
        subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(outdir)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("python-build is not installed")
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"building the distributions failed:\n{exc.stdout}\n{exc.stderr}")
    wheels = sorted(outdir.glob("*.whl"))
    sdists = sorted(outdir.glob("*.tar.gz"))
    assert wheels and sdists
    return wheels[-1], sdists[-1]


def test_wheel_ships_every_license_file(built_archives):
    wheel, _ = built_archives
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        licenses_root = next(
            name.split("/")[0] + "/licenses/" for name in names if ".dist-info/licenses/" in name
        )
        shipped = {
            name[len(licenses_root):]
            for name in names
            if name.startswith(licenses_root) and not name.endswith("/")
        }
        metadata = archive.read(
            next(name for name in names if name.endswith(".dist-info/METADATA"))
        ).decode("utf-8")
    for name in LICENSE_FILES:
        assert name in shipped, f"{name} missing from wheel, shipped: {sorted(shipped)}"
        assert f"License-File: {name}" in metadata, name


def test_sdist_ships_every_license_file(built_archives):
    _, sdist = built_archives
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
    prefix = names[0].split("/")[0] + "/"
    shipped = {name[len(prefix):] for name in names if name.startswith(prefix)}
    for name in LICENSE_FILES:
        assert name in shipped, f"{name} missing from sdist"
