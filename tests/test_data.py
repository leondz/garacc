# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import random
import tempfile
import os

from pathlib import Path
from garak import _config
from garak.exception import GarakException
from garak.data import path as data_path
from garak.data import LocalDataPath


@pytest.fixture
def random_resource_filename(request) -> None:
    with tempfile.NamedTemporaryFile(
        dir=LocalDataPath.ORDERED_SEARCH_PATHS[-1], mode="w", delete=False
    ) as tmpfile:
        tmpfile.write("file data")

    def remove_files():
        for path in LocalDataPath.ORDERED_SEARCH_PATHS:
            rem_path = path / os.path.basename(tmpfile.name)
            if rem_path.exists():
                rem_path.unlink()

    request.addfinalizer(remove_files)

    return os.path.basename(tmpfile.name)


@pytest.mark.parametrize(
    "test_paths",
    [
        [".."],  # parent of data_path
        ["autodan", "..", "..", "configs"],  # exists but outside allowed paths
        [
            "autodan",
            "../../configs",
        ],  # exists outside allowed paths using concatenated segment
        ["../.."],  # outside data_path using concatenated segment
    ],
)
def test_no_relative_escape_joined_path(test_paths):
    target_path = data_path
    with pytest.raises(GarakException) as exc_info:
        for p in test_paths:
            target_path / p
    assert "does not refer to a valid path" in str(exc_info.value)


def test_allow_relative_in_path():
    source = data_path / "autodan" / ".." / "gcg"
    assert source.name == "gcg"


def test_known_resource_found():
    known_filename = "tags.misp.tsv"
    source = data_path / known_filename
    assert source.name == known_filename


def test_local_override(random_resource_filename):
    source = data_path / random_resource_filename
    assert _config.transient.package_dir in source.parents

    data_root_path = _config.transient.data_dir / "data"
    data_root_path.mkdir(parents=True, exist_ok=True)
    with open(
        data_root_path / random_resource_filename, encoding="utf-8", mode="w"
    ) as f:
        f.write("fake data")

    source = data_path / random_resource_filename
    assert _config.transient.data_dir in source.parents


@pytest.fixture
def random_file_tree(request) -> None:
    files = []
    temp_dir = tempfile.mkdtemp(dir=LocalDataPath.ORDERED_SEARCH_PATHS[-1])
    temp_dirname = os.path.basename(temp_dir)
    temp_dir = Path(temp_dir)
    data_dir = LocalDataPath.ORDERED_SEARCH_PATHS[0] / temp_dirname
    data_dir.mkdir()
    testing_temp_dir = temp_dir / "testing"
    testing_temp_dir.mkdir()
    testing_data_dir = data_dir / "testing"
    testing_data_dir.mkdir()

    for i in range(random.randint(1, 10)):
        with tempfile.NamedTemporaryFile(
            dir=testing_temp_dir, suffix=".test", mode="w", delete=False
        ) as tmpfile:
            tmpfile.write("file data")
            files.append(os.path.basename(tmpfile.name))

    override_files = []
    for i in range(random.randint(1, len(files))):
        with open(testing_data_dir / files[i], mode="w") as over_file:
            over_file.write("override data")
            override_files.append(os.path.basename(over_file.name))

    def remove_files():
        for path in LocalDataPath.ORDERED_SEARCH_PATHS:
            for file in files:
                rem_path = path / temp_dirname / "testing" / os.path.basename(file)
                if rem_path.exists():
                    rem_path.unlink()
            rem_path.parent.rmdir()
            rem_path.parent.parent.rmdir()

    request.addfinalizer(remove_files)

    return (temp_dirname, files, override_files)


@pytest.fixture
def symlinked_payload_tree(request) -> None:
    """Simulate a symlinked install (e.g. UV_LINK_MODE=symlink) where files
    under the package data dir are symlinks whose targets live outside any
    ORDERED_SEARCH_PATHS (the package installer's cache)."""
    # external target dir, outside every search path (mimics the UV cache)
    cache_dir = Path(tempfile.mkdtemp())

    # directory under the package data search path holding the symlinks
    pkg_data_dir = Path(tempfile.mkdtemp(dir=LocalDataPath.ORDERED_SEARCH_PATHS[-1]))
    dirname = pkg_data_dir.name

    filenames = []
    for i in range(5):
        target = cache_dir / f"payload_{i}.json"
        with open(target, mode="w", encoding="utf-8") as f:
            f.write(f'{{"payloads": ["p{i}"]}}')
        link = pkg_data_dir / f"payload_{i}.json"
        os.symlink(target, link)
        filenames.append(link.name)

    def remove_files():
        for link in pkg_data_dir.glob("*.json"):
            link.unlink()
        pkg_data_dir.rmdir()
        for target in cache_dir.glob("*.json"):
            target.unlink()
        cache_dir.rmdir()

    request.addfinalizer(remove_files)

    return dirname, filenames


def test_determine_suffix_symlinked_install(symlinked_payload_tree):
    """A symlinked file (target outside search paths) must still resolve to its
    correct suffix under the package data dir, not None."""
    dirname, filenames = symlinked_payload_tree
    link = LocalDataPath.ORDERED_SEARCH_PATHS[-1] / dirname / filenames[0]
    suffix = LocalDataPath(link)._determine_suffix()
    assert suffix == Path(dirname) / filenames[0]


def test_direct_access_symlinked_install(symlinked_payload_tree):
    """Direct '/'-operator access to a symlinked file (target outside search
    paths) must resolve to the file under the package data dir, not raise. The
    pre-fix bug used .resolve() in _eval_paths, following the symlink outside
    ORDERED_SEARCH_PATHS and raising GarakException."""
    dirname, filenames = symlinked_payload_tree
    source = data_path / dirname / filenames[0]
    assert source.name == filenames[0]
    assert LocalDataPath.ORDERED_SEARCH_PATHS[-1] in source.parents


def test_glob_symlinked_install_returns_all(symlinked_payload_tree):
    """Globbing a dir of symlinked files must return every file. The pre-fix
    bug resolved all symlinks outside the search paths, yielding suffix=None
    for each, so the _glob dedup collapsed them to a single file."""
    dirname, filenames = symlinked_payload_tree
    glob_files = (data_path / dirname).glob("*.json")
    assert len(glob_files) == len(filenames)
    assert {f.name for f in glob_files} == set(filenames)


def test_consolidated_glob(random_file_tree):
    dirname, files, override_files = random_file_tree
    glob_files = (data_path / dirname / "testing").glob("*.test")
    found_override_files = []
    for file in glob_files:
        if LocalDataPath.ORDERED_SEARCH_PATHS[0] in file.parents:
            found_override_files.append(file)

    assert len(glob_files) == len(files)
    assert len(found_override_files) == len(override_files)


def test_consolidated_rglob(random_file_tree):
    dirname, files, override_files = random_file_tree
    glob_files = (data_path / dirname).rglob("*.test")
    found_override_files = []
    for file in glob_files:
        if file.is_file() and LocalDataPath.ORDERED_SEARCH_PATHS[0] in file.parents:
            found_override_files.append(file)

    assert len(glob_files) == len(files)
    assert len(found_override_files) == len(override_files)
