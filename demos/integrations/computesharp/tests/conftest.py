"""Session fixtures shared by the ComputeSharp demo tests."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "computesharp_fixture_support",
    Path(__file__).resolve().parent / "fixture_support.py",
)
SUPPORT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(SUPPORT)


@pytest.fixture(scope="session")
def discovery_tools(tmp_path_factory):
    return SUPPORT.build_tools(tmp_path_factory)


@pytest.fixture(scope="session")
def program_tools(discovery_tools, tmp_path_factory):
    return SUPPORT.build_program(discovery_tools, tmp_path_factory)


@pytest.fixture(scope="session")
def baseline_fixture(discovery_tools, tmp_path_factory):
    return SUPPORT.run_fixture(
        discovery_tools, tmp_path_factory.mktemp("baseline-proof")
    )
