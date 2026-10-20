"""Check the native device preflight of the translated-run harness.

A stub backend is compiled against the real ``csbackend.h`` so the ctypes view of
``cs_device_info`` is checked against the C layout. The stub reports a device
selected by an environment variable, or fails to create one.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from fixture_support import ROOT, load_module  # noqa: E402

RUN_TESTS = load_module("computesharp_run_tests", ROOT / "run_tests.py")
PROFILES = json.loads((ROOT / "coverage.json").read_text())["profiles"]
STUB = r"""
#include <stdlib.h>
#include <string.h>
#include "csbackend.h"

uint32_t cs_abi_version(void) { return CS_ABI_VERSION; }

int cs_device_create(void **device, cs_device_info *info, char *error, int32_t capacity) {
    const char *mode = getenv("CROSSTL_STUB_DEVICE");
    if (mode == NULL || strcmp(mode, "missing") == 0) {
        strncpy(error, "no adapter is present", (size_t)capacity - 1);
        return 2;
    }
    memset(info, 0, sizeof(*info));
    info->abi_version = CS_ABI_VERSION;
    info->is_hardware = strcmp(mode, "software") != 0;
    info->supports_fp64 = strcmp(mode, "fp64") == 0;
    info->max_threads_per_group = 1024;
    info->max_group_size_z = 64;
    strcpy(info->name, "Stub device");
    strcpy(info->api, "Metal");
    strcpy(info->driver, "stub driver 1.0");
    *device = info;
    return 0;
}

void cs_device_destroy(void *device) { (void)device; }
"""


@pytest.fixture(scope="module")
def stub_library(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        pytest.skip("a C compiler is required to build the stub backend")
    directory = tmp_path_factory.mktemp("stub-backend")
    source = directory / "stub.c"
    source.write_text(STUB)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    library = directory / f"libstub{suffix}"
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            "-Wall",
            "-Werror",
            "-I",
            str(ROOT / "runtime" / "native"),
            "-o",
            str(library),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return library


def test_matching_device_passes_and_reads_the_c_layout(stub_library, monkeypatch):
    monkeypatch.setenv("CROSSTL_STUB_DEVICE", "hardware")
    report = RUN_TESTS.preflight(stub_library, PROFILES["metal-apple-gpu"])
    assert report["status"] == "passed", report["errors"]
    assert report["abiVersion"] == 1
    assert report["device"] == {
        "name": "Stub device",
        "api": "Metal",
        "driver": "stub driver 1.0",
        "hardware": True,
        "fp64": False,
        "maxThreadsPerGroup": 1024,
    }


def test_missing_device_is_an_infrastructure_blocker(stub_library, monkeypatch):
    monkeypatch.setenv("CROSSTL_STUB_DEVICE", "missing")
    report = RUN_TESTS.preflight(stub_library, PROFILES["metal-apple-gpu"])
    assert report["status"] == "blocked"
    assert "no adapter is present" in report["errors"][0]


@pytest.mark.parametrize(
    "mode, profile, message",
    [
        ("software", "metal-apple-gpu", "requires hardware execution"),
        ("hardware", "opengl-mesa-llvmpipe", "requires software execution"),
        ("fp64", "metal-apple-gpu", "freezes fp64=False"),
    ],
)
def test_capability_mismatches_stop_the_run(
    stub_library, monkeypatch, mode, profile, message
):
    monkeypatch.setenv("CROSSTL_STUB_DEVICE", mode)
    report = RUN_TESTS.preflight(stub_library, PROFILES[profile])
    assert report["status"] == "failed"
    assert any(message in error for error in report["errors"]), report["errors"]


def test_unloadable_library_is_an_infrastructure_blocker(tmp_path):
    report = RUN_TESTS.preflight(tmp_path / "missing.so", PROFILES["metal-apple-gpu"])
    assert report["status"] == "blocked"
