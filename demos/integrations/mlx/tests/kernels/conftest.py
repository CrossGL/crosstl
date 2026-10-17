"""Shared source controls for pinned binary-kernel execution."""

from functools import lru_cache
from pathlib import Path

import pytest

from demos.integrations.mlx.tests.kernels.test_binary_complete_opengl import (
    MLX_BINARY_SOURCE,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)

ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture(scope="session")
def binary_metal_reference(tmp_path_factory):
    @lru_cache(maxsize=None)
    def build(root, target):
        if target != "metal":
            return None, None
        original = tmp_path_factory.mktemp("binary-source-control")
        runner = original / "readback"
        _run(
            [
                "xcrun",
                "swiftc",
                ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
                "-o",
                runner,
            ],
            original,
            "build-runner",
        )
        library = _metal_library(
            root / MLX_BINARY_SOURCE,
            original / "original.metallib",
            root,
            upstream=True,
        )
        return runner, library

    return build
