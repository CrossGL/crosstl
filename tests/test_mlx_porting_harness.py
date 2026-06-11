import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "integrations" / "mlx" / "run_mlx_porting.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("mlx_porting_harness", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MLX_PORTING = _load_harness()


def _payload(*, target, artifacts, toolchain_status="available", runs=()):
    return {
        "artifacts": [
            {"target": target, "path": path, "status": "translated"}
            for path in artifacts
        ],
        "validation": {
            "toolchains": [
                {
                    "target": target,
                    "status": toolchain_status,
                    "tools": [
                        (
                            {"name": "dxc", "path": None, "available": False}
                            if toolchain_status != "available"
                            else {
                                "name": "dxc",
                                "path": "/tools/dxc",
                                "available": True,
                            }
                        )
                    ],
                }
            ],
            "toolchainRuns": list(runs),
        },
    }


def _run(target, path, status="ok", stderr=""):
    return {
        "target": target,
        "path": path,
        "status": status,
        "stderr": stderr,
    }


def test_mlx_porting_parse_args_keeps_toolchain_requirements_independent():
    args = MLX_PORTING.parse_args(
        [
            "--mlx-root",
            "/tmp/mlx",
            "--require-directx-toolchain",
            "--require-opengl-toolchain",
            "--require-vulkan-toolchain",
        ]
    )

    assert args.require_directx_toolchain is True
    assert args.require_opengl_toolchain is True
    assert args.require_vulkan_toolchain is True


def test_required_toolchain_smoke_accepts_multiple_runs_per_artifact():
    payload = _payload(
        target="vulkan",
        artifacts=["out/vulkan/a.spvasm", "out/vulkan/b.spvasm"],
        runs=[
            _run("vulkan", "out/vulkan/a.spvasm"),
            _run("vulkan", "out/vulkan/a.spvasm"),
            _run("vulkan", "out/vulkan/b.spvasm"),
            _run("vulkan", "out/vulkan/b.spvasm"),
        ],
    )

    result = MLX_PORTING._require_target_toolchain_smoke(
        payload,
        "vulkan",
        label="frontier",
    )

    assert result == {
        "target": "vulkan",
        "status": "validated",
        "artifactCount": 2,
        "runCount": 4,
    }


def test_required_toolchain_smoke_reports_unavailable_tools():
    payload = _payload(
        target="directx",
        artifacts=["out/directx/a.hlsl"],
        toolchain_status="unavailable",
    )

    with pytest.raises(MLX_PORTING.PortingCheckError, match="missing: dxc"):
        MLX_PORTING._require_target_toolchain_smoke(
            payload,
            "directx",
            label="frontier",
        )


def test_required_toolchain_smoke_reports_skipped_artifacts():
    payload = _payload(target="opengl", artifacts=["out/opengl/a.glsl"])

    with pytest.raises(MLX_PORTING.PortingCheckError, match="missing or skipped"):
        MLX_PORTING._require_target_toolchain_smoke(
            payload,
            "opengl",
            label="arange",
        )


def test_required_toolchain_smoke_reports_failed_runs():
    payload = _payload(
        target="opengl",
        artifacts=["out/opengl/a.glsl"],
        runs=[
            _run(
                "opengl",
                "out/opengl/a.glsl",
                status="failed",
                stderr="shader validation failed",
            )
        ],
    )

    with pytest.raises(MLX_PORTING.PortingCheckError, match="shader validation failed"):
        MLX_PORTING._require_target_toolchain_smoke(
            payload,
            "opengl",
            label="arange",
        )
