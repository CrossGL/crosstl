"""Public project API coverage for generated native target adapters."""

import pytest

import crosstl.project as project_api
from crosstl.project.native_target_adapters import (
    NATIVE_LOADER_TARGET_ADAPTER_KIND,
    NATIVE_LOADER_TARGET_ADAPTER_VERSION,
    NativeLoaderTargetAdapterError,
    native_loader_target_adapter_targets,
)


def test_project_exports_native_loader_target_adapter_contract():
    assert (
        project_api.NATIVE_LOADER_TARGET_ADAPTER_KIND
        == NATIVE_LOADER_TARGET_ADAPTER_KIND
    )
    assert (
        project_api.NATIVE_LOADER_TARGET_ADAPTER_VERSION
        == NATIVE_LOADER_TARGET_ADAPTER_VERSION
    )
    assert project_api.NativeLoaderTargetAdapterError is NativeLoaderTargetAdapterError
    assert (
        project_api.native_loader_target_adapter_targets
        is native_loader_target_adapter_targets
    )
    assert project_api.native_loader_target_adapter_targets() == (
        "directx",
        "opengl",
    )


@pytest.mark.parametrize("target", (None, "", " directx", 3))
def test_native_loader_target_adapter_rejects_invalid_targets(target):
    with pytest.raises(NativeLoaderTargetAdapterError) as raised:
        project_api.generate_native_loader_target_adapter(target)

    assert raised.value.code == "project.native-loader-target-adapter.target-invalid"
    assert raised.value.path == "$.target"


def test_native_loader_target_adapter_rejects_unsupported_target():
    with pytest.raises(NativeLoaderTargetAdapterError) as raised:
        project_api.generate_native_loader_target_adapter("metal")

    assert raised.value.to_json() == {
        "severity": "error",
        "code": "project.native-loader-target-adapter.target-unsupported",
        "message": "Native loader target adapter is not available for 'metal'.",
        "path": "$.target",
        "details": {
            "target": "metal",
            "supportedTargets": ["directx", "opengl"],
        },
    }
