"""Public project API coverage for native loader dispatch requests."""

import crosstl.project as project_api
from crosstl.project.native_loader_dispatch import (
    NativeLoaderDispatchError,
    build_native_loader_dispatch_request,
)


def test_project_exports_native_loader_dispatch_contract():
    assert project_api.NativeLoaderDispatchError is NativeLoaderDispatchError
    assert (
        project_api.build_native_loader_dispatch_request
        is build_native_loader_dispatch_request
    )
    assert "NativeLoaderDispatchError" in project_api.__all__
    assert "build_native_loader_dispatch_request" in project_api.__all__
    assert issubclass(project_api.NativeLoaderDispatchError, ValueError)
