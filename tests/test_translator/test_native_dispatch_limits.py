from dataclasses import replace
from types import SimpleNamespace

import pytest

from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    OpenGLComputeRuntime,
    _validate_dispatch_limits,
    _VulkanDispatchContext,
    _workgroup_count,
)
from crosstl.project.runtime_verification import (
    RuntimeAdapterDispatchError,
    RuntimeAdapterSetupError,
    RuntimeDispatchGeometry,
)
from tests.test_translator.test_native_runtime_drivers import (
    _directx_dispatch_request,
    _FakeCompushady,
    _native_dispatch_sequence_requests,
    _SequenceOpenGLContext,
)


@pytest.mark.parametrize("target", ["directx", "opengl", "vulkan"])
@pytest.mark.parametrize(
    "size", [(1024, 1, 1), (1, 1024, 1), (1, 1, 64), (8, 8, 16), None]
)
def test_dispatch_limit_boundaries(target, size):
    _validate_dispatch_limits(
        (65535, 65535, 65535),
        size,
        target=target,
        max_count=(65535, 65535, 65535),
        max_size=(1024, 1024, 64),
        max_invocations=1024,
    )


@pytest.mark.parametrize("target", ["directx", "opengl", "vulkan"])
@pytest.mark.parametrize(
    "field", ["workgroupCount", "workgroupSize", "workgroupInvocations"]
)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_dispatch_limit_diagnostics_identify_actual_constraint(target, field, axis):
    counts, size = [1, 1, 1], [1, 1, 1]
    if field == "workgroupCount":
        counts[axis] = 65536
    elif field == "workgroupSize":
        size[axis] = (1025, 1025, 65)[axis]
    else:
        size = [32, 32, 2]
    with pytest.raises(RuntimeAdapterSetupError) as caught:
        _validate_dispatch_limits(
            counts,
            size,
            target=target,
            max_count=(65535, 65535, 65535),
            max_size=(1024, 1024, 64),
            max_invocations=1024,
            node_index=2,
        )
    details = caught.value.details
    assert details["target"] == target and details["nodeIndex"] == 2
    assert details["reasonKind"] == "dispatch-limit-exceeded"
    assert details["field"] == field
    assert details["workgroupCount"] == counts and details["workgroupSize"] == size
    assert details["requested"] > details["maximum"]
    if field != "workgroupInvocations":
        assert details["axis"] == axis
    else:
        assert "axis" not in details


@pytest.mark.parametrize(
    "bad", [None, [], [1, 2], [1, 2, 0], [1, 2, True], [1, 2, 3.5]]
)
def test_missing_or_invalid_capabilities_do_not_default_to_assumed_limits(bad):
    with pytest.raises(RuntimeAdapterSetupError) as caught:
        _validate_dispatch_limits(
            (1, 1, 1),
            (1, 1, 1),
            target="opengl",
            max_count=bad,
            max_size=(1024, 1024, 64),
            max_invocations=1024,
        )
    assert caught.value.details["reasonKind"] == "dispatch-limits-unavailable"


@pytest.mark.parametrize("bad", [(0,), (-1,), (True,), (1.5,), ("2",), (1, 1, 1, 1)])
def test_native_geometry_does_not_clamp_or_coerce_invalid_dimensions(bad):
    request = SimpleNamespace(dispatch=SimpleNamespace(workgroup_count=bad))
    with pytest.raises(RuntimeAdapterSetupError, match="positive integers"):
        _workgroup_count(request, target="OpenGL")


def test_derived_workgroup_counts_use_exact_integer_arithmetic():
    extent = 2**60 + 1
    request = SimpleNamespace(
        dispatch=SimpleNamespace(
            workgroup_count=None, global_size=(extent,), workgroup_size=(2,)
        )
    )
    assert _workgroup_count(request) == (2**59 + 1, 1, 1)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_directx_rejects_over_limit_before_allocating_or_dispatching(tmp_path, axis):
    module = _FakeCompushady()
    runtime = DirectXComputeRuntime(
        module_loader=lambda _: module, platform_name="win32"
    )
    counts = [1, 1, 1]
    counts[axis] = 65536
    request = replace(
        _directx_dispatch_request(tmp_path),
        dispatch=RuntimeDispatchGeometry(workgroup_count=tuple(counts)),
    )
    with pytest.raises(RuntimeAdapterSetupError) as caught:
        runtime.dispatch(None, None, request)
    assert caught.value.details["axis"] == axis
    assert module.buffers == [] and module.computes == []


def test_opengl_preflights_entire_sequence_before_any_device_work(tmp_path):
    context = _SequenceOpenGLContext()
    runtime = OpenGLComputeRuntime(
        module_loader=lambda _: object(), context_factory=lambda _: context
    )
    requests = list(_native_dispatch_sequence_requests(tmp_path, "opengl"))
    requests[1] = replace(
        requests[1], dispatch=RuntimeDispatchGeometry(workgroup_count=(65536, 1, 1))
    )
    with pytest.raises(RuntimeAdapterSetupError) as caught:
        runtime.dispatch_sequence(None, None, requests)
    assert caught.value.details["nodeIndex"] == 1
    assert not context.events and not context.buffers and not context.shaders
    assert context.release_count == 1


@pytest.mark.parametrize(
    "phase,query",
    [
        ("context", 1),
        ("prepare", 2),
        ("dispatch", 3),
        ("synchronize", 4),
        ("readback", 5),
    ],
)
def test_opengl_api_errors_cannot_be_reported_as_success(tmp_path, phase, query):
    class Context(_SequenceOpenGLContext):
        queries = 0

        @property
        def error(self):
            self.queries += 1
            return "GL_INVALID_OPERATION" if self.queries == query else "GL_NO_ERROR"

    context = Context()
    runtime = OpenGLComputeRuntime(
        module_loader=lambda _: object(), context_factory=lambda _: context
    )
    request = _native_dispatch_sequence_requests(tmp_path, "opengl")[0]
    with pytest.raises(RuntimeAdapterDispatchError) as caught:
        runtime.dispatch(None, None, request)
    assert caught.value.details["reasonKind"] == "opengl-api-error"
    assert caught.value.details["phase"] == phase
    assert caught.value.details["glError"] == "GL_INVALID_OPERATION"
    assert context.release_count == 1
    assert all(shader.release_count == 1 for shader in context.shaders)
    assert all(buffer.release_count == 1 for buffer in context.buffers)


def test_opengl_error_query_failure_cannot_be_reported_as_success(tmp_path):
    class Context(_SequenceOpenGLContext):
        @property
        def error(self):
            raise RuntimeError("context error query failed")

    context = Context()
    runtime = OpenGLComputeRuntime(
        module_loader=lambda _: object(), context_factory=lambda _: context
    )
    request = _native_dispatch_sequence_requests(tmp_path, "opengl")[0]
    with pytest.raises(RuntimeAdapterDispatchError) as caught:
        runtime.dispatch(None, None, request)
    assert caught.value.details["reasonKind"] == "opengl-error-query-failed"
    assert caught.value.details["phase"] == "context"
    assert not context.buffers and not context.shaders
    assert context.release_count == 1


def test_vulkan_checks_selected_physical_device_before_logical_device_creation():
    destroyed = []
    limits = SimpleNamespace(
        maxComputeWorkGroupCount=(4, 3, 2),
        maxComputeWorkGroupSize=(32, 16, 8),
        maxComputeWorkGroupInvocations=128,
    )
    vk = SimpleNamespace(
        vkGetPhysicalDeviceProperties=lambda device: SimpleNamespace(limits=limits),
        vkDestroyInstance=lambda instance, allocator: destroyed.append(instance),
    )
    runtime = SimpleNamespace(
        name="vulkan",
        _create_instance=lambda vk: "instance",
        _select_compute_device=lambda vk, instance: ("physical", 0),
        _destroy_instance=lambda vk, instance: vk.vkDestroyInstance(instance, None),
    )
    context = _VulkanDispatchContext(
        vk=vk,
        runtime=runtime,
        shader_code=b"unused",
        entry_point="main",
        buffers=(),
        workgroup_count=(5, 1, 1),
    )
    with pytest.raises(RuntimeAdapterSetupError) as caught:
        context.run()
    assert caught.value.details["target"] == "vulkan"
    assert caught.value.details["maximum"] == 4
    assert caught.value.details["axis"] == 0
    assert destroyed == ["instance"]
