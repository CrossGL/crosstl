"""Reflection reports the workgroup size a pipeline or host sets instead of its placeholder default."""

import re

import pytest

import crosstl
from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.project.native_loader_dispatch import (
    NativeLoaderDispatchError,
    _reflected_workgroup_size,
)
from crosstl.translator.codegen.host_workgroup_size_ids import (
    HOST_WORKGROUP_SIZE_CONSTANT_NAMES,
    HOST_WORKGROUP_SIZE_MACROS,
)
from crosstl.translator.host_workgroup_size import (
    HOST_WORKGROUP_SIZE_KEY,
    HostDefinedWorkgroupSizeError,
    require_static_workgroup_size,
)

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"

# ``reduce_sum`` depends on a threadgroup size that only the host chooses; ``scale`` does not.
REDUCE_SUM = """
kernel void reduce_sum(device const float* src [[buffer(0)]],
                       device float* dst [[buffer(1)]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint gid [[threadgroup_position_in_grid]],
                       uint tsize [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    scratch[tid] = src[gid * tsize + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < tsize; ++i) { total += scratch[i]; }
        dst[gid] = total;
    }
}
"""

SCALE = """
kernel void scale(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    data[id] = data[id] * 2.0f;
}
"""

# A module whose pipeline specializes a size of its own, with a fixed z dimension.
SPECIALIZED_ASSEMBLY = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionModeId %main LocalSizeId %x %y %z
OpName %x "tile_width"
OpDecorate %x SpecId 7
OpDecorate %y SpecId 8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%x = OpSpecConstant %uint 16
%y = OpSpecConstant %uint 4
%z = OpConstant %uint 2
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(HEADER + source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _reflect(tmp_path, code, name, target):
    artifact = tmp_path / name
    artifact.write_text(code)
    record = reflect_target_host_interface(artifact, target=target)
    assert record is not None
    return record


def _entries(record):
    return {entry["name"]: entry for entry in record["entryPoints"]}


def _assert_host_chosen(entry, ids, macros=None):
    config = entry["executionConfig"]
    assert config["local_size_id"] == list(ids)
    assert "local_size" not in config and "local_size_x" not in config
    record = config[HOST_WORKGROUP_SIZE_KEY]
    assert record["specializationIds"] == list(ids)
    assert record.get("macros") == macros
    with pytest.raises(HostDefinedWorkgroupSizeError):
        require_static_workgroup_size(config)
    with pytest.raises(NativeLoaderDispatchError) as error:
        _reflected_workgroup_size(entry)
    assert error.value.code.endswith("workgroup-size-host-defined")


def test_spirv_names_the_host_size_constants_once(tmp_path):
    code = _translate(
        tmp_path, REDUCE_SUM + REDUCE_SUM.replace("reduce_sum", "reduce_max"), "vulkan"
    )
    for name in HOST_WORKGROUP_SIZE_CONSTANT_NAMES:
        assert len(re.findall(rf'OpName %\w+ "{name}"', code)) == 1


def test_spirv_builtin_host_size_is_not_reported_as_its_placeholder(tmp_path):
    record = _reflect(
        tmp_path, _translate(tmp_path, REDUCE_SUM, "vulkan"), "k.spvasm", "vulkan"
    )
    assert record["status"] == "ready"
    entry = _entries(record)["reduce_sum"]
    assert entry["metadata"]["workgroupSizeSource"] == "WorkgroupSize"
    assert "requiredExtensions" not in entry["metadata"]
    _assert_host_chosen(entry, [0, 1, 2])


def test_spirv_local_size_id_entry_reports_the_host_size_and_maintenance4(tmp_path):
    code = _translate(tmp_path, REDUCE_SUM + SCALE, "vulkan")
    assert "OpExecutionModeId" in code
    entries = _entries(_reflect(tmp_path, code, "k.spvasm", "vulkan"))
    host = entries["reduce_sum"]
    assert host["metadata"]["workgroupSizeSource"] == "LocalSizeId"
    assert host["metadata"]["requiredExtensions"] == ["VK_KHR_maintenance4"]
    _assert_host_chosen(host, [0, 1, 2])
    assert entries["scale"]["executionConfig"] == {"local_size": [1, 1, 1]}
    assert _reflected_workgroup_size(entries["scale"]) == (1, 1, 1)


def test_spirv_specialized_size_reports_defaults_and_ids(tmp_path):
    record = _reflect(tmp_path, SPECIALIZED_ASSEMBLY, "k.spvasm", "vulkan")
    entry = _entries(record)["main"]
    assert entry["executionConfig"] == {
        "local_size": [16, 4, 2],
        "local_size_id": [7, 8, None],
    }
    assert entry["metadata"]["requiredExtensions"] == ["VK_KHR_maintenance4"]
    assert _reflected_workgroup_size(entry) == (16, 4, 2)


@pytest.mark.parametrize(
    "assembly",
    [
        SPECIALIZED_ASSEMBLY.replace(
            "LocalSizeId %x %y %z", "LocalSizeId %x %y %missing"
        ),
        SPECIALIZED_ASSEMBLY.replace("OpDecorate %y SpecId 8\n", ""),
        SPECIALIZED_ASSEMBLY.replace(
            "OpExecutionModeId %main LocalSizeId %x %y %z",
            "OpDecorate %size BuiltIn WorkgroupSize",
        ).replace(
            "%main = OpFunction",
            "%v2 = OpTypeVector %uint 2\n%size = OpSpecConstantComposite %v2 %x %y\n%main = OpFunction",
        ),
    ],
    ids=["undefined-component", "specialization-without-id", "two-component-builtin"],
)
def test_spirv_unresolvable_size_is_diagnosed(tmp_path, assembly):
    record = _reflect(tmp_path, assembly, "k.spvasm", "vulkan")
    assert record["status"] == "incomplete"
    assert _entries(record)["main"]["executionConfig"] == {}
    details = [
        diagnostic
        for diagnostic in record["diagnosticRecords"]
        if diagnostic.get("details", {}).get("reasonKind")
        == "workgroup-size-unresolved"
    ]
    assert len(details) == 1


def test_glsl_host_size_reports_its_ids_and_macros(tmp_path):
    record = _reflect(
        tmp_path, _translate(tmp_path, REDUCE_SUM, "opengl"), "k.comp", "opengl"
    )
    assert record["status"] == "ready"
    _assert_host_chosen(
        _entries(record)["main"], [0, 1, 2], list(HOST_WORKGROUP_SIZE_MACROS)
    )


def test_glsl_reads_every_input_layout_declaration(tmp_path):
    source = (
        "#version 450\n"
        "layout(local_size_x = 32, local_size_x_id = 3) in;\n"
        "layout(local_size_y = 2) in;\n"
        "void main() {}\n"
    )
    entry = _entries(_reflect(tmp_path, source, "k.comp", "opengl"))["main"]
    assert entry["executionConfig"] == {
        "local_size_x": 32,
        "local_size_y": 2,
        "local_size_z": 1,
        "local_size_id": [3, None, None],
    }
    assert _reflected_workgroup_size(entry) == (32, 2, 1)


def test_glsl_fixed_size_is_unchanged(tmp_path):
    code = _translate(tmp_path, SCALE, "opengl")
    entry = _entries(_reflect(tmp_path, code, "k.comp", "opengl"))["main"]
    assert entry["executionConfig"] == {
        "local_size_x": 1,
        "local_size_y": 1,
        "local_size_z": 1,
    }
