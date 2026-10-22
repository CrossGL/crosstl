"""Metal kernels keep the host's threadgroup size or refuse targets that fix it in the shader."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.stage_utils import compute_local_size
from crosstl.translator.host_workgroup_size import (
    HOST_WORKGROUP_SIZE_KEY,
    HostDefinedWorkgroupSizeError,
    normalize_threadgroup_size_option,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"

SHARED = HEADER + """
kernel void rotate(device float* data [[buffer(0)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint count [[threads_per_threadgroup]]) {
    threadgroup float tile[256];
    tile[lid] = data[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    data[gid] = tile[(lid + 1) % count];
}
"""

SIMD_HELPER = HEADER + """
float lane_total(float value) { return simd_sum(value); }
kernel void reduce(device float* data [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    data[gid] = lane_total(data[gid]);
}
"""

INDEPENDENT = HEADER + """
float simdLerp(float a, float b) { return a + (b - a) * 0.5; }
kernel void scale(device float* data [[buffer(0)]],
                  uint gid [[thread_position_in_grid]],
                  uint unused [[thread_position_in_threadgroup]]) {
    data[gid] = simdLerp(data[gid], 2.0);
}
"""

SIZES = {
    "vulkan": r"OpExecutionMode %\S+ LocalSize (\d+) (\d+) (\d+)",
    "directx": r"\[numthreads\((\d+), (\d+), (\d+)\)\]",
    "opengl": r"local_size_x = (\d+), local_size_y = (\d+), local_size_z = (\d+)",
}


def _translate(tmp_path, source, backend, **options):
    path = tmp_path / "kernel.metal"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, source_options=options or None)


def _size(code, backend):
    return tuple(int(value) for value in re.search(SIZES[backend], code).groups())


@pytest.mark.parametrize("backend", sorted(SIZES))
@pytest.mark.parametrize(
    "source, entry, construct",
    [
        (SHARED, "rotate", "the threadgroup variable tile"),
        (SIMD_HELPER, "reduce", "calls WaveActiveSum"),
    ],
)
def test_size_dependent_kernel_is_refused_without_the_host_size(
    tmp_path, backend, source, entry, construct
):
    with pytest.raises(HostDefinedWorkgroupSizeError) as raised:
        _translate(tmp_path, source, backend)
    assert raised.value.entry == entry
    assert construct in str(raised.value)
    assert raised.value.project_diagnostic_code == (
        "project.translate.workgroup-size-host-defined"
    )


@pytest.mark.parametrize("backend", sorted(SIZES))
@pytest.mark.parametrize(
    "option, size", [(64, (64, 1, 1)), ((8, 4), (8, 4, 1)), ([4, 2, 2], (4, 2, 2))]
)
def test_threadgroup_size_option_fixes_the_target_size(tmp_path, backend, option, size):
    for source in (SHARED, SIMD_HELPER):
        code = _translate(tmp_path, source, backend, threadgroup_size=option)
        assert _size(code, backend) == size


def test_metal_round_trip_keeps_the_size_with_the_host(tmp_path):
    code = _translate(tmp_path, SHARED, "metal")
    assert "kernel void rotate" in code
    assert "max_total_threads_per_threadgroup" not in code


@pytest.mark.parametrize("option", [None, 64])
def test_size_independent_kernel_keeps_one_invocation_per_group(tmp_path, option):
    # One invocation per workgroup is exact for any dispatch, including a non-uniform dispatchThreads grid.
    for backend in SIZES:
        code = _translate(tmp_path, INDEPENDENT, backend, threadgroup_size=option)
        assert _size(code, backend) == (1, 1, 1)


def test_crossgl_marks_host_sized_kernels(tmp_path):
    crossgl = _translate(tmp_path, SHARED, "cgl")
    assert "@ host_workgroup_size\n" in crossgl
    assert "@ workgroup_size" not in crossgl
    sized = _translate(tmp_path, SHARED, "cgl", threadgroup_size=(32, 1, 1))
    assert "@ host_workgroup_size(32, 1, 1)" in sized
    assert "@ workgroup_size" not in sized


def _compute_stage(source):
    shader = Parser(Lexer(source).get_tokens()).parse()
    (stage,) = list(shader.stages.values())
    return stage


def test_parser_records_dependence_and_explicit_sizes_win():
    dependent = _compute_stage("""
shader main {
    compute {
        @ host_workgroup_size
        @ stage_entry
        void k(uint lid @gl_LocalInvocationID) { workgroupBarrier(); }
    }
}
""")
    record = dependent.execution_config[HOST_WORKGROUP_SIZE_KEY]
    assert record == {"entry": "k", "reason": "calls workgroupBarrier"}
    assert not any(
        attribute.name == "host_workgroup_size"
        for attribute in dependent.entry_point.attributes
    )
    with pytest.raises(HostDefinedWorkgroupSizeError):
        compute_local_size(dependent.execution_config)

    explicit = _compute_stage("""
shader main {
    compute {
        @ host_workgroup_size
        @ workgroup_size(16, 2, 1)
        @ stage_entry
        void k(uint lid @gl_LocalInvocationID) { workgroupBarrier(); }
    }
}
""")
    assert HOST_WORKGROUP_SIZE_KEY not in explicit.execution_config
    assert compute_local_size(explicit.execution_config) == ("16", "2", "1")

    unused = _compute_stage("""
shader main {
    compute {
        @ host_workgroup_size
        @ stage_entry
        void k(uint lid @gl_LocalInvocationID) { }
    }
}
""")
    assert HOST_WORKGROUP_SIZE_KEY not in unused.execution_config
    assert compute_local_size(unused.execution_config) == ("1", "1", "1")

    host_sized = _compute_stage("""
shader main {
    compute {
        @ host_workgroup_size(8, 4, 1)
        @ stage_entry
        void k(uint lid @gl_LocalInvocationIndex) { workgroupBarrier(); }
    }
}
""")
    assert HOST_WORKGROUP_SIZE_KEY not in host_sized.execution_config
    assert compute_local_size(host_sized.execution_config) == ("8", "4", "1")

    host_sized_unused = _compute_stage("""
shader main {
    compute {
        @ host_workgroup_size(8, 4, 1)
        @ stage_entry
        void k(uint gid @gl_GlobalInvocationID) { }
    }
}
""")
    assert host_sized_unused.execution_config.get("numthreads") is None
    assert compute_local_size(host_sized_unused.execution_config) == ("1", "1", "1")


def test_parser_rejects_a_malformed_host_size_mark():
    with pytest.raises(ValueError, match="host_workgroup_size"):
        _compute_stage("""
shader main {
    compute {
        @ host_workgroup_size(8, 4)
        @ stage_entry
        void k() { workgroupBarrier(); }
    }
}
""")


@pytest.mark.parametrize(
    "value", [0, -1, True, "64", 1.5, (1, 2, 3, 4), (), (64, 0), (2, None)]
)
def test_threadgroup_size_option_rejects_non_positive_integers(value):
    with pytest.raises(ValueError, match="threadgroup_size"):
        normalize_threadgroup_size_option(value)


def test_sized_spirv_validates(tmp_path):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not installed")
    code = _translate(tmp_path, SIMD_HELPER, "vulkan", threadgroup_size=64)
    assembly, binary = tmp_path / "kernel.spvasm", tmp_path / "kernel.spv"
    assembly.write_text(code)
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
