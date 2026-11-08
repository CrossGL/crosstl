"""GLSL lets the host set a workgroup size that the source's host chooses at dispatch time."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLSoftwareSubgroupError,
)
from crosstl.translator.codegen.glsl_host_workgroup_size import (
    glsl_workgroup_proof_local_size,
)
from crosstl.translator.codegen.host_workgroup_size_ids import (
    HOST_WORKGROUP_SIZE_MACROS,
    allocate_host_workgroup_size_ids,
    host_sized_workgroup_config,
    is_host_sized_workgroup,
)
from crosstl.translator.host_workgroup_size import HOST_WORKGROUP_SIZE_KEY

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"

ROTATE = HEADER + """
CONSTANTS
kernel void rotate(device float* data [[buffer(0)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint count [[threads_per_threadgroup]]) {
    threadgroup float tile[256];
    tile[lid] = data[gid] * SCALE;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    data[gid] = tile[(lid + 1) % count];
}
"""

TWO_CONSTANTS = """
constant float first [[function_constant(0)]];
constant float second [[function_constant(2)]];
"""

LAYOUT_IDS = re.compile(
    r"local_size_x_id = (\d+),\s*local_size_y_id = (\d+),\s*local_size_z_id = (\d+)"
)
GLSLANG = shutil.which("glslangValidator")


def _rotate(constants="", scale="1.0"):
    return ROTATE.replace("CONSTANTS", constants).replace("SCALE", scale)


def _translate(tmp_path, source, backend, name="kernel.metal"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _glslang(tmp_path, code, *arguments):
    path = tmp_path / "kernel.comp"
    path.write_text(code)
    return subprocess.run(
        [GLSLANG, *arguments, "-S", "comp", str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )


def test_host_sized_kernel_gets_a_specialized_layout_and_source_macros(tmp_path):
    code = _translate(tmp_path, _rotate(), "opengl")
    assert LAYOUT_IDS.search(code).groups() == ("0", "1", "2")
    for macro in HOST_WORKGROUP_SIZE_MACROS:
        assert f"defined({macro})" in code
    assert "local_size_x = CROSSGL_WORKGROUP_SIZE_X" in code
    assert "#error" in code
    assert "uint count = uint(gl_WorkGroupSize.x);" in code


@pytest.mark.skipif(GLSLANG is None, reason="glslangValidator is not installed")
def test_glslang_accepts_both_paths_and_stops_without_the_macros(tmp_path):
    code = _translate(tmp_path, _rotate(), "opengl")

    missing = _glslang(tmp_path, code)
    assert missing.returncode != 0
    assert "CROSSGL_WORKGROUP_SIZE_X" in missing.stdout + missing.stderr

    defines = [f"-D{macro}=4" for macro in HOST_WORKGROUP_SIZE_MACROS]
    source_path = _glslang(tmp_path, code, "-l", *defines)
    assert source_path.returncode == 0, source_path.stdout + source_path.stderr

    vulkan = _glslang(
        tmp_path, code, "-V", "--target-env", "vulkan1.1", "-o", "kernel.spv"
    )
    assert vulkan.returncode == 0, vulkan.stdout + vulkan.stderr
    if shutil.which("spirv-dis"):
        assembly = subprocess.run(
            ["spirv-dis", str(tmp_path / "kernel.spv")],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert "BuiltIn WorkgroupSize" in assembly
        for spec_id in range(3):
            assert re.search(rf"OpDecorate %\w+ SpecId {spec_id}\b", assembly)


def test_glsl_and_spirv_give_the_host_the_same_ids(tmp_path):
    source = _rotate(TWO_CONSTANTS, "(first + second)")
    glsl = _translate(tmp_path, source, "opengl")
    assert LAYOUT_IDS.search(glsl).groups() == ("1", "3", "4")
    spirv = _translate(tmp_path, source, "vulkan")
    composite = re.search(r"OpDecorate %(\w+) BuiltIn WorkgroupSize\b", spirv).group(1)
    components = re.search(
        rf"%{composite} = OpSpecConstantComposite %\w+ %(\w+) %(\w+) %(\w+)", spirv
    ).groups()
    ids = tuple(
        re.search(rf"OpDecorate %{component} SpecId (\d+)", spirv).group(1)
        for component in components
    )
    assert ids == ("1", "3", "4")


def test_explicit_threadgroup_size_keeps_a_fixed_layout(tmp_path):
    path = tmp_path / "kernel.metal"
    path.write_text(_rotate())
    code = crosstl.translate(
        str(path),
        backend="opengl",
        format_output=False,
        source_options={"threadgroup_size": (64, 1, 1)},
    )
    assert "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;" in code
    assert "CROSSGL_WORKGROUP_SIZE_X" not in code


def test_ids_skip_the_programs_specialization_ids():
    assert allocate_host_workgroup_size_ids(set()) == (0, 1, 2)
    assert allocate_host_workgroup_size_ids({0, 2}) == (1, 3, 4)
    mark = {HOST_WORKGROUP_SIZE_KEY: {"entry": "k", "reason": "reads gl_WorkGroupSize"}}
    assert is_host_sized_workgroup(mark)
    assert not is_host_sized_workgroup({**mark, "local_size": [8, 1, 1]})
    assert not is_host_sized_workgroup({**mark, "local_size_x_id": "3"})
    assert host_sized_workgroup_config(mark, (5, 6, 7))["local_size_z_id"] == 7
    assert host_sized_workgroup_config({"local_size_x": 4}, (5, 6, 7)) == {
        "local_size_x": 4
    }


def test_invocation_proofs_assume_no_size_the_pipeline_sets():
    # The local_size values of a specialized size are defaults that the pipeline may replace.
    assert glsl_workgroup_proof_local_size({"local_size_x": "64"}) == ("64", "1", "1")
    assert (
        glsl_workgroup_proof_local_size({"local_size_x": "64", "local_size_x_id": "3"})
        is None
    )
    assert glsl_workgroup_proof_local_size({HOST_WORKGROUP_SIZE_KEY: True}) is None


SPECIALIZED_REDUCTION = """shader Reduction {
    RWStructuredBuffer<uint> outputWords @register(u0);
    compute {
        layout(local_size_x = 32, local_size_x_id = 4) in;
        void main(uint invocation @gl_LocalInvocationIndex) {
            outputWords[invocation] = WaveActiveSum(invocation);
        }
    }
}
"""


def test_software_subgroups_refuse_a_size_the_pipeline_specializes():
    with pytest.raises(OpenGLSoftwareSubgroupError) as raised:
        GLSLCodeGen(software_subgroup_width=32).generate(parse(SPECIALIZED_REDUCTION))
    assert raised.value.reason == "workgroup-size-specialized"
