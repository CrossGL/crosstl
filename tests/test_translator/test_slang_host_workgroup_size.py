"""Slang sizes ``numthreads`` from the host's macros or from module-scope specialization constants.

A kernel whose size only the host chooses reads ``CROSSGL_WORKGROUP_SIZE_{X,Y,Z}`` when the host defines them on the
slangc command line, as the generated HLSL and GLSL do, and otherwise names three ``vk::constant_id`` constants with
the ids the generated GLSL and SPIR-V use, which slangc lowers to SPIR-V ``LocalSizeId``. Targets without
specialization constants refuse the constants (slangc E55205), so a compile without the macros fails closed there.
Specialized sizes name the same kind of constants, declared once per module however many entries name them.
"""

import os
import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.slang_workgroup_size import (
    SlangWorkgroupSizeConstants,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

MACROS = "CROSSGL_WORKGROUP_SIZE_X, CROSSGL_WORKGROUP_SIZE_Y, CROSSGL_WORKGROUP_SIZE_Z"
DEFINED = (
    "defined(CROSSGL_WORKGROUP_SIZE_X) && defined(CROSSGL_WORKGROUP_SIZE_Y) && "
    "defined(CROSSGL_WORKGROUP_SIZE_Z)"
)
CONSTANTS = (
    "crossgl_workgroup_size_x, crossgl_workgroup_size_y, crossgl_workgroup_size_z"
)
HOST_NUMTHREADS = (
    f"#if {DEFINED}\n[numthreads({MACROS})]\n#else\n[numthreads({CONSTANTS})]\n#endif\n"
)


def _declarations(ids):
    lines = "".join(
        f"[vk::constant_id({spec_id})] const int crossgl_workgroup_size_{axis} = 1;\n"
        for axis, spec_id in zip("xyz", ids)
    )
    return f"#if !({DEFINED})\n{lines}#endif\n"


ROTATE = """
shader main {
    compute {
        buffer float* data;
        shared float tile[64];
        @ host_workgroup_size
        void main() {
            uint lid = gl_LocalInvocationIndex;
            uint gid = gl_GlobalInvocationID.x;
            tile[lid] = data[gid];
            barrier();
            data[gid] = tile[(lid + 1u) % gl_WorkGroupSize.x];
        }
    }
}
"""

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"

METAL_KERNELS = HEADER + """
constant float first [[function_constant(0)]];
constant float second [[function_constant(2)]];
kernel void rotate(device float* data [[buffer(0)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_index_in_threadgroup]],
                   uint3 count [[threads_per_threadgroup]]) {
    threadgroup float tile[256];
    tile[lid] = data[gid] * first;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    data[gid] = tile[(lid + 1) % count.x] * second;
}
kernel void shift(device float* data [[buffer(0)]],
                  uint gid [[thread_position_in_grid]],
                  uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float tile[256];
    tile[lid] = data[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    data[gid] = tile[(lid + 2) % 256];
}
kernel void scale(device float* data [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    data[gid] = data[gid] * 2.0f;
}
"""

SPECIALIZED_PAIR = """
shader Probe {
    compute {
        layout(local_size_x = FIRST, local_size_x_id = 4) in;
        buffer float* outp;
        @ stage_entry
        void first() {
            outp[gl_GlobalInvocationID.x] = float(gl_WorkGroupSize.x);
        }
    }
    compute {
        layout(local_size_x = 64, local_size_x_id = 4) in;
        buffer float* outq;
        @ stage_entry
        void second() {
            outq[gl_GlobalInvocationID.x] = float(gl_WorkGroupSize.x);
        }
    }
}
"""


def _translate(tmp_path, source, name="shader.cgl", **options):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(
        str(path),
        backend="slang",
        format_output=False,
        source_options=options or None,
    )


def _slangc():
    configured = os.environ.get("SLANGC")
    if configured and os.path.exists(configured):
        return configured
    return shutil.which("slangc")


def _compile(tmp_path, code, target, *arguments, entries=("main",)):
    shader = tmp_path / "shader.slang"
    shader.write_text(code)
    output = tmp_path / f"shader.{target}"
    command = [_slangc(), str(shader), "-target", target, "-o", str(output)]
    for entry in entries:
        command += ["-entry", entry]
    result = subprocess.run(
        command + list(arguments), capture_output=True, text=True, check=False
    )
    return result, output


def test_host_sized_entry_reads_the_macros_or_three_constants(tmp_path):
    code = _translate(tmp_path, ROTATE)
    assert code.startswith(_declarations((0, 1, 2)))
    assert HOST_NUMTHREADS + '[shader("compute")]\n' in code
    assert "uint3(WorkgroupSize()).x" in code
    assert code.count("vk::constant_id") == 3


def test_host_constants_avoid_the_program_specialization_ids(tmp_path):
    code = _translate(tmp_path, METAL_KERNELS, "kernels.metal")
    # function_constant(0) and (2) hold ids 0 and 2, as in the generated GLSL and SPIR-V.
    assert _declarations((1, 3, 4)) in code
    # Both size-dependent kernels share the constants; the size-independent one keeps one invocation per group.
    assert code.count(HOST_NUMTHREADS) == 2
    assert code.count("const int crossgl_workgroup_size_x") == 1
    assert "[numthreads(1, 1, 1)]" in code
    assert "uint3 count = uint3(WorkgroupSize());" in code


def test_threadgroup_size_option_keeps_literal_numthreads(tmp_path):
    code = _translate(
        tmp_path, METAL_KERNELS, "kernels.metal", threadgroup_size=(64, 1, 1)
    )
    assert "[numthreads(64, 1, 1)]" in code
    assert "CROSSGL_WORKGROUP_SIZE" not in code
    assert "crossgl_workgroup_size" not in code


def test_specialized_entries_share_a_constant_and_split_on_a_new_default(tmp_path):
    shared = _translate(tmp_path, SPECIALIZED_PAIR.replace("FIRST", "64"))
    assert shared.startswith(
        "[vk::constant_id(4)] const int crossgl_workgroup_size_x = 64;\n\n"
    )
    assert shared.count("const int crossgl_workgroup_size_x") == 1
    assert shared.count("[numthreads(crossgl_workgroup_size_x, 1, 1)]") == 2
    split = _translate(tmp_path, SPECIALIZED_PAIR.replace("FIRST", "32"))
    assert split.startswith(
        "[vk::constant_id(4)] const int crossgl_workgroup_size_x = 32;\n"
        "[vk::constant_id(4)] const int crossgl_workgroup_size_x_1 = 64;\n"
    )
    assert "[numthreads(crossgl_workgroup_size_x_1, 1, 1)]" in split


def test_constants_are_named_in_declaration_order():
    constants = SlangWorkgroupSizeConstants(None)
    specialized = {"local_size_x": 8, "local_size_x_id": 3}
    assert constants.numthreads(specialized) == (
        "[numthreads(crossgl_workgroup_size_x, 1, 1)]\n"
    )
    assert constants.numthreads(specialized) == (
        "[numthreads(crossgl_workgroup_size_x, 1, 1)]\n"
    )
    assert constants.numthreads({"local_size": [4, 2, 1]}) == "[numthreads(4, 2, 1)]\n"
    assert constants.declarations() == (
        "[vk::constant_id(3)] const int crossgl_workgroup_size_x = 8;\n\n"
    )
    assert SlangWorkgroupSizeConstants(None).declarations() == ""


def test_workgroup_size_parameter_outside_compute_is_refused():
    from crosstl.translator.codegen.slang_codegen import SlangCodeGen
    from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused

    code = """
    shader P {
        fragment {
            vec4 main(uvec3 size @gl_WorkGroupSize) @gl_FragColor {
                return vec4(float(size.x));
            }
        }
    }
    """
    with pytest.raises(SlangLoweringRefused) as refused:
        SlangCodeGen().generate(Parser(Lexer(code).get_tokens()).parse())
    assert any(
        "gl_WorkGroupSize parameter 'size'" in reason and "WorkgroupSize()" in reason
        for reason in refused.value.reasons
    )


requires_slangc = pytest.mark.skipif(_slangc() is None, reason="slangc is required")
MACRO_DEFINES = (
    "-DCROSSGL_WORKGROUP_SIZE_X=64",
    "-DCROSSGL_WORKGROUP_SIZE_Y=1",
    "-DCROSSGL_WORKGROUP_SIZE_Z=1",
)


@requires_slangc
@pytest.mark.skipif(shutil.which("spirv-dis") is None, reason="spirv-dis is required")
def test_slangc_specializes_spirv_without_the_macros(tmp_path):
    result, binary = _compile(tmp_path, _translate(tmp_path, ROTATE), "spirv")
    assert result.returncode == 0, result.stderr
    disassembly = subprocess.run(
        ["spirv-dis", str(binary)], capture_output=True, text=True, check=True
    ).stdout
    assert re.search(
        r"OpExecutionModeId %main LocalSizeId %crossgl_workgroup_size_x "
        r"%crossgl_workgroup_size_y %crossgl_workgroup_size_z",
        disassembly,
    )
    for spec_id in range(3):
        assert re.search(rf"OpDecorate %\w+ SpecId {spec_id}\b", disassembly)
    if shutil.which("spirv-val"):
        validation = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.3", str(binary)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert validation.returncode == 0, validation.stderr


@requires_slangc
def test_slangc_fixes_the_macro_size_for_hlsl(tmp_path):
    result, output = _compile(
        tmp_path,
        _translate(tmp_path, ROTATE),
        "hlsl",
        "-profile",
        "cs_6_0",
        *MACRO_DEFINES,
    )
    assert result.returncode == 0, result.stderr
    # The macros leave the constants undeclared, so slangc reports no implicit global shader parameter.
    assert "warning" not in result.stderr + result.stdout
    assert "[numthreads(64, 1, 1)]" in output.read_text()


@requires_slangc
def test_slangc_refuses_hlsl_without_the_macros(tmp_path):
    result, _ = _compile(tmp_path, _translate(tmp_path, ROTATE), "hlsl")
    assert result.returncode != 0
    assert "E55205" in result.stderr + result.stdout


@requires_slangc
@pytest.mark.skipif(shutil.which("spirv-val") is None, reason="spirv-val is required")
def test_slangc_compiles_two_entries_that_specialize_one_id(tmp_path):
    code = _translate(tmp_path, SPECIALIZED_PAIR.replace("FIRST", "32"))
    result, binary = _compile(tmp_path, code, "spirv", entries=("first", "second"))
    assert result.returncode == 0, result.stderr
    validation = subprocess.run(
        ["spirv-val", "--target-env", "vulkan1.3", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert validation.returncode == 0, validation.stderr
