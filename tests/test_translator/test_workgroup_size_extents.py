"""Arrays whose extent reads ``gl_WorkGroupSize`` (GLSL 4.60 section 7.1.3: a constant expression)."""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.ast import MemberAccessNode
from crosstl.translator.codegen.lowering_refusals import LoweringRefused
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused
from crosstl.translator.codegen.workgroup_size_extents import (
    fold_workgroup_size_extents,
    program_workgroup_size,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from tests.spirv_friendly_names import translate_spirv

TILE = """#version 450
layout(LAYOUT) in;
layout(std430, binding = 0) buffer Out { float data[]; };
shared float tile[EXTENT];
void main() {
    uint i = gl_LocalInvocationID.x;
    tile[i] = float(i);
    barrier();
    data[gl_GlobalInvocationID.x] = tile[gl_WorkGroupSize.x - 1u - i];
}
"""

FIXED = "local_size_x = 64, local_size_y = 1, local_size_z = 1"
SPECIALIZED = (
    "local_size_x = 64, local_size_y = 1, local_size_z = 1, local_size_x_id = 3"
)

TWO_SIZES = """
shader main {
    shared float tile[gl_WorkGroupSize.x];
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        @ stage_entry
        void first(RWStructuredBuffer<float> data @buffer(0), uint lid @gl_LocalInvocationIndex) {
            tile[lid] = 1.0;
            buffer_store(data, lid, tile[lid]);
        }
    }
    compute {
        layout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in;
        @ stage_entry
        void second(RWStructuredBuffer<float> data @buffer(1), uint lid @gl_LocalInvocationIndex) {
            tile[lid] = 2.0;
            buffer_store(data, lid, tile[lid]);
        }
    }
}
"""

REFUSED = (LoweringRefused, SpirvLoweringRefused)

HELPER = """#version 450
layout(local_size_x = 32) in;
layout(std430, binding = 0) buffer Out { uint data[]; };
uint width() { return gl_WorkGroupSize.x; }
void main() { data[gl_LocalInvocationID.x] = width(); }
"""


def _tile(layout=FIXED, extent="gl_WorkGroupSize.x"):
    return TILE.replace("LAYOUT", layout).replace("EXTENT", extent)


def _write(tmp_path, source, suffix=".comp"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return path


def _translate(tmp_path, source, backend, suffix=".comp"):
    path = _write(tmp_path, source, suffix)
    if backend == "vulkan":
        return translate_spirv(path)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    return result.stdout


def _glslang(tmp_path, code):
    if shutil.which("glslangValidator"):
        shader = tmp_path / "out.comp"
        shader.write_text(code)
        _run(["glslangValidator", "-V", str(shader), "-o", str(tmp_path / "out.spv")])


def _spirv_val(tmp_path, code):
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        asm.write_text(code)
        _run(["spirv-as", "--target-env", "vulkan1.0", str(asm), "-o", str(binary)])
        _run(["spirv-val", "--target-env", "vulkan1.0", str(binary)])


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def test_glsl_declares_the_local_size_before_the_array_that_reads_it(tmp_path):
    code = _translate(tmp_path, _tile(), "opengl")

    layout = f"layout({FIXED}) in;"
    assert "shared float tile[gl_WorkGroupSize.x];" in code
    assert code.index(layout) < code.index("shared float tile")
    _glslang(tmp_path, code)


def test_glsl_declares_the_local_size_before_a_helper_that_reads_it(tmp_path):
    code = _translate(tmp_path, HELPER, "opengl")

    assert code.index("layout(local_size_x = 32") < code.index("uint width()")
    _glslang(tmp_path, code)


def test_glsl_keeps_a_specialized_extent_and_declares_its_ids_first(tmp_path):
    code = _translate(tmp_path, _tile(SPECIALIZED), "opengl")

    assert "shared float tile[gl_WorkGroupSize.x];" in code
    assert code.index("local_size_x_id = 3") < code.index("shared float tile")
    _glslang(tmp_path, code)


@pytest.mark.parametrize(
    "backend, declaration",
    [
        ("metal", "threadgroup float tile[64];"),
        ("directx", "groupshared float tile[64];"),
    ],
)
def test_msl_and_hlsl_size_the_array_with_the_declared_size(
    tmp_path, backend, declaration
):
    code = _translate(tmp_path, _tile(), backend)

    assert declaration in code
    assert "tile[gl_WorkGroupSize" not in code


def test_spirv_sizes_the_workgroup_array_with_a_constant_length(tmp_path):
    code = _translate(tmp_path, _tile(), "vulkan")

    assert "%_arr_float_int_64 = OpTypeArray %float %int_64" in code
    assert "%tile = OpVariable %_ptr_Workgroup__arr_float_int_64 Workgroup" in code
    assert "OpTypePointer Workgroup %_runtimearr" not in code
    _spirv_val(tmp_path, code)


@pytest.mark.parametrize("backend", ["metal", "directx", "vulkan"])
def test_an_extent_expression_of_components_is_folded(tmp_path, backend):
    extent = "gl_WorkGroupSize.x * 2u + gl_WorkGroupSize.y"
    code = _translate(
        tmp_path, _tile("local_size_x = 64, local_size_y = 3", extent), backend
    )

    if backend == "vulkan":
        assert "OpTypeArray %float %int_131" in code
        _spirv_val(tmp_path, code)
    else:
        assert "float tile[131];" in code


@pytest.mark.parametrize("backend", ["metal", "vulkan"])
def test_a_specialized_extent_is_refused(tmp_path, backend):
    with pytest.raises(REFUSED, match=r"gl_WorkGroupSize\.x.*specializes"):
        _translate(tmp_path, _tile(SPECIALIZED), backend)


@pytest.mark.parametrize("backend", ["metal", "directx", "vulkan", "opengl"])
def test_entries_of_different_sizes_leave_a_shared_extent_without_a_size(
    tmp_path, backend
):
    with pytest.raises(REFUSED, match="different"):
        _translate(tmp_path, TWO_SIZES, backend, suffix=".cgl")


def test_the_program_size_reports_why_a_component_has_no_value():
    ast = _parse(TWO_SIZES)

    sizes = program_workgroup_size(ast, "Metal")

    assert "different sizes (8, 64)" in sizes[0]
    assert sizes[1:] == [1, 1]


def test_folding_leaves_the_input_program_unchanged():
    ast = _parse(
        "shader s { shared float tile[gl_WorkGroupSize.y * 4]; compute { "
        "layout(local_size_x = 8, local_size_y = 2) in; void main() { tile[0] = 1.0; } } }"
    )
    refusals = []

    folded = fold_workgroup_size_extents(ast, "Metal", refusals.append)

    assert refusals == []
    assert isinstance(ast.global_variables[0].var_type.size.left, MemberAccessNode)
    assert folded.global_variables[0].var_type.size.value == 8
