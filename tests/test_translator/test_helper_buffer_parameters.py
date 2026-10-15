"""Buffer-attributed helper parameters keep their caller's pointer rebase."""

import shutil
import subprocess

import pytest

from crosstl.translator import parse
from crosstl.translator.ast import NamedType, PointerType
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.helper_buffer_parameters import (
    lower_helper_buffer_parameters,
)

STAGES = {"compute"}

ENTRY = """
    compute {
        @ stage_entry
        void select_value(StructuredBuffer<uint> input @buffer(0),
                          RWStructuredBuffer<uint> output @buffer(1),
                          uvec3 tid @gl_WorkGroupID) {
            input += tid.z;
            CALL;
        }
    }
"""

READER = """
    void run(StructuredBuffer<uint> input @buffer(0),
             RWStructuredBuffer<uint> output @buffer(1)) {
        buffer_store(output, 0, buffer_load(input, 1));
    }
"""


def _source(helpers, call="run(input, output)"):
    return f"shader main {{{ENTRY.replace('CALL', call)}{helpers}\n}}"


def _helper(ast, name="run"):
    (function,) = (function for function in ast.functions if function.name == name)
    return function


def _types(function):
    return {parameter.name: parameter.param_type for parameter in function.parameters}


def test_rebased_entry_buffers_reach_helpers_as_offset_views():
    generated = HLSLCodeGen().generate_entry(parse(_source(READER)), "select_value")
    assert (
        "void run(StructuredBuffer<uint> input, int64_t input_offset, "
        "RWStructuredBuffer<uint> output, int64_t output_offset)" in generated
    )
    assert "output[uint(output_offset)] = input[uint((input_offset + 1))];" in generated
    assert "input_offset += uint(tid.z);" in generated
    assert "run(input, int64_t(input_offset), output, int64_t(0));" in generated
    assert ".Load(" not in generated


def test_lowering_restores_source_pointer_contracts():
    ast = lower_helper_buffer_parameters(
        parse(
            _source(
                READER.replace(
                    "RWStructuredBuffer<uint> output @buffer(1))",
                    "RWStructuredBuffer<uint> output @buffer(1), "
                    "StructuredBuffer<int> strides @buffer(2) @constant)",
                ),
                "run(input, output, input)",
            )
        ),
        STAGES,
    )
    function = _helper(ast)
    types = _types(function)
    assert [
        (types[name].address_space, parameter.qualifiers, parameter.attributes)
        for name, parameter in zip(types, function.parameters)
    ] == [
        ("device", ["const", "device"], []),
        ("device", ["device"], []),
        ("constant", ["const", "constant"], []),
    ]
    assert all(isinstance(value, PointerType) for value in types.values())
    (stage,) = list(ast.stages.values())
    entry = stage.entry_point
    assert all(
        isinstance(parameter.param_type, NamedType)
        for parameter in entry.parameters[:2]
    )


def test_nested_helpers_forward_their_own_rebases():
    outer = """
    void outer(StructuredBuffer<uint> input @buffer(0),
               RWStructuredBuffer<uint> output @buffer(1)) {
        input += 2;
        run(input, output);
    }
    """
    generated = HLSLCodeGen().generate_entry(
        parse(_source(READER + outer, "outer(input + 1, output)")), "select_value"
    )
    assert "input_offset += int(2);" in generated
    assert "run(input, int64_t(input_offset), output, int64_t(output_offset));" in (
        generated
    )
    assert "outer(input, int64_t((input_offset + 1)), output, int64_t(0));" in (
        generated
    )


@pytest.mark.parametrize(
    "body,call,parameter",
    [
        # Reassigning the view itself is not a pointer form the ABI tracks.
        (
            "input = input; buffer_store(output, 0, buffer_load(input, 1));",
            None,
            "input",
        ),
        # Builtins other than indexed loads and stores observe the resource.
        ("buffer_store(output, 0, buffer_dimensions(input));", None, "input"),
        # Stores used as values have no statement form to rewrite.
        (
            "uint value = buffer_store(output, 0, buffer_load(input, 1));",
            None,
            "output",
        ),
        # A local copy has no tracked root at the call site.
        (
            "buffer_store(output, 0, buffer_load(input, 1));",
            "uint copy = 0u; run(input, output); run(copy, output)",
            "input",
        ),
    ],
)
def test_unsupported_uses_keep_the_resource_contract(body, call, parameter):
    helpers = READER.replace("buffer_store(output, 0, buffer_load(input, 1));", body)
    ast = lower_helper_buffer_parameters(
        parse(_source(helpers, call or "run(input, output)")), STAGES
    )
    assert isinstance(_types(_helper(ast))[parameter], NamedType)


def test_helpers_without_buffer_attributes_are_unchanged():
    helpers = READER.replace(" @buffer(0)", "").replace(" @buffer(1)", "")
    ast = lower_helper_buffer_parameters(parse(_source(helpers)), STAGES)
    assert all(isinstance(value, NamedType) for value in _types(_helper(ast)).values())


def test_glsl_emits_only_bound_specializations_of_buffer_helpers(tmp_path):
    generated = GLSLCodeGen().generate_entry(parse(_source(READER)), "select_value")
    # GLSL has no buffer-block parameters: the bound specialization is the only
    # valid form, and an unspecialized `uint input_[]` original cannot compile.
    assert "_[]," not in generated and "_[])" not in generated
    assert generated.count("void run_glsl_input_input_uint_output_output_uint(") == 2
    assert "output_[output_offset] = input_[(input_offset + int(1))];" in generated
    assert (
        "run_glsl_input_input_uint_output_output_uint(int(input_offset), int(0));"
        in generated
    )
    compiler = shutil.which("glslangValidator")
    if compiler is None:
        pytest.skip("glslangValidator is not installed")
    shader = tmp_path / "helper.comp"
    shader.write_text(generated, encoding="utf-8")
    result = subprocess.run(
        [compiler, "--target-env", "opengl", str(shader)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_generated_helper_compiles_with_dxc(tmp_path):
    compiler = shutil.which("dxc")
    if compiler is None:
        pytest.skip("DXC is not installed")
    outer = """
    void outer(StructuredBuffer<uint> input @buffer(0),
               RWStructuredBuffer<uint> output @buffer(1)) {
        input += 2;
        run(input, output);
    }
    """
    source = tmp_path / "helper.hlsl"
    source.write_text(
        HLSLCodeGen().generate_entry(
            parse(_source(READER + outer, "outer(input + 1, output)")),
            "select_value",
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [compiler, "-T", "cs_6_6", "-E", "CSMain", "-WX", str(source)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
