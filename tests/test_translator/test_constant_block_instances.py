"""Constant blocks with an instance name, and ``ConstantBuffer<T>`` declarations that carry a buffer layout."""

import re
import struct

import pytest

import crosstl
from crosstl.translator.ast import (
    ArrayType,
    NamedType,
    ShaderStage,
    StructNode,
    VariableNode,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _translate(tmp_path, source, backend, format_output=False):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=format_output)


def _flat(code):
    return " ".join(code.split())


NAMED = """shader S {
    cbuffer Params @buffer_layout(std140) {
        vec4 v;
        float s;
    } params @binding(2);
    uniform Lights {
        vec4 colour;
    } lights[2];
    fragment {
        cbuffer Local {
            vec4 u;
        } local;
        vec4 main() @ gl_FragColor {
            return params.v * params.s + lights[1].colour + local.u;
        }
    }
}
"""


def test_an_instance_name_declares_a_constant_buffer_of_the_block_struct():
    program = _parse(NAMED)
    assert not getattr(program, "cbuffers", None)
    structs = {struct.name: struct for struct in program.structs}
    assert {"Params", "Lights"} <= set(structs)
    assert not getattr(structs["Params"], "is_cbuffer", False)
    assert [member.name for member in structs["Params"].members] == ["v", "s"]

    params, lights = program.global_variables
    assert isinstance(params, VariableNode) and params.name == "params"
    assert isinstance(params.var_type, NamedType)
    assert params.var_type.name == "ConstantBuffer"
    assert [argument.name for argument in params.var_type.generic_args] == ["Params"]
    assert "uniform" in params.qualifiers
    assert [attribute.name for attribute in params.attributes] == [
        "buffer_layout",
        "binding",
    ]
    assert isinstance(lights.var_type, ArrayType)
    assert lights.var_type.element_type.name == "ConstantBuffer"

    stage = program.stages[ShaderStage.FRAGMENT]
    assert [struct.name for struct in stage.local_structs] == ["Local"]
    assert not stage.local_cbuffers
    assert [variable.name for variable in stage.local_variables] == ["local"]


def test_a_block_without_an_instance_name_stays_a_constant_block():
    program = _parse("""shader S {
        cbuffer Params { vec4 v; };
        fragment { vec4 main() @ gl_FragColor { return v; } }
    }""")
    (block,) = program.cbuffers
    assert isinstance(block, StructNode) and block.is_cbuffer
    assert not program.global_variables


@pytest.mark.parametrize(
    "backend, declarations",
    [
        (
            "opengl",
            [
                "layout(std140, binding = 2) uniform Params",
                "} params;",
                "uniform Lights { vec4 colour; } lights[2];",
            ],
        ),
        (
            "directx",
            ["ConstantBuffer<Lights> lights[2]", "ConstantBuffer<Local> local"],
        ),
    ],
)
def test_every_target_declares_the_instance_it_reads(tmp_path, backend, declarations):
    code = _flat(_translate(tmp_path, NAMED, backend))
    for declaration in declarations:
        assert declaration in code
    assert "params.s" in code or "params.v" in code
    assert "lights[1].colour" in code and "local.u" in code


def test_metal_declares_the_instances_as_constant_references(tmp_path):
    source = NAMED.replace(" + lights[1].colour", "").replace(
        """    uniform Lights {
        vec4 colour;
    } lights[2];
""",
        "",
    )
    code = _flat(_translate(tmp_path, source, "metal"))
    assert "constant Local& local [[buffer(" in code
    assert "local.u" in code
    assert "params_block.params.s" in code


LAYOUT = """shader S {
    cbuffer Params @buffer_layout(LAYOUT) {
        float a;
        vec3 b;
        float c[2];
    } params @binding(1);
    fragment {
        vec4 main() @ gl_FragColor {
            return vec4(params.b, params.a + params.c[1]);
        }
    }
}
"""


def test_hlsl_reads_a_std140_constant_buffer_at_its_std140_offsets(tmp_path):
    code = _flat(_translate(tmp_path, LAYOUT.replace("LAYOUT", "std140"), "directx"))
    assert "ConstantBuffer<Params>" not in code
    assert "cbuffer params_block : register(b1)" in code
    assert "params.a = asfloat(__crossgl_params_block_registers[0].x);" in code
    assert "params.b = asfloat(__crossgl_params_block_registers[1].xyz);" in code


def test_metal_pads_a_std140_constant_buffer_to_its_std140_offsets(tmp_path):
    code = _flat(_translate(tmp_path, LAYOUT.replace("LAYOUT", "std140"), "metal"))
    assert (
        "struct Params { float a; char _pad1[12]; packed_float3 b; char _pad2[4]; "
        "float4 c[2]; };" in code
    )
    assert "constant params_block& params_block [[buffer(1)]]" in code


@pytest.mark.parametrize("backend", ["opengl", "vulkan"])
def test_std140_targets_place_an_hlsl_packed_constant_buffer(tmp_path, backend):
    # Formatted SPIR-V names its types.
    code = _translate(
        tmp_path, LAYOUT.replace("LAYOUT", "hlsl_cbuffer"), backend, format_output=True
    )
    if backend == "opengl":
        assert "uniform params_block" in code
        assert "uintBitsToFloat(crossgl_cbuffer_registers[0].yzw)" in code
    else:
        assert re.search(r"OpMemberDecorate %Params_placed\w* 1 Offset 4\b", code)
        assert re.search(r"OpMemberDecorate %Params_placed\w* 2 Offset 16\b", code)


@pytest.mark.parametrize(
    "backend, layout",
    [("opengl", "std140"), ("vulkan", "std140"), ("directx", "hlsl_cbuffer")],
)
def test_a_constant_buffer_in_the_targets_own_layout_keeps_its_declaration(
    tmp_path, backend, layout
):
    code = _flat(_translate(tmp_path, LAYOUT.replace("LAYOUT", layout), backend))
    assert "params_block" not in code


RUNTIME = """shader P {
    cbuffer Params @buffer_layout(std140) {
        float a;
        vec3 b;
        float c[2];
    } params @binding(1);
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = params.a + params.b.x * 2.0 + params.b.z * 4.0 + params.c[i] * 8.0;
        }
    }
}
"""


def test_metal_output_reads_the_std140_bytes_the_host_uploads(tmp_path):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    code = _translate(tmp_path, RUNTIME, "metal")
    slots = dict(re.findall(r"(\w+) \[\[buffer\((\d+)\)\]\]", code))
    library, error = device.newLibraryWithSource_options_error_(
        code, Metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    entry = re.search(r"kernel void (\w+)", code).group(1)
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error
    # std140: a at 0, b at 16, c[0] at 32 and c[1] at 48 (array stride 16).
    block = struct.pack("<f12x3f4xf12xf12x", 1.0, 2.0, 3.0, 5.0, 7.0, 11.0)
    constants = device.newBufferWithBytes_length_options_(
        block, len(block), Metal.MTLResourceStorageModeShared
    )
    results = device.newBufferWithLength_options_(8, Metal.MTLResourceStorageModeShared)
    (block_slot,) = (slot for name, slot in slots.items() if name != "outp")
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(constants, 0, int(block_slot))
    encoder.setBuffer_offset_atIndex_(results, 0, int(slots["outp"]))
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(2, 1, 1), Metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    assert command.error() is None, command.error()
    data = struct.unpack("<2f", bytes(results.contents().as_buffer(8)))
    assert data == (1.0 + 4.0 + 20.0 + 56.0, 1.0 + 4.0 + 20.0 + 88.0)
