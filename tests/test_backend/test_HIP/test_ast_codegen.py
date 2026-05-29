from crosstl.backend.HIP.HipAst import (
    ConstantMemoryNode,
    HipDevicePropertyNode,
    HipErrorHandlingNode,
    SharedMemoryNode,
    TextureAccessNode,
    VariableNode,
)
from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter
from crosstl.backend.HIP.HipParser import HipProgramNode


def generate_program(*statements):
    return HipToCrossGLConverter().generate(HipProgramNode(list(statements)))


def test_explicit_shared_memory_ast_nodes_emit_workgroup_declarations():
    code = generate_program(
        SharedMemoryNode("float", "scratch"),
        SharedMemoryNode("int", "histogram", "128"),
    )

    assert "var<workgroup> scratch: f32;" in code
    assert "var<workgroup> histogram: array<i32, 128>;" in code
    assert "SharedMemoryNode(" not in code


def test_explicit_constant_memory_ast_nodes_emit_uniform_declarations():
    code = generate_program(
        ConstantMemoryNode("unsigned int", "flags"),
        ConstantMemoryNode("float", "scale", "0.5"),
        ConstantMemoryNode("int", "zero", 0),
    )

    assert "@group(0) @binding(0) var<uniform> flags: u32;" in code
    assert "@group(0) @binding(0) var<uniform> scale: f32 = 0.5;" in code
    assert "@group(0) @binding(0) var<uniform> zero: i32 = 0;" in code
    assert "ConstantMemoryNode(" not in code


def test_explicit_texture_access_ast_nodes_emit_texture_sampling():
    code = generate_program(
        VariableNode("float4", "sampled", TextureAccessNode("tex", ["u", "v"])),
        VariableNode("float4", "sampled_vec", TextureAccessNode("tex", "uv")),
    )

    assert "var sampled: vec4<f32> = texture(tex, vec2<f32>(u, v));" in code
    assert "var sampled_vec: vec4<f32> = texture(tex, uv);" in code
    assert "TextureAccessNode(" not in code


def test_explicit_hip_error_handling_ast_nodes_emit_metadata_expressions():
    code = generate_program(
        VariableNode(
            "const char*", "message", HipErrorHandlingNode("hipGetErrorString", "err")
        ),
        VariableNode(
            "const char*", "name", HipErrorHandlingNode("hipGetErrorName", "err")
        ),
        VariableNode("hipError_t", "status", HipErrorHandlingNode("hipError_t", "err")),
    )

    assert 'var message: ptr<i8> = /* HIP error string: err */ "";' in code
    assert 'var name: ptr<i8> = /* HIP error name: err */ "";' in code
    assert (
        "var status: hipError_t = (/* HIP error status: hipError_t, expr: err */ err);"
        in code
    )
    assert "HipErrorHandlingNode(" not in code


def test_explicit_hip_device_property_ast_nodes_emit_metadata_expressions():
    code = generate_program(
        VariableNode("int", "sms", HipDevicePropertyNode("multiProcessorCount", 0)),
        VariableNode("int", "warp", HipDevicePropertyNode("warpSize")),
    )

    assert (
        "var sms: i32 = (/* HIP device property: multiProcessorCount, device: 0 */ 0);"
        in code
    )
    assert "var warp: i32 = (/* HIP device property: warpSize */ 0);" in code
    assert "HipDevicePropertyNode(" not in code
