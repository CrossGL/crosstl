import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.mojo_codegen import MojoCodeGen
from crosstl.translator.codegen.rust_codegen import RustCodeGen
from crosstl.translator.codegen.slang_codegen import SlangCodeGen

RESOURCE_SHADER = """
shader main {
    struct Particles {
        float4 position;
        float mass;
    };

    RWStructuredBuffer<Particles> particles;

    vertex {
        void main() {
        }
    }
}
"""


@pytest.mark.parametrize(
    ("codegen_cls", "expected"),
    [
        (
            RustCodeGen,
            "static PARTICLES: std::sync::LazyLock<RWStructuredBuffer<Particles>> = "
            "std::sync::LazyLock::new(|| Default::default());",
        ),
        (MojoCodeGen, "var particles: RWStructuredBuffer[Particles]"),
        (CudaCodeGen, "Particles* particles;"),
        (HLSLCodeGen, "RWStructuredBuffer<Particles> particles : register(u0);"),
        (
            GLSLCodeGen,
            "layout(std430, binding = 0) buffer particlesBuffer { Particles particles[]; };",
        ),
        (HipCodeGen, "Particles* particles;"),
        (MetalCodeGen, "device Particles* particles [[buffer(0)]]"),
        (SlangCodeGen, "RWStructuredBuffer<Particles> particles : register(u0);"),
    ],
)
def test_generic_resource_types_preserve_element_type(codegen_cls, expected):
    ast = parse(RESOURCE_SHADER)

    generated_code = codegen_cls().generate(ast)

    assert expected in generated_code


def test_structured_buffer_parameters_preserve_element_type_for_core_backends():
    shader = """
    shader main {
        compute {
            void scale(RWStructuredBuffer<float> data, StructuredBuffer<float> input, uint index) {
                float value = buffer_load(input, index);
                buffer_store(data, index, value * 2.0);
            }
        }
    }
    """
    ast = parse(shader)

    hlsl = HLSLCodeGen().generate(ast)
    assert (
        "void scale(RWStructuredBuffer<float> data, StructuredBuffer<float> input, uint index)"
        in hlsl
    )
    assert "float value = input.Load(index);" in hlsl
    assert "data.Store(index, (value * 2.0));" in hlsl

    metal = MetalCodeGen().generate(ast)
    assert (
        "void scale(device float* data, const device float* input, uint index)" in metal
    )
    assert "float value = input[index];" in metal
    assert "data[index] = value * 2.0;" in metal

    glsl = GLSLCodeGen().generate(ast)
    assert "void scale(float data[], float input_[], uint index)" in glsl
    assert "float value = input_[index];" in glsl
    assert "data[index] = (value * 2.0);" in glsl
    assert "StructuredBuffer" not in glsl

    slang = SlangCodeGen().generate(ast)
    assert (
        "void scale(RWStructuredBuffer<float> data, "
        "StructuredBuffer<float> input, uint index)" in slang
    )
    assert "float value = input.Load(index);" in slang
    assert "data.Store(index, value * 2.0);" in slang
    assert "buffer_load(" not in slang
    assert "buffer_store(" not in slang


def test_buffer_builtin_aliases_do_not_instantiate_same_named_generics():
    shader = """
    shader main {
        RWStructuredBuffer<float> data;
        StructuredBuffer<float> input;

        generic<T> fn buffer_load(T value, uint index) -> T {
            return value;
        }

        generic<T> fn buffer_store(T buffer, uint index, float value) -> void {
        }

        compute {
            void scale(uint index) {
                float value = buffer_load(input, index);
                buffer_store(data, index, value * 2.0);
            }
        }
    }
    """
    ast = parse(shader)

    hlsl = HLSLCodeGen().generate(ast)
    assert "float value = input.Load(index);" in hlsl
    assert "data.Store(index, (value * 2.0));" in hlsl
    assert "buffer_load_" not in hlsl
    assert "buffer_store_" not in hlsl

    metal = MetalCodeGen().generate(ast)
    assert "float value = input[index];" in metal
    assert "data[index] = value * 2.0;" in metal
    assert "buffer_load_" not in metal
    assert "buffer_store_" not in metal

    glsl = GLSLCodeGen().generate(ast)
    assert "float value = input[index];" in glsl
    assert "data[index] = (value * 2.0);" in glsl
    assert "buffer_load_" not in glsl
    assert "buffer_store_" not in glsl
