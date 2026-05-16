import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.hip_codegen import HipCodeGen
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
            "static particles: RWStructuredBuffer<Particles> = Default::default();",
        ),
        (MojoCodeGen, "var particles: RWStructuredBuffer<Particles>"),
        (CudaCodeGen, "RWStructuredBuffer<Particles> particles;"),
        (HipCodeGen, "RWStructuredBuffer<Particles> particles;"),
        (SlangCodeGen, "RWStructuredBuffer<Particles> particles;"),
    ],
)
def test_generic_resource_types_preserve_element_type(codegen_cls, expected):
    ast = parse(RESOURCE_SHADER)

    generated_code = codegen_cls().generate(ast)

    assert expected in generated_code
