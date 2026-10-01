"""Bounds-check exits use immutable, dimension-specific invocation facts."""

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLSoftwareSubgroupError,
)
from tests.test_translator.test_metal_builtin_ownership import _compile


def _source(body, shape=(32, 1, 1), helpers=""):
    return f"""shader ExitBounds {{
        RWStructuredBuffer<float> results @register(u0);
        cbuffer Params @register(b1) {{ int limit; }};
        {helpers}
        compute {{
            @numthreads({shape[0]}, {shape[1]}, {shape[2]})
            void main(uint3 grid @gl_GlobalInvocationID, uint invocation @gl_LocalInvocationIndex) {{
                {body}
                float total = WaveActiveSum(float(invocation));
                results[invocation] = total;
            }}
        }}
    }}"""


@pytest.mark.parametrize(
    "shape,axis", [((32, 1, 1), "y"), ((32, 1, 1), "z"), ((32, 4, 1), "z")]
)
def test_uniform_bounds_exit_compiles(tmp_path, shape, axis):
    code = _source(
        f"const int row = int(grid.{axis}); if (row >= limit) {{ return; }}", shape
    )
    generated = GLSLCodeGen(software_subgroup_width=32).generate_stage(
        parse(code), "compute"
    )
    assert "return;" in generated and "barrier();" in generated
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize(
    "shape,axis", [((32, 1, 1), "x"), ((32, 4, 1), "y"), ((32, 1, 2), "z")]
)
def test_nonuniform_component_cannot_justify_exit(shape, axis):
    with pytest.raises(OpenGLSoftwareSubgroupError):
        GLSLCodeGen(software_subgroup_width=32).generate_stage(
            parse(
                _source(
                    f"const int row = int(grid.{axis}); if (row >= limit) {{ return; }}",
                    shape,
                )
            ),
            "compute",
        )


@pytest.mark.parametrize(
    "body",
    [
        "grid.y = invocation; const int row = int(grid.y); if (row >= limit) { return; }",
        "int row = int(grid.y); row += int(invocation); if (row >= limit) { return; }",
        "int row = int(grid.y); rewrite(row, invocation); if (row >= limit) { return; }",
        "int row = int(grid.y); int& alias = row; alias = int(invocation); if (row >= limit) { return; }",
        "int row = int(grid.y); { int row = int(invocation); if (row >= limit) { return; } }",
        "{ int row = int(grid.y); } int row = int(invocation); if (row >= limit) { return; }",
    ],
)
def test_mutation_aliases_and_shadowing_invalidate_exit_proof(body):
    with pytest.raises(OpenGLSoftwareSubgroupError):
        GLSLCodeGen(software_subgroup_width=32).generate_stage(
            parse(
                _source(
                    body,
                    helpers="void rewrite(inout int value, uint lane) { value = int(lane); }",
                )
            ),
            "compute",
        )


def test_helper_parameter_cannot_impersonate_uniform_builtin():
    code = _source(
        "float value = guarded(grid);",
        helpers="float guarded(uint3 gl_GlobalInvocationID) { if (gl_GlobalInvocationID.y > 0u) { return 0.0; } return WaveActiveSum(float(gl_GlobalInvocationID.x)); }",
    )
    with pytest.raises(OpenGLSoftwareSubgroupError):
        GLSLCodeGen(software_subgroup_width=32).generate_stage(parse(code), "compute")
