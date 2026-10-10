"""Logical-subgroup matrix exchange without native subgroup dependencies."""

import json
import os
import re
import sys
from pathlib import Path

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLCooperativeMatrixError,
    OpenGLSoftwareSubgroupError,
)
from tests.test_translator.test_directx_cooperative_matrix import (
    MATRIX,
    _project_request,
    _source,
)
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile


def _codegen():
    return GLSLCodeGen(
        cooperative_matrix_software_lowering=True, software_subgroup_width=32
    )


def test_software_matrix_compiles_without_hardware_subgroup_dependencies(tmp_path):
    generator = _codegen()
    ast = parse(_source())
    generated = generator.generate_stage(ast, "compute")
    assert generated == generator.generate_stage(ast, "compute")
    assert generated.count("shared vec2 ") == 2
    assert generated.count("barrier();") == 2
    assert generated.count("precise float term") == 16
    assert generated.count("precise float sum") == 2
    assert "gl_Subgroup" not in generated
    assert "subgroupShuffle" not in generated
    assert "GL_KHR_shader_subgroup" not in generated
    assert "CROSSTL_REQUIRED_SUBGROUP_WIDTH" not in generated
    assert "#define CROSSTL_SOFTWARE_SUBGROUP_WIDTH 32" in generated
    _compile(generated, "opengl", tmp_path)
    generator.set_software_subgroup_width(None)
    plain = generator.generate_stage(
        parse("shader Plain { compute { void main() {} } }"), "compute"
    )
    assert "shared vec2 " not in plain
    assert "barrier();" not in plain


@pytest.mark.parametrize("width,height", [(16, 1), (33, 1), (16, 3)])
def test_matrix_requires_complete_logical_subgroups(width, height):
    with pytest.raises(OpenGLSoftwareSubgroupError) as error:
        _codegen().generate_stage(parse(_source(width=width, height=height)), "compute")
    assert error.value.reason == "incomplete-matrix-subgroup"


@pytest.mark.parametrize(
    "body",
    [
        "if (invocation < 16u) { CALL; }",
        "if (invocation == 0u) { return; } CALL;",
        "for (uint step = 0u; step < invocation; ++step) { CALL; }",
        f"{MATRIX} result = invocation == 0u ? CALL : accumulator;",
    ],
)
def test_matrix_rejects_divergent_barriers(body):
    call = "cooperative_matrix_multiply_accumulate(left, right, accumulator)"
    with pytest.raises(OpenGLSoftwareSubgroupError):
        _codegen().generate_stage(parse(_source(body.replace("CALL", call))), "compute")


def test_matrix_checks_transitive_helper_control_flow(tmp_path):
    helper = f"""{MATRIX} product({MATRIX} a, {MATRIX} b, {MATRIX} c, uint count) {{
        for (uint step = 0u; step < count; ++step) {{
            cooperative_matrix_multiply_accumulate(c, a, b, c);
        }}
        return c;
    }}"""
    body = f"""{MATRIX} result = product(left, right, accumulator, 2u);
        outputWords[index] = asuint(cooperative_matrix_element(result, 0));"""
    generated = _codegen().generate_stage(
        parse(_source(body, helpers=helper)), "compute"
    )
    _compile(generated, "opengl", tmp_path)
    with pytest.raises(OpenGLSoftwareSubgroupError):
        _codegen().generate_stage(
            parse(
                _source(
                    body.replace("accumulator, 2u", "accumulator, invocation"),
                    helpers=helper,
                )
            ),
            "compute",
        )


@pytest.mark.parametrize(
    "arguments,reason",
    [
        ("left, right", "invalid-argument-count"),
        ("left, right, 1.0", "missing-operand-contract"),
        (
            "cooperative_matrix_add(left, right), left, right, accumulator",
            "destination-requires-lvalue",
        ),
    ],
)
def test_matrix_rejects_invalid_operands(arguments, reason):
    with pytest.raises(OpenGLCooperativeMatrixError) as error:
        _codegen().generate_stage(
            parse(_source(f"cooperative_matrix_multiply_accumulate({arguments});")),
            "compute",
        )
    assert error.value.reason == reason


@pytest.mark.parametrize("replacement", ["int", "uint"])
def test_matrix_rejects_incompatible_arithmetic(replacement):
    with pytest.raises(OpenGLCooperativeMatrixError) as error:
        _codegen().generate_stage(
            parse(
                _source().replace(
                    "CooperativeMatrix<float,", f"CooperativeMatrix<{replacement},"
                )
            ),
            "compute",
        )
    assert error.value.reason == "incompatible-multiply-accumulate-contract"


def test_matrix_retains_explicit_policy_and_stage_contracts():
    with pytest.raises(OpenGLCooperativeMatrixError) as error:
        GLSLCodeGen(cooperative_matrix_software_lowering=True).generate_stage(
            parse(_source()), "compute"
        )
    assert error.value.reason == "exact-subgroup-width-required"
    with pytest.raises(OpenGLSoftwareSubgroupError) as error:
        _codegen().generate_stage(
            parse(_source().replace("compute {", "fragment {")), "fragment"
        )
    assert error.value.reason == "entry-point-contract-invalid"
    with pytest.raises(OpenGLSoftwareSubgroupError) as error:
        _codegen().generate_stage(
            parse(_source().replace("@numthreads", "@WaveSize(32) @numthreads")),
            "compute",
        )
    assert error.value.reason == "hardware-contract-conflict"


def test_matrix_scratch_identifiers_avoid_source_names(tmp_path):
    generated = _codegen().generate_stage(parse(_source()), "compute")
    names = re.findall(r"shared vec2 (\w+)\[", generated)
    helpers = "\n".join(
        f"float {name}(float value) {{ return value; }}" for name in names
    )
    generated = _codegen().generate_stage(parse(_source(helpers=helpers)), "compute")
    actual = re.findall(r"shared vec2 (\w+)\[", generated)
    assert len(actual) == 2 and not set(names) & set(actual)
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize("shape", [(32, 4, 1), (16, 8, 1)])
def test_public_matrix_package_retains_logical_policy(tmp_path, shape):
    _, request, _ = _project_request(tmp_path, "opengl", shape=shape)
    report = json.loads((tmp_path / "report.json").read_text())
    policies = [value for value in _policies(report)]
    assert len(policies) == 1
    assert policies[0]["effectiveWidth"] == 32
    assert policies[0]["requirements"] == [
        "operation:CooperativeMatrixMultiplyAccumulate"
    ]
    _compile(request.artifact_path.read_text(), "opengl", tmp_path)


def _policies(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "softwareSubgroupPolicy":
                yield child
            else:
                yield from _policies(child)
    elif isinstance(value, list):
        for child in value:
            yield from _policies(child)


@pytest.mark.parametrize("shape", [(32, 4, 1), (16, 8, 1)])
@pytest.mark.parametrize("rounding", [False, True])
def test_public_matrix_executes(tmp_path, shape, rounding):
    if os.environ.get("CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME=1 for native matrix execution"
        )
    if not sys.platform.startswith("linux"):
        pytest.skip("OpenGL execution requires the Linux runtime job")
    _, request, outputs = _project_request(
        tmp_path, "opengl", shape=shape, rounding=rounding
    )
    _execute(request, outputs, tmp_path)


def test_matrix_execution_uses_existing_native_gate():
    from tests.ci_helpers import assert_paths_covered
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    assert (
        workflow.count("tests/test_translator/test_opengl_cooperative_matrix.py") == 1
    )
    assert 'CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME: "1"' in workflow
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_opengl_cooperative_matrix.py",
        )
