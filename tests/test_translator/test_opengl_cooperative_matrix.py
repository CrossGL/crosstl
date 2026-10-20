"""Logical-subgroup matrix exchange without native subgroup dependencies."""

import json
import os
import re
import sys
from pathlib import Path

import pytest

from crosstl.project import build_runtime_artifact_manifest, validate_project_report
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
from tests.test_translator.test_directx_matrix_rounding import (
    PROFILES,
    _edge_case,
    _plain_product_case,
    _profile_case,
)
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile


def _codegen(**options):
    return GLSLCodeGen(
        cooperative_matrix_software_lowering=True, software_subgroup_width=32, **options
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


def test_matrix_plain_products_accumulate_into_positive_zero(tmp_path):
    # Metal's operator* equals simdgroup_multiply_accumulate into a zero
    # matrix; MLX's Winograd transforms chain two: (G * g) * Gt.
    body = f"""{MATRIX} result = cooperative_matrix_multiply(
            cooperative_matrix_multiply(left, right), accumulator);
        outputWords[index * 2u] = floatBitsToUint(cooperative_matrix_element(result, 0));
        outputWords[index * 2u + 1u] = floatBitsToUint(cooperative_matrix_element(result, 1));"""
    generated = _codegen().generate_stage(parse(_source(body)), "compute")
    assert "cooperative_matrix_multiply(" not in generated
    assert generated.count("(float[2](float(0), float(0))))") == 2
    assert generated.count("barrier();") == 2
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize("arguments", ["left", "left, right, accumulator"])
def test_matrix_plain_product_requires_two_operands(arguments):
    body = f"{MATRIX} result = cooperative_matrix_multiply({arguments});"
    with pytest.raises(OpenGLCooperativeMatrixError) as error:
        _codegen().generate_stage(parse(_source(body)), "compute")
    assert error.value.reason == "invalid-argument-count"


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
    policies = [*_policies(report)]
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
    import yaml

    from tests import workflow_text
    from tests.ci_helpers import assert_paths_covered

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    steps = yaml.safe_load(workflow)["jobs"]["portable-host"]["steps"]
    required = next(step for step in steps if step.get("id") == "collective-helpers")
    assert (
        required["run"].count("tests/test_translator/test_opengl_cooperative_matrix.py")
        == 1
    )
    assert required["env"]["CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME"] == "1"
    assert not required.get("continue-on-error")
    for event in ("pull_request", "push"):
        assert_paths_covered(
            workflow_text.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_opengl_cooperative_matrix.py",
        )


@pytest.mark.parametrize("profile", PROFILES)
def test_matrix_rounding_compiles_integer_intermediates(tmp_path, profile):
    generated = _codegen(cooperative_matrix_mma_profile=profile).generate_stage(
        parse(_source()), "compute"
    )
    assert generated.count("#version") == 1
    assert generated.count("SoftFloat Release 3e") == 1
    assert "uint sum0 = floatBitsToUint(accumulator.elements[0]);" in generated
    assert "result.elements[1] = uintBitsToFloat(sum1);" in generated
    assert "precise float term" not in generated
    assert "gl_Subgroup" not in generated
    assert generated.count("barrier();") == 2
    if profile.startswith("fma-"):
        assert "term0_0LeftBits, term0_0RightBits, sum0," in generated
        assert "term0_0Product" not in generated
    else:
        assert "(term0_0LeftBits ^ term0_0RightBits) & 0x80000000u" in generated
        assert "sum0, 0x3f800000u, term0_0Product," in generated
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize("profile", PROFILES)
def test_exact_hardware_subgroup_matrix_retains_rounding(tmp_path, profile):
    from tests.test_translator.test_codegen.test_GLSL_codegen import (
        assert_glsl_compute_validates_if_available,
    )

    source = _source().replace("@numthreads", "@WaveSize(32) @numthreads")
    generated = GLSLCodeGen(
        cooperative_matrix_software_lowering=True,
        cooperative_matrix_mma_profile=profile,
    ).generate_stage(parse(source), "compute")
    assert "subgroupShuffle" in generated
    assert "crossgl_matrix_round_bits" in generated
    assert "uint sum0 = floatBitsToUint(accumulator.elements[0]);" in generated
    assert "result.elements[1] = uintBitsToFloat(sum1);" in generated
    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "matrix_rounding",
        spirv_target="spirv1.3",
        validate_spirv=True,
    )


@pytest.mark.parametrize("profile", [True, 32, "", "nearest", [], {}])
def test_matrix_rounding_rejects_invalid_profiles(profile):
    with pytest.raises(ValueError, match="cooperative_matrix_mma_profile"):
        _codegen(cooperative_matrix_mma_profile=profile)


def test_matrix_rounding_default_and_generation_state():
    ast = parse(_source())
    assert _codegen().generate_stage(ast, "compute") == _codegen(
        cooperative_matrix_mma_profile=None
    ).generate_stage(ast, "compute")
    generator = _codegen(cooperative_matrix_mma_profile="rne-gradual")
    first = generator.generate_stage(ast, "compute")
    assert first == generator.generate_stage(ast, "compute")
    generator.set_cooperative_matrix_mma_profile("fma-rne-flush")
    assert first != generator.generate_stage(ast, "compute")
    generator.set_software_subgroup_width(None)
    plain = generator.generate_stage(
        parse("shader Plain { compute { void main() {} } }"), "compute"
    )
    assert "crossgl_matrix_round_" not in plain
    assert "SoftFloat" not in plain


@pytest.mark.parametrize("name", ["floatBitsToUint", "uintBitsToFloat"])
@pytest.mark.parametrize("kind", ["function", "global", "struct"])
def test_matrix_rounding_rejects_captured_bitcasts(name, kind):
    declarations = {
        "function": f"float {name}(float value) {{ return value; }}",
        "global": f"float {name};",
        "struct": f"struct {name} {{ float value; }};",
    }
    with pytest.raises(OpenGLCooperativeMatrixError) as error:
        _codegen(cooperative_matrix_mma_profile="rne-gradual").generate_stage(
            parse(_source(helpers=declarations[kind])), "compute"
        )
    assert error.value.reason == "rounding-builtin-shadowed"


def test_matrix_rounding_helpers_avoid_user_names(tmp_path):
    source = _source(
        helpers="uint crossgl_matrix_round_bits(uint value) { return value; }"
    ).replace(
        "outputWords[index * 2u] = ",
        "outputWords[index * 2u] = crossgl_matrix_round_bits(0u) + ",
    )
    generated = _codegen(cooperative_matrix_mma_profile="rne-gradual").generate_stage(
        parse(source), "compute"
    )
    names = re.findall(r"uint (\w+)\(uint a, uint b, uint c, bool", generated)
    assert len(names) == 1 and names[0] != "crossgl_matrix_round_bits"
    assert "uint crossgl_matrix_round_bits(uint value)" in generated
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize("profile", PROFILES)
def test_project_matrix_rounding_package_records_profile(tmp_path, profile):
    _, request, _ = _project_request(
        tmp_path, "opengl", mma_profile=profile, word_case=_profile_case(profile)
    )
    path = tmp_path / "report.json"
    report = json.loads(path.read_text())
    assert (
        report["artifacts"][0]["provenance"]["cooperativeMatrixMmaProfile"] == profile
    )
    assert request.artifact["provenance"]["cooperativeMatrixMmaProfile"] == profile
    assert "crossgl_matrix_round_bits" in request.artifact_path.read_text()
    _compile(request.artifact_path.read_text(), "opengl", tmp_path)
    assert validate_project_report(path)["success"]
    assert build_runtime_artifact_manifest(path)["success"]
    for invalid in (
        None,
        "native",
        "rne-flush" if profile == "rne-gradual" else "rne-gradual",
        1,
        [],
    ):
        provenance = report["artifacts"][0]["provenance"]
        if invalid is None:
            provenance.pop("cooperativeMatrixMmaProfile", None)
        else:
            provenance["cooperativeMatrixMmaProfile"] = invalid
        path.write_text(json.dumps(report))
        result = validate_project_report(path)
        assert not result["success"]
        assert "cooperativeMatrixMmaProfile" in json.dumps(result["diagnostics"])


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("shape", [(32, 4, 1), (16, 8, 1)])
def test_project_matrix_rounding_native_package(tmp_path, profile, shape):
    if os.environ.get("CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME=1 for native matrix execution"
        )
    if not sys.platform.startswith("linux"):
        pytest.skip("OpenGL execution requires the Linux runtime job")
    _, request, expected = _project_request(
        tmp_path,
        "opengl",
        shape=shape,
        mma_profile=profile,
        word_case=_profile_case(profile),
    )
    _execute(request, expected, tmp_path)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("inputs", ["edges", "quarter", "rounding"])
def test_project_matrix_rounding_native_edges(tmp_path, profile, inputs):
    if os.environ.get("CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME=1 for native matrix execution"
        )
    if not sys.platform.startswith("linux"):
        pytest.skip("OpenGL execution requires the Linux runtime job")
    _, request, expected = _project_request(
        tmp_path,
        "opengl",
        mma_profile=profile,
        word_case=(
            _edge_case(profile)
            if inputs == "edges"
            else _profile_case(profile, rounding=inputs == "rounding", products=1)
        ),
        single_product=True,
    )
    _execute(request, expected, tmp_path)


@pytest.mark.parametrize("profile", ["fma-rne-gradual", "rne-gradual"])
def test_project_matrix_plain_products_native_package(tmp_path, profile):
    if os.environ.get("CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME=1 for native matrix execution"
        )
    if not sys.platform.startswith("linux"):
        pytest.skip("OpenGL execution requires the Linux runtime job")
    _, request, expected = _project_request(
        tmp_path,
        "opengl",
        mma_profile=profile,
        word_case=_plain_product_case(profile),
        plain_products=True,
    )
    _execute(request, expected, tmp_path)
