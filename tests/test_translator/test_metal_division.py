"""Explicit source division profiles preserve evaluation and narrow results."""

import json
import os
import sys

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalAst import BinaryOpNode, FunctionCallNode
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalDivisionProfileError,
    MetalToCrossGLConverter,
)
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.project import (
    ProjectConfig,
    build_runtime_artifact_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
    validate_project_report,
)
from tests.test_translator.test_division_math import GUARD, REQUIRE_ENV, _oracle, _pairs
from tests.test_translator.test_fused_math import _dispatch
from tests.test_translator.test_metal_builtin_ownership import _compile

SOURCE = """#include <metal_stdlib>
using namespace metal;
using Real = float;
struct Cell { Real value; };
static float record(thread uint& count, float value) { count += 1; return value; }
kernel void divide_profile(const device uint* values [[buffer(0)]],
                           device uint* results [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    Real a = as_type<float>(values[2u * i]);
    float b = as_type<float>(values[2u * i + 1u]);
    uint count = 0;
    float scalar = record(count, a) / b;
    float2 pair = float2(a, b) / float2(b, a);
    float3 triple = metal::precise::divide(float3(a, a, b), float3(b));
    float4 quad = a / float4(b, a, b, a);
    Cell cell; cell.value = a;
    results[4u + 15u * i] = as_type<uint>(scalar);
    results[5u + 15u * i] = as_type<uint>(pair.x);
    results[6u + 15u * i] = as_type<uint>(pair.y);
    results[7u + 15u * i] = as_type<uint>(triple.x);
    results[8u + 15u * i] = as_type<uint>(triple.y);
    results[9u + 15u * i] = as_type<uint>(triple.z);
    results[10u + 15u * i] = as_type<uint>(quad.x);
    results[11u + 15u * i] = as_type<uint>(quad.y);
    results[12u + 15u * i] = as_type<uint>(quad.z);
    results[13u + 15u * i] = as_type<uint>(quad.w);
    results[14u + 15u * i] = as_type<uint>(cell.value / b);
    results[15u + 15u * i] = as_type<uint>((a / b) / a);
    results[16u + 15u * i] = count;
    results[17u + 15u * i] = as_type<uint>(float(bfloat(a / b)));
    results[18u + 15u * i] = as_type<uint>(float(bfloat(a) / bfloat(b)));
}
"""


def _translate(tmp_path, source, target="crossgl", profile="rne-flush"):
    path = tmp_path / "division.metal"
    path.write_text(source, encoding="utf-8")
    return translate(
        str(path),
        backend=target,
        format_output=False,
        source_options={"binary32_division_profile": profile},
    )


@pytest.mark.parametrize("profile", ["rne-gradual", "rne-flush"])
@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_division_profiles_compile(tmp_path, target, profile):
    generated = _translate(tmp_path, SOURCE, target, profile)
    for width in ("", "2", "3", "4"):
        assert f"metal_divide_float{width}(" in generated
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


@pytest.mark.parametrize("profile", [True, 1, "", "toward-zero", [], {}])
def test_division_profile_rejects_invalid_configuration(profile):
    with pytest.raises(ValueError, match="binary32_division_profile"):
        MetalToCrossGLConverter(binary32_division_profile=profile)


def test_division_profile_is_explicit_and_resets_between_modules(tmp_path):
    assert "__crossgl_divide" not in _translate(tmp_path, SOURCE, profile=None)
    converter = MetalToCrossGLConverter(binary32_division_profile="rne-gradual")
    for source, expected in (
        (SOURCE, True),
        ("float value(float a) { return a; }", False),
    ):
        ast = MetalParser(MetalLexer(source).tokenize()).parse()
        assert ("__crossgl_divide_bits" in converter.generate(ast)) == expected


def test_division_helpers_avoid_source_name_collisions(tmp_path):
    source = """
    uint __crossgl_divide_bits(uint a) { return a; }
    float __crossgl_metal_divide_float(float a) { return a; }
    float value(float a, float b) { return a / b; }
    """
    generated = _translate(tmp_path, source)
    assert "uint __crossgl_divide_bits_(uint a, uint b" in generated
    assert "float __crossgl_metal_divide_float_(float a, float b)" in generated


def test_division_builtin_ownership_and_fast_mode(tmp_path):
    source = """
    float divide(float a, float b) { return a + b; }
    float user(float a, float b) { return ::divide(a, b); }
    float builtin(float a, float b) { return metal::divide(a, b); }
    float precise_mode(float a, float b) { return precise::divide(a, b); }
    float fast_mode(float a, float b) { return metal::fast::divide(a, b); }
    """
    generated = _translate(tmp_path, source)
    assert "return divide__metal_overload_1(a, b);" in generated
    assert (
        generated.count("return __crossgl_metal_divide_float(float(a), float(b));") == 2
    )
    assert "return ((a) / (b));" in generated


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_division_builtin_and_reciprocal_compile(tmp_path, target):
    source = SOURCE.replace(
        "record(count, a) / b", "metal::precise::divide(record(count, a), b)"
    ).replace("cell.value / b", "metal::divide(1.0f, b)")
    generated = _translate(tmp_path, source, target)
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


def test_division_keeps_user_operator_and_other_arithmetic_types(tmp_path):
    source = """
    struct Ratio { float x; };
    Ratio crosstl_metal_operator_divide__Ratio__Ratio(Ratio a, Ratio b) {
        Ratio c; c.x = a.x + b.x; return c;
    }
    Ratio custom(Ratio a, Ratio b) { return a / b; }
    int integral(int a, int b) { return a / b; }
    half narrow(half a, half b) { return a / b; }
    """
    generated = _translate(tmp_path, source)
    assert "__crossgl_divide_bits" not in generated
    assert generated.count("return a / b;") == 2
    assert "operator_divide" in generated


@pytest.mark.parametrize(
    "source",
    [
        "float value = 1.0f / 3.0f;",
        "float value = metal::divide(1.0f, 3.0f);",
        "float value(float a, float b) { a /= b++; return a; }",
        "float value(float a, float b) { float2 c(a); c.x /= b; return c.x; }",
    ],
)
def test_division_profile_diagnoses_unrepresentable_contexts(tmp_path, source):
    with pytest.raises(MetalDivisionProfileError) as error:
        _translate(tmp_path, source)
    assert error.value.profile == "rne-flush"
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-division-profile-unsupported"
    )


def test_division_profile_diagnoses_unresolved_type():
    converter = MetalToCrossGLConverter(binary32_division_profile="rne-flush")
    with pytest.raises(MetalDivisionProfileError, match="unresolved"):
        converter.generate_expression(BinaryOpNode("unresolved", "/", "1.0f"))


@pytest.mark.parametrize("profile", ["rne-gradual", "rne-flush"])
def test_division_profile_keeps_sizeof_arithmetic_integral(tmp_path, profile):
    converter = MetalToCrossGLConverter(binary32_division_profile=profile)
    size = FunctionCallNode("sizeof", ["float"])
    assert converter.expression_metal_type(size) == "size_t"
    assert (
        converter.metal_binary_expression_type(BinaryOpNode("8", "/", size))
        == "uint64_t"
    )
    generated = _translate(
        tmp_path,
        "static constant int count = 8 / sizeof(float);\n"
        "float ratio(float a, float b) { return a / b; }",
        profile=profile,
    )
    assert "count = 8 / 4" in generated
    assert "return __crossgl_metal_divide_float(float(a), float(b));" in generated


def test_sizeof_type_does_not_match_qualified_calls():
    converter = MetalToCrossGLConverter()
    converter.generate(
        MetalParser(
            MetalLexer("float identity(float x) { return x; }").tokenize()
        ).parse()
    )
    assert (
        converter.expression_metal_type(FunctionCallNode("other::sizeof", ["float"]))
        is None
    )


COMPOUND_SOURCE = """#include <metal_stdlib>
using namespace metal;
struct Cell { float value; };
kernel void divide_compound(const device uint* values [[buffer(0)]],
                           device uint* results [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    float a = as_type<float>(values[2u * i]);
    float b = as_type<float>(values[2u * i + 1u]);
    float local = a;
    float returned = (local /= b);
    Cell cell; cell.value = a; cell.value /= b;
    float3 vector(a); vector /= b;
    float3 component(a); component[1] /= b;
    bfloat narrow = bfloat(a); bfloat denominator = bfloat(b); narrow /= denominator;
    results[4u + 8u * i] = as_type<uint>(local);
    results[5u + 8u * i] = as_type<uint>(returned);
    results[6u + 8u * i] = as_type<uint>(cell.value);
    results[7u + 8u * i] = as_type<uint>(vector.z);
    results[8u + 8u * i] = as_type<uint>(component[1]);
    results[9u + 8u * i] = as_type<uint>(component[0]);
    results[10u + 8u * i] = as_type<uint>(float(narrow));
    results[11u + 8u * i] = as_type<uint>(1.0f / b);
}
"""

BUFFER_SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void divide_buffer(const device uint* values [[buffer(0)]],
                          device float* results [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    float a = as_type<float>(values[2u * i]);
    float b = as_type<float>(values[2u * i + 1u]);
    results[4u + 2u * i] = a;
    float returned = (results[4u + 2u * i] /= b);
    results[5u + 2u * i] = returned;
}
"""


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_profiled_compound_division_compiles(tmp_path, target):
    generated = _translate(tmp_path, COMPOUND_SOURCE, target)
    assert "crossgl_metal_divide_assign" in generated
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_profiled_buffer_division_compiles(tmp_path, target):
    source = """#include <metal_stdlib>
    using namespace metal;
    kernel void update(device float* values [[buffer(0)]],
                       uint i [[thread_position_in_grid]]) { values[i] /= 3.0f; }
    """
    generated = _translate(tmp_path, source, target)
    assert "crossgl_metal_divide_assign" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("width", [8, 16])
def test_profiled_wide_division_uses_per_lane_helpers(tmp_path, width):
    source = f"""using Wide = metal::vec<float, {width}>;
    Wide value(Wide a, Wide b) {{ Wide c = a / b; c /= b; return c; }}
    """
    generated = _translate(tmp_path, source)
    assert generated.count("__crossgl_metal_divide_float(float(") == width * 2


@pytest.mark.parametrize("profile", ["rne-gradual", "rne-flush"])
def test_project_retains_division_profile_in_report(tmp_path, profile):
    (tmp_path / "division.metal").write_text(SOURCE, encoding="utf-8")
    config = ProjectConfig(
        root=tmp_path,
        targets=("opengl",),
        source_options={"metal": {"binary32_division_profile": profile}},
    )
    report = translate_project(
        config, validate=False, run_toolchains=False, format_output=False
    )
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 1, data["diagnostics"]
    assert (
        data["project"]["sourceOptions"]["metal"]["binary32_division_profile"]
        == profile
    )
    assert data["artifacts"][0]["provenance"]["binary32DivisionProfile"] == profile
    path = tmp_path / "report.json"
    report.write_json(path)
    assert validate_project_report(path)["success"]
    for value in (
        None,
        "rne-flush" if profile == "rne-gradual" else "rne-gradual",
        "toward-zero",
    ):
        if value is None:
            data["artifacts"][0]["provenance"].pop("binary32DivisionProfile", None)
        else:
            data["artifacts"][0]["provenance"]["binary32DivisionProfile"] = value
        path.write_text(json.dumps(data), encoding="utf-8")
        validation = validate_project_report(path)
        assert not validation["success"]
        assert "binary32DivisionProfile" in json.dumps(validation["diagnostics"])


def test_division_profile_resolution_and_runtime_provenance(tmp_path):
    (tmp_path / "division.metal").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "crosstl.toml").write_text(
        """[project]
targets = ["directx", "opengl"]
[project.source_options.metal]
binary32_division_profile = "rne-gradual"
[project.source_options.metal.target_options.opengl.source_patterns."division.metal"]
binary32_division_profile = "rne-flush"
""",
        encoding="utf-8",
    )
    report = translate_project(
        load_project_config(tmp_path),
        validate=False,
        run_toolchains=False,
        format_output=False,
    )
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 2, data["diagnostics"]
    expected = {"directx": "rne-gradual", "opengl": "rne-flush"}
    assert {
        a["target"]: a["provenance"]["binary32DivisionProfile"]
        for a in data["artifacts"]
    } == expected
    path = tmp_path / "report.json"
    report.write_json(path)
    assert validate_project_report(path)["success"]
    manifest = build_runtime_artifact_manifest(path)
    assert manifest["success"], manifest["diagnostics"]
    assert {
        a["target"]: a["provenance"]["binary32DivisionProfile"]
        for a in manifest["artifacts"]
    } == expected
    manifest_path = tmp_path / "artifacts.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    package = build_runtime_package(manifest_path, tmp_path / "package")
    assert package["success"], package


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_division_profile_survives_saved_intermediate(tmp_path, target):
    intermediate = tmp_path / "saved.cgl"
    intermediate.write_text(_translate(tmp_path, SOURCE), encoding="utf-8")
    generated = translate(str(intermediate), backend=target, format_output=False)
    assert "metal_divide_float" in generated
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


def test_ci_runs_source_profiles_once_per_native_target():
    from pathlib import Path

    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate binary32 arithmetic"
    )
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert workflow.count("tests/test_translator/test_metal_division.py") == 1
    assert "tests/test_translator/test_metal_division.py" in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step


def _bfloat(word):
    if (word & 0x7FFFFFFF) > 0x7F800000:
        return (word & 0xFFFF0000) | 0x00400000
    return ((word + 0x7FFF + ((word >> 16) & 1)) >> 16 & 0xFFFF) << 16


@pytest.mark.parametrize("kind", ["operators", "compound", "buffer"])
@pytest.mark.parametrize("profile", ["rne-gradual", "rne-flush", "source"])
def test_division_profiles_execute(tmp_path, profile, kind, monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native division profiles")
    from functools import partial

    from tests.test_translator import test_fused_math as native_math

    monkeypatch.setattr(
        native_math,
        "_compile",
        partial(
            _compile,
            directx_compile_flags=("-enable-16bit-types",),
            metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
        ),
    )
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    if profile == "source" and target != "metal":
        pytest.skip("The original source control requires Metal")
    pairs = _pairs()
    if kind != "operators":
        pairs = pairs[::31]
    source = {
        "operators": SOURCE,
        "compound": COMPOUND_SOURCE,
        "buffer": BUFFER_SOURCE,
    }[kind]
    field_count = {"operators": 15, "compound": 8, "buffer": 2}[kind]
    expected = [GUARD] * 4
    flush = profile != "rne-gradual"
    for a, b in pairs:
        q = _oracle(a, b, flush)
        narrow = _bfloat(_oracle(_bfloat(a), _bfloat(b), flush))
        if kind == "buffer":
            expected.extend([q, q])
        elif kind == "compound":
            expected.extend([q, q, q, q, q, a, narrow, _oracle(0x3F800000, b, flush)])
        else:
            expected.extend(
                [
                    q,
                    q,
                    _oracle(b, a, flush),
                    q,
                    q,
                    _oracle(b, b, flush),
                    q,
                    _oracle(a, a, flush),
                    q,
                    _oracle(a, a, flush),
                    q,
                    _oracle(q, a, flush),
                    1,
                    _bfloat(q),
                    narrow,
                ]
            )
    expected.extend([GUARD] * 4)
    (tmp_path / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    generated = (
        source if profile == "source" else _translate(tmp_path, source, target, profile)
    )
    entry = {
        "operators": "divide_profile",
        "compound": "divide_compound",
        "buffer": "divide_buffer",
    }[kind]
    actual, evidence = _dispatch(
        tmp_path,
        target,
        generated,
        pairs,
        len(expected),
        entry=entry if target == "metal" else None,
        initial_output=[GUARD] * len(expected),
        output_dtype="float32" if kind == "buffer" else "uint32",
    )
    mismatches = []
    for i, (want, got) in enumerate(zip(expected, actual)):
        payload = 4 <= i < len(expected) - 4
        narrowed = (i - 4) % field_count in {
            "operators": (13, 14),
            "compound": (6,),
            "buffer": (),
        }[kind]
        both_nan = (want & 0x7FFFFFFF) > 0x7F800000 and (got & 0x7FFFFFFF) > 0x7F800000
        classification_only = payload and (profile == "source" or narrowed)
        if want != got and not (classification_only and both_nan):
            mismatches.append({"index": i, "expected": want, "actual": got})
    evidence.update(
        profile=profile,
        kind=kind,
        guardCount=8,
        mismatchCount=len(mismatches),
        mismatches=mismatches,
        nanComparison="exact helper results; classification after bfloat conversion and in original source",
    )
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    assert len(actual) == len(expected)
    assert not mismatches, mismatches[:10]
