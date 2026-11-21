"""Metal floating-point directives: contraction controls are carried as precise functions, other settings fail.

A directive that forbids or limits contraction, or forbids reassociation, makes every function of the unit
``@precise`` (crosstl/backend/Metal/floating_point.py). Settings CrossGL cannot express are refused before any
artifact is published.
"""

import json
import re

import pytest

from crosstl import translate
from crosstl.backend.Metal.floating_point import (
    MetalContractionDirectiveError,
    contraction_directive_requires_precise,
)
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project import (
    build_runtime_artifact_manifest,
    load_project_config,
    translate_project,
    validate_project_report,
)

PRECISE_DIRECTIVES = (
    "clang fp contract(off)",
    "clang fp contract(on)",
    "clang fp reassociate(off)",
    "clang fp reassociate(off) contract(off) exceptions(ignore)",
    "clang /* source policy */ fp contract /* gap */ (off)",
    "METAL fp contract(off)",
    "METAL fp contract(on)",
    "STDC FP_CONTRACT OFF",
    "STDC FP_CONTRACT ON",
    "STDC FP_CONTRACT DEFAULT",
    "OPENCL FP_CONTRACT OFF",
)
PERMISSIVE_DIRECTIVES = (
    "clang fp contract(fast)",
    "clang fp contract(fast-honor-pragmas)",
    "clang fp reassociate(on) exceptions(ignore) eval_method(source)",
    "METAL fp contract(fast)",
)
UNSUPPORTED_DIRECTIVES = (
    "clang fp exceptions(strict)",
    "clang fp exceptions(maytrap)",
    "clang fp eval_method(double)",
    "clang fp contract(sometimes)",
    "clang fp unknown(on)",
    "clang fp",
    "clang fp contract(off) junk",
    "METAL fp contract(sometimes)",
    "METAL fp contract",
    "STDC FP_CONTRACT MAYBE",
)
SOURCE = """#include <metal_stdlib>
using namespace metal;
float helper(float a, float b) {
    return a * b - a;
}
kernel void expression(const device float* values [[buffer(0)]],
                       device float* results [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    DIRECTIVE
    float a = values[2u*i];
    float b = values[2u*i+1u];
    results[i] = a*a - b*b + helper(a, b);
}
"""


def spell(directive, form):
    if form == "line":
        return "#pragma " + directive
    prefix = "L" if form == "wide-operator" else ""
    return "_Pragma(" + prefix + json.dumps(directive) + ")"


def write_source(tmp_path, directive_text, name="expression.metal"):
    path = tmp_path / name
    path.write_text(SOURCE.replace("DIRECTIVE", directive_text), encoding="utf-8")
    return path


def crossgl(path):
    return translate(str(path), backend="cgl", format_output=False)


def precise_function_count(code):
    return len(re.findall(r"\)\s*(?:@\s*\w+(?:\([^)]*\))?\s*)*@precise\b", code))


def assert_every_function_precise(code, functions=2):
    assert precise_function_count(code) == functions, code
    assert not re.search(r"#pragma\s+(?:clang|METAL|STDC|OPENCL)\b", code), code


@pytest.mark.parametrize("directive", PRECISE_DIRECTIVES)
def test_directive_classification_requires_precise(directive):
    assert contraction_directive_requires_precise(directive) is True


@pytest.mark.parametrize("directive", PERMISSIVE_DIRECTIVES)
def test_permissive_directives_need_nothing(directive):
    assert contraction_directive_requires_precise(directive) is False


@pytest.mark.parametrize(
    "directive",
    (
        "clang loop unroll(full)",
        "clang loop unroll(full) // contract(off)",
        "custom contract(off)",
        "once",
    ),
)
def test_other_pragmas_are_not_floating_point_directives(directive):
    assert contraction_directive_requires_precise(directive) is None


@pytest.mark.parametrize("directive", PRECISE_DIRECTIVES)
@pytest.mark.parametrize("form", ("line", "operator", "wide-operator"))
def test_contraction_controls_make_every_function_precise(tmp_path, directive, form):
    assert_every_function_precise(
        crossgl(write_source(tmp_path, spell(directive, form)))
    )


@pytest.mark.parametrize("directive", PERMISSIVE_DIRECTIVES)
def test_permissive_directives_keep_functions_unchanged(tmp_path, directive):
    code = crossgl(write_source(tmp_path, spell(directive, "line")))
    assert precise_function_count(code) == 0, code
    assert not re.search(r"#pragma\s+(?:clang|METAL)\b", code), code


@pytest.mark.parametrize("directive", UNSUPPORTED_DIRECTIVES)
@pytest.mark.parametrize("form", ("line", "operator"))
def test_unrepresentable_settings_are_refused(directive, form):
    source = SOURCE.replace("DIRECTIVE", spell(directive, form))
    with pytest.raises(MetalContractionDirectiveError) as error:
        MetalPreprocessor().preprocess(source)
    assert error.value.project_diagnostic_code == (
        "project.translate.metal-contraction-unsupported"
    )
    assert error.value.missing_capabilities == ("metal.source-expression-contraction",)


@pytest.mark.parametrize(
    "source",
    (
        "#pragma clang fp contract(off)\n" + SOURCE.replace("DIRECTIVE", ""),
        "# /* gap */ pragma clang fp contract(off)\n" + SOURCE.replace("DIRECTIVE", ""),
        SOURCE.replace("DIRECTIVE", "{\n#pragma clang fp contract(off)\n}"),
        "template<typename T> T unused(T a) {\n"
        "#pragma clang fp contract(off)\nreturn a*a-a;\n}\n"
        + SOURCE.replace("DIRECTIVE", ""),
        '#define CONTROL _Pragma("clang fp contract(off)")\n'
        + SOURCE.replace("DIRECTIVE", "CONTROL"),
        "#define PRAGMA_INNER(x) _Pragma(#x)\n"
        "#define PRAGMA(x) PRAGMA_INNER(x)\n"
        + SOURCE.replace("DIRECTIVE", "PRAGMA(clang fp contract(off))"),
    ),
)
def test_directive_survives_template_rewriting(tmp_path, source):
    path = tmp_path / "expression.metal"
    path.write_text(source, encoding="utf-8")
    # The unused template is kept as a generic CrossGL function.
    assert_every_function_precise(
        crossgl(path), functions=3 if "unused" in source else 2
    )


@pytest.mark.parametrize(
    "spelling",
    ("#pragma clang fp contract(off)", '_Pragma("clang fp contract(off)")'),
)
def test_parser_without_preprocessing_records_the_directive(spelling):
    tokens = MetalLexer(
        SOURCE.replace("DIRECTIVE", spelling), preprocess=False
    ).tokenize()
    assert MetalParser(tokens).parse().no_contraction is True


@pytest.mark.parametrize(
    "spelling",
    ("#pragma clang fp exceptions(strict)", '_Pragma("clang fp exceptions(strict)")'),
)
def test_parser_without_preprocessing_refuses_unsupported_settings(spelling):
    tokens = MetalLexer(
        SOURCE.replace("DIRECTIVE", spelling), preprocess=False
    ).tokenize()
    with pytest.raises(MetalContractionDirectiveError):
        MetalParser(tokens).parse()


@pytest.mark.parametrize(
    "prefix",
    (
        "// #pragma clang fp contract(off)\n",
        "/*\n#pragma clang fp contract(off)\n*/\n",
        '#define UNUSED _Pragma("clang fp contract(off)")\n',
        "#if 0\n#pragma clang fp contract(off)\n#endif\n",
        '#if 0\n_Pragma("clang fp contract(off)")\n#endif\n',
        '#define TEXT "#pragma clang fp contract(off)"\n',
    ),
)
def test_inactive_contraction_text_does_not_select_a_policy(tmp_path, prefix):
    source = prefix + SOURCE.replace("DIRECTIVE", "")
    tree = MetalParser(MetalLexer(source).tokenize()).parse()
    assert tree.functions[-1].name == "expression"
    assert tree.no_contraction is False
    path = tmp_path / "expression.metal"
    path.write_text(source, encoding="utf-8")
    assert precise_function_count(crossgl(path)) == 0


@pytest.mark.parametrize(
    "spelling",
    (
        "#pragma clang loop unroll(full)",
        "#pragma clang loop unroll(full) // contract(off)",
        '_Pragma("clang loop unroll(full)")',
        "#pragma custom contract(off)",
    ),
)
def test_non_contraction_pragmas_keep_existing_behavior(spelling):
    tree = MetalParser(
        MetalLexer(SOURCE.replace("DIRECTIVE", spelling)).tokenize()
    ).parse()
    assert tree.functions[-1].name == "expression"
    assert tree.no_contraction is False


def test_string_literals_are_not_pragma_operators():
    source = 'constant char* text = "_Pragma(\\"clang fp contract(off)\\")";\n'
    assert MetalPreprocessor().preprocess(source) == source.rstrip()
    tokens = MetalLexer(source, preprocess=False).tokenize()
    assert [value for kind, value in tokens if kind == "STRING"] == [
        '"_Pragma(\\"clang fp contract(off)\\")"'
    ]


@pytest.mark.parametrize(
    "literal",
    (
        r'"escaped \" quote and \\ backslash"',
        r'"_Pragma(\"clang fp contract(off)\")"',
        'R"text(_Pragma("clang fp contract(off)"))text"',
    ),
)
def test_contraction_scanner_keeps_string_tokens_opaque(literal):
    source = "constant char* text = " + literal + ";"
    tokens = MetalLexer(source).tokenize()
    assert [value for kind, value in tokens if kind == "STRING"] == [literal]


def test_included_active_directive_applies(tmp_path):
    (tmp_path / "policy.h").write_text(
        "#pragma clang fp contract(off)\n", encoding="utf-8"
    )
    path = tmp_path / "entry.metal"
    path.write_text(
        '#include "policy.h"\n' + SOURCE.replace("DIRECTIVE", ""), encoding="utf-8"
    )
    assert_every_function_precise(crossgl(path))


def test_translations_evaluate_the_arithmetic_as_written(tmp_path):
    path = write_source(tmp_path, "#pragma METAL fp contract(off)")
    metal = translate(str(path), backend="metal", format_output=False)
    # Each precise function scopes the directive to its own body.
    assert metal.count("#pragma clang fp contract(off)") == 2
    for backend in ("opengl", "directx"):
        code = translate(str(path), backend=backend, format_output=False)
        # The stored value is captured in a precise local, so no operation reaching it is contracted.
        assert re.search(r"results\[i\] = \(crossglPreciseValue\w* = ", code), code
    spirv = translate(str(path), backend="vulkan", format_output=False)
    decorated = set(re.findall(r"OpDecorate %(\d+) NoContraction", spirv))
    arithmetic = re.findall(r"%(\w+) = Op(?:FMul|FSub|FAdd) ", spirv)
    assert arithmetic and set(arithmetic) <= decorated, spirv


@pytest.mark.parametrize("target", ("metal", "opengl", "directx", "cgl"))
def test_translation_does_not_publish_unrepresentable_settings(tmp_path, target):
    path = write_source(tmp_path, "#pragma clang fp exceptions(strict)")
    output = tmp_path / "artifact"
    with pytest.raises(MetalContractionDirectiveError):
        translate(
            str(path), backend=target, save_shader=str(output), format_output=False
        )
    assert not output.exists()


def test_project_report_exposes_unsupported_setting_for_each_target(tmp_path):
    write_source(tmp_path, "#pragma clang fp eval_method(double)")
    (tmp_path / "crosstl.toml").write_text(
        '[project]\ntargets = ["metal", "opengl", "directx"]\n', encoding="utf-8"
    )
    report = translate_project(load_project_config(tmp_path), format_output=False)
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 0
    diagnostics = [
        item
        for item in data["diagnostics"]
        if item["code"] == "project.translate.metal-contraction-unsupported"
    ]
    assert {item["target"] for item in diagnostics} == {"metal", "opengl", "directx"}
    for item in diagnostics:
        assert item["severity"] == "error"
        assert item["missingCapabilities"] == ["metal.source-expression-contraction"]
        assert "eval_method(double)" in item["message"]
    path = tmp_path / "report.json"
    report.write_json(path)
    validation = validate_project_report(path)
    assert not validation["success"]
    assert not build_runtime_artifact_manifest(path)["success"]
    assert all(item["status"] == "failed" for item in data["artifacts"])
    assert all(not (tmp_path / item["path"]).exists() for item in data["artifacts"])
