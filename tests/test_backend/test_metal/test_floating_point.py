"""Unsupported source contraction controls fail before artifact publication."""

import json

import pytest

from crosstl import translate
from crosstl.backend.Metal.floating_point import MetalContractionDirectiveError
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project import (
    build_runtime_artifact_manifest,
    load_project_config,
    translate_project,
    validate_project_report,
)

DIRECTIVES = (
    "clang fp contract(off)",
    "clang fp contract(on)",
    "clang fp contract(fast)",
    "clang fp reassociate(off) contract(off) exceptions(ignore)",
    "clang /* source policy */ fp contract /* gap */ (off)",
    "STDC FP_CONTRACT OFF",
    "STDC FP_CONTRACT ON",
    "STDC FP_CONTRACT DEFAULT",
    "OPENCL FP_CONTRACT OFF",
)
SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void expression(const device float* values [[buffer(0)]],
                       device float* results [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    DIRECTIVE
    float a = values[2u*i];
    float b = values[2u*i+1u];
    results[i] = a*a - b*b;
}
"""


@pytest.mark.parametrize("directive", DIRECTIVES)
@pytest.mark.parametrize("form", ("line", "operator", "wide-operator"))
def test_contraction_controls_are_not_discarded(directive, form):
    spelling = "#pragma " + directive
    if form != "line":
        prefix = "L" if form == "wide-operator" else ""
        spelling = "_Pragma(" + prefix + json.dumps(directive) + ")"
    source = SOURCE.replace("DIRECTIVE", spelling)
    with pytest.raises(MetalContractionDirectiveError) as error:
        MetalPreprocessor().preprocess(source)
    assert error.value.project_diagnostic_code == (
        "project.translate.metal-contraction-unsupported"
    )
    assert error.value.missing_capabilities == ("metal.source-expression-contraction",)
    assert "original source backend" in str(error.value)
    assert "-fno-fast-math are not equivalent" in str(error.value)


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
def test_contraction_check_precedes_template_rewriting(source):
    with pytest.raises(MetalContractionDirectiveError):
        MetalPreprocessor().preprocess(source)


@pytest.mark.parametrize(
    "spelling",
    ("#pragma clang fp contract(off)", '_Pragma("clang fp contract(off)")'),
)
def test_parser_without_preprocessing_also_rejects_contraction(spelling):
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
def test_inactive_contraction_text_does_not_select_a_policy(prefix):
    source = prefix + SOURCE.replace("DIRECTIVE", "")
    tree = MetalParser(MetalLexer(source).tokenize()).parse()
    assert tree.functions[-1].name == "expression"


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


def test_included_active_directive_is_not_discarded(tmp_path):
    (tmp_path / "policy.h").write_text(
        "#pragma clang fp contract(off)\n", encoding="utf-8"
    )
    path = tmp_path / "entry.metal"
    source = '#include "policy.h"\n' + SOURCE.replace("DIRECTIVE", "")
    with pytest.raises(MetalContractionDirectiveError):
        MetalPreprocessor(include_paths=[str(tmp_path)]).preprocess(
            source, file_path=str(path)
        )


@pytest.mark.parametrize("target", ("metal", "opengl", "directx", "cgl"))
def test_translation_does_not_publish_unpreserved_contraction(tmp_path, target):
    source = tmp_path / "expression.metal"
    source.write_text(
        SOURCE.replace("DIRECTIVE", "#pragma clang fp contract(off)"),
        encoding="utf-8",
    )
    output = tmp_path / "artifact"
    with pytest.raises(MetalContractionDirectiveError):
        translate(
            str(source), backend=target, save_shader=str(output), format_output=False
        )
    assert not output.exists()


def test_project_report_exposes_contraction_failure_for_each_target(tmp_path):
    (tmp_path / "expression.metal").write_text(
        SOURCE.replace("DIRECTIVE", "#pragma clang fp contract(off)"),
        encoding="utf-8",
    )
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
        assert "original source backend" in item["message"]
    path = tmp_path / "report.json"
    report.write_json(path)
    validation = validate_project_report(path)
    assert not validation["success"]
    assert any(
        item["code"] == "project.translate.metal-contraction-unsupported"
        for item in validation["diagnostics"]
    )
    assert not build_runtime_artifact_manifest(path)["success"]
    assert all(item["status"] == "failed" for item in data["artifacts"])
    assert all(not (tmp_path / item["path"]).exists() for item in data["artifacts"])
