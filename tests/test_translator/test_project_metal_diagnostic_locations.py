"""Diagnostics found in preprocessed Metal text report the line the user wrote.

The preprocessor drops directive lines and inlines included files, and later
passes rewrite declarations, so offsets in that text name no source position
by themselves. ``HLSLPreprocessor.line_origins`` records where each output line
comes from, and ``crosstl.project.preprocessed_locations`` maps offsets back.
"""

import textwrap
from pathlib import Path

import pytest

from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project import load_project_config
from crosstl.project import pipeline as project_pipeline
from crosstl.project import preprocessed_locations, translate_project
from crosstl.project.source_closure_memo import PREPROCESSOR_RUN_MEMO

WORK_LIMIT_CONFIG = """
[project]
targets = ["directx"]
output_dir = "out"

[project.source_options.metal]
max_template_materialization_work = 2
"""

LAUNCH_TEMPLATE = """
template <typename T>
[[kernel]] void launch(device T* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = T(gid);
}
"""

INSTANTIATIONS = """
template [[host_name("launch_a")]] [[kernel]] decltype(launch<float>) launch<float>;
template [[host_name("launch_b")]] [[kernel]] decltype(launch<int>) launch<int>;
template [[host_name("launch_c")]] [[kernel]] decltype(launch<uint>) launch<uint>;
"""


@pytest.fixture(autouse=True)
def _fresh_location_state():
    preprocessed_locations.clear()
    PREPROCESSOR_RUN_MEMO.clear()
    yield
    preprocessed_locations.clear()
    PREPROCESSOR_RUN_MEMO.clear()


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")


def _work_limit_location(repo):
    payload = translate_project(load_project_config(repo)).to_json()
    diagnostics = [
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["code"] == "project.translate.metal-template-specialization"
    ]
    assert len(diagnostics) == 1, payload["diagnostics"]
    return diagnostics[0]["location"], diagnostics[0]["message"]


def _line_of(path, needle):
    lines = path.read_text(encoding="utf-8").splitlines()
    return next(index for index, line in enumerate(lines, 1) if needle in line)


def test_hlsl_preprocessor_records_the_origin_of_every_output_line(tmp_path):
    header = tmp_path / "inc" / "values.h"
    _write(header, "#define TWO 2\nint two = TWO;\n")
    main = tmp_path / "main.hlsl"
    _write(
        main,
        """
        #define ONE 1
        #include "inc/values.h"
        /* a comment
           over two lines */ int one = ONE;
        #line 100 "renamed.hlsl"
        int three = 3;
        """,
    )
    preprocessor = HLSLPreprocessor()
    text = preprocessor.preprocess(main.read_text(), file_path=str(main))

    lines = text.split("\n")
    assert len(lines) == len(preprocessor.line_origins)
    by_text = dict(zip(lines, preprocessor.line_origins))
    include_site = (str(main), 2, None)
    # The header path joins the include directive's own spelling ("inc/values.h"), so on Windows it can mix
    # separators; compare it as a path.
    header_file, header_line, header_site = by_text["int two = 2;"]
    assert (Path(header_file), header_line, header_site) == (header, 2, include_site)
    assert by_text["   over two lines */ int one = 1;"] == (str(main), 4, None)
    # #line changes the presumed line (__LINE__), not the physical origin.
    assert by_text["int three = 3;"] == (str(main), 6, None)


def test_preprocessor_memo_ignores_and_restores_line_origins(tmp_path):
    main = tmp_path / "k.metal"
    _write(main, "#define A 1\nint a = A;\n")
    first = MetalPreprocessor()
    first.line_origins = [("stale", 9, None)]
    text = project_pipeline._metal_preprocess_without_template_materialization(
        first, main.read_text(), file_path=str(main)
    )
    hits = PREPROCESSOR_RUN_MEMO.hits

    second = MetalPreprocessor()
    again = project_pipeline._metal_preprocess_without_template_materialization(
        second, main.read_text(), file_path=str(main)
    )

    assert again == text == "int a = 1;"
    assert PREPROCESSOR_RUN_MEMO.hits == hits + 1
    assert second.line_origins == first.line_origins == [(str(main), 2, None)]


def test_work_limit_diagnostic_names_the_declaration_after_dropped_defines(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "crosstl.toml", WORK_LIMIT_CONFIG)
    unit = repo / "k.metal"
    _write(
        unit,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "#define UNUSED_A 1\n#define UNUSED_B 2\n" + LAUNCH_TEMPLATE + INSTANTIATIONS,
    )

    location, message = _work_limit_location(repo)

    line = _line_of(unit, '"launch_c"')
    assert line == 13
    declaration = unit.read_text().splitlines()[line - 1]
    assert location["file"] == "k.metal"
    assert (location["line"], location["column"]) == (line, 1)
    assert (location["endLine"], location["endColumn"]) == (line, len(declaration) + 1)
    assert location["length"] == len(declaration)
    offset = location["offset"]
    assert unit.read_text()[offset : offset + location["length"]] == declaration
    assert f"Source context: k.metal:{line}:1." in message


def test_work_limit_diagnostic_skips_lines_from_an_included_header(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "crosstl.toml", WORK_LIMIT_CONFIG)
    _write(
        repo / "include" / "helpers.h",
        "\n".join(
            f"inline int helper_{index}() {{ return {index}; }}" for index in range(8)
        )
        + "\n",
    )
    unit = repo / "k.metal"
    _write(
        unit,
        '#include <metal_stdlib>\n#include "include/helpers.h"\nusing namespace metal;\n'
        + LAUNCH_TEMPLATE
        + INSTANTIATIONS,
    )

    location, _message = _work_limit_location(repo)

    assert location["file"] == "k.metal"
    assert (location["line"], location["column"]) == (_line_of(unit, '"launch_c"'), 1)


def test_work_limit_diagnostic_names_a_declaration_in_a_project_header(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "crosstl.toml", WORK_LIMIT_CONFIG)
    header = repo / "kernels" / "instantiations.h"
    _write(header, "#define UNUSED 1\n\n" + INSTANTIATIONS)
    _write(
        repo / "k.metal",
        "#include <metal_stdlib>\nusing namespace metal;\n"
        + LAUNCH_TEMPLATE
        + '\n#include "kernels/instantiations.h"\n',
    )

    location, message = _work_limit_location(repo)

    line = _line_of(header, '"launch_c"')
    assert location["file"] == "kernels/instantiations.h"
    assert (location["line"], location["column"]) == (line, 1)
    assert f"Source context: kernels/instantiations.h:{line}:1." in message


def test_header_outside_the_project_maps_to_its_include_directive(tmp_path):
    outside = tmp_path / "outside"
    _write(
        outside / "defs.h", "\n".join(f"int value_{i} = {i};" for i in range(5)) + "\n"
    )
    unit = tmp_path / "repo" / "src" / "k.metal"
    _write(unit, "#define ONE 1\n#include <defs.h>\nint one = ONE;\n")
    preprocessor = MetalPreprocessor(include_paths=[str(outside)])
    text = project_pipeline._metal_preprocess_without_template_materialization(
        preprocessor, unit.read_text(), file_path=str(unit)
    )

    in_header = text.index("int value_3")
    span = preprocessed_locations.source_span(
        str(unit), "src/k.metal", text, in_header, 5
    )
    assert (span.file, span.line, span.column) == ("src/k.metal", 2, 1)

    own_line = text.index("one = 1")
    span = preprocessed_locations.source_span(
        str(unit), "src/k.metal", text, own_line, 3
    )
    # The line was macro-expanded, so the span starts at the first non-blank column of its source line.
    assert (span.file, span.line, span.column) == ("src/k.metal", 3, 1)


def test_rewritten_text_maps_through_the_preprocessor_output(tmp_path):
    unit = tmp_path / "k.metal"
    _write(
        unit,
        """
        #define SCALE 2
        // first
        int keep_a = 1;
        int rewrite_me = SCALE;
        int keep_b = 3;
        """,
    )
    preprocessor = MetalPreprocessor()
    text = project_pipeline._metal_preprocess_without_template_materialization(
        preprocessor, unit.read_text(), file_path=str(unit)
    )
    rewritten = text.replace(
        "int rewrite_me = 2;", "int rewritten = 2;\nint extra = 0;"
    )

    span = preprocessed_locations.source_span(
        str(unit), "k.metal", rewritten, rewritten.index("keep_b"), 6
    )
    assert (span.line, span.column) == (5, 5)
    assert unit.read_text()[span.offset : span.end_offset] == "keep_b"

    span = preprocessed_locations.source_span(
        str(unit), "k.metal", rewritten, rewritten.index("extra"), 5
    )
    assert (span.line, span.column) == (4, 1)


def test_the_units_own_source_is_left_to_the_caller(tmp_path):
    unit = tmp_path / "k.metal"
    _write(unit, "#define A 1\nint a = A;\n")
    source = unit.read_text()
    project_pipeline._metal_preprocess_without_template_materialization(
        MetalPreprocessor(), source, file_path=str(unit)
    )

    assert (
        preprocessed_locations.source_span(str(unit), "k.metal", source, 12, 3) is None
    )
