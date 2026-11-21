"""A translation that selects one entry skips the functions it does not write in the undeclared-name checks.

The DirectX, Metal and OpenGL generators write only the functions a selected entry reaches, and importers leave a
call they cannot resolve unchanged in a body no entry evaluates. The undeclared type, identifier and function
checks therefore skip the functions no stage entry reaches when an entry is selected, while a whole-program
translation, which writes every plain function, still refuses an undeclared name in any of them.
"""

from pathlib import Path

import pytest

import crosstl
from crosstl.project import load_project_config, translate_project
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import (
    UndeclaredFunctionError,
    UndeclaredIdentifierError,
    UndeclaredTypeError,
    apply_rule_carrying,
)
from crosstl.translator.semantics.declared_identifiers import (
    recorded_undeclared_identifier,
)
from crosstl.translator.semantics.declared_types import (
    dead_function_ids,
    entry_dropped_function_ids,
)

ENTRY_SCOPED_TARGETS = ["directx", "metal", "opengl"]

_PROGRAM = """
shader P {{
    float unused(float x) {{ {body} }}
    compute {{
        buffer float* outp;
        void main() {{ outp[0] = outp[3]{call}; }}
    }}
}}
"""

UNREACHED = {
    "call": _PROGRAM.format(body="return nosuch(x);", call=""),
    "type": _PROGRAM.format(body="foo f; return x;", call=""),
    "identifier": _PROGRAM.format(body="return x + nosuchvar;", call=""),
}
REACHED_CALL = _PROGRAM.format(body="return nosuch(x);", call=" + unused(1.0)")
UNDECLARED_NAMES = {"call": "nosuch", "type": "foo", "identifier": "nosuchvar"}
ERRORS = {
    "call": UndeclaredFunctionError,
    "type": UndeclaredTypeError,
    "identifier": UndeclaredIdentifierError,
}


def _program(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _source_file(tmp_path, source):
    path = tmp_path / "kernel.cgl"
    path.write_text(source, encoding="utf-8")
    return path


def test_only_an_entry_scoped_translation_drops_the_unreached_functions():
    program = _program(UNREACHED["call"])
    assert entry_dropped_function_ids(program) == set()
    assert entry_dropped_function_ids(program, "main") == dead_function_ids(program)
    assert dead_function_ids(program) != set()


@pytest.mark.parametrize("kind", sorted(UNREACHED))
@pytest.mark.parametrize("target", ENTRY_SCOPED_TARGETS)
def test_rule_carrying_for_a_selected_entry_skips_unreached_bodies(kind, target):
    program = _program(UNREACHED[kind])
    apply_rule_carrying(program, "cgl", target, "main")
    assert recorded_undeclared_identifier(program) is None


@pytest.mark.parametrize("kind", ["call", "type"])
@pytest.mark.parametrize("target", ENTRY_SCOPED_TARGETS)
def test_whole_program_rule_carrying_still_refuses_unreached_bodies(kind, target):
    with pytest.raises(ERRORS[kind]):
        apply_rule_carrying(_program(UNREACHED[kind]), "cgl", target)


@pytest.mark.parametrize("target", ENTRY_SCOPED_TARGETS)
def test_whole_program_rule_carrying_still_records_unreached_identifiers(target):
    program = _program(UNREACHED["identifier"])
    apply_rule_carrying(program, "cgl", target)
    assert isinstance(
        recorded_undeclared_identifier(program), UndeclaredIdentifierError
    )


@pytest.mark.parametrize("kind", sorted(UNREACHED))
@pytest.mark.parametrize("target", ENTRY_SCOPED_TARGETS)
def test_entry_scoped_translation_writes_no_unreached_undeclared_name(
    tmp_path, kind, target
):
    path = _source_file(tmp_path, UNREACHED[kind])
    code = crosstl.translate(str(path), backend=target, entry_point="main")
    assert UNDECLARED_NAMES[kind] not in code
    assert "unused" not in code


@pytest.mark.parametrize("target", ENTRY_SCOPED_TARGETS)
def test_entry_scoped_translation_refuses_a_reached_undeclared_call(tmp_path, target):
    path = _source_file(tmp_path, REACHED_CALL)
    with pytest.raises(UndeclaredFunctionError):
        crosstl.translate(str(path), backend=target, entry_point="main")


@pytest.mark.parametrize("target", ENTRY_SCOPED_TARGETS)
def test_whole_program_translation_refuses_an_unreached_undeclared_call(
    tmp_path, target
):
    path = _source_file(tmp_path, UNREACHED["call"])
    with pytest.raises(UndeclaredFunctionError):
        crosstl.translate(str(path), backend=target)


@pytest.mark.parametrize("target", ENTRY_SCOPED_TARGETS)
def test_project_entry_selection_translates_a_source_with_an_unreached_undeclared_call(
    tmp_path, target
):
    source = tmp_path / "src" / "kernel.cgl"
    source.parent.mkdir()
    source.write_text(UNREACHED["call"], encoding="utf-8")
    config = tmp_path / "crosstl.toml"
    config.write_text(
        "\n".join(
            [
                "[project]",
                'source_roots = ["src"]',
                'include = ["src/kernel.cgl"]',
                f'targets = ["{target}"]',
                'output_dir = "out"',
                "",
                "[project.entry_points]",
                '"src/kernel.cgl" = "main"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    report = translate_project(
        load_project_config(tmp_path, config),
        targets=(target,),
        output_dir="out",
        format_output=False,
        validate=False,
        run_toolchains=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 1
    written = [
        path.read_text(encoding="utf-8")
        for path in Path(tmp_path, "out").rglob("*")
        if path.is_file() and path.suffix != ".json"
    ]
    assert written and all("nosuch" not in text for text in written)
