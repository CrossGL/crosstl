"""A CrossGL function body that does not parse refuses the translation.

Before, every ``.cgl`` input (and every Mojo import) was parsed without ``strict_function_bodies``, which skips an
unparseable body and keeps the function with an empty one, so ``float helper(float x) { float y = x * ; return y; }``
was emitted as ``float helper(float x) {}``: valid-looking output that returns an undefined value.
"""

import textwrap

import pytest

import crosstl
from crosstl.project import translate_project
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import CrossGLFunctionBodyParseError, Parser

PROGRAM = textwrap.dedent("""
    shader BrokenHelper {
        struct FragmentOutput {
            vec4 color;
        }
        float helper(float x) {
            float y = x * ;
            return y;
        }
        fragment {
            FragmentOutput main() {
                FragmentOutput output;
                output.color = vec4(helper(1.0), 0.0, 0.0, 1.0);
                return output;
            }
        }
    }
    """)

DIAGNOSTIC_CODE = "project.translate.crossgl-function-body-parse-failed"


def tokens(text):
    return Lexer(text).tokens


def test_the_strict_parser_refuses_an_unparseable_body():
    with pytest.raises(CrossGLFunctionBodyParseError) as raised:
        Parser(tokens(PROGRAM), strict_function_bodies=True).parse()
    assert raised.value.function_name == "helper"
    assert raised.value.project_diagnostic_code == DIAGNOSTIC_CODE
    assert "could not parse the body of function 'helper'" in str(raised.value)


@pytest.mark.parametrize(
    "backend", ["opengl", "webgl", "directx", "metal", "vulkan", "wgsl", "cgl"]
)
def test_translating_a_program_with_an_unparseable_body_is_refused(tmp_path, backend):
    source = tmp_path / "broken_helper.cgl"
    source.write_text(PROGRAM)
    with pytest.raises(CrossGLFunctionBodyParseError):
        crosstl.translate(str(source), backend=backend)


def test_a_caller_cannot_opt_a_cgl_translation_out_of_strict_bodies(tmp_path):
    source = tmp_path / "broken_helper.cgl"
    source.write_text(PROGRAM)
    with pytest.raises(CrossGLFunctionBodyParseError):
        crosstl.translate(
            str(source),
            backend="metal",
            source_options={"strict_function_bodies": False},
        )


def test_a_project_unit_with_an_unparseable_body_fails_with_a_diagnostic(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "broken_helper.cgl").write_text(PROGRAM)

    payload = translate_project(repo, targets=["metal"], output_dir="out").to_json()

    assert [artifact["status"] for artifact in payload["artifacts"]] == ["failed"]
    assert payload["summary"]["diagnosticsByCode"] == {DIAGNOSTIC_CODE: 1}
    assert not (repo / payload["artifacts"][0]["path"]).exists()


def test_a_parseable_program_still_translates(tmp_path):
    source = tmp_path / "helper.cgl"
    source.write_text(PROGRAM.replace("x * ;", "x * 2.0;"))
    output = crosstl.translate(str(source), backend="metal")
    assert "x * 2.0" in output
