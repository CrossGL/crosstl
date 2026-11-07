"""HLSL expression statements rooted in a conditional, an access or a cast keep their evaluation.

The importer wrote ``// Unhandled statement type: TernaryOpNode`` for ``c ? bump() : reset();`` and did the same for
``data[bump()];``, so the calls in them never ran on any target. Such statements are expression statements
(C++14 section 6.2 [stmt.expr]) and are now imported as ``e;``. A literal statement such as ``1;`` does nothing and
is dropped.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

HLSL = """
RWStructuredBuffer<int> data : register(u0);
struct S { int v; };
static int counter;
static S s;
int bump() { counter += 1; return counter; }
int reset() { counter = 0; return counter; }
[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    int a = data[0];
    (a > 0) ? bump() : reset();
    data[1];
    data[bump()];
    s.v;
    (float)bump();
    (void)((a > 0) ? bump() : 0);
    1;
    data[2] = counter;
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "statements.hlsl"
    path.write_text(HLSL)
    return crosstl.translate(str(path), backend=backend)


def _main(output):
    return output.split("CSMain" if "CSMain" in output else "main", 1)[-1]


def _lines(body):
    return [line.strip() for line in body.splitlines() if line.strip()]


def test_crossgl_keeps_every_statement(tmp_path):
    lines = _lines(_main(_translate(tmp_path, "cgl")))
    start = lines.index("int a = data[0];")
    assert lines[start + 1 : start + 8] == [
        "a > 0 ? bump() : reset();",
        "data[1];",
        "data[bump()];",
        "s.v;",
        "float(bump());",
        "a > 0 ? bump() : 0;",
        "data[2] = counter;",
    ]


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx"])
def test_calls_survive_on_every_target(tmp_path, backend):
    body = _main(_translate(tmp_path, backend))
    assert "Unhandled" not in body
    assert len(re.findall(r"\breset\(", body)) == 1, body
    assert len(re.findall(r"\bbump\(", body)) == 4, body


def test_spirv_calls_every_function(tmp_path):
    # bump() four times and reset() once; neither helper calls another function.
    assert _translate(tmp_path, "vulkan").count("OpFunctionCall") == 5


def test_spirv_output_validates(tmp_path):
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        pytest.skip("SPIRV-Tools are not installed")
    (tmp_path / "out.spvasm").write_text(_translate(tmp_path, "vulkan"))
    for command in (
        [spirv_as, "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"],
        [spirv_val, "--target-env", "vulkan1.1", "out.spv"],
    ):
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_glsl_output_validates(tmp_path):
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator is not installed")
    (tmp_path / "out.comp").write_text(_translate(tmp_path, "opengl"))
    result = subprocess.run(
        [validator, "out.comp"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout
