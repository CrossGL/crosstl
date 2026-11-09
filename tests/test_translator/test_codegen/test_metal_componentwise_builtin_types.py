"""Metal types componentwise builtin calls such as ``exp(x)`` by their arguments.

The Metal generator had no result type for ``exp``, ``sin``, ``sqrt`` and the other componentwise floating-point
builtins, so ``1 + exp(x)`` took the ``int`` of its literal. A floating-point builtin around it then converted the
``float4`` sum to ``float``: ``log(float(1 + exp(x)))``, which Metal rejects. Each case below fails on that code.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl


def _program(declarations, expression):
    return textwrap.dedent(f"""
        shader main {{
            compute {{
                RWStructuredBuffer<vec4> dst @buffer(0);
                RWStructuredBuffer<int> counts @buffer(1);
                @numthreads(1, 1, 1)
                void main(uvec3 id @ gl_GlobalInvocationID) {{
                    vec4 x = dst[id.x];
                    int n = counts[id.x];
                    {declarations}
                    dst[id.x] = {expression};
                }}
            }}
        }}
    """).strip()


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _compile(tmp_path, code):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")
    path = tmp_path / "shader.metal"
    path.write_text(code)
    result = subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(path), "-o", "/dev/null"],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "expression",
    [
        "log(1 + exp(x))",
        "log(exp(x) + 1)",
        "sqrt(fract(x) + 2)",
        "log2(1 + pow(x, float4(2.0)))",
        "exp(2 * sin(x) + 1)",
    ],
)
def test_vector_builtin_results_keep_their_type(tmp_path, expression):
    code = _translate(tmp_path, _program("", expression.replace("float4(", "vec4(")))

    assert f"dst[id.x] = {expression};" in code
    _compile(tmp_path, code)


def test_integer_operands_are_still_converted(tmp_path):
    code = _translate(
        tmp_path, _program("float s = sin(2 + n) + log(1 + exp(n));", "x * s")
    )

    # ``2 + n`` is an int and ``exp(n)`` a float, so only the former is converted.
    assert "float s = sin(float(2 + n)) + log(1 + exp(float(n)));" in code
    _compile(tmp_path, code)
