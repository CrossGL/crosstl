"""Metal refuses a derivative a kernel reaches through any overload a call can select (issue #3168).

The check followed only the last overload of a called name, so ``slope(vec2)`` with ``dFdx`` was missed when
``slope(float)`` came after it, and the Metal compiler then rejected the kernel. A call now reaches the overloads that
accept its argument count, narrowed by the argument types when they decide.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_codegen import UnsupportedMetalFeatureError

DERIVATIVE = "vec2 slope(vec2 x) { return dFdx(x); }"
PLAIN = "float slope(float x) { return x * 2.0; }"


def _shader(first, second, call):
    return f"""
    shader S {{
        buffer float values[];
        {first}
        {second}
        compute {{
            void main() {{
                uint i = gl_GlobalInvocationID.x;
                values[i] = {call};
            }}
        }}
    }}
    """


def _metal(tmp_path, source):
    path = tmp_path / "slope.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


@pytest.mark.parametrize("order", [(DERIVATIVE, PLAIN), (PLAIN, DERIVATIVE)])
def test_derivative_overload_is_refused_in_either_order(tmp_path, order):
    with pytest.raises(UnsupportedMetalFeatureError, match="dFdx through 'slope'"):
        _metal(tmp_path, _shader(*order, "slope(vec2(values[i], 1.0)).x"))


@pytest.mark.parametrize("order", [(DERIVATIVE, PLAIN), (PLAIN, DERIVATIVE)])
def test_a_call_that_selects_the_plain_overload_translates(tmp_path, order):
    code = _metal(tmp_path, _shader(*order, "slope(values[i])"))
    if shutil.which("xcrun"):
        source = tmp_path / "slope.metal"
        source.write_text(code)
        result = subprocess.run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(source),
                "-o",
                str(tmp_path / "s.air"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
