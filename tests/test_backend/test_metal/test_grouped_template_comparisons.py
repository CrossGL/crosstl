"""Comparisons against parenthesized operands, and names of value-argument specializations.

The Metal parser decides whether ``name <`` opens a template argument list by scanning ahead
for the closing ``>``. That scan counted a ``>`` nested in parentheses, so ``i < MAX(a, b)``,
which expands to ``i < ((a) > (b) ? (a) : (b))``, looked like ``i<...>(b)``: the parser took a
template argument list, consumed the rest of the function and failed with ``Expected RPAREN``.
A ``>`` inside parentheses does not end a template argument list (C++ [temp.names]), and a
closing parenthesis opened before the ``<`` means the ``<`` is a comparison.

Explicit specializations were named by dropping punctuation from their arguments, so
``pick<(N > 2)>`` and ``pick<(N < 2)>`` were both emitted as ``pick_N_2``, and ``twice<int>``
reused the name of an ordinary function ``twice_int``. A taken name now gets a numbered suffix.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

COMPARISONS = textwrap.dedent("""\
    #include <metal_stdlib>
    using namespace metal;
    #define MIN(x, y) ((x) < (y) ? (x) : (y))
    #define MAX(x, y) ((x) > (y) ? (x) : (y))

    struct Args { int len; int top_k; };

    kernel void k(constant Args & args [[buffer(0)]],
                  device int * dst [[buffer(1)]],
                  uint tid [[thread_position_in_grid]])
    {
        const int a = int(tid);
        const int b = args.top_k;
        const int len0 = MIN(args.len, MAX(0, b - a));
        int r = 0;
        if (a < MAX(a, b)) { r += 1; }
        const int s = (a) < ((b) > (len0) ? (b) : (len0));
        int low = a > len0 ? a - len0 : 0;
        dst[tid] = r + s + low;
    }
    """)

SPECIALIZATIONS = textwrap.dedent("""\
    #include <metal_stdlib>
    using namespace metal;

    constant int N = 3;

    template <bool B>
    int pick(int x) { return B ? x : 0; }

    template <typename T>
    T twice(T x) { return x + x; }

    int twice_int(int x) { return x * 3; }

    kernel void k(device int * dst [[buffer(0)]],
                  uint tid [[thread_position_in_grid]])
    {
        int v = int(tid);
        dst[0] = pick<(N > 2)>(v);
        dst[1] = pick<(N < 2)>(v);
        dst[2] = twice<int>(v);
        dst[3] = twice_int(v);
    }
    """)


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_comparison_with_parenthesized_greater_than_is_not_a_template(tmp_path):
    code = _translate(tmp_path, COMPARISONS, "cgl")
    assert "if (a < (a > b ? a : b))" in code
    assert "const int s = a < (b > len0 ? b : len0);" in code
    assert "int low = a > len0 ? a - len0 : 0;" in code


def test_parenthesized_template_argument_with_greater_than_is_a_template(tmp_path):
    code = _translate(tmp_path, SPECIALIZATIONS, "metal")
    assert "inline int pick_N_2(int x) {\n    return N > 2 ? x : 0;" in code
    assert "inline int pick_N_2_2(int x) {\n    return N < 2 ? x : 0;" in code
    assert "dst[0] = pick_N_2(v);" in code
    assert "dst[1] = pick_N_2_2(v);" in code


def test_specialization_does_not_take_an_ordinary_function_name(tmp_path):
    code = _translate(tmp_path, SPECIALIZATIONS, "metal")
    assert "int twice_int(int x) {\n    return x * 3;" in code
    assert "inline int twice_int_2(int x) {\n    return x + x;" in code
    assert "dst[2] = twice_int_2(v);" in code
    assert "dst[3] = twice_int(v);" in code


_VALIDATORS = {
    "metal": (
        "kernel_out.metal",
        [["xcrun", "metal", "-c", "kernel_out.metal", "-o", "kernel_out.air"]],
    ),
    "opengl": (
        "kernel_out.comp",
        [["glslangValidator", "-S", "comp", "kernel_out.comp"]],
    ),
    "vulkan": (
        "kernel_out.spvasm",
        [
            [
                "spirv-as",
                "--target-env",
                "vulkan1.1",
                "kernel_out.spvasm",
                "-o",
                "k.spv",
            ],
            ["spirv-val", "--target-env", "vulkan1.1", "k.spv"],
        ],
    ),
    "directx": (
        "kernel_out.hlsl",
        [["dxc", "-T", "cs_6_2", "-E", "CSMain", "kernel_out.hlsl"]],
    ),
}


def _metal_compiler_available():
    if shutil.which("xcrun") is None:
        return False
    probe = subprocess.run(
        ["xcrun", "-f", "metal"], capture_output=True, text=True, timeout=60
    )
    return probe.returncode == 0


@pytest.mark.parametrize("name", ["comparisons", "specializations"])
@pytest.mark.parametrize("backend", sorted(_VALIDATORS))
def test_outputs_pass_native_checkers(tmp_path, backend, name):
    source = COMPARISONS if name == "comparisons" else SPECIALIZATIONS
    file_name, commands = _VALIDATORS[backend]
    if backend == "metal" and not _metal_compiler_available():
        pytest.skip("xcrun metal not installed")
    missing = [command[0] for command in commands if shutil.which(command[0]) is None]
    if missing:
        pytest.skip(f"{', '.join(missing)} not installed")
    options = {}
    if backend != "metal":
        options["source_options"] = {"threadgroup_size": (64, 1, 1)}
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    code = crosstl.translate(str(path), backend=backend, format_output=False, **options)
    (tmp_path / file_name).write_text(code, encoding="utf-8")
    for command in commands:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300, cwd=tmp_path
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
