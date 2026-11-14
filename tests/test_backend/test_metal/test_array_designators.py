"""C array designators in Metal sources resolve to positional CrossGL initializer lists.

Clang accepts ``[i] = v`` (C11 6.7.9) and the GNU range ``[first ... last] = v`` in MSL. A designator sets the
current index, positional elements continue after it, unnamed elements are zero, and an array of unknown size
takes its length from the largest index initialized.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.import_refusals import MetalImportRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

PROBE = """
#include <metal_stdlib>
using namespace metal;
struct P { float x; int y; };
constant float table[5] = {[1] = 2.0f, [3 ... 4] = 7.0f};
kernel void probe(device float *o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    constexpr short NQ = 3;
    float S[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };
    half h[4] = {[2] = 1.0h, 3.0h};
    float2 v[3] = {[1] = float2(1.0f, 2.0f)};
    P p[3] = {[2] = {1.0f, 2}};
    int u[] = {[3] = 9, 4};
    int m[2][3] = {[1] = {[2] = 5}};
    o[i] = table[i % 5] + S[i % 3] + float(h[i % 4]) + v[i % 3].y + p[i % 3].x
        + float(u[i % 5]) + float(m[1][i % 3]);
}
"""

EXPECTED_CROSSGL = (
    "constant float[5] table = {float(0), 2.0f, float(0), 7.0f, 7.0f};",
    "float[NQ] S = {(-FLT_MAX) / 2, (-FLT_MAX) / 2, (-FLT_MAX) / 2};",
    "float16[4] h = {float16(0), float16(0), float16(1.0), float16(3.0)};",
    "vec2[3] v = {vec2(0), vec2(1.0f, 2.0f)};",
    "P[3] p = {{}, {}, {1.0f, 2}};",
    "int[] u = {int(0), int(0), int(0), 9, 4};",
    "int[2][3] m = {{}, {int(0), int(0), 5}};",
)


def write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def translate(tmp_path, backend, source=PROBE):
    return crosstl.translate(
        str(write(tmp_path, "probe.metal", source)),
        backend=backend,
        format_output=False,
    )


def run_tool(command, **kwargs):
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=180, **kwargs
    )
    diagnostics = result.stdout + result.stderr
    if result.returncode != 0 and "missing Metal Toolchain" in diagnostics:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, diagnostics


def test_designators_resolve_to_positional_lists(tmp_path):
    crossgl = translate(tmp_path, "cgl")
    for declaration in EXPECTED_CROSSGL:
        assert declaration in crossgl, crossgl
    assert "..." not in crossgl
    Parser(Lexer(crossgl).get_tokens()).parse()


def test_glsl_receives_every_element(tmp_path):
    glsl = translate(tmp_path, "opengl")
    assert "float S[NQ] = float[3](" in glsl
    assert "int u[5] = int[5](int(0), int(0), int(0), 9, 4);" in glsl
    assert "P p[3] = P[3](P(0.0, 0), P(0.0, 0), P(1.0, 2));" in glsl
    glslang = shutil.which("glslangValidator")
    if glslang is not None:
        run_tool([glslang, "-S", "comp", str(write(tmp_path, "probe.comp", glsl))])


def test_spirv_output_validates(tmp_path):
    spirv = translate(tmp_path, "vulkan")
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        pytest.skip("SPIRV-Tools is not installed")
    binary = tmp_path / "probe.spv"
    assembly = write(tmp_path, "probe.spvasm", spirv)
    run_tool([spirv_as, "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)])
    run_tool([spirv_val, "--target-env", "vulkan1.1", str(binary)])


def test_metal_output_compiles(tmp_path):
    metal = translate(tmp_path, "metal")
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")
    shader = write(tmp_path, "out.metal", metal)
    run_tool([xcrun, "-sdk", "macosx", "metal", "-c", str(shader)], cwd=tmp_path)


def test_hlsl_output_compiles(tmp_path):
    hlsl = translate(tmp_path, "directx")
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")
    shader = write(tmp_path, "probe.hlsl", hlsl)
    run_tool(
        [dxc, "-T", "cs_6_2", "-enable-16bit-types", "-E", "CSMain", str(shader)],
        cwd=tmp_path,
    )


def kernel_with(declaration):
    return f"""
    #include <metal_stdlib>
    using namespace metal;
    struct P {{ float x; int y; }};
    float next(device float *o) {{ return o[0]++; }}
    kernel void probe(device float *o [[buffer(0)]], uint i [[thread_position_in_grid]]) {{
        {declaration}
        o[i] = a[0];
    }}
    """


@pytest.mark.parametrize(
    ("declaration", "reason"),
    [
        ("float a[3] = {[1] = 1.0f, [1] = 2.0f};", "two initializers of index 1"),
        ("float a[3] = {[1] = 1.0f, 2.0f, 3.0f};", "index 3 outside the array"),
        ("float a[3] = {[0 ... 2] = next(o)};", "may have side effects"),
        ("float a[3] = {[2 ... 1] = 1.0f};", "empty designator range"),
        ("P a[2] = {[1].x = 1.0f};", "designator chain"),
        ("float a[3] = {[i] = 1.0f};", "not an integral constant"),
    ],
)
def test_unresolvable_designators_are_refused(tmp_path, declaration, reason):
    with pytest.raises(MetalImportRefused, match=reason) as raised:
        translate(tmp_path, "cgl", kernel_with(declaration))
    assert raised.value.code == "project.translate.metal-import-unsupported"


def test_template_extent_parameter_is_refused_in_generic_output(tmp_path):
    source = """
    #include <metal_stdlib>
    using namespace metal;
    template <short N>
    float total() {
        float s[N] = {[0 ... N - 1] = 1.0f};
        return s[0];
    }
    """
    with pytest.raises(
        MetalImportRefused, match="extent that is not an integral constant"
    ):
        translate(tmp_path, "cgl", source)


def test_designated_range_in_a_specialized_template(tmp_path):
    source = """
    #include <metal_stdlib>
    using namespace metal;
    template <short N>
    float total(float x) {
        float s[N] = {[0 ... N - 1] = 1.0f};
        float r = x;
        for (short k = 0; k < N; ++k) { r += s[k]; }
        return r;
    }
    kernel void probe(device float *o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
        o[i] = total<4>(o[i]);
    }
    """
    glsl = translate(tmp_path, "opengl", source)
    assert "float[4](1.0, 1.0, 1.0, 1.0)" in glsl
