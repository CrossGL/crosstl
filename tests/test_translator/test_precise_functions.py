"""A CrossGL ``@precise`` function evaluates all of its floating-point arithmetic as written.

HLSL keeps that by declaring the function's value locals ``precise`` and capturing each value it stores elsewhere
and each condition it branches on in a precise local (crosstl/translator/codegen/hlsl_precise_functions.py).
SPIR-V decorates every arithmetic result of the function ``NoContraction``. Execution evidence: the conformance
cells u_fmul_rn and c_fp_contract_off on d3d12-warp, vulkan-moltenvk and vulkan-kosmickrisp.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

PRECISE_KERNEL = """@compute
@precise
@workgroup_size(64, 1, 1)
fn k(
    @group(0) @binding(0) var<storage, read_write> a: array<f32>,
    @group(0) @binding(1) var<storage, read_write> o: array<f32>
) {
    var i: i32 = gl_GlobalInvocationID.x;
    var acc: f32 = 0.0f;
    for (var j: i32 = 0; ((a[i] * f32(j)) < a[i + 1]); (j++)) {
        acc = (acc + (a[i] * a[i + 2]));
    }
    if (((a[i] * a[i]) - a[i + 3]) > 0.0f) {
        o[i] = ((a[i] * a[i]) - acc);
    } else {
        o[i] = ((acc > 0.0f) ? ((acc * a[i]) - a[i + 1]) : acc);
    }
}
"""

CONTRACTED_KERNEL = PRECISE_KERNEL.replace("@precise\n", "")


def translate(tmp_path, source, backend):
    path = tmp_path / "k.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_hlsl_precise_function_declares_its_locals_precise(tmp_path):
    hlsl = translate(tmp_path, PRECISE_KERNEL, "directx")
    assert re.search(r"\bprecise float acc\b", hlsl)
    assert re.search(r"\bprecise int j\b", hlsl)


def test_hlsl_precise_function_captures_stores_and_conditions(tmp_path):
    hlsl = translate(tmp_path, PRECISE_KERNEL, "directx")
    stores = re.findall(r"o\[i\] = \((crossglPreciseValue\w*) = ", hlsl)
    conditions = re.findall(r"\((crossglPreciseCondition\w*) = ", hlsl)
    assert len(stores) == 2, hlsl
    # The loop condition, the if condition and the selection's condition.
    assert len(conditions) == 3, hlsl
    for name in stores:
        assert re.search(rf"\bprecise float {name};", hlsl)
    for name in conditions:
        assert re.search(rf"\bprecise bool {name};", hlsl)
    # The captured condition stays parenthesized, so it is not read as a mistyped comparison.
    assert re.search(r"if \(\(crossglPreciseCondition\w* = ", hlsl)


def test_hlsl_precise_void_function_has_no_precise_result(tmp_path):
    hlsl = translate(tmp_path, PRECISE_KERNEL, "directx")
    assert "precise void" not in hlsl
    assert re.search(r"\[numthreads\(64, 1, 1\)\]\s*void\s+\w+", hlsl)


def test_hlsl_function_that_is_not_precise_is_unchanged(tmp_path):
    hlsl = translate(tmp_path, CONTRACTED_KERNEL, "directx")
    assert "crossglPrecise" not in hlsl
    assert not re.search(r"\bprecise\b", hlsl)


def test_dxc_keeps_every_operation_of_a_precise_function_exact(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")
    for source, fast_expected in ((PRECISE_KERNEL, False), (CONTRACTED_KERNEL, True)):
        hlsl_path = tmp_path / "k.hlsl"
        hlsl_path.write_text(translate(tmp_path, source, "directx"))
        listing = tmp_path / "k.ll"
        entry = re.search(r"\]\s*void\s+(\w+)\s*\(", hlsl_path.read_text()).group(1)
        subprocess.run(
            [dxc, "-T", "cs_6_0", "-E", entry, str(hlsl_path), "-Fc", str(listing)],
            check=True,
            capture_output=True,
        )
        operations = re.findall(r"= (fadd|fsub|fmul|fcmp)( fast)?", listing.read_text())
        assert operations
        assert any(fast for _, fast in operations) is fast_expected


def test_glsl_precise_function_captures_stores_to_entry_buffer_parameters(tmp_path):
    # ``o`` is a parameter of the entry, but GLSL declares it as a global buffer, so a store to it is captured.
    glsl = translate(tmp_path, PRECISE_KERNEL, "opengl")
    stores = re.findall(r"o\[i\] = \((crossglPreciseValue\w*) = ", glsl)
    assert len(stores) == 2, glsl
    for name in stores:
        assert re.search(rf"\bprecise float {name};", glsl)


def test_spirv_precise_function_decorates_its_arithmetic_no_contraction(tmp_path):
    spirv = translate(tmp_path, PRECISE_KERNEL, "vulkan")
    decorated = set(re.findall(r"OpDecorate (%\S+) NoContraction", spirv))
    arithmetic = re.findall(r"(%\S+) = OpF(?:Mul|Add|Sub)\b", spirv)
    assert arithmetic
    assert set(arithmetic) <= decorated


def test_spirv_function_that_is_not_precise_has_no_contraction_decorations(tmp_path):
    spirv = translate(tmp_path, CONTRACTED_KERNEL, "vulkan")
    assert "NoContraction" not in spirv
