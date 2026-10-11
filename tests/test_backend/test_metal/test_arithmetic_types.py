"""Preserve source floating families through inferred Metal arithmetic."""

import shutil
import subprocess

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalArithmeticTypeResolutionError,
    MetalToCrossGLConverter,
)
from crosstl.project import translate_project


@pytest.mark.parametrize("operator", ("+", "-", "*", "/"))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("bfloat", "bfloat", "bfloat"),
        ("bfloat16_t", "bfloat16", "bfloat"),
        *[
            ("bfloat", kind, "bfloat")
            for kind in (
                "bool",
                "char",
                "uchar",
                "short",
                "ushort",
                "int",
                "uint",
                "long",
                "ulong",
            )
        ],
        ("bfloat", "float", "float"),
        ("bfloat2", "int", "bfloat2"),
        ("bfloat3", "bfloat", "bfloat3"),
        ("bfloat4", "bfloat4", "bfloat4"),
        ("float2", "bfloat", "float2"),
    ],
)
def test_bfloat_arithmetic_keeps_native_result_family(
    operator, reverse, left, right, expected
):
    if reverse:
        left, right = right, left
    result, _ = MetalToCrossGLConverter().metal_bfloat_arithmetic_plan(
        operator, left, right
    )
    assert result == expected


@pytest.mark.parametrize("operator", ("+", "-", "*", "/", "==", "<"))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize(
    "left,right",
    [
        ("bfloat", "half"),
        ("bfloat2", "half"),
        ("bfloat2", "float"),
        ("bfloat2", "float2"),
        ("bfloat2", "bfloat3"),
        ("int2", "bfloat"),
    ],
)
def test_bfloat_arithmetic_rejects_invalid_native_type_combinations(
    operator, reverse, left, right
):
    if reverse:
        left, right = right, left
    with pytest.raises(MetalArithmeticTypeResolutionError) as raised:
        MetalToCrossGLConverter().metal_bfloat_arithmetic_plan(operator, left, right)
    assert raised.value.operand_types == (left, right)
    assert raised.value.operator == operator
    assert (
        raised.value.project_diagnostic_code
        == "project.translate.metal-arithmetic-type-invalid"
    )


@pytest.mark.parametrize(
    "operand", ("bfloat*", "const device bfloat16_t*", "bfloat[4]")
)
@pytest.mark.parametrize("reverse", (False, True))
def test_bfloat_arithmetic_does_not_convert_pointer_operands(operand, reverse):
    types = ("uint", operand) if reverse else (operand, "uint")
    assert MetalToCrossGLConverter().metal_bfloat_arithmetic_plan("+", *types) is None


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("alias", (False, True))
def test_bfloat_inferred_nested_arithmetic_survives_saved_source(
    tmp_path, target, alias
):
    dtype = "Narrow" if alias else "bfloat"
    source = tmp_path / "original.metal"
    source.write_text(f"""#include <metal_stdlib>
using namespace metal;
using Narrow = bfloat;
{dtype} identity({dtype} value) {{ return value; }}
kernel void evaluate(const device ushort* words [[buffer(0)]],
                     device uint* results [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {{
    {dtype} x = as_type<bfloat>(words[i]);
    uint counter = 1u;
    auto y = counter++ / (1 + identity(x));
    y += counter++;
    results[i] = uint(as_type<ushort>(y)) + counter;
}}
""")
    intermediate = tmp_path / "saved.cgl"
    cgl = translate(str(source), backend="cgl", format_output=False)
    intermediate.write_text(cgl)
    assert "bfloat16 y = bfloat16(counter++) / (bfloat16(1) + identity(x));" in cgl
    assert "y += bfloat16(counter++);" in cgl
    assert cgl.count("counter++") == 2
    generated = translate(str(source), backend=target, format_output=False)
    assert (
        translate(str(intermediate), backend=target, format_output=False) == generated
    )
    assert "round_half" not in generated
    if target == "metal" and shutil.which("xcrun"):
        for name, code in (("original", source.read_text()), ("generated", generated)):
            path = tmp_path / f"{name}.metal"
            path.write_text(code)
            result = subprocess.run(
                [
                    "xcrun",
                    "--sdk",
                    "macosx",
                    "metal",
                    "-std=metal3.1",
                    "-Werror",
                    "-fno-fast-math",
                    "-c",
                    str(path),
                    "-o",
                    str(path.with_suffix(".air")),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, result.stdout + result.stderr


def test_bfloat_mixed_half_is_rejected_before_target_generation(tmp_path):
    source = tmp_path / "invalid.metal"
    source.write_text("""#include <metal_stdlib>
using namespace metal;
kernel void evaluate(device float* output [[buffer(0)]]) {
    bfloat x = bfloat(1.0f);
    half y = half(1.0f);
    output[0] = float(x + y);
}
""")
    with pytest.raises(MetalArithmeticTypeResolutionError, match="unordered"):
        translate(str(source), backend="opengl", format_output=False)
    payload = translate_project(
        tmp_path, targets=["opengl"], output_dir="out"
    ).to_json()
    assert payload["summary"]["translatedCount"] == 0, payload
    assert payload["summary"]["failedCount"] == 1, payload
    assert payload["summary"]["diagnosticsByCode"] == {
        "project.translate.metal-arithmetic-type-invalid": 1,
    }
    assert payload["summary"]["missingCapabilityCounts"] == {
        "metal.source-arithmetic-types": 1,
    }
