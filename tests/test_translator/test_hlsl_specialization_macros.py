"""HLSL reads specialization constants that have no default from host macros.

Direct3D compiles HLSL once per variant, so a specialization constant the host must supply (an MSL function
constant without a value) becomes a ``static const`` initialised from ``CROSSGL_SPECIALIZE_<name>`` behind an
``#error`` guard, instead of a ``DirectXSpecializationConstantError``.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.hlsl_specialization_macros import (
    HLSL_SPECIALIZATION_MACRO_RE,
    hlsl_specialization_macro_name,
    render_hlsl_required_specialization_constant,
)

CROSSGL_SOURCE = """
shader RequiredSpecialization {
    constant bool has_w @function_constant(20);
    constant int mode @constant_id(21);
    constant uint stride @function_constant(22);
    constant float scale @function_constant(23);
    constant float bias @function_constant(24) = 0.25;

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        void main(RWStructuredBuffer<float> output @ binding(0)) {
            output[stride] = has_w ? scale * float(mode) + bias : 0.0;
        }
    }
}
"""

METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;
constant bool forward [[function_constant(1)]];
constant int width [[function_constant(2)]];

kernel void scale_kernel(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    out[i] = forward ? float(width) : -1.0;
}
"""

DEFINES = {
    "CROSSGL_SPECIALIZE_has_w": "true",
    "CROSSGL_SPECIALIZE_mode": "3",
    "CROSSGL_SPECIALIZE_stride": "2u",
    "CROSSGL_SPECIALIZE_scale": "0.5",
}


def _generate():
    return HLSLCodeGen().generate(crosstl.translator.parse(CROSSGL_SOURCE))


def test_required_constants_read_guarded_host_macros():
    generated = _generate()

    for name, hlsl_type, constant_id in (
        ("has_w", "bool", 20),
        ("mode", "int", 21),
        ("stride", "uint", 22),
        ("scale", "float", 23),
    ):
        macro = f"CROSSGL_SPECIALIZE_{name}"
        assert f"#if !defined({macro})\n#error" in generated
        assert f"(id {constant_id}) has no default: define {macro}" in generated
        assert f"static const {hlsl_type} {name} = ({hlsl_type})({macro});" in generated
        assert f"specialization constant id {constant_id} fixed to its default" not in (
            generated
        )
    # A constant with a default keeps the existing frozen-default form.
    assert "specialization constant id 24 fixed to its default" in generated
    assert "static const float bias = 0.25;" in generated
    assert "output[stride] = (has_w ? ((scale * float(mode)) + bias) : 0.0);" in (
        generated
    )


def test_metal_function_constants_without_values_translate_to_hlsl(tmp_path):
    source = tmp_path / "kernel.metal"
    source.write_text(METAL_SOURCE, encoding="utf-8")

    generated = crosstl.translate(
        str(source), backend="directx", source_backend="metal"
    )

    assert (
        "static const bool forward = (bool)(CROSSGL_SPECIALIZE_forward);" in generated
    )
    assert "static const int width = (int)(CROSSGL_SPECIALIZE_width);" in generated
    assert "(forward ? float(width) : -1.0)" in generated


def test_hlsl_import_honours_the_guard():
    generated = _generate()

    with pytest.raises(SyntaxError, match="define CROSSGL_SPECIALIZE_has_w"):
        HLSLParser(HLSLLexer(generated).tokenize()).parse()

    lexer = HLSLLexer(generated, defines=DEFINES)
    HLSLParser(lexer.tokenize()).parse()


def test_host_reflection_reports_required_macro_constants(tmp_path):
    artifact = tmp_path / "required.hlsl"
    artifact.write_text(_generate(), encoding="utf-8")

    reflected = reflect_target_host_interface(
        artifact, target="directx", stage="compute"
    )

    required = {
        record["name"]: record for record in reflected["specializationConstants"]
    }
    assert sorted(required) == ["has_w", "mode", "scale", "stride"]
    assert required["mode"]["id"] == 21
    assert required["mode"]["macro"] == "CROSSGL_SPECIALIZE_mode"
    assert required["mode"]["dtype"] == "int"
    assert all(record["required"] is True for record in required.values())
    assert all(record["status"] == "required" for record in required.values())
    assert all(
        record["source"] == "hlsl.preprocessor_macro" for record in required.values()
    )
    # Required constants are not reported again as plain constants; the defaulted one is.
    assert [constant["name"] for constant in reflected["constants"]] == ["bias"]


def test_render_round_trips_through_the_reflection_pattern():
    rendered = render_hlsl_required_specialization_constant("count", 7, "uint")

    match = HLSL_SPECIALIZATION_MACRO_RE.search(rendered)

    assert match is not None
    assert match.group("name") == "count"
    assert match.group("id") == "7"
    assert match.group("type") == "uint"
    assert match.group("macro") == hlsl_specialization_macro_name("count")
    assert hlsl_specialization_macro_name("count") == "CROSSGL_SPECIALIZE_count"


@pytest.mark.skipif(shutil.which("dxc") is None, reason="dxc is not available")
def test_dxc_requires_the_macros_and_uses_their_values(tmp_path):
    hlsl_path = tmp_path / "required.hlsl"
    hlsl_path.write_text(_generate(), encoding="utf-8")
    base = [shutil.which("dxc"), "-T", "cs_6_0", "-E", "CSMain", str(hlsl_path)]

    missing = subprocess.run(
        [*base, "-Fo", str(tmp_path / "missing.dxil")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0
    assert "define CROSSGL_SPECIALIZE_has_w" in missing.stderr + missing.stdout

    listing = tmp_path / "specialized.ll"
    defines = [
        argument
        for key, value in DEFINES.items()
        for argument in ("-D", f"{key}={value}")
    ]
    specialized = subprocess.run(
        [
            *base,
            *defines,
            "-Fo",
            str(tmp_path / "specialized.dxil"),
            "-Fc",
            str(listing),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert specialized.returncode == 0, specialized.stderr
    text = listing.read_text(encoding="utf-8")
    # 0.5 * 3 + 0.25 stored at index 2: the macros, not zero or a default, reach the program.
    assert "float 1.750000e+00" in text
    assert "$Globals" not in text
