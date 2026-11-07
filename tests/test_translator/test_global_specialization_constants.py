"""Module-scope specialization constants keep their ids in Slang and WGSL.

``constant T name @function_constant(N);`` (the Metal importer's spelling of an MSL function constant) is a
module-scope variable, not an ``ast.constants`` entry. Slang printed it as ``static const T name;`` (rejected by
slangc) and WGSL as a ``var<private>`` read as zero; a defaulted one lost its id, and constants computed from one
froze. They are now planned with the program's constants.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.specialization_constants import (
    SpecializationConstantError,
)
from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen

CROSSGL_SOURCE = """
shader GlobalSpecialization {
    constant bool has_w @function_constant(20);
    constant float bias @function_constant(24) = 0.25;
    constant int width @constant_id(25);
    constant int span = width * 2 + 1;
    constant float scaled = float(span) * bias;
    constant int plain = 7;

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        void main(RWStructuredBuffer<float> output @ binding(0)) {
            output[0] = has_w ? scaled + float(span + plain) : 0.0;
        }
    }
}
"""

METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;
constant bool forward [[function_constant(1)]];
constant int tile [[function_constant(2)]];
constant int span = tile * 3 - 1;

kernel void scale_kernel(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    out[i] = forward ? float(span) : -1.0;
}
"""


def _translate(backend, source=CROSSGL_SOURCE):
    codegen = {"slang": SlangCodeGen, "wgsl": WGSLCodeGen}[backend]()
    return codegen.generate(crosstl.translator.parse(source))


def test_slang_declares_module_scope_specialization_constants():
    generated = _translate("slang")

    assert "[vk::constant_id(20)] const bool has_w;" in generated
    assert "[vk::constant_id(24)] const float bias = 0.25;" in generated
    assert "[vk::constant_id(25)] const int width;" in generated
    # Integer constants that follow one fold to specialization-constant operations; others evaluate at run time.
    assert "static const int span = width * 2 + 1;" in generated
    assert "static float scaled = float(span) * bias;" in generated
    assert "static const int plain = 7;" in generated
    assert "static const bool has_w" not in generated
    assert "static const float bias" not in generated


def test_wgsl_declares_module_scope_specialization_constants_as_overrides():
    generated = _translate("wgsl")

    assert "@id(20)\noverride has_w: bool;" in generated
    assert "@id(24)\noverride bias: f32 = 0.25;" in generated
    assert "@id(25)\noverride width: i32;" in generated
    assert "override span: i32 = ((width * 2) + 1);" in generated
    assert "override scaled: f32 = (f32(span) * bias);" in generated
    assert "const plain: i32 = 7;" in generated
    assert "var<private> has_w" not in generated
    assert "const bias" not in generated


def test_metal_function_constants_reach_slang_and_wgsl(tmp_path):
    source = tmp_path / "kernel.metal"
    source.write_text(METAL_SOURCE, encoding="utf-8")

    slang = crosstl.translate(str(source), backend="slang", source_backend="metal")
    wgsl = crosstl.translate(str(source), backend="wgsl", source_backend="metal")

    assert "[vk::constant_id(1)] const bool forward;" in slang
    assert "[vk::constant_id(2)] const int tile;" in slang
    assert "static const int span = tile * 3 - 1;" in slang
    assert "@id(1)\noverride forward: bool;" in wgsl
    assert "@id(2)\noverride tile: i32;" in wgsl
    assert "override span: i32 = ((tile * 3) - 1);" in wgsl


@pytest.mark.parametrize("backend", ["slang", "wgsl"])
def test_module_scope_and_constant_ids_must_differ(backend):
    source = """
    shader DuplicateSpecialization {
        constant bool flag @function_constant(4);
        const int mode @constant_id(4) = 1;

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main(RWStructuredBuffer<float> output @ binding(0)) {
                output[0] = flag ? float(mode) : 0.0;
            }
        }
    }
    """

    with pytest.raises(SpecializationConstantError) as excinfo:
        _translate(backend, source)

    assert excinfo.value.kind == "duplicate-id"
    assert excinfo.value.specialization_id == 4


def test_wgsl_refuses_a_local_array_sized_by_a_dependent_constant():
    source = """
    shader DependentArraySize {
        constant int count @constant_id(3) = 4;
        constant int doubled = count * 2;

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main(RWStructuredBuffer<float> output @ binding(0)) {
                float tmp[doubled];
                tmp[0] = 1.0;
                output[0] = tmp[0];
            }
        }
    }
    """

    # WGSL takes an override-expression element count only for var<workgroup>; the
    # array used to be sized by the frozen default.
    with pytest.raises(SpecializationConstantError) as excinfo:
        _translate("wgsl", source)

    assert excinfo.value.kind == "array-size"
    assert excinfo.value.constant_name == "doubled"
    assert "float tmp[doubled];" in _translate("slang", source)


@pytest.mark.skipif(shutil.which("naga") is None, reason="naga is not available")
def test_wgsl_output_validates_with_naga(tmp_path):
    path = tmp_path / "global_specialization.wgsl"
    path.write_text(_translate("wgsl"), encoding="utf-8")

    result = subprocess.run(
        [shutil.which("naga"), str(path)], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("slangc") is None, reason="slangc is not available")
def test_slang_output_compiles_to_spirv_specialization_constants(tmp_path):
    path = tmp_path / "global_specialization.slang"
    path.write_text(_translate("slang"), encoding="utf-8")
    spirv = tmp_path / "global_specialization.spv"

    result = subprocess.run(
        [shutil.which("slangc"), str(path), "-target", "spirv", "-o", str(spirv)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    disassembler = shutil.which("spirv-dis")
    if disassembler is None:
        return
    listing = subprocess.run(
        [disassembler, str(spirv)], check=True, capture_output=True, text=True
    ).stdout
    for spec_id in (20, 24, 25):
        assert f"SpecId {spec_id}" in listing
