"""Explicit function specializations reach every call of their primary template.

``template <typename T> T f(T);`` may be declared without a body when explicit specializations
``template <> float f<float>(float) { ... }`` complete it ([temp.expl.spec]/1). A call ``f(x)`` deduces ``T`` and
runs the specialization; it never names the specialization itself. The importer keeps such a primary as a CrossGL
generic prototype so the specializations stay selectable, and project materialization for targets without
templates uses a specialization's body in place of the primary's, also for helpers that an instantiated kernel
template reaches.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.external_functions import MetalExternalFunctionError
from tests.stage_entry_helpers import joined_entries

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

DECLARED_PRIMARY = """
template <typename T> T ramp(T x);
template <> inline float ramp<float>(float x) { return x > 0.0f ? x : exp(x) - 1.0f; }
template <> inline float4 ramp<float4>(float4 x) {
  float4 r = x;
  r[0] = ramp(x[0]);
  return r;
}
template <typename T>
kernel void k(device T* dst [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  T x = dst[i];
  dst[i] = ramp(x);
}
template [[host_name("k_f32")]] kernel void k<float>(device float*, uint);
template [[host_name("k_f32_4")]] kernel void k<float4>(device float4*, uint);
"""

DEFINED_PRIMARY = """
template <typename T> T twice(T x) { return x; }
template <> float twice<float>(float x) { return x * 2.0f; }
template <typename T>
kernel void k(device T* dst [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  T x = dst[i];
  dst[i] = twice(x);
}
template [[host_name("k_f32")]] kernel void k<float>(device float*, uint);
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(PREFIX + source)
    return joined_entries(path, backend, format_output=False, source_backend="metal")


def _assert_metal_compiles(tmp_path, generated):
    if shutil.which("xcrun") is None:
        return
    artifact = tmp_path / "translated.metal"
    artifact.write_text(generated)
    compiled = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(artifact)]
        + ["-o", str(tmp_path / "translated.air")],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr


def test_declared_primary_is_a_generic_prototype(tmp_path):
    cgl = _translate(tmp_path, DECLARED_PRIMARY, "cgl")
    assert "generic<T> T ramp(T x);" in cgl
    assert "@explicit_specialization(ramp, float)" in cgl
    assert "@explicit_specialization(ramp, vec4)" in cgl


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl"])
def test_calls_of_a_declared_primary_run_the_specializations(tmp_path, backend):
    code = _translate(tmp_path, DECLARED_PRIMARY, backend)
    assert "ramp(" not in code.replace("_ramp(", "")
    assert "ramp_float4(" not in code
    # The nested call inside the float4 specialization deduces float.
    assert "ramp_u3cfloat_u3e(CrossGLMetalVectorIndex_vec4_get(x, 0))" in code
    assert "ramp_u3cfloat4_u3e(x)" in code
    if backend == "metal":
        _assert_metal_compiles(tmp_path, code)


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl"])
@pytest.mark.parametrize(
    "entry, specialization",
    [("k_f32", "ramp_u3cfloat_u3e"), ("k_f32_4", "ramp_u3cfloat4_u3e")],
)
def test_an_entry_scoped_translation_keeps_the_specializations_it_calls(
    tmp_path, backend, entry, specialization
):
    # The entry calls the primary, which only resolves to a specialization when the generic is instantiated, so
    # entry scoping must keep each specialization of a primary it reaches.
    path = tmp_path / "kernel.metal"
    path.write_text(PREFIX + DECLARED_PRIMARY)
    code = crosstl.translate(
        str(path), backend=backend, format_output=False, entry_point=entry
    )
    assert f"{specialization}(" in code
    if backend == "metal":
        _assert_metal_compiles(tmp_path, code)


def _assert_spirv_validates(tmp_path, generated):
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        return
    (tmp_path / "translated.spvasm").write_text(generated)
    binary = str(tmp_path / "translated.spv")
    for command in (
        ["spirv-as", "--target-env", "vulkan1.0"]
        + [str(tmp_path / "translated.spvasm"), "-o", binary],
        ["spirv-val", "--target-env", "vulkan1.0", binary],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def test_spirv_calls_of_a_declared_primary_run_the_specializations(tmp_path):
    # The float4 specialization calls ``ramp`` on a vector element, which the
    # importer reads through a concrete helper; its declared result binds T.
    code = _translate(tmp_path, DECLARED_PRIMARY, "vulkan")
    entries = re.findall(r'OpEntryPoint GLCompute %(\w+) "(\w+)"', code)
    assert [name for _, name in entries] == ["k_f32", "k_f32_4"]
    functions = re.findall(r"%(\w+) = OpFunction %(\w+) ", code)
    calls = re.findall(r"= OpFunctionCall %(\w+) %(\w+)", code)
    called = {callee for _, callee in calls}
    called_types = sorted(result for name, result in functions if name in called)
    # One float and one float4 specialization plus the vector-element helper.
    assert len(called_types) == 3, (functions, calls)
    assert len(set(called_types)) == 2, called_types
    _assert_spirv_validates(tmp_path, code)


def test_redeclared_defined_primary_keeps_one_generic(tmp_path):
    source = "template <typename T> T twice(T x);\n" + DEFINED_PRIMARY
    cgl = _translate(tmp_path, source, "cgl")
    assert cgl.count("generic<T> T twice(T x)") == 1
    assert "@explicit_specialization(twice, float)" in cgl


def test_declared_primary_without_specializations_stays_refused(tmp_path):
    source = """
template <typename T> T ramp(T x);
kernel void k(device float* dst [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  dst[i] = ramp(dst[i]);
}
"""
    with pytest.raises(MetalExternalFunctionError, match="ramp"):
        _translate(tmp_path, source, "cgl")


@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan"])
def test_materialized_kernel_helpers_use_the_specialized_body(tmp_path, backend):
    code = _translate(tmp_path, DEFINED_PRIMARY, backend)
    if backend == "vulkan":
        # The primary only returns x; the specialization multiplies.
        assert "OpFMul" in code
        return
    assert "x * 2.0" in code
    assert "return x;" not in code


VALUE_PRIMARY = """
template <bool upper> inline bool mask(int row, int col, int k);
template <> inline bool mask<true>(int row, int col, int k) { return col - row >= k; }
template <> inline bool mask<false>(int row, int col, int k) { return col - row <= k; }
kernel void k(device int* o [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
  o[p.x] = mask<true>(p.y, p.x, 1) ? 1 : 0;
  o[p.y] = mask<false>(p.y, p.x, 1) ? 1 : 0;
}
"""


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl", "vulkan"])
def test_value_template_calls_keep_naming_their_specializations(tmp_path, backend):
    # CrossGL generics take no value parameters, so the importer binds each call
    # itself; a retained primary made it call a body-less clone ``mask_true``.
    code = _translate(tmp_path, VALUE_PRIMARY, backend)
    assert "mask_true" not in code and "mask_false" not in code
    if backend == "vulkan":
        assert code.count("= OpFunction ") == 3
        return
    assert re.search(r"mask_u3ctrue_u3e\((?:int\()?p\.y", code), code
    assert re.search(r"mask_u3cfalse_u3e\((?:int\()?p\.y", code), code
    if backend == "metal":
        _assert_metal_compiles(tmp_path, code)
