"""Math builtins in uninstantiated Metal function templates resolve per instantiation.

A library header can define ``template <typename T> T erf_approx(T x) { ... exp(-x * x) ... }`` without any kernel
of the including file calling it. Metal resolves ``exp`` only for the instantiations that exist (C++ [temp.res]),
but the importer looked for an ``exp`` overload taking ``T`` itself, found none and refused the whole program. See
:mod:`crosstl.backend.Metal.dependent_builtins`.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.dependent_builtins import (
    NOT_DEPENDENT,
    dependent_builtin_result_type,
    unbound_template_type_names,
)

HELPERS = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
inline T erf_like(T x) {
    T t = 1.0f / (1.0f + 0.3275911f * fabs(x));
    return sign(x) * (1.0f - t * exp(-x * x));
}

template <typename T>
inline T unit_or_zero(T v) {
    return all(v == T(0)) ? v : normalize(clamp(v, T(-1), T(1)));
}

kernel void scale(device float *data [[buffer(0)]],
                  uint tid [[thread_position_in_grid]]) {
    data[tid] = data[tid] * 2.0f;
}
"""

INSTANTIATED = HELPERS.replace(
    "data[tid] = data[tid] * 2.0f;", "data[tid] = erf_like(data[tid]);"
)


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernels.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(
        str(path), backend=backend, source_options={"threadgroup_size": (64, 1, 1)}
    )


class _Function:
    template_parameters = [("typename", "T"), ("int", "N")]


def test_dependent_results_follow_the_builtin_kind():
    names = unbound_template_type_names(_Function, set())
    assert names == {"T"}
    assert unbound_template_type_names(_Function, {"T"}) == frozenset()
    assert dependent_builtin_result_type("same", ["T"], names) == "T"
    assert dependent_builtin_result_type("same", ["float", "vec<T, 4>"], names) == (
        "vec<T, 4>"
    )
    assert dependent_builtin_result_type("bool_reduction", ["T"], names) == "bool"
    assert dependent_builtin_result_type("element", ["T", "T"], names) is None
    assert dependent_builtin_result_type("same", ["float"], names) is NOT_DEPENDENT
    # A name that merely contains the parameter's letters is not the parameter.
    assert dependent_builtin_result_type("same", ["Tf"], names) is NOT_DEPENDENT


def test_uninstantiated_helpers_import_as_generic_functions(tmp_path):
    crossgl = _translate(tmp_path, HELPERS, "cgl")
    assert "generic<T> T erf_like(T x)" in crossgl
    assert "generic<T> T unit_or_zero(T v)" in crossgl
    assert "exp((-x) * x)" in crossgl


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl", "vulkan"])
def test_kernels_beside_uninstantiated_helpers_translate(tmp_path, backend):
    assert _translate(tmp_path, HELPERS, backend).strip()


def test_instantiated_helpers_still_resolve_their_builtins(tmp_path):
    metal = _translate(tmp_path, INSTANTIATED, "metal")
    assert "exp(" in metal
    xcrun = shutil.which("xcrun")
    if (
        xcrun is None
        or subprocess.run(
            [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True
        ).returncode
    ):
        return
    shader = tmp_path / "out.metal"
    shader.write_text(metal, encoding="utf-8")
    completed = subprocess.run(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader),
            "-o",
            str(tmp_path / "out.air"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]


def test_spirv_lowers_the_instantiated_helper_with_fabs(tmp_path):
    spirv = _translate(tmp_path, INSTANTIATED, "vulkan")
    assert "FAbs" in spirv
    assert "Exp" in spirv
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        return
    assembly, module = tmp_path / "out.spvasm", tmp_path / "out.spv"
    assembly.write_text(spirv, encoding="utf-8")
    for command in (
        [spirv_as, "--target-env", "vulkan1.1", str(assembly), "-o", str(module)],
        [spirv_val, "--target-env", "vulkan1.1", str(module)],
    ):
        completed = subprocess.run(command, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
