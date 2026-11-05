"""Metal atomics on a target typed by an open template parameter import as CrossGL generic atomics.

``template <typename AT, typename T> void add_cas(device metal::atomic<AT>* data, ...)`` loads and compares-and-swaps
``data + offset``. The importer required the atomic's value type to be ``int``, ``uint`` or ``float`` while it
imported the template, where it is still ``AT``, and refused every such helper. A parameter the template leaves open
is accepted: generators monomorphize the generic and check each specialization's atomic target before they emit it.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalAtomicLoadLoweringError

KERNEL = """#include <metal_stdlib>
using namespace metal;
template <typename AT, typename T>
static inline void add_cas(device ::metal::atomic<AT>* data, int offset, T value) {
  auto ptr = data + offset;
  auto old = ::metal::atomic_load_explicit(ptr, ::metal::memory_order_relaxed);
  T val;
  do {
    val = old + value;
  } while (!::metal::atomic_compare_exchange_weak_explicit(
      ptr, &old, val, ::metal::memory_order_relaxed, ::metal::memory_order_relaxed));
}
kernel void k(device atomic<int>* data [[buffer(0)]],
              device atomic<float>* fdata [[buffer(1)]],
              uint tid [[thread_position_in_grid]]) {
  add_cas(data, int(tid), 2);
  add_cas(fdata, int(tid), 1.5f);
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_generic_atomic_target_imports_as_a_generic_helper(tmp_path):
    crossgl = _translate(tmp_path, KERNEL, "cgl")

    assert (
        "generic<AT, T> void add_cas(device AT* data, int offset, T value)" in crossgl
    )
    assert "AT old = atomicLoad(ptr[0]);" in crossgl
    assert "atomicCompareExchangeWeak(ptr[0], old, val)" in crossgl


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("metal", ("void add_cas_int_int(", "void add_cas_float_float(")),
        ("directx", ("void add_cas_int_int(", "void add_cas_float_float(")),
        (
            "opengl",
            (
                "void add_cas_int_int_glsl_data_data_int(",
                "void add_cas_float_float_glsl_data_fdata_float(",
            ),
        ),
    ],
)
def test_generic_atomic_helper_specializes_per_atomic_type(tmp_path, backend, expected):
    code = _translate(tmp_path, KERNEL, backend)

    for signature in expected:
        assert signature in code, code
    assert "AT" not in code.replace("ATOMIC", "")


def test_concrete_non_scalar_atomic_target_is_still_refused(tmp_path):
    source = KERNEL.replace(
        "template <typename AT, typename T>\n", "template <typename T>\n"
    ).replace("::metal::atomic<AT>*", "::metal::atomic<bool>*")
    with pytest.raises(MetalAtomicLoadLoweringError):
        _translate(tmp_path, source, "cgl")


def test_generic_atomic_helper_translations_compile(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is not None:
        metal = tmp_path / "out.metal"
        metal.write_text(_translate(tmp_path, KERNEL, "metal"), encoding="utf-8")
        result = _run(
            [xcrun, "-sdk", "macosx", "metal", "-c", str(metal)]
            + ["-o", str(tmp_path / "out.air")]
        )
        if "unable to find utility" not in result.stderr:
            assert result.returncode == 0, result.stderr

    glslang = shutil.which("glslangValidator")
    if glslang is not None:
        glsl = tmp_path / "out.comp"
        glsl.write_text(_translate(tmp_path, KERNEL, "opengl"), encoding="utf-8")
        result = _run([glslang, str(glsl)])
        assert result.returncode == 0, result.stdout + result.stderr
