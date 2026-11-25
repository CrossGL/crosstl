"""CUDA and HIP input: classification intrinsics, unscoped enumerators and buffer memory qualifiers.

Three constructs of a CUDA or HIP kernel reached CrossGL in a form no other target could compile or that lost a
promise of the source:

- ``__isinff``, ``__isnanf``, ``__finitef``, ``__signbitf`` and their double forms were written unchanged, so every
  target but CUDA and HIP called an undeclared function. They are ``int`` built-ins that return 1 or 0 (nvcc writes
  ``selp.u32 %r, 1, 0`` and ``shr.u32 %r, %r, 31``), and now import as the ``i32`` of CrossGL's ``bool`` tests, of
  the operand converted to the parameter type.
- An unscoped enumerator was written bare (``Low``), and CrossGL names enumerators through their enum
  (``Mode::Low``), so the Metal and GLSL output used an undeclared name. A later declaration of the name hides it.
- ``volatile T*`` and ``T* __restrict__`` buffer parameters lost the qualifier. They now import as ``@volatile`` and
  ``@restrict`` on the storage buffer, and the target lowering of kernel parameters keeps both.

HIP's ``__lane_id()`` was written unchanged as well; it is the lane's index in its wavefront, the subgroup invocation
index, as ``warpSize`` already imports as the subgroup size. So were the ``float`` math functions ``acoshf``,
``copysignf``, ``cospif``, ``exp10f``, ``fdimf``, ``ilogbf``, ``ldexpf``, ``log10f``, ``nextafterf``, ``sinpif`` and
the rest of :mod:`crosstl.backend.CUDA.libm_names`; they import as CrossGL's unsuffixed names, and the CUDA and HIP
generators spell ``exp10`` as ``exp10f`` or ``exp10`` by precision instead of ``exp2(x * log2(10))``. The CUDA
generator writes a ``@volatile`` storage buffer, parameter or program-scope pointer, as a pointer to ``volatile``
memory, as the HIP generator does.
"""

import shutil
import subprocess

import pytest

import crosstl

SUFFIXES = [".cu", ".hip"]

CLASSIFICATION = """
extern "C" __global__ void probe(const float* f, const double* d, int* o) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    o[i * 8 + 0] = __isinff(f[i]);
    o[i * 8 + 1] = __isnanf(f[i]);
    o[i * 8 + 2] = __finitef(f[i]);
    o[i * 8 + 3] = __signbitf(f[i]);
    o[i * 8 + 4] = __isinf(d[i]);
    o[i * 8 + 5] = __isnan(d[i]);
    o[i * 8 + 6] = __finite(d[i]);
    o[i * 8 + 7] = __signbit(d[i]);
}
"""

FLOAT_CLASSIFICATION = """
extern "C" __global__ void __launch_bounds__(64) probe(const float* f, int* o) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    o[i] = __isinff(f[i]) + 2 * __isnanf(f[i]) + 4 * __finitef(f[i]) + 8 * __signbitf(f[i]);
}
"""

ENUMS = """
enum Mode { Off, Low = 2, High = Low + 1 };
enum { TILE = 4, SPAN = TILE * 2 };
struct P { int Low; float v[TILE]; };
__device__ int pick(int Off) { return Off + High; }
extern "C" __global__ void __launch_bounds__(64) probe(const int* iin, float* outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    P p;
    p.Low = iin[i];
    Mode m = (iin[i] & 1) != 0 ? Low : High;
    float r = 0.0f;
    switch (m) {
    case Off:
        r = 1.0f;
        break;
    case Low:
        r = 2.0f;
        break;
    default:
        r = 4.0f;
        break;
    }
    int a = Low + 1;
    {
        int Low = 7;
        a += Low;
    }
    int s[SPAN];
    s[0] = High;
    outp[i] = r + (float)(a + pick(1) + p.Low + s[0] + SPAN + int(m));
}
"""

QUALIFIERS = """
extern "C" __global__ void __launch_bounds__(64) probe(
    const float* __restrict__ fin, float* __restrict__ outp, volatile float* flags, float* volatile plain,
    float* other) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = fin[i] * 2.0f + flags[i] + plain[i] + other[i];
}
"""

LIBM = """
extern "C" __global__ void __launch_bounds__(64) probe(const float* a, float* outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = a[i];
    float y = a[i + 1];
    outp[i] = asinhf(x) + acoshf(y) + atanhf(x) + copysignf(x, y) + cospif(x) + sinpif(x);
    outp[i + 1] = exp10f(x) + fdimf(x, y) + (float)ilogbf(y) + ldexpf(x, 3) + log10f(y) + nextafterf(x, y);
}
"""

EXP10_DOUBLE = """
extern "C" __global__ void __launch_bounds__(64) probe(double* d) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    d[i] = exp10(d[i]);
}
"""

# OpenCL C input lowers kernel buffer parameters to program-scope storage buffers, which keep ``@volatile``.
VOLATILE_GLOBAL = """
__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void probe(__global float* fin, __global volatile float* outp) {
    uint i = get_global_id(0);
    outp[i] = fin[i] * 2.0;
}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _flat(code):
    return " ".join(code.split())


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_classification_intrinsics_import_as_integer_tests(tmp_path, suffix):
    code = _flat(_translate(tmp_path, CLASSIFICATION, suffix, "cgl"))
    for expected in [
        "o[((i * 8) + 0)] = i32(isinf(f32(f[i])));",
        "o[((i * 8) + 1)] = i32(isnan(f32(f[i])));",
        "o[((i * 8) + 2)] = i32(isfinite(f32(f[i])));",
        "o[((i * 8) + 3)] = i32(floatBitsToUint(f32(f[i])) >> 31u);",
        "o[((i * 8) + 4)] = i32(isinf(f64(d[i])));",
        "o[((i * 8) + 5)] = i32(isnan(f64(d[i])));",
        "o[((i * 8) + 6)] = i32(isfinite(f64(d[i])));",
        "o[((i * 8) + 7)] = i32(as_type<u64>(f64(d[i])) >> 63u);",
    ]:
        assert expected in code
    assert "__is" not in code and "__finite" not in code and "__signbit" not in code


@pytest.mark.parametrize("suffix", SUFFIXES)
@pytest.mark.parametrize("backend", ["metal", "opengl", "vulkan", "directx"])
def test_classification_intrinsics_reach_every_target(tmp_path, suffix, backend):
    code = _translate(tmp_path, FLOAT_CLASSIFICATION, suffix, backend)
    assert "__isinff" not in code and "__signbitf" not in code
    if backend == "opengl" and shutil.which("glslangValidator"):
        shader = tmp_path / "probe.comp"
        shader.write_text(code)
        result = subprocess.run(
            ["glslangValidator", "-S", "comp", str(shader)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_unscoped_enumerators_are_named_through_their_enum(tmp_path, suffix):
    code = _flat(_translate(tmp_path, ENUMS, suffix, "cgl"))
    for expected in [
        "High = (Mode::Low + 1),",
        "SPAN = (anonymous_enum_0::TILE * 2),",
        "return (Off + Mode::High);",
        "var m: Mode = (((iin[i] & 1) != 0) ? Mode::Low : Mode::High);",
        "case Mode::Off:",
        "case Mode::Low:",
        "var a: i32 = (Mode::Low + 1);",
        "var Low: i32 = 7; a += Low;",
        "s[0] = Mode::High;",
        "p.Low = iin[i];",
    ]:
        assert expected in code


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_unscoped_enumerators_compile_in_glsl(tmp_path, suffix):
    code = _translate(tmp_path, ENUMS, suffix, "opengl")
    assert "Mode_Low" in code
    if shutil.which("glslangValidator") is None:
        pytest.skip("glslangValidator is not installed")
    shader = tmp_path / "probe.comp"
    shader.write_text(code)
    result = subprocess.run(
        ["glslangValidator", "-S", "comp", str(shader)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_buffer_memory_qualifiers_become_attributes(tmp_path, suffix):
    code = _translate(tmp_path, QUALIFIERS, suffix, "cgl")
    lines = {
        line.split(":")[0].split()[-1]: line.strip()
        for line in code.splitlines()
        if "var<storage" in line
    }
    assert lines["fin"].startswith("@restrict @group(0)")
    assert lines["outp"].startswith("@restrict @group(0)")
    assert lines["flags"].startswith("@volatile @group(0)")
    # A volatile pointer variable is not volatile memory, and an unqualified pointer promises nothing.
    assert lines["plain"].startswith("@group(0)")
    assert lines["other"].startswith("@group(0)")


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_buffer_memory_qualifiers_reach_spirv_glsl_and_metal(tmp_path, suffix):
    spirv = _translate(tmp_path, QUALIFIERS, suffix, "vulkan")
    assert "OpMemberDecorate %finBuffer 0 Restrict" in spirv
    assert "OpMemberDecorate %outpBuffer 0 Restrict" in spirv
    assert "OpMemberDecorate %flagsBuffer 0 Volatile" in spirv
    assert "%otherBuffer 0 Restrict" not in spirv
    glsl = _flat(_translate(tmp_path, QUALIFIERS, suffix, "opengl"))
    assert "restrict" in glsl.split("buffer finBuffer")[0].rsplit(";", 1)[-1]
    assert "restrict buffer outpBuffer" in glsl
    assert "volatile buffer flagsBuffer" in glsl
    assert "restrict buffer otherBuffer" not in glsl
    metal = _flat(_translate(tmp_path, QUALIFIERS, suffix, "metal"))
    assert "volatile device float *flags" in metal
    if suffix == ".cu":
        # CUDA kernels reach Metal through the target lowering of kernel parameters, which keeps @restrict.
        assert "device float *__restrict outp" in metal


@pytest.mark.parametrize("suffix", SUFFIXES)
@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_buffer_memory_qualifiers_round_trip(tmp_path, suffix, backend):
    code = _flat(_translate(tmp_path, QUALIFIERS, suffix, backend))
    assert "float* __restrict__ outp" in code
    assert "volatile float* flags" in code
    assert "float* other" in code


def test_hip_lane_id_is_the_subgroup_invocation_index(tmp_path):
    source = """
extern "C" __global__ void __launch_bounds__(64) probe(float* outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = float(__lane_id()) + float(warpSize);
}
"""
    code = _flat(_translate(tmp_path, source, ".hip", "cgl"))
    assert "outp[i] = (f32(WaveGetLaneIndex()) + f32(int(gl_SubgroupSize)));" in code
    metal = _translate(tmp_path, source, ".hip", "metal")
    assert "__lane_id" not in metal
    assert "thread_index_in_simdgroup" in metal


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_float_math_functions_import_without_their_suffix(tmp_path, suffix):
    code = _flat(_translate(tmp_path, LIBM, suffix, "cgl"))
    assert (
        "outp[i] = (((((asinh(x) + acosh(y)) + atanh(x)) + copysign(x, y)) + cospi(x)) + sinpi(x));"
        in code
    )
    assert (
        "outp[(i + 1)] = (((((exp10(x) + fdim(x, y)) + f32(ilogb(y))) + ldexp(x, 3)) + log10(y)) "
        "+ nextafter(x, y));" in code
    )
    for name in (
        "acoshf",
        "copysignf",
        "exp10f",
        "fdimf",
        "ilogbf",
        "ldexpf",
        "nextafterf",
    ):
        assert f"{name}(" not in code


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_float_math_functions_reach_metal(tmp_path, suffix):
    code = _translate(tmp_path, LIBM, suffix, "metal")
    for name in (
        "acosh",
        "copysign",
        "cospi",
        "exp10",
        "fdim",
        "ldexp",
        "log10",
        "nextafter",
    ):
        assert f"{name}(" in code, name
        assert f"{name}f(" not in code, name


@pytest.mark.parametrize("suffix", SUFFIXES)
@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_exp10_keeps_the_math_library_function(tmp_path, suffix, backend):
    code = _translate(tmp_path, LIBM, suffix, backend)
    assert "exp10f(x)" in code
    assert "3.321928094887362" not in code
    code = _translate(tmp_path, EXP10_DOUBLE, suffix, backend)
    assert "exp10(d[i])" in code
    assert "3.321928094887362" not in code


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_volatile_storage_buffers_point_to_volatile_memory(tmp_path, backend):
    code = _translate(tmp_path, VOLATILE_GLOBAL, ".cl", backend)
    assert "__device__ volatile float* outp;" in code
    assert "volatile float* fin" not in code


def _compile(tmp_path, tool, args, code, suffix):
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} is not installed")
    source = tmp_path / f"round_trip{suffix}"
    source.write_text(code)
    result = subprocess.run(
        [tool, *args, str(source), "-o", str(tmp_path / "round_trip.o")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "source",
    [CLASSIFICATION, ENUMS, QUALIFIERS, LIBM, EXP10_DOUBLE],
    ids=["classification", "enums", "qualifiers", "libm", "exp10-double"],
)
def test_nvcc_compiles_the_round_trip(tmp_path, source):
    code = _translate(tmp_path, source, ".cu", "cuda")
    _compile(tmp_path, "nvcc", ["-std=c++17", "-c"], code, ".cu")


@pytest.mark.parametrize(
    "source",
    [CLASSIFICATION, ENUMS, QUALIFIERS, LIBM, EXP10_DOUBLE],
    ids=["classification", "enums", "qualifiers", "libm", "exp10-double"],
)
def test_hipcc_compiles_the_round_trip(tmp_path, source):
    code = _translate(tmp_path, source, ".hip", "hip")
    _compile(
        tmp_path, "hipcc", ["-std=c++17", "-c", "--offload-arch=gfx90a"], code, ".hip"
    )
