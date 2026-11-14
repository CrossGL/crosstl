"""CUDA and HIP output declare named 8- and 16-bit integer vectors as ``char2``..``ushort4``.

CrossGL spells them ``i8vec2``/``u16vec3`` (the GLSL importer's spelling) or ``vec2<i8>`` (the CUDA and HIP
importers' spelling). Both generators mapped only the generic spelling, so a GLSL kernel using ``i8vec2`` came out
with ``i8vec2`` declarations and constructors and plain ``+`` on them, which names no CUDA or HIP type.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.c_narrow_integer_vectors import (
    generic_narrow_vector_spelling,
)

BACKENDS = ("cuda", "hip")

GLSL_KERNEL = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer B { int o[]; } b;
void main() {
    i8vec2 a = i8vec2(int8_t(100), int8_t(-100));
    u8vec3 u = u8vec3(uint8_t(200));
    i16vec4 s = i16vec4(int16_t(30000), int16_t(-2), int16_t(3), int16_t(4));
    u16vec2 w = u16vec2(uint16_t(65535), uint16_t(1));
    i8vec2 a2 = a + a;
    u8vec3 u2 = u * u;
    i16vec4 s2 = s - s.wzyx;
    u16vec2 w2 = w + w;
    b.o[0] = int(a2.x) + int(u2.y) + int(s2.z) + int(w2.x);
}
"""

NATIVE = {
    "i8vec2": "char2",
    "u8vec3": "uchar3",
    "i16vec4": "short4",
    "u16vec2": "ushort2",
}

# Swizzles of generic narrow vectors, which were typed as int and uint vectors (``short4 t = make_int4(...)``).
CGL_SWIZZLES = """shader S {
    compute {
        buffer int o[];
        layout(local_size_x = 1) in;
        void main() {
            vec4<i16> s = vec4<i16>(i16(1), i16(2), i16(3), i16(4));
            vec4<i16> t = s.wzyx;
            s.xy = s.wz;
            vec2<u8> u = vec2<u8>(u8(1), u8(2));
            vec2<u8> v = u.yx + u;
            o[0] = int(t.x) + int(v.x) + int(s.x);
        }
    }
}
"""


def translate(tmp_path, backend, source=GLSL_KERNEL, suffix=".comp"):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize(
    ("spelling", "generic"),
    [
        ("i8vec2", "vec2<i8>"),
        ("u8vec3", "vec3<u8>"),
        ("i16vec4", "vec4<i16>"),
        ("u16vec2", "vec2<u16>"),
    ],
)
def test_named_spellings_take_the_generic_spelling(spelling, generic):
    assert generic_narrow_vector_spelling(spelling) == generic


@pytest.mark.parametrize(
    "spelling", ["i8vec5", "ivec2", "u64vec2", "i32vec2", "vec2<i8>", "i8", None]
)
def test_other_spellings_are_not_named_narrow_vectors(spelling):
    assert generic_narrow_vector_spelling(spelling) is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_glsl_narrow_vectors_use_the_native_vectors(tmp_path, backend):
    code = translate(tmp_path, backend)
    for named, native in NATIVE.items():
        assert f"{native} " in code, native
        assert f"make_{native}(" in code, native
        assert named not in code, named
    assert "char2 a2 = cgl_char2_add(a, a);" in code
    assert "uchar3 u2 = cgl_uchar3_mul(u, u);" in code
    assert "ushort2 w2 = cgl_ushort2_add(w, w);" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_narrow_vector_swizzles_keep_their_element_type(tmp_path, backend):
    code = translate(tmp_path, backend, CGL_SWIZZLES, ".cgl")
    assert "short4 t = make_short4(s.w, s.z, s.y, s.x);" in code
    assert "short2 __crossgl_swizzle_value_0 = make_short2(s.w, s.z);" in code
    assert "uchar2 v = cgl_uchar2_add(make_uchar2(u.y, u.x), u);" in code
    assert "make_int4" not in code
    assert "make_uint2" not in code


def compile_with(tmp_path, code, command, suffix):
    source = tmp_path / f"kernel{suffix}"
    source.write_text(code)
    result = subprocess.run(
        [*command, str(source), "-o", str(tmp_path / "kernel.o")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]


@pytest.mark.parametrize(
    ("source", "suffix"),
    [(GLSL_KERNEL, ".comp"), (CGL_SWIZZLES, ".cgl")],
    ids=["glsl-named", "cgl-swizzles"],
)
def test_nvcc_accepts_narrow_vectors(tmp_path, source, suffix):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")
    code = translate(tmp_path, "cuda", source, suffix)
    compile_with(tmp_path, code, [nvcc, "-std=c++17", "-c"], ".cu")


@pytest.mark.parametrize(
    ("source", "suffix"),
    [(GLSL_KERNEL, ".comp"), (CGL_SWIZZLES, ".cgl")],
    ids=["glsl-named", "cgl-swizzles"],
)
def test_hipcc_accepts_narrow_vectors(tmp_path, source, suffix):
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")
    code = translate(tmp_path, "hip", source, suffix)
    compile_with(
        tmp_path, code, [hipcc, "--offload-arch=gfx1100", "-std=c++17", "-c"], ".hip"
    )
