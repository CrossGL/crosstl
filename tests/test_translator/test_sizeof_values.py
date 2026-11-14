"""CrossGL ``sizeof`` and ``metal_sizeof`` become byte counts, or are refused where the targets disagree.

The generators printed ``sizeof`` unchanged: GLSL has no such operator, SPIR-V refused it, MSL and HLSL measured the
operand with their own layout, and ``sizeof(T)`` reached a generic function's specializations with ``T`` unbound.
"""

import re
import textwrap

import pytest

import crosstl
from crosstl.translator.semantics import SizeofUnsupportedError
from crosstl.translator.semantics.sizeof_values import sizeof_type_bytes

GRAPHICS_TARGETS = ("metal", "directx", "opengl", "vulkan")
SPECIALIZING_TARGETS = ("metal", "directx", "opengl")

PORTABLE = textwrap.dedent("""
    shader main {
        const int WORDS = 16 / sizeof(float);
        compute {
            RWStructuredBuffer<int> out_ @buffer(0);
            @numthreads(1, 1, 1)
            void main(uint3 gid @gl_GlobalInvocationID) {
                half h = half(1.0);
                float arr[4];
                arr[0] = 1.0;
                int total = WORDS + int(sizeof(h)) + int(sizeof(arr)) + int(sizeof(float[2]));
                total = total + int(sizeof(arr[1])) + int(sizeof(int8)) + int(sizeof(uint64)) + int(sizeof(ivec4));
                out_[gid.x] = total;
            }
        }
    }
    """)

GENERIC = textwrap.dedent("""
    shader main {
        generic<T> uint stride_of(T v) {
            T local = v;
            return uint(sizeof(local)) + uint(sizeof(T[3])) * 100u;
        }
        compute {
            RWStructuredBuffer<uint> out_ @buffer(0);
            @numthreads(1, 1, 1)
            void main(uint3 gid @gl_GlobalInvocationID) {
                half h = half(1.0);
                int8 small = int8(1);
                vec4 wide = vec4(1.0);
                out_[gid.x] = stride_of(h) + stride_of(1.0) + stride_of(small) + stride_of(wide);
            }
        }
    }
    """)

METAL_LAYOUT = textwrap.dedent("""
    shader main {
        generic<T> uint msize(T v) {
            return uint(metal_sizeof(T)) + uint(metal_sizeof(v)) * 100u;
        }
        compute {
            RWStructuredBuffer<uint> out_ @buffer(0);
            @numthreads(1, 1, 1)
            void main(uint3 gid @gl_GlobalInvocationID) {
                bool flag = true;
                vec3 p = vec3(1.0);
                mat3 m = mat3(1.0);
                uint total = uint(metal_sizeof(flag)) + uint(metal_sizeof(vec3)) + uint(metal_sizeof(bvec3));
                total = total + uint(metal_sizeof(m)) + uint(metal_sizeof(mat2x3)) + uint(metal_sizeof(vec3[2]));
                total = total + msize(p) + msize(flag);
                out_[gid.x] = total;
            }
        }
    }
    """)

METAL_TEMPLATE = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    template <typename T>
    uint size_of(T value) { return uint(sizeof(T)) * 100u + uint(sizeof(value)); }
    kernel void sizes(device uint* results [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
        half3 three = half3(1.0h);
        bool flag = tid > 0u;
        results[tid] = size_of(float3(0.0)) + size_of(three) + size_of(flag);
    }
    """)

METAL_BFLOAT = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    kernel void sizes(device uint* results [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
        results[tid] = uint(sizeof(bfloat4)) + uint(sizeof(bfloat3)) * 100u;
    }
    """)


def _translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _lines(code, marker):
    return [line.strip() for line in code.splitlines() if marker in line]


def _values(line):
    return [int(value) for value in re.findall(r"u?int\((\d+)\)", line)]


@pytest.mark.parametrize("backend", GRAPHICS_TARGETS)
def test_portable_sizeof_folds_to_byte_counts(tmp_path, backend):
    code = _translate(tmp_path, PORTABLE, backend)
    assert "sizeof" not in code
    if backend == "vulkan":
        constants = set(re.findall(r"OpConstant %\S+ (\d+)$", code, re.MULTILINE))
        assert {"1", "2", "4", "8", "16"} <= constants
        return
    first, second = (
        line for line in _lines(code, "total =") if "WORDS" in line or "total +" in line
    )
    assert _values(first) == [2, 16, 8]
    assert _values(second) == [4, 1, 8, 16]
    assert re.search(r"WORDS = \(?16 / 4\)?;", code)


@pytest.mark.parametrize("backend", SPECIALIZING_TARGETS)
def test_sizeof_of_a_type_parameter_folds_in_each_specialization(tmp_path, backend):
    code = _translate(tmp_path, GENERIC, backend)
    assert "sizeof" not in code
    returns = [_values(line) for line in _lines(code, "100u")]
    # half, float, int8 and vec4 keep their CrossGL widths after the target widens half and int8.
    assert sorted(returns) == sorted([[2, 6], [4, 12], [1, 3], [16, 48]])


@pytest.mark.parametrize("backend", ["vulkan", "wgsl", "slang", "rust", "mojo"])
def test_sizeof_of_a_type_parameter_is_refused_without_folding_specializations(
    tmp_path, backend
):
    with pytest.raises(SizeofUnsupportedError, match=r"type parameter"):
        _translate(tmp_path, GENERIC, backend)


def test_crossgl_output_keeps_sizeof(tmp_path):
    code = _translate(tmp_path, GENERIC, "cgl")
    assert "sizeof(local)" in code and "sizeof(T[3])" in code


@pytest.mark.parametrize(
    ("declaration", "operand", "detail"),
    [
        ("bool flag = true;", "flag", "bool has no portable size"),
        ("vec3 p = vec3(1.0);", "p", "3-component vector"),
        ("mat3 m = mat3(1.0);", "m", "matrix"),
        ("Pair pair;", "pair", "structure's layout"),
        ("min16float f = min16float(1.0);", "f", "minimum-precision"),
    ],
)
@pytest.mark.parametrize("backend", GRAPHICS_TARGETS)
def test_unportable_sizeof_is_refused(tmp_path, backend, declaration, operand, detail):
    source = textwrap.dedent(f"""
        shader main {{
            struct Pair {{ float a; int b; }};
            compute {{
                RWStructuredBuffer<int> out_ @buffer(0);
                @numthreads(1, 1, 1)
                void main(uint3 gid @gl_GlobalInvocationID) {{
                    {declaration}
                    out_[gid.x] = int(sizeof({operand}));
                }}
            }}
        }}
        """)
    with pytest.raises(SizeofUnsupportedError, match=re.escape(detail)) as raised:
        _translate(tmp_path, source, backend)
    assert raised.value.target == backend


def test_cuda_keeps_native_sizeof_of_a_structure(tmp_path):
    source = textwrap.dedent("""
        shader main {
            struct Pair { float a; int b; };
            compute {
                RWStructuredBuffer<int> out_ @buffer(0);
                @numthreads(1, 1, 1)
                void main(uint3 gid @gl_GlobalInvocationID) {
                    Pair pair;
                    out_[gid.x] = int(sizeof(pair)) + int(sizeof(float));
                }
            }
        }
        """)
    code = _translate(tmp_path, source, "cuda")
    assert "sizeof(pair)" in code and "sizeof(float)" not in code


@pytest.mark.parametrize("backend", SPECIALIZING_TARGETS)
def test_metal_sizeof_measures_with_msl_layout(tmp_path, backend):
    code = _translate(tmp_path, METAL_LAYOUT, backend)
    assert "sizeof" not in code
    totals = [_values(line) for line in _lines(code, "total =")]
    assert totals[:2] == [[1, 16, 4], [48, 32, 32]]
    returns = sorted(_values(line) for line in _lines(code, "100u"))
    assert returns == [[1, 1], [16, 16]]


@pytest.mark.parametrize(
    ("type_value", "portable", "metal"),
    [
        ("float", 4, 4),
        ("half", 2, 2),
        ("bfloat16", 2, 2),
        ("bfloat4", 8, 8),
        ("int8", 1, 1),
        ("uint64", 8, 8),
        ("vec2", 8, 8),
        ("i16vec4", 8, 8),
        ("float[3]", 12, 12),
        ("vec3", None, 16),
        ("bfloat3", None, 8),
        ("bvec3", None, 4),
        ("bool", None, 1),
        ("mat3", None, 48),
        ("mat2x3", None, 32),
        ("mat3x2", None, 24),
        ("f16mat3", None, 24),
        ("dmat2", None, None),
        ("min16float", None, None),
        ("Pair", None, None),
    ],
)
def test_layouts(type_value, portable, metal):
    for layout, expected in (("portable", portable), ("metal", metal)):
        if expected is None:
            with pytest.raises(SizeofUnsupportedError):
                sizeof_type_bytes(type_value, layout=layout)
        else:
            assert sizeof_type_bytes(type_value, layout=layout) == expected


def test_metal_sizeof_of_names_the_parser_reads_as_values(tmp_path):
    # bfloat4 and bfloat3 have no type keyword, so the parser reads them as variable references.
    crossgl = _translate(tmp_path, METAL_BFLOAT, "cgl", suffix=".metal")
    assert "sizeof" not in crossgl
    assert "uint(8) + uint(8) * 100u" in crossgl


# float3, half3 and bool, measured with MSL's layout.
METAL_TEMPLATE_SIZES = [[1, 1], [8, 8], [16, 16]]


def test_metal_template_sizeof_keeps_msl_layout_through_crossgl(tmp_path):
    crossgl = _translate(tmp_path, METAL_TEMPLATE, "cgl", suffix=".metal")
    assert "metal_sizeof(T)" in crossgl and "metal_sizeof(value)" in crossgl
    for backend in SPECIALIZING_TARGETS:
        code = _translate(tmp_path, crossgl, backend)
        assert "sizeof" not in code
        returns = sorted(
            _values(line) for line in _lines(code, "100u") if "return" in line
        )
        assert returns == METAL_TEMPLATE_SIZES, backend
    with pytest.raises(SizeofUnsupportedError, match=r"metal_sizeof\(T\)"):
        _translate(tmp_path, crossgl, "vulkan")


@pytest.mark.parametrize("backend", SPECIALIZING_TARGETS)
def test_metal_template_sizeof_translates_directly(tmp_path, backend):
    code = _translate(tmp_path, METAL_TEMPLATE, backend, suffix=".metal")
    assert "sizeof" not in code
    returns = sorted(_values(line) for line in _lines(code, "100u") if "return" in line)
    assert returns == METAL_TEMPLATE_SIZES
