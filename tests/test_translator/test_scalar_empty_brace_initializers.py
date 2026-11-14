"""An empty brace initializer of a scalar is the scalar's zero on every target (#3735).

C++ value-initializes ``float one = {};`` and ``int n = {};`` to zero. CUDA and HIP input import them as
``var one: f32 = {};``, which HLSL output copied (DXC rejects it), WGSL output wrote as ``array()`` (naga rejects it)
and GLSL output refused as an aggregate initializer.
"""

import textwrap

import pytest

import crosstl

KERNEL = textwrap.dedent("""
    extern "C" __global__ void first(float* out) {
        float one = {};
        int n = {};
        unsigned int u = {};
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        out[i] = one + (float)n + (float)u;
    }
""")

ALIASED = textwrap.dedent("""
    typedef float real_t;
    extern "C" __global__ void first(float* out) {
        real_t one = {};
        static __constant__ real_t two;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        out[i] = one + two;
    }
""")

EXPECTED = {
    "directx": ["float one = 0.0;", "int n = 0;", "uint u = 0u;"],
    "opengl": ["float one = 0.0;", "int n = 0;", "uint u = 0u;"],
    "wgsl": ["var one: f32 = 0.0;", "var n: i32 = 0;", "var u: u32 = 0u;"],
    "metal": ["float one = 0.0;", "int n = 0;", "uint u = 0u;"],
}


def translate(tmp_path, source, backend, suffix=".cu"):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
@pytest.mark.parametrize("backend", sorted(EXPECTED))
def test_empty_brace_scalar_initializer_is_zero(tmp_path, backend, suffix):
    output = translate(tmp_path, KERNEL, backend, suffix)
    for line in EXPECTED[backend]:
        assert line in output
    assert "{}" not in output
    assert "array()" not in output


@pytest.mark.parametrize("backend", ["directx", "opengl", "wgsl"])
def test_empty_brace_initializer_of_a_scalar_alias_is_zero(tmp_path, backend):
    output = translate(tmp_path, ALIASED, backend)
    assert "{}" not in output
    assert "array()" not in output
    zero = "f32 = 0.0;" if backend == "wgsl" else " = 0.0;"
    assert output.count(zero) == 2


STAGE_SCOPE = textwrap.dedent("""
    shader P {
        compute {
            buffer float* outp;
            float g = {};
            void main() { outp[0] = g; }
        }
    }
""")

STAGE_SCOPE_EXPECTED = {
    "directx": "static float g = 0.0;",
    "opengl": "float g = 0.0;",
    "wgsl": "var<private> g: f32 = 0.0;",
    "metal": "float g = 0.0;",
}


@pytest.mark.parametrize("backend", sorted(STAGE_SCOPE_EXPECTED))
def test_empty_brace_initializer_of_a_stage_scope_scalar_is_zero(tmp_path, backend):
    # The GLSL generator follows parent links when it orders a stage's globals, so a zero linked back to ``g`` read as
    # ``g`` depending on itself.
    output = translate(tmp_path, STAGE_SCOPE, backend, ".cgl")
    assert STAGE_SCOPE_EXPECTED[backend] in output


NARROW = textwrap.dedent("""
    extern "C" __global__ void first(float* out) {
        short s = SIGNED;
        unsigned char c = UNSIGNED;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        out[i] = (float)s + (float)c;
    }
""")
WIDE = NARROW.replace("short", "long long").replace(
    "unsigned char", "unsigned long long"
)


def _sized(template, braces):
    signed, unsigned = ("{}", "{}") if braces else ("0", "0u")
    return template.replace("UNSIGNED", unsigned).replace("SIGNED", signed)


@pytest.mark.parametrize(
    "template, backend",
    [
        pytest.param(NARROW, backend, id=f"narrow-{backend}")
        for backend in ("directx", "opengl", "metal", "wgsl")
    ]
    # WGSL has no 64-bit integers.
    + [
        pytest.param(WIDE, backend, id=f"wide-{backend}")
        for backend in ("directx", "opengl", "metal")
    ],
)
def test_empty_brace_initializer_of_a_sized_integer_is_its_zero(
    tmp_path, template, backend
):
    zero = translate(tmp_path, _sized(template, braces=False), backend)
    assert translate(tmp_path, _sized(template, braces=True), backend) == zero


def test_crossgl_output_keeps_the_braces(tmp_path):
    output = translate(tmp_path, KERNEL, "cgl")
    assert "var one: f32 = {};" in output


# An empty brace list whose destination is a scalar outside a scalar's declaration (#3751).
ASSIGNED = """
shader P {
    compute {
        buffer float* outp;
        void main() {
            float x = outp[1];
            x = {};
            outp[0] = x;
        }
    }
}
"""

RETURNED = """
shader P {
    float zero() { return {}; }
    compute {
        buffer float* outp;
        void main() { outp[0] = zero(); }
    }
}
"""

MEMBER = """
shader P {
    struct S { float a; float b; };
    compute {
        buffer float* outp;
        void main() {
            S s = {{}, 1.0};
            outp[0] = s.a + s.b;
        }
    }
}
"""

ELEMENT = """
shader P {
    compute {
        buffer float* outp;
        void main() {
            float a[2] = {{}, 1.0};
            outp[0] = a[0] + a[1];
        }
    }
}
"""

POSITION_EXPECTED = {
    "assigned": {
        "directx": "x = 0.0;",
        "opengl": "x = 0.0;",
        "wgsl": "x = 0.0;",
        "metal": "x = 0.0;",
    },
    "returned": {
        "directx": "return 0.0;",
        "opengl": "return 0.0;",
        "wgsl": "return 0.0;",
        "metal": "return 0.0;",
    },
    "member": {
        "directx": "s.a = 0.0;",
        "opengl": "S s = S(0.0, 1.0);",
        "wgsl": "var s: S = S(0.0, 1.0);",
        "metal": "S s = {0.0, 1.0};",
    },
    "element": {
        "directx": "float a[2] = {0.0, 1.0};",
        "opengl": "float a[2] = float[2](0.0, 1.0);",
        "wgsl": "var a: array<f32, 2> = array(0.0, 1.0);",
        "metal": "float a[2] = {0.0, 1.0};",
    },
}
POSITION_SOURCES = {
    "assigned": ASSIGNED,
    "returned": RETURNED,
    "member": MEMBER,
    "element": ELEMENT,
}


@pytest.mark.parametrize("backend", ["directx", "opengl", "wgsl", "metal"])
@pytest.mark.parametrize("position", sorted(POSITION_SOURCES))
def test_empty_brace_list_with_a_scalar_destination_is_zero(
    tmp_path, position, backend
):
    output = translate(tmp_path, POSITION_SOURCES[position], backend, ".cgl")
    assert POSITION_EXPECTED[position][backend] in output
    assert "{}" not in output
    assert "array()" not in output


IMPORTED = textwrap.dedent("""
    struct S { float a; float b; };
    extern "C" __global__ void first(float* out) {
        float x = out[1];
        x = {};
        int n = 3;
        n = {};
        S s = {{}, 1.0f};
        out[0] = x + (float)n + s.a + s.b;
    }
""")


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
@pytest.mark.parametrize("backend", ["directx", "opengl", "wgsl"])
def test_imported_empty_brace_list_with_a_scalar_destination_is_zero(
    tmp_path, backend, suffix
):
    output = translate(tmp_path, IMPORTED, backend, suffix)
    assert "x = 0.0;" in output
    assert "n = 0;" in output
    assert "{}" not in output
    assert "array()" not in output


def test_empty_brace_list_with_a_vector_destination_keeps_its_braces(tmp_path):
    source = """
    shader P {
        compute {
            buffer float* outp;
            void main() {
                float2 v = float2(outp[1], outp[2]);
                v = {};
                outp[0] = v.x;
            }
        }
    }
    """
    output = translate(tmp_path, source, "metal", ".cgl")
    assert "v = {};" in output


def test_empty_brace_list_after_brace_elision_keeps_its_braces(tmp_path):
    # ``1.0`` starts the array member ``a``, so ``{}`` is ``a[1]`` rather than ``b``.
    source = """
    shader P {
        struct S { float a[2]; float b; float c; };
        compute {
            buffer float* outp;
            void main() {
                S s = {1.0, {}, 2.0};
                outp[0] = s.b;
            }
        }
    }
    """
    output = translate(tmp_path, source, "metal", ".cgl")
    assert "{1.0, {}, 2.0}" in output
