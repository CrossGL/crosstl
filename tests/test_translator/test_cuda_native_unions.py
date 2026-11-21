"""CUDA and HIP output write a CrossGL union-layout struct as a C++ union.

A struct marked ``@union_layout`` stands for a union whose members share storage. CUDA and HIP output wrote it as a
``struct``, so a kernel that wrote ``b.f`` and read ``b.u`` read a member that was never written (#4774). The
generators now write a ``union`` whose layout is the recorded one, with a ``static_assert`` that nvcc and hipcc check,
and refuse a union whose recorded layout C++ cannot reproduce.
"""

import re
import shutil
import subprocess

import pytest

from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.refusals import CudaRefusal, HipRefusal

GENERATORS = {"cuda": (CudaCodeGen, CudaRefusal), "hip": (HipCodeGen, HipRefusal)}

BITS = """
    @union_layout(4, 4, little_endian, {abi})
    struct Bits {{
        @union_member_layout(0, 4, 4) float f;
        @union_member_layout(0, 4, 4) uint u;
    }}
"""


def _program(structs, body):
    return f"""
shader main {{
{structs}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* inp;
        buffer uint* out_;

        void main() {{
            uint i = gl_GlobalInvocationID.x;
{body}
        }}
    }}
}}
"""


PUN = """
            Bits b;
            b.f = inp[i];
            out_[i] = b.u;
"""


def _generate(backend, source):
    generator, _refusal = GENERATORS[backend]
    return generator().generate(Parser(Lexer(source).tokens).parse())


def _refused(backend, source, pattern):
    _generator, refusal = GENERATORS[backend]
    with pytest.raises(refusal, match=pattern):
        _generate(backend, source)


def _union_body(code, name):
    match = re.search(
        rf"^union(?: alignas\(\d+\))? {name}\s*\{{(.*?)\}};", code, re.M | re.S
    )
    assert match is not None, code
    return [line.strip() for line in match.group(1).splitlines() if line.strip()]


@pytest.mark.parametrize("backend", sorted(GENERATORS))
@pytest.mark.parametrize("abi", ["cuda", "metal"])
def test_union_layout_struct_is_a_union(backend, abi):
    code = _generate(backend, _program(BITS.format(abi=abi), PUN))

    members = _union_body(code, "Bits")
    assert members[0] == "float f;"
    assert members[1] in ("uint u;", "unsigned int u;")
    assert not re.search(r"^struct Bits\b", code, re.M)
    assert re.search(
        r"static_assert\(sizeof\(Bits\) == 4 && alignof\(Bits\) == 4,", code
    )
    assert "b.f = inp[i];" in code
    assert "out_[i] = b.u;" in code


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_plain_struct_stays_a_struct(backend):
    structs = "    struct Pair { float f; uint u; }"
    body = """
            Pair p;
            p.f = inp[i];
            p.u = 1u;
            out_[i] = p.u;
"""
    code = _generate(backend, _program(structs, body))

    assert re.search(r"^struct Pair\b", code, re.M)
    assert "union" not in code
    assert "static_assert(sizeof(Pair)" not in code


NARROW_MEMBERS = """
    @union_layout(8, 8, little_endian, cuda)
    struct Wide {
        @union_member_layout(0, 8, 4) uint words[2];
        @union_member_layout(0, 8, 1) uint8 bytes[8];
        @union_member_layout(0, 2, 2) uint16 low;
        @union_member_layout(0, 8, 8) vec2 pair;
    }
"""


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_members_narrower_than_the_union_keep_their_own_layout(backend):
    body = """
            Wide w;
            w.pair = vec2(inp[i], inp[i + 1u]);
            out_[i] = w.words[1] + uint(w.bytes[0]) + uint(w.low);
"""
    code = _generate(backend, _program(NARROW_MEMBERS, body))

    assert len(_union_body(code, "Wide")) == 4
    assert "sizeof(Wide) == 8 && alignof(Wide) == 8" in code


def _bits_with(size, alignment, member_f, member_u, byte_order="little_endian"):
    return f"""
    @union_layout({size}, {alignment}, {byte_order}, metal)
    struct Bits {{
        {member_f} float f;
        {member_u} uint u;
    }}
"""


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_over_aligned_union_is_written_alignas(backend):
    structs = _bits_with(
        16, 16, "@union_member_layout(0, 4, 4)", "@union_member_layout(0, 4, 4)"
    )
    code = _generate(backend, _program(structs, PUN))

    assert re.search(r"^union alignas\(16\) Bits\b", code, re.M)
    assert "sizeof(Bits) == 16 && alignof(Bits) == 16" in code


NESTED = """
    struct Pair { float a; float b; }
    @union_layout(8, 8, little_endian, cuda)
    struct Inner {
        @union_member_layout(0, 8, 4) Pair pair;
        @union_member_layout(0, 8, 8) uvec2 words;
    }
    @union_layout(16, 16, little_endian, cuda)
    struct Outer {
        @union_member_layout(0, 8, 8) Inner inner;
        @union_member_layout(0, 16, 16) uvec4 all;
    }
"""


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_struct_and_union_members_take_their_c_layout(backend):
    body = """
            Outer o;
            o.inner.pair.a = inp[i];
            out_[i] = o.all.x + o.inner.words.y;
"""
    code = _generate(backend, _program(NESTED, body))

    assert re.search(r"^struct Pair\b", code, re.M)
    assert _union_body(code, "Inner")
    assert _union_body(code, "Outer")
    assert "sizeof(Inner) == 8 && alignof(Inner) == 8" in code
    assert "sizeof(Outer) == 16 && alignof(Outer) == 16" in code


REFUSED = {
    "metal-vec3-layout": (
        """
    @union_layout(16, 16, little_endian, metal)
    struct Bits {
        @union_member_layout(0, 16, 16) vec3 v;
        @union_member_layout(0, 16, 16) uvec4 u;
    }
""",
        r"member 'v' of type '(vec3|float3)' takes 12 bytes at 4-byte alignment",
    ),
    "member-offset": (
        _bits_with(
            8, 4, "@union_member_layout(0, 4, 4)", "@union_member_layout(4, 4, 4)"
        ),
        r"member 'u' starts at byte 4",
    ),
    "big-endian": (
        _bits_with(
            4,
            4,
            "@union_member_layout(0, 4, 4)",
            "@union_member_layout(0, 4, 4)",
            byte_order="big_endian",
        ),
        r"byte order 'big_endian'",
    ),
    "member-layout-missing": (
        _bits_with(4, 4, "@union_member_layout(0, 4, 4)", ""),
        r"member 'u' has no @union_member_layout",
    ),
    "size-mismatch": (
        _bits_with(
            8, 4, "@union_member_layout(0, 4, 4)", "@union_member_layout(0, 4, 4)"
        ),
        r"recorded size 8 is not the 4 bytes",
    ),
    "alignment-below-members": (
        _bits_with(
            4, 2, "@union_member_layout(0, 4, 4)", "@union_member_layout(0, 4, 4)"
        ),
        r"recorded alignment 2 is below the 4-byte alignment",
    ),
    "constructed-member": (
        """
    @union_layout(8, 8, little_endian, cuda)
    struct Bits {
        @union_member_layout(0, 8, 8) f16vec4 h;
        @union_member_layout(0, 8, 8) uvec2 u;
    }
""",
        r"member 'h' has type '(f16vec4|cgl_half4)'",
    ),
    "double4": (
        """
    @union_layout(32, 16, little_endian, cuda)
    struct Bits {
        @union_member_layout(0, 32, 16) dvec4 d;
        @union_member_layout(0, 16, 16) uvec4 u;
    }
""",
        r"member 'd' has type '(dvec4|double4)', whose alignment is 16 bytes",
    ),
}


@pytest.mark.parametrize("backend", sorted(GENERATORS))
@pytest.mark.parametrize("case", sorted(REFUSED))
def test_union_without_a_cpp_layout_is_refused(backend, case):
    structs, pattern = REFUSED[case]
    body = """
            Bits b;
            out_[i] = 0u;
"""
    _refused(backend, _program(structs, body), pattern)


CONSTRUCTORS = """
            Bits zero = Bits();
            Bits one = Bits(inp[i]);
            Bits copy = Bits(one);
            out_[i] = zero.u + one.u + copy.u;
"""


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_union_constructor_initializes_the_first_member(backend):
    code = _generate(backend, _program(BITS.format(abi="cuda"), CONSTRUCTORS))

    assert "Bits zero = Bits{};" in code
    assert "Bits one = Bits{static_cast<float>(inp[i])};" in code
    assert "Bits copy = one;" in code


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_union_constructor_with_two_arguments_is_refused(backend):
    body = """
            Bits b = Bits(inp[i], 1u);
            out_[i] = b.u;
"""
    _refused(
        backend,
        _program(BITS.format(abi="cuda"), body),
        r"initializes its first member 'f' from one argument",
    )


COMPILED_STRUCTS = BITS.format(abi="metal") + NARROW_MEMBERS + NESTED
COMPILED_BODY = PUN + CONSTRUCTORS.replace("out_[i] =", "out_[i] +=") + """
            Wide w;
            w.pair = vec2(inp[i], inp[i + 1u]);
            Outer o;
            o.inner.pair.a = inp[i];
            out_[i] += w.words[1] + uint(w.bytes[0]) + uint(w.low) + o.all.x;
"""


def _compile(command, tmp_path, name, code):
    source = tmp_path / name
    source.write_text(code)
    result = subprocess.run(
        [*command, str(source), "-o", str(tmp_path / "probe.o")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]


def test_nvcc_accepts_the_unions_and_their_layout_assertions(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")
    _compile(
        [nvcc, "-arch=sm_80", "-std=c++17", "-c"],
        tmp_path,
        "probe.cu",
        _generate("cuda", _program(COMPILED_STRUCTS, COMPILED_BODY)),
    )


def test_hipcc_accepts_the_unions_and_their_layout_assertions(tmp_path):
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")
    _compile(
        [hipcc, "--offload-arch=gfx1100", "-std=c++17", "-c"],
        tmp_path,
        "probe.hip",
        _generate("hip", _program(COMPILED_STRUCTS, COMPILED_BODY)),
    )
