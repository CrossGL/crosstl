"""SPIR-V, WGSL and Slang output of a CrossGL union (``@union_layout``).

The CUDA and Metal importers write a C union as a structure marked ``@union_layout(size, alignment, byte_order,
abi)`` whose members carry ``@union_member_layout(offset, size, alignment)``. The three generators used to ignore the
marks and declare a structure with one member per view, so ``b.f = x; y = b.u;`` read a member that was never
written. They now keep one storage member and reinterpret its bits on each access
(:mod:`crosstl.translator.codegen.spirv_union_storage`), and refuse a view they cannot express exactly.
"""

import re
import shutil
import subprocess

import pytest

from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.codegen.spirv_union_storage import (
    UnionLayoutError,
    lower_union_storage,
)
from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

GENERATORS = {
    "wgsl": WGSLCodeGen,
    "vulkan": VulkanSPIRVCodeGen,
    "slang": SlangCodeGen,
}

BITS = """
@union_layout(4, 4, little_endian, cuda) struct Bits {
    @union_member_layout(0, 4, 4) float f;
    @union_member_layout(0, 4, 4) uint u;
    @union_member_layout(0, 4, 4) int i;
};
"""

PAIR = """
@union_layout(8, 8, little_endian, cuda) struct Pair {
    @union_member_layout(0, 8, 8) vec2 v;
    @union_member_layout(0, 8, 4) uint w[2];
    @union_member_layout(0, 8, 4) float a[2];
};
"""


def program(declarations, body, helpers="", resources=""):
    return f"""
shader UnionViews {{
    {declarations}
    {helpers}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* inp;
        buffer uint* outp;
        {resources}

        void main() {{
            uint idx = gl_GlobalInvocationID.x;
            {body}
        }}
    }}
}}
"""


PUNNING = program(BITS, "Bits b; b.f = inp[idx]; outp[idx] = b.u;")

VIEWS = program(
    BITS + PAIR + "struct Holder { Bits bits; float scale; };",
    """
            Bits b = make(inp[idx]);
            b.f *= 2.0;
            b.u += 1u;
            Holder h;
            h.bits = b;
            h.bits.i = h.bits.i - 2;
            Pair p = {vec2(inp[idx], 0.5)};
            p.w[1] = p.w[1] ^ 1u;
            p.a[0] = p.a[0] + p.v.y;
            Bits z = Bits();
            z.u++;
            outp[idx * 4u] = word(h.bits);
            outp[idx * 4u + 1u] = p.w[0];
            outp[idx * 4u + 2u] = floatBitsToUint(p.v.y);
            outp[idx * 4u + 3u] = z.u + uint(h.bits.i < 0);
    """,
    helpers="""
    Bits make(float value) {
        Bits b = Bits(value);
        return b;
    }

    uint word(Bits b) {
        return b.u;
    }
    """,
)


def parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def generate(backend, source):
    return GENERATORS[backend]().generate(parse(source))


def validate(backend, tmp_path, code):
    if backend == "wgsl":
        tool = shutil.which("naga")
        if tool is None:
            pytest.skip("naga is not installed")
        path = tmp_path / "shader.wgsl"
        path.write_text(code, encoding="utf-8")
        commands = [[tool, str(path)]]
    elif backend == "vulkan":
        if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
            pytest.skip("spirv-as and spirv-val are not installed")
        path = tmp_path / "shader.spvasm"
        path.write_text(code, encoding="utf-8")
        binary = tmp_path / "shader.spv"
        environment = ["--target-env", "vulkan1.1"]
        commands = [
            ["spirv-as", *environment, str(path), "-o", str(binary)],
            ["spirv-val", *environment, str(binary)],
        ]
    else:
        return
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_union_members_share_one_storage_word(backend, tmp_path):
    code = generate(backend, PUNNING)
    assert "CrossGLUnionStorage" in code
    assert ".f =" not in code and ".u;" not in code
    if backend == "vulkan":
        members = re.findall(r'OpMemberName %\w+ (\d+) "(\w+)"', code)
        assert ("0", "CrossGLUnionStorage") in members
        assert not {"f", "u", "i"} & {name for _, name in members}
        assert "OpBitcast" in code
    validate(backend, tmp_path, code)


def test_wgsl_reinterprets_each_view_of_the_storage(tmp_path):
    code = generate("wgsl", VIEWS)
    for expected in (
        "struct Bits {\n    CrossGLUnionStorage: u32,\n};",
        "struct Pair {\n    CrossGLUnionStorage: vec2<u32>,\n};",
        "var b: Bits = Bits(bitcast<u32>(value));",
        "return b.CrossGLUnionStorage;",
        # A compound write of a float view reads the bits, operates in float and stores the bits back.
        "b.CrossGLUnionStorage = bitcast<u32>((bitcast<f32>(b.CrossGLUnionStorage) * 2.0));",
        "b.CrossGLUnionStorage += 1u;",
        "h.bits.CrossGLUnionStorage = u32((i32(h.bits.CrossGLUnionStorage) - 2));",
        # A brace initializer initializes the first member, a vec2 over both words.
        "var p: Pair = Pair(bitcast<vec2<u32>>(vec2<f32>(inp[idx], 0.5)));",
        # An element of a word-array view is one component of the storage.
        "p.CrossGLUnionStorage[1] = (p.CrossGLUnionStorage[1] ^ 1u);",
        "bitcast<u32>((bitcast<f32>(p.CrossGLUnionStorage[0]) + bitcast<vec2<f32>>(p.CrossGLUnionStorage).y))",
        "var z: Bits = Bits(0u);",
    ):
        assert expected in code, expected
    validate("wgsl", tmp_path, code)


@pytest.mark.parametrize("backend", ["vulkan", "slang"])
def test_views_lower_for_spirv_and_slang(backend, tmp_path):
    code = generate(backend, VIEWS)
    if backend == "slang":
        assert "uint2 CrossGLUnionStorage;" in code
        assert "p.CrossGLUnionStorage[1] = p.CrossGLUnionStorage[1] ^ 1u;" in code
        assert (
            "b.CrossGLUnionStorage = asuint(asfloat(b.CrossGLUnionStorage) * 2.0);"
            in code
        )
    validate(backend, tmp_path, code)


RESOURCES = program(
    BITS + "cbuffer Params { Bits scale; };",
    "words[idx].f = inp[idx] * uintBitsToFloat(scale.u); outp[idx] = words[idx].u;",
    resources="buffer Bits* words;",
)


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_unions_in_buffers_and_uniform_blocks_keep_the_source_layout(backend, tmp_path):
    # The storage has the union's size and alignment, so a buffer element or block member is where the host put it.
    code = generate(backend, RESOURCES)
    if backend == "wgsl":
        assert "words[idx].CrossGLUnionStorage = bitcast<u32>(" in code
        assert "outp[idx] = words[idx].CrossGLUnionStorage;" in code
    if backend == "vulkan":
        assert re.search(r"OpMemberDecorate %\w+ 0 Offset 0", code)
    validate(backend, tmp_path, code)


def test_the_callers_program_keeps_its_union():
    ast = parse(PUNNING)
    WGSLCodeGen().generate(ast)
    (bits,) = [struct for struct in ast.structs if struct.name == "Bits"]
    assert [member.name for member in bits.members] == ["f", "u", "i"]


def test_a_program_without_a_union_is_returned_as_is():
    ast = parse(program("struct Plain { float f; };", "outp[idx] = 1u;"))
    assert lower_union_storage(ast, "wgsl") is ast


REFUSALS = {
    "member-view-unsupported": program(
        """
        @union_layout(4, 4, little_endian, cuda) struct Bytes {
            @union_member_layout(0, 4, 4) uint u;
            @union_member_layout(0, 4, 1) uint8 b[4];
        };
        """,
        "Bytes v; v.u = 1u; outp[idx] = uint(v.b[1]);",
    ),
    "whole-array-view-unsupported": program(
        PAIR, "Pair p; uint copy[2] = p.w; outp[idx] = copy[0];"
    ),
    "view-address-unsupported": program(
        BITS, "Bits b; b.u = 1u; uint* q = &b.u; outp[idx] = *q;"
    ),
    "interpreted-assignment-value-unsupported": program(
        BITS, "Bits b; float f = (b.f = 1.0); outp[idx] = b.u;"
    ),
    "interpreted-subcomponent-write-unsupported": program(
        PAIR, "Pair p; p.v.x = 1.0; outp[idx] = p.w[0];"
    ),
    "view-reference-argument-unsupported": program(
        BITS,
        "Bits b; bump(b.f); outp[idx] = b.u;",
        helpers="void bump(inout float value) { value = value + 1.0; }",
    ),
    "union-initializer-shape-unsupported": program(
        BITS, "Bits b = Bits(1.0, 2u); outp[idx] = b.u;"
    ),
}


STAGE_OUTPUT = """
shader UnionOutput {
    @union_layout(4, 4, little_endian, cuda) struct Bits {
        @union_member_layout(0, 4, 4) float f;
        @union_member_layout(0, 4, 4) uint u;
    };
    vertex {
        out Bits tag;
        void main() {
            tag.u = 1u;
            gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        }
    }
}
"""


@pytest.mark.parametrize("backend", sorted(GENERATORS))
@pytest.mark.parametrize(
    "reason, source",
    [*sorted(REFUSALS.items()), ("interface-union-layout-unsupported", STAGE_OUTPUT)],
)
def test_views_without_an_exact_lowering_are_refused(backend, reason, source):
    with pytest.raises(UnionLayoutError) as caught:
        generate(backend, source)
    error = caught.value
    assert error.reason == reason
    assert error.project_diagnostic_code == (
        f"project.translate.{backend}-union-layout-unsupported"
    )
    assert "cannot preserve union" in str(error)
