"""Whole-value ``==`` and ``!=`` on structures and arrays (ADR 0002), written member by member where needed.

GLSL compares structures and arrays as a whole (GLSL 4.60 section 5.9) and CrossGL keeps that meaning. MSL, HLSL,
WGSL, SPIR-V, Slang, CUDA, HIP, Rust and Mojo have no such operator; the generators used to write ``p == q``
verbatim, which every reference compiler rejected, and CUDA and HIP compiled an array comparison as a pointer
comparison.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.aggregate_equality import (
    MAX_LEAF_COMPARISONS,
    AggregateEqualityOperandError,
    AggregateEqualityUnsupportedError,
    lower_aggregate_equality,
)


def program(declarations, body, helpers=""):
    return textwrap.dedent(f"""
        shader P {{
            {declarations}
            compute {{
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                buffer float* outp;
                {helpers}
                void main() {{
                    uint i = gl_GlobalInvocationID.x;
                    {body}
                }}
            }}
        }}
        """)


PAIR = "struct Pair { float a; float b; };"
PAIR_AND_ARRAY_BODY = """
    Pair p;
    p.a = outp[i];
    p.b = 2.0;
    Pair q = p;
    float t[2];
    t[0] = 1.0;
    t[1] = 3.0;
    float u[2] = t;
    float r = 0.0;
    if (p == q) {
        r += 1.0;
    }
    if (t != u) {
        r += 2.0;
    }
    outp[i] = r;
"""

NESTED = """
    const int N = 2;
    struct Inner {
        vec3 v;
        mat2x3 m;
        int c[N];
        bool f;
    };
    struct Pair {
        float a;
        Inner i[2];
    };
"""
NESTED_HELPERS = """
    bool same(Pair x, Pair y) {
        return x == y;
    }
"""
NESTED_BODY = """
    Pair p;
    p.a = outp[i];
    for (int k = 0; k < 2; k++) {
        p.i[k].v = vec3(1.0, 2.0, 3.0);
        p.i[k].m = mat2x3(1.0);
        p.i[k].c[0] = k;
        p.i[k].c[1] = 2;
        p.i[k].f = true;
    }
    Pair q = p;
    float r = 0.0;
    bool flag = outp[i] > 0.0;
    if (!(p == q)) {
        r += 1.0;
    }
    if ((p != q) && flag) {
        r += 2.0;
    }
    if (p.i[0] == q.i[1] || same(p, q)) {
        r += 4.0;
    }
    bool e = p.i[0].c == q.i[0].c;
    outp[i] = e ? r : -r;
"""

LOWERING_BACKENDS = ["metal", "directx", "wgsl", "slang", "cuda", "hip", "rust", "mojo"]


def translate(tmp_path, source, backend, name="eq"):
    path = tmp_path / f"{name}.cgl"
    path.write_text(source, "utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def run_tool(command, **kwargs):
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=300, **kwargs
    )
    diagnostics = "\n".join(
        part for part in (result.stdout, result.stderr) if part.strip()
    )
    if result.returncode != 0 and "missing Metal Toolchain" in diagnostics:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, diagnostics


def require_tool(name):
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name} is not installed")
    return path


def compact(code):
    """``code`` without whitespace."""
    return "".join(code.split())


def bare(code):
    """``code`` without whitespace or parentheses, so that grouping does not matter."""
    return compact(code).replace("(", "").replace(")", "")


@pytest.mark.parametrize("backend", LOWERING_BACKENDS)
def test_structure_and_array_equality_compare_member_by_member(tmp_path, backend):
    code = bare(translate(tmp_path, program(PAIR, PAIR_AND_ARRAY_BODY), backend))
    assert "p==q" not in code and "t!=u" not in code
    assert "p.a==q.a" in code and "p.b==q.b" in code
    assert "t[0]!=u[0]" in code and "t[1]!=u[1]" in code


def test_glsl_output_keeps_whole_value_equality(tmp_path):
    code = bare(translate(tmp_path, program(PAIR, PAIR_AND_ARRAY_BODY), "opengl"))
    assert "p==q" in code and "t!=u" in code
    assert "p.a==q.a" not in code


def test_spirv_compares_scalar_members():
    shader = parse(program(PAIR, PAIR_AND_ARRAY_BODY))
    assert lower_aggregate_equality(shader, "vulkan") == 2


def test_lowering_leaves_glsl_and_crossgl_alone():
    for target in ("opengl", "webgl", "cgl"):
        assert (
            lower_aggregate_equality(parse(program(PAIR, PAIR_AND_ARRAY_BODY)), target)
            == 0
        )


def test_nested_members_reduce_vectors_and_matrix_columns(tmp_path):
    code = compact(
        translate(tmp_path, program(NESTED, NESTED_BODY, NESTED_HELPERS), "metal")
    )
    for expected in (
        "all(x.i[0].v==y.i[0].v)",
        "all(p.i[1].m[1]==q.i[1].m[1])",
        "p.i[1].c[1]==q.i[1].c[1]",
        "p.i[1].f==q.i[1].f",
        "any(p.i[0].m[0]!=q.i[0].m[0])",
        "p.i[0].c[0]==q.i[0].c[0]&&p.i[0].c[1]==q.i[0].c[1]",
        "all(p.i[0].v==q.i[1].v)",
    ):
        assert expected in code, expected
    assert "x==y" not in code and "p==q" not in code and "p!=q" not in code
    # The negated comparison stays one operand of '!'.
    assert "!(p.a==q.a&&" in code


def test_comparison_order_follows_member_declaration_order():
    shader = parse(program(NESTED, NESTED_BODY, NESTED_HELPERS))
    lower_aggregate_equality(shader, "directx")
    helper = next(f for f in shader.stages.values()).local_functions[0]
    returned = helper.body.statements[0].value
    leaves = []

    def collect(node):
        if isinstance(node, A.BinaryOpNode) and node.op == "&&":
            collect(node.left)
            collect(node.right)
        else:
            leaves.append(node)

    collect(returned)
    # a, then per element: v, two matrix columns, two array elements, f.
    assert len(leaves) == 1 + 2 * (1 + 2 + 2 + 1)
    assert isinstance(leaves[0], A.BinaryOpNode)
    assert leaves[0].left.member == "a"


@pytest.mark.parametrize("backend", ["metal", "directx", "vulkan", "wgsl"])
def test_operand_with_side_effects_is_refused(tmp_path, backend):
    helpers = """
        Pair make(float a) {
            Pair p;
            p.a = a;
            p.b = a;
            return p;
        }
    """
    body = """
        Pair q;
        q.a = 1.0;
        q.b = 1.0;
        outp[i] = make(outp[i]) == q ? 1.0 : 0.0;
    """
    with pytest.raises(AggregateEqualityOperandError) as raised:
        translate(tmp_path, program(PAIR, body, helpers), backend)
    assert raised.value.code.endswith("aggregate-equality-impure-operand")


def test_comparison_beyond_the_leaf_limit_is_refused(tmp_path):
    size = MAX_LEAF_COMPARISONS + 1
    body = f"""
        float t[{size}];
        float u[{size}];
        outp[i] = t == u ? 1.0 : 0.0;
    """
    with pytest.raises(AggregateEqualityUnsupportedError) as raised:
        translate(tmp_path, program("", body), "metal")
    assert raised.value.code.endswith("aggregate-equality-unsupported")


def test_struct_constructor_operand_is_refused_rather_than_repeated(tmp_path):
    body = """
        Pair q;
        q.a = 1.0;
        q.b = 2.0;
        outp[i] = q != Pair(1.0, outp[i]++) ? 1.0 : 0.0;
    """
    with pytest.raises(AggregateEqualityOperandError):
        translate(tmp_path, program(PAIR, body), "directx")


# Reference compilers ------------------------------------------------------------------------------------------


def write(tmp_path, backend, suffix):
    path = tmp_path / f"nested.{suffix}"
    path.write_text(
        translate(tmp_path, program(NESTED, NESTED_BODY, NESTED_HELPERS), backend),
        "utf-8",
    )
    return path


def test_nested_comparisons_compile_with_xcrun_metal(tmp_path):
    xcrun = require_tool("xcrun")
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True, timeout=30
    )
    if probe.returncode != 0:
        pytest.skip("xcrun cannot locate metal")
    source = write(tmp_path, "metal", "metal")
    run_tool(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "n.air"),
        ]
    )


def test_nested_comparisons_compile_with_dxc(tmp_path):
    dxc = require_tool("dxc")
    source = write(tmp_path, "directx", "hlsl")
    run_tool([dxc, "-T", "cs_6_0", "-E", "CSMain", str(source)])


def test_nested_comparisons_validate_with_spirv_val(tmp_path):
    spirv_as = require_tool("spirv-as")
    spirv_val = require_tool("spirv-val")
    asm = write(tmp_path, "vulkan", "spvasm")
    binary = tmp_path / "nested.spv"
    run_tool([spirv_as, "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
    run_tool([spirv_val, "--target-env", "vulkan1.1", str(binary)])


def test_nested_comparisons_validate_with_naga(tmp_path):
    naga = require_tool("naga")
    run_tool([naga, str(write(tmp_path, "wgsl", "wgsl"))])


def test_nested_comparisons_compile_with_slangc(tmp_path):
    slangc = require_tool("slangc")
    source = write(tmp_path, "slang", "slang")
    run_tool(
        [
            slangc,
            str(source),
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            "compute",
            "-o",
            str(tmp_path / "nested.slang.spv"),
        ]
    )


def test_glsl_output_keeps_whole_value_comparisons_valid(tmp_path):
    glslang = require_tool("glslangValidator")
    source = write(tmp_path, "opengl", "comp")
    run_tool([glslang, "-S", "comp", str(source)])


def test_metal_array_parameters_in_an_address_space_compare_as_pointers(tmp_path):
    # ``thread float v[4]`` is a pointer in MSL, so ``left != right`` compares addresses and is not expanded.
    path = tmp_path / "decayed.metal"
    path.write_text(textwrap.dedent("""
        #include <metal_stdlib>
        using namespace metal;
        bool differ(thread float left[4], thread float right[4]) {
          return left != right;
        }
        kernel void probe(device uint* output [[buffer(0)]]) {
          float left[4] = {0};
          float right[4] = {1};
          output[0] = differ(left, right) ? 1u : 0u;
        }
        """))
    metal = crosstl.translate(str(path), backend="metal", format_output=False)
    assert "return left != right;" in metal
    assert "left[0] != right[0]" not in metal
