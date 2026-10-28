"""The comma operator, ``(a, b)``, from every C-family source to every target.

CrossGL's ``(a, b)`` evaluates ``a``, discards its value and yields ``b`` (C11 6.5.17, GLSL 4.60 5.9).
C-family targets print it with parentheses; SPIR-V evaluates the operands in order; WGSL, which has no
comma operator, hoists the discarded operands into statements; Rust and Mojo refuse it.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.codegen.lowering_refusals import LoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

PROBE = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int a = iin[i];
            int b = (a += 3, a * 2);
            int c = 0;
            float r = b > 20 ? (c = 1, 2.0) : (c = 3, 4.0);
            int d = (c++, c + a);
            outp[i] = float(b) + r * 64.0 + float(c) * 512.0 + float(d) * 4096.0;
        }
    }
}
"""

# The same probe without a comma in a conditional position, which WGSL can lower.
UNCONDITIONAL_PROBE = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int a = iin[i];
            int b = (a += 3, a * 2);
            int c = 0;
            int d = (c++, c + a);
            (c += 2, d -= 1);
            outp[i] = float(b) + float(c) * 512.0 + float(d) * 4096.0;
        }
    }
}
"""

C_FAMILY_SOURCE = """
    int a = iin[i];
    int b = (a += 3, a * 2);
    int c = 0;
    float r = b > 20 ? (c = 1, 2.0f) : (c = 3, 4.0f);
    int d = (c++, c + a);
    outp[i] = (float)(b) + r * 64.0f + (float)(c) * 512.0f + (float)(d) * 4096.0f;
"""

NATIVE_SOURCES = {
    "hlsl": """
        RWStructuredBuffer<int> iin : register(u0);
        RWStructuredBuffer<float> outp : register(u1);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
            uint i = id.x;
        """ + C_FAMILY_SOURCE + "}\n",
    "metal": """
        #include <metal_stdlib>
        using namespace metal;
        kernel void probe(device int* iin [[buffer(0)]],
                          device float* outp [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
        """ + C_FAMILY_SOURCE + "}\n",
    "cu": (
        'extern "C" __global__ void probe(int* iin, float* outp) {\n'
        "    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        + C_FAMILY_SOURCE
        + "}\n"
    ),
    "hip": (
        "#include <hip/hip_runtime.h>\n"
        'extern "C" __global__ void probe(int* iin, float* outp) {\n'
        "    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        + C_FAMILY_SOURCE
        + "}\n"
    ),
    "cl": (
        "__kernel void probe(__global int* iin, __global float* outp) {\n"
        "    uint i = get_global_id(0);\n" + C_FAMILY_SOURCE + "}\n"
    ),
    "slang": """
        RWStructuredBuffer<int> iin;
        RWStructuredBuffer<float> outp;
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
            uint i = id.x;
        """ + C_FAMILY_SOURCE + "}\n",
    "comp": (
        """
        #version 450
        layout(local_size_x = 1) in;
        layout(std430, binding = 0) buffer In { int iin[]; };
        layout(std430, binding = 1) buffer Out { float outp[]; };
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int a = iin[i];
            int b = (a += 3, a * 2);
            int c = 0;
            float r = b > 20 ? (c = 1, 2.0) : (c = 3, 4.0);
            int d = (c++, c + a);
            outp[i] = float(b) + r * 64.0 + float(c) * 512.0 + float(d) * 4096.0;
        }
        """
    ),
}


def parse_crossgl(source):
    return Parser(Lexer(source).get_tokens()).parse()


def write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def translate(tmp_path, backend, source=PROBE, name="probe.cgl"):
    return crosstl.translate(str(write(tmp_path, name, source)), backend=backend)


def comma_nodes(ast):
    return [node for node in ast.walk() if isinstance(node, A.CommaExpressionNode)]


def require_tool(name):
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name} is not installed")
    return path


def run_tool(command, **kwargs):
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=180, **kwargs
    )
    diagnostics = "\n".join(
        part for part in (result.stdout, result.stderr) if part.strip()
    )
    if result.returncode != 0 and "missing Metal Toolchain" in diagnostics:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, diagnostics


def test_parenthesized_comma_parses_as_comma_expression():
    ast = parse_crossgl(PROBE)
    commas = comma_nodes(ast)
    assert len(commas) == 4
    assert all(len(node.expressions) == 2 for node in commas)
    assert isinstance(commas[0].expressions[0], A.AssignmentNode)


def test_trailing_comma_keeps_the_tuple_literal():
    ast = parse_crossgl("shader S { compute { void main() { int x[2] = (1, 2,); } } }")
    assert not comma_nodes(ast)
    assert any(isinstance(node, A.ArrayLiteralNode) for node in ast.walk())


def test_hlsl_splits_a_comma_initializer_into_statements(tmp_path):
    # DXC warns on ``int b = (a, b);`` (-Wcomma-in-init), so the discarded operand runs first.
    generated = translate(tmp_path, "directx")
    body = generated[generated.index("void CSMain") :]
    order = ["a += 3;", "int b = (a * 2);", "int c = 0;", "c++;", "int d = (c + a);"]
    positions = [body.index(statement) for statement in order]
    assert positions == sorted(positions)
    assert "? ((c = 1), 2.0) : ((c = 3), 4.0)" in body


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("opengl", "int b = ((a += 3), (a * 2));"),
        ("metal", "int b = ((a += 3), a * 2);"),
        ("slang", "int b = ((a += 3), a * 2);"),
        ("cuda", "int b = ((a += 3), (a * 2));"),
        ("hip", "int b = ((a += 3), (a * 2));"),
    ],
)
def test_c_family_targets_print_the_operator(tmp_path, backend, expected):
    generated = translate(tmp_path, backend)
    assert expected in generated
    assert "CommaExpressionNode" not in generated
    # The ternary arms keep their own parentheses, so the comma cannot split them.
    assert "? ((c = 1), 2.0" in generated


def test_hlsl_output_compiles_with_dxc(tmp_path):
    dxc = require_tool("dxc")
    shader = write(tmp_path, "probe.hlsl", translate(tmp_path, "directx"))
    run_tool([dxc, "-T", "cs_6_0", "-E", "CSMain", "-WX", str(shader)], cwd=tmp_path)


def test_glsl_output_compiles_with_glslang(tmp_path):
    glslang = require_tool("glslangValidator")
    shader = write(tmp_path, "probe.comp", translate(tmp_path, "opengl"))
    run_tool(
        [glslang, "--target-env", "opengl", str(shader), "-o", str(tmp_path / "p.spv")]
    )


def test_metal_output_compiles_with_xcrun_metal(tmp_path):
    xcrun = require_tool("xcrun")
    shader = write(tmp_path, "probe.metal", translate(tmp_path, "metal"))
    run_tool(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(shader)],
        cwd=tmp_path,
    )


def test_spirv_evaluates_operands_in_order_and_branches_on_effectful_arms(tmp_path):
    generated = translate(tmp_path, "vulkan")
    assert "OpSelectionMerge" in generated
    assert "CommaExpressionNode" not in generated
    spirv_as = require_tool("spirv-as")
    spirv_val = require_tool("spirv-val")
    source = write(tmp_path, "probe.spvasm", generated)
    binary = tmp_path / "probe.spv"
    run_tool([spirv_as, "--target-env", "vulkan1.1", str(source), "-o", str(binary)])
    run_tool([spirv_val, "--target-env", "vulkan1.1", str(binary)])


def test_wgsl_hoists_discarded_operands_into_statements(tmp_path):
    generated = translate(tmp_path, "wgsl", UNCONDITIONAL_PROBE)
    body = generated[generated.index("fn compute_main") :]
    order = [
        "a += 3;",
        "var b: i32 = (a * 2);",
        "var c: i32 = 0;",
        "var d: i32 = (c + a);",
        "c += 2;",
        "d -= 1;",
    ]
    positions = [body.index(statement) for statement in order]
    assert positions == sorted(positions)
    naga = shutil.which("naga")
    if naga is not None:
        run_tool([naga, str(write(tmp_path, "probe.wgsl", generated))])


def test_wgsl_refuses_an_effectful_comma_in_a_ternary_arm(tmp_path):
    with pytest.raises(ValueError, match="comma expression whose discarded operand"):
        translate(tmp_path, "wgsl")


def test_wgsl_drops_a_pure_discarded_operand_in_a_ternary_arm(tmp_path):
    source = PROBE.replace("(c = 1, 2.0)", "(c, 2.0)").replace(
        "(c = 3, 4.0)", "(c, 4.0)"
    )
    generated = translate(tmp_path, "wgsl", source)
    assert "var r: f32 = select(4.0, 2.0, (b > 20));" in generated


@pytest.mark.parametrize(
    ("backend", "code"),
    [("rust", "rust-lowering-unsupported"), ("mojo", "mojo-lowering-unsupported")],
)
def test_targets_without_a_lowering_refuse_with_a_typed_diagnostic(
    tmp_path, backend, code
):
    with pytest.raises(LoweringRefused, match="comma expression") as raised:
        translate(tmp_path, backend)
    assert raised.value.code.endswith(code)


@pytest.mark.parametrize("extension", sorted(NATIVE_SOURCES))
def test_importers_emit_the_crossgl_comma_operator(tmp_path, extension):
    crossgl = crosstl.translate(
        str(write(tmp_path, f"probe.{extension}", NATIVE_SOURCES[extension])),
        backend="cgl",
    )
    assert "comma expression" not in crossgl.lower()
    assert "{a += 3" not in crossgl
    commas = comma_nodes(parse_crossgl(crossgl))
    assert len(commas) == 4, crossgl
    first = commas[0].expressions
    assert isinstance(first[0], A.AssignmentNode) and first[0].operator == "+="


# A comma's value has its last operand's type, so conversions of that value follow the operand.
TYPED_RESULT_PROBE = """
shader Probe {
    struct FragmentInput {
        vec2 uv;
    }
    struct FragmentOutput {
        vec4 color;
    }
    fragment {
        uniform uint count;
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            int c = 0;
            float f = (c++, count);
            float g = (c += 2, count) * 1.5;
            vec2 w = (c += 2, input.uv * 2.0);
            output.color = vec4((c++, w), g, float(c) + f);
            return output;
        }
    }
}
"""


def test_hlsl_converts_an_unsigned_comma_value_with_the_rounding_helper(tmp_path):
    generated = translate(tmp_path, "directx", TYPED_RESULT_PROBE)
    assert "(__crossgl_uint_to_float1(((c += 2), count)) * 1.5)" in generated
    dxc = shutil.which("dxc")
    if dxc is not None:
        shader = write(tmp_path, "typed.hlsl", generated)
        run_tool([dxc, "-T", "ps_6_0", "-E", "PSMain", str(shader)], cwd=tmp_path)


@pytest.mark.parametrize("backend", ["opengl", "webgl"])
def test_glsl_converts_an_unsigned_comma_value_explicitly(tmp_path, backend):
    # GLSL ES has no implicit uint-to-float conversion (GLSL ES 3.00 section 4.1.10).
    generated = translate(tmp_path, backend, TYPED_RESULT_PROBE)
    assert "float(((c += 2), count)) * 1.5" in generated
    assert "float f = float(((c++), count));" in generated
    glslang = shutil.which("glslangValidator")
    if glslang is not None:
        run_tool([glslang, "-S", "frag", str(write(tmp_path, "typed.frag", generated))])


def test_hlsl_parenthesized_values_round_trip_as_comma_operator(tmp_path):
    # DXC evaluates ``(1, 0, 0, 1)`` to ``1`` and splats it; it is not ``float4(1, 0, 0, 1)``.
    source = write(
        tmp_path,
        "values.hlsl",
        """
        RWStructuredBuffer<float4> o : register(u0);
        [numthreads(1, 1, 1)]
        void main() {
            float4 c = 0;
            int k = 0;
            c = (k += 2, k);
            o[0] = c;
        }
        """,
    )
    crossgl = crosstl.translate(str(source), backend="cgl")
    assert "c = (k += 2, k);" in crossgl
    assert "{k += 2" not in crossgl
    hlsl = crosstl.translate(str(source), backend="directx")
    # ``k += 2`` keeps the emitter's wrapping signed arithmetic.
    assert "c = ((k = int((uint(k) + 2u))), k);" in hlsl


def test_hlsl_return_and_statement_commas_stay_one_expression(tmp_path):
    source = write(
        tmp_path,
        "statements.hlsl",
        """
        RWStructuredBuffer<int> o : register(u0);
        int f(int x) { int y = x; return y++, y * 3; }
        [numthreads(1, 1, 1)]
        void main() {
            int a = 1, b = 2;
            a++, b += 2;
            o[0] = f(a) + b;
        }
        """,
    )
    crossgl = crosstl.translate(str(source), backend="cgl")
    assert "return (y++, y * 3);" in crossgl
    assert "(a++, b += 2);" in crossgl
    assert len(comma_nodes(parse_crossgl(crossgl))) == 2


def test_cuda_parenthesized_identifier_before_plus_is_not_a_cast(tmp_path):
    source = write(
        tmp_path,
        "cast.cu",
        """
        __global__ void k(float* o, int b, float r) {
            o[0] = (float)(b) + r;
            o[1] = (float)(b) - r;
        }
        """,
    )
    crossgl = crosstl.translate(str(source), backend="cgl")
    assert "o[0] = (f32(b) + r);" in crossgl
    assert "o[1] = (f32(b) - r);" in crossgl


# Narrow postfix updates lower to an update followed by the old value; a comma that discards them
# must not leave that value unused.
DISCARDED_NARROW_UPDATES = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int16_t a = int16_t(1);
            int16_t b = int16_t(2);
            (a++, b--);
            int c = (a++, b);
            for (int k = 0; k < 2; (k++, a--)) { }
            outp[i] = int(a) + int(b) + c;
        }
    }
}
"""


def test_for_update_comma_is_the_update_list():
    ast = parse_crossgl(DISCARDED_NARROW_UPDATES)
    loops = [node for node in ast.walk() if isinstance(node, A.ForNode)]
    assert len(loops) == 1
    update = loops[0].update
    assert isinstance(update, list) and len(update) == 2
    assert not any(isinstance(item, A.CommaExpressionNode) for item in update)


def test_metal_discarded_comma_operands_compile_with_werror(tmp_path):
    generated = translate(tmp_path, "metal", DISCARDED_NARROW_UPDATES)
    assert "(void)(" in generated
    xcrun = require_tool("xcrun")
    shader = write(tmp_path, "probe.metal", generated)
    result = subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-Werror", "-c", str(shader)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    diagnostics = result.stdout + result.stderr
    if result.returncode != 0 and "missing Metal Toolchain" in diagnostics:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, diagnostics


def test_glsl_discarded_comma_operands_drop_the_old_value(tmp_path):
    generated = translate(tmp_path, "opengl", DISCARDED_NARROW_UPDATES)
    body = generated[generated.index("void main") :]
    assert "((a = bitfieldExtract((a + 1), 0, 16))," in body
    # A discarded postfix update keeps no reconstruction of its old value.
    assert "(a - 1)" not in body.split("int c")[0]


@pytest.mark.parametrize(
    ("extension", "source", "expected"),
    [
        (
            ".mojo",
            "fn pair(a: Int, b: Int) -> Int:\n    var t = (a, b)\n    return t[1]\n",
            "(a, b,)",
        ),
        (
            ".rs",
            "fn pair(a: i32, b: i32) -> i32 {\n    let t = (a, b);\n    t.1\n}\n",
            "(a, b,)",
        ),
    ],
)
def test_tuple_importers_keep_the_tuple_literal(tmp_path, extension, source, expected):
    # Without the trailing comma, CrossGL would read the tuple as the comma operator.
    path = write(tmp_path, f"source{extension}", source)
    generated = crosstl.translate(str(path), backend="cgl")
    assert expected in generated
    parse_crossgl(generated)
