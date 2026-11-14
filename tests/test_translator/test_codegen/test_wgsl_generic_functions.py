"""WGSL output monomorphizes CrossGL generic functions.

WGSL has no user-declared generic functions (WGSL section 10.1 declares functions
with concrete parameter and return types only), so the generator emits one
concrete function per instantiation that the module calls.
"""

import re
import shutil
import subprocess

import pytest

from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def generate_wgsl(source):
    ast = Parser(Lexer(source).get_tokens()).parse()
    return WGSLCodeGen().generate(ast)


def validate_with_naga(tmp_path, code):
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    path = tmp_path / "shader.wgsl"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [naga, str(path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def compute_shader(helpers, body):
    return f"""
shader GenericProbe {{
    {helpers}
    compute {{
        layout(std430, binding = 0) buffer Data {{ float values[]; }} data;
        layout(local_size_x = 1) in;
        void main() {{
            {body}
        }}
    }}
}}
"""


def function_signatures(code):
    return [line for line in code.splitlines() if line.startswith("fn ")]


def assert_no_generic_spelling(code):
    for line in function_signatures(code):
        assert "<T" not in line.split("(")[0], line
        assert not re.search(r":\s*T\b|->\s*T\b", line), line


MIXED_GENERICS = compute_shader(
    """
    struct Pair { float a; int b; }
    T twice<T>(T x) { return x + x; }
    T quad<T>(T x) { return twice(twice<T>(x)); }
    T pick<T>(vec2<T> v) { return v.x; }
    float pick(float v) { return v * 3.0; }
    T zero<T>() { T z = T(0); return z; }
    vec2<T> splat2<T>(T v) { return vec2<T>(v, v); }
    T ident<T>(T v) { return v; }
    T add<T>(T a, T b) { return a + b; }
    """,
    """
            float a = twice<float>(1);
            float q = quad(2.0);
            int c = pick(ivec2(1, 2));
            uint u = pick(uvec2(3u, 4u));
            float p = pick(1.0);
            float pv = pick(vec2(1.0, 2.0));
            int z = zero<int>();
            vec2 s = splat2(1.5);
            Pair pr;
            pr.a = 1.0;
            Pair pr2 = ident(pr);
            vec3 v3 = twice(vec3(1.0));
            float sum = add(a, 1);
            data.values[0] = a + q + float(c) + float(u) + p + pv + float(z)
                + s.x + pr2.a + v3.y + sum;
    """,
)


def test_generic_functions_emit_one_concrete_function_per_instantiation(tmp_path):
    code = generate_wgsl(MIXED_GENERICS)

    signatures = function_signatures(code)
    assert signatures == [
        "fn pick(v: f32) -> f32 {",
        "fn twice_f32(x: f32) -> f32 {",
        "fn quad_f32(x: f32) -> f32 {",
        "fn pick_i32(v: vec2<i32>) -> i32 {",
        "fn pick_u32(v: vec2<u32>) -> u32 {",
        "fn pick_f32(v: vec2<f32>) -> f32 {",
        "fn zero_i32() -> i32 {",
        "fn splat2_f32(v: f32) -> vec2<f32> {",
        "fn ident_Pair(v: Pair) -> Pair {",
        "fn twice_vec3_f32(x: vec3<f32>) -> vec3<f32> {",
        "fn add_f32(a: f32, b: f32) -> f32 {",
        "fn compute_main() {",
    ]
    assert_no_generic_spelling(code)
    # Calls inside a specialization resolve to specializations too.
    assert "return twice_f32(twice_f32(x));" in code
    # Generic constructors and local types take the concrete type.
    assert "var z: i32 = i32(0);" in code
    assert "return vec2<f32>(v, v);" in code
    # Explicit type arguments convert the argument; a literal never decides T.
    assert "var a: f32 = twice_f32(f32(1));" in code
    assert "var sum: f32 = add_f32(a, f32(1));" in code
    # The exact concrete overload wins over the generic one.
    assert "var p: f32 = pick(1.0);" in code
    assert "var pv: f32 = pick_f32(vec2<f32>(1.0, 2.0));" in code
    validate_with_naga(tmp_path, code)


def test_generic_function_output_is_deterministic():
    assert generate_wgsl(MIXED_GENERICS) == generate_wgsl(MIXED_GENERICS)


def test_uncalled_generic_function_is_not_emitted(tmp_path):
    code = generate_wgsl(
        compute_shader(
            "T twice<T>(T x) { return x + x; }",
            "data.values[0] = 1.0;",
        )
    )

    assert function_signatures(code) == ["fn compute_main() {"]
    validate_with_naga(tmp_path, code)


def test_stage_local_generic_function_is_specialized(tmp_path):
    code = generate_wgsl("""
shader StageLocalGeneric {
    compute {
        layout(std430, binding = 0) buffer Data { float values[]; } data;
        layout(local_size_x = 1) in;
        T twice<T>(T x) { return x + x; }
        void main() { data.values[0] = twice(1.5); }
    }
}
""")

    assert "fn twice_f32(x: f32) -> f32 {" in code
    assert "data.values[0] = twice_f32(1.5);" in code
    assert_no_generic_spelling(code)
    validate_with_naga(tmp_path, code)


def test_literal_arguments_deduce_their_own_type(tmp_path):
    code = generate_wgsl(
        compute_shader(
            "T twice<T>(T x) { return x + x; }",
            """
            float a = twice(1.0);
            int b = twice(2);
            uint c = twice(3u);
            data.values[0] = a + float(b) + float(c);
            """,
        )
    )

    assert function_signatures(code)[:3] == [
        "fn twice_f32(x: f32) -> f32 {",
        "fn twice_i32(x: i32) -> i32 {",
        "fn twice_u32(x: u32) -> u32 {",
    ]
    validate_with_naga(tmp_path, code)


def test_multiple_and_defaulted_type_parameters(tmp_path):
    code = generate_wgsl(
        compute_shader(
            """
            U convert2<T, U>(T x, U y) { return U(x) + y; }
            T fromInt<T = float>(int x) { return T(x); }
            """,
            """
            float a = convert2(1u, 2.0);
            float b = fromInt(3);
            int c = fromInt<int>(4);
            data.values[0] = a + b + float(c);
            """,
        )
    )

    assert "fn convert2_u32_f32(x: u32, y: f32) -> f32 {" in code
    assert "fn fromInt_f32(x: i32) -> f32 {" in code
    assert "fn fromInt_i32(x: i32) -> i32 {" in code
    assert "return (f32(x) + y);" in code
    validate_with_naga(tmp_path, code)


def test_generic_function_shared_by_two_stages(tmp_path):
    code = generate_wgsl("""
shader SharedGeneric {
    T square<T>(T x) { return x * x; }
    vertex {
        vec4 main(vec3 position @ POSITION) @ gl_Position {
            return vec4(square(position), 1.0);
        }
    }
    fragment {
        vec4 main() @ gl_FragColor { return vec4(square(0.5)); }
    }
}
""")

    assert "fn square_vec3_f32(x: vec3<f32>) -> vec3<f32> {" in code
    assert "fn square_f32(x: f32) -> f32 {" in code
    assert code.count("fn square_f32(") == 1
    validate_with_naga(tmp_path, code)


@pytest.mark.parametrize(
    "helpers,body,message",
    [
        (
            "T add<T>(T a, T b) { return a + b; }",
            "vec3 v = add(vec3(1.0), ivec3(1));",
            "WGSL target cannot infer concrete type arguments for generic "
            "function 'add' (T) from call arguments (vec3<f32>, vec3<i32>)",
        ),
        (
            "T zero<T>() { return T(0); }",
            "float z = zero();",
            "WGSL target cannot infer concrete type arguments for generic "
            "function 'zero' (T) from call arguments ()",
        ),
        (
            "T first<T>(T x) { return x; } U first<U>(U y) { return y; }",
            "float f = first(1.0);",
            "WGSL target found an ambiguous call to generic function 'first' "
            "with arguments (f32)",
        ),
    ],
    ids=["vector-conflict", "no-deducible-argument", "ambiguous-overloads"],
)
def test_unspecializable_generic_calls_fail_closed(helpers, body, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        generate_wgsl(compute_shader(helpers, body))


def test_generic_function_with_resource_parameter_fails_closed():
    source = """
shader ResourceGeneric {
    vec4 sampleAt<T>(sampler2D tex, T uv) { return texture(tex, uv); }
    fragment {
        uniform sampler2D colorMap;
        vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
            return sampleAt(colorMap, uv);
        }
    }
}
"""
    with pytest.raises(
        ValueError,
        match="WGSL target cannot specialize generic function 'sampleAt' with "
        "texture, sampler or buffer parameters",
    ):
        generate_wgsl(source)
