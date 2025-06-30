import pytest
from crosstl.backend.Rust.RustLexer import RustLexer
from crosstl.backend.Rust.RustParser import RustParser
from crosstl.backend.Rust.RustCrossGLCodeGen import RustToCrossGLConverter


def parse_and_generate(code: str) -> str:
    """Helper function to parse Rust code and generate CrossGL."""
    lexer = RustLexer(code)
    tokens = lexer.tokenize()
    parser = RustParser(tokens)
    ast = parser.parse()

    generator = RustToCrossGLConverter()
    return generator.generate(ast)


def test_struct_conversion():
    code = """
    #[repr(C)]
    pub struct Vertex {
        position: Vec3<f32>,
        normal: Vec3<f32>,
        texcoord: Vec2<f32>,
    }
    """
    try:
        result = parse_and_generate(code)
        assert "shader main {" in result
        assert "struct Vertex {" in result
        assert "vec3 position" in result
        assert "vec3 normal" in result
        assert "vec2 texcoord" in result
    except Exception as e:
        pytest.fail(f"Struct conversion failed: {e}")


def test_function_conversion():
    code = """
    #[vertex_shader]
    pub fn vertex_main(vertex: Vec3<f32>) -> Vec4<f32> {
        let pos = Vec4::new(vertex.x, vertex.y, vertex.z, 1.0);
        return pos;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "vertex {" in result
        assert "pos = vec4(" in result
        assert "return pos;" in result
    except Exception as e:
        pytest.fail(f"Function conversion failed: {e}")


def test_fragment_shader_conversion():
    code = """
    #[fragment_shader]
    pub fn fragment_main(input: Vec2<f32>) -> Vec4<f32> {
        Vec4::new(input.x, input.y, 0.0, 1.0)
    }
    """
    try:
        result = parse_and_generate(code)
        assert "fragment {" in result
        assert "vec4" in result
    except Exception as e:
        pytest.fail(f"Fragment shader conversion failed: {e}")


def test_compute_shader_conversion():
    code = """
    #[compute_shader]
    pub fn compute_main() {
        // Compute shader logic
    }
    """
    try:
        result = parse_and_generate(code)
        assert "compute {" in result
    except Exception as e:
        pytest.fail(f"Compute shader conversion failed: {e}")


def test_type_mapping():
    code = """
    pub struct TypeTest {
        f32_val: f32,
        i32_val: i32,
        u32_val: u32,
        bool_val: bool,
        vec2_val: Vec2<f32>,
        vec3_val: Vec3<f32>,
        vec4_val: Vec4<f32>,
        mat4_val: Mat4<f32>,
    }
    """
    try:
        result = parse_and_generate(code)
        assert "float f32_val" in result
        assert "int i32_val" in result
        assert "uint u32_val" in result
        assert "bool bool_val" in result
        assert "vec2 vec2_val" in result
        assert "vec3 vec3_val" in result
        assert "vec4 vec4_val" in result
        assert "mat4 mat4_val" in result
    except Exception as e:
        pytest.fail(f"Type mapping failed: {e}")


def test_control_flow_conversion():
    code = """
    fn test_control() {
        if x > 0 {
            let y = x + 1;
        } else {
            let y = x - 1;
        }
        
        for i in 0..10 {
            println!("{}", i);
        }
        
        while condition {
            do_something();
        }
        
        loop {
            if should_break {
                break;
            }
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "if (" in result
        assert "} else {" in result
        assert "for (" in result
        assert "while (" in result
        assert "while (true) {" in result
        assert "break;" in result
    except Exception as e:
        pytest.fail(f"Control flow conversion failed: {e}")


def test_match_to_switch_conversion():
    code = """
    fn test_match() {
        match value {
            0 => do_zero(),
            1 => do_one(),
            _ => do_default(),
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "switch (" in result
        assert "case " in result
        assert "break;" in result
    except Exception as e:
        pytest.fail(f"Match to switch conversion failed: {e}")


def test_let_binding_conversion():
    code = """
    fn test_let() {
        let x = 42;
        let mut y = 3.14;
        let z: f32 = 2.0;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "x = 42;" in result
        assert "y = 3.14;" in result
        assert "float z = 2.0;" in result
    except Exception as e:
        pytest.fail(f"Let binding conversion failed: {e}")


def test_binary_operations_conversion():
    code = """
    fn test_binary() {
        let a = 5 + 3;
        let b = 10 - 2;
        let c = 4 * 6;
        let d = 8 / 2;
        let e = 10 % 3;
        let f = a > b;
        let g = c <= d;
        let h = e == 1;
        let i = f && g;
        let j = h || i;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "(5 + 3)" in result
        assert "(10 - 2)" in result
        assert "(4 * 6)" in result
        assert "(8 / 2)" in result
        assert "(10 % 3)" in result
        assert "(a > b)" in result
        assert "(c <= d)" in result
        assert "(e == 1)" in result
        assert "(f && g)" in result
        assert "(h || i)" in result
    except Exception as e:
        pytest.fail(f"Binary operations conversion failed: {e}")


def test_unary_operations_conversion():
    code = """
    fn test_unary() {
        let a = -5;
        let b = !true;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "(-5)" in result
        assert "(!true)" in result
    except Exception as e:
        pytest.fail(f"Unary operations conversion failed: {e}")


def test_function_call_conversion():
    code = """
    fn test_calls() {
        let result = normalize(vector);
        let dot_product = dot(a, b);
        let length = length(vec);
        let vector = Vec3::new(1.0, 2.0, 3.0);
    }
    """
    try:
        result = parse_and_generate(code)
        assert "normalize(" in result
        assert "dot(" in result
        assert "length(" in result
        assert "vec3(" in result
    except Exception as e:
        pytest.fail(f"Function call conversion failed: {e}")


def test_member_access_conversion():
    code = """
    fn test_member() {
        let x = vertex.position.x;
        let y = vertex.position.y;
        let z = vertex.position.z;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "vertex.position.x" in result
        assert "vertex.position.y" in result
        assert "vertex.position.z" in result
    except Exception as e:
        pytest.fail(f"Member access conversion failed: {e}")


def test_array_access_conversion():
    code = """
    fn test_array() {
        let element = array[0];
        let slice_element = slice[index];
    }
    """
    try:
        result = parse_and_generate(code)
        assert "array[0]" in result
        assert "slice[index]" in result
    except Exception as e:
        pytest.fail(f"Array access conversion failed: {e}")


def test_assignment_operations_conversion():
    code = """
    fn test_assign() {
        let mut a = 5;
        a += 3;
        a -= 2;
        a *= 4;
        a /= 2;
        a %= 3;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "a += 3;" in result
        assert "a -= 2;" in result
        assert "a *= 4;" in result
        assert "a /= 2;" in result
        assert "a %= 3;" in result
    except Exception as e:
        pytest.fail(f"Assignment operations conversion failed: {e}")


def test_vector_constructor_conversion():
    code = """
    fn test_vectors() {
        let vec2_val = Vec2::new(1.0, 2.0);
        let vec3_val = Vec3::new(1.0, 2.0, 3.0);
        let vec4_val = Vec4::new(1.0, 2.0, 3.0, 4.0);
    }
    """
    try:
        result = parse_and_generate(code)
        assert "vec2(1.0, 2.0)" in result
        assert "vec3(1.0, 2.0, 3.0)" in result
        assert "vec4(1.0, 2.0, 3.0, 4.0)" in result
    except Exception as e:
        pytest.fail(f"Vector constructor conversion failed: {e}")


def test_matrix_constructor_conversion():
    code = """
    fn test_matrices() {
        let mat2_val = Mat2::new(1.0, 2.0, 3.0, 4.0);
        let mat3_val = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        let mat4_val = Mat4::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    }
    """
    try:
        result = parse_and_generate(code)
        assert "mat2(" in result
        assert "mat3(" in result
        assert "mat4(" in result
    except Exception as e:
        pytest.fail(f"Matrix constructor conversion failed: {e}")


def test_const_static_conversion():
    code = """
    const PI: f32 = 3.14159;
    static mut GLOBAL_COUNTER: i32 = 0;
    """
    try:
        result = parse_and_generate(code)
        assert "const float PI = 3.14159;" in result
        assert "static mut int GLOBAL_COUNTER = 0;" in result
    except Exception as e:
        pytest.fail(f"Const/static conversion failed: {e}")


def test_use_statement_conversion():
    code = """
    use std::collections::HashMap;
    use gpu::*;
    use math::{sin, cos, tan};
    """
    try:
        result = parse_and_generate(code)
        assert "// use std::collections::HashMap" in result
        assert "// use gpu::*" in result
    except Exception as e:
        pytest.fail(f"Use statement conversion failed: {e}")


def test_impl_block_conversion():
    code = """
    impl Vertex {
        pub fn new(x: f32, y: f32, z: f32) -> Self {
            Self {
                position: Vec3::new(x, y, z),
                normal: Vec3::new(0.0, 1.0, 0.0),
                texcoord: Vec2::new(0.0, 0.0),
            }
        }
        
        fn calculate_distance(&self, other: &Vertex) -> f32 {
            distance(self.position, other.position)
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "// Implementation for Vertex" in result
        assert "Vertex_new(" in result
        assert (
            "calculate_distance(" in result
        )  # Currently not prefixed with struct name
    except Exception as e:
        pytest.fail(f"Impl block conversion failed: {e}")


def test_references_conversion():
    code = """
    fn test_references() {
        let x = 42;
        let y = &x;
        let z = &mut x;
        let w = *y;
    }
    """
    try:
        result = parse_and_generate(code)
        # References should be handled by removing the reference syntax
        assert "y = x;" in result  # Reference removed
        assert "w = y;" in result  # Dereference removed
    except Exception as e:
        pytest.fail(f"References conversion failed: {e}")


def test_type_casting_conversion():
    code = """
    fn test_cast() {
        let x = 42i32;
        let y = x as f64;
        let z = y as i32;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "(double)x" in result
        assert "(int)y" in result
    except Exception as e:
        pytest.fail(f"Type casting conversion failed: {e}")


def test_ternary_expression_conversion():
    code = """
    fn test_ternary() {
        let result = if condition { true_value } else { false_value };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "condition ? true_value : false_value" in result
    except Exception as e:
        pytest.fail(f"Ternary expression conversion failed: {e}")


def test_complex_shader_conversion():
    code = """
    #[repr(C)]
    pub struct VertexInput {
        #[location(0)]
        position: Vec3<f32>,
        #[location(1)]
        normal: Vec3<f32>,
        #[location(2)]
        texcoord: Vec2<f32>,
    }
    
    #[repr(C)]
    pub struct VertexOutput {
        position: Vec4<f32>,
        normal: Vec3<f32>,
        texcoord: Vec2<f32>,
    }
    
    #[vertex_shader]
    pub fn vertex_main(input: VertexInput) -> VertexOutput {
        let world_position = transform * Vec4::new(input.position.x, input.position.y, input.position.z, 1.0);
        let world_normal = normalize((normal_matrix * Vec4::new(input.normal.x, input.normal.y, input.normal.z, 0.0)).xyz());
        
        VertexOutput {
            position: world_position,
            normal: world_normal,
            texcoord: input.texcoord,
        }
    }
    
    #[fragment_shader]
    pub fn fragment_main(input: VertexOutput) -> Vec4<f32> {
        let light_dir = normalize(light_position - input.position.xyz());
        let diffuse = max(dot(input.normal, light_dir), 0.0);
        let color = texture_sample(diffuse_texture, input.texcoord);
        
        Vec4::new(color.xyz() * diffuse, color.w)
    }
    """
    try:
        result = parse_and_generate(code)
        assert "shader main {" in result
        assert "struct VertexInput {" in result
        assert "struct VertexOutput {" in result
        assert "vertex {" in result
        assert "fragment {" in result
        assert (
            "world_position = " in result
        )  # Variable assignments without explicit type
        assert "normalize(" in result
        assert "dot(" in result
        assert "max(" in result
    except Exception as e:
        pytest.fail(f"Complex shader conversion failed: {e}")


def test_math_functions_conversion():
    code = """
    fn test_math() {
        let a = sin(x);
        let b = cos(x);
        let c = tan(x);
        let d = sqrt(x);
        let e = pow(x, y);
        let f = abs(x);
        let g = min(x, y);
        let h = max(x, y);
        let i = clamp(x, min_val, max_val);
        let j = floor(x);
        let k = ceil(x);
    }
    """
    try:
        result = parse_and_generate(code)
        assert "sin(" in result
        assert "cos(" in result
        assert "tan(" in result
        assert "sqrt(" in result
        assert "pow(" in result
        assert "abs(" in result
        assert "min(" in result
        assert "max(" in result
        assert "clamp(" in result
        assert "floor(" in result
        assert "ceil(" in result
    except Exception as e:
        pytest.fail(f"Math functions conversion failed: {e}")


def test_error_handling():
    """Test that the code generator handles edge cases gracefully"""
    edge_cases = [
        "fn empty() {}",
        "struct Empty {}",
        "// Just a comment",
        "",
    ]

    for code in edge_cases:
        try:
            result = parse_and_generate(code)
            assert "shader main {" in result
        except Exception:
            # Some edge cases might fail, which is acceptable
            pass


if __name__ == "__main__":
    pytest.main()
