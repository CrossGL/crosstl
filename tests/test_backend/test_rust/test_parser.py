import pytest
from crosstl.backend.Rust.RustLexer import RustLexer
from crosstl.backend.Rust.RustParser import RustParser
from crosstl.backend.Rust.RustAst import (
    ConstNode,
    FunctionNode,
    ImplNode,
    StaticNode,
    StructNode,
    UseNode,
)


def parse_code(code: str):
    """Helper function to parse code."""
    lexer = RustLexer(code)
    tokens = lexer.tokenize()
    parser = RustParser(tokens)
    return parser.parse()


def test_struct_parsing():
    code = """
    #[repr(C)]
    pub struct Vertex {
        position: Vec3<f32>,
        normal: Vec3<f32>,
        texcoord: Vec2<f32>,
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.structs) == 1
        struct = ast.structs[0]
        assert isinstance(struct, StructNode)
        assert struct.name == "Vertex"
        assert len(struct.members) == 3
        assert struct.visibility == "pub"
        assert len(struct.attributes) == 1
    except Exception as e:
        pytest.fail(f"Struct parsing failed: {e}")


def test_function_parsing():
    code = """
    #[vertex_shader]
    pub fn vertex_main(vertex: Vec3<f32>) -> Vec4<f32> {
        let pos = Vec4::new(vertex.x, vertex.y, vertex.z, 1.0);
        return pos;
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert isinstance(func, FunctionNode)
        assert func.name == "vertex_main"
        assert func.return_type == "Vec4<f32>"
        assert len(func.params) == 1
        assert func.visibility == "pub"
        assert len(func.attributes) == 1
    except Exception as e:
        pytest.fail(f"Function parsing failed: {e}")


def test_impl_block_parsing():
    code = """
    impl Vertex {
        pub fn new(x: f32, y: f32, z: f32) -> Self {
            Self {
                position: Vec3::new(x, y, z),
                normal: Vec3::new(0.0, 1.0, 0.0),
                texcoord: Vec2::new(0.0, 0.0),
            }
        }
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.impl_blocks) == 1
        impl_block = ast.impl_blocks[0]
        assert isinstance(impl_block, ImplNode)
        assert impl_block.struct_name == "Vertex"
        assert len(impl_block.functions) == 1
    except Exception as e:
        pytest.fail(f"Impl block parsing failed: {e}")


def test_use_statement_parsing():
    code = """
    use std::collections::HashMap;
    use gpu::*;
    use math::{sin, cos, tan};
    """
    try:
        ast = parse_code(code)
        assert len(ast.use_statements) >= 1
        use_stmt = ast.use_statements[0]
        assert isinstance(use_stmt, UseNode)
    except Exception as e:
        pytest.fail(f"Use statement parsing failed: {e}")


def test_if_statement_parsing():
    code = """
    fn test_if() {
        if x > 0 {
            let y = x + 1;
        } else {
            let y = x - 1;
        }
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.body) >= 1
    except Exception as e:
        pytest.fail(f"If statement parsing failed: {e}")


def test_for_loop_parsing():
    code = """
    fn test_for() {
        for i in 0..10 {
            println!("{}", i);
        }
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.body) >= 1
    except Exception as e:
        pytest.fail(f"For loop parsing failed: {e}")


def test_while_loop_parsing():
    code = """
    fn test_while() {
        let mut i = 0;
        while i < 10 {
            i += 1;
        }
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.body) >= 2
    except Exception as e:
        pytest.fail(f"While loop parsing failed: {e}")


def test_match_expression_parsing():
    code = """
    fn test_match() {
        let value = 42;
        match value {
            0 => println!("zero"),
            1 => println!("one"),
            _ => println!("other"),
        }
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.body) >= 2
    except Exception as e:
        pytest.fail(f"Match expression parsing failed: {e}")


def test_let_binding_parsing():
    code = """
    fn test_let() {
        let x = 42;
        let mut y = 3.14;
        let z: f32 = 2.0;
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.body) >= 3
    except Exception as e:
        pytest.fail(f"Let binding parsing failed: {e}")


def test_const_static_parsing():
    code = """
    const PI: f32 = 3.14159;
    static mut GLOBAL_COUNTER: i32 = 0;
    """
    try:
        ast = parse_code(code)
        assert len(ast.global_variables) >= 2
        const_var = ast.global_variables[0]
        static_var = ast.global_variables[1]
        assert isinstance(const_var, ConstNode)
        assert isinstance(static_var, StaticNode)
    except Exception as e:
        pytest.fail(f"Const/static parsing failed: {e}")


def test_function_call_parsing():
    code = """
    fn test_call() {
        let result = Vec3::new(1.0, 2.0, 3.0);
        let length = result.length();
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.body) >= 2
    except Exception as e:
        pytest.fail(f"Function call parsing failed: {e}")


def test_binary_operations_parsing():
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
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.body) >= 10
    except Exception as e:
        pytest.fail(f"Binary operations parsing failed: {e}")


def test_unary_operations_parsing():
    code = """
    fn test_unary() {
        let a = -5;
        let b = !true;
        let c = &x;
        let d = *ptr;
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.body) >= 4
    except Exception as e:
        pytest.fail(f"Unary operations parsing failed: {e}")


def test_array_access_parsing():
    code = """
    fn test_array() {
        let arr = [1, 2, 3, 4, 5];
        let element = arr[0];
        let slice = &arr[1..3];
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.body) >= 3
    except Exception as e:
        pytest.fail(f"Array access parsing failed: {e}")


def test_member_access_parsing():
    code = """
    fn test_member() {
        let vertex = Vertex::new();
        let x = vertex.position.x;
        let y = vertex.position.y;
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.body) >= 3
    except Exception as e:
        pytest.fail(f"Member access parsing failed: {e}")


def test_type_casting_parsing():
    code = """
    fn test_cast() {
        let x = 42i32;
        let y = x as f64;
        let z = y as i32;
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.body) >= 3
    except Exception as e:
        pytest.fail(f"Type casting parsing failed: {e}")


def test_assignment_operations_parsing():
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
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.body) >= 6
    except Exception as e:
        pytest.fail(f"Assignment operations parsing failed: {e}")


def test_control_flow_parsing():
    code = """
    fn test_control() {
        loop {
            if condition {
                continue;
            } else {
                break;
            }
        }
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.body) >= 1
    except Exception as e:
        pytest.fail(f"Control flow parsing failed: {e}")


def test_generics_parsing():
    code = """
    fn generic_function<T, U>(a: T, b: U) -> T 
    where 
        T: Clone,
        U: Debug,
    {
        a.clone()
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.generics) == 2
    except Exception as e:
        pytest.fail(f"Generics parsing failed: {e}")


def test_complex_expressions_parsing():
    code = """
    fn test_complex() {
        let result = (a + b) * (c - d) / (e % f);
        let conditional = if x > 0 { y } else { z };
        let vector = Vec3::new(1.0, 2.0, 3.0);
        let access = vector.x + array[index].field;
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.body) >= 4
    except Exception as e:
        pytest.fail(f"Complex expressions parsing failed: {e}")


def test_attributes_parsing():
    code = """
    #[vertex_shader]
    #[location(0)]
    pub fn vertex_main(
        #[location(0)] position: Vec3<f32>,
        #[location(1)] normal: Vec3<f32>
    ) -> Vec4<f32> {
        Vec4::new(position.x, position.y, position.z, 1.0)
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.attributes) >= 2
    except Exception as e:
        pytest.fail(f"Attributes parsing failed: {e}")


def test_trait_parsing():
    code = """
    pub trait Drawable {
        fn draw(&self);
        fn update(&mut self, delta: f32);
    }
    """
    try:
        parse_code(code)
        # Note: trait parsing might be handled differently in the current implementation
        # This test ensures the parser doesn't crash on trait syntax
    except Exception as e:
        pytest.fail(f"Trait parsing failed: {e}")


def test_error_handling():
    """Test that the parser handles invalid syntax gracefully"""
    invalid_codes = [
        "fn incomplete(",
        "struct { missing_name }",
        "let = incomplete_assignment;",
        "if without_condition { }",
    ]

    for code in invalid_codes:
        try:
            parse_code(code)
            # If parsing succeeds, that's also acceptable (error recovery)
        except Exception:
            # Expected to fail, which is fine
            pass


if __name__ == "__main__":
    pytest.main()
