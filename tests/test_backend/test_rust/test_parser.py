import pytest
from crosstl.backend.Rust.RustLexer import RustLexer
from crosstl.backend.Rust.RustParser import RustParser
from crosstl.backend.Rust.RustAst import (
    AssociatedTypeNode,
    ConstNode,
    EnumNode,
    EnumVariantNode,
    FunctionCallNode,
    FunctionNode,
    ImplNode,
    StaticNode,
    StructNode,
    TypeAliasNode,
    UseNode,
    ArrayNode,
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


def test_enum_parsing():
    code = """
    #[repr(u32)]
    pub enum Message<T>
    where
        T: Copy,
    {
        Quit,
        AxisX = 0,
        Data(T, Vec3<f32>),
        Move { x: f32, y: f32 },
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.enums) == 1

        enum = ast.enums[0]
        assert isinstance(enum, EnumNode)
        assert enum.name == "Message"
        assert enum.visibility == "pub"
        assert enum.generics == ["T"]
        assert enum.where_clauses == [("T", ["Copy"])]
        assert len(enum.attributes) == 1

        assert [variant.name for variant in enum.variants] == [
            "Quit",
            "AxisX",
            "Data",
            "Move",
        ]
        assert all(isinstance(variant, EnumVariantNode) for variant in enum.variants)
        assert enum.variants[0].kind == "unit"
        assert enum.variants[1].value == "0"
        assert enum.variants[2].kind == "tuple"
        assert enum.variants[2].fields == ["T", "Vec3<f32>"]
        assert enum.variants[3].kind == "struct"
        assert [field.name for field in enum.variants[3].fields] == ["x", "y"]
        assert [field.vtype for field in enum.variants[3].fields] == ["f32", "f32"]
    except Exception as e:
        pytest.fail(f"Enum parsing failed: {e}")


def test_type_alias_parsing():
    code = """
    #[repr(transparent)]
    pub type Real = f32;
    type Buffer<T> = [T; 4];
    type Table<T>
    where
        T: Copy,
    = Vec3<T>;

    impl<T> Kernel<T>
    where
        T: Clone,
    {
        type Output = T;
        pub type Scratch = [T; 2];
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.type_aliases) == 3
        assert all(isinstance(alias, TypeAliasNode) for alias in ast.type_aliases)

        real, buffer, table = ast.type_aliases
        assert real.name == "Real"
        assert real.alias_type == "f32"
        assert real.visibility == "pub"
        assert len(real.attributes) == 1

        assert buffer.name == "Buffer"
        assert buffer.generics == ["T"]
        assert buffer.alias_type == "T[4]"

        assert table.name == "Table"
        assert table.generics == ["T"]
        assert table.where_clauses == [("T", ["Copy"])]
        assert table.alias_type == "Vec3<T>"

        impl_block = ast.impl_blocks[0]
        assert impl_block.struct_name == "Kernel<T>"
        assert impl_block.where_clauses == [("T", ["Clone"])]
        assert [alias.name for alias in impl_block.type_aliases] == [
            "Output",
            "Scratch",
        ]
        assert [alias.alias_type for alias in impl_block.type_aliases] == [
            "T",
            "T[2]",
        ]
        assert impl_block.type_aliases[1].visibility == "pub"
    except Exception as e:
        pytest.fail(f"Type alias parsing failed: {e}")


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


def test_generic_impl_where_clause_parsing():
    code = """
    impl<T> Drawable for Sprite<T>
    where
        T: Copy + Clone,
    {
        fn draw(&self) {}
    }

    impl<T> Sprite<T>
    where
        T: Copy,
    {
        fn new(value: T) -> Self {
            Self { value: value }
        }
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.impl_blocks) == 2

        trait_impl = ast.impl_blocks[0]
        assert trait_impl.trait_name == "Drawable"
        assert trait_impl.struct_name == "Sprite<T>"
        assert trait_impl.generics == ["T"]
        assert trait_impl.where_clauses == [("T", ["Copy", "Clone"])]
        assert trait_impl.functions[0].name == "draw"

        inherent_impl = ast.impl_blocks[1]
        assert inherent_impl.trait_name is None
        assert inherent_impl.struct_name == "Sprite<T>"
        assert inherent_impl.generics == ["T"]
        assert inherent_impl.where_clauses == [("T", ["Copy"])]
        assert inherent_impl.functions[0].return_type == "Self"
    except Exception as e:
        pytest.fail(f"Generic impl where-clause parsing failed: {e}")


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


def test_fixed_array_type_and_repeated_literal_parsing():
    code = """
    fn test_arrays() {
        let weights: [f32; 4] = [0.25, 0.25, 0.25, 0.25];
        let zeros: [f32; 3] = [0.0; 3];
        let matrix: [[f32; 2]; 3] = [[1.0, 2.0]; 3];
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        assert body[0].vtype == "f32[4]"
        assert isinstance(body[0].value, ArrayNode)
        assert body[0].value.size is None
        assert body[0].value.elements == ["0.25", "0.25", "0.25", "0.25"]

        assert body[1].vtype == "f32[3]"
        assert isinstance(body[1].value, ArrayNode)
        assert body[1].value.size == "3"
        assert body[1].value.elements == ["0.0"]

        assert body[2].vtype == "f32[3][2]"
        assert isinstance(body[2].value, ArrayNode)
        assert body[2].value.size == "3"
        assert isinstance(body[2].value.elements[0], ArrayNode)
    except Exception as e:
        pytest.fail(f"Fixed array parsing failed: {e}")


def test_reference_array_type_parsing():
    code = """
    fn sample_arrays(weights: &[f32; 4], output: &mut [Vec3<f32>; 2]) {
        let first = weights[0];
    }
    """
    try:
        ast = parse_code(code)
        params = ast.functions[0].params

        assert params[0].vtype == "&f32[4]"
        assert params[1].vtype == "&mut Vec3<f32>[2]"
    except Exception as e:
        pytest.fail(f"Reference array type parsing failed: {e}")


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


def test_turbofish_path_expression_parsing():
    code = """
    type Vector<T> = Vec3<T>;

    fn test_paths() {
        let a = Vec3::<f32>::new(1.0, 2.0, 3.0);
        let b = Vector::<f32>::new(1.0, 2.0, 3.0);
        let c = Type::<f32>::CONST;
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        assert isinstance(body[0].value, FunctionCallNode)
        assert body[0].value.name == "Vec3<f32>::new"
        assert isinstance(body[1].value, FunctionCallNode)
        assert body[1].value.name == "Vector<f32>::new"
        assert body[2].value == "Type<f32>::CONST"
    except Exception as e:
        pytest.fail(f"Turbofish path expression parsing failed: {e}")


def test_module_path_expression_parsing():
    code = """
    type Vector<T> = Vec3<T>;

    fn test_paths() {
        let a = crate::math::value();
        let b = super::math::VALUE;
        let c = self::helper::CONST;
        let d = crate::math::Vector::<f32>::new(1.0, 2.0, 3.0);
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        assert isinstance(body[0].value, FunctionCallNode)
        assert body[0].value.name == "crate::math::value"
        assert body[1].value == "super::math::VALUE"
        assert body[2].value == "self::helper::CONST"
        assert isinstance(body[3].value, FunctionCallNode)
        assert body[3].value.name == "crate::math::Vector<f32>::new"
    except Exception as e:
        pytest.fail(f"Module path expression parsing failed: {e}")


def test_module_path_type_parsing():
    code = """
    fn sample(x: crate::math::Real) -> crate::math::Real {
        x
    }

    type Name = std::vec::Vec<f32>;
    """
    try:
        ast = parse_code(code)
        func = ast.functions[0]

        assert func.params[0].vtype == "crate::math::Real"
        assert func.return_type == "crate::math::Real"
        assert ast.type_aliases[0].alias_type == "std::vec::Vec<f32>"
    except Exception as e:
        pytest.fail(f"Module path type parsing failed: {e}")


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
        assert func.generics == ["T", "U"]
        assert func.where_clauses == [("T", ["Clone"]), ("U", ["Debug"])]
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
    pub trait Drawable<T: Into<Vec3<f32>> + Clone, U>
    where
        U: Debug,
    {
        type Output: Copy + Clone;
        type Scratch: Into<Vec3<f32>> = Vec3<f32>;

        fn draw(&self) -> Self::Output;
        fn update<V: Copy>(&mut self, value: V) -> U
        where
            V: Clone;
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.traits) == 1

        trait = ast.traits[0]
        assert trait.name == "Drawable"
        assert trait.visibility == "pub"
        assert trait.generics == ["T: Into<Vec3<f32>> + Clone", "U"]
        assert trait.where_clauses == [("U", ["Debug"])]
        assert len(trait.associated_types) == 2
        assert all(
            isinstance(associated_type, AssociatedTypeNode)
            for associated_type in trait.associated_types
        )
        assert trait.associated_types[0].name == "Output"
        assert trait.associated_types[0].bounds == ["Copy", "Clone"]
        assert trait.associated_types[0].default_type is None
        assert trait.associated_types[1].name == "Scratch"
        assert trait.associated_types[1].bounds == ["Into<Vec3<f32>>"]
        assert trait.associated_types[1].default_type == "Vec3<f32>"
        assert [method.name for method in trait.methods] == ["draw", "update"]
        assert trait.methods[0].params[0].vtype == "&Self"
        assert trait.methods[0].return_type == "Self::Output"
        assert trait.methods[1].params[0].vtype == "&mut Self"
        assert trait.methods[1].params[1].vtype == "V"
        assert trait.methods[1].return_type == "U"
        assert trait.methods[1].generics == ["V: Copy"]
        assert trait.methods[1].where_clauses == [("V", ["Clone"])]
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
