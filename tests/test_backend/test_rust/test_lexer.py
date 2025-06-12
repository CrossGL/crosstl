import pytest
from typing import List
from crosstl.backend.Rust.RustLexer import RustLexer


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = RustLexer(code)
    return lexer.tokenize()


def test_struct_tokenization():
    code = """
    #[repr(C)]
    pub struct Vertex {
        position: Vec3<f32>,
        normal: Vec3<f32>,
        texcoord: Vec2<f32>,
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Struct tokenization not implemented.")


def test_function_tokenization():
    code = """
    #[vertex_shader]
    pub fn vertex_main(vertex: &Vertex) -> Vec4<f32> {
        let pos = Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
        pos
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Function tokenization not implemented.")


def test_if_tokenization():
    code = """
    fn fragment_main(input: &FragmentInput) -> Vec4<f32> {
        if input.color.x > 0.5 {
            Vec4::new(1.0, 0.0, 0.0, 1.0)
        } else {
            Vec4::new(0.0, 1.0, 0.0, 1.0)
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("If statement tokenization not implemented.")


def test_for_loop_tokenization():
    code = """
    fn main() {
        for i in 0..10 {
            println!("{}", i);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("For loop tokenization not implemented.")


def test_while_loop_tokenization():
    code = """
    fn main() {
        let mut i = 0;
        while i < 10 {
            i += 1;
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("While loop tokenization not implemented.")


def test_loop_tokenization():
    code = """
    fn main() {
        loop {
            break;
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Loop tokenization not implemented.")


def test_match_tokenization():
    code = """
    fn main() {
        let value = 42;
        match value {
            0 => println!("zero"),
            1..=10 => println!("one to ten"),
            _ => println!("something else"),
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Match expression tokenization not implemented.")


def test_let_binding_tokenization():
    code = """
    fn main() {
        let x = 42;
        let mut y = 3.14;
        let z: f32 = 2.0;
        let (a, b) = (1, 2);
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Let binding tokenization not implemented.")


def test_use_statement_tokenization():
    code = """
    use std::collections::HashMap;
    use gpu::*;
    use math::{sin, cos, tan};
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Use statement tokenization not implemented.")


def test_impl_block_tokenization():
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
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Impl block tokenization not implemented.")


def test_trait_tokenization():
    code = """
    pub trait Drawable {
        fn draw(&self);
        fn update(&mut self, delta: f32);
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Trait tokenization not implemented.")


def test_const_static_tokenization():
    code = """
    const PI: f32 = 3.14159;
    static mut GLOBAL_COUNTER: i32 = 0;
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Const/static tokenization not implemented.")


def test_attributes_tokenization():
    code = """
    #[vertex_shader]
    #[location(0)]
    #[derive(Debug, Clone)]
    #[repr(C)]
    pub struct MyStruct {
        #[location(0)]
        position: Vec3<f32>,
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Attributes tokenization not implemented.")


def test_assignment_operators_tokenization():
    code = """
    fn main() {
        let mut a = 5;
        a += 3;
        a -= 2;
        a *= 4;
        a /= 2;
        a %= 3;
        a ^= 1;
        a |= 2;
        a &= 7;
        a <<= 1;
        a >>= 2;
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Assignment operators tokenization not implemented.")


def test_bitwise_operators_tokenization():
    code = """
    fn main() {
        let val = 0x01;
        let result = val | 0x02;
        let result2 = val & 0x04;
        let result3 = val ^ 0x08;
        let result4 = !val;
        let result5 = val << 2;
        let result6 = val >> 1;
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Bitwise operators tokenization not implemented.")


def test_logical_operators_tokenization():
    code = """
    fn main() {
        let val_0 = true;
        let val_1 = val_0 && false;
        let val_2 = val_0 || false;
        let val_3 = !val_0;
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Logical operators tokenization not implemented.")


def test_references_tokenization():
    code = """
    fn main() {
        let x = 42;
        let y = &x;
        let z = &mut x;
        let w = *y;
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("References tokenization not implemented.")


def test_generics_tokenization():
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
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Generics tokenization not implemented.")


def test_array_slice_tokenization():
    code = """
    fn main() {
        let arr: [i32; 5] = [1, 2, 3, 4, 5];
        let slice = &arr[1..3];
        let element = arr[0];
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Array/slice tokenization not implemented.")


def test_tuple_tokenization():
    code = """
    fn main() {
        let tup: (i32, f64, u8) = (500, 6.4, 1);
        let (x, y, z) = tup;
        let first = tup.0;
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Tuple tokenization not implemented.")


def test_closure_tokenization():
    code = """
    fn main() {
        let add = |x, y| x + y;
        let result = add(5, 3);
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Closure tokenization not implemented.")


def test_comments_tokenization():
    code = """
    // Single line comment
    fn main() {
        let a = 5; // End of line comment
        /*
        Multi-line comment
        spanning multiple lines
        */
        let b = 10;
    }
    """
    try:
        tokens = tokenize_code(code)
        # Comments should be tokenized but filtered out in lexer
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Comments tokenization not implemented.")


def test_string_literals_tokenization():
    code = """
    fn main() {
        let message = "Hello, Rust!";
        let path = "path/to/file.txt";
        let raw_string = r#"This is a "raw" string"#;
        let char_lit = 'a';
    }
    """
    try:
        tokens = tokenize_code(code)
        string_literals = [token[1] for token in tokens if token[0] == "STRING"]
        assert (
            '"Hello, Rust!"' in string_literals
        ), "String literal not tokenized correctly"
        assert (
            '"path/to/file.txt"' in string_literals
        ), "String literal not tokenized correctly"
    except SyntaxError:
        pytest.fail("String literals tokenization not implemented.")


def test_numeric_literals_tokenization():
    code = """
    fn main() {
        let decimal = 42;
        let hex = 0x2A;
        let binary = 0b101010;
        let octal = 0o52;
        let float_val = 3.14159;
        let scientific = 1.23e-4;
        let float_suffix = 3.14f32;
        let int_suffix = 42i32;
    }
    """
    try:
        tokens = tokenize_code(code)
        numbers = [token[1] for token in tokens if token[0] == "NUMBER"]
        assert "42" in numbers, "Decimal literal not tokenized correctly"
        assert "3.14159" in numbers, "Float literal not tokenized correctly"
    except SyntaxError:
        pytest.fail("Numeric literals tokenization not implemented.")


def test_type_casting_tokenization():
    code = """
    fn main() {
        let x = 42i32;
        let y = x as f64;
        let z = y as i32;
    }
    """
    try:
        tokens = tokenize_code(code)
        assert len(tokens) > 0
    except SyntaxError:
        pytest.fail("Type casting tokenization not implemented.")


def test_mod_tokenization():
    code = """
    let a = 10 % 3;
    """
    tokens = tokenize_code(code)

    has_mod = False
    for token in tokens:
        if token == ("MODULO", "%"):
            has_mod = True
            break

    assert has_mod, "Modulus operator (%) not tokenized correctly"


def test_bitwise_not_tokenization():
    code = """
    let a = !5;
    """
    tokens = tokenize_code(code)

    has_not = False
    for token in tokens:
        if token == ("EXCLAMATION", "!"):
            has_not = True
            break

    assert has_not, "Logical NOT operator (!) not tokenized correctly"


def test_range_tokenization():
    code = """
    fn main() {
        for i in 0..10 {
            println!("{}", i);
        }
        for i in 0..=10 {
            println!("{}", i);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        has_range = any(token[0] == "RANGE" for token in tokens)
        has_range_inclusive = any(token[0] == "RANGE_INCLUSIVE" for token in tokens)
        assert has_range, "Range operator (..) not tokenized correctly"
        assert (
            has_range_inclusive
        ), "Inclusive range operator (..=) not tokenized correctly"
    except SyntaxError:
        pytest.fail("Range operators tokenization not implemented.")


if __name__ == "__main__":
    pytest.main()
