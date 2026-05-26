import pytest
import crosstl.translator
import shutil
import subprocess
import textwrap
from pathlib import Path
from crosstl.translator.parser import Parser
from crosstl.translator.lexer import Lexer
from crosstl.translator.ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    ExecutionModel,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    PrimitiveType,
    ShaderNode,
    VariableNode,
)
from crosstl.translator.codegen.rust_codegen import RustCodeGen
from typing import List


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = Lexer(code)
    return lexer.get_tokens()


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST.

    Args:
        tokens (List): The list of tokens to parse
    Returns:
        AST: The abstract syntax tree generated from the parser
    """
    parser = Parser(tokens)
    return parser.parse()


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = RustCodeGen()
    return codegen.generate(ast_node)


RUST_SMOKE_STUBS = """
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_parens)]

mod math {
    use std::marker::PhantomData;
    use std::ops::{Add, Div, Mul, Neg, Sub};

    #[derive(Debug, Clone, Copy, Default)]
    pub struct Vec2<T> {
        pub x: T,
        pub y: T,
    }

    impl<T> Vec2<T> {
        pub fn new(x: T, y: T) -> Self {
            Self { x, y }
        }
    }

    #[derive(Debug, Clone, Copy, Default)]
    pub struct Vec3<T> {
        pub x: T,
        pub y: T,
        pub z: T,
    }

    impl<T> Vec3<T> {
        pub fn new(x: T, y: T, z: T) -> Self {
            Self { x, y, z }
        }
    }

    impl Add for Vec3<f32> {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
        }
    }

    impl Sub for Vec3<f32> {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
        }
    }

    impl Mul for Vec3<f32> {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
        }
    }

    impl Mul<f32> for Vec3<f32> {
        type Output = Self;

        fn mul(self, rhs: f32) -> Self::Output {
            Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
        }
    }

    impl Mul<Vec3<f32>> for f32 {
        type Output = Vec3<f32>;

        fn mul(self, rhs: Vec3<f32>) -> Self::Output {
            Vec3::new(self * rhs.x, self * rhs.y, self * rhs.z)
        }
    }

    impl Div<f32> for Vec3<f32> {
        type Output = Self;

        fn div(self, rhs: f32) -> Self::Output {
            Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
        }
    }

    impl Neg for Vec3<f32> {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self::new(-self.x, -self.y, -self.z)
        }
    }

    #[derive(Debug, Clone, Copy, Default)]
    pub struct Vec4<T> {
        pub x: T,
        pub y: T,
        pub z: T,
        pub w: T,
    }

    impl<T> Vec4<T> {
        pub fn new(x: T, y: T, z: T, w: T) -> Self {
            Self { x, y, z, w }
        }
    }

    impl Add for Vec4<f32> {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self::new(
                self.x + rhs.x,
                self.y + rhs.y,
                self.z + rhs.z,
                self.w + rhs.w,
            )
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Mat3<T>(PhantomData<T>);

    impl<T> Default for Mat3<T> {
        fn default() -> Self {
            Self(PhantomData)
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Mat4<T>(PhantomData<T>);

    impl<T> Default for Mat4<T> {
        fn default() -> Self {
            Self(PhantomData)
        }
    }

    impl Mul<Vec3<f32>> for Mat3<f32> {
        type Output = Vec3<f32>;

        fn mul(self, rhs: Vec3<f32>) -> Self::Output {
            rhs
        }
    }

    impl Mul<Vec4<f32>> for Mat4<f32> {
        type Output = Vec4<f32>;

        fn mul(self, rhs: Vec4<f32>) -> Self::Output {
            rhs
        }
    }

    pub fn dot(left: Vec3<f32>, right: Vec3<f32>) -> f32 {
        left.x * right.x + left.y * right.y + left.z * right.z
    }

    pub fn normalize(value: Vec3<f32>) -> Vec3<f32> {
        value
    }

    pub fn reflect(incident: Vec3<f32>, normal: Vec3<f32>) -> Vec3<f32> {
        incident - normal * (2.0 * dot(incident, normal))
    }

    pub fn max(left: f32, right: f32) -> f32 {
        left.max(right)
    }

    pub fn min(left: f32, right: f32) -> f32 {
        left.min(right)
    }

    pub fn pow(left: f32, right: f32) -> f32 {
        left.powf(right)
    }

    pub fn sqrt(value: f32) -> f32 {
        value.sqrt()
    }

    pub fn floor(value: f32) -> f32 {
        value.floor()
    }

    pub fn smoothstep(_edge0: f32, _edge1: f32, value: f32) -> f32 {
        value
    }

    pub trait Lerp<Rhs = Self, T = f32> {
        type Output;

        fn lerp(self, rhs: Rhs, t: T) -> Self::Output;
    }

    impl Lerp for f32 {
        type Output = f32;

        fn lerp(self, rhs: f32, t: f32) -> Self::Output {
            self + (rhs - self) * t
        }
    }

    impl Lerp for Vec3<f32> {
        type Output = Vec3<f32>;

        fn lerp(self, rhs: Vec3<f32>, t: f32) -> Self::Output {
            self + (rhs - self) * t
        }
    }

    pub fn lerp<Left, Right, Weight>(
        left: Left,
        right: Right,
        weight: Weight,
    ) -> <Left as Lerp<Right, Weight>>::Output
    where
        Left: Lerp<Right, Weight>,
    {
        left.lerp(right, weight)
    }
}

mod gpu {
    use super::math::{Vec2, Vec3, Vec4};
    use std::marker::PhantomData;

    #[derive(Debug, Clone, Copy)]
    pub struct Texture1D<T>(PhantomData<T>);

    impl<T> Default for Texture1D<T> {
        fn default() -> Self {
            Self(PhantomData)
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Texture1DArray<T>(PhantomData<T>);

    impl<T> Default for Texture1DArray<T> {
        fn default() -> Self {
            Self(PhantomData)
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Texture2D<T>(PhantomData<T>);

    impl<T> Default for Texture2D<T> {
        fn default() -> Self {
            Self(PhantomData)
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Texture2DArray<T>(PhantomData<T>);

    impl<T> Default for Texture2DArray<T> {
        fn default() -> Self {
            Self(PhantomData)
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Texture3D<T>(PhantomData<T>);

    impl<T> Default for Texture3D<T> {
        fn default() -> Self {
            Self(PhantomData)
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct TextureCube<T>(PhantomData<T>);

    impl<T> Default for TextureCube<T> {
        fn default() -> Self {
            Self(PhantomData)
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct TextureCubeArray<T>(PhantomData<T>);

    impl<T> Default for TextureCubeArray<T> {
        fn default() -> Self {
            Self(PhantomData)
        }
    }

    pub trait TextureLike {}
    impl<T> TextureLike for Texture1D<T> {}
    impl<T> TextureLike for Texture1DArray<T> {}
    impl<T> TextureLike for Texture2D<T> {}
    impl<T> TextureLike for Texture2DArray<T> {}
    impl<T> TextureLike for Texture3D<T> {}
    impl<T> TextureLike for TextureCube<T> {}
    impl<T> TextureLike for TextureCubeArray<T> {}

    pub trait SampleCoord {}
    impl SampleCoord for f32 {}
    impl SampleCoord for Vec2<f32> {}
    impl SampleCoord for Vec3<f32> {}
    impl SampleCoord for Vec4<f32> {}

    pub fn sample<Texture, Coord>(_texture: Texture, _coord: Coord) -> Vec4<f32>
    where
        Texture: TextureLike,
        Coord: SampleCoord,
    {
        Vec4::default()
    }

    pub fn sample_lod<Texture, Coord, Lod>(
        _texture: Texture,
        _coord: Coord,
        _lod: Lod,
    ) -> Vec4<f32>
    where
        Texture: TextureLike,
        Coord: SampleCoord,
    {
        Vec4::default()
    }

    pub fn sample_grad<Texture, Coord, Grad>(
        _texture: Texture,
        _coord: Coord,
        _ddx: Grad,
        _ddy: Grad,
    ) -> Vec4<f32>
    where
        Texture: TextureLike,
        Coord: SampleCoord,
    {
        Vec4::default()
    }

    pub fn sample_offset<Texture, Coord, Offset>(
        _texture: Texture,
        _coord: Coord,
        _offset: Offset,
    ) -> Vec4<f32>
    where
        Texture: TextureLike,
        Coord: SampleCoord,
    {
        Vec4::default()
    }
}
"""


def rustc_or_skip():
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc is not installed")
    return rustc


def rust_smoke_source(generated_code: str) -> str:
    stripped_lines = [
        line
        for line in generated_code.splitlines()
        if line.strip()
        not in {
            "#[vertex_shader]",
            "#[fragment_shader]",
            "#[compute_shader]",
        }
    ]
    return textwrap.dedent(RUST_SMOKE_STUBS) + "\n" + "\n".join(stripped_lines)


def assert_generated_rust_smoke_compiles(generated_code: str, tmp_path):
    source = rust_smoke_source(generated_code)
    source_path = tmp_path / "generated.rs"
    output_path = tmp_path / "libgenerated.rlib"
    source_path.write_text(source, encoding="utf-8")

    result = subprocess.run(
        [
            rustc_or_skip(),
            "--edition=2021",
            "--crate-type=lib",
            str(source_path),
            "-o",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + "\n\n" + source


def test_struct():
    code = """
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "struct VSInput" in generated_code
        assert "struct VSOutput" in generated_code
        assert "pub texCoord: Vec2<f32>" in generated_code
        assert "pub color: Vec4<f32>" in generated_code
        assert "#[repr(C)]" in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Rust struct codegen not implemented.")


def test_unsized_struct_arrays_do_not_derive_copy():
    code = """
    struct DynamicPayload {
        float weights[];
        vec3 colors[2];
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "#[derive(Debug, Clone, Default)]" in generated_code
    assert "#[derive(Debug, Clone, Copy, Default)]" not in generated_code
    assert "pub weights: Vec<f32>," in generated_code
    assert "pub colors: [Vec3<f32>; 2]," in generated_code


def test_legacy_generic_wrappers_emit_rust_generic_params():
    code = """
    shader GenericWrappers {
        generic<T, U> struct Pair {
            left: T;
            right: U;
        }

        generic<T> struct Wrapper {
            item: Option<int>;
            result: Result<T, int>;
        }

        generic<T> fn identity(value: T) -> T {
            return value;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub struct Pair<T, U>" in generated_code
    assert "pub left: T," in generated_code
    assert "pub right: U," in generated_code
    assert "pub struct Wrapper<T>" in generated_code
    assert "pub item: Option<i32>," in generated_code
    assert "pub result: Result<T, i32>," in generated_code
    assert "pub fn identity<T>(value: T) -> T" in generated_code


def test_traits_and_generic_constraints_emit_rust_and_smoke_compile(tmp_path):
    code = """
    shader GenericBounds {
        trait Numeric {
            fn zero() -> Self;
            fn add(self, other: Self) -> Self;
            fn checked(self) -> Maybe<Self>;
        }

        struct ScalarBox<T: Numeric> {
            value: T;
        }

        generic<T: Numeric> enum Maybe {
            Some(T)
        }

        generic<T: Numeric + Copy> fn add_zero(value: T) -> T {
            return value.add(T::zero());
        }

        generic<T: Numeric> fn is_same(a: T, b: T) -> bool {
            return a == b;
        }

        generic<T: Numeric + Copy> fn operator_mix(a: T, b: T, c: T, d: T) -> T {
            return (a + b) - (c / d);
        }

        generic<T: Numeric> fn divide(a: T, b: T) -> T {
            return a / b;
        }

        generic<T: Numeric> fn call_divide(a: T, b: T) -> T {
            return divide(a, b);
        }

        generic<T: Numeric> fn make_box(a: T, b: T) -> ScalarBox<T> {
            return ScalarBox { value: a - b };
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub trait Numeric: Sized {" in generated_code
    assert "fn zero() -> Self;" in generated_code
    assert "fn add(self, other: Self) -> Self;" in generated_code
    assert "fn checked(self) -> Maybe<Self>;" in generated_code
    assert "impl Numeric for f32" not in generated_code
    assert "pub struct ScalarBox<T: Numeric>" in generated_code
    assert "pub enum Maybe<T: Numeric>" in generated_code
    assert "impl<T: Numeric + Default> Default for Maybe<T>" in generated_code
    assert "pub fn add_zero<T: Numeric + Copy>(value: T) -> T" in generated_code
    assert "return Numeric::add(value, T::zero());" in generated_code
    assert (
        "pub fn is_same<T: Numeric + PartialEq>(a: T, b: T) -> bool" in generated_code
    )
    assert (
        "pub fn operator_mix<T: Numeric + Copy + "
        "std::ops::Add<Output = T> + std::ops::Div<Output = T> + "
        "std::ops::Sub<Output = T>>"
    ) in generated_code
    assert (
        "pub fn divide<T: Numeric + std::ops::Div<Output = T>>(a: T, b: T) -> T"
        in generated_code
    )
    assert (
        "pub fn call_divide<T: Numeric + std::ops::Div<Output = T>>(a: T, b: T) -> T"
        in generated_code
    )
    assert "pub fn make_box<T: Numeric + std::ops::Sub<Output = T>>" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_canonical_numeric_trait_emits_primitive_impls_and_smoke_compiles(tmp_path):
    code = """
    shader PrimitiveNumericBounds {
        trait Numeric {
            fn add(self, other: Self) -> Self;
            fn mul(self, other: Self) -> Self;
            fn zero() -> Self;
            fn one() -> Self;
        }

        struct ScalarBox<T: Numeric> {
            value: T;
        }

        fn make_float_box(value: float) -> ScalarBox<float> {
            return ScalarBox { value: value };
        }

        fn sum_float(a: float, b: float) -> float {
            return Numeric::add(a, b);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "impl Numeric for f32" in generated_code
    assert "impl Numeric for i32" in generated_code
    assert "pub fn make_float_box(value: f32) -> ScalarBox<f32>" in generated_code
    assert "pub fn sum_float(a: f32, b: f32) -> f32" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_generic_sqrt_adds_trait_bound_and_smoke_compiles(tmp_path):
    code = """
    shader GenericSqrt {
        trait Numeric {
            fn add(self, other: Self) -> Self;
            fn mul(self, other: Self) -> Self;
            fn zero() -> Self;
            fn one() -> Self;
        }

        generic<T: Numeric> fn root_energy(value: T) -> T {
            return sqrt(value);
        }

        fn root_float(value: float) -> float {
            return sqrt(value);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub trait CglSqrt" in generated_code
    assert "impl CglSqrt for f32" in generated_code
    assert "impl CglSqrt for i32" in generated_code
    assert "pub fn root_energy<T: Numeric + CglSqrt>(value: T) -> T" in generated_code
    assert "return CglSqrt::cgl_sqrt(value);" in generated_code
    assert "pub fn root_float(value: f32) -> f32" in generated_code
    assert "return sqrt(value);" in generated_code
    assert "pub fn root_float<T" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_generated_structs_emit_new_constructors_and_smoke_compile(tmp_path):
    code = """
    shader StructConstructors {
        uniform float weight;

        generic<T> struct Vec3 {
            x: T;
            y: T;
            z: T;
        }

        struct Packed {
            color: Vec3<float>;
            weight: float;
        }

        fn make_vec() -> Vec3<float> {
            return vec3(1.0, 2.0, 3.0);
        }

        fn make_packed() -> Packed {
            return Packed::new(vec3(0.25, 0.5, 0.75), 1.0);
        }

        fn shade(light: vec3) -> vec3 {
            let unit = normalize(light);
            return vec3(unit.x, unit.y, unit.z);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "impl<T> Vec3<T> {" in generated_code
    assert "pub fn new(x: T, y: T, z: T) -> Self" in generated_code
    assert "impl Packed {" in generated_code
    assert "pub fn new(color: Vec3<f32>, weight_value: f32) -> Self" in generated_code
    assert "Self { color, weight: weight_value }" in generated_code
    assert "return Vec3::<f32>::new(1.0, 2.0, 3.0);" in generated_code
    assert (
        "return Packed::new(Vec3::<f32>::new(0.25, 0.5, 0.75), 1.0);" in generated_code
    )
    assert "pub fn shade(light: math::Vec3<f32>) -> math::Vec3<f32>" in generated_code
    assert "let unit: math::Vec3<f32> = normalize(light);" in generated_code
    assert "return math::Vec3::<f32>::new(unit.x, unit.y, unit.z);" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_nested_enum_wrapper_structs_emit_rust_enums():
    code = """
    shader GenericWrappers {
        generic<T> struct Option {
            enum OptionType {
                Some(T),
                None
            }
            OptionType variant;
        }

        generic<T, E> struct Result {
            enum ResultType {
                Ok(T),
                Err(E)
            }
            ResultType variant;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "#[derive(Debug, Clone, Copy, Default)]\npub enum Option<T>" in generated_code
    )
    assert "Some(T)," in generated_code
    assert "    #[default]\n    None," in generated_code
    assert "#[derive(Debug, Clone, Copy)]\npub enum Result<T, E>" in generated_code
    assert "Ok(T)," in generated_code
    assert "Err(E)," in generated_code
    assert "impl<T: Default, E: Default> Default for Result<T, E>" in generated_code
    assert "Self::Ok(Default::default())" in generated_code
    assert "pub struct Option<T>" not in generated_code
    assert "pub struct Result<T, E>" not in generated_code
    assert "pub variant: OptionType," not in generated_code
    assert "pub variant: ResultType," not in generated_code


def test_top_level_enums_emit_rust_variants():
    code = """
    shader EnumCases {
        enum MathError {
            DivisionByZero,
            InvalidInput
        }

        enum LightingModel {
            Phong { ambient: vec3, shininess: float },
            Toon { base_color: vec3 }
        }

        enum ShaderError {
            MathError(MathError),
            TextureError(str),
            InvalidState
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "#[derive(Debug, Clone, Copy, Default)]\npub enum MathError {" in generated_code
    )
    assert "    #[default]\n    DivisionByZero," in generated_code
    assert "DivisionByZero," in generated_code
    assert "InvalidInput," in generated_code
    assert "#[derive(Debug, Clone, Copy)]\npub enum LightingModel {" in generated_code
    assert "Phong { ambient: Vec3<f32>, shininess: f32 }," in generated_code
    assert "Toon { base_color: Vec3<f32> }," in generated_code
    assert "impl Default for LightingModel {" in generated_code
    assert (
        "Self::Phong { ambient: Default::default(), " "shininess: Default::default() }"
    ) in generated_code
    assert (
        "#[derive(Debug, Clone, Copy, Default)]\npub enum ShaderError {"
        in generated_code
    )
    assert "MathError(MathError)," in generated_code
    assert "TextureError(&'static str)," in generated_code
    assert "    #[default]\n    InvalidState," in generated_code
    assert "InvalidState," in generated_code


def test_top_level_enum_discriminants_emit_rust_values():
    code = """
    shader EnumCases {
        enum OpCode {
            Add = 1,
            Multiply = 2
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub enum OpCode {" in generated_code
    assert "Add = 1," in generated_code
    assert "Multiply = 2," in generated_code


def test_advanced_generic_pattern_matching_example_emits_referenced_enums():
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "examples/advanced/GenericPatternMatching.cgl").read_text()

    generated_code = generate_code(parse_code(tokenize_code(source)))

    assert (
        "#[derive(Debug, Clone, Copy, Default)]\npub enum Option<T>" in generated_code
    )
    assert "    #[default]\n    None," in generated_code
    assert "#[derive(Debug, Clone, Copy)]\npub enum Result<T, E>" in generated_code
    assert "impl<T: Default, E: Default> Default for Result<T, E>" in generated_code
    assert "pub struct Option<T>" not in generated_code
    assert "pub struct Result<T, E>" not in generated_code
    assert (
        "#[derive(Debug, Clone, Copy, Default)]\npub enum MathError {" in generated_code
    )
    assert "DivisionByZero," in generated_code
    assert (
        "#[derive(Debug, Clone, Copy, Default)]\npub enum ShaderError {"
        in generated_code
    )
    assert "TextureError(&'static str)," in generated_code
    assert "#[derive(Debug, Clone, Copy)]\npub enum LightingModel {" in generated_code
    assert "impl Default for LightingModel {" in generated_code
    assert (
        "PBR { albedo: math::Vec3<f32>, metallic: f32, roughness: f32, ao: f32 },"
        in generated_code
    )
    assert (
        "#[derive(Debug, Clone, Copy, Default)]\npub enum VectorOp {" in generated_code
    )
    assert "Normalize," in generated_code
    assert "pub struct Geometry {" in generated_code
    assert "pub struct Material {" in generated_code
    assert (
        "pub fn validate_material(material: Material) -> Result<i32, ShaderError>"
        in generated_code
    )
    assert "match validate_material(material)" in generated_code
    assert "material.validate()" not in generated_code
    assert (
        "#[derive(Debug, Clone, Copy, Default)]\npub enum RenderOutput {"
        in generated_code
    )
    assert "Success(RenderedFrame)," in generated_code
    assert "Clear { color: Vec4<f32>, depth: f32 }," in generated_code


def test_advanced_generic_pattern_matching_example_smoke_compiles(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "examples/advanced/GenericPatternMatching.cgl").read_text()

    generated_code = generate_code(parse_code(tokenize_code(source)))

    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_generated_rust_enum_defaults_smoke_compile_with_rustc(tmp_path):
    code = """
    shader EnumSmoke {
        generic<T> struct Option {
            enum OptionType {
                Some(T),
                None
            }
            OptionType variant;
        }

        generic<T, E> struct Result {
            enum ResultType {
                Ok(T),
                Err(E)
            }
            ResultType variant;
        }

        enum MathError {
            DivisionByZero,
            InvalidInput
        }

        enum LightingModel {
            Phong { ambient: vec3, shininess: float },
            Toon { base_color: vec3 }
        }

        struct RenderState {
            Option<int> material_id;
            Result<int, MathError> status;
            LightingModel lighting_model;
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_unsized_cbuffer_arrays_do_not_derive_copy():
    code = """
    cbuffer DynamicBlock {
        float weights[];
        vec3 colors[2];
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "#[derive(Debug, Clone)]" in generated_code
    assert "#[derive(Debug, Clone, Copy)]" not in generated_code
    assert "pub weights: Vec<f32>," in generated_code
    assert "pub colors: [Vec3<f32>; 2]," in generated_code
    assert "static WEIGHTS: Vec<f32> = Vec::new();" in generated_code
    assert "static WEIGHTS: Vec<f32> = Default::default();" not in generated_code


def test_cbuffer_scalar_statics_use_const_default_literals():
    code = """
    cbuffer Material {
        float exposure;
        int count;
        uint mask;
        bool enabled;
        float values[4];
        int indices[2];
        bool flags[3];
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "static EXPOSURE: f32 = 0.0;" in generated_code
    assert "static COUNT: i32 = 0;" in generated_code
    assert "static MASK: u32 = 0;" in generated_code
    assert "static ENABLED: bool = false;" in generated_code
    assert "static VALUES: [f32; 4] = [0.0; 4];" in generated_code
    assert "static INDICES: [i32; 2] = [0; 2];" in generated_code
    assert "static FLAGS: [bool; 3] = [false; 3];" in generated_code
    assert "static EXPOSURE: f32 = Default::default();" not in generated_code
    assert "static VALUES: [f32; 4] = Default::default();" not in generated_code


def test_vector_matrix_array_statics_use_zeroed_static_fallback():
    code = """
    vec3 globalDirs[2];
    dvec2 globalPrecise[3];
    mat4 globalTransforms[2];

    cbuffer Material {
        vec3 directions[2];
        mat4 transforms[2];
        dvec2 precise[3];
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "static GLOBAL_DIRS: [Vec3<f32>; 2] = " "unsafe { std::mem::zeroed() };"
    ) in generated_code
    assert (
        "static GLOBAL_PRECISE: [Vec2<f64>; 3] = " "unsafe { std::mem::zeroed() };"
    ) in generated_code
    assert (
        "static GLOBAL_TRANSFORMS: [Mat4<f32>; 2] = " "unsafe { std::mem::zeroed() };"
    ) in generated_code
    assert (
        "static DIRECTIONS: [Vec3<f32>; 2] = " "unsafe { std::mem::zeroed() };"
    ) in generated_code
    assert (
        "static TRANSFORMS: [Mat4<f32>; 2] = " "unsafe { std::mem::zeroed() };"
    ) in generated_code
    assert (
        "static PRECISE: [Vec2<f64>; 3] = " "unsafe { std::mem::zeroed() };"
    ) in generated_code
    assert (
        "static GLOBAL_DIRS: [Vec3<f32>; 2] = Default::default();" not in generated_code
    )
    assert (
        "static TRANSFORMS: [Mat4<f32>; 2] = Default::default();" not in generated_code
    )


def test_default_composite_statics_use_lazy_lock():
    code = """
    vec3 direction;
    mat4 transform;

    cbuffer Material {
        vec3 tint;
        mat4 model;
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "static DIRECTION: std::sync::LazyLock<Vec3<f32>> = "
        "std::sync::LazyLock::new(|| Default::default());"
    ) in generated_code
    assert (
        "static TRANSFORM: std::sync::LazyLock<Mat4<f32>> = "
        "std::sync::LazyLock::new(|| Default::default());"
    ) in generated_code
    assert (
        "static TINT: std::sync::LazyLock<Vec3<f32>> = "
        "std::sync::LazyLock::new(|| Default::default());"
    ) in generated_code
    assert (
        "static MODEL: std::sync::LazyLock<Mat4<f32>> = "
        "std::sync::LazyLock::new(|| Default::default());"
    ) in generated_code
    assert "static DIRECTION: Vec3<f32> = Default::default();" not in generated_code
    assert "static MODEL: Mat4<f32> = Default::default();" not in generated_code


def test_lazy_lock_static_use_sites_are_dereferenced():
    code = """
    vec3 direction;
    float values[] = {1.0, 2.0};

    shader LazyUse {
        compute {
            float main() {
                return direction.x + values[0];
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "return ((*DIRECTION).x + (*VALUES)[0]);" in generated_code
    assert "direction.x + values[0]" not in generated_code


def test_colliding_static_symbol_names_are_suffixed_and_referenced():
    code = """
    float fooBar = 1.0;
    float foo_bar = 2.0;
    vec3 lightDir;
    vec3 light_dir;

    shader StaticCollision {
        compute {
            float main() {
                return fooBar + foo_bar + lightDir.x + light_dir.y;
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "static FOO_BAR: f32 = 1.0;" in generated_code
    assert "static FOO_BAR_2: f32 = 2.0;" in generated_code
    assert "static LIGHT_DIR: std::sync::LazyLock<Vec3<f32>>" in generated_code
    assert "static LIGHT_DIR_2: std::sync::LazyLock<Vec3<f32>>" in generated_code
    assert (
        "return (((FOO_BAR + FOO_BAR_2) + (*LIGHT_DIR).x) + (*LIGHT_DIR_2).y);"
        in generated_code
    )
    assert "static FOO_BAR: f32 = 2.0;" not in generated_code
    assert "fooBar + foo_bar" not in generated_code


def test_lazy_lock_static_name_shadowed_by_local_is_not_dereferenced():
    code = """
    vec3 direction;

    shader LazyUse {
        compute {
            vec3 main() {
                vec3 direction = vec3(1.0, 2.0, 3.0);
                return direction;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let direction: Vec3<f32>" in generated_code
    assert "return direction;" in generated_code
    assert "return *direction;" not in generated_code


def test_generated_rust_lazy_lock_statics_smoke_compile_with_rustc(tmp_path):
    shaders = [
        """
        vec3 direction;
        float values[] = {1.0, 2.0};

        shader LazyUse {
            compute {
                float main(int index) {
                    return direction.x + values[index];
                }
            }
        }
        """,
        """
        sampler3D volumeMap;

        shader TextureUse {
            fragment {
                vec4 main(vec3 uv) @ gl_FragColor {
                    return texture(volumeMap, uv);
                }
            }
        }
        """,
        """
        shader MatrixUse {
            vertex {
                uniform mat4 transform;

                vec4 main(vec3 position) {
                    transform * vec4(position, 1.0)
                }
            }
        }
        """,
    ]

    for shader in shaders:
        generated_code = generate_code(parse_code(tokenize_code(shader)))
        assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_generated_rust_match_expressions_smoke_compile_with_rustc(tmp_path):
    shaders = [
        """
        shader MatchSmoke {
            generic<T, E> struct Result {
                enum ResultType {
                    Ok(T),
                    Err(E)
                }
                ResultType variant;
            }

            enum MathError {
                InvalidInput
            }

            enum VectorOp {
                Add,
                Cross
            }

            fn convert(op: VectorOp) -> Result<int, MathError> {
                match op {
                    VectorOp::Add => Result::Ok(1),
                    _ => Result::Err(MathError::InvalidInput),
                }
            }
        }
        """,
        """
        shader MatchInitSmoke {
            enum Mode {
                Bright,
                Dark
            }

            fn choose(mode: Mode) -> vec3 {
                let color = match mode {
                    Mode::Bright => vec3(1.0, 1.0, 1.0),
                    _ => vec3(0.0, 0.0, 0.0),
                };
                color
            }
        }
        """,
    ]

    for shader in shaders:
        generated_code = generate_code(parse_code(tokenize_code(shader)))

        assert "match " in generated_code
        assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_generated_rust_rich_match_patterns_smoke_compile_with_rustc(tmp_path):
    code = """
    shader RichPatternSmoke {
        generic<T, E> struct Result {
            enum ResultType {
                Ok(T),
                Err(E)
            }
            ResultType variant;
        }

        enum MathError {
            InvalidInput
        }

        enum LightingModel {
            Phong {
                ambient: vec3,
                diffuse: vec3,
                specular: vec3,
                shininess: float
            },
            Toon { base_color: vec3 }
        }

        fn unwrap_result(result: Result<int, MathError>) -> int {
            match result {
                Result::Ok(value) => value,
                Result::Err(_) => 0,
            }
        }

        fn classify_model(model: LightingModel) -> int {
            match model {
                LightingModel::Phong { ambient, diffuse, specular, shininess } if shininess > 0.0 => 1,
                LightingModel::Toon { base_color, .. } => 2,
            }
        }

        fn guarded_literal(mode: int) -> int {
            match mode {
                0 if mode > 0 => 1,
                _ => 2,
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "Result::Ok(value) =>" in generated_code
    assert "Result::Err(_) =>" in generated_code
    assert (
        "LightingModel::Phong { ambient: _ambient, diffuse: _diffuse, "
        "specular: _specular, shininess } if (shininess > 0.0) =>" in generated_code
    )
    assert "LightingModel::Toon { base_color: _base_color, .. } =>" in generated_code
    assert "0 if (mode > 0) =>" in generated_code
    assert "_ => unreachable!()," in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_generated_rust_duplicate_stage_main_names_smoke_compile_with_rustc(tmp_path):
    code = """
    shader MultiStageSmoke {
        vertex {
            vec4 main(vec3 position) {
                vec4(position, 1.0)
            }
        }

        fragment {
            vec4 main(vec4 color) {
                color
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub fn vertex_main(position: Vec3<f32>) -> Vec4<f32>" in generated_code
    assert "pub fn fragment_main(color: Vec4<f32>) -> Vec4<f32>" in generated_code
    assert "pub fn main(" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_initialized_vector_and_dynamic_array_statics_use_lazy_lock():
    code = """
    vec3 globalDirs[3] = {
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0)
    };
    dvec2 preciseDirs[2] = {dvec2(1.0, 2.0)};
    mat2 transforms[2] = {mat2(1.0, 0.0, 0.0, 1.0)};
    float dynamicWeights[] = {1.0, 2.0};
    vec3 dynamicDirs[] = {vec3(0.0, 0.0, 1.0)};
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "static GLOBAL_DIRS: std::sync::LazyLock<[Vec3<f32>; 3]> = "
        "std::sync::LazyLock::new(|| ["
        "Vec3::<f32>::new(1.0, 0.0, 0.0), "
        "Vec3::<f32>::new(0.0, 1.0, 0.0), "
        "unsafe { std::mem::zeroed() }]);"
    ) in generated_code
    assert (
        "static PRECISE_DIRS: std::sync::LazyLock<[Vec2<f64>; 2]> = "
        "std::sync::LazyLock::new(|| ["
        "Vec2::<f64>::new(1.0, 2.0), "
        "unsafe { std::mem::zeroed() }]);"
    ) in generated_code
    assert (
        "static TRANSFORMS: std::sync::LazyLock<[Mat2<f32>; 2]> = "
        "std::sync::LazyLock::new(|| ["
        "Mat2::<f32>::new(1.0, 0.0, 0.0, 1.0), "
        "unsafe { std::mem::zeroed() }]);"
    ) in generated_code
    assert (
        "static DYNAMIC_WEIGHTS: std::sync::LazyLock<Vec<f32>> = "
        "std::sync::LazyLock::new(|| vec![1.0, 2.0]);"
    ) in generated_code
    assert (
        "static DYNAMIC_DIRS: std::sync::LazyLock<Vec<Vec3<f32>>> = "
        "std::sync::LazyLock::new(|| "
        "vec![Vec3::<f32>::new(0.0, 0.0, 1.0)]);"
    ) in generated_code
    assert (
        "static GLOBAL_DIRS: [Vec3<f32>; 3] = [Vec3::<f32>::new" not in generated_code
    )
    assert "static DYNAMIC_WEIGHTS: Vec<f32> = vec!" not in generated_code


def test_legacy_global_array_nodes_emit_static_declarations():
    ast = ShaderNode(
        "Legacy",
        ExecutionModel.GENERAL_PURPOSE,
        global_variables=[
            ArrayNode(PrimitiveType("float"), "weights", 4),
            ArrayNode(PrimitiveType("int"), "indices", 2),
            ArrayNode(PrimitiveType("float"), "dynamicWeights"),
            ArrayNode(
                PrimitiveType("float"),
                "initialized",
                4,
                initial_value=ArrayLiteralNode(
                    [
                        LiteralNode(1.0, PrimitiveType("float")),
                        LiteralNode(2.0, PrimitiveType("float")),
                    ]
                ),
            ),
        ],
    )

    generated_code = generate_code(ast)

    assert "static WEIGHTS: [f32; 4] = [0.0; 4];" in generated_code
    assert "static INDICES: [i32; 2] = [0; 2];" in generated_code
    assert "static DYNAMIC_WEIGHTS: Vec<f32> = Vec::new();" in generated_code
    assert "static INITIALIZED: [f32; 4] = [1.0, 2.0, 0.0, 0.0];" in generated_code
    assert "\nlet weights:" not in generated_code
    assert "\nlet initialized:" not in generated_code


def test_basic_shader():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert generated_code is not None
        assert "fn vertex_main(" in generated_code
        assert "fn fragment_main(" in generated_code
        assert "#[vertex_shader]" in generated_code
        assert "#[fragment_shader]" in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Rust basic shader codegen not implemented.")


def test_local_struct_outputs_initialize_with_default_before_field_assignment():
    code = """
    shader main {
        struct Out {
            vec4 color @ COLOR;
        };
        vertex {
            Out main() {
                Out output;
                output.color = vec4(1.0, 0.0, 0.0, 1.0);
                return output;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "#[derive(Debug, Clone, Copy, Default)]" in generated_code
    assert "let mut output: Out = Default::default();" in generated_code
    assert "output.color = Vec4::<f32>::new(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "let mut output: Out;" not in generated_code


def test_if_statement():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                if (input.texCoord.x > 0.5) {
                    output.color = vec4(1.0, 1.0, 1.0, 1.0);
                } else {
                    output.color = vec4(0.0, 0.0, 0.0, 1.0);
                }
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                let brightness = texture(iChannel0, input.color.xy).r;
                let bloom = max(0.0, brightness - 0.5);
                if (bloom > 0.5) {
                    bloom = 0.5;
                } else {
                    bloom = 0.0;
                }
                let texColor = texture(iChannel0, input.color.xy).rgb;
                let colorWithBloom = texColor + vec3(bloom);
                return vec4(colorWithBloom, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "if " in generated_code
        assert "else" in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("If statement codegen not implemented.")


def test_sampler3d_maps_to_texture3d_in_rust():
    code = """
    shader Sampler3DProbe {
        sampler3D volumeMap;
        fragment {
            vec4 main(vec3 uv) @ gl_FragColor {
                return texture(volumeMap, uv);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "static VOLUME_MAP: std::sync::LazyLock<Texture3D<f32>> = "
        "std::sync::LazyLock::new(|| Default::default());"
    ) in generated_code
    assert (
        "static VOLUME_MAP: Texture3D<f32> = Default::default();" not in generated_code
    )
    assert "static VOLUME_MAP: sampler3D" not in generated_code
    assert "return sample(*VOLUME_MAP, uv);" in generated_code


def test_sampler_array_and_cube_families_map_to_rust_textures_and_compile(tmp_path):
    code = """
    shader SamplerFamilyProbe {
        sampler1D rampMap;
        sampler1DArray rampArrayMap;
        sampler2DArray arrayMap;
        samplerCube envMap;
        samplerCubeArray probeMap;

        fragment {
            vec4 main(float x, vec2 xLayer, vec3 uvw, vec3 dir, vec4 probeCoord) @ gl_FragColor {
                let ramp = texture(rampMap, x);
                let rampLayer = texture(rampArrayMap, xLayer);
                let layer = texture(arrayMap, uvw);
                let env = texture(envMap, dir);
                let probe = texture(probeMap, probeCoord);
                return ramp + rampLayer + layer + env + probe;
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "static RAMP_MAP: std::sync::LazyLock<Texture1D<f32>>" in generated_code
    assert (
        "static RAMP_ARRAY_MAP: std::sync::LazyLock<Texture1DArray<f32>>"
        in generated_code
    )
    assert (
        "static ARRAY_MAP: std::sync::LazyLock<Texture2DArray<f32>>" in generated_code
    )
    assert "static ENV_MAP: std::sync::LazyLock<TextureCube<f32>>" in generated_code
    assert (
        "static PROBE_MAP: std::sync::LazyLock<TextureCubeArray<f32>>" in generated_code
    )
    assert "sampler1D" not in generated_code
    assert "sampler1DArray" not in generated_code
    assert "sampler2DArray" not in generated_code
    assert "samplerCube" not in generated_code
    assert "samplerCubeArray" not in generated_code
    assert "let ramp: Vec4<f32> = sample(*RAMP_MAP, x);" in generated_code
    assert (
        "let rampLayer: Vec4<f32> = sample(*RAMP_ARRAY_MAP, xLayer);" in generated_code
    )
    assert "let layer: Vec4<f32> = sample(*ARRAY_MAP, uvw);" in generated_code
    assert "let env: Vec4<f32> = sample(*ENV_MAP, dir);" in generated_code
    assert "let probe: Vec4<f32> = sample(*PROBE_MAP, probeCoord);" in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_texture_lod_grad_offset_calls_map_to_rust_helpers_and_compile(tmp_path):
    code = """
    shader TextureAdvancedSamplingProbe {
        sampler2D mainTexture;

        fragment {
            vec4 main(vec2 uv, vec2 ddx, vec2 ddy, ivec2 offset) @ gl_FragColor {
                let lodColor = textureLod(mainTexture, uv, 2.0);
                let gradColor = textureGrad(mainTexture, uv, ddx, ddy);
                let offsetColor = textureOffset(mainTexture, uv, offset);
                return lodColor + gradColor + offsetColor;
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "let lodColor: Vec4<f32> = sample_lod(*MAIN_TEXTURE, uv, 2.0);"
        in generated_code
    )
    assert (
        "let gradColor: Vec4<f32> = sample_grad(*MAIN_TEXTURE, uv, ddx, ddy);"
        in generated_code
    )
    assert (
        "let offsetColor: Vec4<f32> = sample_offset(*MAIN_TEXTURE, uv, offset);"
        in generated_code
    )
    assert "textureLod" not in generated_code
    assert "textureGrad" not in generated_code
    assert "textureOffset" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_for_statement():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                for (int i = 0; i < 10; i++) {
                    output.color = vec4(1.0, 1.0, 1.0, 1.0);
                }
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert (
            "while " in generated_code
        )  # Rust codegen converts for loops to while loops
        assert "let mut i: i32 = 0;" in generated_code
        assert "let mut i: i32 = 0;;" not in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("For statement codegen not implemented.")


def test_for_continue_emits_update_before_continue_in_rust():
    code = """
    shader main {
        compute {
            void main() {
                int total = 0;
                for (int i = 0; i < 4; i++) {
                    if (i == 1) {
                        continue;
                    }
                    for (int j = 0; j < 2; j++) {
                        if (j == 0) {
                            continue;
                        }
                        total += j;
                    }
                    total += i;
                }
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "if (i == 1) {\n            i += 1;\n            continue;\n        }"
        in generated_code
    )
    assert (
        "if (j == 0) {\n                j += 1;\n                continue;\n            }"
        in generated_code
    )
    assert (
        "if (j == 0) {\n                i += 1;\n                continue;\n            }"
        not in generated_code
    )
    assert "DoWhileNode" not in generated_code


def test_for_in_statement_lowers_to_rust_ranges_and_scopes_loop_contexts():
    code = """
    shader main {
        compute {
            void main() {
                int total = 0;
                for i in 4 {
                    if (i == 1) {
                        continue;
                    }
                    total += i;
                }
                for j in 2..5 {
                    total += j;
                }
                for k in 1..=4 {
                    total += k;
                }
                for (int outer = 0; outer < 2; outer++) {
                    for inner in 3 {
                        if (inner == 1) {
                            continue;
                        }
                        total += inner;
                    }
                }
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for i in 0..4 {" in generated_code
    assert "for j in 2..5 {" in generated_code
    assert "for k in 1..=4 {" in generated_code
    assert "total += i;" in generated_code
    assert "total += j;" in generated_code
    assert "total += k;" in generated_code
    assert "total += inner;" in generated_code
    assert (
        "if (inner == 1) {\n                continue;\n            }" in generated_code
    )
    assert "outer += 1;\n                continue;" not in generated_code
    assert "ForInNode" not in generated_code
    assert "RangeNode" not in generated_code


def test_while_statement_lowers_to_rust_while_and_scopes_loop_contexts():
    code = """
    shader main {
        compute {
            void main() {
                int value = 0;
                while (value < 4) {
                    if (value == 2) {
                        continue;
                    }
                    value += 1;
                }
                for (int i = 0; i < 3; i++) {
                    int j = 0;
                    while (j < 2) {
                        if (j == 1) {
                            continue;
                        }
                        j += 1;
                    }
                }
                do {
                    int k = 0;
                    while (k < 2) {
                        if (k == 1) {
                            break;
                        }
                        k += 1;
                    }
                    value += 1;
                } while (value < 8);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "while (value < 4) {" in generated_code
    assert "while (j < 2) {" in generated_code
    assert "if (value == 2) {\n            continue;\n        }" in generated_code
    assert "if (j == 1) {\n                continue;\n            }" in generated_code
    assert (
        "if (j == 1) {\n                i += 1;\n                continue;"
        not in generated_code
    )
    assert "__cgl_do_break_0 = true;" not in generated_code
    assert "WhileNode" not in generated_code


def test_loop_statement_lowers_to_rust_loop_and_scopes_loop_contexts():
    code = """
    shader main {
        compute {
            void main() {
                int value = 0;
                loop {
                    value += 1;
                    if (value == 2) {
                        continue;
                    }
                    if (value > 3) {
                        break;
                    }
                }
                for (int i = 0; i < 3; i++) {
                    loop {
                        if (i == 1) {
                            continue;
                        }
                        break;
                    }
                }
                do {
                    loop {
                        if (value == 4) {
                            break;
                        }
                        continue;
                    }
                    value += 1;
                } while (value < 8);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "loop {" in generated_code
    assert "value += 1;" in generated_code
    assert "if (value == 2) {\n            continue;\n        }" in generated_code
    assert "if (i == 1) {\n                continue;\n            }" in generated_code
    assert (
        "if (i == 1) {\n                i += 1;\n                continue;"
        not in generated_code
    )
    assert "__cgl_do_break_0 = true;" not in generated_code
    assert "LoopNode" not in generated_code


def test_increment_and_decrement_emit_rust_assignment_updates():
    code = """
    shader main {
        compute {
            void main() {
                int i = 0;
                i++;
                ++i;
                i--;
                --i;
                for (int j = 0; j < 2; j++) {
                    i++;
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "i += 1;" in generated_code
    assert "i -= 1;" in generated_code
    assert "j += 1;" in generated_code
    assert "++i" not in generated_code
    assert "i++" not in generated_code
    assert "--i" not in generated_code
    assert "i--" not in generated_code
    assert "++j" not in generated_code


def test_increment_and_decrement_initializers_preserve_rust_value_order():
    code = """
    shader main {
        compute {
            void main() {
                int i = 0;
                int pre = ++i;
                int post = i++;
                int pre_dec = --i;
                int post_dec = i--;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    lines = [line.strip() for line in generated_code.splitlines()]

    def contains_adjacent(first, second):
        return any(
            current == first and following == second
            for current, following in zip(lines, lines[1:])
        )

    assert contains_adjacent("i += 1;", "let pre: i32 = i;")
    assert contains_adjacent("let post: i32 = i;", "i += 1;")
    assert contains_adjacent("i -= 1;", "let pre_dec: i32 = i;")
    assert contains_adjacent("let post_dec: i32 = i;", "i -= 1;")
    assert "let pre: i32 = i += 1;" not in generated_code
    assert "let post: i32 = i += 1;" not in generated_code
    assert "let pre_dec: i32 = i -= 1;" not in generated_code
    assert "let post_dec: i32 = i -= 1;" not in generated_code


def test_do_while_statement_lowers_to_rust_loop_with_condition_after_body():
    code = """
    shader main {
        compute {
            void main() {
                int value = 0;
                do {
                    value += 1;
                    if (value == 2) {
                        continue;
                    }
                    if (value == 4) {
                        break;
                    }
                    value += 2;
                } while (value < 8);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let mut __cgl_do_break_0: bool = false;" in generated_code
    assert "loop {\n        loop {" in generated_code
    assert "value += 1;" in generated_code
    assert "if (value == 2) {\n                break;\n            }" in generated_code
    assert (
        "if (value == 4) {\n"
        "                __cgl_do_break_0 = true;\n"
        "                break;\n"
        "            }"
    ) in generated_code
    assert "if !((value < 8))" in generated_code
    assert "DoWhileNode" not in generated_code


def test_bool_string_and_char_literals_emit_rust_syntax():
    code = """
    shader main {
        compute {
            void main() {
                bool enabled = true;
                bool disabled = false;
                string label = "debug";
                char marker = 'x';
                if (enabled && !disabled) {
                    label = "active";
                    marker = 'y';
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    entry_point = next(iter(ast.stages.values())).entry_point

    assert entry_point.body.statements[0].initial_value.value is True
    assert entry_point.body.statements[1].initial_value.value is False

    generated_code = generate_code(ast)

    assert "let enabled: bool = true;" in generated_code
    assert "let disabled: bool = false;" in generated_code
    assert 'let mut label: &\'static str = "debug";' in generated_code
    assert "let mut marker: char = 'x';" in generated_code
    assert 'label = "active";' in generated_code
    assert "marker = 'y';" in generated_code
    assert "True" not in generated_code
    assert "False" not in generated_code


def test_local_bindings_are_mutable_only_when_reassigned_or_mutated():
    code = """
    shader LocalMutability {
        struct Output {
            value: int;
        }

        void main(int index) {
            int immutableValue = 1;
            int reassignedValue = 2;
            Output output;
            float values[2];

            reassignedValue = immutableValue;
            output.value = reassignedValue;
            values[index] = 1.0;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "let immutableValue: i32 = 1;" in generated_code
    assert "let mut reassignedValue: i32 = 2;" in generated_code
    assert "let mut output: Output = Default::default();" in generated_code
    assert "let mut values: [f32; 2];" in generated_code
    assert "let mut immutableValue" not in generated_code


def test_direct_literal_nodes_emit_rust_escaping():
    codegen = RustCodeGen()

    assert (
        codegen.generate_expression(LiteralNode(True, PrimitiveType("bool"))) == "true"
    )
    assert (
        codegen.generate_expression(LiteralNode('debug"name', PrimitiveType("string")))
        == '"debug\\"name"'
    )
    assert (
        codegen.generate_expression(LiteralNode("'", PrimitiveType("char"))) == "'\\''"
    )


def test_else_if_statement():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                if (input.texCoord.x > 0.5) {
                    output.color = vec4(1.0, 1.0, 1.0, 1.0);
                } else if (input.texCoord.x < 0.5) {
                    output.color = vec4(0.0, 0.0, 0.0, 1.0);
                } else {
                    output.color = vec4(0.5, 0.5, 0.5, 1.0);
                }
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                let brightness = texture(iChannel0, input.color.xy).r;
                let bloom = max(0.0, brightness - 0.5);
                if (bloom > 0.5) {
                    bloom = 0.5;
                } else if (bloom < 0.5) {
                    bloom = 0.0;
                } else {
                    bloom = 0.5;
                }
                let texColor = texture(iChannel0, input.color.xy).rgb;
                let colorWithBloom = texColor + vec3(bloom);
                return vec4(colorWithBloom, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "else if " in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Else if codegen not implemented.")


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    float result = add(1.0, 2.0);
                }
                
                float add(float a, float b) {
                    return a + b;
                }
            }
            """,
            "add(1.0, 2.0)",
        )
    ],
)
def test_function_call(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = RustCodeGen()
    generated_code = code_gen.generate(ast)
    assert expected_output in generated_code


def test_builtin_function_call_names_are_mapped():
    code = """
    shader main {
        fragment {
            float main() {
                float a = mix(0.0, 1.0, 0.25);
                float b = mod(5.0, 2.0);
                float c = frac(1.25);
                float d = saturate(2.0);
                float e = saturate(frac(1.75));
                float f = inversesqrt(4.0);
                return a + b + c + d + e + f;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "lerp(0.0, 1.0, 0.25)" in generated_code
    assert "modulo(5.0, 2.0)" in generated_code
    assert "fract(1.25)" in generated_code
    assert "clamp(2.0, 0.0, 1.0)" in generated_code
    assert "clamp(fract(1.75), 0.0, 1.0)" in generated_code
    assert "rsqrt(4.0)" in generated_code
    assert "mix(0.0, 1.0, 0.25)" not in generated_code
    assert "mod(5.0, 2.0)" not in generated_code
    assert "frac(1.25)" not in generated_code
    assert "inversesqrt(" not in generated_code
    assert "saturate(" not in generated_code


def test_builtin_function_calls_infer_rust_value_types_and_smoke_compile(tmp_path):
    code = """
    shader BuiltinInference {
        fragment {
            uniform sampler2D main_texture;

            vec4 main(vec3 normal, vec3 light, vec3 albedo, vec2 uv) {
                let unit = normalize(normal);
                let reflected = reflect(-unit, light);
                let ndotl = max(0.0, dot(reflected, light));
                let gloss = pow(ndotl, 2.0);
                let quantized = floor(gloss);
                let smoothed = smoothstep(0.0, 1.0, quantized);
                let mixed = mix(vec3(0.04), albedo, smoothed);
                let tex = texture(main_texture, uv);
                let color = mixed + tex.rgb * smoothed;
                return vec4(color, tex.a);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "let unit: Vec3<f32> = normalize(normal);" in generated_code
    assert "let reflected: Vec3<f32> = reflect((-unit), light);" in generated_code
    assert "let ndotl: f32 = max(0.0, dot(reflected, light));" in generated_code
    assert "let mixed: Vec3<f32> = lerp(" in generated_code
    assert "let tex: Vec4<f32> = sample(*MAIN_TEXTURE, uv);" in generated_code
    assert "let unit: f32 = normalize" not in generated_code
    assert "let reflected: f32 = reflect" not in generated_code
    assert "let mixed: f32 = lerp" not in generated_code
    assert "let tex: f32 = sample" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_scalar_left_vector_arithmetic_lowers_to_lane_constructors(tmp_path):
    code = """
    shader ScalarVectorOps {
        fragment {
            float nextScale() {
                return 1.0;
            }

            vec3 nextColor() {
                return vec3(0.25, 0.5, 0.75);
            }

            vec4 main(vec3 color, float exposure) {
                let inverted = 1.0 - color;
                let reciprocal = 1.0 / color;
                let scaled = exposure * color;
                let shifted = exposure + color;
                let complex = nextScale() - nextColor();
                let combined = inverted + reciprocal + scaled + shifted + complex;
                return vec4(combined, 1.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "let inverted: Vec3<f32> = "
        "Vec3::<f32>::new((1.0 - color.x), (1.0 - color.y), (1.0 - color.z));"
        in generated_code
    )
    assert (
        "let reciprocal: Vec3<f32> = "
        "Vec3::<f32>::new((1.0 / color.x), (1.0 / color.y), (1.0 / color.z));"
        in generated_code
    )
    assert (
        "let scaled: Vec3<f32> = "
        "Vec3::<f32>::new((exposure * color.x), "
        "(exposure * color.y), (exposure * color.z));" in generated_code
    )
    assert (
        "let shifted: Vec3<f32> = "
        "Vec3::<f32>::new((exposure + color.x), "
        "(exposure + color.y), (exposure + color.z));" in generated_code
    )
    assert (
        "let complex: Vec3<f32> = "
        "{ let __cgl_vec_arg_0 = nextScale(); let __cgl_vec_arg_1 = nextColor(); "
        "Vec3::<f32>::new((__cgl_vec_arg_0 - __cgl_vec_arg_1.x), "
        "(__cgl_vec_arg_0 - __cgl_vec_arg_1.y), "
        "(__cgl_vec_arg_0 - __cgl_vec_arg_1.z)) };" in generated_code
    )
    assert "(1.0 - color)" not in generated_code
    assert "(1.0 / color)" not in generated_code
    assert "(exposure * color)" not in generated_code
    assert "(exposure + color)" not in generated_code
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_lambda_call_emits_native_rust_closure():
    code = """
    shader main {
        void apply(Values values) {
            let mapped = map(values, lambda(x, { prepare(x); return (x + 1); }));
            let folded = fold(values, 0, lambda(int acc, int x, (acc + x)));
            let parsed = map(values, lambda(Result<i32, i32> value, { return value; }));
            let always = lambda(true);
        }
    }
    """

    ast = crosstl.translator.parse(code)
    generated_code = generate_code(ast)

    assert "map(values, |x| { prepare(x); return (x + 1); })" in generated_code
    assert "fold(values, 0, |acc: i32, x: i32| (acc + x))" in generated_code
    assert "map(values, |value: Result<i32, i32>| { return value; })" in generated_code
    assert "|| true" in generated_code
    assert "lambda(" not in generated_code


def test_user_defined_mix_function_is_not_lowered_to_lerp():
    code = """
    shader main {
        fragment {
            float mix(float x, float y, float t) {
                return x + y + t;
            }

            float main() {
                float adjusted = mix(0.0, 1.0, 0.25);
                return adjusted;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "pub fn mix(x: f32, y: f32, t: f32) -> f32" in generated_code
    assert "let adjusted: f32 = mix(0.0, 1.0, 0.25);" in generated_code
    assert "let adjusted: f32 = lerp(0.0, 1.0, 0.25);" not in generated_code


def test_bool_scalar_mix_lowers_to_rust_select():
    code = """
    shader main {
        fragment {
            float nextX() {
                return 0.0;
            }

            float nextY() {
                return 1.0;
            }

            bool nextFlag() {
                return true;
            }

            float main() {
                bool flag = true;
                float literal = mix(0.0, 1.0, flag);
                float complexValue = mix(nextX(), nextY(), nextFlag());
                return literal + complexValue;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let literal: f32 = (if flag { 1.0 } else { 0.0 });" in generated_code
    assert (
        "let complexValue: f32 = "
        "(if nextFlag() { nextY() } else { nextX() });" in generated_code
    )
    assert "lerp(0.0, 1.0, flag)" not in generated_code
    assert "lerp(nextX(), nextY(), nextFlag())" not in generated_code
    assert "mix(0.0, 1.0, flag)" not in generated_code
    assert "mix(nextX(), nextY(), nextFlag())" not in generated_code


def test_bool_vector_mix_lowers_to_rust_lane_selectors():
    code = """
    bvec3 makeMask() {
        return bvec3(true, false, true);
    }

    vec3 makeA() {
        return vec3(1.0, 2.0, 3.0);
    }

    vec3 makeB() {
        return vec3(4.0, 5.0, 6.0);
    }

    void probe(bvec3 mask, vec3 a, vec3 b) {
        vec3 selected = mix(a, b, mask);
        vec3 complexSelected = mix(makeA(), makeB(), makeMask());
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let selected: Vec3<f32> = "
        "Vec3::<f32>::new((if mask.x { b.x } else { a.x }), "
        "(if mask.y { b.y } else { a.y }), "
        "(if mask.z { b.z } else { a.z }));" in generated_code
    )
    assert (
        "let complexSelected: Vec3<f32> = "
        "{ let __cgl_vec_arg_0 = makeMask(); let __cgl_vec_arg_1 = makeB(); "
        "let __cgl_vec_arg_2 = makeA(); "
        "Vec3::<f32>::new((if __cgl_vec_arg_0.x { __cgl_vec_arg_1.x } "
        "else { __cgl_vec_arg_2.x }), "
        "(if __cgl_vec_arg_0.y { __cgl_vec_arg_1.y } "
        "else { __cgl_vec_arg_2.y }), "
        "(if __cgl_vec_arg_0.z { __cgl_vec_arg_1.z } "
        "else { __cgl_vec_arg_2.z })) };" in generated_code
    )
    assert "lerp(a, b, mask)" not in generated_code
    assert "lerp(makeA(), makeB(), makeMask())" not in generated_code
    assert "mix(a, b, mask)" not in generated_code
    assert "mix(makeA(), makeB(), makeMask())" not in generated_code


def test_user_defined_function_call_scalar_args_cast_to_param_types():
    code = """
    float takesScalars(float weight, double precise, uint count, bool enabled) {
        return weight;
    }

    void probe(int index, float weight, bool ready) {
        float result = takesScalars(index, weight, index, index);
        float fromBool = takesScalars(ready, 1.0, 2, ready);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "pub fn takesScalars(weight: f32, precise: f64, count: u32, "
        "enabled: bool) -> f32" in generated_code
    )
    assert (
        "let result: f32 = "
        "takesScalars((index as f32), (weight as f64), "
        "(index as u32), (index != 0));" in generated_code
    )
    assert (
        "let fromBool: f32 = "
        "takesScalars((if ready { 1.0 } else { 0.0 }), "
        "(1.0 as f64), 2, ready);" in generated_code
    )
    assert "takesScalars(index, weight, index, index)" not in generated_code
    assert "takesScalars(ready, 1.0, 2, ready)" not in generated_code


def test_user_defined_function_call_vector_args_cast_to_param_types():
    code = """
    float consumeVectors(vec2 coords, uvec3 ids, bvec2 mask) {
        return coords.x;
    }

    void probe(ivec2 pixel, vec3 weights, vec2 amount, bool ready) {
        float same = consumeVectors(amount, uvec3(1, 2, 3), bvec2(ready, false));
        float converted = consumeVectors(pixel, weights, amount);
        float constructed = consumeVectors(
            ivec2(1, 2),
            vec3(1.0, 2.0, 3.0),
            vec2(1.0, 0.0)
        );
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let same: f32 = "
        "consumeVectors(amount, Vec3::<u32>::new(1, 2, 3), "
        "Vec2::<bool>::new(ready, false));" in generated_code
    )
    assert (
        "let converted: f32 = "
        "consumeVectors(Vec2::<f32>::new((pixel.x as f32), (pixel.y as f32)), "
        "Vec3::<u32>::new((weights.x as u32), (weights.y as u32), "
        "(weights.z as u32)), "
        "Vec2::<bool>::new((amount.x != 0.0), (amount.y != 0.0)));" in generated_code
    )
    assert (
        "consumeVectors({ let __cgl_vec_arg_0 = Vec2::<i32>::new(1, 2); "
        "Vec2::<f32>::new((__cgl_vec_arg_0.x as f32), "
        "(__cgl_vec_arg_0.y as f32)) }, "
        "{ let __cgl_vec_arg_1 = Vec3::<f32>::new(1.0, 2.0, 3.0); "
        "Vec3::<u32>::new((__cgl_vec_arg_1.x as u32), "
        "(__cgl_vec_arg_1.y as u32), (__cgl_vec_arg_1.z as u32)) }, "
        "{ let __cgl_vec_arg_2 = Vec2::<f32>::new(1.0, 0.0); "
        "Vec2::<bool>::new((__cgl_vec_arg_2.x != 0.0), "
        "(__cgl_vec_arg_2.y != 0.0)) });" in generated_code
    )
    assert "consumeVectors(pixel, weights, amount)" not in generated_code
    assert "consumeVectors(Vec2::<i32>::new(1, 2)" not in generated_code


def test_user_defined_function_call_matrix_args_cast_to_param_types():
    code = """
    float consumeMatrices(dmat2 precise, mat2 regular) {
        return regular.c0.x;
    }

    void probe(mat2 transform, dmat2 preciseInput, float weight, int index) {
        float same = consumeMatrices(preciseInput, transform);
        float converted = consumeMatrices(transform, preciseInput);
        float constructed = consumeMatrices(
            mat2(weight, index, 2, 1.0),
            dmat2(1.0, 0.0, 0.0, 1.0)
        );
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let same: f32 = consumeMatrices(preciseInput, transform);" in generated_code
    assert (
        "let converted: f32 = "
        "consumeMatrices(Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)), "
        "Mat2::<f32>::new((preciseInput.c0.x as f32), "
        "(preciseInput.c0.y as f32), (preciseInput.c1.x as f32), "
        "(preciseInput.c1.y as f32)));" in generated_code
    )
    assert (
        "consumeMatrices({ let __cgl_mat_arg_0 = "
        "Mat2::<f32>::new(weight, (index as f32), 2.0, 1.0); "
        "Mat2::<f64>::new((__cgl_mat_arg_0.c0.x as f64), "
        "(__cgl_mat_arg_0.c0.y as f64), "
        "(__cgl_mat_arg_0.c1.x as f64), "
        "(__cgl_mat_arg_0.c1.y as f64)) }, "
        "{ let __cgl_mat_arg_1 = Mat2::<f64>::new(1.0, 0.0, 0.0, 1.0); "
        "Mat2::<f32>::new((__cgl_mat_arg_1.c0.x as f32), "
        "(__cgl_mat_arg_1.c0.y as f32), "
        "(__cgl_mat_arg_1.c1.x as f32), "
        "(__cgl_mat_arg_1.c1.y as f32)) });" in generated_code
    )
    assert "consumeMatrices(transform, preciseInput)" not in generated_code
    assert "consumeMatrices(Mat2::<f32>::new(weight" not in generated_code


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    a |= 2;
                }
            }
            """,
            "a |= 2",
        )
    ],
)
def test_assignment_or_operator(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = RustCodeGen()
    generated_code = code_gen.generate(ast)
    assert expected_output in generated_code


def test_assignment_modulus_operator():
    code = """
    shader main {
        vertex {
            void main() {
                int a = 10;
                a %= 3;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "a %= 3" in generated_code or "a = a % 3" in generated_code
    except SyntaxError:
        pytest.fail("Assignment modulus operator codegen not implemented.")


def test_assignment_xor_operator():
    code = """
    shader main {
        vertex {
            void main() {
                int a = 5;
                a ^= 3;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "a ^= 3" in generated_code or "a = a ^ 3" in generated_code
    except SyntaxError:
        pytest.fail("Assignment XOR operator codegen not implemented.")


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    a <<= 2;
                    a >>= 1;
                }
            }
            """,
            ["a <<= 2", "a >>= 1"],
        )
    ],
)
def test_assignment_shift_operators(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = RustCodeGen()
    generated_code = code_gen.generate(ast)
    for output in expected_output:
        assert output in generated_code


@pytest.mark.parametrize(
    "shader, expected_outputs",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    int b = 2;
                    int c = a | b;
                    int d = a & b;
                    int e = a ^ b;
                }
            }
            """,
            ["a | b", "a & b", "a ^ b"],
        )
    ],
)
def test_bitwise_operators(shader, expected_outputs):
    ast = crosstl.translator.parse(shader)
    code_gen = RustCodeGen()
    generated_code = code_gen.generate(ast)
    for expected in expected_outputs:
        assert expected in generated_code


def test_bitwise_and_operator():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(f32(i32(input.texCoord.x * 100.0) & 15), 
                                    f32(i32(input.texCoord.y * 100.0) & 15), 
                                    0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.color.rgb, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise AND codegen not implemented")


def test_double_data_type():
    code = """
    shader DoubleShader {
        struct VSInput {
            double texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            double color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = input.texCoord * 2.0;
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.color, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "f64" in generated_code  # Rust uses f64 for double
        print(generated_code)
    except SyntaxError:
        pytest.fail("Double data type not supported.")


@pytest.mark.parametrize(
    "shader, expected_outputs",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    int b = 2;
                    int c = a << b;
                    int d = a >> b;
                }
            }
            """,
            ["a << b", "a >> b"],
        )
    ],
)
def test_shift_operators(shader, expected_outputs):
    ast = crosstl.translator.parse(shader)
    code_gen = RustCodeGen()
    generated_code = code_gen.generate(ast)
    for expected in expected_outputs:
        assert expected in generated_code


def test_bitwise_or_operator():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(f32(i32(input.texCoord.x * 100.0) | 15), 
                                    f32(i32(input.texCoord.y * 100.0) | 15), 
                                    0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.color.rgb, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise OR codegen not implemented")


def test_ternary_operator():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(input.texCoord.x > 0.5 ? 1.0 : 0.0,
                                    input.texCoord.y > 0.5 ? 1.0 : 0.0,
                                    0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert (
            "if" in generated_code and "else" in generated_code
        )  # Rust converts ternary to if-else
        print(generated_code)
    except SyntaxError:
        pytest.fail("Ternary operator codegen not implemented")


def test_vector_constructor():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(1.0, 0.5, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "Vec4::<f32>::new" in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Vector constructor codegen not implemented")


def test_double_vector_and_matrix_types_emit_rust_names():
    code = """
    shader main {
        compute {
            void main() {
                dvec2 preciseUV = dvec2(1.0, 2.0);
                bvec2 mask = bvec2(true, false);
                bvec3 flags;
                mat2 transform = mat2(1.0, 0.0, 0.0, 1.0);
                mat3x4 affine;
                dmat2 precise = dmat2(1.0, 0.0, 0.0, 1.0);
                dmat4x3 jacobian;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let preciseUV: Vec2<f64> = Vec2::<f64>::new(1.0, 2.0);" in generated_code
    assert "let mask: Vec2<bool> = Vec2::<bool>::new(true, false);" in generated_code
    assert "let flags: Vec3<bool>;" in generated_code
    assert (
        "let transform: Mat2<f32> = " "Mat2::<f32>::new(1.0, 0.0, 0.0, 1.0);"
    ) in generated_code
    assert "let affine: Mat3x4<f32>;" in generated_code
    assert (
        "let precise: Mat2<f64> = " "Mat2::<f64>::new(1.0, 0.0, 0.0, 1.0);"
    ) in generated_code
    assert "let jacobian: Mat4x3<f64>;" in generated_code
    assert "dvec2(" not in generated_code
    assert "bvec2(" not in generated_code
    assert "bool2" not in generated_code
    assert "dmat2(" not in generated_code
    assert "MatrixType(" not in generated_code


def test_matrix_constructors_flatten_vector_args_once():
    code = """
    vec2 makeCol0() {
        return vec2(1.0, 0.0);
    }

    vec2 makeCol1() {
        return vec2(0.0, 1.0);
    }

    void probe() {
        vec2 a = vec2(1.0, 0.0);
        vec2 b = vec2(0.0, 1.0);
        mat2 localMatrix = mat2(a, b);
        mat2 complexMatrix = mat2(makeCol0(), makeCol1());
        mat2 mixedMatrix = mat2(makeCol0().xy, b);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let localMatrix: Mat2<f32> = "
        "Mat2::<f32>::new(a.x, a.y, b.x, b.y);" in generated_code
    )
    assert (
        "let complexMatrix: Mat2<f32> = "
        "{ let __cgl_vec_arg_0 = makeCol0(); let __cgl_vec_arg_1 = makeCol1(); "
        "Mat2::<f32>::new(__cgl_vec_arg_0.x, __cgl_vec_arg_0.y, "
        "__cgl_vec_arg_1.x, __cgl_vec_arg_1.y) };" in generated_code
    )
    assert (
        "let mixedMatrix: Mat2<f32> = "
        "{ let __cgl_vec_arg_2 = makeCol0(); "
        "Mat2::<f32>::new(__cgl_vec_arg_2.x, __cgl_vec_arg_2.y, b.x, b.y) };"
        in generated_code
    )
    assert "Mat2::<f32>::new(a, b)" not in generated_code
    assert "Mat2::<f32>::new(makeCol0(), makeCol1())" not in generated_code
    assert "makeCol0().xy" not in generated_code


def test_mixed_scalar_matrix_constructor_lanes_cast_to_component_type():
    code = """
    void probe(float weight, int index, uint count) {
        mat2 mixedFloat = mat2(weight, index, 2, 1.0);
        dmat2 mixedDouble = dmat2(weight, index, 2, 1.0);
        mat2 fromIntVector = mat2(ivec2(index, 2), vec2(weight, 1.0));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let mixedFloat: Mat2<f32> = "
        "Mat2::<f32>::new(weight, (index as f32), 2.0, 1.0);" in generated_code
    )
    assert (
        "let mixedDouble: Mat2<f64> = "
        "Mat2::<f64>::new((weight as f64), (index as f64), 2.0, 1.0);" in generated_code
    )
    assert (
        "let fromIntVector: Mat2<f32> = "
        "{ let __cgl_vec_arg_0 = Vec2::<i32>::new(index, 2); "
        "let __cgl_vec_arg_1 = Vec2::<f32>::new(weight, 1.0); "
        "Mat2::<f32>::new((__cgl_vec_arg_0.x as f32), "
        "(__cgl_vec_arg_0.y as f32), __cgl_vec_arg_1.x, __cgl_vec_arg_1.y) };"
        in generated_code
    )
    assert "Mat2::<f32>::new(weight, index, 2, 1.0)" not in generated_code
    assert "Mat2::<f64>::new(weight, index, 2, 1.0)" not in generated_code
    assert "Mat2::<f32>::new(__cgl_vec_arg_0.x, __cgl_vec_arg_0.y" not in generated_code


def test_inferred_matrix_constructor_bindings_use_matrix_types():
    code = """
    void probe() {
        let transform = mat2(1.0, 0.0, 0.0, 1.0);
        let affine = mat3x4(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0
        );
        let precise = dmat2(1.0, 0.0, 0.0, 1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let transform: Mat2<f32> = "
        "Mat2::<f32>::new(1.0, 0.0, 0.0, 1.0);" in generated_code
    )
    assert "let affine: Mat3x4<f32> = Mat3x4::<f32>::new(" in generated_code
    assert (
        "let precise: Mat2<f64> = "
        "Mat2::<f64>::new(1.0, 0.0, 0.0, 1.0);" in generated_code
    )
    assert "let transform: f32 = Mat2" not in generated_code
    assert "let affine: f32 = Mat3x4" not in generated_code
    assert "let precise: f32 = Mat2::<f64>" not in generated_code


def test_inferred_matrix_binary_bindings_prefer_matrix_operands():
    code = """
    void probe() {
        let leftScaled = mat2(1.0, 0.0, 0.0, 1.0) * 2.0;
        let rightScaled = 2.0 * mat2(1.0, 0.0, 0.0, 1.0);
        let combined = mat2(1.0, 0.0, 0.0, 1.0)
            + mat2(2.0, 0.0, 0.0, 2.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let leftScaled: Mat2<f32> = "
        "(Mat2::<f32>::new(1.0, 0.0, 0.0, 1.0) * 2.0);" in generated_code
    )
    assert (
        "let rightScaled: Mat2<f32> = "
        "(2.0 * Mat2::<f32>::new(1.0, 0.0, 0.0, 1.0));" in generated_code
    )
    assert "let combined: Mat2<f32> = " in generated_code
    assert "let rightScaled: f32 = " not in generated_code


def test_mixed_vector_and_matrix_binary_operands_promote_before_operator():
    code = """
    void probe(ivec2 pixel, vec2 amount, mat2 transform, dmat2 preciseInput) {
        let inferredVec = pixel + amount;
        let inferredVecReverse = amount + pixel;
        vec2 declaredVec = pixel + amount;
        let inferredMat = transform + preciseInput;
        let inferredMatReverse = preciseInput + transform;
        dmat2 declaredMat = transform + preciseInput;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let inferredVec: Vec2<f32> = "
        "(Vec2::<f32>::new((pixel.x as f32), (pixel.y as f32)) + amount);"
        in generated_code
    )
    assert (
        "let inferredVecReverse: Vec2<f32> = "
        "(amount + Vec2::<f32>::new((pixel.x as f32), (pixel.y as f32)));"
        in generated_code
    )
    assert (
        "let declaredVec: Vec2<f32> = "
        "(Vec2::<f32>::new((pixel.x as f32), (pixel.y as f32)) + amount);"
        in generated_code
    )
    assert (
        "let inferredMat: Mat2<f64> = "
        "(Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)) + preciseInput);" in generated_code
    )
    assert (
        "let inferredMatReverse: Mat2<f64> = "
        "(preciseInput + Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)));" in generated_code
    )
    assert (
        "let declaredMat: Mat2<f64> = "
        "(Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)) + preciseInput);" in generated_code
    )
    assert "(pixel + amount)" not in generated_code
    assert "(transform + preciseInput)" not in generated_code
    assert "let inferredVec: Vec2<i32>" not in generated_code
    assert "let inferredMat: Mat2<f32>" not in generated_code
    assert "{ let __cgl_vec_arg_" not in generated_code
    assert "{ let __cgl_mat_arg_" not in generated_code


def test_mixed_vector_and_matrix_scalar_binary_operands_cast_to_components():
    code = """
    void probe(
        ivec2 pixel,
        vec2 amount,
        mat2 transform,
        dmat2 preciseInput,
        int index,
        float weight,
        double precise
    ) {
        let inferredVec = pixel + weight;
        let inferredVecReverse = weight + pixel;
        vec2 declaredVec = pixel + weight;
        let inferredDoubleVec = amount + precise;
        let indexedVec = amount + index;
        let inferredMat = transform + precise;
        let inferredMatReverse = precise + transform;
        dmat2 declaredMat = transform + precise;
        let indexedMat = transform + index;
        let sameVec = amount + weight;
        let sameMat = preciseInput + precise;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let inferredVec: Vec2<f32> = "
        "(Vec2::<f32>::new((pixel.x as f32), (pixel.y as f32)) + weight);"
        in generated_code
    )
    assert (
        "let inferredVecReverse: Vec2<f32> = "
        "Vec2::<f32>::new((weight + (pixel.x as f32)), "
        "(weight + (pixel.y as f32)));" in generated_code
    )
    assert (
        "let declaredVec: Vec2<f32> = "
        "(Vec2::<f32>::new((pixel.x as f32), (pixel.y as f32)) + weight);"
        in generated_code
    )
    assert (
        "let inferredDoubleVec: Vec2<f64> = "
        "(Vec2::<f64>::new((amount.x as f64), (amount.y as f64)) + precise);"
        in generated_code
    )
    assert "let indexedVec: Vec2<f32> = (amount + (index as f32));" in generated_code
    assert (
        "let inferredMat: Mat2<f64> = "
        "(Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)) + precise);" in generated_code
    )
    assert (
        "let inferredMatReverse: Mat2<f64> = "
        "(precise + Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)));" in generated_code
    )
    assert (
        "let declaredMat: Mat2<f64> = "
        "(Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)) + precise);" in generated_code
    )
    assert "let indexedMat: Mat2<f32> = (transform + (index as f32));" in generated_code
    assert "let sameVec: Vec2<f32> = (amount + weight);" in generated_code
    assert "let sameMat: Mat2<f64> = (preciseInput + precise);" in generated_code
    assert "(pixel + weight)" not in generated_code
    assert "(transform + precise)" not in generated_code
    assert "{ let __cgl_vec_arg_" not in generated_code
    assert "{ let __cgl_mat_arg_" not in generated_code


def test_matrix_vector_binary_operands_promote_components_and_result_size():
    code = """
    void probe(
        mat2 transform,
        dmat2 preciseTransform,
        vec2 amount,
        dvec2 preciseAmount,
        ivec2 pixel,
        mat2x3 affine,
        vec2 uv,
        vec3 normal
    ) {
        let inferredMatVec = transform * preciseAmount;
        let inferredMatVecReverseComponents = preciseTransform * amount;
        let inferredMatIntVec = transform * pixel;
        dvec2 declaredMatVec = transform * amount;
        let inferredVecMat = preciseAmount * transform;
        let inferredVecMatReverseComponents = amount * preciseTransform;
        dvec2 declaredVecMat = amount * transform;
        let rectMatVec = affine * uv;
        let rectVecMat = normal * affine;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let inferredMatVec: Vec2<f64> = "
        "(Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)) * preciseAmount);" in generated_code
    )
    assert (
        "let inferredMatVecReverseComponents: Vec2<f64> = "
        "(preciseTransform * Vec2::<f64>::new((amount.x as f64), "
        "(amount.y as f64)));" in generated_code
    )
    assert (
        "let inferredMatIntVec: Vec2<f32> = "
        "(transform * Vec2::<f32>::new((pixel.x as f32), "
        "(pixel.y as f32)));" in generated_code
    )
    assert (
        "let declaredMatVec: Vec2<f64> = "
        "(Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)) * Vec2::<f64>::new((amount.x as f64), "
        "(amount.y as f64)));" in generated_code
    )
    assert (
        "let inferredVecMat: Vec2<f64> = "
        "(preciseAmount * Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)));" in generated_code
    )
    assert (
        "let inferredVecMatReverseComponents: Vec2<f64> = "
        "(Vec2::<f64>::new((amount.x as f64), (amount.y as f64)) "
        "* preciseTransform);" in generated_code
    )
    assert (
        "let declaredVecMat: Vec2<f64> = "
        "(Vec2::<f64>::new((amount.x as f64), (amount.y as f64)) "
        "* Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)));" in generated_code
    )
    assert "let rectMatVec: Vec3<f32> = (affine * uv);" in generated_code
    assert "let rectVecMat: Vec2<f32> = (normal * affine);" in generated_code
    assert "let inferredMatVecReverseComponents: Vec2<f32>" not in generated_code
    assert "let inferredMatIntVec: Vec2<i32>" not in generated_code
    assert "let rectMatVec: Vec2<f32>" not in generated_code
    assert "let rectVecMat: Vec3<f32>" not in generated_code
    assert "{ let __cgl_vec_arg_" not in generated_code
    assert "{ let __cgl_mat_arg_" not in generated_code


def test_vector_comparison_binary_operands_emit_boolean_lanes():
    code = """
    float makeWeight() {
        return 0.5;
    }

    vec2 makeAmount() {
        return vec2(1.0, 2.0);
    }

    void probe(
        int index,
        uint count,
        float weight,
        double precise,
        ivec2 pixel,
        vec2 amount,
        bvec2 mask
    ) {
        let scalarCompare = index < weight;
        let scalarEqual = count == index;
        bool declaredCompare = index < precise;
        let vectorEqual = pixel == amount;
        let vectorNotEqual = amount != pixel;
        bvec2 declaredMask = pixel < amount;
        let vectorScalarCompare = pixel < weight;
        let scalarVectorCompare = weight < pixel;
        let boolVectorEqual = mask == pixel;
        let complexCompare = makeAmount() > makeWeight();
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let scalarCompare: bool = ((index as f32) < weight);" in generated_code
    assert "let scalarEqual: bool = (count == (index as u32));" in generated_code
    assert "let declaredCompare: bool = ((index as f64) < precise);" in generated_code
    assert (
        "let vectorEqual: Vec2<bool> = "
        "Vec2::<bool>::new(((pixel.x as f32) == amount.x), "
        "((pixel.y as f32) == amount.y));" in generated_code
    )
    assert (
        "let vectorNotEqual: Vec2<bool> = "
        "Vec2::<bool>::new((amount.x != (pixel.x as f32)), "
        "(amount.y != (pixel.y as f32)));" in generated_code
    )
    assert (
        "let declaredMask: Vec2<bool> = "
        "Vec2::<bool>::new(((pixel.x as f32) < amount.x), "
        "((pixel.y as f32) < amount.y));" in generated_code
    )
    assert (
        "let vectorScalarCompare: Vec2<bool> = "
        "Vec2::<bool>::new(((pixel.x as f32) < weight), "
        "((pixel.y as f32) < weight));" in generated_code
    )
    assert (
        "let scalarVectorCompare: Vec2<bool> = "
        "Vec2::<bool>::new((weight < (pixel.x as f32)), "
        "(weight < (pixel.y as f32)));" in generated_code
    )
    assert (
        "let boolVectorEqual: Vec2<bool> = "
        "Vec2::<bool>::new(((mask.x as i32) == pixel.x), "
        "((mask.y as i32) == pixel.y));" in generated_code
    )
    assert (
        "let complexCompare: Vec2<bool> = "
        "{ let __cgl_vec_arg_0 = makeAmount(); "
        "let __cgl_vec_arg_1 = makeWeight(); "
        "Vec2::<bool>::new((__cgl_vec_arg_0.x > __cgl_vec_arg_1), "
        "(__cgl_vec_arg_0.y > __cgl_vec_arg_1)) };" in generated_code
    )
    assert "let vectorEqual: Vec2<i32>" not in generated_code
    assert "let vectorNotEqual: Vec2<f32>" not in generated_code
    assert "(pixel < amount)" not in generated_code


def test_bool_vector_logical_binary_operands_emit_boolean_lanes():
    code = """
    bvec3 makeMask() {
        return bvec3(true, false, true);
    }

    bool makeFlag() {
        return true;
    }

    void probe(bool ready, bvec3 a, bvec3 b) {
        bool scalarAnd = ready && makeFlag();
        let both = a && b;
        let either = a || b;
        bvec3 scalarRight = a && ready;
        bvec3 scalarLeft = makeFlag() || b;
        let complex = makeMask() && makeFlag();
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let scalarAnd: bool = (ready && makeFlag());" in generated_code
    assert (
        "let both: Vec3<bool> = "
        "Vec3::<bool>::new((a.x && b.x), (a.y && b.y), (a.z && b.z));" in generated_code
    )
    assert (
        "let either: Vec3<bool> = "
        "Vec3::<bool>::new((a.x || b.x), (a.y || b.y), (a.z || b.z));" in generated_code
    )
    assert (
        "let scalarRight: Vec3<bool> = "
        "Vec3::<bool>::new((a.x && ready), (a.y && ready), (a.z && ready));"
        in generated_code
    )
    assert (
        "let scalarLeft: Vec3<bool> = "
        "{ let __cgl_vec_arg_0 = makeFlag(); "
        "Vec3::<bool>::new((__cgl_vec_arg_0 || b.x), "
        "(__cgl_vec_arg_0 || b.y), (__cgl_vec_arg_0 || b.z)) };" in generated_code
    )
    assert (
        "let complex: Vec3<bool> = "
        "{ let __cgl_vec_arg_1 = makeMask(); "
        "let __cgl_vec_arg_2 = makeFlag(); "
        "Vec3::<bool>::new((__cgl_vec_arg_1.x && __cgl_vec_arg_2), "
        "(__cgl_vec_arg_1.y && __cgl_vec_arg_2), "
        "(__cgl_vec_arg_1.z && __cgl_vec_arg_2)) };" in generated_code
    )
    assert "let both: Vec3<bool> = (a && b);" not in generated_code
    assert "let either: Vec3<bool> = (a || b);" not in generated_code
    assert "let scalarRight: Vec3<bool> = (a && ready);" not in generated_code
    assert "let scalarLeft: Vec3<bool> = (makeFlag() || b);" not in generated_code


def test_bool_vector_unary_not_emits_boolean_lanes():
    code = """
    bvec3 makeMask() {
        return bvec3(true, false, true);
    }

    void probe(bool flag, bvec2 mask2, bvec3 mask3, bvec4 mask4) {
        bool invFlag = !flag;
        bvec2 inv2 = !mask2;
        let inv3 = !mask3;
        bvec4 inv4 = !makeMask().xyzz;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let invFlag: bool = (!flag);" in generated_code
    assert (
        "let inv2: Vec2<bool> = "
        "Vec2::<bool>::new((!mask2.x), (!mask2.y));" in generated_code
    )
    assert (
        "let inv3: Vec3<bool> = "
        "Vec3::<bool>::new((!mask3.x), (!mask3.y), (!mask3.z));" in generated_code
    )
    assert (
        "let inv4: Vec4<bool> = "
        "{ let __cgl_vec_arg_0 = makeMask(); "
        "Vec4::<bool>::new((!__cgl_vec_arg_0.x), (!__cgl_vec_arg_0.y), "
        "(!__cgl_vec_arg_0.z), (!__cgl_vec_arg_0.z)) };" in generated_code
    )
    assert "let inv2: Vec2<bool> = (!mask2);" not in generated_code
    assert "let inv3: Vec3<bool> = (!mask3);" not in generated_code
    assert "!{ let __cgl_swizzle_" not in generated_code


def test_inferred_scalar_constructor_bindings_use_cast_types():
    code = """
    void probe() {
        let index = int(1.2);
        let unsignedIndex = uint(2);
        let precise = double(1.0);
        let enabled = bool(1);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let index: i32 = (1.2 as i32);" in generated_code
    assert "let unsignedIndex: u32 = (2 as u32);" in generated_code
    assert "let precise: f64 = (1.0 as f64);" in generated_code
    assert "let enabled: bool = (1 != 0);" in generated_code
    assert "let index: f32 = " not in generated_code
    assert "let enabled: f32 = " not in generated_code


def test_inferred_scalar_binary_bindings_promote_cast_operands():
    code = """
    void probe() {
        let rightUint = 3 + uint(2);
        let rightDouble = 2.0 + double(1.0);
        let compared = int(1.2) < 3;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let rightUint: u32 = (3 + (2 as u32));" in generated_code
    assert "let rightDouble: f64 = ((2.0 as f64) + (1.0 as f64));" in generated_code
    assert "let compared: bool = ((1.2 as i32) < 3);" in generated_code
    assert "let rightUint: i32 = " not in generated_code
    assert "let rightDouble: f32 = " not in generated_code


def test_mixed_scalar_binary_literals_match_float_operand_types():
    code = """
    void probe() {
        let intFloat = 3 + float(1);
        let floatInt = float(1) + 3;
        let intDouble = 2 + double(1.0);
        let doubleInt = double(1.0) + 2;
        let intUint = 3 + uint(2);
        let compareFloat = 3 < float(4);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let intFloat: f32 = (3.0 + (1 as f32));" in generated_code
    assert "let floatInt: f32 = ((1 as f32) + 3.0);" in generated_code
    assert "let intDouble: f64 = (2.0 + (1.0 as f64));" in generated_code
    assert "let doubleInt: f64 = ((1.0 as f64) + 2.0);" in generated_code
    assert "let intUint: u32 = (3 + (2 as u32));" in generated_code
    assert "let compareFloat: bool = (3.0 < (4 as f32));" in generated_code
    assert "(3 + (1 as f32))" not in generated_code
    assert "(3 < (4 as f32))" not in generated_code


def test_mixed_scalar_binary_variables_cast_to_promoted_operand_types():
    code = """
    void probe(int index, uint count, float weight, double precise) {
        let intFloat = index + float(1);
        let floatInt = weight + index;
        let intDouble = index + precise;
        let uintInt = count + index;
        let compareFloat = index < weight;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let intFloat: f32 = ((index as f32) + (1 as f32));" in generated_code
    assert "let floatInt: f32 = (weight + (index as f32));" in generated_code
    assert "let intDouble: f64 = ((index as f64) + precise);" in generated_code
    assert "let uintInt: u32 = (count + (index as u32));" in generated_code
    assert "let compareFloat: bool = ((index as f32) < weight);" in generated_code
    assert "(weight + index)" not in generated_code
    assert "(index < weight)" not in generated_code


def test_mixed_scalar_compound_assignments_cast_rhs_to_lhs_type():
    code = """
    void probe(int index, uint countInput, float weightInput, double preciseInput) {
        float weight = weightInput;
        double precise = preciseInput;
        uint count = countInput;
        weight += index;
        precise += index;
        count += index;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "weight += (index as f32);" in generated_code
    assert "precise += (index as f64);" in generated_code
    assert "count += (index as u32);" in generated_code
    assert "weight += index;" not in generated_code
    assert "precise += index;" not in generated_code
    assert "count += index;" not in generated_code


def test_mixed_scalar_simple_assignments_cast_rhs_to_lhs_type():
    code = """
    void probe(int index, uint countInput, float weightInput, double preciseInput) {
        float declaredWeight = index;
        double declaredPrecise = index;
        uint declaredCount = index;

        float weight = weightInput;
        double precise = preciseInput;
        uint count = countInput;
        weight = index;
        precise = index;
        count = index;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let declaredWeight: f32 = (index as f32);" in generated_code
    assert "let declaredPrecise: f64 = (index as f64);" in generated_code
    assert "let declaredCount: u32 = (index as u32);" in generated_code
    assert "weight = (index as f32);" in generated_code
    assert "precise = (index as f64);" in generated_code
    assert "count = (index as u32);" in generated_code
    assert "let declaredWeight: f32 = index;" not in generated_code
    assert "let declaredPrecise: f64 = index;" not in generated_code
    assert "let declaredCount: u32 = index;" not in generated_code
    assert "weight = index;" not in generated_code
    assert "precise = index;" not in generated_code
    assert "count = index;" not in generated_code


def test_vector_and_matrix_assignments_cast_rhs_to_lhs_component_types():
    code = """
    void probe(ivec2 pixel, vec2 amount, mat2 transform, dmat2 preciseInput) {
        vec2 declaredCoords = pixel;
        bvec2 declaredMask = amount;
        dmat2 declaredPrecise = transform;
        mat2 declaredRegular = preciseInput;

        vec2 coords = amount;
        dmat2 precise = preciseInput;
        coords = pixel;
        declaredMask = amount;
        precise = transform;
        declaredRegular = preciseInput;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let declaredCoords: Vec2<f32> = "
        "Vec2::<f32>::new((pixel.x as f32), (pixel.y as f32));" in generated_code
    )
    assert (
        "let mut declaredMask: Vec2<bool> = "
        "Vec2::<bool>::new((amount.x != 0.0), (amount.y != 0.0));" in generated_code
    )
    assert (
        "let declaredPrecise: Mat2<f64> = "
        "Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64));" in generated_code
    )
    assert (
        "let mut declaredRegular: Mat2<f32> = "
        "Mat2::<f32>::new((preciseInput.c0.x as f32), "
        "(preciseInput.c0.y as f32), (preciseInput.c1.x as f32), "
        "(preciseInput.c1.y as f32));" in generated_code
    )
    assert "let mut coords: Vec2<f32> = amount;" in generated_code
    assert "let mut precise: Mat2<f64> = preciseInput;" in generated_code
    assert (
        "coords = Vec2::<f32>::new((pixel.x as f32), (pixel.y as f32));"
        in generated_code
    )
    assert (
        "declaredMask = Vec2::<bool>::new((amount.x != 0.0), "
        "(amount.y != 0.0));" in generated_code
    )
    assert (
        "precise = Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64));" in generated_code
    )
    assert (
        "declaredRegular = Mat2::<f32>::new((preciseInput.c0.x as f32), "
        "(preciseInput.c0.y as f32), (preciseInput.c1.x as f32), "
        "(preciseInput.c1.y as f32));" in generated_code
    )
    assert "let declaredCoords: Vec2<f32> = pixel;" not in generated_code
    assert "coords = pixel;" not in generated_code
    assert "precise = transform;" not in generated_code
    assert "declaredRegular = preciseInput;" not in generated_code


def test_mixed_scalar_returns_cast_value_to_declared_return_type():
    code = """
    float toWeight(int index) {
        return index;
    }

    double toPrecise(int index) {
        return index;
    }

    uint toCount(int index) {
        return index;
    }

    bool toEnabled(int index) {
        return index;
    }

    float boolWeight(bool enabled) {
        return enabled;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "return (index as f32);" in generated_code
    assert "return (index as f64);" in generated_code
    assert "return (index as u32);" in generated_code
    assert "return (index != 0);" in generated_code
    assert "return (if enabled { 1.0 } else { 0.0 });" in generated_code
    assert "return index;" not in generated_code
    assert "return enabled;" not in generated_code


def test_numeric_conditions_compare_against_zero_in_rust_bool_sites():
    code = """
    float choose(int index, uint count, float inputWeight, bool ready) {
        int localIndex = index;
        float weight = inputWeight;
        if (localIndex) {
            weight = 1.0;
        } else if (count) {
            weight = 2.0;
        }
        while (weight) {
            weight -= 1.0;
        }
        for (int i = 3; i; i--) {
            weight += 1.0;
        }
        do {
            localIndex -= 1;
        } while (localIndex);
        let selected = localIndex ? weight : 0.0;
        if (!localIndex || ready && count) {
            weight = selected;
        }
        return weight;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "if (localIndex != 0) {" in generated_code
    assert "} else if (count != 0) {" in generated_code
    assert "while (weight != 0.0) {" in generated_code
    assert "while (i != 0) {" in generated_code
    assert "if !((localIndex != 0)) {" in generated_code
    assert (
        "let selected: f32 = (if (localIndex != 0) { weight } else { 0.0 });"
        in generated_code
    )
    assert "if (!((localIndex != 0)) || (ready && (count != 0))) {" in generated_code
    assert "if localIndex {" not in generated_code
    assert "else if count {" not in generated_code
    assert "while weight {" not in generated_code
    assert "while i {" not in generated_code


def test_mixed_scalar_ternary_branches_cast_to_result_type():
    code = """
    float chooseFloat(bool ready, int index, uint count, float weight) {
        float picked = ready ? index : weight;
        float reversed = ready ? weight : index;
        uint pickedCount = ready ? index : count;
        let inferred = ready ? index : weight;
        return ready ? weight : index;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let picked: f32 = (if ready { (index as f32) } else { weight });"
        in generated_code
    )
    assert (
        "let reversed: f32 = (if ready { weight } else { (index as f32) });"
        in generated_code
    )
    assert (
        "let pickedCount: u32 = (if ready { (index as u32) } else { count });"
        in generated_code
    )
    assert (
        "let inferred: f32 = (if ready { (index as f32) } else { weight });"
        in generated_code
    )
    assert "return (if ready { weight } else { (index as f32) });" in generated_code
    assert (
        "let picked: f32 = ((if ready { index } else { weight }) as f32);"
        not in generated_code
    )
    assert (
        "let pickedCount: u32 = ((if ready { index } else { count }) as u32);"
        not in generated_code
    )
    assert "return (if ready { weight } else { index });" not in generated_code


def test_vector_and_matrix_ternary_branches_cast_to_result_type():
    code = """
    vec2 chooseVec(bool ready, ivec2 pixel, vec2 amount) {
        vec2 declared = ready ? pixel : amount;
        bvec2 mask = ready ? amount : pixel;
        declared = ready ? amount : pixel;
        return ready ? pixel : amount;
    }

    dmat2 chooseMat(bool ready, mat2 transform, dmat2 preciseInput) {
        dmat2 precise = ready ? transform : preciseInput;
        mat2 regular = ready ? preciseInput : transform;
        precise = ready ? transform : preciseInput;
        return ready ? transform : preciseInput;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let mut declared: Vec2<f32> = "
        "(if ready { Vec2::<f32>::new((pixel.x as f32), "
        "(pixel.y as f32)) } else { amount });" in generated_code
    )
    assert (
        "let mask: Vec2<bool> = "
        "(if ready { Vec2::<bool>::new((amount.x != 0.0), "
        "(amount.y != 0.0)) } else { Vec2::<bool>::new((pixel.x != 0), "
        "(pixel.y != 0)) });" in generated_code
    )
    assert (
        "declared = (if ready { amount } else { "
        "Vec2::<f32>::new((pixel.x as f32), (pixel.y as f32)) });" in generated_code
    )
    assert (
        "return (if ready { Vec2::<f32>::new((pixel.x as f32), "
        "(pixel.y as f32)) } else { amount });" in generated_code
    )
    assert (
        "let mut precise: Mat2<f64> = "
        "(if ready { Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)) } else { preciseInput });" in generated_code
    )
    assert (
        "let regular: Mat2<f32> = "
        "(if ready { Mat2::<f32>::new((preciseInput.c0.x as f32), "
        "(preciseInput.c0.y as f32), (preciseInput.c1.x as f32), "
        "(preciseInput.c1.y as f32)) } else { transform });" in generated_code
    )
    assert (
        "precise = (if ready { Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)) } else { preciseInput });" in generated_code
    )
    assert (
        "return (if ready { Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)) } else { preciseInput });" in generated_code
    )
    assert "{ let __cgl_vec_arg_" not in generated_code
    assert "{ let __cgl_mat_arg_" not in generated_code
    assert "if ready { pixel } else { amount }" not in generated_code
    assert "if ready { transform } else { preciseInput }" not in generated_code


def test_bool_vector_ternary_conditions_emit_lane_selectors():
    code = """
    bvec3 makeMask() {
        return bvec3(true, false, true);
    }

    vec3 makeA() {
        return vec3(1.0, 2.0, 3.0);
    }

    float makeScalar() {
        return 0.5;
    }

    void probe(bool cond, bvec3 mask, vec3 a, vec3 b) {
        vec3 scalarSelected = cond ? a : b;
        vec3 laneSelected = mask ? a : b;
        vec3 scalarArms = mask ? 1.0 : 0.0;
        vec3 mixedArm = mask ? a : 0.0;
        let inferred = mask ? a : 0.0;
        let complex = makeMask() ? makeA() : makeScalar();
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let scalarSelected: Vec3<f32> = (if cond { a } else { b });" in generated_code
    )
    assert (
        "let laneSelected: Vec3<f32> = "
        "Vec3::<f32>::new((if mask.x { a.x } else { b.x }), "
        "(if mask.y { a.y } else { b.y }), "
        "(if mask.z { a.z } else { b.z }));" in generated_code
    )
    assert (
        "let scalarArms: Vec3<f32> = "
        "Vec3::<f32>::new((if mask.x { 1.0 } else { 0.0 }), "
        "(if mask.y { 1.0 } else { 0.0 }), "
        "(if mask.z { 1.0 } else { 0.0 }));" in generated_code
    )
    assert (
        "let mixedArm: Vec3<f32> = "
        "Vec3::<f32>::new((if mask.x { a.x } else { 0.0 }), "
        "(if mask.y { a.y } else { 0.0 }), "
        "(if mask.z { a.z } else { 0.0 }));" in generated_code
    )
    assert (
        "let inferred: Vec3<f32> = "
        "Vec3::<f32>::new((if mask.x { a.x } else { 0.0 }), "
        "(if mask.y { a.y } else { 0.0 }), "
        "(if mask.z { a.z } else { 0.0 }));" in generated_code
    )
    assert (
        "let complex: Vec3<f32> = "
        "{ let __cgl_vec_arg_0 = makeMask(); let __cgl_vec_arg_1 = makeA(); "
        "let __cgl_vec_arg_2 = makeScalar(); "
        "Vec3::<f32>::new((if __cgl_vec_arg_0.x { __cgl_vec_arg_1.x } "
        "else { __cgl_vec_arg_2 }), "
        "(if __cgl_vec_arg_0.y { __cgl_vec_arg_1.y } else { __cgl_vec_arg_2 }), "
        "(if __cgl_vec_arg_0.z { __cgl_vec_arg_1.z } else { __cgl_vec_arg_2 })) };"
        in generated_code
    )
    assert "(if mask { a } else { b })" not in generated_code
    assert "(if mask { 1.0 } else { 0.0 })" not in generated_code
    assert "(if makeMask() { makeA() } else { makeScalar() })" not in generated_code


def test_inferred_vector_and_matrix_ternaries_promote_branch_component_types():
    code = """
    void infer(bool ready, ivec2 pixel, vec2 amount, mat2 transform, dmat2 preciseInput) {
        let inferredVec = ready ? pixel : amount;
        let inferredVecReverse = ready ? amount : pixel;
        let inferredMat = ready ? transform : preciseInput;
        let inferredMatReverse = ready ? preciseInput : transform;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let inferredVec: Vec2<f32> = "
        "(if ready { Vec2::<f32>::new((pixel.x as f32), "
        "(pixel.y as f32)) } else { amount });" in generated_code
    )
    assert (
        "let inferredVecReverse: Vec2<f32> = "
        "(if ready { amount } else { Vec2::<f32>::new((pixel.x as f32), "
        "(pixel.y as f32)) });" in generated_code
    )
    assert (
        "let inferredMat: Mat2<f64> = "
        "(if ready { Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)) } else { preciseInput });" in generated_code
    )
    assert (
        "let inferredMatReverse: Mat2<f64> = "
        "(if ready { preciseInput } else { Mat2::<f64>::new((transform.c0.x as f64), "
        "(transform.c0.y as f64), (transform.c1.x as f64), "
        "(transform.c1.y as f64)) });" in generated_code
    )
    assert "let inferredVec: Vec2<i32>" not in generated_code
    assert "let inferredMat: Mat2<f32>" not in generated_code


def test_generic_vector_constructors_emit_rust_names():
    code = """
    shader main {
        compute {
            void main() {
                vec2<f64> precise = vec2<f64>(1.0, 2.0);
                vec3<i32> index = vec3<i32>(1, 2, 3);
                vec4<u32> mask = vec4<u32>(1, 2, 3, 4);
                vec2<bool> flags = vec2<bool>(true, false);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let precise: Vec2<f64> = Vec2::<f64>::new(1.0, 2.0);" in generated_code
    assert "let index: Vec3<i32> = Vec3::<i32>::new(1, 2, 3);" in generated_code
    assert "let mask: Vec4<u32> = Vec4::<u32>::new(1, 2, 3, 4);" in generated_code
    assert "let flags: Vec2<bool> = Vec2::<bool>::new(true, false);" in generated_code
    assert "Vec2<f64>::new" not in generated_code
    assert "Vec3<i32>::new" not in generated_code
    assert "Vec4<u32>::new" not in generated_code
    assert "Vec2<bool>::new" not in generated_code
    assert "vec2<" not in generated_code
    assert "vec3<" not in generated_code
    assert "vec4<" not in generated_code


def test_inferred_let_vector_declarations_preserve_vector_type():
    code = """
    shader main {
        compute {
            void main() {
                let value = vec4(1.0, 2.0, 3.0, 4.0);
                let x = value.x;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let value: Vec4<f32> = Vec4::<f32>::new(1.0, 2.0, 3.0, 4.0);" in generated_code
    )
    assert "let x: f32 = value.x;" in generated_code
    assert "let value: f32 = Vec4" not in generated_code


def test_scalar_vector_constructors_splat_rust_values():
    code = """
    shader main {
        compute {
            void main() {
                float weight = 0.5;
                int index = 1;
                bool enabled = true;
                vec4 base = vec4(1.0);
                vec3 scaled = vec3(weight);
                ivec2 pixel = ivec2(index);
                bvec4 mask = bvec4(enabled);
                vec4 offset = vec4(0.25, 0.5, 0.75, 1.0);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let base: Vec4<f32> = Vec4::<f32>::new(1.0, 1.0, 1.0, 1.0);" in generated_code
    )
    assert (
        "let scaled: Vec3<f32> = Vec3::<f32>::new(weight, weight, weight);"
        in generated_code
    )
    assert "let pixel: Vec2<i32> = Vec2::<i32>::new(index, index);" in generated_code
    assert (
        "let mask: Vec4<bool> = "
        "Vec4::<bool>::new(enabled, enabled, enabled, enabled);" in generated_code
    )
    assert (
        "let offset: Vec4<f32> = "
        "Vec4::<f32>::new(0.25, 0.5, 0.75, 1.0);" in generated_code
    )
    assert "Vec4::<f32>::new(1.0);" not in generated_code
    assert "Vec3::<f32>::new(weight);" not in generated_code
    assert "Vec2::<i32>::new(index);" not in generated_code
    assert "Vec4::<bool>::new(enabled);" not in generated_code


def test_complex_scalar_vector_constructor_splats_use_single_evaluation_blocks():
    code = """
    float makeWeight() {
        return 0.5;
    }

    int nextIndex() {
        return 1;
    }

    bool nextFlag() {
        return true;
    }

    void probe() {
        vec3 fromCall = vec3(makeWeight());
        vec3 fromCast = vec3(nextIndex());
        bvec4 fromBoolCall = bvec4(nextFlag());
        bvec2 fromInt = bvec2(nextIndex());
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let fromCall: Vec3<f32> = "
        "{ let __cgl_vec_arg_0 = makeWeight(); "
        "Vec3::<f32>::new(__cgl_vec_arg_0, __cgl_vec_arg_0, "
        "__cgl_vec_arg_0) };" in generated_code
    )
    assert (
        "let fromCast: Vec3<f32> = "
        "{ let __cgl_vec_arg_1 = (nextIndex() as f32); "
        "Vec3::<f32>::new(__cgl_vec_arg_1, __cgl_vec_arg_1, "
        "__cgl_vec_arg_1) };" in generated_code
    )
    assert (
        "let fromBoolCall: Vec4<bool> = "
        "{ let __cgl_vec_arg_2 = nextFlag(); "
        "Vec4::<bool>::new(__cgl_vec_arg_2, __cgl_vec_arg_2, "
        "__cgl_vec_arg_2, __cgl_vec_arg_2) };" in generated_code
    )
    assert (
        "let fromInt: Vec2<bool> = "
        "{ let __cgl_vec_arg_3 = (nextIndex() != 0); "
        "Vec2::<bool>::new(__cgl_vec_arg_3, __cgl_vec_arg_3) };" in generated_code
    )
    assert (
        "Vec3::<f32>::new(makeWeight(), makeWeight(), makeWeight())"
        not in generated_code
    )
    assert (
        "Vec3::<f32>::new((nextIndex() as f32), "
        "(nextIndex() as f32), (nextIndex() as f32))" not in generated_code
    )
    assert "Vec4::<bool>::new(nextFlag(), nextFlag()" not in generated_code
    assert (
        "Vec2::<bool>::new((nextIndex() != 0), (nextIndex() != 0))"
        not in generated_code
    )


def test_mixed_scalar_vector_constructor_lanes_cast_to_component_type():
    code = """
    void probe(float weight, int index, uint count) {
        vec2 mixedFloat = vec2(weight, index);
        uvec3 mixedUint = uvec3(count, index, 2);
        vec3 splatFloat = vec3(index);
        bvec2 boolMask = bvec2(index, weight);
        vec3 fromIntVector = vec3(ivec2(index, 2), weight);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let mixedFloat: Vec2<f32> = "
        "Vec2::<f32>::new(weight, (index as f32));" in generated_code
    )
    assert (
        "let mixedUint: Vec3<u32> = "
        "Vec3::<u32>::new(count, (index as u32), 2);" in generated_code
    )
    assert (
        "let splatFloat: Vec3<f32> = "
        "Vec3::<f32>::new((index as f32), (index as f32), (index as f32));"
        in generated_code
    )
    assert (
        "let boolMask: Vec2<bool> = "
        "Vec2::<bool>::new((index != 0), (weight != 0.0));" in generated_code
    )
    assert (
        "let fromIntVector: Vec3<f32> = "
        "{ let __cgl_vec_arg_0 = Vec2::<i32>::new(index, 2); "
        "Vec3::<f32>::new((__cgl_vec_arg_0.x as f32), "
        "(__cgl_vec_arg_0.y as f32), weight) };" in generated_code
    )
    assert "Vec2::<f32>::new(weight, index)" not in generated_code
    assert "Vec3::<u32>::new(count, index, 2)" not in generated_code
    assert "Vec3::<f32>::new(index, index, index)" not in generated_code
    assert "Vec2::<bool>::new(index, weight)" not in generated_code


def test_composite_vector_constructors_flatten_rust_lanes():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        compute {
            void main(VSInput input) {
                vec2 xy = vec2(1.0, 2.0);
                vec4 color = vec4(xy, 0.0, 1.0);
                vec3 position = vec3(xy, 1.0);
                vec3 rgb = vec3(color.rgb);
                vec4 packed = vec4(input.texCoord, 0.0, 1.0);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let color: Vec4<f32> = "
        "Vec4::<f32>::new(xy.x, xy.y, 0.0, 1.0);" in generated_code
    )
    assert (
        "let position: Vec3<f32> = "
        "Vec3::<f32>::new(xy.x, xy.y, 1.0);" in generated_code
    )
    assert (
        "let rgb: Vec3<f32> = "
        "Vec3::<f32>::new(color.x, color.y, color.z);" in generated_code
    )
    assert (
        "let packed: Vec4<f32> = "
        "Vec4::<f32>::new(input.texCoord.x, input.texCoord.y, 0.0, 1.0);"
        in generated_code
    )
    assert "Vec4::<f32>::new(xy, 0.0, 1.0)" not in generated_code
    assert "Vec3::<f32>::new(xy, 1.0)" not in generated_code
    assert "Vec3::<f32>::new(color.rgb)" not in generated_code
    assert "Vec4::<f32>::new(input.texCoord, 0.0, 1.0)" not in generated_code


def test_complex_vector_constructor_args_use_single_evaluation_blocks():
    code = """
    vec2 makeUv() {
        return vec2(1.0, 2.0);
    }

    vec3 makeNormal() {
        return vec3(0.0, 1.0, 0.0);
    }

    vec4 makeColor() {
        return vec4(1.0, 2.0, 3.0, 4.0);
    }

    void probe() {
        vec4 color = vec4(makeUv(), 0.0, 1.0);
        vec4 packed = vec4(makeNormal(), 1.0);
        vec4 mixed = vec4(makeColor().rg, 0.0, 1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let color: Vec4<f32> = "
        "{ let __cgl_vec_arg_0 = makeUv(); "
        "Vec4::<f32>::new(__cgl_vec_arg_0.x, __cgl_vec_arg_0.y, 0.0, 1.0) };"
        in generated_code
    )
    assert (
        "let packed: Vec4<f32> = "
        "{ let __cgl_vec_arg_1 = makeNormal(); "
        "Vec4::<f32>::new(__cgl_vec_arg_1.x, __cgl_vec_arg_1.y, "
        "__cgl_vec_arg_1.z, 1.0) };" in generated_code
    )
    assert (
        "let mixed: Vec4<f32> = "
        "{ let __cgl_vec_arg_2 = makeColor(); "
        "Vec4::<f32>::new(__cgl_vec_arg_2.x, __cgl_vec_arg_2.y, 0.0, 1.0) };"
        in generated_code
    )
    assert "makeUv().x" not in generated_code
    assert "makeNormal().z" not in generated_code
    assert "makeColor().rg" not in generated_code


def test_standalone_vector_swizzles_emit_rust_constructors():
    code = """
    void probe() {
        vec4 color = vec4(1.0, 2.0, 3.0, 4.0);
        let rgb = color.rgb;
        let bgra = color.bgra;
        let red = color.r;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let rgb: Vec3<f32> = Vec3::<f32>::new(color.x, color.y, color.z);"
        in generated_code
    )
    assert (
        "let bgra: Vec4<f32> = "
        "Vec4::<f32>::new(color.z, color.y, color.x, color.w);" in generated_code
    )
    assert "let red: f32 = color.x;" in generated_code
    assert "color.rgb" not in generated_code
    assert "color.bgra" not in generated_code
    assert "color.r;" not in generated_code


def test_complex_vector_swizzles_use_single_evaluation_blocks():
    code = """
    vec4 makeColor() {
        return vec4(1.0, 2.0, 3.0, 4.0);
    }

    void probe() {
        let rgb = makeColor().rgb;
        let xy = vec4(1.0, 2.0, 3.0, 4.0).xy;
        let red = makeColor().r;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "let rgb: Vec3<f32> = "
        "{ let __cgl_swizzle_0 = makeColor(); "
        "Vec3::<f32>::new(__cgl_swizzle_0.x, __cgl_swizzle_0.y, "
        "__cgl_swizzle_0.z) };" in generated_code
    )
    assert (
        "let xy: Vec2<f32> = "
        "{ let __cgl_swizzle_1 = Vec4::<f32>::new(1.0, 2.0, 3.0, 4.0); "
        "Vec2::<f32>::new(__cgl_swizzle_1.x, __cgl_swizzle_1.y) };" in generated_code
    )
    assert "let red: f32 = makeColor().x;" in generated_code
    assert "makeColor().rgb" not in generated_code
    assert "Vec4::<f32>::new(1.0, 2.0, 3.0, 4.0).xy" not in generated_code


def test_switch_statement_emits_rust_match():
    code = """
    shader main {
        compute {
            int main(int mode) {
                int out = 0;
                switch (mode) {
                    case 0:
                        out = 1;
                        break;
                    default:
                        out = 2;
                }
                return out;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "match mode" in generated_code
    assert "0 =>" in generated_code
    assert "_ =>" in generated_code
    assert "out = 1;" in generated_code
    assert "out = 2;" in generated_code
    assert "return out;" in generated_code
    assert "SwitchNode" not in generated_code
    assert "CaseNode" not in generated_code
    assert "BreakNode" not in generated_code


def test_match_statement_emits_rust_match():
    code = """
    shader main {
        compute {
            int main(int mode) {
                int value = 0;
                match mode {
                    0 => {
                        value = 1;
                    }
                    _ => {
                        value = 2;
                    }
                }
                return value;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "match mode" in generated_code
    assert "0 =>" in generated_code
    assert "_ =>" in generated_code
    assert "value = 1;" in generated_code
    assert "value = 2;" in generated_code
    assert "MatchNode" not in generated_code
    assert "MatchArmNode" not in generated_code


def test_match_path_constructor_struct_and_guarded_patterns_emit_rust_match():
    code = """
    shader PatternCases {
        fn inspect(result: Result<int, Error>, model: LightingModel) -> int {
            match result {
                Result::Ok(value) => value,
                Result::Err(_) => 0,
            }

            match model {
                LightingModel::Phong { ambient, diffuse, specular, shininess } if shininess > 0.0 => 1,
                LightingModel::Toon { base_color, .. } => 2,
            }

            return 0;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "match result" in generated_code
    assert "Result::Ok(value) =>" in generated_code
    assert "Result::Err(_) =>" in generated_code
    assert "match model" in generated_code
    assert (
        "LightingModel::Phong { ambient: _ambient, diffuse: _diffuse, "
        "specular: _specular, shininess } if (shininess > 0.0) =>" in generated_code
    )
    assert "LightingModel::Toon { base_color: _base_color, .. } =>" in generated_code
    assert "MatchNode" not in generated_code
    assert "MatchArmNode" not in generated_code


def test_unused_match_pattern_bindings_are_prefixed_but_used_bindings_are_preserved():
    code = """
    shader PatternWarnings {
        generic<T, E> struct Result {
            enum ResultType {
                Ok(T),
                Err(E)
            }
            ResultType variant;
        }

        enum Command {
            Draw {
                payload: int,
                amount: int
            },
            Skip
        }

        fn unwrap(result: Result<int, int>) -> int {
            match result {
                Result::Ok(value) => value,
                Result::Err(error) => 0,
            }
        }

        fn inspect(command: Command) -> int {
            match command {
                Command::Draw { payload, amount } if amount > 0 => amount,
                Command::Draw { payload, amount } => amount,
                Command::Skip => 0,
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "Result::Ok(value) =>" in generated_code
    assert "Result::Err(_error) =>" in generated_code
    assert (
        "Command::Draw { payload: _payload, amount } if (amount > 0) =>"
        in generated_code
    )
    assert "Command::Draw { payload: _payload, amount } =>" in generated_code
    assert "amount: _amount" not in generated_code


def test_match_arm_block_tail_expression_emits_without_semicolon():
    code = """
    shader PatternCases {
        fn convert(op: VectorOp) -> Result<int, MathError> {
            match op {
                VectorOp::Cross => {
                    Result::Ok(1)
                },
                _ => Result::Err(MathError::InvalidInput),
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "VectorOp::Cross => {\n            Result::Ok(1)\n        }" in generated_code
    )
    assert (
        "_ => {\n            Result::Err(MathError::InvalidInput)\n        }"
        in generated_code
    )
    assert "Result::Ok(1);" not in generated_code
    assert "Result::Err(MathError::InvalidInput);" not in generated_code


def test_keyword_like_let_binding_inside_match_arm_preserves_function_body():
    code = """
    shader KeywordBinding {
        fn process_lighting_model(model: int) -> vec3 {
            match model {
                _ => {
                    let geometry = 1.0;
                    vec3(geometry, geometry, geometry)
                }
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub fn process_lighting_model(model: i32) -> Vec3<f32> {" in generated_code
    assert "match model" in generated_code
    assert "let geometry: f32 = 1.0;" in generated_code
    assert "Vec3::<f32>::new(geometry, geometry, geometry)" in generated_code
    assert (
        "pub fn process_lighting_model(model: i32) -> Vec3<f32> {\n}\n"
        not in generated_code
    )


def test_match_expression_initializer_emits_rust_match_initializer():
    code = """
    shader MatchInitializer {
        fn choose(model: int, fallback: vec3) -> vec3 {
            let processed_normal = match model {
                0 => vec3(1.0, 1.0, 1.0),
                _ => fallback,
            };
            processed_normal
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "let processed_normal: Vec3<f32> = match model {" in generated_code
    assert "0 => {\n            Vec3::<f32>::new(1.0, 1.0, 1.0)" in generated_code
    assert "_ => {\n            fallback" in generated_code
    assert "};\n    processed_normal" in generated_code
    assert "let processed_normal: f32 = match model" not in generated_code


def test_stage_local_structs_and_uniforms_infer_normalized_matrix_vector_types():
    code = """
    shader StageInference {
        generic<T> struct Vec3 {
            T x;
            T y;
            T z;
        }

        vertex {
            struct VertexInput {
                position: vec3,
                normal: vec3,
            }

            struct VertexOutput {
                normal: Vec3<float>,
            }

            uniform mat4 model_matrix;
            uniform mat3 normal_matrix;

            VertexOutput main(VertexInput input) {
                let world_pos = model_matrix * vec4(input.position, 1.0);
                let world_normal = normalize(normal_matrix * input.normal);
                let normal_vec3 = Vec3 { x: world_normal.x, y: world_normal.y, z: world_normal.z };
                VertexOutput { normal: normal_vec3 }
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "pub struct VertexInput" in generated_code
    assert "pub position: math::Vec3<f32>," in generated_code
    assert "pub normal: math::Vec3<f32>," in generated_code
    assert (
        "static MODEL_MATRIX: std::sync::LazyLock<Mat4<f32>> = "
        "std::sync::LazyLock::new(|| Default::default());"
    ) in generated_code
    assert (
        "static NORMAL_MATRIX: std::sync::LazyLock<Mat3<f32>> = "
        "std::sync::LazyLock::new(|| Default::default());"
    ) in generated_code
    assert "static MODEL_MATRIX: Mat4<f32> = Default::default();" not in generated_code
    assert "static NORMAL_MATRIX: Mat3<f32> = Default::default();" not in generated_code
    assert (
        "let world_pos: Vec4<f32> = "
        "(*MODEL_MATRIX * Vec4::<f32>::new("
        "input.position.x, input.position.y, input.position.z, 1.0));" in generated_code
    )
    assert (
        "let world_normal: math::Vec3<f32> = "
        "normalize((*NORMAL_MATRIX * input.normal));" in generated_code
    )
    assert (
        "let normal_vec3: Vec3<f32> = "
        "Vec3 { x: world_normal.x, y: world_normal.y, z: world_normal.z };"
        in generated_code
    )
    assert "let world_pos: f32 = " not in generated_code
    assert "let world_normal: f32 = normalize" not in generated_code
    assert "let normal_vec3: Vec3 = Vec3" not in generated_code


def test_stage_local_uniform_statics_are_emitted_once():
    code = """
    shader StageUniforms {
        vertex {
            uniform mat4 shared_transform;

            vec4 main(vec3 position) {
                shared_transform * vec4(position, 1.0)
            }
        }

        fragment {
            uniform mat4 shared_transform;
            uniform sampler2D main_texture;

            vec4 main(vec2 uv) {
                texture(main_texture, uv)
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert generated_code.count("static SHARED_TRANSFORM:") == 1
    assert (
        "static SHARED_TRANSFORM: std::sync::LazyLock<Mat4<f32>> = "
        "std::sync::LazyLock::new(|| Default::default());"
    ) in generated_code
    assert (
        "static MAIN_TEXTURE: std::sync::LazyLock<Texture2D<f32>> = "
        "std::sync::LazyLock::new(|| Default::default());"
    ) in generated_code
    assert (
        "static SHARED_TRANSFORM: Mat4<f32> = Default::default();" not in generated_code
    )
    assert (
        "static MAIN_TEXTURE: Texture2D<f32> = Default::default();"
        not in generated_code
    )
    assert "static MAIN_TEXTURE: sampler2D" not in generated_code
    assert "(*SHARED_TRANSFORM * Vec4::<f32>::new" in generated_code
    assert "sample(*MAIN_TEXTURE, uv)" in generated_code


def test_method_calls_and_shorthand_path_constructors_emit_rust_syntax():
    code = """
    shader ConstructorCases {
        fn build(v1: Vec3<T>, v2: Vec3<T>, color: vec4, depth: float, op: VectorOp) -> int {
            match op {
                VectorOp::Add => Result::Ok(Vec3 { x: v1.x.add(v2.x), y: v1.y.mul(v2.y), z: v1.z }),
                _ => Result::Ok(RenderOutput::Clear { color, depth }),
            }

            return 0;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "v1.x.add(v2.x)" in generated_code
    assert "v1.y.mul(v2.y)" in generated_code
    assert "None(v2.x)" not in generated_code
    assert "None(v2.y)" not in generated_code
    assert "RenderOutput::Clear { color: color, depth: depth }" in generated_code
    assert "RenderOutput::Clear::new(color, depth)" not in generated_code


def test_non_wildcard_match_fallback_uses_never_expression():
    code = """
    shader PatternCases {
        fn convert(op: VectorOp) -> Result<int, MathError> {
            match op {
                VectorOp::Cross => Result::Ok(1),
                VectorOp::Add => Result::Err(MathError::InvalidInput),
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "_ => unreachable!()," in generated_code
    assert "_ => {}," not in generated_code


def test_exhaustive_bool_result_and_enum_matches_omit_unreachable_fallback(tmp_path):
    code = """
    shader ExhaustivePatternCases {
        generic<T, E> struct Result {
            enum ResultType {
                Ok(T),
                Err(E)
            }
            ResultType variant;
        }

        enum Mode {
            Bright,
            Dark
        }

        enum Command {
            Draw {
                payload: Result<int, int>,
                amount: int
            },
            Skip
        }

        struct Pair {
            left: int,
            right: int
        }

        fn unwrap(result: Result<int, int>) -> int {
            match result {
                Result::Ok(value) => value,
                Result::Err(_) => 0,
            }
        }

        fn choose(mode: Mode) -> int {
            match mode {
                Mode::Bright => 1,
                Mode::Dark => 0,
            }
        }

        fn flag_to_int(flag: bool) -> int {
            match flag {
                true => 1,
                false => 0,
            }
        }

        fn sum_pair(pair: Pair) -> int {
            match pair {
                Pair { left, right } => left + right,
            }
        }

        fn nested(command: Command) -> int {
            match command {
                Command::Draw { payload: Result::Ok(value), amount } => value + amount,
                Command::Draw { payload: Result::Err(_), amount } => amount,
                Command::Skip => 0,
            }
        }

        fn partial(mode: Mode) -> int {
            match mode {
                Mode::Bright => 1,
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    def function_section(name):
        start = generated_code.index(f"pub fn {name}")
        next_start = generated_code.find("\npub fn ", start + 1)
        if next_start == -1:
            return generated_code[start:]
        return generated_code[start:next_start]

    assert "_ => unreachable!()," not in function_section("unwrap")
    assert "_ => unreachable!()," not in function_section("choose")
    assert "_ => unreachable!()," not in function_section("flag_to_int")
    assert "_ => unreachable!()," not in function_section("sum_pair")
    assert "_ => unreachable!()," not in function_section("nested")
    assert "_ => unreachable!()," in function_section("partial")
    assert_generated_rust_smoke_compiles(generated_code, tmp_path)


def test_inferred_generic_struct_literals_and_destructured_members_keep_generic_types():
    code = """
    shader InferredStructLiteral {
        generic<T> struct Vec3 {
            T x;
            T y;
            T z;
        }

        generic<T> struct Matrix3x3 {
            Vec3<T> row0;
            Vec3<T> row1;
            Vec3<T> row2;
        }

        generic<T> fn make_vec(a: T, b: T, c: T) -> Vec3<T> {
            let result = Vec3 { x: a, y: b, z: c };
            return result;
        }

        generic<T> fn determinant(matrix: Matrix3x3<T>) -> T {
            match matrix {
                Matrix3x3 { row0, row1, row2 } => {
                    let a = row0.x;
                    a
                }
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "let result: Vec3<T> = Vec3 { x: a, y: b, z: c };" in generated_code
    assert "let a: T = row0.x;" in generated_code
    assert "let result: f32 = Vec3" not in generated_code
    assert "let a: f32 = row0.x;" not in generated_code


def test_match_guarded_literal_arm_emits_rust_guard():
    code = """
    shader main {
        compute {
            int main(int mode) {
                int value = 0;
                match mode {
                    0 if mode > 0 => {
                        value = 1;
                    }
                    _ => {
                        value = 2;
                    }
                }
                return value;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "0 if (mode > 0) =>" in generated_code
    assert "value = 1;" in generated_code


def test_generic_vector_composite_types_emit_rust_names():
    code = """
    shader GenericComposite {
        struct Packed {
            vec2<f64> precise;
            vec3<i32> index;
            vec4<u32> mask;
            vec2<bool> flags;
        };

        vec2<f64> passthrough(vec2<f64> value, vec3<i32> index, vec4<u32> mask, vec2<bool> flags) {
            vec2<f64> localValues[2];
            localValues[0] = value;
            return localValues[0];
        }

        compute {
            void main() {
                Packed p;
                vec2<f64> values[2];
                values[0] = vec2<f64>(1.0, 2.0);
                vec2<f64> result = passthrough(values[0], vec3<i32>(1, 2, 3), vec4<u32>(1, 2, 3, 4), vec2<bool>(true, false));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "pub precise: Vec2<f64>," in generated_code
    assert "pub index: Vec3<i32>," in generated_code
    assert "pub mask: Vec4<u32>," in generated_code
    assert "pub flags: Vec2<bool>," in generated_code
    assert (
        "pub fn passthrough(value: Vec2<f64>, index: Vec3<i32>, "
        "mask: Vec4<u32>, flags: Vec2<bool>) -> Vec2<f64>"
    ) in generated_code
    assert "let mut localValues: [Vec2<f64>; 2];" in generated_code
    assert "let mut values: [Vec2<f64>; 2];" in generated_code
    assert "LiteralNode(" not in generated_code
    assert "vec2<" not in generated_code


def test_array_access():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        cbuffer Material {
            float values[4];
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(values[0], values[1], values[2], 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "VALUES[0]" in generated_code
        assert "VALUES[1]" in generated_code
        assert "VALUES[2]" in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Array access codegen not implemented")


def test_array_literals_emit_rust_array_initializers():
    code = """
    float globalWeights[4] = {1.0, 2.0};

    float pickScalar(int index) {
        float values[4] = {1.0, 2.0, 3.0, 4.0};
        return values[index];
    }

    float[4] makeValues() {
        return {1.0, 2.0};
    }

    vec3 pickColor(int index) {
        vec3 colors[2] = {vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0)};
        return colors[index];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "static GLOBAL_WEIGHTS: [f32; 4] = " "[1.0, 2.0, 0.0, 0.0];"
    ) in generated_code
    assert "static GLOBAL_WEIGHTS: [f32; 4] = [1.0, 2.0, Default" not in generated_code
    assert "let values: [f32; 4] = [1.0, 2.0, 3.0, 4.0];" in generated_code
    assert (
        "return [1.0, 2.0, Default::default(), Default::default()];"
    ) in generated_code
    assert (
        "let colors: [Vec3<f32>; 2] = "
        "[Vec3::<f32>::new(1.0, 2.0, 3.0), "
        "Vec3::<f32>::new(4.0, 5.0, 6.0)];"
    ) in generated_code
    assert "ArrayLiteralNode" not in generated_code


def test_array_access_casts_non_literal_indices_to_usize():
    code = """
    float globalWeights[4] = {1.0, 2.0};
    vec3 colors[2] = {vec3(1.0, 2.0, 3.0)};

    float pickScalar(int index) {
        float values[4] = {1.0, 2.0, 3.0, 4.0};
        return values[index] + globalWeights[index] + values[0];
    }

    vec3 pickColor(int index) {
        return colors[index];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "values[index as usize]" in generated_code
    assert "GLOBAL_WEIGHTS[index as usize]" in generated_code
    assert "(*COLORS)[index as usize]" in generated_code
    assert "values[0]" in generated_code
    assert "values[0 as usize]" not in generated_code
    assert "values[index]" not in generated_code
    assert "globalWeights[index]" not in generated_code


def test_inferred_array_access_bindings_use_element_types():
    code = """
    float globalWeights[4] = {1.0, 2.0};
    vec3 globalColors[2] = {vec3(1.0, 2.0, 3.0)};

    void probe(int index) {
        float values[4] = {1.0, 2.0, 3.0, 4.0};
        vec3 colors[2] = {
            vec3(1.0, 2.0, 3.0),
            vec3(4.0, 5.0, 6.0)
        };
        let scalar = values[index];
        let globalScalar = globalWeights[index];
        let color = colors[index];
        let globalColor = globalColors[index];
        let literalScalar = values[0];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let scalar: f32 = values[index as usize];" in generated_code
    assert "let globalScalar: f32 = GLOBAL_WEIGHTS[index as usize];" in generated_code
    assert "let color: Vec3<f32> = colors[index as usize];" in generated_code
    assert (
        "let globalColor: Vec3<f32> = (*GLOBAL_COLORS)[index as usize];"
        in generated_code
    )
    assert "let literalScalar: f32 = values[0];" in generated_code
    assert "let scalar: [f32; 4]" not in generated_code
    assert "let color: [Vec3<f32>; 2]" not in generated_code


def test_auto_typed_legacy_bindings_infer_initializer_types():
    ast = ShaderNode(
        "LegacyAutoBindings",
        ExecutionModel.GENERAL_PURPOSE,
        functions=[
            FunctionNode(
                "probe",
                PrimitiveType("void"),
                [],
                [
                    ArrayNode(PrimitiveType("float"), "values", 4),
                    VariableNode(
                        "picked",
                        PrimitiveType("auto"),
                        ArrayAccessNode(
                            IdentifierNode("values"),
                            LiteralNode(0, PrimitiveType("int")),
                        ),
                    ),
                    VariableNode(
                        "flag",
                        "auto",
                        LiteralNode(True, PrimitiveType("bool")),
                    ),
                ],
            )
        ],
    )

    generated_code = generate_code(ast)

    assert "let picked: f32 = values[0];" in generated_code
    assert "let flag: bool = true;" in generated_code
    assert ": auto" not in generated_code


def test_inferred_function_call_bindings_use_user_return_types():
    code = """
    float pickWeight() {
        return 0.5;
    }

    vec3 makeColor() {
        return vec3(1.0, 2.0, 3.0);
    }

    bool isEnabled() {
        return true;
    }

    void probe() {
        let weight = pickWeight();
        let color = makeColor();
        let enabled = isEnabled();
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let weight: f32 = pickWeight();" in generated_code
    assert "let color: Vec3<f32> = makeColor();" in generated_code
    assert "let enabled: bool = isEnabled();" in generated_code
    assert "let color: f32 = makeColor();" not in generated_code
    assert "let enabled: f32 = isEnabled();" not in generated_code


def test_rust_imports():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "use gpu::*;" in generated_code
        assert "use math::*;" in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Rust imports not generated")


def test_let_binding():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                let x = input.texCoord.x;
                let y = input.texCoord.y;
                output.color = vec4(x, y, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "let x:" in generated_code  # Rust let binding syntax
        assert "let y:" in generated_code  # Rust let binding syntax
        print(generated_code)
    except SyntaxError:
        pytest.fail("Let binding codegen not implemented")


def test_rust_attributes():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "#[vertex_shader]" in generated_code
        assert "#[fragment_shader]" in generated_code
        assert "#[repr(C)]" in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Rust attributes not generated")


def test_rust_type_conversions():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                int i = int(input.texCoord.x * 100.0);
                float f = float(i);
                output.color = vec4(f, 0.0, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "let i: i32 = ((input.texCoord.x * 100.0) as i32);" in generated_code
        assert "let f: f32 = (i as f32);" in generated_code
        assert "int(" not in generated_code
        assert "float(" not in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Rust type conversions not implemented")
