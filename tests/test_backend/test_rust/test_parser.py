import pytest

from crosstl.backend.Rust.RustAst import (
    ArrayAccessNode,
    ArrayNode,
    AssignmentNode,
    AssociatedTypeNode,
    AsyncBlockNode,
    AwaitNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CastNode,
    ClosureNode,
    ClosureParameterNode,
    ConditionChainNode,
    ConstBlockNode,
    ConstNode,
    ContinueNode,
    DereferenceNode,
    EnumNode,
    EnumVariantNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    ImplNode,
    LetNode,
    LetPatternConditionNode,
    LoopNode,
    MatchBindingPatternNode,
    MatchesMacroNode,
    MatchNode,
    MatchOrPatternNode,
    MatchRestPatternNode,
    MatchStructPatternNode,
    MemberAccessNode,
    RangeNode,
    ReferenceNode,
    ReturnNode,
    StaticNode,
    StructInitializationNode,
    StructNode,
    TernaryOpNode,
    TryBlockNode,
    TryNode,
    TupleNode,
    TypeAliasNode,
    UnaryOpNode,
    UnsafeBlockNode,
    UseNode,
    WhileNode,
)
from crosstl.backend.Rust.RustLexer import RustLexer
from crosstl.backend.Rust.RustParser import RustParser


def parse_code(code: str):
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


def test_underscore_prefixed_identifier_parsing():
    code = """
    #[rust_gpu::vector::v1]
    pub struct _Struct {
        _field: f32,
    }

    pub fn _fn(_: _Struct, _value: f32) -> f32 {
        return _value;
    }
    """

    ast = parse_code(code)

    assert ast.structs[0].name == "_Struct"
    assert ast.structs[0].members[0].name == "_field"
    assert ast.functions[0].name == "_fn"
    assert [param.name for param in ast.functions[0].params] == ["_", "_value"]


def test_raw_identifier_parsing_normalizes_keyword_names():
    ast = parse_code("pub fn r#type(r#match: u32) -> u32 { r#match }")
    function = ast.functions[0]

    assert function.name == "type"
    assert [param.name for param in function.params] == ["match"]
    assert function.body[0] == "match"


def test_absolute_use_and_type_paths_parse_from_generated_webgpu_bindings():
    code = """
    pub use ::wgc::naga;

    pub struct Binding<'a> {
        strings: &'a [::js_sys::JsString],
        value: ::std::primitive::u32,
    }
    """

    ast = parse_code(code)
    struct = ast.structs[0]

    assert ast.use_statements[0].path == "::wgc::naga"
    assert [member.vtype for member in struct.members] == [
        "&::js_sys::JsString[]",
        "::std::primitive::u32",
    ]


def test_if_expression_continues_through_bitwise_operator_chain():
    code = """
    fn usage(supports_storage_resources: bool) -> wgpu::BufferUsages {
        let usage = if supports_storage_resources {
            wgpu::BufferUsages::STORAGE
        } else {
            wgpu::BufferUsages::UNIFORM
        } | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        usage
    }
    """

    ast = parse_code(code)
    usage = ast.functions[0].body[0]

    assert isinstance(usage, LetNode)
    assert isinstance(usage.value, BinaryOpNode)
    assert usage.value.op == "|"


def test_if_statement_condition_path_does_not_steal_branch_block():
    # Reduced from Rust-GPU/rustc_codegen_spirv codegen_cx/constant.rs.
    code = """
    fn const_ptr_byte_offset(offset: Size) -> Value {
        if offset == Size::ZERO {
            val
        } else {
            result
        }
    }
    """

    ast = parse_code(code)
    if_node = ast.functions[0].body[0]

    assert isinstance(if_node, IfNode)
    assert isinstance(if_node.condition, BinaryOpNode)
    assert if_node.condition.right == "Size::ZERO"
    assert if_node.if_body == ["val"]
    assert if_node.else_body == ["result"]


def test_if_let_condition_does_not_treat_empty_branch_as_struct_literal():
    # Reduced from Rust-GPU/rustc_codegen_spirv codegen_cx/declare.rs.
    code = """
    fn declare(align_override: Option<Align>, attrs: Attrs) {
        if let Some(_align) = align_override {}
        if attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL) {}
    }
    """

    ast = parse_code(code)
    first_if, second_if = ast.functions[0].body

    assert isinstance(first_if, IfNode)
    assert isinstance(first_if.condition, LetPatternConditionNode)
    assert first_if.condition.expression == "align_override"
    assert first_if.if_body == []
    assert isinstance(second_if, IfNode)


def test_match_arm_accepts_leading_or_with_struct_rest_patterns():
    # Reduced from Rust-GPU/rustc_codegen_spirv codegen_cx/type_.rs.
    code = """
    fn type_kind(ty: SpirvType) -> TypeKind {
        match ty {
            SpirvType::Function { .. } => TypeKind::Function,
            | SpirvType::Image { .. }
            | SpirvType::Sampler
            | SpirvType::SampledImage { .. }
                => TypeKind::Token,
        }
    }
    """

    ast = parse_code(code)
    match_node = ast.functions[0].body[0]

    assert isinstance(match_node, MatchNode)
    assert len(match_node.arms) == 2
    assert isinstance(match_node.arms[1].pattern, MatchOrPatternNode)
    assert any(
        isinstance(pattern, MatchStructPatternNode) and pattern.has_rest
        for pattern in match_node.arms[1].pattern.patterns
    )


def test_match_tuple_pattern_accepts_half_open_range():
    # Reduced from Rust-GPU/rustc_codegen_spirv linker/spirt_passes/diagnostics.rs.
    code = """
    fn decode(pair: Pair) {
        match pair {
            (0, Imm::Short(k, w) | Imm::LongStart(k, w))
            | (1.., Imm::LongCont(k, w)) => {
                w
            }
        }
    }
    """

    ast = parse_code(code)
    match_node = ast.functions[0].body[0]
    second_pattern = match_node.arms[0].pattern.patterns[1]

    assert isinstance(second_pattern, TupleNode)
    assert isinstance(second_pattern.elements[0], RangeNode)
    assert second_pattern.elements[0].start == "1"
    assert second_pattern.elements[0].end is None


def test_raw_borrow_and_nested_reference_closure_patterns_parse():
    code = """
    fn raw(reference: &mut T) -> Self {
        unsafe { Self::new(NonNull::new_unchecked(&raw mut *reference)) }
    }

    fn find(handle: u32) {
        let _ = ep_results.iter().find(|&&(_, ty)| ty == handle);
    }
    """

    ast = parse_code(code)

    assert len(ast.functions) == 2
    raw_call = ast.functions[0].body[0].block.expression.args[0].args[0]
    closure = ast.functions[1].body[0].value.args[0]
    assert isinstance(raw_call, ReferenceNode)
    assert isinstance(closure, ClosureNode)
    assert isinstance(closure.params[0].pattern, ReferenceNode)


def test_reference_named_raw_cast_parses_from_rust_cuda_surface():
    # Reduced from Rust-GPU/Rust-CUDA crates/cust/src/surface.rs.
    code = """
    fn surface(raw: CUDA_RESOURCE_DESC) {
        cuSurfObjectCreate(uninit.as_mut_ptr(), &raw as *const _).to_result()?;
    }
    """

    ast = parse_code(code)
    call = ast.functions[0].body[0].expression.name.object
    cast = call.args[1]

    assert isinstance(call, FunctionCallNode)
    assert isinstance(cast, CastNode)
    assert cast.target_type == "*const _"
    assert isinstance(cast.expression, ReferenceNode)
    assert cast.expression.expression == "raw"


def test_for_loop_accepts_destructuring_path_patterns():
    code = """
    fn validate(cases: &[crate::SwitchCase]) {
        for &crate::SwitchCase {
            value: _,
            ref body,
            fall_through: _,
        } in cases {
            validate_block(body)?;
        }
    }
    """

    ast = parse_code(code)
    loop = ast.functions[0].body[0]

    assert isinstance(loop, ForNode)
    assert isinstance(loop.pattern, ReferenceNode)
    assert isinstance(loop.pattern.expression, MatchStructPatternNode)


def test_primitive_named_impl_methods_parse_from_rust_gpu_metadata():
    code = """
    pub struct TestMetadata {
        output_type: OutputType,
        epsilon: Option<f32>,
    }

    enum OutputType { Raw, F32, F64, U32, I32 }

    impl TestMetadata {
        pub fn f32(epsilon: f32) -> Self {
            Self { output_type: OutputType::F32, epsilon: Some(epsilon) }
        }

        pub fn u32() -> Self {
            Self { output_type: OutputType::U32, ..Default::default() }
        }
    }
    """

    ast = parse_code(code)
    impl_block = ast.impl_blocks[0]

    assert [method.name for method in impl_block.methods] == ["f32", "u32"]
    assert impl_block.methods[0].params[0].vtype == "f32"


def test_primitive_named_member_access_parse_from_rust_gpu_codegen():
    # Reduced from:
    # Repo: https://github.com/Rust-GPU/rust-gpu
    # Commit: a27c0363d391a54de1feb9ee6864ad9dff72d243
    # Paths:
    # - crates/rustc_codegen_spirv/src/codegen_cx/declare.rs
    # - crates/rustc_codegen_spirv/src/codegen_cx/entry.rs
    code = """
    impl CodegenCx {
        fn classify(&self, element: Element) -> bool {
            element.ty == self.tcx.types.u32
        }

        fn record(&self) {
            self.buffer.insert(fn_id, self.tcx.types.usize);
        }
    }
    """

    ast = parse_code(code)
    classify, record = ast.impl_blocks[0].methods

    comparison = classify.body[0]
    assert isinstance(comparison, BinaryOpNode)
    assert comparison.op == "=="
    assert isinstance(comparison.right, MemberAccessNode)
    assert comparison.right.member == "u32"
    assert comparison.right.object.member == "types"

    insert_call = record.body[0]
    assert isinstance(insert_call, FunctionCallNode)
    assert insert_call.args[1].member == "usize"
    assert insert_call.args[1].object.member == "types"


def test_unsafe_extern_function_pointer_type_parsing_from_rusty_v8_callbacks():
    # Reduced from denoland/rusty_v8 commit
    # c2bac76486b5db090587e3f40988a8033ce81773 src/function.rs.
    code = """
    pub(crate) type NamedGetterCallbackForAccessor =
        unsafe extern "C" fn(SealedLocal<Name>, *const PropertyCallbackInfo<Value>);

    struct Handle {
        callback: unsafe extern "C" fn(handle: &Handle, data: Option<NonNull<c_void>>),
        fallback: extern "system" fn(u32) -> u32,
    }

    fn install(callback: unsafe fn(u32) -> u32) -> unsafe fn(u32) -> u32 {
        callback
    }
    """

    ast = parse_code(code)

    alias = ast.type_aliases[0]
    assert isinstance(alias, TypeAliasNode)
    assert alias.name == "NamedGetterCallbackForAccessor"
    assert alias.alias_type == (
        'unsafe extern "C" fn(SealedLocal<Name>, ' "*const PropertyCallbackInfo<Value>)"
    )

    struct = ast.structs[0]
    assert [member.vtype for member in struct.members] == [
        'unsafe extern "C" fn(&Handle, Option<NonNull<c_void>>)',
        'extern "system" fn(u32) -> u32',
    ]

    function = ast.functions[0]
    assert function.params[0].vtype == "unsafe fn(u32) -> u32"
    assert function.return_type == "unsafe fn(u32) -> u32"


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


def test_enum_discriminant_expression_parsing():
    code = """
    enum Flags {
        A = crate::flags::A,
        B = Type::<u32>::VALUE,
        C = 1 << 2,
        D = C | 8,
        E = D & 3,
        F = E ^ 1,
        G = -1,
        H = 1u32 as i32,
    }
    """
    try:
        ast = parse_code(code)
        variants = ast.enums[0].variants

        assert variants[0].value == "crate::flags::A"
        assert variants[1].value == "Type<u32>::VALUE"
        assert isinstance(variants[2].value, BinaryOpNode)
        assert variants[2].value.op == "<<"
        assert isinstance(variants[3].value, BinaryOpNode)
        assert variants[3].value.op == "|"
        assert isinstance(variants[4].value, BinaryOpNode)
        assert variants[4].value.op == "&"
        assert isinstance(variants[5].value, BinaryOpNode)
        assert variants[5].value.op == "^"
        assert isinstance(variants[6].value, UnaryOpNode)
        assert isinstance(variants[7].value, CastNode)
    except Exception as e:
        pytest.fail(f"Enum discriminant expression parsing failed: {e}")


def test_braced_macro_invocation_parsing():
    code = r"""
    pub unsafe fn subgroup_add(value: u32) -> u32 {
        let mut result = 0;
        asm! {
            "OpStore {result}",
            value = in(reg) value,
            result = in(reg) &mut result,
        }
        result
    }
    """

    ast = parse_code(code)
    function = ast.functions[0]
    macro_call = function.body[1]

    assert isinstance(macro_call, FunctionCallNode)
    assert macro_call.name == "asm!"
    assert macro_call.macro_delimiter == "LBRACE"
    assert "OpStore" in macro_call.args[0]


def test_macro_rules_definition_is_skipped():
    code = r"""
    macro_rules! enum_repr_from {
        ($ident:ident, $repr:ty) => {
            impl From<$repr> for $ident {
                fn from(value: $repr) -> Self {
                    value as $repr
                }
            }
        };
    }

    fn main() {
        return;
    }
    """

    ast = parse_code(code)

    assert len(ast.functions) == 1
    assert ast.functions[0].name == "main"


def test_local_macro_rules_definition_is_skipped_from_function_body():
    code = r"""
    fn classify(value: u32) -> u32 {
        macro_rules! map_case {
            ($input:expr, $($name:ident => $result:expr),*) => {
                match $input {
                    $($name => $result,)*
                    _ => 0,
                }
            };
        }

        let mapped = map_case!(value, One => 1, Two => 2);
        mapped
    }
    """

    ast = parse_code(code)
    function = ast.functions[0]

    assert function.name == "classify"
    assert len(function.body) == 2
    assert isinstance(function.body[0], LetNode)
    assert function.body[0].name == "mapped"
    assert function.body[1] == "mapped"


def test_local_macro_rules_definition_is_skipped_from_block_expression():
    code = r"""
    fn classify(value: u32) -> u32 {
        unsafe {
            macro_rules! pick {
                ($input:expr) => {
                    $input
                };
            }

            pick!(value)
        }
    }
    """

    ast = parse_code(code)
    unsafe_block = ast.functions[0].body[0]

    assert isinstance(unsafe_block, UnsafeBlockNode)
    assert unsafe_block.block.statements == []
    assert isinstance(unsafe_block.block.returns_value, FunctionCallNode)
    assert unsafe_block.block.returns_value.name == "pick!"


def test_block_local_module_item_is_skipped_from_rust_gpu_type_constraints():
    # Reduced from:
    # Repo: https://github.com/Rust-GPU/rust-gpu
    # Commit: a27c0363d391a54de1feb9ee6864ad9dff72d243
    # Path: crates/rustc_codegen_spirv/src/spirv_type_constraints.rs
    code = """
    pub fn instruction_signatures(op: Op) -> Option<&'static [InstSig<'static>]> {
        mod pat_ctors {
            pub const S: super::StorageClassPat = super::StorageClassPat::S;
            pub use super::TyPat::{Array, Either, Function};
        }

        op
    }
    """

    ast = parse_code(code)
    function = ast.functions[0]

    assert function.name == "instruction_signatures"
    assert function.return_type == "Option<&'static[InstSig<'static>]>"
    assert function.body == ["op"]


def test_match_guard_let_chain_parses_from_rust_gpu_format_args_decompiler():
    # Reduced from:
    # Repo: https://github.com/Rust-GPU/rust-gpu
    # Commit: a27c0363d391a54de1feb9ee6864ad9dff72d243
    # Path: crates/rustc_codegen_spirv/src/builder/format_args_decompiler.rs
    code = """
    fn decode(args: &[Value]) -> Option<(Word, Word)> {
        let split = match *args {
            [template, rt_args_or_tagged_len, ref trailing @ ..]
                if trailing.len() <= 3
                    && matches!(lookup(template.ty), Type::Pointer { .. })
                    && let (Some(template_id), Some(rt_args_or_tagged_len_id)) =
                        (value_id(template), value_id(rt_args_or_tagged_len))
                    && const_str(&[template_id, rt_args_or_tagged_len_id]).is_none() =>
            {
                Some((template_id, rt_args_or_tagged_len_id))
            }
            _ => None,
        };

        split
    }
    """

    ast = parse_code(code)
    function = ast.functions[0]
    split = function.body[0]

    assert isinstance(split, LetNode)
    assert isinstance(split.value, MatchNode)

    guard = split.value.arms[0].guard
    assert isinstance(guard, ConditionChainNode)
    assert len(guard.operands) == 4
    assert isinstance(guard.operands[2], LetPatternConditionNode)


def test_top_level_macro_invocations_are_skipped_from_rust_gpu_spirv_std():
    code = r"""
    bitflags::bitflags! {
        pub struct Semantics: u32 {
            const NONE = 0;
            const ACQUIRE = 0x2;
        }
    }

    impl_scalar! {
        impl UnsignedInteger for u8;
        impl UnsignedInteger for u16;
    }

    fn main() {
        return;
    }
    """

    ast = parse_code(code)

    assert len(ast.functions) == 1
    assert ast.functions[0].name == "main"
    assert ast.structs == []


def test_rust_gpu_path_qualified_attribute_parsing():
    code = r"""
    #[spirv_std_macros::gpu_only]
    #[doc(alias = "OpEmitVertex")]
    #[inline]
    pub unsafe fn emit_vertex() {
        unsafe {
            asm! {
                "OpEmitVertex",
            }
        }
    }
    """

    ast = parse_code(code)
    function = ast.functions[0]
    unsafe_block = function.body[0]
    macro_call = unsafe_block.block.returns_value

    assert [attr.name for attr in function.attributes] == [
        "spirv_std_macros::gpu_only",
        "doc",
        "inline",
    ]
    assert function.name == "emit_vertex"
    assert function.visibility == "pub"
    assert isinstance(unsafe_block, UnsafeBlockNode)
    assert isinstance(macro_call, FunctionCallNode)
    assert macro_call.name == "asm!"
    assert macro_call.macro_delimiter == "LBRACE"


def test_absolute_path_expression_parsing_from_rust_gpu_intrinsics():
    code = """
    fn derivative_ops(value: Vec3<f32>) {
        let direct = ::spirv_std::arch::fwidth(value.x);
        let associated = ::spirv_std::arch::Derivative::dfdx(value);
        ::spirv_std::arch::workgroup_memory_barrier_with_group_sync();
    }
    """

    ast = parse_code(code)
    body = ast.functions[0].body

    assert isinstance(body[0], LetNode)
    assert isinstance(body[0].value, FunctionCallNode)
    assert body[0].value.name == "spirv_std::arch::fwidth"
    assert isinstance(body[1], LetNode)
    assert isinstance(body[1].value, FunctionCallNode)
    assert body[1].value.name == "spirv_std::arch::Derivative::dfdx"
    assert isinstance(body[2], FunctionCallNode)
    assert body[2].name == "spirv_std::arch::workgroup_memory_barrier_with_group_sync"


def test_key_value_attribute_parsing_from_rust_gpu_spirv_std():
    code = r"""
    #[lang = "eh_personality"]
    fn rust_eh_personality() {}

    #[path = "generated.rs"]
    pub fn include_generated() {}
    """

    ast = parse_code(code)

    lang_fn, path_fn = ast.functions
    assert lang_fn.name == "rust_eh_personality"
    assert lang_fn.attributes[0].name == "lang"
    assert lang_fn.attributes[0].args == ['"eh_personality"']

    assert path_fn.name == "include_generated"
    assert path_fn.visibility == "pub"
    assert path_fn.attributes[0].name == "path"
    assert path_fn.attributes[0].args == ['"generated.rs"']


def test_control_flow_expressions_in_match_arm_results():
    code = """
    fn main() {
        'outer: loop {
            let value = match next() {
                0 => return Err(error),
                1 => continue 'outer,
                _ => break 'outer 7,
            };
        }
    }
    """

    ast = parse_code(code)
    outer_loop = ast.functions[0].body[0]
    value_binding = outer_loop.body[0]
    match_value = value_binding.value

    return_arm, continue_arm, break_arm = (arm.body for arm in match_value.arms)

    assert isinstance(return_arm, ReturnNode)
    assert isinstance(return_arm.value, FunctionCallNode)
    assert return_arm.value.name == "Err"

    assert isinstance(continue_arm, ContinueNode)
    assert continue_arm.label == "'outer"

    assert isinstance(break_arm, BreakNode)
    assert break_arm.label == "'outer"
    assert break_arm.value == "7"


def test_rust_gpu_ray_query_parenthesized_statement_macro_parsing():
    code = """
    use glam::Vec3;
    use spirv_std::ray_tracing::{AccelerationStructure, RayFlags, RayQuery};
    use spirv_std::spirv;

    #[spirv(fragment)]
    pub fn main(#[spirv(descriptor_set = 0, binding = 0)] accel: &AccelerationStructure) {
        unsafe {
            spirv_std::ray_query!(let mut handle);
            handle.initialize(accel, RayFlags::NONE, 0, Vec3::ZERO, 0.0, Vec3::ZERO, 0.0);
            let origin: glam::Vec3 = handle.get_world_ray_origin();
        }
    }
    """

    ast = parse_code(code)
    unsafe_block = ast.functions[0].body[0]
    macro_call = unsafe_block.block.statements[0]
    origin = unsafe_block.block.statements[2]

    assert isinstance(unsafe_block, UnsafeBlockNode)
    assert isinstance(macro_call, FunctionCallNode)
    assert macro_call.name == "spirv_std::ray_query!"
    assert macro_call.args == ["let mut handle"]
    assert macro_call.macro_delimiter == "LPAREN"
    assert origin.name == "origin"
    assert origin.vtype == "glam::Vec3"


def test_rust_gpu_compute_shader_method_turbofish_parsing():
    code = """
    fn main() {
        let result = src_range.clone().into_par_iter().map(collatz).collect::<Vec<_>>();
    }
    """

    ast = parse_code(code)
    result = ast.functions[0].body[0]
    collect_call = result.value

    assert isinstance(collect_call, FunctionCallNode)
    assert isinstance(collect_call.name, MemberAccessNode)
    assert collect_call.name.member == "collect<Vec<_>>"


def test_vulkan_shader_examples_compute_atomic_const_generics_parsing():
    code = """
    use spirv_std::arch::atomic_i_add;
    use spirv_std::glam::UVec3;
    use spirv_std::spirv;

    #[repr(C)]
    pub struct UBOOut {
        pub draw_count: i32,
        pub lod_count: [i32; 6],
    }

    #[spirv(compute(threads(16)))]
    pub fn main_cs(
        #[spirv(global_invocation_id)] global_id: UVec3,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] ubo_out: &mut UBOOut,
    ) {
        let lod_level = 0u32;
        unsafe {
            atomic_i_add::<
                i32,
                { spirv_std::memory::Scope::Device as u32 },
                { spirv_std::memory::Semantics::NONE.bits() },
            >(&mut ubo_out.lod_count[lod_level as usize], 1);
        }
    }
    """

    ast = parse_code(code)
    unsafe_block = ast.functions[0].body[1]
    atomic_call = unsafe_block.block.statements[0]

    assert isinstance(unsafe_block, UnsafeBlockNode)
    assert isinstance(atomic_call, FunctionCallNode)
    assert atomic_call.name == (
        "atomic_i_add<i32, {spirv_std::memory::Scope::Device as u32}, "
        "{spirv_std::memory::Semantics::NONE.bits()},>"
    )
    assert isinstance(atomic_call.args[0], ReferenceNode)
    assert isinstance(atomic_call.args[0].expression, ArrayAccessNode)
    assert isinstance(atomic_call.args[0].expression.index, CastNode)
    assert atomic_call.args[0].expression.index.target_type == "usize"


def test_vulkan_shader_examples_emboss_2d_compute_image_macro_parsing():
    # Reduced from https://github.com/Rust-GPU/VulkanShaderExamples commit
    # b29a37eb46802b5ea6882af4808d6887fc184581,
    # shaders/rust/computeshader/emboss/src/lib.rs.
    code = """
    use spirv_std::{glam::{IVec2, UVec3, Vec4, vec4}, spirv, Image};

    #[spirv(compute(threads(16, 16)))]
    pub fn main_cs(
        #[spirv(global_invocation_id)] global_id: UVec3,
        #[spirv(descriptor_set = 0, binding = 0)] input_image: &Image!(
            2D,
            format = rgba8,
            sampled = false
        ),
        #[spirv(descriptor_set = 0, binding = 1)] result_image: &Image!(
            2D,
            format = rgba8,
            sampled = false
        ),
    ) {
        let coord = IVec2::new(global_id.x as i32, global_id.y as i32);
        let rgb: Vec4 = input_image.read(coord);
        let gray = (rgb.x + rgb.y + rgb.z) / 3.0;
        let res = vec4(gray, gray, gray, 1.0);
        unsafe {
            result_image.write(coord, res);
        }
    }
    """

    ast = parse_code(code)
    function = ast.functions[0]

    assert function.attributes[0].args == [
        "compute",
        "(",
        "threads",
        "(",
        "16",
        "16",
        ")",
        ")",
    ]
    assert [(param.name, param.vtype) for param in function.params] == [
        ("global_id", "UVec3"),
        ("input_image", "&Image!(2D, format=rgba8, sampled=false)"),
        ("result_image", "&Image!(2D, format=rgba8, sampled=false)"),
    ]
    assert function.body[0].name == "coord"
    assert isinstance(function.body[1].value, FunctionCallNode)
    assert isinstance(function.body[-1], UnsafeBlockNode)


def test_underscore_parameter_name_parsing():
    code = """
    pub fn fallback(_value: u32) -> u32 {
        return 0;
    }
    """

    ast = parse_code(code)

    assert ast.functions[0].params[0].name == "_value"
    assert ast.functions[0].params[0].vtype == "u32"


def test_cuda_std_tuple_pattern_impl_parameter_parsing():
    code = """
    impl From<(u32, u32)> for GridSize {
        fn from((x, y): (u32, u32)) -> GridSize {
            GridSize::xy(x, y)
        }
    }
    """

    ast = parse_code(code)
    method = ast.impl_blocks[0].functions[0]
    param = method.params[0]

    assert param.name == "_rust_pattern_param_0"
    assert param.vtype == "(u32, u32)"
    assert isinstance(param.pattern, TupleNode)
    assert param.pattern.elements == ["x", "y"]


def test_tuple_struct_and_variant_visibility_parsing():
    code = """
    #[repr(transparent)]
    pub(crate) struct Marker;
    struct Tagged<T>;
    pub(super) struct Pair(pub(crate) f32, pub(self) Vec3<f32>);

    enum Message {
        Data(
            #[location(0)] pub(crate) f32,
            #[location(1)] pub(super) Vec3<f32>,
        ),
        Move { pub(crate) x: f32, pub(super) y: f32 },
    }
    """
    try:
        ast = parse_code(code)

        marker = ast.structs[0]
        assert marker.name == "Marker"
        assert marker.members == []
        assert marker.visibility == "pub(crate)"
        assert [attr.name for attr in marker.attributes] == ["repr"]

        tagged = ast.structs[1]
        assert tagged.name == "Tagged"
        assert tagged.members == []
        assert tagged.generics == ["T"]

        pair = ast.structs[2]
        assert pair.visibility == "pub(super)"
        assert [member.name for member in pair.members] == ["field0", "field1"]
        assert [member.vtype for member in pair.members] == ["f32", "Vec3<f32>"]
        assert [member.visibility for member in pair.members] == [
            "pub(crate)",
            "pub(self)",
        ]

        data = ast.enums[0].variants[0]
        assert data.kind == "tuple"
        assert [field.name for field in data.fields] == ["field0", "field1"]
        assert [field.vtype for field in data.fields] == ["f32", "Vec3<f32>"]
        assert [field.visibility for field in data.fields] == [
            "pub(crate)",
            "pub(super)",
        ]
        assert [[attr.name for attr in field.attributes] for field in data.fields] == [
            ["location"],
            ["location"],
        ]

        move = ast.enums[0].variants[1]
        assert [field.name for field in move.fields] == ["x", "y"]
        assert [field.visibility for field in move.fields] == [
            "pub(crate)",
            "pub(super)",
        ]
    except Exception as e:
        pytest.fail(f"Tuple struct and variant visibility parsing failed: {e}")


def test_struct_where_clause_parsing():
    code = """
    pub struct Buffer<T>
    where
        T: Copy,
    {
        value: T,
    }

    pub(super) struct Pair<T>(T, Vec3<f32>)
    where
        T: Copy + Clone;

    struct Marker<T>
    where
        T: Copy;
    """
    try:
        ast = parse_code(code)

        buffer, pair, marker = ast.structs
        assert buffer.name == "Buffer"
        assert buffer.generics == ["T"]
        assert buffer.where_clauses == [("T", ["Copy"])]
        assert buffer.members[0].vtype == "T"

        assert pair.name == "Pair"
        assert pair.visibility == "pub(super)"
        assert pair.where_clauses == [("T", ["Copy", "Clone"])]
        assert [member.vtype for member in pair.members] == ["T", "Vec3<f32>"]

        assert marker.name == "Marker"
        assert marker.generics == ["T"]
        assert marker.members == []
        assert marker.where_clauses == [("T", ["Copy"])]
    except Exception as e:
        pytest.fail(f"Struct where-clause parsing failed: {e}")


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


def test_attributed_impl_block_parsing_from_spirv_std_image():
    code = """
    #[crate::macros::gen_sample_param_permutations]
    impl<
        SampledType: SampleType<FORMAT, COMPONENTS>,
        const DIM: u32,
        const DEPTH: u32,
        const ARRAYED: u32,
        const MULTISAMPLED: u32,
        const SAMPLED: u32,
        const FORMAT: u32,
        const COMPONENTS: u32,
    > Image<SampledType, DIM, DEPTH, ARRAYED, MULTISAMPLED, SAMPLED, FORMAT, COMPONENTS> {
        pub fn sample<F>(&self, sampler: Sampler) -> SampledType::SampleResult
        where
            F: Float,
        {
            return self.fetch_with(sampler);
        }
    }
    """

    ast = parse_code(code)
    impl_block = ast.impl_blocks[0]

    assert impl_block.struct_name == (
        "Image<SampledType, DIM, DEPTH, ARRAYED, MULTISAMPLED, SAMPLED, FORMAT, "
        "COMPONENTS>"
    )
    assert impl_block.generics == [
        "SampledType: SampleType<FORMAT, COMPONENTS>",
        "const DIM: u32",
        "const DEPTH: u32",
        "const ARRAYED: u32",
        "const MULTISAMPLED: u32",
        "const SAMPLED: u32",
        "const FORMAT: u32",
        "const COMPONENTS: u32",
    ]
    assert impl_block.functions[0].name == "sample"


def test_impl_item_attributes_are_preserved_from_rust_gpu_style_helpers():
    code = """
    struct Kernel;

    impl Kernel {
        #[cfg_attr(target_arch = "spirv", inline)]
        #[doc(alias = "OpImageSampleExplicitLod")]
        pub fn sample(&self) -> u32 {
            return 1;
        }

        #[allow(non_camel_case_types)]
        pub type sample_result = u32;
    }
    """

    ast = parse_code(code)
    impl_block = ast.impl_blocks[0]
    method = impl_block.functions[0]
    alias = impl_block.type_aliases[0]

    assert [attr.name for attr in method.attributes] == ["cfg_attr", "doc"]
    assert method.attributes[0].args == [
        "target_arch",
        "=",
        '"spirv"',
        "inline",
    ]
    assert method.attributes[1].args == ["alias", "=", '"OpImageSampleExplicitLod"']
    assert method.visibility == "pub"

    assert alias.name == "sample_result"
    assert alias.attributes[0].name == "allow"
    assert alias.attributes[0].args == ["non_camel_case_types"]
    assert alias.visibility == "pub"


def test_rust_gpu_block_local_impl_item_is_not_parsed_as_for_loop():
    # Reduced from https://github.com/Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # crates/spirv-std/src/arch/subgroup.rs subgroup_all_equal helper item.
    code = """
    pub fn subgroup_all_equal<T: ScalarComposite>(value: T) -> bool {
        struct Transform(bool);

        impl ScalarOrVectorTransform for Transform {
            fn transform<T: ScalarOrVector>(&mut self, value: T) -> T {
                value
            }
        }

        let mut transform = Transform(true);
        value.transform(&mut transform);
        transform.0
    }
    """

    ast = parse_code(code)
    function = ast.functions[0]

    assert function.name == "subgroup_all_equal"
    assert [type(stmt) for stmt in function.body] == [
        LetNode,
        FunctionCallNode,
        MemberAccessNode,
    ]
    assert function.body[0].name == "transform"
    assert isinstance(function.body[0].value, FunctionCallNode)


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


def test_negative_trait_impl_parsing_from_rust_reference():
    # Source: https://doc.rust-lang.org/reference/items/implementations.html
    code = """
    struct ShaderHandle;

    impl !Send for ShaderHandle {}

    impl<T> !Sync for DeviceBuffer<T>
    where
        T: Copy,
    {}
    """

    ast = parse_code(code)

    send_impl, sync_impl = ast.impl_blocks
    assert send_impl.trait_name == "Send"
    assert send_impl.struct_name == "ShaderHandle"
    assert send_impl.is_negative is True
    assert send_impl.functions == []

    assert sync_impl.generics == ["T"]
    assert sync_impl.trait_name == "Sync"
    assert sync_impl.struct_name == "DeviceBuffer<T>"
    assert sync_impl.where_clauses == [("T", ["Copy"])]
    assert sync_impl.is_negative is True
    assert sync_impl.functions == []


def test_shader_type_token_declaration_names_parsing():
    code = """
    struct Vec2<T> {
        x: T,
        y: T,
    }

    impl<T> Vec2<T> {
        fn x(&self) -> T {
            return self.x;
        }
    }

    enum Option {
        None,
        Some(i32),
        Vec3 { value: i32 },
    }

    trait Result {
        type Vec4;
        fn Some(&self) -> i32;
    }

    type Mat4 = Vec2<f32>;
    """
    try:
        ast = parse_code(code)

        assert ast.structs[0].name == "Vec2"
        assert [member.name for member in ast.structs[0].members] == ["x", "y"]
        assert ast.impl_blocks[0].struct_name == "Vec2<T>"
        assert ast.impl_blocks[0].functions[0].name == "x"
        assert ast.enums[0].name == "Option"
        assert [variant.name for variant in ast.enums[0].variants] == [
            "None",
            "Some",
            "Vec3",
        ]
        assert ast.traits[0].name == "Result"
        assert ast.traits[0].associated_types[0].name == "Vec4"
        assert ast.traits[0].methods[0].name == "Some"
        assert ast.type_aliases[0].name == "Mat4"
    except Exception as e:
        pytest.fail(f"Shader type-token declaration name parsing failed: {e}")


def test_use_statement_parsing():
    code = """
    use std::collections::HashMap;
    use gpu::*;
    use math::{sin, cos, tan};
    use glam::Vec3 as GVec3;
    use crate::math::Mat4 as CMat4;
    use crate::math as shader_math;
    use glam::{Vec3 as GVec3Grouped, Vec4 as GVec4Grouped};
    use crate::math::{Mat4 as CMat4Grouped, Unknown};
    use crate::{math as grouped_math, math::{Vec3 as NestedVec3, Mat4 as NestedMat4}, color::Vec4 as NestedVec4};
    use crate::{math::{*, Vec3 as GlobVec3}, color::*};
    use {shared::*, spirv_std::num_traits::Float};
    pub use crate::shared as shared;
    pub(crate) use crate::internal::Vec4 as InternalVec4;
    """
    try:
        ast = parse_code(code)
        assert len(ast.use_statements) >= 13
        use_stmt = ast.use_statements[0]
        assert isinstance(use_stmt, UseNode)
        assert ast.use_statements[2].items == [
            {"path": "sin", "alias": None},
            {"path": "cos", "alias": None},
            {"path": "tan", "alias": None},
        ]
        assert ast.use_statements[3].path == "glam::Vec3"
        assert ast.use_statements[3].alias == "GVec3"
        assert ast.use_statements[4].path == "crate::math::Mat4"
        assert ast.use_statements[4].alias == "CMat4"
        assert ast.use_statements[5].path == "crate::math"
        assert ast.use_statements[5].alias == "shader_math"
        assert ast.use_statements[6].path == (
            "glam::{Vec3 as GVec3Grouped, Vec4 as GVec4Grouped}"
        )
        assert ast.use_statements[6].items == [
            {"path": "Vec3", "alias": "GVec3Grouped"},
            {"path": "Vec4", "alias": "GVec4Grouped"},
        ]
        assert ast.use_statements[7].items == [
            {"path": "Mat4", "alias": "CMat4Grouped"},
            {"path": "Unknown", "alias": None},
        ]
        assert ast.use_statements[8].items == [
            {"path": "math", "alias": "grouped_math"},
            {"path": "math::Vec3", "alias": "NestedVec3"},
            {"path": "math::Mat4", "alias": "NestedMat4"},
            {"path": "color::Vec4", "alias": "NestedVec4"},
        ]
        assert ast.use_statements[9].items == [
            {"path": "math::*", "alias": None},
            {"path": "math::Vec3", "alias": "GlobVec3"},
            {"path": "color::*", "alias": None},
        ]
        assert ast.use_statements[10].items == [
            {"path": "shared::*", "alias": None},
            {"path": "spirv_std::num_traits::Float", "alias": None},
        ]
        assert ast.use_statements[10].path == (
            "{shared::*, spirv_std::num_traits::Float}"
        )
        assert ast.use_statements[11].visibility == "pub"
        assert ast.use_statements[11].alias == "shared"
        assert ast.use_statements[12].visibility == "pub(crate)"
        assert ast.use_statements[12].alias == "InternalVec4"
    except Exception as e:
        pytest.fail(f"Use statement parsing failed: {e}")


def test_underscore_use_alias_parsing():
    code = """
    use spirv_std as _;
    use glam::{Vec3 as _, Vec4};
    """

    ast = parse_code(code)

    assert ast.use_statements[0].path == "spirv_std"
    assert ast.use_statements[0].alias == "_"
    assert ast.use_statements[1].items == [
        {"path": "Vec3", "alias": "_"},
        {"path": "Vec4", "alias": None},
    ]


def test_vulkan_shader_examples_block_scoped_use_parsing():
    code = """
    fn main_fs(head_index_image: &Image2d, coord: IVec2, node_idx: u32) -> u32 {
        let prev_head_idx = unsafe {
            use spirv_std::memory::{Scope, Semantics};
            use spirv_std::arch::atomic_exchange;
            let current = head_index_image.read(coord).x;
            head_index_image.write(coord, node_idx.into());
            current
        };
        prev_head_idx
    }
    """

    ast = parse_code(code)
    unsafe_block = ast.functions[0].body[0].value

    assert isinstance(unsafe_block, UnsafeBlockNode)
    assert isinstance(unsafe_block.block.statements[0], UseNode)
    assert unsafe_block.block.statements[0].path == (
        "spirv_std::memory::{Scope, Semantics}"
    )
    assert isinstance(unsafe_block.block.statements[1], UseNode)
    assert unsafe_block.block.statements[1].path == "spirv_std::arch::atomic_exchange"
    assert unsafe_block.block.expression == "current"


def test_cfg_attributes_preserved_on_use_and_function_items():
    code = """
    #[cfg(feature = "gpu")]
    pub use crate::gpu::*;

    #[cfg(target_os = "unknown")]
    pub fn shader_entry() -> f32 {
        return 1.0;
    }
    """
    ast = parse_code(code)

    use_stmt = ast.use_statements[0]
    assert use_stmt.visibility == "pub"
    assert use_stmt.path == "crate::gpu::*"
    assert use_stmt.attributes[0].name == "cfg"
    assert use_stmt.attributes[0].args == ["feature", "=", '"gpu"']

    function = ast.functions[0]
    assert function.visibility == "pub"
    assert function.attributes[0].name == "cfg"
    assert function.attributes[0].args == ["target_os", "=", '"unknown"']


def test_keyword_attribute_path_parses_from_rust_cuda_codegen_entrypoint():
    code = """
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn __rustc_codegen_backend() {
        return;
    }
    """
    ast = parse_code(code)

    function = ast.functions[0]
    assert function.name == "__rustc_codegen_backend"
    assert function.attributes[0].name == "unsafe"
    assert function.attributes[0].args == ["no_mangle"]
    assert function.visibility == "pub"
    assert function.is_unsafe is True
    assert function.abi == "C"


def test_inner_and_nested_spirv_attributes_parse_from_rust_gpu_style_source():
    code = """
    #![cfg_attr(target_arch = "spirv", no_std)]
    #![deny(warnings)]

    use core::f32::consts::PI;
    use spirv_std::spirv;

    #[spirv(compute(threads(64)))]
    pub fn main_cs(
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] values: &mut [u32],
    ) {
        let scale = PI as f32;
        values[0] = values[0] + scale as u32;
    }
    """
    ast = parse_code(code)

    assert [use.path for use in ast.use_statements] == [
        "core::f32::consts::PI",
        "spirv_std::spirv",
    ]
    function = ast.functions[0]
    assert function.name == "main_cs"
    assert function.attributes[0].name == "spirv"
    assert function.params[0].attributes[0].args == [
        "storage_buffer",
        "descriptor_set",
        "=",
        "0",
        "binding",
        "=",
        "0",
    ]


def test_rust_gpu_spirv_entry_point_name_attribute_parsing():
    code = """
    use spirv_std::spirv;

    #[spirv(compute(threads(256), entry_point_name = "find_matches_and_compute_f4"))]
    pub fn shader_body() {}

    #[spirv(vertex(entry_point_name = "renamed_vs"))]
    pub fn vertex_body() {}
    """
    ast = parse_code(code)

    compute = ast.functions[0]
    assert compute.attributes[0].args == [
        "compute",
        "(",
        "threads",
        "(",
        "256",
        ")",
        "entry_point_name",
        "=",
        '"find_matches_and_compute_f4"',
        ")",
    ]

    vertex = ast.functions[1]
    assert vertex.attributes[0].args == [
        "vertex",
        "(",
        "entry_point_name",
        "=",
        '"renamed_vs"',
        ")",
    ]


def test_rust_gpu_image_macro_type_parsing():
    code = """
    use spirv_std::{spirv, Image};

    type Image2d = Image!(2D, type=f32, sampled);

    #[spirv(fragment)]
    pub fn main_fs(
        #[spirv(descriptor_set = 0, binding = 0)] sampled_tex: &Image!(2D, type=f32, sampled),
        #[spirv(descriptor_set = 0, binding = 1)] storage_tex: &Image!(2D, format=rgba16f, sampled=false),
    ) {}
    """
    ast = parse_code(code)

    assert ast.type_aliases[0].alias_type == "Image!(2D, type=f32, sampled)"
    function = ast.functions[0]
    assert [param.vtype for param in function.params] == [
        "&Image!(2D, type=f32, sampled)",
        "&Image!(2D, format=rgba16f, sampled=false)",
    ]
    assert function.params[0].attributes[0].args == [
        "descriptor_set",
        "=",
        "0",
        "binding",
        "=",
        "0",
    ]


def test_rust_gpu_extra_image_macro_dimensions_parsing_from_upstream():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # tests/compiletests/ui/image/query/{rect_image_query_size,storage_image_query_size}.rs
    # and tests/compiletests/ui/image/read_subpass.rs.
    code = """
    use spirv_std::{spirv, Image};

    #[spirv(fragment)]
    pub fn main(
        #[spirv(descriptor_set = 0, binding = 0)] rect_storage: &Image!(rect, type=f32, sampled=false),
        #[spirv(descriptor_set = 1, binding = 1)] rect_storage_array: &Image!(rect, type=f32, sampled=false, arrayed),
        #[spirv(descriptor_set = 2, binding = 2)] buffer_image: &Image!(buffer, type=f32, sampled=false),
        #[spirv(descriptor_set = 3, binding = 3)] sampled_buffer: &Image!(buffer, type=u32, sampled),
        #[spirv(descriptor_set = 4, binding = 4, input_attachment_index = 0)] input_attachment: &Image!(subpass, type=f32, sampled=false),
    ) {}
    """
    ast = parse_code(code)

    function = ast.functions[0]
    assert [param.vtype for param in function.params] == [
        "&Image!(rect, type=f32, sampled=false)",
        "&Image!(rect, type=f32, sampled=false, arrayed)",
        "&Image!(buffer, type=f32, sampled=false)",
        "&Image!(buffer, type=u32, sampled)",
        "&Image!(subpass, type=f32, sampled=false)",
    ]
    assert function.params[4].attributes[0].args == [
        "descriptor_set",
        "=",
        "4",
        "binding",
        "=",
        "4",
        "input_attachment_index",
        "=",
        "0",
    ]


def test_rust_gpu_sampled_image_generic_type_parsing():
    code = """
    use spirv_std::{spirv, Image};
    use spirv_std::image::SampledImage;

    #[spirv(fragment)]
    pub fn main_fs(
        #[spirv(descriptor_set = 0, binding = 0)] tex: &SampledImage<Image!(2D, type=f32, sampled)>,
    ) {}
    """
    ast = parse_code(code)

    function = ast.functions[0]
    assert function.params[0].vtype == (
        "&SampledImage<Image!(2D, type = f32, sampled)>"
    )
    assert function.params[0].attributes[0].args == [
        "descriptor_set",
        "=",
        "0",
        "binding",
        "=",
        "0",
    ]


def test_rust_gpu_runtime_descriptor_array_parsing_from_upstream():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # tests/compiletests/ui/storage_class/runtime_descriptor_array.rs.
    code = """
    use spirv_std::spirv;
    use spirv_std::{Image, RuntimeArray, Sampler};

    #[spirv(fragment)]
    pub fn main(
        #[spirv(descriptor_set = 0, binding = 0)] sampler: &Sampler,
        #[spirv(descriptor_set = 0, binding = 1)] slice: &RuntimeArray<Image!(2D, type=f32, sampled)>,
        output: &mut glam::Vec4,
    ) {
        let img = unsafe { slice.index(5) };
        let v2 = glam::Vec2::new(0.0, 1.0);
        let r1: glam::Vec4 = img.sample(*sampler, v2);
        *output = r1;
    }
    """
    ast = parse_code(code)

    function = ast.functions[0]
    assert [param.vtype for param in function.params] == [
        "&Sampler",
        "&RuntimeArray<Image!(2D, type = f32, sampled)>",
        "&mut glam::Vec4",
    ]
    assert function.params[1].attributes[0].args == [
        "descriptor_set",
        "=",
        "0",
        "binding",
        "=",
        "1",
    ]


def test_rust_gpu_builtin_spirv_parameter_attributes_parse():
    code = """
    use spirv_std::spirv;

    #[spirv(fragment)]
    pub fn main_fs(
        #[spirv(frag_coord)] in_frag_coord: Vec4,
        #[spirv(front_facing)] is_front_facing: bool,
    ) {}

    #[spirv(vertex)]
    pub fn main_vs(
        #[spirv(instance_index)] instance_index: u32,
    ) {}

    #[spirv(compute(threads(64)))]
    pub fn main_cs(
        #[spirv(local_invocation_index)] local_index: u32,
    ) {}
    """
    ast = parse_code(code)

    fragment = ast.functions[0]
    assert [param.attributes[0].args for param in fragment.params] == [
        ["frag_coord"],
        ["front_facing"],
    ]

    vertex = ast.functions[1]
    assert vertex.params[0].attributes[0].args == ["instance_index"]

    compute = ast.functions[2]
    assert compute.params[0].attributes[0].args == ["local_invocation_index"]


def test_rust_gpu_reduce_subgroup_builtins_parse_from_upstream_example():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/shaders/reduce/src/lib.rs compute subgroup reduction entry point.
    code = """
    use spirv_std::glam::UVec3;
    use spirv_std::spirv;

    #[spirv(compute(threads(256)))]
    pub fn main(
        #[spirv(global_invocation_id)] global_invocation_id: UVec3,
        #[spirv(local_invocation_id)] local_invocation_id: UVec3,
        #[spirv(subgroup_local_invocation_id)] subgroup_local_invocation_id: u32,
        #[spirv(workgroup_id)] workgroup_id: UVec3,
        #[spirv(subgroup_id)] subgroup_id: u32,
        #[spirv(num_subgroups)] num_subgroups: u32,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[u32],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [u32],
        #[spirv(workgroup)] shared: &mut [u32; 256],
    ) {
        let local_invocation_id_x = local_invocation_id.x as usize;
        if subgroup_local_invocation_id == 0 {
            shared[subgroup_id as usize] = 0;
        }
        if local_invocation_id_x == 0 {
            output[workgroup_id.x as usize] = input[0];
        }
    }
    """

    ast = parse_code(code)
    function = ast.functions[0]

    assert function.attributes[0].args == [
        "compute",
        "(",
        "threads",
        "(",
        "256",
        ")",
        ")",
    ]
    assert [(param.name, param.vtype) for param in function.params] == [
        ("global_invocation_id", "UVec3"),
        ("local_invocation_id", "UVec3"),
        ("subgroup_local_invocation_id", "u32"),
        ("workgroup_id", "UVec3"),
        ("subgroup_id", "u32"),
        ("num_subgroups", "u32"),
        ("input", "&u32[]"),
        ("output", "&mut u32[]"),
        ("shared", "&mut u32[256]"),
    ]
    assert [param.attributes[0].args[0] for param in function.params] == [
        "global_invocation_id",
        "local_invocation_id",
        "subgroup_local_invocation_id",
        "workgroup_id",
        "subgroup_id",
        "num_subgroups",
        "storage_buffer",
        "storage_buffer",
        "workgroup",
    ]


def test_rust_gpu_subgroup_composite_primitive_name_binding_parse():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # tests/compiletests/ui/arch/subgroup/subgroup_composite_all_equals.rs.
    code = """
    fn disassembly(my_struct: MyStruct) -> bool {
        subgroup_all_equal(my_struct)
    }

    #[spirv(compute(threads(32)))]
    pub fn main(inv_id: u32, inv_id_3d: UVec3, output: &mut u32) {
        unsafe {
            let my_struct = MyStruct {
                a: inv_id as f32,
                b: inv_id_3d,
                c: Nested(5i32 - inv_id as i32),
                d: Zst,
            };
            let bool = disassembly(my_struct);
            *output = u32::from(bool);
        }
    }
    """

    ast = parse_code(code)
    unsafe_block = ast.functions[1].body[0]
    bool_binding = unsafe_block.block.statements[1]

    assert isinstance(unsafe_block, UnsafeBlockNode)
    assert isinstance(bool_binding, LetNode)
    assert bool_binding.name == "bool"
    assert isinstance(bool_binding.value, FunctionCallNode)
    assert bool_binding.value.name == "disassembly"


def test_rust_gpu_vector_associated_constants_parse_from_upstream_compiletests():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # tests/compiletests/ui/spirv-attr/matrix-type.rs,
    # tests/compiletests/ui/arch/subgroup/subgroup_composite.rs,
    # and tests/difftests/tests/lang/core/ops/vector_ops/vector_ops-rust/src/lib.rs.
    code = """
    use glam::{UVec3, Vec2, Vec3, Vec4};

    fn main() {
        let zero = glam::Vec3::ZERO;
        let axis = Vec3::X;
        let subgroup_out = UVec3::ZERO;
        let component = Vec4::Y.y;
        let scalar = Vec2::ZERO.x;
    }
    """

    ast = parse_code(code)
    body = ast.functions[0].body

    assert body[0].value == "glam::Vec3::ZERO"
    assert body[1].value == "Vec3::X"
    assert body[2].value == "UVec3::ZERO"
    assert isinstance(body[3].value, MemberAccessNode)
    assert body[3].value.object == "Vec4::Y"
    assert body[3].value.member == "y"
    assert isinstance(body[4].value, MemberAccessNode)
    assert body[4].value.object == "Vec2::ZERO"
    assert body[4].value.member == "x"


def test_rust_gpu_mouse_shader_perp_dot_method_parse_from_upstream_example():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/shaders/mouse-shader/src/lib.rs main_fs background expression.
    code = """
    fn determinant_magnitude(to_frag: Vec2, start_to_end: Vec2) -> f32 {
        to_frag.perp_dot(start_to_end).abs()
    }
    """

    ast = parse_code(code)
    final_call = ast.functions[0].body[0]
    perp_dot_call = final_call.name.object

    assert isinstance(final_call, FunctionCallNode)
    assert isinstance(final_call.name, MemberAccessNode)
    assert final_call.name.member == "abs"
    assert isinstance(perp_dot_call, FunctionCallNode)
    assert isinstance(perp_dot_call.name, MemberAccessNode)
    assert perp_dot_call.name.member == "perp_dot"
    assert perp_dot_call.args == ["start_to_end"]


def test_rust_gpu_compute_collatz_option_chain_parse_from_upstream_example():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/shaders/compute-shader/src/lib.rs Collatz compute entry point.
    code = """
    use glam::UVec3;
    use spirv_std::{glam, spirv};

    pub fn collatz(mut n: u32) -> Option<u32> {
        let mut i = 0;
        if n == 0 {
            return None;
        }
        while n != 1 {
            n = if n.is_multiple_of(2) {
                n / 2
            } else {
                if n >= 0x5555_5555 {
                    return None;
                }
                3 * n + 1
            };
            i += 1;
        }
        Some(i)
    }

    #[spirv(compute(threads(64)))]
    pub fn main_cs(
        #[spirv(global_invocation_id)] id: UVec3,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
        prime_indices: &mut [u32],
    ) {
        let index = id.x as usize;
        prime_indices[index] = collatz(prime_indices[index]).unwrap_or(u32::MAX);
    }
    """

    ast = parse_code(code)
    collatz, main_cs = ast.functions

    assert collatz.name == "collatz"
    assert collatz.return_type == "Option<u32>"
    assert [(param.name, param.vtype) for param in collatz.params] == [("n", "u32")]
    assert isinstance(collatz.body[2], WhileNode)
    loop_assign = collatz.body[2].body[0]
    assert isinstance(loop_assign, AssignmentNode)
    assert isinstance(loop_assign.right, IfNode)
    assert isinstance(loop_assign.right.condition, FunctionCallNode)
    assert loop_assign.right.condition.name.member == "is_multiple_of"

    assert main_cs.attributes[0].args == [
        "compute",
        "(",
        "threads",
        "(",
        "64",
        ")",
        ")",
    ]
    assert [(param.name, param.vtype) for param in main_cs.params] == [
        ("id", "UVec3"),
        ("prime_indices", "&mut u32[]"),
    ]
    assert [param.attributes[0].args[0] for param in main_cs.params] == [
        "global_invocation_id",
        "storage_buffer",
    ]
    output_assignment = main_cs.body[1]
    assert isinstance(output_assignment.left, ArrayAccessNode)
    assert output_assignment.left.array == "prime_indices"
    unwrap_call = output_assignment.right
    assert isinstance(unwrap_call.name, MemberAccessNode)
    assert unwrap_call.name.member == "unwrap_or"
    assert isinstance(unwrap_call.name.object, FunctionCallNode)
    assert unwrap_call.name.object.name == "collatz"
    assert unwrap_call.args == ["u32::MAX"]


def test_restricted_visibility_parsing():
    code = """
    #[inline]
    pub(in crate::math) fn helper() {}

    pub(super) struct Data {
        pub(crate) value: f32,
        pub(self) local: f32,
    }

    pub(in crate::math) type Real = f32;
    pub(super) const PI: f32 = 3.0;

    impl Data {
        pub(super) fn update(&self) {}
        pub(in crate::math) type Output = f32;
    }
    """
    try:
        ast = parse_code(code)

        assert ast.functions[0].visibility == "pub(in crate::math)"
        assert ast.structs[0].visibility == "pub(super)"
        assert [member.visibility for member in ast.structs[0].members] == [
            "pub(crate)",
            "pub(self)",
        ]
        assert ast.type_aliases[0].visibility == "pub(in crate::math)"
        assert ast.global_variables[0].visibility == "pub(super)"
        assert ast.impl_blocks[0].functions[0].visibility == "pub(super)"
        assert ast.impl_blocks[0].type_aliases[0].visibility == ("pub(in crate::math)")
    except Exception as e:
        pytest.fail(f"Restricted visibility parsing failed: {e}")


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


def test_for_loop_pattern_parsing():
    code = """
    fn test_for_patterns() {
        for _ in 0..4 {
            tick();
        }
        for (_, value) in pairs {
            consume(value);
        }
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        assert body[0].pattern == "_"

        tuple_pattern = body[1].pattern
        assert isinstance(tuple_pattern, TupleNode)
        assert tuple_pattern.elements == ["_", "value"]
    except Exception as e:
        pytest.fail(f"For loop pattern parsing failed: {e}")


def test_for_loop_enumerate_iter_pattern_parsing():
    code = """
    fn test_enumerate(values: Buffer) {
        for (i, value) in values.iter().enumerate() {
            consume(i, value);
        }
    }
    """
    try:
        ast = parse_code(code)
        loop = ast.functions[0].body[0]

        assert isinstance(loop.pattern, TupleNode)
        assert loop.pattern.elements == ["i", "value"]
        assert isinstance(loop.iterable, FunctionCallNode)
        assert isinstance(loop.iterable.name, MemberAccessNode)
        assert loop.iterable.name.member == "enumerate"
    except Exception as e:
        pytest.fail(f"For loop enumerate pattern parsing failed: {e}")


def test_for_loop_step_by_range_parsing():
    code = """
    fn test_for_step_by() {
        for i in (0..10).step_by(2) {
            use_index(i);
        }
        for j in (1..=9).step_by(step) {
            use_index(j);
        }
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        first_iterable = body[0].iterable
        assert isinstance(first_iterable, FunctionCallNode)
        assert isinstance(first_iterable.name, MemberAccessNode)
        assert first_iterable.name.member == "step_by"
        assert isinstance(first_iterable.name.object, RangeNode)
        assert first_iterable.name.object.start == "0"
        assert first_iterable.name.object.end == "10"
        assert first_iterable.args == ["2"]

        second_iterable = body[1].iterable
        assert isinstance(second_iterable.name.object, RangeNode)
        assert second_iterable.name.object.inclusive is True
        assert second_iterable.args == ["step"]
    except Exception as e:
        pytest.fail(f"For loop step_by range parsing failed: {e}")


def test_for_loop_rev_range_parsing():
    code = """
    fn test_for_rev(limit: i32) {
        for i in (0..4).rev() {
            use_index(i);
        }
        for j in (1..=4).rev() {
            use_index(j);
        }
        for k in (0..limit).rev() {
            use_index(k);
        }
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        first_iterable = body[0].iterable
        assert isinstance(first_iterable, FunctionCallNode)
        assert isinstance(first_iterable.name, MemberAccessNode)
        assert first_iterable.name.member == "rev"
        assert isinstance(first_iterable.name.object, RangeNode)
        assert first_iterable.name.object.start == "0"
        assert first_iterable.name.object.end == "4"
        assert first_iterable.args == []

        second_range = body[1].iterable.name.object
        assert isinstance(second_range, RangeNode)
        assert second_range.inclusive is True

        third_range = body[2].iterable.name.object
        assert isinstance(third_range, RangeNode)
        assert third_range.end == "limit"
    except Exception as e:
        pytest.fail(f"For loop rev range parsing failed: {e}")


def test_for_loop_step_by_rev_range_parsing():
    code = """
    fn test_for_step_by_rev() {
        for i in (0..10).step_by(2).rev() {
            use_index(i);
        }
    }
    """
    try:
        ast = parse_code(code)
        iterable = ast.functions[0].body[0].iterable

        assert isinstance(iterable, FunctionCallNode)
        assert isinstance(iterable.name, MemberAccessNode)
        assert iterable.name.member == "rev"
        step_call = iterable.name.object
        assert isinstance(step_call, FunctionCallNode)
        assert isinstance(step_call.name, MemberAccessNode)
        assert step_call.name.member == "step_by"
        assert isinstance(step_call.name.object, RangeNode)
        assert step_call.name.object.start == "0"
        assert step_call.name.object.end == "10"
        assert step_call.args == ["2"]
    except Exception as e:
        pytest.fail(f"For loop step_by rev range parsing failed: {e}")


def test_for_loop_range_bounds_parse_additive_expressions():
    code = """
    fn test_for_range_bounds(count: i32, start: i32, end: i32) {
        for i in 0..count + 1 {
            use_index(i);
        }
        for j in start + 1..=end - 1 {
            use_index(j);
        }
    }
    """

    ast = parse_code(code)
    body = ast.functions[0].body

    first_iterable = body[0].iterable
    assert isinstance(first_iterable, RangeNode)
    assert first_iterable.start == "0"
    assert isinstance(first_iterable.end, BinaryOpNode)
    assert first_iterable.end.op == "+"
    assert first_iterable.end.left == "count"
    assert first_iterable.end.right == "1"

    second_iterable = body[1].iterable
    assert isinstance(second_iterable, RangeNode)
    assert second_iterable.inclusive is True
    assert isinstance(second_iterable.start, BinaryOpNode)
    assert second_iterable.start.op == "+"
    assert second_iterable.start.left == "start"
    assert second_iterable.start.right == "1"
    assert isinstance(second_iterable.end, BinaryOpNode)
    assert second_iterable.end.op == "-"
    assert second_iterable.end.left == "end"
    assert second_iterable.end.right == "1"


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


def test_raw_pointer_type_parsing():
    code = """
    const NULL_PTR: *const i32 = null();

    fn copy<T>(src: *const T, dst: *mut T, count: usize);
    """

    ast = parse_code(code)

    assert ast.global_variables[0].vtype == "*const i32"
    assert ast.functions[0].params[0].vtype == "*const T"
    assert ast.functions[0].params[1].vtype == "*mut T"


def test_function_pointer_type_parsing():
    code = """
    struct Position(u32);

    fn use_cmp(cmp: fn(&Position) -> u32) {
        let value = cmp(&Position(0));
    }
    """

    ast = parse_code(code)

    assert ast.functions[0].params[0].vtype == "fn(&Position) -> u32"


def test_named_function_pointer_type_alias_parsing_from_rust_gpu():
    # Reduced from Rust-GPU/rust-gpu crates/rustc_codegen_spirv/src/naga_transpile.rs.
    code = """
    pub type NagaTranspile = fn(
        sess: &Session,
        spv_binary: &[u32],
    ) -> Result<(), ErrorGuaranteed>;
    """

    ast = parse_code(code)

    assert ast.type_aliases[0].alias_type == (
        "fn(&Session, &u32[]) -> Result<(), ErrorGuaranteed>"
    )


def test_nested_qualified_associated_type_generic_from_rust_gpu():
    # Reduced from Rust-GPU/rust-gpu crates/rustc_codegen_spirv/src/lib.rs.
    code = """
    pub type BackendModule = ModuleCodegen<<Self as WriteBackendMethods>::Module>;
    """

    ast = parse_code(code)

    assert ast.type_aliases[0].alias_type == (
        "ModuleCodegen<<Self as WriteBackendMethods>::Module>"
    )


def test_higher_ranked_function_pointer_tuple_struct_field_from_bevy_error_handler():
    # Reduced from https://github.com/bevyengine/bevy commit
    # 67f441da62424f83a1bb6e3f5145034e0583d495,
    # crates/bevy_render/src/error_handler.rs.
    code = """
    pub struct RenderErrorHandler(
        pub for<'a> fn(&'a RenderError, &'a mut World, &'a mut World) -> RenderErrorPolicy,
    );
    """

    ast = parse_code(code)

    assert ast.structs[0].members[0].vtype == (
        "for<'a> fn(&RenderError, &mut World, &mut World) -> RenderErrorPolicy"
    )


def test_double_reference_type_and_expression_parsing():
    code = """
    fn take_slice(value: &&[u32]) {
        let nested = &&123;
    }
    """

    ast = parse_code(code)

    assert ast.functions[0].params[0].vtype == "&&u32[]"
    nested = ast.functions[0].body[0].value
    assert isinstance(nested, ReferenceNode)
    assert isinstance(nested.expression, ReferenceNode)


def test_qualified_associated_path_expression_parsing():
    code = """
    fn add(a: u32, b: u32) -> u32 {
        <u32 as core::ops::Add>::add(a, b)
    }
    """

    ast = parse_code(code)
    call = ast.functions[0].body[0]

    assert isinstance(call, FunctionCallNode)
    assert call.name == "<u32 as core::ops::Add>::add"


def test_anonymous_const_item_parsing():
    code = """
    const _: () = {};
    """

    ast = parse_code(code)

    assert len(ast.global_variables) == 1
    assert isinstance(ast.global_variables[0], ConstNode)
    assert ast.global_variables[0].name == "_"
    assert ast.global_variables[0].vtype == "()"


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


def test_const_expression_array_size_parsing_from_rust_gpu_const_generics():
    code = """
    fn shade<const LANES: usize>(
        values: [Vec4; {LANES + 1}],
        scratch: [[f32; 2]; {LANES + 1}],
    ) {
        let local: [u32; {LANES + 1}];
    }
    """

    ast = parse_code(code)
    function = ast.functions[0]

    assert function.params[0].vtype == "Vec4[LANES+1]"
    assert function.params[1].vtype == "f32[LANES+1][2]"
    assert function.body[0].vtype == "u32[LANES+1]"


def test_cast_array_size_spacing_parsing_from_wgpu_backend_list():
    # Reduced from:
    # Repo: https://github.com/gfx-rs/wgpu
    # Commit: 6fbbb0fbb7e8d546224f84a1efe4337b70654cf6
    # Path: wgpu-types/src/backend.rs
    code = """
    impl Backend {
        pub const ALL: [Backend; Backends::all().bits().count_ones() as usize] = [
            Self::Noop,
        ];
    }
    """

    ast = parse_code(code)
    const = ast.impl_blocks[0].associated_consts[0]

    assert const.vtype == "Backend[Backends::all().bits().count_ones() as usize]"


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


def test_nested_array_generic_reference_type_parsing_from_naga_criterion():
    # Reduced from gfx-rs/naga commit
    # d0f28c0b1a3c772e55e68db1c47eff5131cb6732,
    # benches/criterion.rs parse_glsl inputs parameter.
    code = """
    fn parse_glsl(inputs: &[Box<[u8]>]) {
        return;
    }
    """

    ast = parse_code(code)

    assert ast.functions[0].params[0].vtype == "&Box<[u8]>[]"


def test_lifetime_reference_type_parsing():
    code = """
    fn borrow<'a>(value: &'a Vec3<f32>) -> &'a Vec3<f32> {
        return value;
    }

    fn borrow_mut<'a>(value: &'a mut Vec3<f32>) -> &'a mut Vec3<f32> {
        return value;
    }
    """
    try:
        ast = parse_code(code)
        borrowed, borrowed_mut = ast.functions

        assert borrowed.generics == ["'a"]
        assert borrowed.params[0].vtype == "&Vec3<f32>"
        assert borrowed.return_type == "&Vec3<f32>"
        assert borrowed_mut.generics == ["'a"]
        assert borrowed_mut.params[0].vtype == "&mut Vec3<f32>"
        assert borrowed_mut.return_type == "&mut Vec3<f32>"
    except Exception as e:
        pytest.fail(f"Lifetime reference type parsing failed: {e}")


def test_lifetime_reference_generic_type_argument_spacing_from_corpus():
    code = """
    struct FunctionIter<'a> {
        module: PhantomData<&'a &'a Module>,
        next: Option<&'a Value>,
    }

    fn matrix_motion_transform_from_handle(
        handle: TraversableHandle,
    ) -> Option<&'static MatrixMotionTransform> {
        return core::mem::transmute(handle);
    }
    """

    ast = parse_code(code)

    assert ast.structs[0].members[1].vtype == "Option<&'a Value>"
    assert ast.functions[0].return_type == "Option<&'static MatrixMotionTransform>"


def test_lifetime_receiver_parsing_from_rust_gpu_wgpu_runner():
    # Reduced from https://github.com/Rust-GPU/rust-gpu.git commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/runners/wgpu/src/lib.rs CompiledShaderModules::spv_module_for_entry_point.
    code = """
    struct CompiledShaderModules;

    impl CompiledShaderModules {
        fn spv_module_for_entry_point<'a>(
            &'a self,
            wanted_entry: &str,
        ) -> wgpu::ShaderModuleDescriptor<'a> {
            unreachable!();
        }
    }
    """
    try:
        ast = parse_code(code)
        impl_block = ast.impl_blocks[0]
        method = impl_block.functions[0]

        assert impl_block.struct_name == "CompiledShaderModules"
        assert method.generics == ["'a"]
        assert [(param.name, param.vtype) for param in method.params] == [
            ("self", "&Self"),
            ("wanted_entry", "&str"),
        ]
        assert method.return_type == "wgpu::ShaderModuleDescriptor<'a>"
    except Exception as e:
        pytest.fail(f"Lifetime receiver parsing failed: {e}")


def test_explicit_self_receiver_types_from_rust_reference():
    # The Rust Reference documents typed method receivers such as
    # self: Box<Self> and mutable receiver bindings.
    code = """
    use std::sync::Arc;

    trait ShaderReceivers: Sized {
        fn by_box(self: Box<Self>);
        fn modify(mut self: Box<Self>);
    }

    struct ShaderHandle;

    impl ShaderHandle {
        fn by_arc(self: Arc<Self>) -> Arc<Self> {
            self
        }
    }
    """

    ast = parse_code(code)
    by_box, modify = ast.traits[0].methods
    by_arc = ast.impl_blocks[0].functions[0]

    assert [(param.name, param.vtype) for param in by_box.params] == [
        ("self", "Box<Self>")
    ]
    assert [(param.name, param.vtype) for param in modify.params] == [
        ("self", "Box<Self>")
    ]
    assert modify.params[0].is_mutable is True
    assert [(param.name, param.vtype) for param in by_arc.params] == [
        ("self", "Arc<Self>")
    ]
    assert by_arc.return_type == "Arc<Self>"
    assert by_arc.body == ["self"]


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


def test_char_literal_expression_parsing():
    code = r"""
    fn test_chars(c: char) -> char {
        let letter = 'a';
        let newline = '\n';
        match c {
            'a' => hit_a(),
            '\n' => hit_newline(),
            _ => hit_default(),
        }
        return 'z';
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        assert body[0].value == "'a'"
        assert body[1].value == r"'\n'"

        match_stmt = body[2]
        assert isinstance(match_stmt, MatchNode)
        assert match_stmt.arms[0].pattern == "'a'"
        assert match_stmt.arms[1].pattern == r"'\n'"

        assert isinstance(body[3], ReturnNode)
        assert body[3].value == "'z'"
    except Exception as e:
        pytest.fail(f"Char literal expression parsing failed: {e}")


def test_raw_string_literal_expression_parsing():
    code = r"""
    fn test_raw_strings(s: String) -> String {
        let plain = r"plain raw";
        let quoted = r#"This is a "raw" string"#;
        let hashed = r##"This has "# in it"##;
        match s {
            r#"ok"# => hit_ok(),
            _ => hit_default(),
        }
        return r#"done"#;
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        assert body[0].value == 'r"plain raw"'
        assert body[1].value == 'r#"This is a "raw" string"#'
        assert body[2].value == 'r##"This has "# in it"##'

        match_stmt = body[3]
        assert isinstance(match_stmt, MatchNode)
        assert match_stmt.arms[0].pattern == 'r#"ok"#'

        assert isinstance(body[4], ReturnNode)
        assert body[4].value == 'r#"done"#'
    except Exception as e:
        pytest.fail(f"Raw string literal expression parsing failed: {e}")


def test_byte_literal_expression_parsing():
    code = r"""
    fn test_byte_literals(c: u8) {
        let bytes = b"abc";
        let byte = b'a';
        let newline = b'\n';
        let raw = br#"a "quoted" byte raw"#;
        match c {
            b'a' => hit_a(),
            b'\n' => hit_newline(),
            _ => hit_default(),
        }
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        assert body[0].value == 'b"abc"'
        assert body[1].value == "b'a'"
        assert body[2].value == r"b'\n'"
        assert body[3].value == 'br#"a "quoted" byte raw"#'

        match_stmt = body[4]
        assert isinstance(match_stmt, MatchNode)
        assert match_stmt.arms[0].pattern == "b'a'"
        assert match_stmt.arms[1].pattern == r"b'\n'"
    except Exception as e:
        pytest.fail(f"Byte literal expression parsing failed: {e}")


def test_c_string_literal_expression_parsing_from_rust_gpu_ash_runner():
    # examples/runners/ash/src/device.rs uses Rust C string literals for
    # Vulkan layer names inside a const block.
    code = r"""
    fn validation_layers() {
        let layer_names: &'static [_] = const {
            &[c"VK_LAYER_KHRONOS_validation".as_ptr()]
        };
        let raw_layer = cr#"VK_LAYER_KHRONOS_validation"#;
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        array_ref = body[0].value.block.returns_value
        array = array_ref.expression
        c_string_call = array.elements[0]
        assert c_string_call.name.object == 'c"VK_LAYER_KHRONOS_validation"'
        assert c_string_call.name.member == "as_ptr"
        assert body[1].value == 'cr#"VK_LAYER_KHRONOS_validation"#'
    except Exception as e:
        pytest.fail(f"C string literal expression parsing failed: {e}")


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


def test_option_result_constructor_expression_parsing():
    code = """
    use std::option::Option::{Some, None};

    fn test_option_result(value: i32, err: i32) -> Result<i32, i32> {
        let maybe: Option<i32> = Some(value);
        let empty: Option<i32> = None;
        let ok_value = Ok(value + 1);
        let err_value = Err(err);
        match maybe {
            Some(v) => v,
            None => 0,
        }
        return Ok(value);
    }
    """
    try:
        ast = parse_code(code)
        assert ast.use_statements[0].items == [
            {"path": "Some", "alias": None},
            {"path": "None", "alias": None},
        ]

        body = ast.functions[0].body
        assert body[0].vtype == "Option<i32>"
        assert isinstance(body[0].value, FunctionCallNode)
        assert body[0].value.name == "Some"
        assert body[0].value.args == ["value"]

        assert body[1].vtype == "Option<i32>"
        assert body[1].value == "None"

        assert isinstance(body[2].value, FunctionCallNode)
        assert body[2].value.name == "Ok"
        assert isinstance(body[2].value.args[0], BinaryOpNode)

        assert isinstance(body[3].value, FunctionCallNode)
        assert body[3].value.name == "Err"
        assert body[3].value.args == ["err"]

        match_stmt = body[4]
        assert isinstance(match_stmt, MatchNode)
        assert isinstance(match_stmt.arms[0].pattern, FunctionCallNode)
        assert match_stmt.arms[0].pattern.name == "Some"
        assert match_stmt.arms[0].pattern.args == ["v"]
        assert match_stmt.arms[1].pattern == "None"

        assert isinstance(body[5], ReturnNode)
        assert isinstance(body[5].value, FunctionCallNode)
        assert body[5].value.name == "Ok"
    except Exception as e:
        pytest.fail(f"Option/Result constructor expression parsing failed: {e}")


def test_tuple_constructor_match_pattern_parsing():
    code = """
    fn test_constructor_patterns(pair: Pair, color: Color) {
        match pair {
            Pair(a, b) => use_pair(a, b),
            Pair(x, 3) => use_literal(x),
            _ => use_default(),
        }
        match color {
            Color::Rgb(r, g, b) => use_rgb(r, g, b),
            Color::Red => use_red(),
            _ => use_default(),
        }
    }
    """
    try:
        ast = parse_code(code)
        pair_match = ast.functions[0].body[0]
        color_match = ast.functions[0].body[1]

        assert isinstance(pair_match, MatchNode)
        assert isinstance(pair_match.arms[0].pattern, FunctionCallNode)
        assert pair_match.arms[0].pattern.name == "Pair"
        assert pair_match.arms[0].pattern.args == ["a", "b"]

        assert isinstance(pair_match.arms[1].pattern, FunctionCallNode)
        assert pair_match.arms[1].pattern.name == "Pair"
        assert pair_match.arms[1].pattern.args == ["x", "3"]
        assert pair_match.arms[2].pattern == "_"

        assert isinstance(color_match, MatchNode)
        assert isinstance(color_match.arms[0].pattern, FunctionCallNode)
        assert color_match.arms[0].pattern.name == "Color::Rgb"
        assert color_match.arms[0].pattern.args == ["r", "g", "b"]
        assert color_match.arms[1].pattern == "Color::Red"
        assert color_match.arms[2].pattern == "_"
    except Exception as e:
        pytest.fail(f"Tuple constructor match pattern parsing failed: {e}")


def test_match_reference_scrutinee_with_qualified_variant_patterns_from_rust_gpu():
    code = """
    impl Command {
        pub fn run(&self) -> i32 {
            match &self {
                Self::Install(install) => install.value,
                Self::Build => 1,
                Self::Show => 2,
            }
        }
    }
    """

    ast = parse_code(code)
    impl_function = ast.impl_blocks[0].functions[0]
    match_expr = impl_function.body[0]

    assert isinstance(match_expr, MatchNode)
    assert isinstance(match_expr.expression, ReferenceNode)
    assert match_expr.expression.expression == "self"

    install_pattern = match_expr.arms[0].pattern
    assert isinstance(install_pattern, FunctionCallNode)
    assert install_pattern.name == "Self::Install"
    assert install_pattern.args == ["install"]
    assert match_expr.arms[1].pattern == "Self::Build"
    assert match_expr.arms[2].pattern == "Self::Show"


def test_nested_constructor_match_pattern_parsing():
    code = """
    fn test_nested_constructor_patterns(value: Option<Result<i32, i32>>, pair: Pair) {
        match value {
            Some(Ok(v)) => use_ok(v),
            Some(Err(3)) => use_literal_err(),
            None => use_none(),
        }
        match pair {
            Pair(Some(x), y) => use_pair(x, y),
            Pair(None, 3) => use_none_pair(),
            _ => use_default(),
        }
    }
    """
    try:
        ast = parse_code(code)
        value_match = ast.functions[0].body[0]
        pair_match = ast.functions[0].body[1]

        assert isinstance(value_match, MatchNode)
        some_ok = value_match.arms[0].pattern
        assert isinstance(some_ok, FunctionCallNode)
        assert some_ok.name == "Some"
        assert len(some_ok.args) == 1
        assert isinstance(some_ok.args[0], FunctionCallNode)
        assert some_ok.args[0].name == "Ok"
        assert some_ok.args[0].args == ["v"]

        some_err = value_match.arms[1].pattern
        assert isinstance(some_err, FunctionCallNode)
        assert isinstance(some_err.args[0], FunctionCallNode)
        assert some_err.args[0].name == "Err"
        assert some_err.args[0].args == ["3"]
        assert value_match.arms[2].pattern == "None"

        pair_some = pair_match.arms[0].pattern
        assert isinstance(pair_some, FunctionCallNode)
        assert pair_some.name == "Pair"
        assert isinstance(pair_some.args[0], FunctionCallNode)
        assert pair_some.args[0].name == "Some"
        assert pair_some.args[0].args == ["x"]
        assert pair_some.args[1] == "y"

        pair_none = pair_match.arms[1].pattern
        assert isinstance(pair_none, FunctionCallNode)
        assert pair_none.args == ["None", "3"]
    except Exception as e:
        pytest.fail(f"Nested constructor match pattern parsing failed: {e}")


def test_guarded_nested_constructor_match_pattern_parsing():
    code = """
    fn test_guarded_nested_patterns(value: Option<Result<i32, i32>>) {
        match value {
            Some(Ok(v)) if v > 0 => use_positive(v),
            Some(Err(e)) if e != 0 => use_err(e),
            _ => use_default(),
        }
    }
    """
    try:
        ast = parse_code(code)
        match_stmt = ast.functions[0].body[0]

        assert isinstance(match_stmt, MatchNode)
        some_ok = match_stmt.arms[0].pattern
        assert isinstance(some_ok, FunctionCallNode)
        assert isinstance(some_ok.args[0], FunctionCallNode)
        assert some_ok.args[0].name == "Ok"
        assert some_ok.args[0].args == ["v"]
        assert isinstance(match_stmt.arms[0].guard, BinaryOpNode)
        assert match_stmt.arms[0].guard.op == ">"
        assert match_stmt.arms[0].guard.left == "v"
        assert match_stmt.arms[0].guard.right == "0"

        some_err = match_stmt.arms[1].pattern
        assert isinstance(some_err, FunctionCallNode)
        assert isinstance(some_err.args[0], FunctionCallNode)
        assert some_err.args[0].name == "Err"
        assert some_err.args[0].args == ["e"]
        assert isinstance(match_stmt.arms[1].guard, BinaryOpNode)
        assert match_stmt.arms[1].guard.op == "!="
        assert match_stmt.arms[1].guard.left == "e"
        assert match_stmt.arms[1].guard.right == "0"
    except Exception as e:
        pytest.fail(f"Guarded nested constructor match parsing failed: {e}")


def test_match_or_pattern_parsing():
    code = """
    fn test_or_patterns(mode: i32, color: Color, value: Option<Result<i32, i32>>) {
        match mode {
            0 | 1 => hit_small(),
            _ => hit_default(),
        }
        match color {
            Color::Red | Color::Green => hit_warm(),
            _ => hit_default(),
        }
        match value {
            Some(Ok(v)) | Some(Err(v)) if v > 0 => use_value(v),
            _ => use_default(),
        }
        match value {
            Some(Ok(v) | Err(v)) if v > 0 => use_nested(v),
            _ => use_default(),
        }
    }
    """
    try:
        ast = parse_code(code)
        mode_match = ast.functions[0].body[0]
        color_match = ast.functions[0].body[1]
        value_match = ast.functions[0].body[2]
        nested_value_match = ast.functions[0].body[3]

        assert isinstance(mode_match.arms[0].pattern, MatchOrPatternNode)
        assert mode_match.arms[0].pattern.patterns == ["0", "1"]

        assert isinstance(color_match.arms[0].pattern, MatchOrPatternNode)
        assert color_match.arms[0].pattern.patterns == [
            "Color::Red",
            "Color::Green",
        ]

        value_pattern = value_match.arms[0].pattern
        assert isinstance(value_pattern, MatchOrPatternNode)
        assert len(value_pattern.patterns) == 2
        assert all(
            isinstance(pattern, FunctionCallNode) for pattern in value_pattern.patterns
        )
        assert value_pattern.patterns[0].name == "Some"
        assert value_pattern.patterns[0].args[0].name == "Ok"
        assert value_pattern.patterns[1].name == "Some"
        assert value_pattern.patterns[1].args[0].name == "Err"
        assert isinstance(value_match.arms[0].guard, BinaryOpNode)
        assert value_match.arms[0].guard.op == ">"

        nested_value_pattern = nested_value_match.arms[0].pattern
        assert isinstance(nested_value_pattern, FunctionCallNode)
        assert nested_value_pattern.name == "Some"
        assert isinstance(nested_value_pattern.args[0], MatchOrPatternNode)
        assert nested_value_pattern.args[0].patterns[0].name == "Ok"
        assert nested_value_pattern.args[0].patterns[1].name == "Err"
    except Exception as e:
        pytest.fail(f"Match or-pattern parsing failed: {e}")


def test_match_range_pattern_parsing():
    code = """
    fn test_range_patterns(mode: i32, signed: i32, value: Option<i32>) {
        match mode {
            1..=10 => hit_inclusive(),
            11..20 => hit_exclusive(),
            0 | 30..=40 => hit_or_range(),
            _ => hit_default(),
        }
        match signed {
            -5..=-1 => hit_negative(),
            _ => hit_default(),
        }
        match value {
            Some(1..=3 | 7..10) => hit_nested(),
            _ => hit_default(),
        }
    }
    """
    try:
        ast = parse_code(code)
        mode_match = ast.functions[0].body[0]
        signed_match = ast.functions[0].body[1]
        value_match = ast.functions[0].body[2]

        inclusive = mode_match.arms[0].pattern
        assert isinstance(inclusive, RangeNode)
        assert inclusive.start == "1"
        assert inclusive.end == "10"
        assert inclusive.inclusive is True

        exclusive = mode_match.arms[1].pattern
        assert isinstance(exclusive, RangeNode)
        assert exclusive.start == "11"
        assert exclusive.end == "20"
        assert exclusive.inclusive is False

        or_range = mode_match.arms[2].pattern
        assert isinstance(or_range, MatchOrPatternNode)
        assert or_range.patterns[0] == "0"
        assert isinstance(or_range.patterns[1], RangeNode)
        assert or_range.patterns[1].start == "30"
        assert or_range.patterns[1].end == "40"

        negative = signed_match.arms[0].pattern
        assert isinstance(negative, RangeNode)
        assert isinstance(negative.start, UnaryOpNode)
        assert negative.start.op == "-"
        assert negative.start.operand == "5"
        assert isinstance(negative.end, UnaryOpNode)
        assert negative.end.op == "-"
        assert negative.end.operand == "1"

        nested_arg = value_match.arms[0].pattern.args[0]
        assert isinstance(nested_arg, MatchOrPatternNode)
        assert isinstance(nested_arg.patterns[0], RangeNode)
        assert nested_arg.patterns[0].inclusive is True
        assert isinstance(nested_arg.patterns[1], RangeNode)
        assert nested_arg.patterns[1].inclusive is False
    except Exception as e:
        pytest.fail(f"Match range pattern parsing failed: {e}")


def test_match_grouped_reference_pattern_parsing():
    code = """
    fn test_grouped_reference_patterns(mode: i32, value: &Option<Result<i32, i32>>, nested: Option<&i32>) {
        match mode {
            (0 | 1) => hit_grouped(),
            _ => hit_default(),
        }
        match mode {
            &(2..=4) => hit_ref_range(),
            &mut (5 | 6) => hit_mut_ref_or(),
            _ => hit_default(),
        }
        match value {
            &Some(Ok(v) | Err(v)) if v > 0 => use_value(v),
            _ => use_default(),
        }
        match nested {
            Some(&(1 | 2)) => hit_nested_ref(),
            _ => hit_default(),
        }
    }
    """
    try:
        ast = parse_code(code)
        grouped_match = ast.functions[0].body[0]
        ref_match = ast.functions[0].body[1]
        value_match = ast.functions[0].body[2]
        nested_match = ast.functions[0].body[3]

        grouped_pattern = grouped_match.arms[0].pattern
        assert isinstance(grouped_pattern, MatchOrPatternNode)
        assert grouped_pattern.patterns == ["0", "1"]

        ref_range = ref_match.arms[0].pattern
        assert isinstance(ref_range, ReferenceNode)
        assert isinstance(ref_range.expression, RangeNode)
        assert ref_range.expression.start == "2"
        assert ref_range.expression.end == "4"

        mut_ref_or = ref_match.arms[1].pattern
        assert isinstance(mut_ref_or, ReferenceNode)
        assert mut_ref_or.is_mutable is True
        assert isinstance(mut_ref_or.expression, MatchOrPatternNode)
        assert mut_ref_or.expression.patterns == ["5", "6"]

        value_pattern = value_match.arms[0].pattern
        assert isinstance(value_pattern, ReferenceNode)
        assert value_pattern.expression.name == "Some"
        inner_or = value_pattern.expression.args[0]
        assert isinstance(inner_or, MatchOrPatternNode)
        assert inner_or.patterns[0].name == "Ok"
        assert inner_or.patterns[1].name == "Err"

        nested_arg = nested_match.arms[0].pattern.args[0]
        assert isinstance(nested_arg, ReferenceNode)
        assert isinstance(nested_arg.expression, MatchOrPatternNode)
        assert nested_arg.expression.patterns == ["1", "2"]
    except Exception as e:
        pytest.fail(f"Grouped/reference match pattern parsing failed: {e}")


def test_match_binding_pattern_parsing():
    code = """
    fn test_binding_patterns(mode: i32, value: Option<Result<i32, i32>>, nested: Option<i32>) {
        match mode {
            whole @ 1..=10 if whole > 3 => use_mode(whole),
            label @ (0 | 20) => use_label(label),
            _ @ _ => use_discard(),
        }
        match value {
            outer @ Some(Ok(v) | Err(v)) if outer_ready(outer) && v > 0 => use_result(outer, v),
            _ => use_default(),
        }
        match nested {
            Some(inner @ (1..=3 | 7..10)) => use_inner(inner),
            _ => use_default(),
        }
    }
    """
    try:
        ast = parse_code(code)
        mode_match = ast.functions[0].body[0]
        value_match = ast.functions[0].body[1]
        nested_match = ast.functions[0].body[2]

        whole = mode_match.arms[0].pattern
        assert isinstance(whole, MatchBindingPatternNode)
        assert whole.name == "whole"
        assert isinstance(whole.pattern, RangeNode)
        assert whole.pattern.inclusive is True
        assert isinstance(mode_match.arms[0].guard, BinaryOpNode)

        label = mode_match.arms[1].pattern
        assert isinstance(label, MatchBindingPatternNode)
        assert label.name == "label"
        assert isinstance(label.pattern, MatchOrPatternNode)
        assert label.pattern.patterns == ["0", "20"]

        discard = mode_match.arms[2].pattern
        assert isinstance(discard, MatchBindingPatternNode)
        assert discard.name == "_"
        assert discard.pattern == "_"

        outer = value_match.arms[0].pattern
        assert isinstance(outer, MatchBindingPatternNode)
        assert outer.name == "outer"
        assert isinstance(outer.pattern, FunctionCallNode)
        assert outer.pattern.name == "Some"
        assert isinstance(outer.pattern.args[0], MatchOrPatternNode)
        assert outer.pattern.args[0].patterns[0].name == "Ok"
        assert outer.pattern.args[0].patterns[1].name == "Err"

        inner = nested_match.arms[0].pattern.args[0]
        assert isinstance(inner, MatchBindingPatternNode)
        assert inner.name == "inner"
        assert isinstance(inner.pattern, MatchOrPatternNode)
        assert isinstance(inner.pattern.patterns[0], RangeNode)
        assert isinstance(inner.pattern.patterns[1], RangeNode)
    except Exception as e:
        pytest.fail(f"Match binding pattern parsing failed: {e}")


def test_match_tuple_pattern_parsing():
    code = """
    fn test_tuple_patterns(mode: i32, state: i32, value: Option<i32>) {
        match (mode, state) {
            (0, y) if y > 1 => use_y(y),
            (1..=3, 4 | 5) => hit_range_or(),
            (left @ (6 | 7), right @ 8..10) => use_pair(left, right),
            _ => use_default(),
        }
        match (value, mode) {
            (Some(v), 1) => use_some(v),
            (None, _) => use_none(),
            _ => use_default(),
        }
        match (mode, (state, value)) {
            (9, (kept @ 10..=12, Some(v))) => use_nested(kept, v),
            _ => use_default(),
        }
    }
    """
    try:
        ast = parse_code(code)
        tuple_match = ast.functions[0].body[0]
        option_match = ast.functions[0].body[1]
        nested_match = ast.functions[0].body[2]

        assert isinstance(tuple_match.expression, TupleNode)
        assert tuple_match.expression.elements == ["mode", "state"]

        first_pattern = tuple_match.arms[0].pattern
        assert isinstance(first_pattern, TupleNode)
        assert first_pattern.elements == ["0", "y"]
        assert isinstance(tuple_match.arms[0].guard, BinaryOpNode)

        range_or = tuple_match.arms[1].pattern
        assert isinstance(range_or, TupleNode)
        assert isinstance(range_or.elements[0], RangeNode)
        assert isinstance(range_or.elements[1], MatchOrPatternNode)
        assert range_or.elements[1].patterns == ["4", "5"]

        bindings = tuple_match.arms[2].pattern
        assert isinstance(bindings.elements[0], MatchBindingPatternNode)
        assert bindings.elements[0].name == "left"
        assert isinstance(bindings.elements[1], MatchBindingPatternNode)
        assert bindings.elements[1].name == "right"

        option_pattern = option_match.arms[0].pattern
        assert isinstance(option_pattern, TupleNode)
        assert option_pattern.elements[0].name == "Some"
        assert option_pattern.elements[0].args == ["v"]

        nested_pattern = nested_match.arms[0].pattern
        assert isinstance(nested_pattern, TupleNode)
        assert isinstance(nested_pattern.elements[1], TupleNode)
        nested_inner = nested_pattern.elements[1]
        assert isinstance(nested_inner.elements[0], MatchBindingPatternNode)
        assert nested_inner.elements[1].name == "Some"
    except Exception as e:
        pytest.fail(f"Match tuple pattern parsing failed: {e}")


def test_match_array_pattern_parsing():
    code = """
    fn test_array_patterns(values: [i32; 4], option_values: [Option<i32>; 2], mode: i32, state: i32) {
        match values {
            [first, 2, third @ 3..=5, _] if first > 0 => use_fixed(first, third),
            [0 | 1, ..] => hit_prefix(),
            [head, middle @ .., tail] => use_rest(head, middle, tail),
            [] => use_empty(),
            _ => use_default(),
        }
        match option_values {
            [Some(v), _] => use_some(v),
            _ => use_default(),
        }
        match [mode, state] {
            [0, kept @ 1..=3] => use_literal(kept),
            _ => use_default(),
        }
    }
    """
    try:
        ast = parse_code(code)
        values_match = ast.functions[0].body[0]
        option_match = ast.functions[0].body[1]
        literal_match = ast.functions[0].body[2]

        fixed = values_match.arms[0].pattern
        assert isinstance(fixed, ArrayNode)
        assert fixed.elements[0] == "first"
        assert fixed.elements[1] == "2"
        assert isinstance(fixed.elements[2], MatchBindingPatternNode)
        assert fixed.elements[2].name == "third"
        assert isinstance(fixed.elements[2].pattern, RangeNode)
        assert fixed.elements[3] == "_"

        prefix = values_match.arms[1].pattern
        assert isinstance(prefix.elements[0], MatchOrPatternNode)
        assert isinstance(prefix.elements[1], MatchRestPatternNode)

        rest = values_match.arms[2].pattern
        assert rest.elements[0] == "head"
        assert isinstance(rest.elements[1], MatchBindingPatternNode)
        assert rest.elements[1].name == "middle"
        assert isinstance(rest.elements[1].pattern, MatchRestPatternNode)
        assert rest.elements[2] == "tail"

        empty = values_match.arms[3].pattern
        assert isinstance(empty, ArrayNode)
        assert empty.elements == []

        option_pattern = option_match.arms[0].pattern
        assert isinstance(option_pattern, ArrayNode)
        assert option_pattern.elements[0].name == "Some"
        assert option_pattern.elements[0].args == ["v"]

        assert isinstance(literal_match.expression, ArrayNode)
        literal_pattern = literal_match.arms[0].pattern
        assert isinstance(literal_pattern, ArrayNode)
        assert literal_pattern.elements[0] == "0"
        assert isinstance(literal_pattern.elements[1], MatchBindingPatternNode)
    except Exception as e:
        pytest.fail(f"Match array pattern parsing failed: {e}")


def test_match_struct_pattern_parsing():
    code = """
    fn test_struct_patterns(point: Point, shape: Shape) {
        match point {
            Point { x, y: 0, .. } if x > 0 => use_x(x),
            Point { x: left @ 1..=3, y } => use_pair(left, y),
            _ => use_default(),
        }
        match shape {
            Shape::Circle { radius } if radius > 0 => use_radius(radius),
            Shape::Rect { corner: Point { x, y }, size: 1..=10, .. } => use_rect(x, y),
            Shape::Circle { radius: small @ (1 | 2), .. }
            | Shape::Circle { radius: small @ 3, .. } => use_small(small),
            _ => use_default(),
        }
    }
    """
    try:
        ast = parse_code(code)
        point_match = ast.functions[0].body[0]
        shape_match = ast.functions[0].body[1]

        first_point = point_match.arms[0].pattern
        assert isinstance(first_point, MatchStructPatternNode)
        assert first_point.name == "Point"
        assert first_point.has_rest is True
        assert first_point.fields[0] == ("x", "x")
        assert first_point.fields[1] == ("y", "0")
        assert isinstance(point_match.arms[0].guard, BinaryOpNode)

        second_point = point_match.arms[1].pattern
        assert isinstance(second_point.fields[0][1], MatchBindingPatternNode)
        assert second_point.fields[0][1].name == "left"
        assert isinstance(second_point.fields[0][1].pattern, RangeNode)
        assert second_point.fields[1] == ("y", "y")

        circle = shape_match.arms[0].pattern
        assert isinstance(circle, MatchStructPatternNode)
        assert circle.name == "Shape::Circle"
        assert circle.fields == [("radius", "radius")]

        rect = shape_match.arms[1].pattern
        assert isinstance(rect, MatchStructPatternNode)
        assert rect.name == "Shape::Rect"
        assert rect.has_rest is True
        corner = rect.fields[0][1]
        assert isinstance(corner, MatchStructPatternNode)
        assert corner.name == "Point"
        assert corner.fields == [("x", "x"), ("y", "y")]
        assert isinstance(rect.fields[1][1], RangeNode)

        small = shape_match.arms[2].pattern
        assert isinstance(small, MatchOrPatternNode)
        assert all(
            isinstance(pattern, MatchStructPatternNode) for pattern in small.patterns
        )
        assert small.patterns[0].fields[0][0] == "radius"
        assert isinstance(small.patterns[0].fields[0][1], MatchBindingPatternNode)
    except Exception as e:
        pytest.fail(f"Match struct pattern parsing failed: {e}")


def test_match_binding_modifier_pattern_parsing():
    code = """
    fn test_binding_modifier_patterns(value: Option<i32>, point: Point, pair: Pair, values: [i32; 3]) {
        match value {
            Some(ref v) if v > 0 => use_ref(v),
            Some(mut bounded @ 1..=3) => use_mut(bounded),
            Some(ref mut choice @ (4 | 5)) => use_choice(choice),
            _ => use_default(),
        }
        match point {
            Point { ref x, mut y, z: ref renamed, w: mut kept @ 1..=2, .. } => use_point(x, y, renamed, kept),
            _ => use_default(),
        }
        match pair {
            Pair(ref left, mut right) => use_pair(left, right),
        }
        match values {
            [ref first, mut second @ 2..=3, ..] => use_array(first, second),
            _ => use_default(),
        }
    }
    """
    try:
        ast = parse_code(code)
        value_match = ast.functions[0].body[0]
        point_match = ast.functions[0].body[1]
        pair_match = ast.functions[0].body[2]
        values_match = ast.functions[0].body[3]

        ref_value = value_match.arms[0].pattern.args[0]
        assert ref_value == "v"

        bounded = value_match.arms[1].pattern.args[0]
        assert isinstance(bounded, MatchBindingPatternNode)
        assert bounded.name == "bounded"
        assert isinstance(bounded.pattern, RangeNode)

        choice = value_match.arms[2].pattern.args[0]
        assert isinstance(choice, MatchBindingPatternNode)
        assert choice.name == "choice"
        assert isinstance(choice.pattern, MatchOrPatternNode)

        point_pattern = point_match.arms[0].pattern
        assert isinstance(point_pattern, MatchStructPatternNode)
        assert point_pattern.fields[0] == ("x", "x")
        assert point_pattern.fields[1] == ("y", "y")
        assert point_pattern.fields[2] == ("z", "renamed")
        assert isinstance(point_pattern.fields[3][1], MatchBindingPatternNode)
        assert point_pattern.fields[3][1].name == "kept"

        pair_pattern = pair_match.arms[0].pattern
        assert pair_pattern.args == ["left", "right"]

        array_pattern = values_match.arms[0].pattern
        assert isinstance(array_pattern, ArrayNode)
        assert array_pattern.elements[0] == "first"
        assert isinstance(array_pattern.elements[1], MatchBindingPatternNode)
        assert array_pattern.elements[1].name == "second"
    except Exception as e:
        pytest.fail(f"Match binding modifier pattern parsing failed: {e}")


def test_if_while_let_pattern_condition_parsing():
    code = """
    fn test_let_conditions(value: Option<i32>, point: Point) {
        if let Some(Ok(v) | Err(v)) = value {
            use_value(v);
        } else if let Some(_) = value {
            use_other();
        } else {
            use_default();
        }

        if let Point { ref x, y: 0, .. } = point {
            use_point(x);
        }

        while let Some(item) = next_value() {
            consume(item);
        }
    }
    """
    try:
        ast = parse_code(code)
        function = ast.functions[0]

        first_if = function.body[0]
        assert isinstance(first_if.condition, LetPatternConditionNode)
        assert first_if.condition.expression == "value"
        assert first_if.condition.pattern.name == "Some"
        assert isinstance(first_if.condition.pattern.args[0], MatchOrPatternNode)

        else_if = first_if.else_body[0]
        assert isinstance(else_if, IfNode)
        assert isinstance(else_if.condition, LetPatternConditionNode)
        assert else_if.condition.pattern.name == "Some"
        assert else_if.condition.pattern.args == ["_"]

        point_if = function.body[1]
        assert isinstance(point_if.condition, LetPatternConditionNode)
        assert isinstance(point_if.condition.pattern, MatchStructPatternNode)
        assert point_if.condition.pattern.fields[0] == ("x", "x")
        assert point_if.condition.pattern.fields[1] == ("y", "0")
        assert point_if.condition.pattern.has_rest is True

        while_loop = function.body[2]
        assert isinstance(while_loop, WhileNode)
        assert isinstance(while_loop.condition, LetPatternConditionNode)
        assert while_loop.condition.pattern.name == "Some"
        assert while_loop.condition.pattern.args == ["item"]
    except Exception as e:
        pytest.fail(f"If/while let pattern condition parsing failed: {e}")


def test_let_else_pattern_parsing():
    code = """
    fn test_let_else(value: Option<i32>, pair: Pair, point: Point) -> i32 {
        let Some(v) = value else {
            return 0;
        };
        let Pair(left, bounded @ 1..=3) = pair else {
            return v;
        };
        let Point { ref x, y: 0, .. } = point else {
            return bounded;
        };
        return x;
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        option_let = body[0]
        assert isinstance(option_let, LetNode)
        assert option_let.name.name == "Some"
        assert option_let.name.args == ["v"]
        assert len(option_let.else_body) == 1
        assert isinstance(option_let.else_body[0], ReturnNode)

        pair_let = body[1]
        assert isinstance(pair_let, LetNode)
        assert pair_let.name.name == "Pair"
        assert pair_let.name.args[0] == "left"
        assert isinstance(pair_let.name.args[1], MatchBindingPatternNode)
        assert pair_let.name.args[1].name == "bounded"
        assert isinstance(pair_let.name.args[1].pattern, RangeNode)

        struct_let = body[2]
        assert isinstance(struct_let, LetNode)
        assert isinstance(struct_let.name, MatchStructPatternNode)
        assert struct_let.name.fields[0] == ("x", "x")
        assert struct_let.name.fields[1] == ("y", "0")
        assert struct_let.name.has_rest is True
        assert len(struct_let.else_body) == 1
    except Exception as e:
        pytest.fail(f"Let else pattern parsing failed: {e}")


def test_wgpu_let_else_semicolonless_return_parsing():
    # Reduced from gfx-rs/wgpu commit 6877690dbefa1144c05932dde2d10c52facfee60,
    # examples/bug-repro/01_texture_atomic_bug/src/main.rs App::window_event.
    code = """
    impl App {
        fn window_event(&mut self) {
            let Some(state) = &mut self.state else { return };
            state.render_frame();
        }
    }
    """

    ast = parse_code(code)
    method = ast.impl_blocks[0].methods[0]
    let_else = method.body[0]

    assert isinstance(let_else, LetNode)
    assert let_else.name.name == "Some"
    assert let_else.name.args == ["state"]
    assert isinstance(let_else.value, ReferenceNode)
    assert let_else.value.is_mutable is True
    assert isinstance(let_else.value.expression, MemberAccessNode)
    assert let_else.value.expression.object == "self"
    assert let_else.value.expression.member == "state"
    assert len(let_else.else_body) == 1
    assert isinstance(let_else.else_body[0], ReturnNode)
    assert let_else.else_body[0].value is None


def test_matches_macro_pattern_parsing():
    code = """
    fn test_matches(value: Option<Result<i32, i32>>, mode: i32, point: Point) {
        let ready = matches!(value, Some(Ok(v) | Err(v)) if v > 0);
        let simple = matches!(mode, 0 | 1,);
        if matches!(point, Point { x, y: 0, .. }) {
            use_point();
        }
        while matches!(next_value(), Some(_)) {
            spin();
        }
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        ready = body[0].value
        assert isinstance(ready, MatchesMacroNode)
        assert ready.expression == "value"
        assert ready.pattern.name == "Some"
        assert isinstance(ready.pattern.args[0], MatchOrPatternNode)
        assert isinstance(ready.guard, BinaryOpNode)

        simple = body[1].value
        assert isinstance(simple, MatchesMacroNode)
        assert simple.expression == "mode"
        assert isinstance(simple.pattern, MatchOrPatternNode)
        assert simple.guard is None

        point_if = body[2]
        assert isinstance(point_if.condition, MatchesMacroNode)
        assert isinstance(point_if.condition.pattern, MatchStructPatternNode)
        assert point_if.condition.pattern.fields[1] == ("y", "0")

        loop = body[3]
        assert isinstance(loop, WhileNode)
        assert isinstance(loop.condition, MatchesMacroNode)
        assert loop.condition.pattern.name == "Some"
        assert loop.condition.pattern.args == ["_"]
    except Exception as e:
        pytest.fail(f"Matches macro pattern parsing failed: {e}")


def test_let_condition_chain_parsing():
    code = """
    fn test_condition_chains(value: Option<i32>, other: Option<i32>) {
        if ready && let Some(v) = value && v > 0 {
            use_value(v);
        } else {
            use_default();
        }

        if let Some(a) = value && let Some(b) = other && (a + b) > 0 {
            use_pair(a, b);
        }

        while has_next() && let Some(item) = next_value() && item != 0 {
            consume(item);
        }

        let result = if ready && let Some(v) = value && v > 0 {
            v
        } else {
            0
        };
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        first_if = body[0]
        assert isinstance(first_if.condition, ConditionChainNode)
        assert first_if.condition.operands[0] == "ready"
        assert isinstance(first_if.condition.operands[1], LetPatternConditionNode)
        assert first_if.condition.operands[1].pattern.name == "Some"
        assert isinstance(first_if.condition.operands[2], BinaryOpNode)

        second_if = body[1]
        assert isinstance(second_if.condition, ConditionChainNode)
        assert isinstance(second_if.condition.operands[0], LetPatternConditionNode)
        assert isinstance(second_if.condition.operands[1], LetPatternConditionNode)
        assert isinstance(second_if.condition.operands[2], BinaryOpNode)

        while_loop = body[2]
        assert isinstance(while_loop, WhileNode)
        assert isinstance(while_loop.condition, ConditionChainNode)
        assert isinstance(while_loop.condition.operands[0], FunctionCallNode)
        assert isinstance(while_loop.condition.operands[1], LetPatternConditionNode)

        result_if = body[3].value
        assert isinstance(result_if, IfNode)
        assert isinstance(result_if.condition, ConditionChainNode)
        assert len(result_if.condition.operands) == 3
    except Exception as e:
        pytest.fail(f"Let condition chain parsing failed: {e}")


def test_absolute_path_if_let_pattern_parses_from_rust_cuda_codegen_backend():
    code = """
    fn synthesize_features(args: Args) {
        let mut features = vec![];
        for opt in &args.nvvm_options {
            if let ::nvvm::NvvmOption::Arch(arch) = opt {
                features.extend(arch.all_target_features());
            }
        }
    }
    """
    ast = parse_code(code)
    loop = ast.functions[0].body[1]

    assert isinstance(loop, ForNode)
    condition = loop.body[0].condition
    assert isinstance(condition, LetPatternConditionNode)
    assert isinstance(condition.pattern, FunctionCallNode)
    assert condition.pattern.name == "nvvm::NvvmOption::Arch"
    assert condition.pattern.args == ["arch"]
    assert condition.expression == "opt"


def test_closure_expression_parsing():
    code = """
    fn test_closures(values: Values) {
        let add = |x, y| x + y;
        let typed = |x: i32| x + 1;
        let always = || true;
        let moved = move |v| v * 2;
        let mapped = values.map(|x| x + 1).filter(|x| x > 0);
        let block = |x| {
            let y = x + 1;
            y * 2
        };
        let tupled = |(x, y)| x + y;
        let optioned = |Some(v)| v;
        let pointed = |Point { x, y }| x + y;
        let ignored = |_| true;
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        add = body[0].value
        assert isinstance(add, ClosureNode)
        assert [param.pattern for param in add.params] == ["x", "y"]
        assert isinstance(add.body, BinaryOpNode)

        typed = body[1].value
        assert isinstance(typed, ClosureNode)
        assert isinstance(typed.params[0], ClosureParameterNode)
        assert typed.params[0].pattern == "x"
        assert typed.params[0].param_type == "i32"

        always = body[2].value
        assert isinstance(always, ClosureNode)
        assert always.params == []
        assert always.body == "true"

        moved = body[3].value
        assert isinstance(moved, ClosureNode)
        assert moved.is_move is True
        assert moved.params[0].pattern == "v"

        filtered = body[4].value
        assert isinstance(filtered, FunctionCallNode)
        assert filtered.name.member == "filter"
        assert isinstance(filtered.args[0], ClosureNode)

        mapped = filtered.name.object
        assert isinstance(mapped, FunctionCallNode)
        assert mapped.name.member == "map"
        assert isinstance(mapped.args[0], ClosureNode)

        block = body[5].value
        assert isinstance(block, ClosureNode)
        assert isinstance(block.body, BlockNode)
        assert len(block.body.statements) == 1
        assert isinstance(block.body.statements[0], LetNode)
        assert isinstance(block.body.expression, BinaryOpNode)

        tupled = body[6].value
        assert isinstance(tupled, ClosureNode)
        assert isinstance(tupled.params[0].pattern, TupleNode)
        assert tupled.params[0].pattern.elements == ["x", "y"]

        optioned = body[7].value
        assert isinstance(optioned, ClosureNode)
        assert isinstance(optioned.params[0].pattern, FunctionCallNode)
        assert optioned.params[0].pattern.name == "Some"
        assert optioned.params[0].pattern.args == ["v"]

        pointed = body[8].value
        assert isinstance(pointed, ClosureNode)
        assert isinstance(pointed.params[0].pattern, MatchStructPatternNode)
        assert pointed.params[0].pattern.name == "Point"
        assert pointed.params[0].pattern.fields == [("x", "x"), ("y", "y")]

        ignored = body[9].value
        assert isinstance(ignored, ClosureNode)
        assert ignored.params[0].pattern == "_"
    except Exception as e:
        pytest.fail(f"Closure expression parsing failed: {e}")


def test_closure_return_type_parsing():
    code = """
    fn test_closure_return_types() {
        let parsed = |value: Result<i32, i32>| -> Result<i32, i32> {
            let v: i32 = value?;
            Ok(v)
        };
        let optional = || -> Option<i32> {
            Some(1)
        };
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        parsed = body[0].value
        assert isinstance(parsed, ClosureNode)
        assert parsed.return_type == "Result<i32, i32>"
        assert isinstance(parsed.body, BlockNode)
        assert isinstance(parsed.body.statements[0], LetNode)
        assert isinstance(parsed.body.statements[0].value, TryNode)
        assert isinstance(parsed.body.expression, FunctionCallNode)
        assert parsed.body.expression.name == "Ok"

        optional = body[1].value
        assert isinstance(optional, ClosureNode)
        assert optional.return_type == "Option<i32>"
        assert isinstance(optional.body, BlockNode)
        assert isinstance(optional.body.expression, FunctionCallNode)
        assert optional.body.expression.name == "Some"
    except Exception as e:
        pytest.fail(f"Closure return type parsing failed: {e}")


def test_async_closure_expression_parsing():
    code = """
    fn test_async_closures(values: Values) {
        let async_add = async |x| x + 1;
        let async_moved = async move || true;
        let async_typed = async move |value: Result<i32, i32>| -> Result<i32, i32> {
            Ok(value?)
        };
        let mapped = values.map(async |x| x + 1);
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        async_add = body[0].value
        assert isinstance(async_add, ClosureNode)
        assert async_add.is_async is True
        assert async_add.is_move is False
        assert async_add.params[0].pattern == "x"
        assert isinstance(async_add.body, BinaryOpNode)

        async_moved = body[1].value
        assert isinstance(async_moved, ClosureNode)
        assert async_moved.is_async is True
        assert async_moved.is_move is True
        assert async_moved.params == []
        assert async_moved.body == "true"

        async_typed = body[2].value
        assert isinstance(async_typed, ClosureNode)
        assert async_typed.is_async is True
        assert async_typed.is_move is True
        assert async_typed.params[0].param_type == "Result<i32, i32>"
        assert async_typed.return_type == "Result<i32, i32>"
        assert isinstance(async_typed.body, BlockNode)
        assert isinstance(async_typed.body.expression, FunctionCallNode)
        assert isinstance(async_typed.body.expression.args[0], TryNode)

        mapped = body[3].value
        assert isinstance(mapped, FunctionCallNode)
        assert mapped.name.member == "map"
        assert isinstance(mapped.args[0], ClosureNode)
        assert mapped.args[0].is_async is True
    except Exception as e:
        pytest.fail(f"Async closure expression parsing failed: {e}")


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


def test_if_expression_operand_parsing():
    code = """
    fn test_if_operand(a: bool, b: bool, c: bool, d: bool) {
        let x = a || if b { c } else { d };
    }
    """

    ast = parse_code(code)
    value = ast.functions[0].body[0].value

    assert isinstance(value, BinaryOpNode)
    assert value.op == "||"
    assert isinstance(value.right, TernaryOpNode)
    assert value.right.condition == "b"


def test_match_loop_and_block_expression_operand_parsing():
    code = """
    fn test_expression_operands(a: bool, b: bool, v: i32) {
        let matched = a || match v {
            0 => false,
            _ => true,
        };
        let looped = check(loop {
            break true;
        });
        let blocked = a || { b };
    }
    """

    ast = parse_code(code)
    body = ast.functions[0].body

    matched = body[0].value
    assert isinstance(matched, BinaryOpNode)
    assert matched.op == "||"
    assert isinstance(matched.right, MatchNode)
    assert matched.right.expression == "v"

    looped = body[1].value
    assert isinstance(looped, FunctionCallNode)
    assert isinstance(looped.args[0], LoopNode)

    blocked = body[2].value
    assert isinstance(blocked, BinaryOpNode)
    assert isinstance(blocked.right, BlockNode)
    assert blocked.right.expression == "b"


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
        body = ast.functions[0].body

        assert isinstance(body[0].value, UnaryOpNode)
        assert body[0].value.op == "-"
        assert isinstance(body[1].value, UnaryOpNode)
        assert body[1].value.op == "!"
        assert isinstance(body[2].value, ReferenceNode)
        assert body[2].value.expression == "x"
        assert body[2].value.is_mutable is False
        assert isinstance(body[3].value, DereferenceNode)
        assert body[3].value.expression == "ptr"
    except Exception as e:
        pytest.fail(f"Unary operations parsing failed: {e}")


def test_reference_dereference_expression_parsing():
    code = """
    fn test_reference_deref(ptr: f32, value: f32) -> f32 {
        let shared = &value;
        let writable = &mut value;
        output.value = *ptr;
        return *ptr;
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        assert isinstance(body[0].value, ReferenceNode)
        assert body[0].value.is_mutable is False
        assert body[0].value.expression == "value"

        assert isinstance(body[1].value, ReferenceNode)
        assert body[1].value.is_mutable is True
        assert body[1].value.expression == "value"

        assert isinstance(body[2], AssignmentNode)
        assert isinstance(body[2].right, DereferenceNode)
        assert body[2].right.expression == "ptr"

        assert isinstance(body[3], ReturnNode)
        assert isinstance(body[3].value, DereferenceNode)
        assert body[3].value.expression == "ptr"
    except Exception as e:
        pytest.fail(f"Reference/dereference expression parsing failed: {e}")


def test_array_access_parsing():
    code = """
    fn test_array() {
        let arr = [1, 2, 3, 4, 5];
        let start = 1;
        let end = 3;
        let element = arr[0];
        let slice = &arr[1..3];
        let prefix = &arr[..end];
        let suffix = &arr[start..];
        let full = &arr[..];
    }
    """
    try:
        ast = parse_code(code)
        assert len(ast.functions) == 1
        func = ast.functions[0]
        assert len(func.body) >= 8

        closed_range = func.body[4].value.expression.index
        prefix_range = func.body[5].value.expression.index
        suffix_range = func.body[6].value.expression.index
        full_range = func.body[7].value.expression.index

        assert isinstance(func.body[3].value, ArrayAccessNode)
        assert isinstance(closed_range, RangeNode)
        assert closed_range.start == "1"
        assert closed_range.end == "3"
        assert isinstance(prefix_range, RangeNode)
        assert prefix_range.start is None
        assert prefix_range.end == "end"
        assert isinstance(suffix_range, RangeNode)
        assert suffix_range.start == "start"
        assert suffix_range.end is None
        assert isinstance(full_range, RangeNode)
        assert full_range.start is None
        assert full_range.end is None
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


def test_tuple_field_access_parsing():
    code = """
    fn test_tuple_field(a: i32, b: i32) -> i32 {
        let pair = (a, b);
        let first = pair.0;
        let second = (a, b).1;
        first + second
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body
        first = body[1].value
        second = body[2].value

        assert isinstance(first, MemberAccessNode)
        assert first.object == "pair"
        assert first.member == "0"
        assert isinstance(second, MemberAccessNode)
        assert isinstance(second.object, TupleNode)
        assert second.member == "1"
    except Exception as e:
        pytest.fail(f"Tuple field access parsing failed: {e}")


def test_type_casting_parsing():
    code = """
    fn test_cast() {
        let x = 42i32;
        let y = x as f64;
        let z = y as i32;
        let chained = x as f32 as i32;
        let from_deref = *ptr as f32;
        let from_swizzle = color.xyz() as Vec3<f32>;
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        assert isinstance(body[1].value, CastNode)
        assert body[1].value.target_type == "f64"
        assert isinstance(body[2].value, CastNode)
        assert body[2].value.target_type == "i32"

        chained = body[3].value
        assert isinstance(chained, CastNode)
        assert chained.target_type == "i32"
        assert isinstance(chained.expression, CastNode)
        assert chained.expression.target_type == "f32"

        from_deref = body[4].value
        assert isinstance(from_deref, CastNode)
        assert from_deref.target_type == "f32"
        assert isinstance(from_deref.expression, DereferenceNode)

        from_swizzle = body[5].value
        assert isinstance(from_swizzle, CastNode)
        assert from_swizzle.target_type == "Vec3<f32>"
        assert isinstance(from_swizzle.expression, FunctionCallNode)
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
        a &= 1;
        a |= 2;
        a ^= 3;
        a <<= 1;
        a >>= 2;
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body
        assignments = [stmt for stmt in body if isinstance(stmt, AssignmentNode)]

        assert [stmt.operator for stmt in assignments] == [
            "+=",
            "-=",
            "*=",
            "/=",
            "%=",
            "&=",
            "|=",
            "^=",
            "<<=",
            ">>=",
        ]
    except Exception as e:
        pytest.fail(f"Assignment operations parsing failed: {e}")


def test_assignment_expression_parsing():
    code = """
    fn test_assignment_expr() {
        let mut a = 0;
        (a = 1);
        let unit = (a = 2);
        let compound = (a += 3);
        let block_unit = { a = 4; };
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        assert isinstance(body[1], AssignmentNode)
        assert body[1].operator == "="
        assert body[1].right == "1"

        assert isinstance(body[2], LetNode)
        assert isinstance(body[2].value, AssignmentNode)
        assert body[2].value.operator == "="
        assert body[2].value.right == "2"

        assert isinstance(body[3], LetNode)
        assert isinstance(body[3].value, AssignmentNode)
        assert body[3].value.operator == "+="
        assert body[3].value.right == "3"

        assert isinstance(body[4], LetNode)
        assert isinstance(body[4].value, BlockNode)
        assert isinstance(body[4].value.statements[0], AssignmentNode)
        assert body[4].value.statements[0].operator == "="
        assert body[4].value.statements[0].right == "4"
    except Exception as e:
        pytest.fail(f"Assignment expression parsing failed: {e}")


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


def test_labeled_control_flow_parsing():
    code = """
    fn test_labeled_control() {
        'outer: loop {
            'inner: while ready {
                continue 'outer;
                break 'inner;
            }
            'scan: for i in 0..4 {
                break 'scan;
            }
        }
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        outer = body[0]
        assert isinstance(outer, LoopNode)
        assert outer.label == "'outer"

        inner = outer.body[0]
        assert isinstance(inner, WhileNode)
        assert inner.label == "'inner"
        assert isinstance(inner.body[0], ContinueNode)
        assert inner.body[0].label == "'outer"
        assert isinstance(inner.body[1], BreakNode)
        assert inner.body[1].label == "'inner"

        scan = outer.body[1]
        assert isinstance(scan, ForNode)
        assert scan.label == "'scan"
        assert isinstance(scan.body[0], BreakNode)
        assert scan.body[0].label == "'scan"
    except Exception as e:
        pytest.fail(f"Labeled control flow parsing failed: {e}")


def test_break_value_parsing():
    code = """
    fn test_break_values() {
        loop {
            break result;
        }
        'outer: loop {
            break 'outer value + 1;
        }
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        plain_break = body[0].body[0]
        assert isinstance(plain_break, BreakNode)
        assert plain_break.label is None
        assert plain_break.value == "result"

        labeled_break = body[1].body[0]
        assert isinstance(labeled_break, BreakNode)
        assert labeled_break.label == "'outer"
        assert isinstance(labeled_break.value, BinaryOpNode)
        assert labeled_break.value.op == "+"
        assert labeled_break.value.left == "value"
        assert labeled_break.value.right == "1"
    except Exception as e:
        pytest.fail(f"Break value parsing failed: {e}")


def test_break_result_expression_parsing():
    code = """
    fn test_break_result_expressions() {
        let from_if: i32 = loop {
            break if ready {
                let seed = 1;
                seed + 1
            } else {
                2
            };
        };
        let from_match: i32 = loop {
            break match mode {
                0 => 1,
                _ => {
                    let fallback = 2;
                    fallback
                },
            };
        };
        let from_block: i32 = loop {
            break {
                let base = 3;
                base + 1
            };
        };
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        if_break = body[0].value.body[0]
        assert isinstance(if_break, BreakNode)
        assert isinstance(if_break.value, IfNode)
        assert isinstance(if_break.value.if_body, BlockNode)
        assert len(if_break.value.if_body.statements) == 1
        assert isinstance(if_break.value.if_body.expression, BinaryOpNode)

        match_break = body[1].value.body[0]
        assert isinstance(match_break, BreakNode)
        assert isinstance(match_break.value, MatchNode)
        assert match_break.value.arms[0].body == "1"
        assert isinstance(match_break.value.arms[1].body, BlockNode)
        assert match_break.value.arms[1].body.expression == "fallback"

        block_break = body[2].value.body[0]
        assert isinstance(block_break, BreakNode)
        assert isinstance(block_break.value, BlockNode)
        assert len(block_break.value.statements) == 1
        assert isinstance(block_break.value.expression, BinaryOpNode)
    except Exception as e:
        pytest.fail(f"Break result expression parsing failed: {e}")


def test_loop_expression_let_parsing():
    code = """
    fn test_loop_expression() {
        let value: i32 = loop {
            if ready {
                break 7;
            }
        };
        let labeled: i32 = 'outer: loop {
            break 'outer value + 1;
        };
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        value_loop = body[0].value
        assert isinstance(value_loop, LoopNode)
        value_break = value_loop.body[0].if_body[0]
        assert isinstance(value_break, BreakNode)
        assert value_break.value == "7"

        labeled_loop = body[1].value
        assert isinstance(labeled_loop, LoopNode)
        assert labeled_loop.label == "'outer"
        labeled_break = labeled_loop.body[0]
        assert isinstance(labeled_break, BreakNode)
        assert labeled_break.label == "'outer"
        assert isinstance(labeled_break.value, BinaryOpNode)
    except Exception as e:
        pytest.fail(f"Loop expression let parsing failed: {e}")


def test_loop_expression_assignment_parsing():
    code = """
    fn test_loop_assignment() {
        value = loop {
            break result;
        };
        output.color = 'outer: loop {
            break 'outer value + 1;
        };
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        first_assignment = body[0]
        assert isinstance(first_assignment, AssignmentNode)
        assert isinstance(first_assignment.right, LoopNode)
        first_break = first_assignment.right.body[0]
        assert isinstance(first_break, BreakNode)
        assert first_break.value == "result"

        member_assignment = body[1]
        assert isinstance(member_assignment, AssignmentNode)
        assert isinstance(member_assignment.right, LoopNode)
        assert member_assignment.right.label == "'outer"
        labeled_break = member_assignment.right.body[0]
        assert isinstance(labeled_break, BreakNode)
        assert labeled_break.label == "'outer"
        assert isinstance(labeled_break.value, BinaryOpNode)
    except Exception as e:
        pytest.fail(f"Loop expression assignment parsing failed: {e}")


def test_loop_expression_return_parsing():
    code = """
    fn test_loop_return() -> i32 {
        return loop {
            break result;
        };
        return 'outer: loop {
            break 'outer value + 1;
        };
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        plain_return = body[0]
        assert isinstance(plain_return, ReturnNode)
        assert isinstance(plain_return.value, LoopNode)
        plain_break = plain_return.value.body[0]
        assert isinstance(plain_break, BreakNode)
        assert plain_break.value == "result"

        labeled_return = body[1]
        assert isinstance(labeled_return, ReturnNode)
        assert isinstance(labeled_return.value, LoopNode)
        assert labeled_return.value.label == "'outer"
        labeled_break = labeled_return.value.body[0]
        assert isinstance(labeled_break, BreakNode)
        assert labeled_break.label == "'outer"
        assert isinstance(labeled_break.value, BinaryOpNode)
    except Exception as e:
        pytest.fail(f"Loop expression return parsing failed: {e}")


def test_block_loop_expression_parsing():
    code = """
    fn test_block_loop_expression() -> i32 {
        let value: i32 = {
            let seed = 1;
            loop {
                break seed + 1;
            }
        };
        output.color = {
            'outer: loop {
                break 'outer value + 1;
            }
        };
        return {
            loop {
                break value;
            }
        };
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        let_block = body[0].value
        assert isinstance(let_block, BlockNode)
        assert len(let_block.statements) == 1
        assert isinstance(let_block.expression, LoopNode)
        assert isinstance(let_block.expression.body[0].value, BinaryOpNode)

        assignment_block = body[1].right
        assert isinstance(assignment_block, BlockNode)
        assert isinstance(assignment_block.expression, LoopNode)
        assert assignment_block.expression.label == "'outer"

        return_block = body[2].value
        assert isinstance(return_block, BlockNode)
        assert isinstance(return_block.expression, LoopNode)
        assert return_block.expression.body[0].value == "value"
    except Exception as e:
        pytest.fail(f"Block loop expression parsing failed: {e}")


def test_parsed_block_node_statements_are_lists():
    code = """
    fn parsed_blocks(flag: bool) -> i32 {
        let value = {
            let seed = 1;
            if flag {
                let selected = seed;
                selected
            } else {
                let selected = 0;
                selected
            }
        };
        return value;
    }
    """

    ast = parse_code(code)
    block = ast.functions[0].body[0].value

    assert isinstance(block, BlockNode)
    assert isinstance(block.statements, list)
    assert isinstance(block.expression, IfNode)
    assert isinstance(block.expression.if_body, BlockNode)
    assert isinstance(block.expression.if_body.statements, list)
    assert isinstance(block.expression.else_body, BlockNode)
    assert isinstance(block.expression.else_body.statements, list)


def test_if_expression_result_parsing():
    code = """
    fn test_if_expression() -> i32 {
        let value: i32 = if ready {
            let seed = 1;
            seed + 1
        } else {
            2
        };
        output.color = if value > 1 {
            value + 1
        } else {
            let fallback = 0;
            fallback
        };
        return if done {
            value
        } else {
            let next = value + 1;
            next
        };
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        let_if = body[0].value
        assert isinstance(let_if, IfNode)
        assert isinstance(let_if.if_body, BlockNode)
        assert len(let_if.if_body.statements) == 1
        assert isinstance(let_if.if_body.expression, BinaryOpNode)

        assignment_if = body[1].right
        assert isinstance(assignment_if, IfNode)
        assert isinstance(assignment_if.else_body, BlockNode)
        assert len(assignment_if.else_body.statements) == 1

        return_if = body[2].value
        assert isinstance(return_if, IfNode)
        assert isinstance(return_if.else_body.expression, str)
    except Exception as e:
        pytest.fail(f"If expression result parsing failed: {e}")


def test_match_expression_result_parsing():
    code = """
    fn test_match_expression() -> i32 {
        let value: i32 = match mode {
            0 => 1,
            1 => {
                let seed = 2;
                seed + 1
            },
            _ => 3,
        };
        output.color = match value {
            0 => 0,
            _ => value + 1,
        };
        return match value {
            0 => value,
            _ => {
                let next = value + 1;
                next
            },
        };
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        let_match = body[0].value
        assert isinstance(let_match, MatchNode)
        assert let_match.expression == "mode"
        assert let_match.arms[0].body == "1"
        assert isinstance(let_match.arms[1].body, BlockNode)
        assert len(let_match.arms[1].body.statements) == 1
        assert isinstance(let_match.arms[1].body.expression, BinaryOpNode)

        assignment_match = body[1].right
        assert isinstance(assignment_match, MatchNode)
        assert assignment_match.arms[1].pattern == "_"
        assert isinstance(assignment_match.arms[1].body, BinaryOpNode)

        return_match = body[2].value
        assert isinstance(return_match, MatchNode)
        assert isinstance(return_match.arms[1].body, BlockNode)
        assert return_match.arms[1].body.expression == "next"
    except Exception as e:
        pytest.fail(f"Match expression result parsing failed: {e}")


def test_match_expression_return_arm_parsing():
    code = """
    fn render() {
        let window_surface = match self.window_surface {
            Some(ws) => ws,
            None => return,
        };
        use_value(window_surface);
    }
    """

    ast = parse_code(code)
    match_value = ast.functions[0].body[0].value

    assert isinstance(match_value, MatchNode)
    assert match_value.arms[0].body == "ws"
    assert isinstance(match_value.arms[1].body, ReturnNode)
    assert match_value.arms[1].body.value is None


def test_block_final_if_match_expression_parsing():
    code = """
    fn test_block_final_expression() -> i32 {
        let value: i32 = {
            let seed = 1;
            if ready {
                seed + 1
            } else {
                let fallback = 2;
                fallback
            }
        };
        output.color = {
            match value {
                0 => 0,
                _ => {
                    let next = value + 1;
                    next
                },
            }
        };
        return {
            match value {
                0 => value,
                _ => value + 1,
            }
        };
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        let_block = body[0].value
        assert isinstance(let_block, BlockNode)
        assert len(let_block.statements) == 1
        assert isinstance(let_block.expression, IfNode)
        assert isinstance(let_block.expression.if_body, BlockNode)
        assert isinstance(let_block.expression.if_body.expression, BinaryOpNode)
        assert isinstance(let_block.expression.else_body, BlockNode)
        assert len(let_block.expression.else_body.statements) == 1
        assert let_block.expression.else_body.expression == "fallback"

        assignment_block = body[1].right
        assert isinstance(assignment_block, BlockNode)
        assert isinstance(assignment_block.expression, MatchNode)
        assert assignment_block.expression.arms[1].pattern == "_"
        assert isinstance(assignment_block.expression.arms[1].body, BlockNode)

        return_block = body[2].value
        assert isinstance(return_block, BlockNode)
        assert isinstance(return_block.expression, MatchNode)
        assert isinstance(return_block.expression.arms[1].body, BinaryOpNode)
    except Exception as e:
        pytest.fail(f"Block final if/match expression parsing failed: {e}")


def test_block_expression_semicolon_statement_parsing():
    code = """
    fn test_block_expression_statements() {
        let value: i32 = {
            compute_default();
            fallback
        };
        loop {
            break match mode {
                _ => {
                    compute_unit();
                },
            };
        }
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        let_block = body[0].value
        assert isinstance(let_block, BlockNode)
        assert len(let_block.statements) == 1
        assert isinstance(let_block.statements[0], FunctionCallNode)
        assert let_block.expression == "fallback"

        match_break = body[1].body[0]
        assert isinstance(match_break, BreakNode)
        assert isinstance(match_break.value, MatchNode)
        arm_block = match_break.value.arms[0].body
        assert isinstance(arm_block, BlockNode)
        assert len(arm_block.statements) == 1
        assert isinstance(arm_block.statements[0], FunctionCallNode)
        assert arm_block.expression is None
    except Exception as e:
        pytest.fail(f"Block expression semicolon statement parsing failed: {e}")


def test_block_expression_semicolon_if_match_statement_parsing():
    code = """
    fn test_statement_like_block_expressions() {
        let value: i32 = {
            if ready {
                compute_ready();
                1
            } else {
                compute_fallback();
                2
            };
            match mode {
                0 => {
                    compute_zero();
                    0
                },
                _ => {
                    compute_default();
                    3
                },
            };
            fallback
        };
    }
    """
    try:
        ast = parse_code(code)
        let_block = ast.functions[0].body[0].value

        assert isinstance(let_block, BlockNode)
        assert len(let_block.statements) == 2
        assert let_block.expression == "fallback"

        if_stmt = let_block.statements[0]
        assert isinstance(if_stmt, IfNode)
        assert len(if_stmt.if_body) == 1
        assert isinstance(if_stmt.if_body[0], FunctionCallNode)
        assert len(if_stmt.else_body) == 1
        assert isinstance(if_stmt.else_body[0], FunctionCallNode)

        match_stmt = let_block.statements[1]
        assert isinstance(match_stmt, MatchNode)
        assert isinstance(match_stmt.arms[0].body[0], FunctionCallNode)
        assert isinstance(match_stmt.arms[1].body[0], FunctionCallNode)
    except Exception as e:
        pytest.fail(f"Block expression semicolon if/match parsing failed: {e}")


def test_let_pattern_parsing():
    code = """
    fn test_let_patterns() {
        let _ = compute_discard();
        let (_, y) = (compute_x(), 2);
        let (a, (b, _)) = (1, (2, compute_ignored()));
        let (tx, ty): (f32, i32) = (1.0, 2);
        let (_, (nz, _)): (f32, (i32, Vec3<f32>)) = (
            compute_typed(),
            (3, compute_typed_drop()),
        );
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        assert body[0].name == "_"
        assert isinstance(body[0].value, FunctionCallNode)

        tuple_pattern = body[1].name
        assert isinstance(tuple_pattern, TupleNode)
        assert tuple_pattern.elements == ["_", "y"]
        assert isinstance(body[1].value, TupleNode)

        nested_pattern = body[2].name
        assert isinstance(nested_pattern, TupleNode)
        assert nested_pattern.elements[0] == "a"
        assert isinstance(nested_pattern.elements[1], TupleNode)
        assert nested_pattern.elements[1].elements == ["b", "_"]

        typed_pattern = body[3].name
        assert isinstance(typed_pattern, TupleNode)
        assert body[3].vtype == "(f32, i32)"

        typed_nested_pattern = body[4].name
        assert isinstance(typed_nested_pattern, TupleNode)
        assert isinstance(typed_nested_pattern.elements[1], TupleNode)
        assert body[4].vtype == "(f32, (i32, Vec3<f32>))"
    except Exception as e:
        pytest.fail(f"Let pattern parsing failed: {e}")


def test_tuple_pattern_result_expression_elements_parsing():
    code = """
    fn test_tuple_result_elements() {
        let (from_if, from_loop, from_match, from_block): (i32, i32, i32, i32) = (
            if ready {
                prepare_if();
                1
            } else {
                2
            },
            loop {
                break 3;
            },
            match mode {
                0 => 4,
                _ => {
                    prepare_match();
                    5
                },
            },
            {
                prepare_block();
                6
            },
        );
        let (_, kept): (i32, i32) = (
            if should_drop {
                prepare_drop();
                31
            } else {
                fallback_drop();
                32
            },
            7,
        );
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        value = body[0].value
        assert isinstance(value, TupleNode)
        assert isinstance(value.elements[0], IfNode)
        assert isinstance(value.elements[1], LoopNode)
        assert isinstance(value.elements[2], MatchNode)
        assert isinstance(value.elements[3], BlockNode)

        discarded_value = body[1].value
        assert isinstance(discarded_value, TupleNode)
        assert isinstance(discarded_value.elements[0], IfNode)
        assert discarded_value.elements[1] == "7"
    except Exception as e:
        pytest.fail(f"Tuple pattern result expression parsing failed: {e}")


def test_simple_if_expression_stays_ternary_parsing():
    code = """
    fn test_if_expression() {
        let value = if ready { 1 } else { 2 };
    }
    """
    try:
        ast = parse_code(code)
        value = ast.functions[0].body[0].value
        assert isinstance(value, TernaryOpNode)
    except Exception as e:
        pytest.fail(f"Simple if expression parsing failed: {e}")


def test_try_expression_parsing():
    code = """
    fn try_values(left: Result<i32, i32>, right: Result<i32, i32>) -> Result<i32, i32> {
        let value = left?;
        let wrapped = Ok(right?);
        let field = make_result()?.field;
        return Ok(value);
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        assert isinstance(body[0].value, TryNode)
        assert body[0].value.expression == "left"

        assert isinstance(body[1].value, FunctionCallNode)
        assert body[1].value.name == "Ok"
        assert isinstance(body[1].value.args[0], TryNode)
        assert body[1].value.args[0].expression == "right"

        assert isinstance(body[2].value, MemberAccessNode)
        assert body[2].value.member == "field"
        assert isinstance(body[2].value.object, TryNode)
        assert isinstance(body[2].value.object.expression, FunctionCallNode)
        assert body[2].value.object.expression.name == "make_result"
    except Exception as e:
        pytest.fail(f"Try expression parsing failed: {e}")


def test_qualified_path_try_postfix_parsing_from_wgpu_deno_webgpu():
    # Reduced from:
    # Repo: https://github.com/gfx-rs/wgpu.git
    # Commit: 26e2525f8dea477ef356b80efb6eb1bc1dec120d
    # Path: deno_webgpu/compute_pass.rs set_bind_group.
    code = """
    fn set_bind_group(
        scope: Scope,
        dynamic_offsets: Value,
        prefix: Prefix,
        context: Context,
        options: Options,
    ) -> Result<(), WebIdlError> {
        let offsets = <Option<Vec<u32>>>::convert(
            scope,
            dynamic_offsets,
            prefix,
            context,
            options,
        )?.unwrap_or_default();
        Ok(())
    }
    """

    ast = parse_code(code)
    value = ast.functions[0].body[0].value

    assert isinstance(value, FunctionCallNode)
    assert isinstance(value.name, MemberAccessNode)
    assert value.name.member == "unwrap_or_default"
    assert isinstance(value.name.object, TryNode)
    subject = value.name.object.expression
    assert isinstance(subject, FunctionCallNode)
    assert subject.name == "<Option<Vec<u32>>>::convert"


def test_await_expression_parsing():
    code = """
    fn await_values(future: Result<i32, i32>, object: FutureObject) -> Result<i32, i32> {
        let ready = future.await;
        let value = future.await?;
        let field = object.await.value;
        return Ok(value);
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        assert isinstance(body[0].value, AwaitNode)
        assert body[0].value.expression == "future"

        assert isinstance(body[1].value, TryNode)
        assert isinstance(body[1].value.expression, AwaitNode)
        assert body[1].value.expression.expression == "future"

        assert isinstance(body[2].value, MemberAccessNode)
        assert body[2].value.member == "value"
        assert isinstance(body[2].value.object, AwaitNode)
        assert body[2].value.object.expression == "object"
    except Exception as e:
        pytest.fail(f"Await expression parsing failed: {e}")


def test_async_function_and_block_parsing():
    code = """
    #[vertex_shader]
    pub async fn load(future: Result<i32, i32>) -> Result<i32, i32> {
        return Ok(future.await?);
    }

    impl Loader {
        pub async fn method(&self, future: Result<i32, i32>) -> Result<i32, i32> {
            return Ok(future.await?);
        }
    }

    trait Fetch {
        async fn fetch(&self) -> Result<i32, i32>;
    }

    fn async_blocks(value: Result<i32, i32>) -> Result<i32, i32> {
        let result: i32 = async {
            let v: i32 = value.await?;
            v + 1
        };
        let moved: i32 = async move {
            value.await?
        };
        return Ok(result + moved);
    }
    """
    try:
        ast = parse_code(code)

        load = ast.functions[0]
        assert load.is_async is True
        assert load.visibility == "pub"
        assert load.attributes[0].name == "vertex_shader"

        method = ast.impl_blocks[0].methods[0]
        assert method.is_async is True
        assert method.visibility == "pub"

        trait_method = ast.traits[0].methods[0]
        assert trait_method.is_async is True
        assert trait_method.body == []

        async_blocks = ast.functions[1]
        result_block = async_blocks.body[0].value
        assert isinstance(result_block, AsyncBlockNode)
        assert result_block.is_move is False
        assert isinstance(result_block.block, BlockNode)
        assert isinstance(result_block.block.statements[0], LetNode)
        assert isinstance(result_block.block.statements[0].value, TryNode)
        assert isinstance(result_block.block.statements[0].value.expression, AwaitNode)

        moved_block = async_blocks.body[1].value
        assert isinstance(moved_block, AsyncBlockNode)
        assert moved_block.is_move is True
        assert isinstance(moved_block.block.expression, TryNode)
    except Exception as e:
        pytest.fail(f"Async function and block parsing failed: {e}")


def test_unsafe_extern_function_and_block_parsing():
    code = """
    #[no_mangle]
    pub unsafe extern "C" fn decode(value: Result<i32, i32>) -> Result<i32, i32> {
        let result: i32 = unsafe {
            let v: i32 = value?;
            v + 1
        };
        return Ok(result);
    }

    impl Loader {
        pub unsafe fn method(&self, value: Result<i32, i32>) -> Result<i32, i32> {
            return unsafe {
                value
            };
        }
    }

    trait Fetch {
        unsafe fn fetch(&self) -> i32;
    }

    unsafe extern "C" {
        fn foreign_decode(value: i32) -> i32;
    }

    fn after_extern_block() -> i32 {
        return 1;
    }
    """
    try:
        ast = parse_code(code)

        decode = ast.functions[0]
        assert decode.is_unsafe is True
        assert decode.abi == "C"
        assert decode.visibility == "pub"
        assert decode.attributes[0].name == "no_mangle"

        result_block = decode.body[0].value
        assert isinstance(result_block, UnsafeBlockNode)
        assert isinstance(result_block.block, BlockNode)
        assert isinstance(result_block.block.statements[0], LetNode)
        assert isinstance(result_block.block.statements[0].value, TryNode)

        method = ast.impl_blocks[0].methods[0]
        assert method.is_unsafe is True
        assert method.abi is None
        assert isinstance(method.body[0].value, UnsafeBlockNode)

        trait_method = ast.traits[0].methods[0]
        assert trait_method.is_unsafe is True
        assert trait_method.body == []

        assert ast.functions[1].name == "after_extern_block"
    except Exception as e:
        pytest.fail(f"Unsafe extern function and block parsing failed: {e}")


def test_local_unsafe_extern_block_parsing_from_rust_cuda_thread_intrinsic():
    # Reduced from https://github.com/Rust-GPU/Rust-CUDA commit
    # 103a8d56935c4e0885ff7c3d25402319df1a8e00,
    # crates/cuda_std/src/thread.rs __nvvm_warp_size declaration.
    code = """
    pub fn warp_size() -> u32 {
        unsafe extern "C" {
            fn __nvvm_warp_size() -> u32;
        }

        unsafe {
            __nvvm_warp_size()
        }
    }
    """

    ast = parse_code(code)
    function = ast.functions[0]

    assert function.name == "warp_size"
    assert len(function.body) == 1
    unsafe_block = function.body[0]
    assert isinstance(unsafe_block, UnsafeBlockNode)
    assert unsafe_block.block.expression.name == "__nvvm_warp_size"


def test_associated_type_projection_alias_parsing_from_rust_cuda_gpu_rand():
    # Reduced from https://github.com/Rust-GPU/Rust-CUDA commit
    # 103a8d56935c4e0885ff7c3d25402319df1a8e00,
    # crates/gpu_rand/src/default.rs DefaultRand SeedableRng impl.
    code = """
    struct Xoroshiro128StarStar;
    struct DefaultRand;

    trait SeedableRng {
        type Seed;
    }

    impl SeedableRng for DefaultRand {
        type Seed = <Xoroshiro128StarStar as SeedableRng>::Seed;
    }
    """

    ast = parse_code(code)
    alias = ast.impl_blocks[0].type_aliases[0]

    assert isinstance(alias, TypeAliasNode)
    assert alias.name == "Seed"
    assert alias.alias_type == "<Xoroshiro128StarStar as SeedableRng>::Seed"


def test_const_function_and_block_parsing():
    code = """
    pub const SCALE: i32 = 2;

    pub const unsafe extern "C" fn scale(value: i32) -> i32 {
        let result: i32 = const {
            let base: i32 = SCALE;
            base + value
        };
        return result;
    }

    impl Loader {
        pub const fn method(&self, value: i32) -> i32 {
            return const {
                value + 1
            };
        }
    }
    """
    try:
        ast = parse_code(code)

        assert len(ast.global_variables) == 1
        assert isinstance(ast.global_variables[0], ConstNode)
        assert ast.global_variables[0].name == "SCALE"

        scale = ast.functions[0]
        assert scale.is_const is True
        assert scale.is_unsafe is True
        assert scale.abi == "C"

        result_block = scale.body[0].value
        assert isinstance(result_block, ConstBlockNode)
        assert isinstance(result_block.block, BlockNode)
        assert isinstance(result_block.block.statements[0], LetNode)
        assert isinstance(result_block.block.expression, BinaryOpNode)

        method = ast.impl_blocks[0].methods[0]
        assert method.is_const is True
        assert method.is_unsafe is False
        assert isinstance(method.body[0].value, ConstBlockNode)
    except Exception as e:
        pytest.fail(f"Const function and block parsing failed: {e}")


def test_local_const_static_item_parsing():
    code = """
    fn local_items(value: i32) -> i32 {
        const OFFSET: i32 = 1;
        static mut CACHE: i32 = 2;
        let result: i32 = {
            const INNER: i32 = 3;
            static ENABLED: bool = true;
            value + OFFSET + INNER
        };
        return result;
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        assert isinstance(body[0], ConstNode)
        assert body[0].name == "OFFSET"
        assert body[0].vtype == "i32"

        assert isinstance(body[1], StaticNode)
        assert body[1].name == "CACHE"
        assert body[1].is_mutable is True

        block = body[2].value
        assert isinstance(block, BlockNode)
        assert isinstance(block.statements[0], ConstNode)
        assert block.statements[0].name == "INNER"
        assert isinstance(block.statements[1], StaticNode)
        assert block.statements[1].name == "ENABLED"
        assert block.expression is not None
    except Exception as e:
        pytest.fail(f"Local const/static item parsing failed: {e}")


def test_statement_and_expression_attributes_are_ignored():
    code = """
    fn entry() {
        #[spirv(vertex)]
        let _statement = ();

        let _closure = #[spirv(fragment)]
        || {};

        (
            #[spirv(compute)]
            (1, 2, 3)
        );

        match () {
            #[spirv(fragment)]
            _arm => {}
        }
    }
    """

    ast = parse_code(code)
    body = ast.functions[0].body

    assert isinstance(body[0], LetNode)
    assert body[0].name == "_statement"
    assert isinstance(body[1], LetNode)
    assert isinstance(body[1].value, ClosureNode)
    assert isinstance(body[2], TupleNode)
    assert isinstance(body[3], MatchNode)
    assert body[3].arms[0].pattern == "_arm"


def test_try_expression_host_parsing():
    code = """
    struct Output {
        position: Vec3<f32>,
        value: i32,
    }

    fn try_hosts(
        start: Result<i32, i32>,
        end: Result<i32, i32>,
        value: Result<Option<i32>, i32>,
        position: Result<Vec3<f32>, i32>,
    ) -> Result<i32, i32> {
        for i in start?..end? {
            use_value(i);
        }
        let output = Output {
            position: position?,
            value: 1,
        };
        let mapped: i32 = match value? {
            Some(v) => v,
            None => 0,
        };
        return Ok(mapped);
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        for_loop = body[0]
        assert isinstance(for_loop, ForNode)
        assert isinstance(for_loop.iterable, RangeNode)
        assert isinstance(for_loop.iterable.start, TryNode)
        assert isinstance(for_loop.iterable.end, TryNode)

        output = body[1].value
        assert isinstance(output, StructInitializationNode)
        assert output.fields[0][0] == "position"
        assert isinstance(output.fields[0][1], TryNode)

        mapped_match = body[2].value
        assert isinstance(mapped_match, MatchNode)
        assert isinstance(mapped_match.expression, TryNode)
        assert isinstance(mapped_match.expression.expression, str)
    except Exception as e:
        pytest.fail(f"Try expression host parsing failed: {e}")


def test_for_range_uppercase_bound_does_not_parse_loop_body_as_struct_literal():
    code = """
    struct Output {
        value: i32,
    }

    const AA: usize = 2;

    fn shader_loop() {
        for jj in 0..AA {
            for ii in 0..AA {
                let output = Output {
                    value: ii as i32 + jj as i32,
                };
                use_value(output.value);
            }
        }
    }
    """

    ast = parse_code(code)
    body = ast.functions[0].body

    outer_loop = body[0]
    assert isinstance(outer_loop, ForNode)
    assert isinstance(outer_loop.iterable, RangeNode)
    assert outer_loop.iterable.end == "AA"

    inner_loop = outer_loop.body[0]
    assert isinstance(inner_loop, ForNode)
    assert isinstance(inner_loop.iterable, RangeNode)
    assert inner_loop.iterable.end == "AA"

    output = inner_loop.body[0].value
    assert isinstance(output, StructInitializationNode)
    assert output.struct_name == "Output"


def test_if_uppercase_condition_does_not_parse_body_as_struct_literal():
    code = """
    const SINGLE: bool = false;

    fn shader_branch(value: i32) {
        let mut s: i32 = 0;
        if SINGLE {
            s = value;
        }
        use_value(s);
    }
    """

    ast = parse_code(code)
    body = ast.functions[0].body

    branch = body[1]
    assert isinstance(branch, IfNode)
    assert branch.condition == "SINGLE"
    assert isinstance(branch.if_body[0], AssignmentNode)


def test_for_uppercase_iterable_does_not_parse_loop_body_as_struct_literal():
    code = """
    fn shader_loop() {
        for gid in LAYOUT_RANGE {
            eval_layouts(gid as u32, &mut out);
        }
    }
    """

    ast = parse_code(code)
    loop = ast.functions[0].body[0]

    assert isinstance(loop, ForNode)
    assert loop.pattern == "gid"
    assert loop.iterable == "LAYOUT_RANGE"
    assert isinstance(loop.body[0], FunctionCallNode)


def test_try_block_expression_parsing():
    code = """
    fn try_blocks(value: Result<i32, i32>, maybe: Option<i32>) -> Result<i32, i32> {
        let result = try {
            let v: i32 = value?;
            v + 1
        };
        let nested = Ok(try {
            let v: i32 = value?;
            v
        });
        return Ok(try {
            let v: i32 = value?;
            v
        }?);
    }
    """
    try:
        ast = parse_code(code)
        body = ast.functions[0].body

        result_block = body[0].value
        assert isinstance(result_block, TryBlockNode)
        assert isinstance(result_block.block, BlockNode)
        assert len(result_block.block.statements) == 1
        assert isinstance(result_block.block.statements[0], LetNode)
        assert isinstance(result_block.block.statements[0].value, TryNode)
        assert isinstance(result_block.block.expression, BinaryOpNode)

        nested_call = body[1].value
        assert isinstance(nested_call, FunctionCallNode)
        assert isinstance(nested_call.args[0], TryBlockNode)

        returned = body[2].value
        assert isinstance(returned, FunctionCallNode)
        assert isinstance(returned.args[0], TryNode)
        assert isinstance(returned.args[0].expression, TryBlockNode)
    except Exception as e:
        pytest.fail(f"Try block expression parsing failed: {e}")


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
        type WithScalar<S: Scalar>: CubePrimitive;

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
        assert len(trait.associated_types) == 3
        assert all(
            isinstance(associated_type, AssociatedTypeNode)
            for associated_type in trait.associated_types
        )
        assert trait.associated_types[0].name == "Output"
        assert trait.associated_types[0].generics == []
        assert trait.associated_types[0].bounds == ["Copy", "Clone"]
        assert trait.associated_types[0].default_type is None
        assert trait.associated_types[1].name == "Scratch"
        assert trait.associated_types[1].generics == []
        assert trait.associated_types[1].bounds == ["Into<Vec3<f32>>"]
        assert trait.associated_types[1].default_type == "Vec3<f32>"
        assert trait.associated_types[2].name == "WithScalar"
        assert trait.associated_types[2].generics == ["S: Scalar"]
        assert trait.associated_types[2].bounds == ["CubePrimitive"]
        assert trait.associated_types[2].default_type is None
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


def test_rust_gpu_associated_const_trait_and_impl_parsing_from_spirv_std():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # crates/spirv-std/src/{scalar.rs,scalar_or_vector.rs}.
    code = """
    use core::num::NonZeroUsize;

    pub unsafe trait Integer: Number {
        const WIDTH: usize;
        const SIGNED: bool;
    }

    pub unsafe trait ScalarOrVector {
        type Scalar: Scalar;
        const N: NonZeroUsize;
    }

    unsafe impl Integer for u32 {
        const WIDTH: usize = 32;
        const SIGNED: bool = false;
    }
    """

    ast = parse_code(code)

    integer = ast.traits[0]
    scalar_or_vector = ast.traits[1]
    impl_block = ast.impl_blocks[0]

    assert integer.name == "Integer"
    assert integer.visibility == "pub"
    assert integer.is_unsafe is True
    assert [const.name for const in integer.associated_consts] == ["WIDTH", "SIGNED"]
    assert [const.vtype for const in integer.associated_consts] == ["usize", "bool"]
    assert [const.value for const in integer.associated_consts] == [None, None]

    assert scalar_or_vector.name == "ScalarOrVector"
    assert scalar_or_vector.visibility == "pub"
    assert scalar_or_vector.is_unsafe is True
    assert scalar_or_vector.associated_types[0].name == "Scalar"
    assert [const.name for const in scalar_or_vector.associated_consts] == ["N"]
    assert scalar_or_vector.associated_consts[0].vtype == "NonZeroUsize"
    assert scalar_or_vector.associated_consts[0].value is None

    assert impl_block.trait_name == "Integer"
    assert impl_block.struct_name == "u32"
    assert impl_block.is_unsafe is True
    assert [const.name for const in impl_block.associated_consts] == [
        "WIDTH",
        "SIGNED",
    ]
    assert [const.value for const in impl_block.associated_consts] == [
        "32",
        "false",
    ]


def test_trait_supertrait_bounds_parsing():
    code = """
    trait Shape: Copy + Clone {
        fn distance(self, p: Vec2) -> f32;
    }
    """

    ast = parse_code(code)

    assert ast.traits[0].name == "Shape"
    assert ast.traits[0].supertraits == ["Copy", "Clone"]
    assert ast.traits[0].methods[0].name == "distance"


def test_trait_alias_parsing_from_rust_gpu_invalid_target_compiletest():
    # Reduced from:
    # Repo: https://github.com/Rust-GPU/rust-gpu
    # Path: tests/compiletests/ui/spirv-attr/invalid-target.rs
    code = """
    #![feature(trait_alias)]

    #[spirv_recursive_for_testing(vertex, uniform, descriptor_set = 0)]
    trait _TraitAlias<T> = Copy + Into<T>;
    """
    ast = parse_code(code)
    trait = ast.traits[0]

    assert trait.name == "_TraitAlias"
    assert trait.generics == ["T"]
    assert trait.supertraits == ["Copy", "Into<T>"]
    assert trait.methods == []
    assert trait.is_alias is True
    assert trait.attributes[0].name == "spirv_recursive_for_testing"


def test_struct_initialization_field_shorthand_parsing():
    code = """
    fn make_stroke(shape: Shape, thickness: f32) -> Stroke {
        return Stroke {
            shape,
            thickness,
        };
    }
    """

    ast = parse_code(code)
    value = ast.functions[0].body[0].value

    assert isinstance(value, StructInitializationNode)
    assert value.fields == [("shape", "shape"), ("thickness", "thickness")]


def test_rust_cuda_lowercase_struct_initialization_field_shorthand_parsing():
    # Reduced from Rust-GPU/Rust-CUDA commit
    # 103a8d56935c4e0885ff7c3d25402319df1a8e00,
    # crates/optix/examples/ex03_window/src/gl_util.rs.
    code = """
    #[allow(non_camel_case_types)]
    pub struct f32x2 {
        x: f32,
        y: f32,
    }

    impl f32x2 {
        pub fn new(x: f32, y: f32) -> f32x2 {
            f32x2 { x, y }
        }
    }
    """

    ast = parse_code(code)
    value = ast.impl_blocks[0].methods[0].body[0]

    assert isinstance(value, StructInitializationNode)
    assert value.struct_name == "f32x2"
    assert value.fields == [("x", "x"), ("y", "y")]


def test_qualified_struct_initialization_chains_in_match_arm():
    code = """
    fn shade(i: i32, resolution: Vec3, time: f32, color: &mut Vec4, frag_coord: Vec2) {
        match i {
            0 => two_tweets::Inputs { resolution, time }.main_image(color, frag_coord),
            _ => {}
        }
    }
    """

    ast = parse_code(code)
    arm_call = ast.functions[0].body[0].arms[0].body[0]

    assert isinstance(arm_call, FunctionCallNode)
    assert isinstance(arm_call.name, MemberAccessNode)
    assert arm_call.name.member == "main_image"
    assert isinstance(arm_call.name.object, StructInitializationNode)
    assert arm_call.name.object.struct_name == "two_tweets::Inputs"
    assert arm_call.name.object.fields == [
        ("resolution", "resolution"),
        ("time", "time"),
    ]


def test_builtin_vector_struct_initialization_in_result_match_arm():
    code = """
    fn shade(i: i32, v1: Vec3) {
        match i {
            0 => Result::Ok(Vec3{x: v1.x, y: v1.y, z: v1.z}),
            _ => Result::Err(0),
        }
    }
    """

    ast = parse_code(code)
    arm_call = ast.functions[0].body[0].arms[0].body[0]

    assert isinstance(arm_call, FunctionCallNode)
    assert arm_call.name == "Result::Ok"
    assert isinstance(arm_call.args[0], StructInitializationNode)
    assert arm_call.args[0].struct_name == "Vec3"
    assert [field_name for field_name, _ in arm_call.args[0].fields] == [
        "x",
        "y",
        "z",
    ]


def test_match_arm_block_tail_preserves_following_guarded_struct_arm():
    code = """
    fn shade(model: LightingModel) {
        match model {
            LightingModel::PBR { ao } => {
                let x: f32 = 1.0;
                {
                    let y = x;
                    y
                }
            },
            LightingModel::Toon { levels } if (levels > 0) => use_levels(levels),
            _ => use_default(),
        }
    }
    """

    ast = parse_code(code)
    match_stmt = ast.functions[0].body[0]

    assert isinstance(match_stmt, MatchNode)
    assert len(match_stmt.arms) == 3
    assert isinstance(match_stmt.arms[0].body[-1], BlockNode)
    guarded = match_stmt.arms[1]
    assert isinstance(guarded.pattern, MatchStructPatternNode)
    assert guarded.pattern.name == "LightingModel::Toon"
    assert isinstance(guarded.guard, BinaryOpNode)
    assert guarded.guard.op == ">"


def test_lowercase_self_condition_does_not_parse_body_as_struct_literal():
    code = """
    impl FloatExt for f32 {
        fn step(self, x: f32) -> f32 {
            if x < self {
                0.0
            } else {
                1.0
            }
        }
    }
    """

    ast = parse_code(code)
    branch = ast.impl_blocks[0].functions[0].body[0]

    assert isinstance(branch, IfNode)
    assert isinstance(branch.condition, BinaryOpNode)
    assert branch.condition.left == "x"
    assert branch.condition.op == "<"
    assert branch.condition.right == "self"


def test_qualified_struct_initialization_rest_field_parsing():
    code = """
    fn configure(instance_flags: wgpu::InstanceFlags) {
        let descriptor = wgpu::InstanceDescriptor {
            flags: instance_flags,
            ..Default::default()
        };
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));
    }
    """

    ast = parse_code(code)
    descriptor = ast.functions[0].body[0].value
    adapter_call = ast.functions[0].body[1].value

    assert isinstance(descriptor, StructInitializationNode)
    assert descriptor.struct_name == "wgpu::InstanceDescriptor"
    assert descriptor.fields[0] == ("flags", "instance_flags")
    assert descriptor.fields[1][0] == ".."
    assert isinstance(descriptor.fields[1][1], FunctionCallNode)
    assert descriptor.fields[1][1].name == "Default::default"

    request_call = adapter_call.args[0]
    assert isinstance(request_call.args[0], ReferenceNode)
    assert isinstance(request_call.args[0].expression, StructInitializationNode)
    assert request_call.args[0].expression.struct_name == "wgpu::RequestAdapterOptions"


def test_impl_trait_parameter_type_parsing():
    code = """
    fn fill(shape: impl Shape, color: Vec4) {
        return;
    }
    """

    ast = parse_code(code)

    assert ast.functions[0].params[0].vtype == "impl Shape"
    assert ast.functions[0].params[0].name == "shape"


def test_impl_callable_trait_parameter_type_parsing_from_rust_gpu():
    code = """
    fn setup_shader_crate_with_cargo_toml(
        f: impl FnOnce(&mut dyn rspirv::binary::Consumer) -> std::io::Result<()>,
        mut before_pass: impl FnMut(&'static str, &Module) -> P,
        display: &dyn DifferenceDisplay,
    ) {
        return;
    }

    fn visit<F: FnMut(Value) -> Option<Value>>(f: F) {
        return;
    }
    """

    ast = parse_code(code)

    setup_params = ast.functions[0].params
    assert (
        setup_params[0].vtype
        == "impl FnOnce(&mut dyn rspirv::binary::Consumer) -> std::io::Result<()>"
    )
    assert setup_params[0].name == "f"
    assert setup_params[1].vtype == "impl FnMut(&str, &Module) -> P"
    assert setup_params[1].name == "before_pass"
    assert setup_params[2].vtype == "&dyn DifferenceDisplay"

    assert ast.functions[1].generics == ["F: FnMut(Value) -> Option<Value>"]


def test_wgpu_spawn_impl_future_static_bound_parsing_from_upstream():
    # Reduced from gfx-rs/wgpu commit
    # 26e2525f8dea477ef356b80efb6eb1bc1dec120d,
    # examples/features/src/hello_triangle/mod.rs spawn.
    code = """
    use std::future::Future;

    #[cfg(not(target_arch = "wasm32"))]
    fn spawn(f: impl Future<Output = ()> + 'static) {
        pollster::block_on(f);
    }
    """

    ast = parse_code(code)
    function = ast.functions[0]

    assert function.name == "spawn"
    assert function.params[0].name == "f"
    assert function.params[0].vtype == "impl Future<Output =()> + 'static"
    assert function.attributes[0].name == "cfg"
    assert function.body[0].name == "pollster::block_on"


def test_trait_default_method_body_parsing():
    code = """
    trait ShaderMath {
        fn saturate(x: f32) -> f32 {
            clamp(x, 0.0, 1.0)
        }

        unsafe fn raw(x: f32) -> f32 {
            x
        }

        fn passthrough(x: f32) -> f32;
    }
    """
    try:
        ast = parse_code(code)
        trait = ast.traits[0]

        assert [method.name for method in trait.methods] == [
            "saturate",
            "raw",
            "passthrough",
        ]
        assert len(trait.methods[0].body) == 1
        assert isinstance(trait.methods[0].body[0], FunctionCallNode)
        assert trait.methods[0].body[0].name == "clamp"
        assert trait.methods[1].is_unsafe is True
        assert trait.methods[1].body == ["x"]
        assert trait.methods[2].body == []
    except Exception as e:
        pytest.fail(f"Trait default method body parsing failed: {e}")


def test_for_loop_reference_binding_patterns():
    code = """
    fn scan_refs(values: &[i32], pairs: &[(&i32, &i32)]) {
        for &value in values {
            use_value(value);
        }

        for (index, (&left, &mut right)) in pairs.iter().enumerate() {
            use_pair(index, left, right);
        }

        for &(ref name, tag) in pairs {
            use_pair(name, tag);
        }
    }
    """

    ast = parse_code(code)
    body = ast.functions[0].body

    value_loop = body[0]
    assert isinstance(value_loop, ForNode)
    assert isinstance(value_loop.pattern, ReferenceNode)
    assert value_loop.pattern.expression == "value"
    assert value_loop.pattern.is_mutable is False

    pair_loop = body[1]
    assert isinstance(pair_loop, ForNode)
    assert isinstance(pair_loop.pattern, TupleNode)
    assert pair_loop.pattern.elements[0] == "index"

    pair_pattern = pair_loop.pattern.elements[1]
    assert isinstance(pair_pattern, TupleNode)
    assert isinstance(pair_pattern.elements[0], ReferenceNode)
    assert pair_pattern.elements[0].expression == "left"
    assert pair_pattern.elements[0].is_mutable is False
    assert isinstance(pair_pattern.elements[1], ReferenceNode)
    assert pair_pattern.elements[1].expression == "right"
    assert pair_pattern.elements[1].is_mutable is True

    ref_loop = body[2]
    assert isinstance(ref_loop, ForNode)
    assert isinstance(ref_loop.pattern, ReferenceNode)
    ref_pair_pattern = ref_loop.pattern.expression
    assert isinstance(ref_pair_pattern, TupleNode)
    assert ref_pair_pattern.elements == ["name", "tag"]


def test_spirv_std_fragment_example_parses_deref_output_and_casts():
    code = """
    use spirv_std::spirv;
    use glam::{Vec3, Vec4, vec2, vec3};

    #[spirv(fragment)]
    pub fn main(
        #[spirv(frag_coord)] in_frag_coord: &Vec4,
        #[spirv(push_constant)] constants: &ShaderConstants,
        output: &mut Vec4,
    ) {
        let frag_coord = vec2(in_frag_coord.x, in_frag_coord.y);
        let mut uv = (frag_coord - 0.5 * vec2(constants.width as f32, constants.height as f32))
            / constants.height as f32;
        uv.y = -uv.y;
        let eye_pos = vec3(0.0, 0.0997, 0.2);
        let sun_pos = vec3(0.0, 75.0, -1000.0);
        let dir = get_ray_dir(uv, eye_pos, sun_pos);
        let color = sky(dir, sun_pos);
        *output = tonemap(color).extend(1.0)
    }
    """

    ast = parse_code(code)
    function = ast.functions[0]

    assert function.attributes[0].name == "spirv"
    assert function.attributes[0].args == ["fragment"]
    assert [(param.name, param.vtype) for param in function.params] == [
        ("in_frag_coord", "&Vec4"),
        ("constants", "&ShaderConstants"),
        ("output", "&mut Vec4"),
    ]
    assert function.params[0].attributes[0].args == ["frag_coord"]
    assert function.params[1].attributes[0].args == ["push_constant"]
    assert isinstance(function.body[1].value, BinaryOpNode)
    assert isinstance(function.body[1].value.right, CastNode)
    assert isinstance(function.body[-1], AssignmentNode)
    assert isinstance(function.body[-1].left, DereferenceNode)


def test_spirv_specialization_and_workgroup_attributes_parse_from_dev_guide():
    code = """
    use spirv_std::spirv;

    #[spirv(vertex)]
    fn specialize(
        #[spirv(spec_constant(id = 100))] x_u64_lo: u32,
        #[spirv(spec_constant(id = 101))] x_u64_hi: u32,
    ) {
        let x_u64 = ((x_u64_hi as u64) << 32) | (x_u64_lo as u64);
    }

    #[spirv(compute(threads(32)))]
    fn shared(#[spirv(workgroup)] tile: &mut [Vec4; 4]) {}
    """

    ast = parse_code(code)
    specialize, shared = ast.functions

    assert [param.attributes[0].args for param in specialize.params] == [
        ["spec_constant", "(", "id", "=", "100", ")"],
        ["spec_constant", "(", "id", "=", "101", ")"],
    ]
    assert isinstance(specialize.body[0].value, BinaryOpNode)
    assert specialize.body[0].value.op == "|"

    assert shared.attributes[0].args == [
        "compute",
        "(",
        "threads",
        "(",
        "32",
        ")",
        ")",
    ]
    assert shared.params[0].vtype == "&mut Vec4[4]"
    assert shared.params[0].attributes[0].args == ["workgroup"]


def test_rust_gpu_sky_shader_hex_default_spec_constant_parse():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/shaders/sky-shader/src/lib.rs fragment specialization constant.
    code = """
    use spirv_std::spirv;

    #[spirv(fragment)]
    pub fn main_fs(
        #[spirv(frag_coord)] in_frag_coord: Vec4,
        #[spirv(push_constant)] constants: &ShaderConstants,
        output: &mut Vec4,
        #[spirv(spec_constant(id = 0x5007, default = 100))]
        sun_intensity_extra_spec_const_factor: u32,
    ) {
        let frag_coord = vec2(in_frag_coord.x, in_frag_coord.y);
        *output = fs(constants, frag_coord, sun_intensity_extra_spec_const_factor);
    }
    """

    ast = parse_code(code)
    function = ast.functions[0]

    assert function.name == "main_fs"
    assert function.attributes[0].args == ["fragment"]
    assert function.params[3].name == "sun_intensity_extra_spec_const_factor"
    assert function.params[3].vtype == "u32"
    assert function.params[3].attributes[0].args == [
        "spec_constant",
        "(",
        "id",
        "=",
        "0x5007",
        "default",
        "=",
        "100",
        ")",
    ]
    assert isinstance(function.body[-1].left, DereferenceNode)


def test_rust_gpu_mouse_shader_tuple_struct_destructure_parse():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/shaders/mouse-shader/src/lib.rs Line::distance.
    code = """
    struct Line(Vec2, Vec2);

    impl Shape for Line {
        fn distance(self, p: Vec2) -> f32 {
            let Line(a, b) = self;
            let ap = p - a;
            ap.x
        }
    }
    """

    ast = parse_code(code)
    method = ast.impl_blocks[0].methods[0]
    destructure = method.body[0]

    assert isinstance(destructure, LetNode)
    assert isinstance(destructure.name, FunctionCallNode)
    assert destructure.name.name == "Line"
    assert destructure.name.args == ["a", "b"]
    assert destructure.value == "self"
    assert method.body[1].name == "ap"
    assert isinstance(method.body[1].value, BinaryOpNode)
    assert method.body[1].value.op == "-"


def test_rust_gpu_mouse_shader_max_element_method_parse():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/shaders/mouse-shader/src/lib.rs Rectangle::distance.
    code = """
    impl Shape for Rectangle {
        fn distance(self, p: Vec2) -> f32 {
            let diff = p - self.center;
            let diff = vec2(diff.x.abs(), diff.y.abs());
            (diff - self.size / 2.0).max_element()
        }
    }
    """

    ast = parse_code(code)
    method = ast.impl_blocks[0].methods[0]
    final_expression = method.body[-1]

    assert isinstance(final_expression, FunctionCallNode)
    assert isinstance(final_expression.name, MemberAccessNode)
    assert final_expression.name.member == "max_element"
    assert isinstance(final_expression.name.object, BinaryOpNode)
    assert final_expression.args == []


def test_rust_gpu_reduce_shader_inferred_cast_target_parse():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/shaders/reduce/src/lib.rs subgroup_add EXECUTION const.
    code = """
    use spirv_std::memory::Scope;

    pub unsafe fn subgroup_add(value: u32) -> u32 {
        const EXECUTION: u32 = Scope::Subgroup as _;
        let mut result = 0;
        result
    }
    """

    ast = parse_code(code)
    function = ast.functions[0]
    execution = function.body[0]

    assert isinstance(execution, ConstNode)
    assert execution.name == "EXECUTION"
    assert execution.vtype == "u32"
    assert isinstance(execution.value, CastNode)
    assert execution.value.target_type == "_"
    assert execution.value.expression == "Scope::Subgroup"


def test_rust_gpu_builder_mut_self_receiver_parse():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # crates/spirv-builder/src/lib.rs builder methods.
    code = """
    impl SpirvBuilder {
        pub fn deny_warnings(mut self, v: bool) -> Self {
            self.deny_warnings = v;
            self
        }
    }
    """

    ast = parse_code(code)
    method = ast.impl_blocks[0].methods[0]

    assert method.name == "deny_warnings"
    assert method.params[0].name == "self"
    assert method.params[0].vtype == "Self"
    assert method.params[0].is_mutable is True
    assert method.params[1].name == "v"
    assert method.params[1].vtype == "bool"
    assert isinstance(method.body[0], AssignmentNode)
    assert method.body[-1] == "self"


def test_rust_gpu_struct_literal_field_attributes_parse():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # tests/difftests/tests/lib/src/scaffold/compute/wgpu.rs init_with_features.
    code = """
    fn init() {
        let descriptor = wgpu::InstanceDescriptor {
            #[cfg(target_os = "linux")]
            backends: wgpu::Backends::VULKAN,
            #[cfg(not(target_os = "linux"))]
            backends: wgpu::Backends::PRIMARY,
            flags: Default::default(),
        };
    }
    """

    ast = parse_code(code)
    descriptor = ast.functions[0].body[0]

    assert isinstance(descriptor.value, StructInitializationNode)
    assert descriptor.value.struct_name == "wgpu::InstanceDescriptor"
    assert [field_name for field_name, _ in descriptor.value.fields] == [
        "backends",
        "backends",
        "flags",
    ]


def test_rust_gpu_ash_runner_nested_block_statement_parse():
    # Reduced from https://github.com/Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/runners/ash/src/graphics.rs MyRenderer::render_frame.
    code = """
    impl MyRenderer {
        pub fn render_frame(&mut self, frame: DrawFrame) -> anyhow::Result<()> {
            unsafe {
                let device = &self.device;
                {
                    device.begin_command_buffer(cmd)?;
                    device.end_command_buffer(cmd)?;
                }
                device.queue_submit2(device.main_queue, &[], frame.draw_finished_fence)?;
                Ok(())
            }
        }
    }
    """

    ast = parse_code(code)
    unsafe_block = ast.impl_blocks[0].methods[0].body[0]

    assert isinstance(unsafe_block, UnsafeBlockNode)
    block = unsafe_block.block
    assert isinstance(block.statements[0], LetNode)
    assert isinstance(block.statements[1], BlockNode)
    assert [type(stmt) for stmt in block.statements[1].statements] == [
        TryNode,
        TryNode,
    ]
    assert isinstance(block.statements[2], TryNode)
    assert block.statements[2].expression.name.member == "queue_submit2"
    assert isinstance(block.expression, FunctionCallNode)
    assert block.expression.name == "Ok"


def test_if_let_dereference_conditions_stop_at_block_from_naga():
    # Reduced from gfx-rs/naga commit
    # d0f28c0b1a3c772e55e68db1c47eff5131cb6732,
    # src/front/interpolator.rs and src/proc/index.rs.
    code = """
    impl crate::Binding {
        fn apply_default_interpolation(&mut self, ty: &crate::TypeInner) {
            if let crate::Binding::Location {
                location: _,
                interpolation: ref mut interpolation @ None,
                ref mut sampling,
                second_blend_source: _,
            } = *self {
                match ty.scalar_kind() {
                    Some(crate::ScalarKind::Float) => {
                        *interpolation = Some(crate::Interpolation::Perspective);
                    }
                    Some(crate::ScalarKind::Sint | crate::ScalarKind::Uint) => {
                        *sampling = None;
                    }
                    Some(_) | None => {}
                }
            }
        }
    }

    impl GuardedIndex {
        fn try_resolve_to_constant(&mut self, module: &crate::Module) {
            if let GuardedIndex::Expression(expr) = *self {
                if let Ok(value) = module.to_ctx().eval_expr_to_u32_from(expr) {
                    *self = GuardedIndex::Known(value);
                }
            }
        }
    }
    """

    ast = parse_code(code)
    binding_if = ast.impl_blocks[0].methods[0].body[0]
    guarded_if = ast.impl_blocks[1].methods[0].body[0]

    assert isinstance(binding_if, IfNode)
    assert isinstance(binding_if.condition, LetPatternConditionNode)
    assert type(binding_if.condition.expression).__name__ == "DereferenceNode"
    assert isinstance(binding_if.if_body[0], MatchNode)

    assert isinstance(guarded_if, IfNode)
    assert isinstance(guarded_if.condition, LetPatternConditionNode)
    assert type(guarded_if.condition.expression).__name__ == "DereferenceNode"
    assert isinstance(guarded_if.if_body[0], IfNode)


def test_error_handling():
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
