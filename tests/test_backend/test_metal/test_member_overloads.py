import shutil
import subprocess

import pytest

from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalStaticAssertionError,
    MetalStructMethodError,
)
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

MUTABLE_ELEMS = "Tile_float__elems__metal_receiver_mutable_thread_unqualified"
CONST_ELEMS = "Tile_float__elems__metal_receiver_const_thread_unqualified"


def _target_artifacts(source):
    metal_ast = MetalParser(MetalLexer(source).tokenize()).parse()
    crossgl = MetalToCrossGLConverter().generate(metal_ast)
    target_ast = CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()
    return HLSLCodeGen().generate(target_ast), GLSLCodeGen().generate(target_ast)


def _crossgl_artifact(source):
    metal_ast = MetalParser(MetalLexer(source).tokenize()).parse()
    return MetalToCrossGLConverter().generate(metal_ast)


def _run(command):
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "method_declarations",
    [
        """
        thread T* elems() { return values; }
        const thread T* elems() const { return values; }
        """,
        """
        const thread T* elems() const { return values; }
        thread T* elems() { return values; }
        """,
    ],
    ids=("mutable-first", "const-first"),
)
def test_preprocessor_selects_pointer_member_overload_from_nested_receiver_cv(
    method_declarations,
):
    source = """
    template <typename T>
    struct Tile {
      T values[4];
      METHOD_DECLARATIONS
    };

    struct Consumer {
      Tile<float> tile;

      void mutate() { tile.elems()[0] = 7.0f; }
      float inspect() const { return tile.elems()[0]; }
    };

    Consumer consumer;
    """.replace("METHOD_DECLARATIONS", method_declarations)

    output = MetalPreprocessor().preprocess(source)

    assert f"{MUTABLE_ELEMS}(self.tile)[0] = 7.0f" in output
    assert f"return {CONST_ELEMS}(self.tile)[0]" in output
    assert f"thread float* {MUTABLE_ELEMS}(thread Tile_float& self)" in output
    assert f"const thread float* {CONST_ELEMS}(thread const Tile_float& self)" in output


def test_preprocessor_rejects_write_through_const_member_pointer_result():
    source = """
    struct Tile {
      float values[4];
      thread float* elems() { return values; }
      const thread float* elems() const { return values; }
    };

    void mutate(thread const Tile& tile) {
      tile.elems()[0] = 7.0f;
    }
    """

    with pytest.raises(MetalStructMethodError) as exc_info:
        MetalPreprocessor().preprocess(source)

    error = exc_info.value
    assert error.reason == "readonly-member-pointer-write"
    assert error.receiver_type == "const thread Tile&"
    assert error.return_type == "const thread float*"
    assert error.source_location["line"] == 9
    assert error.candidate_signatures == ("const thread float* elems() const",)


def test_preprocessor_preserves_receiver_address_space_and_reference_contracts():
    source = """
    struct Box {
      int value;
      int get() & thread { return value; }
      int get() & threadgroup { return value; }
      int get() & device { return value; }
      int get() const & constant { return value; }
    };

    int from_thread(thread Box& box) { return box.get(); }
    int from_named_rvalue(thread Box&& box) { return box.get(); }
    int from_threadgroup(threadgroup Box& box) { return box.get(); }
    int from_device(device Box& box) { return box.get(); }
    int from_constant(constant const Box& box) { return box.get(); }
    """

    output = MetalPreprocessor().preprocess(source)

    expected = {
        "from_thread": "Box__get__metal_receiver_mutable_thread_lvalue(box)",
        "from_named_rvalue": "Box__get__metal_receiver_mutable_thread_lvalue(box)",
        "from_threadgroup": "Box__get__metal_receiver_mutable_threadgroup_lvalue(box)",
        "from_device": "Box__get__metal_receiver_mutable_device_lvalue(box)",
        "from_constant": "Box__get__metal_receiver_const_constant_lvalue(box)",
    }
    for function_name, call in expected.items():
        body = output.split(f"int {function_name}", 1)[1].split("}", 1)[0]
        assert call in body

    assert "thread Box& self" in output
    assert "threadgroup Box& self" in output
    assert "device Box& self" in output
    assert "constant const Box& self" in output


def test_receiver_declaration_index_scans_equal_source_snapshot_once(monkeypatch):
    preprocessor = MetalPreprocessor()
    scan_count = 0
    original_scan = preprocessor._scan_receiver_declarations

    def counted_scan(code):
        nonlocal scan_count
        scan_count += 1
        return original_scan(code)

    monkeypatch.setattr(preprocessor, "_scan_receiver_declarations", counted_scan)
    source = """
    struct Box { int get(); };

    int read(thread Box& mutable_box, const device Box* readonly_box) {
      const char* ignored = "constant Box& mutable_box;";
      return mutable_box.get() + readonly_box->get();
    }
    """
    equal_source = "".join((source[: len(source) // 2], source[len(source) // 2 :]))
    mutable_call = source.index("mutable_box.get")
    readonly_call = source.index("readonly_box->get")

    mutable = preprocessor._receiver_contract_for_named_value(
        source,
        "Box",
        "mutable_box",
        mutable_call,
    )
    readonly = preprocessor._receiver_contract_for_named_value(
        source,
        "Box",
        "readonly_box",
        readonly_call,
    )
    repeated = preprocessor._receiver_contract_for_named_value(
        equal_source,
        "Box",
        "mutable_box",
        mutable_call,
    )

    assert mutable is not None
    assert mutable.is_const is False
    assert mutable.indirection == "&"
    assert mutable.address_spaces == ("thread",)
    assert readonly is not None
    assert readonly.is_const is True
    assert readonly.indirection == "*"
    assert readonly.address_spaces == ("device",)
    assert repeated == mutable
    assert scan_count == 1


def test_receiver_declaration_index_preserves_alias_and_lexical_shadowing():
    preprocessor = MetalPreprocessor()
    source = """
    struct Box { int get(); };
    using BoxAlias = Box;

    int read(thread BoxAlias& value) {
      int outer = value.get();
      {
        constant const BoxAlias& value = global_box;
        int inner = value.get();
      }
      {
        threadgroup volatile BoxAlias& value = shared_box;
        int sibling = value.get();
      }
      return outer + value.get();
    }
    """
    aliases = preprocessor._collect_struct_type_aliases(source, {"Box"}, [])
    outer_call = source.index("value.get")
    inner_call = source.index("value.get", outer_call + 1)
    sibling_call = source.index("value.get", inner_call + 1)
    restored_call = source.index("value.get", sibling_call + 1)

    contracts = [
        preprocessor._receiver_contract_for_named_value(
            source,
            "Box",
            "value",
            call_start,
            type_aliases=aliases,
        )
        for call_start in (outer_call, inner_call, sibling_call, restored_call)
    ]

    assert all(contract is not None for contract in contracts)
    outer, inner, sibling, restored = contracts
    assert outer is not None
    assert inner is not None
    assert sibling is not None
    assert restored is not None
    assert outer.is_const is False
    assert outer.indirection == "&"
    assert outer.address_spaces == ("thread",)
    assert inner.is_const is True
    assert inner.indirection == "&"
    assert inner.address_spaces == ("constant",)
    assert sibling.is_const is False
    assert sibling.is_volatile is True
    assert sibling.indirection == "&"
    assert sibling.address_spaces == ("threadgroup",)
    assert restored == outer

    lexical_scopes = preprocessor._find_lexical_brace_scopes(source)
    for declaration in preprocessor._find_receiver_declarations(source)["value"]:
        expected_scope = preprocessor._innermost_lexical_scope(
            lexical_scopes,
            declaration.declaration_position,
            len(source),
        )
        assert (declaration.scope_start, declaration.scope_end) == expected_scope


def test_preprocessor_preserves_contextual_receiver_qualifiers_in_method_body():
    source = """
    struct Transform {
      int apply(int value) & threadgroup { return value; }
      int apply(int value) const & threadgroup { return value + 1; }
    };

    struct Runner {
      int run(
          threadgroup const Transform& operation,
          int value) const {
        return operation.apply(value);
      }
    };
    """

    output = MetalPreprocessor().preprocess(source)

    helper = "Transform__apply__metal_receiver_const_threadgroup_lvalue"
    assert f"return {helper}(operation, value);" in output
    assert "operation.apply" not in output
    assert "threadgroup const Transform& operation" in output

    repeated = MetalPreprocessor().preprocess(output)
    assert f"return {helper}(operation, value);" in repeated
    assert "operation.apply" not in repeated


def test_preprocessor_lowers_static_member_called_through_contextual_receiver():
    source = """
    struct Transform {
      static float apply(float value) { return value; }
    };

    struct Runner {
      float run(thread const Transform& operation, float value) const {
        return operation.apply(value);
      }
    };
    """

    output = MetalPreprocessor().preprocess(source)

    assert "return Transform__apply(value);" in output
    assert "Transform__apply(operation, value)" not in output
    assert "operation.apply" not in output

    repeated = MetalPreprocessor().preprocess(output)
    assert "return Transform__apply(value);" in repeated
    assert "operation.apply" not in repeated


def test_preprocessor_rejects_unknown_contextual_receiver_type():
    source = """
    struct Runner {
      int run(thread const MissingTransform& operation, int value) const {
        return operation.apply(value);
      }
    };
    """

    with pytest.raises(MetalStructMethodError) as exc_info:
        MetalPreprocessor().preprocess(source)

    error = exc_info.value
    assert error.reason == "concrete-receiver-type-unresolved"
    assert error.receiver_type == "thread const MissingTransform&"
    assert error.requested_signature == "operation.apply"


def test_preprocessor_rejects_ambiguous_contextual_receiver_overload():
    source = """
    struct Transform {
      int apply(int value) const { return value; }
      int apply(int value) volatile { return value + 1; }
    };

    struct Runner {
      int run(thread Transform& operation, int value) const {
        return operation.apply(value);
      }
    };
    """

    with pytest.raises(MetalStructMethodError) as exc_info:
        MetalPreprocessor().preprocess(source)

    error = exc_info.value
    assert error.reason == "concrete-overload-ambiguous"
    assert error.receiver_type == "thread Transform&"
    assert error.requested_signature == "Transform::apply(value)"


@pytest.mark.parametrize(
    "reverse_functions", [False, True], ids=("half-first", "float-first")
)
def test_preprocessor_resolves_line_wrapped_qualified_aliases_per_function(
    reverse_functions,
):
    half_function = """
    void consume_half(uint lane) {
      threadgroup half left[4];
      threadgroup half right[4];
      using block_t = HalfBlock;
      block_t block(lane);
      block.mma(left, right);
    }
    """
    float_function = """
    void consume_float(uint lane) {
      threadgroup float left[4];
      threadgroup float right[4];
      using block_t = kernels::
          FloatBlock;
      block_t block(lane);
      block.mma(left, right);
    }
    """
    functions = [half_function, float_function]
    if reverse_functions:
        functions.reverse()
    source = """
    namespace kernels {
    struct FloatBlock {
      FloatBlock(uint lane) { (void)lane; }
      void mma(const threadgroup float* left, const threadgroup float* right) {
        (void)left;
        (void)right;
      }
    };
    }

    struct HalfBlock {
      HalfBlock(uint lane) { (void)lane; }
      void mma(const threadgroup half* left, const threadgroup half* right) {
        (void)left;
        (void)right;
      }
    };
    """ + "\n".join(functions)

    output = MetalPreprocessor().preprocess(source)

    half_body = output.split("void consume_half", 1)[1].split("}", 1)[0]
    float_body = output.split("void consume_float", 1)[1].split("}", 1)[0]
    assert "HalfBlock__mma(block, left, right)" in half_body
    assert "FloatBlock__mma(block, left, right)" in float_body
    assert "FloatBlock__mma(block, left, right)" not in half_body
    assert "HalfBlock__mma(block, left, right)" not in float_body
    assert (
        "void HalfBlock__mma(thread HalfBlock& self, "
        "const threadgroup half* left, const threadgroup half* right)" in output
    )
    assert (
        "void FloatBlock__mma(thread FloatBlock& self, "
        "const threadgroup float* left, const threadgroup float* right)" in output
    )


def test_receiver_contract_does_not_borrow_shadowed_alias_declaration():
    preprocessor = MetalPreprocessor()
    source = """
    struct HalfBlock { void mma() {} };
    using block_t = HalfBlock;

    void consume() {
      block_t block;
      block.mma();
      {
        using block_t = unresolved::Block;
        block_t block;
        block.mma();
      }
    }
    """
    aliases = preprocessor._collect_struct_type_aliases(
        source,
        {"HalfBlock"},
        [],
    )
    outer_call = source.index("block.mma")
    inner_call = source.index("block.mma", outer_call + 1)

    outer = preprocessor._receiver_contract_for_named_value(
        source,
        "HalfBlock",
        "block",
        outer_call,
        type_aliases=aliases,
    )
    inner = preprocessor._receiver_contract_for_named_value(
        source,
        "HalfBlock",
        "block",
        inner_call,
        type_aliases=aliases,
    )

    assert outer is not None
    assert inner is None

    with pytest.raises(MetalStructMethodError) as exc_info:
        MetalPreprocessor().preprocess(source)

    error = exc_info.value
    assert error.reason == "concrete-receiver-type-unresolved"
    assert error.receiver_type == "block_t"
    assert error.method_name == "mma"
    assert error.missing_capabilities == ("metal.member-overload-resolution",)


def test_member_call_suffix_does_not_cross_statement_comparisons():
    source = """
    void select(uint2 value, float low, float high) {
      float selected = value.x < 16 ? low : high;
      bool valid = value.y > (0);
    }
    """
    name_end = source.index("value.x") + len("value.x")

    preprocessor = MetalPreprocessor()
    suffix = preprocessor._member_template_call_suffix(source, name_end)
    output = preprocessor.preprocess(source)

    assert suffix is None
    assert "value.x < 16 ? low : high" in output
    assert "value.y > (0)" in output


def test_preprocessor_materializes_struct_constructor_before_member_lowering():
    source = """
    template <int Width, int Threads>
    struct BlockLoader {
      static constexpr int pack_factor = packed_width<8, 4>();
      int stride;

      BlockLoader(int source_stride)
          : stride(
                Width < Threads ? Width * packed_width<8, 4>()
                                : source_stride * packed_width<8, 4>()) {}

      void load_safe(int limit) const {
        if (limit < Width) {
          return;
        }
      }
    };

    void consume(int source_stride, int limit) {
      using loader_w_t = BlockLoader<32, (2 * 2 * 32)>;
      loader_w_t loader_w(source_stride);
      loader_w.load_safe(limit);
    }
    """

    output = MetalPreprocessor().preprocess(source)

    concrete_name = "BlockLoader_32_2_2_32"
    concrete_body = output.split(f"struct {concrete_name} {{", 1)[1].split("};", 1)[0]
    assert f"{concrete_name}(int source_stride)" in concrete_body
    assert "BlockLoader(int source_stride)" not in concrete_body
    assert (
        f"void {concrete_name}__load_safe(thread const {concrete_name}& self, "
        "int limit)" in output
    )
    assert f"{concrete_name}__load_safe(loader_w, limit)" in output
    assert "loader_w.load_safe" not in output


def test_preprocessor_materializes_line_wrapped_qualified_struct_receiver():
    source = """
    namespace compute {
    template <typename T, int Width>
    struct Tile {
      int bias;

      Tile(int value) : bias(value) {}

      void accumulate(threadgroup T* values) {
        values[Width] += T(bias);
      }
    };
    }

    void run(threadgroup float* values) {
      using tile_t = compute::
          Tile<float, 4>;
      tile_t tile(2);
      tile.accumulate(values);
    }
    """

    output = MetalPreprocessor().preprocess(source)

    concrete_name = "Tile_float_4"
    assert f"using tile_t = {concrete_name};" in output
    assert f"{concrete_name}__accumulate(tile, values)" in output
    assert (
        f"void {concrete_name}__accumulate(thread {concrete_name}& self, "
        "threadgroup float* values)" in output
    )
    assert "tile.accumulate" not in output
    assert "compute::\n          Tile_float_4" not in output


def test_preprocessor_rejects_line_wrapped_mismatched_struct_namespace():
    source = """
    namespace compute {
    template <typename T, int Width>
    struct Tile {
      void accumulate(threadgroup T* values) { values[Width] += T(1); }
    };
    }

    void run(threadgroup float* values) {
      using valid_t = compute::Tile<float, 4>;
      valid_t valid;
      valid.accumulate(values);

      using invalid_t = unrelated::
          Tile<float, 4>;
      invalid_t invalid;
      invalid.accumulate(values);
    }
    """

    with pytest.raises(MetalStructMethodError) as exc_info:
        MetalPreprocessor().preprocess(source)

    error = exc_info.value
    assert error.reason == "concrete-receiver-type-unresolved"
    assert error.receiver_type == "invalid_t"
    assert error.requested_signature == "invalid.accumulate"


def test_materialized_struct_assertion_cannot_use_later_static_member():
    source = """
    template <int Value>
    struct Limits {
      static_assert(reads <= 16, "Unsupported read count.");
      static constant constexpr const short reads = Value;
      int value;
    };

    Limits<8> limits;
    """

    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_explicit_template_struct_instantiations(
        source
    )
    assert "struct Limits_8" in materialized

    with pytest.raises(MetalStaticAssertionError) as exc_info:
        preprocessor._evaluate_static_assertions(materialized)

    assert exc_info.value.reason == "condition-unresolved"
    assert exc_info.value.unresolved_dependencies == ("reads",)


def test_preprocessor_receiver_helper_identity_ignores_parameter_names():
    source = """
    struct Box {
      int pick(int mutable_value) { return mutable_value; }
      int pick(int readonly_value) const { return readonly_value + 1; }
    };

    int read(thread Box& mutable_box, thread const Box& const_box) {
      return mutable_box.pick(1) + const_box.pick(2);
    }
    """

    output = MetalPreprocessor().preprocess(source)

    mutable_helper = "Box__pick__metal_receiver_mutable_thread_unqualified"
    const_helper = "Box__pick__metal_receiver_const_thread_unqualified"
    assert f"{mutable_helper}(mutable_box, 1)" in output
    assert f"{const_helper}(const_box, 2)" in output
    assert f"int {mutable_helper}(thread Box& self, int mutable_value)" in output
    assert f"int {const_helper}(thread const Box& self, int readonly_value)" in output


def test_preprocessor_mutable_nested_member_retains_parent_volatile_state():
    source = """
    struct Inner {
      int get() { return 0; }
      int get() volatile { return 1; }
    };

    struct Outer {
      mutable Inner inner;
      int read() const volatile { return inner.get(); }
    };
    """

    output = MetalPreprocessor().preprocess(source)

    volatile_helper = "Inner__get__metal_receiver_volatile_thread_unqualified"
    outer_body = output.split("int Outer__read", 1)[1].split("}", 1)[0]
    assert f"return {volatile_helper}(self.inner)" in outer_body


def test_preprocessor_reports_equally_preferred_receiver_candidates():
    source = """
    struct Box {
      int get() const { return 1; }
      int get() volatile { return 2; }
    };

    int read(thread Box& box) { return box.get(); }
    """

    with pytest.raises(MetalStructMethodError) as exc_info:
        MetalPreprocessor().preprocess(source)

    error = exc_info.value
    assert error.reason == "concrete-overload-ambiguous"
    assert error.receiver_type == "thread Box&"
    assert error.source_location["line"] == 7
    assert error.missing_capabilities == ("metal.member-overload-resolution",)
    assert {record["candidate"] for record in error.candidate_mismatches} == {
        "int get() const",
        "int get() volatile",
    }
    assert all(record["viable"] for record in error.candidate_mismatches)
    assert all(not record["mismatches"] for record in error.candidate_mismatches)


def test_preprocessor_fails_closed_without_a_provable_receiver_contract():
    preprocessor = MetalPreprocessor()
    struct = preprocessor._find_concrete_struct_definitions("""
        struct Box {
          int get() { return 1; }
          int get() const { return 2; }
        };
        """)[0]
    call = "box.get()"

    with pytest.raises(MetalStructMethodError) as exc_info:
        preprocessor._select_concrete_method_for_call(
            call,
            struct,
            "get",
            call.index("("),
            {},
            {},
            {},
            {"Box": struct},
            allow_template_fallback=False,
            require_static=False,
            call_start=0,
        )

    error = exc_info.value
    assert error.reason == "concrete-overload-no-viable"
    assert error.receiver_type == "<unknown>"
    assert error.source_location["line"] == 1
    assert error.candidate_signatures == ("int get()", "int get() const")
    assert all(not record["viable"] for record in error.candidate_mismatches)
    assert all(
        record["mismatches"]
        == ("receiver cv/reference/address-space state is unavailable",)
        for record in error.candidate_mismatches
    )


def test_parser_retains_return_and_trailing_member_qualifiers():
    source = """
    const thread float* Box::elems() const volatile & thread noexcept(true) {
      return nullptr;
    }
    """
    tokens = MetalLexer(source, preprocess=False).tokenize()
    function = MetalParser(tokens).parse().functions[0]

    assert function.return_type == "float*"
    assert function.return_qualifiers == ["const", "thread"]
    assert function.declaration_qualifiers == ["const", "thread"]
    assert function.method_qualifiers == [
        "const",
        "volatile",
        "&",
        "thread",
        "noexcept",
        "true",
    ]
    assert function.receiver_qualifiers == function.method_qualifiers


def test_codegen_retains_selected_pointer_return_mutability():
    source = """
    template <typename T>
    struct Tile {
      T values[4];
      thread T* elems() { return values; }
      const thread T* elems() const { return values; }
    };

    Tile<float> tile;
    """

    crossgl = _crossgl_artifact(source)

    assert f"RWStructuredBuffer<float> {MUTABLE_ELEMS}(" in crossgl
    assert f"StructuredBuffer<float> {CONST_ELEMS}(" in crossgl


def test_receiver_cv_selection_reaches_validated_target_artifacts(tmp_path):
    source = """
    struct Tile {
      int value;
      int pick() { return value; }
      int pick() const { return value + 1; }
    };

    kernel void select_overload(device int* out [[buffer(0)]]) {
      Tile mutable_tile;
      mutable_tile.value = 2;
      const Tile const_tile = mutable_tile;
      out[0] = mutable_tile.pick();
      out[1] = const_tile.pick();
    }
    """

    hlsl, glsl = _target_artifacts(source)
    mutable_helper = "Tile__pick__metal_receiver_mutable_thread_unqualified"
    const_helper = "Tile__pick__metal_receiver_const_thread_unqualified"
    assert f"{mutable_helper}(mutable_tile)" in hlsl
    assert f"{const_helper}(const_tile)" in hlsl
    assert f"{mutable_helper.replace('__', '_')}(mutable_tile)" in glsl
    assert f"{const_helper.replace('__', '_')}(const_tile)" in glsl

    hlsl_path = tmp_path / "member-cv-overload.hlsl"
    hlsl_path.write_text(hlsl, encoding="utf-8")
    dxc = shutil.which("dxc")
    if dxc is not None:
        _run(
            [
                dxc,
                "-T",
                "cs_6_0",
                "-E",
                "CSMain",
                str(hlsl_path),
                "-Fo",
                str(tmp_path / "member-cv-overload.dxil"),
            ]
        )

    glslang = shutil.which("glslangValidator")
    if glslang is None:
        return
    hlsl_spirv = tmp_path / "member-cv-overload-hlsl.spv"
    _run(
        [
            glslang,
            "-D",
            "-V",
            "-S",
            "comp",
            "-e",
            "CSMain",
            str(hlsl_path),
            "-o",
            str(hlsl_spirv),
        ]
    )

    glsl_path = tmp_path / "member-cv-overload.comp"
    glsl_spirv = tmp_path / "member-cv-overload-glsl.spv"
    glsl_path.write_text(glsl, encoding="utf-8")
    _run(
        [
            glslang,
            "--target-env",
            "opengl",
            "-S",
            "comp",
            str(glsl_path),
            "-o",
            str(glsl_spirv),
        ]
    )

    spirv_val = shutil.which("spirv-val")
    if spirv_val is not None:
        _run([spirv_val, str(hlsl_spirv)])
        _run([spirv_val, "--target-env", "opengl4.5", str(glsl_spirv)])
