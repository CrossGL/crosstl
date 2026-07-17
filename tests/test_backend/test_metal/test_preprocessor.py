import re

import pytest

from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.preprocessor import (
    CONTAINING_SPAN_CACHE_LIMIT,
    SOURCE_ANALYSIS_CACHE_LIMIT,
    MetalMacroExpansionError,
    MetalPreprocessor,
    MetalStatelessGlobalElisionError,
    MetalStaticAssertionError,
    MetalStructMethodError,
    MetalTemplateSpecializationError,
)


def token_values(code, **lexer_options):
    return [value for _, value in MetalLexer(code, **lexer_options).tokenize()]


def test_containing_span_preserves_sorted_lookup_boundaries():
    preprocessor = MetalPreprocessor()
    spans = [(2, 5), (8, 13), (20, 24)]

    assert preprocessor._containing_span(1, spans) is None
    assert preprocessor._containing_span(2, spans) == (2, 5)
    assert preprocessor._containing_span(4, spans) == (2, 5)
    assert preprocessor._containing_span(5, spans) is None
    assert preprocessor._containing_span(12, spans) == (8, 13)


def test_containing_span_preserves_unsorted_overlap_order():
    preprocessor = MetalPreprocessor()
    spans = [(10, 20), (0, 15), (30, 40)]

    assert preprocessor._containing_span(12, spans) == (10, 20)
    assert preprocessor._containing_span(5, spans) == (0, 15)


def test_containing_span_caches_multiple_span_lists_without_thrashing():
    # The template-materialization scan alternates lookups between two different
    # span lists (template-declaration spans and reachable-function spans) at
    # nearly every source position. A single cache slot thrashed between them and
    # rebuilt its sortedness decision on every call, making the scan quadratic on
    # the large expanded MLX kernels. The cache now keeps one entry per span list
    # keyed by identity, so alternating lookups stay correct AND both lists remain
    # cached at the same time.
    preprocessor = MetalPreprocessor()
    declarations = [(0, 5), (10, 15)]
    reachable = [(2, 8), (20, 25)]

    assert preprocessor._containing_span(3, declarations) == (0, 5)
    assert preprocessor._containing_span(4, reachable) == (2, 8)
    assert preprocessor._containing_span(12, declarations) == (10, 15)
    assert preprocessor._containing_span(22, reachable) == (20, 25)
    assert preprocessor._containing_span(7, declarations) is None
    assert preprocessor._containing_span(9, reachable) is None

    # A single-slot cache could only retain the most recent list; the per-list
    # cache retains both so the alternating lookups never rebuild.
    assert len(preprocessor._containing_span_cache) == 2


def test_containing_span_cache_retention_is_bounded():
    preprocessor = MetalPreprocessor()
    span_lists = [
        [(index * 3, index * 3 + 2)] for index in range(CONTAINING_SPAN_CACHE_LIMIT + 5)
    ]

    for spans in span_lists:
        assert preprocessor._containing_span(spans[0][0], spans) == spans[0]

    assert len(preprocessor._containing_span_cache) == CONTAINING_SPAN_CACHE_LIMIT
    assert id(span_lists[0]) not in preprocessor._containing_span_cache
    assert id(span_lists[-1]) in preprocessor._containing_span_cache


def test_source_analysis_cache_reuses_equal_snapshots_and_is_bounded(monkeypatch):
    preprocessor = MetalPreprocessor()
    scan_names = (
        "_scan_comment_and_literal_spans",
        "_scan_lexical_brace_scopes",
        "_scan_template_declaration_spans",
        "_scan_namespace_spans",
    )
    scan_counts = {name: 0 for name in scan_names}

    for scan_name in scan_names:
        original = getattr(preprocessor, scan_name)

        def counted(code, *, _name=scan_name, _original=original):
            scan_counts[_name] += 1
            return _original(code)

        monkeypatch.setattr(preprocessor, scan_name, counted)

    source = """
    namespace example {
    template <typename T>
    T identity(T value) {
        const char* brace = "}";
        return value;
    }
    }
    """
    equal_source = "".join((source[: len(source) // 2], source[len(source) // 2 :]))
    assert equal_source == source
    assert equal_source is not source

    finders = (
        preprocessor._find_comment_and_literal_spans,
        preprocessor._find_lexical_brace_scopes,
        preprocessor._find_template_declaration_spans,
        preprocessor._find_namespace_spans,
    )
    for finder in finders:
        first = finder(source)
        assert finder(equal_source) is first

    assert scan_counts == {name: 1 for name in scan_names}

    for index in range(SOURCE_ANALYSIS_CACHE_LIMIT + 2):
        preprocessor._find_namespace_spans(
            f"namespace generated_{index} {{ int value_{index}; }}"
        )

    assert len(preprocessor._source_analysis_cache) == SOURCE_ANALYSIS_CACHE_LIMIT
    assert source not in preprocessor._source_analysis_cache


def test_alias_free_struct_families_skip_detailed_scope_scans(monkeypatch):
    preprocessor = MetalPreprocessor()
    scan_counts = {"comments": 0, "scopes": 0}
    original_comment_scan = preprocessor._scan_comment_and_literal_spans
    original_scope_scan = preprocessor._scan_lexical_brace_scopes

    def counted_comment_scan(code):
        scan_counts["comments"] += 1
        return original_comment_scan(code)

    def counted_scope_scan(code):
        scan_counts["scopes"] += 1
        return original_scope_scan(code)

    monkeypatch.setattr(
        preprocessor,
        "_scan_comment_and_literal_spans",
        counted_comment_scan,
    )
    monkeypatch.setattr(
        preprocessor,
        "_scan_lexical_brace_scopes",
        counted_scope_scan,
    )
    source = "\n".join(
        [
            *(f"struct Plain{index} {{ float value; }};" for index in range(256)),
            "struct WithAlias { using scalar_t = float; scalar_t value; };",
        ]
    )

    definitions = preprocessor._find_concrete_struct_definitions(source)

    assert len(definitions) == 257
    with_alias = next(item for item in definitions if item.name == "WithAlias")
    assert with_alias.type_aliases == {"scalar_t": "float"}
    assert scan_counts == {"comments": 1, "scopes": 1}


def test_preprocessor_conditional_expansion():
    code = """
    #define ENABLED 1
    #if ENABLED
    int only_enabled = 1;
    #else
    int only_disabled = ;
    #endif
    """

    values = token_values(code)

    assert "only_enabled" in values
    assert "only_disabled" not in values


def test_preprocessor_function_like_macro_expansion():
    code = """
    #define MAKE_FLOAT2(a, b) float2((a), (b))
    float2 value = MAKE_FLOAT2(1.0, 2.0);
    """

    values = token_values(code)

    assert "MAKE_FLOAT2" not in values
    assert "float2" in values
    assert "1.0" in values
    assert "2.0" in values


def test_preprocessor_materializes_mlx_instantiate_kernel_macros():
    code = """
    #define instantiate_arange(name, itype, offset) \\
        instantiate_kernel("arange" #name, arange, itype, offset)

    template <typename T, int Offset>
    [[kernel]] void arange(
        device T* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        out[gid] = T(gid + Offset);
    }

    instantiate_arange(float32, float, 7)
    instantiate_arange(float16, half, 3)
    """

    output = MetalPreprocessor().preprocess(code)

    assert "template <typename T" not in output
    assert "instantiate_kernel" not in output
    assert '[[host_name("arangefloat32")]]' in output
    assert "void arangefloat32(" in output
    assert "device float* out" in output
    assert "out[gid] = float(gid + 7);" in output
    assert '[[host_name("arangefloat16")]]' in output
    assert "void arangefloat16(" in output
    assert "device half* out" in output
    assert "out[gid] = half(gid + 3);" in output


def test_preprocessor_materializes_multiline_project_instantiation_macro():
    code = """
    #define instantiate_copy(name, type) \\
        instantiate_kernel("copy_" #name, copy, type)

    template <typename T>
    [[kernel]] void copy(
        device const T* src [[buffer(0)]],
        device T* dst [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
        dst[gid] = src[gid];
    }

    instantiate_copy(
        float32,
        float)
    """

    output = MetalPreprocessor().preprocess(code)

    assert "instantiate_copy" not in output
    assert "instantiate_kernel" not in output
    assert "template <typename T>" not in output
    assert '[[host_name("copy_float32")]]' in output
    assert "void copy_float32(" in output
    assert "device const float* src" in output
    assert "device float* dst" in output


def test_preprocessor_materializes_mlx_instantiations_with_template_defaults():
    code = """
    #define instantiate_rope(name, type, idx_type) \\
        instantiate_kernel("rope_" #name, rope, type, idx_type)

    template <typename T, typename IdxT, int N = 4>
    [[kernel]] void rope(
        device const T* src [[buffer(0)]],
        device T* dst [[buffer(1)]],
        device const IdxT* positions [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
        IdxT pos = positions[gid];
        dst[gid] = src[pos] + T(N);
    }

    instantiate_rope(float32, float, uint)
    """

    output = MetalPreprocessor().preprocess(code)

    assert "template <typename T" not in output
    assert "instantiate_kernel" not in output
    assert '[[host_name("rope_float32")]]' in output
    assert "void rope_float32(" in output
    assert "device const float* src" in output
    assert "device const uint* positions" in output
    assert "uint pos = positions[gid];" in output
    assert "dst[gid] = src[pos] + float(4);" in output


def test_preprocessor_materializes_struct_template_trait_defaults():
    code = """
    template <typename U>
    struct DefaultAccT {
        using type = float;
    };

    template <>
    struct DefaultAccT<complex64_t> {
        using type = complex64_t;
    };

    template <
        typename T,
        const bool kDoAxpby, /* Do out = alpha * out + beta * bias */
        typename AccT = typename DefaultAccT<T>::type>
    struct Kernel {
        using acc_type = AccT;

        static void run(threadgroup AccT* shared) {
            thread AccT values[4];
            values[0] = AccT(0);
            shared[0] = values[0];
        }
    };

    Kernel<half, false> half_kernel;
    Kernel<complex64_t, true> complex_kernel;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "struct Kernel_half_false" in output
    assert "using acc_type = float;" in output
    assert "void Kernel_half_false__run(threadgroup float* shared)" in output
    assert "thread float values[4];" in output
    assert "struct Kernel_complex64_t_true" in output
    assert "using acc_type = complex64_t;" in output
    assert (
        "void Kernel_complex64_t_true__run(threadgroup complex64_t* shared)" in output
    )


def test_preprocessor_template_parameter_comments_preserve_declarations():
    parameter_text = """
        typename T,
        // The following type controls output.
        typename U /* descriptive prose ... */,
        typename V = float
    """

    parameters = MetalPreprocessor()._parse_template_parameter_list(parameter_text)

    assert [parameter.name for parameter in parameters] == ["T", "U", "V"]
    assert [parameter.is_variadic for parameter in parameters] == [False] * 3
    assert [parameter.default for parameter in parameters] == [None, None, "float"]


def test_preprocessor_resolves_namespace_qualified_template_traits():
    code = """
    namespace A {
    template <typename T>
    struct Trait { using type = float; };
    }

    namespace B {
    template <typename T>
    struct Trait { using type = int; };
    }
    """
    preprocessor = MetalPreprocessor()
    traits = preprocessor._find_template_type_traits(code)

    assert (
        preprocessor._resolve_template_type_trait(
            "typename A::Trait<half>::type", traits
        )
        == "float"
    )
    assert (
        preprocessor._resolve_template_type_trait(
            "typename B::Trait<half>::type", traits
        )
        == "int"
    )
    assert (
        preprocessor._resolve_template_type_trait(
            "typename Trait<half>::type", traits, "A"
        )
        == "float"
    )


def test_preprocessor_resolves_global_trait_in_commented_namespace_declaration():
    code = """
    namespace /* before name */ A /* before body */ {
    template <typename T>
    struct Trait { using type = float; };
    }
    """
    preprocessor = MetalPreprocessor()
    traits = preprocessor._find_template_type_traits(code)

    assert (
        preprocessor._resolve_template_type_trait(
            "typename ::A::Trait<half>::type", traits
        )
        == "float"
    )


def test_preprocessor_resolves_partial_template_trait_specialization():
    code = """
    template <typename T>
    struct Trait { using type = int; };

    template <typename T>
    struct Trait<T*> { using type = T; };
    """
    preprocessor = MetalPreprocessor()
    traits = preprocessor._find_template_type_traits(code)

    assert (
        preprocessor._resolve_template_type_trait(
            "typename Trait<float*>::type", traits
        )
        == "float"
    )
    assert (
        preprocessor._resolve_template_type_trait("typename Trait<float>::type", traits)
        == "int"
    )


def test_preprocessor_matches_boolean_trait_specialization_with_folded_literal():
    code = """
    template <bool Select, typename A, typename B>
    struct ConditionalType { using type = B; };

    template <typename A, typename B>
    struct ConditionalType<true, A, B> { using type = A; };
    """
    preprocessor = MetalPreprocessor()
    traits = preprocessor._find_template_type_traits(code)

    assert (
        preprocessor._resolve_template_type_trait(
            "typename ConditionalType<1, uint, uchar>::type",
            traits,
        )
        == "uint"
    )
    assert (
        preprocessor._resolve_template_type_trait(
            "typename ConditionalType<0, uint, uchar>::type",
            traits,
        )
        == "uchar"
    )


def test_preprocessor_prefers_more_specific_partial_template_trait_specialization():
    code = """
    template <typename T, typename U>
    struct Trait { using type = char; };

    template <typename T>
    struct Trait<T*, T*> { using type = float; };

    template <typename T, typename U>
    struct Trait<T*, U> { using type = int; };
    """
    preprocessor = MetalPreprocessor()
    traits = preprocessor._find_template_type_traits(code)

    assert (
        preprocessor._resolve_template_type_trait(
            "typename Trait<int*, int*>::type", traits
        )
        == "float"
    )


def test_preprocessor_resolves_relative_qualified_trait_in_enclosing_namespace():
    code = """
    namespace B {
    template <typename T>
    struct Trait { using type = int; };
    }

    namespace A {
    namespace B {
    template <typename T>
    struct Trait { using type = float; };
    }
    }
    """
    preprocessor = MetalPreprocessor()
    traits = preprocessor._find_template_type_traits(code)

    assert (
        preprocessor._resolve_template_type_trait(
            "typename B::Trait<half>::type", traits, "A"
        )
        == "float"
    )


def test_preprocessor_preserves_comment_separated_template_default_arguments():
    parameter_text = """
        typename T,
        typename U = Pair /* note */ <T, float>,
        typename V = int
    """
    preprocessor = MetalPreprocessor()

    parameters = preprocessor._parse_template_parameter_list(parameter_text)

    assert [parameter.name for parameter in parameters] == ["T", "U", "V"]
    assert parameters[1].default == "Pair /* note */ <T, float>"
    assert parameters[2].default == "int"

    value_parameters = preprocessor._parse_template_parameter_list(
        "int N, bool enabled = N < 8, typename Result = float"
    )
    assert [parameter.name for parameter in value_parameters] == [
        "N",
        "enabled",
        "Result",
    ]
    assert value_parameters[1].default == "N < 8"


def test_preprocessor_does_not_fall_back_past_variadic_partial_trait():
    code = """
    template <typename T>
    struct Trait { using type = int; };

    template <typename... Ts>
    struct Trait<Pack<Ts...>> { using type = float; };
    """
    preprocessor = MetalPreprocessor()
    traits = preprocessor._find_template_type_traits(code)

    assert (
        preprocessor._resolve_template_type_trait(
            "typename Trait<Pack<half, float>>::type", traits
        )
        is None
    )
    assert (
        preprocessor._resolve_template_type_trait("typename Trait<float>::type", traits)
        == "int"
    )


def test_preprocessor_materializes_multiple_mlx_instantiations_from_one_macro():
    code = """
    #define instantiate_copy_pair(name, type) \\
        instantiate_kernel("s_copy" #name, copy_s, type, type, 1) \\
        instantiate_kernel("v_copy" #name, copy_v, type, type, 1)

    template <typename T, typename U, int N>
    [[kernel]] void copy_s(
        device const T* src [[buffer(0)]],
        device U* dst [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
        dst[gid] = U(src[0] + N);
    }

    template <typename T, typename U, int N>
    [[kernel]] void copy_v(
        device const T* src [[buffer(0)]],
        device U* dst [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
        dst[gid] = U(src[gid] + N);
    }

    instantiate_copy_pair(float32, float)
    """

    output = MetalPreprocessor().preprocess(code)

    assert "instantiate_kernel" not in output
    assert '[[host_name("s_copyfloat32")]]' in output
    assert "void s_copyfloat32(" in output
    assert "device const float* src" in output
    assert "dst[gid] = float(src[0] + 1);" in output
    assert '[[host_name("v_copyfloat32")]]' in output
    assert "void v_copyfloat32(" in output
    assert "dst[gid] = float(src[gid] + 1);" in output


def test_preprocessor_materializes_mlx_decltype_instantiation_entries():
    code = """
    template <typename T>
    [[kernel]] void arange(
        device T* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        out[gid] = T(gid);
    }

    template [[host_name("arangefloat32")]] [[kernel]]
    decltype(arange<float>) arange<float>;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "decltype(arange<float>)" not in output
    assert '[[host_name("arangefloat32")]]' in output
    assert "void arangefloat32(" in output
    assert "device float* out" in output
    assert "out[gid] = float(gid);" in output


def test_preprocessor_materializes_decltype_instantiation_with_joined_host_name():
    code = """
    template <typename T>
    [[kernel]] void arange(
        device T* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        out[gid] = T(gid);
    }

    template [[host_name("arange" "uint32")]] [[kernel]]
    decltype(arange<uint>) arange<uint>;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "decltype(arange<uint>)" not in output
    assert '[[host_name("arangeuint32")]]' in output
    assert "void arangeuint32(" in output
    assert "device uint* out" in output
    assert "out[gid] = uint(gid);" in output


def test_preprocessor_materializes_signature_only_template_instantiation_entries():
    code = """
    template <int N>
    struct ConvParams {
        int shape[N];
    };

    template <typename T, int N>
    [[kernel]] void naive_unfold_Nd(
        const device T* in [[buffer(0)]],
        device T* out [[buffer(1)]],
        const constant ConvParams<N>* params [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
        out[gid] = in[gid] + T(N + params->shape[0]);
    }

    template <typename T, int N>
    [[kernel]] void naive_unfold_transpose_Nd(
        const device T* in [[buffer(0)]],
        device T* out [[buffer(1)]],
        const constant ConvParams<N>* params [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
        out[gid] = in[gid] - T(N + params->shape[0]);
    }

    #define instantiate_naive_unfold_nd(name, itype, n) \\
        template [[host_name("naive_unfold_nd_" #name "_" #n)]] [[kernel]] void \\
        naive_unfold_Nd( \\
            const device itype* in [[buffer(0)]], \\
            device itype* out [[buffer(1)]], \\
            const constant ConvParams<n>* params [[buffer(2)]], \\
            uint gid [[thread_position_in_grid]]); \\
        template [[host_name("naive_unfold_transpose_nd_" #name "_" #n)]] [[kernel]] void \\
        naive_unfold_transpose_Nd( \\
            const device itype* in [[buffer(0)]], \\
            device itype* out [[buffer(1)]], \\
            const constant ConvParams<n>* params [[buffer(2)]], \\
            uint gid [[thread_position_in_grid]]);

    instantiate_naive_unfold_nd(float32, float, 2) instantiate_naive_unfold_nd(float16, half, 3)
    """

    output = MetalPreprocessor().preprocess(code)

    assert '[[host_name("naive_unfold_nd_float32_2")]]' in output
    assert "void naive_unfold_nd_float32_2(" in output
    assert "const device float* in" in output
    assert "device float* out" in output
    assert "const constant ConvParams_2* params" in output
    assert "out[gid] = in[gid] + float(2 + params->shape[0]);" in output
    assert '[[host_name("naive_unfold_transpose_nd_float32_2")]]' in output
    assert "void naive_unfold_transpose_nd_float32_2(" in output
    assert "out[gid] = in[gid] - float(2 + params->shape[0]);" in output
    assert '[[host_name("naive_unfold_nd_float16_3")]]' in output
    assert "void naive_unfold_nd_float16_3(" in output
    assert "const device half* in" in output
    assert "const constant ConvParams_3* params" in output
    assert '[[host_name("naive_unfold_transpose_nd_float16_3")]]' in output
    assert "void naive_unfold_transpose_nd_float16_3(" in output
    assert "template [[host_name" not in output
    assert "naive_unfold_Nd(" not in output
    assert "naive_unfold_transpose_Nd(" not in output
    assert "ConvParams<n>" not in output
    # The ConvParams<N> struct template referenced by the kernel signatures is
    # materialized into concrete structs, with no residual instantiations left.
    assert "struct ConvParams_2 {" in output
    assert "int shape[2];" in output
    assert "struct ConvParams_3 {" in output
    assert "int shape[3];" in output
    assert "ConvParams<2>" not in output
    assert "ConvParams<3>" not in output


def test_preprocessor_materialized_numeric_specialization_preserves_member_names():
    code = """
    struct PackedScale {
        uint8_t bits;
    };

    template <typename T, const int group_size, const int bits>
    [[kernel]] void nvfp4_quantize(
        const device T* in [[buffer(0)]],
        device uint8_t* out [[buffer(1)]]) {
        PackedScale s;
        T sample = in[0];
        uint8_t q_scale = s.bits;
        uint8_t output = q_scale + bits;
        out[0] = output;
    }

    template [[host_name("nvfp4_quantize_float_gs_16_b_4")]] [[kernel]]
    decltype(nvfp4_quantize<float, 16, 4>) nvfp4_quantize<float, 16, 4>;
    """

    output = MetalPreprocessor().preprocess(code)
    ast = MetalParser(MetalLexer(output, preprocess=False).tokenize()).parse()

    assert "decltype(nvfp4_quantize<float, 16, 4>)" not in output
    assert "void nvfp4_quantize_float_gs_16_b_4(" in output
    assert "const device float* in" in output
    assert "float sample = in[0];" in output
    assert "uint8_t q_scale = s.bits;" in output
    assert "uint8_t q_scale = s.4;" not in output
    assert "uint8_t output = q_scale + 4;" in output
    assert [function.name for function in ast.functions] == [
        "nvfp4_quantize_float_gs_16_b_4"
    ]


def test_preprocessor_removes_proven_static_assertion_after_specialization():
    code = """
    inline int checked_simd_size() {
        constexpr int simd_size = 32;
        static_assert(simd_size == 32, "Expected a 32-lane simdgroup.");
        return simd_size;
    }

    template <typename T, const int group_size>
    [[kernel]] void quantize(
        const device T* in [[buffer(0)]],
        device T* out [[buffer(1)]]) {
        constexpr int simd_size = 32;
        static_assert(
            group_size % simd_size == 0,
            "Group size must be divisible by simd size.");
        out[0] = in[0];
    }

    template [[host_name("quantize_float_gs_32")]] [[kernel]]
    decltype(quantize<float, 32>) quantize<float, 32>;
    """

    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
    )
    output = preprocessor._lower_struct_member_functions(materialized)

    assert "int checked_simd_size()" in output
    assert "void quantize_float_gs_32(" in output
    assert "static_assert" not in output
    assert "constexpr int simd_size = 32;" in output


def test_preprocessor_resolves_specialized_constexpr_static_assert_dependencies():
    code = """
    template <int Word, int Bits>
    constexpr int get_pack_factor() {
        constexpr int factor = Word / Bits;
        return factor;
    }

    template <int GroupSize, int Bits>
    [[kernel]] void checked_pack(
        device int* out [[buffer(0)]]) {
        constexpr int reads =
            GroupSize / (get_pack_factor<8, Bits>() * 4);
        static_assert(
            reads * get_pack_factor<8, Bits>() <= GroupSize,
            "Pack reads must fit the group.");
        out[0] = reads;
    }

    template [[host_name("checked_pack_16_4")]] [[kernel]]
    decltype(checked_pack<16, 4>) checked_pack<16, 4>;
    """

    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
    )
    output = preprocessor._lower_struct_member_functions(materialized)

    assert "void checked_pack_16_4(" in output
    assert "static_assert" not in output
    assert "constexpr int reads" in output


def test_preprocessor_caches_nested_constexpr_assert_helpers_per_specialization():
    code = """
    template <int Bits>
    constexpr int pack_factor() {
        return 32 / Bits;
    }

    template <int Bits>
    constexpr int doubled_pack_factor() {
        constexpr int factor = pack_factor<Bits>();
        return factor * 2;
    }

    template <int Bits>
    [[kernel]] void checked_nested_helper(
        device int* out [[buffer(0)]]) {
        constexpr int expected = Bits == 4 ? 16 : 8;
        static_assert(doubled_pack_factor<Bits>() == expected);
        static_assert(doubled_pack_factor<Bits>() == expected);
        out[0] = expected;
    }

    template [[host_name("checked_nested_helper_4")]] [[kernel]]
    decltype(checked_nested_helper<4>) checked_nested_helper<4>;
    template [[host_name("checked_nested_helper_8")]] [[kernel]]
    decltype(checked_nested_helper<8>) checked_nested_helper<8>;
    """

    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
    )
    output = preprocessor._lower_struct_member_functions(materialized)

    assert "void checked_nested_helper_4(" in output
    assert "void checked_nested_helper_8(" in output
    assert "static_assert" not in output
    assert set(preprocessor._static_constexpr_helper_values.values()) == {
        "4",
        "8",
        "16",
    }
    assert len(preprocessor._static_constexpr_helper_values) == 4


def test_preprocessor_constexpr_assert_helper_respects_local_shadowing():
    code = """
    template <int Bits>
    constexpr int pack_factor() {
        return 32 / Bits;
    }

    kernel void shadowed_assertion(
        device int* out [[buffer(0)]],
        int runtime_bits [[threads_per_grid]]) {
        constexpr int Bits = 4;
        {
            int Bits = runtime_bits;
            static_assert(pack_factor<Bits>() == 8);
            out[0] = Bits;
        }
    }
    """

    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
    )
    with pytest.raises(MetalStaticAssertionError) as exc_info:
        preprocessor._lower_struct_member_functions(materialized)

    error = exc_info.value
    assert error.reason == "condition-unresolved"
    assert error.unresolved_dependencies == ("Bits", "pack_factor")


def test_preprocessor_rejects_false_specialized_constexpr_helper_assertion():
    code = """
    template <int Word, int Bits>
    constexpr int pack_factor() {
        return Word / Bits;
    }

    template <int Reads>
    [[kernel]] void rejected_pack(
        device int* out [[buffer(0)]]) {
        static_assert(Reads * pack_factor<8, 4>() <= 16);
        out[0] = Reads;
    }

    template [[host_name("rejected_pack_9")]] [[kernel]]
    decltype(rejected_pack<9>) rejected_pack<9>;
    """

    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
    )
    with pytest.raises(MetalStaticAssertionError) as exc_info:
        preprocessor._lower_struct_member_functions(materialized)

    error = exc_info.value
    assert error.reason == "assertion-failed"
    assert error.resolved_expression == "9 * 2 <= 16"
    assert error.unresolved_dependencies == ()


@pytest.mark.parametrize(
    ("helper", "helper_name"),
    (
        (
            """
            template <int Value>
            constexpr int recursive_value() {
                return recursive_value<Value>();
            }
            """,
            "recursive_value",
        ),
        (
            """
            template <int Value>
            constexpr int runtime_local_value() {
                int local = Value;
                return local;
            }
            """,
            "runtime_local_value",
        ),
    ),
)
def test_preprocessor_keeps_unresolved_constexpr_assert_helpers_fail_closed(
    helper,
    helper_name,
):
    code = helper + f"""
    kernel void unresolved_helper(device int* out [[buffer(0)]]) {{
        static_assert({helper_name}<4>() == 4);
        out[0] = 4;
    }}
    """

    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
    )
    with pytest.raises(MetalStaticAssertionError) as exc_info:
        preprocessor._lower_struct_member_functions(materialized)

    error = exc_info.value
    assert error.reason == "condition-unresolved"
    assert error.unresolved_dependencies == (helper_name,)


def test_preprocessor_short_circuits_unsupported_constexpr_assert_helper():
    code = """
    template <int Value>
    constexpr int unsupported_value() {
        int local = Value;
        return local;
    }

    kernel void short_circuited_helper(device int* out [[buffer(0)]]) {
        static_assert(true || unsupported_value<4>() == 4);
        static_assert(false && unsupported_value<4>() == 4 || true);
        out[0] = 4;
    }
    """

    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
    )
    output = preprocessor._lower_struct_member_functions(materialized)

    assert "static_assert" not in output
    assert "kernel void short_circuited_helper" in output


@pytest.mark.parametrize(
    "type_text",
    (
        "Wrapper<int>",
        "const Wrapper<uint>",
        "metal::vec<int, 1>",
    ),
)
def test_preprocessor_constexpr_helpers_reject_integral_wrapper_types(type_text):
    assert MetalPreprocessor()._static_constexpr_integral_type(type_text) is None


def test_preprocessor_removes_proven_static_assertion_with_logical_chain():
    code = """
    kernel void supported_group_size(device float* out [[buffer(0)]]) {
        constexpr int group_size = 2;
        static_assert(
            group_size == 2 || group_size == 3 || group_size == 4 ||
                group_size == 5 || group_size == 6 || group_size == 8,
            "Unsupported group size.");
        out[0] = 0.0;
    }
    """

    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
    )
    output = preprocessor._lower_struct_member_functions(materialized)

    assert "static_assert" not in output
    assert "kernel void supported_group_size" in output


def test_preprocessor_removes_proven_static_assertion_for_concrete_struct_layout():
    code = """
    struct ScalarPair {
        float first;
        float second;
    };

    struct TaggedPair {
        uchar tag;
        ScalarPair value;
        half weights[2];
    };

    kernel void valid_layout(device float* out [[buffer(0)]]) {
        static_assert(sizeof(ScalarPair) == 8, "Unexpected pair layout.");
        static_assert(sizeof(TaggedPair) == 16, "Unexpected tagged layout.");
        out[0] = 1.0f;
    }
    """

    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
    )
    output = preprocessor._lower_struct_member_functions(materialized)

    assert "static_assert" not in output
    assert "struct ScalarPair" in output
    assert "struct TaggedPair" in output


def test_preprocessor_rejects_false_static_assertion_for_padded_struct_layout():
    code = """
    struct PaddedValue {
        uchar tag;
        float3 value;
    };

    kernel void invalid_layout(device float* out [[buffer(0)]]) {
        static_assert(sizeof(PaddedValue) == 16, "Unexpected padded layout.");
        out[0] = 0.0f;
    }
    """

    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
    )
    with pytest.raises(MetalStaticAssertionError) as exc_info:
        preprocessor._lower_struct_member_functions(materialized)

    error = exc_info.value
    assert error.reason == "assertion-failed"
    assert error.expression == "sizeof(PaddedValue) == 16"
    assert error.resolved_expression == "32 == 16"
    assert error.unresolved_dependencies == ()


@pytest.mark.parametrize(
    "declaration",
    (
        "device float* value;",
        "float first, second;",
        "alignas(16) float value;",
    ),
)
def test_preprocessor_keeps_unsupported_struct_layout_assertions_fail_closed(
    declaration,
):
    code = f"""
    struct UnsupportedLayout {{
        {declaration}
    }};

    kernel void unresolved_layout(device float* out [[buffer(0)]]) {{
        static_assert(sizeof(UnsupportedLayout) == 8, "Layout must be known.");
        out[0] = 0.0f;
    }}
    """

    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
    )
    with pytest.raises(MetalStaticAssertionError) as exc_info:
        preprocessor._lower_struct_member_functions(materialized)

    error = exc_info.value
    assert error.reason == "condition-unresolved"
    assert error.resolved_expression == "sizeof(UnsupportedLayout) == 8"
    assert error.unresolved_dependencies == ("UnsupportedLayout",)


def test_preprocessor_rejects_false_static_assertion_with_source_context():
    code = """
    kernel void invalid_specialization(device float* out [[buffer(0)]]) {
        constexpr int group_size = 31;
        static_assert(
            group_size % 32 == 0,
            "Group size must be divisible by 32.");
        out[0] = 0.0;
    }
    """

    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
    )
    with pytest.raises(MetalStaticAssertionError) as exc_info:
        preprocessor._lower_struct_member_functions(materialized)

    error = exc_info.value
    assert error.project_diagnostic_code == "project.translate.metal-static-assertion"
    assert error.missing_capabilities == ("compile-time.static-assertion",)
    assert error.reason == "assertion-failed"
    assert error.expression == "group_size % 32 == 0"
    assert error.resolved_expression == "31 % 32 == 0"
    assert error.assertion_message == "Group size must be divisible by 32."
    assert error.unresolved_dependencies == ()
    assert error.source_location["line"] == 4
    assert error.source_location["column"] == 9
    assert "evaluated to false" in str(error)
    assert "select specialization values" in error.suggested_action


def test_preprocessor_rejects_false_static_assertion_logical_chain():
    code = """
    kernel void unsupported_group_size(device float* out [[buffer(0)]]) {
        constexpr int group_size = 7;
        static_assert(
            group_size == 2 || group_size == 4 || group_size == 8,
            "Group size must be a supported tile width.");
        out[0] = 0.0;
    }
    """

    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
    )
    with pytest.raises(MetalStaticAssertionError) as exc_info:
        preprocessor._lower_struct_member_functions(materialized)

    error = exc_info.value
    assert error.reason == "assertion-failed"
    assert error.resolved_expression == "7 == 2 || 7 == 4 || 7 == 8"
    assert error.assertion_message == "Group size must be a supported tile width."
    assert error.unresolved_dependencies == ()
    assert error.source_location["line"] == 4
    assert error.source_location["column"] == 9


def test_preprocessor_rejects_unresolved_static_assertion_with_dependencies():
    code = """
    kernel void unresolved_constraint(
        device float* out [[buffer(0)]],
        uint runtime_width [[threads_per_grid]]) {
        static_assert(
            runtime_width == 32,
            "Runtime width must be specialized.");
        out[0] = 0.0;
    }
    """

    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
    )
    with pytest.raises(MetalStaticAssertionError) as exc_info:
        preprocessor._lower_struct_member_functions(materialized)

    error = exc_info.value
    assert error.project_diagnostic_code == "project.translate.metal-static-assertion"
    assert error.reason == "condition-unresolved"
    assert error.expression == "runtime_width == 32"
    assert error.resolved_expression == "runtime_width == 32"
    assert error.assertion_message == "Runtime width must be specialized."
    assert error.unresolved_dependencies == ("runtime_width",)
    assert error.source_location["line"] == 5
    assert error.source_location["column"] == 9
    assert "unresolved dependencies: runtime_width" in str(error)
    assert "constexpr integral values" in error.suggested_action


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        (
            "2 == 2 || 2 == 3 || 2 == 4 || 2 == 5 || 2 == 6 || 2 == 8",
            (True, 1),
        ),
        ("2 != 2 || 2 != 3", (True, 1)),
        ("2 != 2 && 3 != 3", (True, 0)),
        ("1 == 1 || 2 == 3 && 4 == 5", (True, 1)),
        ("(1 == 1 || 2 == 3) && 4 == 5", (True, 0)),
    ),
)
def test_preprocessor_folds_logical_comparison_chains(expression, expected):
    assert (
        MetalPreprocessor()._evaluate_static_integral_expression(expression) == expected
    )


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        ("true || runtime_width == 32", (True, 1)),
        ("false && runtime_width != 32", (True, 0)),
        ("false || runtime_width == 32", (False, None)),
        ("true && runtime_width != 32", (False, None)),
    ),
)
def test_preprocessor_short_circuits_unresolved_logical_operands(expression, expected):
    assert (
        MetalPreprocessor()._evaluate_static_integral_expression(expression) == expected
    )


@pytest.mark.parametrize(
    "assertion",
    (
        "static_assert(true)",
        "static_assert();",
    ),
)
def test_preprocessor_rejects_malformed_static_assertion(assertion):
    code = f"""
    kernel void malformed_assertion(device float* out [[buffer(0)]]) {{
        {assertion}
        out[0] = 0.0;
    }}
    """
    preprocessor = MetalPreprocessor()
    materialized = preprocessor._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
    )

    with pytest.raises(MetalStaticAssertionError) as exc_info:
        preprocessor._lower_struct_member_functions(materialized)

    error = exc_info.value
    assert error.project_diagnostic_code == "project.translate.metal-static-assertion"
    assert error.reason == "assertion-malformed"
    assert error.source_location["line"] == 3
    assert error.source_location["column"] == 9
    assert "well-formed static_assert" in error.suggested_action


def test_preprocessor_materializes_explicit_template_helper_calls():
    code = """
    template <typename T, typename IdxT, int Offset>
    T load_with_offset(device const T* src, IdxT index) {
        return src[index] + T(Offset);
    }

    kernel void copy(device const float* src [[buffer(0)]],
                     device float* dst [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
        dst[gid] = load_with_offset<float, uint, 7>(src, gid);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "load_with_offset<float, uint, 7>" not in output
    assert "load_with_offset_float_uint_7(src, gid)" in output
    assert "float load_with_offset_float_uint_7(" in output
    assert "device const float* src" in output
    assert "uint index" in output
    assert "return src[index] + float(7);" in output


def test_preprocessor_does_not_treat_function_pointer_aliases_as_functor_calls():
    code = """
    typedef void (im2col_t)(thread float*, int);
    typedef void (*RadixFunc)(thread float2*, thread float2*);
    using Transform = void (*)(thread float*, int);

    struct Helper {
      int value() { return 1; }
    };
    """

    output = MetalPreprocessor().preprocess(code)

    assert "typedef void (im2col_t)(thread float*, int);" in output
    assert "typedef void (*RadixFunc)(thread float2*, thread float2*);" in output
    assert "using Transform = void (*)(thread float*, int);" in output


def test_preprocessor_materializes_callable_non_type_template_arguments():
    code = """
    typedef void (*RadixFunc)(thread float2*, thread float2*);

    void radix2(thread float2* values, thread float2* scratch) {
        values[0] = scratch[0];
    }

    void radix4(thread float2* values, thread float2* scratch) {
        values[0] = scratch[0] + scratch[1];
    }

    template <int Radix, RadixFunc Function>
    void apply_radix(thread float2* values, thread float2* scratch) {
        Function(values, scratch);
    }

    kernel void fft(device float2* out [[buffer(0)]]) {
        float2 values[2];
        float2 scratch[2];
        apply_radix<2, radix2>(values, scratch);
        apply_radix<4, radix4>(values, scratch);
        out[0] = values[0];
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "typedef void (*RadixFunc)(thread float2*, thread float2*);" in output
    assert "apply_radix_2_radix2(values, scratch);" in output
    assert "apply_radix_4_radix4(values, scratch);" in output
    assert "void apply_radix_2_radix2(" in output
    assert "radix2(values, scratch);" in output
    assert "void apply_radix_4_radix4(" in output
    assert "radix4(values, scratch);" in output


def test_preprocessor_materializes_helper_calls_from_local_alias_arguments():
    code = """
    template <typename T, typename U, int Count>
    U load_values(const device T* src, thread U* scratch) {
        scratch[0] = U(src[0]);
        return scratch[Count - 2];
    }

    template <typename U, int Count>
    U qdot(const thread U* scratch) {
        return scratch[Count - 2];
    }

    kernel void compute(
        const device float* src [[buffer(0)]],
        device float* dst [[buffer(1)]]) {
        using Scalar = float;
        typedef Scalar U;
        constexpr int Count = 2;
        thread U scratch[Count];
        U loaded = load_values<float, U, Count>(src, scratch);
        dst[0] = qdot<U, Count>(scratch) + loaded;
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "load_values<float, U, Count>" not in output
    assert "qdot<U, Count>" not in output
    assert "load_values_float_float_2(src, scratch)" in output
    assert "qdot_float_2(scratch)" in output
    assert "float load_values_float_float_2(" in output
    assert "float qdot_float_2(" in output
    assert "thread float* scratch" in output
    assert "return scratch[2 - 2];" in output
    assert "load_values_float_U" not in output
    assert "qdot_U" not in output


def test_preprocessor_leaves_large_decltype_instantiation_families_bounded():
    declarations = "\n".join(
        f'template [[host_name("kernel" "{index}")]] [[kernel]] '
        f"decltype(arange<uint>) arange<uint>;"
        for index in range(4)
    )
    code = f"""
    template <typename T>
    [[kernel]] void arange(
        device T* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {{
        out[gid] = T(gid);
    }}

    {declarations}
    """

    output = MetalPreprocessor(max_template_specializations=3).preprocess(code)

    assert "decltype(arange<uint>)" in output
    assert "void kernel0(" not in output
    assert "template <typename T>" in output


def test_preprocessor_materializes_all_project_instantiations_when_limit_not_enforced():
    # Counterpart of the "bounded" test above: template-hostile targets cannot
    # emit residual generic templates, so the project pipeline calls the
    # instantiation materializer with enforce_specialization_limit=False. Every
    # explicit instantiation is then materialized even though the count (4)
    # exceeds max_template_specializations (3); the default path still bails.
    declarations = "\n".join(
        f'template [[host_name("kernel" "{index}")]] [[kernel]] '
        f"decltype(arange<uint>) arange<uint>;"
        for index in range(4)
    )
    code = f"""
    template <typename T>
    [[kernel]] void arange(
        device T* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {{
        out[gid] = T(gid);
    }}

    {declarations}
    """

    preprocessor = MetalPreprocessor(max_template_specializations=3)

    bounded = preprocessor._materialize_project_template_instantiations(code)
    assert "decltype(arange<uint>)" in bounded
    assert "void kernel0(" not in bounded

    materialized = preprocessor._materialize_project_template_instantiations(
        code, enforce_specialization_limit=False
    )
    assert "decltype(arange<uint>)" not in materialized
    assert "template <typename T>" not in materialized
    for index in range(4):
        assert f"void kernel{index}(" in materialized


def test_preprocessor_materializes_only_selected_project_host_names():
    declarations = "\n".join(
        f'template [[host_name("kernel" "{index}")]] [[kernel]] '
        f"decltype(arange<uint>) arange<uint>;"
        for index in range(4)
    )
    code = f"""
    template <typename T>
    [[kernel]] void arange(
        device T* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {{
        out[gid] = T(gid);
    }}

    {declarations}
    """

    materialized = MetalPreprocessor()._materialize_project_template_instantiations(
        code,
        enforce_specialization_limit=False,
        host_names={"kernel2"},
    )

    assert "decltype(arange<uint>)" not in materialized
    assert "template <typename T>" not in materialized
    assert "void kernel2(" in materialized
    for index in (0, 1, 3):
        assert f"void kernel{index}(" not in materialized


def test_preprocessor_materializes_nested_explicit_template_helper_calls():
    code = """
    template <typename T>
    T cast_value(float value) {
        return T(value);
    }

    template <typename T>
    T twice(float value) {
        return cast_value<T>(value) + cast_value<T>(value);
    }

    kernel void copy(device float* dst [[buffer(0)]]) {
        dst[0] = twice<float>(1.0);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "twice<float>" not in output
    assert "cast_value<float>" not in output
    assert "twice_float(1.0)" in output
    assert "float twice_float(float value)" in output
    assert "cast_value_float(value) + cast_value_float(value)" in output
    assert "float cast_value_float(float value)" in output


def test_preprocessor_preserves_operator_comparison_overloads():
    code = """
    struct complex64_t {
        float real;
        float imag;
    };

    template <typename T>
    T operator+(T lhs, T rhs) {
        return lhs + rhs;
    }

    constexpr bool operator>(complex64_t a, complex64_t b) {
        return a.real > b.real;
    }

    constexpr bool operator<=(complex64_t a, complex64_t b) {
        return operator>(b, a);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "operator<=(complex64_t a, complex64_t b)" in output
    assert "return operator>(b, a);" in output
    assert "operator_complex64_t_a_complex64_t_b" not in output


def test_preprocessor_materializes_only_reachable_explicit_template_helper_calls():
    code = """
    template <typename T>
    T cast_value(float value) {
        return T(value);
    }

    void write_value(device float* dst) {
        dst[0] = cast_value<float>(1.0);
    }

    void unused(device half* dst) {
        dst[0] = cast_value<half>(2.0);
    }

    kernel void copy(device float* dst [[buffer(0)]]) {
        write_value(dst);
    }
    """

    output = MetalPreprocessor(max_template_specializations=1).preprocess(code)

    assert "cast_value_float(1.0)" in output
    assert "float cast_value_float(float value)" in output
    assert "cast_value_half" not in output
    assert "cast_value<half>(2.0)" in output


def test_preprocessor_preserves_explicit_template_specialization_calls():
    code = """
    template <typename T>
    T convert_type(float value) {
        return T(value);
    }

    template <> uint convert_type<uint>(float value) {
        return uint(value + 1.0);
    }

    kernel void copy(device uint* dst [[buffer(0)]]) {
        dst[0] = convert_type<uint>(1.0);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "convert_type<uint>(1.0)" in output
    assert "convert_type_uint(1.0)" not in output
    assert "uint convert_type<uint>(float value)" in output
    assert "uint convert_type_uint(float value)" not in output


def test_preprocessor_preserves_explicit_specialization_called_through_local_alias():
    code = """
    template <typename T>
    T convert_type(float value) {
        return T(value);
    }

    template <> uint convert_type<uint>(float value) {
        return uint(value + 1.0);
    }

    kernel void copy(device uint* dst [[buffer(0)]]) {
        using U = uint;
        dst[0] = convert_type<U>(1.0);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "convert_type<U>(1.0)" in output
    assert "convert_type_uint(1.0)" not in output
    assert "uint convert_type<uint>(float value)" in output
    assert "uint convert_type_uint(float value)" not in output


def test_preprocessor_reports_explicit_template_specialization_limit():
    code = """
    template <typename T>
    T cast_value(float value) {
        return T(value);
    }

    kernel void copy(device float* dst [[buffer(0)]]) {
        dst[0] = cast_value<float>(1.0);
        dst[1] = cast_value<half>(2.0);
    }
    """

    with pytest.raises(MetalTemplateSpecializationError) as exc_info:
        MetalPreprocessor(max_template_specializations=1).preprocess(code)

    assert "template specialization limit exceeded" in str(exc_info.value)
    assert (
        getattr(exc_info.value, "project_diagnostic_code", None)
        == "project.translate.metal-template-specialization"
    )
    assert getattr(exc_info.value, "missing_capabilities", None) == (
        "template.specialization",
    )


def test_preprocessor_dedupes_equivalent_explicit_template_helper_signatures():
    code = """
    template <typename T, typename IdxT, int Width>
    T load_value(device const T* src, IdxT index) {
        return src[index + Width];
    }

    kernel void copy(device const float* src [[buffer(0)]],
                     device float* dst [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
        dst[0] = load_value<float, uint, 4>(src, gid);
        dst[1] = load_value< float, uint, 4 >(src, gid);
        dst[2] = load_value<float /* same concrete type */, uint, 4>(src, gid);
    }
    """

    output = MetalPreprocessor(max_template_specializations=1).preprocess(code)

    assert output.count("float load_value_float_uint_4(") == 1
    assert output.count("load_value_float_uint_4(src, gid)") == 3
    assert "load_value<" not in output


def test_preprocessor_dedupes_scan_style_repeated_helper_calls_before_budget():
    calls = "\n".join(
        f"        acc = scan_step<float, int, 4, true>(acc, {index});"
        for index in range(24)
    )
    code = f"""
    template <typename T, typename OffsetT, int Width, bool Inclusive>
    T scan_step(T value, OffsetT offset) {{
        return value + T(offset) + T(Width);
    }}

    kernel void scan(device float* dst [[buffer(0)]]) {{
        float acc = 0.0;
{calls}
        dst[0] = acc;
    }}
    """

    output = MetalPreprocessor(max_template_specializations=1).preprocess(code)

    assert output.count("float scan_step_float_int_4_true(") == 1
    assert output.count("scan_step_float_int_4_true(acc,") == 24
    assert "scan_step<" not in output


def test_preprocessor_reports_template_specialization_limit_details():
    code = """
    template <typename T>
    T cast_value(float value) {
        return T(value);
    }

    kernel void copy(device float* dst [[buffer(0)]]) {
        dst[0] = cast_value<float>(1.0);
        dst[1] = cast_value<half>(2.0);
    }
    """

    with pytest.raises(MetalTemplateSpecializationError) as exc_info:
        MetalPreprocessor(
            max_template_specializations=1,
            template_specialization_limit_source=(
                "project.source_options.metal.max_template_specializations"
            ),
        ).preprocess(code)

    error = exc_info.value
    assert error.limit == 1
    assert error.limit_source == (
        "project.source_options.metal.max_template_specializations"
    )
    assert error.unique_specialization_count == 2
    assert error.requested_signature == "cast_value<half>"
    assert "2 unique concrete signatures requested" in str(error)
    assert "limit 1 from project.source_options.metal.max_template_specializations" in (
        str(error)
    )
    assert "Suggested action:" in str(error)


def test_preprocessor_expands_multiline_function_macro_invocation():
    code = """
    #define DECLARE_TYPED(name, type) type name;
    DECLARE_TYPED(
        value,
        float);
    """

    output = MetalPreprocessor().preprocess(code)

    assert "DECLARE_TYPED(" not in output
    assert "float value;" in output


def test_preprocessor_preserves_ordinary_multiline_function_call():
    code = """
    float make_value(float value) { return value; }
    float result = make_value(
        1.0f);
    """

    output = MetalPreprocessor().preprocess(code)

    assert "float result = make_value(\n        1.0f);" in output


def test_preprocessor_preserves_directives_inside_ordinary_multiline_call():
    code = """
    #define USE_FIRST 1
    float choose(float left, float right) { return left + right; }
    float result = choose(
    #if USE_FIRST
        1.0f,
    #else
        2.0f,
    #endif
        3.0f);
    """

    output = MetalPreprocessor().preprocess(code)

    assert "float result = choose(\n        1.0f,\n        3.0f);" in output
    assert "2.0f" not in output


def test_preprocessor_multiline_macro_matches_single_line_macro_semantics():
    definitions = r"""
    #define ADD(left, right) ((left) + (right))
    #define JOIN(left, right) left ## right
    #define NAME(value) #value
    #define DECLARE(name, expression, ...) \
        float JOIN(name, _value) = ADD(expression, __VA_ARGS__); \
        constant char* JOIN(name, _name) = NAME(name);
    """
    single_line = definitions + r"""
    DECLARE(sample, ADD(1.0f, 2.0f), 3.0f)
    """
    multiline = definitions + r"""
    DECLARE(
        sample,
        ADD(1.0f, 2.0f),
        3.0f)
    """

    single_output = MetalPreprocessor().preprocess(single_line)
    multiline_output = MetalPreprocessor().preprocess(multiline)

    def normalize_whitespace(text):
        return re.sub(r"\s+", " ", text).strip()

    assert normalize_whitespace(multiline_output) == normalize_whitespace(single_output)
    assert "float sample_value" in multiline_output
    assert 'constant char* sample_name = "sample";' in multiline_output


def test_preprocessor_multiline_macro_ignores_argument_comment_and_literal_parens():
    code = r"""
    #define PASS(value) value
    #define ADD(left, right) ((left) + (right))
    constant char* text = PASS(
        "literal ),( text");
    float value = ADD(
        PASS(1.0f /* unmatched ) and , */), // unmatched (
        2.0f);
    """

    output = MetalPreprocessor().preprocess(code)

    assert 'constant char* text = "literal ),( text";' in output
    assert "float value = ((1.0f) + (2.0f));" in output
    assert "PASS(" not in output
    assert "ADD(" not in output


def test_preprocessor_reports_unterminated_multiline_function_macro_call():
    code = """
    #define DECLARE_TYPED(name, type) type name;
    DECLARE_TYPED(
        value,
        float
    """

    with pytest.raises(MetalMacroExpansionError) as exc_info:
        MetalPreprocessor().preprocess(code, file_path="unterminated.metal")

    error = exc_info.value
    assert error.project_diagnostic_code == (
        "project.translate.metal-macro-expansion-invalid"
    )
    assert error.missing_capabilities == ("metal.function-macro-expansion",)
    assert error.macro_name == "DECLARE_TYPED"
    assert error.reason == "unterminated-function-macro-invocation"
    assert error.source_location["file"] == "unterminated.metal"
    assert error.source_location["line"] == 3
    assert error.source_location["column"] == 5
    assert error.source_location["length"] == len("DECLARE_TYPED")


def test_preprocessor_include_with_search_path(tmp_path):
    include_file = tmp_path / "constants.metal"
    include_file.write_text("constant int from_include = 7;\n", encoding="utf-8")
    code = """
    #include "constants.metal"
    int value = from_include;
    """

    values = token_values(
        code,
        include_paths=[str(tmp_path)],
        file_path=str(tmp_path / "main.metal"),
    )

    assert "from_include" in values
    assert '"constants.metal"' not in values


def test_preprocessor_include_honors_pragma_once(tmp_path):
    include_file = tmp_path / "constants.metal"
    include_file.write_text(
        "#pragma once\nconstant int guarded = 7;\n", encoding="utf-8"
    )
    code = """
    #include "constants.metal"
    #include "constants.metal"
    int value = guarded;
    """

    output = MetalPreprocessor(include_paths=[str(tmp_path)]).preprocess(
        code,
        file_path=str(tmp_path / "main.metal"),
    )

    assert output.count("guarded") == 2
    assert "#pragma once" not in output


def test_unresolved_system_include_is_preserved():
    code = """
    #include <metal_stdlib>
    using namespace metal;
    """

    output = MetalPreprocessor().preprocess(code)
    values = token_values(code)

    assert "#include <metal_stdlib>" in output
    assert "#include <metal_stdlib>" in values
    assert "using" in values


def test_dawn_tint_warning_prologue_is_stripped_before_msl():
    # Reduced from:
    # Repo: https://dawn.googlesource.com/dawn
    # Commit: 78a171ad2ed7f7265cfc3dd52e4e7a637a099df0
    # Path: test/tint/bug/tint/2201.wgsl.expected.msl
    code = """
    <dawn>/test/tint/bug/tint/2201.wgsl:9:9 warning: code is unreachable
            let _e16_ = vec2(false, false);
            ^^^^^^^^^

    #include <metal_stdlib>
    using namespace metal;
    kernel void v() {}
    """

    output = MetalPreprocessor().preprocess(code)

    assert output.lstrip().startswith("#include <metal_stdlib>")
    assert "warning: code is unreachable" not in output


def test_diagnostic_error_prologue_is_preserved():
    code = """
    <dawn>/test/tint/broken.wgsl:1:1 error: invalid shader

    #include <metal_stdlib>
    using namespace metal;
    kernel void v() {}
    """

    output = MetalPreprocessor().preprocess(code)

    assert "error: invalid shader" in output


def test_target_conditionals_simulator_macro_defaults_to_device_import():
    code = """
    #include <TargetConditionals.h>
    #ifndef TARGET_OS_SIMULATOR
    #error TARGET_OS_SIMULATOR not defined. Check <TargetConditionals.h>
    #endif
    #if TARGET_OS_SIMULATOR
    int simulator_only = ;
    #else
    int device_import = 1;
    #endif
    """

    values = token_values(code)

    assert "device_import" in values
    assert "simulator_only" not in values


def test_has_include_checks_include_paths_from_metalpetal_header(tmp_path):
    # Reduced from:
    # Repo: https://github.com/MetalPetal/MetalPetal
    # Commit: f9b78897bd4214bb097f352a1bde0a4f4a1e2ddb
    # Path: Frameworks/MetalPetal/MTIContext+Internal.h
    package_header = tmp_path / "MetalPetal" / "MetalPetal.h"
    package_header.parent.mkdir()
    package_header.write_text("// package umbrella\n", encoding="utf-8")
    code = """
    #if __has_include(<MetalPetal/MetalPetal.h>)
    int package_layout = 1;
    #else
    int local_layout = 1;
    #endif
    """

    values = token_values(code, include_paths=[str(tmp_path)])

    assert "package_layout" in values
    assert "local_layout" not in values


def test_strict_missing_include_raises(tmp_path):
    code = '#include "missing.metal"\nint value = 1;'

    with pytest.raises(FileNotFoundError, match="Include not found: missing.metal"):
        MetalLexer(
            code,
            include_paths=[str(tmp_path)],
            strict_preprocessor=True,
            file_path=str(tmp_path / "main.metal"),
        ).tokenize()


def test_parser_uses_preprocessed_conditionals():
    code = """
    #define ENABLE_BODY 1
    #if ENABLE_BODY
    void main() {
        int value = 1;
    }
    #else
    void broken() {
        int value = ;
    }
    #endif
    """

    tokens = MetalLexer(code).tokenize()
    ast = MetalParser(tokens).parse()

    assert ast is not None
    assert [func.name for func in ast.functions] == ["main"]


def test_preprocessor_can_be_disabled_for_raw_directive_tokens():
    code = """
    #define VALUE 1
    int value = VALUE;
    """

    values = token_values(code, preprocess=False)

    assert "#define VALUE 1" in values
    assert "VALUE" in values


def test_find_and_materialize_template_struct():
    # Foundation for the struct-template materializer (issue #1354): detect a
    # struct template and materialize a concrete instantiation by substituting
    # both type and non-type parameters and renaming the declaration.
    code = """
    template <typename T, int N>
    struct Tile {
        T data[N];
        T sum() {
            T total = T(0);
            for (int i = 0; i < N; ++i) {
                total += data[i];
            }
            return total;
        }
    };
    """
    preprocessor = MetalPreprocessor()
    structs = preprocessor._find_template_structs(code)
    assert len(structs) == 1
    struct = structs[0]
    assert struct.name == "Tile"
    assert struct.template_parameters == ["T", "N"]

    materialized = preprocessor._materialize_template_struct_with_name(
        struct, ["float", "4"], "Tile_float_4"
    )
    assert "struct Tile_float_4" in materialized
    assert "float data[4];" in materialized
    assert "for (int i = 0; i < 4; ++i)" in materialized
    assert "float total = float(0);" in materialized
    # Template parameters and the template<> header must not leak through.
    assert "[N]" not in materialized
    assert "template" not in materialized


def test_materialize_template_struct_requires_all_parameters():
    code = """
    template <typename T>
    struct Box {
        T value;
    };
    """
    preprocessor = MetalPreprocessor()
    struct = preprocessor._find_template_structs(code)[0]
    # No arguments for a required parameter -> not materialized.
    assert (
        preprocessor._materialize_template_struct_with_name(struct, [], "Box_x") == ""
    )


def test_preprocessor_materializes_struct_template_instantiation():
    # Wiring the struct-template materializer (issue #1354): a concrete
    # `StructName<args>` type reference is rewritten to a materialized concrete
    # struct, the counterpart of the function-template call materializer.
    code = """
    template <typename T, int N>
    struct Tile {
        T data[N];
    };

    [[kernel]] void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Tile<float, 4> tile;
        tile.data[0] = out[i];
        out[i] = tile.data[0];
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "Tile_float_4 tile;" in output
    assert "struct Tile_float_4 {" in output
    assert "float data[4];" in output
    # The materialized concrete struct is terminated with a semicolon.
    assert output.rstrip().endswith(";")
    # No residual instantiation survives.
    assert "Tile<float, 4>" not in output


def test_preprocessor_materializes_nested_struct_template_instantiations():
    # Nested instantiations resolve through iteration: materializing Outer
    # surfaces a concrete Inner<float, 8> reference, which is then materialized.
    code = """
    template <typename T, int N> struct Inner { T vals[N]; };
    template <typename T, int N> struct Outer { Inner<T, N> inner; };

    [[kernel]] void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Outer<float, 8> o;
        out[i] = o.inner.vals[0];
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "Outer_float_8 o;" in output
    assert "struct Outer_float_8 {" in output
    assert "Inner_float_8 inner;" in output
    assert "struct Inner_float_8 {" in output
    assert "float vals[8];" in output
    assert "Outer<float, 8>" not in output
    assert "Inner<float, 8>" not in output


def test_preprocessor_materializes_nested_struct_with_enclosing_static_constant():
    code = """
    template <int Threads>
    struct Loader {
        static constexpr int reads = 256 / Threads;

        struct alignas(reads * sizeof(float)) ReadVector {
            uint8_t data[reads * sizeof(float)];
        };
    };

    template <int Rows, int Columns>
    struct Kernel {
        static constexpr int thread_count = Rows * Columns;
        using loader_t = Loader<thread_count>;
        loader_t loader;
    };

    Kernel<4, 8> kernel;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "struct Kernel_4_8" in output
    assert "using loader_t = Loader_32;" in output
    assert "struct Loader_32" in output
    assert "alignas(8 * sizeof(float))" in output
    assert "uint8_t data[8 * sizeof(float)];" in output
    assert "Loader_thread_count" not in output


def test_preprocessor_resolves_owner_alias_before_nested_struct_materialization():
    code = """
    template <typename T, int Rows, int Columns>
    struct BaseFrag {
        using value_type = T;
        static constexpr int width = Columns;
        T values[Rows * Columns];
    };

    template <typename T, int Rows, int Columns, typename Frag>
    struct Tile {
        static constexpr int width = Frag::width;
        typename Frag::value_type values[Rows * Columns];
    };

    template <typename T, int Rows, int Columns>
    struct Kernel {
        static constexpr int frag_size = 8;
        using frag_t = BaseFrag<T, frag_size, frag_size>;
        Tile<T, Rows, Columns, frag_t> tile;
    };

    Kernel<float, 4, 1> kernel;
    """
    preprocessor = MetalPreprocessor()

    output = preprocessor.preprocess(code)

    tile_name = "Tile_float_4_1_BaseFrag_float_8_8"
    tile_body = output.split(f"struct {tile_name} {{", 1)[1].split("};", 1)[0]
    assert f"{tile_name} tile;" in output
    assert "BaseFrag_float_8_8::width" in tile_body
    assert "BaseFrag_float_8_8::value_type" in tile_body
    assert "frag_t" not in tile_name
    assert "frag_t" not in tile_body
    assert preprocessor._materialized_struct_specializations[tile_name] == (
        "Tile",
        ("float", "4", "1", "BaseFrag_float_8_8"),
    )
    assert "frag_t" not in repr(
        preprocessor._materialized_struct_specializations[tile_name]
    )


def test_preprocessor_resolves_shadowed_owner_aliases_independently():
    code = """
    template <typename T, int Width>
    struct BaseFrag {
        using value_type = T;
        T values[Width];
    };

    template <typename Frag>
    struct Tile {
        typename Frag::value_type value;
    };

    template <typename T>
    struct NarrowKernel {
        using frag_t = BaseFrag<T, 4>;
        Tile<frag_t> tile;
    };

    template <typename T>
    struct WideKernel {
        using frag_t = BaseFrag<T, 8>;
        Tile<frag_t> tile;
    };

    NarrowKernel<float> narrow;
    WideKernel<half> wide;
    """
    preprocessor = MetalPreprocessor()

    output = preprocessor.preprocess(code)

    expected = {
        "Tile_BaseFrag_float_4": ("Tile", ("BaseFrag_float_4",)),
        "Tile_BaseFrag_half_8": ("Tile", ("BaseFrag_half_8",)),
    }
    tile_specializations = {
        name: provenance
        for name, provenance in (
            preprocessor._materialized_struct_specializations.items()
        )
        if provenance[0] == "Tile"
    }
    assert tile_specializations == expected
    for tile_name in expected:
        tile_body = output.split(f"struct {tile_name} {{", 1)[1].split("};", 1)[0]
        assert f"{tile_name} tile;" in output
        assert "frag_t" not in tile_name
        assert "frag_t" not in tile_body
    assert "frag_t" not in repr(tile_specializations)


def test_preprocessor_dedupes_nested_struct_aliases_by_canonical_target():
    code = """
    template <typename T, int Width>
    struct BaseFrag {
        T values[Width];
    };

    template <typename Frag>
    struct Tile {
        Frag frag;
    };

    template <typename T>
    struct Kernel {
        static constexpr int frag_size = 8;
        using base_frag_t = BaseFrag<T, frag_size>;
        using first_frag_t = base_frag_t;
        using second_frag_t = BaseFrag<T, frag_size>;
        Tile<first_frag_t> first;
        Tile<second_frag_t> second;
    };

    Kernel<float> kernel;
    """
    preprocessor = MetalPreprocessor()

    output = preprocessor.preprocess(code)

    tile_name = "Tile_BaseFrag_float_8"
    tile_body = output.split(f"struct {tile_name} {{", 1)[1].split("};", 1)[0]
    tile_specializations = {
        name: provenance
        for name, provenance in (
            preprocessor._materialized_struct_specializations.items()
        )
        if provenance[0] == "Tile"
    }
    assert output.count(f"struct {tile_name} {{") == 1
    assert f"{tile_name} first;" in output
    assert f"{tile_name} second;" in output
    assert tile_specializations == {
        tile_name: ("Tile", ("BaseFrag_float_8",)),
    }
    for alias in ("base_frag_t", "first_frag_t", "second_frag_t"):
        assert alias not in tile_name
        assert alias not in tile_body
        assert alias not in repr(tile_specializations)


def test_preprocessor_bounds_owner_alias_materialization_work():
    class CountingWorkBudget:
        def __init__(self):
            self.used = 0
            self.contexts = []

        def consume(self, amount, **kwargs):
            self.used += amount
            self.contexts.append(kwargs.get("context"))

    uses = "\n".join(
        f"Kernel<float, {width}> kernel_{width};" for width in range(1, 33)
    )
    code = f"""
    template <typename T, int Width>
    struct BaseFrag {{ T values[Width]; }};

    template <typename Frag>
    struct Tile {{ Frag frag; }};

    template <typename T, int Width>
    struct Kernel {{
        using frag_t = BaseFrag<T, Width>;
        Tile<frag_t> tile;
    }};

    {uses}
    """
    preprocessor = MetalPreprocessor(max_template_specializations=256)
    work_budget = CountingWorkBudget()

    output = preprocessor._materialize_explicit_template_struct_instantiations(
        code,
        work_budget=work_budget,
    )

    assert output.count("struct Tile_BaseFrag_float_") == 32
    assert "struct Tile_frag_t" not in output
    assert work_budget.used == 96
    assert all(
        context.startswith("reachable concrete struct '")
        for context in work_budget.contexts
    )


def test_preprocessor_reports_dependent_owner_alias_at_nested_instantiation():
    code = """
    template <typename Frag>
    struct Tile {
        Frag value;
    };

    struct Kernel {
        using first_frag_t = second_frag_t;
        using second_frag_t = first_frag_t;
        Tile<first_frag_t> tile;
    };
    """

    with pytest.raises(MetalTemplateSpecializationError) as caught:
        MetalPreprocessor().preprocess(code)

    error = caught.value
    assert "owner-scoped alias 'second_frag_t' dependent" in str(error)
    assert error.requested_signature == "Tile<second_frag_t>"
    assert error.source_location == {
        "line": 10,
        "column": 9,
        "offset": 192,
        "length": 18,
        "endLine": 10,
        "endColumn": 27,
        "endOffset": 210,
    }


def test_preprocessor_materializes_nested_tile_dimensions_from_static_constants():
    code = """
    template <int Rows, int Columns>
    struct Tile {
        float values[Rows * Columns];
    };

    template <int BlockRows, int BlockColumns>
    struct Matrix {
        static constexpr int tile_rows = BlockRows / 8;
        static constexpr int tile_columns = BlockColumns / 8;
        using tile_t = Tile<tile_rows, tile_columns>;
        tile_t tile;
    };

    Matrix<64, 32> matrix;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "struct Matrix_64_32" in output
    assert "using tile_t = Tile_8_4;" in output
    assert "struct Tile_8_4" in output
    assert "float values[8 * 4];" in output
    assert "Tile_tile_rows_tile_columns" not in output


def test_preprocessor_concretizes_nested_ternary_stride_arguments():
    code = """
    template <typename T, int Lda, int Ldb>
    struct Block {
        T value;
    };

    template <typename T, int BM, int BN, int BK, bool TransposeA, bool TransposeB>
    struct Kernel {
        static constexpr int padding = 16 / sizeof(T);
        using block_t = Block<
            T,
            TransposeA ? BM + padding : BK + padding,
            TransposeB ? BK + padding : BN + padding>;
        block_t block;
    };

    Kernel<half, 64, 64, 16, false, false> kernel;
    """

    output = MetalPreprocessor().preprocess(code)
    concrete = output.split("struct Kernel_half_64_64_16_false_false", 1)[1].split(
        "};", 1
    )[0]

    expected_alias = "using block_t = Block_half_false_64_8_16_8_false_16_8_64_8;"
    assert expected_alias in concrete
    assert "sizeof_half" not in concrete
    assert "BM" not in concrete
    assert "BN" not in concrete
    assert "BK" not in concrete


def test_materialized_static_layout_substitution_preserves_method_shadowing():
    code = """
    template <int Threads>
    struct Layout {
        static constexpr int width = Threads * 2;

        struct alignas(width) Storage {
            uint8_t data[width];
        };

        int read(int width) const {
            return width;
        }
    };

    Layout<4> layout;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "struct alignas(8) Storage" in output
    assert "uint8_t data[8];" in output
    assert "int Layout_4__read(thread const Layout_4& self, int width)" in output
    assert "return width;" in output


def test_materialized_static_layout_substitutes_equality_operands():
    code = """
    template <int Width>
    struct Layout {
        static constexpr int width = Width;
        bool matches[width == 4 ? 1 : 2];
    };

    Layout<4> layout;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "bool matches[4 == 4 ? 1 : 2];" in output


def test_materialized_static_layout_substitution_leaves_cycles_unresolved():
    code = """
    template <int Threads>
    struct Layout {
        static constexpr int first = second;
        static constexpr int second = first;
        uint8_t data[first];
    };

    Layout<4> layout;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "static constexpr int first = second;" in output
    assert "static constexpr int second = first;" in output
    assert "uint8_t data[first];" in output


def test_materialized_static_constant_folds_concrete_sizeof_for_layout():
    code = """
    template <typename T>
    struct Layout {
        static constexpr int padding = 16 / sizeof(T);
        uint8_t data[padding];
    };

    Layout<float> layout;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "static constexpr int padding = 16 / sizeof(float);" in output
    assert "uint8_t data[4];" in output


def test_preprocessor_resolves_local_sizeof_constants_through_type_aliases():
    code = """
    template <int Padding>
    struct Block {
        uint8_t data[Padding];
    };

    void float_layout() {
        using Scalar = float;
        constexpr int padding = 16 / sizeof(Scalar);
        Block<padding> value;
    }

    void half_layout() {
        using Scalar = half;
        constexpr int padding = 16 / sizeof(Scalar);
        Block<padding> value;
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Block_4 value;" in output
    assert "Block_8 value;" in output
    assert "uint8_t data[4];" in output
    assert "uint8_t data[8];" in output
    assert "Block_padding" not in output


def test_template_argument_static_constant_substitution_skips_qualified_members():
    output = MetalPreprocessor()._substitute_template_argument_static_constants(
        "Other::thread_count + Other :: thread_count + "
        "(true ? thread_count : thread_count)",
        {"thread_count": "32"},
    )

    assert output == ("Other::thread_count + Other :: thread_count + (true ? 32 : 32)")


def test_preprocessor_selects_full_struct_specialization_by_argument_key():
    code = """
    template <int Rows> struct Tile { float values[Rows]; };

    template <typename T, int Rows>
    struct Trait {
        static constexpr int tile_rows = Rows / 2;
        Tile<tile_rows> tile;
        T value;
    };

    template <> struct Trait<bool, 8> { int value; };

    [[kernel]] void k(
        device int* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Trait<bool, 8> specialized;
        Trait<float, 8> primary;
        out[i] = specialized.value + int(primary.value);
    }
    """
    output = MetalPreprocessor().preprocess(code)

    assert "Trait<bool, 8> specialized;" in output
    assert "struct Trait_bool_8 {" not in output
    assert "Trait_float_8 primary;" in output
    assert "struct Trait_float_8" in output
    assert "Tile_4 tile;" in output
    assert "struct Tile_4" in output


def test_preprocessor_matches_full_struct_specialization_across_default_arguments():
    code = """
    template <typename T, typename U = float>
    struct Trait {
        T value;
    };

    template <>
    struct Trait<bool> {
        int value;
    };

    Trait<bool, float> specialized;
    Trait<int> primary;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Trait<bool, float> specialized;" in output
    assert "struct Trait_bool_float {" not in output
    assert "Trait_int primary;" in output
    assert "struct Trait_int" in output


@pytest.mark.parametrize(
    "alias_declaration",
    [
        pytest.param("typedef float U;", id="typedef"),
        pytest.param("using U = float;", id="using"),
        pytest.param(
            "typedef float Base;\n        typedef Base U;",
            id="typedef-chain",
        ),
        pytest.param(
            "using Base = float;\n        using U = Base;",
            id="using-chain",
        ),
    ],
)
def test_preprocessor_resolves_local_alias_before_struct_specialization_selection(
    alias_declaration,
):
    code = f"""
    template <typename T>
    struct Limits {{
        static constexpr constant T finite_min = T(-1);
    }};

    template <>
    struct Limits<float> {{
        static constexpr constant float finite_min =
            -metal::numeric_limits<float>::max();
    }};

    template <typename T>
    [[kernel]] void use_limit(device T* out [[buffer(0)]]) {{
        {alias_declaration}
        out[0] = T(Limits<U>::finite_min);
    }}

    instantiate_kernel("use_limit_float", use_limit, float)
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Limits_U" not in output
    assert "Limits<U>::finite_min" in output
    assert output.count("struct Limits<float>") == 1
    assert "metal::numeric_limits<float>::max()" in output


def test_preprocessor_materializes_struct_through_nearest_local_alias():
    code = """
    template <typename T>
    struct Box {
        T value;
    };

    [[kernel]] void use_box(device float* out [[buffer(0)]]) {
        using U = float;
        Box<U> outer;
        {
            using U = half;
            Box<U> inner;
            out[0] = float(inner.value);
        }
        Box<U> restored;
        out[1] = outer.value + restored.value;
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Box_float outer;" in output
    assert "Box_half inner;" in output
    assert "Box_float restored;" in output
    assert output.count("struct Box_float") == 1
    assert output.count("struct Box_half") == 1
    assert "Box_U" not in output


def test_preprocessor_resolves_function_local_integral_constants_lexically():
    code = """
    template <typename T, int BM, int Loads>
    struct BlockLoader {
        T values[BM * Loads];
    };

    void first() {
        constexpr int WM = 4;
        BlockLoader<float, 32, (32 / (8 * WM)) * 1> outer;
        {
            BlockLoader<float, 32, (32 / (8 * WM)) * 1> before_shadow;
            constexpr int WM = 2;
            BlockLoader<float, 32, (32 / (8 * WM)) * 1> inner;
        }
        BlockLoader<float, 32, (32 / (8 * WM)) * 1> restored;
    }

    void second() {
        constexpr int WM = 1;
        BlockLoader<float, 32, (32 / (8 * WM)) * 1> conflicting;
    }

    void third() {
        constexpr int Width = 4;
        BlockLoader<float, 32, (32 / (8 * Width)) * 1> equivalent;
    }
    """

    output = MetalPreprocessor().preprocess(code)

    outer = "BlockLoader_float_32_32_8_4_1"
    inner = "BlockLoader_float_32_32_8_2_1"
    conflicting = "BlockLoader_float_32_32_8_1_1"
    assert f"{outer} outer;" in output
    assert f"{outer} before_shadow;" in output
    assert f"{inner} inner;" in output
    assert f"{outer} restored;" in output
    assert f"{conflicting} conflicting;" in output
    assert f"{outer} equivalent;" in output
    assert output.count(f"struct {outer} {{") == 1
    assert output.count(f"struct {inner} {{") == 1
    assert output.count(f"struct {conflicting} {{") == 1
    assert "BlockLoader_float_32_32_8_WM_1" not in output


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("true ? 4 : 2", 4),
        ("(false) ? (4) : (2)", 2),
        ("false ? 1 : true ? 5 : 6", 5),
        ("true ? (false ? 1 : 2) : 3", 2),
        ("(false ? 1 : 2) ? 3 : 4", 3),
        ("(true ? 4 : 3) * 2", 8),
        ("(1 & 2) == 0 ? 4 : 5", 4),
        ("1 & (2 == 0) ? 4 : 5", 5),
        ("(!0) < 0 ? 4 : 5", 5),
        ("true ? ((3 < 2) < 1) : 9", 1),
    ],
)
def test_preprocessor_folds_cpp_integral_conditional_expressions(expression, expected):
    assert MetalPreprocessor()._evaluate_static_integral_expression(expression) == (
        True,
        expected,
    )


@pytest.mark.parametrize(
    "expression",
    [
        "true ? 4",
        "true ? : 2",
        "true ? 4 :",
        "true ? 4 : 2 : 1",
        "(true ? 4 : 2",
        "4 if true else 2",
        'true ? 4 : "x"',
        "true ? -1 : 0xffffffff",
        "ns::flag ? 1 : 2",
        "1 & 2 == 0 ? 4 : 5",
        "!0 < 0 ? 4 : 5",
        "true ? 3 < 2 < 1 : 9",
    ],
)
def test_preprocessor_rejects_malformed_integral_conditional_expressions(expression):
    assert MetalPreprocessor()._evaluate_static_integral_expression(expression) == (
        False,
        None,
    )


def test_preprocessor_materializes_nested_struct_from_local_conditional_constants():
    code = """
    template <typename T, int Rows, int Columns>
    struct Tile {
        T values[Rows * Columns];
    };

    template <typename T, int M, int N, bool Transpose>
    [[kernel]] void compute(device T* out [[buffer(0)]]) {
        constexpr short base_rows = M;
        constexpr short base_columns = N;
        constexpr short rows = Transpose ? base_columns : base_rows;
        constexpr short columns = Transpose ? base_rows : base_columns;
        constexpr short half_rows = rows / 2;
        Tile<T, half_rows, columns> tile;
        out[0] = T(sizeof(tile.values));
    }

    instantiate_kernel("compute_float", compute, float, 2, 4, true)
    instantiate_kernel("compute_float_row_major", compute, float, 2, 4, false)
    """

    output = MetalPreprocessor().preprocess(code)

    assert "struct Tile_float_2_2" in output
    assert "Tile_float_2_2 tile;" in output
    assert "struct Tile_float_1_4" in output
    assert "Tile_float_1_4 tile;" in output
    assert "Tile_float_half_rows_columns" not in output
    assert "Tile<T, half_rows, columns>" not in output


@pytest.mark.parametrize(
    ("function_source", "constant_name", "expected_line"),
    [
        (
            """
            void runtime(int input) {
                int WM = input;
                Block<WM> runtime_value;
            }
            """,
            "WM",
            13,
        ),
        (
            """
            void unsupported_call() {
                constexpr int Width = choose_width();
                Block<Width> call_value;
            }
            """,
            "Width",
            13,
        ),
        (
            """
            void cyclic() {
                constexpr int First = Second;
                constexpr int Second = First;
                Block<First> cycle;
            }
            """,
            "First",
            14,
        ),
    ],
)
def test_preprocessor_reports_unproven_function_local_constants(
    function_source, constant_name, expected_line
):
    code = """
    template <int N>
    struct Block {
        int values[N];
    };

    constexpr int choose_width() {
        return 2;
    }
    """ + function_source

    with pytest.raises(MetalTemplateSpecializationError) as caught:
        MetalPreprocessor().preprocess(code)

    error = caught.value
    assert "function-local constant" in str(error)
    assert error.requested_signature == f"Block<{constant_name}>"
    assert error.unresolved_local_constants == (constant_name,)
    assert error.nested_struct_name == "Block"
    assert error.source_location["line"] == expected_line


def test_preprocessor_resolves_struct_alias_in_static_constant_initializer():
    code = """
    template <typename T>
    struct Base {
        static constexpr int count = 2;
    };

    template <typename T>
    struct Tile {
        using Fragment = Base<T>;
        static constexpr int count = Fragment::count;
    };

    Tile<float> value;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "struct Base_float" in output
    assert "struct Tile_float" in output
    assert "static constexpr int count = Base_float::count;" in output
    assert output.count("Fragment::count") == 1


def test_preprocessor_matches_cv_qualified_full_struct_specialization():
    code = """
    template <typename T>
    struct Trait {
        T value;
    };

    template <>
    struct Trait<float const> {
        int value;
    };

    Trait<const float> specialized;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Trait<const float> specialized;" in output
    assert "struct Trait_const_float {" not in output


def test_preprocessor_selects_partial_struct_specialization_by_fixed_argument():
    code = """
    template <int Rows> struct Tile { float values[Rows]; };

    template <typename T, typename U, int Rows>
    struct Trait {
        static constexpr int tile_rows = Rows / 2;
        Tile<tile_rows> tile;
        T value;
    };

    template <typename U, int Rows>
    struct Trait<bool, U, Rows> {
        int value;
    };

    Trait<bool, half, 8> specialized;
    Trait<float, half, 8> primary;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Trait<bool, half, 8> specialized;" in output
    assert "struct Trait_bool_half_8 {" not in output
    assert "Trait_float_half_8 primary;" in output
    assert "struct Trait_float_half_8" in output
    assert "Tile_4 tile;" in output


def test_preprocessor_materializes_unique_partial_for_static_member_owner():
    code = """
    template <typename T, int Rows>
    struct Trait {
        static constexpr int count = Rows;
    };

    template <typename T>
    struct Trait<T, 8> {
        static constexpr int count = 2;
    };

    int count() {
        return Trait<float, 8>::count;
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "struct Trait_float_8 {" in output
    assert "struct Trait_float_8<float, 8>" not in output
    assert "return Trait_float_8::count;" in output


def test_preprocessor_resolves_scalar_alias_from_selected_partial_specialization():
    code = """
    template <bool Select, typename A, typename B>
    struct ConditionalType {
        using type = B;
    };

    template <typename A, typename B>
    struct ConditionalType<true, A, B> {
        using type = A;
    };

    template <int Bits>
    [[kernel]] void select_word(device uint* out [[buffer(0)]]) {
        constexpr int power_of_two = (Bits & (Bits - 1)) == 0;
        using W_T =
            typename ConditionalType<power_of_two, uint, uchar>::type;
        W_T value = W_T(Bits);
        out[0] = uint(value);
    }

    instantiate_kernel("select_word_pow2", select_word, 4)
    instantiate_kernel("select_word_non_pow2", select_word, 3)
    """

    output = MetalPreprocessor().preprocess(code)

    assert len(re.findall(r"using\s+W_T\s*=\s*uint\s*;", output)) == 1
    assert len(re.findall(r"using\s+W_T\s*=\s*uchar\s*;", output)) == 1
    assert "typename ConditionalType_" not in output
    assert "struct ConditionalType_1_uint_uchar" not in output
    assert "struct ConditionalType_0_uint_uchar" not in output


def test_preprocessor_deduces_partial_struct_specialization_pattern():
    code = """
    template <typename T>
    struct Trait {
        T value;
    };

    template <typename T>
    struct Trait<T*> {
        T pointed_value;
    };

    Trait<float*> specialized;
    Trait<float> primary;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Trait<float*> specialized;" in output
    assert "struct Trait_float {" in output
    assert "Trait_float primary;" in output


def test_preprocessor_matches_cv_qualified_partial_struct_specialization():
    code = """
    template <typename T>
    struct Trait {
        T value;
    };

    template <typename T>
    struct Trait<T const> {
        int value;
    };

    Trait<const float> specialized;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Trait<const float> specialized;" in output
    assert "struct Trait_const_float {" not in output


def test_preprocessor_merges_nested_static_constant_contexts():
    code = """
    template <int Width>
    struct Box {
        float values[Width];
    };

    struct Outer {
        static constexpr int width = 4;

        struct Inner {
            static constexpr int unrelated = 1;
            Box<width> box;
        };
    };
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Box_4 box;" in output
    assert "struct Box_4" in output
    assert "Box_width" not in output


def test_preprocessor_prunes_unreferenced_materialized_struct_templates():
    code = """
    template <int Rows>
    struct Tile {
        float values[Rows];
    };

    template <typename T, int Rows>
    struct Layout {
        Tile<Rows> tile;
        T value;
    };

    Layout<float, 4> layout;
    """
    preprocessor = MetalPreprocessor()

    materialized = preprocessor._materialize_explicit_template_struct_instantiations(
        code
    )
    output = preprocessor._prune_unreferenced_template_struct_declarations(materialized)

    assert "template <" not in output
    assert "struct Layout_float_4" in output
    assert "struct Tile_4" in output


def test_preprocessor_keeps_struct_templates_with_unresolved_specialization_use():
    code = """
    template <typename T>
    struct Trait {
        T value;
    };

    template <typename T>
    struct Trait<T*> {
        T pointed_value;
    };

    Trait<float*> specialized;
    """
    preprocessor = MetalPreprocessor()

    materialized = preprocessor._materialize_explicit_template_struct_instantiations(
        code
    )
    output = preprocessor._prune_unreferenced_template_struct_declarations(materialized)

    assert "template <typename T>" in output
    assert "struct Trait<T*>" in output
    assert "Trait<float*> specialized;" in output


def test_preprocessor_struct_materializer_ignores_non_template_structs():
    # Non-template structs are untouched; the materializer only rewrites concrete
    # references to struct templates.
    code = """
    struct Plain { float x; };

    [[kernel]] void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Plain p;
        p.x = out[i];
        out[i] = p.x;
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "struct Plain {" in output
    assert "Plain_" not in output


def test_preprocessor_lowers_method_with_elaborated_struct_return_type():
    code = """
    struct Result { int value; };

    struct Factory {
        struct Result make() {
            Result result;
            result.value = 4;
            return result;
        }
    };

    Factory factory;
    Result result = factory.make();
    """

    output = MetalPreprocessor().preprocess(code)

    assert "struct Result Factory__make(thread Factory& self)" in output
    assert "Result result = Factory__make(factory);" in output
    assert "factory.make()" not in output


def test_preprocessor_lowers_instance_method_with_member_reference():
    # CrossGL structs are data-only, so an instance member function is lowered to
    # a free function taking the receiver `self` by reference, and bare references
    # to the struct's data members inside the body are rewritten to `self.member`.
    code = """
    struct Adder { float bias; float add(float a, float b){ return a + b + bias; } };

    [[kernel]] void k(
        device const float* in [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        Adder adder;
        adder.bias = 1.0;
        out[i] = adder.add(in[i], 2.0);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    # The struct keeps only its data member; the method is gone from the struct.
    assert "struct Adder { float bias;" in output
    assert "float add(" not in output.split("struct Adder")[1].split("}")[0]
    # The method is re-emitted as a free function with a mutable `self` receiver
    # and the member reference qualified.
    assert "float Adder__add(thread Adder& self, float a, float b)" in output
    assert "a + b + self.bias" in output
    # The call site is rewritten to pass the receiver as the first argument.
    assert "Adder__add(adder, in[i], 2.0)" in output
    # A bare data-member access is left untouched.
    assert "adder.bias = 1.0;" in output
    # No dangling method call survives.
    assert "adder.add(" not in output


def test_preprocessor_mutating_method_uses_mutable_receiver_reference():
    code = """
    struct NestedState { int value; };

    struct State {
        int scalar;
        int values[2];
        NestedState nested;

        void mutate(int amount) {
            scalar += amount;
            values[1] = amount;
            nested.value += amount;
        }
    };

    void apply(thread State& state) {
        state.mutate(3);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "void State__mutate(thread State& self, int amount)" in output
    assert "self.scalar += amount;" in output
    assert "self.values[1] = amount;" in output
    assert "self.nested.value += amount;" in output
    assert "State__mutate(state, 3);" in output


def test_preprocessor_const_method_receiver_is_read_only():
    code = """
    struct State {
        int value;
        int read() const { return value; }
    };

    int inspect(thread const State& state) {
        return state.read();
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "int State__read(thread const State& self)" in output
    assert "return State__read(state);" in output
    assert "State__read(thread State& self)" not in output


def test_preprocessor_inlines_direct_reference_accessor_as_original_storage():
    code = """
    struct State {
        int scalar;
        int values[4];
        thread int& current() { return scalar; }
        thread int& element(int index) { return values[index]; }
        int read(int index) { return element(index); }
    };

    void mutate(thread int& value);
    void assign(thread State& state, int index) {
        state.current() = 9;
        state.element(index) += 2;
        ++state.element(index);
        mutate(state.element(index));
        thread int* address = &state.element(index);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "State__element" not in output
    assert "State__current" not in output
    assert "state.scalar = 9;" in output
    assert "state.values[(index)] += 2;" in output
    assert "++state.values[(index)];" in output
    assert "mutate(state.values[(index)]);" in output
    assert "thread int* address = &state.values[(index)];" in output
    assert "return self.values[(index)];" in output


def test_preprocessor_inlines_implicit_reference_accessor_in_helper_argument():
    code = """
    struct Fragment {
        float lanes[2];
    };

    struct FragmentOps {
        static void store(const thread Fragment& fragment, device float* output) {
            output[0] = fragment.lanes[0];
        }
    };

    struct Tile {
        Fragment val_frags[4];
        thread Fragment& frag_at(short i, short j) {
            return val_frags[i * 2 + j];
        }
        const thread Fragment& frag_at(short i, short j) const {
            return val_frags[i * 2 + j];
        }
        void store(short i, short j, device float* output) const {
            FragmentOps::store(frag_at(i, j), output);
        }
    };

    void run(thread Tile& tile, device float* output) {
        tile.store(0, 1, output);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "FragmentOps__store(self.val_frags[(i) * 2 + (j)], output);" in output
    assert "frag_at(i, j)" not in output
    assert "Tile__frag_at" not in output


def test_preprocessor_rejects_unsafe_implicit_reference_accessor_argument():
    code = """
    struct Fragment {
        float lanes[2];
    };

    struct FragmentOps {
        static void store(thread Fragment& fragment, device float* output) {
            output[0] = fragment.lanes[0];
        }
    };

    struct Tile {
        Fragment val_frags[4];
        thread Fragment& frag_at(short i, short j) {
            return val_frags[i * 2 + j];
        }
        void store(short i, short j, device float* output) {
            FragmentOps::store(frag_at(i++, j), output);
        }
    };

    void run(thread Tile& tile, device float* output) {
        tile.store(0, 1, output);
    }
    """

    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)

    error = excinfo.value
    assert error.project_diagnostic_code == "project.translate.metal-struct-method"
    assert error.missing_capabilities == ("struct.reference-return",)
    assert error.struct_name == "Tile"
    assert error.method_name == "frag_at"
    assert error.requested_signature == "Tile::frag_at(i++, j)"
    assert error.reason == "reference-return-identity-unsupported"
    assert "preserving the returned lvalue identity" in str(error)


@pytest.mark.parametrize(
    "alias_declarations",
    [
        pytest.param(
            "using tile_alias = Tile;\nusing tile_type = tile_alias;",
            id="using-chain",
        ),
        pytest.param(
            "typedef Tile tile_alias;\ntypedef tile_alias tile_type;",
            id="typedef-chain",
        ),
    ],
)
def test_preprocessor_inlines_direct_reference_accessor_through_alias_chain(
    alias_declarations,
):
    code = f"""
    struct Fragment {{ float lanes[2]; }};

    struct Tile {{
        using frag_type = Fragment;
        frag_type val_frags[4];
        thread frag_type& frag_at(short i, short j) {{
            return val_frags[i * 2 + j];
        }}
        const thread frag_type& frag_at(short i, short j) const {{
            return val_frags[i * 2 + j];
        }}
    }};

    {alias_declarations}
    void assign(thread tile_type& tile, short i, short j) {{
        tile.frag_at(i, j).lanes[0] = 1.0f;
    }}
    """

    output = MetalPreprocessor().preprocess(code)

    assert "tile.val_frags[(i) * 2 + (j)].lanes[0] = 1.0f;" in output
    assert "tile.frag_at(" not in output
    assert "Tile__frag_at" not in output


def test_preprocessor_inlines_const_reference_accessor_through_alias_chain():
    code = """
    struct Fragment { float lanes[2]; };

    struct Tile {
        using frag_type = Fragment;
        frag_type val_frags[4];
        thread frag_type& frag_at(short i, short j) {
            return val_frags[i * 2 + j];
        }
        const thread frag_type& frag_at(short i, short j) const {
            return val_frags[i * 2 + j];
        }
    };

    using tile_alias = Tile;
    using tile_type = tile_alias;
    float inspect(thread const tile_type& tile, short i, short j) {
        float result = tile.frag_at(i, j).lanes[0];
        return result;
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "float result = tile.val_frags[(i) * 2 + (j)].lanes[0];" in output
    assert "tile.frag_at(" not in output
    assert "Tile__frag_at" not in output


def test_preprocessor_constant_reference_accessor_receiver_fails_closed():
    code = """
    struct Fragment { float lanes[2]; };

    struct Tile {
        using frag_type = Fragment;
        frag_type val_frags[4];
        thread frag_type& frag_at(short i, short j) {
            return val_frags[i * 2 + j];
        }
        const thread frag_type& frag_at(short i, short j) const {
            return val_frags[i * 2 + j];
        }
    };

    using tile_alias = Tile;
    using tile_type = tile_alias;
    float inspect(constant tile_type& tile, short i, short j) {
        float result = tile.frag_at(i, j).lanes[0];
        return result;
    }
    """

    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)

    error = excinfo.value
    assert error.project_diagnostic_code == "project.translate.metal-struct-method"
    assert error.missing_capabilities == ("struct.reference-return",)
    assert error.struct_name == "Tile"
    assert error.method_name == "frag_at"
    assert error.requested_signature == "Tile::frag_at(i, j)"
    assert error.reason == "reference-return-identity-unsupported"
    assert "preserving the returned lvalue identity" in str(error)
    assert error.source_location["length"] > 0


def test_preprocessor_substitutes_nested_const_reference_alias_uses():
    code = """
    struct Tile {
        float2 val_frags[4];
        thread float2& frag_at(short i, short j) {
            return val_frags[i * 2 + j];
        }
        const thread float2& frag_at(short i, short j) const {
            return val_frags[i * 2 + j];
        }
    };

    struct StoreLoop {
        Tile Ctile;
        void store(short i, short j, device float* output) const {
            thread const auto& accum = Ctile.frag_at(i, j);
            for (short k = 0; k < 2; ++k) {
                output[k] = accum[k];
            }
        }
    };

    void run(thread const StoreLoop& loop, device float* output) {
        loop.store(0, 1, output);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "const auto& accum" not in output
    assert "frag_at(i, j)" not in output
    assert "output[k] = self.Ctile.val_frags[(i) * 2 + (j)][k];" in output


def _nested_const_reference_alias_source(
    *,
    accessor_arguments="i, j",
    alias_lifetime="output[0] = accum[0];",
    alias_setup="",
    accessor_declarations=None,
    supporting_declarations="",
    tile_member="Tile Ctile;",
    accessor_receiver="Ctile",
):
    if accessor_declarations is None:
        accessor_declarations = """
        thread float2& frag_at(short i, short j) {
            return val_frags[i * 2 + j];
        }
        const thread float2& frag_at(short i, short j) const {
            return val_frags[i * 2 + j];
        }
        """
    accessor_call = (
        f"{accessor_receiver}frag_at"
        if accessor_receiver.endswith("->")
        else f"{accessor_receiver}.frag_at"
    )
    return f"""
    struct Tile {{
        float2 val_frags[4];
        {accessor_declarations}
    }};
    struct TileCarrier {{ thread Tile* tile; }};
    {supporting_declarations}
    struct StoreLoop {{
        {tile_member}
        void store(short i, short j, device float* output) const {{
            {alias_setup}
            thread const auto& accum =
                {accessor_call}({accessor_arguments});
            {alias_lifetime}
        }}
    }};
    void run(thread const StoreLoop& loop, device float* output) {{
        loop.store(0, 1, output);
    }}
    """


def test_preprocessor_substitutes_nested_const_reference_alias_through_value_chain():
    code = _nested_const_reference_alias_source(
        supporting_declarations="struct TileLayer { Tile tile; };",
        tile_member="TileLayer layer;",
        accessor_receiver="layer.tile",
        alias_lifetime="""
        output[0] = accum[0];
        output[1] = accum[1];
        """,
    )

    output = MetalPreprocessor().preprocess(code)

    assert "const auto& accum" not in output
    assert "layer.tile.frag_at(i, j)" not in output
    assert "output[0] = self.layer.tile.val_frags[(i) * 2 + (j)][0];" in output
    assert "output[1] = self.layer.tile.val_frags[(i) * 2 + (j)][1];" in output


def test_preprocessor_substitutes_sequential_nested_const_reference_aliases():
    code = _nested_const_reference_alias_source(
        alias_lifetime="""
        thread const auto& alternate = Ctile.frag_at(j, i);
        output[0] = accum[0];
        output[1] = alternate[1];
        """,
    )

    output = MetalPreprocessor().preprocess(code)

    assert "const auto& accum" not in output
    assert "const auto& alternate" not in output
    assert "output[0] = self.Ctile.val_frags[(i) * 2 + (j)][0];" in output
    assert "output[1] = self.Ctile.val_frags[(j) * 2 + (i)][1];" in output


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            _nested_const_reference_alias_source(
                tile_member="TileCarrier carrier;",
                accessor_receiver="carrier.tile->",
            ),
            id="pointer-traversal",
        ),
        pytest.param(
            _nested_const_reference_alias_source(
                accessor_declarations="""
                const thread float2& frag_at(short i, short j) const {
                    return val_frags[i * 2 + j];
                }
                const thread float2& frag_at(int i, int j) const {
                    return val_frags[i * 2 + j];
                }
                """,
            ),
            id="ambiguous-overload",
        ),
        pytest.param(
            _nested_const_reference_alias_source(
                alias_lifetime="""
                {
                    float2 accum = float2(0.0f);
                    output[0] = accum[0];
                }
                """,
            ),
            id="shadowed-alias",
        ),
        pytest.param(
            _nested_const_reference_alias_source(
                alias_lifetime="thread const float2* escaped = &accum;",
            ),
            id="reference-escape",
        ),
        pytest.param(
            _nested_const_reference_alias_source(accessor_arguments="i++, j"),
            id="side-effectful-argument",
        ),
        pytest.param(
            _nested_const_reference_alias_source(
                alias_lifetime="i += 1; output[0] = accum[0];",
            ),
            id="captured-index-mutation",
        ),
        pytest.param(
            _nested_const_reference_alias_source(
                alias_lifetime=(
                    "thread short& index_alias = i; "
                    "index_alias += 1; output[0] = accum[0];"
                ),
            ),
            id="captured-index-reference",
        ),
        pytest.param(
            _nested_const_reference_alias_source(
                alias_lifetime="""
                {
                    short i = 0;
                    output[0] = accum[0];
                }
                """,
            ),
            id="captured-index-shadow",
        ),
        pytest.param(
            _nested_const_reference_alias_source(
                alias_setup="thread short& index_alias = i;",
                alias_lifetime="index_alias += 1; output[0] = accum[0];",
            ),
            id="preexisting-captured-index-reference",
        ),
        pytest.param(
            _nested_const_reference_alias_source(
                accessor_arguments="index.row, j",
                alias_setup="Index index{ i };",
                alias_lifetime="index.row += 1; output[0] = accum[0];",
                supporting_declarations="struct Index { short row; };",
            ),
            id="captured-index-member-mutation",
        ),
        pytest.param(
            _nested_const_reference_alias_source(
                alias_lifetime="bump(i); output[0] = accum[0];",
                supporting_declarations="void bump(thread short& value);",
            ),
            id="captured-index-reference-argument",
        ),
    ],
)
def test_preprocessor_nested_const_reference_alias_fails_closed(code):
    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)

    error = excinfo.value
    assert error.project_diagnostic_code == "project.translate.metal-struct-method"
    assert error.missing_capabilities == ("struct.reference-return",)
    assert error.method_name == "frag_at"
    assert error.reason == "reference-return-identity-unsupported"
    assert "preserving the returned lvalue identity" in str(error)


def test_preprocessor_leaves_unknown_struct_member_calls_unchanged():
    code = """
    void invoke(thread ExternalState& state) {
        state.update();
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "state.update();" in output


@pytest.mark.parametrize(
    ("code", "struct_name", "method_name", "signature", "return_type"),
    [
        pytest.param(
            """
            struct Leaf {
                int value;
                thread int& element() { return value; }
            };
            struct Holder { Leaf leaf; };
            void assign(thread Holder& holder) { holder.leaf.element() = 1; }
            """,
            "Leaf",
            "element",
            "Leaf::element()",
            "thread int&",
            id="nested-receiver",
        ),
        pytest.param(
            """
            struct State {
                int value;
                thread int& element() { return value; }
            };
            void assign(thread State* state) { state->element() = 1; }
            """,
            "State",
            "element",
            "State::element()",
            "thread int&",
            id="pointer-receiver",
        ),
        pytest.param(
            """
            struct State {
                static thread int& element(thread int& value) { return value; }
            };
            void assign(thread int& value) { State::element(value) = 1; }
            """,
            "State",
            "element",
            "State::element(value)",
            "thread int&",
            id="concrete-static",
        ),
        pytest.param(
            """
            struct State {
                template <typename T>
                thread T& element(thread T& value) { return value; }
            };
            void assign(thread State& state) {
                int value = 0;
                state.element(value) = 1;
            }
            """,
            "State",
            "element",
            "state.element(value)",
            "thread int&",
            id="template-member",
        ),
        pytest.param(
            """
            struct Select {
                thread int& operator()(thread int& value) { return value; }
            };
            void assign(thread int& value) { Select{}(value) = 1; }
            """,
            "Select",
            "operator()",
            "Select::operator()(value)",
            "thread int&",
            id="temporary-functor",
        ),
        pytest.param(
            """
            struct State {
                int value;
                thread int& operator()(int index) { return value; }
                int read() { return operator()(0); }
            };
            int inspect(thread State& state) { return state.read(); }
            """,
            "State",
            "operator()",
            "State::operator()(0)",
            "thread int&",
            id="internal-operator",
        ),
        pytest.param(
            """
            struct Index { int value; };
            struct State {
                int values[2];
                template <int I>
                thread int& element() { return values[I]; }
                int read() {
                    Index index;
                    return element<index.value>();
                }
            };
            int inspect(thread State& state) { return state.read(); }
            """,
            "State",
            "element",
            "State::element<index.value>()",
            "thread int&",
            id="runtime-value-template",
        ),
        pytest.param(
            """
            struct View {
                device int* values;
                View(device int* values_) : values(values_) {}
                device int& element(uint index) { return values[index]; }
            };
            kernel void assign(device int* values [[buffer(0)]]) {
                View view(values);
                view.element(0) = 1;
            }
            """,
            "View",
            "element",
            "View::element(0)",
            "device int&",
            id="promoted-receiver",
        ),
        pytest.param(
            """
            struct State {
                int values[2];
                thread int& element(int index) { return values[index]; }
            };
            void assign(thread State& state, int index) {
                state.element(index++) = 1;
            }
            """,
            "State",
            "element",
            "State::element(index++)",
            "thread int&",
            id="side-effectful-argument",
        ),
        pytest.param(
            """
            struct State {
                int values[2];
                thread int& element(int index) { return values[index]; }
            };
            void bind(thread State& state) {
                thread int& alias = state.element(0);
            }
            """,
            "State",
            "element",
            "State::element(0)",
            "thread int&",
            id="reference-binding",
        ),
        pytest.param(
            """
            struct State {
                int value;
                thread int& element() { return value; }
                const thread int& element() const { return value; }
            };
            int inspect(thread const State& state) { return state.element(); }
            """,
            "State",
            "element",
            "State::element()",
            "const thread int&",
            id="const-receiver",
        ),
        pytest.param(
            """
            struct State {
                int value;
                const thread int& element() { return value; }
            };
            int inspect(thread State& state) { return state.element(); }
            """,
            "State",
            "element",
            "State::element()",
            "const thread int&",
            id="const-reference-return",
        ),
        pytest.param(
            """
            struct State {
                int value;
                thread int& element() { return value; }
            };
            void assign(constant State& state) { state.element() = 1; }
            """,
            "State",
            "element",
            "State::element()",
            "thread int&",
            id="constant-address-space-receiver",
        ),
        pytest.param(
            """
            struct State {
                int values[2];
                thread int& element(int index) { return values[index]; }
                int element(float index) { return int(index); }
            };
            int inspect(thread State& state, float index) {
                return state.element(index);
            }
            """,
            "State",
            "element",
            "State::element(index)",
            "thread int&",
            id="same-arity-value-overload",
        ),
        pytest.param(
            """
            struct State {
                int values[2];
                thread int& element(int index) { return values[index]; }
                template <typename T>
                int element(T index) { return int(index); }
            };
            int inspect(thread State& state, float index) {
                return state.element(index);
            }
            """,
            "State",
            "element",
            "State::element(index)",
            "thread int&",
            id="same-arity-template-overload",
        ),
    ],
)
def test_preprocessor_called_reference_returning_paths_fail_closed(
    code, struct_name, method_name, signature, return_type
):
    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)

    error = excinfo.value
    assert error.project_diagnostic_code == "project.translate.metal-struct-method"
    assert error.missing_capabilities == ("struct.reference-return",)
    assert error.struct_name == struct_name
    assert error.method_name == method_name
    assert error.requested_signature == signature
    assert error.reason == "reference-return-identity-unsupported"
    assert error.return_type == return_type
    assert error.suggested_action
    assert "preserving the returned lvalue identity" in str(error)
    assert error.source_location["length"] > 0


@pytest.mark.parametrize(
    ("method", "expected_helper"),
    [
        pytest.param(
            "thread int& element() { return value; }",
            None,
            id="concrete",
        ),
        pytest.param(
            "template <typename T> "
            "thread T& element(thread T& value) { return value; }",
            None,
            id="template",
        ),
    ],
)
def test_preprocessor_uncalled_reference_returning_declarations_are_accepted(
    method, expected_helper
):
    code = f"struct State {{ int value; {method} }};\nState state;\n"

    output = MetalPreprocessor().preprocess(code)

    assert "struct State { int value;" in output
    assert "State state;" in output
    if expected_helper is None:
        assert "State__element" not in output
    else:
        assert expected_helper in output


def test_preprocessor_lowers_static_method_without_self_parameter():
    # A static member function is lowered to a free function with NO `self`
    # parameter, and a qualified `S::m(args)` call becomes `S__m(args)`.
    code = """
    struct Maths { static float twice(float a){ return a * 2.0; } };

    [[kernel]] void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        out[i] = Maths::twice(out[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Maths__twice(float a)" in output
    # No `self` is introduced for a static method.
    assert re.search(r"\bMaths\s*(?:&\s*)?self\b", output) is None
    assert "Maths__twice(out[i])" in output
    assert "Maths::twice" not in output


def test_preprocessor_lowers_call_operator_functor():
    # An `operator()` functor is lowered to a free function named
    # `S__operator_call`, and a `var(args)` call where `var` has type S is
    # rewritten to `S__operator_call(var, args)`.
    code = """
    struct Sum { float operator()(float a, float b){ return a + b; } };

    [[kernel]] void k(
        device const float* in [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        out[i] = op(in[i], out[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Sum__operator_call(thread Sum& self, float a, float b)" in output
    assert "Sum__operator_call(op, in[i], out[i])" in output


def test_preprocessor_lowers_materialized_template_functor():
    # After the struct-template materializer produces a concrete `Sum_float`, the
    # member-function lowering pass lowers its `operator()` and rewrites the call.
    code = """
    template <typename U>
    struct Sum { static constexpr constant U init = U(0); U operator()(U a, U b){ return a + b; } };

    [[kernel]] void k(
        device const float* in [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        Sum<float> op;
        float acc = op(in[i], out[i]);
        out[i] = acc;
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "struct Sum_float {" in output
    assert (
        "float Sum_float__operator_call(thread Sum_float& self, float a, float b)"
        in output
    )
    assert "Sum_float__operator_call(op, in[i], out[i])" in output
    # The primary template is left untouched (template methods are out of scope).
    assert "template <typename U>" in output
    # No dangling functor call survives on the concrete instance.
    assert "= op(" not in output


def test_preprocessor_substitutes_materialized_static_bool_and_int_constants():
    code = """
    template <int N>
    struct Reduction {
        static constant bool needs_reduction = N > 1;
        static constexpr int lane_count = N * 2;
        static constant int adjusted_count = base_count + 1;

        int apply(int value) const {
            return needs_reduction ? value + lane_count + adjusted_count : value;
        }
    };

    constant int base_count = 4;

    [[kernel]] void k(device int* out [[buffer(0)]]) {
        Reduction<1> reduction;
        out[0] = reduction.apply(out[0]);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "int Reduction_1__apply(thread const Reduction_1& self, int value)" in output
    assert "return (false) ? value + (2) + (base_count + 1) : value;" in output
    assert "Reduction_1__apply(reduction, out[0])" in output


def test_preprocessor_resolves_transitive_static_initializer_dependencies():
    code = """
    template <int N>
    struct BlockShape {
        static constant int threadsM = N * 8;
        static constant int blockM = threadsM * 4;

        int block_size() const {
            return blockM;
        }
    };

    [[kernel]] void k(device int* out [[buffer(0)]]) {
        BlockShape<1> shape;
        out[0] = shape.block_size();
    }
    """

    output = MetalPreprocessor().preprocess(code)
    lowered = output.split("int BlockShape_1__block_size", 1)[1].split("}", 1)[0]

    assert "return (32);" in lowered
    assert "threadsM" not in lowered
    assert "blockM" not in lowered


def test_preprocessor_static_initializer_substitution_respects_shadowing_and_context():
    code = """
    struct Other { int value; };

    struct Constants {
        static constant int value = 2 + 3;
        int instance_value = 11;

        int parameter_shadow(int value) const {
            return value + instance_value;
        }

        int local_shadow() const {
            int value = 9;
            return value + instance_value;
        }

        int read(Other other) const {
            // keep value in comment
            const char* label = "value";
            return value + other.value + instance_value;
        }
    };
    """

    output = MetalPreprocessor()._lower_struct_member_functions(code)

    assert "return value + self.instance_value;" in output
    assert "int value = 9;\n            return value + self.instance_value;" in output
    assert "// keep value in comment" in output
    assert 'const char* label = "value";' in output
    assert "return (5) + other.value + self.instance_value;" in output
    assert "(11)" not in output


def test_preprocessor_materialized_template_struct_constructor_is_left_in_place():
    # Regression: a materialized template struct (`ReadWriter<T, U>` ->
    # `ReadWriter_float_float`) keeps a CONSTRUCTOR that still carries the
    # ORIGINAL template name `ReadWriter`, because only the struct header is
    # renamed. The `method_name == struct_name` constructor check therefore
    # cannot see it, and its sole "return type" is the qualifier macro
    # `METAL_FUNC`. Such a constructor must be LEFT IN PLACE like every other
    # constructor rather than lowered to a malformed
    # `METAL_FUNC ReadWriter_float_float__ReadWriter(...)` free function, which
    # previously made the materialized source fail to parse with
    # "Unexpected token in expression: CONST".
    code = """
    template <typename T, typename U>
    struct ReadWriter {
        const device T* in;
        int n;

        METAL_FUNC ReadWriter(const device T* in_, const int n_) : in(in_), n(n_) {}

        METAL_FUNC U read() const { return in[n]; }
    };

    [[kernel]] void k(
        const device float* src [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        ReadWriter<float, float> rw(src, 3);
        out[i] = rw.read();
    }
    """
    output = MetalPreprocessor().preprocess(code)
    # The template struct is materialized to a concrete struct.
    assert "struct ReadWriter_float_float {" in output
    # The struct carries a device pointer member, so it takes the pointer-member
    # promotion path: `in` leaves the struct and is threaded through `read` as a
    # promoted pointer parameter, while the scalar `self` still carries `n`.
    assert (
        "ReadWriter_float_float__read(thread const ReadWriter_float_float& self, "
        "const device float* crosstl_ptr_in)" in output
    )
    # The constructor is NOT lowered to a macro-typed free function; it is left
    # in place (rewritten to drop the promoted pointer parameter `in_`).
    assert "ReadWriter_float_float__ReadWriter(" not in output
    assert "METAL_FUNC ReadWriter_float_float__ReadWriter" not in output
    # The construction drops the pointer argument and the call forwards it.
    assert "ReadWriter_float_float rw(3)" in output
    assert "ReadWriter_float_float__read(rw, src)" in output
    # The materialized source parses cleanly (the original bug crashed here).
    ast = MetalParser(MetalLexer(output).tokenize()).parse()
    assert ast is not None


_POINTER_MEMBER_STRUCT = """
using namespace metal;

struct Rw {
  threadgroup float* buf;
  const device float* in;
  device float* out;
  int n;
  int total;

  Rw(threadgroup float* buf_,
     const device float* in_,
     device float* out_,
     const int n_)
      : buf(buf_), in(in_), out(out_), n(n_) {
    total = n_ * 2;
  }

  float load(int i) const {
    return buf[i] + in[i] + float(n) + float(total);
  }

  void store(int i) const {
    out[i] = buf[i];
  }
};

[[kernel]] void k(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]]) {
  threadgroup float shared[64];
  Rw rw = Rw(&shared[0], input, output, 8);
  float x = rw.load(2);
  rw.store(3);
  output[0] = x;
}
"""


def test_pointer_member_promotion_removes_pointer_members_from_struct():
    # A struct carrying device/threadgroup pointer members cannot be passed by
    # value under SPIR-V logical addressing, so its pointer members are promoted
    # OUT of the struct: the residual struct is a pure scalar aggregate and the
    # constructor drops the pointer parameters (and their initializer entries)
    # while still computing the scalar members.
    output = MetalPreprocessor().preprocess(_POINTER_MEMBER_STRUCT)
    struct_body = output.split("struct Rw {", 1)[1].split("};", 1)[0]
    # Scalar members remain; pointer members are gone.
    assert "int n;" in struct_body
    assert "int total;" in struct_body
    assert "float* buf;" not in struct_body
    assert "float* in;" not in struct_body
    assert "float* out;" not in struct_body
    # The constructor keeps only the scalar parameter and its computation.
    assert "Rw(const int n_) : n(n_)" in struct_body
    assert "total = n_ * 2;" in struct_body


def test_pointer_member_promotion_threads_pointers_through_methods():
    # Each instance method is lowered to a free function that takes `self`
    # followed by the promoted pointer parameters; the body references pointer
    # members through the (uniquely prefixed) parameters and scalar members
    # through `self`.
    output = MetalPreprocessor().preprocess(_POINTER_MEMBER_STRUCT)
    assert (
        "float Rw__load(thread const Rw& self, threadgroup float* crosstl_ptr_buf, "
        "const device float* crosstl_ptr_in, device float* crosstl_ptr_out, "
        "int i)" in output
    )
    assert (
        "return crosstl_ptr_buf[i] + crosstl_ptr_in[i] + float(self.n) "
        "+ float(self.total);" in output
    )
    assert (
        "void Rw__store(thread const Rw& self, threadgroup float* crosstl_ptr_buf, "
        "const device float* crosstl_ptr_in, device float* crosstl_ptr_out, "
        "int i)" in output
    )
    assert "crosstl_ptr_out[i] = crosstl_ptr_buf[i];" in output


def test_pointer_member_promotion_resolves_owner_alias_casts():
    code = """
    struct Writer {
      using elem_type = float;
      device elem_type* output;

      Writer(device elem_type* output_) : output(output_) {}

      device elem_type* data() const {
        return (device elem_type*)output;
      }
    };

    [[kernel]] void k(device float* result [[buffer(0)]]) {
      Writer writer(result);
      writer.data()[0] = 1.0f;
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert (
        "device float* Writer__data(thread const Writer& self, "
        "device float* crosstl_ptr_output)" in output
    )
    assert "return (device float*)crosstl_ptr_output;" in output
    assert "Writer__data(writer, result)" in output


def test_pointer_member_promotion_rewrites_construction_and_calls():
    # The construction drops the pointer arguments (the scalar `8` remains) and
    # each method call forwards the construction's pointer expressions in
    # declaration order.
    output = MetalPreprocessor().preprocess(_POINTER_MEMBER_STRUCT)
    assert "Rw rw = Rw(8);" in output
    assert "Rw__load(rw, &shared[0], input, output, 2)" in output
    assert "Rw__store(rw, &shared[0], input, output, 3)" in output
    # The lowered source parses cleanly.
    ast = MetalParser(MetalLexer(output).tokenize()).parse()
    assert ast is not None


def test_pointer_member_promotion_rewrites_internal_method_calls():
    code = """
    struct Writer {
      device float* output;
      int offset;

      Writer(device float* output_, int offset_)
          : output(output_), offset(offset_) {}

      void store(int index, float value) {
        output[offset + index] = value;
      }

      void store_pair(int index, float value) {
        store(index, value);
        this->store(index + 1, value);
      }
    };

    kernel void k(device float* output [[buffer(0)]]) {
      Writer writer(output, 2);
      writer.store_pair(0, 1.0f);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    forwarded_call = "Writer__store(self, crosstl_ptr_output, index, value);"
    assert output.count(forwarded_call) == 1
    assert "Writer__store(self, crosstl_ptr_output, index + 1, value);" in output
    assert "Writer__store_pair(writer, output, 0, 1.0f);" in output
    assert MetalParser(MetalLexer(output).tokenize()).parse() is not None


def test_pointer_member_promotion_via_using_alias():
    # A struct constructed through a local `using` alias
    # (`using T = Struct; T v = T(...)`) is still promoted, and the alias
    # construction drops the pointer arguments.
    code = """
    using namespace metal;

    struct Rw {
      threadgroup float* buf;
      const device float* in;
      int n;
      Rw(threadgroup float* buf_, const device float* in_, const int n_)
          : buf(buf_), in(in_), n(n_) {}
      float load(int i) const { return buf[i] + in[i] + float(n); }
    };

    [[kernel]] void k(
        const device float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint tid [[thread_position_in_grid]]) {
      threadgroup float shared[64];
      using rw_t = Rw;
      rw_t rw = rw_t(&shared[0], input, 8);
      output[tid] = rw.load(int(tid));
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "buf;" not in output.split("struct Rw {", 1)[1].split("};", 1)[0]
    assert "rw_t rw = rw_t(8);" in output
    assert "Rw__load(rw, &shared[0], input, int(tid))" in output


def test_pointer_member_promotion_bails_when_pointer_not_from_ctor_parameter():
    # If a pointer member is NOT sourced directly from a constructor parameter,
    # the construction cannot supply its expression, so the struct is left on the
    # ordinary lowering path (its pointer member stays in the struct).
    code = """
    using namespace metal;

    struct Holder {
      device float* out;
      int n;
      Holder(const int n_) : n(n_) { out = nullptr; }
      float read(int i) const { return out[i] + float(n); }
    };

    [[kernel]] void k(device float* dst [[buffer(0)]]) {
      Holder h = Holder(4);
      dst[0] = h.read(1);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    # `out` stays a member (not promoted) because it is not a constructor
    # parameter; the method keeps the ordinary reference-receiver signature and
    # accesses the pointer through `self.out`.
    assert "device float* out;" in output
    assert "Holder__read(thread const Holder& self, int i)" in output
    assert "self.out[i]" in output
    assert "crosstl_ptr_out" not in output


def test_pointer_member_promotion_leaves_pointerless_structs_unchanged():
    # A struct without pointer members is untouched by the promotion path: it is
    # lowered with no promoted parameters.
    code = """
    using namespace metal;

    struct Adder {
      int bias;
      Adder(const int bias_) : bias(bias_) {}
      int apply(int x) const { return x + bias; }
    };

    [[kernel]] void k(device int* out [[buffer(0)]]) {
      Adder a = Adder(5);
      out[0] = a.apply(3);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "int Adder__apply(thread const Adder& self, int x)" in output
    assert "Adder__apply(a, 3)" in output
    assert "crosstl_ptr" not in output


def test_pointer_member_promotion_resolves_receiver_per_construction():
    # Two kernels reuse the receiver name `rw` for different pointer expressions;
    # each call forwards the pointer expression from its NEAREST preceding
    # construction.
    code = """
    using namespace metal;

    struct Rw {
      const device float* in;
      int n;
      Rw(const device float* in_, const int n_) : in(in_), n(n_) {}
      float load(int i) const { return in[i] + float(n); }
    };

    [[kernel]] void k1(const device float* a [[buffer(0)]], device float* o [[buffer(1)]]) {
      Rw rw = Rw(a, 1);
      o[0] = rw.load(0);
    }

    [[kernel]] void k2(const device float* b [[buffer(0)]], device float* o [[buffer(1)]]) {
      Rw rw = Rw(b, 2);
      o[0] = rw.load(0);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "Rw__load(rw, a, 0)" in output
    assert "Rw__load(rw, b, 0)" in output


def test_preprocessor_lowering_qualifies_member_only_when_not_shadowed():
    # A member reference shadowed by a parameter or a local stays bare; the
    # `this->member` spelling is always rewritten to `self.member`.
    code = """
    struct C { float scale; float apply(float scale){ return this->scale * scale; } };

    [[kernel]] void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        C c;
        c.scale = 2.0;
        out[i] = c.apply(out[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    # `this->scale` -> `self.scale`; the parameter `scale` is NOT rewritten.
    assert "return self.scale * scale;" in output
    assert "C__apply(c, out[i])" in output


def test_preprocessor_lowering_is_noop_without_struct_methods():
    # Regression-safety: a struct with no member functions (and the surrounding
    # kernel) must be byte-identical after the lowering pass.
    code = (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct Plain { float x; float y; };\n"
        "kernel void k(device float* o [[buffer(0)]],"
        " uint i [[thread_position_in_grid]]) {\n"
        "    Plain p;\n"
        "    p.x = 1.0;\n"
        "    o[i] = p.x + p.y;\n"
        "}\n"
    )
    preprocessor = MetalPreprocessor()
    assert preprocessor._lower_struct_member_functions(code) == code


def test_preprocessor_lowering_leaves_method_free_source_untouched():
    # Regression-safety: a kernel with no structs at all is byte-identical.
    code = (
        "kernel void k(device float* o [[buffer(0)]],"
        " uint i [[thread_position_in_grid]]) {\n"
        "    o[i] = o[i] * 2.0;\n"
        "}\n"
    )
    preprocessor = MetalPreprocessor()
    assert preprocessor._lower_struct_member_functions(code) == code


def test_preprocessor_instantiates_template_method_from_buffer_element_arg():
    # A CALLED template member method is instantiated from a buffer-element
    # argument (`in[i]` whose element type is float) and lowered to a concrete
    # free function `S__m__float(thread S& self, float val)`; the call is rewritten.
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val + val; } };

    kernel void k(
        device const float* in [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        out[i] = op.reduce(in[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Sum__reduce__float(thread Sum& self, float val)" in output
    assert "Sum__reduce__float(op, in[i])" in output
    # The struct is data-only; the template method is gone from it.
    assert "template" not in output.split("struct Sum")[1].split("}")[0]
    # No dangling template-method call survives.
    assert "op.reduce(" not in output


def test_preprocessor_instantiates_template_method_from_pointer_arguments():
    code = """
    struct Tile {
      template <typename U>
      void store(device U* dst, int ld) { dst[0] = dst[ld]; }
      template <typename U>
      void load(const device U* src, int ld) { U value = src[ld]; }
    };

    kernel void k(
        device half* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        const constant int& offset [[buffer(2)]]) {
        Tile tile;
        tile.store(output, 1);
        tile.load(input + offset, 1);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Tile__store__float(tile, output, 1)" in output
    assert "Tile__load__half(tile, input + offset, 1)" in output
    assert "tile.store(" not in output
    assert "tile.load(" not in output


def test_preprocessor_infers_addressed_buffer_element_pointer_for_nested_call():
    code = """
    struct Fragment {
      template <typename SrcPtrType>
      static float load(SrcPtrType src) { return float(src[0]); }
    };

    struct Tile {
      template <typename U>
      float load(const device U* src, int i, int j) {
        return Fragment::load(&(src[(i * 8) * 36 + (j * 8)]));
      }
    };

    kernel void k(
        const device half* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
      Tile tile;
      output[index] = tile.load(input, int(index), 1);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert (
        "float Fragment__load__const_device_half_ptr(const device half* src)" in output
    )
    assert (
        "Fragment__load__const_device_half_ptr("
        "&(src[(i * 8) * 36 + (j * 8)]))" in output
    )
    assert "Tile__load__half(tile, input, int(index), 1)" in output
    assert "Fragment::load(" not in output


def test_preprocessor_preserves_const_threadgroup_pointer_in_nested_call():
    code = """
    struct Fragment {
      template <typename SrcPtrType>
      static float load(SrcPtrType src) { return float(src[0]); }
    };

    struct Tile {
      template <typename U>
      float load(const threadgroup U* src, int index) {
        return Fragment::load(&(src[index]));
      }
    };

    kernel void k(
        device float* output [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
      threadgroup half shared_values[64];
      Tile tile;
      output[index] = tile.load(shared_values, int(index));
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert (
        "float Fragment__load__const_threadgroup_half_ptr("
        "const threadgroup half* src)" in output
    )
    assert "Fragment__load__const_threadgroup_half_ptr(&(src[index]))" in output
    assert "Tile__load__half(tile, shared_values, int(index))" in output


def test_preprocessor_preserves_generic_pointer_argument_expressions():
    code = """
    struct Loader {
      template <typename Pointer>
      float load(Pointer src) { return float(src[0]); }
    };

    kernel void k(
        const device half* src [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint offset [[thread_position_in_grid]]) {
      Loader loader;
      out[0] = loader.load(src);
      out[1] = loader.load(src + offset);
      out[2] = loader.load(offset + src);
      out[3] = loader.load(src - offset);
      out[4] = loader.load(&(src[offset]));
    }
    """

    output = MetalPreprocessor().preprocess(code)

    helper = (
        "float Loader__load__const_device_half_ptr("
        "thread Loader& self, const device half* src)"
    )
    assert output.count(helper) == 1
    assert "Loader__load__const_device_half_ptr(loader, src)" in output
    assert "Loader__load__const_device_half_ptr(loader, src + offset)" in output
    assert "Loader__load__const_device_half_ptr(loader, offset + src)" in output
    assert "Loader__load__const_device_half_ptr(loader, src - offset)" in output
    assert "Loader__load__const_device_half_ptr(loader, &(src[offset]))" in output
    assert "Loader__load__half(" not in output


def test_preprocessor_preserves_addressed_pointer_through_auto_locals():
    code = """
    struct Loader {
      template <typename Pointer>
      float load(Pointer src) { return float(src[0]); }
    };

    kernel void k(
        const device half* src [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint offset [[thread_position_in_grid]]) {
      Loader loader;
      auto cast_ptr = static_cast<const device half*>(src);
      auto offset_ptr = cast_ptr + offset;
      auto copied_ptr = offset_ptr;
      out[0] = loader.load(cast_ptr);
      out[1] = loader.load(offset_ptr);
      out[2] = loader.load(copied_ptr);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    helper = (
        "float Loader__load__const_device_half_ptr("
        "thread Loader& self, const device half* src)"
    )
    assert output.count(helper) == 1
    assert "Loader__load__const_device_half_ptr(loader, cast_ptr)" in output
    assert "Loader__load__const_device_half_ptr(loader, offset_ptr)" in output
    assert "Loader__load__const_device_half_ptr(loader, copied_ptr)" in output
    assert "Loader__load__half(" not in output


def test_preprocessor_declared_pointer_parameter_still_binds_pointee():
    code = """
    struct Loader {
      template <typename U>
      float load(const device U* src) { return float(src[0]); }
    };

    kernel void k(
        const device half* src [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint offset [[thread_position_in_grid]]) {
      Loader loader;
      out[0] = loader.load(src);
      out[1] = loader.load(src + offset);
      out[2] = loader.load(offset + src);
      out[3] = loader.load(src - offset);
      out[4] = loader.load(&(src[offset]));
    }
    """

    output = MetalPreprocessor().preprocess(code)

    helper = "float Loader__load__half(thread Loader& self, const device half* src)"
    assert output.count(helper) == 1
    assert "Loader__load__half(loader, src)" in output
    assert "Loader__load__half(loader, src + offset)" in output
    assert "Loader__load__half(loader, offset + src)" in output
    assert "Loader__load__half(loader, src - offset)" in output
    assert "Loader__load__half(loader, &(src[offset]))" in output
    assert "Loader__load__const_device_half_ptr(" not in output


def test_preprocessor_matches_concrete_pointer_template_member_parameters():
    code = """
    struct Loader {
      template <typename Tag>
      float load_pointer(const device float* src, Tag tag) {
        return src[0] + tag;
      }

      template <typename Tag>
      float load_qualified(const volatile device float* src, Tag tag) {
        return src[0] + tag;
      }
    };

    kernel void k(
        const device float* src [[buffer(0)]],
        device float* out [[buffer(1)]]) {
      Loader loader;
      out[0] = loader.load_pointer(src, 1);
      out[1] = loader.load_qualified(src, 1);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert (
        "float Loader__load_pointer__int(thread Loader& self, "
        "const device float* src, int tag)" in output
    )
    assert (
        "float Loader__load_qualified__int(thread Loader& self, "
        "const volatile device float* src, int tag)" in output
    )


@pytest.mark.parametrize(
    ("alias_declarations", "declared_pointee", "argument_pointee"),
    [
        pytest.param(
            "typedef half float16_t;",
            "half",
            "float16_t",
            id="typedef-argument",
        ),
        pytest.param(
            "using scalar_t = half;\nusing source_t = scalar_t;",
            "half",
            "source_t",
            id="using-argument-chain",
        ),
        pytest.param(
            "typedef half scalar_t;\ntypedef scalar_t source_t;",
            "source_t",
            "half",
            id="typedef-declaration-chain",
        ),
    ],
)
def test_preprocessor_matches_global_pointer_pointee_aliases(
    alias_declarations, declared_pointee, argument_pointee
):
    code = f"""
    {alias_declarations}

    struct Loader {{
      template <typename Tag>
      float load(const device {declared_pointee}* src, Tag tag) {{
        return float(src[0]) + tag;
      }}
    }};

    kernel void k(
        const device {argument_pointee}* src [[buffer(0)]],
        device float* out [[buffer(1)]]) {{
      Loader loader;
      out[0] = loader.load(src, 1);
    }}
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Loader__load__int(loader, src, 1)" in output
    assert "loader.load(" not in output


def test_preprocessor_matches_local_pointer_pointee_alias_chain():
    code = """
    struct Loader {
      template <typename Tag>
      float load(const device half* src, Tag tag) {
        return float(src[0]) + tag;
      }
    };

    kernel void k(
        const device half* src [[buffer(0)]],
        device float* out [[buffer(1)]]) {
      typedef half scalar_t;
      using source_t = scalar_t;
      const device source_t* local_src = src;
      Loader loader;
      out[0] = loader.load(local_src, 1);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Loader__load__int(loader, local_src, 1)" in output


def test_preprocessor_resolves_shadowed_pointer_pointee_alias_at_each_call():
    code = """
    using element_t = int;

    struct Loader {
      template <typename Tag>
      int load_half(const device half* src, Tag tag) {
        return int(src[0]) + tag;
      }

      template <typename Tag>
      int load_int(const device int* src, Tag tag) {
        return src[0] + tag;
      }
    };

    kernel void k(
        const device half* half_src [[buffer(0)]],
        const device int* int_src [[buffer(1)]],
        device int* out [[buffer(2)]]) {
      Loader loader;
      {
        using element_t = half;
        const device element_t* inner_src = half_src;
        out[0] = loader.load_half(inner_src, 1);
      }
      const device element_t* outer_src = int_src;
      out[1] = loader.load_int(outer_src, 2);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Loader__load_half__int(loader, inner_src, 1)" in output
    assert "Loader__load_int__int(loader, outer_src, 2)" in output


@pytest.mark.parametrize(
    ("alias_declarations", "argument_pointee"),
    [
        pytest.param("using source_t = int;", "source_t", id="incompatible"),
        pytest.param(
            "using source_t = later_t;\nusing later_t = half;",
            "source_t",
            id="unresolved-forward",
        ),
        pytest.param(
            "using first_t = second_t;\nusing second_t = first_t;",
            "first_t",
            id="cycle",
        ),
    ],
)
def test_preprocessor_rejects_unproved_pointer_pointee_alias_equivalence(
    alias_declarations, argument_pointee
):
    code = f"""
    {alias_declarations}

    struct Loader {{
      template <typename Tag>
      float load(const device half* src, Tag tag) {{ return float(tag); }}
    }};

    kernel void k(
        const device {argument_pointee}* src [[buffer(0)]],
        device float* out [[buffer(1)]]) {{
      Loader loader;
      out[0] = loader.load(src, 1);
    }}
    """

    with pytest.raises(MetalStructMethodError, match="did not bind consistently"):
        MetalPreprocessor().preprocess(code)


def test_preprocessor_allows_pointer_qualification_added_through_pointee_alias():
    code = """
    using readonly_half_t = const half;

    struct Loader {
      template <typename Tag>
      float load(device readonly_half_t* src, Tag tag) {
        return float(src[0]) + tag;
      }
    };

    kernel void k(
        device half* src [[buffer(0)]],
        device float* out [[buffer(1)]]) {
      Loader loader;
      out[0] = loader.load(src, 1);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Loader__load__int(loader, src, 1)" in output


def test_preprocessor_rejects_pointer_qualification_dropped_through_pointee_alias():
    code = """
    using readonly_half_t = const half;

    struct Loader {
      template <typename Tag>
      float load(device half* src, Tag tag) { return float(tag); }
    };

    kernel void k(
        device readonly_half_t* src [[buffer(0)]],
        device float* out [[buffer(1)]]) {
      Loader loader;
      out[0] = loader.load(src, 1);
    }
    """

    with pytest.raises(MetalStructMethodError, match="did not bind consistently"):
        MetalPreprocessor().preprocess(code)


@pytest.mark.parametrize(
    ("declared_type", "actual_parameter"),
    [
        pytest.param(
            "const device float*",
            "const device int* value [[buffer(0)]],",
            id="pointer-pointee",
        ),
        pytest.param(
            "const device float*",
            "const volatile device float* value [[buffer(0)]],",
            id="dropped-volatile",
        ),
    ],
)
def test_preprocessor_rejects_concrete_pointer_template_member_parameter_mismatch(
    declared_type, actual_parameter
):
    code = f"""
    struct Loader {{
      template <typename Tag>
      float load({declared_type} value, Tag tag) {{ return float(tag); }}
    }};

    kernel void k(
        {actual_parameter}
        device float* out [[buffer(1)]]) {{
      Loader loader;
      out[0] = loader.load(value, 1);
    }}
    """

    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)

    error = excinfo.value
    assert error.project_diagnostic_code == "project.translate.metal-struct-method"
    assert error.missing_capabilities == ("struct.template-method",)
    assert error.requested_signature == "loader.load(value, 1)"
    assert "did not bind consistently" in str(error)


def test_preprocessor_rejects_concrete_array_pointee_mismatch():
    code = """
    struct Loader {
      template <typename Tag>
      float load(thread float values[4], Tag tag) { return float(tag); }
    };

    kernel void k(device float* out [[buffer(0)]]) {
      thread int values[4];
      Loader loader;
      out[0] = loader.load(values, 1);
    }
    """

    with pytest.raises(MetalStructMethodError, match="did not bind consistently"):
        MetalPreprocessor().preprocess(code)


def test_preprocessor_validates_dependent_concrete_pointer_pointee_portion():
    source = """
    template <typename Left, typename Right>
    struct Pair { Left first; Right second; };

    struct Loader {
      template <typename Tag>
      float load(const device Pair<Tag, float>* value, Tag tag) {
        return float(tag);
      }
    };

    kernel void k(
        const device Pair<int, RIGHT_TYPE>* value [[buffer(0)]],
        device float* out [[buffer(1)]]) {
      Loader loader;
      out[0] = loader.load(value, 1);
    }
    """

    output = MetalPreprocessor().preprocess(source.replace("RIGHT_TYPE", "float"))
    assert "Loader__load__int(loader, value, 1)" in output

    with pytest.raises(MetalStructMethodError, match="did not bind consistently"):
        MetalPreprocessor().preprocess(source.replace("RIGHT_TYPE", "int"))


def test_preprocessor_selects_template_pointer_overload_by_concrete_pointee():
    code = """
    struct Loader {
      template <typename Tag>
      int load_pointer(const device float* src, Tag tag) {
        return int(src[0]) + tag;
      }

      template <typename Tag>
      int load_pointer(const device int* src, Tag tag) {
        return src[0] + tag;
      }
    };

    kernel void k(
        const device int* src [[buffer(0)]],
        device int* out [[buffer(1)]]) {
      Loader loader;
      out[0] = loader.load_pointer(src, 1);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert (
        "int Loader__load_pointer__int(thread Loader& self, "
        "const device int* src, int tag)" in output
    )
    assert "const device float* src, int tag" not in output


def test_preprocessor_preserves_scalar_conversion_for_concrete_parameter():
    code = """
    struct Loader {
      template <typename Tag>
      float load(float value, Tag tag) { return value + tag; }
    };

    kernel void k(device float* out [[buffer(0)]]) {
      Loader loader;
      out[0] = loader.load(1, 2);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Loader__load__int(loader, 1, 2)" in output
    assert (
        "float Loader__load__int(thread Loader& self, float value, int tag)" in output
    )


def test_preprocessor_instantiates_explicit_value_template_member_call():
    code = """
    struct Reducer {
      template <int N>
      int reduce_many(int best, thread int* values, uint offset) {
        for (int i = 0; i < N; i++) {
          best += values[i] + int(offset);
        }
        return best;
      }
    };

    kernel void k(
        device int* output [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
      Reducer reducer;
      int values[4] = {1, 2, 3, 4};
      output[index] = reducer.template reduce_many<4>(0, values, index);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "int Reducer__reduce_many__4(" in output
    assert "for (int i = 0; i < 4; i++)" in output
    assert "Reducer__reduce_many__4(reducer, 0, values, index)" in output
    assert "reducer.template reduce_many" not in output
    assert "reducer.reduce_many" not in output


def test_preprocessor_binds_float16_threadgroup_array_to_explicit_member_template():
    code = """
    typedef half float16_t;

    struct Tile {
      template <typename U, int StrideX, int StrideY>
      void load(const threadgroup U* src) {
        U value = src[StrideX + StrideY];
      }
    };

    kernel void float_path(device float* output [[buffer(0)]]) {
      threadgroup float Ws[16];
      Tile tile;
      tile.template load<float, 4, 1>(Ws);
      output[0] = Ws[0];
    }

    kernel void float16_path(device float16_t* output [[buffer(0)]]) {
      threadgroup float16_t Ws[16];
      Tile tile;
      tile.template load<float16_t, 4, 1>(Ws + 1);
      output[0] = Ws[0];
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Tile__load__float_4_1(tile, Ws)" in output
    assert "Tile__load__float16_t_4_1(tile, Ws + 1)" in output
    assert "const threadgroup float16_t* src" in output
    assert "tile.template load" not in output


def test_preprocessor_instantiates_template_method_on_nested_struct_field():
    code = """
    struct Tile {
      template <typename T, int Rows, int Cols, int Leading, int Threads>
      void load(threadgroup T* src) {
        src[Rows + Cols + Leading + Threads] = T(1);
      }
    };

    struct BlockMMA {
      Tile Atile;

      void mma(threadgroup float* As) {
        Atile.template load<float, 1, 1, (36), (1)>(As);
      }
    };

    kernel void k(device float* output [[buffer(0)]]) {
      threadgroup float As[64];
      BlockMMA block;
      block.mma(As);
      output[0] = As[0];
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Tile__load__float_1_1_36_1(self.Atile, As)" in output
    assert "void Tile__load__float_1_1_36_1(" in output
    assert "self.Atile.template load" not in output
    assert "self.Atile.load" not in output


def test_preprocessor_instantiates_template_method_on_external_nested_field():
    code = """
    struct Tile {
      template <typename T, int Width>
      void load(threadgroup T* src) { src[Width] = T(1); }
    };

    struct Block { Tile tile; };

    kernel void k(device float* output [[buffer(0)]]) {
      threadgroup float values[8];
      Block block;
      block.tile.template load<float, 4>(values);
      output[0] = values[0];
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Tile__load__float_4(block.tile, values)" in output
    assert "void Tile__load__float_4(" in output
    assert "block.tile.template load" not in output


def test_preprocessor_instantiates_template_method_on_pointer_receiver():
    code = """
    struct Tile {
      template <typename T, int Width>
      void load(threadgroup T* src) { src[Width] = T(1); }
    };

    kernel void k(device float* output [[buffer(0)]]) {
      threadgroup float values[8];
      Tile tile;
      thread Tile* tile_ptr = &tile;
      tile_ptr->template load<float, 4>(values);
      output[0] = values[4];
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Tile__load__float_4(tile_ptr[0], values)" in output
    assert "void Tile__load__float_4(" in output
    assert "tile_ptr->template load" not in output


def test_preprocessor_instantiates_template_method_on_nested_pointer_field():
    code = """
    struct Tile {
      template <typename T, int Width>
      void load(threadgroup T* src) { src[Width] = T(1); }
    };

    struct Holder { thread Tile* tile; };

    kernel void k(device float* output [[buffer(0)]]) {
      threadgroup float values[8];
      Tile tile;
      Holder holder{&tile};
      holder.tile->template load<float, 4>(values);
      output[0] = values[4];
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Tile__load__float_4(holder.tile[0], values)" in output
    assert "void Tile__load__float_4(" in output
    assert "holder.tile->template load" not in output


def test_preprocessor_instantiates_member_template_with_default_arguments():
    code = """
    struct Tile {
      template <typename T = float, int Width = 4>
      void load(threadgroup T* src) { src[Width] = T(1); }
    };

    kernel void k(device float* output [[buffer(0)]]) {
      threadgroup float values[8];
      Tile tile;
      tile.template load<>(values);
      output[0] = values[4];
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Tile__load__float_4(tile, values)" in output
    assert "void Tile__load__float_4(" in output
    assert "tile.template load" not in output


def test_preprocessor_resolves_local_constants_in_template_member_arguments():
    code = """
    template <typename T, T Value>
    struct integral_constant {
      static constexpr T value = Value;
    };

    template <int Value>
    using Int = integral_constant<int, Value>;

    struct Loader {
      template <typename Stride>
      int load(Stride stride) { return stride.value; }

      template <int Width>
      int explicit_width() { return Width; }
    };

    kernel void first(device int* output [[buffer(0)]]) {
      constexpr int Padding = 8;
      Loader loader;
      output[0] = loader.load(Int<Padding>{});
      output[1] = loader.template explicit_width<Padding>();
      {
        constexpr int Padding = 4;
        output[2] = loader.load(Int<Padding>{});
      }
      output[3] = loader.load(Int<Padding>{});
    }

    kernel void second(device int* output [[buffer(0)]]) {
      using Scalar = half;
      constexpr int Padding = 8 + 16 / sizeof(Scalar);
      Loader loader;
      output[0] = loader.load(Int<Padding>{});
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Loader__load__Int_8(loader, Int<Padding>{})" in output
    assert "Loader__load__Int_4(loader, Int<Padding>{})" in output
    assert "Loader__load__Int_16(loader, Int<Padding>{})" in output
    assert "Loader__explicit_width__8(loader)" in output
    assert "int Loader__load__Int_8(thread Loader& self, Int<8> stride)" in output
    assert "int Loader__load__Int_4(thread Loader& self, Int<4> stride)" in output
    assert "int Loader__load__Int_16(thread Loader& self, Int<16> stride)" in output
    assert "int Loader__explicit_width__8(thread Loader& self)" in output


def test_preprocessor_does_not_borrow_template_member_constant_from_sibling():
    code = """
    template <typename T, T Value>
    struct integral_constant { static constexpr T value = Value; };

    template <int Value>
    using Int = integral_constant<int, Value>;

    struct Loader {
      template <typename Stride>
      int load(Stride stride) { return stride.value; }
    };

    kernel void concrete(device int* output [[buffer(0)]]) {
      constexpr int Padding = 8;
      Loader loader;
      output[0] = loader.load(Int<Padding>{});
    }

    kernel void unresolved(
        device int* output [[buffer(0)]],
        constant int& runtime_padding [[buffer(1)]]) {
      const int Padding = runtime_padding;
      Loader loader;
      output[0] = loader.load(Int<Padding>{});
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert output.count("Loader__load__Int_8(loader, Int<Padding>{})") == 1
    assert "Loader__load__Int_Padding(loader, Int<Padding>{})" in output


def test_preprocessor_resolves_local_constant_in_instantiated_member_body():
    code = """
    template <typename T, T Value>
    struct integral_constant { static constexpr T value = Value; };

    template <int Value>
    using Int = integral_constant<int, Value>;

    struct Loader {
      template <typename Stride>
      int consume(Stride stride) { return stride.value; }

      template <typename T>
      int run(T value) {
        constexpr int Padding = 4;
        return value + consume(Int<Padding>{});
      }
    };

    kernel void k(device int* output [[buffer(0)]]) {
      Loader loader;
      output[0] = loader.run(output[0]);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Loader__run__int(loader, output[0])" in output
    assert "Loader__consume__Int_4(self, Int<Padding>{})" in output
    assert "int Loader__consume__Int_4(thread Loader& self, Int<4> stride)" in output
    assert "Loader__consume__Int_Padding" not in output


def test_preprocessor_selects_template_member_overload_by_arity():
    code = """
    struct Tile {
      template <typename U, int StrideX, int StrideY>
      void load(const threadgroup U* src) { U value = src[0]; }

      template <typename U>
      void load(const device U* src, int stride) { U value = src[stride]; }
    };

    kernel void k(
        const device float* input [[buffer(0)]],
        const constant int& stride [[buffer(1)]]) {
        Tile tile;
        tile.load(input, stride);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Tile__load__float(tile, input, stride)" in output
    assert (
        "void Tile__load__float(thread Tile& self, "
        "const device float* src, int stride)" in output
    )
    assert "StrideX" not in output
    assert "StrideY" not in output
    assert "tile.load(" not in output


def test_preprocessor_selects_reference_template_overload_with_pointer_field():
    code = """
    struct Params { const int ldc; const int fdc; };
    struct Transform { float scale; };
    struct Tile {
      template <typename UnaryEpilogue>
      void apply(thread const UnaryEpilogue& op) {}

      template <typename BinaryEpilogue>
      void apply(
          const device half* src,
          const int ldc,
          const int fdc,
          thread const BinaryEpilogue& op) {}
    };

    kernel void k(
        const device half* input [[buffer(0)]],
        const constant Params* params [[buffer(1)]]) {
        Tile tile;
        Transform transform;
        tile.apply(input, params->ldc, params->fdc, transform);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert (
        "Tile__apply__Transform(tile, input, params->ldc, params->fdc, transform)"
        in output
    )
    assert (
        "void Tile__apply__Transform(thread Tile& self, const device half* src, "
        "const int ldc, const int fdc, thread const Transform& op)" in output
    )
    assert "tile.apply(" not in output


def test_preprocessor_resolves_struct_scoped_alias_in_lowered_method():
    code = """
    struct Helper {
      template <typename U>
      static void apply(U value) { U copy = value; }
    };

    struct Wrapper {
      using HelperAlias = Helper;

      template <typename U>
      void apply(U value) { U copy = value; }

      void run(float value) {
        HelperAlias::apply(value);
      }
    };

    kernel void k(device float* output [[buffer(0)]]) {
      Wrapper wrapper;
      wrapper.run(output[0]);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Helper__apply__float(value)" in output
    assert "void Helper__apply__float(float value)" in output
    assert "HelperAlias::apply" not in output
    assert "Wrapper__apply" not in output


@pytest.mark.parametrize(
    "alias_declaration",
    ["using elem_type = T;", "typedef T elem_type;"],
)
def test_preprocessor_resolves_struct_scoped_aliases_in_explicit_pointer_casts(
    alias_declaration,
):
    code = """
    template <typename T>
    struct Tile {
      ALIAS_DECLARATION
      elem_type values[4];

      thread elem_type* elems() {
        return (thread elem_type*)values;
      }

      const thread elem_type* const_elems() const {
        return reinterpret_cast<const thread elem_type*>(values);
      }
    };

    Tile<float> float_tile;
    Tile<int> int_tile;
    """.replace("ALIAS_DECLARATION", alias_declaration)

    output = MetalPreprocessor().preprocess(code)
    float_helper = output.split("Tile_float__elems", 1)[1].split("}", 1)[0]
    int_helper = output.split("Tile_int__elems", 1)[1].split("}", 1)[0]

    assert "return (thread float*)self.values;" in output
    assert "reinterpret_cast<const thread float*>(self.values)" in output
    assert "return (thread int*)self.values;" in output
    assert "reinterpret_cast<const thread int*>(self.values)" in output
    assert output.count("thread elem_type*") == 4
    assert "elem_type" not in float_helper
    assert "thread int*" not in float_helper
    assert "elem_type" not in int_helper
    assert "thread float*" not in int_helper


def test_preprocessor_preserves_local_alias_shadow_in_explicit_cast():
    code = """
    struct Wrapper {
      using value_type = float;

      int cast(int value) {
        using value_type = int;
        float owner_value = static_cast<Wrapper::value_type>(value);
        return static_cast<value_type>(value) + int(owner_value);
      }
    };

    Wrapper wrapper;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "using value_type = int;" in output
    assert "static_cast<float>(value)" in output
    assert "static_cast<value_type>(value)" in output


def test_preprocessor_resolves_qualified_owner_alias_with_pointer_target():
    code = """
    template <typename T>
    struct Tile {
      typedef const thread T* elem_ptr;
      T values[4];

      elem_ptr elems() const {
        return (Tile<T>::elem_ptr)values;
      }
    };

    Tile<float> float_tile;
    Tile<int> int_tile;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "return (const thread float*)self.values;" in output
    assert "return (const thread int*)self.values;" in output
    assert "Tile_float::elem_ptr" not in output
    assert "Tile_int::elem_ptr" not in output


def test_preprocessor_expands_struct_owned_alias_templates():
    code = """
    struct BaseFrag {
      static constexpr short kElems = 4;

      template <typename U>
      using dtype_frag_t = typename metal::vec<U, kElems>;

      template <typename T>
      static dtype_frag_t<T> make(T value) {
        dtype_frag_t<T> result;
        result[0] = value;
        return result;
      }
    };

    kernel void vector_alias(
        device float* float_output [[buffer(0)]],
        device int* int_output [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
      BaseFrag::dtype_frag_t<float> float_value = BaseFrag::make<float>(1.0f);
      BaseFrag::dtype_frag_t<int> int_value = BaseFrag::make<int>(2);
      float_output[gid] = float_value[0];
      int_output[gid] = int_value[0];
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "using dtype_frag_t = typename metal::vec<U, kElems>;" in output
    assert "metal::vec<float,4> float_value" in output
    assert "metal::vec<int,4> int_value" in output
    assert "metal::vec<float,4> BaseFrag__make__float" in output
    assert "metal::vec<int,4> BaseFrag__make__int" in output
    assert "metal::vec<float,4> result" in output
    assert "metal::vec<int,4> result" in output
    assert "BaseFrag::dtype_frag_t<float>" not in output
    assert "BaseFrag::dtype_frag_t<int>" not in output


def test_preprocessor_expands_alias_template_defaults_from_owner_constants():
    code = """
    struct BaseFrag {
      static constexpr short kElems = 4;

      template <typename U, short N = kElems>
      using frag_t = metal::vec<U, N>;

      frag_t<float> values;
    };

    kernel void vector_alias(
        device float* output [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
      BaseFrag::frag_t<float> value;
      output[gid] = value[0];
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "metal::vec<float,4> value" in output
    assert "BaseFrag::frag_t<float>" not in output


def test_preprocessor_expands_alias_templates_exposed_by_owner_aliases():
    preprocessor = MetalPreprocessor()
    definitions = preprocessor._find_concrete_struct_definitions("""
        struct BaseFrag {
          static constexpr short kFragRows = 16;
          static constexpr short kFragCols = 16;
          static constexpr short kElems = (kFragRows * kFragCols) / 32;

          template <typename U>
          using dtype_frag_t = typename metal::vec<U, kElems>;
        };

        struct Tile {
          using Frag = BaseFrag;
          typedef typename Frag::template dtype_frag_t<float> frag_type;
        };
        """)
    structs = {definition.name: definition for definition in definitions}

    resolved = preprocessor._canonicalize_struct_scoped_type(
        "thread frag_type&",
        structs["Tile"],
        structs,
    )

    assert resolved == "thread metal::vec<float,8>&"


def test_preprocessor_keeps_alias_template_chains_with_the_declaring_owner():
    code = """
    struct Producer {
      template <typename U> using storage_t = metal::vec<U, 2>;
      template <typename U> using value_t = storage_t<U>;
    };

    struct Consumer {
      template <typename U> using storage_t = metal::vec<U, 3>;
      Producer::value_t<float> value;
    };
    """

    output = MetalPreprocessor().preprocess(code)

    assert "metal::vec<float,2> value" in output
    assert "metal::vec<float,3> value" not in output
    assert "Producer::storage_t<float>" not in output


def test_preprocessor_disambiguates_same_named_alias_template_owners():
    code = """
    namespace first {
    struct Frag {
      static constexpr short kElems = 2;
      template <typename U> using value_t = metal::vec<U, kElems>;
    };
    }

    namespace second {
    struct Frag {
      static constexpr short kElems = 3;
      template <typename U> using value_t = metal::vec<U, kElems>;
    };
    }

    first::Frag::value_t<float> first_value;
    second::Frag::value_t<float> second_value;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "metal::vec<float,2> first_value" in output
    assert "metal::vec<float,3> second_value" in output


def test_preprocessor_expands_alias_template_with_qualified_owner_constant():
    code = """
    struct Frag {
      static constexpr short kElems = 4;
      template <typename U>
      using value_t = metal::vec<U, Frag::kElems>;
    };

    Frag::value_t<float> value;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "metal::vec<float,4> value" in output


def test_preprocessor_expands_alias_template_with_fully_qualified_owner_constant():
    code = """
    namespace shapes {
    struct Frag {
      static constexpr short kElems = 4;
      template <typename U>
      using value_t = metal::vec<U, shapes::Frag::kElems>;
    };
    }

    shapes::Frag::value_t<float> value;
    """

    output = MetalPreprocessor().preprocess(code)

    assert "metal::vec<float,4> value" in output


def test_preprocessor_records_struct_aliases_and_method_const_qualification():
    pp = MetalPreprocessor()
    definitions = pp._find_concrete_struct_definitions("""
        struct Base {};
        struct Wrapper {
          using BaseAlias = Base;
          typedef float value_type;
          float value() const { return 1.0; }
        };
        """)

    wrapper = next(item for item in definitions if item.name == "Wrapper")

    assert wrapper.type_aliases == {"BaseAlias": "Base", "value_type": "float"}
    assert len(wrapper.methods) == 1
    assert wrapper.methods[0].name == "value"
    assert wrapper.methods[0].is_const is True


def test_preprocessor_lowers_value_only_nested_template_member_call():
    code = """
    struct Frag {
      template <typename T>
      using dtype_frag_t = T;

      template <typename T>
      static T load(dtype_frag_t<T> value, const device T* src) {
        return value + src[0];
      }

      template <typename T>
      static void store(dtype_frag_t<T> value, device T* dst) {
        dst[0] = value;
      }
    };

    struct Index {
      static constexpr int value = 0;
    };

    struct Tile {
      using FragType = Frag;
      typedef typename FragType::template dtype_frag_t<half> frag_type;
      frag_type values[4];

      template <int I>
      frag_type frag_at() {
        return values[I];
      }

      template <int I>
      frag_type frag_at() const {
        return values[I];
      }

      template <typename U>
      void load(const device U* src) {
        Index index;
        values[0] = FragType::load(frag_at<index.value>(), src);
      }

      template <typename U>
      void store(device U* dst) const {
        Index index;
        FragType::store(frag_at<index.value>(), dst);
      }
    };

    kernel void k(
        const device half* input [[buffer(0)]],
        device half* output [[buffer(1)]]) {
      Tile tile;
      tile.load(input);
      tile.store(output);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Tile__frag_at__runtime_values__int(self, index.value)" in output
    assert "Tile__frag_at__runtime_values__int__const(self, index.value)" in output
    assert "half Tile__frag_at__runtime_values__int(thread Tile& self, int I)" in output
    assert (
        "half Tile__frag_at__runtime_values__int__const"
        "(thread const Tile& self, int I)" in output
    )
    assert "half Frag__load__half(half value, const device half* src)" in output
    assert "void Frag__store__half(half value, device half* dst)" in output
    assert "Frag::dtype_frag_t<half>" not in output
    assert "Frag__load__half(" in output
    assert "Frag__store__half(" in output
    assert "frag_at<" not in output
    assert "FragType::load" not in output


def test_preprocessor_does_not_runtime_lower_compile_time_template_body():
    pp = MetalPreprocessor()
    definitions = pp._find_concrete_struct_definitions("""
        struct Tile {
          template <int I>
          int select() {
            if constexpr (I == 0) { return 1; }
            return 2;
          }
        };
        """)
    method = definitions[0].template_methods[0]

    assert pp._runtime_value_template_method_is_safe(method, ["index.value"]) is False


def test_preprocessor_instantiates_template_method_from_typed_local_arg():
    # The argument is a bare local declared `half v = ...;` so the method
    # instantiates with T=half.
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device half* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        half v = out[i];
        out[i] = op.reduce(v);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "half Sum__reduce__half(thread Sum& self, half val)" in output
    assert "Sum__reduce__half(op, v)" in output


def test_preprocessor_instantiates_template_method_from_generic_vector_local():
    code = """
    struct BaseFrag {
      template <typename T>
      static T first(const thread metal::vec<T, 4>& value) {
        return value.x;
      }
    };

    kernel void k(device float* out [[buffer(0)]]) {
      metal::vec<float, 4> value = metal::vec<float, 4>{1.0, 2.0, 3.0, 4.0};
      out[0] = BaseFrag::first(value);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "float BaseFrag__first__float(" in output
    assert "BaseFrag__first__float(value)" in output
    assert "BaseFrag::first" not in output


@pytest.mark.parametrize(
    ("type_text", "expected"),
    [
        ("metal::vec<float, 4>", ("float", 4)),
        ("::metal::vec<float, (4)>", ("float", 4)),
        ("vec<int, 8>", ("int", 8)),
        ("metal::vec<T, 4>", None),
        ("metal::vec<float, N>", None),
        ("metal::vec<float, 0>", None),
        ("metal::vec<float>", None),
    ],
)
def test_preprocessor_recognizes_only_concrete_generic_vector_types(
    type_text, expected
):
    assert MetalPreprocessor()._scalar_and_width(type_text) == expected


def test_preprocessor_does_not_resolve_types_from_future_declarations():
    declarations = {"value": [(40, "float"), (80, "int")]}
    preprocessor = MetalPreprocessor()

    assert preprocessor._resolve_declared_type_at(declarations, "value", 20) is None
    assert preprocessor._resolve_declared_type_at(declarations, "value", 60) == "float"
    assert preprocessor._resolve_declared_type_at(declarations, "value", 100) == "int"


def test_preprocessor_instantiates_template_operator_call_from_typed_local():
    # A template `operator()` is instantiated from the call-site argument types;
    # the struct template parameter U is materialized first, the method's own
    # template parameter T binds from the second argument.
    code = """
    template <typename U>
    struct Op { template <typename T> U operator()(U a, T b){ return a + b; } };

    [[kernel]] void k(
        device const float* in [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        Op<float> op;
        float acc = out[i];
        float x = in[i];
        out[i] = op(acc, x);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "struct Op_float {" in output
    assert (
        "float Op_float__operator_call__float(thread Op_float& self, "
        "float a, float b)" in output
    )
    assert "Op_float__operator_call__float(op, acc, x)" in output
    # No dangling functor call survives.
    assert "= op(" not in output


def test_preprocessor_instantiates_template_operator_call_from_temporary_functor():
    code = """
    struct Select {
      template <typename T>
      T operator()(bool condition, T x, T y) {
        return condition ? x : y;
      }
    };

    [[kernel]] void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        float x = out[i];
        out[i] = Select{}(true, x, 1.0);
        out[i] = Select()(false, x, 2.0);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert (
        "float Select__operator_call__float(thread Select& self, bool condition, "
        "float x, float y)"
    ) in output
    assert (
        "float Select__operator_call__float__temporary(bool condition, "
        "float x, float y)"
    ) in output
    assert "Select__operator_call__float__temporary(true, x, 1.0)" in output
    assert "Select__operator_call__float__temporary(false, x, 2.0)" in output
    assert "Select{}(" not in output
    assert "Select()(" not in output


def test_preprocessor_lowers_temporary_functor_call_with_arithmetic_and_functor_argument():
    # Regression: a template member method called with an argument that is a
    # BINARY ARITHMETIC expression built from complex-typed values and a NESTED
    # functor construction-and-call -- `Log{}(x + i * Sqrt{}(1.0 - x * x))`, the
    # exact shape MLX's complex arccos/arcsin/arctan feed to Log -- must lower.
    # Previously this clean-failed with "could not be inferred conservatively"
    # because argument inference handled neither the enclosing out-of-line
    # `operator()` parameter, the `auto` construction-initialized local, the
    # arithmetic operators, nor the `Sqrt{}(...)` functor-call result type. The
    # inference is now general enough that the outer call resolves to the concrete
    # `complex64_t operator()` overload and lowers to a free function.
    code = """
    struct complex64_t { float real; float imag; };

    constexpr complex64_t operator+(complex64_t a, complex64_t b) {
      return {a.real + b.real, a.imag + b.imag};
    }
    constexpr complex64_t operator-(float a, complex64_t b) {
      return {a - b.real, -b.imag};
    }
    constexpr complex64_t operator*(complex64_t a, complex64_t b) {
      return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
    }

    struct Sqrt {
      template <typename T>
      T operator()(T x) { return metal::precise::sqrt(x); }
      complex64_t operator()(complex64_t x) { return {x.real, x.imag}; }
    };

    struct Log {
      template <typename T>
      T operator()(T x) { return metal::precise::log(x); }
      complex64_t operator()(complex64_t x) { return {x.real, x.imag}; }
    };

    struct ArcCos {
      template <typename T>
      T operator()(T x) { return metal::precise::acos(x); }
      complex64_t operator()(complex64_t x);
    };

    complex64_t ArcCos::operator()(complex64_t x) {
      auto i = complex64_t{0.0, 1.0};
      auto y = Log{}(x + i * Sqrt{}(1.0 - x * x));
      return {y.imag, -y.real};
    }

    [[kernel]] void k(
        device const complex64_t* in [[buffer(0)]],
        device complex64_t* out [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
        ArcCos op;
        out[gid] = op(in[gid]);
    }
    """

    # A single un-inferable argument would raise MetalStructMethodError out of
    # preprocess(); reaching an output at all means the argument was inferred.
    output = MetalPreprocessor().preprocess(code)

    # The outer `Log{}(...)` temporary functor call lowered to a free function;
    # no dangling `Log{}(` temporary and no un-inferred failure survive.
    assert "Log__operator_call__temporary(" in output
    assert "Log{}(" not in output
    assert "Sqrt{}(" not in output
    assert "could not be inferred" not in output
    assert (
        "Log__operator_call__temporary("
        "x + i * Sqrt__operator_call__temporary(1.0 - x * x))"
    ) in output
    assert output.count("Sqrt__operator_call__temporary(1.0 - x * x)") == 1
    # The complex `Sqrt`/`Log` operator() overloads were emitted as free
    # functions (the argument resolved to the concrete complex overload).
    assert "complex64_t Log__operator_call(thread Log& self, complex64_t x)" in output
    assert "complex64_t Sqrt__operator_call(thread Sqrt& self, complex64_t x)" in output


def test_preprocessor_defers_temporaries_for_declared_out_of_line_call_operators():
    code = """
    struct complex64_t { float real; float imag; };

    struct Sqrt {
      complex64_t operator()(complex64_t x);
    };

    struct Log {
      complex64_t operator()(complex64_t x);
    };

    complex64_t Sqrt::operator()(complex64_t x) { return x; }
    complex64_t Log::operator()(complex64_t x) { return Sqrt{}(x); }

    complex64_t nested(complex64_t x) {
      return Log{}(Sqrt{}(x));
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "complex64_t operator()(complex64_t x);" in output
    assert "return Log{}(Sqrt{}(x));" in output


def test_preprocessor_lowers_const_template_operator_on_stateless_temporary():
    code = """
    struct Identity {
      template <typename T>
      T operator()(T value) const { return value; }
    };

    kernel void k(device float* values [[buffer(0)]]) {
      values[0] = Identity{}(values[0]);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert (
        "float Identity__operator_call__float("
        "thread const Identity& self, float value)"
    ) in output
    assert "Identity__operator_call__float__temporary(values[0])" in output
    assert "Identity{}(" not in output


def test_preprocessor_rejects_stateful_temporary_without_dropping_side_effects():
    code = """
    int next_state();
    int next_value();

    struct AddState {
      int state;
      AddState(int value) : state(value) { observe(value); }
      int operator()(int value) const { return state + value; }
    };

    kernel void k(device int* values [[buffer(0)]]) {
      values[0] = AddState{next_state()}(next_value());
    }
    """

    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)

    error = excinfo.value
    assert error.project_diagnostic_code == "project.translate.metal-struct-method"
    assert error.struct_name == "AddState"
    assert error.method_name == "operator()"
    assert error.reason == "temporary-functor-constructor-stateful"
    assert error.requested_signature == "AddState{next_state()}(next_value())"
    assert error.source_location["length"] == len(error.requested_signature)
    assert "next_state()" in str(error)
    assert "next_value()" in str(error)


@pytest.mark.parametrize("functor_name", ["Op", "missing_functor"])
def test_preprocessor_rejects_unresolved_temporary_functor_structured(functor_name):
    code = f"""
    kernel void k(device float* values [[buffer(0)]]) {{
      values[0] = {functor_name}()(values[0]);
    }}
    """

    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)

    error = excinfo.value
    assert error.project_diagnostic_code == "project.translate.metal-struct-method"
    assert error.struct_name == functor_name
    assert error.method_name == "operator()"
    assert error.reason == "temporary-functor-type-unresolved"
    assert error.requested_signature == f"{functor_name}()(values[0])"
    assert error.source_location["length"] == len(error.requested_signature)


def test_preprocessor_rejects_ambiguous_temporary_operator_overload_structured():
    code = """
    struct Convert {
      short operator()(short value) const { return value; }
      long operator()(long value) const { return value; }
    };

    kernel void k(device int* values [[buffer(0)]]) {
      values[0] = Convert{}(values[0]);
    }
    """

    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)

    error = excinfo.value
    assert error.project_diagnostic_code == "project.translate.metal-struct-method"
    assert error.struct_name == "Convert"
    assert error.method_name == "operator()"
    assert error.reason == "concrete-overload-ambiguous"
    assert error.requested_signature == "Convert::operator()(values[0])"
    assert error.source_location["length"] == len("(values[0])")


def test_preprocessor_skips_calls_inside_residual_template_declarations():
    # After a template struct is materialized to a concrete struct (e.g.
    # `CumLogaddexp_float`), the ORIGINAL `template <typename U> struct
    # CumLogaddexp { ... }` declaration is left in the source with its template
    # parameter U still unbound. A functor call inside that residual template
    # body (`LogAddExp{}(a, static_cast<U>(b))`) references the unbound type U,
    # so its argument type cannot be inferred and it must NOT be lowered — only
    # the concrete materialized copy carries lowerable calls. The call-site
    # rewriter must therefore skip residual template declaration spans; otherwise
    # it descends into the template body and clean-fails with
    # "could not be inferred conservatively".
    code = """
    struct LogAddExp {
      template <typename T>
      T operator()(T x, T y) { return x + y; }
    };

    template <typename U>
    struct CumLogaddexp {
      template <typename T>
      U operator()(U a, T b) {
        return LogAddExp{}(a, static_cast<U>(b));
      }
    };

    template [[host_name("k_f")]] [[kernel]] void
    k<float, float, CumLogaddexp<float>>(
        const device float* in [[buffer(0)]],
        device float* out [[buffer(1)]]);

    template <typename T, typename U, typename Op>
    [[kernel]] void k(
        const device T* in [[buffer(0)]],
        device U* out [[buffer(1)]]) {
      Op op;
      out[0] = op(U(in[0]), in[1]);
    }
    """
    # Must not raise MetalStructMethodError for the residual template's call.
    output = MetalPreprocessor().preprocess(code)
    # The concrete instantiation is materialized and its call is lowerable.
    assert "CumLogaddexp_float" in output
    # The residual generic template declaration is left untouched.
    assert "template <typename U>" in output


def test_preprocessor_rewrites_temporary_functor_calls_inside_template_operator():
    code = """
    struct complex64_t {
      float real;
      float imag;
    };

    struct FloorDivide {
      template <typename T>
      T operator()(T x, T y) {
        return x / y;
      }
    };

    struct Remainder {
      template <typename T>
      T operator()(T x, T y) {
        return x % y;
      }
    };

    struct DivMod {
      template <typename T>
      metal::array<T, 2> operator()(T x, T y) {
        return {FloorDivide{}(x, y), Remainder{}(x, y)};
      }
    };

    [[kernel]] void k(
        device complex64_t* a [[buffer(0)]],
        device complex64_t* b [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
      DivMod op;
      auto out = op(a[i], b[i]);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "FloorDivide__operator_call__complex64_t__temporary(x, y)" in output
    assert "Remainder__operator_call__complex64_t__temporary(x, y)" in output
    assert (
        "complex64_t FloorDivide__operator_call__complex64_t"
        "(thread FloorDivide& self, complex64_t x, complex64_t y)"
    ) in output
    assert (
        "complex64_t Remainder__operator_call__complex64_t"
        "(thread Remainder& self, complex64_t x, complex64_t y)"
    ) in output
    assert "FloorDivide{}(" not in output
    assert "Remainder{}(" not in output


def test_preprocessor_preserves_concrete_overload_identity_in_temporary_wrappers():
    code = """
    typedef bfloat bfloat16_t;

    struct FloorDivide {
      template <typename T>
      T operator()(T x, T y) {
        return x / y;
      }

      template <>
      half operator()(half x, half y) {
        return trunc(x / y);
      }

      template <>
      bfloat16_t operator()(bfloat16_t x, bfloat16_t y) {
        return x - y;
      }
    };

    kernel void run(
        device half* values [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
      half half_value = half(index + 2u);
      bfloat16_t bfloat_value = bfloat16_t(index + 2u);
      values[index * 2u] = FloorDivide{}(half_value, half(2.0h));
      values[index * 2u + 1u] = half(FloorDivide{}(
          bfloat_value, bfloat16_t(2.0h)));
    }
    """

    output = MetalPreprocessor().preprocess(code)

    half_wrapper = "FloorDivide__operator_call__metal_overload_1__temporary"
    bfloat_wrapper = "FloorDivide__operator_call__metal_overload_2__temporary"
    assert f"{half_wrapper}(half_value, half(2.0h))" in output
    assert f"{bfloat_wrapper}(bfloat_value, bfloat16_t(2.0h))" in output
    assert (
        f"half {half_wrapper}(half x, half y) "
        "{ FloorDivide self; return FloorDivide__operator_call(self, x, y); }" in output
    )
    assert (
        f"bfloat16_t {bfloat_wrapper}(bfloat16_t x, bfloat16_t y) "
        "{ FloorDivide self; return FloorDivide__operator_call(self, x, y); }" in output
    )
    assert "FloorDivide__operator_call__half" not in output


def test_preprocessor_selects_unsigned_integral_sfinae_overload_for_bool():
    code = """
    struct complex64_t {
      float real;
      float imag;
    };

    struct Remainder {
      template <typename T>
      metal::enable_if_t<metal::is_integral_v<T> & !metal::is_signed_v<T>, T>
      operator()(T x, T y) {
        return x % y;
      }

      template <typename T>
      metal::enable_if_t<metal::is_integral_v<T> & metal::is_signed_v<T>, T>
      operator()(T x, T y) {
        return y;
      }

      template <typename T>
      metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
        return x;
      }

      template <>
      complex64_t operator()(complex64_t x, complex64_t y) {
        return x % y;
      }
    };

    [[kernel]] void k(
        device bool* out [[buffer(0)]],
        device bool* a [[buffer(1)]],
        device bool* b [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
      Remainder op;
      out[i] = op(a[i], b[i]);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "bool Remainder__operator_call__bool" in output
    assert "return x % y;" in output
    assert "return y;" not in output


def test_preprocessor_prefers_explicit_operator_specialization():
    code = """
    struct complex64_t {
      float real;
      float imag;
    };

    struct Remainder {
      template <typename T>
      metal::enable_if_t<metal::is_integral_v<T> & !metal::is_signed_v<T>, T>
      operator()(T x, T y) {
        return x;
      }

      template <typename T>
      metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
        return y;
      }

      template <>
      complex64_t operator()(complex64_t x, complex64_t y) {
        return x % y;
      }
    };

    [[kernel]] void k(
        device complex64_t* out [[buffer(0)]],
        device complex64_t* a [[buffer(1)]],
        device complex64_t* b [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
      Remainder op;
      out[i] = op(a[i], b[i]);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "complex64_t Remainder__operator_call" in output
    assert "return x % y;" in output
    assert "return y;" not in output


def test_preprocessor_instantiates_static_template_method_from_literal():
    # A static template member method called as `S::m(literal)` instantiates from
    # the literal type and lowers WITHOUT a `self` parameter.
    code = """
    struct Maths { template <typename T> static T twice(T a){ return a + a; } };

    [[kernel]] void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        out[i] = Maths::twice(2.5f);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Maths__twice__float(float a)" in output
    assert re.search(r"\bMaths\s*(?:&\s*)?self\b", output) is None
    assert "Maths__twice__float(2.5f)" in output
    assert "Maths::twice" not in output


def test_preprocessor_materializes_static_template_helper_from_concrete_method():
    code = """
    template <typename T, int N>
    struct BlockKernel {
      template <typename U = T>
      static void load(const device T* src, thread U dst[N]) {
        for (int i = 0; i < N; ++i) {
          dst[i] = static_cast<U>(src[i]);
        }
      }

      template <typename U = T>
      static U zero() {
        return U(0);
      }

      static void run(const device T* src, device T* out) {
        thread T values[N];
        load(src, values);
        values[0] += zero();
        out[0] = values[0];
      }
    };

    [[kernel]] void k(
        const device float* src [[buffer(0)]],
        device float* out [[buffer(1)]]) {
      BlockKernel<float, 4>::run(src, out);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "void BlockKernel_float_4__load__float(" in output
    assert "float BlockKernel_float_4__zero__float()" in output
    assert "BlockKernel_float_4__load__float(src, values);" in output
    assert "BlockKernel_float_4__zero__float();" in output
    assert "BlockKernel_float_4__run(src, out);" in output
    # The only unqualified call left belongs to the deliberately retained
    # unspecialized template declaration.
    assert output.count("load(src, values)") == 1


_INTEGRAL_CONSTANT_TEST_CONTRACT = """
    template <typename T, T v>
    struct integral_constant {
      static constexpr T value = v;
    };

    template <int Value>
    using Int = integral_constant<int, Value>;

    template <int start, int stop, int step, typename F>
    constexpr void const_for_loop(F f) {
      if constexpr (start < stop) {
        constexpr auto index = Int<start>{};
        f(index);
        const_for_loop<start + step, stop, step, F>(f);
      }
    }
"""

_INTEGRAL_CONSTANT_MULTIPLY_CONTRACT = """
    template <typename T, T lhs, typename U, U rhs>
    constexpr auto operator*(
        integral_constant<T, lhs>, integral_constant<U, rhs>) {
      return integral_constant<decltype(lhs * rhs), lhs * rhs>{};
    }
"""


_INTEGRAL_CONSTANT_CONVERSION_MEMBERS = """
      using value_type = T;
      constexpr operator value_type() const noexcept { return value; }
"""


def _const_for_loop_preprocessor(
    *, with_multiply=True, with_conversion=False, **kwargs
):
    preprocessor = MetalPreprocessor(**kwargs)
    contract = _INTEGRAL_CONSTANT_TEST_CONTRACT
    if with_conversion:
        contract = contract.replace(
            "      static constexpr T value = v;",
            "      static constexpr T value = v;"
            + _INTEGRAL_CONSTANT_CONVERSION_MEMBERS,
        )
    if with_multiply:
        contract += _INTEGRAL_CONSTANT_MULTIPLY_CONTRACT
    preprocessor._configure_integral_constant_contracts(contract)
    return preprocessor


def test_const_for_loop_requires_verified_helper_contract():
    source = "const_for_loop<0, 2, 1>([&](auto index) { use(index.value); });"

    assert (
        MetalPreprocessor()._lower_concrete_const_for_loop_callbacks(source) == source
    )

    different_helper = _INTEGRAL_CONSTANT_TEST_CONTRACT.replace(
        "if constexpr (start < stop)", "if constexpr (start <= stop)"
    )
    preprocessor = MetalPreprocessor()
    preprocessor._configure_integral_constant_contracts(different_helper)

    assert not preprocessor._const_for_loop_contract_verified
    assert preprocessor._lower_concrete_const_for_loop_callbacks(source) == source


def test_const_for_loop_preserves_qualified_helper_calls():
    source = """other::const_for_loop<0, 2, 1>([&](auto row) {
      const_for_loop<0, 2, 1>([&](auto col) { use(row.value, col.value); });
    });"""

    output = _const_for_loop_preprocessor()._lower_concrete_const_for_loop_callbacks(
        source
    )

    assert output == source


def test_const_for_loop_requires_verified_int_alias_contract():
    source = "const_for_loop<0, 2, 1>([&](auto index) { use(index.value); });"
    unrelated_int = _INTEGRAL_CONSTANT_TEST_CONTRACT.replace(
        "using Int = integral_constant<int, Value>;",
        "struct Int { static constexpr int value = Value + 1; };",
    )
    preprocessor = MetalPreprocessor()
    preprocessor._configure_integral_constant_contracts(unrelated_int)

    assert not preprocessor._int_alias_contract_verified
    assert not preprocessor._const_for_loop_contract_verified
    assert preprocessor._lower_concrete_const_for_loop_callbacks(source) == source

    verified = _const_for_loop_preprocessor()
    assert verified._integral_constant_type_parts("other::Int<4>") is None
    assert (
        verified._integral_constant_type_parts("other::integral_constant<int, 4>")
        is None
    )


def test_const_for_loop_requires_verified_integral_constant_value_contract():
    source = "const_for_loop<0, 2, 1>([&](auto index) { use(index.value); });"
    unrelated_constant = _INTEGRAL_CONSTANT_TEST_CONTRACT.replace(
        "static constexpr T value = v;", "static constexpr T value = v + 1;"
    )
    preprocessor = MetalPreprocessor()
    preprocessor._configure_integral_constant_contracts(unrelated_constant)

    assert not preprocessor._integral_constant_contract_verified
    assert not preprocessor._const_for_loop_contract_verified
    assert preprocessor._lower_concrete_const_for_loop_callbacks(source) == source


def test_const_for_loop_requires_exact_integral_constant_operator_contract():
    source = "const_for_loop<0, 2, 1>([&](auto index) { use(index * Int<4>{}); });"
    unrelated_operator = """
    template <typename T, T lhs, typename U, U rhs>
    constexpr auto operator*(
        integral_constant<T, lhs>, integral_constant<U, rhs>) {
      return integral_constant<int, 7>{};
    }
    """
    preprocessor = MetalPreprocessor()
    preprocessor._configure_integral_constant_contracts(
        _INTEGRAL_CONSTANT_TEST_CONTRACT
        + _INTEGRAL_CONSTANT_MULTIPLY_CONTRACT
        + unrelated_operator
    )

    assert "*" not in preprocessor._integral_constant_binary_operators
    assert preprocessor._lower_concrete_const_for_loop_callbacks(source) == source


def test_preprocessor_expands_nested_const_for_loop_in_template_member_ordered():
    code = _INTEGRAL_CONSTANT_TEST_CONTRACT + _INTEGRAL_CONSTANT_MULTIPLY_CONTRACT + """

    struct FragmentShape {
      static constexpr int RowStride = 4;
      static constexpr int ColStride = 1;
    };

    struct Fragment {
      template <typename U, typename Row, typename Col>
      METAL_FUNC static void load(const device U* src, Row row, Col col) {
        U value = src[row.value + col.value];
      }
    };

    struct ConcreteTile {
      static constexpr int Rows = 2;
      static constexpr int Cols = 2;
      static constexpr int RowStride = FragmentShape::RowStride;
      static constexpr int ColStride = FragmentShape::ColStride;

      template <typename U>
      METAL_FUNC void load(const device U* src) {
        const_for_loop<0, Rows, 1>([&](auto idx_row) {
          const_for_loop<0, Cols, 1>([&](auto idx_col) {
            Fragment::load(
                src,
                idx_row * Int<RowStride>{},
                idx_col * Int<ColStride>{});
          });
        });
      }
    };

    kernel void k(const device float* src [[buffer(0)]]) {
      ConcreteTile tile;
      tile.load(src);
    }
    """

    output = MetalPreprocessor().preprocess(code)
    expected = [
        (
            "Fragment__load__float_integral_constant_int_0_integral_constant_int_0",
            "integral_constant<int,0>",
            "integral_constant<int,0>",
        ),
        (
            "Fragment__load__float_integral_constant_int_0_integral_constant_int_1",
            "integral_constant<int,0>",
            "integral_constant<int,1>",
        ),
        (
            "Fragment__load__float_integral_constant_int_4_integral_constant_int_0",
            "integral_constant<int,4>",
            "integral_constant<int,0>",
        ),
        (
            "Fragment__load__float_integral_constant_int_4_integral_constant_int_1",
            "integral_constant<int,4>",
            "integral_constant<int,1>",
        ),
    ]

    calls = re.findall(r"(?m)^\s+(Fragment__load__[A-Za-z0-9_]+)\(src,$", output)
    definitions = re.findall(
        r"void (Fragment__load__[A-Za-z0-9_]+)\("
        r"const device float\* src, (integral_constant<int,\d+>) row, "
        r"(integral_constant<int,\d+>) col\)",
        output,
    )

    assert calls == [name for name, _, _ in expected]
    assert definitions == expected
    assert "integral_constant<int, 1>{} * Int<(4)>{}" in output
    assert "integral_constant<int, 1>{} * Int<(1)>{}" in output
    assert "src[0 + 0]" in output
    assert "src[0 + 1]" in output
    assert "src[4 + 0]" in output
    assert "src[4 + 1]" in output
    assert "row.value" not in output
    assert "col.value" not in output
    assert output.count("const_for_loop<") == 1
    assert "[&](auto" not in output
    assert "idx_row" not in output
    assert "idx_col" not in output
    assert "METAL_FUNC Fragment__load" not in output


def test_preprocessor_expands_const_for_loop_in_concrete_struct_specialization():
    code = _INTEGRAL_CONSTANT_TEST_CONTRACT + _INTEGRAL_CONSTANT_MULTIPLY_CONTRACT + """
    struct Fragment {
      template <typename U, typename Row, typename Col>
      static void load(const device U* src, Row row, Col col) {
        U value = src[row.value + col.value];
      }
    };

    template <typename T, int Rows_, int Cols_>
    struct Tile {
      static constexpr int Rows = Rows_;
      static constexpr int Cols = Cols_;

      template <typename U>
      void load(const device U* src) {
        const_for_loop<0, Rows, 1>([&](auto row) {
          const_for_loop<0, Cols, 1>([&](auto col) {
            Fragment::load(src, row * Int<4>{}, col * Int<1>{});
          });
        });
      }
    };

    kernel void k(const device float* src [[buffer(0)]]) {
      Tile<float, 2, 2> tile;
      tile.load(src);
    }
    """

    output = MetalPreprocessor().preprocess(code)
    calls = re.findall(
        r"(?m)^\s+Fragment__load__(?P<types>[A-Za-z0-9_]+)\(src,", output
    )

    assert "void Tile_float_2_2__load__float(" in output
    assert calls == [
        "float_integral_constant_int_0_integral_constant_int_0",
        "float_integral_constant_int_0_integral_constant_int_1",
        "float_integral_constant_int_4_integral_constant_int_0",
        "float_integral_constant_int_4_integral_constant_int_1",
    ]


def test_preprocessor_template_member_binding_drops_expression_reference():
    preprocessor = MetalPreprocessor()
    struct = preprocessor._find_concrete_struct_definitions("""
        struct Consumer {
          template <typename T>
          static void consume(thread T& value) { value = T(0); }
        };
        """)[0]
    method = struct.template_methods[0]

    bindings = preprocessor._bind_template_method_parameters(
        method,
        ["float&"],
        owner_struct=struct,
        structs_by_name={struct.name: struct},
    )

    assert bindings == {"T": "float"}


def test_preprocessor_explicit_template_binding_allows_untyped_parameter():
    preprocessor = MetalPreprocessor()
    struct = preprocessor._find_concrete_struct_definitions("""
        struct Consumer {
          template <typename T>
          static T consume(float untyped) { return T(untyped); }
        };
        """)[0]
    method = struct.template_methods[0]

    bindings = preprocessor._bind_template_method_parameters(
        method,
        [None],
        explicit_template_arguments=["int"],
        owner_struct=struct,
        structs_by_name={struct.name: struct},
    )

    assert bindings == {"T": "int"}


@pytest.mark.parametrize(
    ("outer_stop", "after_inner"),
    [
        ("2", "{ int idx_row = 1; out[0] = idx_row; }"),
        ("unresolved_stop", ""),
    ],
    ids=["shadowed-parameter", "unresolved-bounds"],
)
def test_preprocessor_preserves_whole_non_expandable_nested_const_for_loop(
    outer_stop, after_inner
):
    code = _INTEGRAL_CONSTANT_TEST_CONTRACT + """
    struct ConcreteTile {
      template <typename U>
      void load(device U* out) {
        const_for_loop<0, OUTER_STOP, 1>([&](auto idx_row) {
          const_for_loop<0, 2, 1>([&](auto idx_col) {
            out[idx_row.value * 2 + idx_col.value] = U(1);
          });
          AFTER_INNER
        });
      }
    };

    kernel void k(device float* out [[buffer(0)]]) {
      ConcreteTile tile;
      tile.load(out);
    }
    """
    code = code.replace("OUTER_STOP", outer_stop).replace("AFTER_INNER", after_inner)

    output = MetalPreprocessor().preprocess(code)

    assert "void ConcreteTile__load__float(" in output
    assert f"const_for_loop<0, {outer_stop}, 1>" in output
    assert output.count("const_for_loop<") == 3
    assert output.count("[&](auto") == 2
    assert output.count("out[idx_row.value * 2 + idx_col.value]") == 1
    assert "integral_constant<int, 0>{}" not in output


def test_const_for_loop_lowers_nested_callback_returns_to_iteration_escape():
    source = """const_for_loop<0, 2, 1>([&](auto row) {
      if (discard_row(row.value)) { return; }
      const_for_loop<0, 2, 1>([&](auto col) {
        if (discard_col(col.value)) { return; }
        int converted = base + col;
        store(row.value, converted);
      });
      after(row.value);
    });"""

    output = _const_for_loop_preprocessor(
        with_conversion=True
    )._lower_concrete_const_for_loop_callbacks(source)

    assert "const_for_loop<" not in output
    assert "return;" not in output
    assert output.count("do {") == 6
    assert output.count("break;") == 6
    assert output.count("while (false);") == 6
    assert output.count("int converted = base + 0;") == 2
    assert output.count("int converted = base + 1;") == 2
    assert "store(0, converted);" in output
    assert "store(1, converted);" in output
    assert "after(0);" in output
    assert "after(1);" in output


def test_const_for_loop_requires_verified_integral_constant_conversion():
    source = """const_for_loop<0, 2, 1>([&](auto idx) {
      use(offset + idx);
    });"""

    unverified = _const_for_loop_preprocessor(
        with_conversion=False
    )._lower_concrete_const_for_loop_callbacks(source)
    verified = _const_for_loop_preprocessor(
        with_conversion=True
    )._lower_concrete_const_for_loop_callbacks(source)

    assert unverified == source
    assert "const_for_loop<" not in verified
    assert "use(offset + 0);" in verified
    assert "use(offset + 1);" in verified


@pytest.mark.parametrize(
    "callback_body",
    [
        "return idx.value;",
        "for (int i = 0; i < 2; ++i) { if (i) { return; } }",
        "switch (idx.value) { case 0: return; default: break; }",
    ],
    ids=["value-return", "return-in-loop", "return-in-switch"],
)
def test_const_for_loop_preserves_unsupported_callback_return(callback_body):
    source = f"const_for_loop<0, 2, 1>([&](auto idx) {{ {callback_body} }});"

    output = _const_for_loop_preprocessor()._lower_concrete_const_for_loop_callbacks(
        source
    )

    assert output == source


@pytest.mark.parametrize(
    "callback_body",
    [
        "auto nested = [idx](auto value) { return value; };",
        "auto nested = [&](auto idx) { return idx.value; };",
        "for (auto idx : values) { consume(idx); }",
        "consume(&idx);",
        "consume(&idx.value);",
    ],
    ids=[
        "nested-capture",
        "nested-shadow",
        "range-shadow",
        "address-taken",
        "value-address-taken",
    ],
)
def test_const_for_loop_preserves_unsafe_parameter_uses(callback_body):
    source = "const_for_loop<0, 2, 1>([&](auto idx) { " f"{callback_body}" " });"

    output = _const_for_loop_preprocessor()._lower_concrete_const_for_loop_callbacks(
        source
    )

    assert output == source


def test_const_for_loop_parenthesizes_negative_value_substitutions():
    source = """const_for_loop<-1, 0, 1>([&](auto index) {
      use(x-index.value, -index.value);
    });"""

    output = _const_for_loop_preprocessor()._lower_concrete_const_for_loop_callbacks(
        source
    )

    assert "use(x-(-1), -(-1));" in output


def test_integral_constant_parameter_substitution_preserves_nested_lambda_shadow():
    body = """auto nested = [](auto index) { use(index.value); };
    use(index.value);"""
    preprocessor = _const_for_loop_preprocessor()

    output = preprocessor._substitute_integral_constant_parameter_values(
        "integral_constant<int, -1> index", body
    )

    assert output == body


@pytest.mark.parametrize(
    "source",
    [
        "return const_for_loop<0, 2, 1>([&](auto idx) { use(idx.value); });",
        "(const_for_loop<0, 2, 1>([&](auto idx) { use(idx.value); }), 0);",
        "const_for_loop<2, 0, -1>([&](auto idx) { use(idx.value); });",
    ],
    ids=["return-expression", "comma-expression", "descending-step"],
)
def test_const_for_loop_preserves_unsupported_call_context(source):
    output = _const_for_loop_preprocessor()._lower_concrete_const_for_loop_callbacks(
        source
    )

    assert output == source


@pytest.mark.parametrize(
    "outer_bound",
    ["(N > 0 ? N : 0)", "(N >> 1)"],
    ids=["comparison", "right-shift"],
)
def test_const_for_loop_preserves_nested_callbacks_in_symbolic_expression_bounds(
    outer_bound,
):
    source = f"""const_for_loop<0, {outer_bound}, 1>(/* callback */ [&](auto row) {{
      const_for_loop<0, 2, 1>([&](auto col) {{
        use(row.value, col.value);
      }});
    }});"""

    output = _const_for_loop_preprocessor()._lower_concrete_const_for_loop_callbacks(
        source
    )

    assert output == source


def test_const_for_loop_requires_proven_integral_constant_operator():
    source = "const_for_loop<0, 2, 1>([&](auto idx) { " "use(idx * Int<4>{});" " });"

    output = _const_for_loop_preprocessor(
        with_multiply=False
    )._lower_concrete_const_for_loop_callbacks(source)

    assert output == source


def test_const_for_loop_statement_context_skips_directives_and_comments():
    source = """
    #pragma clang loop unroll(full)
    // The directive and comment do not change statement context.
    const_for_loop<0, 2, 1>([&](auto idx) { use(idx.value); });
    """

    output = _const_for_loop_preprocessor()._lower_concrete_const_for_loop_callbacks(
        source
    )

    assert "const_for_loop<" not in output
    assert "use(0);" in output
    assert "use(1);" in output


def test_const_for_loop_expansion_budget_is_cumulative_for_nested_loops():
    source = """
    const_for_loop<0, 2, 1>([&](auto row) {
      const_for_loop<0, 2, 1>([&](auto col) {
        use(row.value, col.value);
      });
    });
    """
    exact = _const_for_loop_preprocessor(max_template_specializations=6)

    output = exact._lower_concrete_const_for_loop_callbacks(source)

    assert "const_for_loop<" not in output
    with pytest.raises(MetalTemplateSpecializationError) as exc_info:
        _const_for_loop_preprocessor(
            max_template_specializations=5
        )._lower_concrete_const_for_loop_callbacks(source)
    assert exc_info.value.required_work_items == 6
    assert "6 cumulative callback expansions requested" in str(exc_info.value)


def test_const_for_loop_expansion_budget_limits_nesting_depth():
    source = """
    const_for_loop<0, 1, 1>([&](auto row) {
      const_for_loop<0, 1, 1>([&](auto col) {
        use(row.value, col.value);
      });
    });
    """

    with pytest.raises(MetalTemplateSpecializationError, match="nesting depth 2"):
        _const_for_loop_preprocessor(
            max_template_specializations=8,
            max_expansion_depth=1,
        )._lower_concrete_const_for_loop_callbacks(source)


def test_const_for_loop_negative_values_have_distinct_specialization_names():
    code = _INTEGRAL_CONSTANT_TEST_CONTRACT + _INTEGRAL_CONSTANT_MULTIPLY_CONTRACT + """
    struct Sink {
      template <typename Index>
      static void take(Index index) { int value = index.value; }
    };
    struct Runner {
      void run() {
        const_for_loop<-1, 2, 1>([&](auto index) {
          Sink::take(index * Int<1>{});
        });
      }
    };
    kernel void k() { Runner runner; runner.run(); }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Sink__take__integral_constant_int_negative_1" in output
    assert "Sink__take__integral_constant_int_0" in output
    assert "Sink__take__integral_constant_int_1" in output
    assert "int value = (-1);" in output
    assert "int value = 0;" in output
    assert "int value = 1;" in output


def test_preprocessor_template_helper_deduction_precedes_default():
    code = """
    template <typename T, int N>
    struct BlockKernel {
      template <typename U = T>
      static void load(const device T* src, thread U dst[N]) {
        for (int i = 0; i < N; ++i) {
          dst[i] = static_cast<U>(src[i]);
        }
      }

      static void run(const device T* src, device half* out) {
        thread half values[N];
        load(src, values);
        out[0] = values[0];
      }
    };

    [[kernel]] void k(
        const device float* src [[buffer(0)]],
        device half* out [[buffer(1)]]) {
      BlockKernel<float, 4>::run(src, out);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "void BlockKernel_float_4__load__half(" in output
    assert "thread half dst[4]" in output
    assert "BlockKernel_float_4__load__half(src, values);" in output
    assert "BlockKernel_float_4__load__float(" not in output


def test_preprocessor_materializes_explicit_static_template_helper_from_method():
    code = """
    template <typename T, int N>
    struct BlockKernel {
      template <typename U = T>
      static void load(const device T* src, thread U dst[N]) {
        for (int i = 0; i < N; ++i) {
          dst[i] = static_cast<U>(src[i]);
        }
      }

      static void run(const device T* src, device T* out) {
        thread T values[N];
        load<float>(src, values);
        out[0] = values[0];
      }
    };

    [[kernel]] void k(
        const device float* src [[buffer(0)]],
        device float* out [[buffer(1)]]) {
      BlockKernel<float, 4>::run(src, out);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "void BlockKernel_float_4__load__float(" in output
    assert "BlockKernel_float_4__load__float(src, values);" in output
    assert "BlockKernel_float_4__run(src, out);" in output
    assert output.count("load<float>(src, values)") == 1


def test_preprocessor_expands_omitted_template_member_defaults():
    code = """
    struct Tag {};
    struct Loader {
      float value;
      Loader(float input) : value(input) {}
    };

    struct Runner {
      using loader_t = Loader;

      template <bool Enabled>
      static float apply(
          thread loader_t& loader,
          float value,
          int bias = 2,
          Tag tag = {}) {
        return Enabled ? loader.value + value + bias : value;
      }

      static float run(float value) {
        thread loader_t loader(value);
        return apply<true>(loader, value);
      }
    };

    kernel void k(device float* output [[buffer(0)]]) {
      output[0] = Runner::run(output[0]);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Runner__apply__true(loader, value, 2, Tag{})" in output
    assert "thread Loader& loader" in output
    assert "int bias, Tag tag)" in output
    assert "int bias = 2" not in output
    assert output.count("apply<true>(loader, value)") == 0

    repeated_output = MetalPreprocessor().preprocess(output)

    assert "Runner__apply__true(loader, value, 2, Tag{})" in repeated_output
    assert "loader_t loader" not in repeated_output


def test_preprocessor_materializes_reference_and_static_member_defaults():
    code = """
    struct Tag {};

    struct Runner {
      static constexpr int default_bias = 3;

      template <typename T>
      static T apply(
          T value,
          const Tag& tag = {},
          int bias = default_bias) {
        return value + T(bias);
      }
    };

    kernel void k(device float* output [[buffer(0)]]) {
      output[0] = Runner::apply(output[0]);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Runner__apply__float(output[0], Tag{}, (3))" in output
    assert "const Tag& tag, int bias)" in output
    assert "const Tag&{}" not in output
    assert "bias = default_bias" not in output


def test_preprocessor_preserves_parameter_named_like_struct_alias():
    code = """
    struct Runner {
      using value_t = float;

      template <typename T>
      static T apply(float value_t, T input) {
        return input + T(value_t);
      }
    };

    kernel void k(device float* output [[buffer(0)]]) {
      output[0] = Runner::apply(output[0], output[0]);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Runner__apply__float(float value_t, float input)" in output
    assert "float float" not in output


def test_preprocessor_preserves_method_local_alias_shadowing_struct_alias():
    code = """
    struct Runner {
      using value_t = float;

      template <bool Enabled>
      static int apply(int input) {
        using value_t = int;
        value_t local = input;
        return Enabled ? local : input;
      }

      static int run(int input) {
        return apply<true>(input);
      }
    };

    kernel void k(device int* output [[buffer(0)]]) {
      output[0] = Runner::run(output[0]);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "using value_t = int" in output
    assert "value_t local = input" in output
    assert "float local = input" not in output


def test_preprocessor_rejects_omitted_required_template_member_parameter():
    code = """
    struct Runner {
      template <bool Enabled>
      static float apply(float value, int required, int bias = 2) {
        return Enabled ? value + required + bias : value;
      }

      static float run(float value) {
        return apply<true>(value);
      }
    };

    kernel void k(device float* output [[buffer(0)]]) {
      output[0] = Runner::run(output[0]);
    }
    """

    with pytest.raises(MetalStructMethodError):
        MetalPreprocessor().preprocess(code)


def test_preprocessor_rejects_conflicting_explicit_member_template_binding():
    code = """
    struct Runner {
      template <typename T>
      static T convert(T value) {
        return value;
      }

      static float run(float value) {
        return convert<int>(value);
      }
    };

    kernel void k(device float* output [[buffer(0)]]) {
      output[0] = Runner::run(output[0]);
    }
    """

    with pytest.raises(MetalStructMethodError, match="did not bind consistently"):
        MetalPreprocessor().preprocess(code)


def test_preprocessor_resolves_qualified_struct_member_alias_for_template_call():
    code = """
    struct Loader {
      float value;
      Loader(float input) : value(input) {}
    };

    struct Runner {
      using loader_t = Loader;

      template <typename U>
      static U apply(thread loader_t& loader, U value, int bias = 2) {
        return loader.value + value + bias;
      }
    };

    kernel void k(device float* output [[buffer(0)]]) {
      using runner_t = Runner;
      using loader_t = typename runner_t::loader_t;
      thread loader_t loader(output[0]);
      output[0] = runner_t::apply(loader, output[0]);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "Runner__apply__float(loader, output[0], 2)" in output
    assert "Runner__apply__float(thread Loader& loader" in output
    assert "runner_t::apply(loader, output[0])" not in output


def test_preprocessor_resolves_member_alias_from_materialization_provenance():
    code = """
    struct Owner {
      using value_t = Materialized_Value_with_tag;
    };

    void use_value() {
      using owner_t = Owner;
      using value_t = typename owner_t::value_t;
      value_t value{};
    }
    """
    preprocessor = MetalPreprocessor()
    structs = preprocessor._find_concrete_struct_definitions(code)
    structs_by_name = {struct.name: struct for struct in structs}
    struct_spans = [struct.span for struct in structs]
    preprocessor._materialized_struct_specializations["Materialized_Value_with_tag"] = (
        "Value",
        ("with-tag",),
    )

    aliases = preprocessor._collect_struct_type_aliases(
        code,
        set(structs_by_name),
        struct_spans,
        structs_by_name,
    )
    variables = preprocessor._collect_aliased_struct_variable_types(
        code,
        aliases,
        struct_spans,
    )

    assert aliases["value_t"][0].target == "Materialized_Value_with_tag"
    assert variables["value"][0][1] == "Materialized_Value_with_tag"


def test_preprocessor_collects_pointer_to_aliased_struct_parameter():
    code = """
    struct Params { int stride; };
    using params_t = Params;

    void use_params(const constant params_t* params) {
      int stride = params->stride;
    }
    """
    preprocessor = MetalPreprocessor()
    structs = preprocessor._find_concrete_struct_definitions(code)
    structs_by_name = {struct.name: struct for struct in structs}
    struct_spans = [struct.span for struct in structs]
    aliases = preprocessor._collect_struct_type_aliases(
        code,
        set(structs_by_name),
        struct_spans,
        structs_by_name,
    )

    variables = preprocessor._collect_aliased_struct_variable_types(
        code,
        aliases,
        struct_spans,
    )
    field_types = preprocessor._struct_field_types_at(
        variables,
        structs_by_name,
        code.index("params->stride"),
    )

    assert variables["params"][0][1] == "Params*"
    assert field_types == {"params->": {"stride": "int"}}


def test_preprocessor_deduces_template_bindings_from_materialization_provenance():
    preprocessor = MetalPreprocessor()
    preprocessor._materialized_struct_specializations["Holder_metal_half_t_true"] = (
        "Holder",
        ("metal::half_t", "true"),
    )
    bindings = {}

    preprocessor._infer_template_parameter_bindings_from_type(
        "Holder<T, Enabled>",
        "Holder_metal_half_t_true",
        {"T", "Enabled"},
        bindings,
    )

    assert bindings == {"T": "metal::half_t", "Enabled": "true"}

    plain_binding = {}
    preprocessor._infer_template_parameter_bindings_from_type(
        "T",
        "Holder_metal_half_t_true",
        {"T"},
        plain_binding,
    )

    assert plain_binding == {"T": "Holder_metal_half_t_true"}


def test_preprocessor_template_method_instantiations_are_deduplicated():
    # Two call sites with the same inferred argument type share ONE concrete
    # free function (deduplicated by struct/method/bindings).
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device const float* in [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        float a = op.reduce(in[i]);
        out[i] = op.reduce(a);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert output.count("float Sum__reduce__float(thread Sum& self, float val)") == 1
    assert "Sum__reduce__float(op, in[i])" in output
    assert "Sum__reduce__float(op, a)" in output


def test_preprocessor_template_method_uninferable_argument_clean_fails():
    # An un-inferable argument shape (a function-call result) raises a clean
    # MetalStructMethodError that PROPAGATES out of preprocess rather than
    # leaving a dangling `op.reduce(...)` / broken output.
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val; } };
    float helper(uint i){ return 1.0; }

    kernel void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        out[i] = op.reduce(helper(i));
    }
    """
    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)
    assert excinfo.value.struct_name == "Sum"
    assert excinfo.value.method_name == "reduce"
    assert (
        excinfo.value.project_diagnostic_code == "project.translate.metal-struct-method"
    )


def test_preprocessor_addressed_element_side_effect_clean_fails_structured():
    code = """
    struct Fragment {
      template <typename SrcPtrType>
      static float load(SrcPtrType src) { return float(src[0]); }
    };

    kernel void k(
        const device half* src [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
      out[index] = Fragment::load(&(src[index++]));
    }
    """

    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)

    error = excinfo.value
    assert error.project_diagnostic_code == "project.translate.metal-struct-method"
    assert error.missing_capabilities == ("struct.template-method",)
    assert error.struct_name == "Fragment"
    assert error.method_name == "load"
    assert error.requested_signature == "Fragment.load(&(src[index++]))"
    assert error.suggested_action
    assert error.source_location["length"] == len("load(&(src[index++]))")
    assert "could not be inferred conservatively" in str(error)


@pytest.mark.parametrize("argument", ["src", "src + offset"])
def test_preprocessor_generic_pointer_without_address_space_clean_fails(argument):
    code = f"""
    struct Loader {{
      template <typename Pointer>
      float load(Pointer src) {{ return src[0]; }}
    }};

    void invoke(float* src, int offset) {{
      Loader loader;
      float value = loader.load({argument});
    }}
    """

    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)

    error = excinfo.value
    assert error.project_diagnostic_code == "project.translate.metal-struct-method"
    assert error.missing_capabilities == ("struct.template-method",)
    assert error.struct_name == "Loader"
    assert error.method_name == "load"
    assert error.requested_signature == f"loader.load({argument})"
    assert "could not be inferred conservatively" in str(error)


@pytest.mark.parametrize(
    "cast_expression",
    [
        "static_cast<float*>(src)",
        "static_cast<const float*>(src)",
        "static_cast<device half**>(src)",
        "static_cast<device thread half*>(src)",
        "static_cast<device device half*>(src)",
    ],
)
def test_preprocessor_generic_pointer_cast_without_single_address_space_clean_fails(
    cast_expression,
):
    code = f"""
    struct Loader {{
      template <typename Pointer>
      float load(Pointer src) {{ return float(src[0]); }}
    }};

    void invoke(const device half* src) {{
      Loader loader;
      float value = loader.load({cast_expression});
    }}
    """

    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)

    error = excinfo.value
    assert error.project_diagnostic_code == "project.translate.metal-struct-method"
    assert error.missing_capabilities == ("struct.template-method",)
    assert error.requested_signature == f"loader.load({cast_expression})"
    assert "could not be inferred conservatively" in str(error)


def test_preprocessor_unqualified_auto_pointer_clean_fails():
    code = """
    struct Loader {
      template <typename Pointer>
      float load(Pointer src) { return float(src[0]); }
    };

    void invoke(const device half* src) {
      Loader loader;
      auto pointer = static_cast<half*>(src);
      float value = loader.load(pointer);
    }
    """

    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)

    error = excinfo.value
    assert error.project_diagnostic_code == "project.translate.metal-struct-method"
    assert error.missing_capabilities == ("struct.template-method",)
    assert error.requested_signature == "loader.load(pointer)"
    assert "could not be inferred conservatively" in str(error)


def test_preprocessor_declared_unqualified_pointer_parameter_clean_fails():
    code = """
    struct Loader {
      template <typename U>
      float load(U* src) { return float(src[0]); }
    };

    kernel void k(
        const device half* src [[buffer(0)]],
        device float* out [[buffer(1)]]) {
      Loader loader;
      out[0] = loader.load(src);
    }
    """

    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)

    error = excinfo.value
    assert error.project_diagnostic_code == "project.translate.metal-struct-method"
    assert error.missing_capabilities == ("struct.template-method",)
    assert error.requested_signature == "loader.load(src)"
    assert "did not bind consistently" in str(error)


def test_preprocessor_template_method_conflicting_binding_clean_fails():
    # The single template parameter T appears in two parameters whose call-site
    # arguments infer to DIFFERENT concrete types (float and int). Rather than
    # silently keeping the first guess, this clean-fails (never guess).
    code = """
    struct D { template <typename T> T pick(T a, T b){ return a; } };

    kernel void k(
        device float* o [[buffer(0)]],
        device int* q [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        D d;
        o[i] = d.pick(o[i], q[i]);
    }
    """
    with pytest.raises(MetalStructMethodError):
        MetalPreprocessor().preprocess(code)


def test_preprocessor_template_method_consistent_multi_param_instantiates():
    # A multi-parameter template method whose parameters consistently bind T to
    # one type instantiates to a single concrete free function.
    code = """
    struct D { template <typename T> T pick(T a, T b){ return a; } };

    kernel void k(
        device float* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        D d;
        o[i] = d.pick(o[i], o[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float D__pick__float(thread D& self, float a, float b)" in output
    assert "D__pick__float(d, o[i], o[i])" in output


def test_preprocessor_template_method_unbound_parameter_clean_fails():
    # A template parameter that cannot be deduced from any argument (here T only
    # appears in the return type) clean-fails rather than guessing.
    code = """
    struct Maker { template <typename T> T make(float a){ return T(a); } };

    kernel void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Maker m;
        out[i] = m.make(out[i]);
    }
    """
    with pytest.raises(MetalStructMethodError):
        MetalPreprocessor().preprocess(code)


def test_preprocessor_template_method_lowering_is_noop_without_template_calls():
    # Regression-safety: a struct that DECLARES a template method but never CALLS
    # it is left untouched (the uninstantiated template would be dropped by the
    # parser exactly as before; no clean-fail, no spurious free function).
    code = (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct Sum { float x; template <typename T> T reduce(T v){ return v; } };\n"
        "kernel void k(device float* o [[buffer(0)]],"
        " uint i [[thread_position_in_grid]]) {\n"
        "    o[i] = o[i] * 2.0;\n"
        "}\n"
    )
    output = MetalPreprocessor().preprocess(code)
    # No instantiation emitted and no exception raised; the data member survives.
    assert "Sum__reduce" not in output
    assert "float x;" in output


def test_preprocessor_template_method_full_pipeline_to_hlsl():
    # End-to-end: the Sum reduction example translates to valid HLSL with the
    # correct T=float instantiation (simd_sum -> WaveActiveSum) and no dangling
    # simd_reduce call.
    from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
    from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
    from crosstl.translator.lexer import Lexer as CrossGLLexer
    from crosstl.translator.parser import Parser as CrossGLParser

    code = (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "template <typename U> struct Sum {"
        " template <typename T> T simd_reduce(T val){ return simd_sum(val); }"
        " U operator()(U a, U b){ return a + b; } };\n"
        "kernel void reduce(device const float* in [[buffer(0)]],"
        " device float* out [[buffer(1)]],"
        " uint i [[thread_position_in_grid]]) {\n"
        "    Sum<float> op;\n"
        "    float v = op.simd_reduce(in[i]);\n"
        "    out[i] = op(v, 0.0);\n"
        "}\n"
    )
    pre = MetalPreprocessor().preprocess(code)
    assert (
        "float Sum_float__simd_reduce__float(thread Sum_float& self, float val)" in pre
    )
    tokens = MetalLexer(pre).tokenize()
    ast = MetalParser(tokens).parse()
    crossgl = MetalToCrossGLConverter().generate(ast)
    parsed = CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()
    hlsl = HLSLCodeGen().generate(parsed)
    # No dangling receiver call and no bare Metal `simd_reduce(` builtin survive
    # (the lowered free-function NAME legitimately contains the substring).
    assert "op.simd_reduce(" not in hlsl
    assert " simd_reduce(" not in hlsl
    assert "WaveActiveSum" in hlsl
    assert "Sum_float__simd_reduce__float(op, in_.Load(i))" in hlsl


def test_preprocessor_instantiates_template_method_from_local_array_subscript():
    # A CALLED template member method whose argument is a LOCAL ARRAY subscript
    # (`totals[i]`, `T totals[N]`) instantiates with T = the array element type.
    # This is the dominant argument shape in MLX reduce, where reduction
    # accumulators are stack arrays.
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device float* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        float totals[4];
        totals[i] = out[i];
        out[i] = op.reduce(totals[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Sum__reduce__float(thread Sum& self, float val)" in output
    assert "Sum__reduce__float(op, totals[i])" in output


def test_preprocessor_instantiates_template_method_from_threadgroup_array_subscript():
    # A `threadgroup U sh[N]` array local subscript is element-typed too.
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device half* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        threadgroup half shared_vals[32];
        out[i] = op.reduce(shared_vals[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "half Sum__reduce__half(thread Sum& self, half val)" in output
    assert "Sum__reduce__half(op, shared_vals[i])" in output


def test_preprocessor_instantiates_template_method_from_member_access_subscript():
    # A member-access subscript `obj.member[i]` is element-typed via the struct
    # field's element type, even when the carrier struct (`Buf`) has NO methods.
    code = """
    struct Buf { device float* data; };
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device float* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        Buf init;
        o[i] = op.reduce(init.data[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Sum__reduce__float(thread Sum& self, float val)" in output
    assert "Sum__reduce__float(op, init.data[i])" in output


def test_preprocessor_instantiates_template_method_from_stdint_array_subscript():
    # C stdint scalar aliases (`int32_t`, `uint8_t`, ...) are recognized element
    # types, so an `int32_t values[N]` array subscript binds a called template
    # member method's parameter to int32_t. This is the shape that blocked MLX
    # scan (`op.simd_exclusive_scan(values[i])` on `int32_t values[N]`).
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device int* out [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        int32_t values[4];
        values[i] = out[i];
        out[i] = op.reduce(values[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "int32_t Sum__reduce__int32_t(thread Sum& self, int32_t val)" in output
    assert "Sum__reduce__int32_t(op, values[i])" in output


def test_infer_argument_type_recognizes_stdint_scalar_aliases():
    # stdint scalar aliases infer as themselves across the subscript / cast /
    # bare-local shapes (they carry known sizes for SFINAE `sizeof(T)`).
    pp = MetalPreprocessor()
    assert pp._infer_argument_type("vals[i]", {"vals": "int32_t"}, {}) == "int32_t"
    assert pp._infer_argument_type("uint8_t(x)", {}, {}) == "uint8_t"
    assert pp._infer_argument_type("v", {}, {"v": "int64_t"}) == "int64_t"


def test_infer_argument_type_recognizes_pointer_arguments_and_offsets():
    pp = MetalPreprocessor()
    buffers = {
        "input": "half",
        "output": "float",
        "same_type_input": "half",
        "other": "int",
    }
    locals_ = {"offset": "int", "wide_offset": "uint64_t"}

    assert pp._infer_argument_type("output", buffers, locals_) == "float"
    assert pp._infer_argument_type("input + offset", buffers, locals_) == "half"
    assert pp._infer_argument_type("wide_offset + input", buffers, locals_) == "half"
    assert pp._infer_argument_type("input - offset", buffers, locals_) == "half"
    assert pp._infer_argument_type("input + offset + 1", buffers, locals_) == "half"
    promoted_locals = {"lane": "short", "stride": "int", "index": "int"}
    assert pp._infer_argument_type("lane * stride", {}, promoted_locals) == "int"
    assert (
        pp._infer_argument_type(
            "input + lane * stride + index", buffers, promoted_locals
        )
        == "half"
    )
    assert (
        pp._infer_argument_type(
            "lane2 * stride", {}, {"lane2": "short2", "stride": "int"}
        )
        is None
    )
    assert pp._infer_argument_type("input + other", buffers, locals_) is None
    assert pp._infer_argument_type("input - same_type_input", buffers, locals_) is None
    assert pp._infer_argument_type("2 - input", buffers, locals_) is None
    assert pp._infer_argument_type("input + 1.0", buffers, locals_) is None
    assert pp._infer_argument_type("input * 2", buffers, locals_) is None
    assert pp._infer_argument_type("input / 2", buffers, locals_) is None
    assert pp._infer_argument_type("input * offset", buffers, locals_) is None


def test_infer_argument_type_preserves_addressed_element_pointer_type():
    pp = MetalPreprocessor()
    code = """
    kernel void k(const device half* src [[buffer(0)]]) {
      threadgroup float scratch[64];
    }
    """
    positioned = pp._collect_buffer_element_types(code, [])
    buffers = pp._flatten_types_at(positioned, len(code))

    assert (
        pp._infer_argument_type(
            "&(src[(i * 8) * 36 + (j * 8)])",
            buffers,
            {"i": "short", "j": "short"},
        )
        == "const device half*"
    )
    assert (
        pp._infer_argument_type("&scratch[i + 1]", buffers, {"i": "ushort"})
        == "threadgroup float*"
    )


def test_infer_argument_type_preserves_qualified_pointer_expressions():
    pp = MetalPreprocessor()
    code = "kernel void k(const device half* src [[buffer(0)]]) {}"
    positioned = pp._collect_buffer_element_types(code, [])
    buffers = pp._flatten_types_at(positioned, len(code))
    locals_ = {"offset": "uint"}

    assert pp._infer_argument_type("src", buffers, locals_) == "const device half*"
    assert (
        pp._infer_argument_type("src + offset", buffers, locals_)
        == "const device half*"
    )
    assert (
        pp._infer_argument_type("offset + src", buffers, locals_)
        == "const device half*"
    )
    assert (
        pp._infer_argument_type("src - offset", buffers, locals_)
        == "const device half*"
    )
    assert pp._infer_argument_type("offset - src", buffers, locals_) is None


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("static_cast<const device half*>(src)", "const device half*"),
        (
            "static_cast<threadgroup volatile float*>(src)",
            "threadgroup volatile float*",
        ),
        ("static_cast<float*>(src)", None),
        ("static_cast<const float*>(src)", None),
        ("static_cast<device float**>(src)", None),
        ("static_cast<device thread float*>(src)", None),
        ("static_cast<device device float*>(src)", None),
    ],
)
def test_infer_argument_type_validates_pointer_static_cast_target(expression, expected):
    assert MetalPreprocessor()._infer_argument_type(expression, {}, {}) == expected


@pytest.mark.parametrize(
    ("local_type", "expected"),
    [
        ("const threadgroup half*", "const threadgroup half*"),
        ("half*", None),
        ("threadgroup half**", None),
        ("device thread half*", None),
    ],
)
def test_infer_argument_type_validates_propagated_local_pointer_type(
    local_type, expected
):
    pp = MetalPreprocessor()
    assert pp._infer_argument_type("pointer", {}, {"pointer": local_type}) == expected
    offset_expected = expected if expected is not None else None
    assert (
        pp._infer_argument_type(
            "pointer + offset", {}, {"pointer": local_type, "offset": "uint"}
        )
        == offset_expected
    )


def test_infer_argument_type_validates_cached_pointer_return_type():
    pp = MetalPreprocessor()
    pp._record_known_member_function_return_type(
        "Source__addressed", "const device half*"
    )
    pp._known_member_function_return_types["Source__unqualified"] = "half*"

    assert (
        pp._infer_argument_type("Source__addressed(src)", {}, {})
        == "const device half*"
    )
    assert pp._infer_argument_type("Source__unqualified(src)", {}, {}) is None


@pytest.mark.parametrize(
    "expression",
    [
        "&(src[index++])",
        "&(src[next(index)])",
        "&(src[index = 0])",
        "&(src[index, 0])",
        "&(src[index ? 1 : 0])",
        "&(src[index][0])",
        "&(src[index] + 1)",
        "&(missing[index])",
    ],
)
def test_infer_argument_type_rejects_unsupported_addressed_elements(expression):
    pp = MetalPreprocessor()
    code = "kernel void k(const device float* src [[buffer(0)]]) {}"
    positioned = pp._collect_buffer_element_types(code, [])
    buffers = pp._flatten_types_at(positioned, len(code))

    assert pp._infer_argument_type(expression, buffers, {"index": "uint"}) is None


def test_infer_argument_type_rejects_address_without_pointer_metadata():
    # An element-only legacy view cannot prove the source address space.
    assert (
        MetalPreprocessor()._infer_argument_type(
            "&(src[index])", {"src": "float"}, {"index": "uint"}
        )
        is None
    )


def test_infer_argument_type_simd_group_builtin_returns_first_arg_type():
    # A SIMD/quad group built-in that moves/combines lane values returns the type
    # of its first argument, so scan's
    # `operator()(val, simd_shuffle_and_fill_up(val, init, i))` can be typed.
    pp = MetalPreprocessor()
    locals_ = {"val": "float", "x": "int32_t"}
    assert (
        pp._infer_argument_type("simd_shuffle_and_fill_up(val, init, i)", {}, locals_)
        == "float"
    )
    assert (
        pp._infer_argument_type("simd_prefix_inclusive_sum(x)", {}, locals_)
        == "int32_t"
    )
    assert pp._infer_argument_type("simd_sum(val)", {}, locals_) == "float"
    # An unrecognized function name is not inferred (never guesses).
    assert pp._infer_argument_type("some_helper(val)", {}, locals_) is None


def test_preprocessor_instantiates_template_method_from_member_access():
    # A bare member access `obj.member` (non-subscript) resolves to the field's
    # declared type.
    code = """
    struct Pt { float x; };
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device float* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        Pt p;
        o[i] = op.reduce(p.x);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Sum__reduce__float(thread Sum& self, float val)" in output
    assert "Sum__reduce__float(op, p.x)" in output


def test_preprocessor_instantiates_template_method_from_pointer_member_access():
    code = """
    struct Params { const int stride; device float* data; };
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        constant const Params* params [[buffer(0)]],
        device int* out [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        out[i] = op.reduce(params->stride);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "int Sum__reduce__int(thread Sum& self, int val)" in output
    assert "Sum__reduce__int(op, params->stride)" in output


def test_preprocessor_instantiates_template_method_from_union_local():
    # A union-typed local (`bool4_or_uint update;`) is recognized by the local
    # scanner, so a bare such local is an inferable call argument.
    code = """
    union bool4_or_uint { bool4 b; unsigned int i; };
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device uint* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        bool4_or_uint update;
        o[i] = op.reduce(update).i;
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert (
        "bool4_or_uint Sum__reduce__bool4_or_uint(thread Sum& self, "
        "bool4_or_uint val)" in output
    )
    assert "Sum__reduce__bool4_or_uint(op, update)" in output


def test_preprocessor_instantiates_template_method_from_struct_local():
    # A struct-typed local of a method-less carrier is an inferable bare argument.
    code = """
    struct Pair { float a; float b; };
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    kernel void k(
        device float* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        Sum op;
        Pair p;
        o[i] = op.reduce(p).a;
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "Pair Sum__reduce__Pair(thread Sum& self, Pair val)" in output
    assert "Sum__reduce__Pair(op, p)" in output


def test_preprocessor_instantiates_template_method_from_function_pointer_parameter():
    # A call argument that subscripts a POINTER PARAMETER of the enclosing
    # function (`device T* p` -> `p[i]` is T) is inferable.
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    float helper(device float* p, uint i) {
        Sum op;
        return op.reduce(p[i]);
    }

    kernel void k(
        device float* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        o[i] = helper(o, i);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "float Sum__reduce__float(thread Sum& self, float val)" in output
    assert "Sum__reduce__float(op, p[i])" in output


def test_preprocessor_instantiates_template_method_from_function_scalar_parameter():
    # A call argument that is a bare SCALAR PARAMETER of the enclosing function is
    # inferable (the local-declaration scanner alone misses comma-terminated
    # parameters).
    code = """
    struct Sum { template <typename T> T reduce(T val){ return val; } };

    half helper(half v) {
        Sum op;
        return op.reduce(v);
    }

    kernel void k(
        device half* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        o[i] = helper(o[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    assert "half Sum__reduce__half(thread Sum& self, half val)" in output
    assert "Sum__reduce__half(op, v)" in output


def test_preprocessor_template_member_lowering_is_deterministic_across_kernels():
    # Regression for a PYTHONHASHSEED-dependent ordering bug: two kernels in one
    # file that each declare a variable named `op` of a DIFFERENT concrete functor
    # type must resolve `op` per-kernel (nearest preceding declaration) instead of
    # collapsing to one set-iteration-order winner. The lowered output is the same
    # regardless of declaration discovery order.
    code = """
    struct SumF { template <typename T> T reduce(T val){ return val + val; } };
    struct MaxF { template <typename T> T reduce(T val){ return val; } };

    kernel void k_sum(
        device float* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        SumF op;
        float totals[4];
        o[i] = op.reduce(totals[i]);
    }

    kernel void k_max(
        device int* o [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
        MaxF op;
        int totals[4];
        o[i] = op.reduce(totals[i]);
    }
    """
    output = MetalPreprocessor().preprocess(code)
    # Each kernel binds `op` to its OWN struct type and element type.
    assert "float SumF__reduce__float(thread SumF& self, float val)" in output
    assert "int MaxF__reduce__int(thread MaxF& self, int val)" in output
    assert "SumF__reduce__float(op, totals[i])" in output
    assert "MaxF__reduce__int(op, totals[i])" in output
    # No cross-kernel mis-binding.
    assert "MaxF__reduce__float" not in output
    assert "SumF__reduce__int" not in output


def test_infer_argument_type_local_array_and_member_subscript():
    # Direct unit coverage of the conservative inference contract for the new
    # shapes, using flat (position-resolved) views like the call site builds.
    pp = MetalPreprocessor()
    buffers = {"totals": "float", "buf": "uint"}
    locals_ = {"acc": "half", "update": "bool4_or_uint"}
    fields = {"init": {"data": "float", "scale": "half"}}
    # Local array / buffer subscript -> element type.
    assert pp._infer_argument_type("totals[i]", buffers, locals_, fields) == "float"
    assert pp._infer_argument_type("buf[i + 1]", buffers, locals_, fields) == "uint"
    # Member-access subscript -> struct field element type.
    assert (
        pp._infer_argument_type("init.data[elem]", buffers, locals_, fields) == "float"
    )
    # Bare member access -> field type.
    assert pp._infer_argument_type("init.scale", buffers, locals_, fields) == "half"
    pointer_fields = {"params->": {"stride": "int", "data": "float*"}}
    assert (
        pp._infer_argument_type("params->stride", buffers, locals_, pointer_fields)
        == "int"
    )
    assert (
        pp._infer_argument_type("params->data[i]", buffers, locals_, pointer_fields)
        == "float"
    )
    assert (
        pp._infer_argument_type("params->data", buffers, locals_, pointer_fields)
        is None
    )
    assert (
        pp._infer_argument_type("params.stride", buffers, locals_, pointer_fields)
        is None
    )
    assert pp._infer_argument_type("init->scale", buffers, locals_, fields) is None
    # Union / struct local -> its declared type.
    assert (
        pp._infer_argument_type("update", buffers, locals_, fields) == "bool4_or_uint"
    )
    # Unknown shapes stay un-inferable (never guess).
    assert pp._infer_argument_type("missing[i]", buffers, locals_, fields) is None
    assert pp._infer_argument_type("init.unknown", buffers, locals_, fields) is None
    assert pp._infer_argument_type("foo()", buffers, locals_, fields) is None


def test_infer_argument_type_builtin_vector_swizzle():
    pp = MetalPreprocessor()
    locals_ = {"dims": "short2", "color": "const float4"}

    assert pp._infer_argument_type("dims.y", {}, locals_) == "short"
    assert pp._infer_argument_type("color.rgb", {}, locals_) == "float3"
    assert pp._infer_argument_type("dims.z", {}, locals_) is None
    assert pp._infer_argument_type("dims.xr", {}, locals_) is None
    assert pp._infer_argument_type("dims->y", {}, locals_) is None


def test_preprocessor_instantiates_template_method_from_vector_component_parameter():
    code = """
    struct BaseFrag {
      template <typename LimX, typename LimY>
      static void load_safe(LimX lim_x, LimY lim_y) {}
    };

    struct Tile {
      using Frag = BaseFrag;

      template <typename U>
      void load_safe(const device U* src, const short2 dims) {
        Frag::load_safe(dims.y, dims.x);
      }
    };

    kernel void k(
        const device float* src [[buffer(0)]],
        short2 dims [[thread_position_in_grid]]) {
      Tile tile;
      tile.load_safe(src, dims);
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "void BaseFrag__load_safe__short_short(short lim_x, short lim_y)" in output
    assert "BaseFrag__load_safe__short_short(dims.y, dims.x)" in output


def test_const_for_loop_parameter_substitution_ignores_member_identifiers():
    output = MetalPreprocessor()._substitute_const_for_loop_parameter(
        "state.idx = idx.value; ptr->idx = Type::idx;", "idx", 3
    )

    assert output == "state.idx = 3; ptr->idx = Type::idx;"


def test_infer_argument_type_arithmetic_construction_and_functor_call():
    # Direct coverage of the arithmetic / construction / functor-call inference
    # rules, using struct metadata parsed from a small snippet exactly as the
    # call site builds it. These are the shapes MLX's complex unary ops feed to
    # template member methods (e.g. `x + i * Sqrt{}(1.0 - x * x)`).
    pp = MetalPreprocessor()
    snippet = """
    struct complex64_t { float real; float imag; };
    struct Sqrt {
        template <typename T> T operator()(T x) { return x; }
        complex64_t operator()(complex64_t x) { return {x.real, x.imag}; }
    };
    struct Square {
        template <typename T> T operator()(T x) { return x * x; }
    };
    struct Real {
        float operator()(complex64_t x) { return x.real; }
    };
    """
    structs = {
        struct.name: struct for struct in pp._find_concrete_struct_definitions(snippet)
    }
    buffers = {"buf": "complex64_t"}
    locals_ = {"x": "complex64_t", "i": "complex64_t", "f": "float", "n": "int"}

    def infer(expr):
        return pp._infer_argument_type(expr, buffers, locals_, {}, structs)

    # Binary arithmetic between identical operand types keeps that type.
    assert infer("x + i") == "complex64_t"
    assert infer("x * i - x") == "complex64_t"
    assert infer("n + n") == "int"
    # An AGGREGATE concrete operand absorbs a numeric literal (its arithmetic
    # operators return the aggregate).
    assert infer("1.0 - x * x") == "complex64_t"
    assert infer("x / 2.0") == "complex64_t"
    assert infer("buf[i] + 1.0") == "complex64_t"
    # A scalar keeps its type when the literal does not out-rank it.
    assert infer("f + 1") == "float"
    # ...but a floating literal meeting an INTEGER scalar promotes to an unknown
    # floating width, so we bail rather than mis-claim `int`.
    assert infer("n + 1.0") is None
    # Two DIFFERENT concrete non-literal types are ambiguous -> never guess.
    assert infer("x + f") is None
    # Brace construction of a recognized type.
    assert infer("complex64_t{0.0, 1.0}") == "complex64_t"
    assert infer("IndexTag<1>{}") == "IndexTag<1>"
    # Functor construction-and-call: concrete overload match, template identity,
    # and an argument-independent (fixed) return type.
    assert infer("Sqrt{}(x)") == "complex64_t"
    assert infer("Sqrt{}(1.0 - x * x)") == "complex64_t"
    assert infer("Square{}(f)") == "float"
    assert infer("Real{}(x)") == "float"
    # The full composed argument from MLX's complex arccos infers conservatively.
    assert infer("x + i * Sqrt{}(1.0 - x * x)") == "complex64_t"
    # Still never guesses for un-inferable shapes (missing struct / operand).
    assert infer("Mystery{}(x)") is None
    assert infer("x + mystery") is None
    # Without struct metadata, construction / functor-call shapes stay un-inferable.
    assert (
        pp._infer_argument_type("complex64_t{0.0, 1.0}", buffers, locals_, {}) is None
    )
    assert pp._infer_argument_type("Sqrt{}(x)", buffers, locals_, {}) is None


def test_collect_buffer_element_types_rejects_non_declaration_subscripts():
    # The array-declaration scanner must NEVER record a bogus element type from a
    # statement that merely matches the `<token> name[` shape — e.g.
    # `return totals[i]` (leading token `return`) or a control-flow subscript.
    # Only a recognized scalar/vector or struct/union element type is accepted.
    pp = MetalPreprocessor()
    code = (
        "struct Pair { float a; };\n"
        "kernel void k(device float* o [[buffer(0)]],"
        " uint i [[thread_position_in_grid]]) {\n"
        "    float totals[4];\n"
        "    Pair items[4];\n"
        "    return totals[i];\n"
        "    if (i) block[i];\n"
        "}\n"
    )
    element_types = pp._collect_buffer_element_types(code, [])
    # totals is a real `float totals[4]` array; the only recorded type is float.
    assert [t.element_type for _, t in element_types.get("totals", [])] == ["float"]
    assert [t.pointer_type for _, t in element_types.get("totals", [])] == [
        "thread float*"
    ]
    # A struct array records its element struct type.
    assert [t.element_type for _, t in element_types.get("items", [])] == ["Pair"]
    assert [t.pointer_type for _, t in element_types.get("items", [])] == [
        "thread Pair*"
    ]
    # `return`/`if` tokens never become element types.
    assert "block" not in element_types
    assert all(
        t.element_type not in {"return", "if", "else", "for", "while"}
        for entries in element_types.values()
        for _, t in entries
    )


def test_collect_buffer_element_types_preserves_leading_const_pointer_qualifiers():
    pp = MetalPreprocessor()
    code = """
    void read(
        const device half* device_src,
        const threadgroup float* group_src) {}
    """

    element_types = pp._collect_buffer_element_types(code, [])

    assert [entry.pointer_type for _, entry in element_types.get("device_src", [])] == [
        "const device half*"
    ]
    assert [entry.pointer_type for _, entry in element_types.get("group_src", [])] == [
        "const threadgroup float*"
    ]


# --------------------------------------------------------------------------- #
# SFINAE-overloaded template member methods (MLX `simd_reduce`, issue #1354).  #
# --------------------------------------------------------------------------- #

# A synthetic struct mirroring MLX's DEFINE_SIMD_REDUCE() macro expansion plus a
# return-type-SFINAE `simd_reduce_impl` (Min/Max style), used across the tests
# below. It exercises BOTH SFINAE layers: the non-type constraint on
# `simd_reduce` (sizeof) and the return-type constraint on `simd_reduce_impl`
# (is_integral_v).
_SFINAE_SIMD_REDUCE_STRUCT = (
    "static constant constexpr const uint8_t simd_size = 32;\n"
    "template <typename U>\n"
    "struct Red {\n"
    "  template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true>\n"
    "  T simd_reduce(T val) { return simd_reduce_impl(val); }\n"
    "  template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true>\n"
    "  T simd_reduce(T val) {\n"
    "    for (short i = simd_size / 2; i > 0; i /= 2) {\n"
    "      val = operator()(val, simd_shuffle_down(val, i));\n"
    "    }\n"
    "    return val;\n"
    "  }\n"
    "  template <typename T>\n"
    "  metal::enable_if_t<metal::is_integral_v<T>, T> simd_reduce_impl(T val) {\n"
    "    return simd_sum(val);\n"
    "  }\n"
    "  template <typename T>\n"
    "  metal::enable_if_t<!metal::is_integral_v<T>, T> simd_reduce_impl(T val) {\n"
    "    return simd_sum(val);\n"
    "  }\n"
    "  U operator()(U a, U b) { return a + b; }\n"
    "};\n"
)


def _sfinae_reduce_source(elem):
    return (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        + _SFINAE_SIMD_REDUCE_STRUCT
        + f"kernel void reduce(device const {elem}* in [[buffer(0)]],\n"
        f"  device {elem}* out [[buffer(1)]],\n"
        "  uint i [[thread_position_in_grid]]) {\n"
        f"    Red<{elem}> op;\n"
        f"    {elem} total = in[i];\n"
        "    out[i] = op.simd_reduce(total);\n"
        "}\n"
    )


def test_sfinae_template_param_list_parses_constraint_separately():
    # The `<` disambiguation: a SFINAE non-type parameter whose constraint embeds
    # a comparison (`sizeof(T) < 8`) is parsed as ONE parameter; the bindable type
    # parameter list is just `T` and the constraint text is recorded apart.
    pp = MetalPreprocessor()
    text = "typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true"
    assert pp._template_parameter_names(text) == ["T"]
    assert pp._template_parameter_constraints(text) == [
        "metal::enable_if_t<sizeof(T) < 8, bool>"
    ]
    # The matching angle for the parameter list is found despite the comparison.
    header = "<" + text + ">"
    assert pp._find_matching_template_param_angle(header, 0) == len(header) - 1
    # The `== 8` variant likewise yields a single bindable `T`, not a spurious one.
    text_eq = "typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true"
    assert pp._template_parameter_names(text_eq) == ["T"]
    # Existing parameter shapes are unchanged.
    assert pp._template_parameter_names("typename U = bool") == ["U"]
    assert pp._template_parameter_defaults("typename U = bool") == {"U": "bool"}
    assert pp._template_parameter_names("typename T, int N") == ["T", "N"]
    assert pp._variadic_template_parameter_names("typename... Args") == {"Args"}


def test_sfinae_simd_reduce_recognized_as_template_methods():
    # Both `simd_reduce` overloads and both `simd_reduce_impl` overloads are
    # recognized as template member methods (bindable `T`), with their SFINAE
    # constraints captured and the return-type SFINAE unwrapped to the value type.
    pp = MetalPreprocessor()
    body = _SFINAE_SIMD_REDUCE_STRUCT.split("struct Red {", 1)[1].rsplit("};", 1)[0]
    (
        _names,
        _types,
        concrete,
        templates,
        _members,
        _ctors,
        _layout_supported,
    ) = pp._split_struct_body("Red", body, 0)
    by_name = {}
    for method in templates:
        by_name.setdefault(method.name, []).append(method)
    assert len(by_name.get("simd_reduce", [])) == 2
    assert len(by_name.get("simd_reduce_impl", [])) == 2
    for method in templates:
        assert method.template_parameters == ["T"]
    reduce_constraints = {
        c for m in by_name["simd_reduce"] for c in m.template_constraints
    }
    assert reduce_constraints == {
        "metal::enable_if_t<sizeof(T) < 8, bool>",
        "metal::enable_if_t<sizeof(T) == 8, bool>",
    }
    for method in by_name["simd_reduce_impl"]:
        # Return-type SFINAE unwrapped to plain `T`; constraint recorded.
        assert method.return_type == "T"
        assert method.return_type_constraint in {
            "metal::is_integral_v<T>",
            "!metal::is_integral_v<T>",
        }


def test_sfinae_constraint_evaluation_size_and_integral_tables():
    # Direct coverage of the size / is_integral tables and constraint evaluation.
    pp = MetalPreprocessor()
    assert pp._sizeof_concrete_type("float") == 4
    assert pp._sizeof_concrete_type("int") == 4
    assert pp._sizeof_concrete_type("long") == 8
    assert pp._sizeof_concrete_type("double") == 8
    assert pp._sizeof_concrete_type("half") == 2
    assert pp._sizeof_concrete_type("float4") == 16
    assert pp._is_integral_concrete_type("int") is True
    assert pp._is_integral_concrete_type("uint") is True
    assert pp._is_integral_concrete_type("float") is False
    assert pp._is_integral_concrete_type("half") is False
    less8 = "metal::enable_if_t<sizeof(T) < 8, bool>"
    eq8 = "metal::enable_if_t<sizeof(T) == 8, bool>"
    assert pp._evaluate_template_constraint(less8, {"T": "float"}) is True
    assert pp._evaluate_template_constraint(less8, {"T": "long"}) is False
    assert pp._evaluate_template_constraint(eq8, {"T": "double"}) is True
    assert pp._evaluate_template_constraint(eq8, {"T": "int"}) is False
    assert (
        pp._evaluate_template_constraint("metal::is_integral_v<T>", {"T": "int"})
        is True
    )
    assert (
        pp._evaluate_template_constraint("!metal::is_integral_v<T>", {"T": "float"})
        is True
    )
    # An unrecognized constraint / type clean-fails (raises, never guesses).
    with pytest.raises(MetalPreprocessor._UnrecognizedConstraint):
        pp._evaluate_template_constraint(
            "metal::is_floating_point_v<T>", {"T": "float"}
        )
    with pytest.raises(MetalPreprocessor._UnrecognizedConstraint):
        pp._evaluate_template_constraint(less8, {"T": "complex64_t"})


@pytest.mark.parametrize(
    "elem,impl_suffix",
    [
        ("float", "float"),  # sizeof 4 -> <8 ; non-integral simd_reduce_impl
        ("int", "int"),  # sizeof 4 -> <8 ; integral simd_reduce_impl
    ],
)
def test_sfinae_simd_reduce_selects_less8_overload_and_lowers_chain(elem, impl_suffix):
    # `T` of size 4 selects the sizeof<8 overload, whose body calls
    # `simd_reduce_impl` — the second SFINAE layer is resolved too, so the whole
    # chain lowers to free functions with no dangling call.
    output = MetalPreprocessor().preprocess(_sfinae_reduce_source(elem))
    reduce_fn = (
        f"{elem} Red_{elem}__simd_reduce__{elem}"
        f"(thread Red_{elem}& self, {elem} val)"
    )
    assert reduce_fn in output
    assert f"Red_{elem}__simd_reduce_impl__{impl_suffix}" in output
    assert f"Red_{elem}__simd_reduce__{elem}(op, total)" in output
    # No dangling receiver call survives.
    assert "op.simd_reduce(" not in output
    # The lowered `simd_reduce` free function calls the LOWERED impl, not a bare
    # `simd_reduce_impl(...)`. (A bare `simd_reduce_impl(` legitimately remains in
    # the leftover PRIMARY template body, which the downstream parser drops.)
    reduce_body = output.split(reduce_fn, 1)[1].split("}", 1)[0]
    assert "self" in reduce_body  # receiver threaded into the internal call
    assert re.search(r"(?<![\w_])simd_reduce_impl\s*\(", reduce_body) is None
    # The only call sites of the bare impl name are inside `template <` blocks.
    for match in re.finditer(r"(?<![\w_])simd_reduce_impl\s*\(", output):
        preceding = output[: match.start()]
        assert "template <" in preceding.rsplit("struct", 1)[-1]


@pytest.mark.parametrize("elem", ["long", "double"])
def test_sfinae_simd_reduce_selects_eq8_overload(elem):
    # `T` of size 8 selects the sizeof==8 overload, whose body combines elements
    # with `operator()` (lowered to the concrete call operator) and never calls
    # `simd_reduce_impl`.
    output = MetalPreprocessor().preprocess(_sfinae_reduce_source(elem))
    reduce_fn = (
        f"{elem} Red_{elem}__simd_reduce__{elem}"
        f"(thread Red_{elem}& self, {elem} val)"
    )
    assert reduce_fn in output
    assert f"Red_{elem}__simd_reduce__{elem}(op, total)" in output
    # The ==8 body's `operator()(...)` is lowered to the concrete free function.
    assert f"Red_{elem}__operator_call(self," in output
    # The ==8 overload does not call simd_reduce_impl, so the lowered free
    # function body contains no impl call and NO lowered impl is emitted.
    reduce_body = output.split(reduce_fn, 1)[1].split("}", 1)[0]
    assert "simd_reduce_impl" not in reduce_body
    assert f"Red_{elem}__simd_reduce_impl" not in output
    assert "op.simd_reduce(" not in output


def test_sfinae_simd_reduce_unrecognized_type_clean_fails():
    # A call whose concrete `T` is a non-scalar type (`complex64_t`) has no
    # recognized sizeof, so no overload's constraint can be evaluated: the method
    # clean-fails with MetalStructMethodError rather than guessing an overload.
    code = (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct complex64_t { float real; float imag; };\n"
        + _SFINAE_SIMD_REDUCE_STRUCT
        + "kernel void reduce(device const complex64_t* in [[buffer(0)]],\n"
        "  device complex64_t* out [[buffer(1)]],\n"
        "  uint i [[thread_position_in_grid]]) {\n"
        "    Red<complex64_t> op;\n"
        "    complex64_t total = in[i];\n"
        "    out[i] = op.simd_reduce(total);\n"
        "}\n"
    )
    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(code)
    assert excinfo.value.method_name == "simd_reduce"
    assert (
        excinfo.value.project_diagnostic_code == "project.translate.metal-struct-method"
    )
    call_text = "simd_reduce(total)"
    call_offset = code.index(call_text)
    call_line_start = code.rfind("\n", 0, call_offset)
    location = excinfo.value.source_location
    assert location["line"] == code.count("\n", 0, call_offset) + 1
    assert location["column"] == call_offset - call_line_start
    assert location["length"] == len(call_text)
    assert location["endLine"] == location["line"]
    assert location["endColumn"] == location["column"] + len(call_text)


def test_sfinae_simd_reduce_full_pipeline_to_hlsl():
    # End-to-end: the sizeof<8 + is_integral chain for T=float translates to HLSL
    # with `simd_sum` -> `WaveActiveSum` and no dangling simd_reduce calls.
    from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
    from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
    from crosstl.translator.lexer import Lexer as CrossGLLexer
    from crosstl.translator.parser import Parser as CrossGLParser

    pre = MetalPreprocessor().preprocess(_sfinae_reduce_source("float"))
    assert "Red_float__simd_reduce__float(thread Red_float& self, float val)" in pre
    tokens = MetalLexer(pre).tokenize()
    ast = MetalParser(tokens).parse()
    crossgl = MetalToCrossGLConverter().generate(ast)
    parsed = CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()
    hlsl = HLSLCodeGen().generate(parsed)
    assert "op.simd_reduce(" not in hlsl
    assert re.search(r"(?<![\w_])simd_reduce\s*\(", hlsl) is None
    assert "WaveActiveSum" in hlsl


def test_preprocessor_elides_stateless_compile_time_global_calls():
    source = """
    struct Logger {
      constexpr Logger(constant char*, constant char*) constant {}

      template <typename... Args>
      void log_debug(constant char*, Args...) const {
        trap();
      }

      template <typename... Args>
      void log_debug(constant char*, Args...) const constant {}
    };

    constant Logger logger("subsystem", "kernel");

    int bump(device int* output) {
      output[0] += 1;
      return output[0];
    }

    kernel void compute(device int* output [[buffer(0)]]) {
      logger.log_debug("value=%d", bump(output));
      logger.log_debug("plain=%d", output[0]);
    }
    """

    output = MetalPreprocessor().preprocess(source)

    assert "constant Logger logger" not in output
    assert "logger.log_debug" not in output
    assert "bump(output);" in output
    assert output.count("bump(output)") == 1
    assert len(re.findall(r"(?m)^\s*output\[0\];$", output)) == 1
    assert "trap();" not in output


def test_preprocessor_elided_call_evaluates_multiple_arguments_once():
    source = """
    struct Logger {
      constexpr Logger() constant {}

      template <typename... Args>
      void log_debug(constant char*, Args...) const constant {}
    };

    constant Logger logger;

    int bump(device int* output) {
      output[0] += 1;
      return output[0];
    }

    kernel void compute(device int* output [[buffer(0)]]) {
      if (output[0] > 0)
        logger.log_debug("values=%d,%d", bump(output), bump(output));
    }
    """

    output = MetalPreprocessor().preprocess(source)

    assert "logger" not in output
    assert output.count("bump(output);") == 2
    assert re.search(
        r"if\s*\(output\[0\]\s*>\s*0\)\s*\{\s*"
        r"bump\(output\);\s*bump\(output\);\s*\}",
        output,
    )


def test_preprocessor_does_not_elide_function_declaration_as_global():
    source = """
    struct Logger {
      constexpr Logger(constant char*) constant {}
    };

    constant Logger make_logger(constant char*);
    """

    output = MetalPreprocessor().preprocess(source)

    assert "constant Logger make_logger(constant char*);" in output


def test_preprocessor_elides_calls_between_extern_declaration_and_definition():
    source = """
    struct Logger {
      constexpr Logger() constant {}
      void log_debug(constant char*) const constant {}
    };

    extern constant Logger logger;

    kernel void compute(device int* output [[buffer(0)]]) {
      logger.log_debug("message");
      output[0] = 1;
    }

    constant Logger logger{};
    """

    output = MetalPreprocessor().preprocess(source)

    assert "extern constant Logger logger" not in output
    assert "constant Logger logger" not in output
    assert "logger.log_debug" not in output
    assert "output[0] = 1;" in output


@pytest.mark.parametrize("member", ["int state;", "uint state : 1;"])
def test_preprocessor_preserves_stateful_compile_time_globals(member):
    source = f"""
    struct Logger {{
      {member}
      constexpr Logger() constant {{}}
    }};

    constant Logger logger;
    kernel void compute() {{}}
    """

    output = MetalPreprocessor().preprocess(source)

    assert "constant Logger logger;" in output
    assert member in output


@pytest.mark.parametrize(
    ("source", "reason", "effect"),
    [
        (
            """
            struct Logger {
              constexpr Logger() constant { trap(); }
            };
            constant Logger logger;
            kernel void compute() {}
            """,
            "constructor-has-effects",
            "trap-or-termination",
        ),
        (
            """
            struct Logger {
              constexpr Logger() constant {}
              void log_debug(constant char*) const constant;
            };
            constant Logger logger;
            kernel void compute() { logger.log_debug("message"); }
            """,
            "method-body-unresolved",
            None,
        ),
        (
            """
            struct Logger {
              constexpr Logger() constant {}
            };
            constant Logger logger;
            kernel void compute() { constant Logger* observed = &logger; }
            """,
            "identity-observed",
            "object-identity",
        ),
        (
            """
            struct Logger {
              constexpr Logger() constant {}
              void log_debug(constant char*) const constant { printf("real"); }
            };
            constant Logger logger;
            kernel void compute() { logger.log_debug("message"); }
            """,
            "method-has-effects",
            "function-call",
        ),
        (
            """
            struct Logger {
              constexpr Logger() constant {}
              ~Logger() { trap(); }
            };
            constant Logger logger;
            kernel void compute() {}
            """,
            "destructor-has-effects",
            "trap-or-termination",
        ),
        (
            """
            struct Logger {
              constexpr Logger() constant {}
              int value() const constant { return 1; }
            };
            constant Logger logger;
            kernel void compute() { logger.value(); }
            """,
            "method-returns-value",
            "returned-value",
        ),
        (
            """
            constant char* category();
            struct Logger {
              constexpr Logger(constant char*) constant {}
            };
            constant Logger logger(category());
            kernel void compute() {}
            """,
            "initializer-effect-unproven",
            "initializer-evaluation",
        ),
    ],
)
def test_preprocessor_stateless_global_proof_fails_closed(source, reason, effect):
    with pytest.raises(MetalStatelessGlobalElisionError) as exc_info:
        MetalPreprocessor().preprocess(source)

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-stateless-global-unsafe"
    )
    assert diagnostic.missing_capabilities == ("metal.stateless-global-elision",)
    assert diagnostic.global_name == "logger"
    assert diagnostic.type_name == "Logger"
    assert diagnostic.reason == reason
    assert diagnostic.effect == effect
    assert diagnostic.source_location["line"] > 0
    assert diagnostic.source_location["column"] > 0
    assert diagnostic.source_location["length"] > 0
