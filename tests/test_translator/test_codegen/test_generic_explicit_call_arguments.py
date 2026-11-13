"""CrossGL calls with explicit generic arguments (``helper<int>(x)``) bind leading parameters, deduce the rest.

The parser keeps an identifier call's explicit arguments in the callee name. Each generator used to miss the generic
definition for that name, drop the definition and emit the call verbatim, which names no function in the target.
"""

import re

import pytest

import crosstl
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused
from crosstl.translator.semantics.declared_identifiers import UndeclaredIdentifierError

C_STYLE_TARGETS = ["directx", "opengl", "metal", "vulkan"]

PROGRAM = """
shader main {
    generic<T> T zero_of() {
        return T(0);
    }

    generic<T> T plus_zero(T value) {
        return zero_of<T>() + value;
    }

    generic<T, U = float> T scale_by(U factor, T value) {
        return value * T(factor);
    }

    generic<T, U = float> T widen(T value) {
        return T(U(value));
    }

    generic<T, U> T convert_to(U value) {
        return T(value);
    }

    compute {
        @ stage_entry
        void k(RWStructuredBuffer<float> values @buffer(0), RWStructuredBuffer<int> ints @buffer(1),
               uint gid @gl_GlobalInvocationID) {
            buffer_store(values, gid, zero_of<float>() + plus_zero(buffer_load(values, gid)));
            buffer_store(ints, gid, scale_by<int>(float16(2), 3) + widen<int>(3));
            buffer_store(ints, gid, convert_to<int>(buffer_load(values, gid)));
        }
    }
}
"""


def translate(tmp_path, source, backend):
    path = tmp_path / "program.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_explicit_generic_arguments_select_specializations(tmp_path, backend):
    code = translate(tmp_path, PROGRAM, backend)
    for specialization in (
        "zero_of_float",
        "plus_zero_float",
        # U is deduced from the float16 argument; its float default does not apply.
        "scale_by_int_float16",
        # U is not deducible, so it takes its default.
        "widen_int_float",
        "convert_to_int_float",
    ):
        assert specialization in code
    for verbatim in ("zero_of<", "scale_by<", "widen<", "convert_to<"):
        assert verbatim not in code
    # The body of plus_zero<float> calls zero_of<float>, not an instantiation for the parameter T itself.
    assert "zero_of_T" not in code


def test_explicit_generic_arguments_lower_to_spirv_calls(tmp_path):
    module = translate(tmp_path, PROGRAM, "vulkan")
    # zero_of<float>, plus_zero<float> (which calls zero_of<float>), scale_by, widen and convert_to.
    assert module.count("OpFunctionCall") == 6


def test_deduced_parameter_keeps_the_argument_type_in_hlsl(tmp_path):
    code = translate(tmp_path, PROGRAM, "directx")
    assert "int scale_by_int_float16(float16_t factor, int value)" in code
    assert "int widen_int_float(int value)" in code


VARIABLE_ARGUMENT_OVERLOADS = """
shader main {
    buffer float* results;
    void inc<T>(inout T value) { value = value + T(1.0); }
    void inc<T>(inout T value, T step) { value = value + step; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        vec2 w = vec2(1.0);
        inc(w, w);
        results[0] = w.x;
    }
    }
}
"""


@pytest.mark.parametrize("backend", ["opengl", "metal", "cuda", "hip"])
def test_variable_argument_is_not_read_as_a_type_argument(tmp_path, backend):
    # inc(w, w) matches the two-parameter overload; the one-parameter overload must not be specialized for ``w``.
    code = translate(tmp_path, VARIABLE_ARGUMENT_OVERLOADS, backend)
    assert "_overload1(w, w)" in code
    assert "inc_w" not in code


EXPRESSION_ARGUMENT_OVERLOADS = """
shader main {
    buffer float* results;
    struct Q { vec2 v; };
    T sel<T>(T a) { return a; }
    T sel<T>(T a, T b) { return b; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        Q q;
        q.v = vec2(results[0]);
        float x = results[1];
        vec2 r = sel(q.v, q.v) + sel(vec2(1.0), q.v);
        results[0] = r.x + sel(2.0, x) + sel(x + 1.0, x);
    }
    }
}
"""


@pytest.mark.parametrize(
    "backend", ["directx", "opengl", "metal", "cuda", "hip", "slang"]
)
def test_expression_argument_is_not_read_as_a_type_argument(tmp_path, backend):
    # sel(q.v, q.v), sel(vec2(1.0), q.v), sel(2.0, x) and sel(x + 1.0, x) all match sel(T a, T b); none of the
    # leading arguments is a type, so no specialization is named after one.
    code = translate(tmp_path, EXPRESSION_ARGUMENT_OVERLOADS, backend)
    specialized = set(re.findall(r"\bsel_(\w+?)_overload[01]\b", code))
    assert specialized and specialized <= {"float", "vec2", "float2"}


def test_expression_argument_overloads_lower_to_spirv(tmp_path):
    module = translate(tmp_path, EXPRESSION_ARGUMENT_OVERLOADS, "vulkan")
    assert "OpFunctionCall" in module


UNRESOLVABLE = """
shader main {
    generic<T, U> T make_value(T seed) {
        return seed + T(U(0));
    }

    compute {
        @ stage_entry
        void k(RWStructuredBuffer<int> ints @buffer(0), uint gid @gl_GlobalInvocationID) {
            buffer_store(ints, gid, CALL);
        }
    }
}
"""


@pytest.mark.parametrize(
    "call",
    [
        # U is neither supplied, deducible nor defaulted.
        "make_value<int>(1)",
        # More explicit arguments than the generic has parameters.
        "make_value<int, float, float>(1)",
    ],
)
@pytest.mark.parametrize("backend", [*C_STYLE_TARGETS, "cuda", "hip", "slang", "mojo"])
def test_unresolved_explicit_generic_call_is_refused(tmp_path, backend, call):
    with pytest.raises(Exception, match="make_value") as error:
        translate(tmp_path, UNRESOLVABLE.replace("CALL", call), backend)
    assert "generic" in str(error.value)


CONSTANT_ARGUMENT_OVERLOADS = """
shader main {
    buffer float* results;
    const float K = 2.0;
    T sel<T>(T a) { return a; }
    T sel<T>(T a, T b) { return b; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        float x = results[0];
        results[0] = sel(K, x);
    }
    }
}
"""

CBUFFER_ARGUMENT_OVERLOADS = """
shader main {
    buffer float* results;
    cbuffer Params {
        vec2 offset;
    };
    T sel<T>(T a) { return a; }
    T sel<T>(T a, T b) { return b; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        results[0] = sel(offset, offset).x;
    }
    }
}
"""

LOCAL_CONSTANT_NAME_OVERLOADS = """
shader main {
    buffer float* results;
    T sel<T>(T a) { return a; }
    T sel<T>(T a, T b) { return b; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        float K = 2.0;
        float x = results[0];
        results[0] = sel(K, x);
    }
    }
}
"""

ALL_BACKENDS = [
    "directx",
    "opengl",
    "metal",
    "cuda",
    "hip",
    "slang",
    "vulkan",
    "wgsl",
    "mojo",
]


def _assert_vulkan_sel_call(code, param_type, expected_operands):
    # translate() formats SPIR-V through spirv-dis when SPIRV-Tools are installed, which indents every line.
    code = "\n".join(line.strip() for line in code.splitlines())
    names = {
        sym_id: name
        for sym_id, name in re.findall(r'^OpName\s+(%\w+)\s+"([^"]+)"', code, re.M)
    }
    member_names = {
        (struct_id, int(index)): name
        for struct_id, index, name in re.findall(
            r'^OpMemberName\s+(%\w+)\s+(\d+)\s+"([^"]+)"', code, re.M
        )
    }
    defs = {
        sym_id: rhs for sym_id, rhs in re.findall(r"^(%\w+)\s*=\s*(.+)$", code, re.M)
    }
    rhs_to_id = {rhs: sym_id for sym_id, rhs in defs.items()}

    float_ty = rhs_to_id["OpTypeFloat 32"]
    target_ty = (
        rhs_to_id[f"OpTypeVector {float_ty} 2"] if param_type == "vec2" else float_ty
    )
    fn_ty = rhs_to_id[f"OpTypeFunction {target_ty} {target_ty} {target_ty}"]

    fn_match = re.search(
        rf"^(%\w+)\s*=\s*OpFunction\s+{re.escape(target_ty)}\s+None\s+{re.escape(fn_ty)}\n"
        rf"%\w+\s*=\s*OpFunctionParameter\s+{re.escape(target_ty)}\n"
        rf"%\w+\s*=\s*OpFunctionParameter\s+{re.escape(target_ty)}\b",
        code,
        re.M,
    )
    assert fn_match is not None
    assert code.count("OpFunctionParameter") == 2
    fn_id = fn_match.group(1)

    call_match = re.search(
        rf"^%\w+\s*=\s*OpFunctionCall\s+{re.escape(target_ty)}\s+{re.escape(fn_id)}\s+(%\w+)\s+(%\w+)\b",
        code,
        re.M,
    )
    assert call_match is not None

    for arg_id, expected in zip(call_match.groups(), expected_operands):
        kind, spec = expected.split(":", 1)
        if kind == "const":
            assert names.get(arg_id) == spec
            assert defs[arg_id].startswith(f"OpConstant {float_ty} ")
        elif kind == "local":
            load_match = re.match(
                rf"^OpLoad\s+{re.escape(float_ty)}\s+(%\w+)$", defs[arg_id]
            )
            assert load_match is not None
            assert names.get(load_match.group(1)) == spec
        elif kind == "member":
            struct_name, member_name = spec.split(".", 1)
            load_match = re.match(
                rf"^OpLoad\s+{re.escape(target_ty)}\s+(%\w+)$", defs[arg_id]
            )
            assert load_match is not None
            chain_match = re.match(
                r"^OpAccessChain\s+%\w+\s+(%\w+)\s+(%\w+)$",
                defs[load_match.group(1)],
            )
            assert chain_match is not None
            var_id, idx_id = chain_match.groups()
            assert names.get(var_id) == struct_name
            idx_val = int(defs[idx_id].split()[-1])
            assert any(
                struct_id == var_id or names.get(struct_id) == struct_name
                for (struct_id, idx), name in member_names.items()
                if idx == idx_val and name == member_name
            )
        elif kind == "var":
            load_match = re.match(r"^OpLoad\s+%\w+\s+(%\w+)$", defs[arg_id])
            assert load_match is not None
            assert names.get(load_match.group(1)) == spec
        else:
            raise AssertionError(f"Unknown expected operand spec: {expected}")


def _assert_sel_overload(
    code,
    backend,
    forbidden_prefix,
    expected_types,
    wgsl_type,
    call_arg_pattern=None,
    vulkan_param_type="float",
    vulkan_operands=None,
):
    assert forbidden_prefix not in code
    assert "overload0" not in code
    if backend == "vulkan":
        if vulkan_operands is not None:
            _assert_vulkan_sel_call(code, vulkan_param_type, vulkan_operands)
        else:
            assert code.count("OpFunctionParameter") == 2
            assert "OpFunctionCall" in code
    elif backend == "wgsl":
        suffix = rf"\({call_arg_pattern}\)" if call_arg_pattern is not None else r"\b"
        specialized = set(re.findall(rf"\bsel_(\w+){suffix}", code))
        assert specialized and all(wgsl_type in name for name in specialized)
    else:
        suffix = rf"\({call_arg_pattern}\)" if call_arg_pattern is not None else r"\b"
        specialized = set(re.findall(rf"\bsel_(\w+?)_overload1{suffix}", code))
        assert specialized and specialized <= expected_types


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_shader_constant_argument_is_not_read_as_type_argument(tmp_path, backend):
    code = translate(tmp_path, CONSTANT_ARGUMENT_OVERLOADS, backend)
    _assert_sel_overload(
        code,
        backend,
        "sel_K",
        {"float", "Float32"},
        "f32",
        r"K,\s*x",
        vulkan_param_type="float",
        vulkan_operands=("const:K", "local:x"),
    )


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_cbuffer_member_argument_is_not_read_as_type_argument(tmp_path, backend):
    code = translate(tmp_path, CBUFFER_ARGUMENT_OVERLOADS, backend)
    _assert_sel_overload(
        code,
        backend,
        "sel_offset",
        {"vec2", "float2", "vec2f", "SIMD"},
        "vec2",
        r"(?:(?:params|_Params)\.)?offset,\s*(?:(?:params|_Params)\.)?offset",
        vulkan_param_type="vec2",
        vulkan_operands=("member:Params.offset", "member:Params.offset"),
    )


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_local_value_named_like_constant_is_not_read_as_type_argument(
    tmp_path, backend
):
    code = translate(tmp_path, LOCAL_CONSTANT_NAME_OVERLOADS, backend)
    _assert_sel_overload(
        code,
        backend,
        "sel_K",
        {"float", "Float32"},
        "f32",
        r"K,\s*x",
        vulkan_param_type="float",
        vulkan_operands=("local:K", "local:x"),
    )


UNDECLARED_ARGUMENT_OVERLOADS = """
shader main {
    buffer float* results;
    T sel<T>(T a) { return a; }
    T sel<T>(T a, T b) { return b; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        float x = results[0];
        results[0] = sel(foo, x);
    }
    }
}
"""

UNDECLARED_INSIDE_GENERIC_OVERLOADS = """
shader main {
    buffer float* results;
    T sel<T>(T a) { return a; }
    T sel<T>(T a, T b) { return b; }
    T wrap<T>(T x) { return sel(foo, x); }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        float x = results[0];
        results[0] = wrap(x);
    }
    }
}
"""

BUILTIN_AND_ENUM_ARGUMENT_OVERLOADS = """
shader main {
    enum Mode { Fast = 1, Slow = 2 };
    buffer uint* results;
    T sel<T>(T a) { return a; }
    T sel<T>(T a, T b) { return b; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        uint3 u = uint3(1u, 2u, 3u);
        uint3 v = sel(gl_GlobalInvocationID, u);
        int m = sel(Mode::Fast, 0);
        results[0] = v.x + uint(m);
    }
    }
}
"""


@pytest.mark.parametrize("backend", [b for b in ALL_BACKENDS if b != "vulkan"])
def test_undeclared_identifier_argument_is_refused_in_generic_call(tmp_path, backend):
    with pytest.raises(UndeclaredIdentifierError, match="foo"):
        translate(tmp_path, UNDECLARED_ARGUMENT_OVERLOADS, backend)


def test_undeclared_identifier_argument_is_refused_in_generic_call_vulkan(tmp_path):
    # SPIR-V raises SpirvLoweringRefused (a TranslationDiagnostic) during codegen,
    # which generate_refusing_undeclared_identifiers preserves rather than replacing
    # with UndeclaredIdentifierError.
    with pytest.raises(SpirvLoweringRefused, match="foo"):
        translate(tmp_path, UNDECLARED_ARGUMENT_OVERLOADS, "vulkan")


@pytest.mark.parametrize("backend", [b for b in ALL_BACKENDS if b != "vulkan"])
def test_undeclared_identifier_does_not_match_enclosing_generic_substitution(
    tmp_path, backend
):
    with pytest.raises(UndeclaredIdentifierError, match="foo"):
        translate(tmp_path, UNDECLARED_INSIDE_GENERIC_OVERLOADS, backend)


def test_undeclared_identifier_does_not_match_enclosing_generic_substitution_vulkan(
    tmp_path,
):
    # SPIR-V raises SpirvLoweringRefused (a TranslationDiagnostic) during codegen,
    # which generate_refusing_undeclared_identifiers preserves rather than replacing
    # with UndeclaredIdentifierError.
    with pytest.raises(SpirvLoweringRefused, match="foo"):
        translate(tmp_path, UNDECLARED_INSIDE_GENERIC_OVERLOADS, "vulkan")


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_builtin_and_enum_member_arguments_are_not_refused_or_read_as_types(
    tmp_path, backend
):
    code = translate(tmp_path, BUILTIN_AND_ENUM_ARGUMENT_OVERLOADS, backend)
    assert "sel_gl_GlobalInvocationID" not in code
    assert "sel_Mode_Fast" not in code
    assert "overload0" not in code
    if backend == "vulkan":
        assert "OpFunctionCall" in code


MATCH_PATTERN_BINDINGS_OVERLOADS = """
shader main {
    enum Outcome { Ok(float), Err(int) };
    struct Pair { float a; float b; };
    buffer float* results;
    T sel<T>(T a) { return a; }
    T sel<T>(T a, T b) { return b; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        Pair p;
        p.a = 1.0;
        p.b = 2.0;
        Outcome o = Outcome::Ok(1.5);
        float s1 = 0.0;
        match o {
            Outcome::Ok(val) => { s1 = sel(val, 1.0); },
            Outcome::Err(_) => { s1 = 0.0; }
        }
        float s2 = 0.0;
        match p {
            Pair { a: pa, b: pb } => { s2 = sel(pa, 2.0); }
        }
        float e = match p {
            Pair { a: xa, b: xb } => sel(xa, 3.0)
        };
        results[0] = s1 + s2 + e;
    }
    }
}
"""


@pytest.mark.parametrize(
    "backend", ["directx", "opengl", "metal", "hip", "slang", "vulkan", "mojo"]
)
def test_match_pattern_bindings_are_not_read_as_type_arguments(tmp_path, backend):
    code = translate(tmp_path, MATCH_PATTERN_BINDINGS_OVERLOADS, backend)
    for pattern_var in ("val", "pa", "xa"):
        _assert_sel_overload(
            code,
            backend,
            f"sel_{pattern_var}",
            {"float", "Float32"},
            "f32",
        )
    if backend == "vulkan":
        assert code.count("OpFunctionCall") >= 3
    else:
        assert len(re.findall(r"=\s*sel_\w+_overload1\(", code)) == 3


CONCRETE_SHARING_GENERIC_NAME = """
shader main {
    buffer float* results;
    T pick_helper<T>(T v) { return v; }
    T combine<T>(T a) { return a; }
    float combine(float a, float b) {
        float scaled = a + b;
        return pick_helper(scaled);
    }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        float x = results[0];
        results[0] = combine(x, 2.0);
    }
    }
}
"""


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_concrete_overload_sharing_generic_name_specializes_callee(tmp_path, backend):
    code = translate(tmp_path, CONCRETE_SHARING_GENERIC_NAME, backend)
    if backend == "vulkan":
        assert code.count("OpFunctionCall") >= 2
    else:
        assert "pick_helper_" in code


STRUCT_ENUM_COLLIDING_EXPLICIT_TYPES = """
shader main {
    enum Mode { Fast = 1, Slow = 2 };
    enum Tag { TagStruct = 1 };
    struct Pair { float x; };
    struct TagStruct { int v; };
    struct FnType { float k; };
    float FnType(float a) { return a; }
    buffer float* results;
    T keep<T>(T a) { return a; }
    T keep<T>(T a, T b) { return b; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        Pair p; p.x = 1.0;
        TagStruct ts; ts.v = 1;
        FnType ft; ft.k = 2.0;
        Mode m = Mode::Fast;
        Mode m1 = keep<Mode>(m);
        Mode m2 = keep(Mode, m);
        Pair p1 = keep<Pair>(p);
        Pair p2 = keep(Pair, p);
        TagStruct t1 = keep<TagStruct>(ts);
        TagStruct t2 = keep(TagStruct, ts);
        FnType f1 = keep<FnType>(ft);
        FnType f2 = keep(FnType, ft);
        results[0] = p1.x + p2.x + float(t1.v + t2.v) + f1.k + f2.k + float(int(m1) + int(m2));
    }
    }
}
"""

ANGLE_STRUCT_ENUM_EXPLICIT_TYPES = """
shader main {
    enum Mode { Fast = 1, Slow = 2 };
    enum Tag { TagStruct = 1 };
    struct Pair { float x; };
    struct TagStruct { int v; };
    struct FnType { float k; };
    float FnType(float a) { return a; }
    buffer float* results;
    T keep<T>(T a) { return a; }
    T keep<T>(T a, T b) { return b; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        Pair p; p.x = 1.0;
        TagStruct ts; ts.v = 1;
        FnType ft; ft.k = 2.0;
        Mode m = Mode::Fast;
        Mode m1 = keep<Mode>(m);
        Pair p1 = keep<Pair>(p);
        TagStruct t1 = keep<TagStruct>(ts);
        FnType f1 = keep<FnType>(ft);
        results[0] = p1.x + float(t1.v) + f1.k + float(int(m1));
    }
    }
}
"""

ALIAS_LEADING_EXPLICIT_TYPE = """
shader main {
    typedef float ScalarAlias;
    buffer float* results;
    T keep<T>(T a) { return a; }
    T keep<T>(T a, T b) { return b; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        float a1 = keep<ScalarAlias>(1.0);
        float a2 = keep(ScalarAlias, 2.0);
        results[0] = a1 + a2;
    }
    }
}
"""

ALIAS_ANGLE_EXPLICIT_TYPE = """
shader main {
    typedef float ScalarAlias;
    buffer float* results;
    T keep<T>(T a) { return a; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        results[0] = keep<ScalarAlias>(1.0);
    }
    }
}
"""


@pytest.mark.parametrize(
    "backend", ["directx", "opengl", "metal", "cuda", "hip", "slang", "mojo"]
)
def test_struct_enum_and_colliding_names_work_as_explicit_type_arguments(
    tmp_path, backend
):
    code = translate(tmp_path, STRUCT_ENUM_COLLIDING_EXPLICIT_TYPES, backend)
    for specialized in ("keep_Mode", "keep_Pair", "keep_TagStruct", "keep_FnType"):
        assert specialized in code


@pytest.mark.parametrize("backend", ["vulkan", "wgsl"])
def test_struct_enum_and_colliding_names_work_with_angle_type_arguments(
    tmp_path, backend
):
    code = translate(tmp_path, ANGLE_STRUCT_ENUM_EXPLICIT_TYPES, backend)
    if backend == "vulkan":
        assert code.count("OpFunctionCall") >= 4
    else:
        for specialized in ("keep_Mode", "keep_Pair", "keep_TagStruct", "keep_FnType"):
            assert specialized in code


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_type_alias_works_as_explicit_type_argument(tmp_path, backend):
    code = translate(tmp_path, ALIAS_LEADING_EXPLICIT_TYPE, backend)
    assert "keep_ScalarAlias" in code


@pytest.mark.parametrize("backend", ["cuda", "hip", "slang", "wgsl", "mojo"])
def test_resolved_type_alias_works_as_angle_type_argument(tmp_path, backend):
    code = translate(tmp_path, ALIAS_ANGLE_EXPLICIT_TYPE, backend)
    assert "keep_" in code


CBUFFER_BLOCK_NAME_OVERLOADS = """
shader main {
    buffer float* results;
    cbuffer Params {
        vec2 offset;
    };
    T sel<T>(T a) { return a; }
    T sel<T>(T a, T b) { return b; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        float x = results[0];
        results[0] = sel(Params, x);
    }
    }
}
"""


@pytest.mark.parametrize("backend", [b for b in ALL_BACKENDS if b != "wgsl"])
def test_cbuffer_block_name_is_not_read_as_type_argument(tmp_path, backend):
    code = translate(tmp_path, CBUFFER_BLOCK_NAME_OVERLOADS, backend)
    assert "sel_Params" not in code
    assert "overload0" not in code
    if backend == "vulkan":
        _assert_vulkan_sel_call(code, "float", ("var:Params", "local:x"))
    else:
        specialized = set(re.findall(r"\bsel_(\w+?)_overload1\(Params,\s*x\)", code))
        assert specialized and specialized <= {"float", "Float32"}


def test_wgsl_cbuffer_block_name_is_not_read_as_type_argument(tmp_path):
    with pytest.raises(ValueError, match="cannot infer concrete type arguments"):
        translate(tmp_path, CBUFFER_BLOCK_NAME_OVERLOADS, "wgsl")
