"""Subscripts of values whose type is a template type parameter.

``reg[i]`` for a parameter ``thread T& reg`` is type-dependent: each instantiation selects its own subscript, so the
importer keeps it in the CrossGL generic body, and every target types it once it has substituted ``T``.
"""

import platform
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.common_ast import ArrayAccessNode, VariableNode
from crosstl.backend.Metal.dependent_builtins import dependent_subscript_selection
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser

PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n"

KERNEL = """
kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  out[i] = 1.0f;
}
"""

LANES = PRELUDE + """
template <typename type4>
void fill_t4(device const float* src, short il, thread type4& reg) {
    for (int i = 0; i < 2; i++) {
        reg[2 * i + 0] = src[il + i];
        reg[2 * i + 1] = src[il + i] * 2.0f;
    }
}
"""

LANES_KERNEL = """
kernel void k(device const float* src [[buffer(0)]],
              device float4* dst [[buffer(1)]],
              uint tid [[thread_position_in_grid]]) {
    float4 r;
    fill_t4(src, (short)tid, r);
    dst[tid] = r;
}
"""

COLUMNS = PRELUDE + """
template <typename M>
void fill(thread M& m, float v) {
    for (int i = 0; i < 16; i++) {
        m[i / 4][i % 4] = v;
    }
}

template <typename T04>
float total(device const float* x, int n, int k) {
    device const T04* x4 = (device const T04*)x;
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        s += x4[i][k];
    }
    return s;
}
"""

COLUMNS_KERNEL = """
kernel void k(device const float* x [[buffer(0)]],
              device float* out [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    float4x4 m;
    fill(m, x[i]);
    out[i] = total<float4>(x, 4, int(i % 4)) + m[1][2];
}
"""

USER_SUBSCRIPT = PRELUDE + """
struct S {
    float v[4];
    thread float& operator[](int i) thread { return v[i]; }
};

template <typename T>
void fill(thread T& reg, float x) {
    for (int i = 0; i < 4; i++) {
        reg[i] = x + float(i);
    }
}

kernel void k(device const float* x [[buffer(0)]],
              device float* out [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    S s;
    fill(s, x[i]);
    out[i] = s.v[2];
}
"""


def _compile_metal_if_available(code, tmp_path):
    xcrun = shutil.which("xcrun")
    if platform.system() != "Darwin" or xcrun is None:
        return
    shader = tmp_path / "out.metal"
    shader.write_text(code)
    subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-std=metal3.1", "-c", str(shader)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


def _translate(tmp_path, source, backend):
    path = tmp_path / "unit.metal"
    path.write_text(source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def test_uninstantiated_template_keeps_dependent_lane_writes(tmp_path):
    code = _translate(tmp_path, LANES + KERNEL, "cgl")

    assert "generic<type4> void fill_t4(" in code
    assert "reg[2 * i + 0] = src[il + i];" in code
    assert "reg[2 * i + 1] = src[il + i] * 2.0f;" in code


def test_nested_and_pointer_element_subscripts_stay_dependent(tmp_path):
    code = _translate(tmp_path, COLUMNS + KERNEL, "cgl")

    assert "generic<M> void fill(" in code
    assert "m[i / 4][i % 4] = v;" in code
    assert "generic<T04> float total(" in code
    assert "s += x4[i][k];" in code


def test_instantiated_dependent_lane_writes_reach_metal(tmp_path):
    code = _translate(tmp_path, LANES + LANES_KERNEL, "metal")

    assert "thread float4& reg" in code
    assert "reg[2 * i + 0] = src[il + i];" in code
    _compile_metal_if_available(code, tmp_path)


def test_instantiated_dependent_columns_reach_metal(tmp_path):
    code = _translate(tmp_path, COLUMNS + COLUMNS_KERNEL, "metal")

    assert "thread float4x4& m" in code
    assert "m[i / 4][i % 4] = v;" in code
    _compile_metal_if_available(code, tmp_path)


@pytest.mark.parametrize("backend", ["cgl", "metal"])
def test_user_subscript_operator_keeps_dependent_subscript_refused(tmp_path, backend):
    with pytest.raises(Exception, match="aggregate-subscript"):
        _translate(tmp_path, USER_SUBSCRIPT, backend)


def _struct(source):
    ast = MetalParser(MetalLexer(source).tokenize()).parse()
    return next(struct for struct in ast.structs if struct.name == "S")


@pytest.mark.parametrize(
    "member",
    [
        "thread float& operator[](int i) thread { return v[i]; }",
        "float operator [ ] (int i) const;",
        "template <typename I> float operator[](I i) const { return v[i]; }",
    ],
)
def test_parser_records_subscript_operators(member):
    assert _struct(
        "struct S { float v[4]; " + member + " };"
    ).declares_subscript_operator


@pytest.mark.parametrize(
    "member",
    [
        "",
        "thread S& operator=(thread const S& other) thread;",
        "float get(int i) const;",
    ],
)
def test_parser_records_no_subscript_operator(member):
    assert not _struct(
        "struct S { float v[4]; " + member + " };"
    ).declares_subscript_operator


def _type_of(types):
    return lambda node: types.get(getattr(node, "name", None))


def test_selection_is_dependent_only_for_unbound_type_parameters():
    reg = VariableNode("T&", "reg")
    selection = dependent_subscript_selection(reg, "T", _type_of({}), frozenset({"T"}))

    assert selection == {"kind": "dependent", "base_type": "T", "selected_type": None}
    assert dependent_subscript_selection(reg, "float4", _type_of({}), {"T"}) is None
    assert dependent_subscript_selection(reg, "T", _type_of({}), frozenset()) is None


def test_element_of_dependent_value_is_dependent():
    column = ArrayAccessNode(VariableNode("M&", "m"), VariableNode("int", "i"))
    selection = dependent_subscript_selection(
        column, None, _type_of({"m": "M&"}), frozenset({"M"})
    )

    assert selection["kind"] == "dependent"
    assert (
        dependent_subscript_selection(column, None, _type_of({"m": "float4x4"}), {"M"})
        is None
    )
