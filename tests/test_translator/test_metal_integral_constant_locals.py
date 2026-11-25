"""Integral-constant locals, alias owners and compile-time loop indices in concrete Metal functions.

An integral constant has no non-static data members, so ``a * b`` between two of them has a result fixed by the
operand types: overload resolution selects the library's operator template ([over.match.oper]). A block-scope local
``const Int<1> mm{};`` has such a type, bound by block scope ([basic.scope.block]). ``Int<F::kRows>`` and ``Int<4>``
name the same type when ``F::kRows`` is 4 ([temp.type]), also when ``F`` is a block-scope or member alias of a
concrete struct ([dcl.typedef]). A ``const_for_loop`` callback parameter is a copy of ``Int<i>{}``, so beside another
integral constant it keeps that type, and a bound such as ``constexpr short TM = Tile::kRows;`` is a constant
expression ([expr.const]).

The kernel below instantiates its load helper once per offset type, so the helper names in the generated code show
that every offset kept its integral-constant type.
"""

import os
import re
import sys

import pytest

from crosstl.backend.Metal.integral_constant_arithmetic import (
    fold_integral_constant_parameter_arithmetic,
)
from crosstl.backend.Metal.integral_constant_locals import (
    fold_concrete_function_integral_constant_arithmetic,
    resolve_const_for_loop_bound_constants,
)
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_INTEGRAL_CONSTANT_LOCALS"
ENTRY = "constant_locals"
WORKGROUP = (32, 1, 1)
GROUPS = 2
ROW = 8
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [((index * 2654435761) % 1999) - 999 for index in range(GROUPS * 32 * ROW)]

PRELUDE = """#include <metal_stdlib>
using namespace metal;

namespace lib {

template <typename T, T v>
struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  using type = integral_constant;
  constexpr operator value_type() const thread noexcept {
    return value;
  }
};

template <int val>
using Int = integral_constant<int, val>;

template <typename T, T tv, typename U, U uv>
constexpr auto operator+(integral_constant<T, tv>, integral_constant<U, uv>) {
  constexpr auto res = tv + uv;
  using res_t = metal::remove_addrspace_t<decltype(res)>;
  return integral_constant<res_t, res>{};
}

template <typename T, T tv, typename U, U uv>
constexpr auto operator*(integral_constant<T, tv>, integral_constant<U, uv>) {
  constexpr auto res = tv * uv;
  using res_t = metal::remove_addrspace_t<decltype(res)>;
  return integral_constant<res_t, res>{};
}

template <int start, int stop, int step, typename F>
constexpr void const_for_loop(F f) {
  if constexpr (start < stop) {
    constexpr auto idx = Int<start>{};
    f(idx);
    const_for_loop<start + step, stop, step, F>(f);
  }
}
"""

SOURCE = PRELUDE + """
struct Frag {
  static constant constexpr short kRows = 4;

  template <typename Off>
  static int load(const device int* src, Off off) {
    src += off;
    return src[0];
  }
};

struct Tile {
  static constant constexpr short kRows = 2;
  using Frag_t = Frag;
};

template <class TileT>
int gather(const device int* src) {
  constexpr short TM = TileT::kRows;
  using F = typename TileT::Frag_t;
  int total = 0;
  const_for_loop<0, TM, 1>([&](auto mm) {
    auto m = mm * Int<F::kRows>{};
    total += F::load(src, m) * (mm + 1);
  });
  return total;
}

} // namespace lib

using namespace lib;

[[kernel]] void constant_locals(
    const device int* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  const device int* row = values + index * 8;
  int a;
  {
    const Int<1> mm{};
    auto m = mm * Int<Frag::kRows>{};
    a = Frag::load(row, m);
  }
  int b;
  {
    Int<0> mm = Int<0>{};
    auto m = mm * Int<Frag::kRows>{} + Int<1>{};
    b = Frag::load(row, m);
  }
  results[index] = gather<Tile>(row) * 100 + a * 10 + b;
}
"""


def _expected():
    results = []
    for index in range(GROUPS * 32):
        row = VALUES[index * ROW : (index + 1) * ROW]
        gathered = row[0] + 2 * row[4]
        results.append(gathered * 100 + row[4] * 10 + row[1])
    return results + GUARD


def _request(root, target):
    source, descriptor, package = _package(
        root, target, "int", WORKGROUP, source=SOURCE, software_subgroups=False
    )
    expected = _expected()
    initial = [-0x21524111] * (len(expected) - len(GUARD)) + GUARD
    inputs = _bound_values(
        descriptor,
        {
            "values": {"dtype": "int32", "shape": [len(VALUES)], "values": VALUES},
            "results": {"dtype": "int32", "shape": [len(initial)], "values": initial},
        },
    )
    outputs = _bound_values(
        descriptor,
        {"results": {"dtype": "int32", "shape": [len(expected)], "values": expected}},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [GROUPS, 1, 1], "workgroupSize": list(WORKGROUP)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


def _contract_preprocessor():
    preprocessor = MetalPreprocessor()
    preprocessor._configure_integral_constant_contracts(PRELUDE)
    assert preprocessor._const_for_loop_contract_verified
    assert preprocessor._integral_constant_binary_operators == {"+", "*"}
    return preprocessor


@pytest.mark.parametrize(
    ("body", "folded"),
    (
        # Each use binds to the declaration of its own block.
        (
            "{ const Int<1> a{}; x = a * Int<4>{}; } { Int<2> a = Int<2>{}; y = a * Int<4>{}; }",
            "{ const Int<1> a{}; x = Int<4>{}; } { Int<2> a = Int<2>{}; y = Int<8>{}; }",
        ),
        (
            "constexpr auto a = Int<3>{}; y = a * Int<2>{};",
            "constexpr auto a = Int<3>{}; y = Int<6>{};",
        ),
        # A folded result is itself an integral-constant local.
        (
            "auto a = Int<2>{} * Int<3>{}; y = a + Int<1>{};",
            "auto a = Int<6>{}; y = Int<7>{};",
        ),
        # A use before the declaration names something else.
        ("y = a * Int<4>{}; const Int<1> a{};", "y = a * Int<4>{}; const Int<1> a{};"),
        # Another declaration of the name, a static local or a lambda parameter keeps every use.
        (
            "const Int<1> a{}; { int a = 3; y = a * Int<4>{}; }",
            "const Int<1> a{}; { int a = 3; y = a * Int<4>{}; }",
        ),
        (
            "static const Int<1> a{}; y = a * Int<4>{};",
            "static const Int<1> a{}; y = a * Int<4>{};",
        ),
        (
            "const Int<1> a{}; f([](int a) { return a * 2; }); y = a * Int<4>{};",
            "const Int<1> a{}; f([](int a) { return a * 2; }); y = a * Int<4>{};",
        ),
        # A converting initializer is no integral-constant local of the declared type.
        (
            "Int<1> a = Int<2>{}; y = a * Int<4>{};",
            "Int<1> a = Int<2>{}; y = a * Int<4>{};",
        ),
        # Member access and unary operands keep their operators.
        (
            "const Int<2> a{}; y = a.value * Int<4>{};",
            "const Int<2> a{}; y = a.value * Int<4>{};",
        ),
        (
            "const Int<2> a{}; y = -a * Int<4>{};",
            "const Int<2> a{}; y = -a * Int<4>{};",
        ),
    ),
)
def test_block_scope_locals_fold_by_scope(body, folded):
    preprocessor = _contract_preprocessor()
    assert (
        fold_integral_constant_parameter_arithmetic(preprocessor, "int x", body)
        == folded
    )


def test_a_parameter_of_the_same_name_keeps_the_local_unfolded():
    preprocessor = _contract_preprocessor()
    body = "{ const Int<1> a{}; y = a * Int<4>{}; }"
    assert (
        fold_integral_constant_parameter_arithmetic(preprocessor, "int a", body) == body
    )


STRUCTS = """
struct Frag { static constant constexpr short kRows = 4; };
struct Tile { static constant constexpr short kRows = 2; using Frag_t = Frag; };
} // namespace lib
"""


@pytest.mark.parametrize(
    ("statement", "folded"),
    (
        ("auto m = Int<1>{} * Int<Frag::kRows>{};", "auto m = Int<4>{};"),
        (
            "using F = typename Tile::Frag_t; auto m = Int<2>{} * Int<F::kRows>{};",
            "using F = typename Tile::Frag_t; auto m = Int<8>{};",
        ),
        (
            "using T = Tile; using F = typename T::Frag_t; auto m = Int<F::kRows>{} * Int<T::kRows>{};",
            "using T = Tile; using F = typename T::Frag_t; auto m = Int<8>{};",
        ),
        # An owner that names no concrete struct keeps its spelling.
        (
            "auto m = Int<1>{} * Int<Missing::kRows>{};",
            "auto m = Int<1>{} * Int<Missing::kRows>{};",
        ),
    ),
)
def test_static_member_arguments_resolve_through_aliases(statement, folded):
    preprocessor = _contract_preprocessor()
    code = PRELUDE + STRUCTS + f"void f() {{ {statement} }}\n"
    output = fold_concrete_function_integral_constant_arithmetic(preprocessor, code)
    assert output.endswith(f"void f() {{ {folded} }}\n")


@pytest.mark.parametrize(
    ("use", "expanded"),
    (
        ("use(idx * Int<4>{});", "use(integral_constant<int, 1>{} * Int<4>{});"),
        ("use(Int<4>{} * idx);", "use(Int<4>{} * integral_constant<int, 1>{});"),
        # The type of a dependent construction is an integral constant too.
        (
            "use(idx * Int<F::kRows>{});",
            "use(integral_constant<int, 1>{} * Int<F::kRows>{});",
        ),
        # Beside a scalar the index converts to its value.
        ("use(idx + 1);", "use(1 + 1);"),
    ),
)
def test_loop_index_keeps_its_type_beside_a_construction(use, expanded):
    preprocessor = _contract_preprocessor()
    source = f"const_for_loop<1, 2, 1>([&](auto idx) {{ {use} }});"
    output = preprocessor._lower_concrete_const_for_loop_callbacks(source)
    assert output == f"{{{{ {expanded} }}}};"


def test_loop_bounds_name_block_constants():
    preprocessor = _contract_preprocessor()
    code = PRELUDE + STRUCTS + """void f() {
  constexpr short TM = Tile::kRows;
  const_for_loop<0, TM, 1>([&](auto i) { use(i.value); });
  {
    constexpr short TM = 3;
    const_for_loop<0, TM, 1>([&](auto i) { use(i.value); });
  }
  const_for_loop<0, unknown, 1>([&](auto i) { use(i.value); });
}
"""
    output = resolve_const_for_loop_bound_constants(preprocessor, code)
    assert output.count("const_for_loop<0, 2, 1>") == 1
    assert output.count("const_for_loop<0, 3, 1>") == 1
    assert output.count("const_for_loop<0, unknown, 1>") == 1


def _translate(root, source, target="directx"):
    (root / "constants.metal").write_text(source)
    return translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("constants.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=WORKGROUP,
        ),
        format_output=False,
    ).to_json()


def test_an_integral_constant_struct_does_not_instantiate_the_operators(tmp_path):
    source = PRELUDE + """
} // namespace lib

using namespace lib;

[[kernel]] void constant_index(device int* out [[buffer(0)]]) {
  auto zero = integral_constant<int, 0>{};
  out[zero] = 1;
}
"""
    report = _translate(tmp_path, source)
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
    generated = next((tmp_path / "out").rglob("*.hlsl")).read_text()
    assert "crosstl_metal_operator_" not in generated


def test_an_unresolvable_static_member_argument_is_refused(tmp_path):
    source = SOURCE.replace(
        "auto m = mm * Int<F::kRows>{};", "auto m = mm * Int<F::kMissing>{};"
    )
    report = _translate(tmp_path, source)
    assert report["summary"]["translatedCount"] == 0
    assert report["diagnostics"], report


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_integral_constant_locals_compile(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    assert "const_for_loop" not in generated
    assert "crosstl_metal_operator_" not in generated
    # Each helper takes an integral-constant offset: the kernel's offsets 4 and 1, the loop's 0 and 4. Targets
    # mangle the helper names differently, but each keeps the offset type's name.
    helpers = set(re.findall(r"\bFrag_+load_+(\w+?)(?:_glsl\w*)?\(", generated))
    offsets = set()
    for helper in helpers:
        match = re.fullmatch(r"(?:Int|integral_constant_int)_(\d+)", helper)
        assert match, helpers
        offsets.add(int(match.group(1)))
    assert offsets == {0, 1, 4} and len(helpers) == 4, helpers
    _compile(generated, target, tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


def test_integral_constant_locals_execute(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native integral-constant locals")
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source, request, expected = _request(tmp_path, target)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry=ENTRY,
        metal_compile_flags=("-Wall", "-Wextra"),
        compare_with_original=target == "metal",
    )


def test_integral_constant_locals_are_required_on_all_native_platforms():
    from pathlib import Path

    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_metal_integral_constant_locals.py" in step
    assert "if:" not in step and "continue-on-error" not in step
