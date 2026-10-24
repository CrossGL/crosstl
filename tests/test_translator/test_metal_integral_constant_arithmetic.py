"""Arithmetic between integral constants, and implicit conversions of aggregates to scalars.

A Metal library may wrap compile-time integers in ``integral_constant<T, v>`` with a conversion operator and free
operator templates such as ``integral_constant<T, tv> * integral_constant<U, uv>``. When both operands are integral
constants, overload resolution picks the template, which returns ``integral_constant<int, tv * uv>``
([over.match.oper]). When only one is, the built-in operator applies after the conversion function
([over.built], [class.conv.fct]). Initializing, assigning, compound-assigning or returning a scalar from such a
value converts it the same way ([dcl.init.general], [expr.ass]), and ``p += c`` on a pointer converts ``c`` to an
integer ([expr.add]).

A ``const thread`` member qualifier only names the object's address space (Metal Shading Language Specification,
section 4.2), and ``metal::remove_addrspace_t`` removes only that qualifier (section 6.2), so neither changes the
contract. MLX's steel attention kernels use these forms in ``BaseMMAFrag::load_safe``.
"""

import os
import sys

import pytest

from crosstl.backend.Metal.integral_constant_arithmetic import (
    fold_integral_constant_parameter_arithmetic,
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

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_INTEGRAL_CONSTANT_ARITHMETIC"
ENTRY = "constant_offsets"
WORKGROUP = (32, 1, 1)
GROUPS = 2
ROW = 8
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [((index * 2654435761) % 1999) - 999 for index in range(GROUPS * 32 * ROW)]

PRELUDE = """#include <metal_stdlib>
using namespace metal;

namespace steel {

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

#define integral_const_binop(__op__, __operator__)          \\
  template <typename T, T tv, typename U, U uv>             \\
  constexpr auto __operator__(                              \\
      integral_constant<T, tv>, integral_constant<U, uv>) { \\
    constexpr auto res = tv __op__ uv;                      \\
    using res_t = metal::remove_addrspace_t<decltype(res)>; \\
    return integral_constant<res_t, res>{};                 \\
  }

integral_const_binop(+, operator+);
integral_const_binop(-, operator-);
integral_const_binop(*, operator*);
integral_const_binop(/, operator/);
"""

SOURCE = PRELUDE + """
template <typename StrX, typename StrY, typename OffX, typename OffY>
int strided(
    const device int* src, StrX str_x, StrY str_y, OffX off_x, OffY off_y) {
  src += off_x * str_x + off_y * str_y;
  int total = src[0];
  src += str_y;
  total += src[0] * str_x;
  total -= str_y * str_x - off_x;
  return total;
}

struct Frag {
  template <typename StrX, typename StrY, typename OffX, typename OffY>
  static int load(
      const device int* src, StrX str_x, StrY str_y, OffX off_x, OffY off_y) {
    src += off_x * str_x + off_y * str_y;
    return src[0] + (off_y - off_x) * str_y / Int<1>{};
  }
};

template <typename Scale>
int scaled(Scale scale) {
  return scale * Int<3>{};
}

} // namespace steel

using namespace steel;

[[kernel]] void constant_offsets(
    const device int* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  const device int* row = values + index * 8;
  int ld = int(index % 3) + 1;
  int a = strided(row, ld, Int<1>{}, 1, 2);
  int b = strided(row, Int<2>{}, Int<1>{}, Int<1>{}, Int<1>{});
  int c = Frag::load(row, Int<3>{}, Int<1>{}, Int<0>{}, Int<2>{});
  results[index] = a * 1000 + b * 10 + c + scaled(Int<4>{});
}
"""


def _expected():
    results = []
    for index in range(GROUPS * 32):
        row = VALUES[index * ROW : (index + 1) * ROW]
        ld = index % 3 + 1
        a = row[ld + 2] + row[ld + 3] * ld - (ld - 1)
        b = row[3] + 2 * row[4] - 1
        c = row[2] + 2
        results.append(a * 1000 + b * 10 + c + 12)
    return results + GUARD


def _request(root, target, source=SOURCE):
    source, descriptor, package = _package(
        root, target, "int", WORKGROUP, source=source, software_subgroups=False
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


def _contract_preprocessor(source=SOURCE):
    preprocessor = MetalPreprocessor()
    preprocessor._configure_integral_constant_contracts(preprocessor.preprocess(source))
    return preprocessor


def test_address_space_spellings_keep_the_integral_constant_contract():
    preprocessor = _contract_preprocessor()
    assert preprocessor._integral_constant_conversion_contract_verified
    assert preprocessor._integral_constant_binary_operators == {"+", "-", "*", "/"}


def test_an_operator_with_another_result_is_not_part_of_the_contract():
    source = SOURCE.replace(
        "return integral_constant<res_t, res>{};",
        "return integral_constant<res_t, res + 1>{};",
    )
    assert not _contract_preprocessor(source)._integral_constant_binary_operators


@pytest.mark.parametrize(
    ("body", "folded"),
    (
        ("return x + a * b;", "return x + Int<6>{};"),
        ("return a * b * x;", "return Int<6>{} * x;"),
        ("return (a + b) * a - Int<1>{};", "return Int<9>{};"),
        ("return a - b - a;", "return Int<(-3)>{};"),
        # `x * a` is evaluated first and `-a` is a unary operand.
        ("return x * a * b;", "return x * a * b;"),
        ("return -a * b;", "return -a * b;"),
        # Member access, a cast and a call keep their operands.
        ("return a.value * b;", "return a.value * b;"),
        ("return (int)a * b;", "return (int)a * b;"),
        ("return f(a) * b;", "return f(a) * b;"),
        # A local redeclaration hides the parameter.
        ("int a = 1; return a * b;", "int a = 1; return a * b;"),
    ),
)
def test_folds_only_complete_integral_constant_subexpressions(body, folded):
    preprocessor = _contract_preprocessor()
    parameters = "Int<2> a, Int<3> b, int x"
    assert (
        fold_integral_constant_parameter_arithmetic(preprocessor, parameters, body)
        == folded
    )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_integral_constant_arithmetic_compiles(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    assert "total += str_y;" not in generated
    assert "= {};" not in generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


def test_integral_constant_arithmetic_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native integral constants")
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


def test_integral_constant_arithmetic_is_required_on_all_native_platforms():
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
    assert "test_metal_integral_constant_arithmetic.py" in step
    assert "if:" not in step and "continue-on-error" not in step


def test_unfolded_aggregate_arithmetic_is_refused(tmp_path):
    source = SOURCE.replace(
        "return integral_constant<res_t, res>{};",
        "return integral_constant<res_t, res + 1>{};",
    )
    (tmp_path / "products.metal").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("products.metal",),
            targets=("directx",),
            output_dir="out",
            workgroup_size=WORKGROUP,
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert any("operator" in item["message"] for item in report["diagnostics"]), report[
        "diagnostics"
    ]
