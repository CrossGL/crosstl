"""Constant static members initialized by converting constructors on Metal.

``static constexpr constant U init = U(0);`` with a record ``U`` needs a
constant expression on Metal. The Metal importer folds the constructor call
to ``U(member values)``. CrossGL that calls a member-wise constructor helper
in a constant initializer folds in the Metal generator to the aggregate the
helper builds, and a helper left without callers is not emitted, because
Metal compiles with ``-Werror``. Entry scoping keeps helpers that retained
records call from member initializers.
"""

import os
import shutil
import sys

import pytest

import crosstl
from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_CONSTANT_INITIALIZER_RUNTIME"
TOOLS = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}

_SOURCE = """#include <metal_stdlib>
using namespace metal;

template <typename T>
struct complex_t {
  T real;
  T imag;

  constexpr complex_t(T real, T imag) thread : real(real), imag(imag) {};

  template <typename U, typename = typename enable_if<is_convertible_v<U, T>>::type>
  constexpr complex_t(U x) thread : real(static_cast<T>(x)), imag(static_cast<T>(0)) {}
};

template <typename U>
struct Sum {
  static constexpr constant U init = U(3);
  float scale;
};

kernel void fill(device complex_t<float>* results [[buffer(0)]],
                 uint i [[thread_position_in_grid]]) {
  Sum<complex_t<float>> op{1.0f};
  complex_t<float> value = Sum<complex_t<float>>::init;
  results[i] = complex_t<float>(value.real + op.scale * float(i), value.imag - 1.0f);
}

kernel void other(device float* out [[buffer(0)]],
                  uint i [[thread_position_in_grid]]) {
  out[i] = float(i);
}
"""


def _translate(tmp_path, target):
    path = tmp_path / "sum.metal"
    path.write_text(_SOURCE, encoding="utf-8")
    return crosstl.translate(str(path), backend=target, format_output=False)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_constructor_initializer_translates_and_compiles(tmp_path, target):
    generated = _translate(tmp_path, target)
    if target == "metal":
        assert "init = complex_t_float{float(float(3)), float(float(0))};" in generated
        # The helper's only call was folded, so it is not emitted.
        assert "_2_int" not in generated
    if shutil.which(TOOLS[target]):
        _, module = _compile(generated, target, tmp_path)
        assert module.is_file() and module.stat().st_size


def test_entry_scoped_round_trip_keeps_initializer_helpers(tmp_path):
    (tmp_path / "sum.metal").write_text(_SOURCE, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("sum.metal",),
            targets=("metal",),
            output_dir="out",
            entry_points={"sum.metal": ("fill",)},
            workgroup_size=(1, 1, 1),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 0, report["diagnostics"]
    (artifact,) = report["artifacts"]
    generated = (tmp_path / artifact["path"]).read_text()
    assert "init = complex_t_float{float(float(3)), float(float(0))};" in generated
    assert "kernel void other" not in generated
    if shutil.which("xcrun"):
        _, module = _compile(generated, "metal", tmp_path)
        assert module.is_file() and module.stat().st_size


_HELPER_SOURCE = """shader main {
    struct Pair {
        float real;
        float imag;
    }

    struct Holder {
        static constant Pair init = make_pair_int(3);
        float scale;
    }

    Pair make_pair_int(int x) @metal_static {
        Pair value;
        value.real = float(x);
        value.imag = float(0);
        return value;
    }

    compute {
        @ stage_entry
        void fill(RWStructuredBuffer<Pair> results @buffer(0), uint i @gl_GlobalInvocationID) {
            Pair value = Holder::init;
            buffer_store(results, i, value);
        }
    }
}
"""


def test_crossgl_helper_call_in_constant_initializer_folds(tmp_path):
    # The Metal importer folds constructor calls itself, so CrossGL written
    # with a member-wise helper call is what exercises the generator fold.
    path = tmp_path / "pair.cgl"
    path.write_text(_HELPER_SOURCE, encoding="utf-8")
    generated = crosstl.translate(str(path), backend="metal", format_output=False)
    assert "init = {float(3), float(0)};" in generated
    assert "make_pair_int" not in generated
    if shutil.which("xcrun"):
        _, module = _compile(generated, "metal", tmp_path)
        assert module.is_file() and module.stat().st_size


def test_non_member_wise_helpers_are_not_folded():
    from crosstl.translator.codegen.metal_constant_initializers import (
        member_wise_constructor_values,
    )

    class Generator:
        functions_by_name = {}
        function_overloads_by_name = {}
        structs_by_name = {}

    from crosstl.translator.ast import FunctionCallNode, IdentifierNode

    call = FunctionCallNode(IdentifierNode("missing"), [])
    assert member_wise_constructor_values(Generator(), call) is None


def test_constructor_initializer_executes_natively(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required Metal initializer execution")
    if sys.platform != "darwin":
        pytest.skip("the Metal round trip runs on macOS")
    source, descriptor, package = _package(
        tmp_path,
        "metal",
        "float",
        (1, 1, 1),
        source=_SOURCE.split("kernel void other")[0],
        software_subgroups=False,
    )
    expected = [value for i in range(4) for value in (3.0 + i, -1.0)]
    outputs = _bound_values(
        descriptor,
        {"results": {"dtype": "float32", "shape": [4, 2], "values": expected}},
    )
    inputs = _bound_values(
        descriptor,
        {"results": {"dtype": "float32", "shape": [4, 2], "values": [0.0] * 8}},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [4, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target="metal",
    )
    assert not request.execution_plan.diagnostics
    _execute(request, outputs, tmp_path, original_source=source, original_entry="fill")
