"""Writes and atomics through same-width 32-bit views of storage words.

A storage view whose element is one 32-bit scalar addresses whole backing
words, so stores are bit casts and atomics operate on the backing word. A
record whose only member is one 32-bit scalar has that scalar's layout, so it
may be the backing element or the view.
"""

import os
import shutil
import struct
import sys

import pytest

import crosstl
from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.codegen.pointer_reinterpret import PointerReinterpretationError
from crosstl.translator.codegen.scalar_wrapper_views import (
    normalize_scalar_wrapper_views,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_STORAGE_WORD_VIEW_RUNTIME"
TARGETS = ("metal", "directx", "opengl")
TOOLS = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}

_PRELUDE = """#include <metal_stdlib>
using namespace metal;

struct Words {
  atomic<uint> val;
};

struct Lanes {
  atomic<float> val;
};

struct Pair {
  float real;
  float imag;
};
"""

_BODIES = {
    # Two float halves added into uint words through wrapper records.
    "wrapper_add": (
        "device Words* words",
        """device Lanes* lanes = reinterpret_cast<device Lanes*>(words);
  observed[2 * i] = atomic_fetch_add_explicit(
      &(lanes[2 * i].val), updates[i].real, memory_order_relaxed);
  observed[2 * i + 1] = atomic_fetch_add_explicit(
      &(lanes[2 * i + 1].val), updates[i].imag, memory_order_relaxed);""",
    ),
    "wrapper_exchange": (
        "device Words* words",
        """device Lanes* lanes = reinterpret_cast<device Lanes*>(words);
  observed[2 * i] = atomic_exchange_explicit(
      &(lanes[2 * i].val), updates[i].real, memory_order_relaxed);
  observed[2 * i + 1] = atomic_load_explicit(
      &(lanes[2 * i + 1].val), memory_order_relaxed);""",
    ),
    "scalar_add": (
        "device atomic<uint>* words",
        """device atomic<float>* lanes = reinterpret_cast<device atomic<float>*>(words);
  observed[2 * i] = atomic_fetch_add_explicit(
      &lanes[2 * i], updates[i].real, memory_order_relaxed);
  observed[2 * i + 1] = atomic_fetch_add_explicit(
      &lanes[2 * i + 1], updates[i].imag, memory_order_relaxed);""",
    ),
    "scalar_store": (
        "device uint* words",
        """device float* lanes = reinterpret_cast<device float*>(words);
  lanes[2 * i] += updates[i].real;
  lanes[2 * i + 1] = updates[i].imag;
  observed[2 * i] = lanes[2 * i];
  observed[2 * i + 1] = lanes[2 * i + 1];""",
    ),
}

_INITIAL = (1.5, -2.25, 3.0, 0.125, 10.0, -0.5)
_UPDATES = ((0.25, 4.0), (-1.0, 0.5), (2.5, -3.75))


def _source(case):
    words, body = _BODIES[case]
    return f"""{_PRELUDE}
kernel void word_views({words} [[buffer(0)]],
                       const device Pair* updates [[buffer(1)]],
                       device float* observed [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {{
  {body}
}}
"""


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _expected(case):
    words, observed = [], []
    for thread, update in enumerate(_UPDATES):
        for lane in range(2):
            initial = _INITIAL[2 * thread + lane]
            value = update[lane]
            if case in {"wrapper_add", "scalar_add"}:
                words.append(_bits(_f32(initial + value)))
                observed.append(initial)
            elif case == "wrapper_exchange":
                words.append(_bits(value if lane == 0 else initial))
                observed.append(initial)
            else:
                stored = _f32(initial + value) if lane == 0 else value
                words.append(_bits(stored))
                observed.append(stored)
    return words, observed


def _translate(tmp_path, source, target):
    path = tmp_path / "views.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=target, format_output=False)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", sorted(_BODIES))
def test_word_view_translates_and_compiles(tmp_path, target, case):
    generated = _translate(tmp_path, _source(case), target)
    if target == "metal":
        # The round trip keeps the view, including a wrapper record view.
        view = "Lanes" if case.startswith("wrapper") else "float"
        assert f"reinterpret_cast<device {view}*>(words)" in generated
    else:
        assert "asfloat(" in generated or "uintBitsToFloat(" in generated
        assert "Lanes" not in generated.split("void", 1)[-1]
    if shutil.which(TOOLS[target]):
        _, module = _compile(generated, target, tmp_path)
        assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_word_view_float_atomic_updates_backing_words(tmp_path, target):
    generated = _translate(tmp_path, _source("wrapper_add"), target)
    if target == "directx":
        assert "InterlockedCompareExchange(storage[index0].val" in generated
        assert "InterlockedCompareExchangeFloatBitwise" not in generated
    else:
        assert "atomicCompSwap(words[crossgl_float_index0].val" in generated
        # The uint backing words are not rewritten as float storage.
        assert "Words words[]" in generated


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_word_view_integer_atomic_uses_backing_word(tmp_path, target):
    source = _PRELUDE + """
struct Counts {
  atomic<uint> val;
};

kernel void maxima(device Words* words [[buffer(0)]],
                   const device uint* updates [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
  device Counts* counts = reinterpret_cast<device Counts*>(words);
  atomic_fetch_max_explicit(&(counts[i].val), updates[i], memory_order_relaxed);
}
"""
    generated = _translate(tmp_path, source, target)
    intrinsic = "InterlockedMax(words[" if target == "directx" else "atomicMax(words["
    assert intrinsic in generated
    assert "].val, " in generated.split(intrinsic, 1)[1]
    if shutil.which(TOOLS[target]):
        _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_word_view_integer_atomic_needs_backing_signedness(tmp_path, target):
    source = """#include <metal_stdlib>
using namespace metal;
kernel void maxima(device atomic<uint>* out [[buffer(0)]],
                   const device int* updates [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
  device atomic<int>* signed_view = reinterpret_cast<device atomic<int>*>(out);
  atomic_fetch_max_explicit(&signed_view[i], updates[i], memory_order_relaxed);
}
"""
    with pytest.raises(PointerReinterpretationError) as error:
        _translate(tmp_path, source, target)
    assert error.value.reason == "atomic-signedness-mismatch"


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_narrow_view_atomics_stay_refused(tmp_path, target):
    source = """#include <metal_stdlib>
using namespace metal;
kernel void bytes(device atomic<uint>* out [[buffer(0)]],
                  uint i [[thread_position_in_grid]]) {
  device atomic<ushort>* halves = reinterpret_cast<device atomic<ushort>*>(out);
  atomic_fetch_add_explicit(&halves[i], ushort(1), memory_order_relaxed);
}
"""
    with pytest.raises(ValueError) as error:
        _translate(tmp_path, source, target)
    if target == "directx":
        assert "requires a scalar int or uint target, got uint16" in str(error.value)
    else:
        assert isinstance(error.value, PointerReinterpretationError)
        assert error.value.reason == "atomic-requires-word-view"


_RECORD_VIEWS = {
    # Two members: the view element is not one word.
    "two_members": (
        "struct Lanes2 { atomic<float> val; atomic<float> other; };",
        "device Lanes2* lanes = reinterpret_cast<device Lanes2*>(out);\n"
        "  atomic_fetch_add_explicit(&(lanes[i].val), 1.0f, memory_order_relaxed);",
    ),
    # A 16-bit member: the record is not a 32-bit word.
    "narrow_member": (
        "struct Halves { half val; };",
        "device Halves* lanes = reinterpret_cast<device Halves*>(out);\n"
        "  lanes[i].val = half(1.0h);",
    ),
    # The record escapes as a whole value, which Metal can still express.
    "whole_record": (
        "struct Plain { float val; };",
        "device Plain* lanes = reinterpret_cast<device Plain*>(out);\n"
        "  Plain copy = lanes[i];\n"
        "  lanes[i + 1].val = copy.val;",
    ),
}


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", sorted(_RECORD_VIEWS))
def test_non_wrapper_record_views_stay_refused(tmp_path, target, case):
    declaration, body = _RECORD_VIEWS[case]
    source = f"""#include <metal_stdlib>
using namespace metal;
{declaration}
kernel void views(device uint* out [[buffer(0)]],
                  uint i [[thread_position_in_grid]]) {{
  {body}
}}
"""
    if target == "metal" and case == "whole_record":
        # A one-scalar record view is native Metal, read whole or by member.
        generated = _translate(tmp_path, source, target)
        assert "reinterpret_cast<device Plain*>(out_)" in generated
        if shutil.which(TOOLS[target]):
            _compile(generated, target, tmp_path)
        return
    with pytest.raises(PointerReinterpretationError) as error:
        _translate(tmp_path, source, target)
    expected = (
        "target-lowering-unavailable"
        if target == "metal"
        else "unsupported-scalar-layout"
    )
    assert error.value.reason == expected


def test_wrapper_view_used_as_whole_record_is_not_rewritten():
    source = """shader main {
    struct Plain { float val; }
    compute {
        void main(RWStructuredBuffer<uint> words @buffer(0)) {
            device Plain* view = (device Plain*)words;
            Plain copy = view[0];
            view[1].val = copy.val;
        }
    }
}
"""
    assert normalize_scalar_wrapper_views(parse(source)) is False


def _request(root, target, case):
    source, descriptor, package = _package(
        root,
        target,
        "float",
        (1, 1, 1),
        source=_source(case),
        software_subgroups=False,
    )
    words, observed = _expected(case)
    inputs = {
        "words": {
            "dtype": "uint32",
            "shape": [6],
            "values": [_bits(value) for value in _INITIAL],
        },
        "updates": {
            "dtype": "float32",
            "shape": [3, 2],
            "values": [lane for pair in _UPDATES for lane in pair],
        },
        "observed": {"dtype": "float32", "shape": [6], "values": [-99.0] * 6},
    }
    outputs = _bound_values(
        descriptor,
        {
            "words": {"dtype": "uint32", "shape": [6], "values": words},
            "observed": {"dtype": "float32", "shape": [6], "values": observed},
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        outputs,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("case", sorted(_BODIES))
def test_word_view_executes_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required word-view execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry="word_views",
    )


_HELPER_SOURCE = _PRELUDE + """
void add_lane(device Lanes* object, float value, uint offset) {
  atomic_fetch_add_explicit(&(object[offset].val), value, memory_order_relaxed);
}

void add_word(device Words* object, uint value, uint offset) {
  atomic_fetch_add_explicit(&(object[offset].val), value, memory_order_relaxed);
}

void add_direct(device Words* object, uint value, uint offset) {
  atomic_fetch_add_explicit(&(object[offset].val), value, memory_order_relaxed);
}

void add_both(device Words* object, device Words* bits, uint offset) {
  add_word(object, 1u, offset);
  add_word(bits, 2u, offset);
}

kernel void helpers(device Words* words [[buffer(0)]],
                    const device Pair* updates [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
  device Lanes* lanes = reinterpret_cast<device Lanes*>(words);
  add_lane(lanes, updates[i].real, 2 * i);
  add_lane(lanes, updates[i].imag, 2 * i + 1);
  add_direct(words, 1u, 2 * i + 6);
  device Words* bits = reinterpret_cast<device Words*>(words);
  add_both(words, bits, 2 * i + 7);
}
"""


def test_wrapper_views_forwarded_to_helpers_normalize_as_scalar_views():
    ast = parse(crosstl.translate(_source_file(_HELPER_SOURCE), backend="cgl"))
    assert normalize_scalar_wrapper_views(ast) is True
    from crosstl.translator.ast import PointerReinterpretNode

    functions = {function.name: function for function in _functions(ast)}

    def pointee(name, index=0):
        return functions[name].parameters[index].param_type.pointee_type.name

    # A helper that receives a view reaches its records only through the member.
    assert pointee("add_lane") == "float"
    assert pointee("add_word") == "uint"
    # A helper whose callers pass only records keeps its declared type.
    assert pointee("add_direct") == "Words"
    assert (pointee("add_both", 0), pointee("add_both", 1)) == ("Words", "uint")
    calls = [
        node
        for node in _nodes(functions["add_both"])
        if getattr(getattr(node, "function", None), "name", None) == "add_word"
    ]
    # The record pointer reaches the rewritten helper through a scalar view.
    assert isinstance(calls[0].arguments[0], PointerReinterpretNode)
    assert not isinstance(calls[1].arguments[0], PointerReinterpretNode)


@pytest.mark.parametrize("target", ("metal", "opengl"))
def test_wrapper_views_forwarded_to_helpers_translate(tmp_path, target):
    generated = _translate(tmp_path, _HELPER_SOURCE, target)
    if target == "metal":
        assert "reinterpret_cast<device Lanes*>(words)" in generated
    else:
        assert "atomicCompSwap(words[" in generated
        assert "atomicAdd(words[" in generated
    if shutil.which(TOOLS[target]):
        _compile(generated, target, tmp_path)


def test_directx_refuses_word_views_passed_to_helpers(tmp_path):
    # Helper resource parameters take their element type from the declaration
    # until call sites solve the physical layout (#4025).
    from crosstl.translator.codegen.directx_codegen import (
        DirectXResourcePointerParameterError,
    )

    with pytest.raises(DirectXResourcePointerParameterError) as error:
        _translate(tmp_path, _HELPER_SOURCE, "directx")
    assert error.value.reason == "element-type-mismatch"


def _source_file(source):
    import tempfile
    from pathlib import Path

    directory = Path(tempfile.mkdtemp(prefix="word-views-"))
    path = directory / "helpers.metal"
    path.write_text(source, encoding="utf-8")
    return str(path)


def _nodes(root):
    from crosstl.translator.codegen.scalar_wrapper_views import _walk

    return [node for node, _ in _walk(root)]


def _functions(ast):
    from crosstl.translator.ast import FunctionNode

    return [node for node in _nodes(ast) if isinstance(node, FunctionNode)]
