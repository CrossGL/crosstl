"""Concrete member overloads whose parameters are references.

An argument's type is a value type, so a local ``float2 x`` passed to
``add(thread float2& d, ...)`` binds the reference directly although the
declared spelling ``float2&`` differs. Overload selection used to compare the
spellings, so every overload set of reference parameters was refused as
ambiguous. A call ``receiver.method(args)`` was also untyped, so an accessor
that returns a reference could not select an overload either.
"""

import re

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import MetalPreprocessor, MetalStructMethodError
from crosstl.backend.Metal.reference_binding import (
    argument_binds_parameter,
    argument_is_lvalue,
    member_call_result,
)

LVALUE_REFERENCE_OVERLOADS = """
#include <metal_stdlib>
using namespace metal;

struct Accumulate {
  static void add(thread float2& d, thread float2& a) { d += a; }
  static void add(thread float4& d, thread float4& a) { d += a * 2.0f; }
};

struct Pair {
  float2 values[2];
  thread float2& at(const short i) thread { return values[i]; }
  const thread float2& at(const short i) const thread { return values[i]; }
};

kernel void k(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  float2 x = float2(float(tid), 1.0f);
  float2 y = float2(2.0f, float(tid));
  Accumulate::add(x, y);
  Pair p;
  p.values[0] = x;
  p.values[1] = y;
  Accumulate::add(p.at(0), p.at(1));
  out[tid] = x.x + p.values[0].y;
}
"""

CONST_REFERENCE_OVERLOADS = """
#include <metal_stdlib>
using namespace metal;

struct Scale {
  static float2 apply(const thread float2& v) { return v * 3.0f; }
  static float4 apply(const thread float4& v) { return v * 5.0f; }
};

kernel void k(device float2* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  float2 x = float2(float(tid), 1.0f);
  out[tid] = Scale::apply(x + x);
}
"""

# `put(a * 2.0f, ...)` cannot bind `thread float&` to a temporary, so C++
# selects `put(int, float)` through a conversion. Conversion ranking is not
# modelled, so the call must stay refused rather than bind the reference.
TEMPORARY_TO_NON_CONST_REFERENCE = """
#include <metal_stdlib>
using namespace metal;

struct Store {
  static void put(thread float& d, float v) { d = v; }
  static void put(int d, float v) { }
};

kernel void k(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  float a = float(tid);
  Store::put(a * 2.0f, 1.0f);
  out[tid] = a;
}
"""

ACCESSORS = """
struct Tile {
  float2 values[4];
  thread float2& at(const short i) thread { return values[i]; }
  const thread float2& at(const short i) const thread { return values[i]; }
  float2 copy(const short i) const thread { return values[i]; }
  float4 wide(const short i) thread { return float4(values[i], values[i]); }
  half4 wide(const short i) const thread { return half4(0); }
  void clear() thread { }
  template <typename U> U as(const short i) thread { return U(values[i]); }
  short count(short i) thread { return i; }
  short count(int i) thread { return short(i); }
  float count(float i) thread { return i; }
};
"""


def _compact(text):
    return re.sub(r"\s+", "", text)


def _structs(preprocessor, source):
    return {
        struct.name: struct
        for struct in preprocessor._find_concrete_struct_definitions(source)
    }


@pytest.fixture(scope="module")
def accessors():
    preprocessor = MetalPreprocessor()
    return preprocessor, _structs(preprocessor, ACCESSORS)


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_lvalues_bind_reference_parameter_overloads(tmp_path, backend):
    source = tmp_path / "accumulate.metal"
    source.write_text(LVALUE_REFERENCE_OVERLOADS)

    output = _compact(
        crosstl.translate(
            str(source), backend=backend, source_backend="metal", format_output=False
        )
    )

    # Both overloads survive with their own parameter types, and both calls
    # pass lvalues: the locals, then the accessor's storage.
    assert re.search(r"Accumulate_+add\(x,y\);", output)
    assert re.search(r"Accumulate_+add\(p\.values\[0\],p\.values\[1\]\);", output)
    assert len(re.findall(r"voidAccumulate_+add\(", output)) >= 2


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_temporary_binds_const_reference_overload(tmp_path, backend):
    source = tmp_path / "scale.metal"
    source.write_text(CONST_REFERENCE_OVERLOADS)

    output = _compact(
        crosstl.translate(
            str(source), backend=backend, source_backend="metal", format_output=False
        )
    )

    assert re.search(r"Scale_+apply\(\(?x\+x\)?\)", output)


def test_temporary_does_not_bind_non_const_reference_overload():
    with pytest.raises(MetalStructMethodError, match="Store::put"):
        MetalPreprocessor().preprocess(TEMPORARY_TO_NON_CONST_REFERENCE)


@pytest.mark.parametrize(
    "receiver, call, expected",
    [
        # A non-const receiver selects the non-const accessor.
        ("thread Tile&", "t.at(1)", ("float2", True)),
        ("Tile", "t.at(i + 1)", ("float2", True)),
        # A const receiver admits only the const accessor.
        ("const thread Tile&", "t.at(1)", ("float2", True)),
        # A by-value return is not an lvalue.
        ("Tile", "t.copy(0)", ("float2", False)),
        # The const and non-const overloads of `wide` return different types.
        ("Tile", "t.wide(0)", ("float4", False)),
        ("const Tile", "t.wide(0)", ("half4", False)),
    ],
)
def test_member_call_result_types_named_receivers(accessors, receiver, call, expected):
    preprocessor, structs = accessors
    locals_ = {"t": receiver, "i": "short"}

    assert member_call_result(preprocessor, call, locals_, structs) == expected


@pytest.mark.parametrize(
    "call",
    [
        "t.clear()",  # void
        "t.as<float2>(0)",  # template member function
        "t.count(1)",  # overloads disagree on the result
        "t.at(0).x",  # not a whole-expression call
        "t.missing(0)",
        "u.at(0)",  # untracked receiver
        "q.at(0)",  # pointer receiver
    ],
)
def test_member_call_result_stays_untyped_when_unknown(accessors, call):
    preprocessor, structs = accessors
    locals_ = {"t": "Tile", "q": "thread Tile*"}

    assert member_call_result(preprocessor, call, locals_, structs) is None


@pytest.mark.parametrize(
    "argument, expected",
    [
        ("x", True),
        ("(x)", True),
        ("values[i]", True),
        ("t.values[2]", True),
        ("*ptr", True),
        ("buffer[i]", True),
        ("t.at(0)", True),
        ("t.copy(0)", False),
        ("x + x", False),
        ("float2(x)", False),
        ("unknown", False),
        ("t.values[0].length()", False),
    ],
)
def test_argument_is_lvalue(accessors, argument, expected):
    preprocessor, structs = accessors
    locals_ = {
        "x": "float2",
        "values": "float2[4]",
        "t": "Tile",
        "ptr": "thread float2*",
        "i": "int",
    }
    buffers = {"buffer": "float2"}

    assert (
        argument_is_lvalue(preprocessor, argument, locals_, buffers, structs)
        is expected
    )


@pytest.mark.parametrize(
    "declared, parameter, argument, expected",
    [
        ("float2&", "thread float2&", "x", True),
        ("float2&", "thread float2&", "x + x", False),
        ("float2&", "const thread float2&", "x + x", True),
        # A const reference outside the thread address space does not bind a
        # temporary.
        ("float2&", "const device float2&", "x + x", False),
        # `const float*&` refers to a non-const pointer.
        ("float*&", "const thread float*&", "x + x", False),
        ("float2&&", "thread float2&&", "x", False),
        ("float4&", "thread float4&", "x", False),
        ("float2", "float2", "x + x", True),
    ],
)
def test_argument_binds_parameter(accessors, declared, parameter, argument, expected):
    preprocessor, structs = accessors
    argument_type = "float*" if declared.startswith("float*") else "float2"
    locals_ = {"x": "float2"}

    assert (
        argument_binds_parameter(
            preprocessor,
            declared,
            argument_type,
            parameter,
            argument,
            locals_,
            {},
            structs,
        )
        is expected
    )
