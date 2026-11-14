"""User function templates that overload a Metal SIMD-group function for an aggregate.

``Pair<U> simd_shuffle_down(Pair<U>, ushort)`` overloads the library function, and its body shuffles the members
through the library. C++ resolves each call per instantiation: a ``Pair<T>`` argument selects the template and the
members fail its deduction, so they reach ``metal::simd_shuffle_down``. The Metal and CrossGL targets kept the
template as one CrossGL generic, so the aggregate call became the library operation on a struct and the member calls
in the template body called the template again, which the Metal generator refused as recursion.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.class_argument_precedence import (
    class_identity_template_outranks,
)
from crosstl.backend.Metal.preprocessor import MetalTemplateSpecializationError
from crosstl.backend.Metal.simd_group_template_overloads import (
    SIMD_GROUP_FUNCTION_NAMES,
    only_simd_group_overloads,
    overloads_simd_group_function,
)

PAIRS = """
#include <metal_stdlib>
using namespace metal;

template <typename U>
struct Pair {
  uint32_t index;
  U val;
};

struct Other {
  float a;
};

inline Other simd_shuffle_down(Other data, ushort delta) {
  return Other{simd_shuffle_down(data.a, delta)};
}

template <typename U>
Pair<U> simd_shuffle_down(Pair<U> data, ushort delta) {
  return Pair<U>{simd_shuffle_down(data.index, delta), simd_shuffle_down(data.val, delta)};
}
EXTRA
template <typename T>
[[kernel]] void shuffle_pairs(
    device T* values [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    uint lane [[thread_index_in_simdgroup]]) {
  Pair<T> best{lane, values[lane]};
  for (uint offset = 16; offset > 0; offset /= 2) {
    Pair<T> neighbor = simd_shuffle_down(best, offset);
    best.val += neighbor.val;
    best.index = max(best.index, neighbor.index);
  }
  values[lane] = RESULT;
  indices[lane] = best.index;
}

template [[host_name("shuffle_pairs_float")]] [[kernel]] void shuffle_pairs<float>(
    device float*, device uint*, uint);
template [[host_name("shuffle_pairs_half")]] [[kernel]] void shuffle_pairs<half>(
    device half*, device uint*, uint);
"""


def _source(extra="", result="best.val"):
    return PAIRS.replace("EXTRA", extra).replace("RESULT", result)


def _translate(tmp_path, code, backend):
    source = tmp_path / "pairs.metal"
    source.write_text(code)
    return crosstl.translate(
        str(source), backend=backend, source_backend="metal", format_output=False
    )


def test_crossgl_calls_the_template_instance_and_the_library_for_members(tmp_path):
    output = _translate(tmp_path, _source(), "cgl")

    assert "Pair_float neighbor = simd_shuffle_down_float(best, offset);" in output
    assert "Pair_half neighbor = simd_shuffle_down_half(best, offset);" in output
    # The aggregate never reaches the library operation, and no generic template remains.
    assert "WaveShuffleDown(best" not in output
    assert "generic<U> Pair<U> simd_shuffle_down" not in output
    body = output[output.index("Pair_float simd_shuffle_down_float(") :]
    body = body[: body.index("}")]
    assert "WaveShuffleDown(data.index" in body
    assert "WaveShuffleDown(data.val" in body
    # The concrete overload for another struct is kept as written.
    assert "WaveShuffleDown(data.a" in output


def test_metal_round_trip_is_not_recursive(tmp_path):
    output = _translate(tmp_path, _source(), "metal")

    assert "simd_shuffle_down_float_fn(best," in output
    assert "simd_shuffle_down(data.index," in output
    assert "simd_shuffle_down(data.val," in output
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return
    metal = tmp_path / "out.metal"
    metal.write_text(output, encoding="utf-8")
    result = subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(metal)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" not in result.stderr:
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_template_hostile_targets_keep_their_instances(tmp_path, backend):
    output = _translate(tmp_path, _source(), backend)

    assert "simd_shuffle_down_float" in output
    assert "simd_shuffle_down_half" in output


@pytest.mark.parametrize("backend", ["metal", "cgl"])
def test_template_competing_with_the_library_for_scalars_is_refused(tmp_path, backend):
    # The Metal compiler reports this call as ambiguous.
    extra = "\ntemplate <typename T>\nT simd_sum(T x) { return x + T(1); }\n"
    with pytest.raises(MetalTemplateSpecializationError, match="competes with"):
        _translate(tmp_path, _source(extra, "simd_sum(best.val)"), backend)


@pytest.mark.parametrize("backend", ["metal", "cgl"])
def test_concrete_overload_of_the_same_struct_is_not_outranked(tmp_path, backend):
    # C++ selects the non-template for Pair<float>; that ranking is not proven here.
    extra = (
        "\ninline Pair<float> simd_shuffle_down(Pair<float> data, ushort delta) {\n"
        "  return data;\n}\n"
    )
    with pytest.raises(MetalTemplateSpecializationError, match="unproven"):
        _translate(tmp_path, _source(extra), backend)


def test_simd_group_names_cover_the_importer_wave_intrinsics():
    from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter

    public = {
        name
        for name in MetalToCrossGLConverter.metal_wave_intrinsics
        if not name.startswith("__")
    }
    assert public <= SIMD_GROUP_FUNCTION_NAMES


class _Template:
    def __init__(self, name, constraints=()):
        self.name = name
        self.template_constraints = list(constraints)


def test_only_unconstrained_simd_group_templates_join_the_pass():
    assert overloads_simd_group_function(_Template("simd_shuffle_down"))
    assert overloads_simd_group_function(_Template("quad_broadcast"))
    assert not overloads_simd_group_function(_Template("shuffle_down"))
    # A constrained template keeps the constrained rules.
    assert not overloads_simd_group_function(
        _Template("simd_sum", ["is_same_v<T, long>"])
    )
    assert only_simd_group_overloads([_Template("simd_shuffle_down")])
    assert not only_simd_group_overloads([])
    assert not only_simd_group_overloads(
        [_Template("simd_shuffle_down"), _Template("simd_shuffle_down", ["x"])]
    )


class _Struct:
    def __init__(self, name, code):
        start = code.index(f"struct {name}")
        open_brace = code.index("{", start)
        self.span = (start, code.index("};", open_brace) + 2)
        self.body_span = (open_brace + 1, code.index("};", open_brace))


class _Context:
    def __init__(self, code, names):
        self.structs_by_name = {name: _Struct(name, code) for name in names}
        self.struct_type_aliases = {}


class _Competitor:
    def __init__(self, arguments):
        self.arguments = arguments
        self.span = (0, 0)


def _outranks(code, structs, argument_types, competitor):
    from crosstl.backend.Metal.preprocessor import MetalPreprocessor

    preprocessor = MetalPreprocessor()
    template = preprocessor._find_template_functions(code)[0]
    return class_identity_template_outranks(
        preprocessor,
        code,
        _Context(code, structs),
        template,
        {"U": "float"},
        argument_types,
        _Competitor(competitor),
        len(code),
    )


PROOF = """
struct Pair_float { uint index; float val; };
struct Other { float a; };
struct Derived : Other { float b; };
template <typename U>
Pair_float pick(Pair_float data, ushort delta) { return data; }
"""


def test_class_identity_proof():
    structs = ["Pair_float", "Other", "Derived"]
    # Identity on the class argument, the same type for the other one.
    assert _outranks(
        PROOF, structs, ["Pair_float", "uint"], ["Other data", "ushort delta"]
    )
    assert _outranks(
        PROOF, structs, ["Pair_float", "uint"], ["float data", "uint16_t delta"]
    )
    # The competitor takes the class itself.
    assert not _outranks(
        PROOF, structs, ["Pair_float", "uint"], ["Pair_float data", "ushort delta"]
    )
    # The other argument is converted differently.
    assert not _outranks(
        PROOF, structs, ["Pair_float", "uint"], ["Other data", "uint delta"]
    )
    # A reference parameter, or an unknown type, is not ranked.
    assert not _outranks(
        PROOF, structs, ["Pair_float", "uint"], ["thread Other& data", "ushort delta"]
    )
    assert not _outranks(
        PROOF, structs, ["Pair_float", "uint"], ["Unknown data", "ushort delta"]
    )


def test_class_identity_proof_needs_a_class_without_base():
    code = PROOF.replace("Pair_float pick(Pair_float", "Derived pick(Derived")
    structs = ["Pair_float", "Other", "Derived"]
    assert not _outranks(
        code, structs, ["Derived", "uint"], ["Other data", "ushort delta"]
    )


RANKED = """
#include <metal_stdlib>
using namespace metal;

template <typename U>
U simd_shuffle_down(U value, PARAMETER offset) {
  return value;
}
"""


class _Call:
    def __init__(self, name):
        self.name = name
        self.span = (0, 0)
        self.arguments = []


def _selected(parameter, argument_types, name="simd_shuffle_down"):
    from crosstl.backend.Metal.preprocessor import MetalPreprocessor
    from crosstl.backend.Metal.simd_group_template_overloads import (
        simd_group_library_overload_selected,
    )

    code = RANKED.replace("PARAMETER", parameter).replace("simd_shuffle_down", name)
    preprocessor = MetalPreprocessor()
    template = preprocessor._find_template_functions(code)[0]
    return simd_group_library_overload_selected(
        preprocessor,
        code,
        _Call(name),
        argument_types,
        template,
        {"U": argument_types[0]},
    )


def test_template_with_an_exact_delta_outranks_the_library():
    # The library converts uint to ushort; the template takes both arguments exactly.
    assert _selected("uint", ["float", "uint"]) is False


def test_library_with_an_exact_delta_outranks_the_template():
    # The template promotes ushort to int; the library takes both arguments exactly.
    assert _selected("int", ["float", "ushort"]) is True


def test_library_takes_no_bool_or_long_data():
    assert _selected("ushort", ["bool", "ushort"]) is False
    assert _selected("ushort", ["long", "ushort"]) is False
    # Vectors are valid data.
    with pytest.raises(MetalTemplateSpecializationError, match="unproven"):
        _selected("ushort", ["float4", "ushort"])


def test_integral_operations_take_no_floating_data():
    single = RANKED.replace(", PARAMETER offset", "")
    from crosstl.backend.Metal.preprocessor import MetalPreprocessor
    from crosstl.backend.Metal.simd_group_template_overloads import (
        simd_group_library_overload_selected,
    )

    for name, expected in (("simd_and", False), ("simd_sum", None)):
        code = single.replace("simd_shuffle_down", name)
        preprocessor = MetalPreprocessor()
        template = preprocessor._find_template_functions(code)[0]

        def select():
            return simd_group_library_overload_selected(
                preprocessor, code, _Call(name), ["float"], template, {"U": "float"}
            )

        if expected is None:
            with pytest.raises(MetalTemplateSpecializationError):
                select()
        else:
            assert select() is expected


def test_converting_deltas_of_different_types_are_unproven():
    with pytest.raises(MetalTemplateSpecializationError, match="unproven"):
        _selected("int", ["float", "uint"])
