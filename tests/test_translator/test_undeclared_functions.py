"""A call to a function the program never declares is refused before any generator copies it into the target."""

import json
import re
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import UndeclaredFunctionError, apply_rule_carrying
from crosstl.translator.semantics.builtin_function_names import (
    BUILTIN_FUNCTION_NAMES,
    CROSSGL_FUNCTION_NAMES,
    CUDA_FUNCTION_NAMES,
    GLSL_EXTENSION_FUNCTION_NAMES,
    GLSL_FUNCTION_NAMES,
    HLSL_FUNCTION_NAMES,
    METAL_FUNCTION_NAMES,
)
from crosstl.translator.semantics.declared_functions import (
    declared_function_names,
    undeclared_function_calls,
)

TARGETS = ["metal", "directx", "opengl", "vulkan"]

UNDECLARED = """
shader Undeclared {
    compute {
        buffer float values[];

        @compute @workgroup_size(1, 1, 1)
        void main() {
            values[0] = undeclaredHelper(values[1]);
        }
    }
}
"""

DECLARED = """
shader Declared {
    struct Pair {
        float a;
        float b;
    }

    float twice(float x) {
        return x * 2.0;
    }

    T pick<T>(T a, T b) {
        return a;
    }

    compute {
        buffer float values[];

        @compute @workgroup_size(1, 1, 1)
        void main() {
            Pair p = Pair(values[1], 2.0);
            vec2 v = vec2(p.a, p.b);
            values[0] = twice(max(v.x, v.y)) + pick<float>(p.a, p.b) + clamp(p.a, 0.0, 1.0);
        }
    }
}
"""

METAL_UNDECLARED = """
#include <metal_stdlib>
using namespace metal;

kernel void run(device float* values [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    values[id] = undeclared_helper(values[id]);
}
"""

HLSL_UNDECLARED = """
RWStructuredBuffer<float> values : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    values[id.x] = undeclared_helper(values[id.x]);
}
"""

GLSL_UNDECLARED = """
#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Values { float values[]; };

void main() {
    values[gl_GlobalInvocationID.x] = undeclared_helper(values[gl_GlobalInvocationID.x]);
}
"""

SOURCES = {
    "metal": ("undeclared.metal", METAL_UNDECLARED),
    "hlsl": ("undeclared.hlsl", HLSL_UNDECLARED),
    "glsl": ("undeclared.comp", GLSL_UNDECLARED),
}


def _program(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def test_undeclared_function_is_reported_by_name():
    calls = undeclared_function_calls(_program(UNDECLARED))
    assert [name for name, _ in calls] == ["undeclaredHelper"]


def test_declared_functions_constructors_and_builtins_are_known():
    program = _program(DECLARED)
    assert {"twice", "pick", "Pair"} <= declared_function_names(program)
    assert undeclared_function_calls(program) == []


@pytest.mark.parametrize("target", TARGETS)
def test_rule_carrying_refuses_an_undeclared_function(target):
    with pytest.raises(UndeclaredFunctionError) as raised:
        apply_rule_carrying(_program(UNDECLARED), "cgl", target)
    error = raised.value
    assert isinstance(error, TranslationDiagnostic)
    assert error.function_name == "undeclaredHelper"
    assert error.code == "project.translate.undeclared-function"
    assert "undeclaredHelper" in str(error)


@pytest.mark.parametrize("target", TARGETS)
def test_translation_refuses_an_undeclared_function(tmp_path, target):
    path = _write(tmp_path, "undeclared.cgl", UNDECLARED)
    with pytest.raises(UndeclaredFunctionError):
        crosstl.translate(path, backend=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("source", sorted(SOURCES))
def test_imported_call_to_an_undeclared_function_is_refused(tmp_path, source, target):
    name, text = SOURCES[source]
    path = _write(tmp_path, name, text)
    with pytest.raises(UndeclaredFunctionError) as raised:
        crosstl.translate(path, backend=target)
    assert raised.value.function_name == "undeclared_helper"


@pytest.mark.parametrize("target", TARGETS)
def test_declared_functions_still_translate(tmp_path, target):
    path = _write(tmp_path, "declared.cgl", DECLARED)
    code = crosstl.translate(path, backend=target)
    assert "undeclared" not in code and code.strip()


DEAD_HELPER = """
shader Dead {
    float unused(float x) {
        return undeclaredHelper(x);
    }

    float used(float x) {
        return x + 1.0;
    }

    compute {
        buffer float values[];

        @compute @workgroup_size(1, 1, 1)
        void main() {
            values[0] = used(1.0);
        }
    }
}
"""

LIVE_HELPER = DEAD_HELPER.replace("values[0] = used(1.0);", "values[0] = unused(1.0);")


def test_calls_in_functions_no_entry_reaches_are_not_checked():
    assert undeclared_function_calls(_program(DEAD_HELPER)) == []


def test_calls_in_functions_an_entry_reaches_are_checked():
    calls = undeclared_function_calls(_program(LIVE_HELPER))
    assert [name for name, _ in calls] == ["undeclaredHelper"]


@pytest.mark.parametrize(
    ("group", "name"),
    [
        (GLSL_FUNCTION_NAMES, "texelFetch"),
        (GLSL_EXTENSION_FUNCTION_NAMES, "nonuniformEXT"),
        (GLSL_EXTENSION_FUNCTION_NAMES, "rayQueryProceedEXT"),
        (HLSL_FUNCTION_NAMES, "saturate"),
        (HLSL_FUNCTION_NAMES, "InterlockedAdd"),
        (METAL_FUNCTION_NAMES, "simd_shuffle"),
        (BUILTIN_FUNCTION_NAMES, "as_type"),
        (CUDA_FUNCTION_NAMES, "sincospif"),
        (CROSSGL_FUNCTION_NAMES, "sizeof"),
        (CROSSGL_FUNCTION_NAMES, "Some"),
        (CROSSGL_FUNCTION_NAMES, "buffer_load"),
        (CROSSGL_FUNCTION_NAMES, "textureCompare"),
        (CROSSGL_FUNCTION_NAMES, "WaveActiveSum"),
        (CROSSGL_FUNCTION_NAMES, "subgroupBallot"),
        (CROSSGL_FUNCTION_NAMES, "workgroupBarrier"),
        (CUDA_FUNCTION_NAMES, "__shfl_down_sync"),
    ],
)
def test_builtin_spellings_need_no_declaration(group, name):
    assert name in group
    assert name in BUILTIN_FUNCTION_NAMES


def test_reserved_names_need_no_declaration():
    program = _program(UNDECLARED.replace("undeclaredHelper", "__builtin_ctz"))
    assert undeclared_function_calls(program) == []


def test_builtin_names_never_include_user_spellings():
    for name in ("undeclaredHelper", "undeclared_helper", "twice", "helper", "main"):
        assert name not in BUILTIN_FUNCTION_NAMES


INVENTORIES = Path(__file__).resolve().parents[1] / "conformance" / "inventory" / "data"


@pytest.mark.parametrize(
    ("language", "group"),
    [
        ("glsl", GLSL_FUNCTION_NAMES),
        ("hlsl", HLSL_FUNCTION_NAMES),
        ("metal", METAL_FUNCTION_NAMES),
    ],
)
def test_language_sets_are_the_inventory_builtin_functions(language, group):
    items = json.loads((INVENTORIES / f"{language}.json").read_text())["items"]
    names = set()
    for item in items:
        if item.get("category") != "builtin-function":
            continue
        name = re.split(r"::|\.", item["name"].split("(")[0])[-1].split("<")[0].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            names.add(name)
    assert group == frozenset(names)


# A generic structure's static member calls a function the program never declares. Live code calls the built-in
# ``min``, so the member is named; the structure itself is named only when live code declares ``Limits<float>``.
TEMPLATE_STATIC = """
shader Templates {
    struct Pair<T> {
        T a;
        T b;
    }

    struct Limits<T> {
        static constant Pair<T> min = makePair(T(0), T(0));
    }

    compute {
        buffer float values[];

        @compute @workgroup_size(1, 1, 1)
        void main() {
            %s
            values[0] = min(values[1], 1.0);
        }
    }
}
"""
UNNAMED_TEMPLATE = TEMPLATE_STATIC % ""
NAMED_TEMPLATE = TEMPLATE_STATIC % "Limits<float> limits;"
INSTANTIATING = ["metal", "directx", "opengl"]

# The Metal importer keeps a partial specialization (``Limits<pair_t<T>>``) as a generic structure whose static
# member constructs ``pair_t<T>`` under its mangled name, which no declaration spells.
METAL_PARTIAL_SPECIALIZATION = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
struct pair_t {
  T a;
  T b;
  constexpr pair_t(T a, T b) thread : a(a), b(b) {}
};

template <typename U>
struct Limits {
  static const constant U min = metal::numeric_limits<U>::lowest();
};

template <typename T>
struct Limits<pair_t<T>> {
  static constexpr constant pair_t<T> min =
      pair_t<T>(-metal::numeric_limits<T>::infinity(), -metal::numeric_limits<T>::infinity());
};

kernel void clamp_low(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  out[i] = min(out[i], 1.0f);
}
"""


@pytest.mark.parametrize("target", INSTANTIATING)
def test_calls_in_generic_structures_no_live_code_names_are_not_checked(target):
    assert undeclared_function_calls(_program(UNNAMED_TEMPLATE), target) == []


@pytest.mark.parametrize("target", INSTANTIATING)
def test_calls_in_generic_structures_live_code_names_are_checked(target):
    calls = undeclared_function_calls(_program(NAMED_TEMPLATE), target)
    assert [name for name, _ in calls] == ["makePair"]


@pytest.mark.parametrize("target", [None, "vulkan", "rust", "slang"])
def test_calls_in_generic_structures_are_checked_for_targets_that_emit_templates(
    target,
):
    calls = undeclared_function_calls(_program(UNNAMED_TEMPLATE), target)
    assert [name for name, _ in calls] == ["makePair"]


@pytest.mark.parametrize("target", INSTANTIATING)
def test_an_unnamed_generic_structure_call_does_not_block_translation(tmp_path, target):
    path = _write(tmp_path, "template.cgl", UNNAMED_TEMPLATE)
    generated = crosstl.translate(path, backend=target)
    assert "makePair" not in generated and generated.strip()


@pytest.mark.parametrize("target", INSTANTIATING + ["vulkan"])
def test_a_named_generic_structure_calling_an_undeclared_function_is_refused(
    tmp_path, target
):
    path = _write(tmp_path, "template.cgl", NAMED_TEMPLATE)
    with pytest.raises(UndeclaredFunctionError, match="makePair"):
        crosstl.translate(path, backend=target)


@pytest.mark.parametrize("target", INSTANTIATING)
def test_an_imported_partial_specialization_does_not_block_translation(
    tmp_path, target
):
    path = _write(tmp_path, "partial.metal", METAL_PARTIAL_SPECIALIZATION)
    generated = crosstl.translate(path, backend=target)
    assert "pair_t_u3cT_u3e" not in generated and generated.strip()
