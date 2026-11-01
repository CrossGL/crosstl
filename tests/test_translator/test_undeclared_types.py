"""A type name the program never declares is refused before any generator copies it into the target."""

import pytest

import crosstl
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import UndeclaredTypeError, apply_rule_carrying
from crosstl.translator.semantics.builtin_type_names import BUILTIN_TYPE_NAMES
from crosstl.translator.semantics.declared_types import (
    declared_type_names,
    undeclared_type_uses,
)

TARGETS = ["metal", "directx", "opengl", "vulkan"]

UNDECLARED = """
shader Undeclared {
    compute {
        buffer float values[];

        @compute @workgroup_size(1, 1, 1)
        void main() {
            Missing m;
            values[0] = 1.0;
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

    enum Mode {
        First,
        Second
    }

    compute {
        buffer float values[];

        @compute @workgroup_size(1, 1, 1)
        void main() {
            Pair p;
            p.a = 1.0;
            p.b = 2.0;
            values[0] = p.a + p.b;
        }
    }
}
"""


def _program(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def test_undeclared_type_is_reported_by_name():
    uses = undeclared_type_uses(_program(UNDECLARED))
    assert [name for name, _ in uses] == ["Missing"]


def test_declared_structs_and_enums_are_known():
    program = _program(DECLARED)
    assert {"Pair", "Mode"} <= declared_type_names(program)
    assert undeclared_type_uses(program) == []


@pytest.mark.parametrize("target", TARGETS)
def test_rule_carrying_refuses_an_undeclared_type(target):
    with pytest.raises(UndeclaredTypeError) as raised:
        apply_rule_carrying(_program(UNDECLARED), "cgl", target)
    error = raised.value
    assert isinstance(error, TranslationDiagnostic)
    assert error.type_name == "Missing"
    assert error.code.endswith("undeclared-type")
    assert "Missing" in str(error)


@pytest.mark.parametrize("target", TARGETS)
def test_translation_refuses_an_undeclared_type(tmp_path, target):
    path = _write(tmp_path, "undeclared.cgl", UNDECLARED)
    with pytest.raises(UndeclaredTypeError):
        crosstl.translate(path, backend=target)


@pytest.mark.parametrize("target", TARGETS)
def test_declared_types_still_translate(tmp_path, target):
    path = _write(tmp_path, "declared.cgl", DECLARED)
    assert "Pair" in crosstl.translate(path, backend=target)


@pytest.mark.parametrize(
    "name",
    [
        "float4",
        "uint3x2",
        "half2",
        "vec3",
        "i64vec2",
        "f16mat4",
        "packed_float3",
        "sampler2DShadow",
        "usamplerBuffer",
        "texture2DArray",
        "uimage2D",
        "Texture2D",
        "RWByteAddressBuffer",
        "RayDesc",
        "SamplerComparisonState",
        "depth2d_array",
        "instance_acceleration_structure",
        "uint64_t",
        "min16float",
    ],
)
def test_builtin_spellings_need_no_declaration(name):
    assert name in BUILTIN_TYPE_NAMES


def test_builtin_names_never_include_user_spellings():
    for name in ("Missing", "Pair", "Foo", "MyTexture", "Word"):
        assert name not in BUILTIN_TYPE_NAMES


DEAD_HELPER = """
shader Dead {
    float unused(float x) {
        Missing m;
        return x;
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

LIBRARY = """
shader Library {
    float helper(Missing m) {
        return 1.0;
    }
}
"""


def test_functions_no_entry_reaches_are_not_checked():
    assert undeclared_type_uses(_program(DEAD_HELPER)) == []


def test_functions_an_entry_calls_are_checked():
    uses = undeclared_type_uses(_program(LIVE_HELPER))
    assert [name for name, _ in uses] == ["Missing"]


def test_every_function_of_a_program_without_entries_is_checked():
    uses = undeclared_type_uses(_program(LIBRARY))
    assert [name for name, _ in uses] == ["Missing"]


def test_a_dead_helper_does_not_block_translation(tmp_path):
    path = _write(tmp_path, "dead.cgl", DEAD_HELPER)
    assert "values" in crosstl.translate(path, backend="opengl")


def test_crossgl_trait_and_payload_types_are_built_in():
    assert {"Self", "str"} <= BUILTIN_TYPE_NAMES


BOUNDED = """
shader Bounded {
    generic<T: Numeric + Copy> fn twice(a: T) -> T {
        return a + a;
    }

    compute {
        buffer float values[];

        @compute @workgroup_size(1, 1, 1)
        void main() {
            values[0] = twice(1.0);
        }
    }
}
"""


def test_trait_bounds_are_not_type_uses():
    assert undeclared_type_uses(_program(BOUNDED)) == []


FEEDBACK_TEXTURE = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer uint* outp;
        void main() {
            %s obj;
            outp[gl_GlobalInvocationID.x] = 0u;
        }
    }
}
"""
FEEDBACK_KINDS = [
    "FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP>",
    "FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIP_REGION_USED>",
    "feedbackTexture2D<SAMPLER_FEEDBACK_MIP_REGION_USED>",
]


@pytest.mark.parametrize("feedback_type", FEEDBACK_KINDS)
def test_sampler_feedback_kinds_are_values_not_type_uses(feedback_type):
    assert undeclared_type_uses(_program(FEEDBACK_TEXTURE % feedback_type)) == []


@pytest.mark.parametrize(
    "feedback_type", ["FeedbackTexture2D<Missing>", "FeedbackTexture2DArray<Missing>"]
)
def test_other_feedback_texture_arguments_are_type_uses(feedback_type):
    uses = undeclared_type_uses(_program(FEEDBACK_TEXTURE % feedback_type))
    assert [name for name, _node in uses] == ["Missing"]


def test_a_feedback_texture_translates_with_its_kind(tmp_path):
    source = FEEDBACK_TEXTURE % "FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP>"
    path = _write(tmp_path, "feedback.cgl", source)
    generated = crosstl.translate(path, backend="directx")
    assert "FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> obj;" in generated


RAY_QUERIES = """
shader RayQueries {
    compute {
        void main() {
            RayQuery<RAY_FLAG_FORCE_OPAQUE> opaque;
            RayQuery<RAY_FLAG_CULL_OPAQUE | RAY_FLAG_SKIP_TRIANGLES> culled;
            StructuredBuffer<Missing> values;
        }
    }
}
"""


def test_ray_query_flags_are_values_not_type_uses():
    # The flags of RayQuery<FLAGS> are values; other template arguments are still types.
    uses = undeclared_type_uses(_program(RAY_QUERIES))
    assert [name for name, _node in uses] == ["Missing"]


# An importer keeps a generic structure next to the specialization it materialized; the template's member alias
# (``Alias``) has no CrossGL declaration.
TEMPLATE_BODY = """
shader Templates {
    struct Holder<T> {
        T value;
        Alias scratch;
    }

    struct Holder_float {
        float value;
    }

    compute {
        buffer float values[];

        @compute @workgroup_size(1, 1, 1)
        void main() {
            %s h;
            h.value = 2.0;
            values[0] = h.value;
        }
    }
}
"""
UNNAMED_TEMPLATE = TEMPLATE_BODY % "Holder_float"
NAMED_TEMPLATE = TEMPLATE_BODY % "Holder<float>"
INSTANTIATING = ["metal", "directx", "opengl"]
# Generic structures reached only through another generic structure live code names are read too.
NESTED_TEMPLATE = """
shader Nested {
    struct Inner<T> {
        Alias scratch;
    }

    struct Outer<T> {
        Inner<T> inner;
        T value;
    }

    compute {
        buffer float values[];

        @compute @workgroup_size(1, 1, 1)
        void main() {
            Outer<float> o;
            o.value = 1.0;
            values[0] = o.value;
        }
    }
}
"""


@pytest.mark.parametrize("target", INSTANTIATING)
def test_generic_structures_no_live_code_names_are_not_checked(target):
    assert undeclared_type_uses(_program(UNNAMED_TEMPLATE), target) == []


@pytest.mark.parametrize("target", INSTANTIATING)
def test_generic_structures_live_code_names_are_checked(target):
    for source in (NAMED_TEMPLATE, NESTED_TEMPLATE):
        uses = undeclared_type_uses(_program(source), target)
        assert [name for name, _node in uses] == ["Alias"]


@pytest.mark.parametrize("target", [None, "vulkan", "rust", "slang"])
def test_generic_structures_are_checked_for_targets_that_emit_templates(target):
    uses = undeclared_type_uses(_program(UNNAMED_TEMPLATE), target)
    assert [name for name, _node in uses] == ["Alias"]


@pytest.mark.parametrize("target", INSTANTIATING)
def test_an_unnamed_generic_structure_does_not_block_translation(tmp_path, target):
    path = _write(tmp_path, "template.cgl", UNNAMED_TEMPLATE)
    generated = crosstl.translate(path, backend=target)
    assert "Holder_float" in generated
    assert "Alias" not in generated


@pytest.mark.parametrize("target", INSTANTIATING + ["vulkan"])
def test_a_named_generic_structure_with_an_undeclared_member_is_refused(
    tmp_path, target
):
    path = _write(tmp_path, "template.cgl", NAMED_TEMPLATE)
    with pytest.raises(UndeclaredTypeError, match="Alias"):
        crosstl.translate(path, backend=target)


CALL_TYPE_ARGUMENT = """
shader P {
    T pick<T>(T x) { return x; }
    compute {
        buffer float* outp;
        void main() { outp[0] = pick<foo>(outp[1]); }
    }
}
"""

DECLARED_CALL_TYPE_ARGUMENTS = """
shader P {
    struct Pair { float a; float b; };
    T pick<T>(T x) { return x; }
    U second<T, U>(T x, U y) { return pick<U>(y); }
    compute {
        buffer float* outp;
        void main() {
            Pair p;
            p.a = pick<float>(outp[1]);
            p.b = second<Pair, float>(pick<Pair>(p), 2.0);
            outp[0] = p.a + p.b;
        }
    }
}
"""


def test_an_undeclared_call_type_argument_is_reported_by_name():
    uses = undeclared_type_uses(_program(CALL_TYPE_ARGUMENT))
    assert [name for name, _ in uses] == ["foo"]


def test_declared_call_type_arguments_are_known():
    assert undeclared_type_uses(_program(DECLARED_CALL_TYPE_ARGUMENTS)) == []


@pytest.mark.parametrize("target", TARGETS + ["rust"])
def test_translation_refuses_an_undeclared_call_type_argument(tmp_path, target):
    path = _write(tmp_path, "call.cgl", CALL_TYPE_ARGUMENT)
    with pytest.raises(UndeclaredTypeError, match="'foo'"):
        crosstl.translate(path, backend=target)
