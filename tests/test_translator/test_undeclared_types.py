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
