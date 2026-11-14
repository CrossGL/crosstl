"""A plain name the program never declares is refused before any generator copies it into the target."""

import pytest

import crosstl
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import (
    UndeclaredIdentifierError,
    apply_rule_carrying,
)
from crosstl.translator.semantics.declared_identifiers import (
    deferring_undeclared_identifier_refusals,
    generate_refusing_undeclared_identifiers,
    recorded_undeclared_identifier,
    undeclared_identifier_uses,
)

TARGETS = ["metal", "directx", "opengl", "vulkan"]

HLSL_UNDECLARED = """
RWStructuredBuffer<float> dst : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    dst[0] = r;
}
"""

GLSL_UNDECLARED = """
#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Values { float dst[]; };

void main() {
    dst[0] = r;
}
"""

METAL_UNDECLARED = """
#include <metal_stdlib>
using namespace metal;

kernel void k(device float* dst [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    dst[0] = r;
}
"""

SOURCES = {
    "undeclared.hlsl": HLSL_UNDECLARED,
    "undeclared.comp": GLSL_UNDECLARED,
    "undeclared.metal": METAL_UNDECLARED,
}

HLSL_DECLARED = """
cbuffer Params : register(b0) {
    float scale;
    uint count;
};

static const float BIAS = 0.5;
groupshared float tile[4];
RWStructuredBuffer<float> dst : register(u0);

float twice(float v) {
    return v * 2.0;
}

[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    tile[gi] = float(id.x) * scale;
    GroupMemoryBarrierWithGroupSync();
    float sum = 0.0;
    for (uint i = 0; i < count; ++i) {
        sum += tile[i % 4];
    }
    switch (gi) {
    case 0:
        float first = sum;
        dst[id.x] = twice(first) + BIAS;
        break;
    default:
        dst[id.x] = sum;
        break;
    }
}
"""

GLSL_DECLARED = """
#version 450
layout(local_size_x = 4) in;
layout(std430, binding = 0) buffer Values { float dst[]; };
layout(std140, binding = 1) uniform Params { float scale; uint count; };
shared float tile[4];
const float BIAS = 0.5;

void main() {
    uint gi = gl_LocalInvocationIndex;
    tile[gi] = float(gl_GlobalInvocationID.x) * scale;
    barrier();
    float sum = 0.0;
    for (uint i = 0u; i < count; ++i) {
        sum += tile[i % 4u];
    }
    dst[gl_GlobalInvocationID.x] = sum + BIAS;
}
"""

METAL_DECLARED = """
#include <metal_stdlib>
using namespace metal;

constant float BIAS = 0.5;

struct Params {
    float scale;
    uint count;
};

kernel void k(device float* dst [[buffer(0)]],
              constant Params& params [[buffer(1)]],
              uint id [[thread_position_in_grid]],
              uint gi [[thread_index_in_threadgroup]]) {
    threadgroup float tile[4];
    tile[gi] = float(id) * params.scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0;
    for (uint i = 0; i < params.count; ++i) {
        sum += tile[i % 4];
    }
    dst[id] = sum + BIAS + M_PI_F * 0.0;
}
"""

DECLARED_SOURCES = {
    "declared.hlsl": HLSL_DECLARED,
    "declared.comp": GLSL_DECLARED,
    "declared.metal": METAL_DECLARED,
}


def _program(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _cgl(body, helpers=""):
    return f"""
shader Scopes {{
    enum Mode {{
        First,
        Second
    }}

    const float LIMIT = 4.0;

    {helpers}

    compute {{
        buffer float values[];

        @compute @workgroup_size(1, 1, 1)
        void main() {{
            {body}
        }}
    }}
}}
"""


def _names(source):
    return [name for name, _, _ in undeclared_identifier_uses(_program(source))]


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


@pytest.mark.parametrize("name", sorted(SOURCES))
@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_importers_refuse_an_undeclared_name(tmp_path, name, target):
    path = _write(tmp_path, name, SOURCES[name])
    with pytest.raises(UndeclaredIdentifierError) as caught:
        crosstl.translate(path, backend=target)
    assert caught.value.identifier == "r"
    assert caught.value.code.endswith("undeclared-identifier")
    assert "'r'" in str(caught.value)


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_spirv_refuses_an_undeclared_name_with_its_own_diagnostic(tmp_path, name):
    # The SPIR-V generator refuses the unknown variable itself, with a typed diagnostic that takes precedence.
    path = _write(tmp_path, name, SOURCES[name])
    with pytest.raises(TranslationDiagnostic) as caught:
        crosstl.translate(path, backend="vulkan")
    assert "variable r" in str(caught.value)


@pytest.mark.parametrize("name", sorted(DECLARED_SOURCES))
@pytest.mark.parametrize("target", TARGETS)
def test_declared_names_still_translate(tmp_path, name, target):
    path = _write(tmp_path, name, DECLARED_SOURCES[name])
    assert crosstl.translate(path, backend=target)


@pytest.mark.parametrize("target", TARGETS)
def test_rule_carrying_records_the_refusal_for_generation(target):
    program = _program(_cgl("values[0] = missing;"))
    apply_rule_carrying(program, "cgl", target)
    error = recorded_undeclared_identifier(program)
    assert isinstance(error, UndeclaredIdentifierError)
    assert (error.identifier, error.function_name) == ("missing", "main")
    with pytest.raises(UndeclaredIdentifierError):
        generate_refusing_undeclared_identifiers(program, lambda: "code")


def test_a_typed_generator_refusal_takes_precedence():
    program = _program(_cgl("values[0] = missing;"))
    apply_rule_carrying(program, "cgl", "metal")

    def refuse():
        raise TranslationDiagnostic("construct has no lowering", target="metal")

    with pytest.raises(TranslationDiagnostic) as caught:
        generate_refusing_undeclared_identifiers(program, refuse)
    assert not isinstance(caught.value, UndeclaredIdentifierError)


def test_an_untyped_generator_failure_becomes_the_recorded_refusal():
    program = _program(_cgl("values[0] = missing;"))
    apply_rule_carrying(program, "cgl", "metal")

    def fail():
        raise KeyError("missing")

    with pytest.raises(UndeclaredIdentifierError) as caught:
        generate_refusing_undeclared_identifiers(program, fail)
    assert isinstance(caught.value.__cause__, KeyError)


def test_a_deferring_caller_receives_the_output_and_the_refusal():
    program = _program(_cgl("values[0] = missing;"))
    apply_rule_carrying(program, "cgl", "metal")
    with deferring_undeclared_identifier_refusals() as deferred:
        assert (
            generate_refusing_undeclared_identifiers(program, lambda: "code") == "code"
        )
    assert [error.identifier for error in deferred] == ["missing"]


def test_a_declared_program_records_nothing():
    program = _program(_cgl("values[0] = LIMIT;"))
    apply_rule_carrying(program, "cgl", "metal")
    assert recorded_undeclared_identifier(program) is None
    assert generate_refusing_undeclared_identifiers(program, lambda: "code") == "code"


def test_copies_of_the_program_keep_the_record():
    import copy

    program = _program(_cgl("values[0] = missing;"))
    apply_rule_carrying(program, "cgl", "metal")
    assert (
        recorded_undeclared_identifier(copy.deepcopy(program)).identifier == "missing"
    )


def test_a_block_local_is_out_of_scope_after_its_block():
    body = """
            if (values[0] > 0.0) {
                float t = 1.0;
                values[1] = t;
            }
            values[0] = t;
    """
    assert _names(_cgl(body)) == ["t"]


def test_a_local_is_out_of_scope_before_its_declaration():
    body = """
            values[0] = later;
            float later = 2.0;
    """
    assert _names(_cgl(body)) == ["later"]


def test_a_loop_variable_is_out_of_scope_after_the_loop():
    body = """
            for (int i = 0; i < 4; i++) {
                values[i] = float(i);
            }
            values[0] = float(i);
    """
    assert _names(_cgl(body)) == ["i"]


def test_a_local_declared_in_one_case_is_visible_in_the_next():
    body = """
            int k = int(values[0]);
            switch (k) {
                case 0:
                    float shared_value = 1.0;
                    values[0] = shared_value;
                    break;
                default:
                    shared_value = 2.0;
                    values[1] = shared_value;
                    break;
            }
    """
    assert _names(_cgl(body)) == []


def test_module_names_builtins_and_constants_need_no_local_declaration():
    body = """
            Mode m = Mode::First;
            float v = LIMIT + M_PI + FLT_MAX * 0.0;
            values[gl_GlobalInvocationID.x] = helper(v) + (true ? 1.0 : 0.0);
    """
    helpers = """
    float helper(float x) {
        return x * 2.0;
    }
    """
    assert _names(_cgl(body, helpers)) == []


def test_a_helper_no_entry_reaches_is_not_checked(tmp_path):
    helpers = """
    float unused(float x) {
        return x * missing;
    }
    """
    source = _cgl("values[0] = 1.0;", helpers)
    assert _names(source) == []
    path = _write(tmp_path, "dead.cgl", source)
    assert "values" in crosstl.translate(path, backend="opengl")


def test_a_helper_an_entry_calls_is_checked():
    helpers = """
    float scaled(float x) {
        return x * missing;
    }
    """
    uses = undeclared_identifier_uses(
        _program(_cgl("values[0] = scaled(1.0);", helpers))
    )
    assert [(name, function) for name, _, function in uses] == [("missing", "scaled")]
