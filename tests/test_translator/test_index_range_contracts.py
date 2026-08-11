import shutil
import subprocess

import pytest

import crosstl
from crosstl.project import (
    load_project_config,
    translate_project,
    validate_project_report,
)
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLIndexTypeError,
)
from crosstl.translator.codegen.index_range_contracts import (
    INDEX_RANGE_STATIC,
    OPENGL_INDEX_PROFILE,
    WEBGL_INDEX_PROFILE,
    IntegerRange,
    decide_index_narrowing,
)
from crosstl.translator.codegen.webgl_codegen import (
    WebGLCodeGen,
    WebGLIndexTypeError,
)


def _parse(shader):
    return crosstl.translator.parse(shader)


def _validate_compute_if_available(source, tmp_path, name):
    validator = shutil.which("glslangValidator")
    if validator is None:
        return False
    path = tmp_path / f"{name}.comp"
    path.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [validator, "-S", "comp", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True


def _validate_compute_spirv_if_available(source, tmp_path, name):
    compiler = shutil.which("glslangValidator")
    validator = shutil.which("spirv-val")
    if compiler is None or validator is None:
        return False
    source_path = tmp_path / f"{name}.comp"
    binary_path = tmp_path / f"{name}.spv"
    source_path.write_text(source, encoding="utf-8")
    compilation = subprocess.run(
        [
            compiler,
            "--target-env",
            "opengl",
            "--target-env",
            "spirv1.3",
            "-S",
            "comp",
            str(source_path),
            "-o",
            str(binary_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr
    validation = subprocess.run(
        [validator, "--target-env", "spv1.3", str(binary_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert validation.returncode == 0, validation.stdout + validation.stderr
    return True


def test_profiles_publish_backend_neutral_32_bit_scalar_contracts():
    scalar_contracts = [
        (item.name, item.signed, item.bits)
        for item in OPENGL_INDEX_PROFILE.scalar_types
    ]
    assert scalar_contracts == [
        ("int", True, 32),
        ("uint", False, 32),
    ]
    assert WEBGL_INDEX_PROFILE.scalar_types == OPENGL_INDEX_PROFILE.scalar_types

    accepted = decide_index_narrowing(
        source_signed=False,
        source_bits=64,
        source_width=1,
        profile=OPENGL_INDEX_PROFILE,
        value_range=IntegerRange(0, 15, INDEX_RANGE_STATIC),
    )
    rejected = decide_index_narrowing(
        source_signed=False,
        source_bits=64,
        source_width=1,
        profile=OPENGL_INDEX_PROFILE,
        value_range=None,
    )
    assert (accepted.action, accepted.target_type.name) == ("convert", "uint")
    assert (rejected.action, rejected.reason) == ("reject", "index-range-unproven")


def test_opengl_fails_closed_for_unproven_wide_runtime_index():
    shader = """
    shader UnprovenWideIndex {
        StructuredBuffer<uint> values @ binding(0);

        uint readValue(uint64_t index) {
            return buffer_load(values, index);
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    diagnostic = exc_info.value
    assert diagnostic.index_type == "uint64_t"
    assert diagnostic.target_index_type == "uint"
    assert diagnostic.indexed_value == "values"
    assert diagnostic.reason == "index-range-unproven"


def test_opengl_rejects_negative_wide_constant_before_narrowing():
    shader = """
    shader NegativeWideIndex {
        uint readValue() {
            uint values[4];
            return values[int64_t(-1)];
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    diagnostic = exc_info.value
    assert diagnostic.index_type == "int64_t"
    assert diagnostic.target_index_type == "int"
    assert diagnostic.indexed_value == "values"
    assert diagnostic.reason == "negative-index"


def test_opengl_normalizes_fixed_arrays_vectors_matrices_and_bounded_values(tmp_path):
    shader = """
    shader FixedIndexKinds {
        RWStructuredBuffer<uint> output @ binding(0);

        compute {
            [numthreads(1, 1, 1)]
            void main() {
                uint values[4];
                vec4 vectorValue = vec4(1.0);
                mat4 matrixValue = mat4(1.0);
                values[uint64_t(7u) & uint64_t(3u)] = uint(vectorValue[uint64_t(2u)]);
                uint result = values[uint64_t(1u)];
                result += uint(matrixValue[int64_t(2)][0].x);
                buffer_store(output, 0, result);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(_parse(shader))

    assert "values[uint((uint64_t(7u) & uint64_t(3u)))]" in generated
    assert "vectorValue[2u]" in generated
    assert "values[1u]" in generated
    assert "matrixValue[2][0]" in generated
    _validate_compute_if_available(generated, tmp_path, "fixed_index_kinds")


def test_opengl_consumes_static_scalar_and_vector_component_flow_ranges(tmp_path):
    shader = """
    shader StaticIndexProofs {
        samplerBuffer texels @ binding(0);
        StructuredBuffer<uint> source @ binding(1);
        RWStructuredBuffer<uint> resources[3] @ binding(2);
        RWStructuredBuffer<uint> output @ binding(5);

        compute {
            [numthreads(4, 1, 1)]
            void main() {
                uint64_t bounded =
                    uint64_t(gl_LocalInvocationID.x) & uint64_t(3u);
                u64vec2 indices = u64vec2(bounded, uint64_t(2u));
                uint localValues[4];
                vec4 vectorValue = vec4(1.0);
                mat4 matrixValue = mat4(1.0);
                localValues[bounded] = buffer_load(source, indices.x);
                uint value = localValues[indices.x];
                value += uint(vectorValue[indices.y]);
                value += uint(matrixValue[int64_t(bounded)][0].x);
                value += uint(texelFetch(texels, int64_t(bounded)).x);
                buffer_store(resources[indices.y], bounded, value);
                buffer_store(output, bounded, value);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(_parse(shader))

    assert "localValues[uint(bounded)]" in generated
    assert "source[uint(indices.x)]" in generated
    assert "vectorValue[uint(indices.y)]" in generated
    assert "matrixValue[int(int64_t(bounded))][0]" in generated
    assert "texelFetch(texels, int(int64_t(bounded)))" in generated
    assert "resources[uint(indices.y)].data[uint(bounded)]" in generated
    assert generated.count("uint(bounded)") == 3
    _validate_compute_spirv_if_available(
        generated,
        tmp_path,
        "static_scalar_and_component_index_proofs",
    )


def test_opengl_consumes_configured_ranges_across_indexed_value_kinds(tmp_path):
    shader = """
    shader ConfiguredIndexProofs {
        samplerBuffer texels @ binding(0);
        StructuredBuffer<uint> source @ binding(1);
        RWStructuredBuffer<uint> resources[3] @ binding(2);
        RWStructuredBuffer<uint> output @ binding(5);

        uint consume(uint64_t scalarIndex, i64vec2 componentIndices) {
            uint localValues[4];
            vec4 vectorValue = vec4(1.0);
            mat4 matrixValue = mat4(1.0);
            localValues[scalarIndex] = buffer_load(source, componentIndices.x);
            uint value = localValues[componentIndices.x];
            value += uint(vectorValue[componentIndices.y]);
            value += uint(matrixValue[componentIndices.x][0].x);
            value += uint(texelFetch(texels, componentIndices.x).x);
            buffer_store(resources[componentIndices.y], scalarIndex, value);
            return value;
        }

        compute {
            [numthreads(1, 1, 1)]
            void main() {
                uint64_t scalarIndex = uint64_t(gl_GlobalInvocationID.x);
                i64vec2 componentIndices = i64vec2(
                    int64_t(gl_GlobalInvocationID.x),
                    int64_t(gl_GlobalInvocationID.y));
                buffer_store(output, scalarIndex, consume(scalarIndex, componentIndices));
            }
        }
    }
    """

    generated = (
        GLSLCodeGen()
        .set_index_range_assertions(
            [
                {
                    "expression": "scalarIndex",
                    "minimum": 0,
                    "maximum": 3,
                },
                {
                    "expression": "componentIndices.x",
                    "function": "consume",
                    "minimum": 0,
                    "maximum": 3,
                },
                {
                    "expression": "componentIndices.y",
                    "function": "consume",
                    "minimum": 0,
                    "maximum": 2,
                },
            ]
        )
        .generate(_parse(shader))
    )

    assert "localValues[uint(scalarIndex)]" in generated
    assert "source[int(componentIndices.x)]" in generated
    assert "vectorValue[int(componentIndices.y)]" in generated
    assert "matrixValue[int(componentIndices.x)][0]" in generated
    assert "texelFetch(texels, int(componentIndices.x))" in generated
    assert "resources[int(componentIndices.y)].data[uint(scalarIndex)]" in generated
    _validate_compute_spirv_if_available(
        generated,
        tmp_path,
        "configured_index_proofs",
    )


def test_opengl_consumes_signed_loop_index_range_once_per_access(tmp_path):
    shader = """
    shader SignedLoopIndexProof {
        RWStructuredBuffer<uint> output @ binding(0);

        compute {
            [numthreads(1, 1, 1)]
            void main() {
                for (int64_t index = int64_t(0); index < int64_t(4); index++) {
                    buffer_store(output, index, uint(index));
                }
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(_parse(shader))

    assert "output_[int(index)] = uint(index);" in generated
    assert generated.count("[int(index)]") == 1
    _validate_compute_spirv_if_available(
        generated,
        tmp_path,
        "signed_loop_index_proof",
    )


def test_opengl_preserves_side_effecting_component_and_resource_indices(tmp_path):
    shader = """
    shader SideEffectingIndexProofs {
        RWStructuredBuffer<uint> resources[3] @ binding(0);
        RWStructuredBuffer<uint> output @ binding(3);

        u64vec2 nextIndices(inout uint64_t cursor) {
            uint64_t current = cursor++;
            return u64vec2(current, current);
        }

        compute {
            [numthreads(1, 1, 1)]
            void main() {
                uint64_t cursor = uint64_t(0u);
                uint value = buffer_load(resources[0], nextIndices(cursor).x);
                buffer_store(resources[nextIndices(cursor).y], 0, value);
                buffer_store(output, 0, uint(cursor));
            }
        }
    }
    """

    generated = (
        GLSLCodeGen()
        .set_index_range_assertions(
            [
                {
                    "expression": "nextIndices(cursor).x",
                    "function": "main",
                    "minimum": 0,
                    "maximum": 2,
                },
                {
                    "expression": "nextIndices(cursor).y",
                    "function": "main",
                    "minimum": 0,
                    "maximum": 2,
                },
            ]
        )
        .generate(_parse(shader))
    )

    assert generated.count("nextIndices(cursor).x") == 1
    assert generated.count("nextIndices(cursor).y") == 1
    assert "resources[uint(nextIndices(cursor).y)].data[0] = value;" in generated
    _validate_compute_spirv_if_available(
        generated,
        tmp_path,
        "side_effecting_index_proofs",
    )


@pytest.mark.parametrize(
    ("declaration", "index", "reason"),
    [
        (
            "int64_t index = -int64_t(gl_LocalInvocationID.x & 1u);",
            "index",
            "negative-index",
        ),
        (
            "uint64_t index = uint64_t(gl_LocalInvocationID.x) & uint64_t(7u);",
            "index",
            "index-range-out-of-bounds",
        ),
        (
            "u64vec2 indices = u64vec2(uint64_t(0u), "
            "uint64_t(gl_LocalInvocationID.x) & uint64_t(7u));",
            "indices.y",
            "index-range-out-of-bounds",
        ),
    ],
)
def test_opengl_rejects_static_flow_ranges_outside_index_contract(
    declaration,
    index,
    reason,
):
    shader = f"""
    shader InvalidStaticIndexProof {{
        uint readValue() {{
            uint values[4];
            {declaration}
            return values[{index}];
        }}
    }}
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    diagnostic = exc_info.value
    assert diagnostic.reason == reason
    assert diagnostic.range_status == "out-of-range"


@pytest.mark.parametrize("container", ["block", "loop"])
def test_opengl_invalidates_static_index_ranges_after_mutation(container):
    mutation = "index = uint64_t(runtime);"
    if container == "block":
        mutation = f"{{ {mutation} }}"
    else:
        mutation = f"for (uint i = 0u; i < 2u; i++) {{ {mutation} }}"
    shader = f"""
    shader MutatedStaticIndexProof {{
        RWStructuredBuffer<uint> output @ binding(0);

        uint readValue(uint runtime) {{
            uint64_t index = uint64_t(runtime) & uint64_t(3u);
            {mutation}
            return buffer_load(output, index);
        }}
    }}
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    assert exc_info.value.reason == "index-range-unproven"


def test_opengl_propagates_mutation_before_block_local_shadowing():
    shader = """
    shader BlockShadowIndexMutation {
        RWStructuredBuffer<uint> output @ binding(0);

        uint readValue(uint runtime) {
            uint64_t index = uint64_t(runtime) & uint64_t(3u);
            {
                index = uint64_t(runtime);
                uint64_t index = uint64_t(0u);
            }
            return buffer_load(output, index);
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    assert exc_info.value.reason == "index-range-unproven"


def test_opengl_preserves_outer_range_across_static_branch_shadowing():
    shader = """
    shader StaticBranchShadowing {
        RWStructuredBuffer<uint> output @ binding(0);

        uint readValue(uint runtime) {
            uint64_t index = uint64_t(runtime);
            if (true) {
                uint64_t index = uint64_t(0u);
            }
            return buffer_load(output, index);
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    assert exc_info.value.reason == "index-range-unproven"


def test_opengl_invalidates_static_index_range_after_vector_swizzle_write():
    shader = """
    shader SwizzleIndexMutation {
        RWStructuredBuffer<uint> output @ binding(0);

        uint readValue(uint runtime) {
            u64vec2 indices = u64vec2(
                uint64_t(runtime) & uint64_t(3u),
                uint64_t(0u));
            indices.xy = u64vec2(uint64_t(runtime), uint64_t(0u));
            return buffer_load(output, indices.x);
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    assert exc_info.value.reason == "index-range-unproven"


def test_opengl_invalidates_vector_component_proof_after_indexed_write():
    shader = """
    shader IndexedComponentMutation {
        RWStructuredBuffer<uint> output @ binding(0);

        uint readValue(uint runtime) {
            u64vec2 indices = u64vec2(uint64_t(0u));
            indices[0] = uint64_t(runtime);
            return buffer_load(output, indices.x);
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    assert exc_info.value.reason == "index-range-unproven"


def test_opengl_tracks_vector_component_compound_assignment_range():
    shader = """
    shader CompoundComponentIndexMutation {
        uint readValue() {
            uint values[4];
            u64vec2 indices = u64vec2(uint64_t(3u), uint64_t(0u));
            indices.x += uint64_t(3u);
            return values[indices.x];
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    assert exc_info.value.reason == "constant-index-out-of-range"


def test_opengl_invalidates_static_index_range_across_switch_cases():
    shader = """
    shader SwitchIndexMutation {
        RWStructuredBuffer<uint> output @ binding(0);

        uint readValue(uint runtime) {
            uint64_t index = uint64_t(runtime) & uint64_t(3u);
            switch (runtime) {
                case 0u:
                    index = uint64_t(runtime);
                    break;
                default:
                    index = uint64_t(0u);
                    break;
            }
            return buffer_load(output, index);
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    assert exc_info.value.reason == "index-range-unproven"


def test_opengl_does_not_apply_do_while_condition_proof_to_first_iteration():
    shader = """
    shader DoWhileIndexMutation {
        RWStructuredBuffer<uint> output @ binding(0);

        uint readValue(uint runtime) {
            uint64_t index = uint64_t(runtime);
            do {
                return buffer_load(output, index);
            } while (bool(index = uint64_t(0u)));
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    assert exc_info.value.reason == "index-range-unproven"


def test_opengl_rejects_canonical_loop_index_mutated_in_body():
    shader = """
    shader MutatedCanonicalLoopIndex {
        uint readValue() {
            uint values[4];
            uint result = 0u;
            for (int64_t index = int64_t(0); index < int64_t(4); index++) {
                result += values[index];
                index = -int64_t(2);
            }
            return result;
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    assert exc_info.value.reason == "index-range-unproven"


def test_opengl_rejects_narrowing_cast_range_that_wraps_negative():
    shader = """
    shader NarrowingCastIndex {
        uint readValue() {
            uint values[256];
            int8_t narrow = int8_t(255);
            int64_t index = int64_t(narrow);
            return values[index];
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    assert exc_info.value.reason == "index-range-unproven"


def test_opengl_propagates_side_effects_through_else_if_fallthrough():
    shader = """
    shader ElseIfIndexMutation {
        RWStructuredBuffer<uint> output @ binding(0);

        bool replaceIndex(inout uint64_t index, uint runtime) {
            index = uint64_t(runtime);
            return false;
        }

        uint readValue(uint runtime) {
            uint64_t index = uint64_t(runtime) & uint64_t(3u);
            if (false) {
            } else if (replaceIndex(index, runtime)) {
                index = uint64_t(0u);
            }
            return buffer_load(output, index);
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    assert exc_info.value.reason == "index-range-unproven"


@pytest.mark.parametrize(
    "loop",
    [
        "for (index = uint64_t(runtime); false;) {}",
        "while (index++ < uint64_t(1u)) {}",
    ],
)
def test_opengl_invalidates_static_index_ranges_after_loop_header_mutation(loop):
    shader = f"""
    shader LoopHeaderIndexMutation {{
        RWStructuredBuffer<uint> output @ binding(0);

        uint readValue(uint runtime) {{
            uint64_t index = uint64_t(runtime) & uint64_t(3u);
            {loop}
            return buffer_load(output, index);
        }}
    }}
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    assert exc_info.value.reason == "index-range-unproven"


def test_opengl_invalidates_static_index_range_after_inout_call():
    shader = """
    shader InOutIndexMutation {
        RWStructuredBuffer<uint> output @ binding(0);

        void replaceIndex(inout uint64_t index, uint runtime) {
            index = uint64_t(runtime);
        }

        uint readValue(uint runtime) {
            uint64_t index = uint64_t(runtime) & uint64_t(3u);
            replaceIndex(index, runtime);
            return buffer_load(output, index);
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    assert exc_info.value.reason == "index-range-unproven"


def test_opengl_invalidates_static_index_range_after_loop_inout_call():
    shader = """
    shader LoopInOutIndexMutation {
        RWStructuredBuffer<uint> output @ binding(0);

        void replaceIndex(inout uint64_t index, uint runtime) {
            index = uint64_t(runtime);
        }

        uint readValue(uint runtime) {
            uint64_t index = uint64_t(runtime) & uint64_t(3u);
            for (uint i = 0u; i < 2u; i++) {
                replaceIndex(index, runtime);
            }
            return buffer_load(output, index);
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    assert exc_info.value.reason == "index-range-unproven"


def test_opengl_asserted_runtime_ssbo_index_is_converted_and_evaluated_once(tmp_path):
    shader = """
    shader RuntimeIndex {
        StructuredBuffer<uint> values @ binding(0);
        RWStructuredBuffer<uint> output @ binding(1);

        compute {
            [numthreads(1, 1, 1)]
            void main() {
                uint64_t index = uint64_t(0u);
                buffer_store(output, 0, buffer_load(values, index++));
            }
        }
    }
    """
    generated = (
        GLSLCodeGen()
        .set_index_range_assertions(
            [{"expression": "index++", "function": "main", "minimum": 0, "maximum": 31}]
        )
        .generate(_parse(shader))
    )

    assert generated.count("index++") == 1
    assert "values[uint((index++))]" in generated
    _validate_compute_spirv_if_available(generated, tmp_path, "runtime_index_once")


def test_opengl_normalizes_texture_buffer_and_nested_resource_alias_indices(tmp_path):
    texture_shader = """
    shader TextureBufferIndex {
        samplerBuffer values @ binding(0);
        vec4 readValue() { return texelFetch(values, uint64_t(2u)); }
    }
    """
    texture_generated = GLSLCodeGen().generate(_parse(texture_shader))
    assert "texelFetch(values, 2)" in texture_generated
    _validate_compute_if_available(texture_generated, tmp_path, "texture_buffer_index")

    runtime_texture_shader = """
    shader RuntimeTextureBufferIndex {
        samplerBuffer values @ binding(0);
        vec4 readValue(uint64_t index) { return texelFetch(values, index); }
    }
    """
    runtime_texture_generated = (
        GLSLCodeGen()
        .set_index_range_assertions(
            [{"expression": "index", "minimum": 0, "maximum": 31}]
        )
        .generate(_parse(runtime_texture_shader))
    )
    assert "texelFetch(values, int(index))" in runtime_texture_generated
    _validate_compute_if_available(
        runtime_texture_generated,
        tmp_path,
        "runtime_texture_buffer_index",
    )

    alias_shader = """
    shader NestedResourceAliasIndex {
        RWStructuredBuffer<uint> values[2] @ binding(0);
        uint readLeaf(RWStructuredBuffer<uint> leaf[], uint64_t which) {
            return buffer_load(leaf[which], 0);
        }
        uint readNested(RWStructuredBuffer<uint> nested[], uint64_t which) {
            return readLeaf(nested, which);
        }
        uint run(uint64_t which) { return readNested(values, which); }
    }
    """
    generated = (
        GLSLCodeGen()
        .set_index_range_assertions(
            [{"expression": "which", "minimum": 0, "maximum": 1}]
        )
        .generate(_parse(alias_shader))
    )
    assert "uint(which)" in generated
    assert "buffer_load" not in generated


def test_opengl_normalizes_component_indices_through_nested_resource_aliases():
    shader = """
    shader NestedResourceComponentIndex {
        RWStructuredBuffer<uint> values[3] @ binding(0);

        uint readLeaf(RWStructuredBuffer<uint> leaf[], u64vec2 selectors) {
            return buffer_load(leaf[selectors.y], selectors.x);
        }

        uint readNested(RWStructuredBuffer<uint> nested[], u64vec2 selectors) {
            return readLeaf(nested, selectors);
        }

        uint run(u64vec2 selectors) {
            return readNested(values, selectors);
        }
    }
    """

    generated = (
        GLSLCodeGen()
        .set_index_range_assertions(
            [
                {
                    "expression": "selectors.x",
                    "minimum": 0,
                    "maximum": 31,
                },
                {
                    "expression": "selectors.y",
                    "minimum": 0,
                    "maximum": 2,
                },
            ]
        )
        .generate(_parse(shader))
    )

    assert "uint(selectors.x)" in generated
    assert "uint(selectors.y)" in generated
    assert "buffer_load" not in generated


@pytest.mark.parametrize(
    ("index", "reason"),
    [
        ("int64_t(-1)", "negative-index"),
        ("uint64_t(4294967296)", "constant-index-out-of-range"),
        ("uint64_t(4u)", "constant-index-out-of-range"),
    ],
)
def test_opengl_rejects_negative_and_out_of_range_fixed_indices(index, reason):
    shader = f"""
    shader InvalidConstantIndex {{
        uint readValue() {{
            uint values[4];
            return values[{index}];
        }}
    }}
    """
    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))
    assert exc_info.value.reason == reason
    assert exc_info.value.range_status == "out-of-range"


def test_webgl_uses_its_profile_and_fails_closed_for_runtime_wide_index():
    constant_shader = """
    shader WebGLConstantIndex {
        uint readValue() { uint values[4]; return values[uint64_t(2u)]; }
    }
    """
    generated = WebGLCodeGen().generate(_parse(constant_shader))
    assert "#version 300 es" in generated
    assert "values[2u]" in generated
    assert "uint64_t" not in generated

    runtime_shader = """
    shader WebGLRuntimeIndex {
        uint readValue(uint64_t index) {
            uint values[4];
            return values[index];
        }
    }
    """
    with pytest.raises(WebGLIndexTypeError) as exc_info:
        WebGLCodeGen().generate(_parse(runtime_shader))
    assert exc_info.value.target_profile == "WebGL 2 / GLSL ES 3.00"
    assert exc_info.value.reason == "index-range-unproven"


def test_project_range_assertion_and_unproven_diagnostic_contract(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = """shader ProjectIndex {
    StructuredBuffer<uint> values @ binding(0);
    uint readValue(uint64_t index) { return buffer_load(values, index); }
}
"""
    (repo / "index.cgl").write_text(source, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        """[project]
include = ["index.cgl"]
targets = ["opengl"]
output_dir = "out"

[[project.index_range_assertions]]
source = "index.cgl"
function = "readValue"
expression = "index"
minimum = 0
maximum = 31
""",
        encoding="utf-8",
    )

    config = load_project_config(repo)
    project_report = translate_project(config, format_output=False)
    report = project_report.to_json()
    assert report["project"]["indexRangeAssertions"] == [
        {
            "source": "index.cgl",
            "function": "readValue",
            "expression": "index",
            "minimum": 0,
            "maximum": 31,
        }
    ]
    assert report["project"]["indexRangeAssertionCount"] == 1
    report_path = repo / "report.json"
    project_report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True
    generated = (repo / report["artifacts"][0]["path"]).read_text(encoding="utf-8")
    assert "values[uint(index)]" in generated

    unproven = translate_project(
        type(config)(
            root=repo,
            include_patterns=("index.cgl",),
            targets=("opengl",),
            output_dir="unproven",
        ),
        format_output=False,
    ).to_json()
    diagnostic = next(
        item
        for item in unproven["diagnostics"]
        if item["code"] == "project.translate.opengl-index-type-unsupported"
    )
    conversion = diagnostic["details"]["indexConversion"]
    assert conversion["sourceType"] == "uint64_t"
    assert conversion["targetProfile"].startswith("OpenGL #version")
    assert conversion["indexedValue"] == "values"
    assert conversion["rangeStatus"] == "unproven"
    assert diagnostic["location"]["line"] == 3
    assert diagnostic["location"]["column"] > 1
