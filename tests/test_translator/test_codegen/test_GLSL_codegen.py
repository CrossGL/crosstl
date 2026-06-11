import re
from typing import List

import pytest

import crosstl.translator
from crosstl.translator.ast import (
    AttributeNode,
    BlockNode,
    ExecutionModel,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    NamedType,
    ParameterNode,
    PreprocessorNode,
    PrimitiveType,
    ReturnNode,
    ShaderNode,
    ShaderStage,
    StructMemberNode,
    StructNode,
    UnaryOpNode,
    VectorType,
)
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def tokenize_code(code: str) -> List:
    lexer = Lexer(code)
    return lexer.get_tokens()


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST.

    Args:
        tokens (List): The list of tokens to parse
    Returns:
        AST: The abstract syntax tree generated from the parser
    """
    parser = Parser(tokens)
    return parser.parse()


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = GLSLCodeGen()
    return codegen.generate(ast_node)


def test_glsl_type_node_renders_expression_generic_arguments():
    type_node = NamedType(
        "LoopedElemToLoc",
        [
            NamedType("DIM"),
            UnaryOpNode("-", LiteralNode(1, PrimitiveType("int"))),
            NamedType("OffsetT"),
            NamedType("General"),
        ],
    )

    assert (
        GLSLCodeGen().convert_type_node_to_string(type_node)
        == "LoopedElemToLoc<DIM, (-1), OffsetT, General>"
    )


def glsl_image_atomic_parameter_diagnostic(operation, resource_type, zero_value):
    return (
        "/* "
        + "un"
        + "supported GLSL image atomic resource call: "
        + f"{operation} on {resource_type} parameter */ {zero_value}"
    )


def glsl_dynamic_texel_offset_diagnostic(category, operation, zero_value):
    return (
        f"/* unsupported GLSL {category}: {operation} "
        "texel offsets must be compile-time integer constants */ "
        f"{zero_value}"
    )


def test_glsl_specialization_constant_metadata_emits_layout():
    shader = """
    shader SpecializationConstants {
        const int LIGHTING_MODEL @constant_id(0) = 0;
    }
    """

    generated = generate_code(parse_code(tokenize_code(shader)))

    assert "layout(constant_id" not in generated
    assert (
        "/* CrossGL fallback: OpenGL source validation cannot preserve "
        "specialization constant id 0 for 'LIGHTING_MODEL'; using the default "
        "literal. */"
    ) in generated
    assert "const int LIGHTING_MODEL = 0;" in generated


def test_glsl_codegen_drops_metal_system_includes_but_preserves_glsl_includes():
    ast = ShaderNode(
        "IncludeFilters",
        ExecutionModel.COMPUTE_KERNEL,
        preprocessors=[
            PreprocessorNode("include", "<metal_math>"),
            PreprocessorNode("include", "<metal_stdlib>"),
            PreprocessorNode("include", "<shared.glsl>"),
        ],
    )

    generated = generate_code(ast)

    assert "#include <metal_math>" not in generated
    assert "#include <metal_stdlib>" not in generated
    assert "#include <shared.glsl>" in generated


def test_glsl_reserved_parameter_name_escapes_without_colliding():
    shader = """
    shader ParamEscaping {
        float combine(float input, float input_) {
            return input + input_;
        }
    }
    """

    generated = generate_code(crosstl.translator.parse(shader))

    assert "float combine(float input_2, float input_)" in generated
    assert "return (input_2 + input_);" in generated
    assert "float input, " not in generated


def test_glsl_tile_image_ext_importer_attribute_roundtrips():
    # Reduced from the backend GLSL importer output for glslang's
    # Test/spv.ext.ShaderTileImage.overlap.frag fixture.
    shader = """
    #version 460
    #extension GL_EXT_shader_tile_image : require
    precision mediump int;
    precision highp float;

    shader TileImage {
        attachmentEXT in_color @location(0) @tileImageEXT @highp[2];

        fragment {
            vec4 main() @location(1) @highp @out_color {
                vec4 out_color;
                out_color = colorAttachmentReadEXT(in_color[0]);
                return out_color;
            }
        }
    }
    """

    generated = generate_code(parse_code(tokenize_code(shader)))

    assert "#extension GL_EXT_shader_tile_image : require" in generated
    assert (
        "layout(location = 0) tileImageEXT highp attachmentEXT in_color[2];"
        in generated
    )
    assert "layout(location = 0) highp attachmentEXT in_color[2];" not in generated
    assert "out_color = colorAttachmentReadEXT(in_color[0]);" in generated


def test_glsl_stencil_ref_return_semantic_emits_export_extension():
    shader = """
    shader StencilOut {
        fragment {
            uint main() @ gl_FragStencilRefEXT {
                return 7;
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(shader)))

    assert "#extension GL_ARB_shader_stencil_export : require" in generated
    assert "gl_FragStencilRefARB = 7;" in generated
    assert "gl_FragStencilRefEXT" not in generated


def test_glsl_workgroup_barrier_builtin_lowers_to_native_barrier():
    shader = """
    shader SynchronizationBuiltins {
        compute {
            void main() {
                barrier();
                memoryBarrier();
                workgroupBarrier();
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert generated_code.count("barrier();") == 2
    assert "memoryBarrier();" in generated_code
    assert "workgroupBarrier();" not in generated_code


def test_glsl_user_defined_workgroup_barrier_is_not_lowered():
    shader = """
    shader SynchronizationShadowing {
        compute {
            void workgroupBarrier() {
                return;
            }

            void main() {
                workgroupBarrier();
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "void workgroupBarrier()" in generated_code
    assert "workgroupBarrier();" in generated_code
    assert "barrier();" not in generated_code


def test_structured_buffer_operations_lower_to_ssbo():
    code = """
    shader StructuredBufferGLSL {
        RWStructuredBuffer<int> values @ binding(3);

        compute {
            void main(uint3 tid @ gl_GlobalInvocationID) {
                uint len;
                int v = buffer_load(values, tid.x);
                buffer_dimensions(values, len);
                buffer_store(values, tid.x, v + 1);
            }
        }
    }
    """
    ast = parse_code(tokenize_code(code))

    generated = generate_code(ast)

    assert (
        "layout(std430, binding = 3) buffer valuesBuffer { int values[]; };"
        in generated
    )
    assert "int v = values[gl_GlobalInvocationID.x];" in generated
    assert "len = values.length();" in generated
    assert "values[gl_GlobalInvocationID.x] = (v + 1);" in generated
    assert "RWStructuredBuffer" not in generated
    assert "buffer_load" not in generated
    assert "buffer_store" not in generated
    assert "buffer_dimensions" not in generated


def test_compute_stage_validates_builtin_parameter_types():
    float_global_id_code = """
    shader BadComputeGlobalInvocationID {
        compute {
            void main(float gid @ gl_GlobalInvocationID) { }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_GlobalInvocationID.*uvec3"):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(float_global_id_code), "compute"
        )

    vector_local_id_code = """
    shader BadComputeLocalInvocationID {
        compute {
            void main(uint2 lid @ gl_LocalInvocationID) { }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_LocalInvocationID.*uvec3"):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(vector_local_id_code), "compute"
        )

    signed_local_index_code = """
    shader BadComputeLocalInvocationIndex {
        compute {
            void main(int index @ gl_LocalInvocationIndex) { }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_LocalInvocationIndex.*scalar uint"):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(signed_local_index_code), "compute"
        )

    valid_code = """
    shader ValidComputeBuiltinParameters {
        compute {
            void main(
                uint3 gid @ gl_GlobalInvocationID,
                uint3 lid @ gl_LocalInvocationID,
                uint3 group @ gl_WorkGroupID,
                uint3 size @ gl_WorkGroupSize,
                uint3 groups @ gl_NumWorkGroups,
                uint index @ gl_LocalInvocationIndex
            ) {
                uint value = gid.x + lid.x + group.x + size.x + groups.x + index;
            }
        }
    }
    """
    generated = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(valid_code), "compute"
    )
    assert "void main()" in generated
    assert (
        "uint value = (((((gl_GlobalInvocationID.x + gl_LocalInvocationID.x) + "
        "gl_WorkGroupID.x) + gl_WorkGroupSize.x) + gl_NumWorkGroups.x) + "
        "gl_LocalInvocationIndex);"
    ) in generated
    assert "gid @ gl_GlobalInvocationID" not in generated


def test_glsl_hlsl_compute_builtin_parameter_aliases_to_glsl_builtins():
    code = """
    shader HLSLComputeBuiltinAliases {
        compute {
            void main(
                uvec3 tid @ SV_DispatchThreadID,
                uvec3 gid @ SV_GroupID,
                uvec3 lid @ SV_GroupThreadID,
                uint groupIndex @ SV_GroupIndex
            ) {
                uint value = tid.x + gid.y + lid.z + groupIndex;
            }
        }
    }
    """

    generated = GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "compute")

    assert "void main()" in generated
    assert (
        "uint value = (((gl_GlobalInvocationID.x + gl_WorkGroupID.y) + "
        "gl_LocalInvocationID.z) + gl_LocalInvocationIndex);"
    ) in generated
    assert re.search(r"\btid\b", generated) is None
    assert re.search(r"\bgid\b", generated) is None
    assert re.search(r"\blid\b", generated) is None
    assert re.search(r"\bgroupIndex\b", generated) is None
    assert "SV_DispatchThreadID" not in generated


def test_glsl_hlsl_graphics_builtin_parameter_aliases_to_glsl_builtins():
    code = """
    shader HLSLGraphicsBuiltinAliases {
        vertex {
            void main(int vertexId @ SV_VertexID, int instanceId @ SV_InstanceID) {
                gl_Position = vec4(float(vertexId + instanceId));
            }
        }
    }
    """

    generated = GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "vertex")

    assert "void main()" in generated
    assert "in int vertexId;" not in generated
    assert "in int instanceId;" not in generated
    assert "float((gl_VertexID + gl_InstanceID))" in generated
    assert "SV_VertexID" not in generated
    assert "SV_InstanceID" not in generated


def test_glsl_unsigned_sample_mask_input_alias_indexes_builtin_array():
    code = """
    shader SampleMaskInputAlias {
        fragment {
            void main(uint coverage @ SV_Coverage) {
                uint value = coverage;
            }
        }
    }
    """

    generated = GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")

    assert "uint coverage = uint(gl_SampleMaskIn[0]);" in generated
    assert "uint(gl_SampleMaskIn);" not in generated
    assert "uint value = coverage;" in generated


def test_glsl_vertex_index_identifier_lowers_to_vertex_id_builtin():
    code = """
    shader VertexIndexAlias {
        vertex {
            void main() {
                gl_Position = vec4(float(gl_VertexIndex), 0.0, 0.0, 1.0);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "vertex")

    assert "float(gl_VertexID)" in generated
    assert "gl_VertexIndex" not in generated


def test_readonly_structured_buffer_uses_readonly_ssbo():
    code = """
    shader StructuredBufferGLSL {
        StructuredBuffer<int> values @ binding(1);

        compute {
            void main(uint3 tid @ gl_GlobalInvocationID) {
                int v = buffer_load(values, tid.x);
            }
        }
    }
    """
    ast = parse_code(tokenize_code(code))

    generated = generate_code(ast)

    assert (
        "layout(std430, binding = 1) readonly buffer valuesBuffer { int values[]; };"
        in generated
    )
    assert "int v = values[gl_GlobalInvocationID.x];" in generated


def test_glsl_structured_buffer_access_metadata_emits_ssbo_qualifiers():
    code = """
    shader StructuredBufferAccessMetadataGLSL {
        RWStructuredBuffer<int> source @binding(1) @readonly;
        RWStructuredBuffer<int> target @binding(2) @access(writeonly);
        RWStructuredBuffer<int> scratch @binding(3) @access(readwrite);

        compute {
            void main(uint3 tid @ gl_GlobalInvocationID) {
                int value = buffer_load(source, tid.x);
                buffer_store(target, tid.x, value);
                int oldValue = buffer_load(scratch, tid.x);
                buffer_store(scratch, tid.x, oldValue + 1);
            }
        }
    }
    """
    ast = parse_code(tokenize_code(code))

    generated = generate_code(ast)

    assert (
        "layout(std430, binding = 1) readonly buffer sourceBuffer "
        "{ int source[]; };" in generated
    )
    assert (
        "layout(std430, binding = 2) writeonly buffer targetBuffer "
        "{ int target[]; };" in generated
    )
    assert (
        "layout(std430, binding = 3) buffer scratchBuffer { int scratch[]; };"
        in generated
    )
    assert "readwrite buffer" not in generated
    assert "int value = source[gl_GlobalInvocationID.x];" in generated
    assert "target[gl_GlobalInvocationID.x] = value;" in generated
    assert "int oldValue = scratch[gl_GlobalInvocationID.x];" in generated
    assert "scratch[gl_GlobalInvocationID.x] = (oldValue + 1);" in generated


@pytest.mark.parametrize(
    ("shader", "match"),
    [
        (
            """
            shader StructuredBufferAccessInvalidWriteonlyLoad {
                RWStructuredBuffer<int> values @writeonly;

                compute {
                    void main() {
                        int value = buffer_load(values, 0);
                    }
                }
            }
            """,
            "requires read-capable SSBO access.*got writeonly",
        ),
        (
            """
            shader StructuredBufferAccessInvalidReadonlyStore {
                RWStructuredBuffer<int> values @readonly;

                compute {
                    void main() {
                        buffer_store(values, 0, 1);
                    }
                }
            }
            """,
            "requires write-capable SSBO access.*got readonly",
        ),
        (
            """
            shader StructuredBufferAccessInvalidTypedStore {
                StructuredBuffer<int> values;

                compute {
                    void main() {
                        buffer_store(values, 0, 1);
                    }
                }
            }
            """,
            "requires write-capable SSBO access.*got readonly",
        ),
        (
            """
            shader StructuredBufferAccessInvalidTypedMetadata {
                StructuredBuffer<int> values @writeonly;

                compute {
                    void main() {
                    }
                }
            }
            """,
            "StructuredBuffer resource 'values' cannot use writeonly",
        ),
    ],
)
def test_glsl_structured_buffer_access_rejects_invalid_operations(shader, match):
    ast = crosstl.translator.parse(shader)

    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate(ast)


@pytest.mark.parametrize(
    ("shader", "match"),
    [
        (
            """
            shader InvalidStorageImageAccessOperand {
                image2D image @access(foo);

                compute {
                    void main() {
                    }
                }
            }
            """,
            "Invalid OpenGL resource access metadata for 'image': access\\(foo\\)",
        ),
        (
            """
            shader MissingStorageImageAccessOperand {
                image2D image @access;

                compute {
                    void main() {
                    }
                }
            }
            """,
            "Invalid OpenGL resource access metadata for 'image': "
            "access\\(<missing>\\)",
        ),
        (
            """
            shader InvalidStructuredBufferAccessOperand {
                RWStructuredBuffer<int> values @access(foo);

                compute {
                    void main() {
                    }
                }
            }
            """,
            "Invalid OpenGL resource access metadata for 'values': access\\(foo\\)",
        ),
        (
            """
            shader InvalidBufferBlockAccessOperand {
                struct Data {
                    uint value;
                };

                Data data @glsl_buffer_block(std430) @access(foo);

                compute {
                    void main() {
                    }
                }
            }
            """,
            "Invalid OpenGL resource access metadata for 'data': access\\(foo\\)",
        ),
        (
            """
            shader InvalidSampledTextureAccessOperand {
                sampler2D tex @access(foo);

                fragment {
                    vec4 main(vec2 uv) @gl_FragColor {
                        return texture(tex, uv);
                    }
                }
            }
            """,
            "Invalid OpenGL resource access metadata for 'tex': access\\(foo\\)",
        ),
        (
            """
            shader InvalidSamplerStateAccessOperand {
                sampler samp @access(foo);
                sampler2D tex;

                fragment {
                    vec4 main(vec2 uv) @gl_FragColor {
                        return texture(tex, samp, uv);
                    }
                }
            }
            """,
            "Invalid OpenGL resource access metadata for 'samp': access\\(foo\\)",
        ),
        (
            """
            shader InvalidTextureParameterAccessOperand {
                sampler2D tex;

                vec4 sampleTex(sampler2D localTex @access(foo), vec2 uv) {
                    return texture(localTex, uv);
                }

                fragment {
                    vec4 main(vec2 uv) @gl_FragColor {
                        return sampleTex(tex, uv);
                    }
                }
            }
            """,
            "Invalid OpenGL resource access metadata for 'localTex': "
            "access\\(foo\\)",
        ),
    ],
)
def test_glsl_resource_access_rejects_invalid_metadata_operands(shader, match):
    ast = crosstl.translator.parse(shader)

    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate(ast)


def test_glsl_buffer_block_access_allows_compatible_operations():
    shader = """
    shader BufferBlockAccessCompatible {
        struct ReadonlyData {
            uint value;
        };
        struct WriteonlyData {
            uint value;
        };
        struct ReadwriteData {
            uint value;
        };

        ReadonlyData readonlyData @glsl_buffer_block(std430) @readonly;
        WriteonlyData writeonlyData @glsl_buffer_block(std430) @writeonly;
        ReadwriteData readwriteData @glsl_buffer_block(std430) @access(readwrite);

        compute {
            void main() {
                uint value = readonlyData.value;
                writeonlyData.value = value;
                uint old = atomicAdd(readwriteData.value, 1u);
                readwriteData.value += old;
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(std430, binding = 0) readonly buffer ReadonlyData" in generated
    assert "layout(std430, binding = 1) writeonly buffer WriteonlyData" in generated
    assert "layout(std430, binding = 2) buffer ReadwriteData" in generated
    assert "readwrite buffer" not in generated
    assert "uint value = readonlyData.value;" in generated
    assert "writeonlyData.value = value;" in generated
    assert "uint old = atomicAdd(readwriteData.value, 1u);" in generated
    assert "readwriteData.value += old;" in generated


@pytest.mark.parametrize(
    ("shader", "match"),
    [
        (
            """
            shader ConflictingGlobalBufferBlockLayouts {
                struct Data {
                    int value;
                };

                Data data[2] @glsl_buffer_block(std430, std140);

                compute {
                    void main() {
                        int value = data[0].value;
                    }
                }
            }
            """,
            "Conflicting OpenGL buffer block memory layout metadata for "
            "'data': std430 differs from std140",
        ),
        (
            """
            shader DuplicateBufferBlockLayouts {
                struct Data {
                    int value;
                };

                Data data @glsl_buffer_block(std430, std430);

                compute {
                    void main() {
                        int value = data.value;
                    }
                }
            }
            """,
            "Duplicate OpenGL buffer block layout metadata for 'data': std430",
        ),
        (
            """
            shader ConflictingStageLocalBufferBlockLayouts {
                struct Data {
                    int value;
                };

                compute {
                    Data localData @glsl_buffer_block(std140, scalar);

                    void main() {
                        int value = localData.value;
                    }
                }
            }
            """,
            "Conflicting OpenGL buffer block memory layout metadata for "
            "'localData': std140 differs from scalar",
        ),
    ],
)
def test_glsl_buffer_block_layout_metadata_rejects_duplicates_and_conflicts(
    shader, match
):
    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


@pytest.mark.parametrize(
    ("shader", "match"),
    [
        (
            """
            shader BufferBlockReadFromWriteonly {
                struct Data {
                    uint value;
                };

                Data data @glsl_buffer_block(std430) @writeonly;

                compute {
                    void main() {
                        uint value = data.value;
                    }
                }
            }
            """,
            "requires read-capable buffer block access.*got writeonly",
        ),
        (
            """
            shader BufferBlockWriteToReadonly {
                struct Data {
                    uint value;
                };

                Data data @glsl_buffer_block(std430) @readonly;

                compute {
                    void main() {
                        data.value = 1u;
                    }
                }
            }
            """,
            "requires write-capable buffer block access.*got readonly",
        ),
        (
            """
            shader BufferBlockCompoundReadonly {
                struct Data {
                    uint value;
                };

                Data data @glsl_buffer_block(std430) @readonly;

                compute {
                    void main() {
                        data.value += 1u;
                    }
                }
            }
            """,
            "requires read-write buffer block access.*got readonly",
        ),
        (
            """
            shader BufferBlockCompoundWriteonly {
                struct Data {
                    uint value;
                };

                Data data @glsl_buffer_block(std430) @writeonly;

                compute {
                    void main() {
                        data.value += 1u;
                    }
                }
            }
            """,
            "requires read-write buffer block access.*got writeonly",
        ),
        (
            """
            shader BufferBlockAtomicReadonly {
                struct Data {
                    uint value;
                };

                Data data @glsl_buffer_block(std430) @readonly;

                compute {
                    void main() {
                        uint old = atomicAdd(data.value, 1u);
                    }
                }
            }
            """,
            "requires read-write buffer block access.*got readonly",
        ),
        (
            """
            shader BufferBlockAtomicWriteonly {
                struct Data {
                    uint value;
                };

                Data data @glsl_buffer_block(std430) @writeonly;

                compute {
                    void main() {
                        uint old = atomicAdd(data.value, 1u);
                    }
                }
            }
            """,
            "requires read-write buffer block access.*got writeonly",
        ),
        (
            """
            shader BufferBlockAtomicReadonlyArrayElement {
                struct Data {
                    uint values[4];
                };

                Data data @glsl_buffer_block(std430) @readonly;

                compute {
                    void main() {
                        uint old = atomicAdd(data.values[0], 1u);
                    }
                }
            }
            """,
            "requires read-write buffer block access.*got readonly",
        ),
    ],
)
def test_glsl_buffer_block_access_rejects_invalid_operations(shader, match):
    ast = crosstl.translator.parse(shader)

    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate(ast)


def test_glsl_buffer_block_nested_and_array_atomically_mutable_members():
    shader = """
    shader BufferBlockNestedAtomicMembers {
        struct Counter {
            uint value;
        };
        struct Data {
            Counter nested;
            Counter items[2];
            uint values[4];
        };

        Data data @glsl_buffer_block(std430) @access(readwrite);

        compute {
            void main() {
                uint nestedOld = atomicAdd(data.nested.value, 1u);
                uint itemOld = atomicAdd(data.items[1].value, nestedOld);
                uint arrayOld = atomicAdd(data.values[0], itemOld);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Counter nested;" in generated
    assert "Counter items[2];" in generated
    assert "uint values[4];" in generated
    assert "uint nestedOld = atomicAdd(data.nested.value, 1u);" in generated
    assert "uint itemOld = atomicAdd(data.items[1].value, nestedOld);" in generated
    assert "uint arrayOld = atomicAdd(data.values[0], itemOld);" in generated


@pytest.mark.parametrize(
    ("member_declarations", "target", "value", "match"),
    [
        (
            "float value;",
            "data.value",
            "1.0",
            r"requires a scalar int or uint buffer block member.*got float",
        ),
        (
            "uvec2 value;",
            "data.value",
            "1u",
            r"requires a scalar int or uint buffer block member.*got uvec2",
        ),
        (
            "mat2 value;",
            "data.value",
            "1.0",
            r"requires a scalar int or uint buffer block member.*got mat2",
        ),
        (
            "Counter nested;",
            "data.nested",
            "1u",
            r"requires a scalar int or uint buffer block member.*got Counter",
        ),
    ],
)
def test_glsl_buffer_block_atomics_reject_non_scalar_integer_members(
    member_declarations,
    target,
    value,
    match,
):
    shader = f"""
    shader BufferBlockInvalidAtomicMember {{
        struct Counter {{
            uint value;
        }};
        struct Data {{
            {member_declarations}
        }};

        Data data @glsl_buffer_block(std430) @access(readwrite);

        compute {{
            void main() {{
                atomicAdd({target}, {value});
            }}
        }}
    }}
    """
    ast = crosstl.translator.parse(shader)

    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate(ast)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            "atomicAdd(data.value, signedDelta);",
            r"requires uint value argument.*signedDelta has type int",
        ),
        (
            "atomicAdd(data.signedValue, unsignedDelta);",
            r"requires int value argument.*unsignedDelta has type uint",
        ),
        (
            "atomicCompSwap(data.value, signedDelta, unsignedDelta);",
            r"requires uint compare argument.*signedDelta has type int",
        ),
        (
            "atomicCompSwap(data.signedValue, signedDelta, unsignedDelta);",
            r"requires int value argument.*unsignedDelta has type uint",
        ),
        (
            "atomicExchange(data.value, boolDelta);",
            r"requires uint value argument.*boolDelta has type bool",
        ),
        (
            "atomicAdd(data.value, vectorDelta);",
            r"requires uint value argument.*vectorDelta has type vec2",
        ),
    ],
)
def test_glsl_buffer_block_atomics_reject_incompatible_value_arguments(call, match):
    shader = f"""
    shader BufferBlockInvalidAtomicArguments {{
        struct Data {{
            uint value;
            int signedValue;
        }};

        Data data @glsl_buffer_block(std430) @access(readwrite);

        compute {{
            void main() {{
                int signedDelta = 1;
                uint unsignedDelta = 1u;
                bool boolDelta = true;
                vec2 vectorDelta = vec2(1.0, 2.0);
                {call}
            }}
        }}
    }}
    """
    ast = crosstl.translator.parse(shader)

    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate(ast)


def test_glsl_structured_buffer_access_rejects_incompatible_helper_calls():
    shader = """
    shader StructuredBufferAccessInvalidHelper {
        RWStructuredBuffer<int> target @writeonly;

        int readValue(RWStructuredBuffer<int> values, uint index) {
            return buffer_load(values, index);
        }

        compute {
            void main() {
                int value = readValue(target, 0u);
            }
        }
    }
    """
    ast = crosstl.translator.parse(shader)

    with pytest.raises(
        ValueError,
        match="function call 'readValue' requires read-capable SSBO access",
    ):
        GLSLCodeGen().generate(ast)


def test_append_consume_structured_buffers_use_sidecar_counters():
    code = """
    shader StructuredBufferAppendConsumeGLSL {
        AppendStructuredBuffer<int> appendValues @ binding(1);
        ConsumeStructuredBuffer<int> consumeValues @ binding(2);

        compute {
            void main(uint value) {
                buffer_append(appendValues, int(value));
                int consumed = buffer_consume(consumeValues);
            }
        }
    }
    """
    ast = parse_code(tokenize_code(code))

    generated = generate_code(ast)

    assert (
        "layout(std430, binding = 1) buffer appendValuesBuffer { int appendValues[]; };"
        in generated
    )
    assert (
        "layout(std430, binding = 2) buffer consumeValuesBuffer { int consumeValues[]; };"
        in generated
    )
    assert (
        "layout(std430, binding = 3) buffer appendValuesCounterBuffer { uint appendValuesCounter[]; };"
        in generated
    )
    assert (
        "layout(std430, binding = 4) buffer consumeValuesCounterBuffer { uint consumeValuesCounter[]; };"
        in generated
    )
    assert (
        "appendValues[atomicAdd(appendValuesCounter[0], 1u)] = "
        "int(main_value_Args_value);" in generated
    )
    assert (
        "int consumed = consumeValues[(atomicAdd(consumeValuesCounter[0], uint(-1)) - 1u)];"
        in generated
    )
    assert "appendValues[0]" not in generated
    assert "consumeValues[0]" not in generated
    assert "buffer_append" not in generated
    assert "buffer_consume" not in generated


def test_append_consume_structured_buffer_helpers_thread_sidecar_counters():
    code = """
    shader StructuredBufferAppendConsumeHelpersGLSL {
        AppendStructuredBuffer<int> appendValues @ binding(1);
        ConsumeStructuredBuffer<int> consumeValues @ binding(2);

        void pushValue(AppendStructuredBuffer<int> values, uint value) {
            buffer_append(values, int(value));
        }

        int popValue(ConsumeStructuredBuffer<int> values) {
            return buffer_consume(values);
        }

        compute {
            void main(uint value) {
                pushValue(appendValues, value);
                int consumed = popValue(consumeValues);
            }
        }
    }
    """
    ast = parse_code(tokenize_code(code))

    generated = generate_code(ast)

    assert (
        "void pushValue(int values[], uint valuesCounter[], uint value) {" in generated
    )
    assert "values[atomicAdd(valuesCounter[0], 1u)] = int(value);" in generated
    assert "int popValue(int values[], uint valuesCounter[]) {" in generated
    assert "return values[(atomicAdd(valuesCounter[0], uint(-1)) - 1u)];" in generated
    assert (
        "pushValue(appendValues, appendValuesCounter, main_value_Args_value);"
        in generated
    )
    assert "int consumed = popValue(consumeValues, consumeValuesCounter);" in generated
    assert "buffer_append" not in generated
    assert "buffer_consume" not in generated


def test_append_structured_buffer_arrays_use_sidecar_counter_arrays():
    code = """
    shader StructuredBufferAppendArrayGLSL {
        AppendStructuredBuffer<int> appendValues[2] @ binding(1);

        compute {
            void main(uint value, uint index) {
                buffer_append(appendValues[index], int(value));
            }
        }
    }
    """
    ast = parse_code(tokenize_code(code))

    generated = generate_code(ast)

    assert (
        "layout(std430, binding = 1) buffer appendValuesBuffer { int data[]; } appendValues[2];"
        in generated
    )
    assert (
        "layout(std430, binding = 3) buffer appendValuesCounterBuffer { uint counter[]; } appendValuesCounters[2];"
        in generated
    )
    assert (
        "appendValues[main_index_Args_index].data[atomicAdd("
        "appendValuesCounters[main_index_Args_index].counter[0], 1u)] = "
        "int(main_value_Args_value);"
        in generated
    )
    assert "buffer_append" not in generated


def test_structured_buffer_arrays_lower_to_ssbo_instance_arrays():
    code = """
    shader StructuredBufferArrayGLSL {
        RWStructuredBuffer<int> buffers[2] @ binding(3);

        compute {
            void main(uint3 tid @ gl_GlobalInvocationID, uint index) {
                uint len;
                int v = buffer_load(buffers[index], tid.x);
                buffer_dimensions(buffers[index], len);
                buffer_store(buffers[index], tid.x, v + 1);
            }
        }
    }
    """
    ast = parse_code(tokenize_code(code))

    generated = generate_code(ast)

    assert (
        "layout(std430, binding = 3) buffer buffersBuffer { int data[]; } buffers[2];"
        in generated
    )
    assert (
        "int v = buffers[main_index_Args_index].data[gl_GlobalInvocationID.x];"
        in generated
    )
    assert "len = buffers[main_index_Args_index].data.length();" in generated
    assert (
        "buffers[main_index_Args_index].data[gl_GlobalInvocationID.x] = (v + 1);"
        in generated
    )
    assert "buffers[index][gl_GlobalInvocationID.x]" not in generated


def test_opengl_structured_buffer_array_binding_overlap_raises():
    code = """
    shader StructuredBufferArrayBindingOverlapGLSL {
        RWStructuredBuffer<int> first[2] @ binding(3);
        RWStructuredBuffer<int> second @ binding(4);

        int read(uint index) {
            return buffer_load(first[1], index) + buffer_load(second, index);
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting OpenGL resource binding for 'second'",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_opengl_texture_array_binding_overlap_raises():
    code = """
    shader TextureArrayBindingOverlapGLSL {
        sampler2D first[2] @binding(3);
        sampler2D second @binding(4);

        vec4 read(vec2 uv) {
            return texture(first[1], uv) + texture(second, uv);
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting OpenGL resource binding for 'second'",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_opengl_image_array_binding_overlap_raises():
    code = """
    shader ImageArrayBindingOverlapGLSL {
        image2D first[2] @binding(3);
        image2D second @binding(4);

        vec4 read(ivec2 pixel) {
            return imageLoad(first[1], pixel) + imageLoad(second, pixel);
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting OpenGL resource binding for 'second'",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_opengl_texture_and_image_bindings_are_independent():
    code = """
    shader TextureImageBindingNamespacesGLSL {
        sampler2D tex @binding(0);
        image2D img @binding(0);
    }
    """

    generated = generate_code(parse_code(tokenize_code(code)))

    assert "layout(binding = 0) uniform sampler2D tex;" in generated
    assert "layout(rgba32f, binding = 0) uniform image2D img;" in generated


def test_opengl_auto_bindings_are_independent_between_resource_namespaces():
    code = """
    shader AutoBindingNamespacesGLSL {
        RWStructuredBuffer<int> values[2];
        sampler2D textures[2];
        image2D images[3];
        sampler2D afterTexture;
        image2D afterImage;
    }
    """

    generated = generate_code(parse_code(tokenize_code(code)))

    assert "layout(std430, binding = 0) buffer valuesBuffer" in generated
    assert "layout(binding = 0) uniform sampler2D textures[2];" in generated
    assert "layout(rgba32f, binding = 0) uniform image2D images[3];" in generated
    assert "layout(binding = 2) uniform sampler2D afterTexture;" in generated
    assert "layout(rgba32f, binding = 3) uniform image2D afterImage;" in generated


def test_opengl_auto_bindings_skip_later_explicit_global_bindings():
    code = """
    shader LateExplicitBindingsGLSL {
        RWStructuredBuffer<int> autoBuffer;
        sampler2D autoTexture;
        image2D autoImage;

        cbuffer AutoBlock {
            vec4 autoTint;
        };

        cbuffer ExplicitBlock @binding(0) {
            vec4 explicitTint;
        };

        RWStructuredBuffer<int> explicitBuffer @binding(0);
        sampler2D explicitTexture @binding(0);
        image2D explicitImage @binding(0);

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                vec4 stored = imageLoad(autoImage, ivec2(0, 0));
                int value = buffer_load(autoBuffer, 0) + buffer_load(explicitBuffer, 0);
                return texture(autoTexture, uv) +
                    texture(explicitTexture, uv) +
                    stored + autoTint + explicitTint + vec4(float(value));
            }
        }
    }
    """

    generated = GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")

    assert "layout(std430, binding = 1) buffer autoBufferBuffer" in generated
    assert "layout(binding = 1) uniform sampler2D autoTexture;" in generated
    assert "layout(rgba32f, binding = 1) uniform image2D autoImage;" in generated
    assert "layout(std140, binding = 1) uniform AutoBlock" in generated
    assert "layout(std430, binding = 0) buffer explicitBufferBuffer" in generated
    assert "layout(binding = 0) uniform sampler2D explicitTexture;" in generated
    assert "layout(rgba32f, binding = 0) uniform image2D explicitImage;" in generated
    assert "layout(std140, binding = 0) uniform ExplicitBlock" in generated


def test_structured_buffer_fixed_width_aliases_lower_to_standard_ssbo_types():
    code = """
    shader AliasStructuredBufferGLSL {
        RWStructuredBuffer<uint16_t> counts @ binding(3);
        StructuredBuffer<int64_t> signedValues @ binding(4);
        RWStructuredBuffer<size_t> offsets @ binding(5);

        uint64_t loadOffset(uint index) {
            uint64_t offset = buffer_load(offsets, index);
            uint16_t count = buffer_load(counts, index);
            int64_t signedValue = buffer_load(signedValues, index);
            buffer_store(counts, index, count + uint16_t(1u));
            return offset + uint64_t(count) + uint64_t(signedValue);
        }
    }
    """

    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "layout(std430, binding = 3) buffer countsBuffer { uint counts[]; };"
        in generated
    )
    assert (
        "layout(std430, binding = 4) readonly buffer signedValuesBuffer { int signedValues[]; };"
        in generated
    )
    assert (
        "layout(std430, binding = 5) buffer offsetsBuffer { uint offsets[]; };"
        in generated
    )
    assert "uint offset = offsets[index];" in generated
    assert "uint count = counts[index];" in generated
    assert "int signedValue = signedValues[index];" in generated
    assert "counts[index] = (count + uint(1u));" in generated
    for invalid_token in ("uint16_t", "int64_t", "uint64_t", "size_t"):
        assert invalid_token not in generated


def test_structured_buffer_alias_helper_call_passes_ssbo_data_array():
    code = """
    shader AliasStructuredBufferHelperGLSL {
        RWStructuredBuffer<uint16_t> counts[2] @ binding(3);

        uint16_t readOne(RWStructuredBuffer<uint16_t> localCounts, uint index) {
            return buffer_load(localCounts, index);
        }

        uint16_t combine(uint which, uint index) {
            return readOne(counts[which], index);
        }
    }
    """

    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "layout(std430, binding = 3) buffer countsBuffer { uint data[]; } counts[2];"
        in generated
    )
    assert "uint readOne(uint localCounts[], uint index)" in generated
    assert "return localCounts[index];" in generated
    assert "return readOne(counts[which].data, index);" in generated
    assert "readOne(counts[which], index)" not in generated
    assert "buffer_load" not in generated
    assert "uint16_t" not in generated


def test_structured_buffer_alias_array_helpers_expand_to_data_array_parameters():
    code = """
    shader AliasStructuredBufferArrayHelperGLSL {
        RWStructuredBuffer<uint16_t> counts[2] @ binding(3);
        StructuredBuffer<size_t> offsets[2] @ binding(5);

        uint16_t readCount(RWStructuredBuffer<uint16_t> localCounts[], uint which, uint index) {
            return buffer_load(localCounts[which], index);
        }

        uint64_t readOffset(StructuredBuffer<size_t> localOffsets[2], uint which, uint index) {
            uint len;
            buffer_dimensions(localOffsets[which], len);
            return buffer_load(localOffsets[which], index) + uint64_t(len);
        }

        void writeCount(RWStructuredBuffer<uint16_t> localCounts[2], uint which, uint index, uint16_t value) {
            buffer_store(localCounts[which], index, value);
        }

        uint64_t combine(uint which, uint index) {
            uint16_t count = readCount(counts, which, index);
            uint64_t offset = readOffset(offsets, which, index);
            writeCount(counts, which, index, count + uint16_t(1u));
            return offset + uint64_t(count);
        }
    }
    """

    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "layout(std430, binding = 3) buffer countsBuffer { uint data[]; } counts[2];"
        in generated
    )
    assert (
        "layout(std430, binding = 5) readonly buffer offsetsBuffer { uint data[]; } offsets[2];"
        in generated
    )
    assert (
        "uint readCount(uint localCounts_0[], uint localCounts_1[], uint which, uint index)"
        in generated
    )
    assert (
        "return ((which == 0u) ? localCounts_0[index] : localCounts_1[index]);"
        in generated
    )
    assert (
        "uint readOffset(uint localOffsets_0[], uint localOffsets_1[], uint which, uint index)"
        in generated
    )
    assert (
        "len = ((which == 0u) ? localOffsets_0.length() : localOffsets_1.length());"
        in generated
    )
    assert (
        "void writeCount(uint localCounts_0[], uint localCounts_1[], uint which, uint index, uint value)"
        in generated
    )
    assert (
        "((which == 0u) ? (localCounts_0[index] = value) : (localCounts_1[index] = value));"
        in generated
    )
    assert (
        "uint count = readCount(counts[0].data, counts[1].data, which, index);"
        in generated
    )
    assert (
        "uint offset = readOffset(offsets[0].data, offsets[1].data, which, index);"
        in generated
    )
    assert (
        "writeCount(counts[0].data, counts[1].data, which, index, (count + uint(1u)));"
        in generated
    )
    assert "readCount(counts, which, index)" not in generated
    assert "readOffset(offsets, which, index)" not in generated
    assert "writeCount(counts, which, index" not in generated
    assert "buffer_load" not in generated
    assert "buffer_store" not in generated
    assert "buffer_dimensions" not in generated
    assert "uint16_t" not in generated
    assert "size_t" not in generated


def test_unsized_structured_buffer_array_helper_emits_diagnostic_fallback():
    code = """
    shader UnsizedStructuredBufferArrayHelperGLSL {
        RWStructuredBuffer<uint16_t> buffers[] @ binding(3);

        uint16_t readOne(RWStructuredBuffer<uint16_t> localBuffers[], uint which, uint index) {
            return buffer_load(localBuffers[which], index);
        }

        void writeOne(RWStructuredBuffer<uint16_t> localBuffers[], uint which, uint index, uint16_t value) {
            buffer_store(localBuffers[which], index, value);
        }

        uint16_t combine(uint which, uint index) {
            uint16_t value = readOne(buffers, which, index);
            writeOne(buffers, which, index, value + uint16_t(1u));
            return value;
        }
    }
    """

    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "layout(std430, binding = 3) buffer buffersBuffer { uint data[]; } buffers[];"
        in generated
    )
    assert "uint readOne(uint which, uint index)" in generated
    assert "void writeOne(uint which, uint index, uint value)" in generated
    assert (
        "unsupported GLSL structured buffer array parameter readOne.localBuffers"
        in generated
    )
    assert (
        "unsupported GLSL structured buffer array parameter writeOne.localBuffers"
        in generated
    )
    assert "return 0u;" in generated
    assert "uint value = readOne(which, index);" in generated
    assert "writeOne(which, index, (value + uint(1u)));" in generated
    assert "readOne(buffers, which, index)" not in generated
    assert "writeOne(buffers, which, index" not in generated
    assert "localBuffers[which][index]" not in generated
    assert "uint localBuffers[]" not in generated
    assert "buffer_load" not in generated
    assert "buffer_store" not in generated
    assert "uint16_t" not in generated


def test_unsized_global_structured_buffer_arrays_infer_static_sizes():
    code = """
    shader UnsizedGlobalStructuredBufferArrayGLSL {
        RWStructuredBuffer<uint16_t> counts[] @ binding(3);
        RWStructuredBuffer<uint16_t> afterCounts;

        uint16_t readCount(RWStructuredBuffer<uint16_t> localCounts[], uint index) {
            return buffer_load(localCounts[1], index);
        }

        uint16_t combine(uint index) {
            uint16_t first = buffer_load(counts[0], index);
            uint16_t second = readCount(counts, index);
            return first + second + buffer_load(afterCounts, index);
        }
    }
    """

    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "layout(std430, binding = 3) buffer countsBuffer { uint data[]; } counts[2];"
        in generated
    )
    assert (
        "layout(std430, binding = 5) buffer afterCountsBuffer { uint afterCounts[]; };"
        in generated
    )
    assert (
        "uint readCount(uint localCounts_0[], uint localCounts_1[], uint index)"
        in generated
    )
    assert "return localCounts_1[index];" in generated
    assert "uint first = counts[0].data[index];" in generated
    assert (
        "uint second = readCount(counts[0].data, counts[1].data, index);" in generated
    )
    assert "buffer_load" not in generated
    assert "counts[];" not in generated
    assert "readCount(counts, index)" not in generated
    assert "uint16_t" not in generated


def test_unsized_global_structured_buffer_direct_index_sizes_dynamic_helper():
    code = """
    shader UnsizedGlobalStructuredBufferDynamicHelperGLSL {
        RWStructuredBuffer<uint> counts[] @ binding(3);
        RWStructuredBuffer<uint> afterCounts;

        uint readLeaf(RWStructuredBuffer<uint> leafCounts[], uint which, uint index) {
            return buffer_load(leafCounts[which], index);
        }

        uint readDynamic(RWStructuredBuffer<uint> localCounts[], uint which, uint index) {
            return readLeaf(localCounts, which, index);
        }

        uint combine(uint which, uint index) {
            return buffer_load(counts[2], index) + readDynamic(counts, which, index) + buffer_load(afterCounts, index);
        }
    }
    """

    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "layout(std430, binding = 3) buffer countsBuffer { uint data[]; } counts[3];"
        in generated
    )
    assert (
        "layout(std430, binding = 6) buffer afterCountsBuffer { uint afterCounts[]; };"
        in generated
    )
    assert (
        "uint readDynamic(uint localCounts_0[], uint localCounts_1[], "
        "uint localCounts_2[], uint which, uint index)" in generated
    )
    assert (
        "uint readLeaf(uint leafCounts_0[], uint leafCounts_1[], "
        "uint leafCounts_2[], uint which, uint index)" in generated
    )
    assert "counts[2].data[index]" in generated
    assert "leafCounts_2[index]" in generated
    assert (
        "readLeaf(localCounts_0, localCounts_1, localCounts_2, which, index)"
        in generated
    )
    assert (
        "readDynamic(counts[0].data, counts[1].data, counts[2].data, which, index)"
        in generated
    )
    assert "unsupported GLSL structured buffer array parameter" not in generated
    assert "readLeaf(localCounts, which, index)" not in generated
    assert "readDynamic(which, index)" not in generated
    assert "buffer_load" not in generated


def test_unsized_global_structured_buffer_array_rejects_mismatched_fixed_helper_size():
    code = """
    shader UnsizedStructuredBufferFixedMismatchGLSL {
        RWStructuredBuffer<uint> counts[] @ binding(3);

        uint readCount(RWStructuredBuffer<uint> localCounts[3], uint index) {
            return buffer_load(localCounts[2], index);
        }

        uint combine(uint index) {
            return buffer_load(counts[3], index) + readCount(counts, index);
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'counts': 4 and 3",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_unsized_global_structured_buffer_array_rejects_multiple_fixed_helper_sizes():
    code = """
    shader UnsizedStructuredBufferMultipleFixedHelpersGLSL {
        RWStructuredBuffer<uint> counts[] @ binding(3);

        uint readThree(RWStructuredBuffer<uint> localCounts[3], uint index) {
            return buffer_load(localCounts[2], index);
        }

        uint readFour(RWStructuredBuffer<uint> localCounts[4], uint index) {
            return buffer_load(localCounts[3], index);
        }

        uint combine(uint index) {
            return readThree(counts, index) + readFour(counts, index);
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'counts': 3 and 4",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_structured_buffer_array_helpers_propagate_nested_fixed_sizes():
    code = """
    shader NestedStructuredBufferArrayHelpersGLSL {
        RWStructuredBuffer<uint16_t> counts[] @ binding(3);
        RWStructuredBuffer<uint16_t> afterCounts;

        uint16_t readLeaf(RWStructuredBuffer<uint16_t> leafCounts[3], uint index) {
            return buffer_load(leafCounts[2], index);
        }

        uint16_t readMid(RWStructuredBuffer<uint16_t> midCounts[], uint index) {
            return readLeaf(midCounts, index);
        }

        uint16_t combine(uint index) {
            return readMid(counts, index) + buffer_load(afterCounts, index);
        }
    }
    """

    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "layout(std430, binding = 3) buffer countsBuffer { uint data[]; } counts[3];"
        in generated
    )
    assert (
        "layout(std430, binding = 6) buffer afterCountsBuffer { uint afterCounts[]; };"
        in generated
    )
    assert (
        "uint readLeaf(uint leafCounts_0[], uint leafCounts_1[], "
        "uint leafCounts_2[], uint index)" in generated
    )
    assert (
        "uint readMid(uint midCounts_0[], uint midCounts_1[], "
        "uint midCounts_2[], uint index)" in generated
    )
    assert "return leafCounts_2[index];" in generated
    assert "return readLeaf(midCounts_0, midCounts_1, midCounts_2, index);" in generated
    assert "readMid(counts[0].data, counts[1].data, counts[2].data, index)" in generated
    assert "readLeaf(midCounts, index)" not in generated
    assert "readMid(counts, index)" not in generated
    assert "buffer_load" not in generated
    assert "uint16_t" not in generated


def test_structured_buffer_array_nested_helpers_reject_mixed_fixed_sizes():
    code = """
    shader NestedStructuredBufferArrayConflictGLSL {
        RWStructuredBuffer<uint> counts[] @ binding(3);

        uint readLeaf(RWStructuredBuffer<uint> leafCounts[3], uint index) {
            return buffer_load(leafCounts[2], index);
        }

        uint readMid(RWStructuredBuffer<uint> midCounts[4], uint index) {
            return readLeaf(midCounts, index);
        }

        uint combine(uint index) {
            return readMid(counts, index);
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'midCounts': 4 and 3",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_glsl_unsigned_integer_literal_suffix_codegen():
    code = """
    shader UIntLiteralCodegen {
        compute {
            void main() {
                uint a = 7u;
                uint b = 0xFu;
                uint c = 0b101U;
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "uint a = 7u;" in generated_code
    assert "uint b = 15u;" in generated_code
    assert "uint c = 5u;" in generated_code
    assert "7, u" not in generated_code


def test_glsl_resource_binding_attributes_are_not_parameter_semantics():
    code = """
    shader BindingAttributes {
        vec4 sampleBound(sampler2D tex @texture(1), sampler samp @sampler(2), vec2 uv) {
            return texture(tex, samp, uv);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "vec4 sampleBound(sampler2D tex, vec2 uv)" in generated_code
    assert "return texture(tex, uv);" in generated_code
    assert "tex texture" not in generated_code
    assert "samp sampler" not in generated_code


def test_glsl_global_resource_binding_attributes_drive_layout_bindings():
    code = """
    shader ExplicitGlobalBindings {
        sampler samp @sampler(5);
        sampler2D explicitTex @texture(3);
        image2D storageImage @binding(7);
        sampler2D registerTex @register(t8);
        sampler2D autoTex;

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                vec4 stored = imageLoad(storageImage, ivec2(0, 0));
                return texture(explicitTex, samp, uv) + texture(registerTex, samp, uv) + texture(autoTex, samp, uv) + stored;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(binding = 3) uniform sampler2D explicitTex;" in generated_code
    assert (
        "layout(rgba32f, binding = 7) uniform image2D storageImage;" in generated_code
    )
    assert "layout(binding = 8) uniform sampler2D registerTex;" in generated_code
    assert "layout(binding = 9) uniform sampler2D autoTex;" in generated_code


def test_glsl_hlsl_sampler_register_metadata_maps_to_texture_binding():
    code = """
    shader LegacyHlslSamplerRegister {
        sampler2D Texture @register(s0);
        sampler2D NextTexture;

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                return texture(Texture, uv) + texture(NextTexture, uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D Texture;" in generated_code
    assert "layout(binding = 1) uniform sampler2D NextTexture;" in generated_code


def test_glsl_resource_binding_aliases_accept_equivalent_metadata():
    code = """
    shader EquivalentResourceBindings {
        cbuffer Constants @binding(4) @register(b4) {
            vec4 tint;
        };

        RWStructuredBuffer<int> values @binding(3) @register(u3);
        StructuredBuffer<int> readValues @binding(5) @register(t5);
        sampler linearSampler @sampler(6) @register(s6);
        sampler2D tex @binding(1) @texture(1) @register(t1);
        image2D img @binding(2) @register(u2);

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                int value = buffer_load(values, 0);
                int readValue = buffer_load(readValues, 0);
                return texture(tex, uv) + imageLoad(img, ivec2(0, 0)) +
                    tint + vec4(float(value + readValue));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(std140, binding = 4) uniform Constants" in generated_code
    assert "layout(std430, binding = 3) buffer valuesBuffer" in generated_code
    assert "layout(std430, binding = 5) readonly buffer readValuesBuffer" in (
        generated_code
    )
    assert "layout(binding = 1) uniform sampler2D tex;" in generated_code
    assert "layout(rgba32f, binding = 2) uniform image2D img;" in generated_code
    assert "sampler linearSampler" not in generated_code


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (
            """
            shader TextureUsesURegister {
                sampler2D tex @register(u2);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return texture(tex, uv);
                    }
                }
            }
            """,
            "Incompatible OpenGL resource register metadata for 'tex': "
            "register u2 uses u-register, expected t-register for texture binding",
        ),
        (
            """
            shader ImageUsesTRegister {
                image2D img @register(t2);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return imageLoad(img, ivec2(0, 0));
                    }
                }
            }
            """,
            "Incompatible OpenGL resource register metadata for 'img': "
            "register t2 uses t-register, expected u-register for image binding",
        ),
        (
            """
            shader RWBufferUsesTRegister {
                RWStructuredBuffer<int> values @register(t2);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return vec4(float(buffer_load(values, 0)));
                    }
                }
            }
            """,
            "Incompatible OpenGL resource register metadata for 'values': "
            "register t2 uses t-register, expected u-register for buffer binding",
        ),
        (
            """
            shader ReadonlyBufferUsesURegister {
                StructuredBuffer<int> values @register(u2);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return vec4(float(buffer_load(values, 0)));
                    }
                }
            }
            """,
            "Incompatible OpenGL resource register metadata for 'values': "
            "register u2 uses u-register, expected t-register for buffer binding",
        ),
        (
            """
            shader CBufferUsesURegister {
                cbuffer Constants @register(u2) {
                    vec4 tint;
                };

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return tint;
                    }
                }
            }
            """,
            "Incompatible OpenGL resource register metadata for 'Constants': "
            "register u2 uses u-register, expected b-register for uniform buffer "
            "binding",
        ),
        (
            """
            shader BufferBlockUsesTRegister {
                struct Data {
                    int value;
                };

                Data data @glsl_buffer_block(std430) @register(t2);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return vec4(float(data.value));
                    }
                }
            }
            """,
            "Incompatible OpenGL resource register metadata for 'data': "
            "register t2 uses t-register, expected u-register for buffer binding",
        ),
        (
            """
            shader SamplerUsesTRegister {
                sampler samp @register(t2);
                sampler2D tex;

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return texture(tex, samp, uv);
                    }
                }
            }
            """,
            "Incompatible OpenGL resource register metadata for 'samp': "
            "register t2 uses t-register, expected s-register for sampler binding",
        ),
        (
            """
            shader StageSamplerUsesURegister {
                sampler2D tex;

                fragment {
                    vec4 main(sampler samp @register(u2), vec2 uv @TEXCOORD0)
                        @gl_FragColor {
                        return texture(tex, samp, uv);
                    }
                }
            }
            """,
            "Incompatible OpenGL resource register metadata for 'samp': "
            "register u2 uses u-register, expected s-register for sampler binding",
        ),
    ],
)
def test_glsl_resource_register_metadata_rejects_wrong_register_class(code, message):
    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")


def test_glsl_compiler_set_zero_buffer_pointer_resource_lowers_to_ssbo():
    # Reduced from CrossGL-Compiler tests/fixtures/StorageBufferComputeShader.cgl.
    code = """
    shader StorageBufferComputeShader {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            layout(set = 0, binding = 0) buffer float* values;

            void main() {
                float x = 1.0 + 2.0;
                values[0] = x;
                return;
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert (
        "layout(std430, binding = 0) buffer valuesBuffer { float values[]; };"
        in generated
    )
    assert "values[0] = x;" in generated
    assert "PointerType" not in generated


def test_glsl_buffer_pointer_helper_parameters_lower_to_array_parameters():
    code = """
    shader BufferPointerHelperParameter {
        float readValue(buffer float* values, int index) {
            return values[index];
        }

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer float* values;

            void main() {
                float value = readValue(values, 0);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert (
        "layout(std430, binding = 0) buffer valuesBuffer { float values[]; };"
        in generated
    )
    assert "float readValue(float values[], int index)" in generated
    assert "float value = readValue(values, 0);" in generated
    assert "PointerType" not in generated


def test_glsl_descriptor_sets_flatten_to_opengl_bindings():
    # Mirrors the compiler OpenGL lowering policy: set N, binding M -> N * 1024 + M.
    code = """
    shader DescriptorSetFlatteningGLSL {
        struct Params {
            vec4 color;
        };

        layout(set = 0, binding = 0) uniform Params frameParams;
        layout(set = 1, binding = 0) uniform Params materialParams;
        layout(set = 0, binding = 1) buffer float* values;
        layout(set = 2, binding = 1) buffer float* history;
        layout(set = 0, binding = 4) uniform sampler2D baseMap;
        layout(set = 1, binding = 4) uniform sampler2D detailMap;

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                values[0] = history[0];
                return texture(baseMap, uv) + texture(detailMap, uv)
                    + frameParams.color + materialParams.color;
            }
        }
    }
    """

    generated = GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")

    assert "layout(std140, binding = 0) uniform Params {" in generated
    assert "} frameParams;" in generated
    assert "layout(std140, binding = 1024) uniform Params {" in generated
    assert "} materialParams;" in generated
    assert (
        "layout(std430, binding = 1) buffer valuesBuffer { float values[]; };"
        in generated
    )
    assert (
        "layout(std430, binding = 2049) buffer historyBuffer { float history[]; };"
        in generated
    )
    assert "layout(binding = 4) uniform sampler2D baseMap;" in generated
    assert "layout(binding = 1028) uniform sampler2D detailMap;" in generated


def test_glsl_descriptor_array_texture_bindings_remap_overlapping_target_slots():
    code = """
    shader MixedTextureCompareDescriptorArrayShader {
        layout(set = 0, binding = 2) uniform sampler2D colorMaps[2];
        layout(set = 0, binding = 3) uniform sampler2DShadow shadowMaps[2];

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                float shadow = textureCompare(shadowMaps[1], uv, 0.5);
                return texture(colorMaps[1], uv) + vec4(shadow);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")

    assert "layout(binding = 2) uniform sampler2D colorMaps[2];" in generated
    assert "layout(binding = 4) uniform sampler2DShadow shadowMaps[2];" in generated
    assert "texture(colorMaps[1], uv)" in generated
    assert "texture(shadowMaps[1], vec3(uv, 0.5))" in generated
    assert "layout(binding = 3) uniform sampler2DShadow shadowMaps[2];" not in generated


def test_glsl_descriptor_array_storage_image_atomic_bindings_remap_overlapping_target_slots():
    code = """
    shader StorageImageAtomicDescriptorArrayShader {
        layout(set = 0, binding = 1, format = r32i) readwrite uniform iimage2D signedCounters[2];
        layout(set = 0, binding = 2, format = r32ui) readwrite uniform uimage2D unsignedCounters[2];

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 0);
                int signedValue = imageAtomicAdd(signedCounters[1], pixel, 1);
                uint unsignedValue = imageAtomicAdd(unsignedCounters[1], pixel, 1u);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "compute")

    assert "layout(r32i, binding = 1) uniform iimage2D signedCounters[2];" in generated
    assert (
        "layout(r32ui, binding = 3) uniform uimage2D unsignedCounters[2];" in generated
    )
    assert "imageAtomicAdd(signedCounters[1], pixel, 1)" in generated
    assert "imageAtomicAdd(unsignedCounters[1], pixel, 1u)" in generated
    assert (
        "layout(r32ui, binding = 2) uniform uimage2D unsignedCounters[2];"
        not in generated
    )


def test_glsl_descriptor_array_storage_image_format_bindings_remap_overlapping_target_slots():
    code = """
    shader StorageImageExplicitFormatDescriptorArrayShader {
        layout(set = 0, binding = 0, format = rgba32f) readwrite uniform image2D colorImages[2];
        layout(set = 0, binding = 1, format = r32ui) readwrite uniform uimage2D labelImages[2];

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 0);
                vec4 color = imageLoad(colorImages[1], pixel);
                uint label = imageLoad(labelImages[1], pixel);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "compute")

    assert "layout(rgba32f, binding = 0) uniform image2D colorImages[2];" in generated
    assert "layout(r32ui, binding = 2) uniform uimage2D labelImages[2];" in generated
    assert "vec4 color = imageLoad(colorImages[1], pixel);" in generated
    assert "uint label = imageLoad(labelImages[1], pixel).x;" in generated
    assert (
        "layout(r32ui, binding = 1) uniform uimage2D labelImages[2];" not in generated
    )


def test_glsl_descriptor_set_flattening_is_attribute_order_independent():
    code = """
    shader DescriptorSetOrderIndependentGLSL {
        sampler2D tex @binding(2) @set(1);

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                return texture(tex, uv);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")

    assert "layout(binding = 1026) uniform sampler2D tex;" in generated


def test_glsl_register_space_flattens_to_opengl_binding():
    code = """
    shader RegisterSpaceFlatteningGLSL {
        image2D img @space(2) @binding(3);

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                return imageLoad(img, ivec2(0, 0));
            }
        }
    }
    """

    generated = GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")

    assert "layout(rgba32f, binding = 2051) uniform image2D img;" in generated


def test_glsl_descriptor_set_register_alias_flattens_to_opengl_binding():
    code = """
    shader DescriptorSetRegisterAliasGLSL {
        struct Params {
            vec4 scale;
        };

        layout(set = 1, register = 2) uniform Params materialParams;

        fragment {
            vec4 main() @gl_FragColor {
                return materialParams.scale;
            }
        }
    }
    """

    generated = GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")

    assert "layout(std140, binding = 1026) uniform Params {" in generated
    assert "} materialParams;" in generated


def test_glsl_stage_resource_descriptor_set_flattens_to_opengl_binding():
    code = """
    shader StageResourceUsesDescriptorSet {
        fragment {
            vec4 main(sampler2D tex @set(1) @binding(2)) @gl_FragColor {
                return texture(tex, vec2(0.0));
            }
        }
    }
    """

    generated = GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")

    assert "layout(binding = 1026) uniform sampler2D tex;" in generated


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (
            """
            shader DuplicateTextureBinding {
                sampler2D tex @binding(1) @binding(1);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return texture(tex, uv);
                    }
                }
            }
            """,
            "Duplicate OpenGL resource binding metadata for 'tex': binding 1",
        ),
        (
            """
            shader DuplicateStageTextureBinding {
                fragment {
                    vec4 main(sampler2D tex @texture(2) @texture(2))
                        @gl_FragColor
                    {
                        return texture(tex, vec2(0.0));
                    }
                }
            }
            """,
            "Duplicate OpenGL resource binding metadata for 'tex': "
            "texture 2 binding 2",
        ),
        (
            """
            shader DuplicateStructuredBufferRegister {
                RWStructuredBuffer<int> values[2] @register(u3) @register(u3);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return vec4(float(buffer_load(values[1], 0)));
                    }
                }
            }
            """,
            "Duplicate OpenGL resource binding metadata for 'values': "
            "register u3 binding 3",
        ),
        (
            """
            shader DuplicateCBufferBinding {
                cbuffer Constants @binding(4) @binding(4) {
                    vec4 tint;
                };

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return tint;
                    }
                }
            }
            """,
            "Duplicate OpenGL resource binding metadata for 'Constants': binding 4",
        ),
    ],
)
def test_glsl_resource_binding_aliases_reject_duplicate_metadata(code, message):
    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (
            """
            shader ConflictingTextureBindings {
                sampler2D tex @binding(1) @register(t2);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return texture(tex, uv);
                    }
                }
            }
            """,
            "Conflicting OpenGL resource binding metadata for 'tex': "
            "binding 1 differs from register t2 binding 2",
        ),
        (
            """
            shader ConflictingImageBindings {
                image2D img @binding(2) @register(u3);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return imageLoad(img, ivec2(0, 0));
                    }
                }
            }
            """,
            "Conflicting OpenGL resource binding metadata for 'img': "
            "binding 2 differs from register u3 binding 3",
        ),
        (
            """
            shader ConflictingStructuredBufferBindings {
                RWStructuredBuffer<int> values @binding(3) @register(u4);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return vec4(float(buffer_load(values, 0)));
                    }
                }
            }
            """,
            "Conflicting OpenGL resource binding metadata for 'values': "
            "binding 3 differs from register u4 binding 4",
        ),
        (
            """
            shader ConflictingCBufferBindings {
                cbuffer Constants @binding(4) @register(b5) {
                    vec4 tint;
                };

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return tint;
                    }
                }
            }
            """,
            "Conflicting OpenGL resource binding metadata for 'Constants': "
            "binding 4 differs from register b5 binding 5",
        ),
        (
            """
            shader ConflictingStageResourceBindings {
                fragment {
                    vec4 main(sampler2D tex @binding(1) @register(t2))
                        @gl_FragColor
                    {
                        return texture(tex, vec2(0.0));
                    }
                }
            }
            """,
            "Conflicting OpenGL resource binding metadata for 'tex': "
            "binding 1 differs from register t2 binding 2",
        ),
    ],
)
def test_glsl_resource_binding_aliases_reject_conflicting_metadata(code, message):
    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (
            """
            shader InvalidTextureBindingOperand {
                sampler2D tex @binding(slot);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return texture(tex, uv);
                    }
                }
            }
            """,
            "Invalid OpenGL resource binding metadata for 'tex': "
            "binding slot must resolve to a concrete integer binding",
        ),
        (
            """
            shader InvalidTextureRegisterOperand {
                sampler2D tex @register(q2);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return texture(tex, uv);
                    }
                }
            }
            """,
            "Invalid OpenGL resource binding metadata for 'tex': "
            "register q2 must use b/s/t/u register syntax or an integer binding",
        ),
        (
            """
            shader InvalidCBufferBindingOperand {
                cbuffer Constants @binding(slot) {
                    vec4 tint;
                };

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return tint;
                    }
                }
            }
            """,
            "Invalid OpenGL resource binding metadata for 'Constants': "
            "binding slot must resolve to a concrete integer binding",
        ),
        (
            """
            shader InvalidStructuredBufferRegisterOperand {
                RWStructuredBuffer<int> values @register(q4);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return vec4(float(buffer_load(values, 0)));
                    }
                }
            }
            """,
            "Invalid OpenGL resource binding metadata for 'values': "
            "register q4 must use b/s/t/u register syntax or an integer binding",
        ),
        (
            """
            shader InvalidStageResourceBindingOperand {
                fragment {
                    vec4 main(sampler2D tex @texture(slot)) @gl_FragColor {
                        return texture(tex, vec2(0.0));
                    }
                }
            }
            """,
            "Invalid OpenGL resource binding metadata for 'tex': "
            "texture slot must resolve to a concrete integer binding",
        ),
        (
            """
            shader InvalidDescriptorSetOperand {
                sampler2D tex @set(space) @binding(2);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return texture(tex, uv);
                    }
                }
            }
            """,
            "Invalid OpenGL resource binding metadata for 'tex': "
            "set space must resolve to a concrete integer binding",
        ),
    ],
)
def test_glsl_resource_binding_operands_reject_non_integer_metadata(code, message):
    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")


def test_glsl_stage_local_resources_emit_global_layout_bindings():
    code = """
    shader StageLocalResourcesGLSL {
        struct LocalData {
            int value;
        };

        fragment {
            uniform sampler2D localTex @binding(2);
            uniform image2D localImage @binding(5);
            LocalData localData @glsl_buffer_block(std430) @binding(8);

            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                vec4 stored = imageLoad(localImage, ivec2(0, 0));
                return texture(localTex, uv) + stored + vec4(float(localData.value));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(binding = 2) uniform sampler2D localTex;" in generated_code
    assert "layout(rgba32f, binding = 5) uniform image2D localImage;" in generated_code
    assert "layout(std430, binding = 8) buffer LocalData" in generated_code
    assert "texture(localTex, uv)" in generated_code
    assert "float(localData.value)" in generated_code


def test_glsl_stage_resource_parameters_emit_global_layout_bindings():
    code = """
    shader StageParameterResourceBindings {
        fragment {
            vec4 main(
                sampler2D tex @binding(1),
                image2D img @binding(3),
                RWStructuredBuffer<int> values @binding(5)
            ) @gl_FragColor {
                return texture(tex, vec2(0.0)) +
                    imageLoad(img, ivec2(0, 0)) +
                    vec4(float(buffer_load(values, 0)));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(binding = 1) uniform sampler2D tex;" in generated_code
    assert "layout(rgba32f, binding = 3) uniform image2D img;" in generated_code
    assert "layout(std430, binding = 5) buffer valuesBuffer" in generated_code
    assert "void main()" in generated_code
    assert "sampler2D tex)" not in generated_code


def test_glsl_stage_resource_parameter_binding_overlap_raises():
    code = """
    shader StageParameterResourceOverlap {
        sampler2D globalTex @binding(1);

        fragment {
            vec4 main(sampler2D textures[2] @binding(0)) @gl_FragColor {
                return texture(textures[1], vec2(0.0)) +
                    texture(globalTex, vec2(0.0));
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting OpenGL resource binding for 'textures': "
            "texture binding 0-1 overlaps 'globalTex' texture binding 1"
        ),
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")


def test_glsl_stage_resource_parameter_auto_binding_skips_explicit_global():
    code = """
    shader StageParameterResourceAutoBinding {
        sampler2D explicitTex @binding(0);

        fragment {
            vec4 main(sampler2D localTex) @gl_FragColor {
                return texture(localTex, vec2(0.0)) +
                    texture(explicitTex, vec2(0.0));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D explicitTex;" in generated_code
    assert "layout(binding = 1) uniform sampler2D localTex;" in generated_code


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (
            """
            shader DuplicateStageInputLocation {
                fragment {
                    vec4 main(vec2 uv @location(0) @location(0)) @gl_FragColor {
                        return vec4(uv, 0.0, 1.0);
                    }
                }
            }
            """,
            "Duplicate OpenGL layout metadata for 'uv': location = 0",
        ),
        (
            """
            shader DuplicateFragmentOutputComponent {
                struct PSOutput {
                    float luminance @location(0) @component(0) @component(0);
                };

                fragment {
                    PSOutput main(vec2 uv @TEXCOORD0) {
                        PSOutput output;
                        output.luminance = uv.x;
                        return output;
                    }
                }
            }
            """,
            "Duplicate OpenGL layout metadata for 'luminance': component = 0",
        ),
    ],
)
def test_glsl_stage_io_layout_metadata_rejects_duplicate_metadata(code, message):
    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")


def test_glsl_shared_stage_resource_parameter_emits_one_global_binding():
    code = """
    shader SharedStageParameterResource {
        vertex {
            void main(sampler2D tex) {
                vec4 color = texture(tex, vec2(0.0));
            }
        }

        fragment {
            vec4 main(sampler2D tex) @gl_FragColor {
                return texture(tex, vec2(0.0));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert generated_code.count("layout(binding = 0) uniform sampler2D tex;") == 1


def test_glsl_fragment_value_parameter_emits_stage_input_declaration():
    code = """
    shader FragmentValueParameterInput {
        sampler2D colorMap;

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                return texture(colorMap, uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(location = 5) in vec2 uv;" in generated_code
    assert "fragColor = texture(colorMap, uv);" in generated_code


def test_glsl_vertex_value_parameter_emits_stage_input_declaration():
    code = """
    shader VertexValueParameterInput {
        vertex {
            void main(vec3 position @POSITION) {
                gl_Position = vec4(position, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "vertex"
    )

    assert "layout(location = 0) in vec3 position;" in generated_code
    assert "gl_Position = vec4(position, 1.0);" in generated_code


def test_glsl_stage_interface_interpolation_precision_qualifiers():
    code = """
    shader FragmentQualifierSmoke {
        flat centroid in vec2 inputUv @location(1) @highp;
        noperspective sample out vec4 fragColor
            @location(0)
            @invariant
            @precise
            @mediump;

        fragment {
            void main() {
                fragColor = vec4(inputUv, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(location = 1) flat centroid in highp vec2 inputUv;" in (
        generated_code
    )
    assert (
        "layout(location = 0) invariant precise noperspective sample "
        "out mediump vec4 fragColor;" in generated_code
    )


def test_glsl_global_fragment_integer_input_adds_required_flat_qualifier():
    # Reduced from KhronosGroup/glslang Test/spv.nonuniform.frag, where
    # fragment-stage integer varyings are declared with flat interpolation.
    code = """
    shader FragmentIntegerInput {
        in int primitiveId @location(1);
        out vec4 fragColor @location(0);

        fragment {
            void main() {
                fragColor = vec4(float(primitiveId));
            }
        }
    }
    """

    combined_code = GLSLCodeGen().generate(crosstl.translator.parse(code))
    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(location = 1) flat in int primitiveId;" in combined_code
    assert "layout(location = 1) flat in int primitiveId;" in fragment_code


def test_glsl_stage_interface_extended_layout_qualifiers():
    code = """
    shader LayoutQualifierSmoke {
        in vec2 inputUv[] @location(0) @component(1);
        out vec2 outUv
            @location(1)
            @stream(0)
            @xfb_buffer(0)
            @xfb_offset(0);
        out vec4 fragColor @location(0) @index(1);

        geometry {
            layout(points) in;
            layout(points, max_vertices = 1) out;

            void main() {
                outUv = inputUv[0];
                EmitVertex();
            }
        }

        fragment {
            void main() {
                fragColor = vec4(1.0);
            }
        }
    }
    """

    ast = crosstl.translator.parse(code)
    geometry_code = GLSLCodeGen().generate_stage(ast, "geometry")
    fragment_code = GLSLCodeGen().generate_stage(ast, "fragment")

    assert "layout(location = 0, component = 1) in vec2 inputUv[];" in geometry_code
    assert (
        "layout(location = 1, stream = 0, xfb_buffer = 0, xfb_offset = 0) "
        "out vec2 outUv;" in geometry_code
    )
    assert "layout(location = 0, index = 1) out vec4 fragColor;" in fragment_code


def test_glsl_stage_interface_unsized_array_suffix_locations_are_reserved():
    conflict = """
    shader TessellationEvaluationInterfaceLocationConflict {
        in vec3 tcGrid[][2] @location(0);
        in vec3 other[] @location(1);

        tessellation_evaluation {
            void main() @domain(quad) @partitioning(equal) @ccw {
                gl_Position = vec4(tcGrid[0][0] + other[0], 1.0);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="locations 0-1"):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(conflict), "tessellation_evaluation"
        )

    non_conflicting = conflict.replace("@location(1)", "@location(2)")
    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(non_conflicting), "tessellation_evaluation"
    )

    assert "layout(location = 0) in vec3 tcGrid[][2];" in generated_code
    assert "layout(location = 2) in vec3 other[];" in generated_code


def test_glsl_fragment_blend_support_layout_qualifiers():
    code = """
    shader AdvancedBlendOutput {
        out vec4 outputColour
            @location(0)
            @blend_support_colordodge
            @highp;

        fragment {
            layout(blend_support_multiply, blend_support_screen) out;

            void main() {
                outputColour = vec4(1.0);
            }
        }
    }
    """

    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert (
        "layout(blend_support_colordodge, blend_support_multiply, "
        "blend_support_screen) out;" in fragment_code
    )
    assert "layout(location = 0) out highp vec4 outputColour;" in fragment_code
    assert "outputColour = vec4(1.0);" in fragment_code


def test_glsl_fragment_component_packed_output_layouts():
    code = """
    shader ComponentPackedOutput {
        out float luminance @location(0) @component(0);
        out vec2 velocity @location(0) @component(1);
        out float coverage @location(0) @component(3);

        fragment {
            void main() {
                luminance = 1.0;
                velocity = vec2(0.5);
                coverage = 0.25;
            }
        }
    }
    """

    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(location = 0, component = 0) out float luminance;" in fragment_code
    assert "layout(location = 0, component = 1) out vec2 velocity;" in fragment_code
    assert "layout(location = 0, component = 3) out float coverage;" in fragment_code
    assert "luminance = 1.0;" in fragment_code
    assert "velocity = vec2(0.5);" in fragment_code
    assert "coverage = 0.25;" in fragment_code
    assert "fragColor" not in fragment_code


def test_glsl_fragment_component_packed_output_overlap_raises():
    code = """
    shader ComponentPackedOutputOverlap {
        out float luminance @location(0) @component(0);
        out vec2 velocity @location(0) @component(0);

        fragment {
            void main() {
                luminance = 1.0;
                velocity = vec2(0.5);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting OpenGL fragment output location for 'velocity': "
            "location 0 components 0-1 overlaps 'luminance' "
            "location 0 component 0"
        ),
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")


def test_glsl_fragment_component_packed_outputs_reserve_auto_locations():
    code = """
    shader ComponentPackedOutputAutoLocation {
        struct PSOutput {
            float luminance @location(0) @component(0);
            vec2 velocity @location(0) @component(1);
            vec4 color;
        };

        fragment {
            PSOutput main() {
                PSOutput output;
                output.luminance = 1.0;
                output.velocity = vec2(0.5);
                output.color = vec4(0.25);
                return output;
            }
        }
    }
    """

    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(location = 0, component = 0) out float luminance;" in fragment_code
    assert "layout(location = 0, component = 1) out vec2 velocity;" in fragment_code
    assert "layout(location = 1) out vec4 color;" in fragment_code


def test_glsl_fragment_indexed_outputs_reject_same_index_location_overlap():
    code = """
    shader IndexedOutputOverlap {
        out vec4 accum @location(0) @index(1);
        out vec4 revealage @location(0) @index(1);

        fragment {
            void main() {
                accum = vec4(1.0);
                revealage = vec4(0.5);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting OpenGL fragment output location for 'revealage': "
            "location 0 components 0-3 overlaps 'accum' "
            "location 0 components 0-3"
        ),
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")


def test_glsl_fragment_indexed_outputs_allow_different_indices():
    code = """
    shader IndexedOutputValid {
        out vec4 accum @location(0) @index(0);
        out vec4 revealage @location(0) @index(1);

        fragment {
            void main() {
                accum = vec4(1.0);
                revealage = vec4(0.5);
            }
        }
    }
    """

    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(location = 0, index = 0) out vec4 accum;" in fragment_code
    assert "layout(location = 0, index = 1) out vec4 revealage;" in fragment_code


def test_glsl_fragment_indexed_outputs_allow_default_and_secondary_index():
    code = """
    shader IndexedOutputDefaultValid {
        out vec4 accum @location(0);
        out vec4 revealage @location(0) @index(1);

        fragment {
            void main() {
                accum = vec4(1.0);
                revealage = vec4(0.5);
            }
        }
    }
    """

    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(location = 0) out vec4 accum;" in fragment_code
    assert "layout(location = 0, index = 1) out vec4 revealage;" in fragment_code


def test_glsl_fragment_sample_mask_output_aliases_to_builtin():
    code = """
    shader SampleMaskOutput {
        out int sampleMask @gl_SampleMask;

        fragment {
            void main() {
                sampleMask = 1;
            }
        }
    }
    """

    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "gl_SampleMask[0] = 1;" in fragment_code
    assert "out int sampleMask;" not in fragment_code
    assert "layout(location = 0) out int sampleMask;" not in fragment_code
    assert "fragColor" not in fragment_code


def test_glsl_fragment_sample_mask_return_aliases_to_builtin():
    code = """
    shader SampleMaskReturn {
        fragment {
            int main() @SV_Coverage {
                return 1;
            }
        }
    }
    """

    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "gl_SampleMask[0] = 1;" in fragment_code
    assert "return;" in fragment_code
    assert "fragColor" not in fragment_code
    assert "layout(location = 0)" not in fragment_code


@pytest.mark.parametrize(
    "depth_layout",
    ["depth_any", "depth_greater", "depth_less", "depth_unchanged"],
)
def test_glsl_conservative_depth_output_layout(depth_layout):
    code = f"""
    shader ConservativeDepthOutput {{
        out float gl_FragDepth @{depth_layout};

        fragment {{
            void main() {{
                gl_FragDepth = 0.5;
            }}
        }}
    }}
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert f"layout({depth_layout}) out float gl_FragDepth;" in generated_code
    assert "\nout float gl_FragDepth;" not in generated_code
    assert "gl_FragDepth = 0.5;" in generated_code


def test_glsl_stage_builtin_parameter_aliases_to_glsl_builtin():
    code = """
    shader StageBuiltinParameterAlias {
        fragment {
            vec4 main(vec4 coord @gl_FragCoord) @gl_FragColor {
                return coord;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "in vec4 coord;" not in generated_code
    assert "fragColor = gl_FragCoord;" in generated_code


def test_glsl_fragment_struct_builtin_input_aliases_to_glsl_builtin():
    code = """
    shader StructFragmentBuiltinInput {
        struct FSInput {
            bool frontFacing @gl_FrontFacing;
            vec2 uv @TEXCOORD0;
        };

        fragment {
            vec4 main(FSInput input) @gl_FragColor {
                return input.frontFacing ? vec4(input.uv, 0.0, 1.0) : vec4(0.0);
            }
        }
    }
    """

    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "in bool frontFacing;" not in fragment_code
    assert "flat in bool frontFacing;" not in fragment_code
    assert "layout(location = 5) in vec2 uv;" in fragment_code
    assert (
        "fragColor = (gl_FrontFacing ? vec4(uv, 0.0, 1.0) : vec4(0.0));"
        in fragment_code
    )
    assert "input.frontFacing" not in fragment_code


def test_glsl_fragment_sample_builtin_parameter_aliases_to_inputs():
    code = """
    shader FragmentSampleBuiltinAliases {
        fragment {
            vec4 main(
                int sampleIndex @SV_SampleIndex,
                int sampleId @sample_id,
                vec2 samplePosition @sample_position
            ) @gl_FragColor {
                return vec4(
                    float(sampleIndex + sampleId),
                    samplePosition.x,
                    samplePosition.y,
                    1.0
                );
            }
        }
    }
    """

    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "in int sampleIndex;" not in fragment_code
    assert "in int sampleId;" not in fragment_code
    assert "in vec2 samplePosition;" not in fragment_code
    assert (
        "fragColor = vec4(float((gl_SampleID + gl_SampleID)), "
        "gl_SamplePosition.x, gl_SamplePosition.y, 1.0);"
    ) in fragment_code


def test_glsl_fragment_coverage_input_aliases_to_sample_mask_in():
    code = """
    shader FragmentCoverageInputAlias {
        fragment {
            vec4 main(int coverage @SV_Coverage) @gl_FragColor {
                return vec4(float(coverage));
            }
        }
    }
    """

    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "in int coverage;" not in fragment_code
    assert "fragColor = vec4(float(gl_SampleMaskIn[0]));" in fragment_code
    assert "float(gl_SampleMask)" not in fragment_code


def test_glsl_fragment_interpolation_helper_aliases_to_builtins():
    code = """
    shader FragmentInterpolationHelpers {
        fragment {
            vec4 main(
                vec4 color @location(0) @sample,
                vec2 offset @location(1),
                int sampleIndex @SV_SampleIndex
            ) @gl_FragColor {
                vec4 sampleColor = interpolate_at_sample(color, sampleIndex);
                vec4 offsetColor = interpolate_at_offset(color, offset);
                vec4 centroidColor = interpolate_at_centroid(color);
                vec4 directSample = interpolateAtSample(color, 0);
                vec4 directOffset = interpolateAtOffset(color, vec2(0.25));
                vec4 directCentroid = interpolateAtCentroid(color);
                return sampleColor
                    + offsetColor
                    + centroidColor
                    + directSample
                    + directOffset
                    + directCentroid;
            }
        }
    }
    """

    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(location = 0) sample in vec4 color;" in fragment_code
    assert "layout(location = 1) in vec2 offset;" in fragment_code
    assert "interpolateAtSample(color, gl_SampleID)" in fragment_code
    assert "interpolateAtOffset(color, offset)" in fragment_code
    assert "interpolateAtCentroid(color)" in fragment_code
    assert "interpolateAtSample(color, 0)" in fragment_code
    assert "interpolateAtOffset(color, vec2(0.25))" in fragment_code
    assert "interpolate_at_" not in fragment_code


def test_glsl_fragment_interpolation_helper_arguments_are_validated():
    wrong_sample_type = """
    shader BadInterpolationSample {
        fragment {
            vec4 main(vec4 color @location(0), vec2 offset @location(1)) @gl_FragColor {
                return interpolate_at_sample(color, offset);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match=(
            "OpenGL interpolation operation 'interpolateAtSample' "
            "requires a scalar integer sample argument"
        ),
    ):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(wrong_sample_type), "fragment"
        )

    wrong_offset_type = """
    shader BadInterpolationOffset {
        fragment {
            vec4 main(
                vec4 color @location(0),
                int sampleIndex @SV_SampleIndex
            ) @gl_FragColor {
                return interpolate_at_offset(color, sampleIndex);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match=(
            "OpenGL interpolation operation 'interpolateAtOffset' "
            "requires a vec2 floating offset argument"
        ),
    ):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(wrong_offset_type), "fragment"
        )

    missing_sample_arg = """
    shader MissingInterpolationSampleArg {
        fragment {
            vec4 main(vec4 color @location(0)) @gl_FragColor {
                return interpolate_at_sample(color);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="OpenGL interpolation operation 'interpolateAtSample' requires 2 arguments",
    ):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(missing_sample_arg), "fragment"
        )

    vertex_stage_use = """
    shader BadInterpolationStage {
        vertex {
            void main(vec4 color @location(0)) {
                gl_Position = interpolate_at_centroid(color);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match=(
            "OpenGL interpolation operation 'interpolateAtCentroid' "
            "is only valid in fragment stages"
        ),
    ):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(vertex_stage_use), "vertex"
        )


def test_glsl_fragment_derivative_helper_aliases_to_builtins():
    code = """
    shader FragmentDerivativeHelpers {
        fragment {
            vec4 main(vec2 uv @location(0)) @gl_FragColor {
                float coarseX = ddx_coarse(uv.x);
                float coarseY = ddy_coarse(uv.y);
                float fineX = ddx_fine(uv.x);
                float fineY = ddy_fine(uv.y);
                float widthFine = fwidth_fine(uv.x);
                float widthCoarse = fwidth_coarse(uv.y);
                vec2 base = ddx(uv) + ddy(uv) + fwidth(uv);
                return vec4(
                    base.x + coarseX + coarseY + fineX,
                    base.y + fineY + widthFine + widthCoarse,
                    dFdx(uv.x),
                    dFdy(uv.y)
                );
            }
        }
    }
    """

    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "float coarseX = dFdxCoarse(uv.x);" in fragment_code
    assert "float coarseY = dFdyCoarse(uv.y);" in fragment_code
    assert "float fineX = dFdxFine(uv.x);" in fragment_code
    assert "float fineY = dFdyFine(uv.y);" in fragment_code
    assert "float widthFine = fwidthFine(uv.x);" in fragment_code
    assert "float widthCoarse = fwidthCoarse(uv.y);" in fragment_code
    assert "vec2 base = ((dFdx(uv) + dFdy(uv)) + fwidth(uv));" in fragment_code
    assert "dFdx(uv.x)" in fragment_code
    assert "dFdy(uv.y)" in fragment_code
    assert "ddx(" not in fragment_code
    assert "ddy(" not in fragment_code
    assert "ddx_" not in fragment_code
    assert "ddy_" not in fragment_code
    assert "fwidth_" not in fragment_code


def test_glsl_fragment_derivative_helpers_forward_through_stage_helpers_and_gradients():
    code = """
    shader FragmentDerivativeHelperForwarding {
        sampler2D colorMap @binding(0);

        fragment {
            vec2 gradientX(vec2 uv) {
                return ddx(uv);
            }

            vec4 sampleWithGradient(vec2 uv) {
                return textureGrad(colorMap, uv, gradientX(uv), ddy(uv));
            }

            vec4 main(vec2 uv @location(0)) @gl_FragColor {
                return sampleWithGradient(uv) + vec4(fwidth_fine(uv), 0.0, 1.0);
            }
        }
    }
    """

    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "vec2 gradientX(vec2 uv)" in fragment_code
    assert "return dFdx(uv);" in fragment_code
    assert "textureGrad(colorMap, uv, gradientX(uv), dFdy(uv))" in fragment_code
    assert "vec4(fwidthFine(uv), 0.0, 1.0)" in fragment_code
    assert "ddx(" not in fragment_code
    assert "ddy(" not in fragment_code
    assert "fwidth_fine" not in fragment_code


def test_glsl_fragment_derivative_helpers_forward_through_top_level_helpers():
    code = """
    shader FragmentTopLevelDerivativeHelperForwarding {
        sampler2D colorMap @binding(0);

        vec2 wrappedGradientX(vec2 uv) {
            return gradientX(uv);
        }

        fragment {
            vec2 gradientX(vec2 uv) {
                return ddx(uv);
            }

            vec4 main(vec2 uv @location(0)) @gl_FragColor {
                return textureGrad(colorMap, uv, wrappedGradientX(uv), ddy(uv));
            }
        }
    }
    """

    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "vec2 gradientX(vec2 uv)" in fragment_code
    assert "return dFdx(uv);" in fragment_code
    assert "vec2 wrappedGradientX(vec2 uv)" in fragment_code
    assert "return gradientX(uv);" in fragment_code
    assert fragment_code.index("vec2 gradientX(vec2 uv)") < fragment_code.index(
        "vec2 wrappedGradientX(vec2 uv)"
    )
    assert "textureGrad(colorMap, uv, wrappedGradientX(uv), dFdy(uv))" in fragment_code
    assert "ddx(" not in fragment_code
    assert "ddy(" not in fragment_code


def test_glsl_fragment_derivative_helper_arguments_are_validated():
    wrong_value_type = """
    shader BadDerivativeValueType {
        fragment {
            vec4 main(int sampleIndex @SV_SampleIndex) @gl_FragColor {
                return vec4(ddx(sampleIndex));
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match=(
            "OpenGL derivative operation 'dFdx' "
            "requires a floating scalar or vector value argument"
        ),
    ):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(wrong_value_type), "fragment"
        )

    missing_arg = """
    shader BadDerivativeArgCount {
        fragment {
            vec4 main(vec2 uv @location(0)) @gl_FragColor {
                return vec4(ddx_fine());
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="OpenGL derivative operation 'dFdxFine' requires 1 argument",
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(missing_arg), "fragment")

    vertex_stage_use = """
    shader BadDerivativeStage {
        vertex {
            void main(vec4 position @location(0)) {
                gl_Position = vec4(ddx(position.x));
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match=(
            "OpenGL derivative operation 'dFdx' " "is only valid in fragment stages"
        ),
    ):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(vertex_stage_use), "vertex"
        )

    vertex_stage_helper_use = """
    shader BadDerivativeHelperStage {
        vertex {
            float slope(vec3 position) {
                return ddx(position.x);
            }

            void main(vec3 position @POSITION) {
                gl_Position = vec4(position, slope(position));
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match=(
            "OpenGL derivative operation 'dFdx' " "is only valid in fragment stages"
        ),
    ):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(vertex_stage_helper_use), "vertex"
        )

    vertex_top_level_helper_use = """
    shader BadTopLevelDerivativeHelperStage {
        float slope(float value) {
            return ddx(value);
        }

        float wrappedSlope(float value) {
            return slope(value);
        }

        vertex {
            void main(vec3 position @POSITION) {
                gl_Position = vec4(position, wrappedSlope(position.x));
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match=(
            "OpenGL helper function 'wrappedSlope' uses fragment-only operations "
            "and is only valid in fragment stages"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(vertex_top_level_helper_use))


def test_glsl_graphics_builtin_parameter_types_are_validated():
    float_vertex_id_code = """
    shader BadVertexBuiltinType {
        vertex {
            void main(float vertexId @gl_VertexID) {
                gl_Position = vec4(0.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_VertexID.*scalar int"):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(float_vertex_id_code), "vertex"
        )

    vector_primitive_id_code = """
    shader BadPrimitiveBuiltinType {
        fragment {
            vec4 main(ivec2 primitiveId @gl_PrimitiveID) @gl_FragColor {
                return vec4(float(primitiveId.x));
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_PrimitiveID.*scalar int"):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(vector_primitive_id_code), "fragment"
        )

    int_front_facing_code = """
    shader BadFrontFacingBuiltinType {
        fragment {
            vec4 main(int frontFacing @gl_FrontFacing) @gl_FragColor {
                return vec4(float(frontFacing));
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_FrontFacing.*scalar bool"):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(int_front_facing_code), "fragment"
        )

    scalar_frag_coord_code = """
    shader BadFragCoordBuiltinType {
        fragment {
            vec4 main(float coord @gl_FragCoord) @gl_FragColor {
                return vec4(coord);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_FragCoord.*vec4"):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(scalar_frag_coord_code), "fragment"
        )

    vec3_point_coord_code = """
    shader BadPointCoordBuiltinType {
        fragment {
            vec4 main(vec3 coord @gl_PointCoord) @gl_FragColor {
                return vec4(coord, 1.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_PointCoord.*vec2"):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(vec3_point_coord_code), "fragment"
        )

    float_sample_id_code = """
    shader BadSampleIdBuiltinType {
        fragment {
            vec4 main(float sampleIndex @SV_SampleIndex) @gl_FragColor {
                return vec4(sampleIndex);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_SampleID.*scalar int"):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(float_sample_id_code), "fragment"
        )

    vec3_sample_position_code = """
    shader BadSamplePositionBuiltinType {
        fragment {
            vec4 main(vec3 samplePosition @sample_position) @gl_FragColor {
                return vec4(samplePosition, 1.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_SamplePosition.*vec2"):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(vec3_sample_position_code), "fragment"
        )

    vec2_coverage_code = """
    shader BadCoverageBuiltinType {
        fragment {
            vec4 main(vec2 coverage @SV_Coverage) @gl_FragColor {
                return vec4(coverage, 0.0, 1.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_SampleMaskIn.*scalar int"):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(vec2_coverage_code), "fragment"
        )

    valid_code = """
    shader ValidGraphicsBuiltins {
        vertex {
            void main(int vertexId @gl_VertexID, int instanceId @gl_InstanceID) {
                gl_Position = vec4(float(vertexId + instanceId));
            }
        }

        fragment {
            vec4 main(
                int primitiveId @gl_PrimitiveID,
                bool frontFacing @gl_FrontFacing,
                vec4 coord @gl_FragCoord,
                vec2 point @gl_PointCoord,
                int sampleIndex @SV_SampleIndex,
                vec2 samplePosition @gl_SamplePosition,
                int coverage @SV_Coverage
            ) @gl_FragColor {
                return frontFacing
                    ? (coord + vec4(
                        point + samplePosition,
                        float(primitiveId + sampleIndex + coverage),
                        1.0
                    ))
                    : vec4(0.0);
            }
        }
    }
    """
    vertex_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(valid_code), "vertex"
    )
    assert "void main()" in vertex_code
    assert "float((gl_VertexID + gl_InstanceID))" in vertex_code
    assert "vertexId @gl_VertexID" not in vertex_code

    fragment_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(valid_code), "fragment"
    )
    assert "void main()" in fragment_code
    assert (
        "fragColor = (gl_FrontFacing ? (gl_FragCoord + "
        "vec4((gl_PointCoord + gl_SamplePosition), "
        "float(((gl_PrimitiveID + gl_SampleID) + gl_SampleMaskIn[0])), "
        "1.0)) : vec4(0.0));"
    ) in fragment_code
    assert "primitiveId @gl_PrimitiveID" not in fragment_code
    assert "coord @gl_FragCoord" not in fragment_code
    assert "sampleIndex @SV_SampleIndex" not in fragment_code
    assert "coverage @SV_Coverage" not in fragment_code


@pytest.mark.parametrize(
    ("stage", "return_type", "semantic", "value"),
    [
        ("vertex", "int", "gl_VertexID", "0"),
        ("fragment", "vec4", "gl_FragCoord", "vec4(0.0)"),
        ("fragment", "bool", "gl_FrontFacing", "true"),
        ("fragment", "vec2", "gl_PointCoord", "vec2(0.0)"),
        ("fragment", "int", "SV_SampleIndex", "0"),
        ("fragment", "vec2", "gl_SamplePosition", "vec2(0.0)"),
        ("fragment", "int", "gl_SampleMaskIn", "0"),
        ("compute", "uvec3", "gl_GlobalInvocationID", "uvec3(0u)"),
    ],
)
def test_function_return_input_only_builtin_semantics_are_rejected(
    stage, return_type, semantic, value
):
    code = f"""
    shader BadReturnBuiltin {{
        {stage} {{
            {return_type} main() @{semantic} {{
                return {value};
            }}
        }}
    }}
    """
    with pytest.raises(ValueError, match=f"{semantic}.*return semantic"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), stage)


@pytest.mark.parametrize(
    ("stage", "semantic"),
    [
        ("vertex", "gl_Position"),
        ("fragment", "gl_FragColor"),
        ("fragment", "TEXCOORD0"),
    ],
)
def test_glsl_void_function_return_semantics_are_rejected(stage, semantic):
    code = f"""
    shader BadVoidReturnSemantic {{
        {stage} {{
            void main() @{semantic} {{ }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=f"{stage}.*{semantic}.*void return type"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), stage)


def test_glsl_compute_metal_max_total_threads_attribute_is_not_return_semantic():
    code = """
    shader MetalThreadgroupMetadataCompute {
        compute {
            void main() @max_total_threads_per_threadgroup(1024) {
                int value = 1;
            }
        }
    }
    """

    generated = GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "compute")

    assert (
        "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;" in generated
    )
    assert "return semantic 'max_total_threads_per_threadgroup'" not in generated
    assert "max_total_threads_per_threadgroup" not in generated


def test_glsl_geometry_inputtopology_metadata_lowers_to_input_layout():
    code = """
    shader GeometryInputTopology {
        geometry {
            void main()
                @inputtopology(triangle)
                @invocations(2)
                @outputtopology(line)
                @maxvertexcount(2)
            {
                gl_Position = gl_in[0].gl_Position;
                EmitVertex();
                EndPrimitive();
            }
        }
    }
    """

    geometry_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "geometry"
    )

    assert "layout(triangles, invocations = 2) in;" in geometry_code
    assert "layout(line_strip, max_vertices = 2) out;" in geometry_code
    assert "inputtopology" not in geometry_code
    assert "@inputtopology" not in geometry_code


def test_glsl_geometry_layout_counts_accept_positive_integer_constants():
    code = """
    shader GeometryIntegerLayout {
        const int INVOCATION_COUNT = 1 + 1;
        const int MAX_VERTEX_COUNT = 2 * 3;

        geometry {
            void main()
                @inputtopology(line_adjacency)
                @invocations(INVOCATION_COUNT)
                @outputtopology(triangle)
                @maxvertexcount(MAX_VERTEX_COUNT)
            {
                gl_Position = gl_in[0].gl_Position;
                EmitVertex();
                EndPrimitive();
            }
        }
    }
    """

    geometry_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "geometry"
    )

    assert (
        "layout(lines_adjacency, invocations = INVOCATION_COUNT) in;" in geometry_code
    )
    assert (
        "layout(triangle_strip, max_vertices = MAX_VERTEX_COUNT) out;" in geometry_code
    )


def test_glsl_geometry_inputtopology_rejects_output_only_topologies():
    code = """
    shader BadGeometryInputTopology {
        geometry {
            void main()
                @inputtopology(triangle_strip)
                @outputtopology(line)
                @maxvertexcount(2)
            { }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="GLSL geometry input topology cannot be lowered: triangle_strip",
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "geometry")


def test_glsl_geometry_outputtopology_rejects_input_only_topologies():
    code = """
    shader BadGeometryOutputTopology {
        geometry {
            void main()
                @inputtopology(line)
                @outputtopology(lines_adjacency)
                @maxvertexcount(2)
            { }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="GLSL geometry output topology cannot be lowered: lines_adjacency",
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "geometry")


@pytest.mark.parametrize(
    ("attribute_line", "message"),
    [
        (
            "@invocations(0)",
            "GLSL geometry invocations layout requires a positive integer constant, got 0",
        ),
        (
            "@invocations(-1)",
            r"GLSL geometry invocations layout requires a positive integer constant, got \(-1\)",
        ),
        (
            "@invocations(1.5)",
            r"GLSL geometry invocations layout requires a positive integer constant, got 1\.5",
        ),
        (
            "@max_vertices(0)",
            "GLSL geometry max_vertices layout requires a positive integer constant, got 0",
        ),
        (
            "@maxvertexcount(0)",
            "GLSL geometry max_vertices layout requires a positive integer constant, got 0",
        ),
        (
            "@maxvertexcount(2.5)",
            r"GLSL geometry max_vertices layout requires a positive integer constant, got 2\.5",
        ),
    ],
)
def test_glsl_geometry_layout_rejects_invalid_count_constants(attribute_line, message):
    fallback_max_vertices = (
        ""
        if attribute_line.startswith(("@max_vertices", "@maxvertexcount"))
        else "@maxvertexcount(2)"
    )
    code = f"""
    shader BadGeometryLayoutCount {{
        geometry {{
            void main()
                @inputtopology(line)
                @outputtopology(line)
                {fallback_max_vertices}
                {attribute_line}
            {{ }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "geometry")


@pytest.mark.parametrize(
    ("stage", "metadata", "message"),
    [
        (
            "geometry",
            "@triangles(1)\n                @outputtopology(line)\n                @maxvertexcount(2)",
            "GLSL stage attribute triangles does not accept arguments",
        ),
        (
            "geometry",
            "@inputtopology(line)\n                @line_strip(1)\n                @maxvertexcount(2)",
            "GLSL stage attribute line_strip does not accept arguments",
        ),
        (
            "tessellation_evaluation",
            "@domain(triangle)\n                @partitioning(equal)\n                @cw(1)",
            "GLSL stage attribute cw does not accept arguments",
        ),
        (
            "tessellation_evaluation",
            "@domain(triangle)\n                @equal_spacing(1)",
            "GLSL stage attribute equal_spacing does not accept arguments",
        ),
    ],
)
def test_glsl_stage_bare_layout_attributes_reject_arguments(stage, metadata, message):
    code = f"""
    shader BadBareStageLayoutOperand {{
        {stage} {{
            void main()
                {metadata}
            {{ }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), stage)


@pytest.mark.parametrize(
    ("stage", "metadata", "message"),
    [
        (
            "geometry",
            "@inputtopology(line)\n"
            "                @triangles\n"
            "                @outputtopology(line)\n"
            "                @maxvertexcount(2)",
            "Conflicting GLSL geometry input topology layout "
            "inputtopology line with triangles",
        ),
        (
            "geometry",
            "@inputtopology(line)\n"
            "                @outputtopology(line)\n"
            "                @triangle_strip\n"
            "                @maxvertexcount(2)",
            "Conflicting GLSL geometry output topology layout "
            "outputtopology line with triangle_strip",
        ),
        (
            "tessellation_evaluation",
            "@domain(triangle)\n"
            "                @quads\n"
            "                @partitioning(equal)",
            "Conflicting GLSL tessellation domain layout domain triangle with quads",
        ),
        (
            "tessellation_evaluation",
            "@domain(triangle)\n"
            "                @partitioning(equal)\n"
            "                @fractional_even_spacing",
            "Conflicting GLSL tessellation partitioning layout "
            "partitioning equal with fractional_even_spacing",
        ),
        (
            "tessellation_evaluation",
            "@domain(triangle)\n" "                @cw\n" "                @ccw",
            "Conflicting GLSL tessellation winding layout cw with ccw",
        ),
    ],
)
def test_glsl_stage_layout_attributes_reject_conflicting_aliases(
    stage, metadata, message
):
    code = f"""
    shader ConflictingStageLayoutAliases {{
        {stage} {{
            void main()
                {metadata}
            {{ }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), stage)


def test_glsl_stage_layout_attributes_accept_equivalent_aliases():
    geometry_code = """
    shader EquivalentGeometryStageLayoutAliases {
        geometry {
            void main()
                @inputtopology(line)
                @lines
                @outputtopology(line)
                @line_strip
                @maxvertexcount(2)
            { }
        }
    }
    """
    tessellation_code = """
    shader EquivalentTessellationStageLayoutAliases {
        tessellation_evaluation {
            void main()
                @domain(triangle)
                @triangles
                @partitioning(equal)
                @equal_spacing
                @outputtopology(triangle_cw)
            { }
        }
    }
    """

    geometry = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(geometry_code), "geometry"
    )
    tessellation = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(tessellation_code), "tessellation_evaluation"
    )

    assert "layout(lines) in;" in geometry
    assert "layout(line_strip, max_vertices = 2) out;" in geometry
    assert "layout(triangles, equal_spacing, cw) in;" in tessellation


@pytest.mark.parametrize(
    ("stage", "metadata", "message"),
    [
        (
            "geometry",
            "@triangles(1)\n                @outputtopology(line)\n                @maxvertexcount(2)",
            "GLSL stage attribute triangles does not accept arguments",
        ),
        (
            "geometry",
            "@inputtopology(line)\n                @line_strip(1)\n                @maxvertexcount(2)",
            "GLSL stage attribute line_strip does not accept arguments",
        ),
        (
            "tessellation_evaluation",
            "@domain(triangle)\n                @partitioning(equal)\n                @cw(1)",
            "GLSL stage attribute cw does not accept arguments",
        ),
        (
            "tessellation_evaluation",
            "@domain(triangle)\n                @equal_spacing(1)",
            "GLSL stage attribute equal_spacing does not accept arguments",
        ),
    ],
)
def test_glsl_stage_bare_layout_attributes_reject_arguments(stage, metadata, message):
    code = f"""
    shader BadBareStageLayoutOperand {{
        {stage} {{
            void main()
                {metadata}
            {{ }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), stage)


@pytest.mark.parametrize(
    ("stage", "metadata", "message"),
    [
        (
            "geometry",
            "@inputtopology(line)\n"
            "                @triangles\n"
            "                @outputtopology(line)\n"
            "                @maxvertexcount(2)",
            "Conflicting GLSL geometry input topology layout "
            "inputtopology line with triangles",
        ),
        (
            "geometry",
            "@inputtopology(line)\n"
            "                @outputtopology(line)\n"
            "                @triangle_strip\n"
            "                @maxvertexcount(2)",
            "Conflicting GLSL geometry output topology layout "
            "outputtopology line with triangle_strip",
        ),
        (
            "tessellation_evaluation",
            "@domain(triangle)\n"
            "                @quads\n"
            "                @partitioning(equal)",
            "Conflicting GLSL tessellation domain layout domain triangle with quads",
        ),
        (
            "tessellation_evaluation",
            "@domain(triangle)\n"
            "                @partitioning(equal)\n"
            "                @fractional_even_spacing",
            "Conflicting GLSL tessellation partitioning layout "
            "partitioning equal with fractional_even_spacing",
        ),
        (
            "tessellation_evaluation",
            "@domain(triangle)\n" "                @cw\n" "                @ccw",
            "Conflicting GLSL tessellation winding layout cw with ccw",
        ),
    ],
)
def test_glsl_stage_layout_attributes_reject_conflicting_aliases(
    stage, metadata, message
):
    code = f"""
    shader ConflictingStageLayoutAliases {{
        {stage} {{
            void main()
                {metadata}
            {{ }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), stage)


def test_glsl_stage_layout_attributes_accept_equivalent_aliases():
    geometry_code = """
    shader EquivalentGeometryStageLayoutAliases {
        geometry {
            void main()
                @inputtopology(line)
                @lines
                @outputtopology(line)
                @line_strip
                @maxvertexcount(2)
            { }
        }
    }
    """
    tessellation_code = """
    shader EquivalentTessellationStageLayoutAliases {
        tessellation_evaluation {
            void main()
                @domain(triangle)
                @triangles
                @partitioning(equal)
                @equal_spacing
                @outputtopology(triangle_cw)
            { }
        }
    }
    """

    geometry = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(geometry_code), "geometry"
    )
    tessellation = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(tessellation_code), "tessellation_evaluation"
    )

    assert "layout(lines) in;" in geometry
    assert "layout(line_strip, max_vertices = 2) out;" in geometry
    assert "layout(triangles, equal_spacing, cw) in;" in tessellation


@pytest.mark.parametrize(
    ("topology", "expected_layout"),
    [
        ("triangle_cw", "layout(triangles, equal_spacing, cw) in;"),
        ("triangle_ccw", "layout(triangles, equal_spacing, ccw) in;"),
        ("point", "layout(triangles, equal_spacing, point_mode) in;"),
    ],
)
def test_glsl_tessellation_evaluation_outputtopology_metadata_lowers_layout_parts(
    topology, expected_layout
):
    code = f"""
    shader TessellationOutputTopology {{
        tessellation_evaluation {{
            void main()
                @domain(triangle)
                @partitioning(equal)
                @outputtopology({topology})
            {{ }}
        }}
    }}
    """

    evaluation_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "tessellation_evaluation"
    )

    assert expected_layout in evaluation_code
    assert "outputtopology" not in evaluation_code


@pytest.mark.parametrize(
    ("domain", "topology", "message"),
    [
        ("isoline", "triangle_cw", "triangle_cw requires triangles domain"),
        ("triangle", "line", "line requires isolines domain"),
        (
            "triangle",
            "triangle_strip",
            "outputtopology cannot be lowered: triangle_strip",
        ),
    ],
)
def test_glsl_tessellation_evaluation_outputtopology_metadata_diagnostics(
    domain, topology, message
):
    code = f"""
    shader BadTessellationOutputTopology {{
        tessellation_evaluation {{
            void main()
                @domain({domain})
                @outputtopology({topology})
            {{ }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_evaluation"
        )


def test_glsl_tessellation_evaluation_outputtopology_rejects_winding_conflict():
    code = """
    shader ConflictingTessellationOutputTopology {
        tessellation_evaluation {
            void main()
                @domain(triangle)
                @outputtopology(triangle_cw)
                @ccw
            { }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting GLSL tessellation outputtopology triangle_cw with winding ccw",
    ):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_evaluation"
        )


def test_glsl_tessellation_control_rejects_patchconstantfunc_metadata():
    code = """
    shader UnsupportedGLSLPatchConstantFunction {
        struct PatchConstants {
            float outer[3] @ gl_TessLevelOuter;
            float inner[1] @ gl_TessLevelInner;
        };

        tessellation_control {
            PatchConstants HSConst() {
                PatchConstants patch;
                return patch;
            }

            void main()
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst)
            { }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "patchconstantfunc 'HSConst' is unsupported; write tessellation "
            "factors directly to gl_TessLevelOuter and gl_TessLevelInner"
        ),
    ):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


def test_glsl_tessellation_control_rejects_malformed_patchconstantfunc_metadata():
    code = """
    shader MalformedGLSLPatchConstantFunction {
        tessellation_control {
            void main()
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst, Other)
            { }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="patchconstantfunc requires exactly one function name",
    ):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


@pytest.mark.parametrize("attribute_name", ["vertices", "outputcontrolpoints"])
def test_glsl_tessellation_control_layout_accepts_positive_integer_constants(
    attribute_name,
):
    code = f"""
    shader TessellationControlLayoutCount {{
        const int PATCH_VERTICES = 1 + 2;

        tessellation_control {{
            void main() @{attribute_name}(PATCH_VERTICES) {{ }}
        }}
    }}
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "tessellation_control"
    )

    assert "layout(vertices = PATCH_VERTICES) out;" in generated_code


@pytest.mark.parametrize(
    ("attribute_line", "message"),
    [
        (
            "@vertices(0)",
            "GLSL tessellation_control vertices layout requires a positive "
            "integer constant, got 0",
        ),
        (
            "@vertices(1.5)",
            r"GLSL tessellation_control vertices layout requires a positive "
            r"integer constant, got 1\.5",
        ),
        (
            "@outputcontrolpoints(0)",
            "GLSL tessellation_control vertices layout requires a positive "
            "integer constant, got 0",
        ),
        (
            "@outputcontrolpoints(-1)",
            r"GLSL tessellation_control vertices layout requires a positive "
            r"integer constant, got \(-1\)",
        ),
        (
            "@outputcontrolpoints(1.5)",
            r"GLSL tessellation_control vertices layout requires a positive "
            r"integer constant, got 1\.5",
        ),
    ],
)
def test_glsl_tessellation_control_layout_rejects_invalid_count_constants(
    attribute_line, message
):
    code = f"""
    shader BadTessellationControlLayoutCount {{
        tessellation_control {{
            void main()
                {attribute_line}
            {{ }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


def test_glsl_tessellation_control_accepts_direct_factor_component_assignments():
    code = """
    shader DirectGLSLTessellationFactorAssignments {
        tessellation_control {
            void main() @outputcontrolpoints(3) {
                gl_TessLevelOuter[0] = 1.0;
                gl_TessLevelOuter[1] = 2;
                gl_TessLevelOuter[3] = 4.0;
                gl_TessLevelInner[0] = 3.0;
                gl_TessLevelInner[1] = 5.0;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "tessellation_control"
    )

    assert "gl_TessLevelOuter[0] = 1.0;" in generated_code
    assert "gl_TessLevelOuter[1] = 2;" in generated_code
    assert "gl_TessLevelOuter[3] = 4.0;" in generated_code
    assert "gl_TessLevelInner[0] = 3.0;" in generated_code
    assert "gl_TessLevelInner[1] = 5.0;" in generated_code


def test_glsl_tessellation_control_accepts_dynamic_integer_factor_component_indices():
    code = """
    shader DynamicGLSLTessellationFactorAssignments {
        tessellation_control {
            void main() @outputcontrolpoints(3) {
                int outerIndex = 2;
                uint innerIndex = 1u;
                gl_TessLevelOuter[outerIndex] = 2.0;
                gl_TessLevelInner[innerIndex] = 3.0;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "tessellation_control"
    )

    assert "int outerIndex = 2;" in generated_code
    assert "uint innerIndex = 1u;" in generated_code
    assert "gl_TessLevelOuter[outerIndex] = 2.0;" in generated_code
    assert "gl_TessLevelInner[innerIndex] = 3.0;" in generated_code


def test_glsl_tessellation_control_allows_top_level_barrier_before_return():
    code = """
    shader ValidGLSLTessellationControlBarrier {
        tessellation_control {
            void main() @outputcontrolpoints(3) {
                gl_TessLevelOuter[0] = 1.0;
                workgroupBarrier();
                gl_TessLevelInner[0] = 1.0;
                return;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "tessellation_control"
    )

    assert "layout(vertices = 3) out;" in generated_code
    assert "barrier();" in generated_code
    assert "workgroupBarrier();" not in generated_code


def test_glsl_tessellation_control_rejects_barrier_inside_control_flow():
    code = """
    shader BadGLSLTessellationControlBarrierControlFlow {
        tessellation_control {
            void main() @outputcontrolpoints(3) {
                if (gl_InvocationID == 0u) {
                    workgroupBarrier();
                }
            }
        }
    }
    """

    with pytest.raises(ValueError, match="barrier\\(\\).*inside control flow"):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


def test_glsl_tessellation_control_rejects_barrier_after_return():
    code = """
    shader BadGLSLTessellationControlBarrierAfterReturn {
        tessellation_control {
            void main() @outputcontrolpoints(3) {
                return;
                workgroupBarrier();
            }
        }
    }
    """

    with pytest.raises(ValueError, match="barrier\\(\\).*after a return"):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


def test_glsl_tessellation_control_rejects_helper_barrier():
    code = """
    shader BadGLSLTessellationControlHelperBarrier {
        void synchronizePatch() {
            workgroupBarrier();
        }

        tessellation_control {
            void main() @outputcontrolpoints(3) {
                synchronizePatch();
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "barrier\\(\\).*directly in main\\(\\); helper function "
            "'synchronizePatch' reaches 'workgroupBarrier'"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(code))


@pytest.mark.parametrize("builtin", ["gl_TessLevelOuter", "gl_TessLevelInner"])
def test_glsl_tessellation_control_rejects_whole_factor_assignment(builtin):
    code = f"""
    shader BadWholeGLSLTessellationFactorAssignment {{
        tessellation_control {{
            void main() @outputcontrolpoints(3) {{
                {builtin} = vec2(1.0, 2.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            rf"GLSL tessellation factor {builtin} assignment requires "
            "an indexed scalar component target"
        ),
    ):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


@pytest.mark.parametrize(
    ("assignment", "expected_message"),
    [
        (
            "gl_TessLevelOuter[4] = 1.0;",
            r"GLSL tessellation factor gl_TessLevelOuter component index 4 "
            r"out of range; valid range is 0..3",
        ),
        (
            "gl_TessLevelOuter[-1] = 1.0;",
            r"GLSL tessellation factor gl_TessLevelOuter component index -1 "
            r"out of range; valid range is 0..3",
        ),
        (
            "gl_TessLevelInner[2] = 1.0;",
            r"GLSL tessellation factor gl_TessLevelInner component index 2 "
            r"out of range; valid range is 0..1",
        ),
        (
            "gl_TessLevelOuter[1.0] = 1.0;",
            r"GLSL tessellation factor gl_TessLevelOuter component index "
            r"requires a scalar integer value, got float",
        ),
    ],
)
def test_glsl_tessellation_control_rejects_malformed_factor_component_indices(
    assignment,
    expected_message,
):
    code = f"""
    shader BadGLSLTessellationFactorComponentIndex {{
        tessellation_control {{
            void main() @outputcontrolpoints(3) {{
                {assignment}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=expected_message):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


@pytest.mark.parametrize(
    ("index_declaration", "assignment", "expected_message"),
    [
        (
            "float outerIndex = 1.0;",
            "gl_TessLevelOuter[outerIndex] = 1.0;",
            r"GLSL tessellation factor gl_TessLevelOuter component index "
            r"requires a scalar integer value, got float",
        ),
        (
            "vec2 outerIndex = vec2(0.0, 1.0);",
            "gl_TessLevelOuter[outerIndex] = 1.0;",
            r"GLSL tessellation factor gl_TessLevelOuter component index "
            r"requires a scalar integer value, got vec2",
        ),
        (
            "uvec2 innerIndex = uvec2(0u, 1u);",
            "gl_TessLevelInner[innerIndex] = 1.0;",
            r"GLSL tessellation factor gl_TessLevelInner component index "
            r"requires a scalar integer value, got uvec2",
        ),
    ],
)
def test_glsl_tessellation_control_rejects_non_integer_dynamic_factor_indices(
    index_declaration,
    assignment,
    expected_message,
):
    code = f"""
    shader BadDynamicGLSLTessellationFactorComponentIndex {{
        tessellation_control {{
            void main() @outputcontrolpoints(3) {{
                {index_declaration}
                {assignment}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=expected_message):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


@pytest.mark.parametrize("builtin", ["gl_TessLevelOuter", "gl_TessLevelInner"])
def test_glsl_tessellation_control_rejects_vector_factor_component_assignment(
    builtin,
):
    code = f"""
    shader BadGLSLTessellationFactorComponentAssignment {{
        tessellation_control {{
            void main() @outputcontrolpoints(3) {{
                {builtin}[0] = vec2(1.0, 2.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            rf"GLSL tessellation factor {builtin} component assignment "
            "requires a scalar numeric value, got vec2"
        ),
    ):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


def test_glsl_direct_return_semantic_ignores_stage_control_attributes():
    code = """
    shader ReturnSemanticWithStageControls {
        vertex {
            vec4 main() @max_vertices(3) @gl_Position {
                return vec4(0.0);
            }
        }

        fragment {
            vec4 main() @max_vertices(3) @gl_FragColor1 {
                return vec4(1.0);
            }
        }

        geometry {
            void main()
                @points
                @invocations(2)
                @outputtopology(triangle_strip)
                @max_vertices(3)
            { }
        }

        tessellation_control {
            void main() @outputcontrolpoints(4) { }
        }

        tessellation_evaluation {
            void main()
                @domain(triangle)
                @partitioning(fractional_odd)
                @cw
                @point_mode
            { }
        }

        mesh {
            void main()
                @triangles
                @max_vertices(64)
                @max_primitives(32)
            {
                SetMeshOutputCounts(64, 32);
            }
        }
    }
    """
    ast = crosstl.translator.parse(code)

    vertex_code = GLSLCodeGen().generate_stage(ast, "vertex")
    assert "gl_Position = vec4(0.0);" in vertex_code
    assert "vertexOutput = vec4(0.0);" not in vertex_code

    fragment_code = GLSLCodeGen().generate_stage(ast, "fragment")
    assert "layout(location = 1) out vec4 fragColor1;" in fragment_code
    assert "fragColor1 = vec4(1.0);" in fragment_code

    geometry_code = GLSLCodeGen().generate_stage(ast, "geometry")
    assert "layout(points, invocations = 2) in;" in geometry_code
    assert "layout(triangle_strip, max_vertices = 3) out;" in geometry_code
    assert geometry_code.index("layout(points") < geometry_code.index("void main()")

    control_code = GLSLCodeGen().generate_stage(ast, "tessellation_control")
    evaluation_code = GLSLCodeGen().generate_stage(ast, "tessellation_evaluation")
    assert "layout(vertices = 4) out;" in control_code
    assert "layout(vertices = 4) out;" not in evaluation_code
    assert (
        "layout(triangles, fractional_odd_spacing, cw, point_mode) in;"
        in evaluation_code
    )
    assert "layout(triangles" not in control_code

    mesh_code = GLSLCodeGen().generate_stage(ast, "mesh")
    assert "layout(triangles, max_vertices = 64, max_primitives = 32) out;" in mesh_code
    assert "SetMeshOutputsEXT(64, 32);" in mesh_code
    assert "SetMeshOutputCounts" not in mesh_code


@pytest.mark.parametrize(
    ("stage", "return_type", "semantic", "value", "expected"),
    [
        ("vertex", "vec4", "gl_FragColor", "vec4(0.0)", "fragment output"),
        ("vertex", "float", "gl_FragDepth", "0.5", "fragment output"),
        ("fragment", "vec4", "gl_Position", "vec4(0.0)", "vertex output"),
        ("fragment", "float", "gl_PointSize", "1.0", "vertex output"),
        ("compute", "vec4", "gl_Position", "vec4(0.0)", "graphics output"),
    ],
)
def test_glsl_function_return_output_builtin_stages_are_validated(
    stage, return_type, semantic, value, expected
):
    code = f"""
    shader BadReturnOutputStage {{
        {stage} {{
            {return_type} main() @{semantic} {{
                return {value};
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=f"{stage}.*{expected}.*{semantic}"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), stage)


def test_glsl_vertex_return_semantic_assigns_position_builtin():
    code = """
    shader VertexReturnPositionSemantic {
        vertex {
            vec4 main(vec3 position @POSITION) @gl_Position {
                return vec4(position, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "vertex"
    )

    assert "layout(location = 0) in vec3 position;" in generated_code
    assert "gl_Position = vec4(position, 1.0);" in generated_code
    assert "return;" in generated_code
    assert "return vec4" not in generated_code


def test_glsl_vertex_return_semantic_assigns_point_size_builtin():
    code = """
    shader VertexReturnPointSizeSemantic {
        vertex {
            float main(float size @TEXCOORD0) @gl_PointSize {
                return size;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "vertex"
    )

    assert "layout(location = 5) in float size;" in generated_code
    assert "gl_PointSize = size;" in generated_code
    assert "return size;" not in generated_code


def test_glsl_vertex_vec4_return_defaults_to_position_builtin():
    code = """
    shader VertexReturnDefaultPosition {
        vertex {
            vec4 main(vec3 position @POSITION) {
                return vec4(position, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "vertex"
    )

    assert "gl_Position = vec4(position, 1.0);" in generated_code
    assert "return vec4" not in generated_code


def test_glsl_vertex_return_varying_semantic_assigns_output():
    code = """
    shader VertexReturnTexcoordSemantic {
        vertex {
            vec2 main(vec2 uv @TEXCOORD0) @TEXCOORD0 {
                return uv;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "vertex"
    )

    assert "layout(location = 5) in vec2 uv;" in generated_code
    assert "layout(location = 5) out vec2 vertexOutput;" in generated_code
    assert "vertexOutput = uv;" in generated_code
    assert "return uv;" not in generated_code


def test_glsl_direct_vertex_output_renames_input_name_collision():
    code = """
    shader DirectVertexOutputNameCollision {
        vertex {
            vec2 main(vec2 vertexOutput @TEXCOORD0) @TEXCOORD1 {
                return vertexOutput;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "vertex"
    )

    assert "layout(location = 5) in vec2 vertexOutput;" in generated_code
    assert "layout(location = 6) out vec2 out_vertexOutput;" in generated_code
    assert "out_vertexOutput = vertexOutput;" in generated_code
    assert "layout(location = 6) out vec2 vertexOutput;" not in generated_code
    assert "\n    vertexOutput = vertexOutput;" not in generated_code


def test_glsl_legacy_output_renames_direct_vertex_output_name_collision():
    code = """
    shader LegacyDirectVertexOutputNameCollision {
        struct VSOutput {
            vec2 vertexOutput;
        };

        vertex {
            vec2 main(vec2 uv @TEXCOORD0) @TEXCOORD0 {
                return uv;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "out vec2 vertexOutput;" in generated_code
    assert "layout(location = 5) out vec2 out_vertexOutput;" in generated_code
    assert "out_vertexOutput = uv;" in generated_code
    assert "layout(location = 5) out vec2 vertexOutput;" not in generated_code


def test_glsl_legacy_output_duplicate_name_different_type_raises():
    code = """
    shader LegacyOutputDuplicateNameDifferentType {
        struct VSOutput {
            vec2 data;
            vec3 data;
        };
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting OpenGL vertex output declaration for 'data': "
            "out vec3 data; differs from out vec2 data;"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(code))


def test_glsl_vertex_stage_does_not_emit_psoutput_fragment_outputs():
    code = """
    shader VertexStageNoPSOutputFragmentOutputs {
        struct PSOutput {
            vec4 color @gl_FragColor;
        };

        vertex {
            void main(vec3 position @POSITION) {
                gl_Position = vec4(position, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "vertex"
    )

    assert "struct PSOutput" in generated_code
    assert "layout(location = 0) in vec3 position;" in generated_code
    assert "layout(location = 0) out vec4 color;" not in generated_code
    assert "out vec4 color;" not in generated_code


def test_glsl_fragment_stage_does_not_emit_unused_vsoutput_vertex_outputs():
    code = """
    shader FragmentStageNoUnusedVSOutputVertexOutputs {
        struct VSOutput {
            vec2 uv @TEXCOORD0;
        };

        fragment {
            vec4 main() @gl_FragColor {
                return vec4(1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "struct VSOutput" in generated_code
    assert "layout(location = 0) out vec4 fragColor;" in generated_code
    assert "out vec2 uv;" not in generated_code
    assert "layout(location = 5) out vec2 uv;" not in generated_code


def test_glsl_fragment_stage_returning_vsoutput_declares_fragment_outputs():
    code = """
    shader FragmentStageReturnsVSOutput {
        struct VSOutput {
            vec4 color @gl_FragColor;
        };

        fragment {
            VSOutput main() {
                return VSOutput(vec4(1.0));
            }
        }
    }
    """

    ast = crosstl.translator.parse(code)
    combined_code = GLSLCodeGen().generate(ast)
    fragment_code = GLSLCodeGen().generate_stage(ast, "fragment")

    assert "layout(location = 0) out vec4 color;" in combined_code
    assert "\nout vec4 color;" not in combined_code
    assert "color = vec4(1.0);" in combined_code
    assert "return VSOutput" not in combined_code

    assert "layout(location = 0) out vec4 color;" in fragment_code
    assert "\nout vec4 color;" not in fragment_code
    assert "color = vec4(1.0);" in fragment_code
    assert "return VSOutput" not in fragment_code


def test_glsl_fragment_stage_declares_shared_vertex_fragment_output_struct():
    code = """
    shader SharedVertexFragmentOutputStruct {
        struct Shared {
            vec4 color @COLOR0;
        };

        vertex {
            Shared main() {
                return Shared(vec4(1.0));
            }
        }

        fragment {
            Shared main(Shared input) {
                return Shared(input.color);
            }
        }
    }
    """

    ast = crosstl.translator.parse(code)
    combined_code = GLSLCodeGen().generate(ast)
    fragment_code = GLSLCodeGen().generate_stage(ast, "fragment")

    assert "layout(location = 13) out vec4 color;" in combined_code
    assert "layout(location = 13) in vec4 in_color;" in combined_code
    assert "layout(location = 13) out vec4 out_color;" in combined_code
    assert "out_color = in_color;" in combined_code
    assert "layout(location = 13) in vec4 color;" not in combined_code

    assert "layout(location = 13) in vec4 color;" in fragment_code
    assert "layout(location = 13) out vec4 out_color;" in fragment_code
    assert "out_color = color;" in fragment_code
    assert "layout(location = 13) out vec4 color;" not in fragment_code


def test_glsl_direct_vertex_varying_links_to_fragment_input_by_location():
    code = """
    shader DirectVertexVaryingLink {
        sampler2D colorMap;

        vertex {
            vec2 main(vec2 uv @TEXCOORD0) @TEXCOORD0 {
                return uv;
            }
        }

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                return texture(colorMap, uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "layout(location = 5) out vec2 vertexOutput;" in generated_code
    assert "layout(location = 5) in vec2 uv;" in generated_code
    assert "vertexOutput = uv;" in generated_code
    assert "fragColor = texture(colorMap, uv);" in generated_code


def test_glsl_multiple_direct_fragment_entries_reuse_color_output_location():
    code = """
    shader MultipleFragmentEntries {
        fragment {
            vec4 main() @gl_FragColor {
                return vec4(1.0, 0.0, 0.0, 1.0);
            }
        }

        fragment overlay {
            vec4 main() @gl_FragColor {
                return vec4(0.0, 1.0, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert generated_code.count("layout(location = 0) out vec4 fragColor;") == 1
    assert "out_fragColor" not in generated_code
    assert "fragColor = vec4(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "fragColor = vec4(0.0, 1.0, 0.0, 1.0);" in generated_code


def test_glsl_color_semantic_maps_to_explicit_stage_io_locations():
    code = """
    shader VertexReturnColorSemantic {
        vertex {
            vec4 main(vec4 color @COLOR) @COLOR {
                return color;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "vertex"
    )

    assert "layout(location = 13) in vec4 color;" in generated_code
    assert "layout(location = 13) out vec4 vertexOutput;" in generated_code
    assert "vertexOutput = color;" in generated_code


def test_glsl_vertex_non_void_return_without_semantic_raises():
    code = """
    shader VertexReturnWithoutSemantic {
        vertex {
            float main(float value @TEXCOORD0) {
                return value;
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "OpenGL vertex stage entry point with non-void return type "
            "'float' requires an output semantic"
        ),
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "vertex")


def test_glsl_direct_fragment_duplicate_input_location_raises():
    code = """
    shader DirectFragmentDuplicateInputs {
        fragment {
            vec4 main(vec2 uv @TEXCOORD0, vec2 uv2 @TEXCOORD0) @gl_FragColor {
                return vec4(uv + uv2, 0.0, 1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting OpenGL fragment input location for 'uv2': "
            "location 5 overlaps 'uv' location 5"
        ),
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")


def test_glsl_vertex_struct_duplicate_input_location_raises():
    code = """
    shader VertexStructDuplicateInputs {
        struct VSInput {
            vec3 position @POSITION;
            vec3 otherPosition @POSITION;
        };

        vertex {
            void main(VSInput input) {
                gl_Position = vec4(input.position + input.otherPosition, 1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting OpenGL vertex input location for 'otherPosition': "
            "location 0 overlaps 'position' location 0"
        ),
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "vertex")


def test_glsl_fragment_struct_duplicate_input_location_raises():
    code = """
    shader FragmentStructDuplicateInputs {
        struct FSInput {
            vec2 uv @TEXCOORD0;
            vec2 uv2 @TEXCOORD0;
        };

        fragment {
            vec4 main(FSInput input) @gl_FragColor {
                return vec4(input.uv + input.uv2, 0.0, 1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting OpenGL fragment input location for 'uv2': "
            "location 5 overlaps 'uv' location 5"
        ),
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")


def test_glsl_fragment_struct_and_value_input_location_overlap_raises():
    code = """
    shader FragmentStructValueInputOverlap {
        struct FSInput {
            vec2 uv @TEXCOORD0;
        };

        fragment {
            vec4 main(FSInput input, vec2 uv2 @TEXCOORD0) @gl_FragColor {
                return vec4(input.uv + uv2, 0.0, 1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting OpenGL fragment input location for 'uv2': "
            "location 5 overlaps 'uv' location 5"
        ),
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")


def test_glsl_fragment_multiple_struct_input_location_overlap_raises():
    code = """
    shader FragmentMultipleStructInputOverlap {
        struct FSInputA {
            vec2 uv @TEXCOORD0;
        };

        struct FSInputB {
            vec2 uv2 @TEXCOORD0;
        };

        fragment {
            vec4 main(FSInputA a, FSInputB b) @gl_FragColor {
                return vec4(a.uv + b.uv2, 0.0, 1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting OpenGL fragment input location for 'uv2': "
            "location 5 overlaps 'uv' location 5"
        ),
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")


def test_glsl_fragment_duplicate_identical_input_declaration_emits_once():
    code = """
    shader FragmentDuplicateIdenticalInput {
        struct FSInput {
            vec2 uv @TEXCOORD0;
        };

        fragment {
            vec4 main(FSInput input, vec2 uv @TEXCOORD0) @gl_FragColor {
                return vec4(input.uv + uv, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert generated_code.count("layout(location = 5) in vec2 uv;") == 1
    assert "fragColor = vec4((uv + uv), 0.0, 1.0);" in generated_code


def test_glsl_vertex_output_location_range_overlap_raises():
    code = """
    shader VertexOutputLocationRangeOverlap {
        struct VSOutput {
            mat4 transform @TEXCOORD0;
            vec4 tangent @TEXCOORD1;
        };

        vertex {
            VSOutput main() {
                VSOutput output;
                output.transform = mat4(1.0);
                output.tangent = vec4(1.0);
                return output;
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting OpenGL vertex output location for 'tangent': "
            "location 6 overlaps 'transform' locations 5-8"
        ),
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(code), "vertex")


def test_glsl_direct_vertex_output_conflicts_with_struct_output_location():
    float_type = PrimitiveType("float")
    vec2_type = VectorType(float_type, 2)
    output_struct = StructNode(
        "VSOutput",
        [StructMemberNode("uv", vec2_type, attributes=[AttributeNode("TEXCOORD0")])],
    )
    struct_entry = FunctionNode(
        "structMain",
        NamedType("VSOutput"),
        [],
        BlockNode([]),
        qualifiers=["vertex"],
    )
    direct_entry = FunctionNode(
        "directMain",
        vec2_type,
        [ParameterNode("uv", vec2_type, attributes=[AttributeNode("TEXCOORD0")])],
        BlockNode([ReturnNode(IdentifierNode("uv"))]),
        attributes=[AttributeNode("TEXCOORD0")],
        qualifiers=["vertex"],
    )
    ast = ShaderNode(
        "DirectVertexOutputConflict",
        ExecutionModel.GRAPHICS_PIPELINE,
        structs=[output_struct],
        functions=[struct_entry, direct_entry],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting OpenGL vertex output location for 'vertexOutput': "
            "location 5 overlaps 'out_uv' location 5"
        ),
    ):
        GLSLCodeGen().generate_stage(ast, "vertex")


def test_glsl_psoutput_duplicate_fragment_output_location_raises():
    code = """
    shader PSOutputDuplicateColor {
        struct PSOutput {
            vec4 color @gl_FragColor;
            vec4 otherColor @gl_FragColor;
        };
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting OpenGL fragment output location for 'otherColor': "
            "location 0 overlaps 'color' location 0"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(code))


def test_glsl_unannotated_fragment_outputs_use_distinct_locations():
    code = """
    shader UnannotatedFragmentOutputs {
        struct PSOutput {
            vec4 color;
            vec4 normalBuffer;
            vec4 positionBuffer;
            float depth;
        };

        fragment {
            PSOutput main() {
                return PSOutput(vec4(1.0), vec4(0.5), vec4(0.25), 0.75);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "layout(location = 0) out vec4 color;" in generated_code
    assert "layout(location = 1) out vec4 normalBuffer;" in generated_code
    assert "layout(location = 2) out vec4 positionBuffer;" in generated_code
    assert "layout(location = 3) out float depth;" in generated_code


def test_glsl_psoutput_fragcolor_zero_maps_to_location_zero():
    code = """
    shader PSOutputFragColorZero {
        struct PSOutput {
            vec4 color0 @gl_FragColor0;
            vec4 color1 @gl_FragColor1;
        };
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "layout(location = 0) out vec4 color0;" in generated_code
    assert "layout(location = 1) out vec4 color1;" in generated_code
    assert "gl_FragColor0 out" not in generated_code


def test_glsl_psoutput_conflicts_with_direct_fragment_output_location():
    code = """
    shader PSOutputDirectFragmentOutputConflict {
        struct PSOutput {
            vec4 color @gl_FragColor;
        };

        fragment {
            vec4 main() @gl_FragColor {
                return vec4(1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting OpenGL fragment output location for 'fragColor': "
            "location 0 overlaps 'color' location 0"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(code))


def test_glsl_direct_fragment_output_renames_declared_output_name_collision():
    code = """
    shader DirectFragmentDeclaredOutputNameCollision {
        struct PSOutput {
            vec4 fragColor @gl_FragColor1;
        };

        fragment {
            vec4 main() @gl_FragColor {
                return vec4(1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "layout(location = 1) out vec4 fragColor;" in generated_code
    assert "layout(location = 0) out vec4 out_fragColor;" in generated_code
    assert "out_fragColor = vec4(1.0);" in generated_code
    assert "layout(location = 0) out vec4 fragColor;" not in generated_code
    assert "\n    fragColor = vec4(1.0);" not in generated_code


def test_glsl_psoutput_depth_member_uses_builtin_without_output_declaration():
    code = """
    shader PSOutputDepth {
        struct PSOutput {
            vec4 color @gl_FragColor;
            float depth @gl_FragDepth;
        };
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "layout(location = 0) out vec4 color;" in generated_code
    assert "out float depth;" not in generated_code
    assert "layout(location = 0) out float depth;" not in generated_code


def test_glsl_fragment_psoutput_return_assigns_color_and_depth():
    code = """
    shader PSOutputReturnDepth {
        struct PSOutput {
            vec4 color @gl_FragColor;
            float depth @gl_FragDepth;
        };

        fragment {
            PSOutput main() {
                PSOutput output;
                output.color = vec4(1.0);
                output.depth = 0.5;
                return output;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "layout(location = 0) out vec4 color;" in generated_code
    assert "layout(location = 0) out PSOutput fragColor;" not in generated_code
    assert "color = vec4(1.0);" in generated_code
    assert "gl_FragDepth = 0.5;" in generated_code
    assert "return output;" not in generated_code


def test_glsl_fragment_psoutput_sample_mask_member_uses_builtin():
    code = """
    shader PSOutputSampleMask {
        struct PSOutput {
            vec4 color @gl_FragColor;
            int mask @gl_SampleMask;
        };

        fragment {
            PSOutput main() {
                PSOutput output;
                output.color = vec4(1.0);
                output.mask = 1;
                return output;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "layout(location = 0) out vec4 color;" in generated_code
    assert "out int mask;" not in generated_code
    assert "layout(location = 0) out int mask;" not in generated_code
    assert "color = vec4(1.0);" in generated_code
    assert "gl_SampleMask[0] = 1;" in generated_code
    assert "return output;" not in generated_code


def test_glsl_custom_fragment_output_struct_declares_outputs():
    code = """
    shader CustomFragmentOutputStruct {
        struct FragOut {
            vec4 color @gl_FragColor;
            float depth @gl_FragDepth;
        };

        fragment {
            FragOut main() {
                FragOut output;
                output.color = vec4(1.0);
                output.depth = 0.5;
                return output;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "layout(location = 0) out vec4 color;" in generated_code
    assert "struct FragOut" in generated_code
    assert "color = vec4(1.0);" in generated_code
    assert "gl_FragDepth = 0.5;" in generated_code
    assert "out float depth;" not in generated_code
    assert "return output;" not in generated_code


def test_glsl_fragment_output_struct_constructor_return_assigns_members():
    code = """
    shader FragmentOutputStructConstructor {
        struct FragOut {
            vec4 color @gl_FragColor;
            float depth @gl_FragDepth;
        };

        fragment {
            FragOut main() {
                return FragOut(vec4(1.0), 0.5);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "layout(location = 0) out vec4 color;" in generated_code
    assert "color = vec4(1.0);" in generated_code
    assert "gl_FragDepth = 0.5;" in generated_code
    assert "return FragOut" not in generated_code


def test_glsl_fragment_output_struct_renames_input_name_collision():
    code = """
    shader FragmentOutputInputNameCollision {
        struct VSOutput {
            vec4 color @COLOR0;
        };

        struct FragOut {
            vec4 color @gl_FragColor;
        };

        fragment {
            FragOut main(VSOutput input) {
                FragOut output;
                output.color = input.color;
                return output;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "layout(location = 13) in vec4 color;" in generated_code
    assert "layout(location = 0) out vec4 out_color;" in generated_code
    assert "out_color = color;" in generated_code
    assert "layout(location = 0) out vec4 color;" not in generated_code
    assert "\n    color = color;" not in generated_code


def test_glsl_fragment_output_struct_renames_value_parameter_name_collision():
    code = """
    shader FragmentOutputValueParameterNameCollision {
        struct FragOut {
            vec4 color @gl_FragColor;
        };

        fragment {
            FragOut main(vec4 color @COLOR0) {
                return FragOut(color);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "layout(location = 13) in vec4 color;" in generated_code
    assert "layout(location = 0) out vec4 out_color;" in generated_code
    assert "out_color = color;" in generated_code
    assert "return FragOut" not in generated_code


def test_glsl_fragment_output_struct_renames_local_variable_name_collision():
    code = """
    shader FragmentOutputLocalNameCollision {
        struct FragOut {
            vec4 color @gl_FragColor;
        };

        fragment {
            FragOut main() {
                FragOut output;
                vec3 color = vec3(0.2, 0.4, 0.6);
                output.color = vec4(color, 1.0);
                return output;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(location = 0) out vec4 out_color;" in generated_code
    assert "vec3 color = vec3(0.2, 0.4, 0.6);" in generated_code
    assert "out_color = vec4(color, 1.0);" in generated_code
    assert "layout(location = 0) out vec4 color;" not in generated_code
    assert "\n    color = vec4(color, 1.0);" not in generated_code


def test_glsl_direct_fragment_output_renames_input_name_collision():
    code = """
    shader DirectFragmentOutputInputNameCollision {
        struct VSOutput {
            vec4 fragColor @COLOR0;
        };

        fragment {
            vec4 main(VSOutput input) @gl_FragColor {
                return input.fragColor;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "layout(location = 13) in vec4 fragColor;" in generated_code
    assert "layout(location = 0) out vec4 out_fragColor;" in generated_code
    assert "out_fragColor = fragColor;" in generated_code
    assert "layout(location = 0) out vec4 fragColor;" not in generated_code


def test_glsl_direct_fragment_output_renames_local_variable_name_collision():
    code = """
    shader DirectFragmentOutputLocalNameCollision {
        fragment {
            vec4 main() @gl_FragColor {
                vec4 fragColor = vec4(1.0);
                return fragColor;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "layout(location = 0) out vec4 out_fragColor;" in generated_code
    assert "vec4 fragColor = vec4(1.0);" in generated_code
    assert "out_fragColor = fragColor;" in generated_code
    assert "layout(location = 0) out vec4 fragColor;" not in generated_code
    assert "\n    fragColor = fragColor;" not in generated_code


def test_glsl_global_uniforms_do_not_consume_resource_bindings():
    code = """
    shader GlobalUniformBindings {
        uniform float exposure;
        vec4 tint;
        sampler2D colorMap;

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                return texture(colorMap, uv) * exposure * tint;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "uniform float exposure;" in generated_code
    assert "uniform vec4 tint;" in generated_code
    assert "layout(std140, binding = 0) float exposure;" not in generated_code
    assert "layout(std140, binding = 1) vec4 tint;" not in generated_code
    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code


def test_glsl_reserved_global_uniform_names_are_aliased():
    code = """
    shader ReservedGlobalUniform {
        uniform float input;
        uniform float input_;

        fragment {
            vec4 main() @gl_FragColor {
                float value = input + input_;
                return vec4(value, input, input_, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "uniform float input_2;" in generated_code
    assert "uniform float input_;" in generated_code
    assert "uniform float input;" not in generated_code
    assert "float value = (input_2 + input_);" in generated_code
    assert "fragColor = vec4(value, input_2, input_, 1.0);" in generated_code


def test_glsl_cbuffer_binding_attributes_drive_layout_bindings():
    code = """
    shader CBufferBindings {
        cbuffer Camera @register(b2) {
            mat4 viewProj;
            mat4 bones[2];
            vec4 viewRow;
        };

        uniform Material @binding(4) {
            vec4 tint;
        };

        cbuffer AutoBlock {
            vec4 color;
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "layout(std140, binding = 2) uniform Camera" in generated_code
    assert "mat4 viewProj;" in generated_code
    assert "mat4 bones[2];" in generated_code
    assert "layout(std140, binding = 4) uniform Material" in generated_code
    assert "layout(std140, binding = 5) uniform AutoBlock" in generated_code
    assert "MatrixType(" not in generated_code


def test_glsl_cbuffer_binding_overlap_raises():
    code = """
    shader CBufferOverlapGLSL {
        cbuffer Camera @binding(2) {
            mat4 viewProj;
        };

        cbuffer Material @register(b2) {
            vec4 tint;
        };
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting OpenGL resource binding for 'Material': "
            "uniform buffer binding 2 overlaps 'Camera' uniform buffer binding 2"
        ),
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_glsl_cbuffer_and_storage_buffer_bindings_are_independent():
    code = """
    shader CBufferStorageBufferNamespacesGLSL {
        cbuffer Constants @binding(0) {
            vec4 tint;
        };

        RWStructuredBuffer<int> values @binding(0);
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "layout(std140, binding = 0) uniform Constants" in generated_code
    assert "layout(std430, binding = 0) buffer valuesBuffer" in generated_code


def test_glsl_hlsl_matrix_aliases_map_to_glsl_names():
    shader = """
    shader MatrixAliasSmoke {
        float3x3 makeTransform(float3x3 input) {
            float3x3 m = float3x3(1.0);
            float3x2 n = float3x2(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);
            return input * m;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "mat3 makeTransform(mat3 input_)" in generated_code
    assert "mat3 m = mat3(1.0);" in generated_code
    assert "mat2x3 n = mat2x3(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);" in generated_code
    assert "return (input_ * m);" in generated_code
    assert "float3x3" not in generated_code
    assert "float3x2" not in generated_code


def test_glsl_hlsl_mul_alias_emits_binary_matrix_multiply():
    # Khronos GLSL 4.60 section 5.9 defines matrix/vector/matrix
    # multiplication through the binary * operator, not a mul() built-in.
    shader = """
    shader MatrixMulAlias {
        vec4 transformColumn(mat4 viewProj, vec4 position) {
            return mul(viewProj, position);
        }

        vec4 transformRow(vec4 position, mat4 viewProj) {
            return mul(position, viewProj);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "return (viewProj * position);" in generated_code
    assert "return (position * viewProj);" in generated_code
    assert "*(viewProj, position)" not in generated_code
    assert "*(position, viewProj)" not in generated_code
    assert "mul(" not in generated_code


def test_glsl_user_defined_mul_function_is_preserved():
    # Khronos GLSL 4.60 permits user-defined function calls; the CrossGL-to-GLSL
    # mul alias should only lower when no user helper shadows the alias name.
    shader = """
    shader UserMul {
        vec3 mul(vec3 value, vec3 scale) {
            return (value * scale) + vec3(0.5);
        }

        fragment {
            vec4 main() @ gl_FragColor {
                vec3 color = mul(vec3(0.25), vec3(2.0));
                return vec4(color, 1.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "vec3 mul(vec3 value, vec3 scale)" in generated_code
    assert "vec3 color = mul(vec3(0.25), vec3(2.0));" in generated_code
    assert "vec3 color = (vec3(0.25) * vec3(2.0));" not in generated_code


def test_glsl_hlsl_mad_alias_emits_multiply_add_expression():
    # Khronos GLSL 4.60 documents fma(x, y, z) as returning x * y + z;
    # shader ports commonly spell the non-fused source form as mad(x, y, z).
    shader = """
    shader MadAlias {
        fragment {
            vec4 main() @ gl_FragColor {
                vec3 albedo;
                vec3 lighting;
                vec3 ambient;
                vec3 color = mad(albedo, lighting, ambient);
                return vec4(color, 1.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "vec3 color = ((albedo * lighting) + ambient);" in generated_code
    assert "mad(" not in generated_code


def test_glsl_user_defined_mad_function_is_preserved():
    shader = """
    shader UserMad {
        vec3 mad(vec3 value, vec3 scale, vec3 bias) {
            return bias + value * scale;
        }

        fragment {
            vec4 main() @ gl_FragColor {
                vec3 color = mad(vec3(0.25), vec3(2.0), vec3(0.5));
                return vec4(color, 1.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "vec3 mad(vec3 value, vec3 scale, vec3 bias)" in generated_code
    assert "vec3 color = mad(vec3(0.25), vec3(2.0), vec3(0.5));" in generated_code
    assert "((vec3(0.25) * vec3(2.0)) + vec3(0.5))" not in generated_code


def test_glsl_hlsl_double_aliases_map_to_glsl_names():
    shader = """
    shader DoubleAliasSmoke {
        double2x2 makePrecise(double2 uv, double2x2 input) {
            double2 p = double2(1.0, 2.0);
            double4x3 jacobian = double4x3(1.0);
            double2x2 m = double2x2(1.0, 0.0, 0.0, 1.0);
            return input * m;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "dmat2 makePrecise(dvec2 uv, dmat2 input_)" in generated_code
    assert "dvec2 p = dvec2(1.0, 2.0);" in generated_code
    assert "dmat3x4 jacobian = dmat3x4(1.0);" in generated_code
    assert "dmat2 m = dmat2(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "return (input_ * m);" in generated_code
    assert "double2" not in generated_code
    assert "double2x2" not in generated_code
    assert "double4x3" not in generated_code


def test_glsl_hlsl_integer_bool_vector_aliases_map_to_glsl_names():
    shader = """
    shader IntegerBoolAliasSmoke {
        int2 offset(int2 input) {
            int2 delta = int2(1, 2);
            return input + delta;
        }

        uint3 grow(uint3 input) {
            uint3 inc = uint3(1u, 2u, 3u);
            return input + inc;
        }

        bool4 flags(bool4 input) {
            bool4 mask = bool4(true, false, true, false);
            return input;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "ivec2 offset(ivec2 input_)" in generated_code
    assert "ivec2 delta = ivec2(1, 2);" in generated_code
    assert "return (input_ + delta);" in generated_code
    assert "uvec3 grow(uvec3 input_)" in generated_code
    assert "uvec3 inc = uvec3(1u, 2u, 3u);" in generated_code
    assert "return (input_ + inc);" in generated_code
    assert "bvec4 flags(bvec4 input_)" in generated_code
    assert "bvec4 mask = bvec4(true, false, true, false);" in generated_code
    assert "int2" not in generated_code
    assert "uint3" not in generated_code
    assert "bool4" not in generated_code


def test_glsl_vector_relational_operators_lower_to_builtin_functions():
    # Khronos GLSL 4.60 requires these built-ins for component-wise vector
    # relational comparisons:
    # https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html#operators
    shader = """
    shader VectorRelationalSmoke {
        bool2 masks(float2 uv, float cutoff) {
            bool2 above = uv > cutoff;
            bool2 below = cutoff < uv;
            bool2 atMost = uv <= float2(1.0);
            bool2 atLeast = float2(0.0) >= uv;
            return above;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "bvec2 masks(vec2 uv, float cutoff)" in generated_code
    assert "bvec2 above = greaterThan(uv, vec2(cutoff));" in generated_code
    assert "bvec2 below = lessThan(vec2(cutoff), uv);" in generated_code
    assert "bvec2 atMost = lessThanEqual(uv, vec2(1.0));" in generated_code
    assert "bvec2 atLeast = greaterThanEqual(vec2(0.0), uv);" in generated_code
    assert "(uv > cutoff)" not in generated_code
    assert "(cutoff < uv)" not in generated_code


def test_glsl_vector_relational_return_expression_uses_bool_vector_result():
    shader = """
    shader VectorRelationalReturn {
        bool3 visible(vec3 color, vec3 threshold) {
            return color >= threshold;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "bvec3 visible(vec3 color, vec3 threshold)" in generated_code
    assert "return greaterThanEqual(color, threshold);" in generated_code
    assert "return (color >= threshold);" not in generated_code


def test_glsl_precision_aliases_lower_to_standard_glsl_types():
    shader = """
    shader PrecisionAliasSmoke {
        half tone(half input) {
            half bias = half(0.5);
            return input + bias;
        }

        half2 pair(half2 input) {
            half2 scale = half2(1.0, 2.0);
            return input * scale;
        }

        min16float3 tint(min16float3 input) {
            min16float3 offset = min16float3(1.0, 2.0, 3.0);
            return input + offset;
        }

        min16uint2 bump(min16uint2 input) {
            min16uint2 inc = min16uint2(1u, 2u);
            return input + inc;
        }

        min12int clampInt(min12int input) {
            min12int one = min12int(1);
            return input + one;
        }

        half3x2 passMatrix(half3x2 input) {
            half3x2 m = half3x2(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);
            return input;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "float tone(float input_)" in generated_code
    assert "float bias = float(0.5);" in generated_code
    assert "vec2 pair(vec2 input_)" in generated_code
    assert "vec2 scale = vec2(1.0, 2.0);" in generated_code
    assert "vec3 tint(vec3 input_)" in generated_code
    assert "vec3 offset = vec3(1.0, 2.0, 3.0);" in generated_code
    assert "uvec2 bump(uvec2 input_)" in generated_code
    assert "uvec2 inc = uvec2(1u, 2u);" in generated_code
    assert "int clampInt(int input_)" in generated_code
    assert "int one = int(1);" in generated_code
    assert "mat2x3 passMatrix(mat2x3 input_)" in generated_code
    assert "mat2x3 m = mat2x3(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);" in generated_code
    for invalid_token in ("half", "min16float", "min16uint", "min12int"):
        assert invalid_token not in generated_code


def test_glsl_float16_ir_aliases_lower_to_standard_glsl_types():
    shader = """
    shader Float16IRSmoke {
        float16 tone(float16 input) {
            float16 bias = float16(0.5);
            return input + bias;
        }

        f16vec2 pair(f16vec2 input) {
            f16vec2 scale = f16vec2(1.0, 2.0);
            return input * scale;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "float tone(float input_)" in generated_code
    assert "float bias = float(0.5);" in generated_code
    assert "vec2 pair(vec2 input_)" in generated_code
    assert "vec2 scale = vec2(1.0, 2.0);" in generated_code
    assert "float16" not in generated_code
    assert "f16vec2" not in generated_code


def test_glsl_float16_matrix_ir_aliases_lower_to_standard_glsl_matrices():
    shader = """
    shader Float16MatrixIRSmoke {
        f16mat3x2 passMatrix(f16mat3x2 input) {
            f16mat3x2 m = f16mat3x2(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);
            return input;
        }

        f16mat3 passSquare(f16mat3 input) {
            f16mat3 m = f16mat3(1.0);
            return input * m;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "mat3x2 passMatrix(mat3x2 input_)" in generated_code
    assert "mat3x2 m = mat3x2(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);" in generated_code
    assert "mat3 passSquare(mat3 input_)" in generated_code
    assert "mat3 m = mat3(1.0);" in generated_code
    assert "return (input_ * m);" in generated_code
    assert "f16mat3x2" not in generated_code
    assert "f16mat3" not in generated_code


def test_glsl_narrow_integer_aliases_lower_to_standard_glsl_integer_types():
    shader = """
    shader NarrowIntegerAliasSmoke {
        int8 signedScalar(int8 input) {
            int8 one = int8(1);
            return input + one;
        }

        uint8 unsignedScalar(uint8 input) {
            uint8 one = uint8(1u);
            return input + one;
        }

        i8vec2 signedPair(i8vec2 input) {
            i8vec2 inc = i8vec2(1, 2);
            return input + inc;
        }

        u8vec3 unsignedTriple(u8vec3 input) {
            u8vec3 inc = u8vec3(1u, 2u, 3u);
            return input + inc;
        }

        short2 signedShort(short2 input) {
            short2 inc = short2(1, 2);
            return input + inc;
        }

        ushort3 unsignedShort(ushort3 input) {
            ushort3 inc = ushort3(1u, 2u, 3u);
            return input + inc;
        }

        char4 signedChar(char4 input) {
            char4 inc = char4(1, 2, 3, 4);
            return input + inc;
        }

        uchar2 unsignedChar(uchar2 input) {
            uchar2 inc = uchar2(1u, 2u);
            return input + inc;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "int signedScalar(int input_)" in generated_code
    assert "int one = int(1);" in generated_code
    assert "uint unsignedScalar(uint input_)" in generated_code
    assert "uint one = uint(1u);" in generated_code
    assert "ivec2 signedPair(ivec2 input_)" in generated_code
    assert "ivec2 inc = ivec2(1, 2);" in generated_code
    assert "uvec3 unsignedTriple(uvec3 input_)" in generated_code
    assert "uvec3 inc = uvec3(1u, 2u, 3u);" in generated_code
    assert "ivec2 signedShort(ivec2 input_)" in generated_code
    assert "uvec3 unsignedShort(uvec3 input_)" in generated_code
    assert "ivec4 signedChar(ivec4 input_)" in generated_code
    assert "uvec2 unsignedChar(uvec2 input_)" in generated_code
    for invalid_token in (
        "int8",
        "uint8",
        "i8vec2",
        "u8vec3",
        "short2",
        "ushort3",
        "char4",
        "uchar2",
    ):
        assert invalid_token not in generated_code


def test_glsl_packed_vector_aliases_lower_to_standard_glsl_vectors():
    shader = """
    shader PackedVectorAliasSmoke {
        packed_float4 color(packed_float4 input) {
            packed_float4 bias = packed_float4(1.0, 2.0, 3.0, 4.0);
            return input + bias;
        }

        packed_half2 halfPair(packed_half2 input) {
            packed_half2 scale = packed_half2(1.0, 2.0);
            return input * scale;
        }

        packed_int3 signedTriple(packed_int3 input) {
            packed_int3 inc = packed_int3(1, 2, 3);
            return input + inc;
        }

        packed_uint4 unsignedQuad(packed_uint4 input) {
            packed_uint4 inc = packed_uint4(1u, 2u, 3u, 4u);
            return input + inc;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "vec4 color(vec4 input_)" in generated_code
    assert "vec4 bias = vec4(1.0, 2.0, 3.0, 4.0);" in generated_code
    assert "vec2 halfPair(vec2 input_)" in generated_code
    assert "vec2 scale = vec2(1.0, 2.0);" in generated_code
    assert "ivec3 signedTriple(ivec3 input_)" in generated_code
    assert "ivec3 inc = ivec3(1, 2, 3);" in generated_code
    assert "uvec4 unsignedQuad(uvec4 input_)" in generated_code
    assert "uvec4 inc = uvec4(1u, 2u, 3u, 4u);" in generated_code
    assert "packed_" not in generated_code


def test_glsl_simd_aliases_lower_to_standard_glsl_types():
    shader = """
    shader SimdAliasSmoke {
        simd_float4 color(simd_float4 input) {
            simd_float4 bias = simd_float4(1.0, 2.0, 3.0, 4.0);
            return input + bias;
        }

        simd_int3 signedTriple(simd_int3 input) {
            simd_int3 inc = simd_int3(1, 2, 3);
            return input + inc;
        }

        simd_uint2 unsignedPair(simd_uint2 input) {
            simd_uint2 inc = simd_uint2(1u, 2u);
            return input + inc;
        }

        simd_float4x4 passSquare(simd_float4x4 input) {
            simd_float4x4 m = simd_float4x4(1.0);
            return input * m;
        }

        simd_float3x2 passMatrix(simd_float3x2 input) {
            simd_float3x2 m = simd_float3x2(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);
            return input;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "vec4 color(vec4 input_)" in generated_code
    assert "vec4 bias = vec4(1.0, 2.0, 3.0, 4.0);" in generated_code
    assert "ivec3 signedTriple(ivec3 input_)" in generated_code
    assert "ivec3 inc = ivec3(1, 2, 3);" in generated_code
    assert "uvec2 unsignedPair(uvec2 input_)" in generated_code
    assert "uvec2 inc = uvec2(1u, 2u);" in generated_code
    assert "mat4 passSquare(mat4 input_)" in generated_code
    assert "mat4 m = mat4(1.0);" in generated_code
    assert "mat2x3 passMatrix(mat2x3 input_)" in generated_code
    assert "mat2x3 m = mat2x3(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);" in generated_code
    assert "simd_" not in generated_code


def test_glsl_fixed_width_scalar_aliases_lower_to_standard_glsl_scalars():
    shader = """
    shader FixedWidthScalarAliasSmoke {
        int8_t signedByte(int8_t input) {
            int8_t one = int8_t(1);
            return input + one;
        }

        uint8_t unsignedByte(uint8_t input) {
            uint8_t one = uint8_t(1u);
            return input + one;
        }

        int16_t signedShortScalar(int16_t input) {
            int16_t one = int16_t(1);
            return input + one;
        }

        uint16_t unsignedShortScalar(uint16_t input) {
            uint16_t one = uint16_t(1u);
            return input + one;
        }

        int32_t signedWord(int32_t input) {
            int32_t one = int32_t(1);
            return input + one;
        }

        uint32_t unsignedWord(uint32_t input) {
            uint32_t one = uint32_t(1u);
            return input + one;
        }

        int64 signedLong(int64 input) {
            int64 one = int64(1);
            return input + one;
        }

        uint64 unsignedLong(uint64 input) {
            uint64 one = uint64(1u);
            return input + one;
        }

        int64_t signedLongT(int64_t input) {
            int64_t one = int64_t(1);
            return input + one;
        }

        uint64_t unsignedLongT(uint64_t input) {
            uint64_t one = uint64_t(1u);
            return input + one;
        }

        size_t sizeValue(size_t input) {
            size_t one = size_t(1u);
            return input + one;
        }

        ptrdiff_t ptrDiff(ptrdiff_t input) {
            ptrdiff_t one = ptrdiff_t(1);
            return input + one;
        }

        long longValue(long input) {
            long one = long(1);
            return input + one;
        }

        ulong ulongValue(ulong input) {
            ulong one = ulong(1u);
            return input + one;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "int signedByte(int input_)" in generated_code
    assert "uint unsignedByte(uint input_)" in generated_code
    assert "int signedShortScalar(int input_)" in generated_code
    assert "uint unsignedShortScalar(uint input_)" in generated_code
    assert "int signedWord(int input_)" in generated_code
    assert "uint unsignedWord(uint input_)" in generated_code
    assert "int signedLong(int input_)" in generated_code
    assert "uint unsignedLong(uint input_)" in generated_code
    assert "int signedLongT(int input_)" in generated_code
    assert "uint unsignedLongT(uint input_)" in generated_code
    assert "uint sizeValue(uint input_)" in generated_code
    assert "int ptrDiff(int input_)" in generated_code
    assert "int longValue(int input_)" in generated_code
    assert "uint ulongValue(uint input_)" in generated_code
    assert "int one = int(1);" in generated_code
    assert "uint one = uint(1u);" in generated_code
    for invalid_token in (
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
        "int64 ",
        "uint64 ",
        "int64_t",
        "uint64_t",
        "size_t",
        "ptrdiff_t",
        "long ",
        "ulong ",
    ):
        assert invalid_token not in generated_code


def test_glsl_fixed_width_scalar_array_aliases_lower_in_aggregate_declarations():
    shader = """
    shader AliasArrayAggregateSmoke {
        int8_t globalBytes[2];
        uint64_t globalCounters[2];

        struct AliasPayload {
            int8_t bytes[2];
            uint16_t words[3];
            int64_t signedValue;
            size_t offsets[2];
        };

        cbuffer AliasConstants @register(b1) {
            int16_t smalls[2];
            uint32_t count;
            ptrdiff_t delta;
        };

        uint64_t bump(uint64_t counters[2], AliasPayload payload) {
            uint64_t localCounters[2];
            uint64_t one = uint64_t(1u);
            localCounters[0] = counters[0] + one;
            return localCounters[0] + payload.offsets[1];
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "int bytes[2];" in generated_code
    assert "uint words[3];" in generated_code
    assert "int signedValue;" in generated_code
    assert "uint offsets[2];" in generated_code
    assert "int globalBytes[2];" in generated_code
    assert "uint globalCounters[2];" in generated_code
    assert "int smalls[2];" in generated_code
    assert "uint count;" in generated_code
    assert "int delta;" in generated_code
    assert "uint bump(uint counters[2], AliasPayload payload)" in generated_code
    assert "uint localCounters[2];" in generated_code
    assert "uint one = uint(1u);" in generated_code
    for invalid_token in (
        "int8_t",
        "uint16_t",
        "int16_t",
        "uint32_t",
        "int64 ",
        "uint64 ",
        "int64_t",
        "uint64_t",
        "size_t",
        "ptrdiff_t",
    ):
        assert invalid_token not in generated_code


def test_glsl_fixed_width_nested_array_aliases_lower_to_standard_glsl_types():
    shader = """
    shader AliasNestedArraySmoke {
        struct AliasGridPayload {
            int8_t bytes[2][3];
            size_t offsets[2][3];
        };

        uint64_t[2] makeCounters() {
            uint64_t counters[2];
            counters[0] = uint64_t(1u);
            counters[1] = counters[0] + uint64_t(2u);
            return counters;
        }

        uint16_t bumpNested(uint16_t values[2][3], int row, int col) {
            uint16_t grid[2][3];
            grid[row][col] = values[row][col] + uint16_t(1u);
            return grid[row][col];
        }

        ptrdiff_t readPayload(AliasGridPayload payload, int row, int col) {
            return ptrdiff_t(payload.bytes[row][col]) +
                ptrdiff_t(payload.offsets[row][col]);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "int bytes[2][3];" in generated_code
    assert "uint offsets[2][3];" in generated_code
    assert "uint[2] makeCounters()" in generated_code
    assert "uint counters[2];" in generated_code
    assert "uint bumpNested(uint values[2][3], int row, int col)" in generated_code
    assert "uint grid[2][3];" in generated_code
    assert (
        "int readPayload(AliasGridPayload payload, int row, int col)" in generated_code
    )
    assert "uint(1u)" in generated_code
    assert "uint(2u)" in generated_code
    assert "int(payload.bytes[row][col])" in generated_code
    assert "int(payload.offsets[row][col])" in generated_code
    assert "int[2] bytes[3]" not in generated_code
    assert "uint[2] offsets[3]" not in generated_code
    for invalid_token in (
        "int8_t",
        "uint16_t",
        "int64 ",
        "uint64 ",
        "uint64_t",
        "size_t",
        "ptrdiff_t",
    ):
        assert invalid_token not in generated_code


def test_glsl_fixed_width_nested_cbuffer_array_aliases_lower_to_standard_types():
    shader = """
    shader AliasNestedCbufferArraySmoke {
        cbuffer AliasNestedConstants @register(b2) {
            int64_t signedGrid[2][3];
            uint16_t unsignedGrid[2][3];
            size_t offsets[2][3];
        };

        uint64_t readConstant(int row, int col) {
            return uint64_t(offsets[row][col]) + uint64_t(unsignedGrid[row][col]);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "layout(std140, binding = 2) uniform AliasNestedConstants" in generated_code
    assert "int signedGrid[2][3];" in generated_code
    assert "uint unsignedGrid[2][3];" in generated_code
    assert "uint offsets[2][3];" in generated_code
    assert "uint readConstant(int row, int col)" in generated_code
    assert "uint(offsets[row][col])" in generated_code
    assert "uint(unsignedGrid[row][col])" in generated_code
    assert "int[2] signedGrid[3]" not in generated_code
    assert "uint[2] offsets[3]" not in generated_code
    for invalid_token in (
        "int64_t",
        "uint16_t",
        "uint64 ",
        "uint64_t",
        "size_t",
    ):
        assert invalid_token not in generated_code


def test_glsl_default_float_image_scalar_and_vector_load_store():
    code = """
    shader DefaultFloatImageLoadStore {
        image2D storageImage;

        float touchScalar(image2D image, ivec2 pixel, float value) {
            float scalarOld = imageLoad(image, pixel);
            imageStore(image, pixel, scalarOld + value);
            return scalarOld;
        }

        vec4 touchVector(image2D image, ivec2 pixel, vec4 value) {
            vec4 vectorOld = imageLoad(image, pixel);
            imageStore(image, pixel, vectorOld + value);
            return vectorOld;
        }

        compute {
            void main() {
                float a = touchScalar(storageImage, ivec2(0, 1), 0.25);
                vec4 b = touchVector(storageImage, ivec2(2, 3), vec4(1.0));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "compute"
    )

    assert "float scalarOld = imageLoad(image, pixel).x;" in generated_code
    assert "imageStore(image, pixel, vec4((scalarOld + value)));" in generated_code
    assert "vec4 vectorOld = imageLoad(image, pixel);" in generated_code
    assert "imageStore(image, pixel, (vectorOld + value));" in generated_code
    assert "vec4 vectorOld = imageLoad(image, pixel).x;" not in generated_code
    assert "imageStore(image, pixel, vec4((vectorOld + value)));" not in generated_code


def test_glsl_default_integer_storage_image_vector_load_store_from_compiler_fixture():
    # Reduced from CrossGL-Compiler
    # tests/frontend/fixtures/StorageImageHIRShader.cgl.
    code = """
    shader StorageImageHIRShader {
        compute {
            layout(set = 0, binding = 0) uniform iimage2D labelImage;
            layout(set = 0, binding = 1) uniform uimage2D maskImage;

            void main() {
                ivec2 pixel = ivec2(0, 1);
                ivec4 label2D = imageLoad(labelImage, pixel);
                uvec4 mask2D = imageLoad(maskImage, pixel);
                imageStore(labelImage, pixel, label2D);
                imageStore(maskImage, pixel, mask2D);
                return;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "compute"
    )

    assert "layout(r32i, binding = 0) uniform iimage2D labelImage;" in generated_code
    assert "layout(r32ui, binding = 1) uniform uimage2D maskImage;" in generated_code
    assert "ivec4 label2D = imageLoad(labelImage, pixel);" in generated_code
    assert "uvec4 mask2D = imageLoad(maskImage, pixel);" in generated_code
    assert "ivec4 label2D = imageLoad(labelImage, pixel).x;" not in generated_code
    assert "uvec4 mask2D = imageLoad(maskImage, pixel).x;" not in generated_code
    assert "imageStore(labelImage, pixel, ivec4(label2D));" in generated_code
    assert "imageStore(maskImage, pixel, uvec4(mask2D));" in generated_code


def test_glsl_explicit_scalar_image_formats_allow_vector_fixture_contexts():
    # Reduced from CrossGL-Compiler
    # tests/fixtures/StorageImageExplicitFormatShader.cgl.
    code = """
    shader StorageImageExplicitFormatShader {
        compute {
            layout(local_size_x = 2, local_size_y = 2, local_size_z = 1) in;
            layout(set = 0, binding = 0, format = r32f) readonly uniform image2D readColor;
            layout(set = 0, binding = 1, format = r32i) readonly uniform iimage2D readLabel;
            layout(set = 0, binding = 2, format = r32ui) readonly uniform uimage2D readMask;
            layout(set = 0, binding = 3, format = r32ui) writeonly uniform uimage2D writeMask;
            layout(set = 0, binding = 4) buffer vec4* colors;
            layout(set = 0, binding = 5) buffer ivec4* labels;

            void main() {
                ivec2 pixel = ivec2(0, 0);
                vec4 color = imageLoad(readColor, pixel);
                ivec4 label = imageLoad(readLabel, pixel);
                uvec4 mask = imageLoad(readMask, pixel);
                imageStore(writeMask, pixel, mask);
                colors[0] = color;
                labels[0] = label;
                return;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "compute"
    )

    assert (
        "layout(r32f, binding = 0) readonly uniform image2D readColor;"
        in generated_code
    )
    assert (
        "layout(r32i, binding = 1) readonly uniform iimage2D readLabel;"
        in generated_code
    )
    assert (
        "layout(r32ui, binding = 2) readonly uniform uimage2D readMask;"
        in generated_code
    )
    assert (
        "layout(r32ui, binding = 3) writeonly uniform uimage2D writeMask;"
        in generated_code
    )
    assert "vec4 color = imageLoad(readColor, pixel);" in generated_code
    assert "ivec4 label = imageLoad(readLabel, pixel);" in generated_code
    assert "uvec4 mask = imageLoad(readMask, pixel);" in generated_code
    assert "imageStore(writeMask, pixel, mask);" in generated_code
    assert "vec4 color = imageLoad(readColor, pixel).x;" not in generated_code
    assert "ivec4 label = imageLoad(readLabel, pixel).x;" not in generated_code
    assert "uvec4 mask = imageLoad(readMask, pixel).x;" not in generated_code
    assert "imageStore(writeMask, pixel, uvec4(mask));" not in generated_code


def test_glsl_rg_image_scalar_and_vector_load_store():
    code = """
    shader RGImageScalarVector {
        image2D rgFloat @rg32f;
        uimage2D rgUnsigned @rg32ui;

        float scalarFloat(image2D image @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloat(image2D image @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsigned(uimage2D image @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsigned(uimage2D image @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float a = scalarFloat(rgFloat, ivec2(0, 1), 0.25);
                vec2 b = vectorFloat(rgFloat, ivec2(2, 3), vec2(1.0));
                uint c = scalarUnsigned(rgUnsigned, ivec2(4, 5), 7u);
                uvec2 d = vectorUnsigned(rgUnsigned, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "compute"
    )

    assert "float oldValue = imageLoad(image, pixel).x;" in generated_code
    assert "uint oldValue = imageLoad(image, pixel).x;" in generated_code
    assert (
        "imageStore(image, pixel, vec4((oldValue + value), 0.0, 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(image, pixel, uvec4((oldValue + value), 0u, 0u, 0u));"
        in generated_code
    )
    assert "vec2 oldValue = imageLoad(image, pixel).xy;" in generated_code
    assert "uvec2 oldValue = imageLoad(image, pixel).xy;" in generated_code
    assert (
        "imageStore(image, pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(image, pixel, uvec4((oldValue + value), 0u, 0u));" in generated_code
    )


def test_input_output():
    code = """
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_if_statement():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;

            if (input.texCoord.x > 0.5) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
            } else {
                output.color = vec4(0.0, 0.0, 0.0, 1.0);
            }

            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            if (bloom > 0.5) {
                bloom = 0.5;
            } else {
                bloom = 0.0;
            }

            // Apply bloom to the texture color
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);

            return vec4(colorWithBloom, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_for_statement():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            for (int i = 0; i < 10; i=i+1) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
            }
            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            for (int i = 0; i < 10; i++) {
                if (i > 0.5) {
                    bloom = 0.5;
                } else {
                    bloom = 0.0;
                }
            }

            // Apply bloom to the texture color
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);

            return vec4(colorWithBloom, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_for_statement_preserves_declaration_initializers():
    shader = """
    shader LoopDeclarationInitializers {
        float helper() {
            const float weights[2];
            int i = 0;
            float total = 0.0;
            for (int i = 0; i < 2; i++) {
                total = total + weights[0];
            }
            for (i = 0; i < 4; i++) {
                if (i == 0) {
                    continue;
                }
                break;
            }
            for (const int fixed = 0; fixed < 0;) {
                total = total + 1.0;
            }
            for (;;) {
                break;
            }
            return total;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const float weights[2];" in generated_code
    assert "for (int i = 0; (i < 2); (++i))" in generated_code
    assert "for (i = 0; (i < 4); (++i))" in generated_code
    assert "for (const int fixed = 0; (fixed < 0); )" in generated_code
    assert "for (; ; )" in generated_code
    assert "continue;" in generated_code
    assert "break;" in generated_code
    assert "for (int i = 0;;" not in generated_code
    assert "for (const int fixed = 0;;" not in generated_code
    assert "BreakNode(" not in generated_code
    assert "ContinueNode(" not in generated_code
    assert "None" not in generated_code


def test_loop_statement_lowers_to_while_true():
    shader = """
    shader LoopNodeSmoke {
        int helper(int limit) {
            int i = 0;
            loop {
                i = i + 1;
                if (i >= limit) {
                    break;
                }
            }
            return i;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "while (true)" in generated_code
    assert "i = (i + 1);" in generated_code
    assert "if ((i >= limit))" in generated_code
    assert "break;" in generated_code
    assert "return i;" in generated_code
    assert "LoopNode(" not in generated_code


def test_do_while_statement_lowers_to_c_style_syntax():
    shader = """
    shader DoWhileNodeSmoke {
        int helper(int limit) {
            int i = 0;
            do {
                i = i + 1;
                if (i >= limit) {
                    break;
                }
            } while (i < 4);
            return i;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "do {" in generated_code
    assert "i = (i + 1);" in generated_code
    assert "if ((i >= limit))" in generated_code
    assert "break;" in generated_code
    assert "} while ((i < 4));" in generated_code
    assert "return i;" in generated_code
    assert "DoWhileNode(" not in generated_code


def test_for_in_statement_lowers_to_counted_loop():
    shader = """
    shader ForInNodeSmoke {
        int helper(int limit) {
            int total = 0;
            for i in limit {
                total = total + i;
            }
            return total;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "for (int i = 0; i < limit; ++i)" in generated_code
    assert "total = (total + i);" in generated_code
    assert "return total;" in generated_code
    assert "ForInNode(" not in generated_code


def test_for_in_range_statement_lowers_to_counted_loop():
    shader = """
    shader ForInRangeNodeSmoke {
        int helper(int limit) {
            int total = 0;
            for i in 2..5 {
                total = total + i;
            }
            for j in 1..=limit {
                total = total + j;
            }
            return total;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "for (int i = 2; i < 5; ++i)" in generated_code
    assert "for (int j = 1; j <= limit; ++j)" in generated_code
    assert "total = (total + i);" in generated_code
    assert "total = (total + j);" in generated_code
    assert "return total;" in generated_code
    assert "RangeNode(" not in generated_code
    assert "ForInNode(" not in generated_code


def test_fragment_return_semantic_writes_output_variable():
    shader = """
    shader FragmentReturnSemanticSmoke {
        fragment {
            vec4 main() @ gl_FragColor {
                int total = 0;
                for i in 1..=3 {
                    total = total + i;
                }
                return vec4(float(total), 0.0, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(location = 0) out vec4 fragColor;" in generated_code
    assert "void main()" in generated_code
    assert "for (int i = 1; i <= 3; ++i)" in generated_code
    assert "fragColor = vec4(float(total), 0.0, 0.0, 1.0);" in generated_code
    assert "return;" in generated_code
    assert "return vec4" not in generated_code
    assert "RangeNode(" not in generated_code


def test_vertex_struct_io_lowers_to_flattened_stage_variables():
    shader = """
    shader VertexStructIOSmoke {
        struct VSInput {
            vec3 position @ POSITION;
            vec2 uv @ TEXCOORD0;
        };

        struct VSOutput {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.position = vec4(input.position, 1.0);
                output.uv = input.uv;
                return output;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(location = 0) in vec3 position;" in generated_code
    assert "layout(location = 5) in vec2 uv;" in generated_code
    assert "layout(location = 5) out vec2 out_uv;" in generated_code
    assert "gl_Position = vec4(position, 1.0);" in generated_code
    assert "out_uv = uv;" in generated_code
    assert "return;" in generated_code
    assert "VSOutput output;" not in generated_code
    assert "input." not in generated_code
    assert "output." not in generated_code
    assert "return output;" not in generated_code
    assert "out vec4 position;" not in generated_code


def test_vertex_struct_output_renames_value_parameter_name_collision():
    shader = """
    shader VertexStructOutputValueParameterNameCollision {
        struct VSOutput {
            vec2 uv @ TEXCOORD1;
        };

        vertex {
            VSOutput main(vec2 uv @ TEXCOORD0) {
                VSOutput output;
                output.uv = uv;
                return output;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "vertex"
    )

    assert "layout(location = 5) in vec2 uv;" in generated_code
    assert "layout(location = 6) out vec2 out_uv;" in generated_code
    assert "out_uv = uv;" in generated_code
    assert "layout(location = 6) out vec2 uv;" not in generated_code
    assert "\n    uv = uv;" not in generated_code


def test_vertex_integer_struct_outputs_are_flat_qualified():
    shader = """
    shader VertexIntegerStructOutputFlat {
        struct VSOutput {
            vec4 position @ gl_Position;
            int layer @ TEXCOORD0;
        };

        vertex {
            VSOutput main() {
                VSOutput output;
                output.position = vec4(0.0, 0.0, 0.0, 1.0);
                output.layer = 1;
                return output;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "vertex"
    )

    assert "layout(location = 5) flat out int layer;" in generated_code
    assert "layer = 1;" in generated_code


def test_glsl_vertex_struct_output_uses_builtin_cull_distance():
    shader = """
    shader VertexCullDistanceBuiltin {
        struct VSOutput {
            vec4 position @ gl_Position;
            float cull[1] @ gl_CullDistance;
        };

        vertex {
            VSOutput main(vec3 position @ POSITION) {
                VSOutput output;
                output.position = vec4(position, 1.0);
                output.cull[0] = position.x;
                return output;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "vertex"
    )

    assert "layout(location = 0) in vec3 position;" in generated_code
    assert "gl_Position = vec4(position, 1.0);" in generated_code
    assert "gl_CullDistance[0] = position.x;" in generated_code
    assert "out float cull[1];" not in generated_code
    assert "cull[0] = position.x;" not in generated_code


def test_vertex_struct_constructor_return_lowers_to_flattened_stage_variables():
    shader = """
    shader VertexStructConstructorReturn {
        struct VSOutput {
            vec4 position @ gl_Position;
            float pointSize @ gl_PointSize;
        };

        vertex {
            VSOutput main() {
                return VSOutput(vec4(1.0), 2.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "gl_Position = vec4(1.0);" in generated_code
    assert "gl_PointSize = 2.0;" in generated_code
    assert "return VSOutput" not in generated_code


def test_fragment_struct_input_lowers_to_flattened_stage_variables():
    shader = """
    shader FragmentStructInputSmoke {
        struct VSOutput {
            vec2 uv @ TEXCOORD0;
            vec3 normal @ NORMAL;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.uv, input.normal.x, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(location = 5) in vec2 uv;" in generated_code
    assert "layout(location = 1) in vec3 normal;" in generated_code
    assert "layout(location = 0) out vec4 fragColor;" in generated_code
    assert "fragColor = vec4(uv, normal.x, 1.0);" in generated_code
    assert "return;" in generated_code
    assert "input." not in generated_code
    assert "out vec2 uv;" not in generated_code
    assert "return vec4" not in generated_code


def test_generate_stage_splits_combined_vertex_fragment_units():
    shader = """
    shader CombinedStageIOSmoke {
        struct VSInput {
            vec3 position @ POSITION;
            vec2 uv @ TEXCOORD0;
        };

        struct VSOutput {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.position = vec4(input.position, 1.0);
                output.uv = input.uv;
                return output;
            }
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.uv, 0.0, 1.0);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generator = GLSLCodeGen()
    vertex_code = generator.generate_stage(ast, "vertex")
    fragment_code = generator.generate_stage(ast, "fragment")

    assert "layout(location = 0) in vec3 position;" in vertex_code
    assert "layout(location = 5) in vec2 uv;" in vertex_code
    assert "layout(location = 5) out vec2 out_uv;" in vertex_code
    assert "gl_Position = vec4(position, 1.0);" in vertex_code
    assert "out_uv = uv;" in vertex_code
    assert "fragColor" not in vertex_code
    assert "layout(location = 5) in vec2 out_uv;" not in vertex_code

    assert "layout(location = 5) in vec2 out_uv;" in fragment_code
    assert "layout(location = 0) out vec4 fragColor;" in fragment_code
    assert "fragColor = vec4(out_uv, 0.0, 1.0);" in fragment_code
    assert "layout(location = 0) in vec3 position;" not in fragment_code
    assert "layout(location = 5) out vec2 out_uv;" not in fragment_code
    assert "gl_Position" not in fragment_code
    assert "input." not in fragment_code


def test_glsl_combined_struct_io_renames_fragment_input_colliding_with_vertex_output():
    shader = """
    shader CombinedStructIOAvoidsSameUnitInputOutputCollision {
        struct VSInput {
            vec3 position @ POSITION;
            vec2 uv @ TEXCOORD0;
        };

        struct VSOutput {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        vertex {
            VSOutput main(VSInput input) {
                return VSOutput(vec4(input.position, 1.0), input.uv);
            }
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.uv, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(location = 5) out vec2 out_uv;" in generated_code
    assert "layout(location = 5) in vec2 in_out_uv;" in generated_code
    assert "fragColor = vec4(in_out_uv, 0.0, 1.0);" in generated_code
    assert "layout(location = 5) in vec2 out_uv;" not in generated_code


def test_glsl_combined_multi_stage_entries_use_distinct_names():
    shader = """
    shader CombinedStageEntryNames {
        struct VSInput {
            vec3 position @ POSITION;
        };

        struct VSOutput {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        vertex {
            VSOutput main(VSInput input) {
                return VSOutput(vec4(input.position, 1.0), vec2(0.5));
            }
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.uv, 0.0, 1.0);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    combined_code = GLSLCodeGen().generate(ast)
    vertex_code = GLSLCodeGen().generate_stage(ast, "vertex")
    fragment_code = GLSLCodeGen().generate_stage(ast, "fragment")

    assert "#ifdef GL_VERTEX_SHADER" in combined_code
    assert "#ifdef GL_FRAGMENT_SHADER" in combined_code
    assert "void vertex_main()" not in combined_code
    assert "void fragment_main()" not in combined_code
    assert combined_code.count("void main()") == 2
    assert combined_code.count("void ") == 2

    assert "void main()" in vertex_code
    assert "void vertex_main()" not in vertex_code
    assert "void main()" in fragment_code
    assert "void fragment_main()" not in fragment_code


def test_glsl_combined_stage_entry_names_avoid_helper_function_collisions():
    shader = """
    shader CombinedEntryNameCollision {
        void vertex_main() { }

        vertex {
            void main() { }
        }

        fragment {
            void main() { }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert generated_code.count("void vertex_main()") == 1
    assert generated_code.count("void main()") == 2
    assert "void vertex_main_2()" not in generated_code
    assert "void fragment_main()" not in generated_code


def test_glsl_stage_guards_are_only_emitted_for_combined_multi_stage_output():
    single_stage_shader = """
    shader SingleComputeStage {
        compute {
            void main() { }
        }
    }
    """
    multi_stage_shader = """
    shader GuardedGraphicsStages {
        vertex {
            void main() { }
        }

        fragment {
            void main() { }
        }
    }
    """

    single_stage_code = GLSLCodeGen().generate(
        crosstl.translator.parse(single_stage_shader)
    )
    multi_stage_code = GLSLCodeGen().generate(
        crosstl.translator.parse(multi_stage_shader)
    )

    assert "#ifdef GL_COMPUTE_SHADER" not in single_stage_code
    assert "#ifdef GL_VERTEX_SHADER" in multi_stage_code
    assert "#ifdef GL_FRAGMENT_SHADER" in multi_stage_code


def test_glsl_combined_single_stage_entry_name_avoids_helper_main():
    shader = """
    shader SingleEntryNameCollision {
        void main() { }

        vertex {
            void main() { }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert generated_code.count("void main()") == 1
    assert "void vertex_main()" in generated_code


def test_glsl_generate_stage_rejects_global_main_helper_collision():
    shader = """
    shader StageMainHelperCollision {
        void main() { }

        fragment {
            vec4 main() @ gl_FragColor {
                return vec4(1.0);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Global helper function 'main'"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_glsl_combined_advanced_stage_entries_use_distinct_names():
    shader = """
    shader CombinedAdvancedStages {
        geometry {
            void main() { }
        }

        tessellation_control {
            void main() { }
        }

        tessellation_evaluation {
            void main() { }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    combined_code = GLSLCodeGen().generate(ast)
    geometry_code = GLSLCodeGen().generate_stage(ast, "geometry")

    assert "#ifdef GL_GEOMETRY_SHADER" in combined_code
    assert "#ifdef GL_TESS_CONTROL_SHADER" in combined_code
    assert "#ifdef GL_TESS_EVALUATION_SHADER" in combined_code
    assert "void geometry_main()" not in combined_code
    assert "void tessellation_control_main()" not in combined_code
    assert "void tessellation_evaluation_main()" not in combined_code
    assert combined_code.count("void main()") == 3

    assert "void main()" in geometry_code
    assert "void geometry_main()" not in geometry_code


def test_glsl_mesh_task_stage_entries_use_distinct_names():
    shader = """
    shader CombinedMeshTaskStages {
        task {
            void main() { }
        }

        mesh {
            void main() { }
        }

        amplification {
            void main() { }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    combined_code = GLSLCodeGen().generate(ast)
    mesh_code = GLSLCodeGen().generate_stage(ast, "mesh")

    assert "#ifdef GL_TASK_SHADER_EXT" in combined_code
    assert "#ifdef GL_MESH_SHADER_EXT" in combined_code
    assert "void task_main()" not in combined_code
    assert "void mesh_main()" not in combined_code
    assert "void amplification_main()" not in combined_code
    assert combined_code.count("void main()") == 3

    assert "void main()" in mesh_code
    assert "void mesh_main()" not in mesh_code


def test_glsl_mesh_task_stage_extensions_and_local_size_layouts():
    shader = """
    shader MeshTaskLocalSizes {
        struct TaskPayload {
            uint meshlet;
        };

        task {
            layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
            @taskPayloadSharedEXT TaskPayload payload;
            void main() {
                payload.meshlet = 0u;
                DispatchMesh(1, 1, 1);
            }
        }

        mesh {
            layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
            layout(triangles, max_vertices = 64, max_primitives = 32) out;
            @taskPayloadSharedEXT TaskPayload payload;
            perprimitive out vec3 primitiveNormal[32];
            out vec4 vertexColor[64];
            void main() {
                SetMeshOutputCounts(64, 32);
                primitiveNormal[0] = vec3(0.0, 0.0, 1.0);
                vertexColor[0] = vec4(float(payload.meshlet));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    combined_code = GLSLCodeGen().generate(ast)
    mesh_code = GLSLCodeGen().generate_stage(ast, "mesh")

    assert "#extension GL_EXT_mesh_shader : require" in combined_code
    assert combined_code.count("#extension GL_EXT_mesh_shader : require") == 1
    assert (
        "layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;"
        in combined_code
    )
    assert (
        "layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;"
        in combined_code
    )
    assert "#extension GL_EXT_mesh_shader : require" in mesh_code
    assert (
        "layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;" in mesh_code
    )
    assert "layout(triangles, max_vertices = 64, max_primitives = 32) out;" in mesh_code
    assert "taskPayloadSharedEXT TaskPayload payload;" in mesh_code
    assert "perprimitiveEXT out vec3 primitiveNormal[32];" in mesh_code
    assert "out vec4 vertexColor[64];" in mesh_code
    assert "SetMeshOutputsEXT(64, 32);" in mesh_code
    assert "primitiveNormal[0] = vec3(0.0, 0.0, 1.0);" in mesh_code
    assert "vertexColor[0] = vec4(float(payload.meshlet));" in mesh_code
    assert "SetMeshOutputCounts" not in mesh_code
    assert "EmitMeshTasksEXT(1, 1, 1);" in combined_code
    assert combined_code.count("taskPayloadSharedEXT TaskPayload payload;") == 1
    assert "DispatchMesh" not in combined_code
    assert (
        "layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;"
        not in mesh_code
    )


def test_glsl_mesh_layout_counts_accept_positive_integer_constants_and_aliases():
    shader = """
    shader MeshLayoutConstants {
        const int MAX_VERTEX_COUNT = 2 * 3;
        const int MAX_PRIMITIVE_COUNT = 1;

        mesh {
            void main()
                @outputtopology(line)
                @maxvertexcount(MAX_VERTEX_COUNT)
                @maxprimitivecount(MAX_PRIMITIVE_COUNT)
                @numthreads(1, 1, 1)
            {
                SetMeshOutputCounts(MAX_VERTEX_COUNT, MAX_PRIMITIVE_COUNT);
            }
        }
    }
    """

    mesh_code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")

    assert (
        "layout(lines, max_vertices = MAX_VERTEX_COUNT, "
        "max_primitives = MAX_PRIMITIVE_COUNT) out;" in mesh_code
    )
    assert "SetMeshOutputsEXT(MAX_VERTEX_COUNT, MAX_PRIMITIVE_COUNT);" in mesh_code


def test_glsl_mesh_layout_accepts_equivalent_aliases():
    shader = """
    shader MeshLayoutEquivalentAliases {
        mesh {
            void main()
                @outputtopology(triangle)
                @triangles
                @max_vertices(3)
                @maxvertexcount(3)
                @max_primitives(1)
                @maxprimitivecount(1)
                @numthreads(1, 1, 1)
            {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """

    mesh_code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")

    assert "layout(triangles, max_vertices = 3, max_primitives = 1) out;" in mesh_code
    assert "SetMeshOutputsEXT(3, 1);" in mesh_code


@pytest.mark.parametrize(
    ("attribute_lines", "message"),
    [
        (
            "@outputtopology(line)\n                @triangles",
            "Conflicting GLSL mesh output topology layout outputtopology line "
            "with triangles",
        ),
        (
            "@max_vertices(3)\n                @maxvertexcount(4)",
            "Conflicting GLSL mesh max_vertices layout max_vertices 3 "
            "with maxvertexcount 4",
        ),
        (
            "@max_primitives(1)\n                @maxprimitivecount(2)",
            "Conflicting GLSL mesh max_primitives layout max_primitives 1 "
            "with maxprimitivecount 2",
        ),
    ],
)
def test_glsl_mesh_layout_rejects_conflicting_aliases(attribute_lines, message):
    fallback_topology = (
        ""
        if "@outputtopology" in attribute_lines
        or any(
            f"@{name}" in attribute_lines for name in ("points", "lines", "triangles")
        )
        else "@outputtopology(triangle)"
    )
    fallback_max_vertices = (
        ""
        if "@max_vertices" in attribute_lines or "@maxvertexcount" in attribute_lines
        else "@max_vertices(3)"
    )
    fallback_max_primitives = (
        ""
        if "@max_primitives" in attribute_lines
        or "@maxprimitivecount" in attribute_lines
        else "@max_primitives(1)"
    )
    shader = f"""
    shader BadMeshLayoutAliases {{
        mesh {{
            void main()
                {fallback_topology}
                {fallback_max_vertices}
                {fallback_max_primitives}
                {attribute_lines}
                @numthreads(1, 1, 1)
            {{ }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


@pytest.mark.parametrize(
    ("attribute_line", "message"),
    [
        (
            "@max_vertices(0)",
            "GLSL mesh max_vertices layout requires a positive integer constant, got 0",
        ),
        (
            "@maxvertexcount(1.5)",
            r"GLSL mesh max_vertices layout requires a positive integer constant, got 1\.5",
        ),
        (
            "@max_primitives(0)",
            "GLSL mesh max_primitives layout requires a positive integer constant, got 0",
        ),
        (
            "@maxprimitivecount(-1)",
            r"GLSL mesh max_primitives layout requires a positive integer constant, got \(-1\)",
        ),
        (
            "@maxprimitivecount(1.5)",
            r"GLSL mesh max_primitives layout requires a positive integer constant, got 1\.5",
        ),
    ],
)
def test_glsl_mesh_layout_rejects_invalid_count_constants(attribute_line, message):
    fallback_max_vertices = (
        ""
        if attribute_line.startswith(("@max_vertices", "@maxvertexcount"))
        else "@max_vertices(3)"
    )
    fallback_max_primitives = (
        ""
        if attribute_line.startswith(("@max_primitives", "@maxprimitivecount"))
        else "@max_primitives(1)"
    )
    shader = f"""
    shader BadMeshLayoutCounts {{
        mesh {{
            void main()
                @outputtopology(triangle)
                {fallback_max_vertices}
                {fallback_max_primitives}
                {attribute_line}
                @numthreads(1, 1, 1)
            {{ }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


def test_glsl_task_dispatch_mesh_payload_argument_assigns_shared_payload_storage():
    shader = """
    shader MeshTaskPayloadDispatch {
        struct TaskPayload {
            uint meshlet;
        };

        task {
            @taskPayloadSharedEXT TaskPayload payload;
            void main() @numthreads(1, 1, 1) {
                TaskPayload localPayload;
                localPayload.meshlet = 7u;
                DispatchMesh(2, 3, 4, localPayload);
            }
        }
    }
    """

    task_code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "task")

    assert "taskPayloadSharedEXT TaskPayload payload;" in task_code
    assert "TaskPayload localPayload;" in task_code
    assert "localPayload.meshlet = 7u;" in task_code
    assert "payload = localPayload;" in task_code
    assert "EmitMeshTasksEXT(2, 3, 4);" in task_code
    assert "EmitMeshTasksEXT(2, 3, 4, localPayload)" not in task_code
    assert "DispatchMesh" not in task_code


def test_glsl_task_dispatch_mesh_accepts_scalar_integer_group_counts():
    shader = """
    shader MeshTaskDispatchCounts {
        task {
            void main() @numthreads(1, 1, 1) {
                uint groupsX = 2u;
                int groupsY = 3;
                DispatchMesh(groupsX, groupsY, 4);
            }
        }
    }
    """

    task_code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "task")

    assert "uint groupsX = 2u;" in task_code
    assert "int groupsY = 3;" in task_code
    assert "EmitMeshTasksEXT(groupsX, groupsY, 4);" in task_code
    assert "DispatchMesh" not in task_code


@pytest.mark.parametrize(
    ("declarations", "dispatch_call", "message"),
    [
        (
            "",
            "DispatchMesh(1.5, 1, 1);",
            r"GLSL DispatchMesh x group count requires a non-negative scalar "
            r"integer value, got float",
        ),
        (
            "",
            "DispatchMesh(1, -1, 1);",
            r"GLSL DispatchMesh y group count requires a non-negative scalar "
            r"integer value, got \(-1\)",
        ),
        (
            "vec2 groups = vec2(1.0);",
            "DispatchMesh(groups, 1, 1);",
            r"GLSL DispatchMesh x group count requires a non-negative scalar "
            r"integer value, got vec2",
        ),
        (
            "bool groups = true;",
            "DispatchMesh(1, 1, groups);",
            r"GLSL DispatchMesh z group count requires a non-negative scalar "
            r"integer value, got bool",
        ),
    ],
)
def test_glsl_task_dispatch_mesh_rejects_invalid_group_counts(
    declarations, dispatch_call, message
):
    shader = f"""
    shader MeshTaskDispatchCounts {{
        task {{
            void main() @numthreads(1, 1, 1) {{
                {declarations}
                {dispatch_call}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "task")


@pytest.mark.parametrize(
    "dispatch_call",
    [
        "DispatchMesh();",
        "DispatchMesh(1, 1);",
        "DispatchMesh(1, 1, 1, 1, 1);",
    ],
)
def test_glsl_task_dispatch_mesh_rejects_wrong_arity(dispatch_call):
    shader = f"""
    shader MeshTaskDispatchArity {{
        task {{
            void main() @numthreads(1, 1, 1) {{
                {dispatch_call}
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            r"GLSL DispatchMesh requires three group counts and optional "
            r"payload argument"
        ),
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "task")


def test_glsl_task_dispatch_mesh_rejects_wrong_arity_in_expression_context():
    shader = """
    shader MeshTaskDispatchArity {
        task {
            void helper(uint v) {
            }

            void main() @numthreads(1, 1, 1) {
                helper(DispatchMesh(1, 1));
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            r"GLSL DispatchMesh requires three group counts and optional "
            r"payload argument"
        ),
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "task")


def test_glsl_task_dispatch_mesh_payload_argument_rejects_ambiguous_shared_payload_targets():
    shader = """
    shader MeshTaskPayloadDispatch {
        struct TaskPayload {
            uint meshlet;
        };

        task {
            @taskPayloadSharedEXT TaskPayload payloadA;
            @taskPayloadSharedEXT TaskPayload payloadB;
            void main() @numthreads(1, 1, 1) {
                TaskPayload localPayload;
                localPayload.meshlet = 7u;
                DispatchMesh(2, 3, 4, localPayload);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)

    with pytest.raises(
        ValueError,
        match=(
            r"Ambiguous GLSL DispatchMesh payload target for TaskPayload: "
            r"payloadA, payloadB"
        ),
    ):
        GLSLCodeGen().generate_stage(ast, "task")


def test_glsl_task_dispatch_mesh_payload_argument_rejects_missing_shared_payload_storage():
    shader = """
    shader MeshTaskPayloadDispatch {
        struct TaskPayload {
            uint meshlet;
        };

        task {
            void main() @numthreads(1, 1, 1) {
                TaskPayload localPayload;
                localPayload.meshlet = 7u;
                DispatchMesh(2, 3, 4, localPayload);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)

    with pytest.raises(
        ValueError,
        match=(
            r"GLSL DispatchMesh payload argument requires "
            r"taskPayloadSharedEXT storage"
        ),
    ):
        GLSLCodeGen().generate_stage(ast, "task")


def test_glsl_task_dispatch_mesh_payload_argument_rejects_mismatched_shared_payload_type():
    shader = """
    shader MeshTaskPayloadDispatch {
        struct TaskPayload {
            uint meshlet;
        };

        struct OtherPayload {
            uint meshlet;
        };

        task {
            @taskPayloadSharedEXT TaskPayload payload;
            void main() @numthreads(1, 1, 1) {
                OtherPayload localPayload;
                localPayload.meshlet = 7u;
                DispatchMesh(2, 3, 4, localPayload);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)

    with pytest.raises(
        ValueError,
        match=(
            r"GLSL DispatchMesh payload argument type OtherPayload does not "
            r"match taskPayloadSharedEXT payload type TaskPayload"
        ),
    ):
        GLSLCodeGen().generate_stage(ast, "task")


def test_glsl_task_dispatch_mesh_payload_argument_allows_named_shared_payload_target():
    shader = """
    shader MeshTaskPayloadDispatch {
        struct TaskPayload {
            uint meshlet;
        };

        task {
            @taskPayloadSharedEXT TaskPayload payloadA;
            @taskPayloadSharedEXT TaskPayload payloadB;
            void main() @numthreads(1, 1, 1) {
                payloadA.meshlet = 7u;
                DispatchMesh(2, 3, 4, payloadA);
            }
        }
    }
    """

    task_code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "task")

    assert "taskPayloadSharedEXT TaskPayload payloadA;" in task_code
    assert "taskPayloadSharedEXT TaskPayload payloadB;" in task_code
    assert "payloadA.meshlet = 7u;" in task_code
    assert "EmitMeshTasksEXT(2, 3, 4);" in task_code
    assert "payloadA = payloadA;" not in task_code
    assert "DispatchMesh" not in task_code


@pytest.mark.parametrize(
    (
        "topology",
        "expected_layout",
        "index_assignment",
        "expected_index_assignment",
    ),
    [
        (
            "point",
            "layout(points, max_vertices = 1, max_primitives = 1) out;",
            "gl_PrimitivePointIndicesEXT[0] = 0u;",
            "gl_PrimitivePointIndicesEXT[0] = 0u;",
        ),
        (
            "line",
            "layout(lines, max_vertices = 2, max_primitives = 1) out;",
            "gl_PrimitiveLineIndicesEXT[0] = uvec2(0u, 1u);",
            "gl_PrimitiveLineIndicesEXT[0] = uvec2(0u, 1u);",
        ),
        (
            "triangle",
            "layout(triangles, max_vertices = 3, max_primitives = 1) out;",
            "gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0u, 1u, 2u);",
            "gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0u, 1u, 2u);",
        ),
    ],
)
def test_glsl_mesh_topology_builtin_index_arrays(
    topology,
    expected_layout,
    index_assignment,
    expected_index_assignment,
):
    max_vertices = {"point": 1, "line": 2, "triangle": 3}[topology]
    shader = f"""
    shader MeshTopologyBuiltins {{
        mesh {{
            void main()
                @outputtopology({topology})
                @max_vertices({max_vertices})
                @max_primitives(1)
                @numthreads(1, 1, 1)
            {{
                SetMeshOutputCounts({max_vertices}, 1);
                gl_MeshVerticesEXT[0].gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
                {index_assignment}
            }}
        }}
    }}
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "mesh"
    )

    assert "#extension GL_EXT_mesh_shader : require" in generated_code
    assert "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;" in (
        generated_code
    )
    assert expected_layout in generated_code
    assert f"SetMeshOutputsEXT({max_vertices}, 1);" in generated_code
    assert (
        "gl_MeshVerticesEXT[0].gl_Position = vec4(0.0, 0.0, 0.0, 1.0);"
        in generated_code
    )
    assert expected_index_assignment in generated_code
    assert "SetMeshOutputCounts" not in generated_code


def test_glsl_mesh_output_signature_parameters_lower_to_native_outputs():
    shader = """
    shader MeshOutputSignature {
        struct MeshVertex {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        struct MeshPrimitive {
            int primitiveId @ gl_PrimitiveID;
            vec3 normal @ NORMAL;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1],
                @primitives out MeshPrimitive prims[1]
            ) @numthreads(32, 1, 1)
              @outputtopology(triangle)
              @max_vertices(3)
              @max_primitives(1)
            {
                SetMeshOutputCounts(3, 1);
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                verts[0].uv = vec2(0.5, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
                prims[0].primitiveId = 7;
                prims[0].normal = vec3(0.0, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "mesh"
    )

    assert "layout(triangles, max_vertices = 3, max_primitives = 1) out;" in (
        generated_code
    )
    assert "layout(location = 5) out vec2 uv[3];" in generated_code
    assert "layout(location = 1) perprimitiveEXT out vec3 normal[1];" in generated_code
    assert "gl_MeshVerticesEXT[0].gl_Position = vec4(0.0, 0.0, 0.0, 1.0);" in (
        generated_code
    )
    assert "uv[0] = vec2(0.5, 1.0);" in generated_code
    assert "gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0u, 1u, 2u);" in generated_code
    assert "gl_MeshPrimitivesEXT[0].gl_PrimitiveID = 7;" in generated_code
    assert "normal[0] = vec3(0.0, 0.0, 1.0);" in generated_code
    assert "in MeshVertex verts[3];" not in generated_code
    assert "in uvec3 tris[1];" not in generated_code
    assert "in MeshPrimitive prims[1];" not in generated_code
    assert "verts[0]" not in generated_code
    assert "tris[0]" not in generated_code
    assert "prims[0]" not in generated_code
    assert "SetMeshOutputCounts" not in generated_code


def test_glsl_mesh_point_index_signature_parameter_accepts_scalar_uint_indices():
    shader = """
    shader MeshPointOutputSignature {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        struct MeshPrimitive {
            int primitiveId @ gl_PrimitiveID;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[1],
                @indices out uint points[1],
                @primitives out MeshPrimitive prims[1]
            ) @numthreads(1, 1, 1)
              @outputtopology(point)
              @max_vertices(1)
              @max_primitives(1)
            {
                SetMeshOutputCounts(1, 1);
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                points[0] = 0u;
                prims[0].primitiveId = 3;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "mesh"
    )

    assert "layout(points, max_vertices = 1, max_primitives = 1) out;" in (
        generated_code
    )
    assert "gl_PrimitivePointIndicesEXT[0] = 0u;" in generated_code
    assert "gl_MeshPrimitivesEXT[0].gl_PrimitiveID = 3;" in generated_code
    assert "points[0]" not in generated_code
    assert "SetMeshOutputCounts" not in generated_code


@pytest.mark.parametrize(
    ("parameter_declarations", "topology", "expected_error"),
    [
        (
            "@vertices out MeshVertex verts, "
            "@indices out uvec3 tris[1], "
            "@primitives out MeshPrimitive prims[1]",
            "triangle",
            r"@vertices 'verts' requires an array with explicit size",
        ),
        (
            "@vertices out MeshVertex verts[3], "
            "@indices out uvec3 tris, "
            "@primitives out MeshPrimitive prims[1]",
            "triangle",
            r"@indices 'tris' requires an array with explicit size",
        ),
        (
            "@vertices out MeshVertex verts[3], "
            "@indices out uvec3 tris[1], "
            "@primitives out MeshPrimitive prims",
            "triangle",
            r"@primitives 'prims' requires an array with explicit size",
        ),
        (
            "@vertices out vec4 verts[3], "
            "@indices out uvec3 tris[1], "
            "@primitives out MeshPrimitive prims[1]",
            "triangle",
            r"@vertices 'verts' requires a struct element type, got vec4",
        ),
        (
            "@vertices out MeshVertex verts[3], "
            "@indices out uvec3 tris[1], "
            "@primitives out vec3 prims[1]",
            "triangle",
            r"@primitives 'prims' requires a struct element type, got vec3",
        ),
        (
            "@vertices out MeshVertex verts[3], "
            "@indices out uvec2 tris[1], "
            "@primitives out MeshPrimitive prims[1]",
            "triangle",
            r"@indices 'tris' for triangles topology requires uvec3 elements, got uvec2",
        ),
        (
            "@vertices out MeshVertex verts[1], "
            "@indices out uvec2 points[1], "
            "@primitives out MeshPrimitive prims[1]",
            "point",
            r"@indices 'points' for points topology requires uint elements, got uvec2",
        ),
    ],
)
def test_glsl_mesh_output_signature_parameters_reject_malformed_shapes(
    parameter_declarations,
    topology,
    expected_error,
):
    max_vertices = {"point": 1, "line": 2, "triangle": 3}[topology]
    shader = f"""
    shader MeshMalformedOutputSignature {{
        struct MeshVertex {{
            vec4 position @ gl_Position;
        }};

        struct MeshPrimitive {{
            int primitiveId @ gl_PrimitiveID;
        }};

        mesh {{
            void main({parameter_declarations})
              @numthreads(1, 1, 1)
              @outputtopology({topology})
              @max_vertices({max_vertices})
              @max_primitives(1)
            {{
                SetMeshOutputCounts({max_vertices}, 1);
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


@pytest.mark.parametrize(
    ("parameter_declarations", "max_vertices", "max_primitives", "expected_error"),
    [
        (
            "@vertices out MeshVertex verts[2], "
            "@indices out uvec3 tris[1], "
            "@primitives out MeshPrimitive prims[1]",
            3,
            1,
            r"@vertices 'verts' array size 2 must match max_vertices 3",
        ),
        (
            "@vertices out MeshVertex verts[3], "
            "@indices out uvec3 tris[2], "
            "@primitives out MeshPrimitive prims[1]",
            3,
            1,
            r"@indices 'tris' array size 2 must match max_primitives 1",
        ),
        (
            "@vertices out MeshVertex verts[3], "
            "@indices out uvec3 tris[1], "
            "@primitives out MeshPrimitive prims[2]",
            3,
            1,
            r"@primitives 'prims' array size 2 must match max_primitives 1",
        ),
    ],
)
def test_glsl_mesh_output_signature_parameters_reject_layout_count_mismatches(
    parameter_declarations,
    max_vertices,
    max_primitives,
    expected_error,
):
    shader = f"""
    shader MeshMismatchedOutputCounts {{
        struct MeshVertex {{
            vec4 position @ gl_Position;
        }};

        struct MeshPrimitive {{
            int primitiveId @ gl_PrimitiveID;
        }};

        mesh {{
            void main({parameter_declarations})
              @numthreads(1, 1, 1)
              @outputtopology(triangle)
              @max_vertices({max_vertices})
              @max_primitives({max_primitives})
            {{
                SetMeshOutputCounts({max_vertices}, {max_primitives});
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


@pytest.mark.parametrize(
    ("set_counts", "expected_error"),
    [
        (
            "SetMeshOutputCounts(4, 1);",
            r"SetMeshOutputCounts vertex count 4 exceeds @vertices array size 3",
        ),
        (
            "SetMeshOutputCounts(3, 2);",
            r"SetMeshOutputCounts primitive count 2 exceeds @primitives array size 1",
        ),
        (
            "SetMeshOutputCounts(3);",
            r"SetMeshOutputCounts requires exactly two arguments",
        ),
        (
            "SetMeshOutputCounts(1.5, 1);",
            r"SetMeshOutputCounts vertex count requires a non-negative scalar "
            r"integer value, got float",
        ),
        (
            "SetMeshOutputCounts(-1, 1);",
            r"SetMeshOutputCounts vertex count requires a non-negative scalar "
            r"integer value, got \(-1\)",
        ),
        (
            "vec2 count = vec2(1.0); SetMeshOutputCounts(count, 1);",
            r"SetMeshOutputCounts vertex count requires a non-negative scalar "
            r"integer value, got vec2",
        ),
        (
            "bool count = true; SetMeshOutputCounts(1, count);",
            r"SetMeshOutputCounts primitive count requires a non-negative "
            r"scalar integer value, got bool",
        ),
    ],
)
def test_glsl_mesh_output_signature_parameters_reject_bad_output_count_calls(
    set_counts,
    expected_error,
):
    shader = f"""
    shader MeshBadOutputCountCalls {{
        struct MeshVertex {{
            vec4 position @ gl_Position;
        }};

        struct MeshPrimitive {{
            int primitiveId @ gl_PrimitiveID;
        }};

        mesh {{
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1],
                @primitives out MeshPrimitive prims[1]
            ) @numthreads(1, 1, 1)
              @outputtopology(triangle)
              @max_vertices(3)
              @max_primitives(1)
            {{
                {set_counts}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


@pytest.mark.parametrize(
    ("mesh_statement", "expected_error"),
    [
        (
            "verts[3].position = vec4(0.0, 0.0, 0.0, 1.0);",
            r"output @vertices 'verts' index 3 out of range; valid range is 0..2",
        ),
        (
            "tris[1] = uvec3(0u, 1u, 2u);",
            r"output @indices 'tris' index 1 out of range; valid range is 0..0",
        ),
        (
            "prims[-1].primitiveId = 7;",
            r"output @primitives 'prims' index -1 out of range; valid range is 0..0",
        ),
    ],
)
def test_glsl_mesh_output_signature_parameters_reject_literal_index_bounds(
    mesh_statement,
    expected_error,
):
    shader = f"""
    shader MeshBadOutputIndex {{
        struct MeshVertex {{
            vec4 position @ gl_Position;
        }};

        struct MeshPrimitive {{
            int primitiveId @ gl_PrimitiveID;
        }};

        mesh {{
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1],
                @primitives out MeshPrimitive prims[1]
            ) @numthreads(1, 1, 1)
              @outputtopology(triangle)
              @max_vertices(3)
              @max_primitives(1)
            {{
                SetMeshOutputCounts(3, 1);
                {mesh_statement}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


@pytest.mark.parametrize(
    ("index_declaration", "mesh_statement", "expected_error"),
    [
        (
            "float vertexIndex = 0.0;",
            "verts[vertexIndex].position = vec4(0.0, 0.0, 0.0, 1.0);",
            r"output @vertices 'verts' index requires a scalar integer value, got float",
        ),
        (
            "float indexIndex = 0.0;",
            "tris[indexIndex] = uvec3(0u, 1u, 2u);",
            r"output @indices 'tris' index requires a scalar integer value, got float",
        ),
        (
            "vec2 primitiveIndex = vec2(0.0, 1.0);",
            "prims[primitiveIndex].primitiveId = 7;",
            r"output @primitives 'prims' index requires a scalar integer value, got vec2",
        ),
    ],
)
def test_glsl_mesh_output_signature_parameters_reject_non_integer_dynamic_indices(
    index_declaration,
    mesh_statement,
    expected_error,
):
    shader = f"""
    shader MeshBadDynamicOutputIndex {{
        struct MeshVertex {{
            vec4 position @ gl_Position;
        }};

        struct MeshPrimitive {{
            int primitiveId @ gl_PrimitiveID;
        }};

        mesh {{
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1],
                @primitives out MeshPrimitive prims[1]
            ) @numthreads(1, 1, 1)
              @outputtopology(triangle)
              @max_vertices(3)
              @max_primitives(1)
            {{
                SetMeshOutputCounts(3, 1);
                {index_declaration}
                {mesh_statement}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


@pytest.mark.parametrize(
    ("helper_call", "expected_error"),
    [
        (
            "SetVertex(3, vec4(0.0, 0.0, 0.0, 1.0));",
            r"SetVertex vertex index 3 out of range; valid range is 0..2",
        ),
        (
            "SetVertex(-1, vec4(0.0, 0.0, 0.0, 1.0));",
            r"SetVertex vertex index -1 out of range; valid range is 0..2",
        ),
        (
            "SetPrimitive(1, uvec3(0u, 1u, 2u));",
            r"SetPrimitive primitive index 1 out of range; valid range is 0..0",
        ),
    ],
)
def test_glsl_mesh_helper_intrinsics_reject_literal_index_bounds(
    helper_call,
    expected_error,
):
    shader = f"""
    shader MeshHelperBadIndex {{
        mesh {{
            void main()
                @numthreads(1, 1, 1)
                @outputtopology(triangle)
                @max_vertices(3)
                @max_primitives(1)
            {{
                SetMeshOutputCounts(3, 1);
                {helper_call}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


@pytest.mark.parametrize(
    ("index_declaration", "helper_call", "expected_error"),
    [
        (
            "float vertexIndex = 0.0;",
            "SetVertex(vertexIndex, vec4(0.0, 0.0, 0.0, 1.0));",
            r"SetVertex vertex index requires a scalar integer value, got float",
        ),
        (
            "uvec2 primitiveIndex = uvec2(0u, 1u);",
            "SetPrimitive(primitiveIndex, uvec3(0u, 1u, 2u));",
            r"SetPrimitive primitive index requires a scalar integer value, got uvec2",
        ),
    ],
)
def test_glsl_mesh_helper_intrinsics_reject_non_integer_dynamic_indices(
    index_declaration,
    helper_call,
    expected_error,
):
    shader = f"""
    shader MeshHelperBadDynamicIndex {{
        mesh {{
            void main()
                @numthreads(1, 1, 1)
                @outputtopology(triangle)
                @max_vertices(3)
                @max_primitives(1)
            {{
                SetMeshOutputCounts(3, 1);
                {index_declaration}
                {helper_call}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


def test_glsl_mesh_whole_output_constructor_assignments_expand_to_native_outputs():
    shader = """
    shader MeshWholeOutputConstructors {
        struct MeshVertex {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        struct MeshPrimitive {
            int primitiveId @ gl_PrimitiveID;
            vec3 normal @ NORMAL;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @primitives out MeshPrimitive prims[1]
            ) @numthreads(1, 1, 1)
              @outputtopology(triangle)
              @max_vertices(3)
              @max_primitives(1)
            {
                SetMeshOutputCounts(3, 1);
                verts[1] = MeshVertex {
                    position: vec4(1.0, 0.0, 0.0, 1.0),
                    uv: vec2(0.25, 0.75)
                };
                prims[0] = MeshPrimitive {
                    primitiveId: 11,
                    normal: vec3(0.0, 1.0, 0.0)
                };
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "mesh"
    )

    assert (
        "gl_MeshVerticesEXT[1].gl_Position = vec4(1.0, 0.0, 0.0, 1.0);"
        in generated_code
    )
    assert "uv[1] = vec2(0.25, 0.75);" in generated_code
    assert "gl_MeshPrimitivesEXT[0].gl_PrimitiveID = 11;" in generated_code
    assert "normal[0] = vec3(0.0, 1.0, 0.0);" in generated_code
    assert "MeshVertex(" not in generated_code
    assert "MeshPrimitive(" not in generated_code
    assert "verts[1]" not in generated_code
    assert "prims[0]" not in generated_code
    assert "SetMeshOutputCounts" not in generated_code


@pytest.mark.parametrize(
    ("bad_assignment", "expected_error"),
    [
        (
            "verts[0] = vec4(0.0, 0.0, 0.0, 1.0);",
            r"GLSL mesh output @vertices 'verts' assignment requires "
            r"MeshVertex value, got vec4",
        ),
        (
            "prims[0] = vec3(0.0, 1.0, 0.0);",
            r"GLSL mesh output @primitives 'prims' assignment requires "
            r"MeshPrimitive value, got vec3",
        ),
        (
            "prims[0] = MeshVertex { "
            "position: vec4(0.0, 0.0, 0.0, 1.0), "
            "uv: vec2(0.25, 0.75) };",
            r"GLSL mesh output @primitives 'prims' assignment requires "
            r"MeshPrimitive value, got MeshVertex",
        ),
        (
            "verts[0] = MeshVertex { " "position: vec4(0.0, 0.0, 0.0, 1.0) };",
            r"GLSL mesh output @vertices 'verts' assignment requires a "
            r"complete MeshVertex constructor value",
        ),
    ],
)
def test_glsl_mesh_whole_output_assignments_reject_mismatched_values(
    bad_assignment,
    expected_error,
):
    shader = f"""
    shader MeshWholeOutputBadValue {{
        struct MeshVertex {{
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        }};

        struct MeshPrimitive {{
            int primitiveId @ gl_PrimitiveID;
            vec3 normal @ NORMAL;
        }};

        mesh {{
            void main(
                @vertices out MeshVertex verts[3],
                @primitives out MeshPrimitive prims[1]
            ) @numthreads(1, 1, 1)
              @outputtopology(triangle)
              @max_vertices(3)
              @max_primitives(1)
            {{
                SetMeshOutputCounts(3, 1);
                {bad_assignment}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


@pytest.mark.parametrize(
    (
        "topology",
        "max_vertices",
        "primitive_value",
        "expected_primitive_assignment",
    ),
    [
        (
            "point",
            1,
            "0u",
            "gl_PrimitivePointIndicesEXT[0] = 0u;",
        ),
        (
            "line",
            2,
            "uvec2(0u, 1u)",
            "gl_PrimitiveLineIndicesEXT[0] = uvec2(0u, 1u);",
        ),
        (
            "triangle",
            3,
            "uvec3(0u, 1u, 2u)",
            "gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0u, 1u, 2u);",
        ),
    ],
)
def test_glsl_mesh_set_vertex_and_set_primitive_helpers_lower_to_builtins(
    topology,
    max_vertices,
    primitive_value,
    expected_primitive_assignment,
):
    shader = f"""
    shader MeshHelperIntrinsics {{
        mesh {{
            void main()
                @numthreads(1, 1, 1)
                @outputtopology({topology})
                @max_vertices({max_vertices})
                @max_primitives(1)
            {{
                vec3 position = vec3(0.0, 0.5, 1.0);
                SetMeshOutputCounts({max_vertices}, 1);
                SetVertex(0, position);
                SetVertex({max_vertices - 1}, vec4(1.0, 0.0, 0.0, 1.0));
                SetPrimitive(0, {primitive_value});
            }}
        }}
    }}
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "mesh"
    )

    assert "gl_MeshVerticesEXT[0].gl_Position = vec4(position, 1.0);" in generated_code
    assert (
        f"gl_MeshVerticesEXT[{max_vertices - 1}].gl_Position = "
        "vec4(1.0, 0.0, 0.0, 1.0);"
    ) in generated_code
    assert expected_primitive_assignment in generated_code
    assert "SetVertex" not in generated_code
    assert "SetPrimitive" not in generated_code
    assert "SetMeshOutputCounts" not in generated_code


def test_glsl_mesh_set_vertex_rejects_invalid_arity():
    shader = """
    shader MeshHelperBadSetVertexArity {
        mesh {
            void main()
                @numthreads(1, 1, 1)
                @outputtopology(point)
                @max_vertices(1)
                @max_primitives(1)
            {
                SetMeshOutputCounts(1, 1);
                SetVertex(0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=r"GLSL mesh SetVertex helper requires exactly two arguments",
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


def test_glsl_mesh_set_vertex_rejects_non_position_vector():
    shader = """
    shader MeshHelperBadSetVertexShape {
        mesh {
            void main()
                @numthreads(1, 1, 1)
                @outputtopology(point)
                @max_vertices(1)
                @max_primitives(1)
            {
                SetMeshOutputCounts(1, 1);
                SetVertex(0, 1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=r"GLSL mesh SetVertex position requires vec3 or vec4, got float",
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


def test_glsl_mesh_set_primitive_rejects_invalid_arity():
    shader = """
    shader MeshHelperBadSetPrimitiveArity {
        mesh {
            void main()
                @numthreads(1, 1, 1)
                @outputtopology(triangle)
                @max_vertices(3)
                @max_primitives(1)
            {
                SetMeshOutputCounts(3, 1);
                SetPrimitive(0, 0u, 1u);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=r"GLSL mesh SetPrimitive helper requires exactly two arguments",
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


@pytest.mark.parametrize(
    ("topology", "max_vertices", "bad_indices", "expected_type", "actual_type"),
    [
        ("point", 1, "uvec2(0u, 1u)", "uint", "uvec2"),
        ("line", 2, "0u", "uvec2", "uint"),
        ("triangle", 3, "uvec2(0u, 1u)", "uvec3", "uvec2"),
    ],
)
def test_glsl_mesh_set_primitive_rejects_topology_mismatched_indices(
    topology,
    max_vertices,
    bad_indices,
    expected_type,
    actual_type,
):
    shader = f"""
    shader MeshHelperBadSetPrimitiveShape {{
        mesh {{
            void main()
                @numthreads(1, 1, 1)
                @outputtopology({topology})
                @max_vertices({max_vertices})
                @max_primitives(1)
            {{
                SetMeshOutputCounts({max_vertices}, 1);
                SetPrimitive(0, {bad_indices});
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            r"GLSL mesh SetPrimitive for .* topology requires "
            f"{expected_type} primitive indices, got {actual_type}"
        ),
    ):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


@pytest.mark.parametrize(
    ("nested_helper_statement", "expected_error"),
    [
        (
            "float sink = SetVertex(0, 1.0);",
            r"GLSL mesh SetVertex position requires vec3 or vec4, got float",
        ),
        (
            "uint sink = SetPrimitive(0, uvec2(0u, 1u));",
            r"GLSL mesh SetPrimitive for .* topology requires "
            r"uvec3 primitive indices, got uvec2",
        ),
        (
            "float sink = SetVertex(0, vec4(0.0, 0.0, 0.0, 1.0));",
            r"GLSL mesh SetVertex helper can only be used as a statement",
        ),
        (
            "uint sink = SetPrimitive(0, uvec3(0u, 1u, 2u));",
            r"GLSL mesh SetPrimitive helper can only be used as a statement",
        ),
    ],
)
def test_glsl_mesh_helpers_reject_expression_contexts(
    nested_helper_statement,
    expected_error,
):
    shader = f"""
    shader MeshHelperExpressionContext {{
        mesh {{
            void main()
                @numthreads(1, 1, 1)
                @outputtopology(triangle)
                @max_vertices(3)
                @max_primitives(1)
            {{
                SetMeshOutputCounts(3, 1);
                {nested_helper_statement}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


def test_glsl_ray_stage_entries_use_distinct_names():
    shader = """
    shader CombinedRayStages {
        struct RayPayload {
            vec4 color;
        };

        struct CallableData {
            vec4 value;
        };

        struct ShaderRecordData {
            uint materialIndex;
        };

        accelerationStructureEXT topLevelAS @binding(0);
        ShaderRecordData shaderRecord @glsl_buffer_block(shaderRecordEXT);

        ray_generation {
            layout(location = 0) @rayPayloadEXT RayPayload rayPayload;
            layout(location = 1) @callableDataEXT CallableData callableData;

            void main() {
                TraceRay(
                    topLevelAS,
                    gl_RayFlagsNoneEXT,
                    0xff,
                    0,
                    1,
                    0,
                    vec3(0.0),
                    0.001,
                    vec3(0.0, 0.0, 1.0),
                    1000.0,
                    0
                );
                callableData.value = vec4(float(shaderRecord.materialIndex));
                CallShader(0, 1);
            }
        }

        ray_closest_hit {
            layout(location = 0) @rayPayloadInEXT RayPayload closestPayload;
            @hitAttributeEXT vec2 hitAttributes;

            void main() {
                closestPayload.color = vec4(hitAttributes, 0.0, 1.0);
            }
        }

        ray_any_hit {
            layout(location = 0) @rayPayloadInEXT RayPayload anyPayload;
            @hitAttributeEXT vec2 anyHitAttributes;

            void main() {
                IgnoreHit();
            }
        }

        ray_miss {
            layout(location = 0) @rayPayloadInEXT RayPayload missPayload;

            void main() {
                missPayload.color = vec4(0.0);
            }
        }

        ray_intersection {
            @hitAttributeEXT vec2 reportedAttributes;

            void main() {
                ReportHit(1.0, gl_HitKindFrontFacingTriangleEXT);
            }
        }

        ray_callable {
            layout(location = 1) @callableDataInEXT CallableData callableInput;

            void main() {
                callableInput.value = vec4(1.0);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    combined_code = GLSLCodeGen().generate(ast)
    ray_miss_code = GLSLCodeGen().generate_stage(ast, "ray_miss")

    assert "#ifdef GL_RAYGEN_SHADER_EXT" in combined_code
    assert "#ifdef GL_CLOSEST_HIT_SHADER_EXT" in combined_code
    assert "#ifdef GL_ANY_HIT_SHADER_EXT" in combined_code
    assert "#ifdef GL_MISS_SHADER_EXT" in combined_code
    assert "#ifdef GL_INTERSECTION_SHADER_EXT" in combined_code
    assert "#ifdef GL_CALLABLE_SHADER_EXT" in combined_code
    assert "void ray_generation_main()" not in combined_code
    assert "void ray_closest_hit_main()" not in combined_code
    assert "void ray_any_hit_main()" not in combined_code
    assert "void ray_miss_main()" not in combined_code
    assert "void ray_intersection_main()" not in combined_code
    assert "void ray_callable_main()" not in combined_code
    assert combined_code.count("void main()") == 6
    assert combined_code.lstrip().startswith("#version 460 core")
    assert "#extension GL_EXT_ray_tracing : require" in combined_code
    assert "#extension GL_EXT_ray_query : require" not in combined_code
    assert (
        "layout(binding = 0) uniform accelerationStructureEXT topLevelAS;"
        in combined_code
    )
    assert "layout(shaderRecordEXT) buffer ShaderRecordData" in combined_code
    assert "layout(shaderRecordEXT, binding" not in combined_code
    assert "uint materialIndex;" in combined_code
    assert (
        "callableData.value = vec4(float(shaderRecord.materialIndex));" in combined_code
    )
    assert "layout(location = 0) rayPayloadEXT RayPayload rayPayload;" in combined_code
    assert (
        "layout(location = 0) rayPayloadInEXT RayPayload closestPayload;"
        in combined_code
    )
    assert (
        "layout(location = 0) rayPayloadInEXT RayPayload anyPayload;" in combined_code
    )
    assert (
        "layout(location = 0) rayPayloadInEXT RayPayload missPayload;" in combined_code
    )
    assert "hitAttributeEXT vec2 hitAttributes;" in combined_code
    assert "hitAttributeEXT vec2 anyHitAttributes;" in combined_code
    assert "hitAttributeEXT vec2 reportedAttributes;" in combined_code
    assert (
        "layout(location = 1) callableDataEXT CallableData callableData;"
        in combined_code
    )
    assert (
        "layout(location = 1) callableDataInEXT CallableData callableInput;"
        in combined_code
    )
    assert "traceRayEXT(" in combined_code
    assert "executeCallableEXT(0, 1);" in combined_code
    assert "ignoreIntersectionEXT;" in combined_code
    assert (
        "reportIntersectionEXT(1.0, gl_HitKindFrontFacingTriangleEXT);" in combined_code
    )
    assert "TraceRay" not in combined_code
    assert "CallShader" not in combined_code
    assert "IgnoreHit" not in combined_code
    assert "ReportHit" not in combined_code

    assert "void main()" in ray_miss_code
    assert "void ray_miss_main()" not in ray_miss_code
    assert ray_miss_code.lstrip().startswith("#version 460 core")
    assert "#extension GL_EXT_ray_tracing : require" in ray_miss_code
    assert "#extension GL_EXT_ray_query : require" not in ray_miss_code
    assert (
        "layout(location = 0) rayPayloadInEXT RayPayload missPayload;" in ray_miss_code
    )
    assert "rayPayloadEXT RayPayload rayPayload;" not in ray_miss_code


def test_glsl_ray_hit_position_fetch_builtin_emits_extension_only_for_users():
    shader = """
    shader RayHitPositionFetch {
        ray_closest_hit {
            void main() {
                vec3 p0 = gl_HitTriangleVertexPositionsEXT[0];
                vec3 p1 = gl_HitTriangleVertexPositionsEXT[1];
                vec3 edge = p1 - p0;
            }
        }

        ray_miss {
            void main() { }
        }
    }
    """
    ast = crosstl.translator.parse(shader)

    closest_hit_code = GLSLCodeGen().generate_stage(ast, "ray_closest_hit")
    miss_code = GLSLCodeGen().generate_stage(ast, "ray_miss")
    combined_code = GLSLCodeGen().generate(ast)

    assert closest_hit_code.lstrip().startswith("#version 460 core")
    assert "#extension GL_EXT_ray_tracing : require" in closest_hit_code
    assert "#extension GL_EXT_ray_tracing_position_fetch : require" in closest_hit_code
    assert "vec3 p0 = gl_HitTriangleVertexPositionsEXT[0];" in closest_hit_code
    assert "vec3 p1 = gl_HitTriangleVertexPositionsEXT[1];" in closest_hit_code
    assert "vec3 edge = (p1 - p0);" in closest_hit_code

    assert miss_code.lstrip().startswith("#version 460 core")
    assert "#extension GL_EXT_ray_tracing : require" in miss_code
    assert "#extension GL_EXT_ray_tracing_position_fetch : require" not in miss_code
    assert "gl_HitTriangleVertexPositionsEXT" not in miss_code

    assert (
        combined_code.count("#extension GL_EXT_ray_tracing_position_fetch : require")
        == 1
    )
    assert "#ifdef GL_CLOSEST_HIT_SHADER_EXT" in combined_code
    assert "#ifdef GL_MISS_SHADER_EXT" in combined_code
    assert "void ray_closest_hit_main()" not in combined_code
    assert "void ray_miss_main()" not in combined_code
    assert combined_code.count("void main()") == 2


def test_glsl_ray_query_methods_lower_to_ext_functions():
    shader = """
    shader RayQueryCompute {
        accelerationStructureEXT topLevelAS @binding(0);

        compute {
            void main() {
                RayQuery<RAY_FLAG_NONE> rq;
                rayQueryInitializeEXT(
                    rq,
                    topLevelAS,
                    gl_RayFlagsNoneEXT,
                    255u,
                    vec3(0.0),
                    0.001,
                    vec3(0.0, 0.0, 1.0),
                    100.0
                );
                bool active = rq.Proceed();
                uint candidateType = rq.CandidateType();
                uint committedType = rq.CommittedType();
                uint primitiveIndex = rq.CandidatePrimitiveIndex();
                uint instanceId = rq.CommittedInstanceID();
                uint geometryIndex = rq.CandidateGeometryIndex();
                vec3 origin = rq.CommittedObjectRayOrigin();
                vec3 direction = rq.CandidateObjectRayDirection();
                float hitT = rq.CommittedRayT();
                rq.Abort();
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert generated_code.lstrip().startswith("#version 460 core")
    assert "#extension GL_EXT_ray_query : require" in generated_code
    assert "#extension GL_EXT_ray_tracing : require" not in generated_code
    assert "layout(binding = 0) uniform accelerationStructureEXT topLevelAS;" in (
        generated_code
    )
    assert "rayQueryEXT rq;" in generated_code
    assert "rayQueryInitializeEXT(" in generated_code
    assert "bool active_ = rayQueryProceedEXT(rq);" in generated_code
    assert (
        "uint candidateType = rayQueryGetIntersectionTypeEXT(rq, false);"
        in generated_code
    )
    assert (
        "uint committedType = rayQueryGetIntersectionTypeEXT(rq, true);"
        in generated_code
    )
    assert (
        "uint primitiveIndex = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);"
        in generated_code
    )
    assert (
        "uint instanceId = rayQueryGetIntersectionInstanceIdEXT(rq, true);"
        in generated_code
    )
    assert (
        "uint geometryIndex = rayQueryGetIntersectionGeometryIndexEXT(rq, false);"
        in generated_code
    )
    assert (
        "vec3 origin = rayQueryGetIntersectionObjectRayOriginEXT(rq, true);"
        in generated_code
    )
    assert (
        "vec3 direction = rayQueryGetIntersectionObjectRayDirectionEXT(rq, false);"
        in generated_code
    )
    assert "float hitT = rayQueryGetIntersectionTEXT(rq, true);" in generated_code
    assert "rayQueryTerminateEXT(rq);" in generated_code
    assert ".Proceed(" not in generated_code
    assert ".CandidateType(" not in generated_code
    assert ".Abort(" not in generated_code


def test_glsl_ray_query_trace_ray_inline_raydesc_lowers_to_initialize_fields():
    shader = """
    shader RayQueryTraceRayInlineRayDesc {
        struct RayDesc {
            vec3 Origin;
            float TMin;
            vec3 Direction;
            float TMax;
        };

        accelerationStructureEXT topLevelAS @binding(0);

        compute {
            void main() {
                RayDesc ray = RayDesc(
                    vec3(0.0, 1.0, 2.0),
                    0.001,
                    vec3(0.0, 0.0, 1.0),
                    100.0
                );
                RayQuery<RAY_FLAG_NONE> rq;
                rq.TraceRayInline(topLevelAS, gl_RayFlagsNoneEXT, 255u, ray);
                bool active = rq.Proceed();
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "#extension GL_EXT_ray_query : require" in generated_code
    assert "rayQueryEXT rq;" in generated_code
    assert (
        "rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsNoneEXT, 255u, "
        "ray.Origin, ray.TMin, ray.Direction, ray.TMax);" in generated_code
    )
    assert (
        "rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsNoneEXT, 255u, ray);"
        not in (generated_code)
    )
    assert "bool active_ = rayQueryProceedEXT(rq);" in generated_code
    assert ".TraceRayInline(" not in generated_code


def test_glsl_ray_query_lowercase_importer_type_lowers_to_ext_functions():
    # Reduced from FidelityFX SDK raytracing_common.hlsl after HLSL import.
    shader = """
    shader LowercaseRayQueryImporterType {
        struct RayDesc {
            vec3 Origin;
            float TMin;
            vec3 Direction;
            float TMax;
        };

        accelerationStructureEXT topLevelAS @binding(0);

        compute {
            void main() {
                RayDesc ray = RayDesc(
                    vec3(0.0),
                    0.01,
                    vec3(0.0, 0.0, 1.0),
                    1024.0
                );
                rayQuery rayQuery;
                rayQuery.TraceRayInline(topLevelAS, 0, 255, ray);
                bool active = rayQuery.Proceed();
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "#extension GL_EXT_ray_query : require" in generated_code
    assert "rayQueryEXT rayQuery;" in generated_code
    assert (
        "rayQueryInitializeEXT(rayQuery, topLevelAS, 0, 255, "
        "ray.Origin, ray.TMin, ray.Direction, ray.TMax);" in generated_code
    )
    assert "bool active_ = rayQueryProceedEXT(rayQuery);" in generated_code
    assert ".TraceRayInline(" not in generated_code


def test_glsl_target_stage_ray_query_extensions_are_scoped_to_emitted_stage():
    shader = """
    shader ScopedRayQueryExtensions {
        vertex {
            void main() { }
        }

        compute {
            void main() {
                RayQuery<RAY_FLAG_NONE> rq;
                bool active = rq.Proceed();
            }
        }
    }
    """
    ast = crosstl.translator.parse(shader)

    vertex_code = GLSLCodeGen().generate_stage(ast, "vertex")
    compute_code = GLSLCodeGen().generate_stage(ast, "compute")
    combined_code = GLSLCodeGen().generate(ast)

    assert vertex_code.lstrip().startswith("#version 450 core")
    assert "#extension GL_EXT_ray_query : require" not in vertex_code
    assert "rayQueryEXT" not in vertex_code
    assert "rayQueryProceedEXT" not in vertex_code

    assert compute_code.lstrip().startswith("#version 460 core")
    assert "#extension GL_EXT_ray_query : require" in compute_code
    assert "rayQueryEXT rq;" in compute_code
    assert "bool active_ = rayQueryProceedEXT(rq);" in compute_code

    assert combined_code.lstrip().startswith("#version 460 core")
    assert combined_code.count("#extension GL_EXT_ray_query : require") == 1
    assert "#ifdef GL_VERTEX_SHADER" in combined_code
    assert "#ifdef GL_COMPUTE_SHADER" in combined_code
    assert "void vertex_main()" not in combined_code
    assert "void compute_main()" not in combined_code
    assert combined_code.count("void main()") == 2


def test_glsl_acceleration_structure_globals_emit_extension_for_non_ray_stage():
    shader = """
    shader AccelerationStructureHeader {
        accelerationStructureEXT topLevelAS @binding(3);

        vertex {
            void main() { }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "vertex"
    )

    assert generated_code.lstrip().startswith("#version 460 core")
    assert "#extension GL_EXT_ray_query : require" in generated_code
    assert "#extension GL_EXT_ray_tracing : require" not in generated_code
    assert (
        "layout(binding = 3) uniform accelerationStructureEXT topLevelAS;"
        in generated_code
    )


def test_glsl_generic_ray_query_member_calls_lower_to_ext_functions():
    shader = """
    shader RayQueryGenericMethods {
        accelerationStructureEXT topLevelAS @binding(0);

        compute {
            void main() {
                RayQuery<RAY_FLAG_NONE> rq;
                rq.Initialize(
                    topLevelAS,
                    gl_RayFlagsNoneEXT,
                    255u,
                    vec3(0.0),
                    0.001,
                    vec3(0.0, 0.0, 1.0),
                    100.0
                );
                vec3 worldOrigin = rq.WorldRayOrigin();
                vec3 worldDirection = rq.WorldRayDirection();
                uint rayFlags = rq.RayFlags();
                float tMin = rq.RayTMin();
                uint customIndex = rq.CommittedInstanceCustomIndex();
                uint sbtOffset =
                    rq.CandidateInstanceShaderBindingTableRecordOffset();
                vec2 barycentrics = rq.CandidateTriangleBarycentrics();
                bool frontFace = rq.CommittedTriangleFrontFace();
                bool aabbOpaque = rq.CandidateAABBOpaque();
                mat3x4 objectToWorld = rq.CommittedObjectToWorld();
                mat3x4 worldToObject = rq.CandidateWorldToObject3x4();
                vec3 trianglePositions[3];
                rq.CandidateTriangleVertexPositions(trianglePositions);
                rq.CommittedTriangleVertexPositions(trianglePositions);
                rq.GenerateIntersection(1.0);
                rq.ConfirmIntersection();
                rq.Terminate();
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "#extension GL_EXT_ray_query : require" in generated_code
    assert "#extension GL_EXT_ray_tracing_position_fetch : require" in generated_code
    assert "rayQueryEXT rq;" in generated_code
    assert "rayQueryInitializeEXT(" in generated_code
    assert "vec3 worldOrigin = rayQueryGetWorldRayOriginEXT(rq);" in generated_code
    assert (
        "vec3 worldDirection = rayQueryGetWorldRayDirectionEXT(rq);" in generated_code
    )
    assert "uint rayFlags = rayQueryGetRayFlagsEXT(rq);" in generated_code
    assert "float tMin = rayQueryGetRayTMinEXT(rq);" in generated_code
    assert (
        "uint customIndex = "
        "rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);" in generated_code
    )
    assert (
        "uint sbtOffset = "
        "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(rq, false);"
        in generated_code
    )
    assert (
        "vec2 barycentrics = rayQueryGetIntersectionBarycentricsEXT(rq, false);"
        in generated_code
    )
    assert (
        "bool frontFace = rayQueryGetIntersectionFrontFaceEXT(rq, true);"
        in generated_code
    )
    assert (
        "bool aabbOpaque = rayQueryGetIntersectionCandidateAABBOpaqueEXT(rq);"
        in generated_code
    )
    assert (
        "mat4x3 objectToWorld = "
        "rayQueryGetIntersectionObjectToWorldEXT(rq, true);" in generated_code
    )
    assert (
        "mat4x3 worldToObject = "
        "rayQueryGetIntersectionWorldToObjectEXT(rq, false);" in generated_code
    )
    assert "vec3 trianglePositions[3];" in generated_code
    assert (
        "rayQueryGetIntersectionTriangleVertexPositionsEXT("
        "rq, false, trianglePositions);" in generated_code
    )
    assert (
        "rayQueryGetIntersectionTriangleVertexPositionsEXT("
        "rq, true, trianglePositions);" in generated_code
    )
    assert "rayQueryGenerateIntersectionEXT(rq, 1.0);" in generated_code
    assert "rayQueryConfirmIntersectionEXT(rq);" in generated_code
    assert "rayQueryTerminateEXT(rq);" in generated_code
    assert ".WorldRayOrigin(" not in generated_code
    assert ".CandidateTriangleVertexPositions(" not in generated_code
    assert ".GenerateIntersection(" not in generated_code
    assert ".ConfirmIntersection(" not in generated_code


def test_glsl_ray_query_member_calls_reject_invalid_arities():
    cases = [
        (
            "rq.Proceed(1u);",
            r"GLSL ray query Proceed expects 0 arguments, got 1",
        ),
        (
            "rq.Initialize(topLevelAS, gl_RayFlagsNoneEXT);",
            r"GLSL ray query Initialize expects 4 or 7 arguments, got 2",
        ),
        (
            "vec3 position = rq.CandidateTriangleVertexPositions();",
            r"GLSL ray query CandidateTriangleVertexPositions requires exactly "
            r"one output-array argument, got 0",
        ),
        (
            "rq.CommittedTriangleVertexPositions("
            "trianglePositions, trianglePositions);",
            r"GLSL ray query CommittedTriangleVertexPositions requires exactly "
            r"one output-array argument, got 2",
        ),
    ]

    for statement, expected_error in cases:
        shader = f"""
        shader RayQueryInvalidArity {{
            accelerationStructureEXT topLevelAS @binding(0);

            compute {{
                void main() {{
                    RayQuery<RAY_FLAG_NONE> rq;
                    vec3 trianglePositions[3];
                    {statement}
                }}
            }}
        }}
        """

        with pytest.raises(ValueError, match=expected_error):
            GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_glsl_ray_query_member_calls_require_ray_query_receiver():
    invalid_receiver_shader = """
    shader RayQueryInvalidReceiver {
        compute {
            void main() {
                float notQuery = 0.0;
                bool stillActive = notQuery.Proceed();
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=r"GLSL ray query Proceed requires a RayQuery receiver, got float",
    ):
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(invalid_receiver_shader), "compute"
        )

    unknown_receiver_shader = """
    shader RayQueryUnknownReceiver {
        compute {
            void main() {
                bool stillActive = externalQuery.Proceed();
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(unknown_receiver_shader), "compute"
    )

    assert "#extension GL_EXT_ray_query : require" not in generated_code
    assert "rayQueryProceedEXT" not in generated_code
    assert "bool stillActive = externalQuery.Proceed();" in generated_code


def test_glsl_shader_record_buffer_rejects_binding_layout():
    shader = """
    shader InvalidShaderRecordBinding {
        struct ShaderRecordData {
            uint materialIndex;
        };

        ShaderRecordData shaderRecord
            @glsl_buffer_block(shaderRecordEXT)
            @binding(3);

        ray_generation {
            void main() { }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="shaderRecordEXT buffer blocks cannot declare binding layout qualifiers",
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_glsl_ray_generation_shader_emits_ray_tracing_extension():
    shader = """
    shader RayGenExtension {
        struct RayPayload {
            vec4 color;
        };

        accelerationStructureEXT topLevelAS @binding(0);

        ray_generation {
            layout(location = 0) @rayPayloadEXT RayPayload payload;

            void main() {
                TraceRay(
                    topLevelAS,
                    gl_RayFlagsNoneEXT,
                    0xff,
                    0,
                    1,
                    0,
                    vec3(0.0),
                    0.001,
                    vec3(0.0, 0.0, 1.0),
                    1000.0,
                    0
                );
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate_stage(ast, "ray_generation")

    assert generated_code.lstrip().startswith("#version 460 core")
    assert "#extension GL_EXT_ray_tracing : require" in generated_code
    assert "#extension GL_EXT_ray_query : require" not in generated_code
    assert "layout(location = 0) rayPayloadEXT RayPayload payload;" in generated_code
    assert (
        "layout(binding = 0) uniform accelerationStructureEXT topLevelAS;"
        in generated_code
    )
    assert "traceRayEXT(" in generated_code
    assert "TraceRay" not in generated_code
    assert "void main()" in generated_code


def test_glsl_ray_query_in_compute_shader_emits_ray_query_extension():
    shader = """
    shader RayQueryComputeExtension {
        accelerationStructureEXT topLevelAS @binding(0);

        compute {
            void main() {
                RayQuery<RAY_FLAG_NONE> rq;
                rayQueryInitializeEXT(
                    rq,
                    topLevelAS,
                    gl_RayFlagsNoneEXT,
                    255u,
                    vec3(0.0),
                    0.001,
                    vec3(0.0, 0.0, 1.0),
                    100.0
                );
                bool active = rq.Proceed();
                if (active) {
                    uint primitiveIndex = rq.CandidatePrimitiveIndex();
                }
                rq.Abort();
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate_stage(ast, "compute")

    assert "#extension GL_EXT_ray_query : require" in generated_code
    assert "#extension GL_EXT_ray_tracing : require" not in generated_code
    assert "rayQueryEXT rq;" in generated_code
    assert "rayQueryInitializeEXT(" in generated_code
    assert "bool active_ = rayQueryProceedEXT(rq);" in generated_code
    assert "if (active_)" in generated_code
    assert (
        "uint primitiveIndex = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);"
        in generated_code
    )
    assert "rayQueryTerminateEXT(rq);" in generated_code
    assert "active" not in generated_code.replace("active_", "")


def test_glsl_mesh_shader_emits_extension_and_layout():
    shader = """
    shader MeshShaderLayout {
        mesh {
            layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
            layout(triangles, max_vertices = 64, max_primitives = 126) out;
            out vec4 vertexColor[64];
            void main() {
                SetMeshOutputCounts(64, 126);
                vertexColor[gl_LocalInvocationIndex] = vec4(1.0);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate_stage(ast, "mesh")

    assert "#extension GL_EXT_mesh_shader : require" in generated_code
    assert (
        "layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;"
        in generated_code
    )
    assert (
        "layout(triangles, max_vertices = 64, max_primitives = 126) out;"
        in generated_code
    )
    assert "out vec4 vertexColor[64];" in generated_code
    assert "SetMeshOutputsEXT(64, 126);" in generated_code
    assert "SetMeshOutputCounts" not in generated_code
    assert "vertexColor[gl_LocalInvocationIndex] = vec4(1.0);" in generated_code
    assert "void main()" in generated_code


def test_glsl_task_shader_with_work_group_id_and_dispatch_mesh():
    shader = """
    shader TaskShaderDispatch {
        struct TaskPayload {
            uint meshletIndices[32];
            uint meshletCount;
        };

        task {
            layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
            @taskPayloadSharedEXT TaskPayload payload;
            void main() {
                uint gid = gl_WorkGroupID.x;
                payload.meshletCount = gid;
                payload.meshletIndices[gl_LocalInvocationIndex] = gl_LocalInvocationIndex;
                DispatchMesh(payload.meshletCount, 1, 1);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate_stage(ast, "task")

    assert "#extension GL_EXT_mesh_shader : require" in generated_code
    assert (
        "layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;"
        in generated_code
    )
    assert "taskPayloadSharedEXT TaskPayload payload;" in generated_code
    assert "uint gid = gl_WorkGroupID.x;" in generated_code
    assert "payload.meshletCount = gid;" in generated_code
    assert (
        "payload.meshletIndices[gl_LocalInvocationIndex] = gl_LocalInvocationIndex;"
        in generated_code
    )
    assert "EmitMeshTasksEXT(payload.meshletCount, 1, 1);" in generated_code
    assert "DispatchMesh" not in generated_code
    assert "void main()" in generated_code


def test_glsl_ray_query_reserved_active_variable_renamed():
    shader = """
    shader RayQueryActiveReserved {
        accelerationStructureEXT topLevelAS @binding(0);

        compute {
            void main() {
                RayQuery<RAY_FLAG_NONE> rq;
                rayQueryInitializeEXT(
                    rq,
                    topLevelAS,
                    gl_RayFlagsNoneEXT,
                    255u,
                    vec3(0.0),
                    0.001,
                    vec3(0.0, 0.0, 1.0),
                    100.0
                );
                bool active = rq.Proceed();
                float notReserved = 1.0;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate_stage(ast, "compute")

    assert "bool active_ = rayQueryProceedEXT(rq);" in generated_code
    assert "float notReserved = 1.0;" in generated_code
    lines = generated_code.splitlines()
    active_uses = [
        line.strip()
        for line in lines
        if "active" in line and "active_" not in line and "extension" not in line
    ]
    assert (
        active_uses == []
    ), f"Found bare 'active' (without underscore) in output: {active_uses}"


def test_glsl_stage_local_helpers_emit_before_entrypoint():
    shader = """
    shader StageLocalHelperOrder {
        fragment {
            vec4 shade(vec2 uv) {
                return vec4(uv, 0.0, 1.0);
            }

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return shade(uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert generated_code.index("vec4 shade(vec2 uv)") < generated_code.index(
        "void main()"
    )
    assert "fragColor = shade(uv);" in generated_code


def test_glsl_stage_local_helpers_order_by_dependencies():
    shader = """
    shader StageLocalHelperDependencyOrder {
        fragment {
            vec4 first(vec2 uv) {
                return second(uv);
            }

            vec4 second(vec2 uv) {
                return vec4(uv, 0.0, 1.0);
            }

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return first(uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    second_index = generated_code.index("vec4 second(vec2 uv)")
    first_index = generated_code.index("vec4 first(vec2 uv)")
    main_index = generated_code.index("void main()")

    assert second_index < first_index < main_index
    assert "return second(uv);" in generated_code
    assert "fragColor = first(uv);" in generated_code


def test_generate_compute_stage_suppresses_graphics_io_declarations():
    shader = """
    shader MixedComputeStageIOSmoke {
        struct VSInput {
            vec3 position @ POSITION;
            vec2 uv @ TEXCOORD0;
        };

        struct VSOutput {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        struct WorkItem {
            int id;
            vec4 value;
        };

        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.position = vec4(input.position, 1.0);
                output.uv = input.uv;
                return output;
            }
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.uv, 0.0, 1.0);
            }
        }

        compute {
            void main() {
                WorkItem item;
                item.id = 1;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    compute_code = GLSLCodeGen().generate_stage(ast, "compute")

    assert "// Compute Shader" in compute_code
    assert "struct WorkItem" in compute_code
    assert "struct VSOutput" in compute_code
    assert "out vec4 position;" not in compute_code
    assert "out vec2 uv;" not in compute_code
    assert "layout(location = 0) in vec3 position;" not in compute_code
    assert "layout(location = 5) in vec2 out_uv;" not in compute_code
    assert "fragColor" not in compute_code
    assert "// Vertex Shader" not in compute_code
    assert "// Fragment Shader" not in compute_code


def test_compute_stage_emits_default_local_size_layout():
    shader = """
    shader ComputeLayoutSmoke {
        compute {
            void main() {
                int value = 1;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    compute_code = GLSLCodeGen().generate_stage(ast, "compute")

    layout = "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;"
    assert layout in compute_code
    assert compute_code.index(layout) < compute_code.index("void main()")


def test_compute_stage_uses_execution_config_local_size():
    shader = """
    shader ComputeConfiguredLayoutSmoke {
        compute {
            void main() {
                int value = 1;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    ast.stages[ShaderStage.COMPUTE].execution_config = {"local_size": (8, 4, 2)}

    compute_code = GLSLCodeGen().generate_stage(ast, "compute")

    assert (
        "layout(local_size_x = 8, local_size_y = 4, local_size_z = 2) in;"
        in compute_code
    )


def test_combined_same_stage_compute_output_keeps_single_native_layout_and_main():
    shader = """
    shader MultiCompute {
        compute {
            layout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in;
            void main() { }
        }

        compute spawn {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() { }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert generated_code.count("layout(local_size_x") == 1
    assert generated_code.count("void main()") == 1
    assert "void compute_main()" in generated_code


@pytest.mark.parametrize("attribute", ["numthreads", "local_size", "workgroup_size"])
def test_compute_stage_uses_function_execution_config_attribute(attribute):
    shader = f"""
    shader ComputeFunctionAttributeLayoutSmoke {{
        compute {{
            void main() @{attribute}(8, 4, 2) {{
                int value = 1;
            }}
        }}
    }}
    """

    compute_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "layout(local_size_x = 8, local_size_y = 4, local_size_z = 2) in;"
        in compute_code
    )
    assert f"@{attribute}" not in compute_code
    assert "void main() @" not in compute_code


@pytest.mark.parametrize(
    ("stage", "body", "message"),
    [
        (
            "compute",
            "void main() @numthreads(8, 1, 1) @workgroup_size(4, 1, 1) { }",
            r"Conflicting numthreads metadata on compute stage entry function "
            r"'main': \(8, 1, 1\) vs \(4, 1, 1\)",
        ),
        (
            "task",
            "void main() @numthreads(8, 1, 1) @workgroup_size(4, 1, 1) { }",
            r"Conflicting numthreads metadata on task stage entry function "
            r"'main': \(8, 1, 1\) vs \(4, 1, 1\)",
        ),
        (
            "mesh",
            """
            void main()
                @outputtopology(triangle)
                @max_vertices(3)
                @max_primitives(1)
                @numthreads(8, 1, 1)
                @workgroup_size(4, 1, 1)
            { }
            """,
            r"Conflicting numthreads metadata on mesh stage entry function "
            r"'main': \(8, 1, 1\) vs \(4, 1, 1\)",
        ),
    ],
)
def test_threadgroup_stages_reject_conflicting_execution_config_metadata(
    stage, body, message
):
    shader = f"""
    shader BadThreadgroupMetadata {{
        {stage} {{
            {body}
        }}
    }}
    """

    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), stage)


def test_while_switch_and_void_return_emit_c_style_syntax():
    shader = """
    shader StatementLeakSmoke {
        void helper() {
            int i = 0;
            while (i < 4) {
                switch (i) {
                    case 0:
                        i = i + 1;
                        continue;
                    default:
                        break;
                }
            }
            return;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "while ((i < 4))" in generated_code
    assert "switch (i)" in generated_code
    assert "case 0:" in generated_code
    assert "default:" in generated_code
    assert "i = (i + 1);" in generated_code
    assert "continue;" in generated_code
    assert "break;" in generated_code
    assert "return;" in generated_code
    assert "WhileNode(" not in generated_code
    assert "SwitchNode(" not in generated_code
    assert "return ;" not in generated_code
    assert "return None;" not in generated_code


def test_switch_fallthrough_and_nested_switch_emit_c_style_syntax():
    shader = """
    shader SwitchEdgeSmoke {
        int helper(int mode, int submode) {
            int value = 0;
            switch (mode) {
                case 0:
                case 1:
                    value = value + 1;
                    break;
                case 2:
                    switch (submode) {
                        case 0:
                            value = value + 2;
                            break;
                        default:
                            value = value + 3;
                            break;
                    }
                    break;
                default:
                    value = value + 4;
                    break;
            }
            return value;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" in generated_code
    assert "case 0:\n        case 1: {" in generated_code
    assert "case 2: {\n            switch (submode)" in generated_code
    assert generated_code.count("default:") == 2
    assert "value = (value + 1);" in generated_code
    assert "value = (value + 2);" in generated_code
    assert "value = (value + 3);" in generated_code
    assert "value = (value + 4);" in generated_code
    assert "return value;" in generated_code
    assert "SwitchNode(" not in generated_code
    assert "CaseNode(" not in generated_code


def test_switch_and_match_case_blocks_scope_local_declarations():
    shader = """
    shader CaseScopeSmoke {
        int switchHelper(int mode) {
            int value = 0;
            switch (mode) {
                case 0:
                case 1:
                    int scoped = value + 1;
                    value = scoped;
                    break;
                default:
                    int scoped = value + 2;
                    value = scoped;
                    break;
            }
            return value;
        }

        int matchHelper(int mode) {
            int value = 0;
            match mode {
                0 => { int scoped = value + 1; value = scoped; }
                1 => { int scoped = value + 2; value = scoped; }
                _ => { int scoped = value + 3; value = scoped; }
            }
            return value;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "case 0:\n        case 1: {" in generated_code
    assert generated_code.count("case 0: {") == 1
    assert generated_code.count("int scoped") == 5
    assert "default: {" in generated_code
    assert "MatchNode(" not in generated_code
    assert "SwitchNode(" not in generated_code


def test_match_literal_and_wildcard_arms_lower_to_switch():
    shader = """
    shader MatchLeakSmoke {
        int helper(int mode) {
            int value = 0;
            match mode {
                0 => { value = 1; }
                1 => { value = 2; }
                _ => { value = 3; }
            }
            return value;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" in generated_code
    assert "case 0:" in generated_code
    assert "case 1:" in generated_code
    assert "default:" in generated_code
    assert "value = 1;" in generated_code
    assert "value = 2;" in generated_code
    assert "value = 3;" in generated_code
    assert generated_code.count("break;") == 3
    assert "return value;" in generated_code
    assert "MatchNode(" not in generated_code
    assert "MatchArmNode(" not in generated_code


def test_match_return_arms_do_not_emit_extra_breaks():
    shader = """
    shader MatchReturnArms {
        int helper(int mode) {
            match mode {
                0 => { return 1; }
                _ => { return 2; }
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" in generated_code
    assert "case 0:" in generated_code
    assert "default:" in generated_code
    assert "return 1;" in generated_code
    assert "return 2;" in generated_code
    assert "break;" not in generated_code
    assert "MatchNode(" not in generated_code


def test_match_guarded_literal_and_wildcard_arms_lower_to_if_chain():
    shader = """
    shader MatchGuardPattern {
        int helper(int mode) {
            int value = 0;
            match mode {
                0 if mode > 0 => { value = 1; }
                0 => { value = 2; }
                _ if mode < 10 => { value = 3; }
                _ => { value = 4; }
            }
            return value;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" not in generated_code
    assert "if (((mode == 0) &&" in generated_code
    assert "mode > 0" in generated_code
    assert "else if ((mode == 0))" in generated_code
    assert "else if" in generated_code
    assert "mode < 10" in generated_code
    assert "else {" in generated_code
    assert "value = 1;" in generated_code
    assert "value = 2;" in generated_code
    assert "value = 3;" in generated_code
    assert "value = 4;" in generated_code
    assert "MatchNode(" not in generated_code


def test_match_identifier_binding_arm_lowers_to_scoped_else_body():
    shader = """
    shader MatchBindingPattern {
        int helper(int mode) {
            int value = 0;
            match mode {
                0 => { value = 1; }
                other => { value = other; }
            }
            return value;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" not in generated_code
    assert "if ((mode == 0))" in generated_code
    assert "else {" in generated_code
    assert "int other = mode;" in generated_code
    assert "value = other;" in generated_code
    assert "MatchNode(" not in generated_code


def test_match_guarded_identifier_binding_falls_through_to_later_arm():
    shader = """
    shader MatchGuardedBindingPattern {
        int helper(int mode) {
            int value = 0;
            match mode {
                0 => { value = 1; }
                candidate if candidate > 2 => { value = candidate; }
                _ => { value = 7; }
            }
            return value;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" not in generated_code
    assert "if ((mode == 0))" in generated_code
    assert "else {" in generated_code
    assert "int candidate = mode;" in generated_code
    assert "candidate > 2" in generated_code
    assert "value = candidate;" in generated_code
    assert "value = 7;" in generated_code
    assert "MatchNode(" not in generated_code


def test_match_plain_struct_pattern_binds_fields():
    shader = """
    shader MatchStructPattern {
        struct Pair {
            int left;
            int right;
        };

        int helper(Pair pair) {
            int value = 0;
            match pair {
                Pair { left, right } => { value = left + right; }
            }
            return value;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "int left = pair.left;" in generated_code
    assert "int right = pair.right;" in generated_code
    assert "value = (left + right);" in generated_code
    assert "MatchNode(" not in generated_code


def test_match_enum_path_pattern_lowers_to_integer_constants():
    shader = """
    shader MatchEnumPathPattern {
        enum Mode {
            Add,
            Multiply = 4,
            Divide
        }

        int helper(Mode mode) {
            int value = Mode::Divide;
            match mode {
                Mode::Add => { value = 1; }
                Mode::Multiply => { value = 2; }
                _ => { value = 3; }
            }
            return value;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int Mode_Add = 0;" in generated_code
    assert "const int Mode_Multiply = 4;" in generated_code
    assert "const int Mode_Divide = 5;" in generated_code
    assert "int helper(int mode)" in generated_code
    assert "int value = Mode_Divide;" in generated_code
    assert "switch (mode)" not in generated_code
    assert "if ((mode == Mode_Add))" in generated_code
    assert "else if ((mode == Mode_Multiply))" in generated_code
    assert "value = 3;" in generated_code
    assert "MatchNode(" not in generated_code


def test_match_struct_enum_pattern_binds_named_payload_fields():
    shader = """
    shader MatchStructEnumPattern {
        enum LightingModel {
            Phong {
                ambient: vec3,
                diffuse: vec3,
                shininess: float
            },
            Toon {
                base_color: vec3,
                levels: int
            }
        }

        vec3 shade(LightingModel model) {
            vec3 result = vec3(0.0);
            match model {
                LightingModel::Phong { ambient, diffuse, shininess } => {
                    result = ambient + diffuse * shininess;
                },
                LightingModel::Toon { base_color, .. } => {
                    result = base_color;
                }
            }
            return result;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int LightingModel_Phong = 0;" in generated_code
    assert "const int LightingModel_Toon = 1;" in generated_code
    assert "struct LightingModel {" in generated_code
    assert "int variant;" in generated_code
    assert "vec3 ambient;" in generated_code
    assert "vec3 base_color;" in generated_code
    assert "vec3 shade(LightingModel model)" in generated_code
    assert "if ((model.variant == LightingModel_Phong))" in generated_code
    assert "else if ((model.variant == LightingModel_Toon))" in generated_code
    assert "vec3 ambient = model.ambient;" in generated_code
    assert "float shininess = model.shininess;" in generated_code
    assert "vec3 base_color = model.base_color;" in generated_code
    assert "LightingModel::" not in generated_code
    assert "MatchNode(" not in generated_code


def test_match_tuple_enum_pattern_binds_payload_fields():
    shader = """
    shader MatchTupleEnumPattern {
        enum MaybeInt {
            Value(int),
            Pair(int, float),
            Missing
        }

        int read(MaybeInt item) {
            match item {
                MaybeInt::Value(value) => { return value; },
                MaybeInt::Pair(left, scale) => { return left + int(scale); },
                MaybeInt::Missing => { return 0; }
            }
        }

        MaybeInt make(int value) {
            return MaybeInt::Value(value);
        }

        MaybeInt none() {
            return MaybeInt::Missing;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int MaybeInt_Value = 0;" in generated_code
    assert "const int MaybeInt_Pair = 1;" in generated_code
    assert "const int MaybeInt_Missing = 2;" in generated_code
    assert "struct MaybeInt {" in generated_code
    assert "int Value_0;" in generated_code
    assert "int Pair_0;" in generated_code
    assert "float Pair_1;" in generated_code
    assert "MaybeInt MaybeInt_Value_make(int payload0)" in generated_code
    assert "MaybeInt MaybeInt_Missing_make()" in generated_code
    assert "if ((item.variant == MaybeInt_Value))" in generated_code
    assert "else if ((item.variant == MaybeInt_Pair))" in generated_code
    assert "int value = item.Value_0;" in generated_code
    assert "int left = item.Pair_0;" in generated_code
    assert "float scale = item.Pair_1;" in generated_code
    assert "return MaybeInt_Value_make(value);" in generated_code
    assert "return MaybeInt_Missing_make();" in generated_code
    assert "MaybeInt::" not in generated_code
    assert "MatchNode(" not in generated_code


def test_string_payload_enum_maps_to_shader_token_type():
    shader = """
    shader StringPayloadEnum {
        enum ShaderError {
            TextureError(str),
            BufferError(str),
            InvalidState
        }

        ShaderError texture_error(int code) {
            return ShaderError::TextureError(code);
        }

        ShaderError invalid_state() {
            return ShaderError::InvalidState;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct ShaderError {" in generated_code
    assert "int TextureError_0;" in generated_code
    assert "int BufferError_0;" in generated_code
    assert "ShaderError ShaderError_TextureError_make(int payload0)" in generated_code
    assert "ShaderError ShaderError_InvalidState_make()" in generated_code
    assert "return ShaderError_TextureError_make(code);" in generated_code
    assert "return ShaderError_InvalidState_make();" in generated_code
    assert re.search(r"\bstr\b", generated_code) is None


def test_tuple_enum_constructor_wrong_arity_raises():
    shader = """
    shader TupleEnumConstructorArity {
        enum MaybeInt {
            Value(int),
            Missing
        }

        MaybeInt make() {
            return MaybeInt::Value();
        }
    }
    """

    with pytest.raises(
        ValueError, match="Enum constructor MaybeInt::Value expects 1 arguments, got 0"
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_generic_enum_struct_concrete_match_and_constructors():
    shader = """
    shader GenericEnumConcrete {
        generic<T> struct Option {
            enum OptionType { Some(T), None }
            OptionType variant;
        }

        generic<T, E> struct Result {
            enum ResultType { Ok(T), Err(E) }
            ResultType variant;
        }

        enum MathError {
            DivisionByZero
        }

        Option<int> some(int value) {
            return Option::Some(value);
        }

        Option<int> none() {
            return Option::None;
        }

        int read_option(Option<int> item) {
            match item {
                Option::Some(value) => { return value; },
                Option::None => { return 0; }
            }
        }

        Result<int, MathError> make_result(bool ok) {
            match ok {
                true => { return Result::Ok(1); },
                false => { return Result::Err(MathError::DivisionByZero); }
            }
        }

        int read_result(Result<int, MathError> item) {
            match item {
                Result::Ok(value) => { return value; },
                Result::Err(_) => { return 0; }
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int Option_Some = 0;" in generated_code
    assert "const int Result_Ok = 0;" in generated_code
    assert "struct Option_int {" in generated_code
    assert "struct Result_int_MathError {" in generated_code
    assert "struct Option {" not in generated_code
    assert "struct Result {" not in generated_code
    assert "Option_int Option_int_Some_make(int payload0)" in generated_code
    assert "Option_int Option_int_None_make()" in generated_code
    assert (
        "Result_int_MathError Result_int_MathError_Ok_make(int payload0)"
        in generated_code
    )
    assert (
        "Result_int_MathError Result_int_MathError_Err_make(int payload0)"
        in generated_code
    )
    assert "Option_int some(int value)" in generated_code
    assert "Option_int none()" in generated_code
    assert "Result_int_MathError make_result(bool ok)" in generated_code
    assert "int read_result(Result_int_MathError item)" in generated_code
    assert "return Option_int_Some_make(value);" in generated_code
    assert "return Option_int_None_make();" in generated_code
    assert "return Result_int_MathError_Ok_make(1);" in generated_code
    assert (
        "return Result_int_MathError_Err_make(MathError_DivisionByZero);"
        in generated_code
    )
    assert "if ((item.variant == Option_Some))" in generated_code
    assert "else if ((item.variant == Option_None))" in generated_code
    assert "if ((item.variant == Result_Ok))" in generated_code
    assert "else if ((item.variant == Result_Err))" in generated_code
    assert "int value = item.Some_0;" in generated_code
    assert "int value = item.Ok_0;" in generated_code
    assert "Option<" not in generated_code
    assert "Result<" not in generated_code
    assert "Option::" not in generated_code
    assert "Result::" not in generated_code


def test_generic_enum_function_call_match_infers_result_type_once():
    shader = """
    shader FunctionCallResultMatch {
        generic<T, E> struct Result {
            enum ResultType { Ok(T), Err(E) }
            ResultType variant;
        }

        enum MathError {
            DivisionByZero
        }

        Result<int, MathError> make_result(bool ok) {
            match ok {
                true => { return Result::Ok(7); },
                false => { return Result::Err(MathError::DivisionByZero); }
            }
        }

        int read(bool ok) {
            match make_result(ok) {
                Result::Ok(value) => { return value; },
                Result::Err(_) => { return -1; }
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Result_int_MathError make_result(bool ok)" in generated_code
    assert (
        "Result_int_MathError cgl_match_subject_0 = make_result(ok);" in generated_code
    )
    assert generated_code.count("make_result(ok)") == 1
    assert "if ((cgl_match_subject_0.variant == Result_Ok))" in generated_code
    assert "else if ((cgl_match_subject_0.variant == Result_Err))" in generated_code
    assert "int value = cgl_match_subject_0.Ok_0;" in generated_code
    assert "__crossgl_match_subject" not in generated_code
    assert "Result::" not in generated_code
    assert "MatchNode(" not in generated_code


def test_generic_enum_local_function_call_type_infers_match_subject():
    shader = """
    shader LocalResultMatch {
        generic<T, E> struct Result {
            enum ResultType { Ok(T), Err(E) }
            ResultType variant;
        }

        enum MathError {
            DivisionByZero
        }

        Result<int, MathError> make_result(bool ok) {
            match ok {
                true => { return Result::Ok(7); },
                false => { return Result::Err(MathError::DivisionByZero); }
            }
        }

        int read(bool ok) {
            let item = make_result(ok);
            match item {
                Result::Ok(value) => { return value; },
                Result::Err(_) => { return -1; }
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Result_int_MathError item = make_result(ok);" in generated_code
    assert "float item = make_result(ok);" not in generated_code
    assert "if ((item.variant == Result_Ok))" in generated_code
    assert "else if ((item.variant == Result_Err))" in generated_code
    assert "int value = item.Ok_0;" in generated_code
    assert "Result<" not in generated_code
    assert "ConstructorNode(" not in generated_code
    assert "MatchNode(" not in generated_code


def test_generic_enum_match_expression_initializes_vector_local():
    shader = """
    shader MatchExpressionValue {
        generic<T, E> struct Result {
            enum ResultType { Ok(T), Err(E) }
            ResultType variant;
        }

        enum MathError {
            DivisionByZero
        }

        Result<vec3, MathError> make_result(bool ok) {
            match ok {
                true => { return Result::Ok(vec3(1.0, 2.0, 3.0)); },
                false => { return Result::Err(MathError::DivisionByZero); }
            }
        }

        vec3 read(bool ok, vec3 fallback) {
            let value = match make_result(ok) {
                Result::Ok(actual) => actual,
                Result::Err(_) => fallback
            };
            return value;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Result_vec3_MathError make_result(bool ok)" in generated_code
    assert "vec3 read(bool ok, vec3 fallback)" in generated_code
    assert "vec3 value;" in generated_code
    assert (
        "Result_vec3_MathError cgl_match_subject_0 = make_result(ok);" in generated_code
    )
    assert "if ((cgl_match_subject_0.variant == Result_Ok))" in generated_code
    assert "else if ((cgl_match_subject_0.variant == Result_Err))" in generated_code
    assert "vec3 actual = cgl_match_subject_0.Ok_0;" in generated_code
    assert "__crossgl_match_subject" not in generated_code
    assert "value = actual;" in generated_code
    assert "value = fallback;" in generated_code
    assert "float value = MatchNode" not in generated_code
    assert "MatchNode(" not in generated_code


def test_enum_struct_variant_constructor_expression():
    shader = """
    shader EnumStructVariantConstructor {
        enum RenderOutput {
            Clear { color: vec4, depth: float },
            StateSet
        }

        RenderOutput clear(vec4 color, float depth) {
            return RenderOutput::Clear { color, depth };
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "RenderOutput RenderOutput_Clear_make(vec4 payload0, float payload1)"
        in generated_code
    )
    assert "RenderOutput clear(vec4 color, float depth)" in generated_code
    assert "return RenderOutput_Clear_make(color, depth);" in generated_code
    assert "ConstructorNode(" not in generated_code
    assert "RenderOutput::" not in generated_code


def test_tail_expression_returns_struct_constructor_and_vector():
    shader = """
    shader TailExpressionReturn {
        struct Pair {
            value: vec3,
            weight: float
        }

        Pair make_pair(vec3 value, float weight) {
            Pair { value, weight }
        }

        vec4 make_color(vec3 color) {
            vec4(color, 1.0)
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Pair make_pair(vec3 value, float weight)" in generated_code
    assert "return Pair(value, weight);" in generated_code
    assert "vec4 make_color(vec3 color)" in generated_code
    assert "return vec4(color, 1.0);" in generated_code
    assert "ConstructorNode(" not in generated_code


def test_stage_tail_struct_constructor_returns_stage_output():
    shader = """
    shader StageTailReturn {
        vertex {
            struct VertexInput {
                position: vec3,
                color: vec4
            }
            struct VertexOutput {
                position: vec4,
                color: vec4
            }
            VertexOutput main(VertexInput input) {
                VertexOutput {
                    position: vec4(input.position, 1.0),
                    color: input.color
                }
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct VertexOutput {" not in generated_code
    assert "void main()" in generated_code
    assert "out_position = vec4(position, 1.0);" in generated_code
    assert "out_color = color;" in generated_code
    assert "return;" in generated_code
    assert "ConstructorNode(" not in generated_code


def test_flattened_vertex_output_struct_is_kept_when_helper_uses_type():
    shader = """
    shader StageOutputHelperUse {
        struct VertexOutput {
            position: vec4,
            color: vec4
        }

        float readColor(VertexOutput value) {
            return value.color.x;
        }

        vertex {
            VertexOutput main() {
                return VertexOutput(vec4(1.0), vec4(0.25));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct VertexOutput {" in generated_code
    assert "float readColor(VertexOutput value)" in generated_code
    assert "position = vec4(1.0);" in generated_code
    assert "color = vec4(0.25);" in generated_code
    assert "return VertexOutput" not in generated_code


def test_glsl_stage_io_multidimensional_arrays_use_c_style_declarators():
    shader = """
    shader StageIOMultidimensionalArrays {
        vertex {
            struct VertexInput {
                vec3 positions[2][3];
                ivec2 ids[2][2];
            }

            struct VertexOutput {
                vec4 colors[2][3];
                vec4 position @ gl_Position;
            }

            VertexOutput main(VertexInput input) {
                VertexOutput output;
                output.colors[1][2] = vec4(input.positions[0][1], 1.0);
                output.position = output.colors[1][2];
                return output;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "in vec3 positions[2][3];" in generated_code
    assert "in ivec2 ids[2][2];" in generated_code
    assert "out vec4 colors[2][3];" in generated_code
    assert "vec3[2][3] positions" not in generated_code
    assert "ivec2[2][2] ids" not in generated_code
    assert "vec4[2][3] colors" not in generated_code


def test_glsl_interface_block_instance_preserves_all_array_dimensions():
    shader = """
    shader GLSLInterfaceBlockInstanceArray {
        @glsl_interface_block(in) @glsl_interface_instance(vertexIn) @glsl_interface_array(2, 3)
        struct VertexIn {
            ivec2 ids[2][2];
            vec3 positions[2][3];
        };

        @glsl_interface_block(out) @glsl_interface_instance(fragmentOut)
        struct FragmentOut {
            vec4 colors[2][3];
        };

        vertex {
            void main() {
                gl_Position = vec4(vertexIn[1][2].positions[0][1], 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "} vertexIn[2][3];" in generated_code
    assert "ivec2 ids[2][2];" in generated_code
    assert "vec3 positions[2][3];" in generated_code
    assert "vec4 colors[2][3];" in generated_code


def test_glsl_stage_io_multidimensional_array_locations_are_reserved():
    shader = """
    shader StageIOMultidimensionalArrayLocationConflict {
        vertex {
            struct VertexInput {
                vec4 first[2][3] @location(0);
                vec4 second @location(5);
            }

            void main(VertexInput input) {
                gl_Position = input.first[0][0] + input.second;
            }
        }
    }
    """

    with pytest.raises(ValueError, match="locations 0-5"):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_glsl_stage_io_multidimensional_arrays_keep_interpolation_qualifiers():
    shader = """
    shader StageIOInterpolationArrays {
        vertex {
            struct VertexInput {
                vec3 position @location(0);
            }

            struct VertexOutput {
                ivec2 ids[2][2] @location(1);
                vec4 samples[2] @location(5) @sample;
                vec3 centers[2] @location(7) @centroid;
                vec2 noPersp[2] @location(9) @noperspective;
                vec4 position @gl_Position;
            }

            VertexOutput main(VertexInput input) {
                VertexOutput output;
                output.ids[0][0] = ivec2(1, 2);
                output.samples[0] = vec4(input.position, 1.0);
                output.centers[0] = input.position;
                output.noPersp[0] = input.position.xy;
                output.position = vec4(input.position, 1.0);
                return output;
            }
        }

        fragment {
            vec4 main(
                ivec2 ids[2][2] @location(1),
                vec4 samples[2] @location(5) @sample,
                vec3 centers[2] @location(7) @centroid,
                vec2 noPersp[2] @location(9) @noperspective
            ) @gl_FragColor {
                return samples[0]
                    + vec4(centers[0], 1.0)
                    + vec4(noPersp[0], 0.0, 1.0)
                    + vec4(ids[0][0], 0.0, 1.0);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    vertex_code = GLSLCodeGen().generate_stage(ast, "vertex")
    fragment_code = GLSLCodeGen().generate_stage(ast, "fragment")

    assert "layout(location = 1) flat out ivec2 ids[2][2];" in vertex_code
    assert "layout(location = 5) sample out vec4 samples[2];" in vertex_code
    assert "layout(location = 7) centroid out vec3 centers[2];" in vertex_code
    assert "layout(location = 9) noperspective out vec2 noPersp[2];" in vertex_code
    assert "layout(location = 1) flat in ivec2 ids[2][2];" in fragment_code
    assert "layout(location = 5) sample in vec4 samples[2];" in fragment_code
    assert "layout(location = 7) centroid in vec3 centers[2];" in fragment_code
    assert "layout(location = 9) noperspective in vec2 noPersp[2];" in fragment_code
    assert "ivec2[2][2] ids" not in vertex_code
    assert "vec4[2] samples" not in vertex_code
    assert "ivec2[2][2] ids" not in fragment_code
    assert "vec4[2] samples" not in fragment_code


def test_trait_self_return_does_not_emit_generic_enum_specialization():
    shader = """
    shader TraitSelfOption {
        generic<T> struct Option {
            enum OptionType { Some(T), None }
            OptionType variant;
        }

        trait VectorOps {
            fn normalize(self) -> Option<Self>;
        }

        int passthrough(int value) {
            return value;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct Option_Self" not in generated_code
    assert "Option_Self" not in generated_code
    assert "Self Some_0" not in generated_code
    assert "int passthrough(int value)" in generated_code


def test_generic_function_call_emits_concrete_specialization():
    shader = """
    shader GenericFunctionSpecialization {
        generic<T> fn add_one(value: T) -> T {
            return value.add(T::one());
        }

        float use_add_one(float value) {
            return add_one(value);
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "float add_one_float(float value)" in generated_code
    assert "return (value + 1.0);" in generated_code
    assert "return add_one_float(value);" in generated_code
    assert "T add_one(T value)" not in generated_code
    assert "return add_one(value);" not in generated_code
    assert ".add(" not in generated_code
    assert "T::one" not in generated_code


def test_generic_function_call_without_inferred_type_raises_diagnostic():
    shader = """
    shader GenericFunctionUninferred {
        generic<T> fn zero() -> T {
            return T::zero();
        }

        float use_zero() {
            return zero();
        }
    }
    """

    with pytest.raises(ValueError) as exc_info:
        GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "cannot infer concrete template arguments for generic function 'zero'"
        in str(exc_info.value)
    )
    assert (
        getattr(exc_info.value, "project_diagnostic_code", None)
        == "project.translate.opengl-template-type-unresolved"
    )


def test_generic_struct_concrete_constructor_and_member_access():
    shader = """
    shader GenericStructConcrete {
        generic<T> struct Box {
            value: T;
        }

        generic<T> struct PairBox {
            first: Box<T>;
            second: T;
        }

        Box<int> make(int value) {
            return Box { value: value };
        }

        int read(Box<int> item) {
            return item.value;
        }

        PairBox<int> make_pair(int value) {
            return PairBox {
                first: Box { value: value },
                second: value
            };
        }

        int read_pair(PairBox<int> item) {
            return item.first.value + item.second;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct Box_int {" in generated_code
    assert "int value;" in generated_code
    assert "struct PairBox_int {" in generated_code
    assert "Box_int first;" in generated_code
    assert "int second;" in generated_code
    assert "struct Box {" not in generated_code
    assert "struct PairBox {" not in generated_code
    assert "Box_int make(int value)" in generated_code
    assert "return Box_int(value);" in generated_code
    assert "int read(Box_int item)" in generated_code
    assert "return item.value;" in generated_code
    assert "PairBox_int make_pair(int value)" in generated_code
    assert "return PairBox_int(Box_int(value), value);" in generated_code
    assert "int read_pair(PairBox_int item)" in generated_code
    assert "item.first.value + item.second" in generated_code
    assert "Box<" not in generated_code
    assert "PairBox<" not in generated_code
    assert "ConstructorNode(" not in generated_code


def test_struct_constructor_and_inferred_generic_constructor_local():
    shader = """
    shader StructConstructors {
        struct Geometry {
            center: vec3;
            normal: vec3;
        }

        generic<T> struct Box {
            value: T;
        }

        Geometry make_geometry(vec3 center, vec3 normal) {
            return Geometry { center: center, normal: normal };
        }

        Box<float> make_box(vec3 value) {
            let normalized = normalize(value);
            let wrapped = Box { value: normalized.x };
            return wrapped;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct Geometry {" in generated_code
    assert "struct Box_float {" in generated_code
    assert "Geometry make_geometry(vec3 center, vec3 normal)" in generated_code
    assert "return Geometry(center, normal);" in generated_code
    assert "Box_float make_box(vec3 value)" in generated_code
    assert "vec3 normalized = normalize(value);" in generated_code
    assert "Box_float wrapped = Box_float(normalized.x);" in generated_code
    assert "return wrapped;" in generated_code
    assert "Option_Self normalized" not in generated_code
    assert "Box<" not in generated_code
    assert "ConstructorNode(" not in generated_code


def test_else_statement():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;

            if (input.texCoord.x > 0.5) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
            } else {
                output.color = vec4(0.0, 0.0, 0.0, 1.0);
            }

            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            if (bloom > 0.5) {
                bloom = 0.5;
            } else {
                bloom = 0.0;
            }

            // Apply bloom to the texture color
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);

            return vec4(colorWithBloom, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_codegen_emits_version_when_absent_in_crossgl():
    shader = """
    shader simple {
        struct VSOut {
            vec4 position @ gl_Position;
        };

        vertex {
            VSOut main() {
                VSOut o;
                o.position = vec4(0.0, 0.0, 0.0, 1.0);
                return o;
            }
        }
    }
    """
    ast = crosstl.translator.parse(shader)
    glsl_code = GLSLCodeGen().generate(ast)
    assert glsl_code.lstrip().startswith("#version 450 core")


def test_geometry_stage_entrypoint():
    code = """
    shader geom {
        geometry {
            void main() { }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "void main()" in generated


def test_glsl_texture_and_atomic_builtins():
    code = """
    shader main {
        compute {
            void main() {
                sampler2D tex;
                image2D img;
                atomic_uint counter;
                vec4 g = textureGatherOffset(tex, vec2(0.5), ivec2(1));
                uint v = imageAtomicAdd(img, ivec2(0, 0), 1);
                uint c = atomicCounterIncrement(counter);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "textureGatherOffset" in generated
    assert "imageAtomicAdd" in generated
    assert "atomicCounterIncrement" in generated


def test_glsl_lowers_compiler_workgroup_atomic_resource_types():
    code = """
    shader OpenGLAtomicAddShader {
        const int GROUP_SIZE = 4;

        compute {
            layout(local_size_x = GROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
            layout(set = 0, binding = 0) buffer atomic<int>* counters;
            layout(set = 0, binding = 1) buffer atomic<uint>* unsignedCounters;
            var<workgroup> tile: atomic<int>[GROUP_SIZE];
            var<workgroup> unsignedTile: atomic<uint>[GROUP_SIZE];

            void main() {
                uint index = gl_LocalInvocationID.x;
                uint unsignedDelta = index;
                atomicAdd(counters[index], 1);
                atomicAdd(unsignedCounters[index], unsignedDelta);
                atomicAdd(tile[index], 1);
                atomicAdd(unsignedTile[index], unsignedDelta);
                return;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    generated = generate_code(ast)

    assert "buffer countersBuffer { int counters[]; };" in generated
    assert "buffer unsignedCountersBuffer { uint unsignedCounters[]; };" in generated
    assert "shared int tile[GROUP_SIZE];" in generated
    assert "shared uint unsignedTile[GROUP_SIZE];" in generated
    assert "atomic<int>" not in generated
    assert "atomic<uint>" not in generated


def test_glsl_wave_and_mesh_intrinsics():
    code = """
    shader main {
        compute {
            void main() {
                uint v;
                uint sum = WaveActiveSum(v);
                SetMeshOutputCounts(64, 32);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "#extension GL_KHR_shader_subgroup_basic : require" in generated
    assert "#extension GL_KHR_shader_subgroup_arithmetic : require" in generated
    assert "uint sum = subgroupAdd(v);" in generated
    assert "WaveActiveSum" not in generated
    assert "SetMeshOutputsEXT" in generated
    assert "SetMeshOutputCounts" not in generated


def test_glsl_wave_intrinsics_lower_to_khr_subgroup_builtins():
    code = """
    shader GLSLWaveIntrinsics {
        compute {
            void main() {
                uint lane = WaveGetLaneIndex();
                uint count = WaveGetLaneCount();
                bool firstLane = WaveIsFirstLane();
                uint sumValue = WaveActiveSum(lane);
                uint productValue = WaveActiveProduct(lane + 1u);
                uint minValue = WaveActiveMin(sumValue);
                uint maxValue = WaveActiveMax(productValue);
                uint andValue = WaveActiveBitAnd(maxValue);
                uint orValue = WaveActiveBitOr(andValue);
                uint xorValue = WaveActiveBitXor(orValue);
                uint prefixSum = WavePrefixSum(xorValue);
                uint prefixProduct = WavePrefixProduct(lane + 1u);
                bool anyLane = WaveActiveAnyTrue(prefixSum > 0u);
                bool allLane = WaveActiveAllTrue(prefixProduct > 0u);
                uvec4 ballot = WaveActiveBallot(anyLane);
                uint broadcast = WaveReadLaneAt(prefixSum, 0u);
                uint firstValue = WaveReadLaneFirst(broadcast);
                uint quadX = QuadReadAcrossX(firstValue);
                uint quadY = QuadReadAcrossY(quadX);
                uint quadDiagonal = QuadReadAcrossDiagonal(quadY);
                uint quadLane = QuadReadLaneAt(quadDiagonal, 0u);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    for extension in [
        "#extension GL_KHR_shader_subgroup_basic : require",
        "#extension GL_KHR_shader_subgroup_vote : require",
        "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "#extension GL_KHR_shader_subgroup_ballot : require",
        "#extension GL_KHR_shader_subgroup_shuffle : require",
        "#extension GL_KHR_shader_subgroup_quad : require",
    ]:
        assert extension in generated

    for expected in [
        "uint lane = gl_SubgroupInvocationID;",
        "uint count = gl_SubgroupSize;",
        "bool firstLane = subgroupElect();",
        "uint sumValue = subgroupAdd(lane);",
        "uint productValue = subgroupMul((lane + 1u));",
        "uint minValue = subgroupMin(sumValue);",
        "uint maxValue = subgroupMax(productValue);",
        "uint andValue = subgroupAnd(maxValue);",
        "uint orValue = subgroupOr(andValue);",
        "uint xorValue = subgroupXor(orValue);",
        "uint prefixSum = subgroupExclusiveAdd(xorValue);",
        "uint prefixProduct = subgroupExclusiveMul((lane + 1u));",
        "bool anyLane = subgroupAny((prefixSum > 0u));",
        "bool allLane = subgroupAll((prefixProduct > 0u));",
        "uvec4 ballot = subgroupBallot(anyLane);",
        "uint broadcast = subgroupShuffle(prefixSum, 0u);",
        "uint firstValue = subgroupBroadcastFirst(broadcast);",
        "uint quadX = subgroupQuadSwapHorizontal(firstValue);",
        "uint quadY = subgroupQuadSwapVertical(quadX);",
        "uint quadDiagonal = subgroupQuadSwapDiagonal(quadY);",
        "uint quadLane = subgroupQuadBroadcast(quadDiagonal, 0u);",
    ]:
        assert expected in generated

    assert "WaveActive" not in generated
    assert "WavePrefix" not in generated
    assert "WaveReadLane" not in generated
    assert "QuadRead" not in generated


def test_glsl_lowers_metal_simd_group_helpers_to_khr_subgroup_builtins():
    code = """
    shader GLSLMetalSimdGroupHelpers {
        compute {
            void main() {
                uint lane = 1u;
                uint sumValue = __metal_simd_sum(lane);
                uint productValue = __metal_simd_product(lane + 1u);
                uint minValue = __metal_simd_min(sumValue);
                uint maxValue = __metal_simd_max(productValue);
                uint xorValue = __metal_simd_xor(maxValue);
                uint prefixExclusive = __metal_simd_prefix_exclusive_sum(xorValue);
                uint prefixInclusive = __metal_simd_prefix_inclusive_sum(prefixExclusive);
                uint inclusiveProduct = __metal_simd_prefix_inclusive_product(lane + 1u);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    assert "#extension GL_KHR_shader_subgroup_basic : require" in generated
    assert "#extension GL_KHR_shader_subgroup_arithmetic : require" in generated
    for expected in [
        "uint sumValue = subgroupAdd(lane);",
        "uint productValue = subgroupMul((lane + 1u));",
        "uint minValue = subgroupMin(sumValue);",
        "uint maxValue = subgroupMax(productValue);",
        "uint xorValue = subgroupXor(maxValue);",
        "uint prefixExclusive = subgroupExclusiveAdd(xorValue);",
        "uint prefixInclusive = subgroupInclusiveAdd(prefixExclusive);",
        "uint inclusiveProduct = subgroupInclusiveMul((lane + 1u));",
    ]:
        assert expected in generated

    assert "__metal_simd" not in generated


def test_glsl_metal_simd_group_helper_diagnostics_replace_unresolved_calls():
    code = """
    shader GLSLMetalSimdGroupHelperDiagnostics {
        compute {
            void main() {
                uint lane = 1u;
                bool flag = true;
                float badSum = __metal_simd_sum(flag);
                uint badArity = __metal_simd_max(lane, lane);
                uint unsupported = __metal_simd_shuffle_down(lane, 1u);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    assert (
        "__metal_simd_sum requires a numeric scalar or vector value argument: "
        "flag has type bool"
    ) in generated
    assert "__metal_simd_max expects 1 arguments, got 2" in generated
    assert (
        "/* GLSL wave intrinsic diagnostic: __metal_simd_shuffle_down "
        "has no OpenGL subgroup equivalent */ 0u"
    ) in generated
    assert "subgroupAdd(flag)" not in generated
    assert "subgroupMax(lane, lane)" not in generated
    assert "__metal_simd_shuffle_down(" not in generated


def test_glsl_wave_intrinsic_invalid_arities_and_wave_match_lowering():
    code = """
    shader GLSLWaveDiagnostics {
        compute {
            void main() {
                uint lane = WaveGetLaneIndex(1u);
                uint missing = WaveReadLaneAt(lane);
                uvec4 matchMask = WaveMatch(lane);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    assert (
        "uint lane = /* GLSL wave intrinsic diagnostic: WaveGetLaneIndex "
        "expects 0 arguments, got 1 */ 0u;" in generated
    )
    assert (
        "uint missing = /* GLSL wave intrinsic diagnostic: WaveReadLaneAt "
        "expects 2 arguments, got 1 */ 0u;" in generated
    )
    assert "uvec4 crossglWaveMatch(uint value)" in generated
    assert "uvec4 matchMask = crossglWaveMatch(lane);" in generated
    assert "WaveOpNode" not in generated


def test_glsl_wave_intrinsic_type_mismatches_emit_diagnostics():
    code = """
    shader GLSLWaveTypeDiagnostics {
        compute {
            void main() {
                bool flag;
                float value;
                vec2 values;
                mat2 transform;
                float badSum = WaveActiveSum(flag);
                uint badBits = WaveActiveBitAnd(value);
                bool badVote = WaveActiveAnyTrue(values);
                uint badLane = WaveReadLaneAt(1u, value);
                uvec4 badMatch = WaveMatch(transform);
                uint badMultiMask = WaveMultiPrefixSum(1u, values);
                uint badMultiCount = WaveMultiPrefixCountBits(value, uvec4(1u));
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    for expected in [
        (
            "WaveActiveSum requires a numeric scalar or vector value argument: "
            "flag has type bool"
        ),
        (
            "WaveActiveBitAnd requires an integer or boolean scalar or vector "
            "value argument: value has type float"
        ),
        (
            "WaveActiveAnyTrue requires a boolean scalar value argument: "
            "values has type vec2"
        ),
        (
            "WaveReadLaneAt requires a scalar integer lane argument: "
            "value has type float"
        ),
        (
            "WaveMatch requires a scalar or vector value argument: "
            "transform has type mat2"
        ),
        (
            "WaveMultiPrefixSum requires a uvec4 partition mask argument: "
            "values has type vec2"
        ),
        (
            "WaveMultiPrefixCountBits requires a boolean scalar value argument: "
            "value has type float"
        ),
    ]:
        assert expected in generated

    assert "subgroupAdd(flag)" not in generated
    assert "subgroupAnd(value)" not in generated
    assert "subgroupAny(values)" not in generated
    assert "subgroupShuffle(1u, value)" not in generated
    assert "crossglWaveMatch(transform)" not in generated
    assert "crossglWaveMultiPrefixSum(1u, values)" not in generated
    assert "crossglWaveMultiPrefixCountBits(value, uvec4(1u))" not in generated


def test_glsl_additional_wave_ballot_count_intrinsics_lower_or_diagnose():
    code = """
    shader GLSLAdditionalWaveIntrinsics {
        compute {
            void main() {
                uint lane = WaveGetLaneIndex();
                mat2 transform;
                vec2 values;
                bool equalLane = WaveActiveAllEqual(lane);
                uint activeCount = WaveActiveCountBits(lane > 0u);
                uint prefixCount = WavePrefixCountBits(lane > 1u);
                bool badEqual = WaveActiveAllEqual(transform);
                uint badCount = WaveActiveCountBits(values);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    assert "#extension GL_KHR_shader_subgroup_basic : require" in generated
    assert "#extension GL_KHR_shader_subgroup_vote : require" in generated
    assert "#extension GL_KHR_shader_subgroup_ballot : require" in generated
    assert "bool equalLane = subgroupAllEqual(lane);" in generated
    assert (
        "uint activeCount = "
        "subgroupBallotBitCount(subgroupBallot((lane > 0u)));" in generated
    )
    assert (
        "uint prefixCount = "
        "subgroupBallotExclusiveBitCount(subgroupBallot((lane > 1u)));" in generated
    )
    assert (
        "WaveActiveAllEqual requires a scalar or vector value argument: "
        "transform has type mat2"
    ) in generated
    assert (
        "WaveActiveCountBits requires a boolean scalar value argument: "
        "values has type vec2"
    ) in generated
    assert "WaveOpNode" not in generated
    assert "WaveActiveAllEqual(" not in generated
    assert "WaveActiveCountBits(" not in generated
    assert "WavePrefixCountBits(" not in generated


def test_glsl_wave_match_and_multiprefix_forms_lower_to_helpers():
    code = """
    shader GLSLWaveMultiPrefixDiagnostics {
        compute {
            void main() {
                uint lane = WaveGetLaneIndex();
                uvec4 mask = WaveActiveBallot(lane > 0u);
                uvec4 matchMask = WaveMatch(lane);
                uint sumValue = WaveMultiPrefixSum(lane, mask);
                uint productValue = WaveMultiPrefixProduct(lane + 1u, mask);
                uint countValue = WaveMultiPrefixCountBits(lane > 1u, mask);
                uint andValue = WaveMultiPrefixBitAnd(lane, mask);
                uint orValue = WaveMultiPrefixBitOr(lane, mask);
                uint xorValue = WaveMultiPrefixBitXor(lane, mask);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    assert "#extension GL_KHR_shader_subgroup_ballot : require" in generated
    assert "#extension GL_KHR_shader_subgroup_shuffle : require" in generated
    assert "uvec4 mask = subgroupBallot((lane > 0u));" in generated
    for expected in [
        "bool crossglWaveMaskContains(uvec4 mask, uint lane)",
        "void crossglWaveMaskSet(inout uvec4 mask, uint lane)",
        "uvec4 crossglWaveMatch(uint value)",
        "uint crossglWaveMultiPrefixCountBits(bool value, uvec4 mask)",
        "uvec4 matchMask = crossglWaveMatch(lane);",
        "uint sumValue = crossglWaveMultiPrefixSum(lane, mask);",
        "uint productValue = crossglWaveMultiPrefixProduct((lane + 1u), mask);",
        "uint countValue = crossglWaveMultiPrefixCountBits((lane > 1u), mask);",
        "uint andValue = crossglWaveMultiPrefixBitAnd(lane, mask);",
        "uint orValue = crossglWaveMultiPrefixBitOr(lane, mask);",
        "uint xorValue = crossglWaveMultiPrefixBitXor(lane, mask);",
        "result += subgroupShuffle(value, lane);",
        "result += subgroupShuffle(laneValue, lane);",
    ]:
        assert expected in generated

    assert "subgroupClustered" not in generated
    assert "subgroupInclusive" not in generated
    assert "WaveOpNode" not in generated


def test_else_if_statement():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;

            if (input.texCoord.x > 0.5) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
            } else if (input.texCoord.x < 0.5) {
                output.color = vec4(0.0, 0.0, 0.0, 1.0);
            } else {
                output.color = vec4(0.5, 0.5, 0.5, 1.0);
            }

            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            if (bloom > 0.5) {
                bloom = 0.5;
            } else if (bloom < 0.5) {
                bloom = 0.1;
            } else {
                bloom = 0.0;
            }


            // Apply bloom to the texture color
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);

            return vec4(colorWithBloom, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_function_call():
    code = """
    shader perlinNoise {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    vec2 perlinNoise(vec2 uv) {
        return vec2(0.0, 0.0);
    }

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            vec2 noise = perlinNoise(input.texCoord);
            if (input.texCoord.x > 0.5) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
            } else {
                output.color = vec4(0.0, 0.0, 0.0, 1.0);
            }

            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            vec2 noise = perlinNoise(input.color.xy);
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            if (bloom > 0.5) {
                bloom = 0.5;
            } else {
                bloom = 0.0;
            }

            // Apply bloom to the texture color
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);

            return vec4(colorWithBloom, 1.0);
        }
    }
}

    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_assignment_shift_operators():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            int x = 2;
            x >>= 1;
            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            int x = 2;
            x <<= 1;

            // Apply bloom to the texture color
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);

            return vec4(colorWithBloom, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_bitwise_operators():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            int x = 2;
            x = x & 1;
            x = x | 1;
            x = x + 1;

            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            int x = 2+ 2;
            x = x + 1;
            x = x + 1;
            x = x + 1;

            // Apply bloom to the texture color
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);

            return vec4(colorWithBloom, 1.0);
        }
    }
}

    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Bitwise Shift parsing not implemented.")


def test_opengl_array_handling(array_test_data):
    """Test the OpenGL code generator's handling of array types and array access."""
    code = """
    shader main {
    struct Particle {
        vec3 position;
        vec3 velocity;
    };

    struct Material {
        float values[4];  // Fixed-size array
        vec3 colors[];    // Dynamic array
    };

    cbuffer Constants {
        float weights[8];
        int indices[10];
    };

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;

            // Array access in various forms
            float value = weights[2];
            int index = indices[5];

            // Array member access
            Material material;
            float x = material.values[0];
            vec3 color = material.colors[index];

            // Nested array access
            Particle particles[10];
            vec3 pos = particles[3].position;

            // Array access in expressions
            float sum = weights[0] + weights[1] + weights[2];

            return output;
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)

        # Use the fixture data for verification
        for expected in array_test_data["glsl"]["array_type_declarations"]:
            assert expected in generated_code

        for expected in array_test_data["glsl"]["array_access"]:
            assert expected in generated_code

    except SyntaxError as e:
        pytest.fail(f"OpenGL array codegen failed: {e}")


def test_opengl_cbuffer_members_are_not_global_variables():
    code = """
    shader CBufferScope {
        cbuffer Constants {
            float weights[8];
            vec3 colors[2];
        };

        compute {
            void main() {
                float value = weights[2];
                vec3 color = colors[1];
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    generated_code = GLSLCodeGen().generate_stage(ast, "compute")

    assert [buffer.name for buffer in getattr(ast, "cbuffers", [])] == ["Constants"]
    assert [var.name for var in ast.global_variables] == []
    assert "layout(std140, binding = 0) uniform Constants" in generated_code
    assert "float weights[8];" in generated_code
    assert "vec3 colors[2];" in generated_code
    assert "float value = weights[2];" in generated_code
    assert "vec3 color = colors[1];" in generated_code
    assert "layout(std140, binding = 0) float weights[8];" not in generated_code


def test_opengl_duplicate_cbuffer_members_error():
    code = """
    shader CBufferScope {
        cbuffer Camera {
            float value;
        };

        compute {
            void main() {
                float x = value;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    ast.cbuffers.append(
        StructNode(
            name="Lighting",
            members=[
                StructMemberNode(name="value", member_type=PrimitiveType("float"))
            ],
        )
    )

    with pytest.raises(
        ValueError,
        match="Ambiguous cbuffer member name\\(s\\) in OpenGL output: value",
    ):
        GLSLCodeGen().generate_stage(ast, "compute")


def test_opengl_duplicate_cbuffer_names_error():
    code = """
    shader CBufferScope {
        cbuffer Camera {
            float exposure;
        };

        compute {
            void main() {
                float x = exposure;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    ast.cbuffers.append(
        StructNode(
            name="Camera",
            members=[
                StructMemberNode(name="gamma", member_type=PrimitiveType("float"))
            ],
        )
    )

    with pytest.raises(
        ValueError,
        match="Duplicate cbuffer name\\(s\\) in OpenGL output: Camera",
    ):
        GLSLCodeGen().generate_stage(ast, "compute")


def test_opengl_cbuffer_name_conflicts_with_struct_error():
    code = """
    shader CBufferScope {
        struct Camera {
            float exposure;
        };

        cbuffer Lighting {
            float gamma;
        };

        compute {
            void main() {
                float x = gamma;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    ast.cbuffers[0].name = "Camera"

    with pytest.raises(
        ValueError,
        match="Cbuffer name\\(s\\) conflict with existing OpenGL declaration\\(s\\): Camera",
    ):
        GLSLCodeGen().generate_stage(ast, "compute")


@pytest.mark.parametrize(
    "array_shader, expected_outputs",
    [
        (
            """
            shader TestArrayShader {
                struct DataArray {
                    float values[4];
                };

                void main() {
                    DataArray data;
                    data.values[2] = 1.0;

                    float arr[5];
                    arr[0] = 1.0;
                }
            }
            """,
            ["float values[4]", "data.values[2]", "arr[0]"],
        )
    ],
)
def test_array_handling(array_shader, expected_outputs):
    try:
        ast = crosstl.translator.parse(array_shader)
        code_gen = GLSLCodeGen()
        generated_code = code_gen.generate(ast)

        for expected in expected_outputs:
            assert expected in generated_code
    except Exception as e:
        pytest.fail(f"OpenGL array codegen failed: {e}")


def test_opengl_local_array_declarations_use_glsl_order():
    shader = """
    shader TestShader {
        void main() {
            vec3 localColors[4];
            float weights[8];
            localColors[0] = vec3(1.0, 0.0, 0.0);
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "vec3 localColors[4];" in generated_code
    assert "float weights[8];" in generated_code
    assert "vec3[4] localColors" not in generated_code
    assert "float[8] weights" not in generated_code


def test_opengl_array_parameters_use_glsl_order():
    shader = """
    shader TestShader {
        float accumulate(float weights[4], vec3 colors[2]) {
            return weights[0] + colors[1].x;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "float accumulate(float weights[4], vec3 colors[2])" in generated_code
    assert "float[4] weights" not in generated_code
    assert "vec3[2] colors" not in generated_code


def test_opengl_helper_parameters_preserve_out_and_inout_qualifiers():
    shader = """
    shader OutParamGLSL {
        void fill(out float value) {
            value = 1.0;
        }

        void accumulate(inout float value) {
            value += 1.0;
        }

        compute {
            void main() {
                float x = 0.0;
                fill(x);
                accumulate(x);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "void fill(out float value)" in generated_code
    assert "void accumulate(inout float value)" in generated_code
    assert "void fill(float value)" not in generated_code
    assert "void accumulate(float value)" not in generated_code


def test_opengl_non_resource_arrays_preserve_expression_sizes():
    shader = """
    shader ArrayExpressionSizes {
        struct Payload {
            vec3 colors[(2 + 1) * 2];
            float weights[+6];
        };

        float accumulate(float values[(2 + 1) * 2], vec3 normals[+6]) {
            float localWeights[(2 + 1) * 2];
            vec3 localNormals[+6];
            return values[2] + normals[2].x + localWeights[2] + localNormals[2].x;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "vec3 colors[((2 + 1) * 2)];" in generated_code
    assert "float weights[(+6)];" in generated_code
    assert (
        "float accumulate(float values[((2 + 1) * 2)], vec3 normals[(+6)])"
        in generated_code
    )
    assert "float localWeights[((2 + 1) * 2)];" in generated_code
    assert "vec3 localNormals[(+6)];" in generated_code
    assert "float values[]" not in generated_code
    assert "float localWeights[]" not in generated_code


def test_opengl_sampler_globals_are_uniform_resources():
    shader = """
    shader TextureShader {
        sampler2D colorMap;
        samplerCube envMap;

        struct VSOutput {
            vec2 uv;
            vec3 normal;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 color = texture(colorMap, uv);
                vec4 env = texture(envMap, normal);
                return color + env;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCube envMap;" in generated_code
    assert "layout(std140, binding = 0) sampler2D" not in generated_code
    assert "in vec2 uv;" in generated_code
    assert "in vec3 normal;" in generated_code
    assert "VectorType(" not in generated_code


def test_opengl_version_330_omits_explicit_resource_binding_layouts():
    shader = """
    #version 330
    shader Binding330Resources {
        sampler2D texture0 @binding(0);

        cbuffer Uniforms @binding(0) {
            vec4 colDiffuse;
        };

        fragment {
            vec4 main() @gl_FragColor {
                return texture(texture0, vec2(0.0)) * colDiffuse;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "#version 330" in generated_code
    assert "uniform sampler2D texture0;" in generated_code
    assert "layout(std140) uniform Uniforms" in generated_code
    assert "binding =" not in generated_code


def test_opengl_version_330_keeps_resource_bindings_with_420pack_extension():
    shader = """
    #version 330
    #extension GL_ARB_shading_language_420pack : require
    shader Binding330ResourcesWithExtension {
        sampler2D texture0 @binding(0);

        cbuffer Uniforms @binding(0) {
            vec4 colDiffuse;
        };

        fragment {
            vec4 main() @gl_FragColor {
                return texture(texture0, vec2(0.0)) * colDiffuse;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "#version 330" in generated_code
    assert "#extension GL_ARB_shading_language_420pack : require" in generated_code
    assert "layout(binding = 0) uniform sampler2D texture0;" in generated_code
    assert "layout(std140, binding = 0) uniform Uniforms" in generated_code


def test_opengl_rejects_non_resource_shadow_of_global_resource():
    shader = """
    shader ResourceShadow {
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv;
        };

        vec4 shade(float colorMap, FSInput input) {
            float linearSampler = 1.0;
            return vec4(colorMap + linearSampler);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return shade(1.0, input);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Non-resource local declaration\\(s\\) shadow OpenGL global "
            "resource\\(s\\): colorMap, linearSampler"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_rejects_texture_call_with_non_resource_argument():
    shader = """
    shader InvalidTextureArgument {
        struct FSInput {
            vec2 uv;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float value = 1.0;
                return texture(value, input.uv);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "OpenGL texture operation 'texture' requires a declared texture "
            "or image resource argument: value"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            "texture(colorMap)",
            "OpenGL texture operation 'texture' requires at least 2 "
            "argument\\(s\\), got 1",
        ),
        (
            "textureLod(colorMap, input.uv)",
            "OpenGL texture operation 'textureLod' requires at least 3 "
            "argument\\(s\\), got 2",
        ),
        (
            "textureGrad(colorMap, input.uv, input.uv)",
            "OpenGL texture operation 'textureGrad' requires at least 4 "
            "argument\\(s\\), got 3",
        ),
        (
            "texture(colorMap, linearSampler)",
            "OpenGL texture operation 'texture' requires at least 3 "
            "argument\\(s\\), got 2",
        ),
    ],
)
def test_opengl_rejects_texture_call_with_too_few_arguments(call, match):
    shader = f"""
    shader InvalidTextureArity {{
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {{
            vec2 uv;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expression", "operation", "max_count", "arg_count"),
    [
        (
            "vec4(float(textureSamples(msTex, input.layer)))",
            "textureSamples",
            1,
            2,
        ),
        (
            "vec4(float(imageSamples(msTex, input.layer)))",
            "imageSamples",
            1,
            2,
        ),
        (
            "vec4(float(textureQueryLevels(colorMap, input.layer)))",
            "textureQueryLevels",
            1,
            2,
        ),
        (
            "vec4(vec2(imageSize(colorImage, input.layer)), 0.0, 1.0)",
            "imageSize",
            1,
            2,
        ),
        (
            "vec4(textureSize(colorMap, input.layer, input.layer), 0.0, 1.0)",
            "textureSize",
            2,
            3,
        ),
    ],
)
def test_opengl_rejects_resource_query_call_with_too_many_arguments(
    return_expression, operation, max_count, arg_count
):
    shader = f"""
    shader InvalidTextureQueryArity {{
        sampler2D colorMap;
        sampler2DMS msTex;
        image2D colorImage;

        struct FSInput {{
            ivec2 pixel;
            int layer;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expression};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL texture operation '{operation}' accepts at most "
            f"{max_count} argument\\(s\\), got {arg_count}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("statement", "operation", "max_count", "arg_count"),
    [
        (
            "vec4 fetched = texelFetch(colorMap, input.pixel, input.layer, input.layer);",
            "texelFetch",
            3,
            4,
        ),
        (
            "vec4 fetched = texelFetchOffset(colorMap, input.pixel, input.layer, input.offset, input.layer);",
            "texelFetchOffset",
            4,
            5,
        ),
        (
            "vec4 color = imageLoad(colorImage, input.pixel, input.layer);",
            "imageLoad",
            2,
            3,
        ),
        (
            "imageStore(colorImage, input.pixel, vec4(1.0), input.layer);",
            "imageStore",
            3,
            4,
        ),
        (
            "uint oldValue = imageAtomicAdd(counterImage, input.pixel, input.amount, input.amount);",
            "imageAtomicAdd",
            3,
            4,
        ),
        (
            "uint oldValue = imageAtomicCompSwap(counterImage, input.pixel, input.amount, input.amount, input.amount);",
            "imageAtomicCompSwap",
            4,
            5,
        ),
    ],
)
def test_opengl_rejects_fixed_resource_call_with_too_many_arguments(
    statement, operation, max_count, arg_count
):
    shader = f"""
    shader InvalidFixedResourceArity {{
        sampler2D colorMap;
        image2D colorImage;
        uimage2D counterImage;

        struct FSInput {{
            ivec2 pixel;
            ivec2 offset;
            int layer;
            uint amount;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                {statement}
                return vec4(1.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL texture operation '{operation}' accepts at most "
            f"{max_count} argument\\(s\\), got {arg_count}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expression", "operation", "max_count", "arg_count"),
    [
        (
            "texture(colorMap, input.uv, input.bias, input.bias)",
            "texture",
            3,
            4,
        ),
        (
            "textureLod(colorMap, input.uv, input.bias, input.bias)",
            "textureLod",
            3,
            4,
        ),
        (
            "textureGrad(colorMap, input.uv, input.ddx, input.ddy, input.bias)",
            "textureGrad",
            4,
            5,
        ),
        (
            "textureOffset(colorMap, input.uv, input.offset, input.bias, input.bias)",
            "textureOffset",
            4,
            5,
        ),
        (
            "vec4(textureCompare(shadowMap, compareSampler, input.uv, input.depth, input.bias))",
            "textureCompare",
            4,
            5,
        ),
        (
            "textureGather(colorMap, input.uv, input.component, input.component)",
            "textureGather",
            3,
            4,
        ),
        (
            "textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depth, input.offset, input.component)",
            "textureGatherCompareOffset",
            5,
            6,
        ),
    ],
)
def test_opengl_rejects_texture_sampling_call_with_too_many_arguments(
    return_expression, operation, max_count, arg_count
):
    shader = f"""
    shader InvalidTextureSamplingArity {{
        sampler2D colorMap;
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec2 ddx;
            vec2 ddy;
            ivec2 offset;
            float bias;
            float depth;
            int component;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expression};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL texture operation '{operation}' accepts at most "
            f"{max_count} argument\\(s\\), got {arg_count}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_rejects_texture_gather_offsets_with_ambiguous_argument_count():
    shader = """
    shader InvalidTextureGatherOffsetsArity {
        sampler2D colorMap;

        struct FSInput {
            vec2 uv;
            ivec2 offset;
            int component;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return textureGatherOffsets(
                    colorMap,
                    input.uv,
                    input.offset,
                    input.offset,
                    input.component
                );
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "OpenGL texture operation 'textureGatherOffsets' accepts "
            "3, 4, 6, 7 argument\\(s\\), got 5"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("statement", "operation"),
    [
        ("vec4 color = imageLoad(colorMap, input.pixel);", "imageLoad"),
        ("imageStore(colorMap, input.pixel, vec4(1.0));", "imageStore"),
        ("ivec2 size = imageSize(colorMap);", "imageSize"),
        (
            "uint oldValue = imageAtomicAdd(colorMap, input.pixel, input.amount);",
            "imageAtomicAdd",
        ),
    ],
)
def test_opengl_rejects_image_call_with_sampled_texture_argument(statement, operation):
    shader = f"""
    shader InvalidImageResource {{
        sampler2D colorMap;

        struct FSInput {{
            ivec2 pixel;
            uint amount;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                {statement}
                return vec4(1.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL image operation '{operation}' requires a storage "
            "image resource argument: colorMap"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("helper", "helper_call", "match"),
    [
        (
            "vec4 misuse(sampler sampleState, vec2 uv) { return texture(sampleState, uv); }",
            "misuse(linearSampler, input.uv)",
            "OpenGL texture operation 'texture' requires a declared texture "
            "or image resource argument: sampleState",
        ),
        (
            "vec4 misuse(sampler sampleState, ivec2 pixel) { return imageLoad(sampleState, pixel); }",
            "misuse(linearSampler, input.pixel)",
            "OpenGL image operation 'imageLoad' requires a storage image "
            "resource argument: sampleState",
        ),
    ],
)
def test_opengl_rejects_sampler_parameter_as_texture_or_image_operand(
    helper, helper_call, match
):
    shader = f"""
    shader InvalidSamplerOperand {{
        sampler linearSampler;

        struct FSInput {{
            vec2 uv;
            ivec2 pixel;
        }};

        {helper}

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {helper_call};
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        ("texelFetch(colorMap, input.uv, 0)", "texelFetch"),
        ("imageLoad(colorImage, input.uv)", "imageLoad"),
    ],
)
def test_opengl_rejects_float_coordinate_for_integer_resource_operation(
    call, operation
):
    shader = f"""
    shader InvalidResourceCoordinate {{
        sampler2D colorMap;
        image2D colorImage;

        struct FSInput {{
            vec2 uv;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL resource operation '{operation}' requires an integer "
            "coordinate argument: input.uv has type vec2"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation", "dimension"),
    [
        ("texelFetch(arrayMap, input.pixel2, 0)", "texelFetch", 3),
        ("imageLoad(volumeImage, input.pixel2)", "imageLoad", 3),
        ("imageLoad(colorImage, input.pixel3)", "imageLoad", 2),
    ],
)
def test_opengl_rejects_wrong_coordinate_dimension_for_resource_operation(
    call, operation, dimension
):
    shader = f"""
    shader InvalidResourceCoordinateDimension {{
        sampler2DArray arrayMap;
        image2D colorImage;
        image3D volumeImage;

        struct FSInput {{
            ivec2 pixel2;
            ivec3 pixel3;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL resource operation '{operation}' requires a "
            f"{dimension}D integer coordinate"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        ("textureOffset(arrayMap, input.uvLayer, input.offset3)", "textureOffset"),
        (
            "texelFetchOffset(arrayMap, input.pixelLayer, 0, input.offset3)",
            "texelFetchOffset",
        ),
    ],
)
def test_opengl_rejects_wrong_offset_dimension_for_resource_operation(call, operation):
    shader = f"""
    shader InvalidResourceOffsetDimension {{
        sampler2DArray arrayMap;

        struct FSInput {{
            vec3 uvLayer;
            ivec3 pixelLayer;
            ivec3 offset3;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(f"OpenGL resource operation '{operation}' requires a 2D integer offset"),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        (
            "textureProjOffset(arrayMap, input.projLayer, input.offset3)",
            "textureProjOffset",
        ),
        (
            "textureGatherOffset(arrayMap, input.uvLayer, input.offset3)",
            "textureGatherOffset",
        ),
        (
            "textureGatherOffsets(arrayMap, input.uvLayer, input.offset3, input.offset3, input.offset3, input.offset3)",
            "textureGatherOffsets",
        ),
        (
            "vec4(textureCompareOffset(shadowMap, compareSampler, input.uv, input.depth, input.offset3))",
            "textureCompareOffset",
        ),
        (
            "vec4(textureCompareProjGradOffset(shadowMap, compareSampler, input.projCoord, input.depth, input.ddx, input.ddy, input.offset3))",
            "textureCompareProjGradOffset",
        ),
        (
            "textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depth, input.offset3)",
            "textureGatherCompareOffset",
        ),
        (
            "textureGatherCompareOffsets(shadowMap, compareSampler, input.uv, input.depth, input.offset3, input.offset3, input.offset3, input.offset3)",
            "textureGatherCompareOffsets",
        ),
    ],
)
def test_opengl_rejects_wrong_offset_dimension_for_extended_resource_operation(
    call, operation
):
    shader = f"""
    shader InvalidExtendedResourceOffsetDimension {{
        sampler2DArray arrayMap;
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec3 uvLayer;
            vec3 projCoord;
            vec4 projLayer;
            vec2 ddx;
            vec2 ddy;
            float depth;
            ivec3 offset3;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(f"OpenGL resource operation '{operation}' requires a 2D integer offset"),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_rejects_float_offset_for_resource_operation():
    shader = """
    shader InvalidResourceOffsetType {
        sampler2D colorMap;

        struct FSInput {
            vec2 uv;
            vec2 offset;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return textureOffset(colorMap, input.uv, input.offset);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "OpenGL resource operation 'textureOffset' requires an integer "
            "offset argument"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        (
            "textureGatherOffset(colorMap, input.uv, input.offset)",
            "textureGatherOffset",
        ),
        (
            "textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depth, input.offset)",
            "textureGatherCompareOffset",
        ),
        (
            "textureGatherCompareOffsets(shadowMap, compareSampler, input.uv, input.depth, input.offset, input.offset, input.offset, input.offset)",
            "textureGatherCompareOffsets",
        ),
    ],
)
def test_opengl_rejects_float_offset_for_extended_resource_operation(call, operation):
    shader = f"""
    shader InvalidExtendedResourceOffsetType {{
        sampler2D colorMap;
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec2 offset;
            float depth;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL resource operation '{operation}' requires an integer "
            "offset argument"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation", "dimension"),
    [
        ("textureGrad(colorMap, input.uv, input.ddx3, input.ddy3)", "textureGrad", 2),
        (
            "textureGrad(cubeMap, input.direction, input.ddx2, input.ddy2)",
            "textureGrad",
            3,
        ),
        (
            "textureGradOffset(colorMap, input.uv, input.ddx3, input.ddy3, input.offset2)",
            "textureGradOffset",
            2,
        ),
        (
            "textureProjGrad(colorMap, input.projCoord, input.ddx3, input.ddy3)",
            "textureProjGrad",
            2,
        ),
        (
            "vec4(textureCompareGrad(shadowMap, compareSampler, input.uv, input.depth, input.ddx3, input.ddy3))",
            "textureCompareGrad",
            2,
        ),
        (
            "vec4(textureCompareProjGradOffset(shadowMap, compareSampler, input.projCoord, input.depth, input.ddx3, input.ddy3, input.offset2))",
            "textureCompareProjGradOffset",
            2,
        ),
    ],
)
def test_opengl_rejects_wrong_gradient_dimension_for_resource_operation(
    call, operation, dimension
):
    shader = f"""
    shader InvalidResourceGradientDimension {{
        sampler2D colorMap;
        samplerCube cubeMap;
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec3 direction;
            vec3 projCoord;
            vec2 ddx2;
            vec2 ddy2;
            vec3 ddx3;
            vec3 ddy3;
            ivec2 offset2;
            float depth;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL resource operation '{operation}' requires a "
            f"{dimension}D floating gradient"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_rejects_integer_gradient_for_resource_operation():
    shader = """
    shader InvalidResourceGradientType {
        sampler2D colorMap;

        struct FSInput {
            vec2 uv;
            ivec2 pixel;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return textureGrad(colorMap, input.uv, input.pixel, input.pixel);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "OpenGL resource operation 'textureGrad' requires a floating "
            "gradient argument"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation", "type_name"),
    [
        (
            "textureLod(colorMap, linearSampler, input.uv, input.lodVec)",
            "textureLod",
            "vec2",
        ),
        (
            "textureLodOffset(colorMap, linearSampler, input.uv, input.lodVec, input.offset)",
            "textureLodOffset",
            "vec2",
        ),
        (
            "textureProjLod(colorMap, linearSampler, input.projCoord, input.lodVec)",
            "textureProjLod",
            "vec2",
        ),
        (
            "vec4(textureCompareLod(shadowMap, compareSampler, input.uv, input.depth, input.lodVec))",
            "textureCompareLod",
            "vec2",
        ),
        (
            "vec4(textureCompareProjLodOffset(shadowMap, compareSampler, input.projCoord, input.depth, input.lodVec, input.offset))",
            "textureCompareProjLodOffset",
            "vec2",
        ),
        (
            "textureLod(colorMap, linearSampler, input.uv, colorMap)",
            "textureLod",
            "sampler2D",
        ),
    ],
)
def test_opengl_rejects_non_scalar_numeric_lod_argument(call, operation, type_name):
    shader = f"""
    shader InvalidTextureLodArgument {{
        sampler2D colorMap;
        sampler2DShadow shadowMap;
        sampler linearSampler;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec3 projCoord;
            vec2 lodVec;
            ivec2 offset;
            float depth;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL texture LOD operation '{operation}' requires a scalar "
            f"numeric lod argument: .* has type {type_name}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expr", "operation", "type_name"),
    [
        (
            "texelFetch(colorMap, input.pixel, input.floatLevel)",
            "texelFetch",
            "float",
        ),
        (
            "texelFetchOffset(colorMap, input.pixel, input.levelVec, input.offset)",
            "texelFetchOffset",
            "ivec2",
        ),
        (
            "vec4(textureSize(colorMap, input.floatLevel), 0.0, 1.0)",
            "textureSize",
            "float",
        ),
        (
            "texelFetch(colorMap, input.pixel, colorMap)",
            "texelFetch",
            "sampler2D",
        ),
    ],
)
def test_opengl_rejects_non_scalar_integer_mip_level_argument(
    return_expr, operation, type_name
):
    shader = f"""
    shader InvalidTextureMipLevelArgument {{
        sampler2D colorMap;

        struct FSInput {{
            ivec2 pixel;
            ivec2 offset;
            ivec2 levelVec;
            float floatLevel;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expr};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL resource operation '{operation}' requires a scalar integer "
            f"mip/sample level argument: .* has type {type_name}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expr", "type_name"),
    [
        ("texelFetch(msTex, input.pixel, input.floatSample)", "float"),
        ("texelFetch(msArray, input.pixelLayer, input.sampleVec)", "ivec2"),
        ("texelFetch(msTex, input.pixel, msTex)", "sampler2DMS"),
    ],
)
def test_opengl_rejects_non_scalar_integer_multisample_sample_index(
    return_expr, type_name
):
    shader = f"""
    shader InvalidMultisampleTexelFetchSampleIndex {{
        sampler2DMS msTex;
        sampler2DMSArray msArray;

        struct FSInput {{
            ivec2 pixel;
            ivec3 pixelLayer;
            ivec2 sampleVec;
            float floatSample;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expr};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "OpenGL multisample texel fetch operation 'texelFetch' requires a "
            f"scalar integer sample index argument: .* has type {type_name}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expr", "helper", "operation", "type_name"),
    [
        (
            "textureGather(colorMap, linearSampler, input.uv, input.floatComponent)",
            "",
            "textureGather",
            "float",
        ),
        (
            "textureGatherOffset(colorMap, linearSampler, input.uv, input.offset, input.componentVec)",
            "",
            "textureGatherOffset",
            "ivec2",
        ),
        (
            "textureGatherOffsets(colorMap, linearSampler, input.uv, input.offset, input.offset, input.offset, input.offset, colorMap)",
            "",
            "textureGatherOffsets",
            "sampler2D",
        ),
        (
            "vec4(1.0)",
            "vec4 invalidArrayGather(sampler2D tex, sampler s, vec2 uv, ivec2 offsets[4], float component) { return textureGatherOffsets(tex, s, uv, offsets, component); }",
            "textureGatherOffsets",
            "float",
        ),
    ],
)
def test_opengl_rejects_non_scalar_integer_gather_component_argument(
    return_expr, helper, operation, type_name
):
    shader = f"""
    shader InvalidTextureGatherComponentArgument {{
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {{
            vec2 uv;
            ivec2 offset;
            ivec2 componentVec;
            float floatComponent;
        }};

        {helper}

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expr};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL texture gather operation '{operation}' requires a scalar "
            f"integer component argument: .* has type {type_name}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expr", "operation", "type_name"),
    [
        (
            "texture(colorMap, linearSampler, input.uv, input.biasVec)",
            "texture",
            "vec2",
        ),
        (
            "textureOffset(colorMap, linearSampler, input.uv, input.offset, input.biasVec)",
            "textureOffset",
            "vec2",
        ),
        (
            "textureProj(colorMap, linearSampler, input.projCoord, input.biasVec)",
            "textureProj",
            "vec2",
        ),
        (
            "textureProjOffset(colorMap, linearSampler, input.projCoord, input.offset, colorMap)",
            "textureProjOffset",
            "sampler2D",
        ),
    ],
)
def test_opengl_rejects_non_scalar_numeric_bias_argument(
    return_expr, operation, type_name
):
    shader = f"""
    shader InvalidTextureBiasArgument {{
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {{
            vec2 uv;
            vec3 projCoord;
            vec2 biasVec;
            ivec2 offset;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expr};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL texture bias operation '{operation}' requires a scalar "
            f"numeric bias argument: .* has type {type_name}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_rejects_non_floating_query_lod_coordinate():
    shader = """
    shader InvalidTextureQueryLodCoordinate {
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {
            ivec2 pixel;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return vec4(textureQueryLod(colorMap, linearSampler, input.pixel), 0.0, 1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "OpenGL texture query operation 'textureQueryLod' requires a "
            "floating coordinate argument: .* has type ivec2"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("query_call", "dimension", "resource_type", "type_name"),
    [
        (
            "textureQueryLod(layerMap, linearSampler, input.uv)",
            3,
            "sampler2DArray",
            "vec2",
        ),
        (
            "textureQueryLod(cubeMap, linearSampler, input.uv)",
            3,
            "samplerCube",
            "vec2",
        ),
        (
            "textureQueryLod(cubeArray, linearSampler, input.direction)",
            4,
            "samplerCubeArray",
            "vec3",
        ),
    ],
)
def test_opengl_rejects_wrong_query_lod_coordinate_dimension(
    query_call, dimension, resource_type, type_name
):
    shader = f"""
    shader InvalidTextureQueryLodCoordinateDimension {{
        sampler2DArray layerMap;
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler linearSampler;

        struct FSInput {{
            vec2 uv;
            vec3 direction;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return vec4({query_call}, 0.0, 1.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "OpenGL texture query operation 'textureQueryLod' requires a "
            f"{dimension}D floating coordinate for {resource_type}: "
            f".* has type {type_name}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation", "type_name"),
    [
        (
            "vec4(textureCompare(shadowMap, compareSampler, input.uv, input.depthVec))",
            "textureCompare",
            "vec2",
        ),
        (
            "vec4(textureCompareProjGradOffset(shadowMap, compareSampler, input.projCoord, input.depthVec, input.ddx, input.ddy, input.offset))",
            "textureCompareProjGradOffset",
            "vec2",
        ),
        (
            "textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depthVec, input.offset)",
            "textureGatherCompareOffset",
            "vec2",
        ),
        (
            "vec4(textureCompare(shadowMap, compareSampler, input.uv, input.layer))",
            "textureCompare",
            "int",
        ),
    ],
)
def test_opengl_rejects_non_scalar_float_compare_argument(call, operation, type_name):
    shader = f"""
    shader InvalidTextureCompareArgument {{
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec3 projCoord;
            vec2 depthVec;
            vec2 ddx;
            vec2 ddy;
            ivec2 offset;
            int layer;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL texture compare operation '{operation}' requires a scalar "
            f"floating compare argument: .* has type {type_name}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_texture_array_resources_and_indexed_sampling():
    shader = """
    shader TextureArrayShader {
        sampler2D textures[4];
        samplerCube envMap;

        struct VSOutput {
            vec2 uv;
            vec3 normal;
            int layer;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 color = texture(textures[layer], uv);
                vec4 env = texture(envMap, normal);
                return color + env;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "layout(binding = 4) uniform samplerCube envMap;" in generated_code
    assert "texture(textures[layer], uv)" in generated_code
    assert "texture(envMap, normal)" in generated_code


def test_glsl_separate_texture_sampler_arrays_preserve_nonuniform_indexing():
    shader = """
    shader SeparateTextureSamplerNonuniform {
        texture2d textures[8] @binding(0);
        sampler samplers[8] @binding(1);

        struct VSOutput {
            vec2 uv;
            uint materialIndex;
        };

        vec4 sampleMaterial(
            texture2d textures[8],
            sampler samplers[8],
            uint materialIndex,
            vec2 uv
        ) {
            return texture(
                textures[nonuniformEXT(materialIndex)],
                samplers[nonuniformEXT(materialIndex)],
                uv
            );
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleMaterial(
                    textures,
                    samplers,
                    input.materialIndex,
                    input.uv
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "#extension GL_EXT_nonuniform_qualifier : require" in generated_code
    assert "layout(binding = 0) uniform texture2D textures[8];" in generated_code
    assert "layout(binding = 1) uniform sampler samplers[8];" in generated_code
    assert (
        "vec4 sampleMaterial(texture2D textures[8], sampler samplers[8], "
        "uint materialIndex, vec2 uv)"
    ) in generated_code
    assert (
        "texture(sampler2D(textures[nonuniformEXT(materialIndex)], "
        "samplers[nonuniformEXT(materialIndex)]), uv)"
    ) in generated_code
    assert "in vec2 uv;" in generated_code
    assert "flat in uint materialIndex;" in generated_code
    assert "sampleMaterial(textures, samplers, materialIndex, uv)" in generated_code
    assert "texture(textures[nonuniformEXT(materialIndex)], uv)" not in generated_code
    assert "samplers[nonuniformEXT(materialIndex)]" in generated_code


def test_opengl_fixed_texture_array_keeps_declared_size_with_constant_indices():
    shader = """
    shader FixedArrayConstantIndex {
        const int LAYER = 2;
        sampler2D textures[6];
        sampler samplers[6];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[6], sampler samplers[6], vec2 uv) {
            return texture(textures[LAYER], samplers[LAYER], uv) + texture(textures[1 + 2], samplers[1 + 2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int LAYER = 2;" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[6];" in generated_code
    assert "layout(binding = 6) uniform sampler2D afterTexture;" in generated_code
    assert "vec4 sampleLayer(sampler2D textures[6], vec2 uv)" in generated_code
    assert "texture(textures[LAYER], uv)" in generated_code
    assert "texture(textures[(1 + 2)], uv)" in generated_code
    assert "sampleLayer(textures, uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[4];" not in generated_code
    assert "layout(binding = 4) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[LAYER]" not in generated_code
    assert "samplers[(1 + 2)]" not in generated_code


def test_opengl_fixed_texture_array_resolves_constant_declared_size_for_bindings():
    shader = """
    shader ConstSizedResourceArrays {
        const int BASE_COUNT = 2;
        const int TEXTURE_COUNT = BASE_COUNT * 3;
        sampler2D textures[TEXTURE_COUNT];
        sampler samplers[TEXTURE_COUNT];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[TEXTURE_COUNT], sampler samplers[TEXTURE_COUNT], vec2 uv) {
            return texture(textures[2], samplers[2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int BASE_COUNT = 2;" in generated_code
    assert "const int TEXTURE_COUNT = (BASE_COUNT * 3);" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2D textures[TEXTURE_COUNT];"
        in generated_code
    )
    assert "layout(binding = 6) uniform sampler2D afterTexture;" in generated_code
    assert (
        "vec4 sampleLayer(sampler2D textures[TEXTURE_COUNT], vec2 uv)" in generated_code
    )
    assert "texture(textures[2], uv)" in generated_code
    assert "sampleLayer(textures, uv)" in generated_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code


def test_opengl_fixed_texture_array_resolves_inline_declared_size_expression_for_bindings():
    shader = """
    shader ExprSizedResourceArrays {
        sampler2D textures[2 * 3];
        sampler samplers[2 * 3];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[2 * 3], sampler samplers[2 * 3], vec2 uv) {
            return texture(textures[2], samplers[2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(binding = 0) uniform sampler2D textures[(2 * 3)];" in generated_code
    assert "layout(binding = 6) uniform sampler2D afterTexture;" in generated_code
    assert "vec4 sampleLayer(sampler2D textures[(2 * 3)], vec2 uv)" in generated_code
    assert "texture(textures[2], uv)" in generated_code
    assert "sampleLayer(textures, uv)" in generated_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "[None]" not in generated_code


def test_opengl_fixed_texture_arrays_preserve_parenthesized_and_unary_declared_sizes():
    shader = """
    shader ParenthesizedSizedResourceArrays {
        sampler2D textures[(2 + 1) * 2];
        sampler samplers[(2 + 1) * 2];
        sampler2D unaryTextures[+6];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[(2 + 1) * 2], sampler samplers[(2 + 1) * 2], sampler2D unaryTextures[+6], vec2 uv) {
            return texture(textures[2], samplers[2], uv) + texture(unaryTextures[2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, unaryTextures, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2D textures[((2 + 1) * 2)];"
        in generated_code
    )
    assert (
        "layout(binding = 6) uniform sampler2D unaryTextures[(+6)];" in generated_code
    )
    assert "layout(binding = 12) uniform sampler2D afterTexture;" in generated_code
    assert (
        "vec4 sampleLayer(sampler2D textures[((2 + 1) * 2)], sampler2D unaryTextures[(+6)], vec2 uv)"
        in generated_code
    )
    assert "sampleLayer(textures, unaryTextures, uv)" in generated_code
    assert "texture(textures[2], uv)" in generated_code
    assert "texture(unaryTextures[2], uv)" in generated_code
    assert "layout(binding = 6) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "[None]" not in generated_code


def test_opengl_texture_array_helper_parameter_keeps_sampler_array():
    shader = """
    shader TextureArrayHelper {
        sampler2D textures[4];

        struct VSOutput {
            vec2 uv;
            int layer;
        };

        vec4 sampleLayer(sampler2D textures[4], int layer, vec2 uv) {
            return texture(textures[layer], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, layer, uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "vec4 sampleLayer(sampler2D textures[4], int layer, vec2 uv)" in generated_code
    )
    assert "texture(textures[layer], uv)" in generated_code
    assert "sampleLayer(textures, layer, uv)" in generated_code


def test_opengl_texture_array_helper_parameter_filters_indexed_sampler_array():
    shader = """
    shader SamplerArrayHelper {
        sampler2D textures[4];
        sampler samplers[4];

        struct VSOutput {
            vec2 uv;
            int layer;
        };

        vec4 sampleLayer(sampler2D textures[4], sampler samplers[4], int layer, vec2 uv) {
            return texture(textures[layer], samplers[layer], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, layer, uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert (
        "vec4 sampleLayer(sampler2D textures[4], int layer, vec2 uv)" in generated_code
    )
    assert "texture(textures[layer], uv)" in generated_code
    assert "sampleLayer(textures, layer, uv)" in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[layer]" not in generated_code


def test_opengl_unsized_texture_array_infers_helper_size_and_filters_sampler_array():
    shader = """
    shader UnsizedSampledResourceArrays {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            vec4 color = texture(textures[2], samplers[2], uv);
            vec4 other = texture(textures[1], samplers[1], uv);
            return color + other;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D textures[3];" in generated_code
    assert "layout(binding = 3) uniform sampler2D afterTexture;" in generated_code
    assert "vec4 sampleLayer(sampler2D textures[3], vec2 uv)" in generated_code
    assert "texture(textures[2], uv)" in generated_code
    assert "texture(textures[1], uv)" in generated_code
    assert "sampleLayer(textures, uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[];" not in generated_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[2]" not in generated_code


def test_opengl_unsized_texture_array_infers_transitive_helper_size():
    shader = """
    shader MultiHopUnsizedSampledResources {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleDeep(sampler2D textures[], sampler samplers[], vec2 uv) {
            vec4 high = texture(textures[4], samplers[4], uv);
            vec4 low = texture(textures[1], samplers[1], uv);
            return high + low;
        }

        vec4 sampleMid(sampler2D textures[], sampler samplers[], vec2 uv) {
            return sampleDeep(textures, samplers, uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleMid(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D textures[5];" in generated_code
    assert "layout(binding = 5) uniform sampler2D afterTexture;" in generated_code
    assert "vec4 sampleDeep(sampler2D textures[5], vec2 uv)" in generated_code
    assert "vec4 sampleMid(sampler2D textures[5], vec2 uv)" in generated_code
    assert "texture(textures[4], uv)" in generated_code
    assert "texture(textures[1], uv)" in generated_code
    assert "sampleDeep(textures, uv)" in generated_code
    assert "sampleMid(textures, uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[];" not in generated_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[4]" not in generated_code


def test_opengl_unsized_texture_array_preserves_dynamic_indexing():
    shader = """
    shader MixedIndexedUnsizedSampledResources {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
            int layer;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], int layer, vec2 uv) {
            vec4 dynamicColor = texture(textures[layer], samplers[layer], uv);
            vec4 fixedColor = texture(textures[3], samplers[3], uv);
            return dynamicColor + fixedColor;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, layer, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "layout(binding = 4) uniform sampler2D afterTexture;" in generated_code
    assert (
        "vec4 sampleLayer(sampler2D textures[4], int layer, vec2 uv)" in generated_code
    )
    assert "texture(textures[layer], uv)" in generated_code
    assert "texture(textures[3], uv)" in generated_code
    assert "sampleLayer(textures, layer, uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[];" not in generated_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[layer]" not in generated_code


def test_opengl_unsized_texture_array_ignores_unsupported_indices():
    dynamic_shader = """
    shader DynamicOnlyUnsizedSampledResources {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
            int layer;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], int layer, vec2 uv) {
            return texture(textures[layer], samplers[layer], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, layer, uv) + texture(afterTexture, uv);
            }
        }
    }
    """
    negative_shader = """
    shader NegativeIndexedUnsizedSampledResources {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[-1], samplers[-1], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    dynamic_code = GLSLCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = GLSLCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "layout(binding = 0) uniform sampler2D textures[1];" in dynamic_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" in dynamic_code
    assert "vec4 sampleLayer(sampler2D textures[1], int layer, vec2 uv)" in dynamic_code
    assert "texture(textures[layer], uv)" in dynamic_code
    assert "layout(binding = 0) uniform sampler2D textures[];" not in dynamic_code
    assert "layout(binding = 2) uniform sampler2D afterTexture;" not in dynamic_code
    assert "sampler samplers" not in dynamic_code
    assert "samplers[layer]" not in dynamic_code

    assert "layout(binding = 0) uniform sampler2D textures[1];" in negative_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" in negative_code
    assert "vec4 sampleLayer(sampler2D textures[1], vec2 uv)" in negative_code
    assert "texture(textures[0], uv)" in negative_code
    assert "texture(textures[(-1)], uv)" not in negative_code
    assert "layout(binding = 0) uniform sampler2D textures[];" not in negative_code
    assert "layout(binding = 0) uniform sampler2D textures[0];" not in negative_code
    assert "layout(binding = 0) uniform sampler2D afterTexture;" not in negative_code
    assert "sampler samplers" not in negative_code


def test_opengl_unsized_texture_array_infers_constant_expression_size():
    shader = """
    shader ExprIndexedUnsizedSampledResources {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[1 + 2], samplers[1 + 2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "layout(binding = 4) uniform sampler2D afterTexture;" in generated_code
    assert "vec4 sampleLayer(sampler2D textures[4], vec2 uv)" in generated_code
    assert "texture(textures[(1 + 2)], uv)" in generated_code
    assert "sampleLayer(textures, uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[];" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[(1 + 2)]" not in generated_code


def test_opengl_unsized_texture_array_infers_named_constant_size():
    shader = """
    shader ConstIndexedUnsizedSampledResources {
        const int BASE = 1;
        const int LAYER = BASE + 2;
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[LAYER], samplers[LAYER], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int BASE = 1;" in generated_code
    assert "const int LAYER = (BASE + 2);" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "layout(binding = 4) uniform sampler2D afterTexture;" in generated_code
    assert "vec4 sampleLayer(sampler2D textures[4], vec2 uv)" in generated_code
    assert "texture(textures[LAYER], uv)" in generated_code
    assert "sampleLayer(textures, uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[];" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[LAYER]" not in generated_code


def test_opengl_unsized_texture_array_ignores_shadowed_constant_name():
    shader = """
    shader ShadowedConstIndex {
        const int LAYER = 3;
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
            int layer;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], int LAYER, vec2 uv) {
            return texture(textures[LAYER], samplers[LAYER], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, layer, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int LAYER = 3;" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[1];" in generated_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" in generated_code
    assert (
        "vec4 sampleLayer(sampler2D textures[1], int LAYER, vec2 uv)" in generated_code
    )
    assert "texture(textures[LAYER], uv)" in generated_code
    assert "sampleLayer(textures, layer, uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[];" not in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[4];" not in generated_code
    assert "layout(binding = 4) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[LAYER]" not in generated_code


def test_opengl_fixed_texture_array_global_conflicts_with_fixed_parameter_size():
    shader = """
    shader FixedSampledGlobalMismatch {
        sampler2D textures[4];
        sampler samplers[4];

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleThree(sampler2D textures[3], sampler samplers[3], vec2 uv) {
            return texture(textures[2], samplers[2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleThree(textures, samplers, uv);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_opengl_fixed_texture_array_global_widens_unsized_parameter_size():
    shader = """
    shader FixedSampledGlobalToUnsizedHelper {
        sampler2D textures[4];
        sampler samplers[4];

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleUnsized(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[2], samplers[2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleUnsized(textures, samplers, uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "vec4 sampleUnsized(sampler2D textures[4], vec2 uv)" in generated_code
    assert "texture(textures[2], uv)" in generated_code
    assert "sampleUnsized(textures, uv)" in generated_code
    assert "vec4 sampleUnsized(sampler2D textures[3], vec2 uv)" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[2]" not in generated_code


def test_opengl_fixed_texture_array_global_direct_index_out_of_bounds_raises():
    shader = """
    shader FixedSampledGlobalDirectOutOfBounds {
        sampler2D textures[4];
        sampler samplers[4];

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return texture(textures[4], samplers[4], uv);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_opengl_fixed_texture_array_parameter_direct_index_out_of_bounds_raises():
    shader = """
    shader FixedSampledParamDirectOutOfBounds {
        vec4 sampleLayer(sampler2D textures[4], sampler samplers[4], vec2 uv) {
            return texture(textures[4], samplers[4], uv);
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return vec4(0.0);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_opengl_fixed_texture_array_global_const_index_out_of_bounds_raises():
    shader = """
    shader FixedSampledGlobalConstIndexOutOfBounds {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return texture(textures[COUNT], samplers[COUNT], uv);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_opengl_fixed_texture_array_parameter_const_index_out_of_bounds_raises():
    shader = """
    shader FixedSampledParamConstIndexOutOfBounds {
        const int COUNT = 4;

        vec4 sampleLayer(sampler2D textures[4], sampler samplers[4], vec2 uv) {
            return texture(textures[COUNT], samplers[COUNT], uv);
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return vec4(0.0);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_opengl_fixed_texture_array_const_index_and_shadowing_generate():
    shader = """
    shader FixedSampledConstIndexWithinBounds {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        vec4 sampleConst(sampler2D textures[4], sampler samplers[4], vec2 uv) {
            return texture(textures[COUNT - 1], samplers[COUNT - 1], uv);
        }

        vec4 sampleShadowed(sampler2D textures[4], sampler samplers[4], vec2 uv) {
            int COUNT = 0;
            return texture(textures[COUNT], samplers[COUNT], uv);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                int COUNT = 0;
                vec4 direct = texture(textures[COUNT], samplers[COUNT], input.uv);
                return direct + sampleConst(textures, samplers, input.uv) + sampleShadowed(textures, samplers, input.uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "vec4 sampleConst(sampler2D textures[4], vec2 uv)" in generated_code
    assert "vec4 sampleShadowed(sampler2D textures[4], vec2 uv)" in generated_code
    assert "texture(textures[(COUNT - 1)], uv)" in generated_code
    assert "texture(textures[COUNT], uv)" in generated_code
    assert "sampleConst(textures, uv)" in generated_code
    assert "sampleShadowed(textures, uv)" in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[COUNT]" not in generated_code
    assert "samplers[(COUNT - 1)]" not in generated_code
    assert "sampler2D textures[5]" not in generated_code


def test_opengl_transitive_texture_array_shadowed_const_index_stays_dynamic():
    shader = """
    shader TransitiveSampledShadowedConstIndex {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv) {
            int COUNT = 0;
            return texture(textures[COUNT], samplers[COUNT], uv);
        }

        vec4 passThrough(sampler2D textures[], sampler samplers[], vec2 uv) {
            int COUNT = 0;
            vec4 sampled = texture(textures[COUNT], samplers[COUNT], uv);
            return sampled + leaf(textures, samplers, uv);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return passThrough(textures, samplers, input.uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "vec4 leaf(sampler2D textures[4], vec2 uv)" in generated_code
    assert "vec4 passThrough(sampler2D textures[4], vec2 uv)" in generated_code
    assert "return texture(textures[COUNT], uv);" in generated_code
    assert "vec4 sampled = texture(textures[COUNT], uv);" in generated_code
    assert "return (sampled + leaf(textures, uv));" in generated_code
    assert "fragColor = passThrough(textures, uv);" in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[COUNT]" not in generated_code
    assert "sampler2D textures[5]" not in generated_code


def test_opengl_transitive_texture_array_unshadowed_const_index_conflict_raises():
    shader = """
    shader TransitiveSampledUnshadowedConstIndexConflict {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[COUNT], samplers[COUNT], uv);
        }

        vec4 passThrough(sampler2D textures[], sampler samplers[], vec2 uv) {
            int COUNT = 0;
            return leaf(textures, samplers, uv);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return passThrough(textures, samplers, input.uv);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_opengl_texture_array_helper_operation_variants_filter_sampler_array():
    shader = """
    shader TextureArrayOps {
        sampler2D textures[4];
        sampler samplers[4];

        struct VSOutput {
            vec2 uv;
            ivec2 pixel;
            int layer;
        };

        vec4 sampleOps(sampler2D textures[4], sampler samplers[4], int layer, vec2 uv, ivec2 pixel) {
            vec4 lodColor = textureLod(textures[layer], samplers[layer], uv, 2.0);
            vec4 gradColor = textureGrad(textures[layer], samplers[layer], uv, vec2(0.1), vec2(0.2));
            vec4 gathered = textureGather(textures[layer], samplers[layer], uv);
            vec4 fetched = texelFetch(textures[layer], pixel, 0);
            return lodColor + gradColor + gathered + fetched;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleOps(textures, samplers, layer, uv, pixel);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert (
        "vec4 sampleOps(sampler2D textures[4], int layer, vec2 uv, ivec2 pixel)"
        in generated_code
    )
    assert "textureLod(textures[layer], uv, 2.0)" in generated_code
    assert "textureGrad(textures[layer], uv, vec2(0.1), vec2(0.2))" in generated_code
    assert "textureGather(textures[layer], uv)" in generated_code
    assert "texelFetch(textures[layer], pixel, 0)" in generated_code
    assert "sampleOps(textures, layer, uv, pixel)" in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[layer]" not in generated_code


def test_opengl_array_texture_types_and_shadow_compare_coordinates():
    shader = """
    shader ArrayTextureTypes {
        sampler2DArray colorArray;
        sampler2DArrayShadow shadowArray;
        samplerCubeShadow cubeShadow;
        sampler arraySampler;
        sampler shadowSampler;

        struct VSOutput {
            vec3 uvLayer;
            ivec3 pixelLayer;
            vec3 direction;
            float depth;
        };

        vec4 sampleArray(sampler2DArray tex, sampler s, vec3 uvLayer, ivec3 pixelLayer) {
            vec4 color = texture(tex, s, uvLayer);
            vec4 lodColor = textureLod(tex, s, uvLayer, 1.0);
            vec4 gradColor = textureGrad(tex, s, uvLayer, vec2(0.1), vec2(0.2));
            vec4 gathered = textureGather(tex, s, uvLayer);
            vec4 fetched = texelFetch(tex, pixelLayer, 0);
            return color + lodColor + gradColor + gathered + fetched;
        }

        float sampleShadowArray(sampler2DArrayShadow tex, sampler s, vec3 uvLayer, float depth) {
            return textureCompare(tex, s, uvLayer, depth);
        }

        float sampleCubeShadow(samplerCubeShadow tex, sampler s, vec3 direction, float depth) {
            return textureCompare(tex, s, direction, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float shadow = sampleShadowArray(shadowArray, shadowSampler, uvLayer, depth);
                float cube = sampleCubeShadow(cubeShadow, shadowSampler, direction, depth);
                return sampleArray(colorArray, arraySampler, uvLayer, pixelLayer) * shadow * cube;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2DArray colorArray;" in generated_code
    assert (
        "layout(binding = 1) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert "layout(binding = 2) uniform samplerCubeShadow cubeShadow;" in generated_code
    assert (
        "vec4 sampleArray(sampler2DArray tex, vec3 uvLayer, ivec3 pixelLayer)"
        in generated_code
    )
    assert "texture(tex, uvLayer)" in generated_code
    assert "textureLod(tex, uvLayer, 1.0)" in generated_code
    assert "textureGrad(tex, uvLayer, vec2(0.1), vec2(0.2))" in generated_code
    assert "textureGather(tex, uvLayer)" in generated_code
    assert "texelFetch(tex, pixelLayer, 0)" in generated_code
    assert (
        "float sampleShadowArray(sampler2DArrayShadow tex, vec3 uvLayer, float depth)"
        in generated_code
    )
    assert "texture(tex, vec4(uvLayer, depth))" in generated_code
    assert (
        "float sampleCubeShadow(samplerCubeShadow tex, vec3 direction, float depth)"
        in generated_code
    )
    assert "texture(tex, vec4(direction, depth))" in generated_code
    assert "sampler s" not in generated_code
    assert "arraySampler" not in generated_code
    assert "shadowSampler" not in generated_code
    assert "vec3(uvLayer, depth)" not in generated_code
    assert "vec3(direction, depth)" not in generated_code


def test_opengl_cube_array_and_multisample_texture_types():
    shader = """
    shader CubeMsResources {
        samplerCubeArray cubeArray;
        samplerCubeArrayShadow cubeArrayShadow;
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler cubeSampler;
        sampler shadowSampler;

        struct VSOutput {
            vec4 cubeLayer;
            ivec2 pixel;
            ivec3 pixelLayer;
            int sampleIndex;
            float depth;
        };

        vec4 sampleCubeArray(samplerCubeArray tex, sampler s, vec4 cubeLayer) {
            return texture(tex, s, cubeLayer) + textureLod(tex, s, cubeLayer, 2.0);
        }

        float sampleCubeArrayShadow(samplerCubeArrayShadow tex, sampler s, vec4 cubeLayer, float depth) {
            return textureCompare(tex, s, cubeLayer, depth);
        }

        vec4 fetchMs(sampler2DMS tex, sampler2DMSArray texArray, ivec2 pixel, ivec3 pixelLayer, int sampleIndex) {
            return texelFetch(tex, pixel, sampleIndex) + texelFetch(texArray, pixelLayer, sampleIndex);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float shadow = sampleCubeArrayShadow(cubeArrayShadow, shadowSampler, cubeLayer, depth);
                return sampleCubeArray(cubeArray, cubeSampler, cubeLayer) * shadow + fetchMs(msTex, msArray, pixel, pixelLayer, sampleIndex);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform samplerCubeArray cubeArray;" in generated_code
    assert (
        "layout(binding = 1) uniform samplerCubeArrayShadow cubeArrayShadow;"
        in generated_code
    )
    assert "layout(binding = 2) uniform sampler2DMS msTex;" in generated_code
    assert "layout(binding = 3) uniform sampler2DMSArray msArray;" in generated_code
    assert (
        "vec4 sampleCubeArray(samplerCubeArray tex, vec4 cubeLayer)" in generated_code
    )
    assert "texture(tex, cubeLayer)" in generated_code
    assert "textureLod(tex, cubeLayer, 2.0)" in generated_code
    assert (
        "float sampleCubeArrayShadow(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth)"
        in generated_code
    )
    assert "texture(tex, cubeLayer, depth)" in generated_code
    assert (
        "vec4 fetchMs(sampler2DMS tex, sampler2DMSArray texArray, ivec2 pixel, ivec3 pixelLayer, int sampleIndex)"
        in generated_code
    )
    assert "texelFetch(tex, pixel, sampleIndex)" in generated_code
    assert "texelFetch(texArray, pixelLayer, sampleIndex)" in generated_code
    assert "cubeSampler" not in generated_code
    assert "shadowSampler" not in generated_code
    assert "vec4(cubeLayer, depth)" not in generated_code


def test_opengl_cube_array_texture_grad_gather_filter_sampler_arguments():
    shader = """
    shader CubeArrayGradGather {
        samplerCubeArray cubeArray;
        samplerCubeArray cubeArrays[4];
        sampler cubeSampler;
        sampler cubeSamplers[4];

        struct FSInput {
            vec4 cubeLayer @ TEXCOORD0;
            vec3 ddx @ TEXCOORD1;
            vec3 ddy @ TEXCOORD2;
        };

        vec4 sampleCubeArrayOps(samplerCubeArray tex, sampler s, vec4 cubeLayer, vec3 ddx, vec3 ddy) {
            vec4 gradColor = textureGrad(tex, s, cubeLayer, ddx, ddy);
            vec4 gathered = textureGather(tex, s, cubeLayer);
            return gradColor + gathered;
        }

        vec4 sampleCubeArrayElements(samplerCubeArray cubeArrays[], sampler cubeSamplers[], vec4 cubeLayer, vec3 ddx, vec3 ddy) {
            vec4 gradColor = textureGrad(cubeArrays[2], cubeSamplers[2], cubeLayer, ddx, ddy);
            vec4 gathered = textureGather(cubeArrays[3], cubeSamplers[3], cubeLayer);
            return gradColor + gathered;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return sampleCubeArrayOps(cubeArray, cubeSampler, input.cubeLayer, input.ddx, input.ddy)
                    + sampleCubeArrayElements(cubeArrays, cubeSamplers, input.cubeLayer, input.ddx, input.ddy);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform samplerCubeArray cubeArray;" in generated_code
    assert (
        "layout(binding = 1) uniform samplerCubeArray cubeArrays[4];" in generated_code
    )
    assert (
        "vec4 sampleCubeArrayOps(samplerCubeArray tex, vec4 cubeLayer, vec3 ddx, vec3 ddy)"
        in generated_code
    )
    assert "textureGrad(tex, cubeLayer, ddx, ddy)" in generated_code
    assert "textureGather(tex, cubeLayer)" in generated_code
    assert (
        "vec4 sampleCubeArrayElements(samplerCubeArray cubeArrays[4], vec4 cubeLayer, vec3 ddx, vec3 ddy)"
        in generated_code
    )
    assert "textureGrad(cubeArrays[2], cubeLayer, ddx, ddy)" in generated_code
    assert "textureGather(cubeArrays[3], cubeLayer)" in generated_code
    assert "sampleCubeArrayOps(cubeArray, cubeLayer, ddx, ddy)" in generated_code
    assert "sampleCubeArrayElements(cubeArrays, cubeLayer, ddx, ddy)" in generated_code
    assert "cubeSampler" not in generated_code
    assert "cubeSamplers" not in generated_code


def test_opengl_texture_sampling_capability_descriptors():
    codegen = GLSLCodeGen()

    def expect(texture_type, **overrides):
        expected = {
            "texture_type": codegen.resource_base_type(texture_type),
            "gather": False,
            "gather_offset": False,
            "sample_offset": False,
            "compare_offset": False,
            "compare_lod": False,
            "compare_grad": False,
            "compare_lod_offset": False,
            "compare_grad_offset": False,
            "gather_compare_offset": False,
        }
        expected.update(overrides)
        assert codegen.texture_sampling_capabilities(texture_type) == expected

    expect(
        "sampler2D",
        gather=True,
        gather_offset=True,
        sample_offset=True,
    )
    expect("samplerCube", gather=True)
    expect(
        "sampler2DArray[4]",
        gather=True,
        gather_offset=True,
        sample_offset=True,
    )
    expect(
        "sampler2DShadow",
        sample_offset=True,
        compare_offset=True,
        compare_lod=True,
        compare_grad=True,
        compare_lod_offset=True,
        compare_grad_offset=True,
        gather_compare_offset=True,
    )
    expect(
        "sampler2DArrayShadow",
        sample_offset=True,
        compare_offset=True,
        compare_grad=True,
        compare_grad_offset=True,
        gather_compare_offset=True,
    )
    expect("samplerCubeShadow", compare_grad=True)
    expect("sampler2DMS")


def test_opengl_texture_dimension_descriptors():
    codegen = GLSLCodeGen()

    def expect(texture_type, **overrides):
        expected = {
            "texture_type": codegen.resource_base_type(texture_type),
            "coordinate_dimension": None,
            "offset_dimension": None,
            "sample_offset_dimension": None,
            "texel_fetch_offset_dimension": None,
            "gather_offset_dimension": None,
            "compare_offset_dimension": None,
            "compare_lod_offset_dimension": None,
            "compare_grad_offset_dimension": None,
            "gather_compare_offset_dimension": None,
            "gradient_dimension": None,
            "query_lod_coordinate_dimension": None,
        }
        expected.update(overrides)
        assert codegen.texture_dimension_descriptor(texture_type) == expected

    expect(
        "sampler2DArray[4]",
        coordinate_dimension=3,
        offset_dimension=2,
        sample_offset_dimension=2,
        texel_fetch_offset_dimension=2,
        gather_offset_dimension=2,
        gradient_dimension=2,
        query_lod_coordinate_dimension=3,
    )
    expect(
        "sampler1DArray",
        coordinate_dimension=2,
        offset_dimension=1,
        sample_offset_dimension=1,
        texel_fetch_offset_dimension=1,
        gradient_dimension=1,
        query_lod_coordinate_dimension=2,
    )
    expect(
        "sampler2DShadow",
        offset_dimension=2,
        sample_offset_dimension=2,
        texel_fetch_offset_dimension=2,
        compare_offset_dimension=2,
        compare_lod_offset_dimension=2,
        compare_grad_offset_dimension=2,
        gather_compare_offset_dimension=2,
        gradient_dimension=2,
        query_lod_coordinate_dimension=2,
    )
    expect(
        "samplerCubeArray",
        gradient_dimension=3,
        query_lod_coordinate_dimension=4,
    )
    expect("sampler2DMSArray", coordinate_dimension=3)
    expect("image3D", coordinate_dimension=3)


def test_opengl_texture_gather_offset_variants_filter_sampler_arguments():
    shader = """
    shader GatherOffsetVariants {
        sampler2D colorMap;
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            ivec2 offset @ TEXCOORD2;
            ivec2 offset0 @ TEXCOORD3;
            ivec2 offset1 @ TEXCOORD4;
            ivec2 offset2 @ TEXCOORD5;
            ivec2 offset3 @ TEXCOORD6;
            int component @ TEXCOORD7;
        };

        vec4 gatherOps(
            sampler2D tex,
            sampler2DArray layers,
            sampler s,
            vec2 uv,
            vec3 uvLayer,
            ivec2 offset,
            ivec2 offset0,
            ivec2 offset1,
            ivec2 offset2,
            ivec2 offset3,
            int component
        ) {
            vec4 green = textureGather(tex, s, uv, 1);
            vec4 dynamic = textureGather(tex, s, uv, component);
            vec4 offsetGather = textureGatherOffset(tex, s, uv, offset, 3);
            vec4 dynamicOffset = textureGatherOffset(tex, s, uv, offset, component);
            vec4 offsetsGather = textureGatherOffsets(
                layers,
                s,
                uvLayer,
                offset0,
                offset1,
                offset2,
                offset3,
                component
            );
            return green + dynamic + offsetGather + dynamicOffset + offsetsGather;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherOps(
                    colorMap,
                    layerMap,
                    linearSampler,
                    input.uv,
                    input.uvLayer,
                    input.offset,
                    input.offset0,
                    input.offset1,
                    input.offset2,
                    input.offset3,
                    input.component
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler2DArray layerMap;" in generated_code
    assert (
        "vec4 gatherOps(sampler2D tex, sampler2DArray layers, vec2 uv, vec3 uvLayer, ivec2 offset, ivec2 offset0, ivec2 offset1, ivec2 offset2, ivec2 offset3, int component)"
        in generated_code
    )
    assert "textureGather(tex, uv, 1)" in generated_code
    assert (
        "component == 0 ? textureGather(tex, uv, 0) : "
        "component == 1 ? textureGather(tex, uv, 1) : "
        "component == 2 ? textureGather(tex, uv, 2) : textureGather(tex, uv, 3)"
        in generated_code
    )
    assert (
        generated_code.count(
            glsl_dynamic_texel_offset_diagnostic(
                "texture gather", "textureGatherOffset", "vec4(0.0)"
            )
        )
        == 2
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "texture gather", "textureGatherOffsets", "vec4(0.0)"
        )
        in generated_code
    )
    assert "textureGather(tex, uv, component)" not in generated_code
    assert "textureGatherOffset(tex, uv, offset, component)" not in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code
    assert (
        "gatherOps(colorMap, layerMap, uv, uvLayer, offset, offset0, offset1, offset2, offset3, component)"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert ", s, uv" not in generated_code
    assert ", s, uvLayer" not in generated_code


def test_opengl_texture_gather_compare_offsets_filter_sampler_arguments():
    shader = """
    shader GatherCompareOffsets {
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth @ TEXCOORD1;
            ivec2 offset0 @ TEXCOORD2;
            ivec2 offset1 @ TEXCOORD3;
            ivec2 offset2 @ TEXCOORD4;
            ivec2 offset3 @ TEXCOORD5;
        };

        vec4 gatherShadowOffsets(
            sampler2DShadow tex,
            sampler s,
            vec2 uv,
            float depth,
            ivec2 offsets[4],
            ivec2 offset0,
            ivec2 offset1,
            ivec2 offset2,
            ivec2 offset3
        ) {
            vec4 arrayOffsets = textureGatherCompareOffsets(
                tex,
                s,
                uv,
                depth,
                offsets
            );
            vec4 directOffsets = textureGatherCompareOffsets(
                tex,
                s,
                uv,
                depth,
                offset0,
                offset1,
                offset2,
                offset3
            );
            return arrayOffsets + directOffsets;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                ivec2 offsets[4];
                offsets[0] = input.offset0;
                offsets[1] = input.offset1;
                offsets[2] = input.offset2;
                offsets[3] = input.offset3;
                return gatherShadowOffsets(
                    shadowMap,
                    compareSampler,
                    input.uv,
                    input.depth,
                    offsets,
                    input.offset0,
                    input.offset1,
                    input.offset2,
                    input.offset3
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "sampler compareSampler" not in generated_code
    assert "sampler s" not in generated_code
    assert "textureGatherCompareOffsets(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert (
        generated_code.count(
            glsl_dynamic_texel_offset_diagnostic(
                "texture gather compare",
                "textureGatherCompareOffsets",
                "vec4(0.0)",
            )
        )
        == 2
    )


def test_opengl_texture_gather_offsets_accept_const_array_offsets():
    shader = """
    shader ConstGatherOffsets {
        sampler2D colorMap;
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth @ TEXCOORD1;
            int component @ TEXCOORD2;
        };

        vec4 gatherColor(sampler2D tex, vec2 uv, int component) {
            const ivec2 offsets[4] = {
                ivec2(-1, -1),
                ivec2(1, -1),
                ivec2(-1, 1),
                ivec2(1, 1)
            };
            return textureGatherOffsets(tex, uv, offsets, component);
        }

        vec4 gatherShadow(sampler2DShadow tex, sampler s, vec2 uv, float depth) {
            const int left = -1;
            const ivec2 offsets[4] = {
                ivec2(left, -1),
                ivec2(1, -1),
                ivec2(left, 1),
                ivec2(1, 1)
            };
            return textureGatherCompareOffsets(tex, s, uv, depth, offsets);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherColor(colorMap, input.uv, input.component)
                    + gatherShadow(shadowMap, compareSampler, input.uv, input.depth);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "sampler compareSampler" not in generated_code
    assert "sampler s" not in generated_code
    assert "const ivec2 offsets[4] = {" in generated_code
    assert "textureGatherOffsets(" not in generated_code
    assert "textureGatherCompareOffsets(" not in generated_code
    assert (
        "/* unsupported GLSL texture gather: textureGatherOffsets "
        "texel offsets must be compile-time integer constants */" not in generated_code
    )
    assert (
        "/* unsupported GLSL texture gather compare: textureGatherCompareOffsets "
        "texel offsets must be compile-time integer constants */" not in generated_code
    )
    assert "textureGatherOffset(tex, uv, offsets[0], 0).x" in generated_code
    assert "textureGatherOffset(tex, uv, offsets[3], 3).w" in generated_code
    assert "textureGatherOffset(tex, uv, depth, offsets[0]).x" in generated_code
    assert "textureGatherOffset(tex, uv, depth, offsets[3]).w" in generated_code


def test_opengl_texture_gather_offsets_mix_literal_and_dynamic_offsets():
    shader = """
    shader GatherOffsetMixed {
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            ivec2 dynamic0 @ TEXCOORD1;
            ivec2 dynamic1 @ TEXCOORD2;
            int component @ TEXCOORD3;
        };

        vec4 gatherOps(
            sampler2DArray layers,
            sampler s,
            vec3 uvLayer,
            ivec2 dynamic0,
            ivec2 dynamic1,
            int component
        ) {
            return textureGatherOffsets(
                layers,
                s,
                uvLayer,
                ivec2(-1, 0),
                dynamic0,
                ivec2(1, -1),
                dynamic1,
                component
            );
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherOps(
                    layerMap,
                    linearSampler,
                    input.uvLayer,
                    input.dynamic0,
                    input.dynamic1,
                    input.component
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DArray layerMap;" in generated_code
    assert (
        "vec4 gatherOps(sampler2DArray layers, vec3 uvLayer, ivec2 dynamic0, ivec2 dynamic1, int component)"
        in generated_code
    )
    assert "textureGatherOffsets(" not in generated_code
    assert "linearSampler" not in generated_code
    assert ", s, uvLayer" not in generated_code
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "texture gather", "textureGatherOffsets", "vec4(0.0)"
        )
        in generated_code
    )
    assert "textureGatherOffset(" not in generated_code
    assert (
        "gatherOps(layerMap, uvLayer, dynamic0, dynamic1, component)" in generated_code
    )


def test_opengl_direct_stage_gather_offsets_use_input_members():
    shader = """
    shader DirectStageGatherOffsets {
        sampler2D colorMap;
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            ivec2 offset @ TEXCOORD2;
            ivec2 offset0 @ TEXCOORD3;
            ivec2 offset1 @ TEXCOORD4;
            ivec2 offset2 @ TEXCOORD5;
            ivec2 offset3 @ TEXCOORD6;
            int component @ TEXCOORD7;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 dynamic = textureGather(colorMap, linearSampler, input.uv, input.component);
                vec4 dynamicOffset = textureGatherOffset(colorMap, linearSampler, input.uv, input.offset, input.component);
                vec4 offsetsGather = textureGatherOffsets(
                    layerMap,
                    linearSampler,
                    input.uvLayer,
                    input.offset0,
                    input.offset1,
                    input.offset2,
                    input.offset3,
                    input.component
                );
                return dynamic + dynamicOffset + offsetsGather;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler2DArray layerMap;" in generated_code
    assert (
        "vec4 dynamic = (component == 0 ? textureGather(colorMap, uv, 0) : "
        "component == 1 ? textureGather(colorMap, uv, 1) : "
        "component == 2 ? textureGather(colorMap, uv, 2) : "
        "textureGather(colorMap, uv, 3));" in generated_code
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "texture gather", "textureGatherOffset", "vec4(0.0)"
        )
        in generated_code
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "texture gather", "textureGatherOffsets", "vec4(0.0)"
        )
        in generated_code
    )
    assert "textureGather(colorMap, uv, component)" not in generated_code
    assert "textureGatherOffset(colorMap, uv, offset, component)" not in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code
    assert "input.uv" not in generated_code
    assert "input.offset" not in generated_code


def test_opengl_cube_texture_gather_dynamic_components_use_constant_branches():
    shader = """
    shader CubeGatherDynamicComponent {
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler cubeSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            int component @ TEXCOORD2;
        };

        vec4 gatherCubeOps(
            samplerCube cubeMap,
            samplerCubeArray cubeArray,
            sampler s,
            vec3 direction,
            vec4 cubeLayer,
            int component
        ) {
            vec4 cubeGather = textureGather(cubeMap, s, direction, component);
            vec4 cubeArrayGather = textureGather(cubeArray, s, cubeLayer, component);
            return cubeGather + cubeArrayGather;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherCubeOps(
                    cubeMap,
                    cubeArray,
                    cubeSampler,
                    input.direction,
                    input.cubeLayer,
                    input.component
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform samplerCube cubeMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCubeArray cubeArray;" in generated_code
    assert (
        "vec4 gatherCubeOps(samplerCube cubeMap, samplerCubeArray cubeArray, vec3 direction, vec4 cubeLayer, int component)"
        in generated_code
    )
    assert (
        "vec4 cubeGather = (component == 0 ? textureGather(cubeMap, direction, 0) : "
        "component == 1 ? textureGather(cubeMap, direction, 1) : "
        "component == 2 ? textureGather(cubeMap, direction, 2) : "
        "textureGather(cubeMap, direction, 3));" in generated_code
    )
    assert (
        "vec4 cubeArrayGather = (component == 0 ? textureGather(cubeArray, cubeLayer, 0) : "
        "component == 1 ? textureGather(cubeArray, cubeLayer, 1) : "
        "component == 2 ? textureGather(cubeArray, cubeLayer, 2) : "
        "textureGather(cubeArray, cubeLayer, 3));" in generated_code
    )
    assert (
        "gatherCubeOps(cubeMap, cubeArray, direction, cubeLayer, component)"
        in generated_code
    )
    assert "cubeSampler" not in generated_code
    assert "textureGather(cubeMap, direction, component)" not in generated_code
    assert "textureGather(cubeArray, cubeLayer, component)" not in generated_code


def test_opengl_cube_texture_gather_offsets_emit_diagnostics():
    shader = """
    shader UnsupportedCubeGatherOffsets {
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler cubeSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            ivec2 offset @ TEXCOORD2;
            ivec2 offset0 @ TEXCOORD3;
            ivec2 offset1 @ TEXCOORD4;
            ivec2 offset2 @ TEXCOORD5;
            ivec2 offset3 @ TEXCOORD6;
            int component @ TEXCOORD7;
        };

        vec4 gatherCube(
            samplerCube tex,
            sampler s,
            vec3 direction,
            ivec2 offset,
            ivec2 offset0,
            ivec2 offset1,
            ivec2 offset2,
            ivec2 offset3,
            int component
        ) {
            vec4 offsetGather = textureGatherOffset(tex, s, direction, offset, component);
            vec4 offsetsGather = textureGatherOffsets(tex, s, direction, offset0, offset1, offset2, offset3, component);
            return offsetGather + offsetsGather;
        }

        vec4 gatherCubeArray(
            samplerCubeArray tex,
            sampler s,
            vec4 cubeLayer,
            ivec2 offset,
            ivec2 offset0,
            ivec2 offset1,
            ivec2 offset2,
            ivec2 offset3,
            int component
        ) {
            vec4 offsetGather = textureGatherOffset(tex, s, cubeLayer, offset, component);
            vec4 offsetsGather = textureGatherOffsets(tex, s, cubeLayer, offset0, offset1, offset2, offset3, component);
            return offsetGather + offsetsGather;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherCube(
                    cubeMap,
                    cubeSampler,
                    input.direction,
                    input.offset,
                    input.offset0,
                    input.offset1,
                    input.offset2,
                    input.offset3,
                    input.component
                ) + gatherCubeArray(
                    cubeArray,
                    cubeSampler,
                    input.cubeLayer,
                    input.offset,
                    input.offset0,
                    input.offset1,
                    input.offset2,
                    input.offset3,
                    input.component
                ) + textureGatherOffset(cubeMap, input.direction, input.offset, input.component)
                    + textureGatherOffset(cubeArray, input.cubeLayer, input.offset, input.component)
                    + textureGatherOffsets(cubeMap, input.direction, input.offset0, input.offset1, input.offset2, input.offset3, input.component)
                    + textureGatherOffsets(cubeArray, input.cubeLayer, input.offset0, input.offset1, input.offset2, input.offset3, input.component);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    offset_diagnostic = (
        "/* unsupported GLSL texture gather: textureGatherOffset "
        "offsets require 2D or 2D-array textures */ vec4(0.0)"
    )
    offsets_diagnostic = (
        "/* unsupported GLSL texture gather: textureGatherOffsets "
        "offsets require 2D or 2D-array textures */ vec4(0.0)"
    )
    assert "layout(binding = 0) uniform samplerCube cubeMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCubeArray cubeArray;" in generated_code
    assert (
        "vec4 gatherCube(samplerCube tex, vec3 direction, ivec2 offset, ivec2 offset0, ivec2 offset1, ivec2 offset2, ivec2 offset3, int component)"
        in generated_code
    )
    assert (
        "vec4 gatherCubeArray(samplerCubeArray tex, vec4 cubeLayer, ivec2 offset, ivec2 offset0, ivec2 offset1, ivec2 offset2, ivec2 offset3, int component)"
        in generated_code
    )
    assert generated_code.count(offset_diagnostic) == 4
    assert generated_code.count(offsets_diagnostic) == 4
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code
    assert "cubeSampler" not in generated_code


def test_opengl_unsupported_dimension_texture_gather_emits_diagnostics():
    shader = """
    shader UnsupportedDimensionGather {
        sampler1D lineMap;
        sampler3D volumeMap;
        sampler linearSampler;

        struct FSInput {
            float u @ TEXCOORD0;
            vec3 volumeUv @ TEXCOORD1;
            int component @ TEXCOORD2;
        };

        vec4 gatherLine(sampler1D tex, sampler s, float u, int component) {
            vec4 fixedGather = textureGather(tex, s, u, 2);
            vec4 dynamicGather = textureGather(tex, s, u, component);
            return fixedGather + dynamicGather;
        }

        vec4 gatherVolume(sampler3D tex, sampler s, vec3 volumeUv, int component) {
            vec4 fixedGather = textureGather(tex, s, volumeUv, 1);
            vec4 dynamicGather = textureGather(tex, s, volumeUv, component);
            return fixedGather + dynamicGather;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherLine(lineMap, linearSampler, input.u, input.component)
                    + gatherVolume(volumeMap, linearSampler, input.volumeUv, input.component)
                    + textureGather(lineMap, input.u, input.component)
                    + textureGather(volumeMap, input.volumeUv, input.component);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported GLSL texture gather: textureGather requires 2D, "
        "2D-array, cube, or cube-array textures */ vec4(0.0)"
    )
    assert "layout(binding = 0) uniform sampler1D lineMap;" in generated_code
    assert "layout(binding = 1) uniform sampler3D volumeMap;" in generated_code
    assert "vec4 gatherLine(sampler1D tex, float u, int component)" in generated_code
    assert (
        "vec4 gatherVolume(sampler3D tex, vec3 volumeUv, int component)"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 6
    assert "textureGather(" not in generated_code
    assert "linearSampler" not in generated_code


def test_opengl_texture_lod_grad_offsets_filter_sampler_arguments():
    shader = """
    shader TextureSampleOffsets {
        sampler2D colorMap;
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 offsetOps(
            sampler2D tex,
            sampler2DArray layers,
            sampler s,
            vec2 uv,
            vec3 uvLayer,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec4 plain = textureOffset(tex, s, uv, offset);
            vec4 lodSample = textureLodOffset(tex, s, uv, lod, offset);
            vec4 gradSample = textureGradOffset(tex, s, uv, ddx, ddy, offset);
            vec4 arrayLod = textureLodOffset(layers, s, uvLayer, lod, offset);
            vec4 arrayGrad = textureGradOffset(layers, s, uvLayer, ddx, ddy, offset);
            return plain + lodSample + gradSample + arrayLod + arrayGrad;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return offsetOps(
                    colorMap,
                    layerMap,
                    linearSampler,
                    input.uv,
                    input.uvLayer,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "vec4 offsetOps(sampler2D tex, sampler2DArray layers, vec2 uv, vec3 uvLayer, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "texture offset", "textureOffset", "vec4(0.0)"
        )
        in generated_code
    )
    assert (
        generated_code.count(
            glsl_dynamic_texel_offset_diagnostic(
                "texture offset", "textureLodOffset", "vec4(0.0)"
            )
        )
        == 2
    )
    assert (
        generated_code.count(
            glsl_dynamic_texel_offset_diagnostic(
                "texture offset", "textureGradOffset", "vec4(0.0)"
            )
        )
        == 2
    )
    assert (
        "offsetOps(colorMap, layerMap, uv, uvLayer, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert ", s, uv" not in generated_code
    assert ", s, uvLayer" not in generated_code
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code


def test_opengl_cube_texture_sample_offsets_emit_diagnostics():
    shader = """
    shader CubeTextureSampleOffsetDiagnostics {
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        samplerCubeShadow shadowMap;
        samplerCubeArrayShadow shadowArray;
        sampler linearSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 cubeOffsets(
            samplerCube tex,
            sampler s,
            vec3 direction,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            vec4 plain = textureOffset(tex, s, direction, offset);
            vec4 lodSample = textureLodOffset(tex, s, direction, lod, offset);
            vec4 gradSample = textureGradOffset(tex, s, direction, ddx, ddy, offset);
            return plain + lodSample + gradSample;
        }

        vec4 cubeArrayOffsets(
            samplerCubeArray tex,
            sampler s,
            vec4 cubeLayer,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            vec4 plain = textureOffset(tex, s, cubeLayer, offset);
            vec4 lodSample = textureLodOffset(tex, s, cubeLayer, lod, offset);
            vec4 gradSample = textureGradOffset(tex, s, cubeLayer, ddx, ddy, offset);
            return plain + lodSample + gradSample;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 sampled = cubeOffsets(
                    cubeMap,
                    linearSampler,
                    input.direction,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + cubeArrayOffsets(
                    cubeArray,
                    linearSampler,
                    input.cubeLayer,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                )
                    + textureOffset(cubeMap, input.direction, input.offset)
                    + textureLodOffset(cubeMap, input.direction, input.lod, input.offset)
                    + textureGradOffset(cubeMap, input.direction, input.ddx, input.ddy, input.offset)
                    + textureOffset(cubeArray, input.cubeLayer, input.offset)
                    + textureLodOffset(cubeArray, input.cubeLayer, input.lod, input.offset)
                    + textureGradOffset(cubeArray, input.cubeLayer, input.ddx, input.ddy, input.offset);
                float compared = textureCompareOffset(shadowMap, input.direction, input.depth, input.offset)
                    + textureCompareLodOffset(shadowMap, input.direction, input.depth, input.lod, input.offset)
                    + textureCompareGradOffset(shadowMap, input.direction, input.depth, input.ddx, input.ddy, input.offset)
                    + textureCompareOffset(shadowArray, input.cubeLayer, input.depth, input.offset)
                    + textureCompareLodOffset(shadowArray, input.cubeLayer, input.depth, input.lod, input.offset)
                    + textureCompareGradOffset(shadowArray, input.cubeLayer, input.depth, input.ddx, input.ddy, input.offset);
                return sampled + vec4(compared);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform samplerCube cubeMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCubeArray cubeArray;" in generated_code
    assert "layout(binding = 2) uniform samplerCubeShadow shadowMap;" in generated_code
    assert (
        "layout(binding = 3) uniform samplerCubeArrayShadow shadowArray;"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert (
        "vec4 cubeOffsets(samplerCube tex, vec3 direction, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "vec4 cubeArrayOffsets(samplerCubeArray tex, vec4 cubeLayer, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture offset: textureOffset offsets require 1D, 2D, 2D-array, 3D, or planar shadow samplers */ vec4(0.0)"
        )
        == 4
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture offset: textureLodOffset offsets require 1D, 2D, 2D-array, 3D, or planar shadow samplers */ vec4(0.0)"
        )
        == 4
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture offset: textureGradOffset offsets require 1D, 2D, 2D-array, 3D, or planar shadow samplers */ vec4(0.0)"
        )
        == 4
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture compare: textureCompareOffset offsets require 2D or 2D-array shadow samplers */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture compare: textureCompareLodOffset explicit LOD offsets require 2D shadow samplers */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture compare: textureCompareGradOffset explicit gradient offsets require 2D or 2D-array shadow samplers */ 0.0"
        )
        == 2
    )
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code
    assert "textureCompareOffset(" not in generated_code
    assert "textureCompareLodOffset(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_opengl_direct_stage_sample_offsets_and_texel_fetch_offset_use_input_members():
    shader = """
    shader DirectStageSampleOffsets {
        sampler2D colorMap;
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            ivec2 pixel @ TEXCOORD2;
            ivec3 pixelLayer @ TEXCOORD3;
            float lod;
            vec2 ddx @ TEXCOORD4;
            vec2 ddy @ TEXCOORD5;
            ivec2 offset @ TEXCOORD6;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 plain = textureOffset(colorMap, linearSampler, input.uv, input.offset);
                vec4 lodSample = textureLodOffset(colorMap, linearSampler, input.uv, input.lod, input.offset);
                vec4 gradSample = textureGradOffset(colorMap, linearSampler, input.uv, input.ddx, input.ddy, input.offset);
                vec4 arrayPlain = textureOffset(layerMap, linearSampler, input.uvLayer, input.offset);
                vec4 arrayLod = textureLodOffset(layerMap, linearSampler, input.uvLayer, input.lod, input.offset);
                vec4 arrayGrad = textureGradOffset(layerMap, linearSampler, input.uvLayer, input.ddx, input.ddy, input.offset);
                vec4 fetched = texelFetchOffset(colorMap, input.pixel, int(input.lod), input.offset);
                vec4 fetchedLayer = texelFetchOffset(layerMap, input.pixelLayer, int(input.lod), input.offset);
                return plain + lodSample + gradSample + arrayPlain + arrayLod + arrayGrad + fetched + fetchedLayer;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler2DArray layerMap;" in generated_code
    assert (
        generated_code.count(
            glsl_dynamic_texel_offset_diagnostic(
                "texture offset", "textureOffset", "vec4(0.0)"
            )
        )
        == 2
    )
    assert (
        generated_code.count(
            glsl_dynamic_texel_offset_diagnostic(
                "texture offset", "textureLodOffset", "vec4(0.0)"
            )
        )
        == 2
    )
    assert (
        generated_code.count(
            glsl_dynamic_texel_offset_diagnostic(
                "texture offset", "textureGradOffset", "vec4(0.0)"
            )
        )
        == 2
    )
    assert (
        generated_code.count(
            glsl_dynamic_texel_offset_diagnostic(
                "texel fetch offset", "texelFetchOffset", "vec4(0.0)"
            )
        )
        == 2
    )
    assert "linearSampler" not in generated_code
    assert "input.uv" not in generated_code
    assert "input.pixel" not in generated_code
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code
    assert "texelFetchOffset(" not in generated_code


def test_opengl_projected_texture_variants_filter_sampler_arguments():
    shader = """
    shader TextureProjectionVariants {
        sampler2D colorMap;
        sampler3D volumeMap;
        sampler linearSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 xyzq @ TEXCOORD2;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        vec4 projectedOps(
            sampler2D tex,
            sampler3D volume,
            sampler s,
            vec3 uvq,
            vec4 uvqw,
            vec4 xyzq,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec4 projected = textureProj(tex, s, uvq);
            vec4 projectedBias = textureProj(tex, s, uvqw, 0.25);
            vec4 volumeProjected = textureProj(volume, s, xyzq);
            vec4 projectedOffset = textureProjOffset(tex, s, uvq, offset);
            vec4 projectedOffsetBias = textureProjOffset(tex, s, uvq, offset, 0.5);
            vec4 projectedLod = textureProjLod(tex, s, uvq, 2.0);
            vec4 projectedLodOffset = textureProjLodOffset(tex, s, uvq, 3.0, offset);
            vec4 projectedGrad = textureProjGrad(tex, s, uvq, ddx, ddy);
            vec4 projectedGradOffset = textureProjGradOffset(tex, s, uvq, ddx, ddy, offset);
            return projected
                + projectedBias
                + volumeProjected
                + projectedOffset
                + projectedOffsetBias
                + projectedLod
                + projectedLodOffset
                + projectedGrad
                + projectedGradOffset;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return projectedOps(
                    colorMap,
                    volumeMap,
                    linearSampler,
                    input.uvq,
                    input.uvqw,
                    input.xyzq,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "vec4 projectedOps(sampler2D tex, sampler3D volume, vec3 uvq, vec4 uvqw, vec4 xyzq, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert "textureProj(tex, uvq)" in generated_code
    assert "textureProj(tex, uvqw, 0.25)" in generated_code
    assert "textureProj(volume, xyzq)" in generated_code
    assert "textureProjLod(tex, uvq, 2.0)" in generated_code
    assert "textureProjGrad(tex, uvq, ddx, ddy)" in generated_code
    assert (
        generated_code.count(
            glsl_dynamic_texel_offset_diagnostic(
                "projected texture", "textureProjOffset", "vec4(0.0)"
            )
        )
        == 2
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjLodOffset", "vec4(0.0)"
        )
        in generated_code
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjGradOffset", "vec4(0.0)"
        )
        in generated_code
    )
    assert (
        "projectedOps(colorMap, volumeMap, uvq, uvqw, xyzq, ddx, ddy, offset)"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert ", s, uvq" not in generated_code
    assert ", s, uvqw" not in generated_code
    assert ", s, xyzq" not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_opengl_texture_offset_non_const_extension_allows_dynamic_offsets():
    shader = """
    #version 450 core
    #extension GL_EXT_texture_offset_non_const : enable

    shader NonConstTextureOffset {
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvq @ TEXCOORD1;
            ivec2 offset @ TEXCOORD2;
            int lod @ TEXCOORD3;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 sampled = textureOffset(colorMap, linearSampler, input.uv, input.offset);
                vec4 fetched = texelFetchOffset(colorMap, ivec2(input.uv), input.lod, input.offset);
                vec4 projected = textureProjOffset(colorMap, linearSampler, input.uvq, input.offset);
                return sampled + fetched + projected;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "#extension GL_EXT_texture_offset_non_const : enable" in generated_code
    assert "vec4 sampled = textureOffset(colorMap, uv, offset);" in generated_code
    assert (
        "vec4 fetched = texelFetchOffset(colorMap, ivec2(uv), lod, offset);"
        in generated_code
    )
    assert (
        "vec4 projected = textureProjOffset(colorMap, uvq, offset);" in generated_code
    )
    assert "texel offsets must be compile-time integer constants" not in generated_code
    assert "linearSampler" not in generated_code


def test_opengl_direct_projected_texture_stage_input_members():
    shader = """
    shader DirectProjectedTexture {
        sampler2D colorMap;
        sampler3D volumeMap;
        sampler linearSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 xyzq @ TEXCOORD2;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 projected = textureProj(colorMap, linearSampler, input.uvq);
                vec4 projectedBias = textureProj(colorMap, linearSampler, input.uvqw, 0.25);
                vec4 volumeProjected = textureProj(volumeMap, linearSampler, input.xyzq);
                vec4 projectedOffset = textureProjOffset(colorMap, linearSampler, input.uvq, input.offset);
                vec4 projectedLod = textureProjLod(colorMap, linearSampler, input.uvq, input.lod);
                vec4 projectedLodOffset = textureProjLodOffset(colorMap, linearSampler, input.uvq, input.lod, input.offset);
                vec4 projectedGrad = textureProjGrad(colorMap, linearSampler, input.uvq, input.ddx, input.ddy);
                vec4 projectedGradOffset = textureProjGradOffset(colorMap, linearSampler, input.uvq, input.ddx, input.ddy, input.offset);
                return projected
                    + projectedBias
                    + volumeProjected
                    + projectedOffset
                    + projectedLod
                    + projectedLodOffset
                    + projectedGrad
                    + projectedGradOffset;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler3D volumeMap;" in generated_code
    assert "vec4 projected = textureProj(colorMap, uvq);" in generated_code
    assert "vec4 projectedBias = textureProj(colorMap, uvqw, 0.25);" in generated_code
    assert "vec4 volumeProjected = textureProj(volumeMap, xyzq);" in generated_code
    assert "vec4 projectedLod = textureProjLod(colorMap, uvq, lod);" in generated_code
    assert (
        "vec4 projectedGrad = textureProjGrad(colorMap, uvq, ddx, ddy);"
        in generated_code
    )
    assert (
        "vec4 projectedOffset = "
        + glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjOffset", "vec4(0.0)"
        )
        + ";"
        in generated_code
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjLodOffset", "vec4(0.0)"
        )
        in generated_code
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjGradOffset", "vec4(0.0)"
        )
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert ", input." not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_opengl_direct_projected_array_texture_stage_input_members():
    shader = """
    shader DirectProjectedArrayTexture {
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec4 uvLayerQ @ TEXCOORD0;
            float lod;
            vec2 ddx @ TEXCOORD1;
            vec2 ddy @ TEXCOORD2;
            ivec2 offset @ TEXCOORD3;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 projected = textureProj(layerMap, linearSampler, input.uvLayerQ);
                vec4 projectedOffset = textureProjOffset(layerMap, linearSampler, input.uvLayerQ, input.offset);
                vec4 projectedLod = textureProjLod(layerMap, linearSampler, input.uvLayerQ, input.lod);
                vec4 projectedLodOffset = textureProjLodOffset(layerMap, linearSampler, input.uvLayerQ, input.lod, input.offset);
                vec4 projectedGrad = textureProjGrad(layerMap, linearSampler, input.uvLayerQ, input.ddx, input.ddy);
                vec4 projectedGradOffset = textureProjGradOffset(layerMap, linearSampler, input.uvLayerQ, input.ddx, input.ddy, input.offset);
                return projected + projectedOffset + projectedLod + projectedLodOffset + projectedGrad + projectedGradOffset;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DArray layerMap;" in generated_code
    assert "vec4 projected = textureProj(layerMap, uvLayerQ);" in generated_code
    assert (
        "vec4 projectedLod = textureProjLod(layerMap, uvLayerQ, lod);" in generated_code
    )
    assert (
        "vec4 projectedGrad = textureProjGrad(layerMap, uvLayerQ, ddx, ddy);"
        in generated_code
    )
    assert (
        "vec4 projectedOffset = "
        + glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjOffset", "vec4(0.0)"
        )
        + ";"
        in generated_code
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjLodOffset", "vec4(0.0)"
        )
        in generated_code
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjGradOffset", "vec4(0.0)"
        )
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert ", input." not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_opengl_implicit_projected_array_texture_stage_input_members_strip_samplers():
    shader = """
    shader DirectImplicitProjectedArrayTexture {
        sampler2DArray layerMap;

        struct FSInput {
            vec4 uvLayerQ @ TEXCOORD0;
            float lod;
            vec2 ddx @ TEXCOORD1;
            vec2 ddy @ TEXCOORD2;
            ivec2 offset @ TEXCOORD3;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 projected = textureProj(layerMap, input.uvLayerQ);
                vec4 projectedOffset = textureProjOffset(layerMap, input.uvLayerQ, input.offset);
                vec4 projectedLod = textureProjLod(layerMap, input.uvLayerQ, input.lod);
                vec4 projectedLodOffset = textureProjLodOffset(layerMap, input.uvLayerQ, input.lod, input.offset);
                vec4 projectedGrad = textureProjGrad(layerMap, input.uvLayerQ, input.ddx, input.ddy);
                vec4 projectedGradOffset = textureProjGradOffset(layerMap, input.uvLayerQ, input.ddx, input.ddy, input.offset);
                return projected + projectedOffset + projectedLod + projectedLodOffset + projectedGrad + projectedGradOffset;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DArray layerMap;" in generated_code
    assert "vec4 projected = textureProj(layerMap, uvLayerQ);" in generated_code
    assert (
        "vec4 projectedLod = textureProjLod(layerMap, uvLayerQ, lod);" in generated_code
    )
    assert (
        "vec4 projectedGrad = textureProjGrad(layerMap, uvLayerQ, ddx, ddy);"
        in generated_code
    )
    assert (
        "vec4 projectedOffset = "
        + glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjOffset", "vec4(0.0)"
        )
        + ";"
        in generated_code
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjLodOffset", "vec4(0.0)"
        )
        in generated_code
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjGradOffset", "vec4(0.0)"
        )
        in generated_code
    )
    assert "input." not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_opengl_projected_array_texture_resource_arrays_strip_samplers():
    shader = """
    shader ProjectedArrayTextureResourceArrays {
        sampler2DArray layerMaps[4];
        sampler linearSamplers[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec4 uvLayerQ @ TEXCOORD1;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 projectedLeaf(
            sampler2DArray maps[],
            sampler samplers[],
            int layer,
            vec4 uvLayerQ,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec4 fixedProj = textureProj(maps[2], samplers[2], uvLayerQ);
            vec4 dynamicOffset = textureProjOffset(maps[layer], samplers[layer], uvLayerQ, offset);
            vec4 fixedLod = textureProjLod(maps[1], samplers[1], uvLayerQ, lod);
            vec4 dynamicLodOffset = textureProjLodOffset(maps[layer], samplers[layer], uvLayerQ, lod, offset);
            vec4 fixedGrad = textureProjGrad(maps[3], samplers[3], uvLayerQ, ddx, ddy);
            vec4 dynamicGradOffset = textureProjGradOffset(maps[layer], samplers[layer], uvLayerQ, ddx, ddy, offset);
            return fixedProj + dynamicOffset + fixedLod + dynamicLodOffset + fixedGrad + dynamicGradOffset;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return projectedLeaf(
                    layerMaps,
                    linearSamplers,
                    input.layer,
                    input.uvLayerQ,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DArray layerMaps[4];" in generated_code
    assert (
        "vec4 projectedLeaf(sampler2DArray maps[4], int layer, vec4 uvLayerQ, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert "vec4 fixedProj = textureProj(maps[2], uvLayerQ);" in generated_code
    assert "vec4 fixedLod = textureProjLod(maps[1], uvLayerQ, lod);" in generated_code
    assert (
        "vec4 fixedGrad = textureProjGrad(maps[3], uvLayerQ, ddx, ddy);"
        in generated_code
    )
    assert (
        "vec4 dynamicOffset = "
        + glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjOffset", "vec4(0.0)"
        )
        + ";"
        in generated_code
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjLodOffset", "vec4(0.0)"
        )
        in generated_code
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjGradOffset", "vec4(0.0)"
        )
        in generated_code
    )
    assert (
        "projectedLeaf(layerMaps, layer, uvLayerQ, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert "linearSamplers" not in generated_code
    assert "samplers" not in generated_code
    assert "input." not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_opengl_implicit_projected_array_texture_resource_arrays_strip_samplers():
    shader = """
    shader ImplicitProjectedArrayTextureResourceArrays {
        sampler2DArray layerMaps[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec4 uvLayerQ @ TEXCOORD1;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 projectedLeaf(
            sampler2DArray maps[],
            int layer,
            vec4 uvLayerQ,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec4 fixedProj = textureProj(maps[2], uvLayerQ);
            vec4 dynamicOffset = textureProjOffset(maps[layer], uvLayerQ, offset);
            vec4 fixedLod = textureProjLod(maps[1], uvLayerQ, lod);
            vec4 dynamicLodOffset = textureProjLodOffset(maps[layer], uvLayerQ, lod, offset);
            vec4 fixedGrad = textureProjGrad(maps[3], uvLayerQ, ddx, ddy);
            vec4 dynamicGradOffset = textureProjGradOffset(maps[layer], uvLayerQ, ddx, ddy, offset);
            return fixedProj + dynamicOffset + fixedLod + dynamicLodOffset + fixedGrad + dynamicGradOffset;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return projectedLeaf(
                    layerMaps,
                    input.layer,
                    input.uvLayerQ,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DArray layerMaps[4];" in generated_code
    assert (
        "vec4 projectedLeaf(sampler2DArray maps[4], int layer, vec4 uvLayerQ, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert "vec4 fixedProj = textureProj(maps[2], uvLayerQ);" in generated_code
    assert "vec4 fixedLod = textureProjLod(maps[1], uvLayerQ, lod);" in generated_code
    assert (
        "vec4 fixedGrad = textureProjGrad(maps[3], uvLayerQ, ddx, ddy);"
        in generated_code
    )
    assert (
        "vec4 dynamicOffset = "
        + glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjOffset", "vec4(0.0)"
        )
        + ";"
        in generated_code
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjLodOffset", "vec4(0.0)"
        )
        in generated_code
    )
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "projected texture", "textureProjGradOffset", "vec4(0.0)"
        )
        in generated_code
    )
    assert (
        "projectedLeaf(layerMaps, layer, uvLayerQ, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert "input." not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_opengl_implicit_projected_stage_input_members_strip_samplers():
    shader = """
    shader DirectImplicitProjection {
        sampler2D colorMap;
        sampler3D volumeMap;
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 xyzq @ TEXCOORD2;
            vec4 uvLayerQ @ TEXCOORD3;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD4;
            vec2 ddy @ TEXCOORD5;
            ivec2 offset @ TEXCOORD6;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 color = textureProj(colorMap, input.uvq);
                vec4 colorLod = textureProjLod(colorMap, input.uvqw, input.lod);
                vec4 volume = textureProj(volumeMap, input.xyzq);
                float shadow = textureCompareProj(shadowMap, input.uvq, input.depth);
                float shadowGrad = textureCompareProjGrad(shadowArray, input.uvLayerQ, input.depth, input.ddx, input.ddy);
                return color + colorLod + volume + vec4(shadow + shadowGrad);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler3D volumeMap;" in generated_code
    assert "layout(binding = 2) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "layout(binding = 3) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert "vec4 color = textureProj(colorMap, uvq);" in generated_code
    assert "vec4 colorLod = textureProjLod(colorMap, uvqw, lod);" in generated_code
    assert "vec4 volume = textureProj(volumeMap, xyzq);" in generated_code
    assert (
        "float shadow = texture(shadowMap, vec3(uvq.xy / uvq.z, depth));"
        in generated_code
    )
    assert (
        "float shadowGrad = textureGrad(shadowArray, vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), ddx, ddy);"
        in generated_code
    )
    assert "unsupported GLSL texture compare" not in generated_code
    assert "textureCompareProj(" not in generated_code
    assert "input." not in generated_code


def test_opengl_projected_shadow_compare_variants_filter_sampler_arguments():
    shader = """
    shader ProjectedShadowCompareVariants {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 uvLayerQ @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        float projectedShadow(
            sampler2DShadow tex,
            sampler s,
            vec3 uvq,
            vec4 uvqw,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, s, uvq, depth);
            float projectedW = textureCompareProj(tex, s, uvqw, depth);
            float offsetProjected = textureCompareProjOffset(tex, s, uvq, depth, offset);
            float lodProjected = textureCompareProjLod(tex, s, uvq, depth, lod);
            float lodOffsetProjected = textureCompareProjLodOffset(tex, s, uvq, depth, lod, offset);
            float gradProjected = textureCompareProjGrad(tex, s, uvq, depth, ddx, ddy);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, s, uvq, depth, ddx, ddy, offset);
            return projected + projectedW + offsetProjected + lodProjected + lodOffsetProjected + gradProjected + gradOffsetProjected;
        }

        float projectedArrayShadow(
            sampler2DArrayShadow tex,
            sampler s,
            vec4 uvLayerQ,
            float depth,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, s, uvLayerQ, depth);
            float offsetProjected = textureCompareProjOffset(tex, s, uvLayerQ, depth, offset);
            float gradProjected = textureCompareProjGrad(tex, s, uvLayerQ, depth, ddx, ddy);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, s, uvLayerQ, depth, ddx, ddy, offset);
            return projected + offsetProjected + gradProjected + gradOffsetProjected;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float shadow = projectedShadow(
                    shadowMap,
                    compareSampler,
                    input.uvq,
                    input.uvqw,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float arrayShadow = projectedArrayShadow(
                    shadowArray,
                    compareSampler,
                    input.uvLayerQ,
                    input.depth,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                return vec4(shadow + arrayShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "layout(binding = 1) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "float projectedShadow(sampler2DShadow tex, vec3 uvq, vec4 uvqw, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "float projected = texture(tex, vec3(uvq.xy / uvq.z, depth));" in generated_code
    )
    assert (
        "float projectedW = texture(tex, vec3(uvqw.xy / uvqw.w, depth));"
        in generated_code
    )
    assert (
        "float offsetProjected = /* unsupported GLSL texture compare: "
        "textureCompareProjOffset texel offsets must be compile-time integer constants "
        "*/ 0.0;" in generated_code
    )
    assert (
        "float lodProjected = textureLod(tex, vec3(uvq.xy / uvq.z, depth), lod);"
        in generated_code
    )
    assert (
        "float lodOffsetProjected = "
        + glsl_dynamic_texel_offset_diagnostic(
            "texture compare", "textureCompareProjLodOffset", "0.0"
        )
        + ";"
        in generated_code
    )
    assert (
        "float gradProjected = textureGrad(tex, vec3(uvq.xy / uvq.z, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetProjected = "
        + glsl_dynamic_texel_offset_diagnostic(
            "texture compare", "textureCompareProjGradOffset", "0.0"
        )
        + ";"
        in generated_code
    )
    assert (
        "projectedShadow(shadowMap, uvq, uvqw, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "float projectedArrayShadow(sampler2DArrayShadow tex, vec4 uvLayerQ, float depth, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "float projected = texture(tex, vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth));"
        in generated_code
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture compare: textureCompareProjOffset texel offsets must be compile-time integer constants */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            glsl_dynamic_texel_offset_diagnostic(
                "texture compare", "textureCompareProjGradOffset", "0.0"
            )
        )
        == 2
    )
    assert (
        "float gradProjected = textureGrad(tex, vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetProjected = "
        + glsl_dynamic_texel_offset_diagnostic(
            "texture compare", "textureCompareProjGradOffset", "0.0"
        )
        + ";"
        in generated_code
    )
    assert (
        "projectedArrayShadow(shadowArray, uvLayerQ, depth, ddx, ddy, offset)"
        in generated_code
    )
    assert "compareSampler" not in generated_code
    assert "textureCompareProj(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code


def test_opengl_direct_projected_shadow_compare_stage_input_members():
    shader = """
    shader DirectProjectedShadowCompare {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 uvLayerQ @ TEXCOORD2;
            float depth;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float planar = textureCompareProj(shadowMap, compareSampler, input.uvq, input.depth);
                float planarOffset = textureCompareProjOffset(shadowMap, compareSampler, input.uvqw, input.depth, input.offset);
                float arrayGrad = textureCompareProjGrad(shadowArray, compareSampler, input.uvLayerQ, input.depth, input.ddx, input.ddy);
                return vec4(planar + planarOffset + arrayGrad);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "layout(binding = 1) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "float planar = texture(shadowMap, vec3(uvq.xy / uvq.z, depth));"
        in generated_code
    )
    assert (
        "float planarOffset = /* unsupported GLSL texture compare: "
        "textureCompareProjOffset texel offsets must be compile-time integer constants "
        "*/ 0.0;" in generated_code
    )
    assert (
        "float arrayGrad = textureGrad(shadowArray, vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), ddx, ddy);"
        in generated_code
    )
    assert "compareSampler" not in generated_code
    assert "textureCompareProjOffset(" not in generated_code
    assert "textureCompareProj(" not in generated_code


def test_opengl_projected_shadow_compare_resource_arrays_strip_sampler_arguments():
    shader = """
    shader ProjectedShadowResourceArrays {
        sampler2DShadow shadowMaps[4];
        sampler2DArrayShadow shadowArrays[4];
        sampler shadowSamplers[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec3 uvq @ TEXCOORD1;
            vec4 uvLayerQ @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        float projectedLeaf(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            sampler shadowSamplers[],
            int layer,
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float planar = textureCompareProj(shadowMaps[layer], shadowSamplers[layer], uvq, depth);
            float planarOffset = textureCompareProjOffset(shadowMaps[1], shadowSamplers[1], uvq, depth, offset);
            float planarLod = textureCompareProjLod(shadowMaps[2], shadowSamplers[2], uvq, depth, lod);
            float planarGradOffset = textureCompareProjGradOffset(shadowMaps[layer], shadowSamplers[layer], uvq, depth, ddx, ddy, offset);
            float arrayProjected = textureCompareProj(shadowArrays[2], shadowSamplers[2], uvLayerQ, depth);
            float arrayOffset = textureCompareProjOffset(shadowArrays[layer], shadowSamplers[layer], uvLayerQ, depth, offset);
            float arrayGrad = textureCompareProjGrad(shadowArrays[1], shadowSamplers[1], uvLayerQ, depth, ddx, ddy);
            return planar + planarOffset + planarLod + planarGradOffset + arrayProjected + arrayOffset + arrayGrad;
        }

        float projectedWrapper(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            sampler shadowSamplers[],
            int layer,
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            return projectedLeaf(shadowMaps, shadowArrays, shadowSamplers, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return projectedWrapper(
                    shadowMaps,
                    shadowArrays,
                    shadowSamplers,
                    input.layer,
                    input.uvq,
                    input.uvLayerQ,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert (
        "layout(binding = 4) uniform sampler2DArrayShadow shadowArrays[4];"
        in generated_code
    )
    assert (
        "float projectedLeaf(sampler2DShadow shadowMaps[4], sampler2DArrayShadow shadowArrays[4], int layer, vec3 uvq, vec4 uvLayerQ, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "float planar = texture(shadowMaps[layer], vec3(uvq.xy / uvq.z, depth));"
        in generated_code
    )
    assert (
        "float planarOffset = /* unsupported GLSL texture compare: "
        "textureCompareProjOffset texel offsets must be compile-time integer constants "
        "*/ 0.0;" in generated_code
    )
    assert (
        "float planarLod = textureLod(shadowMaps[2], vec3(uvq.xy / uvq.z, depth), lod);"
        in generated_code
    )
    assert (
        "float planarGradOffset = "
        + glsl_dynamic_texel_offset_diagnostic(
            "texture compare", "textureCompareProjGradOffset", "0.0"
        )
        + ";"
        in generated_code
    )
    assert (
        "float arrayProjected = texture(shadowArrays[2], vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth));"
        in generated_code
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture compare: textureCompareProjOffset texel offsets must be compile-time integer constants */ 0.0"
        )
        == 2
    )
    assert (
        "float arrayGrad = textureGrad(shadowArrays[1], vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float projectedWrapper(sampler2DShadow shadowMaps[4], sampler2DArrayShadow shadowArrays[4], int layer, vec3 uvq, vec4 uvLayerQ, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "projectedLeaf(shadowMaps, shadowArrays, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedWrapper(shadowMaps, shadowArrays, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[" not in generated_code
    assert "textureCompareProj(" not in generated_code
    assert "textureGradOffset(" not in generated_code


def test_opengl_implicit_projected_shadow_compare_resource_arrays_strip_samplers():
    shader = """
    shader ImplicitProjectedShadowResourceArrays {
        sampler2DShadow shadowMaps[4];
        sampler2DArrayShadow shadowArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec3 uvq @ TEXCOORD1;
            vec4 uvLayerQ @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        float projectedLeaf(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            int layer,
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float planar = textureCompareProj(shadowMaps[layer], uvq, depth);
            float planarOffset = textureCompareProjOffset(shadowMaps[1], uvq, depth, offset);
            float planarLod = textureCompareProjLod(shadowMaps[2], uvq, depth, lod);
            float planarGradOffset = textureCompareProjGradOffset(shadowMaps[layer], uvq, depth, ddx, ddy, offset);
            float arrayProjected = textureCompareProj(shadowArrays[2], uvLayerQ, depth);
            float arrayOffset = textureCompareProjOffset(shadowArrays[layer], uvLayerQ, depth, offset);
            float arrayGrad = textureCompareProjGrad(shadowArrays[1], uvLayerQ, depth, ddx, ddy);
            return planar + planarOffset + planarLod + planarGradOffset + arrayProjected + arrayOffset + arrayGrad;
        }

        float projectedWrapper(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            int layer,
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            return projectedLeaf(shadowMaps, shadowArrays, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return projectedWrapper(
                    shadowMaps,
                    shadowArrays,
                    input.layer,
                    input.uvq,
                    input.uvLayerQ,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert (
        "layout(binding = 4) uniform sampler2DArrayShadow shadowArrays[4];"
        in generated_code
    )
    assert (
        "float projectedLeaf(sampler2DShadow shadowMaps[4], sampler2DArrayShadow shadowArrays[4], int layer, vec3 uvq, vec4 uvLayerQ, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "float planar = texture(shadowMaps[layer], vec3(uvq.xy / uvq.z, depth));"
        in generated_code
    )
    assert (
        "float planarOffset = /* unsupported GLSL texture compare: "
        "textureCompareProjOffset texel offsets must be compile-time integer constants "
        "*/ 0.0;" in generated_code
    )
    assert (
        "float planarLod = textureLod(shadowMaps[2], vec3(uvq.xy / uvq.z, depth), lod);"
        in generated_code
    )
    assert (
        "float planarGradOffset = "
        + glsl_dynamic_texel_offset_diagnostic(
            "texture compare", "textureCompareProjGradOffset", "0.0"
        )
        + ";"
        in generated_code
    )
    assert (
        "float arrayProjected = texture(shadowArrays[2], vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth));"
        in generated_code
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture compare: textureCompareProjOffset texel offsets must be compile-time integer constants */ 0.0"
        )
        == 2
    )
    assert (
        "float arrayGrad = textureGrad(shadowArrays[1], vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float projectedWrapper(sampler2DShadow shadowMaps[4], sampler2DArrayShadow shadowArrays[4], int layer, vec3 uvq, vec4 uvLayerQ, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "projectedLeaf(shadowMaps, shadowArrays, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedWrapper(shadowMaps, shadowArrays, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert "sampler shadowMapsSampler" not in generated_code
    assert "sampler shadowArraysSampler" not in generated_code
    assert "textureCompareProj(" not in generated_code
    assert "textureGradOffset(" not in generated_code


def test_opengl_unsized_projected_shadow_compare_arrays_infer_transitive_constant_size():
    shader = """
    shader UnsizedProjectedShadowResources {
        const int LAYER = 4;
        sampler2DShadow shadowMaps[];
        sampler2DArrayShadow shadowArrays[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvLayerQ @ TEXCOORD1;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        float shadowDeep(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            sampler shadowSamplers[],
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float planarHigh = textureCompareProj(shadowMaps[LAYER], shadowSamplers[LAYER], uvq, depth);
            float planarLow = textureCompareProjOffset(shadowMaps[1], shadowSamplers[1], uvq, depth, offset);
            float arrayHigh = textureCompareProjGrad(shadowArrays[2 * 2], shadowSamplers[2 * 2], uvLayerQ, depth, ddx, ddy);
            float arrayLow = textureCompareProjOffset(shadowArrays[3], shadowSamplers[3], uvLayerQ, depth, offset);
            return planarHigh + planarLow + arrayHigh + arrayLow;
        }

        float shadowMid(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            sampler shadowSamplers[],
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            return shadowDeep(shadowMaps, shadowArrays, shadowSamplers, uvq, uvLayerQ, depth, lod, ddx, ddy, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float arrayShadow = shadowMid(
                    shadowMaps,
                    shadowArrays,
                    shadowSamplers,
                    input.uvq,
                    input.uvLayerQ,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uvq.xy, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[5];" in generated_code
    )
    assert (
        "layout(binding = 5) uniform sampler2DArrayShadow shadowArrays[5];"
        in generated_code
    )
    assert "layout(binding = 10) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowDeep(sampler2DShadow shadowMaps[5], sampler2DArrayShadow shadowArrays[5], "
        "vec3 uvq, vec4 uvLayerQ, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
    ) in generated_code
    assert (
        "float shadowMid(sampler2DShadow shadowMaps[5], sampler2DArrayShadow shadowArrays[5], "
        "vec3 uvq, vec4 uvLayerQ, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
    ) in generated_code
    assert (
        "float planarHigh = texture(shadowMaps[LAYER], vec3(uvq.xy / uvq.z, depth));"
        in generated_code
    )
    assert (
        "float planarLow = /* unsupported GLSL texture compare: "
        "textureCompareProjOffset texel offsets must be compile-time integer constants "
        "*/ 0.0;" in generated_code
    )
    assert (
        "float arrayHigh = textureGrad(shadowArrays[(2 * 2)], "
        "vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), ddx, ddy);"
    ) in generated_code
    assert (
        generated_code.count(
            "/* unsupported GLSL texture compare: textureCompareProjOffset texel offsets must be compile-time integer constants */ 0.0"
        )
        == 2
    )
    assert (
        "shadowDeep(shadowMaps, shadowArrays, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "shadowMid(shadowMaps, shadowArrays, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "float singleShadow = texture(afterShadow, vec3(uvq.xy, depth));"
        in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers" not in generated_code
    assert "afterSampler" not in generated_code
    assert "textureCompareProj(" not in generated_code


def test_opengl_projected_array_shadow_lod_offset_reports_unsupported():
    shader = """
    shader ProjectedArrayShadowLodOffsetUnsupported {
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec4 uvLayerQ @ TEXCOORD0;
            float depth;
            float lod;
            ivec2 offset @ TEXCOORD1;
        };

        float projectedArrayShadow(
            sampler2DArrayShadow tex,
            sampler s,
            vec4 uvLayerQ,
            float depth,
            float lod,
            ivec2 offset
        ) {
            float rejected = textureCompareProjLodOffset(tex, s, uvLayerQ, depth, lod, offset);
            return rejected;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float shadow = projectedArrayShadow(
                    shadowArray,
                    compareSampler,
                    input.uvLayerQ,
                    input.depth,
                    input.lod,
                    input.offset
                );
                return vec4(shadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "float rejected = /* unsupported GLSL texture compare: textureCompareProjLodOffset projected explicit LOD is not supported for sampler2DArrayShadow */ 0.0;"
        in generated_code
    )
    assert "textureLodOffset(" not in generated_code


def test_opengl_projected_cube_shadow_compare_lowers_supported_basic_form():
    shader = """
    shader ProjectedCubeShadowDiagnostics {
        samplerCubeShadow cubeMap;
        samplerCubeArrayShadow cubeArray;
        sampler compareSampler;

        struct FSInput {
            vec4 cubeProj @ TEXCOORD0;
            vec4 cubeLayerProj @ TEXCOORD1;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        float cubeProjected(
            samplerCubeShadow tex,
            sampler s,
            vec4 cubeProj,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, s, cubeProj, depth);
            float offsetProjected = textureCompareProjOffset(tex, s, cubeProj, depth, offset);
            float lodOffsetProjected = textureCompareProjLodOffset(tex, s, cubeProj, depth, lod, offset);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, s, cubeProj, depth, ddx, ddy, offset);
            return projected + offsetProjected + lodOffsetProjected + gradOffsetProjected;
        }

        float cubeArrayProjected(
            samplerCubeArrayShadow tex,
            sampler s,
            vec4 cubeLayerProj,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, s, cubeLayerProj, depth);
            float offsetProjected = textureCompareProjOffset(tex, s, cubeLayerProj, depth, offset);
            float lodOffsetProjected = textureCompareProjLodOffset(tex, s, cubeLayerProj, depth, lod, offset);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, s, cubeLayerProj, depth, ddx, ddy, offset);
            return projected + offsetProjected + lodOffsetProjected + gradOffsetProjected;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return cubeProjected(
                    cubeMap,
                    compareSampler,
                    input.cubeProj,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + cubeArrayProjected(
                    cubeArray,
                    compareSampler,
                    input.cubeLayerProj,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(binding = 0) uniform samplerCubeShadow cubeMap;" in generated_code
    assert (
        "layout(binding = 1) uniform samplerCubeArrayShadow cubeArray;"
        in generated_code
    )
    assert (
        "float cubeProjected(samplerCubeShadow tex, vec4 cubeProj, float depth, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "float cubeArrayProjected(samplerCubeArrayShadow tex, vec4 cubeLayerProj, float depth, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "float projected = texture(tex, vec4(cubeProj.xyz / cubeProj.w, depth));"
        in generated_code
    )
    expected_cube_diagnostics = {
        "textureCompareProjOffset": "offsets require 2D or 2D-array shadow samplers",
        "textureCompareProjLodOffset": (
            "explicit LOD offsets require 2D shadow samplers"
        ),
        "textureCompareProjGradOffset": (
            "explicit gradient offsets require 2D or 2D-array shadow samplers"
        ),
    }
    for func_name, reason in expected_cube_diagnostics.items():
        assert (
            generated_code.count(
                f"/* unsupported GLSL texture compare: {func_name} {reason} */ 0.0"
            )
            == 1
        )
    coordinate_diagnostic_counts = {
        "textureCompareProj": 1,
        "textureCompareProjOffset": 1,
        "textureCompareProjLodOffset": 1,
        "textureCompareProjGradOffset": 1,
    }
    for func_name, count in coordinate_diagnostic_counts.items():
        assert (
            generated_code.count(
                f"/* unsupported GLSL texture compare: {func_name} requires sampler2DShadow vec3/vec4 or sampler2DArrayShadow vec4 projection coordinates */ 0.0"
            )
            == count
        )
    assert "compareSampler" not in generated_code
    assert "textureCompareProj(" not in generated_code
    assert "textureCompareProjOffset(" not in generated_code
    assert "textureCompareProjLodOffset(" not in generated_code
    assert "textureCompareProjGradOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code


def test_opengl_direct_projected_cube_texture_lowers_supported_color_forms():
    shader = """
    shader DirectProjectedCubeDiagnostics {
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler linearSampler;

        struct FSInput {
            vec4 cubeProj @ TEXCOORD0;
            vec4 cubeArrayProj @ TEXCOORD1;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 cubeProjected = textureProj(cubeMap, linearSampler, input.cubeProj);
                vec4 cubeLod = textureProjLod(cubeMap, linearSampler, input.cubeProj, input.lod);
                vec4 cubeGrad = textureProjGrad(cubeMap, linearSampler, input.cubeProj, input.ddx, input.ddy);
                vec4 cubeLodOffset = textureProjLodOffset(cubeMap, linearSampler, input.cubeProj, input.lod, input.offset);
                vec4 cubeGradOffset = textureProjGradOffset(cubeMap, linearSampler, input.cubeProj, input.ddx, input.ddy, input.offset);
                vec4 cubeArrayProjected = textureProj(cubeArray, linearSampler, input.cubeArrayProj);
                vec4 cubeArrayLodOffset = textureProjLodOffset(cubeArray, linearSampler, input.cubeArrayProj, input.lod, input.offset);
                vec4 cubeArrayGradOffset = textureProjGradOffset(cubeArray, linearSampler, input.cubeArrayProj, input.ddx, input.ddy, input.offset);
                vec4 implicitCube = textureProj(cubeMap, input.cubeProj);
                vec4 implicitCubeGrad = textureProjGrad(cubeMap, input.cubeProj, input.ddx, input.ddy);
                vec4 implicitCubeArray = textureProjLod(cubeArray, input.cubeArrayProj, input.lod);
                return cubeProjected + cubeLod + cubeGrad + cubeLodOffset + cubeGradOffset + cubeArrayProjected + cubeArrayLodOffset + cubeArrayGradOffset + implicitCube + implicitCubeGrad + implicitCubeArray;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform samplerCube cubeMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCubeArray cubeArray;" in generated_code
    assert (
        "vec4 cubeProjected = texture(cubeMap, cubeProj.xyz / cubeProj.w);"
        in generated_code
    )
    assert (
        "vec4 cubeLod = textureLod(cubeMap, cubeProj.xyz / cubeProj.w, lod);"
        in generated_code
    )
    assert (
        "vec4 cubeGrad = textureGrad(cubeMap, cubeProj.xyz / cubeProj.w, ddx, ddy);"
        in generated_code
    )
    assert (
        "vec4 implicitCube = texture(cubeMap, cubeProj.xyz / cubeProj.w);"
        in generated_code
    )
    assert (
        "vec4 implicitCubeGrad = textureGrad(cubeMap, cubeProj.xyz / cubeProj.w, ddx, ddy);"
        in generated_code
    )
    assert (
        "/* unsupported GLSL projected texture: textureProjLodOffset offsets require 1D, 2D, 2D-array, 3D, or planar shadow samplers */ vec4(0.0)"
        in generated_code
    )
    assert (
        "/* unsupported GLSL projected texture: textureProjGradOffset offsets require 1D, 2D, 2D-array, 3D, or planar shadow samplers */ vec4(0.0)"
        in generated_code
    )
    coordinate_diagnostic_counts = {
        "textureProj": 1,
        "textureProjLodOffset": 1,
        "textureProjGradOffset": 1,
        "textureProjLod": 1,
    }
    for func_name, count in coordinate_diagnostic_counts.items():
        assert (
            generated_code.count(
                f"/* unsupported GLSL projected texture: {func_name} requires 1D, 2D, 2D-array, or 3D projection coordinates */ vec4(0.0)"
            )
            == count
        )
    assert "linearSampler" not in generated_code
    assert "textureProj(cubeMap" not in generated_code
    assert "textureProj(cubeArray" not in generated_code


def test_opengl_projected_cube_texture_resource_arrays_lower_supported_color_forms():
    shader = """
    shader ProjectedCubeArrayDiagnostics {
        samplerCube cubeMaps[4];
        samplerCubeArray cubeArrays[4];
        samplerCubeShadow shadowCubes[4];
        samplerCubeArrayShadow shadowCubeArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec4 cubeProj @ TEXCOORD1;
            vec4 cubeArrayProj @ TEXCOORD2;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD3;
            vec3 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        vec4 projectedCube(
            samplerCube maps[],
            int layer,
            vec4 cubeProj,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            vec4 fixedProjected = textureProj(maps[2], cubeProj);
            vec4 dynamicLodProjected = textureProjLod(maps[layer], cubeProj, lod);
            vec4 dynamicGradOffsetProjected = textureProjGradOffset(maps[layer], cubeProj, ddx, ddy, offset);
            return fixedProjected + dynamicLodProjected + dynamicGradOffsetProjected;
        }

        vec4 projectedCubeArray(
            samplerCubeArray maps[],
            int layer,
            vec4 cubeArrayProj,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            vec4 fixedProjected = textureProj(maps[2], cubeArrayProj);
            vec4 dynamicLodProjected = textureProjLod(maps[layer], cubeArrayProj, lod);
            vec4 dynamicGradOffsetProjected = textureProjGradOffset(maps[layer], cubeArrayProj, ddx, ddy, offset);
            return fixedProjected + dynamicLodProjected + dynamicGradOffsetProjected;
        }

        vec4 projectedShadow(
            samplerCubeShadow maps[],
            int layer,
            vec4 cubeProj,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float fixedProjected = textureCompareProj(maps[2], cubeProj, depth);
            float dynamicLodProjected = textureCompareProjLod(maps[layer], cubeProj, depth, lod);
            float dynamicGradOffsetProjected = textureCompareProjGradOffset(maps[layer], cubeProj, depth, ddx, ddy, offset);
            return vec4(fixedProjected + dynamicLodProjected + dynamicGradOffsetProjected);
        }

        vec4 projectedShadowArray(
            samplerCubeArrayShadow maps[],
            int layer,
            vec4 cubeArrayProj,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float fixedProjected = textureCompareProj(maps[2], cubeArrayProj, depth);
            float dynamicLodProjected = textureCompareProjLod(maps[layer], cubeArrayProj, depth, lod);
            float dynamicGradOffsetProjected = textureCompareProjGradOffset(maps[layer], cubeArrayProj, depth, ddx, ddy, offset);
            return vec4(fixedProjected + dynamicLodProjected + dynamicGradOffsetProjected);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 globalProjected = textureProj(cubeMaps[input.layer], input.cubeProj);
                vec4 globalArrayProjected = textureProjLod(cubeArrays[2], input.cubeArrayProj, input.lod);
                float globalShadow = textureCompareProj(shadowCubes[input.layer], input.cubeProj, input.depth);
                float globalArrayShadow = textureCompareProjLod(shadowCubeArrays[2], input.cubeArrayProj, input.depth, input.lod);
                return globalProjected + globalArrayProjected + vec4(globalShadow + globalArrayShadow)
                    + projectedCube(
                        cubeMaps,
                        input.layer,
                        input.cubeProj,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    )
                    + projectedCubeArray(
                        cubeArrays,
                        input.layer,
                        input.cubeArrayProj,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    )
                    + projectedShadow(
                        shadowCubes,
                        input.layer,
                        input.cubeProj,
                        input.depth,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    )
                    + projectedShadowArray(
                        shadowCubeArrays,
                        input.layer,
                        input.cubeArrayProj,
                        input.depth,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform samplerCube cubeMaps[4];" in generated_code
    assert (
        "layout(binding = 4) uniform samplerCubeArray cubeArrays[4];" in generated_code
    )
    assert (
        "layout(binding = 8) uniform samplerCubeShadow shadowCubes[4];"
        in generated_code
    )
    assert (
        "layout(binding = 12) uniform samplerCubeArrayShadow shadowCubeArrays[4];"
        in generated_code
    )
    assert (
        "vec4 projectedCube(samplerCube maps[4], int layer, vec4 cubeProj, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "vec4 projectedCubeArray(samplerCubeArray maps[4], int layer, vec4 cubeArrayProj, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "vec4 projectedShadow(samplerCubeShadow maps[4], int layer, vec4 cubeProj, float depth, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "vec4 projectedShadowArray(samplerCubeArrayShadow maps[4], int layer, vec4 cubeArrayProj, float depth, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "vec4 fixedProjected = texture(maps[2], cubeProj.xyz / cubeProj.w);"
        in generated_code
    )
    assert (
        "vec4 dynamicLodProjected = textureLod(maps[layer], cubeProj.xyz / cubeProj.w, lod);"
        in generated_code
    )
    assert (
        "vec4 globalProjected = texture(cubeMaps[layer], cubeProj.xyz / cubeProj.w);"
        in generated_code
    )
    assert (
        "/* unsupported GLSL projected texture: textureProjGradOffset offsets require 1D, 2D, 2D-array, 3D, or planar shadow samplers */ vec4(0.0)"
        in generated_code
    )
    projected_counts = {
        "textureProj": 1,
        "textureProjLod": 2,
        "textureProjGradOffset": 1,
    }
    for func_name, count in projected_counts.items():
        assert (
            generated_code.count(
                f"/* unsupported GLSL projected texture: {func_name} requires 1D, 2D, 2D-array, or 3D projection coordinates */ vec4(0.0)"
            )
            == count
        )
    assert (
        "float fixedProjected = texture(maps[2], vec4(cubeProj.xyz / cubeProj.w, depth));"
        in generated_code
    )
    assert (
        "float globalShadow = texture(shadowCubes[layer], vec4(cubeProj.xyz / cubeProj.w, depth));"
        in generated_code
    )
    assert (
        "/* unsupported GLSL texture compare: textureCompareProjLod explicit LOD requires 2D shadow samplers */ 0.0"
        in generated_code
    )
    assert (
        "/* unsupported GLSL texture compare: textureCompareProjGradOffset explicit gradient offsets require 2D or 2D-array shadow samplers */ 0.0"
        in generated_code
    )
    compare_counts = {
        "textureCompareProj": 1,
        "textureCompareProjLod": 2,
        "textureCompareProjGradOffset": 1,
    }
    for func_name, count in compare_counts.items():
        assert (
            generated_code.count(
                f"/* unsupported GLSL texture compare: {func_name} requires sampler2DShadow vec3/vec4 or sampler2DArrayShadow vec4 projection coordinates */ 0.0"
            )
            == count
        )
    assert (
        "projectedCube(cubeMaps, layer, cubeProj, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedCubeArray(cubeArrays, layer, cubeArrayProj, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedShadow(shadowCubes, layer, cubeProj, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedShadowArray(shadowCubeArrays, layer, cubeArrayProj, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert "textureProj(" not in generated_code
    assert "textureCompareProj(" not in generated_code
    assert "textureGrad(" not in generated_code


def test_opengl_integer_projected_textures_lower_supported_forms_and_diagnose_cube_arrays():
    shader = """
    shader IntegerProjectedTextureDiagnostics {
        isampler2D signedMap;
        usampler2DArray unsignedLayerMap;
        isamplerCube signedCube;
        usamplerCube unsignedCube;
        isamplerCubeArray signedCubeArray;
        usamplerCubeArray unsignedCubeArray;
        sampler linearSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvLayerQ @ TEXCOORD1;
            vec4 cubeProj @ TEXCOORD2;
            float lod;
            vec3 ddx @ TEXCOORD3;
            vec3 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        ivec4 signedProjected(
            isampler2D map,
            isamplerCube cube,
            isamplerCubeArray cubeArray,
            sampler s,
            vec3 uvq,
            vec4 cubeProj,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            ivec4 planar = textureProj(map, s, uvq);
            ivec4 cubeProjected = textureProj(cube, s, cubeProj);
            ivec4 cubeLod = textureProjLod(cube, cubeProj, lod);
            ivec4 cubeGrad = textureProjGrad(cube, s, cubeProj, ddx, ddy);
            ivec4 cubeOffsetRejected = textureProjGradOffset(cube, s, cubeProj, ddx, ddy, offset);
            ivec4 cubeArrayRejected = textureProj(cubeArray, s, cubeProj);
            return planar + cubeProjected + cubeLod + cubeGrad + cubeOffsetRejected + cubeArrayRejected;
        }

        uvec4 unsignedProjected(
            usampler2DArray mapArray,
            usamplerCube cube,
            usamplerCubeArray cubeArray,
            sampler s,
            vec4 uvLayerQ,
            vec4 cubeProj,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            uvec4 planarArray = textureProj(mapArray, s, uvLayerQ);
            uvec4 cubeProjected = textureProj(cube, s, cubeProj);
            uvec4 cubeLod = textureProjLod(cube, cubeProj, lod);
            uvec4 cubeGrad = textureProjGrad(cube, s, cubeProj, ddx, ddy);
            uvec4 cubeOffsetRejected = textureProjGradOffset(cube, s, cubeProj, ddx, ddy, offset);
            uvec4 cubeArrayRejected = textureProj(cubeArray, s, cubeProj);
            return planarArray + cubeProjected + cubeLod + cubeGrad + cubeOffsetRejected + cubeArrayRejected;
        }

        fragment {
            ivec4 main(FSInput input) @ COLOR0 {
                uvec4 unsignedResult = unsignedProjected(
                    unsignedLayerMap,
                    unsignedCube,
                    unsignedCubeArray,
                    linearSampler,
                    input.uvLayerQ,
                    input.cubeProj,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                return signedProjected(
                    signedMap,
                    signedCube,
                    signedCubeArray,
                    linearSampler,
                    input.uvq,
                    input.cubeProj,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + ivec4(unsignedResult);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform isampler2D signedMap;" in generated_code
    assert (
        "layout(binding = 1) uniform usampler2DArray unsignedLayerMap;"
        in generated_code
    )
    assert "layout(binding = 2) uniform isamplerCube signedCube;" in generated_code
    assert "layout(binding = 3) uniform usamplerCube unsignedCube;" in generated_code
    assert (
        "layout(binding = 4) uniform isamplerCubeArray signedCubeArray;"
        in generated_code
    )
    assert (
        "layout(binding = 5) uniform usamplerCubeArray unsignedCubeArray;"
        in generated_code
    )
    assert (
        "ivec4 signedProjected(isampler2D map, isamplerCube cube, isamplerCubeArray cubeArray, vec3 uvq, vec4 cubeProj, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "uvec4 unsignedProjected(usampler2DArray mapArray, usamplerCube cube, usamplerCubeArray cubeArray, vec4 uvLayerQ, vec4 cubeProj, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert "ivec4 planar = textureProj(map, uvq);" in generated_code
    assert "uvec4 planarArray = textureProj(mapArray, uvLayerQ);" in generated_code
    assert (
        "ivec4 cubeProjected = texture(cube, cubeProj.xyz / cubeProj.w);"
        in generated_code
    )
    assert (
        "uvec4 cubeProjected = texture(cube, cubeProj.xyz / cubeProj.w);"
        in generated_code
    )
    assert (
        "ivec4 cubeLod = textureLod(cube, cubeProj.xyz / cubeProj.w, lod);"
        in generated_code
    )
    assert (
        "uvec4 cubeLod = textureLod(cube, cubeProj.xyz / cubeProj.w, lod);"
        in generated_code
    )
    assert (
        "ivec4 cubeGrad = textureGrad(cube, cubeProj.xyz / cubeProj.w, ddx, ddy);"
        in generated_code
    )
    assert (
        "uvec4 cubeGrad = textureGrad(cube, cubeProj.xyz / cubeProj.w, ddx, ddy);"
        in generated_code
    )
    assert (
        "ivec4 cubeOffsetRejected = /* unsupported GLSL projected texture: textureProjGradOffset offsets require 1D, 2D, 2D-array, 3D, or planar shadow samplers */ ivec4(0);"
        in generated_code
    )
    assert (
        "uvec4 cubeOffsetRejected = /* unsupported GLSL projected texture: textureProjGradOffset offsets require 1D, 2D, 2D-array, 3D, or planar shadow samplers */ uvec4(0u);"
        in generated_code
    )
    assert (
        "ivec4 cubeArrayRejected = /* unsupported GLSL projected texture: textureProj requires 1D, 2D, 2D-array, or 3D projection coordinates */ ivec4(0);"
        in generated_code
    )
    assert (
        "uvec4 cubeArrayRejected = /* unsupported GLSL projected texture: textureProj requires 1D, 2D, 2D-array, or 3D projection coordinates */ uvec4(0u);"
        in generated_code
    )
    assert (
        "unsignedProjected(unsignedLayerMap, unsignedCube, unsignedCubeArray, uvLayerQ, cubeProj, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "signedProjected(signedMap, signedCube, signedCubeArray, uvq, cubeProj, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert "textureProj(cube" not in generated_code
    assert "textureProj(cubeArray" not in generated_code


def test_opengl_integer_sampler_offsets_and_gathers_strip_sampler_arguments():
    shader = """
    shader IntegerTextureGatherOffset {
        isampler2D signedMap;
        usampler2DArray unsignedLayers;
        isamplerCube signedCube;
        usamplerCubeArray unsignedCubeArray;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec3 direction @ TEXCOORD2;
            vec4 cubeLayer @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
            int component @ TEXCOORD5;
        };

        fragment {
            ivec4 main(FSInput input) @ COLOR0 {
                ivec4 offsetSigned = textureOffset(signedMap, linearSampler, input.uv, input.offset);
                uvec4 offsetUnsigned = textureLodOffset(unsignedLayers, linearSampler, input.uvLayer, 1.0, input.offset);
                ivec4 gatheredSigned = textureGather(signedMap, linearSampler, input.uv, input.component);
                uvec4 gatheredUnsigned = textureGatherOffset(unsignedLayers, linearSampler, input.uvLayer, input.offset, input.component);
                ivec4 gatheredCube = textureGather(signedCube, linearSampler, input.direction);
                uvec4 gatheredCubeArray = textureGather(unsignedCubeArray, linearSampler, input.cubeLayer);
                return offsetSigned + gatheredSigned + gatheredCube + ivec4(offsetUnsigned + gatheredUnsigned + gatheredCubeArray);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform isampler2D signedMap;" in generated_code
    assert (
        "layout(binding = 1) uniform usampler2DArray unsignedLayers;" in generated_code
    )
    assert "layout(binding = 2) uniform isamplerCube signedCube;" in generated_code
    assert (
        "layout(binding = 3) uniform usamplerCubeArray unsignedCubeArray;"
        in generated_code
    )
    assert (
        "ivec4 offsetSigned = "
        + glsl_dynamic_texel_offset_diagnostic(
            "texture offset", "textureOffset", "ivec4(0)"
        )
        + ";"
        in generated_code
    )
    assert (
        "uvec4 offsetUnsigned = "
        + glsl_dynamic_texel_offset_diagnostic(
            "texture offset", "textureLodOffset", "uvec4(0u)"
        )
        + ";"
        in generated_code
    )
    assert "textureGather(signedMap, uv, 0)" in generated_code
    assert "textureGather(signedMap, uv, 1)" in generated_code
    assert "textureGather(signedMap, uv, 2)" in generated_code
    assert "textureGather(signedMap, uv, 3)" in generated_code
    assert (
        "uvec4 gatheredUnsigned = "
        + glsl_dynamic_texel_offset_diagnostic(
            "texture gather", "textureGatherOffset", "uvec4(0u)"
        )
        + ";"
        in generated_code
    )
    assert (
        "ivec4 gatheredCube = textureGather(signedCube, direction);" in generated_code
    )
    assert (
        "uvec4 gatheredCubeArray = textureGather(unsignedCubeArray, cubeLayer);"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert ", linearSampler" not in generated_code
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGatherOffset(" not in generated_code


def test_opengl_shadow_gather_compare_offsets_filter_sampler_arguments():
    shader = """
    shader ShadowGatherCompareOffsets {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float depth;
            ivec2 offset @ TEXCOORD3;
        };

        vec4 gatherShadow(sampler2DShadow tex, sampler s, vec2 uv, float depth, ivec2 offset) {
            vec4 gathered = textureGatherCompare(tex, s, uv, depth);
            vec4 offsetGathered = textureGatherCompareOffset(tex, s, uv, depth, offset);
            float offsetCompared = textureCompareOffset(tex, s, uv, depth, offset);
            return gathered + offsetGathered + vec4(offsetCompared);
        }

        vec4 gatherShadowArray(sampler2DArrayShadow tex, sampler s, vec3 uvLayer, float depth, ivec2 offset) {
            vec4 gathered = textureGatherCompare(tex, s, uvLayer, depth);
            vec4 offsetGathered = textureGatherCompareOffset(tex, s, uvLayer, depth, offset);
            float offsetCompared = textureCompareOffset(tex, s, uvLayer, depth, offset);
            return gathered + offsetGathered + vec4(offsetCompared);
        }

        vec4 gatherCubeShadowArray(samplerCubeArrayShadow tex, sampler s, vec4 cubeLayer, float depth) {
            return textureGatherCompare(tex, s, cubeLayer, depth);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherShadow(shadowMap, compareSampler, input.uv, input.depth, input.offset)
                    + gatherShadowArray(shadowArray, compareSampler, input.uvLayer, input.depth, input.offset)
                    + gatherCubeShadowArray(cubeShadowArray, compareSampler, input.cubeLayer, input.depth);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "layout(binding = 1) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "layout(binding = 2) uniform samplerCubeArrayShadow cubeShadowArray;"
        in generated_code
    )
    assert (
        "vec4 gatherShadow(sampler2DShadow tex, vec2 uv, float depth, ivec2 offset)"
        in generated_code
    )
    assert "textureGather(tex, uv, depth)" in generated_code
    assert (
        "vec4 offsetGathered = "
        + glsl_dynamic_texel_offset_diagnostic(
            "texture gather compare", "textureGatherCompareOffset", "vec4(0.0)"
        )
        + ";"
        in generated_code
    )
    assert (
        "float offsetCompared = /* unsupported GLSL texture compare: "
        "textureCompareOffset texel offsets must be compile-time integer constants "
        "*/ 0.0;" in generated_code
    )
    assert (
        "vec4 gatherShadowArray(sampler2DArrayShadow tex, vec3 uvLayer, float depth, ivec2 offset)"
        in generated_code
    )
    assert "textureGather(tex, uvLayer, depth)" in generated_code
    assert (
        generated_code.count(
            glsl_dynamic_texel_offset_diagnostic(
                "texture gather compare",
                "textureGatherCompareOffset",
                "vec4(0.0)",
            )
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture compare: textureCompareOffset texel offsets must be compile-time integer constants */ 0.0"
        )
        == 2
    )
    assert (
        "vec4 gatherCubeShadowArray(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth)"
        in generated_code
    )
    assert "return textureGather(tex, cubeLayer, depth);" in generated_code
    assert "gatherShadow(shadowMap, uv, depth, offset)" in generated_code
    assert "gatherShadowArray(shadowArray, uvLayer, depth, offset)" in generated_code
    assert "gatherCubeShadowArray(cubeShadowArray, cubeLayer, depth)" in generated_code
    assert "compareSampler" not in generated_code
    assert "textureGatherCompare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code
    assert "textureCompareOffset(" not in generated_code
    assert "textureGatherOffset(" not in generated_code


def test_opengl_shadow_compare_offset_literal_offsets_lower_to_texture_offset():
    shader = """
    shader ShadowCompareLiteralOffset {
        sampler2DShadow shadowMap;
        sampler compareSampler;

        vec4 sampleShadow(sampler2DShadow tex, sampler s, vec2 uv, float depth) {
            float compared = textureCompareOffset(tex, s, uv, depth, ivec2(1, 0));
            return vec4(compared);
        }

        fragment {
            vec4 main() @ gl_FragColor {
                return sampleShadow(shadowMap, compareSampler, vec2(0.5), 0.25);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "float compared = textureOffset(tex, vec3(uv, depth), ivec2(1, 0));"
        in generated_code
    )
    assert "compareSampler" not in generated_code
    assert "unsupported GLSL texture compare" not in generated_code


def test_opengl_cube_shadow_gather_compare_supports_cube_and_cube_array():
    shader = """
    shader CubeShadowGatherCompare {
        samplerCubeShadow cubeShadow;
        samplerCubeArrayShadow cubeShadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
        };

        vec4 gatherCubeShadow(samplerCubeShadow tex, sampler s, vec3 direction, float depth) {
            return textureGatherCompare(tex, s, direction, depth);
        }

        vec4 gatherCubeArrayShadow(samplerCubeArrayShadow tex, sampler s, vec4 cubeLayer, float depth) {
            return textureGatherCompare(tex, s, cubeLayer, depth);
        }

        vec4 implicitCubeShadow(samplerCubeShadow tex, vec3 direction, float depth) {
            return textureGatherCompare(tex, direction, depth);
        }

        vec4 implicitCubeArrayShadow(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth) {
            return textureGatherCompare(tex, cubeLayer, depth);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherCubeShadow(cubeShadow, compareSampler, input.direction, input.depth)
                    + gatherCubeArrayShadow(cubeShadowArray, compareSampler, input.cubeLayer, input.depth)
                    + implicitCubeShadow(cubeShadow, input.direction, input.depth)
                    + implicitCubeArrayShadow(cubeShadowArray, input.cubeLayer, input.depth)
                    + textureGatherCompare(cubeShadow, compareSampler, input.direction, input.depth)
                    + textureGatherCompare(cubeShadowArray, compareSampler, input.cubeLayer, input.depth)
                    + textureGatherCompare(cubeShadow, input.direction, input.depth)
                    + textureGatherCompare(cubeShadowArray, input.cubeLayer, input.depth);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform samplerCubeShadow cubeShadow;" in generated_code
    assert (
        "layout(binding = 1) uniform samplerCubeArrayShadow cubeShadowArray;"
        in generated_code
    )
    assert (
        "vec4 gatherCubeShadow(samplerCubeShadow tex, vec3 direction, float depth)"
        in generated_code
    )
    assert "return textureGather(tex, direction, depth);" in generated_code
    assert (
        "vec4 gatherCubeArrayShadow(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth)"
        in generated_code
    )
    assert "return textureGather(tex, cubeLayer, depth);" in generated_code
    assert (
        "vec4 implicitCubeShadow(samplerCubeShadow tex, vec3 direction, float depth)"
        in generated_code
    )
    assert (
        "vec4 implicitCubeArrayShadow(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth)"
        in generated_code
    )
    assert "gatherCubeShadow(cubeShadow, direction, depth)" in generated_code
    assert "gatherCubeArrayShadow(cubeShadowArray, cubeLayer, depth)" in generated_code
    assert "textureGather(cubeShadow, direction, depth)" in generated_code
    assert "textureGather(cubeShadowArray, cubeLayer, depth)" in generated_code
    assert "compareSampler" not in generated_code
    assert "unsupported GLSL texture gather compare" not in generated_code
    assert "textureGatherCompare(" not in generated_code
    assert "input." not in generated_code


def test_opengl_direct_shadow_gather_compare_stage_input_members():
    explicit_shader = """
    shader DirectShadowGatherCompare {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float depth;
            ivec2 offset @ TEXCOORD3;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 planar = textureGatherCompare(shadowMap, compareSampler, input.uv, input.depth);
                vec4 planarOffset = textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depth, input.offset);
                vec4 arrayGather = textureGatherCompare(shadowArray, compareSampler, input.uvLayer, input.depth);
                vec4 arrayOffset = textureGatherCompareOffset(shadowArray, compareSampler, input.uvLayer, input.depth, input.offset);
                vec4 cubeArrayGather = textureGatherCompare(cubeShadowArray, compareSampler, input.cubeLayer, input.depth);
                return planar + planarOffset + arrayGather + arrayOffset + cubeArrayGather;
            }
        }
    }
    """
    implicit_shader = """
    shader DirectImplicitShadowGatherCompare {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float depth;
            ivec2 offset @ TEXCOORD3;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 planar = textureGatherCompare(shadowMap, input.uv, input.depth);
                vec4 planarOffset = textureGatherCompareOffset(shadowMap, input.uv, input.depth, input.offset);
                vec4 arrayGather = textureGatherCompare(shadowArray, input.uvLayer, input.depth);
                vec4 arrayOffset = textureGatherCompareOffset(shadowArray, input.uvLayer, input.depth, input.offset);
                vec4 cubeArrayGather = textureGatherCompare(cubeShadowArray, input.cubeLayer, input.depth);
                return planar + planarOffset + arrayGather + arrayOffset + cubeArrayGather;
            }
        }
    }
    """

    explicit_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(explicit_shader), "fragment"
    )
    implicit_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(implicit_shader), "fragment"
    )

    for generated_code in (explicit_code, implicit_code):
        assert (
            "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
        )
        assert (
            "layout(binding = 1) uniform sampler2DArrayShadow shadowArray;"
            in generated_code
        )
        assert (
            "layout(binding = 2) uniform samplerCubeArrayShadow cubeShadowArray;"
            in generated_code
        )
        assert "vec4 planar = textureGather(shadowMap, uv, depth);" in generated_code
        assert (
            "vec4 planarOffset = "
            + glsl_dynamic_texel_offset_diagnostic(
                "texture gather compare",
                "textureGatherCompareOffset",
                "vec4(0.0)",
            )
            + ";"
            in generated_code
        )
        assert (
            "vec4 arrayGather = textureGather(shadowArray, uvLayer, depth);"
            in generated_code
        )
        assert (
            "vec4 arrayOffset = "
            + glsl_dynamic_texel_offset_diagnostic(
                "texture gather compare",
                "textureGatherCompareOffset",
                "vec4(0.0)",
            )
            + ";"
            in generated_code
        )
        assert (
            "vec4 cubeArrayGather = textureGather(cubeShadowArray, cubeLayer, depth);"
            in generated_code
        )
        assert "textureGatherCompare(" not in generated_code
        assert "textureGatherCompareOffset(" not in generated_code
        assert "textureGatherOffset(" not in generated_code
        assert "compareSampler" not in generated_code
        assert "input." not in generated_code


def test_opengl_shadow_gather_compare_resource_arrays_strip_samplers():
    explicit_shader = """
    shader ShadowGatherCompareResourceArrays {
        sampler2DShadow shadowMaps[4];
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeArrays[4];
        sampler compareSamplers[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            vec4 cubeLayer @ TEXCOORD3;
            float depth;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 gatherLayer(
            sampler2DShadow maps[],
            sampler2DArrayShadow arrays[],
            samplerCubeArrayShadow cubes[],
            sampler samplers[],
            int layer,
            vec2 uv,
            vec3 uvLayer,
            vec4 cubeLayer,
            float depth,
            ivec2 offset
        ) {
            vec4 fixedPlanar = textureGatherCompare(maps[2], samplers[2], uv, depth);
            vec4 dynamicPlanarOffset = textureGatherCompareOffset(maps[layer], samplers[layer], uv, depth, offset);
            vec4 fixedArray = textureGatherCompare(arrays[1], samplers[1], uvLayer, depth);
            vec4 dynamicArrayOffset = textureGatherCompareOffset(arrays[layer], samplers[layer], uvLayer, depth, offset);
            vec4 fixedCubeArray = textureGatherCompare(cubes[3], samplers[3], cubeLayer, depth);
            vec4 dynamicCubeArray = textureGatherCompare(cubes[layer], samplers[layer], cubeLayer, depth);
            return fixedPlanar + dynamicPlanarOffset + fixedArray + dynamicArrayOffset + fixedCubeArray + dynamicCubeArray;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherLayer(
                    shadowMaps,
                    shadowArrays,
                    cubeArrays,
                    compareSamplers,
                    input.layer,
                    input.uv,
                    input.uvLayer,
                    input.cubeLayer,
                    input.depth,
                    input.offset
                );
            }
        }
    }
    """
    implicit_shader = """
    shader ImplicitShadowGatherCompareResourceArrays {
        sampler2DShadow shadowMaps[4];
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            vec4 cubeLayer @ TEXCOORD3;
            float depth;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 gatherLayer(
            sampler2DShadow maps[],
            sampler2DArrayShadow arrays[],
            samplerCubeArrayShadow cubes[],
            int layer,
            vec2 uv,
            vec3 uvLayer,
            vec4 cubeLayer,
            float depth,
            ivec2 offset
        ) {
            vec4 fixedPlanar = textureGatherCompare(maps[2], uv, depth);
            vec4 dynamicPlanarOffset = textureGatherCompareOffset(maps[layer], uv, depth, offset);
            vec4 fixedArray = textureGatherCompare(arrays[1], uvLayer, depth);
            vec4 dynamicArrayOffset = textureGatherCompareOffset(arrays[layer], uvLayer, depth, offset);
            vec4 fixedCubeArray = textureGatherCompare(cubes[3], cubeLayer, depth);
            vec4 dynamicCubeArray = textureGatherCompare(cubes[layer], cubeLayer, depth);
            return fixedPlanar + dynamicPlanarOffset + fixedArray + dynamicArrayOffset + fixedCubeArray + dynamicCubeArray;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherLayer(
                    shadowMaps,
                    shadowArrays,
                    cubeArrays,
                    input.layer,
                    input.uv,
                    input.uvLayer,
                    input.cubeLayer,
                    input.depth,
                    input.offset
                );
            }
        }
    }
    """

    explicit_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(explicit_shader), "fragment"
    )
    implicit_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(implicit_shader), "fragment"
    )

    for generated_code in (explicit_code, implicit_code):
        assert (
            "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];"
            in generated_code
        )
        assert (
            "layout(binding = 4) uniform sampler2DArrayShadow shadowArrays[4];"
            in generated_code
        )
        assert (
            "layout(binding = 8) uniform samplerCubeArrayShadow cubeArrays[4];"
            in generated_code
        )
        assert (
            "vec4 gatherLayer(sampler2DShadow maps[4], sampler2DArrayShadow arrays[4], samplerCubeArrayShadow cubes[4], int layer, vec2 uv, vec3 uvLayer, vec4 cubeLayer, float depth, ivec2 offset)"
            in generated_code
        )
        assert "vec4 fixedPlanar = textureGather(maps[2], uv, depth);" in generated_code
        assert (
            "vec4 dynamicPlanarOffset = "
            + glsl_dynamic_texel_offset_diagnostic(
                "texture gather compare",
                "textureGatherCompareOffset",
                "vec4(0.0)",
            )
            + ";"
            in generated_code
        )
        assert (
            "vec4 fixedArray = textureGather(arrays[1], uvLayer, depth);"
            in generated_code
        )
        assert (
            "vec4 dynamicArrayOffset = "
            + glsl_dynamic_texel_offset_diagnostic(
                "texture gather compare",
                "textureGatherCompareOffset",
                "vec4(0.0)",
            )
            + ";"
            in generated_code
        )
        assert (
            "vec4 fixedCubeArray = textureGather(cubes[3], cubeLayer, depth);"
            in generated_code
        )
        assert (
            "vec4 dynamicCubeArray = textureGather(cubes[layer], cubeLayer, depth);"
            in generated_code
        )
        assert (
            "gatherLayer(shadowMaps, shadowArrays, cubeArrays, layer, uv, uvLayer, cubeLayer, depth, offset)"
            in generated_code
        )
        assert "compareSamplers" not in generated_code
        assert "samplers" not in generated_code
        assert "textureGatherCompare(" not in generated_code
        assert "textureGatherCompareOffset(" not in generated_code
        assert "textureGatherOffset(" not in generated_code
        assert "input." not in generated_code


def test_opengl_unsupported_cube_shadow_gather_compare_offsets_emit_diagnostics():
    shader = """
    shader UnsupportedCubeGatherCompareOffset {
        samplerCubeShadow cubeMap;
        samplerCubeArrayShadow cubeArray;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            ivec2 offset @ TEXCOORD2;
        };

        vec4 cubeOffset(
            samplerCubeShadow tex,
            vec3 direction,
            float depth,
            ivec2 offset
        ) {
            return textureGatherCompareOffset(tex, direction, depth, offset);
        }

        vec4 cubeArrayOffset(
            samplerCubeArrayShadow tex,
            vec4 cubeLayer,
            float depth,
            ivec2 offset
        ) {
            return textureGatherCompareOffset(tex, cubeLayer, depth, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return cubeOffset(cubeMap, input.direction, input.depth, input.offset)
                    + cubeArrayOffset(cubeArray, input.cubeLayer, input.depth, input.offset)
                    + textureGatherCompareOffset(cubeArray, input.cubeLayer, input.depth, input.offset);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported GLSL texture gather compare: "
        "textureGatherCompareOffset offsets require 2D or 2D-array shadow samplers */ "
        "vec4(0.0)"
    )
    assert "layout(binding = 0) uniform samplerCubeShadow cubeMap;" in generated_code
    assert (
        "layout(binding = 1) uniform samplerCubeArrayShadow cubeArray;"
        in generated_code
    )
    assert (
        "vec4 cubeOffset(samplerCubeShadow tex, vec3 direction, float depth, ivec2 offset)"
        in generated_code
    )
    assert (
        "vec4 cubeArrayOffset(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth, ivec2 offset)"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 3
    assert "cubeOffset(cubeMap, direction, depth, offset)" in generated_code
    assert "cubeArrayOffset(cubeArray, cubeLayer, depth, offset)" in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code
    assert "input." not in generated_code


def test_opengl_shadow_compare_lod_grad_filter_sampler_arguments():
    shader = """
    shader ShadowCompareLodGrad {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        float compareShadow(
            sampler2DShadow tex,
            sampler s,
            vec2 uv,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float lodValue = textureCompareLod(tex, s, uv, depth, lod);
            float lodOffsetValue = textureCompareLodOffset(tex, s, uv, depth, lod, offset);
            float gradValue = textureCompareGrad(tex, s, uv, depth, ddx, ddy);
            float gradOffsetValue = textureCompareGradOffset(tex, s, uv, depth, ddx, ddy, offset);
            return lodValue + lodOffsetValue + gradValue + gradOffsetValue;
        }

        float compareShadowArray(
            sampler2DArrayShadow tex,
            sampler s,
            vec3 uvLayer,
            float depth,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float gradValue = textureCompareGrad(tex, s, uvLayer, depth, ddx, ddy);
            float gradOffsetValue = textureCompareGradOffset(tex, s, uvLayer, depth, ddx, ddy, offset);
            return gradValue + gradOffsetValue;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float shadow = compareShadow(
                    shadowMap,
                    compareSampler,
                    input.uv,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float arrayShadow = compareShadowArray(
                    shadowArray,
                    compareSampler,
                    input.uvLayer,
                    input.depth,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                return vec4(shadow + arrayShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "layout(binding = 1) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "float compareShadow(sampler2DShadow tex, vec2 uv, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert "float lodValue = textureLod(tex, vec3(uv, depth), lod);" in generated_code
    assert (
        "float lodOffsetValue = "
        + glsl_dynamic_texel_offset_diagnostic(
            "texture compare", "textureCompareLodOffset", "0.0"
        )
        + ";"
        in generated_code
    )
    assert (
        "float gradValue = textureGrad(tex, vec3(uv, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetValue = "
        + glsl_dynamic_texel_offset_diagnostic(
            "texture compare", "textureCompareGradOffset", "0.0"
        )
        + ";"
        in generated_code
    )
    assert (
        "float compareShadowArray(sampler2DArrayShadow tex, vec3 uvLayer, float depth, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "float gradValue = textureGrad(tex, vec4(uvLayer, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetValue = "
        + glsl_dynamic_texel_offset_diagnostic(
            "texture compare", "textureCompareGradOffset", "0.0"
        )
        + ";"
        in generated_code
    )
    assert (
        "compareShadow(shadowMap, uv, depth, lod, ddx, ddy, offset)" in generated_code
    )
    assert (
        "compareShadowArray(shadowArray, uvLayer, depth, ddx, ddy, offset)"
        in generated_code
    )
    assert "compareSampler" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGrad(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code


def test_opengl_nested_implicit_shadow_compare_lod_grad_strips_samplers():
    shader = """
    shader NestedShadowCompareQuery {
        sampler2DShadow shadowMap;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
            vec2 ddx @ TEXCOORD1;
            vec2 ddy @ TEXCOORD2;
            ivec2 offset @ TEXCOORD3;
        };

        float shadowOps(
            sampler2DShadow tex,
            vec2 uv,
            float depth,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec2 lod = textureQueryLod(tex, uv);
            float cmp = textureCompareLod(tex, uv, depth, lod.x);
            float grad = textureCompareGradOffset(tex, uv, depth, ddx, ddy, offset);
            return cmp + grad + lod.y;
        }

        float wrappedShadow(
            sampler2DShadow tex,
            vec2 uv,
            float depth,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            return shadowOps(tex, uv, depth, ddx, ddy, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return vec4(wrappedShadow(
                    shadowMap,
                    input.uv,
                    input.depth,
                    input.ddx,
                    input.ddy,
                    input.offset
                ));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "float shadowOps(sampler2DShadow tex, vec2 uv, float depth, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert "vec2 lod = textureQueryLod(tex, uv);" in generated_code
    assert "float cmp = textureLod(tex, vec3(uv, depth), lod.x);" in generated_code
    assert (
        "float grad = "
        + glsl_dynamic_texel_offset_diagnostic(
            "texture compare", "textureCompareGradOffset", "0.0"
        )
        + ";"
        in generated_code
    )
    assert (
        "float wrappedShadow(sampler2DShadow tex, vec2 uv, float depth, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert "return shadowOps(tex, uv, depth, ddx, ddy, offset);" in generated_code
    assert "wrappedShadow(shadowMap, uv, depth, ddx, ddy, offset)" in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code


def test_opengl_array_shadow_compare_lod_reports_unsupported():
    shader = """
    shader ArrayShadowCompareLodUnsupported {
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            float depth;
            float lod;
        };

        float compareShadowArray(
            sampler2DArrayShadow tex,
            sampler s,
            vec3 uvLayer,
            float depth,
            float lod
        ) {
            float rejected = textureCompareLod(tex, s, uvLayer, depth, lod);
            return rejected;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float shadow = compareShadowArray(
                    shadowArray,
                    compareSampler,
                    input.uvLayer,
                    input.depth,
                    input.lod
                );
                return vec4(shadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "float rejected = /* unsupported GLSL texture compare: textureCompareLod explicit LOD requires 2D shadow samplers */ 0.0;"
        in generated_code
    )
    assert "textureLod(" not in generated_code


def test_opengl_array_shadow_compare_lod_offset_reports_unsupported():
    shader = """
    shader ArrayShadowCompareLodOffsetUnsupported {
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            float depth;
            float lod;
            ivec2 offset @ TEXCOORD1;
        };

        float compareShadowArray(
            sampler2DArrayShadow tex,
            sampler s,
            vec3 uvLayer,
            float depth,
            float lod,
            ivec2 offset
        ) {
            float rejected = textureCompareLodOffset(tex, s, uvLayer, depth, lod, offset);
            return rejected;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float shadow = compareShadowArray(
                    shadowArray,
                    compareSampler,
                    input.uvLayer,
                    input.depth,
                    input.lod,
                    input.offset
                );
                return vec4(shadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "float rejected = /* unsupported GLSL texture compare: textureCompareLodOffset explicit LOD offsets require 2D shadow samplers */ 0.0;"
        in generated_code
    )
    assert "textureLodOffset(" not in generated_code


def test_opengl_cube_shadow_compare_lod_offset_diagnostics():
    shader = """
    shader CubeShadowCompareLodOffsetDiagnostics {
        samplerCubeShadow cubeShadow;
        samplerCubeArrayShadow cubeShadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        float compareCubeShadow(
            samplerCubeShadow tex,
            sampler s,
            vec3 direction,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float lodRejected = textureCompareLod(tex, s, direction, depth, lod);
            float lodOffsetRejected = textureCompareLodOffset(tex, s, direction, depth, lod, offset);
            float gradValue = textureCompareGrad(tex, s, direction, depth, ddx, ddy);
            float gradOffsetRejected = textureCompareGradOffset(tex, s, direction, depth, ddx, ddy, offset);
            return lodRejected + lodOffsetRejected + gradValue + gradOffsetRejected;
        }

        float compareCubeArrayShadow(
            samplerCubeArrayShadow tex,
            sampler s,
            vec4 cubeLayer,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float lodRejected = textureCompareLod(tex, s, cubeLayer, depth, lod);
            float lodOffsetRejected = textureCompareLodOffset(tex, s, cubeLayer, depth, lod, offset);
            float gradRejected = textureCompareGrad(tex, s, cubeLayer, depth, ddx, ddy);
            float gradOffsetRejected = textureCompareGradOffset(tex, s, cubeLayer, depth, ddx, ddy, offset);
            return lodRejected + lodOffsetRejected + gradRejected + gradOffsetRejected;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return vec4(
                    compareCubeShadow(
                        cubeShadow,
                        compareSampler,
                        input.direction,
                        input.depth,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    )
                    + compareCubeArrayShadow(
                        cubeShadowArray,
                        compareSampler,
                        input.cubeLayer,
                        input.depth,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    )
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "float lodRejected = /* unsupported GLSL texture compare: textureCompareLod explicit LOD requires 2D shadow samplers */ 0.0;"
        in generated_code
    )
    assert (
        "float lodOffsetRejected = /* unsupported GLSL texture compare: textureCompareLodOffset explicit LOD offsets require 2D shadow samplers */ 0.0;"
        in generated_code
    )
    assert (
        "float gradValue = textureGrad(tex, vec4(direction, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetRejected = /* unsupported GLSL texture compare: textureCompareGradOffset explicit gradient offsets require 2D or 2D-array shadow samplers */ 0.0;"
        in generated_code
    )
    assert (
        "float gradRejected = /* unsupported GLSL texture compare: textureCompareGrad explicit gradients require 2D, 2D-array, or cube shadow samplers */ 0.0;"
        in generated_code
    )
    assert "textureLod(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareLodOffset(" not in generated_code
    assert "textureCompareGrad(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code
    assert "compareSampler" not in generated_code


def test_opengl_array_shadow_texture_resource_arrays_keep_compare_coordinates():
    shader = """
    shader ArrayShadowTextureResourceArrays {
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeShadowArrays[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
        };

        float sampleArrayLayer(sampler2DArrayShadow shadowArrays[], sampler shadowSamplers[], vec3 uvLayer, float depth) {
            return textureCompare(shadowArrays[2], shadowSamplers[2], uvLayer, depth);
        }

        float sampleCubeLayer(samplerCubeArrayShadow cubeShadowArrays[], sampler shadowSamplers[], vec4 cubeLayer, float depth) {
            return textureCompare(cubeShadowArrays[3], shadowSamplers[3], cubeLayer, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return sampleArrayLayer(shadowArrays, shadowSamplers, input.uvLayer, input.depth) + sampleCubeLayer(cubeShadowArrays, shadowSamplers, input.cubeLayer, input.depth);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DArrayShadow shadowArrays[4];"
        in generated_code
    )
    assert (
        "layout(binding = 4) uniform samplerCubeArrayShadow cubeShadowArrays[4];"
        in generated_code
    )
    assert (
        "float sampleArrayLayer(sampler2DArrayShadow shadowArrays[4], vec3 uvLayer, float depth)"
        in generated_code
    )
    assert "texture(shadowArrays[2], vec4(uvLayer, depth))" in generated_code
    assert (
        "float sampleCubeLayer(samplerCubeArrayShadow cubeShadowArrays[4], vec4 cubeLayer, float depth)"
        in generated_code
    )
    assert "texture(cubeShadowArrays[3], cubeLayer, depth)" in generated_code
    assert "sampleArrayLayer(shadowArrays, uvLayer, depth)" in generated_code
    assert "sampleCubeLayer(cubeShadowArrays, cubeLayer, depth)" in generated_code
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[2]" not in generated_code
    assert "shadowSamplers[3]" not in generated_code
    assert "textureCompare(" not in generated_code
    assert "vec3(uvLayer, depth)" not in generated_code
    assert "vec4(cubeLayer, depth)" not in generated_code


def test_opengl_array_shadow_compare_lod_grad_resource_arrays():
    shader = """
    shader ShadowCompareResourceArrays {
        sampler2DShadow shadowMaps[4];
        sampler2DArrayShadow shadowArrays[4];
        sampler shadowSamplers[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        float shadowLayer(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            sampler shadowSamplers[],
            int layer,
            vec2 uv,
            vec3 uvLayer,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float planarLod = textureCompareLod(shadowMaps[layer], shadowSamplers[layer], uv, depth, lod);
            float planarGrad = textureCompareGradOffset(shadowMaps[1], shadowSamplers[1], uv, depth, ddx, ddy, offset);
            float arrayLod = textureCompareLod(shadowArrays[2], shadowSamplers[2], uvLayer, depth, lod);
            float arrayGrad = textureCompareGradOffset(shadowArrays[layer], shadowSamplers[layer], uvLayer, depth, ddx, ddy, offset);
            return planarLod + planarGrad + arrayLod + arrayGrad;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return shadowLayer(
                    shadowMaps,
                    shadowArrays,
                    shadowSamplers,
                    input.layer,
                    input.uv,
                    input.uvLayer,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert (
        "layout(binding = 4) uniform sampler2DArrayShadow shadowArrays[4];"
        in generated_code
    )
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[4], sampler2DArrayShadow shadowArrays[4], int layer, vec2 uv, vec3 uvLayer, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "float planarLod = textureLod(shadowMaps[layer], vec3(uv, depth), lod);"
        in generated_code
    )
    assert (
        "float planarGrad = "
        + glsl_dynamic_texel_offset_diagnostic(
            "texture compare", "textureCompareGradOffset", "0.0"
        )
        + ";"
        in generated_code
    )
    assert (
        "float arrayLod = /* unsupported GLSL texture compare: textureCompareLod explicit LOD requires 2D shadow samplers */ 0.0;"
        in generated_code
    )
    assert (
        "float arrayGrad = "
        + glsl_dynamic_texel_offset_diagnostic(
            "texture compare", "textureCompareGradOffset", "0.0"
        )
        + ";"
        in generated_code
    )
    assert (
        "shadowLayer(shadowMaps, shadowArrays, layer, uv, uvLayer, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code


def test_opengl_cube_shadow_compare_lod_grad_resource_arrays():
    shader = """
    shader CubeShadowCompareResourceArrays {
        samplerCubeShadow cubeMaps[4];
        samplerCubeArrayShadow cubeArrays[4];
        sampler shadowSamplers[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec3 direction @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD3;
            vec3 ddy @ TEXCOORD4;
        };

        float cubeShadowLayer(
            samplerCubeShadow cubeMaps[],
            samplerCubeArrayShadow cubeArrays[],
            sampler shadowSamplers[],
            int layer,
            vec3 direction,
            vec4 cubeLayer,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy
        ) {
            float cubeLod = textureCompareLod(cubeMaps[layer], shadowSamplers[layer], direction, depth, lod);
            float cubeGrad = textureCompareGrad(cubeMaps[1], shadowSamplers[1], direction, depth, ddx, ddy);
            float cubeArrayLod = textureCompareLod(cubeArrays[2], shadowSamplers[2], cubeLayer, depth, lod);
            float cubeArrayGrad = textureCompareGrad(cubeArrays[layer], shadowSamplers[layer], cubeLayer, depth, ddx, ddy);
            return cubeLod + cubeGrad + cubeArrayLod + cubeArrayGrad;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return cubeShadowLayer(
                    cubeMaps,
                    cubeArrays,
                    shadowSamplers,
                    input.layer,
                    input.direction,
                    input.cubeLayer,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform samplerCubeShadow cubeMaps[4];" in generated_code
    )
    assert (
        "layout(binding = 4) uniform samplerCubeArrayShadow cubeArrays[4];"
        in generated_code
    )
    assert (
        "float cubeShadowLayer(samplerCubeShadow cubeMaps[4], samplerCubeArrayShadow cubeArrays[4], int layer, vec3 direction, vec4 cubeLayer, float depth, float lod, vec3 ddx, vec3 ddy)"
        in generated_code
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture compare: textureCompareLod explicit LOD requires 2D shadow samplers */ 0.0"
        )
        == 2
    )
    assert (
        "float cubeGrad = textureGrad(cubeMaps[1], vec4(direction, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float cubeArrayGrad = /* unsupported GLSL texture compare: textureCompareGrad explicit gradients require 2D, 2D-array, or cube shadow samplers */ 0.0;"
        in generated_code
    )
    assert (
        "cubeShadowLayer(cubeMaps, cubeArrays, layer, direction, cubeLayer, depth, lod, ddx, ddy)"
        in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGrad(" not in generated_code


def test_opengl_array_shadow_texture_resource_arrays_reject_mismatched_fixed_helper_size():
    shader = """
    shader ArrayShadowTextureResourceArrayMismatch {
        sampler2DArrayShadow shadowArrays[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            float depth;
        };

        float sampleArrayLayer(sampler2DArrayShadow shadowArrays[3], sampler shadowSamplers[3], vec3 uvLayer, float depth) {
            return textureCompare(shadowArrays[2], shadowSamplers[2], uvLayer, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return sampleArrayLayer(shadowArrays, shadowSamplers, input.uvLayer, input.depth);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'shadowArrays': 4 and 3",
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_storage_image_load_store():
    shader = """
    shader StorageImages {
        image2D outputImage;
        image3D volumeImage;
        image2DArray layerImage;

        vec4 touchImages(image2D outImg, image3D volume, image2DArray layers, ivec2 pixel, ivec3 voxel, ivec3 pixelLayer) {
            vec4 color = imageLoad(outImg, pixel);
            vec4 volumeColor = imageLoad(volume, voxel);
            vec4 layerColor = imageLoad(layers, pixelLayer);
            imageStore(outImg, pixel, color + layerColor);
            imageStore(volume, voxel, volumeColor);
            imageStore(layers, pixelLayer, color);
            return color + volumeColor + layerColor;
        }

        compute {
            void main() {
                vec4 result = touchImages(outputImage, volumeImage, layerImage, ivec2(0, 1), ivec3(0, 1, 2), ivec3(3, 4, 5));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(rgba32f, binding = 0) uniform image2D outputImage;" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image3D volumeImage;" in generated_code
    assert (
        "layout(rgba32f, binding = 2) uniform image2DArray layerImage;"
        in generated_code
    )
    assert (
        "vec4 touchImages(image2D outImg, image3D volume, image2DArray layers, ivec2 pixel, ivec3 voxel, ivec3 pixelLayer)"
        in generated_code
    )
    assert "vec4 color = imageLoad(outImg, pixel);" in generated_code
    assert "vec4 volumeColor = imageLoad(volume, voxel);" in generated_code
    assert "vec4 layerColor = imageLoad(layers, pixelLayer);" in generated_code
    assert "imageStore(outImg, pixel, (color + layerColor));" in generated_code
    assert "imageStore(volume, voxel, volumeColor);" in generated_code
    assert "imageStore(layers, pixelLayer, color);" in generated_code


def test_opengl_direct_stage_image_load_store_and_atomics_use_input_members():
    shader = """
    shader DirectStageImageOps {
        image2D outImg;
        image3D volume;
        image2DArray layers;
        uimage2D counters;
        iimage2DArray signedLayers;

        struct FSInput {
            ivec2 pixel @ TEXCOORD0;
            ivec3 voxel @ TEXCOORD1;
            ivec3 pixelLayer @ TEXCOORD2;
            uint amount;
            int signedAmount;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 color = imageLoad(outImg, input.pixel);
                vec4 volumeColor = imageLoad(volume, input.voxel);
                vec4 layerColor = imageLoad(layers, input.pixelLayer);
                imageStore(outImg, input.pixel, color + layerColor);
                imageStore(volume, input.voxel, volumeColor);
                imageStore(layers, input.pixelLayer, color);
                uint previous = imageAtomicAdd(counters, input.pixel, input.amount);
                int previousSigned = imageAtomicMin(signedLayers, input.pixelLayer, input.signedAmount);
                return color + volumeColor + layerColor + vec4(float(previous + uint(previousSigned)));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(rgba32f, binding = 0) uniform image2D outImg;" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image3D volume;" in generated_code
    assert "layout(rgba32f, binding = 2) uniform image2DArray layers;" in generated_code
    assert "layout(r32ui, binding = 3) uniform uimage2D counters;" in generated_code
    assert (
        "layout(r32i, binding = 4) uniform iimage2DArray signedLayers;"
        in generated_code
    )
    assert "vec4 color = imageLoad(outImg, pixel);" in generated_code
    assert "vec4 volumeColor = imageLoad(volume, voxel);" in generated_code
    assert "vec4 layerColor = imageLoad(layers, pixelLayer);" in generated_code
    assert "imageStore(outImg, pixel, (color + layerColor));" in generated_code
    assert "imageStore(volume, voxel, volumeColor);" in generated_code
    assert "imageStore(layers, pixelLayer, color);" in generated_code
    assert "uint previous = imageAtomicAdd(counters, pixel, amount);" in generated_code
    assert (
        "int previousSigned = imageAtomicMin(signedLayers, pixelLayer, signedAmount);"
        in generated_code
    )
    assert "input.pixel" not in generated_code
    assert "input.voxel" not in generated_code
    assert "input.pixelLayer" not in generated_code


def test_opengl_direct_stage_explicit_image_formats_use_input_members():
    shader = """
    shader DirectStageImageFormats {
        image2D scalarFloat @r32f;
        image2D rgFloat @rg32f;
        uimage2D unsignedScalar @r32ui;
        uimage2DArray unsignedLayers @rg32ui;
        iimage3D signedVolume @r32i;

        struct FSInput {
            ivec2 pixel @ TEXCOORD0;
            ivec3 voxel @ TEXCOORD1;
            ivec3 pixelLayer @ TEXCOORD2;
            float amount;
            uint unsignedAmount;
            int signedAmount;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float scalarValue = imageLoad(scalarFloat, input.pixel);
                vec2 rgValue = imageLoad(rgFloat, input.pixel);
                uint unsignedValue = imageLoad(unsignedScalar, input.pixel);
                uvec2 layerValue = imageLoad(unsignedLayers, input.pixelLayer);
                int signedValue = imageLoad(signedVolume, input.voxel);
                imageStore(scalarFloat, input.pixel, scalarValue + input.amount);
                imageStore(rgFloat, input.pixel, rgValue + vec2(input.amount));
                imageStore(unsignedScalar, input.pixel, unsignedValue + input.unsignedAmount);
                imageStore(unsignedLayers, input.pixelLayer, layerValue + uvec2(input.unsignedAmount));
                imageStore(signedVolume, input.voxel, signedValue + input.signedAmount);
                return vec4(scalarValue + rgValue.x + float(unsignedValue + layerValue.x) + float(signedValue));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r32f, binding = 0) uniform image2D scalarFloat;" in generated_code
    assert "layout(rg32f, binding = 1) uniform image2D rgFloat;" in generated_code
    assert (
        "layout(r32ui, binding = 2) uniform uimage2D unsignedScalar;" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 3) uniform uimage2DArray unsignedLayers;"
        in generated_code
    )
    assert "layout(r32i, binding = 4) uniform iimage3D signedVolume;" in generated_code
    assert "float scalarValue = imageLoad(scalarFloat, pixel).x;" in generated_code
    assert "vec2 rgValue = imageLoad(rgFloat, pixel).xy;" in generated_code
    assert "uint unsignedValue = imageLoad(unsignedScalar, pixel).x;" in generated_code
    assert (
        "uvec2 layerValue = imageLoad(unsignedLayers, pixelLayer).xy;" in generated_code
    )
    assert "int signedValue = imageLoad(signedVolume, voxel).x;" in generated_code
    assert (
        "imageStore(scalarFloat, pixel, vec4((scalarValue + amount)));"
        in generated_code
    )
    assert (
        "imageStore(rgFloat, pixel, vec4((rgValue + vec2(amount)), 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(unsignedScalar, pixel, uvec4((unsignedValue + unsignedAmount)));"
        in generated_code
    )
    assert (
        "imageStore(unsignedLayers, pixelLayer, uvec4((layerValue + uvec2(unsignedAmount)), 0u, 0u));"
        in generated_code
    )
    assert (
        "imageStore(signedVolume, voxel, ivec4((signedValue + signedAmount)));"
        in generated_code
    )
    assert "input.pixel" not in generated_code
    assert "input.voxel" not in generated_code
    assert "input.pixelLayer" not in generated_code


def test_opengl_direct_stage_image_compare_swap_use_input_members():
    shader = """
    shader DirectStageImageCompareSwap {
        uimage3D volumeCounters;
        iimage2DArray layerCounters;

        struct FSInput {
            ivec3 voxel @ TEXCOORD0;
            ivec3 pixelLayer @ TEXCOORD1;
            uint expected;
            uint replacement;
            int signedExpected;
            int signedReplacement;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                uint volumeOld = imageAtomicCompSwap(volumeCounters, input.voxel, input.expected, input.replacement);
                int layerOld = imageAtomicCompSwap(layerCounters, input.pixelLayer, input.signedExpected, input.signedReplacement);
                return vec4(float(volumeOld), float(layerOld), 0.0, 1.0);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(r32ui, binding = 0) uniform uimage3D volumeCounters;" in generated_code
    )
    assert (
        "layout(r32i, binding = 1) uniform iimage2DArray layerCounters;"
        in generated_code
    )
    assert (
        "uint volumeOld = imageAtomicCompSwap(volumeCounters, voxel, expected, replacement);"
        in generated_code
    )
    assert (
        "int layerOld = imageAtomicCompSwap(layerCounters, pixelLayer, signedExpected, signedReplacement);"
        in generated_code
    )
    assert "input.voxel" not in generated_code
    assert "input.pixelLayer" not in generated_code
    assert "input.expected" not in generated_code
    assert "input.signedExpected" not in generated_code


def test_opengl_storage_image_format_layouts():
    shader = """
    shader ImageFormats {
        sampler2D sampledColor;
        image2D colorOut;
        image3D volumeOut;
        image2DArray layerOut;
        iimage2D signedOut;
        iimage3D signedVolumeOut;
        iimage2DArray signedLayerOut;
        uimage2D unsignedOut;
        uimage3D unsignedVolumeOut;
        uimage2DArray unsignedLayerOut;
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D sampledColor;" in generated_code
    assert "layout(rgba32f, binding = 0) uniform image2D colorOut;" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image3D volumeOut;" in generated_code
    assert (
        "layout(rgba32f, binding = 2) uniform image2DArray layerOut;" in generated_code
    )
    assert "layout(r32i, binding = 3) uniform iimage2D signedOut;" in generated_code
    assert (
        "layout(r32i, binding = 4) uniform iimage3D signedVolumeOut;" in generated_code
    )
    assert (
        "layout(r32i, binding = 5) uniform iimage2DArray signedLayerOut;"
        in generated_code
    )
    assert "layout(r32ui, binding = 6) uniform uimage2D unsignedOut;" in generated_code
    assert (
        "layout(r32ui, binding = 7) uniform uimage3D unsignedVolumeOut;"
        in generated_code
    )
    assert (
        "layout(r32ui, binding = 8) uniform uimage2DArray unsignedLayerOut;"
        in generated_code
    )


def test_opengl_explicit_storage_image_format_layouts():
    shader = """
    shader ExplicitImageFormats {
        image2D halfColor @rgba16f;
        image3D normalizedVolume @ rgba8;
        image2DArray layerColor @format(rgba16f);
        iimage2D signedImage @ r32i;
        uimage2D counters @r32ui;
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(rgba16f, binding = 0) uniform image2D halfColor;" in generated_code
    assert (
        "layout(rgba8, binding = 1) uniform image3D normalizedVolume;" in generated_code
    )
    assert (
        "layout(rgba16f, binding = 2) uniform image2DArray layerColor;"
        in generated_code
    )
    assert "layout(r32i, binding = 3) uniform iimage2D signedImage;" in generated_code
    assert "layout(r32ui, binding = 4) uniform uimage2D counters;" in generated_code


def test_opengl_image_format_metadata_accepts_equivalent_aliases():
    shader = """
    shader EquivalentImageFormatAliases {
        image2D scalar @r32f @format(r32f);

        fragment {
            vec4 main(image2D target[2] @format(r32ui) @r32ui @binding(4)) @gl_FragColor {
                imageStore(target[1], ivec2(0, 0), 7u);
                float loaded = imageLoad(scalar, ivec2(0, 0));
                return vec4(loaded);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(r32f, binding = 0) uniform image2D scalar;" in generated_code
    assert "layout(r32ui, binding = 4) uniform uimage2D target[2];" in generated_code
    assert "imageStore(target[1], ivec2(0, 0), uvec4(7u));" in generated_code
    assert "float loaded = imageLoad(scalar, ivec2(0, 0)).x;" in generated_code
    assert "fragColor = vec4(loaded);" in generated_code
    assert "vec4 main(" not in generated_code


@pytest.mark.parametrize(
    ("shader", "message"),
    [
        (
            """
            shader DuplicateGlobalDirectImageFormat {
                image2D scalar @r32f @r32f;

                compute {
                    void main() {
                    }
                }
            }
            """,
            "Duplicate OpenGL image format metadata for 'scalar': @r32f",
        ),
        (
            """
            shader DuplicateGlobalFormatImageFormat {
                image2D scalar @format(r32f) @format(r32f);

                compute {
                    void main() {
                    }
                }
            }
            """,
            "Duplicate OpenGL image format metadata for 'scalar': format r32f",
        ),
        (
            """
            shader DuplicateStageParameterDirectImageFormat {
                fragment {
                    vec4 main(image2D target[2] @r32ui @r32ui) @gl_FragColor {
                        imageStore(target[1], ivec2(0, 0), 7u);
                        return vec4(1.0);
                    }
                }
            }
            """,
            "Duplicate OpenGL image format metadata for 'target': @r32ui",
        ),
        (
            """
            shader DuplicateStageParameterFormatImageFormat {
                fragment {
                    vec4 main(image2D target[2] @format(r32ui) @format(r32ui)) @gl_FragColor {
                        imageStore(target[1], ivec2(0, 0), 7u);
                        return vec4(1.0);
                    }
                }
            }
            """,
            "Duplicate OpenGL image format metadata for 'target': format r32ui",
        ),
    ],
)
def test_opengl_image_format_metadata_rejects_duplicate_metadata(shader, message):
    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


@pytest.mark.parametrize(
    ("shader", "message"),
    [
        (
            """
            shader InvalidGlobalImageFormat {
                image2D colorOut @format(rgba64f);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return imageLoad(colorOut, ivec2(uv));
                    }
                }
            }
            """,
            "Invalid OpenGL image format metadata for 'colorOut': "
            "format rgba64f is not a supported storage image format",
        ),
        (
            """
            shader InvalidParameterImageFormat {
                vec4 loadColor(image2D image @format(slot), ivec2 pixel) {
                    return imageLoad(image, pixel);
                }

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return vec4(0.0);
                    }
                }
            }
            """,
            "Invalid OpenGL image format metadata for 'image': "
            "format slot is not a supported storage image format",
        ),
        (
            """
            shader SampledTextureFormatMetadata {
                sampler2D colorTex @format(rgba8);

                fragment {
                    vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                        return texture(colorTex, uv);
                    }
                }
            }
            """,
            "Image format metadata on global variable 'colorTex' requires "
            "storage image type: @rgba8 on sampler2D",
        ),
        (
            """
            shader SampledTextureParameterFormatMetadata {
                vec4 sampleColor(sampler2D colorTex @format(rgba8), vec2 uv) {
                    return texture(colorTex, uv);
                }
            }
            """,
            "Image format metadata on parameter 'colorTex' of function "
            "'sampleColor' requires storage image type: @rgba8 on sampler2D",
        ),
    ],
)
def test_opengl_image_format_metadata_rejects_invalid_targets(shader, message):
    with pytest.raises(ValueError, match=message):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_explicit_scalar_image_formats():
    shader = """
    shader ExplicitScalarImageFormats {
        image2D scalarFloat @r32f;
        image3D signedVolume @ r32i;
        image2DArray unsignedLayers @format(r32ui);

        float touchFloat(image2D image @r32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        int touchSigned(image3D image @r32i, ivec3 voxel, int value) {
            int oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uint touchUnsigned(image2DArray image @format(r32ui), ivec3 pixelLayer, uint value) {
            uint oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float a = touchFloat(scalarFloat, ivec2(0, 1), 0.5);
                int b = touchSigned(signedVolume, ivec3(1, 2, 3), -4);
                uint c = touchUnsigned(unsignedLayers, ivec3(4, 5, 6), 7);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r32f, binding = 0) uniform image2D scalarFloat;" in generated_code
    assert "layout(r32i, binding = 1) uniform iimage3D signedVolume;" in generated_code
    assert (
        "layout(r32ui, binding = 2) uniform uimage2DArray unsignedLayers;"
        in generated_code
    )
    assert "float touchFloat(image2D image, ivec2 pixel, float value)" in generated_code
    assert "int touchSigned(iimage3D image, ivec3 voxel, int value)" in generated_code
    assert (
        "uint touchUnsigned(uimage2DArray image, ivec3 pixelLayer, uint value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(image, pixel).x;" in generated_code
    assert "imageStore(image, pixel, vec4((oldValue + value)));" in generated_code
    assert "int oldValue = imageLoad(image, voxel).x;" in generated_code
    assert "imageStore(image, voxel, ivec4((oldValue + value)));" in generated_code
    assert "uint oldValue = imageLoad(image, pixelLayer).x;" in generated_code
    assert "imageStore(image, pixelLayer, uvec4((oldValue + value)));" in generated_code


def test_opengl_explicit_rg_image_formats():
    shader = """
    shader ExplicitRGImageFormats {
        image2D rgFloat @rg32f;
        image3D rgSigned @format(rg32i);
        image2DArray rgUnsigned @ rg32ui;

        vec2 touchFloat(image2D image @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        ivec2 touchSigned(image3D image @rg32i, ivec3 voxel, ivec2 value) {
            ivec2 oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uvec2 touchUnsigned(image2DArray image @rg32ui, ivec3 pixelLayer, uvec2 value) {
            uvec2 oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(rg32f, binding = 0) uniform image2D rgFloat;" in generated_code
    assert "layout(rg32i, binding = 1) uniform iimage3D rgSigned;" in generated_code
    assert (
        "layout(rg32ui, binding = 2) uniform uimage2DArray rgUnsigned;"
        in generated_code
    )
    assert "vec2 touchFloat(image2D image, ivec2 pixel, vec2 value)" in generated_code
    assert (
        "ivec2 touchSigned(iimage3D image, ivec3 voxel, ivec2 value)" in generated_code
    )
    assert (
        "uvec2 touchUnsigned(uimage2DArray image, ivec3 pixelLayer, uvec2 value)"
        in generated_code
    )
    assert "vec2 oldValue = imageLoad(image, pixel).xy;" in generated_code
    assert "ivec2 oldValue = imageLoad(image, voxel).xy;" in generated_code
    assert "uvec2 oldValue = imageLoad(image, pixelLayer).xy;" in generated_code
    assert (
        "imageStore(image, pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(image, voxel, ivec4((oldValue + value), 0, 0));" in generated_code
    )
    assert (
        "imageStore(image, pixelLayer, uvec4((oldValue + value), 0u, 0u));"
        in generated_code
    )
    assert "layout(rg32i, binding = 1) uniform image3D" not in generated_code
    assert "layout(rg32ui, binding = 2) uniform image2DArray" not in generated_code
    assert "imageLoad(image, pixel).x;" not in generated_code
    assert "imageStore(image, pixel, (oldValue + value));" not in generated_code


def test_opengl_explicit_narrow_rg_image_formats():
    shader = """
    shader ExplicitNarrowRGImageFormats {
        image2D rg8Float @rg8;
        image2D rg8Snorm @rg8_snorm;
        image3D rg16Float @format(rg16);
        image2D rg16Snorm @rg16_snorm;
        image2DArray rg16Half @ rg16f;
        image2D rg8Signed @rg8i;
        image3D rg16Signed @format(rg16i);
        image2D rg8Unsigned @rg8ui;
        image2DArray rg16Unsigned @format(rg16ui);

        vec2 touchFloat(image2D image @rg8, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchHalf(image2DArray image @rg16f, ivec3 pixelLayer, vec2 value) {
            vec2 oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }

        ivec2 touchSigned(image3D image @rg16i, ivec3 voxel, ivec2 value) {
            ivec2 oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uvec2 touchUnsigned(image2D image @rg8ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(rg8, binding = 0) uniform image2D rg8Float;" in generated_code
    assert "layout(rg8_snorm, binding = 1) uniform image2D rg8Snorm;" in generated_code
    assert "layout(rg16, binding = 2) uniform image3D rg16Float;" in generated_code
    assert (
        "layout(rg16_snorm, binding = 3) uniform image2D rg16Snorm;" in generated_code
    )
    assert "layout(rg16f, binding = 4) uniform image2DArray rg16Half;" in generated_code
    assert "layout(rg8i, binding = 5) uniform iimage2D rg8Signed;" in generated_code
    assert "layout(rg16i, binding = 6) uniform iimage3D rg16Signed;" in generated_code
    assert "layout(rg8ui, binding = 7) uniform uimage2D rg8Unsigned;" in generated_code
    assert (
        "layout(rg16ui, binding = 8) uniform uimage2DArray rg16Unsigned;"
        in generated_code
    )
    assert "vec2 touchFloat(image2D image, ivec2 pixel, vec2 value)" in generated_code
    assert (
        "vec2 touchHalf(image2DArray image, ivec3 pixelLayer, vec2 value)"
        in generated_code
    )
    assert (
        "ivec2 touchSigned(iimage3D image, ivec3 voxel, ivec2 value)" in generated_code
    )
    assert (
        "uvec2 touchUnsigned(uimage2D image, ivec2 pixel, uvec2 value)"
        in generated_code
    )
    assert "vec2 oldValue = imageLoad(image, pixel).xy;" in generated_code
    assert "vec2 oldValue = imageLoad(image, pixelLayer).xy;" in generated_code
    assert "ivec2 oldValue = imageLoad(image, voxel).xy;" in generated_code
    assert "uvec2 oldValue = imageLoad(image, pixel).xy;" in generated_code
    assert (
        "imageStore(image, pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(image, pixelLayer, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(image, voxel, ivec4((oldValue + value), 0, 0));" in generated_code
    )
    assert (
        "imageStore(image, pixel, uvec4((oldValue + value), 0u, 0u));" in generated_code
    )
    assert "layout(rg8i, binding = 5) uniform image2D" not in generated_code
    assert "layout(rg8ui, binding = 7) uniform image2D" not in generated_code
    assert "imageLoad(image, pixel).x;" not in generated_code
    assert "imageStore(image, pixel, (oldValue + value));" not in generated_code


def test_opengl_explicit_rgba_float_image_formats():
    shader = """
    shader ExplicitRGBAFloatFormats {
        image2D rgba8Color @rgba8;
        image2D rgba8Snorm @rgba8_snorm;
        image3D rgba16Color @format(rgba16);
        image2D rgba16Snorm @rgba16_snorm;
        image2DArray rgba16Half @ rgba16f;
        image3D rgba32Float @format(rgba32f);

        vec4 touchColor(image2D image @rgba8, ivec2 pixel, vec4 value) {
            vec4 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        vec4 touchHalf(image2DArray image @rgba16f, ivec3 pixelLayer, vec4 value) {
            vec4 oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }

        vec4 touchFloat(image3D image @rgba32f, ivec3 voxel, vec4 value) {
            vec4 oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        vec4 typedOverride(iimage2D image @rgba16f, ivec2 pixel, vec4 value) {
            vec4 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(rgba8, binding = 0) uniform image2D rgba8Color;" in generated_code
    assert (
        "layout(rgba8_snorm, binding = 1) uniform image2D rgba8Snorm;" in generated_code
    )
    assert "layout(rgba16, binding = 2) uniform image3D rgba16Color;" in generated_code
    assert (
        "layout(rgba16_snorm, binding = 3) uniform image2D rgba16Snorm;"
        in generated_code
    )
    assert (
        "layout(rgba16f, binding = 4) uniform image2DArray rgba16Half;"
        in generated_code
    )
    assert "layout(rgba32f, binding = 5) uniform image3D rgba32Float;" in generated_code
    assert "vec4 touchColor(image2D image, ivec2 pixel, vec4 value)" in generated_code
    assert (
        "vec4 touchHalf(image2DArray image, ivec3 pixelLayer, vec4 value)"
        in generated_code
    )
    assert "vec4 touchFloat(image3D image, ivec3 voxel, vec4 value)" in generated_code
    assert (
        "vec4 typedOverride(image2D image, ivec2 pixel, vec4 value)" in generated_code
    )
    assert "vec4 oldValue = imageLoad(image, pixel);" in generated_code
    assert "vec4 oldValue = imageLoad(image, pixelLayer);" in generated_code
    assert "vec4 oldValue = imageLoad(image, voxel);" in generated_code
    assert "imageStore(image, pixel, (oldValue + value));" in generated_code
    assert "imageStore(image, pixelLayer, (oldValue + value));" in generated_code
    assert "imageStore(image, voxel, (oldValue + value));" in generated_code
    assert "vec4 typedOverride(iimage2D image" not in generated_code
    assert "imageLoad(image, pixel).x" not in generated_code
    assert "imageLoad(image, pixel).xy" not in generated_code
    assert "imageStore(image, pixel, vec4(" not in generated_code


def test_opengl_formatted_image_arrays_preserve_format_metadata():
    shader = """
    shader FormattedImageArrays {
        image2D counters @r32ui[2];
        image2D rgPairs @rg16f[3];
        image3D rgbaVolumes @rgba16f[2];
        image2D afterCounters @r32ui;
        sampler2D sampled;

        uint touchCounters(image2D images[2] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[1], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[3] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec4 touchVolumes(image3D images[2] @rgba16f, ivec3 voxel, vec4 value) {
            vec4 oldValue = imageLoad(images[1], voxel);
            imageStore(images[0], voxel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
                vec2 b = touchPairs(rgPairs, ivec2(2, 3), vec2(0.5));
                vec4 c = touchVolumes(rgbaVolumes, ivec3(1, 2, 3), vec4(1.0));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r32ui, binding = 0) uniform uimage2D counters[2];" in generated_code
    assert "layout(rg16f, binding = 2) uniform image2D rgPairs[3];" in generated_code
    assert (
        "layout(rgba16f, binding = 5) uniform image3D rgbaVolumes[2];" in generated_code
    )
    assert (
        "layout(r32ui, binding = 7) uniform uimage2D afterCounters;" in generated_code
    )
    assert "layout(binding = 0) uniform sampler2D sampled;" in generated_code
    assert (
        "uint touchCounters(uimage2D images[2], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "vec2 touchPairs(image2D images[3], ivec2 pixel, vec2 value)" in generated_code
    )
    assert (
        "vec4 touchVolumes(image3D images[2], ivec3 voxel, vec4 value)"
        in generated_code
    )
    assert "uint oldValue = imageLoad(images[1], pixel).x;" in generated_code
    assert "imageStore(images[0], pixel, uvec4((oldValue + value)));" in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "imageStore(images[1], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert "vec4 oldValue = imageLoad(images[1], voxel);" in generated_code
    assert "imageStore(images[0], voxel, (oldValue + value));" in generated_code
    assert "layout(r32ui, binding = 0) uniform image2D counters" not in generated_code
    assert "layout(rg16f, binding = 2) uniform uimage2D rgPairs" not in generated_code
    assert "uint oldValue = imageLoad(images[1], pixel);" not in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).x;" not in generated_code


def test_opengl_rg_image_arrays_respect_scalar_and_vector_context():
    shader = """
    shader RGImageArrayContext {
        image2D rgFloatImages @rg32f[3];
        uimage2D rgUnsignedImages @rg32ui[2];

        float scalarFloat(image2D images[3] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[1], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloat(image2D images[3] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsigned(uimage2D images[2] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[1], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsigned(uimage2D images[2] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[1], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloat(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloat(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsigned(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsigned(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[3];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 3) uniform uimage2D rgUnsignedImages[2];"
        in generated_code
    )
    assert (
        "float scalarFloat(image2D images[3], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "vec2 vectorFloat(image2D images[3], ivec2 pixel, vec2 value)" in generated_code
    )
    assert (
        "uint scalarUnsigned(uimage2D images[2], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uvec2 vectorUnsigned(uimage2D images[2], ivec2 pixel, uvec2 value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[1], pixel).x;" in generated_code
    assert (
        "imageStore(images[0], pixel, vec4((oldValue + value), 0.0, 0.0, 0.0));"
        in generated_code
    )
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "imageStore(images[1], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert "uint oldValue = imageLoad(images[1], pixel).x;" in generated_code
    assert (
        "imageStore(images[0], pixel, uvec4((oldValue + value), 0u, 0u, 0u));"
        in generated_code
    )
    assert "uvec2 oldValue = imageLoad(images[1], pixel).xy;" in generated_code
    assert (
        "imageStore(images[0], pixel, uvec4((oldValue + value), 0u, 0u));"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[1], pixel);" not in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).x;" not in generated_code
    assert "uint oldValue = imageLoad(images[1], pixel);" not in generated_code
    assert "uvec2 oldValue = imageLoad(images[1], pixel).x;" not in generated_code


def test_opengl_inferred_rg_image_arrays_respect_scalar_context():
    shader = """
    shader RGImageArrayInferredScalarContext {
        const int COUNT = 3;
        const int LAYER = COUNT - 1;
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[COUNT];
        image2D afterImages @rg32f;

        float scalarFloat(image2D images[] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloat(image2D images[] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsigned(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsigned(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloat(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloat(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsigned(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsigned(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "const int COUNT = 3;" in generated_code
    assert "const int LAYER = (COUNT - 1);" in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[3];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 3) uniform uimage2D rgUnsignedImages[COUNT];"
        in generated_code
    )
    assert "layout(rg32f, binding = 6) uniform image2D afterImages;" in generated_code
    assert (
        "float scalarFloat(image2D images[3], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "vec2 vectorFloat(image2D images[3], ivec2 pixel, vec2 value)" in generated_code
    )
    assert (
        "uint scalarUnsigned(uimage2D images[COUNT], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uvec2 vectorUnsigned(uimage2D images[COUNT], ivec2 pixel, uvec2 value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[LAYER], pixel).x;" in generated_code
    assert (
        "imageStore(images[0], pixel, vec4((oldValue + value), 0.0, 0.0, 0.0));"
        in generated_code
    )
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "imageStore(images[1], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert "uint oldValue = imageLoad(images[LAYER], pixel).x;" in generated_code
    assert (
        "imageStore(images[0], pixel, uvec4((oldValue + value), 0u, 0u, 0u));"
        in generated_code
    )
    assert "uvec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "imageStore(images[1], pixel, uvec4((oldValue + value), 0u, 0u));"
        in generated_code
    )
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[];"
        not in generated_code
    )
    assert (
        "layout(rg32f, binding = 3) uniform image2D afterImages;" not in generated_code
    )
    assert "float oldValue = imageLoad(images[LAYER], pixel);" not in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).x;" not in generated_code
    assert "uint oldValue = imageLoad(images[LAYER], pixel);" not in generated_code
    assert "uvec2 oldValue = imageLoad(images[2], pixel).x;" not in generated_code


def test_opengl_transitive_rg_image_arrays_share_call_site_size():
    shader = """
    shader TransitiveRGImageArrayScalarContext {
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];
        image2D afterImages @rg32f;

        float scalarFloatDeep(image2D images[] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        float scalarFloatMid(image2D images[] @rg32f, ivec2 pixel, float value) {
            return scalarFloatDeep(images, pixel, value);
        }

        vec2 vectorFloatDeep(image2D images[] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloatMid(image2D images[] @rg32f, ivec2 pixel, vec2 value) {
            return vectorFloatDeep(images, pixel, value);
        }

        uint scalarUnsignedDeep(uimage2D images[] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsignedMid(uimage2D images[] @rg32ui, ivec2 pixel, uint value) {
            return scalarUnsignedDeep(images, pixel, value);
        }

        uvec2 vectorUnsignedDeep(uimage2D images[] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsignedMid(uimage2D images[] @rg32ui, ivec2 pixel, uvec2 value) {
            return vectorUnsignedDeep(images, pixel, value);
        }

        compute {
            void main() {
                float sf = scalarFloatMid(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloatMid(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsignedMid(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsignedMid(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 4) uniform uimage2D rgUnsignedImages[4];"
        in generated_code
    )
    assert "layout(rg32f, binding = 8) uniform image2D afterImages;" in generated_code
    assert (
        "float scalarFloatDeep(image2D images[4], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "float scalarFloatMid(image2D images[4], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "vec2 vectorFloatDeep(image2D images[4], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert (
        "vec2 vectorFloatMid(image2D images[4], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedDeep(uimage2D images[4], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uvec2 vectorUnsignedDeep(uimage2D images[4], ivec2 pixel, uvec2 value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert "uint oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert "uvec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "imageStore(images[1], pixel, vec4((oldValue + value), 0.0, 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(images[0], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(images[1], pixel, uvec4((oldValue + value), 0u, 0u, 0u));"
        in generated_code
    )
    assert (
        "imageStore(images[0], pixel, uvec4((oldValue + value), 0u, 0u));"
        in generated_code
    )
    assert "vec2 vectorFloatDeep(image2D images[3]" not in generated_code
    assert "uvec2 vectorUnsignedDeep(uimage2D images[3]" not in generated_code


def test_opengl_fixed_rg_image_array_parameters_size_unsized_globals():
    shader = """
    shader FixedParamRGImageArrayContext {
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];

        float scalarFloatFixed(image2D images[4] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloatFixed(image2D images[4] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsignedFixed(uimage2D images[4] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsignedFixed(uimage2D images[4] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloatFixed(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloatFixed(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsignedFixed(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsignedFixed(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 4) uniform uimage2D rgUnsignedImages[4];"
        in generated_code
    )
    assert (
        "float scalarFloatFixed(image2D images[4], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "vec2 vectorFloatFixed(image2D images[4], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(uimage2D images[4], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uvec2 vectorUnsignedFixed(uimage2D images[4], ivec2 pixel, uvec2 value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert "uint oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert "uvec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[];"
        not in generated_code
    )
    assert (
        "layout(rg32ui, binding = 1) uniform uimage2D rgUnsignedImages[];"
        not in generated_code
    )


def test_opengl_const_sized_rg_image_array_parameters_size_unsized_globals():
    shader = """
    shader FixedConstParamRGImageArrayContext {
        const int COUNT = 4;
        const int LAST = COUNT - 1;
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];

        float scalarFloatFixed(image2D images[COUNT] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[LAST], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloatFixed(image2D images[COUNT] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsignedFixed(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[LAST], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsignedFixed(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloatFixed(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloatFixed(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsignedFixed(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsignedFixed(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "const int COUNT = 4;" in generated_code
    assert "const int LAST = (COUNT - 1);" in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 4) uniform uimage2D rgUnsignedImages[4];"
        in generated_code
    )
    assert (
        "float scalarFloatFixed(image2D images[COUNT], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "vec2 vectorFloatFixed(image2D images[COUNT], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(uimage2D images[COUNT], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uvec2 vectorUnsignedFixed(uimage2D images[COUNT], ivec2 pixel, uvec2 value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[LAST], pixel).x;" in generated_code
    assert "uint oldValue = imageLoad(images[LAST], pixel).x;" in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert "uvec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert "float scalarFloatFixed(image2D images[4]" not in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[];"
        not in generated_code
    )


def test_opengl_expr_sized_rg_image_array_parameters_size_unsized_globals():
    shader = """
    shader FixedExprParamRGImageArrayContext {
        const int COUNT = 3;
        const int UINT_COUNT = 2;
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];

        float scalarFloatFixed(image2D images[(COUNT + 1)] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[COUNT], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloatFixed(image2D images[(COUNT + 1)] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsignedFixed(uimage2D images[(UINT_COUNT * 2)] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsignedFixed(uimage2D images[(UINT_COUNT * 2)] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloatFixed(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloatFixed(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsignedFixed(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsignedFixed(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "const int COUNT = 3;" in generated_code
    assert "const int UINT_COUNT = 2;" in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 4) uniform uimage2D rgUnsignedImages[4];"
        in generated_code
    )
    assert (
        "float scalarFloatFixed(image2D images[(COUNT + 1)], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "vec2 vectorFloatFixed(image2D images[(COUNT + 1)], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(uimage2D images[(UINT_COUNT * 2)], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uvec2 vectorUnsignedFixed(uimage2D images[(UINT_COUNT * 2)], ivec2 pixel, uvec2 value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[COUNT], pixel).x;" in generated_code
    assert "uint oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert "uvec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert "float scalarFloatFixed(image2D images[4]" not in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[];"
        not in generated_code
    )


def test_opengl_conflicting_fixed_rg_image_array_sizes_raise():
    shader = """
    shader ConflictingFixedRGImageArrayContext {
        image2D rgFloatImages @rg32f[];

        float touchFour(image2D images[4] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[3], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchThree(image2D images[3] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float a = touchFour(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 b = touchThree(rgFloatImages, ivec2(2, 3), vec2(1.0));
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_direct_rg_image_array_index_conflicts_with_fixed_parameter_size():
    shader = """
    shader DirectIndexFixedConflict {
        image2D rgFloatImages @rg32f[];

        float touchFour(image2D images[4] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[3], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float direct = imageLoad(rgFloatImages[5], pixel);
                float helper = touchFour(rgFloatImages, pixel, direct);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_direct_rg_image_array_index_within_fixed_parameter_size():
    shader = """
    shader DirectIndexWithinFixedSize {
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];

        float touchFour(image2D images[4] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[3], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uint touchUnsignedFour(uimage2D images[4] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float directFloat = imageLoad(rgFloatImages[2], pixel);
                uint directUint = imageLoad(rgUnsignedImages[1], pixel);
                float helperFloat = touchFour(rgFloatImages, pixel, directFloat);
                uint helperUint = touchUnsignedFour(rgUnsignedImages, pixel, directUint);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 4) uniform uimage2D rgUnsignedImages[4];"
        in generated_code
    )
    assert (
        "float touchFour(image2D images[4], ivec2 pixel, float value)" in generated_code
    )
    assert (
        "uint touchUnsignedFour(uimage2D images[4], ivec2 pixel, uint value)"
        in generated_code
    )
    assert "float directFloat = imageLoad(rgFloatImages[2], pixel).x;" in generated_code
    assert (
        "uint directUint = imageLoad(rgUnsignedImages[1], pixel).x;" in generated_code
    )
    assert "float oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert "uint oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[3];"
        not in generated_code
    )
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[];"
        not in generated_code
    )


def test_opengl_fixed_rg_image_array_global_conflicts_with_fixed_parameter_size():
    shader = """
    shader FixedGlobalToMismatchedFixedHelper {
        image2D rgFloatImages @rg32f[4];

        float touchThree(image2D images[3] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float a = touchThree(rgFloatImages, ivec2(0, 1), 0.25);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_fixed_rg_image_array_global_widens_unsized_parameter_size():
    shader = """
    shader FixedGlobalToUnsizedHelper {
        image2D rgFloatImages @rg32f[4];

        float touchUnsized(image2D images[] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float a = touchUnsized(rgFloatImages, ivec2(0, 1), 0.25);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert (
        "float touchUnsized(image2D images[4], ivec2 pixel, float value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[2], pixel).x;" in generated_code
    assert (
        "float touchUnsized(image2D images[3], ivec2 pixel, float value)"
        not in generated_code
    )


def test_opengl_fixed_rg_image_array_global_direct_index_out_of_bounds_raises():
    shader = """
    shader FixedGlobalDirectIndexOutOfBounds {
        image2D rgFloatImages @rg32f[4];

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float value = imageLoad(rgFloatImages[4], pixel);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_fixed_rg_image_array_parameter_direct_index_out_of_bounds_raises():
    shader = """
    shader FixedParameterDirectIndexOutOfBounds {
        float touch(image2D images[4] @rg32f, ivec2 pixel) {
            return imageLoad(images[4], pixel);
        }

        compute {
            void main() {
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_fixed_rg_image_array_global_const_index_out_of_bounds_raises():
    shader = """
    shader FixedGlobalConstIndexOutOfBounds {
        const int COUNT = 4;
        image2D rgFloatImages @rg32f[4];

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float value = imageLoad(rgFloatImages[COUNT], pixel);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_fixed_rg_image_array_parameter_const_index_out_of_bounds_raises():
    shader = """
    shader FixedParameterConstIndexOutOfBounds {
        const int COUNT = 4;

        float touch(image2D images[4] @rg32f, ivec2 pixel) {
            return imageLoad(images[COUNT], pixel);
        }

        compute {
            void main() {
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_fixed_rg_image_array_const_index_within_bounds_generates():
    shader = """
    shader FixedConstIndexWithinBounds {
        const int COUNT = 4;
        image2D rgFloatImages @rg32f[4];

        float touch(image2D images[4] @rg32f, ivec2 pixel) {
            return imageLoad(images[COUNT - 1], pixel);
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float direct = imageLoad(rgFloatImages[COUNT - 1], pixel);
                float helper = touch(rgFloatImages, pixel);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "const int COUNT = 4;" in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert "float touch(image2D images[4], ivec2 pixel)" in generated_code
    assert "return imageLoad(images[(COUNT - 1)], pixel).x;" in generated_code
    assert (
        "float direct = imageLoad(rgFloatImages[(COUNT - 1)], pixel).x;"
        in generated_code
    )
    assert "float touch(image2D images[5], ivec2 pixel)" not in generated_code


def test_opengl_fixed_rg_image_array_shadowed_const_index_stays_dynamic():
    shader = """
    shader FixedShadowedConstIndex {
        const int COUNT = 4;
        image2D rgFloatImages @rg32f[4];

        float touch(image2D images[4] @rg32f, ivec2 pixel) {
            int COUNT = 0;
            return imageLoad(images[COUNT], pixel);
        }

        compute {
            void main() {
                int COUNT = 0;
                ivec2 pixel = ivec2(0, 1);
                float direct = imageLoad(rgFloatImages[COUNT], pixel);
                float helper = touch(rgFloatImages, pixel);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 3
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert "float touch(image2D images[4], ivec2 pixel)" in generated_code
    assert "return imageLoad(images[COUNT], pixel).x;" in generated_code
    assert "float direct = imageLoad(rgFloatImages[COUNT], pixel).x;" in generated_code
    assert "float touch(image2D images[5], ivec2 pixel)" not in generated_code


def test_opengl_transitive_rg_image_array_shadowed_const_index_stays_dynamic():
    shader = """
    shader TransitiveShadowedConstIndex {
        const int COUNT = 4;
        image2D rgFloatImages @rg32f[4];

        float leaf(image2D images[] @rg32f, ivec2 pixel) {
            int COUNT = 0;
            return imageLoad(images[COUNT], pixel);
        }

        float passThrough(image2D images[] @rg32f, ivec2 pixel) {
            int COUNT = 0;
            float sampled = imageLoad(images[COUNT], pixel);
            return sampled + leaf(images, pixel);
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float value = passThrough(rgFloatImages, pixel);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 4
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert "float leaf(image2D images[4], ivec2 pixel)" in generated_code
    assert "float passThrough(image2D images[4], ivec2 pixel)" in generated_code
    assert "return imageLoad(images[COUNT], pixel).x;" in generated_code
    assert "float sampled = imageLoad(images[COUNT], pixel).x;" in generated_code
    assert "float leaf(image2D images[5], ivec2 pixel)" not in generated_code
    assert "float passThrough(image2D images[5], ivec2 pixel)" not in generated_code


def test_opengl_transitive_rg_image_array_unshadowed_const_index_conflict_raises():
    shader = """
    shader TransitiveUnshadowedConstIndexConflict {
        const int COUNT = 4;
        image2D rgFloatImages @rg32f[4];

        float leaf(image2D images[] @rg32f, ivec2 pixel) {
            return imageLoad(images[COUNT], pixel);
        }

        float passThrough(image2D images[] @rg32f, ivec2 pixel) {
            int COUNT = 0;
            return leaf(images, pixel);
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float value = passThrough(rgFloatImages, pixel);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_shadowed_rg_image_array_constant_keeps_scalar_context():
    shader = """
    shader ShadowedRGImageArrayScalarContext {
        const int LAYER = 3;
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];
        image2D afterImages @rg32f;

        float scalarFloat(image2D images[] @rg32f, ivec2 pixel, float value) {
            int LAYER = 0;
            float oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsigned(uimage2D images[] @rg32ui, ivec2 pixel, uint value) {
            int LAYER = 0;
            uint oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int LAYER = 3;" in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[1];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 1) uniform uimage2D rgUnsignedImages[1];"
        in generated_code
    )
    assert "layout(rg32f, binding = 2) uniform image2D afterImages;" in generated_code
    assert (
        "float scalarFloat(image2D images[1], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "uint scalarUnsigned(uimage2D images[1], ivec2 pixel, uint value)"
        in generated_code
    )
    assert "int LAYER = 0;" in generated_code
    assert "float oldValue = imageLoad(images[LAYER], pixel).x;" in generated_code
    assert "uint oldValue = imageLoad(images[LAYER], pixel).x;" in generated_code
    assert (
        "imageStore(images[0], pixel, vec4((oldValue + value), 0.0, 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(images[0], pixel, uvec4((oldValue + value), 0u, 0u, 0u));"
        in generated_code
    )
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];"
        not in generated_code
    )
    assert (
        "layout(rg32f, binding = 4) uniform image2D afterImages;" not in generated_code
    )
    assert "float oldValue = imageLoad(images[LAYER], pixel);" not in generated_code
    assert "uint oldValue = imageLoad(images[LAYER], pixel);" not in generated_code


def test_opengl_formatted_image_arrays_preserve_expression_sizes():
    shader = """
    shader ExprFormattedImageArrays {
        image2D counters @r32ui[(1 + 1) * 2];
        image2D rgPairs @rg16f[+3];
        image2D afterCounters @r32ui;
        sampler2D sampled;

        uint touchCounters(image2D images[(1 + 1) * 2] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[+3] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
                vec2 b = touchPairs(rgPairs, ivec2(2, 3), vec2(0.5));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[((1 + 1) * 2)];"
        in generated_code
    )
    assert "layout(rg16f, binding = 4) uniform image2D rgPairs[(+3)];" in generated_code
    assert (
        "layout(r32ui, binding = 7) uniform uimage2D afterCounters;" in generated_code
    )
    assert "layout(binding = 0) uniform sampler2D sampled;" in generated_code
    assert (
        "uint touchCounters(uimage2D images[((1 + 1) * 2)], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "vec2 touchPairs(image2D images[(+3)], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert "uint oldValue = imageLoad(images[2], pixel).x;" in generated_code
    assert "imageStore(images[1], pixel, uvec4((oldValue + value)));" in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "imageStore(images[1], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert (
        "uint a = touchCounters__glsl_images_counters(ivec2(1, 2), 3);"
        in generated_code
    )
    assert (
        "vec2 b = touchPairs__glsl_images_rgPairs(ivec2(2, 3), vec2(0.5));"
        in generated_code
    )
    assert (
        "layout(r32ui, binding = 4) uniform uimage2D afterCounters;"
        not in generated_code
    )
    assert "layout(r32ui, binding = 0) uniform image2D counters" not in generated_code
    assert "layout(rg16f, binding = 4) uniform uimage2D rgPairs" not in generated_code
    assert "uint oldValue = imageLoad(images[2], pixel);" not in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).x;" not in generated_code


def test_opengl_unsized_formatted_image_arrays_preserve_format_metadata():
    shader = """
    shader UnsizedFormattedImageArrays {
        image2D counters @r32ui[];
        image2D rgPairs @rg16f[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[0], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[0], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r32ui, binding = 0) uniform uimage2D counters[1];" in generated_code
    assert "layout(rg16f, binding = 1) uniform image2D rgPairs[1];" in generated_code
    assert (
        "layout(r32ui, binding = 2) uniform uimage2D afterCounters;" in generated_code
    )
    assert (
        "uint touchCounters(uimage2D images[1], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "vec2 touchPairs(image2D images[1], ivec2 pixel, vec2 value)" in generated_code
    )
    assert "uint oldValue = imageLoad(images[0], pixel).x;" in generated_code
    assert "imageStore(images[0], pixel, uvec4((oldValue + value)));" in generated_code
    assert "vec2 oldValue = imageLoad(images[0], pixel).xy;" in generated_code
    assert (
        "imageStore(images[0], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert "layout(r32ui, binding = 0) uniform image2D counters" not in generated_code
    assert "layout(rg16f, binding = 1) uniform uimage2D rgPairs" not in generated_code
    assert "uint oldValue = imageLoad(images[0], pixel);" not in generated_code
    assert "vec2 oldValue = imageLoad(images[0], pixel).x;" not in generated_code


def test_opengl_formatted_image_arrays_infer_named_constant_size():
    shader = """
    shader ConstSizedFormattedImageArrays {
        const int COUNT = 3;
        const int LAYER = COUNT - 1;
        image2D counters @r32ui[COUNT];
        image2D rgPairs @rg16f[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[COUNT] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
                vec2 b = touchPairs(rgPairs, ivec2(2, 3), vec2(0.5));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int COUNT = 3;" in generated_code
    assert "const int LAYER = (COUNT - 1);" in generated_code
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[COUNT];" in generated_code
    )
    assert "layout(rg16f, binding = 3) uniform image2D rgPairs[3];" in generated_code
    assert (
        "layout(r32ui, binding = 6) uniform uimage2D afterCounters;" in generated_code
    )
    assert (
        "uint touchCounters(uimage2D images[COUNT], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "vec2 touchPairs(image2D images[3], ivec2 pixel, vec2 value)" in generated_code
    )
    assert "uint oldValue = imageLoad(images[LAYER], pixel).x;" in generated_code
    assert "imageStore(images[1], pixel, uvec4((oldValue + value)));" in generated_code
    assert "vec2 oldValue = imageLoad(images[LAYER], pixel).xy;" in generated_code
    assert (
        "imageStore(images[0], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert (
        "uint a = touchCounters__glsl_images_counters(ivec2(1, 2), 3);"
        in generated_code
    )
    assert (
        "vec2 b = touchPairs__glsl_images_rgPairs(ivec2(2, 3), vec2(0.5));"
        in generated_code
    )
    assert "layout(rg16f, binding = 3) uniform image2D rgPairs[];" not in generated_code
    assert (
        "layout(r32ui, binding = 2) uniform uimage2D afterCounters;"
        not in generated_code
    )
    assert "layout(r32ui, binding = 0) uniform image2D counters" not in generated_code
    assert "layout(rg16f, binding = 3) uniform uimage2D rgPairs" not in generated_code
    assert "uint oldValue = imageLoad(images[LAYER], pixel);" not in generated_code
    assert "vec2 oldValue = imageLoad(images[LAYER], pixel).x;" not in generated_code


def test_opengl_formatted_image_arrays_ignore_shadowed_local_constant():
    shader = """
    shader ShadowedImageConstIndex {
        const int LAYER = 3;
        image2D counters @r32ui[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            int LAYER = 0;
            uint oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int LAYER = 3;" in generated_code
    assert "layout(r32ui, binding = 0) uniform uimage2D counters[1];" in generated_code
    assert (
        "layout(r32ui, binding = 1) uniform uimage2D afterCounters;" in generated_code
    )
    assert (
        "uint touchCounters(uimage2D images[1], ivec2 pixel, uint value)"
        in generated_code
    )
    assert "int LAYER = 0;" in generated_code
    assert "uint oldValue = imageLoad(images[LAYER], pixel).x;" in generated_code
    assert "imageStore(images[0], pixel, uvec4((oldValue + value)));" in generated_code
    assert (
        "uint a = touchCounters__glsl_images_counters(ivec2(1, 2), 3);"
        in generated_code
    )
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[4];" not in generated_code
    )
    assert (
        "layout(r32ui, binding = 4) uniform uimage2D afterCounters;"
        not in generated_code
    )
    assert "uint oldValue = imageLoad(images[LAYER], pixel);" not in generated_code


def test_opengl_formatted_image_arrays_infer_transitive_helper_size():
    shader = """
    shader TransitiveFormattedImageArrays {
        image2D counters @r32ui[];
        image2D rgPairs @rg16f[];
        image2D afterCounters @r32ui;

        uint touchCountersDeep(image2D images[] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uint touchCountersMid(image2D images[] @r32ui, ivec2 pixel, uint value) {
            return touchCountersDeep(images, pixel, value);
        }

        vec2 touchPairsDeep(image2D images[] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairsMid(image2D images[] @rg16f, ivec2 pixel, vec2 value) {
            return touchPairsDeep(images, pixel, value);
        }

        compute {
            void main() {
                uint a = touchCountersMid(counters, ivec2(1, 2), 3);
                vec2 b = touchPairsMid(rgPairs, ivec2(2, 3), vec2(0.5));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(r32ui, binding = 0) uniform uimage2D counters[4];" in generated_code
    assert "layout(rg16f, binding = 4) uniform image2D rgPairs[3];" in generated_code
    assert (
        "layout(r32ui, binding = 7) uniform uimage2D afterCounters;" in generated_code
    )
    assert (
        "uint touchCountersDeep(uimage2D images[4], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint touchCountersMid(uimage2D images[4], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "vec2 touchPairsDeep(image2D images[3], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert (
        "vec2 touchPairsMid(image2D images[3], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert "uint oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert "imageStore(images[1], pixel, uvec4((oldValue + value)));" in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "imageStore(images[0], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert "return touchCountersDeep(images, pixel, value);" in generated_code
    assert "return touchPairsDeep(images, pixel, value);" in generated_code
    assert (
        "uint a = touchCountersMid__glsl_images_counters(ivec2(1, 2), 3);"
        in generated_code
    )
    assert (
        "vec2 b = touchPairsMid__glsl_images_rgPairs(ivec2(2, 3), vec2(0.5));"
        in generated_code
    )
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[];" not in generated_code
    )
    assert "layout(rg16f, binding = 4) uniform image2D rgPairs[];" not in generated_code
    assert (
        "layout(r32ui, binding = 2) uniform uimage2D afterCounters;"
        not in generated_code
    )
    assert "uint oldValue = imageLoad(images[3], pixel);" not in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).x;" not in generated_code


def test_opengl_formatted_image_arrays_ignore_unsupported_indices():
    dynamic_shader = """
    shader DynamicOnlyFormattedImageArrays {
        image2D counters @r32ui[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, int layer, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[layer], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, 0, ivec2(1, 2), 3);
            }
        }
    }
    """
    negative_shader = """
    shader NegativeIndexedFormattedImageArrays {
        image2D counters @r32ui[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[-1], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
            }
        }
    }
    """

    dynamic_code = GLSLCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = GLSLCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "layout(r32ui, binding = 0) uniform uimage2D counters[1];" in dynamic_code
    assert "layout(r32ui, binding = 1) uniform uimage2D afterCounters;" in dynamic_code
    assert (
        "uint touchCounters(uimage2D images[1], int layer, ivec2 pixel, uint value)"
        in dynamic_code
    )
    assert "uint oldValue = imageLoad(images[layer], pixel).x;" in dynamic_code
    assert "imageStore(images[0], pixel, uvec4((oldValue + value)));" in dynamic_code
    assert (
        "uint a = touchCounters__glsl_images_counters(0, ivec2(1, 2), 3);"
        in dynamic_code
    )
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[2];" not in dynamic_code
    )
    assert (
        "layout(r32ui, binding = 2) uniform uimage2D afterCounters;" not in dynamic_code
    )
    assert "uint oldValue = imageLoad(images[layer], pixel);" not in dynamic_code

    assert "layout(r32ui, binding = 0) uniform uimage2D counters[1];" in negative_code
    assert "layout(r32ui, binding = 1) uniform uimage2D afterCounters;" in negative_code
    assert (
        "uint touchCounters(uimage2D images[1], ivec2 pixel, uint value)"
        in negative_code
    )
    assert "uint oldValue = imageLoad(images[0], pixel).x;" in negative_code
    assert "imageStore(images[0], pixel, uvec4((oldValue + value)));" in negative_code
    assert (
        "uint a = touchCounters__glsl_images_counters(ivec2(1, 2), 3);" in negative_code
    )
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[0];" not in negative_code
    )
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D afterCounters;"
        not in negative_code
    )
    assert "uint oldValue = imageLoad(images[(-1)], pixel)" not in negative_code


def test_opengl_formatted_image_arrays_ignore_function_call_indices():
    shader = """
    shader FunctionIndexedFormattedImageArrays {
        image2D counters @r32ui[];
        image2D afterCounters @r32ui;

        int getLayer() {
            return 0;
        }

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[getLayer()], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(r32ui, binding = 0) uniform uimage2D counters[1];" in generated_code
    assert (
        "layout(r32ui, binding = 1) uniform uimage2D afterCounters;" in generated_code
    )
    assert "int getLayer()" in generated_code
    assert (
        "uint touchCounters(uimage2D images[1], ivec2 pixel, uint value)"
        in generated_code
    )
    assert "uint oldValue = imageLoad(images[getLayer()], pixel).x;" in generated_code
    assert "imageStore(images[0], pixel, uvec4((oldValue + value)));" in generated_code
    assert (
        "uint a = touchCounters__glsl_images_counters(ivec2(1, 2), 3);"
        in generated_code
    )
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[];" not in generated_code
    )
    assert (
        "layout(r32ui, binding = 2) uniform uimage2D afterCounters;"
        not in generated_code
    )
    assert "uint oldValue = imageLoad(images[getLayer()], pixel);" not in generated_code


def test_opengl_formatted_image_arrays_infer_local_constant_alias_size():
    shader = """
    shader LocalConstAliasFormattedImageArrays {
        const int GLOBAL = 2;
        image2D counters @r32ui[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            const int LOCAL = GLOBAL + 1;
            uint oldValue = imageLoad(images[LOCAL], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int GLOBAL = 2;" in generated_code
    assert "layout(r32ui, binding = 0) uniform uimage2D counters[4];" in generated_code
    assert (
        "layout(r32ui, binding = 4) uniform uimage2D afterCounters;" in generated_code
    )
    assert (
        "uint touchCounters(uimage2D images[4], ivec2 pixel, uint value)"
        in generated_code
    )
    assert "const int LOCAL = (GLOBAL + 1);" in generated_code
    assert "uint oldValue = imageLoad(images[LOCAL], pixel).x;" in generated_code
    assert "imageStore(images[0], pixel, uvec4((oldValue + value)));" in generated_code
    assert (
        "uint a = touchCounters__glsl_images_counters(ivec2(1, 2), 3);"
        in generated_code
    )
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[];" not in generated_code
    )
    assert (
        "layout(r32ui, binding = 1) uniform uimage2D afterCounters;"
        not in generated_code
    )
    assert "    int LOCAL = (GLOBAL + 1);" not in generated_code
    assert "uint oldValue = imageLoad(images[LOCAL], pixel);" not in generated_code


def test_opengl_explicit_scalar_float_image_formats():
    shader = """
    shader ExplicitScalarFloatFormats {
        image2D normalized8 @r8;
        image3D normalized16 @format(r16);
        image2DArray halfLayers @ r16f;
        image2D signedNormalized @r8_snorm;

        float touchR8(image2D image @r8, ivec2 pixel, float value) {
            float oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        float touchR16(image3D image @r16, ivec3 voxel, float value) {
            float oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        float touchR16f(image2DArray image @r16f, ivec3 pixelLayer, float value) {
            float oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r8, binding = 0) uniform image2D normalized8;" in generated_code
    assert "layout(r16, binding = 1) uniform image3D normalized16;" in generated_code
    assert (
        "layout(r16f, binding = 2) uniform image2DArray halfLayers;" in generated_code
    )
    assert (
        "layout(r8_snorm, binding = 3) uniform image2D signedNormalized;"
        in generated_code
    )
    assert "float touchR8(image2D image, ivec2 pixel, float value)" in generated_code
    assert "float touchR16(image3D image, ivec3 voxel, float value)" in generated_code
    assert (
        "float touchR16f(image2DArray image, ivec3 pixelLayer, float value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(image, pixel).x;" in generated_code
    assert "float oldValue = imageLoad(image, voxel).x;" in generated_code
    assert "float oldValue = imageLoad(image, pixelLayer).x;" in generated_code
    assert "imageStore(image, pixel, vec4((oldValue + value)));" in generated_code
    assert "imageStore(image, voxel, vec4((oldValue + value)));" in generated_code
    assert "imageStore(image, pixelLayer, vec4((oldValue + value)));" in generated_code
    assert "imageStore(image, pixel, (oldValue + value));" not in generated_code
    assert "imageLoad(image, pixel);" not in generated_code


def test_opengl_explicit_narrow_integer_image_formats():
    shader = """
    shader ExplicitNarrowIntegerFormats {
        image2D signed8 @r8i;
        image3D signed16 @format(r16i);
        image2D unsigned8 @ r8ui;
        image2DArray unsigned16 @format(r16ui);

        int loadSigned8(image2D image @r8i, ivec2 pixel, int value) {
            int oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        int loadSigned16(image3D image @r16i, ivec3 voxel, int value) {
            int oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uint loadUnsigned8(image2D image @r8ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        uint loadUnsigned16(image2DArray image @r16ui, ivec3 pixelLayer, uint value) {
            uint oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r8i, binding = 0) uniform iimage2D signed8;" in generated_code
    assert "layout(r16i, binding = 1) uniform iimage3D signed16;" in generated_code
    assert "layout(r8ui, binding = 2) uniform uimage2D unsigned8;" in generated_code
    assert (
        "layout(r16ui, binding = 3) uniform uimage2DArray unsigned16;" in generated_code
    )
    assert "int loadSigned8(iimage2D image, ivec2 pixel, int value)" in generated_code
    assert "int loadSigned16(iimage3D image, ivec3 voxel, int value)" in generated_code
    assert (
        "uint loadUnsigned8(uimage2D image, ivec2 pixel, uint value)" in generated_code
    )
    assert (
        "uint loadUnsigned16(uimage2DArray image, ivec3 pixelLayer, uint value)"
        in generated_code
    )
    assert "int oldValue = imageLoad(image, pixel).x;" in generated_code
    assert "int oldValue = imageLoad(image, voxel).x;" in generated_code
    assert "uint oldValue = imageLoad(image, pixel).x;" in generated_code
    assert "uint oldValue = imageLoad(image, pixelLayer).x;" in generated_code
    assert "imageStore(image, pixel, ivec4((oldValue + value)));" in generated_code
    assert "imageStore(image, voxel, ivec4((oldValue + value)));" in generated_code
    assert "imageStore(image, pixel, uvec4((oldValue + value)));" in generated_code
    assert "imageStore(image, pixelLayer, uvec4((oldValue + value)));" in generated_code
    assert "layout(r8i, binding = 0) uniform image2D" not in generated_code
    assert "layout(r8ui, binding = 2) uniform image2D" not in generated_code
    assert "imageStore(image, pixel, (oldValue + value));" not in generated_code


def test_opengl_explicit_integer_image_formats_use_integer_image_types():
    shader = """
    shader ExplicitAtomicFormats {
        image2D unsignedCounters @r32ui;
        image2D signedCounters @r32i;
        image3D unsignedVolume @format(r32ui);
        image2DArray signedLayers @format(r32i);

        uint addUnsigned(image2D image @r32ui, ivec2 pixel, uint value) {
            return imageAtomicAdd(image, pixel, value);
        }

        int minSigned(image2D image @r32i, ivec2 pixel, int value) {
            return imageAtomicMin(image, pixel, value);
        }

        uint swapVolume(image3D image @r32ui, ivec3 voxel, uint expected, uint value) {
            return imageAtomicCompSwap(image, voxel, expected, value);
        }

        int exchangeLayer(image2DArray image @r32i, ivec3 pixelLayer, int value) {
            return imageAtomicExchange(image, pixelLayer, value);
        }

        compute {
            void main() {
                uint a = addUnsigned(unsignedCounters, ivec2(0, 0), 1);
                int b = minSigned(signedCounters, ivec2(1, 0), -1);
                uint c = swapVolume(unsignedVolume, ivec3(0, 1, 2), 3, 4);
                int d = exchangeLayer(signedLayers, ivec3(2, 3, 4), 5);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(r32ui, binding = 0) uniform uimage2D unsignedCounters;"
        in generated_code
    )
    assert (
        "layout(r32i, binding = 1) uniform iimage2D signedCounters;" in generated_code
    )
    assert (
        "layout(r32ui, binding = 2) uniform uimage3D unsignedVolume;" in generated_code
    )
    assert (
        "layout(r32i, binding = 3) uniform iimage2DArray signedLayers;"
        in generated_code
    )
    assert "uint addUnsigned(uimage2D image, ivec2 pixel, uint value)" in generated_code
    assert "int minSigned(iimage2D image, ivec2 pixel, int value)" in generated_code
    assert (
        "uint swapVolume(uimage3D image, ivec3 voxel, uint expected, uint value)"
        in generated_code
    )
    assert (
        "int exchangeLayer(iimage2DArray image, ivec3 pixelLayer, int value)"
        in generated_code
    )
    assert (
        f"return {glsl_image_atomic_parameter_diagnostic('imageAtomicAdd', 'uimage2D', '0u')};"
        in generated_code
    )
    assert (
        f"return {glsl_image_atomic_parameter_diagnostic('imageAtomicMin', 'iimage2D', '0')};"
        in generated_code
    )
    assert (
        f"return {glsl_image_atomic_parameter_diagnostic('imageAtomicCompSwap', 'uimage3D', '0u')};"
        in generated_code
    )
    assert (
        f"return {glsl_image_atomic_parameter_diagnostic('imageAtomicExchange', 'iimage2DArray', '0')};"
        in generated_code
    )


def test_opengl_explicit_vector_integer_image_formats():
    shader = """
    shader ExplicitVectorIntegerFormats {
        image2D unsignedColor @rgba32ui;
        image3D signedVolume @format(rgba32i);
        image2DArray unsignedLayers @ rgba16ui;

        uvec4 touchUnsigned(image2D image @rgba32ui, ivec2 pixel, uvec4 value) {
            uvec4 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        ivec4 touchSigned(image3D image @format(rgba32i), ivec3 voxel, ivec4 value) {
            ivec4 oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uvec4 touchLayers(image2DArray image @rgba16ui, ivec3 pixelLayer, uvec4 value) {
            uvec4 oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(rgba32ui, binding = 0) uniform uimage2D unsignedColor;"
        in generated_code
    )
    assert (
        "layout(rgba32i, binding = 1) uniform iimage3D signedVolume;" in generated_code
    )
    assert (
        "layout(rgba16ui, binding = 2) uniform uimage2DArray unsignedLayers;"
        in generated_code
    )
    assert (
        "uvec4 touchUnsigned(uimage2D image, ivec2 pixel, uvec4 value)"
        in generated_code
    )
    assert (
        "ivec4 touchSigned(iimage3D image, ivec3 voxel, ivec4 value)" in generated_code
    )
    assert (
        "uvec4 touchLayers(uimage2DArray image, ivec3 pixelLayer, uvec4 value)"
        in generated_code
    )
    assert "uvec4 oldValue = imageLoad(image, pixel);" in generated_code
    assert "ivec4 oldValue = imageLoad(image, voxel);" in generated_code
    assert "uvec4 oldValue = imageLoad(image, pixelLayer);" in generated_code
    assert "imageStore(image, pixel, (oldValue + value));" in generated_code
    assert "imageStore(image, voxel, (oldValue + value));" in generated_code
    assert "imageStore(image, pixelLayer, (oldValue + value));" in generated_code
    assert "layout(rgba32ui, binding = 0) uniform image2D" not in generated_code
    assert "layout(rgba32i, binding = 1) uniform image3D" not in generated_code
    assert "imageLoad(image, pixel).x" not in generated_code
    assert "uvec4((oldValue + value))" not in generated_code
    assert "ivec4((oldValue + value))" not in generated_code


def test_opengl_integer_image_atomic_add():
    shader = """
    shader AtomicImages {
        uimage2D counters;
        iimage2D signedCounters;

        uint addCounter(uimage2D image, ivec2 pixel, uint value) {
            uint previous = imageAtomicAdd(image, pixel, value);
            return previous;
        }

        int addSignedCounter(iimage2D image, ivec2 pixel, int value) {
            int previous = imageAtomicAdd(image, pixel, value);
            return previous;
        }

        compute {
            void main() {
                uint oldValue = addCounter(counters, ivec2(0, 1), 2);
                int oldSigned = addSignedCounter(signedCounters, ivec2(2, 3), -1);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r32ui, binding = 0) uniform uimage2D counters;" in generated_code
    assert (
        "layout(r32i, binding = 1) uniform iimage2D signedCounters;" in generated_code
    )
    assert "uint addCounter(uimage2D image, ivec2 pixel, uint value)" in generated_code
    assert (
        "int addSignedCounter(iimage2D image, ivec2 pixel, int value)" in generated_code
    )
    assert (
        "uint previous = "
        f"{glsl_image_atomic_parameter_diagnostic('imageAtomicAdd', 'uimage2D', '0u')};"
        in generated_code
    )
    assert (
        "int previous = "
        f"{glsl_image_atomic_parameter_diagnostic('imageAtomicAdd', 'iimage2D', '0')};"
        in generated_code
    )


def test_opengl_integer_image_atomic_operations():
    shader = """
    shader AtomicOps {
        uimage2D counters;
        iimage2D signedCounters;

        uint unsignedOps(uimage2D image, ivec2 pixel, uint value) {
            uint minValue = imageAtomicMin(image, pixel, value);
            uint maxValue = imageAtomicMax(image, pixel, value);
            uint andValue = imageAtomicAnd(image, pixel, value);
            uint orValue = imageAtomicOr(image, pixel, value);
            uint xorValue = imageAtomicXor(image, pixel, value);
            uint exchanged = imageAtomicExchange(image, pixel, value);
            return minValue + maxValue + andValue + orValue + xorValue + exchanged;
        }

        int signedOps(iimage2D image, ivec2 pixel, int value) {
            int minValue = imageAtomicMin(image, pixel, value);
            int maxValue = imageAtomicMax(image, pixel, value);
            int exchanged = imageAtomicExchange(image, pixel, value);
            return minValue + maxValue + exchanged;
        }

        compute {
            void main() {
                uint unsignedResult = unsignedOps(counters, ivec2(0, 1), 3);
                int signedResult = signedOps(signedCounters, ivec2(2, 3), -4);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    for operation in [
        "imageAtomicMin",
        "imageAtomicMax",
        "imageAtomicAnd",
        "imageAtomicOr",
        "imageAtomicXor",
        "imageAtomicExchange",
    ]:
        assert (
            glsl_image_atomic_parameter_diagnostic(operation, "uimage2D", "0u")
            in generated_code
        )

    for operation in [
        "imageAtomicMin",
        "imageAtomicMax",
        "imageAtomicExchange",
    ]:
        assert (
            glsl_image_atomic_parameter_diagnostic(operation, "iimage2D", "0")
            in generated_code
        )

    assert "uint unsignedOps(uimage2D image, ivec2 pixel, uint value)" in generated_code
    assert "int signedOps(iimage2D image, ivec2 pixel, int value)" in generated_code


def test_opengl_integer_image_atomic_compare_swap():
    shader = """
    shader AtomicCompareSwap {
        uimage2D counters;
        iimage2D signedCounters;

        uint compareUnsigned(uimage2D image, ivec2 pixel, uint expected, uint replacement) {
            uint previous = imageAtomicCompSwap(image, pixel, expected, replacement);
            return previous;
        }

        int compareSigned(iimage2D image, ivec2 pixel, int expected, int replacement) {
            int previous = imageAtomicCompSwap(image, pixel, expected, replacement);
            return previous;
        }

        compute {
            void main() {
                uint oldValue = compareUnsigned(counters, ivec2(0, 1), 2, 3);
                int oldSigned = compareSigned(signedCounters, ivec2(2, 3), -1, 4);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "uint compareUnsigned(uimage2D image, ivec2 pixel, uint expected, uint replacement)"
        in generated_code
    )
    assert (
        "int compareSigned(iimage2D image, ivec2 pixel, int expected, int replacement)"
        in generated_code
    )
    assert (
        "uint previous = "
        f"{glsl_image_atomic_parameter_diagnostic('imageAtomicCompSwap', 'uimage2D', '0u')};"
        in generated_code
    )
    assert (
        "int previous = "
        f"{glsl_image_atomic_parameter_diagnostic('imageAtomicCompSwap', 'iimage2D', '0')};"
        in generated_code
    )


def test_opengl_integer_image_dimension_atomics():
    shader = """
    shader TypedImageDimensions {
        uimage3D volumeCounters;
        iimage3D signedVolumeCounters;
        iimage2DArray layerCounters;
        uimage2DArray unsignedLayerCounters;

        uint touchVolume(uimage3D image, ivec3 voxel, uint value) {
            uint oldValue = imageAtomicAdd(image, voxel, value);
            uint swapped = imageAtomicCompSwap(image, voxel, oldValue, value);
            return oldValue + swapped;
        }

        int touchLayers(iimage2DArray image, ivec3 pixelLayer, int value) {
            int oldValue = imageAtomicMin(image, pixelLayer, value);
            int swapped = imageAtomicCompSwap(image, pixelLayer, oldValue, value);
            return oldValue + swapped;
        }

        compute {
            void main() {
                uint volumeResult = touchVolume(volumeCounters, ivec3(0, 1, 2), 3);
                int layerResult = touchLayers(layerCounters, ivec3(4, 5, 6), -7);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(r32ui, binding = 0) uniform uimage3D volumeCounters;" in generated_code
    )
    assert (
        "layout(r32i, binding = 1) uniform iimage3D signedVolumeCounters;"
        in generated_code
    )
    assert (
        "layout(r32i, binding = 2) uniform iimage2DArray layerCounters;"
        in generated_code
    )
    assert (
        "layout(r32ui, binding = 3) uniform uimage2DArray unsignedLayerCounters;"
        in generated_code
    )
    assert "uint touchVolume(uimage3D image, ivec3 voxel, uint value)" in generated_code
    assert (
        "int touchLayers(iimage2DArray image, ivec3 pixelLayer, int value)"
        in generated_code
    )
    assert (
        "uint oldValue = "
        f"{glsl_image_atomic_parameter_diagnostic('imageAtomicAdd', 'uimage3D', '0u')};"
        in generated_code
    )
    assert (
        "uint swapped = "
        f"{glsl_image_atomic_parameter_diagnostic('imageAtomicCompSwap', 'uimage3D', '0u')};"
        in generated_code
    )
    assert (
        "int oldValue = "
        f"{glsl_image_atomic_parameter_diagnostic('imageAtomicMin', 'iimage2DArray', '0')};"
        in generated_code
    )
    assert (
        "int swapped = "
        f"{glsl_image_atomic_parameter_diagnostic('imageAtomicCompSwap', 'iimage2DArray', '0')};"
        in generated_code
    )


def test_opengl_integer_image_scalar_load_store():
    shader = """
    shader IntegerImageLoadStore {
        uimage2D counters;
        iimage3D signedVolume;
        uimage2DArray layerCounters;

        uint touch2D(uimage2D image, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        int touch3D(iimage3D image, ivec3 voxel, int value) {
            int oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uint touchLayer(uimage2DArray image, ivec3 pixelLayer, uint value) {
            uint oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touch2D(counters, ivec2(0, 1), 2);
                int b = touch3D(signedVolume, ivec3(1, 2, 3), -4);
                uint c = touchLayer(layerCounters, ivec3(4, 5, 6), 7);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r32ui, binding = 0) uniform uimage2D counters;" in generated_code
    assert "layout(r32i, binding = 1) uniform iimage3D signedVolume;" in generated_code
    assert (
        "layout(r32ui, binding = 2) uniform uimage2DArray layerCounters;"
        in generated_code
    )
    assert "uint oldValue = imageLoad(image, pixel).x;" in generated_code
    assert "imageStore(image, pixel, uvec4((oldValue + value)));" in generated_code
    assert "int oldValue = imageLoad(image, voxel).x;" in generated_code
    assert "imageStore(image, voxel, ivec4((oldValue + value)));" in generated_code
    assert "uint oldValue = imageLoad(image, pixelLayer).x;" in generated_code
    assert "imageStore(image, pixelLayer, uvec4((oldValue + value)));" in generated_code


def test_opengl_storage_image_size_queries_use_image_size():
    shader = """
    shader StorageImageSizeQueries {
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;
        uimage2D uintImage;
        iimage3D intVolume;

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        ivec2 queryImage2D(image2D image) {
            return textureSize(image, 0) + imageSize(image);
        }

        ivec3 queryImage3D(image3D image) {
            return textureSize(image, 0) + imageSize(image);
        }

        ivec3 queryImageArray(image2DArray image) {
            return textureSize(image, 0) + imageSize(image);
        }

        ivec2 queryUintImage(uimage2D image) {
            return textureSize(image, 0) + imageSize(image);
        }

        ivec3 queryIntVolume(iimage3D image) {
            return textureSize(image, 0) + imageSize(image);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                ivec2 a = textureSize(colorImage, 0) + imageSize(colorImage);
                ivec3 b = textureSize(volumeImage, 0) + imageSize(volumeImage);
                ivec3 c = textureSize(layerImage, 0) + imageSize(layerImage);
                ivec2 d = textureSize(uintImage, 0) + imageSize(uintImage);
                ivec3 e = textureSize(intVolume, 0) + imageSize(intVolume);
                return vec4(a.x + b.x + c.z + d.x + e.z);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(rgba32f, binding = 0) uniform image2D colorImage;" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image3D volumeImage;" in generated_code
    assert (
        "layout(rgba32f, binding = 2) uniform image2DArray layerImage;"
        in generated_code
    )
    assert "layout(r32ui, binding = 3) uniform uimage2D uintImage;" in generated_code
    assert "layout(r32i, binding = 4) uniform iimage3D intVolume;" in generated_code
    assert "ivec2 queryImage2D(image2D image)" in generated_code
    assert "ivec3 queryImageArray(image2DArray image)" in generated_code
    assert "return (imageSize(image) + imageSize(image));" in generated_code
    assert (
        "ivec2 a = (imageSize(colorImage) + imageSize(colorImage));" in generated_code
    )
    assert (
        "ivec3 b = (imageSize(volumeImage) + imageSize(volumeImage));" in generated_code
    )
    assert (
        "ivec3 c = (imageSize(layerImage) + imageSize(layerImage));" in generated_code
    )
    assert "ivec2 d = (imageSize(uintImage) + imageSize(uintImage));" in generated_code
    assert "ivec3 e = (imageSize(intVolume) + imageSize(intVolume));" in generated_code
    assert "textureSize(" not in generated_code


def test_opengl_texture_query_resource_descriptors():
    codegen = GLSLCodeGen()
    codegen.texture_variable_types = {
        "colorImage": "image2D",
        "colorMap": "sampler2D",
        "msTex": "sampler2DMS",
        "msArray": "sampler2DMSArray",
    }

    assert codegen.texture_query_resource_descriptor("colorImage") == {
        "texture_type": "image2D",
        "storage_image": True,
        "multisample": False,
        "size_descriptor": {"function_name": "imageSize"},
    }
    assert codegen.texture_query_resource_descriptor("msArray") == {
        "texture_type": "sampler2DMSArray",
        "storage_image": False,
        "multisample": True,
        "size_descriptor": {"function_name": "textureSize"},
    }
    assert codegen.texture_query_resource_descriptor("colorMap") == {
        "texture_type": "sampler2D",
        "storage_image": False,
        "multisample": False,
        "size_descriptor": {"function_name": "textureSize"},
    }
    assert codegen.texture_query_size_descriptor("sampler2D") == {
        "function_name": "textureSize"
    }
    assert codegen.texture_query_size_descriptor(None) is None
    assert (
        codegen.texture_query_levels_expression("colorImage")
        == "/* unsupported GLSL texture query: textureQueryLevels on image2D */ 0"
    )
    assert codegen.texture_query_levels_expression("msTex") == "1"
    assert codegen.texture_query_levels_expression("colorMap") is None
    assert (
        codegen.texture_samples_expression("textureSamples", "colorMap")
        == "/* unsupported GLSL texture samples query: requires multisample sampler */ 0"
    )
    assert (
        codegen.texture_samples_expression("imageSamples", "msTex")
        == "textureSamples(msTex)"
    )
    assert codegen.texture_samples_expression("textureSamples", "msTex") is None
    assert (
        codegen.texture_size_query_expression("colorImage") == "imageSize(colorImage)"
    )
    assert codegen.texture_size_query_expression("colorMap") is None


def test_opengl_storage_image_levels_and_lod_queries_emit_diagnostics():
    shader = """
    shader StorageImageInvalidQueries {
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;
        uimage2D uintImage;
        iimage3D intVolume;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvw @ TEXCOORD1;
        };

        int imageLevels(image2D image) {
            return textureQueryLevels(image);
        }

        vec2 imageLod(image2D image, vec2 uv) {
            return textureQueryLod(image, uv);
        }

        int volumeLevels(image3D image) {
            return textureQueryLevels(image);
        }

        vec2 volumeLod(image3D image, vec3 uvw) {
            return textureQueryLod(image, uvw);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                int levels = textureQueryLevels(colorImage)
                    + textureQueryLevels(volumeImage)
                    + textureQueryLevels(layerImage)
                    + textureQueryLevels(uintImage)
                    + textureQueryLevels(intVolume);
                vec2 lod = textureQueryLod(colorImage, input.uv)
                    + textureQueryLod(volumeImage, input.uvw)
                    + imageLod(colorImage, input.uv)
                    + volumeLod(volumeImage, input.uvw);
                return vec4(float(levels) + lod.x);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        generated_code.count(
            "/* unsupported GLSL texture query: textureQueryLevels on image2D */ 0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture query: textureQueryLevels on image3D */ 0"
        )
        == 2
    )
    assert (
        "/* unsupported GLSL texture query: textureQueryLevels on image2DArray */ 0"
        in generated_code
    )
    assert (
        "/* unsupported GLSL texture query: textureQueryLevels on uimage2D */ 0"
        in generated_code
    )
    assert (
        "/* unsupported GLSL texture query: textureQueryLevels on iimage3D */ 0"
        in generated_code
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture query: textureQueryLod on image2D */ vec2(0.0)"
        )
        == 3
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture query: textureQueryLod on image3D */ vec2(0.0)"
        )
        == 3
    )
    assert "textureQueryLevels(" not in generated_code
    assert "textureQueryLod(" not in generated_code


def test_opengl_texture_query_functions():
    shader = """
    shader TextureQueries {
        sampler2D colorMap;
        sampler2DArray layerMap;
        sampler2DMS msMap;
        sampler linearSampler;

        struct VSOutput {
            vec2 uv;
        };

        ivec2 query2D(sampler2D tex, sampler s, vec2 uv) {
            ivec2 size = textureSize(tex, 1);
            int levels = textureQueryLevels(tex);
            vec2 lod = textureQueryLod(tex, s, uv);
            return size + ivec2(levels) + ivec2(lod);
        }

        ivec3 queryArray(sampler2DArray tex) {
            return textureSize(tex, 0);
        }

        ivec2 queryMs(sampler2DMS tex) {
            return textureSize(tex);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                ivec2 q = query2D(colorMap, linearSampler, uv);
                ivec3 qa = queryArray(layerMap);
                ivec2 qm = queryMs(msMap);
                return vec4(q.x + qa.z + qm.x);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler2DArray layerMap;" in generated_code
    assert "layout(binding = 2) uniform sampler2DMS msMap;" in generated_code
    assert "ivec2 query2D(sampler2D tex, vec2 uv)" in generated_code
    assert "ivec2 size = textureSize(tex, 1);" in generated_code
    assert "int levels = textureQueryLevels(tex);" in generated_code
    assert "vec2 lod = textureQueryLod(tex, uv);" in generated_code
    assert "ivec3 queryArray(sampler2DArray tex)" in generated_code
    assert "return textureSize(tex, 0);" in generated_code
    assert "ivec2 queryMs(sampler2DMS tex)" in generated_code
    assert "return textureSize(tex);" in generated_code
    assert "linearSampler" not in generated_code
    assert "textureQueryLod(tex, s, uv)" not in generated_code


def test_opengl_multisample_texture_samples_queries_keep_builtin():
    shader = """
    shader MultisampleSamplesQuery {
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler2DMS msTextures[4];
        sampler2DMSArray msArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        int querySamples(sampler2DMS tex, sampler2DMSArray texArray) {
            return textureSamples(tex) + textureSamples(texArray);
        }

        int queryArraySamples(sampler2DMS textures[], sampler2DMSArray arrays[], int layer) {
            return textureSamples(textures[2]) + textureSamples(arrays[layer]);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return float(querySamples(msTex, msArray)
                    + queryArraySamples(msTextures, msArrays, input.layer));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DMS msTex;" in generated_code
    assert "layout(binding = 1) uniform sampler2DMSArray msArray;" in generated_code
    assert "layout(binding = 2) uniform sampler2DMS msTextures[4];" in generated_code
    assert "layout(binding = 6) uniform sampler2DMSArray msArrays[4];" in generated_code
    assert (
        "int querySamples(sampler2DMS tex, sampler2DMSArray texArray)" in generated_code
    )
    assert "return (textureSamples(tex) + textureSamples(texArray));" in generated_code
    assert (
        "int queryArraySamples(sampler2DMS textures[4], sampler2DMSArray arrays[4], int layer)"
        in generated_code
    )
    assert (
        "return (textureSamples(textures[2]) + textureSamples(arrays[layer]));"
        in generated_code
    )
    assert (
        "querySamples(msTex, msArray) + queryArraySamples(msTextures, msArrays, layer)"
        in generated_code
    )
    assert "input." not in generated_code


def test_opengl_integer_multisample_queries_and_texel_fetches_lower():
    shader = """
    shader IntegerMultisampleQueries {
        isampler2DMS signedMs;
        usampler2DMSArray unsignedMsArray;
        isampler2DMS signedTextures[4];
        usampler2DMSArray unsignedArrays[4];

        struct FSInput {
            ivec2 pixel @ TEXCOORD0;
            ivec3 pixelLayer @ TEXCOORD1;
            int sampleIndex;
            int layer @ TEXCOORD2;
        };

        ivec4 fetchSigned(isampler2DMS tex, ivec2 pixel, int sampleIndex) {
            return texelFetch(tex, pixel, sampleIndex);
        }

        uvec4 fetchUnsigned(usampler2DMSArray tex, ivec3 pixelLayer, int sampleIndex) {
            return texelFetch(tex, pixelLayer, sampleIndex);
        }

        int querySigned(isampler2DMS tex) {
            return textureSamples(tex) + imageSamples(tex) + textureQueryLevels(tex);
        }

        int queryArrays(isampler2DMS textures[], usampler2DMSArray arrays[], int layer) {
            return textureSamples(textures[layer])
                + imageSamples(arrays[2])
                + textureQueryLevels(textures[1])
                + textureQueryLevels(arrays[layer]);
        }

        fragment {
            ivec4 main(FSInput input) @ COLOR0 {
                uvec4 unsignedFetched = fetchUnsigned(
                    unsignedMsArray,
                    input.pixelLayer,
                    input.sampleIndex
                );
                int directSamples = textureSamples(signedMs) + imageSamples(unsignedMsArray);
                int directLevels = textureQueryLevels(signedMs) + textureQueryLevels(unsignedMsArray);
                return fetchSigned(signedMs, input.pixel, input.sampleIndex)
                    + ivec4(texelFetch(signedTextures[input.layer], input.pixel, input.sampleIndex))
                    + ivec4(texelFetch(unsignedArrays[2], input.pixelLayer, input.sampleIndex))
                    + ivec4(unsignedFetched)
                    + ivec4(directSamples + directLevels + querySigned(signedMs)
                        + queryArrays(signedTextures, unsignedArrays, input.layer));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform isampler2DMS signedMs;" in generated_code
    assert (
        "layout(binding = 1) uniform usampler2DMSArray unsignedMsArray;"
        in generated_code
    )
    assert (
        "layout(binding = 2) uniform isampler2DMS signedTextures[4];" in generated_code
    )
    assert (
        "layout(binding = 6) uniform usampler2DMSArray unsignedArrays[4];"
        in generated_code
    )
    assert (
        "ivec4 fetchSigned(isampler2DMS tex, ivec2 pixel, int sampleIndex)"
        in generated_code
    )
    assert (
        "uvec4 fetchUnsigned(usampler2DMSArray tex, ivec3 pixelLayer, int sampleIndex)"
        in generated_code
    )
    assert "int querySigned(isampler2DMS tex)" in generated_code
    assert (
        "int queryArrays(isampler2DMS textures[4], usampler2DMSArray arrays[4], int layer)"
        in generated_code
    )
    assert "return texelFetch(tex, pixel, sampleIndex);" in generated_code
    assert "return texelFetch(tex, pixelLayer, sampleIndex);" in generated_code
    assert "textureSamples(tex)" in generated_code
    assert "textureSamples(textures[layer])" in generated_code
    assert "textureSamples(arrays[2])" in generated_code
    assert (
        "int directSamples = (textureSamples(signedMs) + textureSamples(unsignedMsArray));"
        in generated_code
    )
    assert "int directLevels = (1 + 1);" in generated_code
    assert "texelFetch(signedTextures[layer], pixel, sampleIndex)" in generated_code
    assert "texelFetch(unsignedArrays[2], pixelLayer, sampleIndex)" in generated_code
    assert "textureQueryLevels(" not in generated_code
    assert "imageSamples(" not in generated_code
    assert "unsupported GLSL texture samples query" not in generated_code
    assert "input." not in generated_code


def test_opengl_multisample_image_samples_queries_use_texture_samples_builtin():
    shader = """
    shader MultisampleImageSamplesQuery {
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler2DMS msTextures[4];
        sampler2DMSArray msArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        int querySamples(sampler2DMS tex, sampler2DMSArray texArray) {
            return imageSamples(tex) + imageSamples(texArray);
        }

        int queryArraySamples(sampler2DMS textures[], sampler2DMSArray arrays[], int layer) {
            return imageSamples(textures[2]) + imageSamples(arrays[layer]);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return float(imageSamples(msTex)
                    + imageSamples(msArray)
                    + querySamples(msTex, msArray)
                    + queryArraySamples(msTextures, msArrays, input.layer));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DMS msTex;" in generated_code
    assert "layout(binding = 1) uniform sampler2DMSArray msArray;" in generated_code
    assert "layout(binding = 2) uniform sampler2DMS msTextures[4];" in generated_code
    assert "layout(binding = 6) uniform sampler2DMSArray msArrays[4];" in generated_code
    assert (
        "int querySamples(sampler2DMS tex, sampler2DMSArray texArray)" in generated_code
    )
    assert "return (textureSamples(tex) + textureSamples(texArray));" in generated_code
    assert (
        "int queryArraySamples(sampler2DMS textures[4], sampler2DMSArray arrays[4], int layer)"
        in generated_code
    )
    assert (
        "return (textureSamples(textures[2]) + textureSamples(arrays[layer]));"
        in generated_code
    )
    assert "textureSamples(msTex) + textureSamples(msArray)" in generated_code
    assert "queryArraySamples(msTextures, msArrays, layer)" in generated_code
    assert "imageSamples(" not in generated_code
    assert "unsupported GLSL texture samples query" not in generated_code
    assert "input." not in generated_code


def test_opengl_integer_multisample_invalid_sampling_and_offsets_emit_typed_diagnostics():
    shader = """
    shader IntegerMultisampleDiagnostics {
        isampler2DMS signedMs;
        usampler2DMSArray unsignedMsArray;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            ivec2 pixel @ TEXCOORD2;
            ivec3 pixelLayer @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
            int sampleIndex;
        };

        ivec4 invalidSigned(
            isampler2DMS tex,
            sampler s,
            vec2 uv,
            ivec2 pixel,
            int sampleIndex,
            ivec2 offset
        ) {
            ivec4 sampled = texture(tex, s, uv);
            ivec4 lod = textureLod(tex, uv, 0.0);
            ivec4 gathered = textureGather(tex, s, uv);
            ivec4 offsetFetch = texelFetchOffset(tex, pixel, sampleIndex, offset);
            return sampled + lod + gathered + offsetFetch;
        }

        uvec4 invalidUnsigned(
            usampler2DMSArray tex,
            sampler s,
            vec3 uvLayer,
            ivec3 pixelLayer,
            int sampleIndex,
            ivec2 offset
        ) {
            uvec4 sampled = texture(tex, s, uvLayer);
            uvec4 lod = textureLod(tex, uvLayer, 0.0);
            uvec4 gathered = textureGather(tex, s, uvLayer);
            uvec4 offsetFetch = texelFetchOffset(tex, pixelLayer, sampleIndex, offset);
            return sampled + lod + gathered + offsetFetch;
        }

        fragment {
            ivec4 main(FSInput input) @ COLOR0 {
                ivec4 signedResult = invalidSigned(
                    signedMs,
                    linearSampler,
                    input.uv,
                    input.pixel,
                    input.sampleIndex,
                    input.offset
                );
                uvec4 unsignedResult = invalidUnsigned(
                    unsignedMsArray,
                    linearSampler,
                    input.uvLayer,
                    input.pixelLayer,
                    input.sampleIndex,
                    input.offset
                );
                return signedResult + ivec4(unsignedResult)
                    + texture(signedMs, input.uv)
                    + ivec4(texture(unsignedMsArray, input.uvLayer))
                    + texelFetchOffset(signedMs, input.pixel, input.sampleIndex, input.offset)
                    + ivec4(texelFetchOffset(unsignedMsArray, input.pixelLayer, input.sampleIndex, input.offset));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform isampler2DMS signedMs;" in generated_code
    assert (
        "layout(binding = 1) uniform usampler2DMSArray unsignedMsArray;"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert (
        generated_code.count(
            "unsupported GLSL multisample texture call: texture on isampler2DMS */ ivec4(0)"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported GLSL multisample texture call: textureLod on isampler2DMS */ ivec4(0)"
        )
        == 1
    )
    assert (
        generated_code.count(
            "unsupported GLSL multisample texture call: textureGather on isampler2DMS */ ivec4(0)"
        )
        == 1
    )
    assert (
        generated_code.count(
            "unsupported GLSL texel fetch offset: multisample texture isampler2DMS does not support offsets */ ivec4(0)"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported GLSL multisample texture call: texture on usampler2DMSArray */ uvec4(0u)"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported GLSL multisample texture call: textureLod on usampler2DMSArray */ uvec4(0u)"
        )
        == 1
    )
    assert (
        generated_code.count(
            "unsupported GLSL multisample texture call: textureGather on usampler2DMSArray */ uvec4(0u)"
        )
        == 1
    )
    assert (
        generated_code.count(
            "unsupported GLSL texel fetch offset: multisample texture usampler2DMSArray does not support offsets */ uvec4(0u)"
        )
        == 2
    )
    assert "texture(tex," not in generated_code
    assert "textureLod(tex," not in generated_code
    assert "textureGather(tex," not in generated_code
    assert "texelFetchOffset(tex," not in generated_code
    assert "texture(signedMs," not in generated_code
    assert "texture(unsignedMsArray," not in generated_code
    assert "texelFetchOffset(signedMs," not in generated_code
    assert "texelFetchOffset(unsignedMsArray," not in generated_code
    assert "input." not in generated_code


def test_opengl_integer_sampler_1d_3d_array_size_and_offsets_lower():
    shader = """
    shader IntegerSamplerCoordinateRanks {
        isampler1D signedLine;
        usampler1DArray unsignedLineArray;
        isampler3D signedVolume;
        usampler2DArray unsignedLayerMap;

        struct FSInput {
            int x @ TEXCOORD0;
            ivec2 lineLayer @ TEXCOORD1;
            ivec3 voxel @ TEXCOORD2;
            ivec3 pixelLayer @ TEXCOORD3;
            int offset1 @ TEXCOORD4;
            ivec2 offset2 @ TEXCOORD5;
            ivec3 offset3 @ TEXCOORD6;
            int lod;
        };

        ivec4 fetchSignedLine(isampler1D tex, int x, int lod, int offset) {
            return texelFetchOffset(tex, x, lod, offset);
        }

        uvec4 fetchUnsignedLineArray(usampler1DArray tex, ivec2 pixelLayer, int lod, int offset) {
            return texelFetchOffset(tex, pixelLayer, lod, offset);
        }

        ivec4 fetchSignedVolume(isampler3D tex, ivec3 voxel, int lod, ivec3 offset) {
            return texelFetchOffset(tex, voxel, lod, offset);
        }

        uvec4 fetchUnsignedLayer(usampler2DArray tex, ivec3 pixelLayer, int lod, ivec2 offset) {
            return texelFetchOffset(tex, pixelLayer, lod, offset);
        }

        fragment {
            ivec4 main(FSInput input) @ COLOR0 {
                int signedLineSize = textureSize(signedLine, input.lod);
                ivec2 unsignedLineSize = textureSize(unsignedLineArray, input.lod);
                ivec3 signedVolumeSize = textureSize(signedVolume, input.lod);
                ivec3 unsignedLayerSize = textureSize(unsignedLayerMap, input.lod);
                return fetchSignedLine(signedLine, input.x, input.lod, input.offset1)
                    + ivec4(fetchUnsignedLineArray(
                        unsignedLineArray,
                        input.lineLayer,
                        input.lod,
                        input.offset1
                    ))
                    + fetchSignedVolume(signedVolume, input.voxel, input.lod, input.offset3)
                    + ivec4(fetchUnsignedLayer(
                        unsignedLayerMap,
                        input.pixelLayer,
                        input.lod,
                        input.offset2
                    ))
                    + ivec4(
                        signedLineSize
                            + unsignedLineSize.y
                            + signedVolumeSize.z
                            + unsignedLayerSize.z
                    );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform isampler1D signedLine;" in generated_code
    assert (
        "layout(binding = 1) uniform usampler1DArray unsignedLineArray;"
        in generated_code
    )
    assert "layout(binding = 2) uniform isampler3D signedVolume;" in generated_code
    assert (
        "layout(binding = 3) uniform usampler2DArray unsignedLayerMap;"
        in generated_code
    )
    assert (
        "ivec4 fetchSignedLine(isampler1D tex, int x, int lod, int offset)"
        in generated_code
    )
    assert (
        "uvec4 fetchUnsignedLineArray(usampler1DArray tex, ivec2 pixelLayer, int lod, int offset)"
        in generated_code
    )
    assert (
        "ivec4 fetchSignedVolume(isampler3D tex, ivec3 voxel, int lod, ivec3 offset)"
        in generated_code
    )
    assert (
        "uvec4 fetchUnsignedLayer(usampler2DArray tex, ivec3 pixelLayer, int lod, ivec2 offset)"
        in generated_code
    )
    assert (
        generated_code.count(
            glsl_dynamic_texel_offset_diagnostic(
                "texel fetch offset", "texelFetchOffset", "ivec4(0)"
            )
        )
        == 2
    )
    assert (
        generated_code.count(
            glsl_dynamic_texel_offset_diagnostic(
                "texel fetch offset", "texelFetchOffset", "uvec4(0u)"
            )
        )
        == 2
    )
    assert "int signedLineSize = textureSize(signedLine, lod);" in generated_code
    assert (
        "ivec2 unsignedLineSize = textureSize(unsignedLineArray, lod);"
        in generated_code
    )
    assert "ivec3 signedVolumeSize = textureSize(signedVolume, lod);" in generated_code
    assert (
        "ivec3 unsignedLayerSize = textureSize(unsignedLayerMap, lod);"
        in generated_code
    )
    assert "input." not in generated_code
    assert "texelFetchOffset(" not in generated_code


@pytest.mark.parametrize(
    ("texture_decl", "return_expr", "match"),
    [
        (
            "isampler3D tex;",
            "ivec4(textureSize(tex))",
            "OpenGL texture operation 'textureSize' requires 2 argument\\(s\\) for isampler3D, got 1",
        ),
        (
            "usampler1DArray tex;",
            "ivec4(textureSize(tex), 0, 0)",
            "OpenGL texture operation 'textureSize' requires 2 argument\\(s\\) for usampler1DArray, got 1",
        ),
        (
            "isampler2DMS tex;",
            "ivec4(textureSize(tex, 0), 0, 0)",
            "OpenGL texture operation 'textureSize' accepts 1 argument\\(s\\) for isampler2DMS, got 2",
        ),
        (
            "usampler2DMSArray tex;",
            "ivec4(textureSize(tex, 0), 0)",
            "OpenGL texture operation 'textureSize' accepts 1 argument\\(s\\) for usampler2DMSArray, got 2",
        ),
    ],
)
def test_opengl_integer_texture_size_resource_specific_arity(
    texture_decl, return_expr, match
):
    shader = f"""
    shader InvalidIntegerTextureSizeArity {{
        {texture_decl}

        fragment {{
            ivec4 main() @ COLOR0 {{
                return {return_expr};
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


@pytest.mark.parametrize(
    ("texture_decl", "return_expr", "match"),
    [
        (
            "isampler1D tex;",
            "texelFetchOffset(tex, input.pixel2, input.lod, input.offset1)",
            "OpenGL resource operation 'texelFetchOffset' requires a 1D integer coordinate",
        ),
        (
            "usampler1DArray tex;",
            "ivec4(texelFetchOffset(tex, input.pixel1, input.lod, input.offset1))",
            "OpenGL resource operation 'texelFetchOffset' requires a 2D integer coordinate",
        ),
        (
            "isampler3D tex;",
            "texelFetchOffset(tex, input.pixel2, input.lod, input.offset3)",
            "OpenGL resource operation 'texelFetchOffset' requires a 3D integer coordinate",
        ),
        (
            "usampler2DArray tex;",
            "ivec4(texelFetchOffset(tex, input.pixel2, input.lod, input.offset2))",
            "OpenGL resource operation 'texelFetchOffset' requires a 3D integer coordinate",
        ),
        (
            "isampler1DArray tex;",
            "texelFetchOffset(tex, input.pixel2, input.lod, input.offset2)",
            "OpenGL resource operation 'texelFetchOffset' requires a 1D integer offset",
        ),
        (
            "usampler2DArray tex;",
            "ivec4(texelFetchOffset(tex, input.pixel3, input.lod, input.offset3))",
            "OpenGL resource operation 'texelFetchOffset' requires a 2D integer offset",
        ),
        (
            "isampler3D tex;",
            "texelFetchOffset(tex, input.pixel3, input.lod, input.offset2)",
            "OpenGL resource operation 'texelFetchOffset' requires a 3D integer offset",
        ),
    ],
)
def test_opengl_integer_texel_fetch_offset_rank_diagnostics(
    texture_decl, return_expr, match
):
    shader = f"""
    shader InvalidIntegerTexelFetchOffsetRanks {{
        {texture_decl}

        struct FSInput {{
            int pixel1;
            ivec2 pixel2;
            ivec3 pixel3;
            int offset1;
            ivec2 offset2;
            ivec3 offset3;
            int lod;
        }};

        fragment {{
            ivec4 main(FSInput input) @ COLOR0 {{
                return {return_expr};
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_opengl_non_multisample_texture_samples_emit_diagnostics():
    shader = """
    shader NonMultisampleSamplesQuery {
        sampler2D colorMap;
        sampler2DArray layerMap;
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        sampler2D textures[4];

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        int querySamples(
            sampler2D tex,
            sampler2DArray arrayTex,
            samplerCube cubeTex,
            samplerCubeArray cubeArrayTex,
            sampler2DShadow shadowTex,
            sampler2DArrayShadow shadowArrayTex
        ) {
            return textureSamples(tex)
                + textureSamples(arrayTex)
                + textureSamples(cubeTex)
                + textureSamples(cubeArrayTex)
                + textureSamples(shadowTex)
                + textureSamples(shadowArrayTex);
        }

        int queryResourceArray(sampler2D textures[], int layer) {
            return textureSamples(textures[2]) + textureSamples(textures[layer]);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                int direct = textureSamples(colorMap) + textureSamples(shadowArray);
                return float(querySamples(
                    colorMap,
                    layerMap,
                    cubeMap,
                    cubeArray,
                    shadowMap,
                    shadowArray
                ) + queryResourceArray(textures, input.layer) + direct);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported GLSL texture samples query: requires multisample sampler */ 0"
    )
    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler2DArray layerMap;" in generated_code
    assert "layout(binding = 2) uniform samplerCube cubeMap;" in generated_code
    assert "layout(binding = 3) uniform samplerCubeArray cubeArray;" in generated_code
    assert "layout(binding = 6) uniform sampler2D textures[4];" in generated_code
    assert generated_code.count(diagnostic) == 10
    assert "textureSamples(" not in generated_code
    assert "input." not in generated_code


def test_opengl_storage_image_samples_queries_emit_diagnostics():
    shader = """
    shader StorageImageSamplesQueries {
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;
        uimage2D uintImage;
        iimage3D intVolume;

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        int samples2D(image2D image) {
            return imageSamples(image) + textureSamples(image);
        }

        int samples3D(image3D image) {
            return imageSamples(image) + textureSamples(image);
        }

        int samplesArray(image2DArray image) {
            return imageSamples(image) + textureSamples(image);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                int samples = imageSamples(colorImage)
                    + imageSamples(volumeImage)
                    + imageSamples(layerImage)
                    + imageSamples(uintImage)
                    + imageSamples(intVolume)
                    + textureSamples(colorImage)
                    + textureSamples(volumeImage);
                return float(samples
                    + samples2D(colorImage)
                    + samples3D(volumeImage)
                    + samplesArray(layerImage));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported GLSL texture samples query: requires multisample sampler */ 0"
    )
    assert "layout(rgba32f, binding = 0) uniform image2D colorImage;" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image3D volumeImage;" in generated_code
    assert (
        "layout(rgba32f, binding = 2) uniform image2DArray layerImage;"
        in generated_code
    )
    assert "layout(r32ui, binding = 3) uniform uimage2D uintImage;" in generated_code
    assert "layout(r32i, binding = 4) uniform iimage3D intVolume;" in generated_code
    assert generated_code.count(diagnostic) == 19
    assert "imageSamples(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "input." not in generated_code


def test_opengl_storage_image_sampling_and_fetch_emit_diagnostics():
    shader = """
    shader StorageImageInvalidSampleFetch {
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvw @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            ivec2 pixel @ TEXCOORD3;
            ivec3 voxel @ TEXCOORD4;
            ivec3 pixelLayer @ TEXCOORD5;
            ivec2 offset2 @ TEXCOORD6;
            ivec3 offset3 @ TEXCOORD7;
            int lod;
        };

        vec4 invalid2D(image2D image, vec2 uv, ivec2 pixel, int lod, ivec2 offset) {
            return texture(image, uv)
                + textureLod(image, uv, float(lod))
                + textureGather(image, uv)
                + texelFetch(image, pixel, lod)
                + texelFetchOffset(image, pixel, lod, offset);
        }

        vec4 invalid3D(image3D image, vec3 uvw, ivec3 voxel, int lod, ivec3 offset) {
            return texture(image, uvw)
                + textureLod(image, uvw, float(lod))
                + textureGather(image, uvw)
                + texelFetch(image, voxel, lod)
                + texelFetchOffset(image, voxel, lod, offset);
        }

        vec4 invalidArray(image2DArray image, vec3 uvLayer, ivec3 pixelLayer, int lod, ivec2 offset) {
            return texture(image, uvLayer)
                + textureLod(image, uvLayer, float(lod))
                + textureGather(image, uvLayer)
                + texelFetch(image, pixelLayer, lod)
                + texelFetchOffset(image, pixelLayer, lod, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return invalid2D(colorImage, input.uv, input.pixel, input.lod, input.offset2)
                    + invalid3D(volumeImage, input.uvw, input.voxel, input.lod, input.offset3)
                    + invalidArray(layerImage, input.uvLayer, input.pixelLayer, input.lod, input.offset2)
                    + texture(colorImage, input.uv)
                    + textureLod(volumeImage, input.uvw, float(input.lod))
                    + textureGather(layerImage, input.uvLayer)
                    + texelFetch(colorImage, input.pixel, input.lod)
                    + texelFetchOffset(layerImage, input.pixelLayer, input.lod, input.offset2);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = "unsupported GLSL storage image texture operation"
    assert "layout(rgba32f, binding = 0) uniform image2D colorImage;" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image3D volumeImage;" in generated_code
    assert (
        "layout(rgba32f, binding = 2) uniform image2DArray layerImage;"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 35
    for operation in {
        "texture",
        "textureLod",
        "textureGather",
        "texelFetch",
        "texelFetchOffset",
    }:
        assert f"{operation} on image2D" in generated_code
    assert "texture(" not in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGather(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "texelFetchOffset(" not in generated_code
    assert "input." not in generated_code


def test_opengl_storage_image_compare_calls_emit_diagnostics():
    shader = """
    shader StorageImageInvalidCompare {
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvw @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
            float depth;
            float lod;
        };

        float compareImplicit(image2D image, vec2 uv, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset) {
            return textureCompare(image, uv, depth)
                + textureCompareOffset(image, uv, depth, offset)
                + textureCompareLod(image, uv, depth, lod)
                + textureCompareLodOffset(image, uv, depth, lod, offset)
                + textureCompareGrad(image, uv, depth, ddx, ddy)
                + textureCompareGradOffset(image, uv, depth, ddx, ddy, offset);
        }

        float compareExplicit(image3D image, sampler s, vec3 uvw, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset) {
            return textureCompare(image, s, uvw, depth)
                + textureCompareOffset(image, s, uvw, depth, offset)
                + textureCompareLod(image, s, uvw, depth, lod)
                + textureCompareLodOffset(image, s, uvw, depth, lod, offset)
                + textureCompareGrad(image, s, uvw, depth, ddx, ddy)
                + textureCompareGradOffset(image, s, uvw, depth, ddx, ddy, offset);
        }

        vec4 gatherExplicit(image2DArray image, sampler s, vec3 uvLayer, float depth, ivec2 offset) {
            return textureGatherCompare(image, s, uvLayer, depth)
                + textureGatherCompareOffset(image, s, uvLayer, depth, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float cmp = compareImplicit(colorImage, input.uv, input.depth, input.lod, input.ddx, input.ddy, input.offset)
                    + compareExplicit(volumeImage, compareSampler, input.uvw, input.depth, input.lod, input.ddx, input.ddy, input.offset)
                    + textureCompare(colorImage, input.uv, input.depth)
                    + textureCompare(volumeImage, compareSampler, input.uvw, input.depth);
                return vec4(cmp)
                    + gatherExplicit(layerImage, compareSampler, input.uvLayer, input.depth, input.offset)
                    + textureGatherCompare(layerImage, compareSampler, input.uvLayer, input.depth)
                    + textureGatherCompareOffset(layerImage, compareSampler, input.uvLayer, input.depth, input.offset);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    comparison_diagnostic = "unsupported GLSL storage image texture comparison"
    gather_diagnostic = "unsupported GLSL storage image texture operation"
    assert "layout(rgba32f, binding = 0) uniform image2D colorImage;" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image3D volumeImage;" in generated_code
    assert (
        "layout(rgba32f, binding = 2) uniform image2DArray layerImage;"
        in generated_code
    )
    assert generated_code.count(comparison_diagnostic) == 26
    assert generated_code.count(gather_diagnostic) == 6
    assert "textureCompare(" not in generated_code
    assert "textureGatherCompare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code
    assert "input." not in generated_code


def test_opengl_multisample_texture_query_levels_lower_to_single_level():
    shader = """
    shader MultisampleLevelQueries {
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler2DMS msTextures[4];
        sampler2DMSArray msArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        int levels2D(sampler2DMS tex) {
            return textureQueryLevels(tex);
        }

        int levelsArray(sampler2DMSArray tex) {
            return textureQueryLevels(tex);
        }

        int levelsResourceArrays(sampler2DMS textures[], sampler2DMSArray arrays[], int layer) {
            return textureQueryLevels(textures[2]) + textureQueryLevels(arrays[layer]);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                int c = textureQueryLevels(msTex);
                int d = textureQueryLevels(msArray);
                int e = textureQueryLevels(msTextures[2]);
                int f = textureQueryLevels(msArrays[input.layer]);
                return float(c + d + e + f + levels2D(msTex) + levelsArray(msArray)
                    + levelsResourceArrays(msTextures, msArrays, input.layer));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DMS msTex;" in generated_code
    assert "layout(binding = 1) uniform sampler2DMSArray msArray;" in generated_code
    assert "layout(binding = 2) uniform sampler2DMS msTextures[4];" in generated_code
    assert "layout(binding = 6) uniform sampler2DMSArray msArrays[4];" in generated_code
    assert "int levels2D(sampler2DMS tex)" in generated_code
    assert "int levelsArray(sampler2DMSArray tex)" in generated_code
    assert (
        "int levelsResourceArrays(sampler2DMS textures[4], sampler2DMSArray arrays[4], int layer)"
        in generated_code
    )
    assert generated_code.count("return 1;") == 2
    assert "return (1 + 1);" in generated_code
    assert "int c = 1;" in generated_code
    assert "int d = 1;" in generated_code
    assert "int e = 1;" in generated_code
    assert "int f = 1;" in generated_code
    assert "textureQueryLevels(" not in generated_code


def test_opengl_multisample_sampling_operations_emit_diagnostics():
    shader = """
    shader UnsupportedMultisampleSampling {
        sampler2DMS msTex;
        sampler2DMSArray msArray;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 invalid2D(sampler2DMS tex, vec2 uv, vec2 ddx, vec2 ddy, ivec2 offset) {
            vec4 sampled = texture(tex, uv);
            vec4 lod = textureLod(tex, uv, 0.0);
            vec4 grad = textureGrad(tex, uv, ddx, ddy);
            vec4 gathered = textureGather(tex, uv);
            vec4 offsetGathered = textureGatherOffset(tex, uv, offset);
            return sampled + lod + grad + gathered + offsetGathered;
        }

        vec4 invalidArray(sampler2DMSArray tex, vec3 uvLayer, vec2 ddx, vec2 ddy, ivec2 offset) {
            vec4 sampled = texture(tex, uvLayer);
            vec4 lod = textureLod(tex, uvLayer, 0.0);
            vec4 grad = textureGrad(tex, uvLayer, ddx, ddy);
            vec4 gathered = textureGather(tex, uvLayer);
            vec4 offsetGathered = textureGatherOffset(tex, uvLayer, offset);
            return sampled + lod + grad + gathered + offsetGathered;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return invalid2D(msTex, input.uv, input.ddx, input.ddy, input.offset)
                    + invalidArray(msArray, input.uvLayer, input.ddx, input.ddy, input.offset);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DMS msTex;" in generated_code
    assert "layout(binding = 1) uniform sampler2DMSArray msArray;" in generated_code
    assert (
        "vec4 invalid2D(sampler2DMS tex, vec2 uv, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "vec4 invalidArray(sampler2DMSArray tex, vec3 uvLayer, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )

    for func_name in {
        "texture",
        "textureLod",
        "textureGrad",
        "textureGather",
        "textureGatherOffset",
    }:
        assert (
            f"unsupported GLSL multisample texture call: {func_name} on sampler2DMS"
            in generated_code
        )
        assert (
            f"unsupported GLSL multisample texture call: {func_name} on sampler2DMSArray"
            in generated_code
        )

    for invalid_call in {
        "texture(tex,",
        "textureLod(tex,",
        "textureGrad(tex,",
        "textureGather(tex,",
        "textureGatherOffset(tex,",
    }:
        assert invalid_call not in generated_code


def test_opengl_multisample_compare_operations_emit_diagnostics():
    shader = """
    shader UnsupportedMultisampleCompare {
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec4 proj @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        float invalidCompare2D(sampler2DMS tex, sampler s, vec2 uv, vec4 proj, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset) {
            return textureCompare(tex, s, uv, depth)
                + textureCompareOffset(tex, s, uv, depth, offset)
                + textureCompareLod(tex, s, uv, depth, lod)
                + textureCompareLodOffset(tex, s, uv, depth, lod, offset)
                + textureCompareGrad(tex, s, uv, depth, ddx, ddy)
                + textureCompareGradOffset(tex, s, uv, depth, ddx, ddy, offset)
                + textureCompareProj(tex, s, proj, depth)
                + textureCompareProjLod(tex, s, proj, depth, lod)
                + textureCompareProjGradOffset(tex, s, proj, depth, ddx, ddy, offset);
        }

        float invalidCompareArray(sampler2DMSArray tex, sampler s, vec3 uvLayer, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset) {
            return textureCompare(tex, s, uvLayer, depth)
                + textureCompareOffset(tex, s, uvLayer, depth, offset)
                + textureCompareLod(tex, s, uvLayer, depth, lod)
                + textureCompareLodOffset(tex, s, uvLayer, depth, lod, offset)
                + textureCompareGrad(tex, s, uvLayer, depth, ddx, ddy)
                + textureCompareGradOffset(tex, s, uvLayer, depth, ddx, ddy, offset);
        }

        vec4 invalidGather2D(sampler2DMS tex, sampler s, vec2 uv, float depth, ivec2 offset) {
            return textureGatherCompare(tex, s, uv, depth)
                + textureGatherCompareOffset(tex, s, uv, depth, offset);
        }

        vec4 invalidGatherArray(sampler2DMSArray tex, sampler s, vec3 uvLayer, float depth, ivec2 offset) {
            return textureGatherCompare(tex, s, uvLayer, depth)
                + textureGatherCompareOffset(tex, s, uvLayer, depth, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float cmp = invalidCompare2D(msTex, linearSampler, input.uv, input.proj, input.depth, input.lod, input.ddx, input.ddy, input.offset)
                    + invalidCompareArray(msArray, linearSampler, input.uvLayer, input.depth, input.lod, input.ddx, input.ddy, input.offset);
                return vec4(cmp)
                    + invalidGather2D(msTex, linearSampler, input.uv, input.depth, input.offset)
                    + invalidGatherArray(msArray, linearSampler, input.uvLayer, input.depth, input.offset);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DMS msTex;" in generated_code
    assert "layout(binding = 1) uniform sampler2DMSArray msArray;" in generated_code
    assert "linearSampler" not in generated_code

    for func_name in {
        "textureCompare",
        "textureCompareOffset",
        "textureCompareLod",
        "textureCompareLodOffset",
        "textureCompareGrad",
        "textureCompareGradOffset",
    }:
        assert (
            f"unsupported GLSL multisample texture comparison: {func_name} on sampler2DMS"
            in generated_code
        )
        assert (
            f"unsupported GLSL multisample texture comparison: {func_name} on sampler2DMSArray"
            in generated_code
        )

    for func_name in {
        "textureCompareProj",
        "textureCompareProjLod",
        "textureCompareProjGradOffset",
    }:
        assert (
            f"unsupported GLSL multisample texture comparison: {func_name} on sampler2DMS"
            in generated_code
        )

    for func_name in {"textureGatherCompare", "textureGatherCompareOffset"}:
        assert (
            f"unsupported GLSL multisample texture gather comparison: {func_name} on sampler2DMS"
            in generated_code
        )
        assert (
            f"unsupported GLSL multisample texture gather comparison: {func_name} on sampler2DMSArray"
            in generated_code
        )

    for invalid_call in {
        "texture(tex,",
        "textureOffset(tex,",
        "textureLod(tex,",
        "textureLodOffset(tex,",
        "textureGrad(tex,",
        "textureGradOffset(tex,",
        "textureGather(tex,",
        "textureGatherOffset(tex,",
    }:
        assert invalid_call not in generated_code


def test_opengl_multisample_texture_query_lod_emits_diagnostics():
    shader = """
    shader UnsupportedMultisampleQueryLod {
        sampler2DMS msTex;
        sampler2DMSArray msArray;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
        };

        vec2 query2D(sampler2DMS tex, vec2 uv) {
            return textureQueryLod(tex, uv);
        }

        vec2 queryArray(sampler2DMSArray tex, vec3 uvLayer) {
            return textureQueryLod(tex, uvLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = textureQueryLod(msTex, input.uv);
                vec2 b = textureQueryLod(msArray, input.uvLayer);
                vec2 c = query2D(msTex, input.uv);
                vec2 d = queryArray(msArray, input.uvLayer);
                return vec4(a + b + c + d, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DMS msTex;" in generated_code
    assert "layout(binding = 1) uniform sampler2DMSArray msArray;" in generated_code
    assert "vec2 query2D(sampler2DMS tex, vec2 uv)" in generated_code
    assert "vec2 queryArray(sampler2DMSArray tex, vec3 uvLayer)" in generated_code
    assert "textureQueryLod(" not in generated_code
    assert (
        generated_code.count(
            "unsupported GLSL multisample texture query: textureQueryLod on sampler2DMS */ vec2(0.0)"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported GLSL multisample texture query: textureQueryLod on sampler2DMSArray"
        )
        == 2
    )


def test_opengl_multisample_texel_fetch_offsets_emit_diagnostics():
    shader = """
    shader UnsupportedMultisampleTexelFetchOffset {
        sampler2DMS msTex;
        sampler2DMSArray msArray;

        struct FSInput {
            ivec2 pixel @ TEXCOORD0;
            ivec3 pixelLayer @ TEXCOORD1;
            ivec2 offset @ TEXCOORD2;
            int sampleIndex;
        };

        vec4 offset2D(sampler2DMS tex, ivec2 pixel, int sampleIndex, ivec2 offset) {
            return texelFetchOffset(tex, pixel, sampleIndex, offset);
        }

        vec4 offsetArray(sampler2DMSArray tex, ivec3 pixelLayer, int sampleIndex, ivec2 offset) {
            return texelFetchOffset(tex, pixelLayer, sampleIndex, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return offset2D(msTex, input.pixel, input.sampleIndex, input.offset)
                    + offsetArray(msArray, input.pixelLayer, input.sampleIndex, input.offset)
                    + texelFetchOffset(msTex, input.pixel, input.sampleIndex, input.offset)
                    + texelFetchOffset(msArray, input.pixelLayer, input.sampleIndex, input.offset);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DMS msTex;" in generated_code
    assert "layout(binding = 1) uniform sampler2DMSArray msArray;" in generated_code
    assert (
        "vec4 offset2D(sampler2DMS tex, ivec2 pixel, int sampleIndex, ivec2 offset)"
        in generated_code
    )
    assert (
        "vec4 offsetArray(sampler2DMSArray tex, ivec3 pixelLayer, int sampleIndex, ivec2 offset)"
        in generated_code
    )
    assert (
        generated_code.count(
            "unsupported GLSL texel fetch offset: multisample texture sampler2DMS does not support offsets"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported GLSL texel fetch offset: multisample texture sampler2DMSArray does not support offsets"
        )
        == 2
    )
    assert "texelFetchOffset(" not in generated_code


def test_opengl_cube_texel_fetches_emit_diagnostics():
    shader = """
    shader UnsupportedCubeTexelFetch {
        samplerCube cubeMap;
        samplerCubeArray cubeArray;

        struct FSInput {
            ivec3 cubeCoord @ TEXCOORD0;
            ivec4 cubeLayerCoord @ TEXCOORD1;
            ivec3 offset @ TEXCOORD2;
            int lod;
        };

        vec4 fetchCube(samplerCube tex, ivec3 cubeCoord, int lod, ivec3 offset) {
            vec4 plain = texelFetch(tex, cubeCoord, lod);
            vec4 offsetFetch = texelFetchOffset(tex, cubeCoord, lod, offset);
            return plain + offsetFetch;
        }

        vec4 fetchCubeArray(samplerCubeArray tex, ivec4 cubeLayerCoord, int lod, ivec3 offset) {
            vec4 plain = texelFetch(tex, cubeLayerCoord, lod);
            vec4 offsetFetch = texelFetchOffset(tex, cubeLayerCoord, lod, offset);
            return plain + offsetFetch;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return fetchCube(cubeMap, input.cubeCoord, input.lod, input.offset)
                    + fetchCubeArray(cubeArray, input.cubeLayerCoord, input.lod, input.offset)
                    + texelFetch(cubeMap, input.cubeCoord, input.lod)
                    + texelFetch(cubeArray, input.cubeLayerCoord, input.lod)
                    + texelFetchOffset(cubeMap, input.cubeCoord, input.lod, input.offset)
                    + texelFetchOffset(cubeArray, input.cubeLayerCoord, input.lod, input.offset);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform samplerCube cubeMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCubeArray cubeArray;" in generated_code
    assert (
        "vec4 fetchCube(samplerCube tex, ivec3 cubeCoord, int lod, ivec3 offset)"
        in generated_code
    )
    assert (
        "vec4 fetchCubeArray(samplerCubeArray tex, ivec4 cubeLayerCoord, int lod, ivec3 offset)"
        in generated_code
    )
    assert (
        generated_code.count(
            "unsupported GLSL texel fetch: texelFetch on samplerCube */ vec4(0.0)"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported GLSL texel fetch: texelFetchOffset on samplerCube */ vec4(0.0)"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported GLSL texel fetch: texelFetch on samplerCubeArray"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported GLSL texel fetch: texelFetchOffset on samplerCubeArray"
        )
        == 2
    )
    assert "texelFetch(" not in generated_code
    assert "texelFetchOffset(" not in generated_code


def test_opengl_array_shadow_texture_query_functions():
    shader = """
    shader ArrayShadowTextureQueries {
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeShadowArrays[4];

        ivec3 query2DArrayShadow(sampler2DArrayShadow tex) {
            ivec3 size = textureSize(tex, 1);
            int levels = textureQueryLevels(tex);
            return size + ivec3(levels);
        }

        ivec3 queryCubeArrayShadow(samplerCubeArrayShadow tex) {
            ivec3 size = textureSize(tex, 0);
            int levels = textureQueryLevels(tex);
            return size + ivec3(levels);
        }

        ivec3 queryArrayElements(sampler2DArrayShadow shadowArrays[], samplerCubeArrayShadow cubeShadowArrays[]) {
            ivec3 arraySize = textureSize(shadowArrays[2], 1);
            ivec3 cubeSize = textureSize(cubeShadowArrays[3], 0);
            int arrayLevels = textureQueryLevels(shadowArrays[2]);
            int cubeLevels = textureQueryLevels(cubeShadowArrays[3]);
            return arraySize + cubeSize + ivec3(arrayLevels + cubeLevels);
        }

        fragment {
            vec4 main() @ gl_FragColor {
                ivec3 a = query2DArrayShadow(shadowArray);
                ivec3 b = queryCubeArrayShadow(cubeShadowArray);
                ivec3 c = queryArrayElements(shadowArrays, cubeShadowArrays);
                return vec4(float(a.x + b.y + c.z));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "layout(binding = 1) uniform samplerCubeArrayShadow cubeShadowArray;"
        in generated_code
    )
    assert (
        "layout(binding = 2) uniform sampler2DArrayShadow shadowArrays[4];"
        in generated_code
    )
    assert (
        "layout(binding = 6) uniform samplerCubeArrayShadow cubeShadowArrays[4];"
        in generated_code
    )
    assert "ivec3 query2DArrayShadow(sampler2DArrayShadow tex)" in generated_code
    assert "ivec3 queryCubeArrayShadow(samplerCubeArrayShadow tex)" in generated_code
    assert (
        "ivec3 queryArrayElements(sampler2DArrayShadow shadowArrays[4], samplerCubeArrayShadow cubeShadowArrays[4])"
        in generated_code
    )
    assert "ivec3 size = textureSize(tex, 1);" in generated_code
    assert "ivec3 size = textureSize(tex, 0);" in generated_code
    assert "int levels = textureQueryLevels(tex);" in generated_code
    assert "ivec3 arraySize = textureSize(shadowArrays[2], 1);" in generated_code
    assert "ivec3 cubeSize = textureSize(cubeShadowArrays[3], 0);" in generated_code
    assert "int arrayLevels = textureQueryLevels(shadowArrays[2]);" in generated_code
    assert "int cubeLevels = textureQueryLevels(cubeShadowArrays[3]);" in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_direct_stage_texture_queries_mix_size_levels_and_lod():
    shader = """
    shader DirectTextureQueries {
        sampler2D colorMap;
        sampler2DArray layerMap;
        samplerCubeArray cubeArray;
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float lod;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                ivec2 colorSize = textureSize(colorMap, 1);
                ivec3 layerSize = textureSize(layerMap, 2);
                ivec3 cubeSize = textureSize(cubeArray, 0);
                ivec2 shadowSize = textureSize(shadowMap, 0);
                ivec3 shadowArraySize = textureSize(shadowArray, 1);
                ivec3 cubeShadowSize = textureSize(cubeShadowArray, 0);
                int colorLevels = textureQueryLevels(colorMap);
                int shadowLevels = textureQueryLevels(shadowArray);
                vec2 colorLod = textureQueryLod(colorMap, linearSampler, input.uv);
                vec2 layerLod = textureQueryLod(layerMap, linearSampler, input.uvLayer);
                vec2 cubeLod = textureQueryLod(cubeArray, linearSampler, input.cubeLayer);
                vec2 implicitLayerLod = textureQueryLod(layerMap, input.uvLayer);
                vec2 implicitCubeShadowLod = textureQueryLod(cubeShadowArray, input.cubeLayer);
                float total = float(colorSize.x + layerSize.z + cubeSize.z + shadowSize.x + shadowArraySize.z + cubeShadowSize.z + colorLevels + shadowLevels);
                return vec4(total + colorLod.x + layerLod.y + cubeLod.x + implicitLayerLod.x + implicitCubeShadowLod.y);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler2DArray layerMap;" in generated_code
    assert "layout(binding = 2) uniform samplerCubeArray cubeArray;" in generated_code
    assert "layout(binding = 3) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "layout(binding = 4) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "layout(binding = 5) uniform samplerCubeArrayShadow cubeShadowArray;"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert "ivec2 colorSize = textureSize(colorMap, 1);" in generated_code
    assert "ivec3 layerSize = textureSize(layerMap, 2);" in generated_code
    assert "ivec3 cubeSize = textureSize(cubeArray, 0);" in generated_code
    assert "ivec2 shadowSize = textureSize(shadowMap, 0);" in generated_code
    assert "ivec3 shadowArraySize = textureSize(shadowArray, 1);" in generated_code
    assert "ivec3 cubeShadowSize = textureSize(cubeShadowArray, 0);" in generated_code
    assert "int colorLevels = textureQueryLevels(colorMap);" in generated_code
    assert "int shadowLevels = textureQueryLevels(shadowArray);" in generated_code
    assert "vec2 colorLod = textureQueryLod(colorMap, uv);" in generated_code
    assert "vec2 layerLod = textureQueryLod(layerMap, uvLayer.xy);" in generated_code
    assert "vec2 cubeLod = textureQueryLod(cubeArray, cubeLayer.xyz);" in generated_code
    assert (
        "vec2 implicitLayerLod = textureQueryLod(layerMap, uvLayer.xy);"
        in generated_code
    )
    assert (
        "vec2 implicitCubeShadowLod = textureQueryLod(cubeShadowArray, cubeLayer.xyz);"
        in generated_code
    )
    assert "textureQueryLod(layerMap, uvLayer);" not in generated_code
    assert "textureQueryLod(cubeArray, cubeLayer);" not in generated_code
    assert "input." not in generated_code


def test_opengl_array_texture_query_lod_uses_non_layer_coordinates():
    shader = """
    shader ArrayTextureQueryLod {
        sampler2DArray layerMap;
        samplerCubeArray cubeArray;
        sampler2DArray layerMaps[4];
        samplerCubeArray cubeArrays[4];
        sampler linearSampler;
        sampler linearSamplers[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
        };

        vec2 queryArrayLod(sampler2DArray tex, sampler s, vec3 uvLayer) {
            return textureQueryLod(tex, s, uvLayer);
        }

        vec2 queryCubeArrayLod(samplerCubeArray tex, sampler s, vec4 cubeLayer) {
            return textureQueryLod(tex, s, cubeLayer);
        }

        vec2 queryArrayElementLod(sampler2DArray layerMaps[], sampler linearSamplers[], vec3 uvLayer) {
            return textureQueryLod(layerMaps[2], linearSamplers[2], uvLayer);
        }

        vec2 queryCubeArrayElementLod(samplerCubeArray cubeArrays[], sampler linearSamplers[], vec4 cubeLayer) {
            return textureQueryLod(cubeArrays[3], linearSamplers[3], cubeLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = queryArrayLod(layerMap, linearSampler, input.uvLayer);
                vec2 b = queryCubeArrayLod(cubeArray, linearSampler, input.cubeLayer);
                vec2 c = queryArrayElementLod(layerMaps, linearSamplers, input.uvLayer);
                vec2 d = queryCubeArrayElementLod(cubeArrays, linearSamplers, input.cubeLayer);
                return vec4(a.x + b.y, c.x + d.y, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(binding = 0) uniform sampler2DArray layerMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCubeArray cubeArray;" in generated_code
    assert "layout(binding = 2) uniform sampler2DArray layerMaps[4];" in generated_code
    assert (
        "layout(binding = 6) uniform samplerCubeArray cubeArrays[4];" in generated_code
    )
    assert "vec2 queryArrayLod(sampler2DArray tex, vec3 uvLayer)" in generated_code
    assert "return textureQueryLod(tex, uvLayer.xy);" in generated_code
    assert (
        "vec2 queryCubeArrayLod(samplerCubeArray tex, vec4 cubeLayer)" in generated_code
    )
    assert "return textureQueryLod(tex, cubeLayer.xyz);" in generated_code
    assert (
        "vec2 queryArrayElementLod(sampler2DArray layerMaps[4], vec3 uvLayer)"
        in generated_code
    )
    assert "return textureQueryLod(layerMaps[2], uvLayer.xy);" in generated_code
    assert (
        "vec2 queryCubeArrayElementLod(samplerCubeArray cubeArrays[4], vec4 cubeLayer)"
        in generated_code
    )
    assert "return textureQueryLod(cubeArrays[3], cubeLayer.xyz);" in generated_code
    assert "queryArrayLod(layerMap, uvLayer)" in generated_code
    assert "queryCubeArrayLod(cubeArray, cubeLayer)" in generated_code
    assert "queryArrayElementLod(layerMaps, uvLayer)" in generated_code
    assert "queryCubeArrayElementLod(cubeArrays, cubeLayer)" in generated_code
    assert "linearSampler" not in generated_code
    assert "linearSamplers" not in generated_code
    assert "textureQueryLod(tex, uvLayer);" not in generated_code
    assert "textureQueryLod(tex, cubeLayer);" not in generated_code


def test_opengl_nested_array_texture_query_lod_keeps_combined_samplers():
    shader = """
    shader NestedArrayTextureQueryLod {
        sampler2DArray layerMaps[4];
        samplerCubeArray cubeArrays[4];
        sampler linearSamplers[4];
        sampler cubeArraysSampler;

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
        };

        vec2 explicitLeaf(vec3 uvLayer) {
            return textureQueryLod(layerMaps[2], linearSamplers[2], uvLayer);
        }

        vec2 explicitMid(vec3 uvLayer) {
            return explicitLeaf(uvLayer);
        }

        vec2 implicitLeaf(vec4 cubeLayer) {
            return textureQueryLod(cubeArrays[3], cubeLayer);
        }

        vec2 implicitMid(vec4 cubeLayer) {
            return implicitLeaf(cubeLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = explicitMid(input.uvLayer);
                vec2 b = implicitMid(input.cubeLayer);
                return vec4(a.x + b.y, a.y + b.x, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DArray layerMaps[4];" in generated_code
    assert (
        "layout(binding = 4) uniform samplerCubeArray cubeArrays[4];" in generated_code
    )
    assert "vec2 explicitLeaf(vec3 uvLayer)" in generated_code
    assert "vec2 explicitMid(vec3 uvLayer)" in generated_code
    assert "return explicitLeaf(uvLayer);" in generated_code
    assert "return textureQueryLod(layerMaps[2], uvLayer.xy);" in generated_code
    assert "vec2 implicitLeaf(vec4 cubeLayer)" in generated_code
    assert "vec2 implicitMid(vec4 cubeLayer)" in generated_code
    assert "return implicitLeaf(cubeLayer);" in generated_code
    assert "return textureQueryLod(cubeArrays[3], cubeLayer.xyz);" in generated_code
    assert "explicitMid(uvLayer)" in generated_code
    assert "implicitMid(cubeLayer)" in generated_code
    assert "linearSamplers" not in generated_code
    assert "cubeArraysSampler" not in generated_code
    assert "textureQueryLod(layerMaps[2], uvLayer);" not in generated_code
    assert "textureQueryLod(cubeArrays[3], cubeLayer);" not in generated_code


def test_opengl_shadow_array_texture_query_lod_uses_non_layer_coordinates():
    shader = """
    shader ShadowArrayTextureQueryLod {
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeShadowArrays[4];
        sampler linearSampler;
        sampler linearSamplers[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
        };

        vec2 queryArrayLod(sampler2DArrayShadow tex, sampler s, vec3 uvLayer) {
            return textureQueryLod(tex, s, uvLayer);
        }

        vec2 queryCubeArrayLod(samplerCubeArrayShadow tex, sampler s, vec4 cubeLayer) {
            return textureQueryLod(tex, s, cubeLayer);
        }

        vec2 queryArrayElementLod(sampler2DArrayShadow shadowArrays[], sampler linearSamplers[], vec3 uvLayer) {
            return textureQueryLod(shadowArrays[2], linearSamplers[2], uvLayer);
        }

        vec2 queryCubeArrayElementLod(samplerCubeArrayShadow cubeShadowArrays[], sampler linearSamplers[], vec4 cubeLayer) {
            return textureQueryLod(cubeShadowArrays[3], linearSamplers[3], cubeLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = queryArrayLod(shadowArray, linearSampler, input.uvLayer);
                vec2 b = queryCubeArrayLod(cubeShadowArray, linearSampler, input.cubeLayer);
                vec2 c = queryArrayElementLod(shadowArrays, linearSamplers, input.uvLayer);
                vec2 d = queryCubeArrayElementLod(cubeShadowArrays, linearSamplers, input.cubeLayer);
                return vec4(a.x + b.y, c.x + d.y, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "layout(binding = 1) uniform samplerCubeArrayShadow cubeShadowArray;"
        in generated_code
    )
    assert (
        "layout(binding = 2) uniform sampler2DArrayShadow shadowArrays[4];"
        in generated_code
    )
    assert (
        "layout(binding = 6) uniform samplerCubeArrayShadow cubeShadowArrays[4];"
        in generated_code
    )
    assert (
        "vec2 queryArrayLod(sampler2DArrayShadow tex, vec3 uvLayer)" in generated_code
    )
    assert "return textureQueryLod(tex, uvLayer.xy);" in generated_code
    assert (
        "vec2 queryCubeArrayLod(samplerCubeArrayShadow tex, vec4 cubeLayer)"
        in generated_code
    )
    assert "return textureQueryLod(tex, cubeLayer.xyz);" in generated_code
    assert (
        "vec2 queryArrayElementLod(sampler2DArrayShadow shadowArrays[4], vec3 uvLayer)"
        in generated_code
    )
    assert "return textureQueryLod(shadowArrays[2], uvLayer.xy);" in generated_code
    assert (
        "vec2 queryCubeArrayElementLod(samplerCubeArrayShadow cubeShadowArrays[4], vec4 cubeLayer)"
        in generated_code
    )
    assert (
        "return textureQueryLod(cubeShadowArrays[3], cubeLayer.xyz);" in generated_code
    )
    assert "queryArrayLod(shadowArray, uvLayer)" in generated_code
    assert "queryCubeArrayLod(cubeShadowArray, cubeLayer)" in generated_code
    assert "queryArrayElementLod(shadowArrays, uvLayer)" in generated_code
    assert "queryCubeArrayElementLod(cubeShadowArrays, cubeLayer)" in generated_code
    assert "linearSampler" not in generated_code
    assert "linearSamplers" not in generated_code
    assert "textureQueryLod(tex, uvLayer);" not in generated_code
    assert "textureQueryLod(tex, cubeLayer);" not in generated_code


def test_opengl_mixed_cube_shadow_query_lod_and_compare_keep_coordinate_shapes():
    shader = """
    shader MixedCubeShadowQueryLodCompare {
        samplerCubeShadow cubeShadow;
        samplerCubeArrayShadow cubeShadowArray;
        sampler shadowSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
        };

        float inspectCubeShadow(
            samplerCubeShadow tex,
            sampler s,
            vec3 direction,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy
        ) {
            vec2 lodValue = textureQueryLod(tex, s, direction);
            float cmp = textureCompare(tex, s, direction, depth);
            float cmpLod = textureCompareLod(tex, s, direction, depth, lod);
            float grad = textureCompareGrad(tex, s, direction, depth, ddx, ddy);
            return lodValue.x + cmp + cmpLod + grad;
        }

        float inspectCubeArrayShadow(
            samplerCubeArrayShadow tex,
            sampler s,
            vec4 cubeLayer,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy
        ) {
            vec2 lodValue = textureQueryLod(tex, s, cubeLayer);
            float cmp = textureCompare(tex, s, cubeLayer, depth);
            float cmpLod = textureCompareLod(tex, s, cubeLayer, depth, lod);
            float grad = textureCompareGrad(tex, s, cubeLayer, depth, ddx, ddy);
            return lodValue.y + cmp + cmpLod + grad;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return inspectCubeShadow(
                    cubeShadow,
                    shadowSampler,
                    input.direction,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy
                ) + inspectCubeArrayShadow(
                    cubeShadowArray,
                    shadowSampler,
                    input.cubeLayer,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(binding = 0) uniform samplerCubeShadow cubeShadow;" in generated_code
    assert (
        "layout(binding = 1) uniform samplerCubeArrayShadow cubeShadowArray;"
        in generated_code
    )
    assert "shadowSampler" not in generated_code
    assert (
        "float inspectCubeShadow(samplerCubeShadow tex, vec3 direction, float depth, float lod, vec3 ddx, vec3 ddy)"
        in generated_code
    )
    assert "vec2 lodValue = textureQueryLod(tex, direction);" in generated_code
    assert "float cmp = texture(tex, vec4(direction, depth));" in generated_code
    assert (
        "float cmpLod = /* unsupported GLSL texture compare: textureCompareLod explicit LOD requires 2D shadow samplers */ 0.0;"
        in generated_code
    )
    assert (
        "float grad = textureGrad(tex, vec4(direction, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float inspectCubeArrayShadow(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth, float lod, vec3 ddx, vec3 ddy)"
        in generated_code
    )
    assert "vec2 lodValue = textureQueryLod(tex, cubeLayer.xyz);" in generated_code
    assert "float cmp = texture(tex, cubeLayer, depth);" in generated_code
    assert (
        "float grad = /* unsupported GLSL texture compare: textureCompareGrad explicit gradients require 2D, 2D-array, or cube shadow samplers */ 0.0;"
        in generated_code
    )
    assert (
        "inspectCubeShadow(cubeShadow, direction, depth, lod, ddx, ddy)"
        in generated_code
    )
    assert (
        "inspectCubeArrayShadow(cubeShadowArray, cubeLayer, depth, lod, ddx, ddy)"
        in generated_code
    )
    assert "textureCompare(" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGrad(" not in generated_code


def test_opengl_implicit_array_texture_query_lod_uses_non_layer_coordinates():
    shader = """
    shader ImplicitArrayTextureQueryLod {
        sampler2DArray layerMap;
        samplerCubeArray cubeArray;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler2DArray layerMaps[4];
        samplerCubeArrayShadow cubeShadowArrays[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
        };

        vec2 queryArrayLod(sampler2DArray tex, vec3 uvLayer) {
            return textureQueryLod(tex, uvLayer);
        }

        vec2 queryCubeArrayLod(samplerCubeArray tex, vec4 cubeLayer) {
            return textureQueryLod(tex, cubeLayer);
        }

        vec2 queryShadowArrayLod(sampler2DArrayShadow tex, vec3 uvLayer) {
            return textureQueryLod(tex, uvLayer);
        }

        vec2 queryShadowCubeArrayLod(samplerCubeArrayShadow tex, vec4 cubeLayer) {
            return textureQueryLod(tex, cubeLayer);
        }

        vec2 queryArrayElementLod(sampler2DArray layerMaps[], vec3 uvLayer) {
            return textureQueryLod(layerMaps[2], uvLayer);
        }

        vec2 queryShadowCubeArrayElementLod(samplerCubeArrayShadow cubeShadowArrays[], vec4 cubeLayer) {
            return textureQueryLod(cubeShadowArrays[3], cubeLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = queryArrayLod(layerMap, input.uvLayer);
                vec2 b = queryCubeArrayLod(cubeArray, input.cubeLayer);
                vec2 c = queryShadowArrayLod(shadowArray, input.uvLayer);
                vec2 d = queryShadowCubeArrayLod(cubeShadowArray, input.cubeLayer);
                vec2 e = queryArrayElementLod(layerMaps, input.uvLayer);
                vec2 f = queryShadowCubeArrayElementLod(cubeShadowArrays, input.cubeLayer);
                return vec4(a.x + b.y, c.x + d.y, e.x + f.y, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(binding = 0) uniform sampler2DArray layerMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCubeArray cubeArray;" in generated_code
    assert (
        "layout(binding = 2) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "layout(binding = 3) uniform samplerCubeArrayShadow cubeShadowArray;"
        in generated_code
    )
    assert "layout(binding = 4) uniform sampler2DArray layerMaps[4];" in generated_code
    assert (
        "layout(binding = 8) uniform samplerCubeArrayShadow cubeShadowArrays[4];"
        in generated_code
    )
    assert "vec2 queryArrayLod(sampler2DArray tex, vec3 uvLayer)" in generated_code
    assert (
        "vec2 queryCubeArrayLod(samplerCubeArray tex, vec4 cubeLayer)" in generated_code
    )
    assert (
        "vec2 queryShadowArrayLod(sampler2DArrayShadow tex, vec3 uvLayer)"
        in generated_code
    )
    assert (
        "vec2 queryShadowCubeArrayLod(samplerCubeArrayShadow tex, vec4 cubeLayer)"
        in generated_code
    )
    assert (
        "vec2 queryArrayElementLod(sampler2DArray layerMaps[4], vec3 uvLayer)"
        in generated_code
    )
    assert (
        "vec2 queryShadowCubeArrayElementLod(samplerCubeArrayShadow cubeShadowArrays[4], vec4 cubeLayer)"
        in generated_code
    )
    assert generated_code.count("return textureQueryLod(tex, uvLayer.xy);") == 2
    assert generated_code.count("return textureQueryLod(tex, cubeLayer.xyz);") == 2
    assert "return textureQueryLod(layerMaps[2], uvLayer.xy);" in generated_code
    assert (
        "return textureQueryLod(cubeShadowArrays[3], cubeLayer.xyz);" in generated_code
    )
    assert "textureQueryLod(tex, uvLayer);" not in generated_code
    assert "textureQueryLod(tex, cubeLayer);" not in generated_code
    assert "textureQueryLod(layerMaps[2], uvLayer);" not in generated_code
    assert "textureQueryLod(cubeShadowArrays[3], cubeLayer);" not in generated_code


def test_opengl_texture_operation_variants_passthrough():
    shader = """
    shader TextureOps {
        sampler2D colorMap;

        struct VSOutput {
            vec2 uv;
            ivec2 pixel;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 lodColor = textureLod(colorMap, uv, 1.0);
                vec4 gradColor = textureGrad(colorMap, uv, vec2(0.1), vec2(0.2));
                vec4 fetched = texelFetch(colorMap, pixel, 0);
                vec4 gathered = textureGather(colorMap, uv);
                return lodColor + gradColor + fetched + gathered;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "textureLod(colorMap, uv, 1.0)" in generated_code
    assert "textureGrad(colorMap, uv, vec2(0.1), vec2(0.2))" in generated_code
    assert "texelFetch(colorMap, pixel, 0)" in generated_code
    assert "textureGather(colorMap, uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code


def test_opengl_hlsl_texture_aliases_map_to_glsl_names():
    shader = """
    shader TextureAliases {
        sampler2D colorMap;

        void main() {
            vec2 uv;
            vec4 color = tex2Dlod(colorMap, uv, 1.0);
            vec4 grad = tex2Dgrad(colorMap, uv, vec2(0.1), vec2(0.2));
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "textureLod(colorMap, uv, 1.0)" in generated_code
    assert "textureGrad(colorMap, uv, vec2(0.1), vec2(0.2))" in generated_code
    assert "tex2Dlod(" not in generated_code
    assert "tex2Dgrad(" not in generated_code


def test_opengl_hlsl_saturate_alias_emits_three_argument_clamp():
    # Microsoft HLSL saturate clamps to [0, 1]:
    # https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-saturate
    # Khronos GLSL 4.60 section 8.3 defines clamp(x, minVal, maxVal)
    # as min(max(x, minVal), maxVal):
    # https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html#common-functions
    shader = """
    shader SaturateAlias {
        struct FSInput {
            float weight @ TEXCOORD0;
            vec3 color @ COLOR0;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float alpha = saturate(input.weight);
                vec3 rgb = saturate(input.color);
                return vec4(rgb, alpha);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "float alpha = clamp(weight, 0.0, 1.0);" in generated_code
    assert "vec3 rgb = clamp(color, 0.0, 1.0);" in generated_code
    assert "saturate(" not in generated_code
    assert "clamp(weight)" not in generated_code
    assert "clamp(color)" not in generated_code


def test_opengl_hlsl_saturate_alias_renames_local_clamp_shadow():
    shader = """
    shader SaturateAliasLocalClampShadow {
        fragment {
            vec4 main(float weight @ TEXCOORD0) @ gl_FragColor {
                float clamp = weight;
                float alpha = saturate(weight);
                return vec4(alpha + clamp);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "float clamp_ = weight;" in generated_code
    assert "float alpha = clamp(weight, 0.0, 1.0);" in generated_code
    assert "vec4((alpha + clamp_))" in generated_code
    assert "float clamp = weight;" not in generated_code
    assert "saturate(" not in generated_code


def test_opengl_user_defined_saturate_function_is_preserved():
    shader = """
    shader UserSaturate {
        float saturate(float value) {
            return value * 2.0;
        }

        fragment {
            vec4 main(float value @ TEXCOORD0) @ gl_FragColor {
                float adjusted = saturate(value);
                return vec4(adjusted);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "float saturate(float value)" in generated_code
    assert "float adjusted = saturate(value);" in generated_code
    assert "clamp(" not in generated_code


def test_opengl_hlsl_lerp_alias_renames_local_mix_shadow():
    shader = """
    shader LerpAliasLocalMixShadow {
        fragment {
            vec4 main(float weight @ TEXCOORD0) @ gl_FragColor {
                vec3 cool = vec3(0.1, 0.2, 0.8);
                vec3 warm = vec3(1.0, 0.4, 0.1);
                float mix = weight;
                vec3 color = lerp(cool, warm, weight);
                return vec4(color + vec3(mix), 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "float mix_ = weight;" in generated_code
    assert "vec3 color = mix(cool, warm, weight);" in generated_code
    assert "vec3(mix_)" in generated_code
    assert "float mix = weight;" not in generated_code


def test_opengl_hlsl_aliases_rename_parameter_target_shadows():
    shader = """
    shader AliasParameterTargetShadow {
        vec3 shade(vec3 cool, vec3 warm, float mix, float clamp) {
            vec3 color = lerp(cool, warm, mix);
            float alpha = saturate(clamp);
            return color * alpha;
        }

        fragment {
            vec4 main(float weight @ TEXCOORD0) @ gl_FragColor {
                vec3 color = shade(vec3(0.1), vec3(0.9), weight, weight);
                return vec4(color, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "vec3 shade(vec3 cool, vec3 warm, float mix_, float clamp_)" in generated_code
    )
    assert "vec3 color = mix(cool, warm, mix_);" in generated_code
    assert "float alpha = clamp(clamp_, 0.0, 1.0);" in generated_code
    assert "return (color * alpha);" in generated_code
    assert "float mix, float clamp" not in generated_code
    assert "lerp(" not in generated_code
    assert "saturate(" not in generated_code


def test_opengl_hlsl_rcp_alias_emits_reciprocal_expression():
    # Microsoft HLSL rcp computes a per-component reciprocal:
    # https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/rcp
    # Khronos GLSL 4.60 defines division for scalar/vector floating-point
    # operands:
    # https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html#operators
    shader = """
    shader RcpAlias {
        fragment {
            vec4 main(vec3 value, float scalar) @ gl_FragColor {
                vec3 invValue = rcp(value);
                float invScalar = rcp(scalar);
                return vec4(invValue, invScalar);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "vec3 invValue = (1.0 / value);" in generated_code
    assert "float invScalar = (1.0 / scalar);" in generated_code
    assert "rcp(" not in generated_code


def test_opengl_user_defined_rcp_function_is_preserved():
    shader = """
    shader UserRcp {
        float rcp(float value) {
            return value + 1.0;
        }

        fragment {
            vec4 main(float value) @ gl_FragColor {
                float adjusted = rcp(value);
                return vec4(adjusted);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "float rcp(float value)" in generated_code
    assert "float adjusted = rcp(value);" in generated_code
    assert "1.0 / value" not in generated_code


def test_opengl_hlsl_bitcast_aliases_map_to_glsl_bit_functions():
    # Khronos GLSL 4.60 section 8.3 defines bit-preserving float/int
    # conversion built-ins:
    # https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html#common-functions
    shader = """
    shader BitcastAlias {
        fragment {
            vec4 main(float value, uint bits, vec2 pair) @ gl_FragColor {
                uint packed = asuint(value);
                int signedBits = asint(value);
                float restored = asfloat(bits);
                uvec2 pairBits = asuint(pair);
                vec2 restoredPair = asfloat(pairBits);
                float roundTrip = asfloat(asuint(value));
                return vec4(
                    restored + roundTrip + float(packed) + float(signedBits),
                    restoredPair
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "uint packed = floatBitsToUint(value);" in generated_code
    assert "int signedBits = floatBitsToInt(value);" in generated_code
    assert "float restored = uintBitsToFloat(bits);" in generated_code
    assert "uvec2 pairBits = floatBitsToUint(pair);" in generated_code
    assert "vec2 restoredPair = uintBitsToFloat(pairBits);" in generated_code
    assert (
        "float roundTrip = uintBitsToFloat(floatBitsToUint(value));" in generated_code
    )
    assert "asuint(" not in generated_code
    assert "asint(" not in generated_code
    assert "asfloat(" not in generated_code


def test_opengl_hlsl_bitcast_aliases_elide_same_type_conversions():
    shader = """
    shader BitcastAliasIdentity {
        fragment {
            vec4 main(float value, int signedBits, uint bits, ivec2 signedPair, uvec2 pair) @ gl_FragColor {
                float sameFloat = asfloat(value);
                int sameInt = asint(signedBits);
                uint sameUint = asuint(bits);
                uvec2 samePair = asuint(pair);
                uint unsignedBits = asuint(signedBits);
                uvec2 unsignedPair = asuint(signedPair);
                int signedFromUint = asint(bits);
                ivec2 signedFromPair = asint(pair);
                return vec4(
                    sameFloat + float(sameInt + sameUint + samePair.x),
                    float(unsignedBits + unsignedPair.x),
                    float(signedFromUint + signedFromPair.x),
                    1.0
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "float sameFloat = value;" in generated_code
    assert "int sameInt = signedBits;" in generated_code
    assert "uint sameUint = bits;" in generated_code
    assert "uvec2 samePair = pair;" in generated_code
    assert "uint unsignedBits = uint(signedBits);" in generated_code
    assert "uvec2 unsignedPair = uvec2(signedPair);" in generated_code
    assert "int signedFromUint = int(bits);" in generated_code
    assert "ivec2 signedFromPair = ivec2(pair);" in generated_code
    assert "BitsTo" not in generated_code
    assert "asuint(" not in generated_code
    assert "asint(" not in generated_code
    assert "asfloat(" not in generated_code


def test_opengl_bfloat16_asuint_alias_lowers_to_uint_payload():
    shader = """
    shader BFloat16BitcastAlias {
        uint packScalar(bfloat16_t value) {
            return asuint(value);
        }

        uint packNamedAlias(bfloat value) {
            return asuint(value);
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "uint packScalar(float value)" in generated_code
    assert "uint packNamedAlias(float value)" in generated_code
    assert "return (floatBitsToUint(value) >> 16u);" in generated_code
    assert "bfloat16_t" not in generated_code
    assert "bfloat " not in generated_code
    assert "asuint(" not in generated_code


def test_opengl_bfloat16_as_type_alias_lowers_from_uint_payload():
    shader = """
    shader BFloat16AsTypeAlias {
        bfloat16_t unpackScalar(uint bits) {
            return as_type<bfloat16_t>(bits);
        }

        bfloat unpackNamedAlias(uint bits) {
            return as_type<bfloat>(bits);
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "float unpackScalar(uint bits)" in generated_code
    assert "float unpackNamedAlias(uint bits)" in generated_code
    assert "return uintBitsToFloat((bits << 16u));" in generated_code
    assert "bfloat16_t" not in generated_code
    assert "bfloat " not in generated_code
    assert "as_type<" not in generated_code


def test_opengl_empty_struct_emits_placeholder_member():
    shader = """
    shader EmptyStruct {
        struct Marker {
        }

        float pass_through(float value) {
            return value;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct Marker {\n    int _crossgl_empty;\n};" in generated_code
    assert "struct Marker {}" not in generated_code


def test_opengl_skips_uncalled_helpers_with_unresolved_template_types():
    shader = """
    shader UncalledTemplateHelper {
        T ceildiv(T n, U m) {
            return (n + m - 1) / m;
        }

        compute {
            void main() {
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "T ceildiv" not in generated_code
    assert " U " not in generated_code


def test_opengl_hlsl_bitcast_alias_renames_local_target_shadow():
    shader = """
    shader BitcastAliasLocalShadow {
        fragment {
            vec4 main(float value) @ gl_FragColor {
                uint floatBitsToUint = 1u;
                uint packed = asuint(value);
                return vec4(float(floatBitsToUint + packed));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "uint floatBitsToUint_ = 1u;" in generated_code
    assert "uint packed = floatBitsToUint(value);" in generated_code
    assert "float((floatBitsToUint_ + packed))" in generated_code
    assert "uint floatBitsToUint = 1u;" not in generated_code
    assert "asuint(" not in generated_code


def test_opengl_user_defined_bitcast_alias_function_is_preserved():
    shader = """
    shader UserBitcastAlias {
        uint asuint(float value) {
            return uint(value);
        }

        fragment {
            vec4 main(float value) @ gl_FragColor {
                uint packed = asuint(value);
                return vec4(float(packed));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "uint asuint(float value)" in generated_code
    assert "uint packed = asuint(value);" in generated_code
    assert "floatBitsToUint" not in generated_code


def test_opengl_hlsl_clip_alias_emits_discard_guards():
    # Microsoft HLSL clip discards the current pixel if any component is
    # less than zero:
    # https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-clip
    # Khronos GLSL 4.60 uses the parameterless discard statement in fragment
    # shaders:
    # https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html#jumps
    shader = """
    shader ClipAlias {
        struct FSInput {
            float alpha @ TEXCOORD0;
            vec3 distances @ TEXCOORD1;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                clip(input.alpha - 0.5);
                clip(input.distances);
                return vec4(1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "if ((alpha - 0.5) < 0.0) {\n        discard;\n    }" in generated_code
    assert (
        "if (any(lessThan(distances, vec3(0.0)))) {\n" "        discard;\n" "    }"
    ) in generated_code
    assert "discard(" not in generated_code
    assert "clip(" not in generated_code


def test_opengl_user_defined_clip_function_is_preserved():
    shader = """
    shader UserClip {
        float clip(float value) {
            return value * 2.0;
        }

        struct FSInput {
            float alpha @ TEXCOORD0;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float adjusted = clip(input.alpha);
                return vec4(adjusted);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "float clip(float value)" in generated_code
    assert "float adjusted = clip(alpha);" in generated_code
    assert "discard" not in generated_code


def test_opengl_clip_alias_is_rejected_outside_fragment_stage():
    shader = """
    shader VertexClip {
        vertex {
            vec4 main(vec3 position @ POSITION) @ gl_Position {
                clip(position.x);
                return vec4(position, 1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="OpenGL clip alias lowers to discard and is only valid in fragment",
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_explicit_sampler_argument_is_dropped():
    shader = """
    shader ExplicitSampler {
        sampler2D colorMap;
        sampler linearSampler;
        samplerCube envMap;

        struct VSOutput {
            vec2 uv;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return texture(colorMap, linearSampler, uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCube envMap;" in generated_code
    assert "linearSampler" not in generated_code
    assert "texture(colorMap, uv)" in generated_code


def test_opengl_sampler_parameter_is_dropped():
    shader = """
    shader SamplerParameter {
        sampler2D colorMap;
        sampler linearSampler;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleColor(sampler sampleState, vec2 uv) {
            return texture(colorMap, sampleState, uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleColor(linearSampler, uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "vec4 sampleColor(vec2 uv)" in generated_code
    assert "texture(colorMap, uv)" in generated_code
    assert "sampleColor(uv)" in generated_code
    assert "linearSampler" not in generated_code
    assert "sampleState" not in generated_code


def test_opengl_texture_parameter_keeps_combined_sampler():
    shader = """
    shader TextureParameter {
        sampler2D colorMap;
        sampler linearSampler;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleColor(sampler2D tex, sampler sampleState, vec2 uv) {
            return texture(tex, sampleState, uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleColor(colorMap, linearSampler, uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "vec4 sampleColor(sampler2D tex, vec2 uv)" in generated_code
    assert "texture(tex, uv)" in generated_code
    assert "sampleColor(colorMap, uv)" in generated_code
    assert "linearSampler" not in generated_code
    assert "sampleState" not in generated_code


def test_opengl_struct_member_sampler_expression_is_dropped():
    shader = """
    shader StructSamplerExpression {
        sampler2D colorMap;

        struct SamplerPack {
            sampler samplers[2];
        };

        vec4 samplePacked(sampler2D tex, SamplerPack pack, int index, vec2 uv) {
            return texture(tex, pack.samplers[index], uv);
        }

        fragment {
            vec4 main() @ gl_FragColor {
                SamplerPack pack;
                return samplePacked(colorMap, pack, 0, vec2(0.5));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct SamplerPack" in generated_code
    assert "sampler samplers[2];" in generated_code
    assert (
        "vec4 samplePacked(sampler2D tex, SamplerPack pack, int index, vec2 uv)"
        in generated_code
    )
    assert "return texture(tex, uv);" in generated_code
    assert "texture(tex, pack.samplers[index], uv)" not in generated_code


def test_opengl_implicit_sampler_for_texture_parameter():
    shader = """
    shader TextureParameter {
        sampler2D colorMap;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleColor(sampler2D tex, vec2 uv) {
            return texture(tex, uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleColor(colorMap, uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "vec4 sampleColor(sampler2D tex, vec2 uv)" in generated_code
    assert "texture(tex, uv)" in generated_code
    assert "sampleColor(colorMap, uv)" in generated_code


def test_opengl_shadow_texture_compare():
    shader = """
    shader ShadowTexture {
        sampler2DShadow shadowMap;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return textureCompare(shadowMap, uv, depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert "texture(shadowMap, vec3(uv, depth))" in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_shadow_texture_array_compare():
    shader = """
    shader ShadowTextureArray {
        sampler2DShadow shadowMaps[4];

        struct VSOutput {
            vec2 uv;
            float depth;
            int layer;
        };

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return textureCompare(shadowMaps[layer], uv, depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert "texture(shadowMaps[layer], vec3(uv, depth))" in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_shadow_texture_array_compare_filters_indexed_sampler_array():
    shader = """
    shader ShadowSamplerArrayHelper {
        sampler2DShadow shadowMaps[4];
        sampler shadowSamplers[4];

        struct VSOutput {
            vec2 uv;
            float depth;
            int layer;
        };

        float shadowLayer(sampler2DShadow shadowMaps[4], sampler shadowSamplers[4], int layer, vec2 uv, float depth) {
            return textureCompare(shadowMaps[layer], shadowSamplers[layer], uv, depth);
        }

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return shadowLayer(shadowMaps, shadowSamplers, layer, uv, depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[4], int layer, vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[layer], vec3(uv, depth))" in generated_code
    assert "shadowLayer(shadowMaps, layer, uv, depth)" in generated_code
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[layer]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_fixed_shadow_texture_array_keeps_declared_size_with_constant_indices():
    shader = """
    shader FixedShadowArrayConstantIndex {
        const int LAYER = 2;
        sampler2DShadow shadowMaps[6];
        sampler shadowSamplers[6];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[6], sampler shadowSamplers[6], vec2 uv, float depth) {
            return textureCompare(shadowMaps[LAYER], shadowSamplers[LAYER], uv, depth) + textureCompare(shadowMaps[1 + 2], shadowSamplers[1 + 2], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int LAYER = 2;" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[6];" in generated_code
    )
    assert "layout(binding = 6) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[6], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[LAYER], vec3(uv, depth))" in generated_code
    assert "texture(shadowMaps[(1 + 2)], vec3(uv, depth))" in generated_code
    assert "shadowLayer(shadowMaps, uv, depth)" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];"
        not in generated_code
    )
    assert (
        "layout(binding = 4) uniform sampler2DShadow afterShadow;" not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "shadowSamplers[LAYER]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_fixed_shadow_texture_array_resolves_constant_declared_size_for_bindings():
    shader = """
    shader ConstSizedShadowResourceArrays {
        const int BASE_COUNT = 2;
        const int SHADOW_COUNT = BASE_COUNT * 3;
        sampler2DShadow shadowMaps[SHADOW_COUNT];
        sampler shadowSamplers[SHADOW_COUNT];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[SHADOW_COUNT], sampler shadowSamplers[SHADOW_COUNT], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2], shadowSamplers[2], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int BASE_COUNT = 2;" in generated_code
    assert "const int SHADOW_COUNT = (BASE_COUNT * 3);" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[SHADOW_COUNT];"
        in generated_code
    )
    assert "layout(binding = 6) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[SHADOW_COUNT], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[2], vec3(uv, depth))" in generated_code
    assert "shadowLayer(shadowMaps, uv, depth)" in generated_code
    assert (
        "layout(binding = 1) uniform sampler2DShadow afterShadow;" not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_fixed_shadow_texture_array_resolves_inline_declared_size_expression_for_bindings():
    shader = """
    shader ExprSizedShadowResourceArrays {
        sampler2DShadow shadowMaps[2 * 3];
        sampler shadowSamplers[2 * 3];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[2 * 3], sampler shadowSamplers[2 * 3], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2], shadowSamplers[2], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[(2 * 3)];"
        in generated_code
    )
    assert "layout(binding = 6) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[(2 * 3)], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[2], vec3(uv, depth))" in generated_code
    assert "shadowLayer(shadowMaps, uv, depth)" in generated_code
    assert (
        "layout(binding = 1) uniform sampler2DShadow afterShadow;" not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "[None]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_fixed_shadow_texture_arrays_preserve_parenthesized_and_unary_declared_sizes():
    shader = """
    shader ParenthesizedSizedShadowResourceArrays {
        sampler2DShadow shadowMaps[(2 + 1) * 2];
        sampler shadowSamplers[(2 + 1) * 2];
        sampler2DShadow unaryShadowMaps[+6];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[(2 + 1) * 2], sampler shadowSamplers[(2 + 1) * 2], sampler2DShadow unaryShadowMaps[+6], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2], shadowSamplers[2], uv, depth) + textureCompare(unaryShadowMaps[2], shadowSamplers[2], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, unaryShadowMaps, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[((2 + 1) * 2)];"
        in generated_code
    )
    assert (
        "layout(binding = 6) uniform sampler2DShadow unaryShadowMaps[(+6)];"
        in generated_code
    )
    assert "layout(binding = 12) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[((2 + 1) * 2)], sampler2DShadow unaryShadowMaps[(+6)], vec2 uv, float depth)"
        in generated_code
    )
    assert "shadowLayer(shadowMaps, unaryShadowMaps, uv, depth)" in generated_code
    assert "texture(shadowMaps[2], vec3(uv, depth))" in generated_code
    assert "texture(unaryShadowMaps[2], vec3(uv, depth))" in generated_code
    assert (
        "layout(binding = 6) uniform sampler2DShadow afterShadow;" not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "[None]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_unsized_shadow_texture_array_infers_helper_size_and_filters_sampler_array():
    shader = """
    shader UnsizedShadowSamplerArrayHelper {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            float high = textureCompare(shadowMaps[3], shadowSamplers[3], uv, depth);
            float low = textureCompare(shadowMaps[1], shadowSamplers[1], uv, depth);
            return high + low;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert "layout(binding = 4) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[4], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[3], vec3(uv, depth))" in generated_code
    assert "texture(shadowMaps[1], vec3(uv, depth))" in generated_code
    assert "texture(afterShadow, vec3(uv, depth))" in generated_code
    assert "shadowLayer(shadowMaps, uv, depth)" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[];"
        not in generated_code
    )
    assert (
        "layout(binding = 1) uniform sampler2DShadow afterShadow;" not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[3]" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_unsized_shadow_texture_array_infers_transitive_helper_size():
    shader = """
    shader MultiHopUnsizedShadowResources {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowDeep(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            float high = textureCompare(shadowMaps[4], shadowSamplers[4], uv, depth);
            float low = textureCompare(shadowMaps[1], shadowSamplers[1], uv, depth);
            return high + low;
        }

        float shadowMid(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return shadowDeep(shadowMaps, shadowSamplers, uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowMid(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[5];" in generated_code
    )
    assert "layout(binding = 5) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowDeep(sampler2DShadow shadowMaps[5], vec2 uv, float depth)"
        in generated_code
    )
    assert (
        "float shadowMid(sampler2DShadow shadowMaps[5], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[4], vec3(uv, depth))" in generated_code
    assert "texture(shadowMaps[1], vec3(uv, depth))" in generated_code
    assert "shadowDeep(shadowMaps, uv, depth)" in generated_code
    assert "shadowMid(shadowMaps, uv, depth)" in generated_code
    assert "texture(afterShadow, vec3(uv, depth))" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[];"
        not in generated_code
    )
    assert (
        "layout(binding = 1) uniform sampler2DShadow afterShadow;" not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[4]" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_unsized_shadow_texture_array_preserves_dynamic_indexing():
    shader = """
    shader MixedIndexedUnsizedShadowResources {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
            int layer;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], int layer, vec2 uv, float depth) {
            float dynamicShadow = textureCompare(shadowMaps[layer], shadowSamplers[layer], uv, depth);
            float fixedShadow = textureCompare(shadowMaps[3], shadowSamplers[3], uv, depth);
            return dynamicShadow + fixedShadow;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, layer, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert "layout(binding = 4) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[4], int layer, vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[layer], vec3(uv, depth))" in generated_code
    assert "texture(shadowMaps[3], vec3(uv, depth))" in generated_code
    assert "shadowLayer(shadowMaps, layer, uv, depth)" in generated_code
    assert "texture(afterShadow, vec3(uv, depth))" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[];"
        not in generated_code
    )
    assert (
        "layout(binding = 1) uniform sampler2DShadow afterShadow;" not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[layer]" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_unsized_shadow_texture_array_ignores_unsupported_indices():
    dynamic_shader = """
    shader DynamicOnlyUnsizedShadowResources {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
            int layer;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], int layer, vec2 uv, float depth) {
            return textureCompare(shadowMaps[layer], shadowSamplers[layer], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, layer, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """
    negative_shader = """
    shader NegativeIndexedUnsizedShadowResources {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return textureCompare(shadowMaps[-1], shadowSamplers[-1], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    dynamic_code = GLSLCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = GLSLCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "layout(binding = 0) uniform sampler2DShadow shadowMaps[1];" in dynamic_code
    assert "layout(binding = 1) uniform sampler2DShadow afterShadow;" in dynamic_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[1], int layer, vec2 uv, float depth)"
        in dynamic_code
    )
    assert "texture(shadowMaps[layer], vec3(uv, depth))" in dynamic_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[];" not in dynamic_code
    )
    assert (
        "layout(binding = 2) uniform sampler2DShadow afterShadow;" not in dynamic_code
    )
    assert "sampler shadowSamplers" not in dynamic_code
    assert "shadowSamplers[layer]" not in dynamic_code
    assert "textureCompare(" not in dynamic_code

    assert "layout(binding = 0) uniform sampler2DShadow shadowMaps[1];" in negative_code
    assert "layout(binding = 1) uniform sampler2DShadow afterShadow;" in negative_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[1], vec2 uv, float depth)"
        in negative_code
    )
    assert "texture(shadowMaps[0], vec3(uv, depth))" in negative_code
    assert "texture(shadowMaps[(-1)], vec3(uv, depth))" not in negative_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[0];"
        not in negative_code
    )
    assert (
        "layout(binding = 0) uniform sampler2DShadow afterShadow;" not in negative_code
    )
    assert "sampler shadowSamplers" not in negative_code
    assert "textureCompare(" not in negative_code


def test_opengl_unsized_shadow_texture_array_infers_constant_expression_size():
    shader = """
    shader ExprIndexedUnsizedShadowResources {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2 * 2], shadowSamplers[2 * 2], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[5];" in generated_code
    )
    assert "layout(binding = 5) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[5], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[(2 * 2)], vec3(uv, depth))" in generated_code
    assert "texture(afterShadow, vec3(uv, depth))" in generated_code
    assert "shadowLayer(shadowMaps, uv, depth)" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[];"
        not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_unsized_shadow_texture_array_infers_named_constant_size():
    shader = """
    shader ConstIndexedUnsizedShadowResources {
        const int BASE = 2;
        const int LAYER = BASE * 2;
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return textureCompare(shadowMaps[LAYER], shadowSamplers[LAYER], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int BASE = 2;" in generated_code
    assert "const int LAYER = (BASE * 2);" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[5];" in generated_code
    )
    assert "layout(binding = 5) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[5], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[LAYER], vec3(uv, depth))" in generated_code
    assert "texture(afterShadow, vec3(uv, depth))" in generated_code
    assert "shadowLayer(shadowMaps, uv, depth)" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[];"
        not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_fixed_shadow_texture_array_rejects_mismatched_fixed_helper_size():
    shader = """
    shader FixedShadowGlobalMismatch {
        sampler2DShadow shadowMaps[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        float shadowThree(sampler2DShadow shadowMaps[3], sampler shadowSamplers[3], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2], shadowSamplers[2], uv, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return shadowThree(shadowMaps, shadowSamplers, input.uv, input.depth);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'shadowMaps': 4 and 3",
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_fixed_shadow_texture_array_widens_unsized_helper():
    shader = """
    shader FixedShadowGlobalToUnsizedHelper {
        sampler2DShadow shadowMaps[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        float shadowUnsized(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2], shadowSamplers[2], uv, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return shadowUnsized(shadowMaps, shadowSamplers, input.uv, input.depth);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert (
        "float shadowUnsized(sampler2DShadow shadowMaps[4], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[2], vec3(uv, depth))" in generated_code
    assert "shadowUnsized(shadowMaps, uv, depth)" in generated_code
    assert "sampler shadowSamplers" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_transitive_shadow_texture_array_shadowed_const_index_stays_dynamic():
    shader = """
    shader TransitiveShadowSamplerShadowedConstIndex {
        const int COUNT = 4;
        sampler2DShadow shadowMaps[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        float leaf(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            int COUNT = 0;
            return textureCompare(shadowMaps[COUNT], shadowSamplers[COUNT], uv, depth);
        }

        float passThrough(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            int COUNT = 0;
            float sampled = textureCompare(shadowMaps[COUNT], shadowSamplers[COUNT], uv, depth);
            return sampled + leaf(shadowMaps, shadowSamplers, uv, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return passThrough(shadowMaps, shadowSamplers, input.uv, input.depth);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int COUNT = 4;" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert (
        "float leaf(sampler2DShadow shadowMaps[4], vec2 uv, float depth)"
        in generated_code
    )
    assert (
        "float passThrough(sampler2DShadow shadowMaps[4], vec2 uv, float depth)"
        in generated_code
    )
    assert generated_code.count("int COUNT = 0;") == 2
    assert "texture(shadowMaps[COUNT], vec3(uv, depth))" in generated_code
    assert "leaf(shadowMaps, uv, depth)" in generated_code
    assert "passThrough(shadowMaps, uv, depth)" in generated_code
    assert "sampler shadowSamplers" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_transitive_shadow_texture_array_unshadowed_const_index_conflict_raises():
    shader = """
    shader TransitiveShadowSamplerUnshadowedConstIndexConflict {
        const int COUNT = 4;
        sampler2DShadow shadowMaps[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        float leaf(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return textureCompare(shadowMaps[COUNT], shadowSamplers[COUNT], uv, depth);
        }

        float passThrough(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            int COUNT = 0;
            return leaf(shadowMaps, shadowSamplers, uv, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return passThrough(shadowMaps, shadowSamplers, input.uv, input.depth);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'shadowMaps': 4 and 5",
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_implicit_sampler_for_shadow_texture_parameter():
    shader = """
    shader ShadowParameter {
        sampler2DShadow shadowMap;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float sampleShadow(sampler2DShadow tex, vec2 uv, float depth) {
            return textureCompare(tex, uv, depth);
        }

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return sampleShadow(shadowMap, uv, depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "float sampleShadow(sampler2DShadow tex, vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(tex, vec3(uv, depth))" in generated_code
    assert "sampleShadow(shadowMap, uv, depth)" in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_shadow_texture_parameter_keeps_combined_sampler():
    shader = """
    shader ShadowParameter {
        sampler2DShadow shadowMap;
        sampler shadowSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float sampleShadow(sampler2DShadow tex, sampler compareSampler, vec2 uv, float depth) {
            return textureCompare(tex, compareSampler, uv, depth);
        }

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return sampleShadow(shadowMap, shadowSampler, uv, depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "float sampleShadow(sampler2DShadow tex, vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(tex, vec3(uv, depth))" in generated_code
    assert "sampleShadow(shadowMap, uv, depth)" in generated_code
    assert "shadowSampler" not in generated_code
    assert "compareSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_shadow_compare_sampler_parameter_is_dropped():
    shader = """
    shader ShadowHelper {
        sampler2DShadow shadowMap;
        sampler shadowSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float sampleShadow(sampler compareSampler, vec2 uv, float depth) {
            return textureCompare(shadowMap, compareSampler, uv, depth);
        }

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return sampleShadow(shadowSampler, uv, depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert "float sampleShadow(vec2 uv, float depth)" in generated_code
    assert "texture(shadowMap, vec3(uv, depth))" in generated_code
    assert "sampleShadow(uv, depth)" in generated_code
    assert "shadowSampler" not in generated_code
    assert "compareSampler" not in generated_code
    assert "textureCompare(" not in generated_code


@pytest.mark.parametrize(
    "shader, expected_outputs",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    int b = 2;
                    int c = a << b;
                    int d = a >> b;
                }
            }
            """,
            ["a << b", "a >> b"],
        )
    ],
)
def test_shift_operators(shader, expected_outputs):
    ast = crosstl.translator.parse(shader)
    code_gen = GLSLCodeGen()
    generated_code = code_gen.generate(ast)

    for expected in expected_outputs:
        assert expected in generated_code


def test_glsl_sampler_1d_array_sampling_and_queries():
    shader = """
    shader OneDArrayTexture {
        sampler1DArray lineArray;
        sampler linearSampler;

        vec4 sampleLineArray(
            sampler1DArray tex,
            sampler samp,
            vec2 uvLayer,
            ivec2 pixelLayer,
            int lod,
            int offset
        ) {
            ivec2 dims = textureSize(tex, lod);
            int levels = textureQueryLevels(tex);
            vec2 lodInfo = textureQueryLod(tex, samp, uvLayer);
            return texture(tex, samp, uvLayer)
                + textureLod(tex, samp, uvLayer, lod)
                + texelFetch(tex, pixelLayer, lod)
                + texelFetchOffset(tex, pixelLayer, lod, offset)
                + vec4(dims.x, dims.y, levels, lodInfo.x + lodInfo.y);
        }

        fragment {
            vec4 main() @gl_FragColor {
                return sampleLineArray(
                    lineArray,
                    linearSampler,
                    vec2(0.5, 0.0),
                    ivec2(4, 0),
                    0,
                    1
                );
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler1DArray lineArray;" in generated_code
    assert "sampler linearSampler" not in generated_code
    assert (
        "vec4 sampleLineArray(sampler1DArray tex, vec2 uvLayer, ivec2 pixelLayer, int lod, int offset)"
        in generated_code
    )
    assert "ivec2 dims = textureSize(tex, lod);" in generated_code
    assert "int levels = textureQueryLevels(tex);" in generated_code
    assert "vec2 lodInfo = textureQueryLod(tex, uvLayer.x);" in generated_code
    assert "texture(tex, uvLayer)" in generated_code
    assert "textureLod(tex, uvLayer, lod)" in generated_code
    assert "texelFetch(tex, pixelLayer, lod)" in generated_code
    assert (
        glsl_dynamic_texel_offset_diagnostic(
            "texel fetch offset", "texelFetchOffset", "vec4(0.0)"
        )
        in generated_code
    )
    assert "texelFetchOffset(" not in generated_code
    assert "textureQueryLod(tex, uvLayer);" not in generated_code
    assert (
        "sampleLineArray(lineArray, vec2(0.5, 0.0), ivec2(4, 0), 0, 1)"
        in generated_code
    )


def test_glsl_image_1d_and_1d_array_storage_operations():
    shader = """
    shader OneDStorageImages {
        image1D line;
        image1DArray layers;
        uimage1D counters @r32ui;
        uimage1DArray layerCounters @r32ui;

        float touchLine(image1D image, int x, float value) {
            float oldValue = imageLoad(image, x);
            imageStore(image, x, oldValue + value);
            return oldValue;
        }

        vec4 touchLayer(image1DArray image, ivec2 coord, vec4 value) {
            vec4 oldValue = imageLoad(image, coord);
            imageStore(image, coord, oldValue + value);
            return oldValue;
        }

        uint addCounter(uimage1D image @r32ui, int x, uint value) {
            return imageAtomicAdd(image, x, value);
        }

        uint exchangeLayer(uimage1DArray image @r32ui, ivec2 coord, uint value) {
            return imageAtomicExchange(image, coord, value);
        }

        uint compareLayer(uimage1DArray image @r32ui, ivec2 coord, uint expected, uint value) {
            return imageAtomicCompSwap(image, coord, expected, value);
        }

        compute {
            void main() {
                int sizeLine = imageSize(line);
                ivec2 sizeLayer = imageSize(layers);
                float a = touchLine(line, 1, 0.25);
                vec4 b = touchLayer(layers, ivec2(2, 3), vec4(1.0));
                uint c = addCounter(counters, 4, 7u);
                uint d = exchangeLayer(layerCounters, ivec2(5, 6), 8u);
                uint e = compareLayer(layerCounters, ivec2(7, 8), d, c);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(rgba32f, binding = 0) uniform image1D line;" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image1DArray layers;" in generated_code
    assert "layout(r32ui, binding = 2) uniform uimage1D counters;" in generated_code
    assert (
        "layout(r32ui, binding = 3) uniform uimage1DArray layerCounters;"
        in generated_code
    )
    assert "float oldValue = imageLoad(image, x).x;" in generated_code
    assert "imageStore(image, x, vec4((oldValue + value)));" in generated_code
    assert "vec4 oldValue = imageLoad(image, coord);" in generated_code
    assert "imageStore(image, coord, (oldValue + value));" in generated_code
    assert (
        f"return {glsl_image_atomic_parameter_diagnostic('imageAtomicAdd', 'uimage1D', '0u')};"
        in generated_code
    )
    assert (
        f"return {glsl_image_atomic_parameter_diagnostic('imageAtomicExchange', 'uimage1DArray', '0u')};"
        in generated_code
    )
    assert (
        f"return {glsl_image_atomic_parameter_diagnostic('imageAtomicCompSwap', 'uimage1DArray', '0u')};"
        in generated_code
    )


def test_glsl_cube_storage_images_lower_load_store_and_atomics():
    shader = """
    shader GLSLCubeStorageImages {
        imageCube cube;
        imageCubeArray cubeArray @rgba16f;
        iimageCube signedCube @r32i;
        uimageCubeArray unsignedCubeArray @r32ui;

        float touchCube(imageCube image, ivec3 coord, float value) {
            float oldValue = imageLoad(image, coord);
            imageStore(image, coord, oldValue + value);
            return oldValue;
        }

        vec4 touchCubeArray(imageCubeArray image @rgba16f, ivec3 coord, vec4 value) {
            vec4 oldLayer = imageLoad(image, coord);
            imageStore(image, coord, oldLayer + value);
            return oldLayer;
        }

        compute {
            void main() {
                ivec2 cubeSize = imageSize(cube);
                ivec3 arraySize = imageSize(cubeArray);
                float a = touchCube(cube, ivec3(0, 1, 2), 0.5);
                vec4 b = touchCubeArray(cubeArray, ivec3(3, 4, 5), vec4(a));
                int signedOld = imageAtomicExchange(signedCube, ivec3(1, 2, 3), -1);
                imageStore(signedCube, ivec3(2, 3, 4), signedOld + 1);
                uint unsignedOld = imageAtomicAdd(unsignedCubeArray, ivec3(4, 5, 6), 7u);
                imageStore(unsignedCubeArray, ivec3(5, 6, 7), unsignedOld + 1u);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(rgba32f, binding = 0) uniform imageCube cube;" in generated_code
    assert (
        "layout(rgba16f, binding = 1) uniform imageCubeArray cubeArray;"
        in generated_code
    )
    assert "layout(r32i, binding = 2) uniform iimageCube signedCube;" in generated_code
    assert (
        "layout(r32ui, binding = 3) uniform uimageCubeArray unsignedCubeArray;"
        in generated_code
    )
    assert "ivec2 cubeSize = imageSize(cube);" in generated_code
    assert "ivec3 arraySize = imageSize(cubeArray);" in generated_code
    assert "float oldValue = imageLoad(image, coord).x;" in generated_code
    assert "imageStore(image, coord, vec4((oldValue + value)));" in generated_code
    assert "vec4 oldLayer = imageLoad(image, coord);" in generated_code
    assert "imageStore(image, coord, (oldLayer + value));" in generated_code
    assert (
        "int signedOld = imageAtomicExchange(signedCube, ivec3(1, 2, 3), (-1));"
        in generated_code
    )
    assert (
        "imageStore(signedCube, ivec3(2, 3, 4), ivec4((signedOld + 1)));"
        in generated_code
    )
    assert (
        "uint unsignedOld = imageAtomicAdd(unsignedCubeArray, ivec3(4, 5, 6), 7u);"
        in generated_code
    )
    assert (
        "imageStore(unsignedCubeArray, ivec3(5, 6, 7), uvec4((unsignedOld + 1u)));"
        in generated_code
    )


def test_glsl_cube_storage_image_access_rejects_indexed_writeonly_load():
    shader = """
    shader GLSLCubeStorageImageAccessInvalid {
        imageCubeArray targets[2] @writeonly;

        compute {
            void main() {
                vec4 value = imageLoad(targets[0], ivec3(0, 1, 2));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)

    with pytest.raises(ValueError, match="requires read-capable storage image access"):
        GLSLCodeGen().generate(ast)


def test_glsl_multisample_storage_images_lower_load_store_and_atomics():
    shader = """
    shader GLSLMultisampleStorageImages {
        image2DMS colorImage @rgba16f;
        uimage2DMS counters @r32ui;
        image2DMSArray layered @rgba16f;

        vec4 touch(
            image2DMS image @rgba16f,
            uimage2DMS counterImage @r32ui,
            ivec2 pixel,
            int sampleIndex,
            vec4 value,
            uint count
        ) {
            vec4 oldColor = imageLoad(image, pixel, sampleIndex);
            uint oldCount = imageLoad(counterImage, pixel, sampleIndex);
            imageStore(image, pixel, sampleIndex, oldColor + value);
            imageStore(counterImage, pixel, sampleIndex, oldCount + count);
            uint atomicOld = imageAtomicAdd(counterImage, pixel, sampleIndex, count);
            uint exchanged = imageAtomicExchange(counterImage, pixel, sampleIndex, atomicOld + count);
            uint swapped = imageAtomicCompSwap(counterImage, pixel, sampleIndex, exchanged, count);
            return oldColor + vec4(float(oldCount + atomicOld + swapped));
        }

        vec4 touchLayer(image2DMSArray image @rgba16f, ivec3 pixelLayer, int sampleIndex, vec4 value) {
            vec4 oldLayer = imageLoad(image, pixelLayer, sampleIndex);
            imageStore(image, pixelLayer, sampleIndex, oldLayer + value);
            return oldLayer;
        }

        compute {
            void main() {
                vec4 color = touch(colorImage, counters, ivec2(0, 1), 2, vec4(1.0), 3u);
                vec4 layerColor = touchLayer(layered, ivec3(2, 3, 1), 0, color);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(rgba16f, binding = 0) uniform image2DMS colorImage;" in generated_code
    )
    assert "layout(r32ui, binding = 1) uniform uimage2DMS counters;" in generated_code
    assert (
        "layout(rgba16f, binding = 2) uniform image2DMSArray layered;" in generated_code
    )
    assert (
        "vec4 touch(image2DMS image, uimage2DMS counterImage, ivec2 pixel, int sampleIndex, vec4 value, uint count)"
        in generated_code
    )
    assert "vec4 oldColor = imageLoad(image, pixel, sampleIndex);" in generated_code
    assert (
        "uint oldCount = imageLoad(counterImage, pixel, sampleIndex).x;"
        in generated_code
    )
    assert (
        "imageStore(image, pixel, sampleIndex, (oldColor + value));" in generated_code
    )
    assert (
        "imageStore(counterImage, pixel, sampleIndex, uvec4((oldCount + count)));"
        in generated_code
    )
    assert (
        "uint atomicOld = "
        f"{glsl_image_atomic_parameter_diagnostic('imageAtomicAdd', 'uimage2DMS', '0u')};"
        in generated_code
    )
    assert (
        "uint exchanged = "
        f"{glsl_image_atomic_parameter_diagnostic('imageAtomicExchange', 'uimage2DMS', '0u')};"
        in generated_code
    )
    assert (
        "uint swapped = "
        f"{glsl_image_atomic_parameter_diagnostic('imageAtomicCompSwap', 'uimage2DMS', '0u')};"
        in generated_code
    )
    assert (
        "vec4 oldLayer = imageLoad(image, pixelLayer, sampleIndex);" in generated_code
    )
    assert (
        "imageStore(image, pixelLayer, sampleIndex, (oldLayer + value));"
        in generated_code
    )


@pytest.mark.parametrize(
    ("shader", "match"),
    [
        (
            """
            shader StorageImageLoadMatrixResultInvalid {
                image2D image @rg32f;

                compute {
                    void main() {
                        mat2 value = imageLoad(image, ivec2(0, 0));
                    }
                }
            }
            """,
            "requires scalar or vector result context.*got mat2",
        ),
        (
            """
            shader StorageImageLoadMatrixReturnInvalid {
                image2D image;

                mat4 readPixel() {
                    return imageLoad(image, ivec2(0, 0));
                }

                compute {
                    void main() {
                    }
                }
            }
            """,
            "requires scalar or vector result context.*got mat4",
        ),
    ],
)
def test_glsl_storage_image_load_rejects_matrix_result_contexts(shader, match):
    ast = crosstl.translator.parse(shader)

    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate(ast)


@pytest.mark.parametrize(
    ("shader", "match"),
    [
        (
            """
            shader StorageImageStoreMatrixConstructorInvalid {
                image2D image @rg32f;

                compute {
                    void main() {
                        imageStore(image, ivec2(0, 0), mat2(1.0));
                    }
                }
            }
            """,
            "requires scalar or vector value.*has type mat2",
        ),
        (
            """
            shader StorageImageStoreMatrixVariableInvalid {
                image2D image;

                compute {
                    void main() {
                        mat4 value = mat4(1.0);
                        imageStore(image, ivec2(0, 0), value);
                    }
                }
            }
            """,
            "requires scalar or vector value.*has type mat4",
        ),
    ],
)
def test_glsl_storage_image_store_rejects_matrix_values(shader, match):
    ast = crosstl.translator.parse(shader)

    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate(ast)


def test_glsl_storage_image_access_attributes_emit_parameter_qualifiers():
    shader = """
    shader StorageImageAccess {
        image2D outImage @writeonly;

        float readPixel(image2D image @readonly, ivec2 pixel) {
            return imageLoad(image, pixel);
        }

        void writePixel(image2D image @writeonly, ivec2 pixel, vec4 value) {
            imageStore(image, pixel, value);
        }

        compute {
            void main() {
                imageStore(outImage, ivec2(0, 0), vec4(1.0));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(rgba32f, binding = 0) writeonly uniform image2D outImage;"
        in generated_code
    )
    assert "float readPixel(readonly image2D image, ivec2 pixel)" in generated_code
    assert (
        "void writePixel(writeonly image2D image, ivec2 pixel, vec4 value)"
        in generated_code
    )
    assert "image2D image readonly" not in generated_code
    assert "image2D image writeonly" not in generated_code


def test_glsl_storage_image_access_function_attributes_emit_qualifiers():
    shader = """
    shader StorageImageAccessFunctionSpelling {
        image2D source @access(readonly);
        image2D target @access(writeonly);

        float readPixel(image2D image @access(readonly), ivec2 pixel) {
            return imageLoad(image, pixel);
        }

        void writePixel(image2D image @access(writeonly), ivec2 pixel, vec4 value) {
            imageStore(image, pixel, value);
        }

        compute {
            void main() {
                float value = readPixel(source, ivec2(0, 0));
                writePixel(target, ivec2(0, 0), vec4(value));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(rgba32f, binding = 0) readonly uniform image2D source;"
        in generated_code
    )
    assert (
        "layout(rgba32f, binding = 1) writeonly uniform image2D target;"
        in generated_code
    )
    assert "float readPixel(readonly image2D image, ivec2 pixel)" in generated_code
    assert (
        "void writePixel(writeonly image2D image, ivec2 pixel, vec4 value)"
        in generated_code
    )


def test_glsl_storage_image_access_accepts_equivalent_aliases():
    shader = """
    shader StorageImageAccessEquivalentAliases {
        image2D source @readonly @access(readonly);

        fragment {
            vec4 main(
                image2D target[2] @writeonly @access(writeonly) @binding(4)
            ) @gl_FragColor {
                float loaded = imageLoad(source, ivec2(0, 0));
                imageStore(target[1], ivec2(0, 0), vec4(loaded));
                return vec4(loaded);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "layout(rgba32f, binding = 0) readonly uniform image2D source;"
        in generated_code
    )
    assert (
        "layout(rgba32f, binding = 4) writeonly uniform image2D target[2];"
        in generated_code
    )
    assert "float loaded = imageLoad(source, ivec2(0, 0)).x;" in generated_code
    assert "imageStore(target[1], ivec2(0, 0), vec4(loaded));" in generated_code


def test_glsl_storage_image_access_readwrite_operations_stay_qualifier_free():
    shader = """
    shader StorageImageReadwriteOperations {
        uimage2D counters @r32ui @access(readwrite);

        uint bump(uimage2D image @r32ui @access(readwrite), ivec2 pixel, uint value) {
            uint oldValue = imageAtomicAdd(image, pixel, value);
            imageStore(image, pixel, oldValue + 1u);
            return imageLoad(image, pixel);
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 0);
                uint directValue = imageLoad(counters, pixel);
                imageStore(counters, pixel, directValue + 1u);
                uint oldValue = bump(counters, pixel, 2u);
                imageStore(counters, ivec2(1, 0), oldValue + 3u);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r32ui, binding = 0) uniform uimage2D counters;" in generated_code
    assert "readwrite uniform uimage2D" not in generated_code
    assert "uint bump(uimage2D image, ivec2 pixel, uint value)" in generated_code
    assert "readwrite uimage2D image" not in generated_code
    assert "uint directValue = imageLoad(counters, pixel).x;" in generated_code
    assert "imageStore(counters, pixel, uvec4((directValue + 1u)));" in generated_code
    assert "uint oldValue = imageAtomicAdd(counters, pixel, value);" in generated_code
    assert "imageStore(counters, pixel, uvec4((oldValue + 1u)));" in generated_code
    assert "return imageLoad(counters, pixel).x;" in generated_code


@pytest.mark.parametrize(
    ("shader", "match"),
    [
        (
            """
            shader StorageImageAccessConflictGlobal {
                image2D image @readonly @writeonly;

                compute {
                    void main() {
                    }
                }
            }
            """,
            "Conflicting resource access metadata on global variable 'image': "
            "@readonly, @writeonly",
        ),
        (
            """
            shader StorageImageAccessConflictParameter {
                void writePixel(image2D image @access(readonly) @writeonly) {
                }

                compute {
                    void main() {
                    }
                }
            }
            """,
            "Conflicting resource access metadata on parameter 'image' of "
            "function 'writePixel': @readonly, @writeonly",
        ),
        (
            """
            shader StorageImageAccessConflictReadWrite {
                image2D image @readwrite @readonly;

                compute {
                    void main() {
                    }
                }
            }
            """,
            "Conflicting resource access metadata on global variable 'image': "
            "@readonly, @readwrite",
        ),
    ],
)
def test_glsl_storage_image_access_rejects_conflicting_metadata(shader, match):
    with pytest.raises(ValueError, match=match):
        ast = crosstl.translator.parse(shader)
        GLSLCodeGen().generate(ast)


@pytest.mark.parametrize(
    ("shader", "message"),
    [
        (
            """
            shader DuplicateGlobalDirectImageAccess {
                image2D image @readonly @readonly;

                compute {
                    void main() {
                    }
                }
            }
            """,
            "Duplicate OpenGL resource access metadata for 'image': @readonly",
        ),
        (
            """
            shader DuplicateGlobalFunctionImageAccess {
                image2D image @access(readonly) @access(readonly);

                compute {
                    void main() {
                    }
                }
            }
            """,
            "Duplicate OpenGL resource access metadata for 'image': "
            "@access(readonly)",
        ),
        (
            """
            shader DuplicateStageParameterDirectImageAccess {
                fragment {
                    vec4 main(image2D target[2] @writeonly @writeonly) @gl_FragColor {
                        return vec4(1.0);
                    }
                }
            }
            """,
            "Duplicate OpenGL resource access metadata for 'target': @writeonly",
        ),
        (
            """
            shader DuplicateStageParameterFunctionImageAccess {
                fragment {
                    vec4 main(image2D target[2] @access(writeonly) @access(writeonly)) @gl_FragColor {
                        return vec4(1.0);
                    }
                }
            }
            """,
            "Duplicate OpenGL resource access metadata for 'target': "
            "@access(writeonly)",
        ),
    ],
)
def test_glsl_storage_image_access_rejects_duplicate_metadata(shader, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


@pytest.mark.parametrize(
    ("shader", "match"),
    [
        (
            """
            shader StorageImageAccessInvalidLoad {
                float readPixel(image2D image @writeonly, ivec2 pixel) {
                    return imageLoad(image, pixel);
                }

                compute {
                    void main() {
                    }
                }
            }
            """,
            "requires read-capable storage image access",
        ),
        (
            """
            shader StorageImageAccessInvalidStore {
                void writePixel(image2D image @readonly, ivec2 pixel, vec4 value) {
                    imageStore(image, pixel, value);
                }

                compute {
                    void main() {
                    }
                }
            }
            """,
            "requires write-capable storage image access",
        ),
        (
            """
            shader StorageImageAccessInvalidAtomic {
                uint addCounter(uimage2D image @r32ui @readonly, ivec2 pixel, uint value) {
                    return imageAtomicAdd(image, pixel, value);
                }

                compute {
                    void main() {
                    }
                }
            }
            """,
            "requires read-write storage image access",
        ),
        (
            """
            shader StorageImageAccessInvalidGlobal {
                image2D outImage @writeonly;

                compute {
                    void main() {
                        float value = imageLoad(outImage, ivec2(0, 0));
                    }
                }
            }
            """,
            "requires read-capable storage image access",
        ),
    ],
)
def test_glsl_storage_image_access_rejects_invalid_operations(shader, match):
    ast = crosstl.translator.parse(shader)

    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate(ast)


def test_glsl_storage_image_access_allows_compatible_helper_calls():
    shader = """
    shader StorageImageAccessHelperValid {
        image2D source @readonly;
        image2D target @writeonly;

        float readPixel(image2D image, ivec2 pixel) {
            return imageLoad(image, pixel);
        }

        void writePixel(image2D image, ivec2 pixel, vec4 value) {
            imageStore(image, pixel, value);
        }

        compute {
            void main() {
                float value = readPixel(source, ivec2(0, 0));
                writePixel(target, ivec2(0, 0), vec4(value));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "float value = readPixel__glsl_image_source(ivec2(0, 0));" in generated_code
    assert "writePixel__glsl_image_target(ivec2(0, 0), vec4(value));" in generated_code


def test_glsl_storage_image_helper_parameter_metadata_contracts_allow_compatible_calls():
    shader = """
    shader StorageImageHelperMetadataContractsValid {
        image2D counters @r32ui[];
        image2D target;

        int queryCounter(image2D images[] @r32ui, ivec2 pixel) {
            return imageSize(images[0]).x;
        }

        void sizeOnlyWrite(image2D image @writeonly) {
            ivec2 dims = imageSize(image);
        }

        compute {
            void main() {
                int count = queryCounter(counters, ivec2(0, 1));
                sizeOnlyWrite(target);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(r32ui, binding = 0) uniform uimage2D counters[1];" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image2D target;" in generated_code
    assert "int queryCounter__glsl_images_counters(ivec2 pixel)" in generated_code
    assert "void sizeOnlyWrite__glsl_image_target()" in generated_code
    assert (
        "int count = queryCounter__glsl_images_counters(ivec2(0, 1));" in generated_code
    )
    assert "sizeOnlyWrite__glsl_image_target();" in generated_code


def test_glsl_storage_image_helper_contracts_specialize_array_elements_transitively():
    shader = """
    shader StorageImageHelperArrayElementSpecializationValid {
        image2D counters @r32ui[2];

        int queryElement(image2D image @r32ui) {
            return imageSize(image).x;
        }

        int queryViaArray(image2D images[] @r32ui) {
            return queryElement(images[0]);
        }

        compute {
            void main() {
                int directCount = queryElement(counters[0]);
                int nestedCount = queryViaArray(counters);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(r32ui, binding = 0) uniform uimage2D counters[2];" in generated_code
    assert "int queryElement__glsl_image_counters_0()" in generated_code
    assert "return imageSize(counters[0]).x;" in generated_code
    assert "int queryElement__glsl_image_counters()" not in generated_code
    assert "return imageSize(counters).x;" not in generated_code
    assert "int queryViaArray__glsl_images_counters()" in generated_code
    assert "return queryElement__glsl_image_counters_0();" in generated_code
    assert "int directCount = queryElement__glsl_image_counters_0();" in generated_code
    assert "int nestedCount = queryViaArray__glsl_images_counters();" in generated_code


def test_glsl_storage_image_helper_dynamic_array_elements_keep_index_parameters():
    shader = """
    shader StorageImageHelperDynamicArrayElementValid {
        image2D counters @r32ui[2];

        int queryElement(image2D image @r32ui) {
            return imageSize(image).x;
        }

        int queryViaDynamic(image2D images[] @r32ui, int layer) {
            return queryElement(images[layer]);
        }

        int queryViaInitializer(image2D images[] @r32ui, int layer) {
            int count = queryElement(images[layer]);
            return count;
        }

        int queryViaAssignment(image2D images[] @r32ui, int layer) {
            int count = 0;
            count = queryElement(images[layer]);
            return count;
        }

        void storeElement(image2D image @r32ui, ivec2 pixel, uint value) {
            imageStore(image, pixel, value);
        }

        void storeViaExpression(image2D images[] @r32ui, int layer) {
            storeElement(images[layer], ivec2(0, 0), uint(layer));
        }

        compute {
            void main() {
                int directCount = queryElement(counters[0]);
                int nestedCount = queryViaDynamic(counters, 1);
                int initializedCount = queryViaInitializer(counters, 0);
                int assignedCount = queryViaAssignment(counters, 1);
                storeViaExpression(counters, 1);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(r32ui, binding = 0) uniform uimage2D counters[2];" in generated_code
    assert "int queryElement__glsl_image_counters_0()" in generated_code
    assert "return imageSize(counters[0]).x;" in generated_code
    assert "int queryElement__glsl_image_counters_1()" in generated_code
    assert "return imageSize(counters[1]).x;" in generated_code
    assert "int queryViaDynamic__glsl_images_counters(int layer)" in generated_code
    assert "switch (layer)" in generated_code
    assert "return queryElement__glsl_image_counters_0();" in generated_code
    assert "return queryElement__glsl_image_counters_1();" in generated_code
    assert "default:\n        return 0;" in generated_code
    assert "return queryElement(counters[layer]);" not in generated_code
    assert "int queryViaInitializer__glsl_images_counters(int layer)" in generated_code
    assert "int count;\n    switch (layer)" in generated_code
    assert "count = queryElement__glsl_image_counters_0();" in generated_code
    assert "count = queryElement__glsl_image_counters_1();" in generated_code
    assert "int queryViaAssignment__glsl_images_counters(int layer)" in generated_code
    assert "void storeElement__glsl_image_counters_0(" in generated_code
    assert "void storeElement__glsl_image_counters_1(" in generated_code
    assert "imageStore(counters[0], pixel, uvec4(value));" in generated_code
    assert "imageStore(counters[1], pixel, uvec4(value));" in generated_code
    assert "void storeViaExpression__glsl_images_counters(int layer)" in generated_code
    assert (
        "storeElement__glsl_image_counters_0(ivec2(0, 0), uint(layer));"
        in generated_code
    )
    assert (
        "storeElement__glsl_image_counters_1(ivec2(0, 0), uint(layer));"
        in generated_code
    )
    assert "queryElement__glsl_image_counters_layer" not in generated_code
    assert "storeElement__glsl_image_counters_layer" not in generated_code
    assert "return imageSize(counters[layer]).x;" not in generated_code
    assert "imageStore(counters[layer], pixel, uvec4(value));" not in generated_code
    assert "int directCount = queryElement__glsl_image_counters_0();" in generated_code
    assert (
        "int nestedCount = queryViaDynamic__glsl_images_counters(1);" in generated_code
    )
    assert (
        "int initializedCount = queryViaInitializer__glsl_images_counters(0);"
        in generated_code
    )
    assert (
        "int assignedCount = queryViaAssignment__glsl_images_counters(1);"
        in generated_code
    )
    assert "storeViaExpression__glsl_images_counters(1);" in generated_code


def test_glsl_advanced_storage_image_helper_dynamic_array_elements_dispatch():
    shader = """
    shader GLSLAdvancedDynamicImageArrayElementValid {
        image2DMS msImages @rgba16f[2];
        uimage2DMS msCounters @r32ui[2];
        imageCube cubeImages @rgba16f[2];
        imageCubeArray cubeLayerImages @rgba16f[2];

        vec4 touchMS(image2DMS image @rgba16f, ivec2 pixel, int sampleIndex, vec4 value) {
            vec4 oldValue = imageLoad(image, pixel, sampleIndex);
            imageStore(image, pixel, sampleIndex, oldValue + value);
            return oldValue;
        }

        uint bumpMS(uimage2DMS image @r32ui, ivec2 pixel, int sampleIndex, uint value) {
            return imageAtomicAdd(image, pixel, sampleIndex, value);
        }

        vec4 touchCube(imageCube image @rgba16f, ivec3 coord, vec4 value) {
            vec4 oldValue = imageLoad(image, coord);
            imageStore(image, coord, oldValue + value);
            return oldValue;
        }

        vec4 touchCubeLayer(imageCubeArray image @rgba16f, ivec3 coord, vec4 value) {
            vec4 oldValue = imageLoad(image, coord);
            imageStore(image, coord, oldValue + value);
            return oldValue;
        }

        vec4 viaMS(image2DMS images[] @rgba16f, int layer, ivec2 pixel, int sampleIndex, vec4 value) {
            return touchMS(images[layer], pixel, sampleIndex, value);
        }

        uint viaCounter(uimage2DMS counters[] @r32ui, int layer, ivec2 pixel, int sampleIndex, uint value) {
            return bumpMS(counters[layer], pixel, sampleIndex, value);
        }

        vec4 viaCube(imageCube images[] @rgba16f, int layer, ivec3 coord, vec4 value) {
            return touchCube(images[layer], coord, value);
        }

        vec4 viaCubeLayer(imageCubeArray images[] @rgba16f, int layer, ivec3 coord, vec4 value) {
            return touchCubeLayer(images[layer], coord, value);
        }

        compute {
            void main() {
                vec4 msValue = viaMS(msImages, 1, ivec2(2, 3), 0, vec4(1.0));
                uint count = viaCounter(msCounters, 1, ivec2(3, 4), 1, 5u);
                vec4 cubeValue = viaCube(cubeImages, 1, ivec3(2, 3, 4), msValue);
                vec4 layerValue = viaCubeLayer(cubeLayerImages, 1, ivec3(3, 4, 5), cubeValue + vec4(float(count)));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(rgba16f, binding = 0) uniform image2DMS msImages[2];" in generated_code
    )
    assert (
        "layout(r32ui, binding = 2) uniform uimage2DMS msCounters[2];" in generated_code
    )
    assert (
        "layout(rgba16f, binding = 4) uniform imageCube cubeImages[2];"
        in generated_code
    )
    assert (
        "layout(rgba16f, binding = 6) uniform imageCubeArray cubeLayerImages[2];"
        in generated_code
    )
    assert "switch (layer)" in generated_code
    assert "vec4 touchMS__glsl_image_msImages_0(" in generated_code
    assert "vec4 touchMS__glsl_image_msImages_1(" in generated_code
    assert "return touchMS__glsl_image_msImages_0(pixel, sampleIndex, value);" in (
        generated_code
    )
    assert "return touchMS__glsl_image_msImages_1(pixel, sampleIndex, value);" in (
        generated_code
    )
    assert "uint bumpMS__glsl_image_msCounters_0(" in generated_code
    assert "uint bumpMS__glsl_image_msCounters_1(" in generated_code
    assert "return bumpMS__glsl_image_msCounters_0(pixel, sampleIndex, value);" in (
        generated_code
    )
    assert "return bumpMS__glsl_image_msCounters_1(pixel, sampleIndex, value);" in (
        generated_code
    )
    assert "return touchCube__glsl_image_cubeImages_0(coord, value);" in generated_code
    assert "return touchCube__glsl_image_cubeImages_1(coord, value);" in generated_code
    assert (
        "return touchCubeLayer__glsl_image_cubeLayerImages_0(coord, value);"
        in generated_code
    )
    assert (
        "return touchCubeLayer__glsl_image_cubeLayerImages_1(coord, value);"
        in generated_code
    )
    assert "touchMS__glsl_image_msImages_layer" not in generated_code
    assert "bumpMS__glsl_image_msCounters_layer" not in generated_code
    assert "touchCube__glsl_image_cubeImages_layer" not in generated_code
    assert "touchCubeLayer__glsl_image_cubeLayerImages_layer" not in generated_code
    assert "imageLoad(msImages[layer], pixel, sampleIndex)" not in generated_code
    assert "imageLoad(cubeImages[layer], coord)" not in generated_code


def test_glsl_multisample_storage_image_helper_array_elements_specialize_operations():
    shader = """
    shader GLSLMultisampleImageArrayElementSpecializationValid {
        image2DMS images @rgba16f[2];
        uimage2DMS counters @r32ui[2];

        vec4 touch(image2DMS image @rgba16f, ivec2 pixel, int sampleIndex, vec4 value) {
            vec4 oldValue = imageLoad(image, pixel, sampleIndex);
            imageStore(image, pixel, sampleIndex, oldValue + value);
            return oldValue;
        }

        uint bump(uimage2DMS image @r32ui, ivec2 pixel, int sampleIndex, uint value) {
            return imageAtomicAdd(image, pixel, sampleIndex, value);
        }

        vec4 viaArray(image2DMS targets[] @rgba16f, ivec2 pixel, int sampleIndex, vec4 value) {
            return touch(targets[1], pixel, sampleIndex, value);
        }

        uint viaCounter(uimage2DMS targets[] @r32ui, ivec2 pixel, int sampleIndex, uint value) {
            return bump(targets[1], pixel, sampleIndex, value);
        }

        compute {
            void main() {
                vec4 directValue = touch(images[0], ivec2(0, 1), 2, vec4(1.0));
                vec4 nestedValue = viaArray(images, ivec2(2, 3), 0, directValue);
                uint directCount = bump(counters[0], ivec2(1, 2), 3, 4u);
                uint nestedCount = viaCounter(counters, ivec2(3, 4), 1, directCount);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(rgba16f, binding = 0) uniform image2DMS images[2];" in generated_code
    assert (
        "layout(r32ui, binding = 2) uniform uimage2DMS counters[2];" in generated_code
    )
    assert "vec4 touch__glsl_image_images_0(" in generated_code
    assert "vec4 touch__glsl_image_images_1(" in generated_code
    assert "imageLoad(images[0], pixel, sampleIndex)" in generated_code
    assert (
        "imageStore(images[1], pixel, sampleIndex, (oldValue + value));"
        in generated_code
    )
    assert "uint bump__glsl_image_counters_0(" in generated_code
    assert "uint bump__glsl_image_counters_1(" in generated_code
    assert (
        "return imageAtomicAdd(counters[0], pixel, sampleIndex, value);"
        in generated_code
    )
    assert (
        "return imageAtomicAdd(counters[1], pixel, sampleIndex, value);"
        in generated_code
    )
    assert (
        "return touch__glsl_image_images_1(pixel, sampleIndex, value);"
        in generated_code
    )
    assert (
        "return bump__glsl_image_counters_1(pixel, sampleIndex, value);"
        in generated_code
    )
    assert "vec4 directValue = touch__glsl_image_images_0(" in generated_code
    assert "vec4 nestedValue = viaArray__glsl_targets_images(" in generated_code
    assert "uint directCount = bump__glsl_image_counters_0(" in generated_code
    assert "uint nestedCount = viaCounter__glsl_targets_counters(" in generated_code


def test_glsl_cube_storage_image_helper_array_elements_specialize_load_store():
    shader = """
    shader GLSLCubeImageArrayElementSpecializationValid {
        imageCube cubeImages @rgba16f[2];
        imageCubeArray cubeLayerImages @rgba16f[2];

        vec4 touchCube(imageCube image @rgba16f, ivec3 coord, vec4 value) {
            vec4 oldValue = imageLoad(image, coord);
            imageStore(image, coord, oldValue + value);
            return oldValue;
        }

        vec4 touchCubeLayer(imageCubeArray image @rgba16f, ivec3 coord, vec4 value) {
            vec4 oldValue = imageLoad(image, coord);
            imageStore(image, coord, oldValue + value);
            return oldValue;
        }

        vec4 viaCube(imageCube images[] @rgba16f, ivec3 coord, vec4 value) {
            return touchCube(images[1], coord, value);
        }

        vec4 viaCubeLayer(imageCubeArray images[] @rgba16f, ivec3 coord, vec4 value) {
            return touchCubeLayer(images[1], coord, value);
        }

        compute {
            void main() {
                vec4 directCube = touchCube(cubeImages[0], ivec3(0, 1, 2), vec4(1.0));
                vec4 nestedCube = viaCube(cubeImages, ivec3(2, 3, 4), directCube);
                vec4 directLayer = touchCubeLayer(cubeLayerImages[0], ivec3(1, 2, 3), nestedCube);
                vec4 nestedLayer = viaCubeLayer(cubeLayerImages, ivec3(3, 4, 5), directLayer);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(rgba16f, binding = 0) uniform imageCube cubeImages[2];"
        in generated_code
    )
    assert (
        "layout(rgba16f, binding = 2) uniform imageCubeArray cubeLayerImages[2];"
        in generated_code
    )
    assert "vec4 touchCube__glsl_image_cubeImages_0(" in generated_code
    assert "vec4 touchCube__glsl_image_cubeImages_1(" in generated_code
    assert "imageLoad(cubeImages[0], coord)" in generated_code
    assert "imageStore(cubeImages[1], coord, (oldValue + value));" in generated_code
    assert "vec4 touchCubeLayer__glsl_image_cubeLayerImages_0(" in generated_code
    assert "vec4 touchCubeLayer__glsl_image_cubeLayerImages_1(" in generated_code
    assert "imageLoad(cubeLayerImages[0], coord)" in generated_code
    assert (
        "imageStore(cubeLayerImages[1], coord, (oldValue + value));" in generated_code
    )
    assert "return touchCube__glsl_image_cubeImages_1(coord, value);" in generated_code
    assert (
        "return touchCubeLayer__glsl_image_cubeLayerImages_1(coord, value);"
        in generated_code
    )
    assert "vec4 directCube = touchCube__glsl_image_cubeImages_0(" in generated_code
    assert "vec4 nestedCube = viaCube__glsl_images_cubeImages(" in generated_code
    assert (
        "vec4 directLayer = touchCubeLayer__glsl_image_cubeLayerImages_0("
        in generated_code
    )
    assert (
        "vec4 nestedLayer = viaCubeLayer__glsl_images_cubeLayerImages("
        in generated_code
    )


@pytest.mark.parametrize(
    ("shader", "match"),
    [
        (
            """
            shader StorageImageHelperFormatContractInvalid {
                image2D rgImages @rg32f[];

                int queryCounter(image2D images[] @r32ui, ivec2 pixel) {
                    return imageSize(images[0]).x;
                }

                compute {
                    void main() {
                        int count = queryCounter(rgImages, ivec2(0, 1));
                    }
                }
            }
            """,
            "function call 'queryCounter' requires r32ui storage image format "
            "for argument rgImages passed to parameter images: got rg32f",
        ),
        (
            """
            shader StorageImageHelperArrayElementFormatContractInvalid {
                image2D rgImages @rg32f[2];

                int queryElement(image2D image @r32ui) {
                    return imageSize(image).x;
                }

                compute {
                    void main() {
                        int count = queryElement(rgImages[0]);
                    }
                }
            }
            """,
            "function call 'queryElement' requires r32ui storage image format "
            "for argument rgImages\\[0\\] passed to parameter image: got rg32f",
        ),
        (
            """
            shader StorageImageHelperMultisampleArrayElementFormatContractInvalid {
                image2DMS rgImages @rg32f[2];

                vec4 touch(image2DMS image @rgba16f, ivec2 pixel, int sampleIndex) {
                    return imageLoad(image, pixel, sampleIndex);
                }

                compute {
                    void main() {
                        vec4 value = touch(rgImages[0], ivec2(0, 1), 2);
                    }
                }
            }
            """,
            "function call 'touch' requires rgba16f storage image format "
            "for argument rgImages\\[0\\] passed to parameter image: got rg32f",
        ),
        (
            """
            shader StorageImageHelperTransitiveElementFormatContractInvalid {
                image2D rgImages @rg32f[2];

                int leaf(image2D image @r32ui) {
                    return imageSize(image).x;
                }

                int mid(image2D images[]) {
                    return leaf(images[0]);
                }

                compute {
                    void main() {
                        int count = mid(rgImages);
                    }
                }
            }
            """,
            "function call 'leaf' requires r32ui storage image format "
            "for argument images\\[0\\] passed to parameter image: got rg32f",
        ),
        (
            """
            shader StorageImageHelperDynamicElementFormatContractInvalid {
                image2D rgImages @rg32f[2];

                int leaf(image2D image @r32ui) {
                    return imageSize(image).x;
                }

                int mid(image2D images[], int layer) {
                    return leaf(images[layer]);
                }

                compute {
                    void main() {
                        int count = mid(rgImages, 1);
                    }
                }
            }
            """,
            "function call 'leaf' requires r32ui storage image format "
            "for argument images\\[layer\\] passed to parameter image: got rg32f",
        ),
        (
            """
            shader StorageImageHelperTypeContractInvalid {
                image2D colorImages[];

                int queryCounter(uimage2D images[], ivec2 pixel) {
                    return imageSize(images[0]).x;
                }

                compute {
                    void main() {
                        int count = queryCounter(colorImages, ivec2(0, 1));
                    }
                }
            }
            """,
            "function call 'queryCounter' requires uimage2D storage image "
            "for argument colorImages passed to parameter images: got image2D",
        ),
        (
            """
            shader StorageImageHelperAccessContractInvalid {
                image2D source @readonly;

                void sizeOnlyWrite(image2D image @writeonly) {
                    ivec2 dims = imageSize(image);
                }

                compute {
                    void main() {
                        sizeOnlyWrite(source);
                    }
                }
            }
            """,
            "function call 'sizeOnlyWrite' requires write-capable storage image "
            "access for argument source passed to parameter image: got readonly",
        ),
        (
            """
            shader StorageImageHelperTransitiveAccessContractInvalid {
                image2D source @readonly;

                void leaf(image2D image @writeonly) {
                    ivec2 dims = imageSize(image);
                }

                void mid(image2D image) {
                    leaf(image);
                }

                compute {
                    void main() {
                        mid(source);
                    }
                }
            }
            """,
            "function call 'leaf' requires write-capable storage image access "
            "for argument image passed to parameter image: got readonly",
        ),
    ],
)
def test_glsl_storage_image_helper_parameter_metadata_contracts_reject_mismatches(
    shader, match
):
    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("shader", "match"),
    [
        (
            """
            shader StorageImageAccessHelperInvalidRead {
                image2D target @writeonly;

                float readPixel(image2D image, ivec2 pixel) {
                    return imageLoad(image, pixel);
                }

                compute {
                    void main() {
                        float value = readPixel(target, ivec2(0, 0));
                    }
                }
            }
            """,
            "function call 'readPixel' requires read-capable storage image access",
        ),
        (
            """
            shader StorageImageAccessHelperInvalidWrite {
                image2D source @readonly;

                void writePixel(image2D image, ivec2 pixel, vec4 value) {
                    imageStore(image, pixel, value);
                }

                compute {
                    void main() {
                        writePixel(source, ivec2(0, 0), vec4(1.0));
                    }
                }
            }
            """,
            "function call 'writePixel' requires write-capable storage image access",
        ),
        (
            """
            shader StorageImageAccessHelperInvalidAtomic {
                uimage2D source @r32ui @readonly;

                uint addCounter(uimage2D image @r32ui, ivec2 pixel, uint value) {
                    return imageAtomicAdd(image, pixel, value);
                }

                compute {
                    void main() {
                        uint value = addCounter(source, ivec2(0, 0), 1u);
                    }
                }
            }
            """,
            "function call 'addCounter' requires read-write storage image access",
        ),
        (
            """
            shader StorageImageAccessHelperInvalidTransitive {
                image2D target @writeonly;

                float leaf(image2D image, ivec2 pixel) {
                    return imageLoad(image, pixel);
                }

                float mid(image2D image, ivec2 pixel) {
                    return leaf(image, pixel);
                }

                compute {
                    void main() {
                        float value = mid(target, ivec2(0, 0));
                    }
                }
            }
            """,
            "function call 'mid' requires read-capable storage image access",
        ),
    ],
)
def test_glsl_storage_image_access_rejects_incompatible_helper_calls(shader, match):
    ast = crosstl.translator.parse(shader)

    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate(ast)


def test_glsl_wave_subgroup_ballot_and_shuffle_in_fragment_shader():
    code = """
    shader SubgroupFragment {
        struct FSInput {
            float value @ TEXCOORD0;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                uint lane = WaveGetLaneIndex();
                uvec4 ballot = WaveActiveBallot(input.value > 0.5);
                float shuffled = WaveReadLaneAt(input.value, 0u);
                float first = WaveReadLaneFirst(input.value);
                return vec4(shuffled, first, float(lane), 1.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    assert "#extension GL_KHR_shader_subgroup_basic : require" in generated
    assert "#extension GL_KHR_shader_subgroup_ballot : require" in generated
    assert "#extension GL_KHR_shader_subgroup_shuffle : require" in generated
    assert "uint lane = gl_SubgroupInvocationID;" in generated
    assert "uvec4 ballot = subgroupBallot((value > 0.5));" in generated
    assert "float shuffled = subgroupShuffle(value, 0u);" in generated
    assert "float first = subgroupBroadcastFirst(value);" in generated
    assert "WaveGetLaneIndex" not in generated
    assert "WaveActiveBallot" not in generated
    assert "WaveReadLaneAt" not in generated
    assert "WaveReadLaneFirst" not in generated


def test_glsl_texture_gather_basic_emits_texture_gather():
    code = """
    shader TextureGatherBasic {
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 gathered = textureGather(colorMap, linearSampler, input.uv);
                vec4 greenChannel = textureGather(colorMap, linearSampler, input.uv, 1);
                return gathered + greenChannel;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    assert "textureGather(colorMap, uv)" in generated
    assert "textureGather(colorMap, uv, 1)" in generated
    assert "linearSampler" not in generated


def test_glsl_texture_gather_offset_emits_texture_gather_offset():
    code = """
    shader TextureGatherOffsetBasic {
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 gathered = textureGatherOffset(colorMap, linearSampler, input.uv, ivec2(1, 0));
                return gathered;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    assert "textureGatherOffset(colorMap, uv, ivec2(1, 0))" in generated
    assert "linearSampler" not in generated


def test_glsl_texture_proj_basic_emits_texture_proj():
    code = """
    shader TextureProjBasic {
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {
            vec3 projCoord @ TEXCOORD0;
            vec4 projCoord4 @ TEXCOORD1;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 proj2d = textureProj(colorMap, linearSampler, input.projCoord);
                vec4 proj2d4 = textureProj(colorMap, linearSampler, input.projCoord4);
                return proj2d + proj2d4;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    assert "textureProj(colorMap, projCoord)" in generated
    assert "textureProj(colorMap, projCoord4)" in generated
    assert "linearSampler" not in generated


def test_glsl_texture_proj_lod_emits_texture_proj_lod():
    code = """
    shader TextureProjLod {
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {
            vec3 projCoord @ TEXCOORD0;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 projLod = textureProjLod(colorMap, linearSampler, input.projCoord, 2.0);
                return projLod;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    assert "textureProjLod(colorMap, projCoord, 2.0)" in generated
    assert "linearSampler" not in generated


def test_glsl_wave_ops_with_active_variable_name_reservation():
    code = """
    shader WaveWithActiveVar {
        compute {
            void main() {
                uint active = WaveGetLaneIndex();
                uint sum = WaveActiveSum(active);
                bool isActive = WaveActiveAnyTrue(sum > 0u);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    assert "#extension GL_KHR_shader_subgroup_basic : require" in generated
    assert "#extension GL_KHR_shader_subgroup_arithmetic : require" in generated
    assert "#extension GL_KHR_shader_subgroup_vote : require" in generated
    assert "uint active_ = gl_SubgroupInvocationID;" in generated
    assert "uint sum = subgroupAdd(active_);" in generated
    assert "bool isActive = subgroupAny((sum > 0u));" in generated
    assert "WaveGetLaneIndex" not in generated
    assert "WaveActiveSum" not in generated
    assert "WaveActiveAnyTrue" not in generated


def test_glsl_struct_member_name_reservation_updates_member_accesses():
    code = """
    shader ReservedStructMember {
        struct Params {
            int max_particles;
        };

        struct Particle {
            bool active;
            int type;
        };

        struct ParticleBuffer {
            Particle particles[4];
        };

        compute {
            uniform Params params;
            buffer ParticleBuffer particle_buffer;

            void main() {
                Particle particle = particle_buffer.particles[0];
                if (params.max_particles > 0 && !particle.active) {
                    particle.active = true;
                }
                if (particle_buffer.particles[1].active) {
                    particle.type = 1;
                }
                particle_buffer.particles[0] = particle;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    assert "uniform Params params;" in generated
    assert "layout(std430, binding = 0) buffer ParticleBuffer" in generated
    assert "} particle_buffer;" in generated
    assert "bool active_;" in generated
    assert "bool active;" not in generated
    assert "if (((params.max_particles > 0) && (!particle.active_)))" in generated
    assert "particle.active_ = true;" in generated
    assert "if (particle_buffer.particles[1].active_)" in generated
    assert "particle.active" not in generated.replace("particle.active_", "")
    assert ".active" not in generated.replace(".active_", "")


def test_glsl_subgroup_ops_combined_with_texture_gather():
    code = """
    shader SubgroupTextureGather {
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 gathered = textureGather(colorMap, linearSampler, input.uv);
                float avg = WaveActiveSum(gathered.r);
                uint laneCount = WaveGetLaneCount();
                float normalized = avg / float(laneCount);
                return vec4(normalized, normalized, normalized, 1.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    assert "textureGather(colorMap, uv)" in generated
    assert "#extension GL_KHR_shader_subgroup_basic : require" in generated
    assert "#extension GL_KHR_shader_subgroup_arithmetic : require" in generated
    assert "float avg = subgroupAdd(gathered.r);" in generated
    assert "uint laneCount = gl_SubgroupSize;" in generated
    assert "linearSampler" not in generated
    assert "WaveActiveSum" not in generated
    assert "WaveGetLaneCount" not in generated


if __name__ == "__main__":
    pytest.main()
