import re
from typing import List

import pytest

import crosstl.translator
from crosstl.translator.ast import PrimitiveType, StructMemberNode, StructNode
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
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
    codegen = MetalCodeGen()
    return codegen.generate(ast_node)


def test_metal_synchronization_builtins_lower_to_threadgroup_barriers():
    shader = """
    shader SynchronizationBuiltins {
        compute {
            void main() {
                barrier();
                memoryBarrier();
                workgroupBarrier();
                groupMemoryBarrier();
                memoryBarrierShared();
                memoryBarrierBuffer();
                deviceMemoryBarrier();
                memoryBarrierImage();
                allMemoryBarrier();
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert generated_code.count("threadgroup_barrier(mem_flags::mem_threadgroup);") == 4
    assert generated_code.count("threadgroup_barrier(mem_flags::mem_device);") == 3
    assert "threadgroup_barrier(mem_flags::mem_texture);" in generated_code
    assert (
        "threadgroup_barrier(mem_flags::mem_device | "
        "mem_flags::mem_threadgroup | mem_flags::mem_texture);"
    ) in generated_code
    assert "barrier();" not in generated_code
    assert "memoryBarrier();" not in generated_code
    assert "workgroupBarrier();" not in generated_code
    assert "groupMemoryBarrier();" not in generated_code
    assert "memoryBarrierShared();" not in generated_code
    assert "memoryBarrierBuffer();" not in generated_code
    assert "deviceMemoryBarrier();" not in generated_code
    assert "memoryBarrierImage();" not in generated_code
    assert "allMemoryBarrier();" not in generated_code


def test_metal_user_defined_synchronization_names_are_not_lowered():
    shader = """
    shader SynchronizationShadowing {
        compute {
            void barrier() {
                return;
            }

            void memoryBarrier() {
                return;
            }

            void workgroupBarrier() {
                return;
            }

            void groupMemoryBarrier() {
                return;
            }

            void memoryBarrierShared() {
                return;
            }

            void memoryBarrierBuffer() {
                return;
            }

            void deviceMemoryBarrier() {
                return;
            }

            void memoryBarrierImage() {
                return;
            }

            void allMemoryBarrier() {
                return;
            }

            void main() {
                barrier();
                memoryBarrier();
                workgroupBarrier();
                groupMemoryBarrier();
                memoryBarrierShared();
                memoryBarrierBuffer();
                deviceMemoryBarrier();
                memoryBarrierImage();
                allMemoryBarrier();
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "void barrier()" in generated_code
    assert "void memoryBarrier()" in generated_code
    assert "void workgroupBarrier()" in generated_code
    assert "void groupMemoryBarrier()" in generated_code
    assert "void memoryBarrierShared()" in generated_code
    assert "void memoryBarrierBuffer()" in generated_code
    assert "void deviceMemoryBarrier()" in generated_code
    assert "void memoryBarrierImage()" in generated_code
    assert "void allMemoryBarrier()" in generated_code
    assert "barrier();" in generated_code
    assert "memoryBarrier();" in generated_code
    assert "workgroupBarrier();" in generated_code
    assert "groupMemoryBarrier();" in generated_code
    assert "memoryBarrierShared();" in generated_code
    assert "memoryBarrierBuffer();" in generated_code
    assert "deviceMemoryBarrier();" in generated_code
    assert "memoryBarrierImage();" in generated_code
    assert "allMemoryBarrier();" in generated_code
    assert "threadgroup_barrier(" not in generated_code


def test_metal_shared_local_variables_use_threadgroup_address_space():
    shader = """
    shader SharedLocalStorage {
        compute {
            void main() {
                shared int data[4];
                int scratch[4] @threadgroup;
                data[0] = 1;
                scratch[0] = data[0];
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "threadgroup int data[4];" in generated_code
    assert "threadgroup int scratch[4];" in generated_code
    assert "\n    int data[4];" not in generated_code
    assert "\n    int scratch[4];" not in generated_code


def test_metal_threadgroup_array_helpers_preserve_address_space_and_barriers():
    shader = """
    shader ThreadgroupHelperArray {
        void writeScratch(threadgroup float scratch[64], uint index, float value) {
            scratch[index] = value;
        }

        float readScratch(threadgroup float scratch[64], uint index) {
            return scratch[index];
        }

        compute {
            void main() {
                shared float scratch[64];
                uint index = gl_LocalInvocationIndex;
                writeScratch(scratch, index, float(index));
                groupMemoryBarrier();
                memoryBarrierShared();
                float value = readScratch(scratch, index);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        parse_code(tokenize_code(shader)), "compute"
    )

    assert "void writeScratch(threadgroup float scratch[64]" in generated_code
    assert "float readScratch(threadgroup float scratch[64]" in generated_code
    assert "threadgroup float scratch[64];" in generated_code
    assert (
        "uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]]" in generated_code
    )
    assert generated_code.count("threadgroup_barrier(mem_flags::mem_threadgroup);") == 2
    assert "groupMemoryBarrier();" not in generated_code
    assert "memoryBarrierShared();" not in generated_code
    assert "writeScratch(scratch, index, float(index));" in generated_code
    assert "float value = readScratch(scratch, index);" in generated_code


def test_metal_parameter_address_space_qualifiers_lower_for_pointer_reference_and_array_types():
    shader = """
    shader MetalAddressSpaceParameters {
        struct Payload {
            float value;
        };

        void update(
            threadgroup Payload* payload,
            device float values[],
            constant int& count,
            constant Payload& payloadRef,
            device Payload& mut payloadOut
        ) {
            float tmp = values[count] + payloadRef.value;
            payloadOut.value = tmp;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert (
        "void update(threadgroup Payload* payload, device float values[], "
        "constant int& count, constant Payload& payloadRef, "
        "device Payload& payloadOut)"
    ) in generated_code
    assert "float tmp = values[count] + payloadRef.value;" in generated_code
    assert "payloadOut.value = tmp;" in generated_code
    assert "PointerType" not in generated_code
    assert "ReferenceType" not in generated_code


def test_metal_raw_address_space_stage_buffers_lower_to_bindable_pointers():
    shader = """
    shader MetalRawAddressSpaceBuffers {
        void update(device float values[], constant uint& count) {
            if (count > 0u) {
                values[0] = values[0] + 1.0;
            }
        }

        compute {
            void main(device float values[] @buffer(0), constant uint& count @buffer(1)) {
                update(values, count);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        parse_code(tokenize_code(shader)), "compute"
    )

    assert "void update(device float values[], constant uint& count)" in generated_code
    assert (
        "kernel void kernel_main(device float* values [[buffer(0)]], "
        "constant uint& count [[buffer(1)]])"
    ) in generated_code
    assert "device float values[] [[buffer(0)]]" not in generated_code


def test_metal_pointer_typed_member_access_uses_arrow_operator():
    shader = """
    shader MetalPointerMemberAccess {
        struct Payload {
            float value;
        };

        compute {
            void main(device Payload* payload @buffer(0), device float values[] @buffer(1)) {
                payload.value = values[0];
                float value = payload.value;
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        parse_code(tokenize_code(shader)), "compute"
    )

    assert "device Payload* payload [[buffer(0)]]" in generated_code
    assert "payload->value = values[0];" in generated_code
    assert "float value = payload->value;" in generated_code
    assert "payload.value" not in generated_code


def test_metal_indexed_pointer_member_access_uses_element_dot_operator():
    shader = """
    shader MetalIndexedPointerMemberAccess {
        struct Payload {
            float value;
        };

        compute {
            void main(device Payload* payloads @buffer(0), device float values[] @buffer(1)) {
                payloads[0].value = values[0];
                float value = payloads[0].value;
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        parse_code(tokenize_code(shader)), "compute"
    )

    assert "device Payload* payloads [[buffer(0)]]" in generated_code
    assert "payloads[0].value = values[0];" in generated_code
    assert "float value = payloads[0].value;" in generated_code
    assert "payloads[0]->value" not in generated_code


def test_metal_readonly_raw_buffer_parameters_use_const_device_address_space():
    shader = """
    shader MetalReadonlyRawBuffers {
        struct Payload {
            float value;
        };

        compute {
            void main(
                readonly device Payload* payload @buffer(0),
                readonly device float values[] @buffer(1),
                constant uint& count @buffer(2)
            ) {
                float value = payload.value + values[count];
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        parse_code(tokenize_code(shader)), "compute"
    )

    assert "const device Payload* payload [[buffer(0)]]" in generated_code
    assert "const device float* values [[buffer(1)]]" in generated_code
    assert "constant uint& count [[buffer(2)]]" in generated_code
    assert "float value = payload->value + values[count];" in generated_code


def test_metal_readonly_raw_buffer_stores_emit_diagnostics():
    shader = """
    shader MetalReadonlyRawBufferStores {
        struct Payload {
            float value;
        };

        compute {
            void main(
                readonly device Payload* payload @buffer(0),
                readonly device float values[] @buffer(1)
            ) {
                payload.value = 1.0;
                values[0] = 2.0;
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        parse_code(tokenize_code(shader)), "compute"
    )

    assert (
        "/* unsupported Metal raw buffer store: readonly buffer 'payload' cannot be written */"
        in generated_code
    )
    assert (
        "/* unsupported Metal raw buffer store: readonly buffer 'values' cannot be written */"
        in generated_code
    )
    assert "payload->value = 1.0;" not in generated_code
    assert "values[0] = 2.0;" not in generated_code


def test_metal_readonly_raw_buffer_qualifiers_propagate_to_helpers():
    shader = """
    shader MetalReadonlyRawBufferHelpers {
        struct Payload {
            float value;
        };

        float readPayload(
            readonly device Payload* payload,
            readonly device float values[],
            constant uint& index
        ) {
            return payload.value + values[index];
        }

        void rejectWrite(readonly device Payload* payload, readonly device float values[]) {
            payload.value = 1.0;
            values[0] = 2.0;
        }

        compute {
            void main(
                readonly device Payload* payload @buffer(0),
                readonly device float values[] @buffer(1),
                constant uint& index @buffer(2)
            ) {
                float value = readPayload(payload, values, index);
                rejectWrite(payload, values);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        parse_code(tokenize_code(shader)), "compute"
    )

    assert (
        "float readPayload(const device Payload* payload, "
        "const device float values[], constant uint& index)"
    ) in generated_code
    assert (
        "void rejectWrite(const device Payload* payload, "
        "const device float values[])"
    ) in generated_code
    assert "return payload->value + values[index];" in generated_code
    assert (
        "kernel void kernel_main(const device Payload* payload [[buffer(0)]], "
        "const device float* values [[buffer(1)]], "
        "constant uint& index [[buffer(2)]])"
    ) in generated_code
    assert (
        "/* unsupported Metal raw buffer store: readonly buffer 'payload' cannot be written */"
        in generated_code
    )
    assert (
        "/* unsupported Metal raw buffer store: readonly buffer 'values' cannot be written */"
        in generated_code
    )
    assert "payload->value = 1.0;" not in generated_code
    assert "values[0] = 2.0;" not in generated_code


def test_metal_readonly_raw_buffer_calls_to_mutable_helpers_emit_diagnostic():
    shader = """
    shader MetalReadonlyRawBufferMutableHelperCall {
        struct Payload {
            float value;
        };

        void mutate(device Payload* payload, device float values[]) {
            payload.value = values[0];
        }

        compute {
            void main(
                readonly device Payload* payload @buffer(0),
                readonly device float values[] @buffer(1)
            ) {
                mutate(payload, values);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        parse_code(tokenize_code(shader)), "compute"
    )

    assert (
        "void mutate(device Payload* payload, device float values[])" in generated_code
    )
    assert "kernel void kernel_main(const device Payload* payload [[buffer(0)]]," in (
        generated_code
    )
    assert (
        "/* unsupported Metal raw buffer call: readonly buffer 'payload' cannot be "
        "passed to mutable parameter 'payload' of 'mutate' */"
    ) in generated_code
    assert "mutate(payload, values);" not in generated_code


def test_metal_incompatible_helper_address_spaces_emit_diagnostics():
    shader = """
    shader MetalAddressSpaceMismatchCalls {
        struct Payload {
            float value;
        };

        void useThreadgroup(threadgroup Payload& scratch) {
            scratch.value = 1.0;
        }

        void useDevice(device Payload& payload) {
            payload.value = 2.0;
        }

        compute {
            void main(device Payload* payload @buffer(0)) {
                threadgroup Payload scratch;
                useThreadgroup(payload[0]);
                useDevice(scratch);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        parse_code(tokenize_code(shader)), "compute"
    )

    assert "void useThreadgroup(threadgroup Payload& scratch)" in generated_code
    assert "void useDevice(device Payload& payload)" in generated_code
    assert "threadgroup Payload scratch;" in generated_code
    assert (
        "/* unsupported Metal address-space call: argument 'payload' uses device "
        "address space but parameter 'scratch' of 'useThreadgroup' requires "
        "threadgroup */"
    ) in generated_code
    assert (
        "/* unsupported Metal address-space call: argument 'scratch' uses "
        "threadgroup address space but parameter 'payload' of 'useDevice' "
        "requires device */"
    ) in generated_code
    assert "useThreadgroup(payload[0]);" not in generated_code
    assert "useDevice(scratch);" not in generated_code


def test_metal_parameter_address_space_qualifiers_reject_conflicts():
    shader = """
    shader MetalConflictingAddressSpaceParameters {
        void update(device threadgroup float* values) { }
    }
    """

    with pytest.raises(ValueError, match="Conflicting address space"):
        generate_code(parse_code(tokenize_code(shader)))


def test_metal_precision_aliases_map_to_native_metal_types():
    shader = """
    shader PrecisionAliasSmoke {
        float16 tone(float16 input) {
            float16 bias = float16(0.5);
            return input + bias;
        }

        f16vec2 pair(f16vec2 input) {
            f16vec2 scale = f16vec2(1.0, 2.0);
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

        min16float3x2 passMatrix(min16float3x2 input) {
            min16float3x2 m = min16float3x2(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);
            return input;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "half tone(half input)" in generated_code
    assert "half bias = half(0.5);" in generated_code
    assert "half2 pair(half2 input)" in generated_code
    assert "half2 scale = half2(1.0, 2.0);" in generated_code
    assert "half3 tint(half3 input)" in generated_code
    assert "half3 offset = half3(1.0, 2.0, 3.0);" in generated_code
    assert "uint2 bump(uint2 input)" in generated_code
    assert "uint2 inc = uint2(1u, 2u);" in generated_code
    assert "int clampInt(int input)" in generated_code
    assert "int one = int(1);" in generated_code
    assert "half3x2 passMatrix(half3x2 input)" in generated_code
    assert "half3x2 m = half3x2(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);" in generated_code
    for invalid_token in (
        "float16",
        "f16vec2",
        "min16float",
        "min16uint",
        "min12int",
    ):
        assert invalid_token not in generated_code


def test_metal_float16_matrix_ir_aliases_map_to_half_matrices():
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

    assert "half2x3 passMatrix(half2x3 input)" in generated_code
    assert "half2x3 m = half2x3(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);" in generated_code
    assert "half3x3 passSquare(half3x3 input)" in generated_code
    assert "half3x3 m = half3x3(1.0);" in generated_code
    assert "return input * m;" in generated_code
    assert "f16mat3x2" not in generated_code
    assert "f16mat3" not in generated_code


def test_metal_narrow_integer_aliases_map_to_valid_metal_integer_types():
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

    assert "int signedScalar(int input)" in generated_code
    assert "int one = int(1);" in generated_code
    assert "uint unsignedScalar(uint input)" in generated_code
    assert "uint one = uint(1u);" in generated_code
    assert "int2 signedPair(int2 input)" in generated_code
    assert "int2 inc = int2(1, 2);" in generated_code
    assert "uint3 unsignedTriple(uint3 input)" in generated_code
    assert "uint3 inc = uint3(1u, 2u, 3u);" in generated_code
    assert "int2 signedShort(int2 input)" in generated_code
    assert "int2 inc = int2(1, 2);" in generated_code
    assert "uint3 unsignedShort(uint3 input)" in generated_code
    assert "uint3 inc = uint3(1u, 2u, 3u);" in generated_code
    assert "int4 signedChar(int4 input)" in generated_code
    assert "int4 inc = int4(1, 2, 3, 4);" in generated_code
    assert "uint2 unsignedChar(uint2 input)" in generated_code
    assert "uint2 inc = uint2(1u, 2u);" in generated_code
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


def test_metal_packed_vector_aliases_map_to_standard_metal_vectors():
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

    assert "float4 color(float4 input)" in generated_code
    assert "float4 bias = float4(1.0, 2.0, 3.0, 4.0);" in generated_code
    assert "half2 halfPair(half2 input)" in generated_code
    assert "half2 scale = half2(1.0, 2.0);" in generated_code
    assert "int3 signedTriple(int3 input)" in generated_code
    assert "int3 inc = int3(1, 2, 3);" in generated_code
    assert "uint4 unsignedQuad(uint4 input)" in generated_code
    assert "uint4 inc = uint4(1u, 2u, 3u, 4u);" in generated_code
    assert "packed_" not in generated_code


def test_metal_simd_aliases_map_to_standard_metal_types():
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

    assert "float4 color(float4 input)" in generated_code
    assert "float4 bias = float4(1.0, 2.0, 3.0, 4.0);" in generated_code
    assert "int3 signedTriple(int3 input)" in generated_code
    assert "int3 inc = int3(1, 2, 3);" in generated_code
    assert "uint2 unsignedPair(uint2 input)" in generated_code
    assert "uint2 inc = uint2(1u, 2u);" in generated_code
    assert "float4x4 passSquare(float4x4 input)" in generated_code
    assert "float4x4 m = float4x4(1.0);" in generated_code
    assert "float3x2 passMatrix(float3x2 input)" in generated_code
    assert "float3x2 m = float3x2(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);" in generated_code
    assert "simd_" not in generated_code


def test_metal_fixed_width_scalar_aliases_map_to_valid_metal_scalars():
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

    assert "int signedByte(int input)" in generated_code
    assert "uint unsignedByte(uint input)" in generated_code
    assert "int signedShortScalar(int input)" in generated_code
    assert "uint unsignedShortScalar(uint input)" in generated_code
    assert "int signedWord(int input)" in generated_code
    assert "uint unsignedWord(uint input)" in generated_code
    assert "int64_t signedLong(int64_t input)" in generated_code
    assert "uint64_t unsignedLong(uint64_t input)" in generated_code
    assert "int64_t signedLongT(int64_t input)" in generated_code
    assert "uint64_t unsignedLongT(uint64_t input)" in generated_code
    assert "uint64_t sizeValue(uint64_t input)" in generated_code
    assert "int64_t ptrDiff(int64_t input)" in generated_code
    assert "int64_t longValue(int64_t input)" in generated_code
    assert "uint64_t ulongValue(uint64_t input)" in generated_code
    assert "int one = int(1);" in generated_code
    assert "uint one = uint(1u);" in generated_code
    assert "int64_t one = int64_t(1);" in generated_code
    assert "uint64_t one = uint64_t(1u);" in generated_code
    for invalid_token in (
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
        "int64 ",
        "uint64 ",
        "size_t",
        "ptrdiff_t",
        "long ",
        "ulong ",
    ):
        assert invalid_token not in generated_code


def test_metal_fixed_width_scalar_array_aliases_map_in_aggregate_declarations():
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
    assert "int64_t signedValue;" in generated_code
    assert "uint64_t offsets[2];" in generated_code
    assert "int globalBytes[2];" in generated_code
    assert "uint64_t globalCounters[2];" in generated_code
    assert "int smalls[2];" in generated_code
    assert "uint count;" in generated_code
    assert "int64_t delta;" in generated_code
    assert "uint64_t bump(uint64_t counters[2], AliasPayload payload)" in generated_code
    assert "uint64_t localCounters[2];" in generated_code
    assert "uint64_t one = uint64_t(1u);" in generated_code
    for invalid_token in (
        "int8_t",
        "uint16_t",
        "int16_t",
        "uint32_t",
        "int64 ",
        "uint64 ",
        "size_t",
        "ptrdiff_t",
    ):
        assert invalid_token not in generated_code


def test_metal_fixed_width_nested_array_aliases_map_to_valid_metal_types():
    shader = """
    shader AliasNestedArraySmoke {
        struct AliasGridPayload {
            int8_t bytes[2][3];
            size_t offsets[2][3];
        };

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
    assert "uint64_t offsets[2][3];" in generated_code
    assert "uint bumpNested(uint values[2][3], int row, int col)" in generated_code
    assert "uint grid[2][3];" in generated_code
    assert "uint(1u)" in generated_code
    assert (
        "int64_t readPayload(AliasGridPayload payload, int row, int col)"
        in generated_code
    )
    assert "int64_t(payload.bytes[row][col])" in generated_code
    assert "int64_t(payload.offsets[row][col])" in generated_code
    assert "int[2] bytes[3]" not in generated_code
    assert "uint64_t[2] offsets[3]" not in generated_code
    for invalid_token in (
        "int8_t",
        "uint16_t",
        "uint64 ",
        "size_t",
        "ptrdiff_t",
    ):
        assert invalid_token not in generated_code


def test_metal_array_return_aliases_are_rejected_with_clear_diagnostic():
    shader = """
    shader AliasReturnArraySmoke {
        uint64_t[2] makeCounters() {
            uint64_t counters[2];
            return counters;
        }
    }
    """

    with pytest.raises(
        ValueError, match="Metal output does not support C-style array return"
    ):
        generate_code(parse_code(tokenize_code(shader)))


def test_metal_fixed_width_nested_cbuffer_array_aliases_map_to_valid_types():
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

    assert "struct AliasNestedConstants" in generated_code
    assert "int64_t signedGrid[2][3];" in generated_code
    assert "uint unsignedGrid[2][3];" in generated_code
    assert "uint64_t offsets[2][3];" in generated_code
    assert "constant AliasNestedConstants& aliasNestedConstants" in generated_code
    assert "uint64_t readConstant(int row, int col" in generated_code
    assert "uint64_t(aliasNestedConstants.offsets[row][col])" in generated_code
    assert "uint64_t(aliasNestedConstants.unsignedGrid[row][col])" in generated_code
    assert "int64_t[2] signedGrid[3]" not in generated_code
    assert "uint[2] unsignedGrid[3]" not in generated_code
    for invalid_token in (
        "uint16_t",
        "uint64 ",
        "size_t",
    ):
        assert invalid_token not in generated_code


def test_metal_structured_buffer_fixed_width_aliases_map_to_device_pointers():
    shader = """
    shader AliasStructuredBufferMetal {
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

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "device uint* counts" in generated_code
    assert "const device int64_t* signedValues" in generated_code
    assert "device uint64_t* offsets" in generated_code
    assert "uint count = counts[index];" in generated_code
    assert "counts[index] = count + uint(1u);" in generated_code
    assert "uint16_t" not in generated_code
    assert "size_t" not in generated_code


def test_metal_structured_buffer_alias_arrays_infer_helper_parameter_sizes():
    shader = """
    shader AliasStructuredBufferArrayMetal {
        RWStructuredBuffer<uint16_t> counts[2] @ binding(3);
        StructuredBuffer<size_t> offsets[2] @ binding(5);

        uint16_t readCount(RWStructuredBuffer<uint16_t> localCounts[], uint which, uint index) {
            return buffer_load(localCounts[which], index);
        }

        uint64_t readOffset(StructuredBuffer<size_t> localOffsets[2], uint which, uint index) {
            return buffer_load(localOffsets[which], index);
        }

        uint64_t combine(uint which, uint index) {
            uint16_t count = readCount(counts, which, index);
            uint64_t offset = readOffset(offsets, which, index);
            buffer_store(counts[which], index, count + uint16_t(1u));
            return offset + uint64_t(count);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert (
        "uint readCount(array<device uint*, 2> localCounts, uint which, uint index)"
        in generated_code
    )
    assert (
        "uint64_t readOffset(array<const device uint64_t*, 2> localOffsets, uint which, uint index)"
        in generated_code
    )
    assert (
        "uint64_t combine(uint which, uint index, array<device uint*, 2> counts, array<const device uint64_t*, 2> offsets)"
        in generated_code
    )
    assert "return localCounts[which][index];" in generated_code
    assert "counts[which][index] = count + uint(1u);" in generated_code
    assert "device uint* localCounts" not in generated_code
    assert "uint16_t" not in generated_code
    assert "size_t" not in generated_code


def test_metal_unsized_structured_buffer_arrays_infer_helper_size():
    shader = """
    shader UnsizedStructuredBufferArrayMetal {
        RWStructuredBuffer<uint16_t> counts[] @ binding(3);
        StructuredBuffer<size_t> offsets[] @ binding(8);
        RWStructuredBuffer<uint16_t> afterCounts;

        uint16_t readCount(RWStructuredBuffer<uint16_t> localCounts[], uint index) {
            return buffer_load(localCounts[2], index);
        }

        uint64_t readOffset(StructuredBuffer<size_t> localOffsets[], uint index) {
            return buffer_load(localOffsets[1], index);
        }

        uint64_t combine(uint index) {
            uint16_t count = readCount(counts, index);
            uint64_t offset = readOffset(offsets, index);
            return offset + uint64_t(count) + uint64_t(buffer_load(afterCounts, index));
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "array<device uint*, 3> counts" in generated_code
    assert "device uint* afterCounts" in generated_code
    assert "array<const device uint64_t*, 2> offsets" in generated_code
    assert (
        "uint readCount(array<device uint*, 3> localCounts, uint index)"
        in generated_code
    )
    assert (
        "uint64_t readOffset(array<const device uint64_t*, 2> localOffsets, uint index)"
        in generated_code
    )
    assert "return localCounts[2][index];" in generated_code
    assert "return localOffsets[1][index];" in generated_code
    assert "uint count = readCount(counts, index);" in generated_code
    assert "uint64_t offset = readOffset(offsets, index);" in generated_code
    assert "uint16_t" not in generated_code
    assert "size_t" not in generated_code


def test_metal_dynamic_only_unsized_structured_buffer_arrays_keep_fallback_size():
    shader = """
    shader DynamicOnlyUnsizedStructuredBufferArrayMetal {
        RWStructuredBuffer<uint> counts[] @ binding(3);
        RWStructuredBuffer<uint> afterCounts;

        uint readCount(RWStructuredBuffer<uint> localCounts[], uint which, uint index) {
            return buffer_load(localCounts[which], index);
        }

        uint combine(uint which, uint index) {
            return readCount(counts, which, index) + buffer_load(afterCounts, index);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "array<device uint*, 1> counts" in generated_code
    assert "device uint* afterCounts" in generated_code
    assert (
        "uint readCount(array<device uint*, 1> localCounts, uint which, uint index)"
        in generated_code
    )
    assert "return localCounts[which][index];" in generated_code
    assert (
        "return readCount(counts, which, index) + afterCounts[index];" in generated_code
    )
    assert "array<device uint*, 2> counts" not in generated_code


def test_metal_unsized_structured_buffer_array_rejects_multiple_fixed_helper_sizes():
    shader = """
    shader UnsizedStructuredBufferMultipleFixedHelpersMetal {
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
        generate_code(parse_code(tokenize_code(shader)))


def test_metal_structured_buffer_array_helpers_propagate_nested_fixed_sizes():
    shader = """
    shader NestedStructuredBufferArrayHelpersMetal {
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

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "array<device uint*, 3> counts" in generated_code
    assert "device uint* afterCounts" in generated_code
    assert (
        "uint readLeaf(array<device uint*, 3> leafCounts, uint index)" in generated_code
    )
    assert (
        "uint readMid(array<device uint*, 3> midCounts, uint index)" in generated_code
    )
    assert "return leafCounts[2][index];" in generated_code
    assert "return readLeaf(midCounts, index);" in generated_code
    assert "return readMid(counts, index) + afterCounts[index];" in generated_code
    assert "array<device uint*, 1> counts" not in generated_code
    assert "uint16_t" not in generated_code


def test_metal_structured_buffer_array_nested_helpers_reject_mixed_fixed_sizes():
    shader = """
    shader NestedStructuredBufferArrayConflictMetal {
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
        generate_code(parse_code(tokenize_code(shader)))


def test_metal_glsl_buffer_block_fixed_width_aliases_lower_to_pointer_offsets():
    shader = """
    shader AliasGlslBufferBlockMetal {
        struct AliasBlock {
            uint16_t words[2];
            size_t offsets[];
        };

        AliasBlock aliasBlock @glsl_buffer_block(std430) @binding(7);

        uint64_t readAlias(uint index) {
            uint16_t word = aliasBlock.words[1];
            return uint64_t(word) + uint64_t(aliasBlock.offsets[index]);
        }

        void writeAlias(uint index, uint16_t word, size_t offset) {
            aliasBlock.words[1] = word;
            aliasBlock.offsets[index] = offset;
            aliasBlock.words[0] += uint16_t(1u);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "device uchar* aliasBlock" in generated_code
    assert (
        "uint word = (*reinterpret_cast<const device uint*>(aliasBlock + 4));"
        in generated_code
    )
    assert (
        "uint64_t((*reinterpret_cast<const device uint*>(aliasBlock + (8 + index * 4))))"
        in generated_code
    )
    assert "void writeAlias(uint index, uint word, uint64_t offset" in generated_code
    assert (
        "(*reinterpret_cast<device uint*>(aliasBlock + 4)) = uint(word);"
        in generated_code
    )
    assert (
        "(*reinterpret_cast<device uint*>(aliasBlock + (8 + index * 4))) = uint(offset);"
        in generated_code
    )
    assert (
        "(*reinterpret_cast<device uint*>(aliasBlock + 0)) = uint(((*reinterpret_cast<const device uint*>(aliasBlock + 0)) + uint(1u)));"
        in generated_code
    )
    assert "unsupported Metal GLSL buffer block" not in generated_code
    assert "uint16_t" not in generated_code
    assert "size_t" not in generated_code


def test_metal_glsl_buffer_block_atomic_compare_helpers_cover_signed_and_parameters():
    shader = """
    shader BufferBlockAtomicCompareMetal {
        struct AtomicBlock {
            uint counter;
            uint bins[];
        };

        struct SignedAtomicBlock {
            int counter;
            int bins[];
        };

        AtomicBlock atomicBlock @glsl_buffer_block(std430) @binding(17);
        SignedAtomicBlock signedAtomicBlock @glsl_buffer_block(std430) @binding(18);

        uint swapUnsigned(
            AtomicBlock block @glsl_buffer_block(std430),
            uint index,
            uint compareValue,
            uint value
        ) {
            return atomicCompSwap(block.bins[index], compareValue, value);
        }

        int swapSigned(
            SignedAtomicBlock block @glsl_buffer_block(std430),
            uint index,
            int compareValue,
            int value
        ) {
            return atomicCompSwap(block.bins[index], compareValue, value);
        }

        compute {
            void main() {
                uint resultU = swapUnsigned(
                    atomicBlock,
                    1u,
                    atomicBlock.counter,
                    9u
                );
                int resultS = swapSigned(
                    signedAtomicBlock,
                    1u,
                    signedAtomicBlock.counter,
                    -3
                );
                uint oldCounter = atomicCompSwap(
                    atomicBlock.counter,
                    0u,
                    resultU
                );
                int oldSignedCounter = atomicCompSwap(
                    signedAtomicBlock.counter,
                    0,
                    resultS
                );
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert (
        "uint __crossgl_buffer_atomic_compare_exchange_uint("
        "device uchar* buffer, uint offset, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "int __crossgl_buffer_atomic_compare_exchange_int("
        "device uchar* buffer, uint offset, int compareValue, int value)"
        in generated_code
    )
    assert (
        "device atomic_uint* target = reinterpret_cast<device atomic_uint*>(buffer + offset);"
        in generated_code
    )
    assert (
        "device atomic_int* target = reinterpret_cast<device atomic_int*>(buffer + offset);"
        in generated_code
    )
    assert (
        "uint swapUnsigned(device uchar* block, uint index, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "int swapSigned(device uchar* block, uint index, int compareValue, int value)"
        in generated_code
    )
    assert (
        "return __crossgl_buffer_atomic_compare_exchange_uint(block, (4 + index * 4), compareValue, value);"
        in generated_code
    )
    assert (
        "return __crossgl_buffer_atomic_compare_exchange_int(block, (4 + index * 4), compareValue, value);"
        in generated_code
    )
    assert (
        "uint resultU = swapUnsigned(atomicBlock, 1u, "
        "(*reinterpret_cast<const device uint*>(atomicBlock + 0)), 9u);"
        in generated_code
    )
    assert (
        "int resultS = swapSigned(signedAtomicBlock, 1u, "
        "(*reinterpret_cast<const device int*>(signedAtomicBlock + 0)), -3);"
        in generated_code
    )
    assert (
        "uint oldCounter = __crossgl_buffer_atomic_compare_exchange_uint(atomicBlock, 0, 0u, resultU);"
        in generated_code
    )
    assert (
        "int oldSignedCounter = __crossgl_buffer_atomic_compare_exchange_int(signedAtomicBlock, 0, 0, resultS);"
        in generated_code
    )
    assert "atomicCompSwap(" not in generated_code
    assert "atomic_compare_exchange_strong_explicit" not in generated_code


def test_structured_buffer_operations_lower_to_device_buffer():
    code = """
    shader StructuredBufferMetal {
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

    assert "device int* values [[buffer(3)]]" in generated
    assert "constant uint* valuesLength [[buffer(4)]]" in generated
    assert "int v = values[tid.x];" in generated
    assert "len = valuesLength[0];" in generated
    assert "values[tid.x] = v + 1;" in generated
    assert "RWStructuredBuffer" not in generated
    assert "buffer_load" not in generated
    assert "buffer_store" not in generated
    assert "buffer_dimensions" not in generated
    assert "unsupported Metal buffer dimensions" not in generated


def test_readonly_structured_buffer_uses_const_device_pointer():
    code = """
    shader StructuredBufferMetal {
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

    assert "const device int* values [[buffer(1)]]" in generated
    assert "int v = values[tid.x];" in generated


def test_structured_buffer_dimensions_helpers_thread_length_sidecars():
    code = """
    shader StructuredBufferDimensionsHelperMetal {
        RWStructuredBuffer<int> values @ binding(1);

        uint countValuesInner(RWStructuredBuffer<int> localValues) {
            uint len;
            buffer_dimensions(localValues, len);
            return len;
        }

        uint countValues(RWStructuredBuffer<int> localValues) {
            return countValuesInner(localValues);
        }

        compute {
            void main() {
                uint len = countValues(values);
            }
        }
    }
    """
    ast = parse_code(tokenize_code(code))

    generated = generate_code(ast)

    assert "device int* values [[buffer(1)]]" in generated
    assert "constant uint* valuesLength [[buffer(2)]]" in generated
    assert (
        "uint countValuesInner(device int* localValues, constant uint* localValuesLength)"
        in generated
    )
    assert (
        "uint countValues(device int* localValues, constant uint* localValuesLength)"
        in generated
    )
    assert "len = localValuesLength[0];" in generated
    assert "return countValuesInner(localValues, localValuesLength);" in generated
    assert "uint len = countValues(values, valuesLength);" in generated
    assert "unsupported Metal buffer dimensions" not in generated
    assert "buffer_dimensions" not in generated


def test_append_consume_structured_buffers_use_sidecar_counters():
    code = """
    shader StructuredBufferAppendConsumeMetal {
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

    assert "device int* appendValues [[buffer(1)]]" in generated
    assert "device int* consumeValues [[buffer(2)]]" in generated
    assert "device atomic_uint* appendValuesCounter [[buffer(3)]]" in generated
    assert "device atomic_uint* consumeValuesCounter [[buffer(4)]]" in generated
    assert (
        "appendValues[atomic_fetch_add_explicit(appendValuesCounter, 1u, memory_order_relaxed)] = int(value);"
        in generated
    )
    assert (
        "int consumed = consumeValues[(atomic_fetch_sub_explicit(consumeValuesCounter, 1u, memory_order_relaxed) - 1u)];"
        in generated
    )
    assert "unsupported Metal buffer append" not in generated
    assert "unsupported Metal buffer consume" not in generated
    assert "buffer_append" not in generated
    assert "buffer_consume" not in generated


def test_append_consume_structured_buffer_helpers_thread_sidecar_counters():
    code = """
    shader StructuredBufferAppendConsumeHelpersMetal {
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
        "void pushValue(device int* values, device atomic_uint* valuesCounter, uint value)"
        in generated
    )
    assert (
        "values[atomic_fetch_add_explicit(valuesCounter, 1u, memory_order_relaxed)] = int(value);"
        in generated
    )
    assert (
        "int popValue(device int* values, device atomic_uint* valuesCounter)"
        in generated
    )
    assert (
        "return values[(atomic_fetch_sub_explicit(valuesCounter, 1u, memory_order_relaxed) - 1u)];"
        in generated
    )
    assert "pushValue(appendValues, appendValuesCounter, value);" in generated
    assert "int consumed = popValue(consumeValues, consumeValuesCounter);" in generated
    assert "buffer_append" not in generated
    assert "buffer_consume" not in generated


def test_structured_buffer_arrays_lower_to_device_pointer_arrays():
    code = """
    shader StructuredBufferArrayMetal {
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

    assert "array<device int*, 2> buffers [[buffer(3)]]" in generated
    assert "array<constant uint*, 2> buffersLength [[buffer(5)]]" in generated
    assert "int v = buffers[index][tid.x];" in generated
    assert "len = buffersLength[index][0];" in generated
    assert "buffers[index][tid.x] = v + 1;" in generated
    assert "unsupported Metal buffer dimensions" not in generated


def test_append_structured_buffer_arrays_use_sidecar_counter_arrays():
    code = """
    shader StructuredBufferAppendArrayMetal {
        AppendStructuredBuffer<int> appendValues[2] @ binding(1);

        void pushOne(AppendStructuredBuffer<int> values[2], uint which, uint value) {
            buffer_append(values[which], int(value));
        }

        compute {
            void main(uint index, uint value) {
                buffer_append(appendValues[index], int(value));
                pushOne(appendValues, index, value);
            }
        }
    }
    """
    ast = parse_code(tokenize_code(code))

    generated = generate_code(ast)

    assert "array<device int*, 2> appendValues [[buffer(1)]]" in generated
    assert (
        "array<device atomic_uint*, 2> appendValuesCounter [[buffer(3)]]" in generated
    )
    assert (
        "void pushOne(array<device int*, 2> values, array<device atomic_uint*, 2> valuesCounter, uint which, uint value)"
        in generated
    )
    assert (
        "appendValues[index][atomic_fetch_add_explicit(appendValuesCounter[index], 1u, memory_order_relaxed)] = int(value);"
        in generated
    )
    assert (
        "values[which][atomic_fetch_add_explicit(valuesCounter[which], 1u, memory_order_relaxed)] = int(value);"
        in generated
    )
    assert "pushOne(appendValues, appendValuesCounter, index, value);" in generated
    assert "buffer_append" not in generated


def test_metal_structured_buffer_array_binding_overlap_raises():
    code = """
    shader StructuredBufferArrayBindingOverlapMetal {
        RWStructuredBuffer<int> first[2] @ binding(3);
        RWStructuredBuffer<int> second @ binding(4);

        int read(uint index) {
            return buffer_load(first[1], index) + buffer_load(second, index);
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting Metal resource binding for 'second'",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_metal_texture_array_binding_overlap_raises():
    code = """
    shader TextureArrayBindingOverlapMetal {
        sampler2D first[2] @texture(3);
        sampler2D second @texture(4);

        vec4 read(vec2 uv) {
            return texture(first[1], uv) + texture(second, uv);
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting Metal resource binding for 'second'",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_metal_sampler_array_binding_overlap_raises():
    code = """
    shader SamplerArrayBindingOverlapMetal {
        sampler first[2] @sampler(1);
        sampler second @sampler(2);
        sampler2D tex;

        vec4 read(vec2 uv) {
            return texture(tex, first[1], uv) + texture(tex, second, uv);
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting Metal resource binding for 'second'",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_metal_texture_and_image_bindings_share_texture_namespace():
    code = """
    shader TextureImageBindingOverlapMetal {
        sampler2D textures[2] @texture(3);
        image2D storageImage @binding(4);

        vec4 read(vec2 uv, ivec2 pixel) {
            return texture(textures[1], uv) + imageLoad(storageImage, pixel);
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting Metal resource binding for 'storageImage'",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_metal_texture_sampler_and_buffer_bindings_are_independent():
    code = """
    shader ResourceBindingNamespacesMetal {
        sampler2D colorMap @binding(0);
        sampler linearSampler @binding(0);
        RWStructuredBuffer<int> values @binding(0);

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                return texture(colorMap, linearSampler, uv) + vec4(float(buffer_load(values, 0)));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert "device int* values [[buffer(0)]]" in generated_code
    assert "colorMap.sample(linearSampler, uv)" in generated_code
    assert "values[0]" in generated_code


def test_metal_auto_bindings_skip_later_explicit_global_bindings():
    code = """
    shader LateExplicitBindingsMetal {
        sampler2D autoTexture;
        image2D autoImage;
        RWStructuredBuffer<int> autoBuffer;
        sampler autoSampler;

        cbuffer AutoBlock {
            vec4 autoTint;
        };

        cbuffer ExplicitBlock @buffer(0) {
            vec4 explicitTint;
        };

        sampler2D explicitTexture @texture(0);
        image2D explicitImage @texture(1);
        RWStructuredBuffer<int> explicitBuffer @buffer(2);
        sampler explicitSampler @sampler(0);

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                vec4 stored = imageLoad(autoImage, ivec2(0, 0));
                int value = buffer_load(autoBuffer, 0) + buffer_load(explicitBuffer, 0);
                return texture(autoTexture, autoSampler, uv) +
                    texture(explicitTexture, explicitSampler, uv) +
                    stored + autoTint + explicitTint + vec4(float(value));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "constant AutoBlock& autoBlock [[buffer(1)]]" in generated_code
    assert "constant ExplicitBlock& explicitBlock [[buffer(0)]]" in generated_code
    assert "device int* autoBuffer [[buffer(3)]]" in generated_code
    assert "device int* explicitBuffer [[buffer(2)]]" in generated_code
    assert "texture2d<float> autoTexture [[texture(2)]]" in generated_code
    assert (
        "texture2d<float, access::read_write> autoImage [[texture(3)]]"
        in generated_code
    )
    assert "texture2d<float> explicitTexture [[texture(0)]]" in generated_code
    assert (
        "texture2d<float, access::read_write> explicitImage [[texture(1)]]"
        in generated_code
    )
    assert "sampler autoSampler [[sampler(1)]]" in generated_code
    assert "sampler explicitSampler [[sampler(0)]]" in generated_code


def test_metal_unsigned_integer_literal_suffix_codegen():
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


def test_metal_resource_binding_attributes_are_not_parameter_semantics():
    code = """
    shader BindingAttributes {
        vec4 sampleBound(sampler2D tex @texture(1), sampler samp @sampler(2), sampler2D registered @register(t3), vec2 uv) {
            return texture(tex, samp, uv) + texture(registered, uv);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "float4 sampleBound(texture2d<float> tex, sampler samp, texture2d<float> registered, float2 uv)"
        in generated_code
    )
    assert "tex.sample(samp, uv)" in generated_code
    assert (
        "registered.sample(sampler(mag_filter::linear, min_filter::linear), uv)"
        in generated_code
    )
    assert "[[texture]]" not in generated_code
    assert "[[sampler]]" not in generated_code
    assert "[[register]]" not in generated_code


def test_metal_global_resource_binding_attributes_drive_indices():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "texture2d<float> explicitTex [[texture(3)]]" in generated_code
    assert (
        "texture2d<float, access::read_write> storageImage [[texture(7)]]"
        in generated_code
    )
    assert "texture2d<float> registerTex [[texture(8)]]" in generated_code
    assert "texture2d<float> autoTex [[texture(9)]]" in generated_code
    assert "sampler samp [[sampler(5)]]" in generated_code


def test_metal_stage_resource_binding_attributes_drive_indices():
    code = """
    shader ExplicitStageBindings {
        fragment {
            vec4 main(sampler2D tex @texture(4), sampler samp @sampler(6), vec2 uv @TEXCOORD0) @gl_FragColor {
                return texture(tex, samp, uv);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "texture2d<float> tex [[texture(4)]]" in generated_code
    assert "sampler samp [[sampler(6)]]" in generated_code


def test_metal_stage_local_resources_emit_entry_parameters():
    code = """
    shader StageLocalResourcesMetal {
        fragment {
            uniform sampler2D localTex @texture(2);
            uniform sampler localSampler @sampler(4);
            uniform image2D localImage @texture(5);

            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                vec4 stored = imageLoad(localImage, ivec2(0, 0));
                return texture(localTex, localSampler, uv) + stored;
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "texture2d<float> localTex [[texture(2)]]" in generated_code
    assert "sampler localSampler [[sampler(4)]]" in generated_code
    assert (
        "texture2d<float, access::read_write> localImage [[texture(5)]]"
        in generated_code
    )
    assert "localTex.sample(localSampler, uv)" in generated_code


def test_metal_stage_parameter_texture_array_binding_overlap_raises():
    code = """
    shader StageParameterTextureArrayOverlap {
        fragment {
            vec4 main(
                sampler2D textures[2] @texture(0),
                sampler2D other @texture(1),
                vec2 uv @TEXCOORD0
            ) @gl_FragColor {
                return texture(textures[0], uv) + texture(other, uv);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting Metal resource binding for 'other'",
    ):
        MetalCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")


def test_metal_stage_parameter_binding_conflicts_with_global_resource():
    code = """
    shader StageParameterGlobalBindingOverlap {
        sampler2D globalTex @texture(0);

        fragment {
            vec4 main(sampler2D localTex @texture(0), vec2 uv @TEXCOORD0) @gl_FragColor {
                return texture(localTex, uv) + texture(globalTex, uv);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting Metal resource binding for 'localTex'",
    ):
        MetalCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")


def test_metal_cbuffer_binding_attributes_drive_buffer_indices():
    code = """
    shader CBufferBindings {
        cbuffer Camera @register(b2) {
            vec4 viewRow;
        };

        uniform Material @buffer(4) {
            vec4 tint;
        };

        cbuffer AutoBlock {
            vec4 color;
        };

        fragment {
            vec4 main() @gl_FragColor {
                return vec4(1.0);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "constant Camera& camera [[buffer(2)]]" in generated_code
    assert "constant Material& material [[buffer(4)]]" in generated_code
    assert "constant AutoBlock& autoBlock [[buffer(5)]]" in generated_code


def test_metal_cbuffer_reserves_implicit_structured_buffer_indices():
    code = """
    shader CBufferStructuredBufferSlots {
        cbuffer Constants {
            float scale;
        };

        RWStructuredBuffer<int> values;

        compute {
            void main(uint3 tid @ gl_GlobalInvocationID) {
                int value = buffer_load(values, tid.x);
                buffer_store(values, tid.x, int(float(value) * scale));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(code), "compute"
    )

    assert "constant Constants& constants [[buffer(0)]]" in generated_code
    assert "device int* values [[buffer(1)]]" in generated_code
    assert "[[buffer(0)]], device int* values [[buffer(0)]]" not in generated_code


def test_metal_cbuffer_and_structured_buffer_binding_overlap_raises():
    code = """
    shader CBufferStructuredBufferOverlap {
        cbuffer Constants @binding(0) {
            float scale;
        };

        RWStructuredBuffer<int> values @binding(0);

        compute {
            void main(uint3 tid @ gl_GlobalInvocationID) {
                int value = buffer_load(values, tid.x);
                buffer_store(values, tid.x, int(float(value) * scale));
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting Metal resource binding for 'values'",
    ):
        MetalCodeGen().generate_stage(crosstl.translator.parse(code), "compute")


def test_struct():
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
        pytest.fail("Struct codegen not implemented.")


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
        pytest.fail("if statement codegen not implemented.")


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

            for (int i = 0; i < 10; i++) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
            }
            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
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
        pytest.fail("for statement codegen not implemented.")


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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "const float weights[2];" in generated_code
    assert "for (int i = 0; i < 2; ++i)" in generated_code
    assert "for (i = 0; i < 4; ++i)" in generated_code
    assert "for (const int fixed = 0; fixed < 0; )" in generated_code
    assert "for (; ; )" in generated_code
    assert "continue;" in generated_code
    assert "break;" in generated_code
    assert "for (i; i < 2; ++i)" not in generated_code
    assert "for (fixed; fixed < 0; )" not in generated_code
    assert "BreakNode(" not in generated_code
    assert "ContinueNode(" not in generated_code


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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "while (true)" in generated_code
    assert "i = i + 1;" in generated_code
    assert "if (i >= limit)" in generated_code
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "do {" in generated_code
    assert "i = i + 1;" in generated_code
    assert "if (i >= limit)" in generated_code
    assert "break;" in generated_code
    assert "} while (i < 4);" in generated_code
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "for (int i = 0; i < limit; ++i)" in generated_code
    assert "total = total + i;" in generated_code
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "for (int i = 2; i < 5; ++i)" in generated_code
    assert "for (int j = 1; j <= limit; ++j)" in generated_code
    assert "total = total + i;" in generated_code
    assert "total = total + j;" in generated_code
    assert "return total;" in generated_code
    assert "RangeNode(" not in generated_code
    assert "ForInNode(" not in generated_code


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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "while (i < 4)" in generated_code
    assert "switch (i)" in generated_code
    assert "case 0:" in generated_code
    assert "default:" in generated_code
    assert "i = i + 1;" in generated_code
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" in generated_code
    assert "case 0:\n        case 1: {" in generated_code
    assert "case 2: {\n            switch (submode)" in generated_code
    assert generated_code.count("default:") == 2
    assert "value = value + 1;" in generated_code
    assert "value = value + 2;" in generated_code
    assert "value = value + 3;" in generated_code
    assert "value = value + 4;" in generated_code
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "int left = pair.left;" in generated_code
    assert "int right = pair.right;" in generated_code
    assert "value = left + right;" in generated_code
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int Mode_Add = 0;" in generated_code
    assert "static const int Mode_Multiply = 4;" in generated_code
    assert "static const int Mode_Divide = 5;" in generated_code
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int LightingModel_Phong = 0;" in generated_code
    assert "static const int LightingModel_Toon = 1;" in generated_code
    assert "struct LightingModel {" in generated_code
    assert "int variant;" in generated_code
    assert "float3 ambient;" in generated_code
    assert "float3 base_color;" in generated_code
    assert "float3 shade(LightingModel model)" in generated_code
    assert "if ((model.variant == LightingModel_Phong))" in generated_code
    assert "else if ((model.variant == LightingModel_Toon))" in generated_code
    assert "float3 ambient = model.ambient;" in generated_code
    assert "float shininess = model.shininess;" in generated_code
    assert "float3 base_color = model.base_color;" in generated_code
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int MaybeInt_Value = 0;" in generated_code
    assert "static const int MaybeInt_Pair = 1;" in generated_code
    assert "static const int MaybeInt_Missing = 2;" in generated_code
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

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
        MetalCodeGen().generate(crosstl.translator.parse(shader))


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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int Option_Some = 0;" in generated_code
    assert "static const int Result_Ok = 0;" in generated_code
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "Result_int_MathError make_result(bool ok)" in generated_code
    assert (
        "Result_int_MathError __crossgl_match_subject_0 = make_result(ok);"
        in generated_code
    )
    assert generated_code.count("make_result(ok)") == 1
    assert "if ((__crossgl_match_subject_0.variant == Result_Ok))" in generated_code
    assert (
        "else if ((__crossgl_match_subject_0.variant == Result_Err))" in generated_code
    )
    assert "int value = __crossgl_match_subject_0.Ok_0;" in generated_code
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "Result_float3_MathError make_result(bool ok)" in generated_code
    assert "float3 read(bool ok, float3 fallback)" in generated_code
    assert "float3 value;" in generated_code
    assert (
        "Result_float3_MathError __crossgl_match_subject_0 = make_result(ok);"
        in generated_code
    )
    assert "if ((__crossgl_match_subject_0.variant == Result_Ok))" in generated_code
    assert (
        "else if ((__crossgl_match_subject_0.variant == Result_Err))" in generated_code
    )
    assert "float3 actual = __crossgl_match_subject_0.Ok_0;" in generated_code
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "RenderOutput RenderOutput_Clear_make(float4 payload0, float payload1)"
        in generated_code
    )
    assert "RenderOutput clear(float4 color, float depth)" in generated_code
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "Pair make_pair(float3 value, float weight)" in generated_code
    assert "return Pair{value, weight};" in generated_code
    assert "float4 make_color(float3 color)" in generated_code
    assert "return float4(color, 1.0);" in generated_code
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct VertexInput {" in generated_code
    assert "struct VertexOutput {" in generated_code
    assert (
        "vertex VertexOutput vertex_main(VertexInput input [[stage_in]])"
        in generated_code
    )
    assert (
        "return VertexOutput{float4(input.position, 1.0), input.color};"
        in generated_code
    )
    assert "ConstructorNode(" not in generated_code


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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "float add_one_float(float value)" in generated_code
    assert "return (value + 1.0);" in generated_code
    assert "return add_one_float(value);" in generated_code
    assert "T add_one(T value)" not in generated_code
    assert "return add_one(value);" not in generated_code
    assert ".add(" not in generated_code
    assert "T::one" not in generated_code


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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct Box_int {" in generated_code
    assert "int value;" in generated_code
    assert "struct PairBox_int {" in generated_code
    assert "Box_int first;" in generated_code
    assert "int second;" in generated_code
    assert "struct Box {" not in generated_code
    assert "struct PairBox {" not in generated_code
    assert "Box_int make(int value)" in generated_code
    assert "return Box_int{value};" in generated_code
    assert "int read(Box_int item)" in generated_code
    assert "return item.value;" in generated_code
    assert "PairBox_int make_pair(int value)" in generated_code
    assert "return PairBox_int{Box_int{value}, value};" in generated_code
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct Geometry {" in generated_code
    assert "struct Box_float {" in generated_code
    assert "Geometry make_geometry(float3 center, float3 normal)" in generated_code
    assert "return Geometry{center, normal};" in generated_code
    assert "Box_float make_box(float3 value)" in generated_code
    assert "float3 normalized = normalize(value);" in generated_code
    assert "Box_float wrapped = Box_float{normalized.x};" in generated_code
    assert "return wrapped;" in generated_code
    assert "Option_Self normalized" not in generated_code
    assert "Box<" not in generated_code
    assert "ConstructorNode(" not in generated_code


def test_ray_payload_semantics():
    code = """
    shader rt {
        struct Payload {
            vec3 color;
        };
        ray_generation {
            void main(Payload payload @ payload) {
                payload.color = vec3(1.0, 0.0, 0.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    assert "kernel void kernel_main(device Payload& payload)" in generated
    assert "payload.color = float3(1.0, 0.0, 0.0);" in generated
    assert "[[payload]]" not in generated


def test_ray_hit_and_callable_stages_lower_to_visible_functions():
    code = """
    shader rt {
        struct Payload {
            vec3 color;
        };
        struct HitAttrib {
            vec2 bary;
        };
        ray_any_hit {
            void main(Payload payload @ payload, HitAttrib attr @ hit_attribute) {
                payload.color = vec3(attr.bary, 0.0);
            }
        }
        ray_closest_hit {
            void main(Payload payload @ payload, HitAttrib attr @ hit_attribute) {
                payload.color = vec3(attr.bary, 1.0);
            }
        }
        ray_miss {
            void main(Payload payload @ payload) {
                payload.color = vec3(0.0, 0.0, 0.0);
            }
        }
        ray_callable {
            void main(Payload payload @ payload) {
                payload.color = vec3(1.0, 1.0, 1.0);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "[[visible]] void anyhit_main("
        "ray_data Payload& payload [[payload]], HitAttrib attr)"
    ) in generated
    assert (
        "[[visible]] void closesthit_main("
        "ray_data Payload& payload [[payload]], HitAttrib attr)"
    ) in generated
    assert (
        "[[visible]] void miss_main(ray_data Payload& payload [[payload]])"
    ) in generated
    assert (
        "[[visible]] void callable_main(ray_data Payload& payload [[payload]])"
    ) in generated
    assert "anyhit void" not in generated
    assert "closesthit void" not in generated
    assert "miss void" not in generated
    assert "callable void" not in generated
    assert "[[hit_attribute]]" not in generated


def test_ray_callable_data_semantic_uses_ray_data_address_space():
    code = """
    shader rt {
        struct Payload {
            vec3 color;
        };
        ray_callable {
            void main(Payload data @ callable_data) {
                data.color = vec3(1.0, 1.0, 1.0);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "[[visible]] void callable_main(ray_data Payload& data)" in generated
    assert "data.color = float3(1.0, 1.0, 1.0);" in generated
    assert "[[callable_data]]" not in generated
    assert "Payload data" not in generated


def test_metal_acceleration_structure_globals_lower_to_buffer_parameters():
    code = """
    shader rt {
        accelerationStructureEXT topLevelAS @binding(2);
        ray_generation {
            void main() { }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "using namespace metal::raytracing;" in generated
    assert (
        "kernel void kernel_main("
        "instance_acceleration_structure topLevelAS [[buffer(2)]])"
    ) in generated
    assert "accelerationStructureEXT topLevelAS;" not in generated
    assert "instance_acceleration_structure topLevelAS;" not in generated


def test_metal_trace_ray_full_signature_lowers_to_intersector_query():
    code = """
    shader rt {
        accelerationStructureEXT topLevelAS @binding(0);
        ray_generation {
            void main() {
                TraceRay(
                    topLevelAS,
                    0,
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
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "using namespace metal::raytracing;" in generated
    assert "instance_acceleration_structure topLevelAS [[buffer(0)]]" in generated
    assert "ray __crossgl_ray_0;" in generated
    assert "__crossgl_ray_0.origin = float3(0.0);" in generated
    assert "__crossgl_ray_0.direction = float3(0.0, 0.0, 1.0);" in generated
    assert "__crossgl_ray_0.min_distance = 0.001;" in generated
    assert "__crossgl_ray_0.max_distance = 1000.0;" in generated
    assert (
        "intersector<triangle_data, instancing> __crossgl_intersector_1;" in generated
    )
    assert (
        "intersection_result<triangle_data, instancing> __crossgl_intersection_2 = "
        "__crossgl_intersector_1.intersect(__crossgl_ray_0, topLevelAS, 255);"
    ) in generated
    assert "(void)__crossgl_intersection_2;" in generated
    assert "TraceRay" not in generated


def test_metal_trace_ray_uses_single_intersection_function_table():
    code = """
    shader rt {
        accelerationStructureEXT topLevelAS @binding(0);
        intersection_function_table<instancing> intersectionFunctions @binding(1);

        ray_generation {
            void main() {
                TraceRay(
                    topLevelAS,
                    0,
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
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "intersector<instancing> __crossgl_intersector_1;" in generated
    assert (
        "intersection_result<instancing> __crossgl_intersection_2 = "
        "__crossgl_intersector_1.intersect("
        "__crossgl_ray_0, topLevelAS, 255, intersectionFunctions);"
    ) in generated


def test_metal_trace_ray_threads_intersection_function_table_through_helpers():
    code = """
    shader rt {
        accelerationStructureEXT topLevelAS @binding(0);
        intersection_function_table<instancing> intersectionFunctions @binding(1);

        void shoot(vec3 origin, vec3 direction) {
            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                origin,
                0.001,
                direction,
                1000.0,
                0
            );
        }

        ray_generation {
            void main() {
                shoot(vec3(0.0), vec3(0.0, 0.0, 1.0));
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void shoot(float3 origin, float3 direction, "
        "instance_acceleration_structure topLevelAS, "
        "intersection_function_table<instancing> intersectionFunctions)"
    ) in generated
    assert (
        "__crossgl_intersector_1.intersect("
        "__crossgl_ray_0, topLevelAS, 255, intersectionFunctions);"
    ) in generated
    assert (
        "shoot(float3(0.0), float3(0.0, 0.0, 1.0), "
        "topLevelAS, intersectionFunctions);"
    ) in generated


def test_metal_trace_ray_forwards_thread_local_payload_with_intersection_table():
    code = """
    shader rt {
        struct Payload {
            vec3 color;
        };

        accelerationStructureEXT topLevelAS @binding(0);
        intersection_function_table<instancing> intersectionFunctions @binding(1);

        ray_generation {
            void main() {
                Payload payload;
                payload.color = vec3(1.0, 0.0, 0.0);
                TraceRay(
                    topLevelAS,
                    0,
                    0xff,
                    0,
                    1,
                    0,
                    vec3(0.0),
                    0.001,
                    vec3(0.0, 0.0, 1.0),
                    1000.0,
                    payload
                );
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "Payload payload;" in generated
    assert (
        "intersection_result<instancing> __crossgl_intersection_2 = "
        "__crossgl_intersector_1.intersect("
        "__crossgl_ray_0, topLevelAS, 255, intersectionFunctions, payload);"
    ) in generated
    assert "unsupported Metal ray tracing intrinsic: TraceRay payload" not in generated


def test_metal_trace_ray_forwards_helper_parameter_payload():
    code = """
    shader rt {
        struct Payload {
            vec3 color;
        };

        accelerationStructureEXT topLevelAS @binding(0);
        intersection_function_table<instancing> intersectionFunctions @binding(1);

        void shoot(Payload payload, vec3 origin, vec3 direction) {
            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                origin,
                0.001,
                direction,
                1000.0,
                payload
            );
        }

        ray_generation {
            void main() {
                Payload payload;
                shoot(payload, vec3(0.0), vec3(0.0, 0.0, 1.0));
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void shoot(Payload payload, float3 origin, float3 direction, "
        "instance_acceleration_structure topLevelAS, "
        "intersection_function_table<instancing> intersectionFunctions)"
    ) in generated
    assert (
        "__crossgl_intersector_1.intersect("
        "__crossgl_ray_0, topLevelAS, 255, intersectionFunctions, payload);"
    ) in generated
    assert (
        "shoot(payload, float3(0.0), float3(0.0, 0.0, 1.0), "
        "topLevelAS, intersectionFunctions);"
    ) in generated


def test_metal_trace_ray_payload_without_intersection_table_emits_diagnostic():
    code = """
    shader rt {
        struct Payload {
            vec3 color;
        };

        accelerationStructureEXT topLevelAS @binding(0);

        ray_generation {
            void main() {
                Payload payload;
                TraceRay(
                    topLevelAS,
                    0,
                    0xff,
                    0,
                    1,
                    0,
                    vec3(0.0),
                    0.001,
                    vec3(0.0, 0.0, 1.0),
                    1000.0,
                    payload
                );
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "__crossgl_intersector_1.intersect(__crossgl_ray_0, topLevelAS, 255);"
        in generated
    )
    assert (
        "unsupported Metal ray tracing intrinsic: TraceRay payload - "
        "payload forwarding requires a compatible intersection_function_table"
    ) in generated
    assert "intersectionFunctions, payload" not in generated


def test_metal_trace_ray_rejects_non_thread_payload_parameter():
    code = """
    shader rt {
        struct Payload {
            vec3 color;
        };

        accelerationStructureEXT topLevelAS @binding(0);
        intersection_function_table<instancing> intersectionFunctions @binding(1);

        ray_generation {
            void main(Payload payload @payload) {
                TraceRay(
                    topLevelAS,
                    0,
                    0xff,
                    0,
                    1,
                    0,
                    vec3(0.0),
                    0.001,
                    vec3(0.0, 0.0, 1.0),
                    1000.0,
                    payload
                );
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "kernel void kernel_main(device Payload& payload" in generated
    assert (
        "__crossgl_intersector_1.intersect("
        "__crossgl_ray_0, topLevelAS, 255, intersectionFunctions);"
    ) in generated
    assert (
        "payload forwarding requires a thread-local payload lvalue; "
        "ray payload parameters use a non-thread address space"
    ) in generated
    assert "intersectionFunctions, payload" not in generated


def test_metal_trace_ray_skips_incompatible_intersection_function_table():
    code = """
    shader rt {
        accelerationStructureEXT topLevelAS @binding(0);
        intersection_function_table<triangle_data> primitiveIntersectionFunctions
            @binding(1);

        ray_generation {
            void main() {
                TraceRay(
                    topLevelAS,
                    0,
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
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "__crossgl_intersector_1.intersect(__crossgl_ray_0, topLevelAS, 255);"
        in generated
    )
    assert "primitiveIntersectionFunctions)" not in generated


def test_metal_trace_ray_uses_primitive_acceleration_structure_shape():
    code = """
    shader rt {
        primitive_acceleration_structure primitiveAS @binding(0);
        intersection_function_table<triangle_data> intersectionFunctions @binding(1);

        ray_generation {
            void main() {
                TraceRay(
                    primitiveAS,
                    0,
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
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "primitive_acceleration_structure primitiveAS [[buffer(0)]]" in generated
    assert "intersector<triangle_data> __crossgl_intersector_1;" in generated
    assert (
        "intersection_result<triangle_data> __crossgl_intersection_2 = "
        "__crossgl_intersector_1.intersect("
        "__crossgl_ray_0, primitiveAS, intersectionFunctions);"
    ) in generated
    assert "primitiveAS, 0xff" not in generated


def test_metal_acceleration_structure_globals_thread_through_helpers():
    code = """
    shader rt {
        accelerationStructureEXT topLevelAS @binding(0);

        void shoot(vec3 origin, vec3 direction) {
            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                origin,
                0.001,
                direction,
                1000.0,
                0
            );
        }

        ray_generation {
            void main() {
                shoot(vec3(0.0), vec3(0.0, 0.0, 1.0));
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void shoot(float3 origin, float3 direction, "
        "instance_acceleration_structure topLevelAS)"
    ) in generated
    assert "shoot(float3(0.0), float3(0.0, 0.0, 1.0), topLevelAS);" in generated
    assert "shoot(float3(0.0), float3(0.0, 0.0, 1.0));" not in generated


def test_metal_unsupported_ray_intrinsics_emit_compile_safe_diagnostics():
    code = """
    shader rt {
        ray_any_hit {
            void main() {
                IgnoreHit();
                AcceptHitAndEndSearch();
            }
        }
        ray_intersection {
            bool main() @triangle {
                ReportHit(1.0, 0);
                return true;
            }
        }
        ray_callable {
            void main() {
                CallShader(0, 1);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "unsupported Metal ray tracing intrinsic: IgnoreHit" in generated
    assert "unsupported Metal ray tracing intrinsic: AcceptHitAndEndSearch" in generated
    assert "unsupported Metal ray tracing intrinsic: ReportHit" in generated
    assert "unsupported Metal ray tracing intrinsic: CallShader" in generated
    assert "IgnoreHit();" not in generated
    assert "AcceptHitAndEndSearch();" not in generated
    assert "ReportHit(" not in generated
    assert "CallShader(" not in generated


def test_metal_callshader_lowers_to_visible_function_table_call():
    code = """
    shader rt {
        struct CallableData {
            vec4 color;
        };

        visible_function_table<CallableData> callables @binding(1);

        ray_generation {
            void main() {
                CallableData data;
                data.color = vec4(1.0);
                CallShader(0, data);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "visible_function_table<void(thread CallableData&)> callables [[buffer(1)]]"
        in generated
    )
    assert "callables[0](data);" in generated
    assert "unsupported Metal ray tracing intrinsic: CallShader" not in generated
    assert "CallShader(" not in generated


def test_metal_callshader_visible_function_table_threads_through_helpers():
    code = """
    shader rt {
        struct CallableData {
            vec4 color;
        };

        visible_function_table<CallableData> callables @binding(2);

        void invoke(uint shaderIndex, CallableData data) {
            CallShader(shaderIndex, data);
        }

        ray_generation {
            void main() {
                CallableData data;
                data.color = vec4(1.0);
                invoke(3u, data);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void invoke(uint shaderIndex, CallableData data, "
        "visible_function_table<void(thread CallableData&)> callables)"
    ) in generated
    assert "callables[shaderIndex](data);" in generated
    assert "invoke(3u, data, callables);" in generated
    assert "invoke(3u, data);" not in generated


def test_metal_callshader_accepts_explicit_visible_function_table_argument():
    code = """
    shader rt {
        struct CallableData {
            vec4 color;
        };

        visible_function_table<CallableData> primaryCallables @binding(1);
        visible_function_table<CallableData> secondaryCallables @binding(2);

        ray_generation {
            void main() {
                CallableData data;
                CallShader(secondaryCallables, 4u, data);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "secondaryCallables[4u](data);" in generated
    assert "unsupported Metal ray tracing intrinsic: CallShader" not in generated


def test_metal_intersection_function_table_globals_lower_to_buffer_parameters():
    code = """
    shader rt {
        intersection_function_table<instancing> intersectionFunctions @binding(3);

        ray_generation {
            void main() {
                uint count = intersectionFunctions.size();
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "using namespace metal::raytracing;" in generated
    assert (
        "kernel void kernel_main("
        "intersection_function_table<instancing> intersectionFunctions [[buffer(3)]])"
    ) in generated
    assert "uint count = intersectionFunctions.size();" in generated
    assert (
        "intersection_function_table<instancing> intersectionFunctions;"
        not in generated
    )


def test_metal_intersection_function_table_globals_thread_through_helpers():
    code = """
    shader rt {
        intersection_function_table<instancing> intersectionFunctions @binding(4);

        uint tableSize() {
            return intersectionFunctions.size();
        }

        ray_generation {
            void main() {
                uint count = tableSize();
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "uint tableSize("
        "intersection_function_table<instancing> intersectionFunctions)"
    ) in generated
    assert "return intersectionFunctions.size();" in generated
    assert "uint count = tableSize(intersectionFunctions);" in generated
    assert "uint count = tableSize();" not in generated


def test_metal_visible_function_table_stage_parameter_maps_signature_and_binding():
    code = """
    shader rt {
        struct CallableData {
            vec4 color;
        };

        ray_generation {
            void main(visible_function_table<CallableData> callables @binding(1)) {
                CallableData data;
                CallShader(callables, 0, data);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "kernel void kernel_main("
        "visible_function_table<void(thread CallableData&)> callables [[buffer(1)]])"
    ) in generated
    assert "callables[0](data);" in generated
    assert "visible_function_table<CallableData> callables" not in generated


def test_metal_intersection_function_table_stage_parameter_gets_buffer_binding():
    code = """
    shader rt {
        ray_generation {
            void main(
                intersection_function_table<instancing> intersectionFunctions
                    @binding(3)
            ) {
                uint count = intersectionFunctions.size();
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "kernel void kernel_main("
        "intersection_function_table<instancing> intersectionFunctions [[buffer(3)]])"
    ) in generated
    assert "uint count = intersectionFunctions.size();" in generated


def test_metal_arrayed_visible_function_table_emits_compile_safe_diagnostic():
    code = """
    shader rt {
        struct CallableData {
            vec4 color;
        };

        visible_function_table<CallableData> callables[2] @binding(1);

        void invoke(uint shaderIndex, CallableData data) {
            CallShader(callables[0], shaderIndex, data);
        }

        ray_generation {
            void main() {
                CallableData data;
                invoke(1u, data);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "arrays of visible_function_table are not valid Metal buffer parameters" in (
        generated
    )
    assert "array<visible_function_table" not in generated
    assert "callables[0][shaderIndex](data)" not in generated
    assert "void invoke(uint shaderIndex, CallableData data)" in generated
    assert "invoke(1u, data);" in generated


def test_metal_arrayed_intersection_function_table_emits_compile_safe_diagnostic():
    code = """
    shader rt {
        intersection_function_table<instancing> intersectionFunctions[2] @binding(3);

        uint tableSize() {
            return intersectionFunctions[0].size();
        }

        ray_generation {
            void main() {
                uint count = tableSize();
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "arrays of intersection_function_table are not valid Metal buffer parameters"
        in generated
    )
    assert "array<intersection_function_table" not in generated
    assert "return 0 /* unsupported Metal ray tracing resource:" in generated
    assert "intersectionFunctions[0].size()" not in generated


def test_metal_arrayed_visible_function_table_parameter_emits_compile_safe_diagnostic():
    code = """
    shader rt {
        struct CallableData {
            vec4 color;
        };

        ray_generation {
            void main(visible_function_table<CallableData> callables[2] @binding(1)) {
                CallableData data;
                CallShader(callables[0], 0, data);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "arrays of visible_function_table are not valid Metal buffer parameters" in (
        generated
    )
    assert (
        "visible_function_table<void(thread CallableData&)> callables [[buffer(1)]]"
        not in (generated)
    )
    assert "callables[0][0](data)" not in generated


def test_metal_arrayed_intersection_function_table_parameter_emits_diagnostic():
    code = """
    shader rt {
        ray_generation {
            void main(
                intersection_function_table<instancing> intersectionFunctions[2]
                    @binding(3)
            ) {
                uint count = intersectionFunctions[0].size();
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "arrays of intersection_function_table are not valid Metal buffer parameters"
        in generated
    )
    assert (
        "intersection_function_table<instancing> intersectionFunctions [[buffer(3)]]"
        not in (generated)
    )
    assert "uint count = 0 /* unsupported Metal ray tracing resource:" in generated
    assert "intersectionFunctions[0].size()" not in generated


def test_ray_intersection_stage_lowers_to_metal_intersection_attribute():
    code = """
    shader rt {
        struct Payload {
            vec3 color;
        };
        ray_intersection {
            bool main(Payload payload @ payload) @triangle {
                payload.color = vec3(1.0, 0.0, 0.0);
                return true;
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "[[intersection(triangle)]] bool "
        "intersection_main(ray_data Payload& payload [[payload]])"
    ) in generated
    assert "payload.color = float3(1.0, 0.0, 0.0);" in generated
    assert "intersection bool" not in generated


def test_ray_intersection_stage_accepts_bounding_box_result_struct():
    code = """
    shader rt {
        struct Payload {
            vec3 color;
        };
        struct BoundsHit {
            bool accept @ accept_intersection;
            float distance @ distance;
        };
        ray_intersection {
            BoundsHit main(Payload payload @ payload) @bounding_box {
                payload.color = vec3(1.0, 0.0, 0.0);
                return BoundsHit { accept: true, distance: 1.0 };
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "bool accept [[accept_intersection]];" in generated
    assert "float distance [[distance]];" in generated
    assert (
        "[[intersection(bounding_box)]] BoundsHit "
        "intersection_main(ray_data Payload& payload [[payload]])"
    ) in generated
    assert "return BoundsHit{true, 1.0};" in generated


def test_mesh_object_stage_codegen():
    code = """
    shader meshpipe {
        object {
            void main() { }
        }
        mesh {
            void main() { }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "[[object]] void object_main" in generated
    assert "[[mesh]] void mesh_main" in generated
    assert "object void object_main" not in generated
    assert "mesh void mesh_main" not in generated


def test_metal_mesh_object_payload_parameters_use_object_data_address_space():
    code = """
    shader meshpipe {
        struct Payload {
            vec4 color;
        };

        object {
            void main(Payload payload @payload)
                @max_total_threads_per_threadgroup(32)
            {
                payload.color = vec4(1.0, 0.0, 0.0, 1.0);
                DispatchMesh(uvec3(1, 1, 1));
            }
        }

        mesh {
            void main(Payload payload @payload)
                @max_total_threads_per_threadgroup(32)
            {
                vec4 color = payload.color;
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "object_data Payload& payload [[payload]]" in generated
    assert "const object_data Payload& payload [[payload]]" in generated
    assert "payload.color = float4(1.0, 0.0, 0.0, 1.0);" in generated
    assert "float4 color = payload.color;" in generated
    assert "Payload payload [[payload]]" not in generated


def test_metal_mesh_object_payload_helper_address_space_and_const_writes():
    code = """
    shader meshpipe {
        struct Payload {
            vec4 color;
        };

        void mutate(thread Payload& localPayload) {
            localPayload.color = vec4(0.5, 0.5, 0.5, 1.0);
        }

        object {
            void main(Payload payload @payload)
                @max_total_threads_per_threadgroup(32)
            {
                mutate(payload);
                payload.color = vec4(1.0, 0.0, 0.0, 1.0);
                DispatchMesh(1, 1, 1);
            }
        }

        mesh {
            void main(Payload payload @payload)
                @max_total_threads_per_threadgroup(32)
            {
                mutate(payload);
                payload.color = vec4(0.0, 1.0, 0.0, 1.0);
                vec4 color = payload.color;
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "object_data Payload& payload [[payload]]" in generated
    assert "const object_data Payload& payload [[payload]]" in generated
    assert (
        generated.count(
            "/* unsupported Metal address-space call: argument 'payload' uses "
            "object_data address space but parameter 'localPayload' of 'mutate' "
            "requires thread */"
        )
        == 2
    )
    assert "payload.color = float4(1.0, 0.0, 0.0, 1.0);" in generated
    assert (
        "/* unsupported Metal mesh payload store: mesh payload 'payload' is const "
        "object_data in mesh stages */"
    ) in generated
    assert "payload.color = float4(0.0, 1.0, 0.0, 1.0);" not in generated
    assert "mutate(payload);" not in generated
    assert "float4 color = payload.color;" in generated


def test_metal_mesh_payload_dispatch_argument_generates_object_data_payload():
    code = """
    shader meshpipe {
        struct MeshPayload {
            uint meshlet;
        };

        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        task {
            void main() @numthreads(1, 1, 1) {
                groupshared MeshPayload payload;
                payload.meshlet = 7u;
                DispatchMesh(1, 1, 1, payload);
            }
        }

        mesh {
            void main(
                @mesh_payload in MeshPayload payload,
                @vertices out MeshVertex verts[1],
                @indices out uint points[1]
            ) @numthreads(1, 1, 1) @outputtopology(point) {
                SetMeshOutputCounts(1, 1);
                verts[0].position = vec4(float(payload.meshlet), 0.0, 0.0, 1.0);
                points[0] = 0u;
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "object_data MeshPayload& _crossglMeshPayload [[payload]], "
        "mesh_grid_properties _crossglMeshGrid"
    ) in generated
    assert "const object_data MeshPayload& payload [[payload]]" in generated
    assert "threadgroup MeshPayload payload;" in generated
    assert "_crossglMeshPayload = payload;" in generated
    assert "_crossglMeshGrid.set_threadgroups_per_grid(uint3(1, 1, 1));" in generated
    assert "MeshPayload payload [[mesh_payload]]" not in generated
    assert "DispatchMesh" not in generated


def test_metal_mesh_payload_dispatch_helpers_forward_object_data_context():
    code = """
    shader meshpipe {
        struct MeshPayload {
            uint meshlet;
        };

        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        task {
            void dispatchOne(threadgroup MeshPayload& payload) {
                DispatchMesh(1, 1, 1, payload);
            }

            void launch(threadgroup MeshPayload& payload) {
                dispatchOne(payload);
            }

            void main() @numthreads(1, 1, 1) {
                groupshared MeshPayload payload;
                payload.meshlet = 7u;
                launch(payload);
            }
        }

        mesh {
            void main(
                @mesh_payload in MeshPayload payload,
                @vertices out MeshVertex verts[1],
                @indices out uint points[1]
            ) @numthreads(1, 1, 1) @outputtopology(point) {
                SetMeshOutputCounts(1, 1);
                verts[0].position = vec4(float(payload.meshlet), 0.0, 0.0, 1.0);
                points[0] = 0u;
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void dispatchOne(threadgroup MeshPayload& payload, "
        "object_data MeshPayload& _crossglMeshPayload, "
        "mesh_grid_properties _crossglMeshGrid)"
    ) in generated
    assert (
        "void launch(threadgroup MeshPayload& payload, "
        "object_data MeshPayload& _crossglMeshPayload, "
        "mesh_grid_properties _crossglMeshGrid)"
    ) in generated
    assert (
        "object_data MeshPayload& _crossglMeshPayload [[payload]], "
        "mesh_grid_properties _crossglMeshGrid"
    ) in generated
    assert "dispatchOne(payload, _crossglMeshPayload, _crossglMeshGrid);" in generated
    assert "launch(payload, _crossglMeshPayload, _crossglMeshGrid);" in generated
    assert "_crossglMeshPayload = payload;" in generated
    assert "_crossglMeshGrid.set_threadgroups_per_grid(uint3(1, 1, 1));" in generated
    assert "unsupported Metal mesh dispatch helper call" not in generated
    assert "unsupported Metal mesh payload dispatch" not in generated
    assert "DispatchMesh" not in generated


def test_metal_mesh_dispatch_helper_calls_receive_payload_and_grid_context():
    code = """
    shader meshpipe {
        struct MeshPayload {
            uint meshlet;
        };

        void issue(threadgroup MeshPayload& payload) {
            DispatchMesh(1, 1, 1, payload);
        }

        void wrapper(threadgroup MeshPayload& payload) {
            issue(payload);
        }

        task {
            void main() @numthreads(1, 1, 1) {
                groupshared MeshPayload payload;
                payload.meshlet = 7u;
                wrapper(payload);
            }
        }

        mesh {
            void main(
                @mesh_payload in MeshPayload payload,
                @vertices out MeshVertex verts[1],
                @indices out uint points[1]
            ) @numthreads(1, 1, 1) @outputtopology(point) {
                SetMeshOutputCounts(1, 1);
            }
        }

        struct MeshVertex {
            vec4 position @ gl_Position;
        };
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void issue(threadgroup MeshPayload& payload, "
        "object_data MeshPayload& _crossglMeshPayload, "
        "mesh_grid_properties _crossglMeshGrid)"
    ) in generated
    assert (
        "void wrapper(threadgroup MeshPayload& payload, "
        "object_data MeshPayload& _crossglMeshPayload, "
        "mesh_grid_properties _crossglMeshGrid)"
    ) in generated
    assert "issue(payload, _crossglMeshPayload, _crossglMeshGrid);" in generated
    assert "wrapper(payload, _crossglMeshPayload, _crossglMeshGrid);" in generated
    assert "_crossglMeshPayload = payload;" in generated
    assert "_crossglMeshGrid.set_threadgroups_per_grid(uint3(1, 1, 1));" in generated
    assert "DispatchMesh" not in generated


def test_metal_mesh_dispatch_helper_call_without_stage_context_is_diagnostic():
    code = """
    shader meshpipe {
        struct MeshPayload {
            uint meshlet;
        };

        void issue(threadgroup MeshPayload& payload) {
            DispatchMesh(1, 1, 1, payload);
        }

        compute {
            void main() {
                groupshared MeshPayload payload;
                issue(payload);
            }
        }

        mesh {
            void main(
                @mesh_payload in MeshPayload payload,
                @vertices out MeshVertex verts[1],
                @indices out uint points[1]
            ) @numthreads(1, 1, 1) @outputtopology(point) {
                SetMeshOutputCounts(1, 1);
            }
        }

        struct MeshVertex {
            vec4 position @ gl_Position;
        };
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "/* unsupported Metal mesh dispatch helper call: function 'issue' "
        "requires an object_data payload context */"
    ) in generated
    assert "issue(payload);" not in generated
    assert "DispatchMesh" not in generated


def test_metal_mesh_payload_dispatch_accepts_threadgroup_array_elements():
    code = """
    shader meshpipe {
        struct MeshPayload {
            uint meshlet;
            uint lane;
        };

        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        task {
            void main() @numthreads(1, 1, 1) {
                groupshared MeshPayload payloads[2];
                groupshared MeshPayload copied;
                payloads[0].meshlet = 3u;
                payloads[0].lane = 4u;
                copied = payloads[0];
                payloads[1] = copied;
                DispatchMesh(1, 1, 1, payloads[1]);
            }
        }

        mesh {
            void main(
                @mesh_payload in MeshPayload payload,
                @vertices out MeshVertex verts[1],
                @indices out uint points[1]
            ) @numthreads(1, 1, 1) @outputtopology(point) {
                SetMeshOutputCounts(1, 1);
                verts[0].position =
                    vec4(float(payload.meshlet + payload.lane), 0.0, 0.0, 1.0);
                points[0] = 0u;
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "threadgroup MeshPayload payloads[2];" in generated
    assert "threadgroup MeshPayload copied;" in generated
    assert "copied = payloads[0];" in generated
    assert "payloads[1] = copied;" in generated
    assert "_crossglMeshPayload = payloads[1];" in generated
    assert "_crossglMeshGrid.set_threadgroups_per_grid(uint3(1, 1, 1));" in generated
    assert "unsupported Metal mesh payload dispatch" not in generated
    assert "DispatchMesh" not in generated


def test_metal_mesh_payload_dispatch_accepts_threadgroup_member_sources():
    code = """
    shader meshpipe {
        struct MeshPayload {
            uint meshlet;
            uint lane;
        };

        struct PayloadBlock {
            MeshPayload payloads[2];
            MeshPayload active;
        };

        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        task {
            void main() @numthreads(1, 1, 1) {
                groupshared PayloadBlock block;
                block.payloads[0].meshlet = 3u;
                block.payloads[0].lane = 4u;
                block.active = block.payloads[0];
                DispatchMesh(1, 1, 1, block.active);
                DispatchMesh(1, 1, 1, block.payloads[0]);
            }
        }

        mesh {
            void main(
                @mesh_payload in MeshPayload payload,
                @vertices out MeshVertex verts[1],
                @indices out uint points[1]
            ) @numthreads(1, 1, 1) @outputtopology(point) {
                SetMeshOutputCounts(1, 1);
                verts[0].position =
                    vec4(float(payload.meshlet + payload.lane), 0.0, 0.0, 1.0);
                points[0] = 0u;
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "threadgroup PayloadBlock block;" in generated
    assert "block.active = block.payloads[0];" in generated
    assert "_crossglMeshPayload = block.active;" in generated
    assert "_crossglMeshPayload = block.payloads[0];" in generated
    assert (
        generated.count("_crossglMeshGrid.set_threadgroups_per_grid(uint3(1, 1, 1));")
        == 2
    )
    assert "unsupported Metal mesh payload dispatch" not in generated
    assert "DispatchMesh" not in generated


def test_metal_mesh_payload_dispatch_helper_accepts_threadgroup_member_sources():
    code = """
    shader meshpipe {
        struct MeshPayload {
            uint meshlet;
        };

        struct PayloadBlock {
            MeshPayload active;
        };

        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        task {
            void issue(threadgroup MeshPayload& payload) {
                DispatchMesh(1, 1, 1, payload);
            }

            void main() @numthreads(1, 1, 1) {
                groupshared PayloadBlock block;
                block.active.meshlet = 7u;
                issue(block.active);
            }
        }

        mesh {
            void main(
                @mesh_payload in MeshPayload payload,
                @vertices out MeshVertex verts[1],
                @indices out uint points[1]
            ) @numthreads(1, 1, 1) @outputtopology(point) {
                SetMeshOutputCounts(1, 1);
                verts[0].position = vec4(float(payload.meshlet), 0.0, 0.0, 1.0);
                points[0] = 0u;
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "void issue(threadgroup MeshPayload& payload, " in generated
    assert "threadgroup PayloadBlock block;" in generated
    assert "issue(block.active, _crossglMeshPayload, _crossglMeshGrid);" in generated
    assert "_crossglMeshPayload = payload;" in generated
    assert "_crossglMeshGrid.set_threadgroups_per_grid(uint3(1, 1, 1));" in generated
    assert "unsupported Metal mesh payload dispatch" not in generated
    assert "DispatchMesh" not in generated


def test_metal_mesh_payload_dispatch_generated_name_avoids_local_collision():
    code = """
    shader meshpipe {
        struct MeshPayload {
            uint meshlet;
        };

        task {
            void main() @numthreads(1, 1, 1) {
                groupshared MeshPayload _crossglMeshPayload;
                _crossglMeshPayload.meshlet = 7u;
                DispatchMesh(1, 1, 1, _crossglMeshPayload);
            }
        }

        mesh {
            void main(MeshPayload payload @mesh_payload)
                @numthreads(1, 1, 1)
                @max_vertices(1)
                @max_primitives(1)
                @outputtopology(point)
            {
                SetMeshOutputCounts(1, 1);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "object_data MeshPayload& _crossglMeshPayload_1 [[payload]]" in generated
    assert "_crossglMeshPayload_1 = _crossglMeshPayload;" in generated
    assert "const object_data MeshPayload& payload [[payload]]" in generated


def test_metal_mesh_payload_dispatch_type_mismatch_is_rejected():
    code = """
    shader meshpipe {
        struct TaskPayload {
            uint meshlet;
        };

        struct MeshPayload {
            uint meshlet;
        };

        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        task {
            void main() @numthreads(1, 1, 1) {
                groupshared TaskPayload payload;
                DispatchMesh(1, 1, 1, payload);
            }
        }

        mesh {
            void main(
                @mesh_payload in MeshPayload payload,
                @vertices out MeshVertex verts[1],
                @indices out uint points[1]
            ) @numthreads(1, 1, 1) @outputtopology(point) {
                SetMeshOutputCounts(1, 1);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="DispatchMesh payload type.*MeshPayload"):
        generate_code(parse_code(tokenize_code(code)))


def test_metal_mesh_payload_dispatch_rejects_invalid_sources():
    code = """
    shader meshpipe {
        struct MeshPayload {
            uint meshlet;
        };

        MeshPayload makePayload() {
            MeshPayload payload;
            payload.meshlet = 1u;
            return payload;
        }

        task {
            void main() @numthreads(1, 1, 1) {
                MeshPayload threadPayload;
                DispatchMesh(1, 1, 1, makePayload());
                DispatchMesh(1, 1, 1, threadPayload);
            }
        }

        mesh {
            void main(MeshPayload payload @mesh_payload)
                @numthreads(1, 1, 1)
                @max_vertices(1)
                @max_primitives(1)
                @outputtopology(point)
            {
                SetMeshOutputCounts(1, 1);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "object_data MeshPayload& _crossglMeshPayload [[payload]]" in generated
    assert (
        "/* unsupported Metal mesh payload dispatch: payload argument must be a "
        "threadgroup lvalue */"
    ) in generated
    assert (
        "/* unsupported Metal mesh payload dispatch: payload argument "
        "'threadPayload' uses thread address space; DispatchMesh payload requires "
        "a threadgroup lvalue */"
    ) in generated
    assert "_crossglMeshPayload = makePayload();" not in generated
    assert "_crossglMeshPayload = threadPayload;" not in generated
    assert generated.count("_crossglMeshGrid.set_threadgroups_per_grid(") == 2
    assert "DispatchMesh(" not in generated


def test_metal_mesh_object_stage_attributes_and_threadgroup_limits():
    code = """
    shader meshpipe {
        object {
            layout(local_size_x = 8, local_size_y = 4, local_size_z = 2) in;
            void main() { }
        }

        mesh {
            void main() @max_total_threads_per_threadgroup(96) { }
        }
    }
    """
    ast = parse_code(tokenize_code(code))
    generated = generate_code(ast)
    mesh_code = MetalCodeGen().generate_stage(ast, "mesh")

    assert (
        "[[object, max_total_threads_per_threadgroup(64)]] void object_main"
        in generated
    )
    assert "[[mesh, max_total_threads_per_threadgroup(96)]] void mesh_main" in generated
    assert "[[mesh, max_total_threads_per_threadgroup(96)]] void mesh_main" in mesh_code
    assert "[[object, max_total_threads_per_threadgroup(64)]]" not in mesh_code


def test_metal_mesh_stage_output_signature_and_counts():
    code = """
    shader meshpipe {
        mesh {
            void main()
                @max_total_threads_per_threadgroup(32)
                @max_vertices(64)
                @max_primitives(32)
                @outputtopology(triangle)
            {
                SetMeshOutputCounts(64, 12);
            }
        }
    }
    """
    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")

    assert "struct _CrossGLMetalMeshVertex_mesh_main" in generated
    assert (
        "mesh<_CrossGLMetalMeshVertex_mesh_main, void, 64, 32, topology::triangle> "
        "_crossglMeshOut"
    ) in generated
    assert "_crossglMeshOut.set_primitive_count(12);" in generated
    assert "SetMeshOutputCounts" not in generated


def test_metal_mesh_stage_vertex_and_index_writes_lower_to_mesh_output_methods():
    code = """
    shader meshpipe {
        mesh {
            void main()
                @max_total_threads_per_threadgroup(32)
                @max_vertices(64)
                @max_primitives(32)
                @outputtopology(triangle)
            {
                vec3 position = vec3(0.0, 0.0, 0.0);
                SetMeshOutputCounts(3, 1);
                SetVertex(0, position);
                SetPrimitive(0, 0);
            }
        }
    }
    """
    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")

    assert (
        "_crossglMeshOut.set_vertex(0, "
        "_CrossGLMetalMeshVertex_mesh_main{float4(position, 1.0)});"
    ) in generated
    assert "_crossglMeshOut.set_index(0, 0);" in generated
    assert "SetVertex" not in generated
    assert "SetPrimitive" not in generated


def test_metal_mesh_stage_vector_line_primitive_writes_use_set_indices():
    code = """
    shader meshpipe {
        mesh {
            void main()
                @max_total_threads_per_threadgroup(32)
                @max_vertices(64)
                @max_primitives(32)
                @outputtopology(line)
            {
                uvec2 edge = uvec2(0, 1);
                SetMeshOutputCounts(2, 1);
                SetPrimitive(0, edge);
            }
        }
    }
    """
    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")

    assert "_crossglMeshOut.set_indices((0) * 2, uchar2(edge));" in generated
    assert "_crossglMeshOut.set_index(0, edge);" not in generated
    assert "SetPrimitive" not in generated


def test_metal_mesh_stage_vector_triangle_primitive_writes_expand_to_indices():
    code = """
    shader meshpipe {
        mesh {
            void main()
                @max_total_threads_per_threadgroup(32)
                @max_vertices(64)
                @max_primitives(32)
                @outputtopology(triangle)
            {
                uvec3 tri = uvec3(0, 1, 2);
                SetMeshOutputCounts(3, 1);
                SetPrimitive(0, tri);
            }
        }
    }
    """
    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")

    assert "_crossglMeshOut.set_index((0) * 3, tri.x);" in generated
    assert "_crossglMeshOut.set_index(((0) * 3) + 1, tri.y);" in generated
    assert "_crossglMeshOut.set_index(((0) * 3) + 2, tri.z);" in generated
    assert "_crossglMeshOut.set_indices((0) * 3" not in generated
    assert "SetPrimitive" not in generated


def test_metal_mesh_stage_primitive_struct_setter_uses_set_primitive():
    code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        struct MeshPrimitive {
            uint layer @ gl_PrimitiveID;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1],
                @primitives out MeshPrimitive prims[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                MeshVertex outVertex;
                outVertex.position = vec4(0.0, 0.0, 0.0, 1.0);
                MeshPrimitive outPrimitive;
                outPrimitive.layer = 7u;
                SetMeshOutputCounts(3, 1);
                SetVertex(0, outVertex);
                SetPrimitive(0, outPrimitive);
                SetIndex(0, uvec3(0u, 1u, 2u));
            }
        }
    }
    """
    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")

    assert (
        "mesh<MeshVertex, MeshPrimitive, 3, 1, topology::triangle> _crossglMeshOut"
        in generated
    )
    assert "_crossglMeshOut.set_vertex(0, outVertex);" in generated
    assert "_crossglMeshOut.set_primitive(0, outPrimitive);" in generated
    assert "_crossglMeshOut.set_index(0, uint3(0u, 1u, 2u).x);" in generated
    assert "_crossglMeshOut.set_index((0) + 1, uint3(0u, 1u, 2u).y);" in generated
    assert "_crossglMeshOut.set_index((0) + 2, uint3(0u, 1u, 2u).z);" in generated
    assert "_crossglMeshOut.set_index(0, outPrimitive);" not in generated
    assert "SetPrimitive" not in generated
    assert "SetIndex" not in generated


def test_metal_mesh_output_signature_roles_lower_to_mesh_output_parameter():
    code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        struct MeshPrimitive {
            uint layer @ gl_PrimitiveID;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1],
                @primitives out MeshPrimitive prims[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                MeshVertex outVertex;
                outVertex.position = vec4(0.0, 0.0, 0.0, 1.0);
                outVertex.uv = vec2(0.0);
                MeshPrimitive outPrimitive;
                outPrimitive.layer = 0u;

                SetMeshOutputCounts(3, 1);
                verts[0] = outVertex;
                tris[0] = uvec3(0u, 1u, 2u);
                prims[0] = outPrimitive;
            }
        }
    }
    """
    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")

    assert (
        "mesh<MeshVertex, MeshPrimitive, 3, 1, topology::triangle> _crossglMeshOut"
        in generated
    )
    assert "MeshVertex verts[3]" not in generated
    assert "uint3 tris[1]" not in generated
    assert "MeshPrimitive prims[1]" not in generated
    assert "_crossglMeshOut.set_vertex(0, outVertex);" in generated
    assert "_crossglMeshOut.set_index((0) * 3, uint3(0u, 1u, 2u).x);" in generated
    assert "_crossglMeshOut.set_index(((0) * 3) + 1, uint3(0u, 1u, 2u).y);" in generated
    assert "_crossglMeshOut.set_index(((0) * 3) + 2, uint3(0u, 1u, 2u).z);" in generated
    assert "_crossglMeshOut.set_primitive(0, outPrimitive);" in generated
    assert "[[vertices]]" not in generated
    assert "[[indices]]" not in generated
    assert "[[primitives]]" not in generated


def test_metal_mesh_output_signature_single_member_writes_lower_to_setters():
    code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """
    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")

    assert (
        "_crossglMeshOut.set_vertex(" "0, MeshVertex{float4(0.0, 0.0, 0.0, 1.0)});"
    ) in generated
    assert "_crossglMeshOut.set_index((0) * 3, uint3(0u, 1u, 2u).x);" in generated
    assert "verts[0].position" not in generated
    assert "tris[0]" not in generated


def test_metal_mesh_output_signature_multi_member_partial_writes_use_accumulator():
    code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                verts[0].uv = vec2(0.5, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """
    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")

    assert "MeshVertex _crossglMeshVertices_verts_i_0 = {};" in generated
    assert (
        "_crossglMeshVertices_verts_i_0.position = " "float4(0.0, 0.0, 0.0, 1.0);"
    ) in generated
    assert "_crossglMeshOut.set_vertex(0, _crossglMeshVertices_verts_i_0);" in generated
    assert "_crossglMeshVertices_verts_i_0.uv = float2(0.5, 1.0);" in generated
    assert (
        "unsupported Metal mesh output assignment: vertices output 'verts'"
        not in generated
    )
    assert "verts[0].position" not in generated
    assert "verts[0].uv" not in generated
    assert "_crossglMeshOut.set_index((0) * 3, uint3(0u, 1u, 2u).x);" in generated


def test_metal_mesh_output_variable_member_writes_use_indexed_accumulators():
    code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        struct MeshPrimitive {
            uint layer @ gl_PrimitiveID;
            vec2 bary @ TEXCOORD1;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[4],
                @indices out uvec3 tris[2],
                @primitives out MeshPrimitive prims[2]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                uint vertexIndex = 1u;
                uint primitiveIndex = 0u;
                SetMeshOutputCounts(4, 2);
                verts[vertexIndex].position = vec4(1.0, 0.0, 0.0, 1.0);
                verts[vertexIndex].uv = vec2(0.5, 1.0);
                tris[primitiveIndex] = uvec3(0u, 1u, 2u);
                prims[primitiveIndex].layer = 2u;
                prims[primitiveIndex].bary = vec2(0.25, 0.75);
            }
        }
    }
    """
    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")

    assert (
        "mesh<MeshVertex, MeshPrimitive, 4, 2, topology::triangle> _crossglMeshOut"
        in generated
    )
    assert "MeshVertex _crossglMeshVertices_verts_vertexIndex = {};" in generated
    assert (
        "MeshPrimitive _crossglMeshPrimitives_prims_primitiveIndex = {};" in generated
    )
    assert (
        "_crossglMeshVertices_verts_vertexIndex.position = "
        "float4(1.0, 0.0, 0.0, 1.0);"
    ) in generated
    assert (
        "_crossglMeshOut.set_vertex(vertexIndex, "
        "_crossglMeshVertices_verts_vertexIndex);"
    ) in generated
    assert "_crossglMeshVertices_verts_vertexIndex.uv = float2(0.5, 1.0);" in (
        generated
    )
    assert "_crossglMeshOut.set_index((primitiveIndex) * 3, uint3(0u, 1u, 2u).x);" in (
        generated
    )
    assert (
        "_crossglMeshOut.set_index(((primitiveIndex) * 3) + 1, " "uint3(0u, 1u, 2u).y);"
    ) in generated
    assert (
        "_crossglMeshOut.set_index(((primitiveIndex) * 3) + 2, " "uint3(0u, 1u, 2u).z);"
    ) in generated
    assert "_crossglMeshPrimitives_prims_primitiveIndex.layer = 2u;" in generated
    assert (
        "_crossglMeshOut.set_primitive(primitiveIndex, "
        "_crossglMeshPrimitives_prims_primitiveIndex);"
    ) in generated
    assert (
        "_crossglMeshPrimitives_prims_primitiveIndex.bary = float2(0.25, 0.75);"
        in generated
    )
    assert "verts[vertexIndex]" not in generated
    assert "tris[primitiveIndex]" not in generated
    assert "prims[primitiveIndex]" not in generated
    assert "unsupported Metal mesh output assignment" not in generated


def test_metal_mesh_output_writes_validate_order_and_literal_bounds():
    write_before_count_code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                SetMeshOutputCounts(3, 1);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="verts.*after SetMeshOutputCounts"):
        MetalCodeGen().generate_stage(
            parse_code(tokenize_code(write_before_count_code)), "mesh"
        )

    declared_bound_code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
                verts[3].position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="verts.*declared array size"):
        MetalCodeGen().generate_stage(
            parse_code(tokenize_code(declared_bound_code)), "mesh"
        )

    active_count_bound_code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(2, 1);
                verts[2].position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="verts.*numVertices"):
        MetalCodeGen().generate_stage(
            parse_code(tokenize_code(active_count_bound_code)), "mesh"
        )


def test_metal_mesh_set_output_counts_validates_literal_bounds():
    code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(4, 1);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="numVertices.*declared output count"):
        MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")


def test_metal_mesh_set_output_counts_branch_dominance_allows_post_branch_writes():
    code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                if (true) {
                    SetMeshOutputCounts(3, 1);
                } else {
                    SetMeshOutputCounts(3, 1);
                }
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """

    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")

    assert generated.count("_crossglMeshOut.set_primitive_count(1);") == 2
    assert (
        "_crossglMeshOut.set_vertex(0, MeshVertex{float4(0.0, 0.0, 0.0, 1.0)});"
        in generated
    )
    assert "_crossglMeshOut.set_index((0) * 3, uint3(0u, 1u, 2u).x);" in generated


def test_metal_mesh_set_output_counts_rejects_non_dominating_branch_counts():
    missing_else_code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                if (true) {
                    SetMeshOutputCounts(3, 1);
                }
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="verts.*after SetMeshOutputCounts"):
        MetalCodeGen().generate_stage(
            parse_code(tokenize_code(missing_else_code)), "mesh"
        )

    divergent_counts_code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                if (true) {
                    SetMeshOutputCounts(2, 1);
                } else {
                    SetMeshOutputCounts(3, 1);
                }
                verts[2].position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="verts.*numVertices"):
        MetalCodeGen().generate_stage(
            parse_code(tokenize_code(divergent_counts_code)), "mesh"
        )


def test_metal_mesh_set_output_counts_switch_dominance_handles_breaks():
    code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                switch (0) {
                    case 0:
                        SetMeshOutputCounts(3, 1);
                        break;
                    default:
                        SetMeshOutputCounts(3, 1);
                        break;
                }
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """

    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")

    assert generated.count("_crossglMeshOut.set_primitive_count(1);") == 2
    assert (
        "_crossglMeshOut.set_vertex(0, MeshVertex{float4(0.0, 0.0, 0.0, 1.0)});"
        in generated
    )


def test_metal_mesh_set_output_counts_rejects_non_exhaustive_switch_counts():
    code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                switch (0) {
                    case 0:
                        SetMeshOutputCounts(3, 1);
                        break;
                }
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="verts.*after SetMeshOutputCounts"):
        MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")


def test_metal_mesh_set_output_counts_loop_break_dominance_allows_post_loop_writes():
    code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                loop {
                    SetMeshOutputCounts(3, 1);
                    break;
                }
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """

    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")

    assert "while (true)" in generated
    assert "_crossglMeshOut.set_primitive_count(1);" in generated
    assert (
        "_crossglMeshOut.set_vertex(0, MeshVertex{float4(0.0, 0.0, 0.0, 1.0)});"
        in generated
    )


def test_metal_mesh_set_output_counts_while_true_break_dominance_allows_post_loop_writes():
    code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                while (true) {
                    SetMeshOutputCounts(3, 1);
                    break;
                }
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """

    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")

    assert "while (true)" in generated
    assert (
        "_crossglMeshOut.set_vertex(0, MeshVertex{float4(0.0, 0.0, 0.0, 1.0)});"
        in generated
    )


def test_metal_mesh_set_output_counts_do_while_false_dominates_post_loop_writes():
    code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                do {
                    SetMeshOutputCounts(3, 1);
                } while (false);
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """

    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")

    assert "do {" in generated
    assert "_crossglMeshOut.set_primitive_count(1);" in generated
    assert (
        "_crossglMeshOut.set_vertex(0, MeshVertex{float4(0.0, 0.0, 0.0, 1.0)});"
        in generated
    )


def test_metal_mesh_set_output_counts_rejects_skippable_loop_counts():
    code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                int i = 0;
                while (i < 1) {
                    SetMeshOutputCounts(3, 1);
                    break;
                }
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="verts.*after SetMeshOutputCounts"):
        MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")


def test_metal_mesh_set_output_counts_rejects_uncounted_loop_break_path():
    code = """
    shader meshpipe {
        struct MeshVertex {
            vec4 position @ gl_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                loop {
                    if (true) {
                        break;
                    }
                    SetMeshOutputCounts(3, 1);
                    break;
                }
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="verts.*after SetMeshOutputCounts"):
        MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")


def test_metal_mesh_stage_output_signature_avoids_generated_name_collisions():
    code = """
    shader meshpipe {
        struct _CrossGLMetalMeshVertex_mesh_main {
            vec4 color;
        };

        mesh {
            void main(int _crossglMeshOut)
                @max_total_threads_per_threadgroup(32)
                @max_vertices(64)
                @max_primitives(32)
                @outputtopology(triangle)
            {
                SetMeshOutputCounts(64, 12);
            }
        }
    }
    """
    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "mesh")

    assert "struct _CrossGLMetalMeshVertex_mesh_main_1" in generated
    assert "_crossglMeshOut_1.set_primitive_count(12);" in generated
    assert "int _crossglMeshOut, mesh<" in generated
    assert "topology::triangle> _crossglMeshOut_1" in generated


def test_metal_object_stage_dispatch_mesh_sets_grid_properties():
    code = """
    shader meshpipe {
        object {
            void main() @max_total_threads_per_threadgroup(32) {
                DispatchMesh(2, 3, 4);
            }
        }
    }
    """
    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "object")

    assert "mesh_grid_properties _crossglMeshGrid" in generated
    assert "_crossglMeshGrid.set_threadgroups_per_grid(uint3(2, 3, 4));" in generated
    assert "DispatchMesh" not in generated


def test_metal_object_stage_dispatch_mesh_grid_name_avoids_parameter_collision():
    code = """
    shader meshpipe {
        object {
            void main(int _crossglMeshGrid) @max_total_threads_per_threadgroup(32) {
                DispatchMesh(2, 3, 4);
            }
        }
    }
    """
    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "object")

    assert "int _crossglMeshGrid, mesh_grid_properties _crossglMeshGrid_1" in generated
    assert "_crossglMeshGrid_1.set_threadgroups_per_grid(uint3(2, 3, 4));" in generated


def test_metal_object_stage_dispatch_mesh_accepts_vector_grid():
    code = """
    shader meshpipe {
        object {
            void main() @max_total_threads_per_threadgroup(32) {
                DispatchMesh(uvec3(2, 3, 4));
            }
        }
    }
    """
    generated = MetalCodeGen().generate_stage(parse_code(tokenize_code(code)), "object")

    assert "mesh_grid_properties _crossglMeshGrid" in generated
    assert "_crossglMeshGrid.set_threadgroups_per_grid(uint3(2, 3, 4));" in generated
    assert "DispatchMesh" not in generated


def test_metal_rejects_unsupported_geometry_and_tessellation_stages():
    geometry_code = """
    shader geometry_stage {
        geometry {
            void main() { }
        }
    }
    """
    tessellation_code = """
    shader tessellation_stage {
        tessellation_control {
            void main() { }
        }

        tessellation_evaluation {
            void main() { }
        }
    }
    """

    with pytest.raises(ValueError, match="geometry"):
        MetalCodeGen().generate_stage(
            crosstl.translator.parse(geometry_code), "geometry"
        )

    with pytest.raises(ValueError, match="tessellation_control"):
        MetalCodeGen().generate(crosstl.translator.parse(tessellation_code))


def test_anyhit_stage_codegen():
    code = """
    shader rt {
        anyhit {
            void main() { }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "[[visible]] void anyhit_main" in generated
    assert "anyhit void anyhit_main" not in generated


def test_generate_stage_filters_combined_vertex_fragment_units():
    code = """
    shader combined {
        struct VSInput {
            vec3 position @ POSITION;
            vec2 uv @ TEXCOORD0;
        };

        struct VSOutput {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        float adjust(float value) {
            return value + 1.0;
        }

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
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generator = MetalCodeGen()

    vertex_code = generator.generate_stage(ast, "vertex")
    fragment_code = generator.generate_stage(ast, "fragment")

    assert "float adjust(float value)" in vertex_code
    assert "float adjust(float value)" in fragment_code
    assert "vertex VSOutput vertex_main" in vertex_code
    assert "fragment float4 fragment_main" not in vertex_code
    assert "fragment float4 fragment_main" in fragment_code
    assert "vertex VSOutput vertex_main" not in fragment_code


def test_metal_stage_local_helpers_order_by_dependencies_before_entrypoint():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    second_index = generated_code.index("float4 second(float2 uv)")
    first_index = generated_code.index("float4 first(float2 uv)")
    entry_index = generated_code.index("fragment float4 fragment_main")

    assert second_index < first_index < entry_index
    assert "return second(uv);" in generated_code
    assert "return first(uv);" in generated_code


def test_metal_stage_entry_name_avoids_global_helper_collision():
    shader = """
    shader StageEntryGlobalHelperCollision {
        vec4 fragment_main(vec2 uv) {
            return vec4(uv, 0.0, 1.0);
        }

        fragment {
            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return vec4(uv, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "float4 fragment_main(float2 uv)" in generated_code
    assert (
        "fragment float4 fragment_main_2(float2 uv [[attribute(5)]])" in generated_code
    )


def test_metal_stage_entry_name_avoids_local_helper_collision():
    shader = """
    shader StageEntryLocalHelperCollision {
        fragment {
            vec4 fragment_main(vec2 uv) {
                return vec4(uv, 0.0, 1.0);
            }

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return fragment_main(uv);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "float4 fragment_main(float2 uv)" in generated_code
    assert (
        "fragment float4 fragment_main_2(float2 uv [[attribute(5)]])" in generated_code
    )
    assert "return fragment_main(uv);" in generated_code


def test_metal_atomic_fetch_codegen():
    code = """
    shader main {
        compute {
            void main() {
                atomic_int counter;
                int old = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "threadgroup atomic_int counter;" in generated
    assert (
        "int old = atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);"
        in generated
    )
    assert "atomic_fetch_add_explicit(counter" not in generated


def test_metal_atomic_local_initializers_use_atomic_store():
    code = """
    shader main {
        compute {
            void main() {
                atomic_uint counter = 0;
                uint old = atomic_fetch_add_explicit(
                    counter,
                    1u,
                    memory_order_relaxed
                );
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "threadgroup atomic_uint counter;" in generated
    assert "atomic_store_explicit(&counter, 0, memory_order_relaxed);" in generated
    assert (
        "uint old = atomic_fetch_add_explicit(&counter, 1u, memory_order_relaxed);"
        in generated
    )
    assert "threadgroup atomic_uint counter = 0;" not in generated


def test_metal_threadgroup_atomic_array_elements_use_address_arguments():
    code = """
    shader main {
        uint bump(threadgroup atomic_uint counters[64], uint index) {
            return atomic_fetch_add_explicit(
                counters[index],
                1u,
                memory_order_relaxed
            );
        }

        compute {
            void main() {
                shared atomic_uint counters[64];
                uint index = gl_LocalInvocationIndex;
                atomic_store_explicit(
                    counters[index],
                    0u,
                    memory_order_relaxed
                );
                barrier();
                uint oldValue = bump(counters, index);
                groupMemoryBarrier();
                uint currentValue = atomic_load_explicit(
                    counters[index],
                    memory_order_relaxed
                );
                atomic_exchange_explicit(
                    counters[index],
                    oldValue + currentValue,
                    memory_order_relaxed
                );
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "uint bump(threadgroup atomic_uint counters[64], uint index)" in generated
    assert "threadgroup atomic_uint counters[64];" in generated
    assert (
        "return atomic_fetch_add_explicit(&counters[index], 1u, memory_order_relaxed);"
        in generated
    )
    assert (
        "atomic_store_explicit(&counters[index], 0u, memory_order_relaxed);"
        in generated
    )
    assert (
        "uint currentValue = atomic_load_explicit(&counters[index], memory_order_relaxed);"
        in generated
    )
    assert (
        "atomic_exchange_explicit(&counters[index], oldValue + currentValue, memory_order_relaxed);"
        in generated
    )
    assert generated.count("threadgroup_barrier(mem_flags::mem_threadgroup);") == 2
    assert "atomic_fetch_add_explicit(counters[index]" not in generated
    assert "atomic_store_explicit(counters[index]" not in generated
    assert "atomic_load_explicit(counters[index]" not in generated
    assert "atomic_exchange_explicit(counters[index]" not in generated


def test_metal_atomic_pointer_targets_cover_signed_and_bool_operations():
    code = """
    shader MetalAtomicPointerValidation {
        int bumpDevice(device atomic_int* counters, uint index, int delta) {
            return atomic_fetch_add_explicit(
                counters + index,
                delta,
                memory_order_relaxed
            );
        }

        int reduceThreadgroup(
            threadgroup atomic_int* counters,
            uint index,
            int value
        ) {
            return atomic_fetch_min_explicit(
                counters[index],
                value,
                memory_order_relaxed
            );
        }

        bool exchangeFlag(device atomic_bool* flags, uint index, bool value) {
            return atomic_exchange_explicit(
                flags + index,
                value,
                memory_order_relaxed
            );
        }

        compute {
            void main(
                device atomic_int* counters @buffer(0),
                device atomic_bool* flags @buffer(1),
                uint index @gl_LocalInvocationIndex
            ) {
                shared atomic_int scratch[64];
                atomic_store_explicit(scratch[index], 7, memory_order_relaxed);
                atomic_store_explicit(counters[index], 0, memory_order_relaxed);
                int oldDevice = bumpDevice(counters, index, 1);
                int oldScratch = reduceThreadgroup(scratch, index, oldDevice);
                int loaded = atomic_load_explicit(
                    counters + index,
                    memory_order_relaxed
                );
                bool wasSet = exchangeFlag(flags, index, true);
                bool isSet = atomic_load_explicit(
                    flags + index,
                    memory_order_relaxed
                );
                atomic_store_explicit(
                    flags[index],
                    wasSet && isSet,
                    memory_order_relaxed
                );
                atomic_store_explicit(
                    counters[index],
                    oldScratch + loaded,
                    memory_order_relaxed
                );
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "int bumpDevice(device atomic_int* counters, uint index, int delta)"
        in generated
    )
    assert (
        "return atomic_fetch_add_explicit(counters + index, delta, memory_order_relaxed);"
        in generated
    )
    assert (
        "return atomic_fetch_min_explicit(&counters[index], value, memory_order_relaxed);"
        in generated
    )
    assert (
        "return atomic_exchange_explicit(flags + index, value, memory_order_relaxed);"
        in generated
    )
    assert "threadgroup atomic_int scratch[64];" in generated
    assert (
        "atomic_store_explicit(&scratch[index], 7, memory_order_relaxed);" in generated
    )
    assert (
        "atomic_store_explicit(&counters[index], 0, memory_order_relaxed);" in generated
    )
    assert (
        "int loaded = atomic_load_explicit(counters + index, memory_order_relaxed);"
        in generated
    )
    assert (
        "atomic_store_explicit(&flags[index], wasSet && isSet, memory_order_relaxed);"
        in generated
    )
    assert (
        "atomic_store_explicit(&counters[index], oldScratch + loaded, memory_order_relaxed);"
        in generated
    )
    assert "atomic_fetch_add_explicit(&counters + index" not in generated
    assert "atomic_exchange_explicit(&flags + index" not in generated
    assert "atomic_store_explicit(counters[index]" not in generated
    assert "atomic_store_explicit(flags[index]" not in generated


def test_metal_atomic_compare_exchange_addresses_targets_and_expected_values():
    code = """
    shader main {
        bool claim(
            threadgroup atomic_uint counters[64],
            thread uint expectedValues[64],
            uint index,
            uint desired
        ) {
            return atomic_compare_exchange_weak_explicit(
                counters[index],
                expectedValues[index],
                desired,
                memory_order_relaxed,
                memory_order_relaxed
            );
        }

        bool claimPtr(
            threadgroup atomic_uint* counters,
            thread uint* expectedValues,
            uint index,
            uint desired
        ) {
            return atomic_compare_exchange_strong_explicit(
                counters[index],
                expectedValues[index],
                desired,
                memory_order_relaxed,
                memory_order_relaxed
            );
        }

        bool claimRaw(
            threadgroup atomic_uint* counter,
            thread uint* expected,
            uint desired
        ) {
            return atomic_compare_exchange_weak_explicit(
                counter,
                expected,
                desired,
                memory_order_relaxed,
                memory_order_relaxed
            );
        }

        compute {
            void main() {
                shared atomic_uint counters[64];
                uint expectedValues[64];
                uint index = gl_LocalInvocationIndex;
                expectedValues[index] = 0u;
                atomic_store_explicit(counters[index], 0u, memory_order_relaxed);
                bool helperClaimed = claim(counters, expectedValues, index, 1u);
                bool pointerClaimed = claimPtr(counters, expectedValues, index, 2u);
                uint expected = helperClaimed ? 1u : 0u;
                bool claimed = atomic_compare_exchange_strong_explicit(
                    counters[index],
                    expected,
                    2u,
                    memory_order_relaxed,
                    memory_order_relaxed
                );
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert "bool claim(threadgroup atomic_uint counters[64]" in generated
    assert "bool claimPtr(threadgroup atomic_uint* counters" in generated
    indexed_compare_exchange = (
        "return atomic_compare_exchange_weak_explicit(&counters[index], "
        "&expectedValues[index], desired, memory_order_relaxed, "
        "memory_order_relaxed);"
    )
    assert generated.count(indexed_compare_exchange) == 2
    assert (
        "return atomic_compare_exchange_weak_explicit(counter, expected, "
        "desired, memory_order_relaxed, memory_order_relaxed);"
    ) in generated
    assert (
        "bool claimed = atomic_compare_exchange_weak_explicit(&counters[index], "
        "&expected, 2u, memory_order_relaxed, memory_order_relaxed);"
    ) in generated
    assert (
        "atomic_compare_exchange_weak_explicit(&counters[index], expectedValues[index]"
        not in generated
    )
    assert (
        "atomic_compare_exchange_weak_explicit(&counters[index], expected,"
        not in generated
    )
    assert "atomic_compare_exchange_weak_explicit(&counter, &expected," not in generated
    assert "atomic_compare_exchange_strong_explicit" not in generated


def test_metal_device_atomic_compare_exchange_preserves_expected_address_space():
    code = """
    shader MetalDeviceAtomicCompareExchangeValidation {
        bool claimDevice(
            device atomic_uint* counters,
            thread uint* expectedValues,
            uint index,
            uint desired
        ) {
            return atomic_compare_exchange_strong_explicit(
                counters + index,
                expectedValues + index,
                desired,
                memory_order_relaxed,
                memory_order_relaxed
            );
        }

        bool rejectDeviceExpected(
            device atomic_uint* counters,
            device uint* expectedValues,
            uint index,
            uint desired
        ) {
            return atomic_compare_exchange_weak_explicit(
                counters + index,
                expectedValues + index,
                desired,
                memory_order_relaxed,
                memory_order_relaxed
            );
        }

        compute {
            void main(
                device atomic_uint* counters @buffer(0),
                device uint* deviceExpected @buffer(1),
                uint index @gl_LocalInvocationIndex
            ) {
                uint expectedValues[64];
                expectedValues[index] = 0u;
                bool pointerClaimed = claimDevice(
                    counters,
                    expectedValues,
                    index,
                    1u
                );
                bool rejected = rejectDeviceExpected(
                    counters,
                    deviceExpected,
                    index,
                    2u
                );
                uint expected = pointerClaimed ? 1u : 0u;
                bool directClaimed = atomic_compare_exchange_strong_explicit(
                    counters[index],
                    expected,
                    3u,
                    memory_order_relaxed,
                    memory_order_relaxed
                );
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "bool claimDevice(device atomic_uint* counters, thread uint* expectedValues, "
        "uint index, uint desired)"
    ) in generated
    assert (
        "return atomic_compare_exchange_weak_explicit(counters + index, "
        "expectedValues + index, desired, memory_order_relaxed, "
        "memory_order_relaxed);"
    ) in generated
    assert (
        "return false /* unsupported Metal atomic compare-exchange expected pointer: "
        "expected storage 'expectedValues' uses device address space; Metal requires "
        "thread storage */;"
    ) in generated
    assert (
        "bool directClaimed = atomic_compare_exchange_weak_explicit(&counters[index], "
        "&expected, 3u, memory_order_relaxed, memory_order_relaxed);"
    ) in generated
    assert (
        "atomic_compare_exchange_weak_explicit(counters + index, deviceExpected"
        not in generated
    )
    assert (
        "atomic_compare_exchange_weak_explicit(counters + index, expectedValues + index"
        in generated
    )
    assert "atomic_compare_exchange_strong_explicit" not in generated


def test_compute_builtin_semantics_roundtrip():
    code = """
    shader cs {
        compute {
            void main(uvec3 gid @ gl_GlobalInvocationID,
                      uvec3 lid @ gl_LocalInvocationID,
                      uvec3 group @ gl_WorkGroupID,
                      uint idx @ gl_LocalInvocationIndex,
                      uvec3 size @ gl_WorkGroupSize,
                      uvec3 groups @ gl_NumWorkGroups) { }
        }
    }
    """
    ast = crosstl.translator.parse(code)
    generated = MetalCodeGen().generate(ast)
    for expected in [
        "uint3 gid [[thread_position_in_grid]]",
        "uint3 lid [[thread_position_in_threadgroup]]",
        "uint3 group [[threadgroup_position_in_grid]]",
        "uint idx [[thread_index_in_threadgroup]]",
        "uint3 size [[threads_per_threadgroup]]",
        "uint3 groups [[threadgroups_per_grid]]",
    ]:
        assert expected in generated


@pytest.mark.parametrize(
    ("param_type", "semantic", "metal_semantic", "expected_type"),
    [
        ("float", "gl_GlobalInvocationID", "thread_position_in_grid", "uint3"),
        ("uvec2", "gl_LocalInvocationID", "thread_position_in_threadgroup", "uint3"),
        ("int", "gl_LocalInvocationIndex", "thread_index_in_threadgroup", "uint"),
        ("vec3", "gl_NumWorkGroups", "threadgroups_per_grid", "uint3"),
    ],
)
def test_compute_builtin_parameter_types_are_validated(
    param_type, semantic, metal_semantic, expected_type
):
    code = f"""
    shader cs {{
        compute {{
            void main({param_type} value @ {semantic}) {{ }}
        }}
    }}
    """
    ast = crosstl.translator.parse(code)
    with pytest.raises(ValueError, match=f"{metal_semantic}.*{expected_type}"):
        MetalCodeGen().generate_stage(ast, "compute")


def test_compute_direct_builtin_references_inject_metal_parameters():
    code = """
    shader cs {
        compute {
            void main() {
                uint gx = gl_GlobalInvocationID.x;
                uint lx = gl_LocalInvocationID.x;
                uint group = gl_WorkGroupID.x;
                uint index = gl_LocalInvocationIndex;
                uint size = gl_WorkGroupSize.x;
                uint groups = gl_NumWorkGroups.x;
            }
        }
    }
    """
    ast = crosstl.translator.parse(code)
    generated = MetalCodeGen().generate_stage(ast, "compute")

    assert "uint3 gl_GlobalInvocationID [[thread_position_in_grid]]" in generated
    assert "uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]]" in generated
    assert "uint3 gl_WorkGroupID [[threadgroup_position_in_grid]]" in generated
    assert "uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]]" in generated
    assert "uint3 gl_WorkGroupSize [[threads_per_threadgroup]]" in generated
    assert "uint3 gl_NumWorkGroups [[threadgroups_per_grid]]" in generated
    assert "uint gx = gl_GlobalInvocationID.x;" in generated


def test_graphics_builtin_parameter_semantics_roundtrip():
    code = """
    shader GraphicsBuiltinMetal {
        struct VSInput {
            vec3 position @ POSITION;
        };

        struct VSOutput {
            vec4 position @ gl_Position;
        };

        struct FSOutput {
            vec4 color @ gl_FragColor;
        };

        vertex {
            VSOutput main(
                VSInput input,
                uint vertexID @ gl_VertexID,
                uint instanceID @ gl_InstanceID
            ) {
                VSOutput output;
                output.position = vec4(input.position, 1.0);
                output.position.x = output.position.x + float(vertexID + instanceID);
                return output;
            }
        }

        fragment {
            FSOutput main(
                VSOutput input,
                uint primitiveID @ gl_PrimitiveID,
                bool frontFacing @ gl_FrontFacing,
                vec4 coord @ gl_FragCoord,
                vec2 point @ gl_PointCoord
            ) {
                FSOutput output;
                output.color = coord + vec4(point, float(primitiveID), 1.0);
                if (frontFacing) {
                    output.color.x = output.color.x + 1.0;
                }
                return output;
            }
        }
    }
    """
    ast = crosstl.translator.parse(code)

    vertex_code = MetalCodeGen().generate_stage(ast, "vertex")
    assert "uint vertexID [[vertex_id]]" in vertex_code
    assert "uint instanceID [[instance_id]]" in vertex_code

    fragment_code = MetalCodeGen().generate_stage(ast, "fragment")
    assert "uint primitiveID [[primitive_id]]" in fragment_code
    assert "bool frontFacing [[is_front_facing]]" in fragment_code
    assert "float4 coord [[position]]" in fragment_code
    assert "float2 point [[point_coord]]" in fragment_code


@pytest.mark.parametrize(
    ("stage", "param_type", "semantic", "metal_semantic", "expected_type"),
    [
        ("vertex", "int", "gl_VertexID", "vertex_id", "uint"),
        ("vertex", "float", "gl_InstanceID", "instance_id", "uint"),
        ("fragment", "int", "gl_PrimitiveID", "primitive_id", "uint"),
        ("fragment", "int", "gl_FrontFacing", "is_front_facing", "bool"),
        ("fragment", "vec3", "gl_FragCoord", "position", "float4"),
        ("fragment", "vec3", "gl_PointCoord", "point_coord", "float2"),
    ],
)
def test_graphics_builtin_parameter_types_are_validated(
    stage, param_type, semantic, metal_semantic, expected_type
):
    output_struct = (
        "struct VSOutput { vec4 position @ gl_Position; };"
        if stage == "vertex"
        else "struct FSOutput { vec4 color @ gl_FragColor; };"
    )
    return_type = "VSOutput" if stage == "vertex" else "FSOutput"
    body = (
        "VSOutput output; output.position = vec4(0.0); return output;"
        if stage == "vertex"
        else "FSOutput output; output.color = vec4(0.0); return output;"
    )
    code = f"""
    shader BadGraphicsBuiltinType {{
        {output_struct}
        {stage} {{
            {return_type} main({param_type} value @ {semantic}) {{
                {body}
            }}
        }}
    }}
    """
    ast = crosstl.translator.parse(code)
    with pytest.raises(ValueError, match=f"{metal_semantic}.*{expected_type}"):
        MetalCodeGen().generate_stage(ast, stage)


def test_struct_builtin_member_semantics_are_validated():
    valid_code = """
    shader ValidStructBuiltins {
        struct VSOutput {
            vec4 position @ gl_Position;
            float pointSize @ gl_PointSize;
        };

        struct FSInput {
            vec4 coord @ gl_FragCoord;
            vec2 point @ gl_PointCoord;
            bool frontFacing @ gl_FrontFacing;
            uint primitiveID @ gl_PrimitiveID;
        };

        struct FSOutput {
            vec4 color @ gl_FragColor;
            float depth @ gl_FragDepth;
        };

        vertex {
            VSOutput main() {
                VSOutput output;
                output.position = vec4(0.0);
                output.pointSize = 1.0;
                return output;
            }
        }

        fragment {
            FSOutput main(FSInput input) {
                FSOutput output;
                output.color = input.coord + vec4(input.point, float(input.primitiveID), 1.0);
                output.depth = 0.5;
                return output;
            }
        }
    }
    """
    generated = MetalCodeGen().generate(crosstl.translator.parse(valid_code))
    assert "float4 position [[position]];" in generated
    assert "float pointSize [[point_size]];" in generated
    assert "float4 coord [[position]];" in generated
    assert "float2 point [[point_coord]];" in generated
    assert "bool frontFacing [[is_front_facing]];" in generated
    assert "uint primitiveID [[primitive_id]];" in generated
    assert "float4 color [[color(0)]];" in generated
    assert "float depth [[depth(any)]];" in generated


@pytest.mark.parametrize(
    ("member_decl", "metal_semantic", "expected_type"),
    [
        ("vec3 position @ gl_Position", "position", "float4"),
        ("vec3 color @ gl_FragColor", "color\\(0\\)", "float4"),
        ("vec2 depth @ gl_FragDepth", "depth\\(any\\)", "float"),
        ("int frontFacing @ gl_FrontFacing", "is_front_facing", "bool"),
        ("vec3 point @ gl_PointCoord", "point_coord", "float2"),
        ("int primitiveID @ gl_PrimitiveID", "primitive_id", "uint"),
        ("vec2 pointSize @ gl_PointSize", "point_size", "float"),
    ],
)
def test_struct_builtin_member_types_are_validated(
    member_decl, metal_semantic, expected_type
):
    code = f"""
    shader BadStructBuiltin {{
        struct BadIO {{
            {member_decl};
        }};

        fragment {{
            vec4 main() @ gl_FragColor {{
                return vec4(0.0);
            }}
        }}
    }}
    """
    ast = crosstl.translator.parse(code)
    with pytest.raises(ValueError, match=f"{metal_semantic}.*{expected_type}"):
        MetalCodeGen().generate(ast)


def test_function_return_builtin_semantics_are_validated():
    valid_code = """
    shader ValidReturnBuiltins {
        vertex {
            vec4 main() @ gl_Position {
                return vec4(0.0);
            }
        }

        fragment {
            vec4 main() @ gl_FragColor1 {
                return vec4(1.0);
            }
        }
    }
    """
    ast = crosstl.translator.parse(valid_code)
    assert "vertex float4 vertex_main()" in MetalCodeGen().generate_stage(ast, "vertex")
    fragment_code = MetalCodeGen().generate_stage(ast, "fragment")
    assert "struct fragment_main_Return" in fragment_code
    assert "float4 color [[color(1)]];" in fragment_code
    assert "fragment fragment_main_Return fragment_main()" in fragment_code
    assert "return fragment_main_Return{float4(1.0)};" in fragment_code


def test_function_return_semantics_lower_to_output_structs_when_required():
    depth_code = """
    shader DepthReturnBuiltin {
        fragment {
            float main() @ gl_FragDepth {
                return 0.5;
            }
        }
    }
    """
    depth_output = MetalCodeGen().generate_stage(
        crosstl.translator.parse(depth_code), "fragment"
    )
    assert "struct fragment_main_Return" in depth_output
    assert "float depth [[depth(any)]];" in depth_output
    assert "fragment fragment_main_Return fragment_main()" in depth_output
    assert "return fragment_main_Return{0.5};" in depth_output


def test_function_return_semantic_ignores_stage_control_attributes():
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
    }
    """
    ast = crosstl.translator.parse(code)

    vertex_code = MetalCodeGen().generate_stage(ast, "vertex")
    assert "vertex float4 vertex_main()" in vertex_code

    fragment_code = MetalCodeGen().generate_stage(ast, "fragment")
    assert "struct fragment_main_Return" in fragment_code
    assert "float4 color [[color(1)]];" in fragment_code
    assert "return fragment_main_Return{float4(1.0)};" in fragment_code


@pytest.mark.parametrize(
    ("stage", "semantic"),
    [
        ("vertex", "gl_Position"),
        ("fragment", "gl_FragColor"),
        ("fragment", "TEXCOORD0"),
    ],
)
def test_void_function_return_semantics_are_rejected(stage, semantic):
    code = f"""
    shader BadVoidReturnSemantic {{
        {stage} {{
            void main() @ {semantic} {{ }}
        }}
    }}
    """
    ast = crosstl.translator.parse(code)

    with pytest.raises(ValueError, match=f"{stage}.*{semantic}.*void return type"):
        MetalCodeGen().generate_stage(ast, stage)


@pytest.mark.parametrize(
    ("stage", "return_type", "semantic", "metal_semantic", "expected_type"),
    [
        ("vertex", "int", "gl_Position", "position", "float4"),
        ("fragment", "vec3", "gl_FragColor", "color\\(0\\)", "float4"),
        ("fragment", "vec4", "gl_FragDepth", "depth\\(any\\)", "float"),
    ],
)
def test_function_return_builtin_types_are_validated(
    stage, return_type, semantic, metal_semantic, expected_type
):
    value = {
        "int": "0",
        "vec2": "vec2(0.0)",
        "vec3": "vec3(0.0)",
        "vec4": "vec4(0.0)",
    }[return_type]
    code = f"""
    shader BadReturnBuiltin {{
        {stage} {{
            {return_type} main() @ {semantic} {{
                return {value};
            }}
        }}
    }}
    """
    ast = crosstl.translator.parse(code)
    with pytest.raises(ValueError, match=f"{metal_semantic}.*{expected_type}"):
        MetalCodeGen().generate_stage(ast, stage)


@pytest.mark.parametrize(
    ("stage", "return_type", "semantic", "value"),
    [
        ("vertex", "uint", "gl_VertexID", "0u"),
        ("vertex", "float", "gl_PointSize", "2.0"),
        ("fragment", "vec4", "gl_FragCoord", "vec4(0.0)"),
        ("fragment", "bool", "gl_FrontFacing", "true"),
        ("fragment", "vec2", "gl_PointCoord", "vec2(0.0)"),
    ],
)
def test_input_only_builtin_semantics_are_rejected_on_function_returns(
    stage, return_type, semantic, value
):
    code = f"""
    shader BadInputReturnBuiltin {{
        {stage} {{
            {return_type} main() @ {semantic} {{
                return {value};
            }}
        }}
    }}
    """
    ast = crosstl.translator.parse(code)
    with pytest.raises(ValueError, match=f"{semantic}.*return semantic"):
        MetalCodeGen().generate_stage(ast, stage)


@pytest.mark.parametrize(
    ("stage", "return_type", "semantic", "value", "expected"),
    [
        ("vertex", "vec4", "gl_FragColor", "vec4(0.0)", "fragment output"),
        ("vertex", "float", "gl_FragDepth", "0.5", "fragment output"),
        ("fragment", "vec4", "gl_Position", "vec4(0.0)", "vertex output"),
    ],
)
def test_function_return_output_builtin_stages_are_validated(
    stage, return_type, semantic, value, expected
):
    code = f"""
    shader BadReturnOutputStage {{
        {stage} {{
            {return_type} main() @ {semantic} {{
                return {value};
            }}
        }}
    }}
    """
    ast = crosstl.translator.parse(code)

    with pytest.raises(ValueError, match=f"{stage}.*{expected}.*{semantic}"):
        MetalCodeGen().generate_stage(ast, stage)


def test_metal_raytrace_and_mesh_intrinsics():
    code = """
    shader main {
        ray_generation {
            void main() {
                TraceRay();
            }
        }
        mesh {
            void main() {
                SetMeshOutputCounts(64, 32);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "unsupported Metal ray tracing intrinsic: TraceRay" in generated
    assert "TraceRay();" not in generated
    assert "SetMeshOutputCounts" in generated


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
                bloom = 0.0;
            } else {
                bloom = 0.5;
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
        pytest.fail("else if codegen not implemented.")


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    float result = add(1.0, 2.0);
                }
                
                float add(float a, float b) {
                    return a + b;
                }
            }
            """,
            "add(1.0, 2.0)",
        )
    ],
)
def test_function_call(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = MetalCodeGen()
    generated_code = code_gen.generate(ast)

    assert expected_output in generated_code


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    a |= 2;
                }
            }
            """,
            "a |= 2",
        )
    ],
)
def test_assignment_or_operator(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = MetalCodeGen()
    generated_code = code_gen.generate(ast)

    assert expected_output in generated_code


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    a <<= 2;
                    a >>= 1;
                }
            }
            """,
            ["a <<= 2", "a >>= 1"],
        )
    ],
)
def test_assignment_shift_operators(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = MetalCodeGen()
    generated_code = code_gen.generate(ast)

    for output in expected_output:
        assert output in generated_code


@pytest.mark.parametrize(
    "shader, expected_outputs",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    int b = 2;
                    int c = a | b;
                    int d = a & b;
                    int e = a ^ b;
                }
            }
            """,
            ["a | b", "a & b", "a ^ b"],
        )
    ],
)
def test_bitwise_operators(shader, expected_outputs):
    ast = crosstl.translator.parse(shader)
    code_gen = MetalCodeGen()
    generated_code = code_gen.generate(ast)

    for expected in expected_outputs:
        assert expected in generated_code


def test_bitwise_or_operator():
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
            // Use bitwise AND on texture coordinates (for testing purposes)
            output.color = vec4(float(int(input.texCoord.x * 100.0) | 15), 
                                float(int(input.texCoord.y * 100.0) | 15), 
                                0.0, 1.0);
            return output;
        }
    }
    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Simple fragment shader to display the result of the AND operation
            return vec4(input.color.rgb, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise OR codegen not implemented")


def test_metal_texture_types():
    """Test proper conversion of sampler types to Metal texture types."""
    code = """
    shader main {
        sampler2D albedoMap;
        sampler2D environmentMap;
        sampler2D depthMap;
        
        struct VSOutput {
            vec2 texCoord;
            vec4 position @ gl_Position;
        };
        
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 albedo = texture(albedoMap, input.texCoord);
                vec3 normal = normalize(vec3(0.0, 1.0, 0.0));
                vec4 reflection = texture(environmentMap, normalize(normal));
                float depth = texture(depthMap, input.texCoord).r;
                
                return albedo * depth;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        # Verify proper Metal texture types
        assert "texture2d<float> albedoMap" in generated_code
        assert "texture2d<float> environmentMap" in generated_code
        assert "texture2d<float> depthMap" in generated_code

        # Verify sampling operations
        # Just check for texture operations, not specific syntax
        assert "albedoMap" in generated_code
        assert "normalize" in generated_code
    except SyntaxError as e:
        pytest.fail(f"Metal texture type conversion failed: {e}")


def test_metal_attributes_semantics():
    """Test the conversion of CrossGL semantics to Metal attributes."""
    code = """
    shader main {
        struct VSInput {
            vec3 position @ POSITION;
            vec3 normal @ NORMAL;
            vec2 texCoord @ TEXCOORD0;
        };
        
        struct VSOutput {
            vec4 position @ gl_Position;
            vec3 worldNormal;
            vec2 texCoord;
        };
        
        struct FSOutput {
            vec4 color @ gl_FragColor;
        };
        
        vertex {
            VSOutput main(VSInput input, uint vertexID @ gl_VertexID) {
                VSOutput output;
                output.position = vec4(input.position, 1.0);
                output.worldNormal = input.normal;
                output.texCoord = input.texCoord;
                return output;
            }
        }
        
        fragment {
            FSOutput main(VSOutput input) {
                FSOutput output;
                output.color = vec4(normalize(input.worldNormal) * 0.5 + 0.5, 1.0);
                return output;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        # Check Metal attributes for vertex inputs
        assert "[[attribute(0)]]" in generated_code  # POSITION
        assert "[[attribute(1)]]" in generated_code  # NORMAL
        assert "[[attribute(5)]]" in generated_code  # TEXCOORD0

        # Check Metal attributes for vertex outputs
        assert "[[position]]" in generated_code  # gl_Position

        # Check Metal attributes for fragment outputs
        assert "[[color(0)]]" in generated_code  # gl_FragColor

        # Check vertex ID
        assert "uint vertexID [[vertex_id]]" in generated_code
    except SyntaxError as e:
        pytest.fail(f"Metal attributes/semantics conversion failed: {e}")


def test_metal_vector_type_conversions():
    """Test proper conversion of vector types to Metal types."""
    code = """
    shader main {
        struct TestTypes {
            vec2 vec2Field;
            vec3 vec3Field;
            vec4 vec4Field;
            ivec2 ivec2Field;
            ivec3 ivec3Field;
            ivec4 ivec4Field;
            mat2 mat2Field;
            mat3 mat3Field;
            mat4 mat4Field;
        };
        
        vertex {
            TestTypes main() {
                TestTypes output;
                output.vec2Field = vec2(1.0, 2.0);
                output.vec3Field = vec3(1.0, 2.0, 3.0);
                output.vec4Field = vec4(1.0, 2.0, 3.0, 4.0);
                output.ivec2Field = ivec2(1, 2);
                output.ivec3Field = ivec3(1, 2, 3);
                output.ivec4Field = ivec4(1, 2, 3, 4);
                
                // Set matrix fields
                output.mat2Field = mat2(1.0, 0.0, 0.0, 1.0);
                output.mat3Field = mat3(1.0);
                output.mat4Field = mat4(1.0);
                
                return output;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        # Check vector type conversions
        assert "float2 vec2Field" in generated_code
        assert "float3 vec3Field" in generated_code
        assert "float4 vec4Field" in generated_code
        assert "int2 ivec2Field" in generated_code
        assert "int3 ivec3Field" in generated_code
        assert "int4 ivec4Field" in generated_code

        # Check matrix type conversions
        assert "float2x2 mat2Field" in generated_code
        assert "float3x3 mat3Field" in generated_code
        assert "float4x4 mat4Field" in generated_code
    except SyntaxError as e:
        pytest.fail(f"Metal vector type conversion failed: {e}")


def test_metal_texture_sampling():
    """Test the proper conversion of texture sampling operations to Metal."""
    code = """
    shader main {
        sampler2D colorMap;
        sampler2D normalMap;
        sampler2D envMap;
        
        struct VSOutput {
            vec2 texCoord;
            vec3 normal;
            vec3 viewDir;
        };
        
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                // Basic sampling
                vec4 color = texture(colorMap, input.texCoord);
                
                // Component selection after sampling
                vec3 normalTS = texture(normalMap, input.texCoord).rgb;
                
                // Sampling with direction
                vec3 reflectDir = normalize(input.normal);
                vec4 reflectionColor = texture(envMap, input.texCoord);
                
                // Combined result
                return color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        # Check texture declarations
        assert "texture2d<float> colorMap" in generated_code
        assert "texture2d<float> normalMap" in generated_code
        assert "texture2d<float> envMap" in generated_code

        # Check that some form of texture access is happening
        # Metal uses either .sample() or other methods
        assert "colorMap" in generated_code and "texCoord" in generated_code
        assert "normalMap" in generated_code
        assert "envMap" in generated_code
    except SyntaxError as e:
        pytest.fail(f"Metal texture sampling conversion failed: {e}")


def test_metal_constant_buffer():
    """Test the conversion of constant buffers to Metal."""
    code = """
    shader main {
        struct MaterialParams {
            vec4 baseColor;
            float metallic;
            float roughness;
            vec2 textureScale;
        };
        
        MaterialParams material;
        
        struct VSOutput {
            vec2 texCoord;
        };
        
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec2 scaledTexCoord = input.texCoord * material.textureScale;
                vec4 finalColor = material.baseColor;
                finalColor.a = finalColor.a * (1.0 - material.roughness);
                finalColor.rgb = finalColor.rgb * material.metallic;
                return finalColor;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        # Check material struct declaration
        assert "struct MaterialParams" in generated_code

        # Check member access
        assert "material.baseColor" in generated_code
        assert "material.metallic" in generated_code
        assert "material.roughness" in generated_code
        assert "material.textureScale" in generated_code
    except SyntaxError as e:
        pytest.fail(f"Metal constant buffer conversion failed: {e}")


def test_metal_array_handling(array_test_data):
    """Test the Metal code generator's handling of array types and array access."""
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
        for expected in array_test_data["metal"]["array_type_declarations"]:
            assert expected in generated_code

        for expected in array_test_data["metal"]["array_access"]:
            assert expected in generated_code

    except SyntaxError as e:
        pytest.fail(f"Metal array codegen failed: {e}")


def test_metal_cbuffer_members_are_bound_as_constant_buffer_parameters():
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
    generated_code = MetalCodeGen().generate_stage(ast, "compute")

    assert [buffer.name for buffer in getattr(ast, "cbuffers", [])] == ["Constants"]
    assert [var.name for var in ast.global_variables] == []
    assert "struct Constants" in generated_code
    assert "float weights[8];" in generated_code
    assert "float3 colors[2];" in generated_code
    assert "constant Constants& constants [[buffer(0)]]" in generated_code
    assert "float value = constants.weights[2];" in generated_code
    assert "float3 color = constants.colors[1];" in generated_code


def test_metal_cbuffer_parameter_names_are_unique():
    code = """
    shader CBufferScope {
        cbuffer Constants {
            float exposure;
        };

        cbuffer constants {
            float gamma;
        };

        compute {
            void main() {
                float value = exposure + gamma;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    generated_code = MetalCodeGen().generate_stage(ast, "compute")

    assert "constant Constants& constants [[buffer(0)]]" in generated_code
    assert "constant constants& constants1 [[buffer(1)]]" in generated_code
    assert "float value = constants.exposure + constants1.gamma;" in generated_code


def test_metal_cbuffer_parameter_avoids_stage_input_name():
    code = """
    shader CBufferScope {
        struct VSInput {
            vec3 position @ POSITION;
        };

        struct VSOutput {
            vec4 position @ SV_POSITION;
        };

        cbuffer Input {
            float scale;
        };

        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.position = vec4(input.position * scale, 1.0);
                return output;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    generated_code = MetalCodeGen().generate_stage(ast, "vertex")

    assert "VSInput input [[stage_in]]" in generated_code
    assert "constant Input& input1 [[buffer(0)]]" in generated_code
    assert "input.position * input1.scale" in generated_code


def test_metal_cbuffer_parameter_avoids_texture_resource_name():
    code = """
    shader CBufferScope {
        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        cbuffer DiffuseMap {
            float exposure;
        };

        sampler2D diffuseMap;

        fragment {
            vec4 main(FSInput input) {
                return texture(diffuseMap, input.uv) * exposure;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    generated_code = MetalCodeGen().generate_stage(ast, "fragment")

    assert "constant DiffuseMap& diffuseMap1 [[buffer(0)]]" in generated_code
    assert "texture2d<float> diffuseMap [[texture(0)]]" in generated_code
    assert "diffuseMap1.exposure" in generated_code


def test_metal_helper_threads_cbuffer_parameter():
    code = """
    shader CBufferScope {
        cbuffer Constants {
            float scale;
        };

        float applyScale(float value) {
            return value * scale;
        }

        compute {
            void main() {
                float value = applyScale(2.0);
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    generated_code = MetalCodeGen().generate_stage(ast, "compute")

    assert (
        "float applyScale(float value, constant Constants& constants)" in generated_code
    )
    assert "return value * constants.scale;" in generated_code
    assert "float value = applyScale(2.0, constants);" in generated_code


def test_metal_transitively_threads_cbuffer_parameter():
    code = """
    shader CBufferScope {
        cbuffer Constants {
            float scale;
        };

        float readScale(float value) {
            return value * scale;
        }

        float applyScale(float value) {
            return readScale(value);
        }

        compute {
            void main() {
                float value = applyScale(2.0);
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    generated_code = MetalCodeGen().generate_stage(ast, "compute")

    assert (
        "float readScale(float value, constant Constants& constants)" in generated_code
    )
    assert (
        "float applyScale(float value, constant Constants& constants)" in generated_code
    )
    assert "return readScale(value, constants);" in generated_code
    assert "float value = applyScale(2.0, constants);" in generated_code


def test_metal_helper_local_cbuffer_member_name_does_not_thread_parameter():
    code = """
    shader CBufferScope {
        cbuffer Constants {
            float scale;
        };

        float applyScale(float value) {
            float scale = 2.0;
            return value * scale;
        }

        compute {
            void main() {
                float value = applyScale(2.0);
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    generated_code = MetalCodeGen().generate_stage(ast, "compute")

    assert "float applyScale(float value)" in generated_code
    assert "return value * scale;" in generated_code
    assert "float value = applyScale(2.0);" in generated_code


def test_metal_helper_threads_global_texture_parameter():
    code = """
    shader TextureHelperScope {
        sampler2D colorMap;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        vec4 sampleColor(vec2 uv) {
            return texture(colorMap, uv);
        }

        fragment {
            vec4 main(FSInput input) {
                return sampleColor(input.uv);
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    generated_code = MetalCodeGen().generate_stage(ast, "fragment")

    assert "float4 sampleColor(float2 uv, texture2d<float> colorMap)" in generated_code
    assert (
        "return colorMap.sample(sampler(mag_filter::linear, min_filter::linear), uv);"
        in generated_code
    )
    assert "return sampleColor(input.uv, colorMap);" in generated_code


def test_metal_helper_threads_global_texture_and_implicit_sampler_parameters():
    code = """
    shader TextureHelperScope {
        sampler2D colorMap;
        sampler colorMapSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        vec4 sampleColor(vec2 uv) {
            return texture(colorMap, uv);
        }

        fragment {
            vec4 main(FSInput input) {
                return sampleColor(input.uv);
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    generated_code = MetalCodeGen().generate_stage(ast, "fragment")

    assert (
        "float4 sampleColor(float2 uv, texture2d<float> colorMap, sampler colorMapSampler)"
        in generated_code
    )
    assert "return colorMap.sample(colorMapSampler, uv);" in generated_code
    assert "return sampleColor(input.uv, colorMap, colorMapSampler);" in generated_code


def test_metal_transitively_threads_global_texture_parameter():
    code = """
    shader TextureHelperScope {
        sampler2D colorMap;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        vec4 sampleColor(vec2 uv) {
            return texture(colorMap, uv);
        }

        vec4 shade(vec2 uv) {
            return sampleColor(uv);
        }

        fragment {
            vec4 main(FSInput input) {
                return shade(input.uv);
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    generated_code = MetalCodeGen().generate_stage(ast, "fragment")

    assert "float4 sampleColor(float2 uv, texture2d<float> colorMap)" in generated_code
    assert "float4 shade(float2 uv, texture2d<float> colorMap)" in generated_code
    assert "return sampleColor(uv, colorMap);" in generated_code
    assert "return shade(input.uv, colorMap);" in generated_code


def test_metal_ambiguous_cbuffer_member_reference_errors():
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
        match="Ambiguous cbuffer member reference 'value' appears in multiple cbuffers",
    ):
        MetalCodeGen().generate_stage(ast, "compute")


def test_metal_duplicate_cbuffer_names_error():
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
        match="Duplicate cbuffer name\\(s\\) in Metal output: Camera",
    ):
        MetalCodeGen().generate_stage(ast, "compute")


def test_metal_cbuffer_name_conflicts_with_struct_error():
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
        match="Cbuffer name\\(s\\) conflict with existing Metal declaration\\(s\\): Camera",
    ):
        MetalCodeGen().generate_stage(ast, "compute")


def test_metal_array_member_semantics():
    code = """
    struct VertexData {
        float weights[4] @ TEXCOORD0;
        vec3 colors[] @ COLOR;
    };
    """

    ast = crosstl.translator.parse(code)
    generated_code = MetalCodeGen().generate(ast)

    assert "float weights[4] [[attribute(5)]];" in generated_code
    assert "array<float3> colors [[COLOR]];" in generated_code


def test_metal_local_array_declarations_use_c_style_order():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "float3 localColors[4];" in generated_code
    assert "float weights[8];" in generated_code
    assert "float3[4] localColors" not in generated_code
    assert "float[8] weights" not in generated_code


def test_metal_array_parameters_use_c_style_order():
    shader = """
    shader TestShader {
        float accumulate(float weights[4], vec3 colors[2]) {
            return weights[0] + colors[1].x;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "float weights[4]" in generated_code
    assert "float3 colors[2]" in generated_code
    assert "float weights[4] [[stage_in]]" not in generated_code
    assert "float3 colors[2] [[stage_in]]" not in generated_code
    assert "float[4] weights" not in generated_code
    assert "float3[2] colors" not in generated_code


def test_metal_non_resource_arrays_preserve_expression_sizes():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "float3 colors[(2 + 1) * 2];" in generated_code
    assert "float weights[+6];" in generated_code
    assert (
        "float accumulate(float values[(2 + 1) * 2], float3 normals[+6])"
        in generated_code
    )
    assert "float localWeights[(2 + 1) * 2];" in generated_code
    assert "float3 localNormals[+6];" in generated_code
    assert "float values[]" not in generated_code
    assert "float localWeights[]" not in generated_code
    assert "2 + 1 * 2" not in generated_code


def test_metal_sampler_cube_uses_cube_texture_type():
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
                vec4 color = texture(colorMap, input.uv);
                vec4 env = texture(envMap, input.normal);
                return color + env;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texturecube<float> envMap [[texture(1)]]" in generated_code
    assert "texture2d<float> envMap" not in generated_code
    assert "colorMap.sample" in generated_code
    assert "envMap.sample" in generated_code


def test_metal_rejects_non_resource_shadow_of_global_resource():
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
            "Non-resource local declaration\\(s\\) shadow Metal global "
            "resource\\(s\\): colorMap, linearSampler"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


def test_metal_rejects_texture_call_with_non_resource_argument():
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
            "Metal texture operation 'texture' requires a declared texture "
            "or image resource argument: value"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            "texture(colorMap)",
            "Metal texture operation 'texture' requires at least 2 "
            "argument\\(s\\), got 1",
        ),
        (
            "textureLod(colorMap, input.uv)",
            "Metal texture operation 'textureLod' requires at least 3 "
            "argument\\(s\\), got 2",
        ),
        (
            "textureGrad(colorMap, input.uv, input.uv)",
            "Metal texture operation 'textureGrad' requires at least 4 "
            "argument\\(s\\), got 3",
        ),
        (
            "texture(colorMap, linearSampler)",
            "Metal texture operation 'texture' requires at least 3 "
            "argument\\(s\\), got 2",
        ),
    ],
)
def test_metal_rejects_texture_call_with_too_few_arguments(call, match):
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
        MetalCodeGen().generate(crosstl.translator.parse(shader))


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
def test_metal_rejects_resource_query_call_with_too_many_arguments(
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
            f"Metal texture operation '{operation}' accepts at most "
            f"{max_count} argument\\(s\\), got {arg_count}"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


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
def test_metal_rejects_fixed_resource_call_with_too_many_arguments(
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
            f"Metal texture operation '{operation}' accepts at most "
            f"{max_count} argument\\(s\\), got {arg_count}"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


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
def test_metal_rejects_texture_sampling_call_with_too_many_arguments(
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
            f"Metal texture operation '{operation}' accepts at most "
            f"{max_count} argument\\(s\\), got {arg_count}"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


def test_metal_rejects_texture_gather_offsets_with_ambiguous_argument_count():
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
            "Metal texture operation 'textureGatherOffsets' accepts "
            "3, 4, 6, 7 argument\\(s\\), got 5"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


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
def test_metal_rejects_image_call_with_sampled_texture_argument(statement, operation):
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
            f"Metal image operation '{operation}' requires a storage "
            "image resource argument: colorMap"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("helper", "helper_call", "match"),
    [
        (
            "vec4 misuse(sampler sampleState, vec2 uv) { return texture(sampleState, uv); }",
            "misuse(linearSampler, input.uv)",
            "Metal texture operation 'texture' requires a declared texture "
            "or image resource argument: sampleState",
        ),
        (
            "vec4 misuse(sampler sampleState, ivec2 pixel) { return imageLoad(sampleState, pixel); }",
            "misuse(linearSampler, input.pixel)",
            "Metal image operation 'imageLoad' requires a storage image "
            "resource argument: sampleState",
        ),
    ],
)
def test_metal_rejects_sampler_parameter_as_texture_or_image_operand(
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
        MetalCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        ("texelFetch(colorMap, input.uv, 0)", "texelFetch"),
        ("imageLoad(colorImage, input.uv)", "imageLoad"),
    ],
)
def test_metal_rejects_float_coordinate_for_integer_resource_operation(call, operation):
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
            f"Metal resource operation '{operation}' requires an integer "
            "coordinate argument: input.uv has type float2"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation", "dimension"),
    [
        ("texelFetch(arrayMap, input.pixel2, 0)", "texelFetch", 3),
        ("imageLoad(volumeImage, input.pixel2)", "imageLoad", 3),
        ("imageLoad(colorImage, input.pixel3)", "imageLoad", 2),
    ],
)
def test_metal_rejects_wrong_coordinate_dimension_for_resource_operation(
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
            f"Metal resource operation '{operation}' requires a "
            f"{dimension}D integer coordinate"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


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
def test_metal_rejects_wrong_offset_dimension_for_resource_operation(call, operation):
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
        match=(f"Metal resource operation '{operation}' requires a 2D integer offset"),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


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
    ],
)
def test_metal_rejects_wrong_offset_dimension_for_extended_resource_operation(
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
        match=(f"Metal resource operation '{operation}' requires a 2D integer offset"),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


def test_metal_rejects_float_offset_for_resource_operation():
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
            "Metal resource operation 'textureOffset' requires an integer "
            "offset argument"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


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
    ],
)
def test_metal_rejects_float_offset_for_extended_resource_operation(call, operation):
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
            f"Metal resource operation '{operation}' requires an integer "
            "offset argument"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


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
def test_metal_rejects_wrong_gradient_dimension_for_resource_operation(
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
            f"Metal resource operation '{operation}' requires a "
            f"{dimension}D floating gradient"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


def test_metal_rejects_integer_gradient_for_resource_operation():
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
            "Metal resource operation 'textureGrad' requires a floating "
            "gradient argument"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation", "type_name"),
    [
        (
            "textureLod(colorMap, linearSampler, input.uv, input.lodVec)",
            "textureLod",
            "float2",
        ),
        (
            "textureLodOffset(colorMap, linearSampler, input.uv, input.lodVec, input.offset)",
            "textureLodOffset",
            "float2",
        ),
        (
            "textureProjLod(colorMap, linearSampler, input.projCoord, input.lodVec)",
            "textureProjLod",
            "float2",
        ),
        (
            "vec4(textureCompareLod(shadowMap, compareSampler, input.uv, input.depth, input.lodVec))",
            "textureCompareLod",
            "float2",
        ),
        (
            "vec4(textureCompareProjLodOffset(shadowMap, compareSampler, input.projCoord, input.depth, input.lodVec, input.offset))",
            "textureCompareProjLodOffset",
            "float2",
        ),
        (
            "textureLod(colorMap, linearSampler, input.uv, colorMap)",
            "textureLod",
            "texture2d<float>",
        ),
    ],
)
def test_metal_rejects_non_scalar_numeric_lod_argument(call, operation, type_name):
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
            f"Metal texture LOD operation '{operation}' requires a scalar "
            f"numeric lod argument: .* has type {type_name}"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


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
            "int2",
        ),
        (
            "vec4(textureSize(colorMap, input.floatLevel), 0.0, 1.0)",
            "textureSize",
            "float",
        ),
        (
            "texelFetch(colorMap, input.pixel, colorMap)",
            "texelFetch",
            "texture2d<float>",
        ),
    ],
)
def test_metal_rejects_non_scalar_integer_mip_level_argument(
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
            f"Metal resource operation '{operation}' requires a scalar integer "
            f"mip/sample level argument: .* has type {type_name}"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expr", "type_name"),
    [
        ("texelFetch(msTex, input.pixel, input.floatSample)", "float"),
        ("texelFetch(msArray, input.pixelLayer, input.sampleVec)", "int2"),
        ("texelFetch(msTex, input.pixel, msTex)", "texture2d_ms<float>"),
    ],
)
def test_metal_rejects_non_scalar_integer_multisample_sample_index(
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
            "Metal multisample texel fetch operation 'texelFetch' requires a "
            f"scalar integer sample index argument: .* has type {type_name}"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


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
            "int2",
        ),
        (
            "textureGatherOffsets(colorMap, linearSampler, input.uv, input.offset, input.offset, input.offset, input.offset, colorMap)",
            "",
            "textureGatherOffsets",
            "texture2d<float>",
        ),
        (
            "vec4(1.0)",
            "vec4 invalidArrayGather(sampler2D tex, sampler s, vec2 uv, ivec2 offsets[4], float component) { return textureGatherOffsets(tex, s, uv, offsets, component); }",
            "textureGatherOffsets",
            "float",
        ),
    ],
)
def test_metal_rejects_non_scalar_integer_gather_component_argument(
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
            f"Metal texture gather operation '{operation}' requires a scalar "
            f"integer component argument: .* has type {type_name}"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expr", "operation", "type_name"),
    [
        (
            "texture(colorMap, linearSampler, input.uv, input.biasVec)",
            "texture",
            "float2",
        ),
        (
            "textureOffset(colorMap, linearSampler, input.uv, input.offset, input.biasVec)",
            "textureOffset",
            "float2",
        ),
        (
            "textureProj(colorMap, linearSampler, input.projCoord, input.biasVec)",
            "textureProj",
            "float2",
        ),
        (
            "textureProjOffset(colorMap, linearSampler, input.projCoord, input.offset, colorMap)",
            "textureProjOffset",
            "texture2d<float>",
        ),
    ],
)
def test_metal_rejects_non_scalar_numeric_bias_argument(
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
            f"Metal texture bias operation '{operation}' requires a scalar "
            f"numeric bias argument: .* has type {type_name}"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


def test_metal_texture_bias_variants_use_bias_options():
    shader = """
    shader TextureBiasVariants {
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv;
            ivec2 offset;
            float bias;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 biased = texture(colorMap, linearSampler, input.uv, input.bias);
                vec4 offsetBiased = textureOffset(
                    colorMap,
                    linearSampler,
                    input.uv,
                    input.offset,
                    input.bias
                );
                return biased + offsetBiased;
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "colorMap.sample(linearSampler, input.uv, bias(input.bias))" in generated_code
    )
    assert (
        "colorMap.sample(linearSampler, input.uv, bias(input.bias), input.offset)"
        in generated_code
    )


def test_metal_rejects_non_floating_query_lod_coordinate():
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
            "Metal texture query operation 'textureQueryLod' requires a "
            "floating coordinate argument: .* has type int2"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("query_call", "dimension", "resource_type", "type_name"),
    [
        (
            "textureQueryLod(layerMap, linearSampler, input.uv)",
            3,
            "texture2d_array<float>",
            "float2",
        ),
        (
            "textureQueryLod(cubeMap, linearSampler, input.uv)",
            3,
            "texturecube<float>",
            "float2",
        ),
        (
            "textureQueryLod(cubeArray, linearSampler, input.direction)",
            4,
            "texturecube_array<float>",
            "float3",
        ),
    ],
)
def test_metal_rejects_wrong_query_lod_coordinate_dimension(
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
            "Metal texture query operation 'textureQueryLod' requires a "
            f"{dimension}D floating coordinate for {resource_type}: "
            f".* has type {type_name}"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation", "type_name"),
    [
        (
            "vec4(textureCompare(shadowMap, compareSampler, input.uv, input.depthVec))",
            "textureCompare",
            "float2",
        ),
        (
            "vec4(textureCompareProjGradOffset(shadowMap, compareSampler, input.projCoord, input.depthVec, input.ddx, input.ddy, input.offset))",
            "textureCompareProjGradOffset",
            "float2",
        ),
        (
            "textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depthVec, input.offset)",
            "textureGatherCompareOffset",
            "float2",
        ),
        (
            "vec4(textureCompare(shadowMap, compareSampler, input.uv, input.layer))",
            "textureCompare",
            "int",
        ),
    ],
)
def test_metal_rejects_non_scalar_float_compare_argument(call, operation, type_name):
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
            f"Metal texture compare operation '{operation}' requires a scalar "
            f"floating compare argument: .* has type {type_name}"
        ),
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


def test_metal_texture_array_resources_and_indexed_sampling():
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
                vec4 color = texture(textures[input.layer], input.uv);
                vec4 env = texture(envMap, input.normal);
                return color + env;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "texturecube<float> envMap [[texture(4)]]" in generated_code
    assert (
        "textures[input.layer].sample(sampler(mag_filter::linear, min_filter::linear), input.uv)"
        in generated_code
    )
    assert "envMap.sample" in generated_code


def test_metal_fixed_texture_and_sampler_arrays_keep_declared_size_with_constant_indices():
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
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int LAYER = 2;" in generated_code
    assert "array<texture2d<float>, 6> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(6)]]" in generated_code
    assert "array<sampler, 6> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 6> textures, array<sampler, 6> samplers"
        in generated_code
    )
    assert "textures[LAYER].sample(samplers[LAYER], uv)" in generated_code
    assert "textures[1 + 2].sample(samplers[1 + 2], uv)" in generated_code
    assert "array<texture2d<float>, 4> textures" not in generated_code
    assert "texture2d<float> afterTexture [[texture(4)]]" not in generated_code


def test_metal_fixed_texture_and_sampler_arrays_resolve_constant_declared_size_for_bindings():
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
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int BASE_COUNT = 2;" in generated_code
    assert "constant int TEXTURE_COUNT = BASE_COUNT * 3;" in generated_code
    assert (
        "array<texture2d<float>, TEXTURE_COUNT> textures [[texture(0)]]"
        in generated_code
    )
    assert "texture2d<float> afterTexture [[texture(6)]]" in generated_code
    assert "array<sampler, TEXTURE_COUNT> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, TEXTURE_COUNT> textures, array<sampler, TEXTURE_COUNT> samplers"
        in generated_code
    )
    assert "textures[2].sample(samplers[2], uv)" in generated_code
    assert "texture2d<float> afterTexture [[texture(1)]]" not in generated_code


def test_metal_fixed_texture_and_sampler_arrays_resolve_inline_declared_size_expression_for_bindings():
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
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<texture2d<float>, 2 * 3> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(6)]]" in generated_code
    assert "array<sampler, 2 * 3> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 2 * 3> textures, array<sampler, 2 * 3> samplers"
        in generated_code
    )
    assert "textures[2].sample(samplers[2], uv)" in generated_code
    assert "texture2d<float> afterTexture [[texture(1)]]" not in generated_code
    assert "BinaryOpNode" not in generated_code
    assert "None" not in generated_code


def test_metal_fixed_texture_arrays_preserve_parenthesized_and_unary_declared_sizes():
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
                return sampleLayer(textures, samplers, unaryTextures, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "array<texture2d<float>, (2 + 1) * 2> textures [[texture(0)]]" in generated_code
    )
    assert "array<texture2d<float>, +6> unaryTextures [[texture(6)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(12)]]" in generated_code
    assert "array<sampler, (2 + 1) * 2> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, (2 + 1) * 2> textures, array<sampler, (2 + 1) * 2> samplers"
        in generated_code
    )
    assert "textures[2].sample(samplers[2], uv)" in generated_code
    assert "unaryTextures[2].sample" in generated_code
    assert "texture2d<float> afterTexture [[texture(6)]]" not in generated_code
    assert "2 + 1 * 2" not in generated_code
    assert "BinaryOpNode" not in generated_code
    assert "None" not in generated_code


def test_metal_storage_image_load_store():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<float, access::read_write> outputImage [[texture(0)]], texture3d<float, access::read_write> volumeImage [[texture(1)]], texture2d_array<float, access::read_write> layerImage [[texture(2)]])"
        in generated_code
    )
    assert (
        "float4 touchImages(texture2d<float, access::read_write> outImg, texture3d<float, access::read_write> volume, texture2d_array<float, access::read_write> layers, int2 pixel, int3 voxel, int3 pixelLayer)"
        in generated_code
    )
    assert "float4 color = outImg.read(uint2(pixel));" in generated_code
    assert "float4 volumeColor = volume.read(uint3(voxel));" in generated_code
    assert (
        "float4 layerColor = layers.read(uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "outImg.write(color + layerColor, uint2(pixel));" in generated_code
    assert "volume.write(volumeColor, uint3(voxel));" in generated_code
    assert (
        "layers.write(color, uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "kernel void kernel_main(," not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_unsigned_constructor_image_coordinates_are_not_rewrapped():
    shader = """
    shader UnsignedConstructorImageCoordinates {
        image2D outputImage;
        image3D volumeImage;

        compute {
            void main() {
                vec4 color = imageLoad(outputImage, uvec2(1u, 2u));
                vec4 volumeColor = imageLoad(volumeImage, uvec3(1u, 2u, 3u));
                imageStore(outputImage, uvec2(1u, 2u), color);
                imageStore(volumeImage, uvec3(1u, 2u, 3u), volumeColor);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "float4 color = outputImage.read(uint2(1u, 2u));" in generated_code
    assert "float4 volumeColor = volumeImage.read(uint3(1u, 2u, 3u));" in generated_code
    assert "outputImage.write(color, uint2(1u, 2u));" in generated_code
    assert "volumeImage.write(volumeColor, uint3(1u, 2u, 3u));" in generated_code
    assert "uint2(uint2" not in generated_code
    assert "uint3(uint3" not in generated_code


def test_metal_direct_stage_image_load_store_and_atomics_use_input_members():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "fragment float4 fragment_main(FSInput input [[stage_in]], texture2d<float, access::read_write> outImg [[texture(0)]], texture3d<float, access::read_write> volume [[texture(1)]], texture2d_array<float, access::read_write> layers [[texture(2)]], texture2d<uint, access::read_write> counters [[texture(3)]], texture2d_array<int, access::read_write> signedLayers [[texture(4)]])"
        in generated_code
    )
    assert "float4 color = outImg.read(uint2(input.pixel));" in generated_code
    assert "float4 volumeColor = volume.read(uint3(input.voxel));" in generated_code
    assert (
        "float4 layerColor = layers.read(uint2(input.pixelLayer.xy), uint(input.pixelLayer.z));"
        in generated_code
    )
    assert "outImg.write(color + layerColor, uint2(input.pixel));" in generated_code
    assert "volume.write(volumeColor, uint3(input.voxel));" in generated_code
    assert (
        "layers.write(color, uint2(input.pixelLayer.xy), uint(input.pixelLayer.z));"
        in generated_code
    )
    assert (
        "uint previous = counters.atomic_fetch_add(uint2(input.pixel), input.amount).x;"
        in generated_code
    )
    assert (
        "int previousSigned = signedLayers.atomic_fetch_min(uint2(input.pixelLayer.xy), uint(input.pixelLayer.z), input.signedAmount).x;"
        in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code
    assert "imageAtomicAdd(" not in generated_code
    assert "imageAtomicMin(" not in generated_code


def test_metal_direct_stage_explicit_image_formats_use_input_members():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "fragment float4 fragment_main(FSInput input [[stage_in]], texture2d<float, access::read_write> scalarFloat [[texture(0)]], texture2d<float, access::read_write> rgFloat [[texture(1)]], texture2d<uint, access::read_write> unsignedScalar [[texture(2)]], texture2d_array<uint, access::read_write> unsignedLayers [[texture(3)]], texture3d<int, access::read_write> signedVolume [[texture(4)]])"
        in generated_code
    )
    assert (
        "float scalarValue = scalarFloat.read(uint2(input.pixel)).x;" in generated_code
    )
    assert "float2 rgValue = rgFloat.read(uint2(input.pixel)).xy;" in generated_code
    assert (
        "uint unsignedValue = unsignedScalar.read(uint2(input.pixel)).x;"
        in generated_code
    )
    assert (
        "uint2 layerValue = unsignedLayers.read(uint2(input.pixelLayer.xy), uint(input.pixelLayer.z)).xy;"
        in generated_code
    )
    assert (
        "int signedValue = signedVolume.read(uint3(input.voxel)).x;" in generated_code
    )
    assert (
        "scalarFloat.write(float4(scalarValue + input.amount), uint2(input.pixel));"
        in generated_code
    )
    assert (
        "rgFloat.write(float4(rgValue + float2(input.amount), 0.0, 0.0), uint2(input.pixel));"
        in generated_code
    )
    assert (
        "unsignedScalar.write(uint4(unsignedValue + input.unsignedAmount), uint2(input.pixel));"
        in generated_code
    )
    assert (
        "unsignedLayers.write(uint4(layerValue + uint2(input.unsignedAmount), 0u, 0u), uint2(input.pixelLayer.xy), uint(input.pixelLayer.z));"
        in generated_code
    )
    assert (
        "signedVolume.write(int4(signedValue + input.signedAmount), uint3(input.voxel));"
        in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_image_atomic_compare_descriptor_for_integer_storage_textures():
    codegen = MetalCodeGen()

    uint_array_descriptor = codegen.image_atomic_compare_descriptor(
        "texture1d_array<uint, access::read>"
    )
    assert uint_array_descriptor == {
        "helper_name": "imageAtomicCompSwap_uimage1DArray",
        "return_type": "uint",
        "vector_type": "uint4",
        "coord_type": "int2",
        "exchange_expr": (
            "image.atomic_compare_exchange_weak(uint(coord.x), uint(coord.y), &original, value)"
        ),
    }

    int_layer_descriptor = codegen.image_atomic_compare_descriptor(
        "texture2d_array<int, access::write>"
    )
    assert int_layer_descriptor["helper_name"] == "imageAtomicCompSwap_iimage2DArray"
    assert int_layer_descriptor["return_type"] == "int"
    assert int_layer_descriptor["vector_type"] == "int4"
    assert int_layer_descriptor["coord_type"] == "int3"
    assert (
        int_layer_descriptor["exchange_expr"]
        == "image.atomic_compare_exchange_weak(uint2(coord.xy), uint(coord.z), &original, value)"
    )

    assert (
        codegen.image_atomic_compare_descriptor("texture2d<float, access::read_write>")
        is None
    )


def test_metal_direct_stage_image_compare_swap_use_input_members():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "fragment float4 fragment_main(FSInput input [[stage_in]], texture3d<uint, access::read_write> volumeCounters [[texture(0)]], texture2d_array<int, access::read_write> layerCounters [[texture(1)]])"
        in generated_code
    )
    assert (
        "uint imageAtomicCompSwap_uimage3D(texture3d<uint, access::read_write> image, int3 coord, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicCompSwap_iimage2DArray(texture2d_array<int, access::read_write> image, int3 coord, int compareValue, int value)"
        in generated_code
    )
    assert (
        "image.atomic_compare_exchange_weak(uint3(coord), &original, value)"
        in generated_code
    )
    assert (
        "image.atomic_compare_exchange_weak(uint2(coord.xy), uint(coord.z), &original, value)"
        in generated_code
    )
    assert (
        "uint volumeOld = imageAtomicCompSwap_uimage3D(volumeCounters, input.voxel, input.expected, input.replacement);"
        in generated_code
    )
    assert (
        "int layerOld = imageAtomicCompSwap_iimage2DArray(layerCounters, input.pixelLayer, input.signedExpected, input.signedReplacement);"
        in generated_code
    )
    assert "imageAtomicCompSwap(volumeCounters" not in generated_code
    assert "imageAtomicCompSwap(layerCounters" not in generated_code


def test_metal_integer_image_atomic_add():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<uint, access::read_write> counters [[texture(0)]], texture2d<int, access::read_write> signedCounters [[texture(1)]])"
        in generated_code
    )
    assert (
        "uint addCounter(texture2d<uint, access::read_write> image, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "int addSignedCounter(texture2d<int, access::read_write> image, int2 pixel, int value)"
        in generated_code
    )
    assert (
        "uint previous = image.atomic_fetch_add(uint2(pixel), value).x;"
        in generated_code
    )
    assert (
        "int previous = image.atomic_fetch_add(uint2(pixel), value).x;"
        in generated_code
    )
    assert "imageAtomicAdd(" not in generated_code


def test_metal_integer_image_atomic_operations():
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
    generated_code = MetalCodeGen().generate(ast)

    for method in [
        "atomic_fetch_min",
        "atomic_fetch_max",
        "atomic_fetch_and",
        "atomic_fetch_or",
        "atomic_fetch_xor",
        "atomic_exchange",
    ]:
        assert f"image.{method}(uint2(pixel), value).x" in generated_code

    assert (
        "uint unsignedOps(texture2d<uint, access::read_write> image, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "int signedOps(texture2d<int, access::read_write> image, int2 pixel, int value)"
        in generated_code
    )
    assert "imageAtomicMin(" not in generated_code
    assert "imageAtomicMax(" not in generated_code
    assert "imageAtomicExchange(" not in generated_code


def test_metal_integer_image_atomic_compare_swap():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "uint imageAtomicCompSwap_uimage2D(texture2d<uint, access::read_write> image, int2 coord, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicCompSwap_iimage2D(texture2d<int, access::read_write> image, int2 coord, int compareValue, int value)"
        in generated_code
    )
    assert "uint4 original;" in generated_code
    assert "int4 original;" in generated_code
    assert "original.x = compareValue;" in generated_code
    assert (
        "image.atomic_compare_exchange_weak(uint2(coord), &original, value)"
        in generated_code
    )
    assert "return original.x;" in generated_code
    assert (
        "uint previous = imageAtomicCompSwap_uimage2D(image, pixel, expected, replacement);"
        in generated_code
    )
    assert (
        "int previous = imageAtomicCompSwap_iimage2D(image, pixel, expected, replacement);"
        in generated_code
    )
    assert "imageAtomicCompSwap(image" not in generated_code


def test_metal_integer_image_dimension_atomics():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture3d<uint, access::read_write> volumeCounters [[texture(0)]], texture3d<int, access::read_write> signedVolumeCounters [[texture(1)]], texture2d_array<int, access::read_write> layerCounters [[texture(2)]], texture2d_array<uint, access::read_write> unsignedLayerCounters [[texture(3)]])"
        in generated_code
    )
    assert (
        "uint imageAtomicCompSwap_uimage3D(texture3d<uint, access::read_write> image, int3 coord, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicCompSwap_iimage2DArray(texture2d_array<int, access::read_write> image, int3 coord, int compareValue, int value)"
        in generated_code
    )
    assert (
        "image.atomic_compare_exchange_weak(uint3(coord), &original, value)"
        in generated_code
    )
    assert (
        "image.atomic_compare_exchange_weak(uint2(coord.xy), uint(coord.z), &original, value)"
        in generated_code
    )
    assert (
        "uint oldValue = image.atomic_fetch_add(uint3(voxel), value).x;"
        in generated_code
    )
    assert (
        "int oldValue = image.atomic_fetch_min(uint2(pixelLayer.xy), uint(pixelLayer.z), value).x;"
        in generated_code
    )
    assert (
        "uint swapped = imageAtomicCompSwap_uimage3D(image, voxel, oldValue, value);"
        in generated_code
    )
    assert (
        "int swapped = imageAtomicCompSwap_iimage2DArray(image, pixelLayer, oldValue, value);"
        in generated_code
    )
    assert "imageAtomicAdd(" not in generated_code
    assert "imageAtomicMin(" not in generated_code
    assert "imageAtomicCompSwap(image" not in generated_code


def test_metal_resource_array_image_atomics_use_indexed_textures():
    shader = """
    shader ResourceArrayImageAtomics {
        uimage2D counters @r32ui[2];
        iimage2D signedCounters @r32i[2];

        uint addCounter(uimage2D images[2] @r32ui, int index, ivec2 pixel, uint value) {
            return imageAtomicAdd(images[index], pixel, value);
        }

        int swapSigned(
            iimage2D images[2] @r32i,
            int index,
            ivec2 pixel,
            int expected,
            int value
        ) {
            return imageAtomicCompSwap(images[index], pixel, expected, value);
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                uint oldValue = addCounter(counters, 1, pixel, 2u);
                int oldSigned = swapSigned(signedCounters, 0, pixel, 3, int(oldValue));
                imageStore(counters[0], pixel, oldValue + uint(oldSigned));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate_stage(ast, "compute")

    assert (
        "kernel void kernel_main(array<texture2d<uint, access::read_write>, 2> counters [[texture(0)]], array<texture2d<int, access::read_write>, 2> signedCounters [[texture(2)]])"
        in generated_code
    )
    assert (
        "uint addCounter(array<texture2d<uint, access::read_write>, 2> images, int index, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "int swapSigned(array<texture2d<int, access::read_write>, 2> images, int index, int2 pixel, int expected, int value)"
        in generated_code
    )
    assert (
        "int imageAtomicCompSwap_iimage2D(texture2d<int, access::read_write> image, int2 coord, int compareValue, int value)"
        in generated_code
    )
    assert (
        "return images[index].atomic_fetch_add(uint2(pixel), value).x;"
        in generated_code
    )
    assert (
        "return imageAtomicCompSwap_iimage2D(images[index], pixel, expected, value);"
        in generated_code
    )
    assert (
        "counters[0].write(uint4(oldValue + uint(oldSigned)), uint2(pixel));"
        in generated_code
    )
    assert "imageAtomicAdd(" not in generated_code
    assert "imageAtomicCompSwap(images" not in generated_code


def test_metal_integer_image_scalar_load_store():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<uint, access::read_write> counters [[texture(0)]], texture3d<int, access::read_write> signedVolume [[texture(1)]], texture2d_array<uint, access::read_write> layerCounters [[texture(2)]])"
        in generated_code
    )
    assert "uint oldValue = image.read(uint2(pixel)).x;" in generated_code
    assert "image.write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "int oldValue = image.read(uint3(voxel)).x;" in generated_code
    assert "image.write(int4(oldValue + value), uint3(voxel));" in generated_code
    assert (
        "uint oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z)).x;"
        in generated_code
    )
    assert (
        "image.write(uint4(oldValue + value), uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_default_float_image_scalar_and_vector_load_store():
    shader = """
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

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "float scalarOld = image.read(uint2(pixel)).x;" in generated_code
    assert "image.write(float4(scalarOld + value), uint2(pixel));" in generated_code
    assert "float4 vectorOld = image.read(uint2(pixel));" in generated_code
    assert "image.write(vectorOld + value, uint2(pixel));" in generated_code
    assert "float4 vectorOld = image.read(uint2(pixel)).x;" not in generated_code
    assert "image.write(float4(vectorOld + value), uint2(pixel));" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_rg_image_scalar_and_vector_load_store():
    shader = """
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "float oldValue = image.read(uint2(pixel)).x;" in generated_code
    assert "uint oldValue = image.read(uint2(pixel)).x;" in generated_code
    assert (
        "image.write(float4(oldValue + value, 0.0, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert (
        "image.write(uint4(oldValue + value, 0u, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert "float2 oldValue = image.read(uint2(pixel)).xy;" in generated_code
    assert "uint2 oldValue = image.read(uint2(pixel)).xy;" in generated_code
    assert (
        "image.write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert (
        "image.write(uint4(oldValue + value, 0u, 0u), uint2(pixel));" in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_explicit_scalar_image_formats():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<float, access::read_write> scalarFloat [[texture(0)]], texture3d<int, access::read_write> signedVolume [[texture(1)]], texture2d_array<uint, access::read_write> unsignedLayers [[texture(2)]])"
        in generated_code
    )
    assert (
        "float touchFloat(texture2d<float, access::read_write> image, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "int touchSigned(texture3d<int, access::read_write> image, int3 voxel, int value)"
        in generated_code
    )
    assert (
        "uint touchUnsigned(texture2d_array<uint, access::read_write> image, int3 pixelLayer, uint value)"
        in generated_code
    )
    assert "float oldValue = image.read(uint2(pixel)).x;" in generated_code
    assert "image.write(float4(oldValue + value), uint2(pixel));" in generated_code
    assert "int oldValue = image.read(uint3(voxel)).x;" in generated_code
    assert "image.write(int4(oldValue + value), uint3(voxel));" in generated_code
    assert (
        "uint oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z)).x;"
        in generated_code
    )
    assert (
        "image.write(uint4(oldValue + value), uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "[[r32" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_explicit_rg_image_formats():
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

        compute {
            void main() {
                vec2 a = touchFloat(rgFloat, ivec2(0, 1), vec2(0.5, 1.5));
                ivec2 b = touchSigned(rgSigned, ivec3(1, 2, 3), ivec2(-4, 5));
                uvec2 c = touchUnsigned(rgUnsigned, ivec3(4, 5, 6), uvec2(7, 8));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<float, access::read_write> rgFloat [[texture(0)]], texture3d<int, access::read_write> rgSigned [[texture(1)]], texture2d_array<uint, access::read_write> rgUnsigned [[texture(2)]])"
        in generated_code
    )
    assert (
        "float2 touchFloat(texture2d<float, access::read_write> image, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "int2 touchSigned(texture3d<int, access::read_write> image, int3 voxel, int2 value)"
        in generated_code
    )
    assert (
        "uint2 touchUnsigned(texture2d_array<uint, access::read_write> image, int3 pixelLayer, uint2 value)"
        in generated_code
    )
    assert "float2 oldValue = image.read(uint2(pixel)).xy;" in generated_code
    assert "int2 oldValue = image.read(uint3(voxel)).xy;" in generated_code
    assert (
        "uint2 oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z)).xy;"
        in generated_code
    )
    assert (
        "image.write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "image.write(int4(oldValue + value, 0, 0), uint3(voxel));" in generated_code
    assert (
        "image.write(uint4(oldValue + value, 0u, 0u), uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "texture2d<float4, access::read_write> rgFloat" not in generated_code
    assert ".read(uint2(pixel)).x;" not in generated_code
    assert "image.write(oldValue + value" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_explicit_narrow_rg_image_formats():
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

        compute {
            void main() {
                vec2 a = touchFloat(rg8Float, ivec2(0, 1), vec2(0.5, 1.5));
                vec2 b = touchHalf(rg16Half, ivec3(1, 2, 3), vec2(2.5, 3.5));
                ivec2 c = touchSigned(rg16Signed, ivec3(2, 3, 4), ivec2(-4, 5));
                uvec2 d = touchUnsigned(rg8Unsigned, ivec2(4, 5), uvec2(7, 8));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "texture2d<float, access::read_write> rg8Float [[texture(0)]]" in generated_code
    )
    assert (
        "texture2d<float, access::read_write> rg8Snorm [[texture(1)]]" in generated_code
    )
    assert (
        "texture3d<float, access::read_write> rg16Float [[texture(2)]]"
        in generated_code
    )
    assert (
        "texture2d<float, access::read_write> rg16Snorm [[texture(3)]]"
        in generated_code
    )
    assert (
        "texture2d_array<float, access::read_write> rg16Half [[texture(4)]]"
        in generated_code
    )
    assert (
        "texture2d<int, access::read_write> rg8Signed [[texture(5)]]" in generated_code
    )
    assert (
        "texture3d<int, access::read_write> rg16Signed [[texture(6)]]" in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> rg8Unsigned [[texture(7)]]"
        in generated_code
    )
    assert (
        "texture2d_array<uint, access::read_write> rg16Unsigned [[texture(8)]]"
        in generated_code
    )
    assert (
        "float2 touchFloat(texture2d<float, access::read_write> image, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float2 touchHalf(texture2d_array<float, access::read_write> image, int3 pixelLayer, float2 value)"
        in generated_code
    )
    assert (
        "int2 touchSigned(texture3d<int, access::read_write> image, int3 voxel, int2 value)"
        in generated_code
    )
    assert (
        "uint2 touchUnsigned(texture2d<uint, access::read_write> image, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float2 oldValue = image.read(uint2(pixel)).xy;" in generated_code
    assert (
        "float2 oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z)).xy;"
        in generated_code
    )
    assert "int2 oldValue = image.read(uint3(voxel)).xy;" in generated_code
    assert "uint2 oldValue = image.read(uint2(pixel)).xy;" in generated_code
    assert (
        "image.write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert (
        "image.write(float4(oldValue + value, 0.0, 0.0), uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "image.write(int4(oldValue + value, 0, 0), uint3(voxel));" in generated_code
    assert (
        "image.write(uint4(oldValue + value, 0u, 0u), uint2(pixel));" in generated_code
    )
    assert "texture2d<float4, access::read_write> rg8Float" not in generated_code
    assert ".read(uint2(pixel)).x;" not in generated_code
    assert "image.write(oldValue + value" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_explicit_rgba_float_image_formats():
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

        compute {
            void main() {
                vec4 a = touchColor(rgba8Color, ivec2(0, 1), vec4(0.5));
                vec4 b = touchHalf(rgba16Half, ivec3(1, 2, 3), vec4(0.25));
                vec4 c = touchFloat(rgba32Float, ivec3(2, 3, 4), vec4(1.0));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "texture2d<float, access::read_write> rgba8Color [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture2d<float, access::read_write> rgba8Snorm [[texture(1)]]"
        in generated_code
    )
    assert (
        "texture3d<float, access::read_write> rgba16Color [[texture(2)]]"
        in generated_code
    )
    assert (
        "texture2d<float, access::read_write> rgba16Snorm [[texture(3)]]"
        in generated_code
    )
    assert (
        "texture2d_array<float, access::read_write> rgba16Half [[texture(4)]]"
        in generated_code
    )
    assert (
        "texture3d<float, access::read_write> rgba32Float [[texture(5)]]"
        in generated_code
    )
    assert (
        "float4 touchColor(texture2d<float, access::read_write> image, int2 pixel, float4 value)"
        in generated_code
    )
    assert (
        "float4 touchHalf(texture2d_array<float, access::read_write> image, int3 pixelLayer, float4 value)"
        in generated_code
    )
    assert (
        "float4 touchFloat(texture3d<float, access::read_write> image, int3 voxel, float4 value)"
        in generated_code
    )
    assert (
        "float4 typedOverride(texture2d<float, access::read_write> image, int2 pixel, float4 value)"
        in generated_code
    )
    assert "float4 oldValue = image.read(uint2(pixel));" in generated_code
    assert (
        "float4 oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "float4 oldValue = image.read(uint3(voxel));" in generated_code
    assert "image.write(oldValue + value, uint2(pixel));" in generated_code
    assert (
        "image.write(oldValue + value, uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "image.write(oldValue + value, uint3(voxel));" in generated_code
    assert "texture2d<int, access::read_write> image" not in generated_code
    assert "image.read(uint2(pixel)).x" not in generated_code
    assert "image.read(uint2(pixel)).xy" not in generated_code
    assert "image.write(float4(oldValue + value" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_formatted_image_arrays_preserve_format_metadata():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "array<texture2d<uint, access::read_write>, 2> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 3> rgPairs [[texture(2)]]"
        in generated_code
    )
    assert (
        "array<texture3d<float, access::read_write>, 2> rgbaVolumes [[texture(5)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(7)]]"
        in generated_code
    )
    assert "texture2d<float> sampled [[texture(8)]]" in generated_code
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, 2> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float4 touchVolumes(array<texture3d<float, access::read_write>, 2> images, int3 voxel, float4 value)"
        in generated_code
    )
    assert "uint oldValue = images[1].read(uint2(pixel)).x;" in generated_code
    assert "images[0].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[1].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "float4 oldValue = images[1].read(uint3(voxel));" in generated_code
    assert "images[0].write(oldValue + value, uint3(voxel));" in generated_code
    assert (
        "texture2d<float, access::read_write> counters [[texture(0)]]"
        not in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 2> counters" not in generated_code
    )
    assert "images[1].read(uint2(pixel));" not in generated_code
    assert "images[2].read(uint2(pixel)).x;" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_rg_image_arrays_respect_scalar_and_vector_context():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "array<texture2d<float, access::read_write>, 3> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 2> rgUnsignedImages [[texture(3)]]"
        in generated_code
    )
    assert (
        "float scalarFloat(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloat(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsigned(array<texture2d<uint, access::read_write>, 2> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsigned(array<texture2d<uint, access::read_write>, 2> images, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[1].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[0].write(float4(oldValue + value, 0.0, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[1].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "uint oldValue = images[1].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[0].write(uint4(oldValue + value, 0u, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert "uint2 oldValue = images[1].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[0].write(uint4(oldValue + value, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert "float oldValue = images[1].read(uint2(pixel));" not in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).x;" not in generated_code
    assert "uint oldValue = images[1].read(uint2(pixel));" not in generated_code
    assert "uint2 oldValue = images[1].read(uint2(pixel)).x;" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_inferred_rg_image_arrays_respect_scalar_context():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "constant int COUNT = 3;" in generated_code
    assert "constant int LAYER = COUNT - 1;" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 3> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, COUNT> rgUnsignedImages [[texture(3)]]"
        in generated_code
    )
    assert (
        "texture2d<float, access::read_write> afterImages [[texture(6)]]"
        in generated_code
    )
    assert (
        "float scalarFloat(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloat(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsigned(array<texture2d<uint, access::read_write>, COUNT> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsigned(array<texture2d<uint, access::read_write>, COUNT> images, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[LAYER].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[0].write(float4(oldValue + value, 0.0, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[1].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "uint oldValue = images[LAYER].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[0].write(uint4(oldValue + value, 0u, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert "uint2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[1].write(uint4(oldValue + value, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 1> rgFloatImages"
        not in generated_code
    )
    assert (
        "texture2d<float, access::read_write> afterImages [[texture(3)]]"
        not in generated_code
    )
    assert "float oldValue = images[LAYER].read(uint2(pixel));" not in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).x;" not in generated_code
    assert "uint oldValue = images[LAYER].read(uint2(pixel));" not in generated_code
    assert "uint2 oldValue = images[2].read(uint2(pixel)).x;" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_transitive_rg_image_arrays_share_call_site_size():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 4> rgUnsignedImages [[texture(4)]]"
        in generated_code
    )
    assert (
        "texture2d<float, access::read_write> afterImages [[texture(8)]]"
        in generated_code
    )
    assert (
        "float scalarFloatDeep(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float scalarFloatMid(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatDeep(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatMid(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedDeep(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedDeep(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[1].write(float4(oldValue + value, 0.0, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[0].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "uint oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[1].write(uint4(oldValue + value, 0u, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert "uint2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[0].write(uint4(oldValue + value, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert (
        "float2 vectorFloatDeep(array<texture2d<float, access::read_write>, 3> images"
        not in generated_code
    )
    assert (
        "uint2 vectorUnsignedDeep(array<texture2d<uint, access::read_write>, 3> images"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_fixed_rg_image_array_parameters_size_unsized_globals():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 4> rgUnsignedImages [[texture(4)]]"
        in generated_code
    )
    assert (
        "float scalarFloatFixed(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatFixed(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedFixed(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert "uint oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert "uint2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 1> rgFloatImages"
        not in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 1> rgUnsignedImages"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_const_sized_rg_image_array_parameters_size_unsized_globals():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "constant int COUNT = 4;" in generated_code
    assert "constant int LAST = COUNT - 1;" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 4> rgUnsignedImages [[texture(4)]]"
        in generated_code
    )
    assert (
        "float scalarFloatFixed(array<texture2d<float, access::read_write>, COUNT> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatFixed(array<texture2d<float, access::read_write>, COUNT> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(array<texture2d<uint, access::read_write>, COUNT> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedFixed(array<texture2d<uint, access::read_write>, COUNT> images, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[LAST].read(uint2(pixel)).x;" in generated_code
    assert "uint oldValue = images[LAST].read(uint2(pixel)).x;" in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert "uint2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "float scalarFloatFixed(array<texture2d<float, access::read_write>, 4> images"
        not in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 1> rgFloatImages"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_expr_sized_rg_image_array_parameters_size_unsized_globals():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "constant int COUNT = 3;" in generated_code
    assert "constant int UINT_COUNT = 2;" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 4> rgUnsignedImages [[texture(4)]]"
        in generated_code
    )
    assert (
        "float scalarFloatFixed(array<texture2d<float, access::read_write>, COUNT + 1> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatFixed(array<texture2d<float, access::read_write>, COUNT + 1> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(array<texture2d<uint, access::read_write>, UINT_COUNT * 2> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedFixed(array<texture2d<uint, access::read_write>, UINT_COUNT * 2> images, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[COUNT].read(uint2(pixel)).x;" in generated_code
    assert "uint oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert "uint2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "float scalarFloatFixed(array<texture2d<float, access::read_write>, 4> images"
        not in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 1> rgFloatImages"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_conflicting_fixed_rg_image_array_sizes_raise():
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
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_direct_rg_image_array_index_conflicts_with_fixed_parameter_size():
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
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_direct_rg_image_array_index_within_fixed_parameter_size():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 4> rgUnsignedImages [[texture(4)]]"
        in generated_code
    )
    assert (
        "float touchFour(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "uint touchUnsignedFour(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float directFloat = rgFloatImages[2].read(uint2(pixel)).x;" in generated_code
    )
    assert (
        "uint directUint = rgUnsignedImages[1].read(uint2(pixel)).x;" in generated_code
    )
    assert "float oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert "uint oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 3> rgFloatImages"
        not in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 1> rgFloatImages"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_fixed_rg_image_array_global_conflicts_with_fixed_parameter_size():
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
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_fixed_rg_image_array_global_widens_unsized_parameter_size():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "float touchUnsized(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float value)"
        in generated_code
    )
    assert "float oldValue = images[2].read(uint2(pixel)).x;" in generated_code
    assert (
        "float touchUnsized(array<texture2d<float, access::read_write>, 3> images"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_fixed_rg_image_array_global_direct_index_out_of_bounds_raises():
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
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_fixed_rg_image_array_parameter_direct_index_out_of_bounds_raises():
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
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_fixed_rg_image_array_global_const_index_out_of_bounds_raises():
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
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_fixed_rg_image_array_parameter_const_index_out_of_bounds_raises():
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
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_fixed_rg_image_array_const_index_within_bounds_generates():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "constant int COUNT = 4;" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "float touch(array<texture2d<float, access::read_write>, 4> images, int2 pixel)"
        in generated_code
    )
    assert "return images[COUNT - 1].read(uint2(pixel)).x;" in generated_code
    assert (
        "float direct = rgFloatImages[COUNT - 1].read(uint2(pixel)).x;"
        in generated_code
    )
    assert (
        "float touch(array<texture2d<float, access::read_write>, 5> images"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code


def test_metal_fixed_rg_image_array_shadowed_const_index_stays_dynamic():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "constant int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "float touch(array<texture2d<float, access::read_write>, 4> images, int2 pixel)"
        in generated_code
    )
    assert "return images[COUNT].read(uint2(pixel)).x;" in generated_code
    assert "float direct = rgFloatImages[COUNT].read(uint2(pixel)).x;" in generated_code
    assert (
        "float touch(array<texture2d<float, access::read_write>, 5> images"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code


def test_metal_transitive_rg_image_array_shadowed_const_index_stays_dynamic():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "constant int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "float leaf(array<texture2d<float, access::read_write>, 4> images, int2 pixel)"
        in generated_code
    )
    assert (
        "float passThrough(array<texture2d<float, access::read_write>, 4> images, int2 pixel)"
        in generated_code
    )
    assert "return images[COUNT].read(uint2(pixel)).x;" in generated_code
    assert "float sampled = images[COUNT].read(uint2(pixel)).x;" in generated_code
    assert (
        "float leaf(array<texture2d<float, access::read_write>, 5> images"
        not in generated_code
    )
    assert (
        "float passThrough(array<texture2d<float, access::read_write>, 5> images"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code


def test_metal_transitive_rg_image_array_unshadowed_const_index_conflict_raises():
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
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_shadowed_rg_image_array_constant_keeps_scalar_context():
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

        compute {
            void main() {
                float sf = scalarFloat(rgFloatImages, ivec2(0, 1), 0.25);
                uint su = scalarUnsigned(rgUnsignedImages, ivec2(4, 5), 7u);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "constant int LAYER = 3;" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 1> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 1> rgUnsignedImages [[texture(1)]]"
        in generated_code
    )
    assert (
        "texture2d<float, access::read_write> afterImages [[texture(2)]]"
        in generated_code
    )
    assert (
        "float scalarFloat(array<texture2d<float, access::read_write>, 1> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "uint scalarUnsigned(array<texture2d<uint, access::read_write>, 1> images, int2 pixel, uint value)"
        in generated_code
    )
    assert "int LAYER = 0;" in generated_code
    assert "float oldValue = images[LAYER].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[0].write(float4(oldValue + value, 0.0, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "uint oldValue = images[LAYER].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[0].write(uint4(oldValue + value, 0u, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages"
        not in generated_code
    )
    assert (
        "texture2d<float, access::read_write> afterImages [[texture(4)]]"
        not in generated_code
    )
    assert "float oldValue = images[LAYER].read(uint2(pixel));" not in generated_code
    assert "uint oldValue = images[LAYER].read(uint2(pixel));" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_formatted_image_arrays_preserve_expression_sizes():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "array<texture2d<uint, access::read_write>, (1 + 1) * 2> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, +3> rgPairs [[texture(4)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(7)]]"
        in generated_code
    )
    assert "texture2d<float> sampled [[texture(8)]]" in generated_code
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, (1 + 1) * 2> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(array<texture2d<float, access::read_write>, +3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[2].read(uint2(pixel)).x;" in generated_code
    assert "images[1].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[1].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert "float2 b = touchPairs(rgPairs, int2(2, 3), float2(0.5));" in generated_code
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(4)]]"
        not in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, (1 + 1) * 2> counters"
        not in generated_code
    )
    assert "2 + 1 * 2" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_unsized_formatted_image_arrays_preserve_format_metadata():
    shader = """
    shader UnsizedFormattedImageArrays {
        image2D counters @r32ui[];
        image2D rgPairs @rg16f[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
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

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "array<texture2d<uint, access::read_write>, 4> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 3> rgPairs [[texture(4)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(7)]]"
        in generated_code
    )
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert "images[1].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[0].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> counters [[texture(0)]]"
        not in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 1> counters" not in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 1> rgPairs" not in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> images, int2 pixel" not in generated_code
    )
    assert (
        "texture2d<float, access::read_write> images, int2 pixel" not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_formatted_image_arrays_infer_named_constant_size():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int COUNT = 3;" in generated_code
    assert "constant int LAYER = COUNT - 1;" in generated_code
    assert (
        "array<texture2d<uint, access::read_write>, COUNT> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 3> rgPairs [[texture(3)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(6)]]"
        in generated_code
    )
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, COUNT> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[LAYER].read(uint2(pixel)).x;" in generated_code
    assert "images[1].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "float2 oldValue = images[LAYER].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[0].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert "float2 b = touchPairs(rgPairs, int2(2, 3), float2(0.5));" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 1> rgPairs" not in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(2)]]"
        not in generated_code
    )
    assert (
        "texture2d<float, access::read_write> counters [[texture(0)]]"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_formatted_image_arrays_ignore_shadowed_local_constant():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int LAYER = 3;" in generated_code
    assert (
        "array<texture2d<uint, access::read_write>, 1> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(1)]]"
        in generated_code
    )
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, 1> images, int2 pixel, uint value)"
        in generated_code
    )
    assert "int LAYER = 0;" in generated_code
    assert "uint oldValue = images[LAYER].read(uint2(pixel)).x;" in generated_code
    assert "images[0].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert (
        "array<texture2d<uint, access::read_write>, 4> counters" not in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(4)]]"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_formatted_image_arrays_infer_transitive_helper_size():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "array<texture2d<uint, access::read_write>, 4> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 3> rgPairs [[texture(4)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(7)]]"
        in generated_code
    )
    assert (
        "uint touchCountersDeep(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint touchCountersMid(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairsDeep(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float2 touchPairsMid(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert "images[1].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[0].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "return touchCountersDeep(images, pixel, value);" in generated_code
    assert "return touchPairsDeep(images, pixel, value);" in generated_code
    assert "uint a = touchCountersMid(counters, int2(1, 2), 3);" in generated_code
    assert (
        "float2 b = touchPairsMid(rgPairs, int2(2, 3), float2(0.5));" in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 1> counters" not in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 1> rgPairs" not in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(2)]]"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_formatted_image_arrays_ignore_unsupported_indices():
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

    dynamic_code = MetalCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = MetalCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert (
        "array<texture2d<uint, access::read_write>, 1> counters [[texture(0)]]"
        in dynamic_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(1)]]"
        in dynamic_code
    )
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, 1> images, int layer, int2 pixel, uint value)"
        in dynamic_code
    )
    assert "uint oldValue = images[layer].read(uint2(pixel)).x;" in dynamic_code
    assert "images[0].write(uint4(oldValue + value), uint2(pixel));" in dynamic_code
    assert "uint a = touchCounters(counters, 0, int2(1, 2), 3);" in dynamic_code
    assert "array<texture2d<uint, access::read_write>, 2> counters" not in dynamic_code
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(2)]]"
        not in dynamic_code
    )
    assert "imageLoad(" not in dynamic_code
    assert "imageStore(" not in dynamic_code

    assert (
        "array<texture2d<uint, access::read_write>, 1> counters [[texture(0)]]"
        in negative_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(1)]]"
        in negative_code
    )
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, 1> images, int2 pixel, uint value)"
        in negative_code
    )
    assert "uint oldValue = images[-1].read(uint2(pixel)).x;" in negative_code
    assert "images[0].write(uint4(oldValue + value), uint2(pixel));" in negative_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in negative_code
    assert "array<texture2d<uint, access::read_write>, 0> counters" not in negative_code
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(0)]]"
        not in negative_code
    )
    assert "imageLoad(" not in negative_code
    assert "imageStore(" not in negative_code


def test_metal_formatted_image_arrays_ignore_function_call_indices():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "array<texture2d<uint, access::read_write>, 1> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(1)]]"
        in generated_code
    )
    assert "int getLayer()" in generated_code
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, 1> images, int2 pixel, uint value)"
        in generated_code
    )
    assert "uint oldValue = images[getLayer()].read(uint2(pixel)).x;" in generated_code
    assert "images[0].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert (
        "array<texture2d<uint, access::read_write>, 2> counters" not in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(2)]]"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_formatted_image_arrays_infer_local_constant_alias_size():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int GLOBAL = 2;" in generated_code
    assert (
        "array<texture2d<uint, access::read_write>, 4> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(4)]]"
        in generated_code
    )
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint value)"
        in generated_code
    )
    assert "const int LOCAL = GLOBAL + 1;" in generated_code
    assert "uint oldValue = images[LOCAL].read(uint2(pixel)).x;" in generated_code
    assert "images[0].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert (
        "array<texture2d<uint, access::read_write>, 1> counters" not in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(1)]]"
        not in generated_code
    )
    assert "    int LOCAL = GLOBAL + 1;" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_explicit_scalar_float_image_formats():
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

        compute {
            void main() {
                float a = touchR8(normalized8, ivec2(0, 1), 0.125);
                float b = touchR16(normalized16, ivec3(1, 2, 3), 0.25);
                float c = touchR16f(halfLayers, ivec3(4, 5, 6), 0.5);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<float, access::read_write> normalized8 [[texture(0)]], texture3d<float, access::read_write> normalized16 [[texture(1)]], texture2d_array<float, access::read_write> halfLayers [[texture(2)]], texture2d<float, access::read_write> signedNormalized [[texture(3)]])"
        in generated_code
    )
    assert (
        "float touchR8(texture2d<float, access::read_write> image, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float touchR16(texture3d<float, access::read_write> image, int3 voxel, float value)"
        in generated_code
    )
    assert (
        "float touchR16f(texture2d_array<float, access::read_write> image, int3 pixelLayer, float value)"
        in generated_code
    )
    assert "float oldValue = image.read(uint2(pixel)).x;" in generated_code
    assert "float oldValue = image.read(uint3(voxel)).x;" in generated_code
    assert (
        "float oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z)).x;"
        in generated_code
    )
    assert "image.write(float4(oldValue + value), uint2(pixel));" in generated_code
    assert "image.write(float4(oldValue + value), uint3(voxel));" in generated_code
    assert (
        "image.write(float4(oldValue + value), uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert ".read(uint2(pixel));" not in generated_code
    assert "image.write(oldValue + value" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_explicit_narrow_integer_image_formats():
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

        compute {
            void main() {
                int a = loadSigned8(signed8, ivec2(0, 1), -2);
                int b = loadSigned16(signed16, ivec3(1, 2, 3), -4);
                uint c = loadUnsigned8(unsigned8, ivec2(2, 3), 5);
                uint d = loadUnsigned16(unsigned16, ivec3(4, 5, 6), 7);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<int, access::read_write> signed8 [[texture(0)]], texture3d<int, access::read_write> signed16 [[texture(1)]], texture2d<uint, access::read_write> unsigned8 [[texture(2)]], texture2d_array<uint, access::read_write> unsigned16 [[texture(3)]])"
        in generated_code
    )
    assert (
        "int loadSigned8(texture2d<int, access::read_write> image, int2 pixel, int value)"
        in generated_code
    )
    assert (
        "int loadSigned16(texture3d<int, access::read_write> image, int3 voxel, int value)"
        in generated_code
    )
    assert (
        "uint loadUnsigned8(texture2d<uint, access::read_write> image, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint loadUnsigned16(texture2d_array<uint, access::read_write> image, int3 pixelLayer, uint value)"
        in generated_code
    )
    assert "int oldValue = image.read(uint2(pixel)).x;" in generated_code
    assert "int oldValue = image.read(uint3(voxel)).x;" in generated_code
    assert "uint oldValue = image.read(uint2(pixel)).x;" in generated_code
    assert (
        "uint oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z)).x;"
        in generated_code
    )
    assert "image.write(int4(oldValue + value), uint2(pixel));" in generated_code
    assert "image.write(int4(oldValue + value), uint3(voxel));" in generated_code
    assert "image.write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert (
        "image.write(uint4(oldValue + value), uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "texture2d<float, access::read_write> signed8" not in generated_code
    assert "texture3d<float, access::read_write> signed16" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_explicit_integer_image_formats_use_texture_atomics():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<uint, access::read_write> unsignedCounters [[texture(0)]], texture2d<int, access::read_write> signedCounters [[texture(1)]], texture3d<uint, access::read_write> unsignedVolume [[texture(2)]], texture2d_array<int, access::read_write> signedLayers [[texture(3)]])"
        in generated_code
    )
    assert (
        "uint addUnsigned(texture2d<uint, access::read_write> image, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "int minSigned(texture2d<int, access::read_write> image, int2 pixel, int value)"
        in generated_code
    )
    assert (
        "uint swapVolume(texture3d<uint, access::read_write> image, int3 voxel, uint expected, uint value)"
        in generated_code
    )
    assert (
        "int exchangeLayer(texture2d_array<int, access::read_write> image, int3 pixelLayer, int value)"
        in generated_code
    )
    assert "return image.atomic_fetch_add(uint2(pixel), value).x;" in generated_code
    assert "return image.atomic_fetch_min(uint2(pixel), value).x;" in generated_code
    assert (
        "return imageAtomicCompSwap_uimage3D(image, voxel, expected, value);"
        in generated_code
    )
    assert (
        "return image.atomic_exchange(uint2(pixelLayer.xy), uint(pixelLayer.z), value).x;"
        in generated_code
    )
    assert "imageAtomicAdd(" not in generated_code
    assert "imageAtomicMin(" not in generated_code
    assert "imageAtomicExchange(" not in generated_code
    assert "imageAtomicCompSwap(image" not in generated_code


def test_metal_explicit_vector_integer_image_formats():
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

        compute {
            void main() {
                uvec4 a = touchUnsigned(unsignedColor, ivec2(0, 1), uvec4(1, 2, 3, 4));
                ivec4 b = touchSigned(signedVolume, ivec3(1, 2, 3), ivec4(-1, -2, -3, -4));
                uvec4 c = touchLayers(unsignedLayers, ivec3(4, 5, 6), uvec4(5, 6, 7, 8));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<uint, access::read_write> unsignedColor [[texture(0)]], texture3d<int, access::read_write> signedVolume [[texture(1)]], texture2d_array<uint, access::read_write> unsignedLayers [[texture(2)]])"
        in generated_code
    )
    assert (
        "uint4 touchUnsigned(texture2d<uint, access::read_write> image, int2 pixel, uint4 value)"
        in generated_code
    )
    assert (
        "int4 touchSigned(texture3d<int, access::read_write> image, int3 voxel, int4 value)"
        in generated_code
    )
    assert (
        "uint4 touchLayers(texture2d_array<uint, access::read_write> image, int3 pixelLayer, uint4 value)"
        in generated_code
    )
    assert "uint4 oldValue = image.read(uint2(pixel));" in generated_code
    assert "int4 oldValue = image.read(uint3(voxel));" in generated_code
    assert (
        "uint4 oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "image.write(oldValue + value, uint2(pixel));" in generated_code
    assert "image.write(oldValue + value, uint3(voxel));" in generated_code
    assert (
        "image.write(oldValue + value, uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "texture2d<float, access::read_write> unsignedColor" not in generated_code
    assert "texture3d<float, access::read_write> signedVolume" not in generated_code
    assert ".read(uint2(pixel)).x" not in generated_code
    assert "uint4(oldValue + value)" not in generated_code
    assert "int4(oldValue + value)" not in generated_code


def test_metal_texture_array_helper_parameter_uses_resource_array():
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
                return sampleLayer(textures, input.layer, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "float4 sampleLayer(array<texture2d<float>, 4> textures" in generated_code
    assert (
        "textures[layer].sample(sampler(mag_filter::linear, min_filter::linear), uv)"
        in generated_code
    )
    assert "texture2d<float> textures[4] [[stage_in]]" not in generated_code


def test_metal_texture_array_helper_parameter_uses_indexed_sampler_array():
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
                return sampleLayer(textures, samplers, input.layer, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 4> textures, array<sampler, 4> samplers"
        in generated_code
    )
    assert "textures[layer].sample(samplers[layer], uv)" in generated_code
    assert "sampler samplers[4] [[stage_in]]" not in generated_code


def test_metal_unsized_texture_and_sampler_arrays_infer_helper_size():
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
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "array<texture2d<float>, 3> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(3)]]" in generated_code
    assert "array<sampler, 3> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 3> textures, array<sampler, 3> samplers"
        in generated_code
    )
    assert "textures[2].sample(samplers[2], uv)" in generated_code
    assert "textures[1].sample(samplers[1], uv)" in generated_code
    assert "array<texture2d<float>, 1> textures" not in generated_code
    assert "array<sampler, 1> samplers" not in generated_code


def test_metal_unsized_texture_and_sampler_arrays_infer_transitive_helper_size():
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
                return sampleMid(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "array<texture2d<float>, 5> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(5)]]" in generated_code
    assert "array<sampler, 5> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleDeep(array<texture2d<float>, 5> textures, array<sampler, 5> samplers"
        in generated_code
    )
    assert (
        "float4 sampleMid(array<texture2d<float>, 5> textures, array<sampler, 5> samplers"
        in generated_code
    )
    assert "textures[4].sample(samplers[4], uv)" in generated_code
    assert "textures[1].sample(samplers[1], uv)" in generated_code
    assert "sampleDeep(textures, samplers, uv)" in generated_code
    assert "sampleMid(textures, samplers, input.uv)" in generated_code
    assert "array<texture2d<float>, 1> textures" not in generated_code
    assert "array<sampler, 1> samplers" not in generated_code


def test_metal_unsized_texture_and_sampler_arrays_preserve_dynamic_indexing():
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
                return sampleLayer(textures, samplers, input.layer, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(4)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 4> textures, array<sampler, 4> samplers"
        in generated_code
    )
    assert "textures[layer].sample(samplers[layer], uv)" in generated_code
    assert "textures[3].sample(samplers[3], uv)" in generated_code
    assert "sampleLayer(textures, samplers, input.layer, input.uv)" in generated_code
    assert "array<texture2d<float>, 1> textures" not in generated_code
    assert "array<sampler, 1> samplers" not in generated_code


def test_metal_unsized_texture_and_sampler_arrays_ignore_unsupported_indices():
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
                return sampleLayer(textures, samplers, input.layer, input.uv) + texture(afterTexture, input.uv);
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
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    dynamic_code = MetalCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = MetalCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "array<texture2d<float>, 1> textures [[texture(0)]]" in dynamic_code
    assert "texture2d<float> afterTexture [[texture(1)]]" in dynamic_code
    assert "array<sampler, 1> samplers [[sampler(0)]]" in dynamic_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 1> textures, array<sampler, 1> samplers"
        in dynamic_code
    )
    assert "textures[layer].sample(samplers[layer], uv)" in dynamic_code
    assert "array<texture2d<float>, 2> textures" not in dynamic_code
    assert "texture2d<float> afterTexture [[texture(2)]]" not in dynamic_code

    assert "array<texture2d<float>, 1> textures [[texture(0)]]" in negative_code
    assert "texture2d<float> afterTexture [[texture(1)]]" in negative_code
    assert "array<sampler, 1> samplers [[sampler(0)]]" in negative_code
    assert "textures[-1].sample(samplers[-1], uv)" in negative_code
    assert "array<texture2d<float>, 0> textures" not in negative_code
    assert "texture2d<float> afterTexture [[texture(0)]]" not in negative_code


def test_metal_unsized_texture_and_sampler_arrays_infer_constant_expression_size():
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
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(4)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 4> textures, array<sampler, 4> samplers"
        in generated_code
    )
    assert "textures[1 + 2].sample(samplers[1 + 2], uv)" in generated_code
    assert "array<texture2d<float>, 1> textures" not in generated_code


def test_metal_unsized_texture_and_sampler_arrays_infer_named_constant_size():
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
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int BASE = 1;" in generated_code
    assert "constant int LAYER = BASE + 2;" in generated_code
    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(4)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 4> textures, array<sampler, 4> samplers"
        in generated_code
    )
    assert "textures[LAYER].sample(samplers[LAYER], uv)" in generated_code
    assert "array<texture2d<float>, 1> textures" not in generated_code


def test_metal_unsized_texture_and_sampler_arrays_ignore_shadowed_constant_name():
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
                return sampleLayer(textures, samplers, input.layer, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int LAYER = 3;" in generated_code
    assert "array<texture2d<float>, 1> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(1)]]" in generated_code
    assert "array<sampler, 1> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 1> textures, array<sampler, 1> samplers, int LAYER"
        in generated_code
    )
    assert "textures[LAYER].sample(samplers[LAYER], uv)" in generated_code
    assert "array<texture2d<float>, 4> textures" not in generated_code
    assert "texture2d<float> afterTexture [[texture(4)]]" not in generated_code


def test_metal_fixed_texture_array_global_conflicts_with_fixed_parameter_size():
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
                return sampleThree(textures, samplers, input.uv);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_metal_fixed_texture_array_global_widens_unsized_parameter_size():
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
                return sampleUnsized(textures, samplers, input.uv);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleUnsized(array<texture2d<float>, 4> textures, array<sampler, 4> samplers, float2 uv)"
        in generated_code
    )
    assert "textures[2].sample(samplers[2], uv)" in generated_code
    assert "sampleUnsized(textures, samplers, input.uv)" in generated_code
    assert (
        "float4 sampleUnsized(array<texture2d<float>, 3> textures" not in generated_code
    )


def test_metal_fixed_texture_array_global_direct_index_out_of_bounds_raises():
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
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_metal_fixed_texture_array_parameter_direct_index_out_of_bounds_raises():
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
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_metal_fixed_texture_array_global_const_index_out_of_bounds_raises():
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
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_metal_fixed_texture_array_parameter_const_index_out_of_bounds_raises():
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
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_metal_fixed_texture_array_const_index_and_shadowing_generate():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "constant int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleConst(array<texture2d<float>, 4> textures, array<sampler, 4> samplers, float2 uv)"
        in generated_code
    )
    assert (
        "float4 sampleShadowed(array<texture2d<float>, 4> textures, array<sampler, 4> samplers, float2 uv)"
        in generated_code
    )
    assert "textures[COUNT - 1].sample(samplers[COUNT - 1], uv)" in generated_code
    assert "textures[COUNT].sample(samplers[COUNT], uv)" in generated_code
    assert "sampleConst(textures, samplers, input.uv)" in generated_code
    assert "sampleShadowed(textures, samplers, input.uv)" in generated_code
    assert "array<texture2d<float>, 5> textures" not in generated_code
    assert "array<sampler, 5> samplers" not in generated_code


def test_metal_transitive_texture_array_shadowed_const_index_stays_dynamic():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "constant int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 leaf(array<texture2d<float>, 4> textures, array<sampler, 4> samplers, float2 uv)"
        in generated_code
    )
    assert (
        "float4 passThrough(array<texture2d<float>, 4> textures, array<sampler, 4> samplers, float2 uv)"
        in generated_code
    )
    assert "return textures[COUNT].sample(samplers[COUNT], uv);" in generated_code
    assert (
        "float4 sampled = textures[COUNT].sample(samplers[COUNT], uv);"
        in generated_code
    )
    assert "return sampled + leaf(textures, samplers, uv);" in generated_code
    assert "return passThrough(textures, samplers, input.uv);" in generated_code
    assert "array<texture2d<float>, 5> textures" not in generated_code
    assert "array<sampler, 5> samplers" not in generated_code


def test_metal_transitive_texture_array_unshadowed_const_index_conflict_raises():
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
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_metal_texture_array_helper_operation_variants():
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
                return sampleOps(textures, samplers, input.layer, input.uv, input.pixel);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleOps(array<texture2d<float>, 4> textures, array<sampler, 4> samplers, int layer, float2 uv, int2 pixel)"
        in generated_code
    )
    assert "textures[layer].sample(samplers[layer], uv, level(2.0))" in generated_code
    assert (
        "textures[layer].sample(samplers[layer], uv, gradient2d(float2(0.1), float2(0.2)))"
        in generated_code
    )
    assert "textures[layer].gather(samplers[layer], uv)" in generated_code
    assert "textures[layer].read(pixel, 0)" in generated_code
    assert (
        "sampleOps(textures, samplers, input.layer, input.uv, input.pixel)"
        in generated_code
    )
    assert "int layer [[stage_in]]" not in generated_code
    assert "float2 uv [[stage_in]]" not in generated_code
    assert "int2 pixel [[stage_in]]" not in generated_code


def test_metal_array_texture_types_split_coordinates_and_layers():
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
                float shadow = sampleShadowArray(shadowArray, shadowSampler, input.uvLayer, input.depth);
                float cube = sampleCubeShadow(cubeShadow, shadowSampler, input.direction, input.depth);
                return sampleArray(colorArray, arraySampler, input.uvLayer, input.pixelLayer) * shadow * cube;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "texture2d_array<float> colorArray [[texture(0)]]" in generated_code
    assert "depth2d_array<float> shadowArray [[texture(1)]]" in generated_code
    assert "depthcube<float> cubeShadow [[texture(2)]]" in generated_code
    assert "sampler arraySampler [[sampler(0)]]" in generated_code
    assert "sampler shadowSampler [[sampler(1)]]" in generated_code
    assert (
        "float4 sampleArray(texture2d_array<float> tex, sampler s, float3 uvLayer, int3 pixelLayer)"
        in generated_code
    )
    assert "tex.sample(s, uvLayer.xy, uint(uvLayer.z))" in generated_code
    assert "tex.sample(s, uvLayer.xy, uint(uvLayer.z), level(1.0))" in generated_code
    assert (
        "tex.sample(s, uvLayer.xy, uint(uvLayer.z), gradient2d(float2(0.1), float2(0.2)))"
        in generated_code
    )
    assert "tex.gather(s, uvLayer.xy, uint(uvLayer.z))" in generated_code
    assert "tex.read(pixelLayer.xy, uint(pixelLayer.z), 0)" in generated_code
    assert (
        "float sampleShadowArray(depth2d_array<float> tex, sampler s, float3 uvLayer, float depth)"
        in generated_code
    )
    assert "tex.sample_compare(s, uvLayer.xy, uint(uvLayer.z), depth)" in generated_code
    assert (
        "float sampleCubeShadow(depthcube<float> tex, sampler s, float3 direction, float depth)"
        in generated_code
    )
    assert "tex.sample_compare(s, direction, depth)" in generated_code
    assert ".sample(s, uvLayer)" not in generated_code
    assert ".sample_compare(s, uvLayer, depth)" not in generated_code


def test_metal_cube_array_and_multisample_texture_types():
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
                float shadow = sampleCubeArrayShadow(cubeArrayShadow, shadowSampler, input.cubeLayer, input.depth);
                return sampleCubeArray(cubeArray, cubeSampler, input.cubeLayer) * shadow + fetchMs(msTex, msArray, input.pixel, input.pixelLayer, input.sampleIndex);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "texturecube_array<float> cubeArray [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeArrayShadow [[texture(1)]]" in generated_code
    assert "texture2d_ms<float> msTex [[texture(2)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(3)]]" in generated_code
    assert "sampler cubeSampler [[sampler(0)]]" in generated_code
    assert "sampler shadowSampler [[sampler(1)]]" in generated_code
    assert (
        "float4 sampleCubeArray(texturecube_array<float> tex, sampler s, float4 cubeLayer)"
        in generated_code
    )
    assert "tex.sample(s, cubeLayer.xyz, uint(cubeLayer.w))" in generated_code
    assert (
        "tex.sample(s, cubeLayer.xyz, uint(cubeLayer.w), level(2.0))" in generated_code
    )
    assert (
        "float sampleCubeArrayShadow(depthcube_array<float> tex, sampler s, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth)"
        in generated_code
    )
    assert (
        "float4 fetchMs(texture2d_ms<float> tex, texture2d_ms_array<float> texArray, int2 pixel, int3 pixelLayer, int sampleIndex)"
        in generated_code
    )
    assert "tex.read(pixel, uint(sampleIndex))" in generated_code
    assert (
        "texArray.read(pixelLayer.xy, uint(pixelLayer.z), uint(sampleIndex))"
        in generated_code
    )
    assert "tex.sample(s, cubeLayer)" not in generated_code
    assert "tex.sample_compare(s, cubeLayer, depth)" not in generated_code


def test_metal_cube_array_texture_grad_gather_uses_cube_gradients():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texturecube_array<float> cubeArray [[texture(0)]]" in generated_code
    assert (
        "array<texturecube_array<float>, 4> cubeArrays [[texture(1)]]" in generated_code
    )
    assert "sampler cubeSampler [[sampler(0)]]" in generated_code
    assert "array<sampler, 4> cubeSamplers [[sampler(1)]]" in generated_code
    assert (
        "float4 sampleCubeArrayOps(texturecube_array<float> tex, sampler s, float4 cubeLayer, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "tex.sample(s, cubeLayer.xyz, uint(cubeLayer.w), gradientcube(ddx, ddy))"
        in generated_code
    )
    assert "tex.gather(s, cubeLayer.xyz, uint(cubeLayer.w))" in generated_code
    assert (
        "float4 sampleCubeArrayElements(array<texturecube_array<float>, 4> cubeArrays, array<sampler, 4> cubeSamplers, float4 cubeLayer, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "cubeArrays[2].sample(cubeSamplers[2], cubeLayer.xyz, uint(cubeLayer.w), gradientcube(ddx, ddy))"
        in generated_code
    )
    assert (
        "cubeArrays[3].gather(cubeSamplers[3], cubeLayer.xyz, uint(cubeLayer.w))"
        in generated_code
    )
    assert (
        "sampleCubeArrayOps(cubeArray, cubeSampler, input.cubeLayer, input.ddx, input.ddy)"
        in generated_code
    )
    assert (
        "sampleCubeArrayElements(cubeArrays, cubeSamplers, input.cubeLayer, input.ddx, input.ddy)"
        in generated_code
    )
    assert "gradient2d(ddx, ddy)" not in generated_code
    assert ".sample(s, cubeLayer, gradientcube" not in generated_code


def test_metal_texture_grad_uses_dimension_specific_gradient_options():
    shader = """
    shader MetalGradientFamily {
        sampler2D colorMap;
        samplerCube cubeMap;
        sampler3D volumeMap;
        sampler colorSampler;
        sampler cubeSampler;
        sampler volumeSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 direction @ TEXCOORD1;
            vec3 volumeUv @ TEXCOORD2;
            vec2 ddx2 @ TEXCOORD3;
            vec2 ddy2 @ TEXCOORD4;
            vec3 ddx3 @ TEXCOORD5;
            vec3 ddy3 @ TEXCOORD6;
        };

        vec4 sample2DGrad(sampler2D tex, sampler s, vec2 uv, vec2 ddx, vec2 ddy) {
            return textureGrad(tex, s, uv, ddx, ddy);
        }

        vec4 sampleCubeGrad(samplerCube tex, sampler s, vec3 direction, vec3 ddx, vec3 ddy) {
            return textureGrad(tex, s, direction, ddx, ddy);
        }

        vec4 sampleVolumeGrad(sampler3D tex, sampler s, vec3 volumeUv, vec3 ddx, vec3 ddy) {
            return textureGrad(tex, s, volumeUv, ddx, ddy);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return sample2DGrad(colorMap, colorSampler, input.uv, input.ddx2, input.ddy2)
                    + sampleCubeGrad(cubeMap, cubeSampler, input.direction, input.ddx3, input.ddy3)
                    + sampleVolumeGrad(volumeMap, volumeSampler, input.volumeUv, input.ddx3, input.ddy3);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texturecube<float> cubeMap [[texture(1)]]" in generated_code
    assert "texture3d<float> volumeMap [[texture(2)]]" in generated_code
    assert "sampler colorSampler [[sampler(0)]]" in generated_code
    assert "sampler cubeSampler [[sampler(1)]]" in generated_code
    assert "sampler volumeSampler [[sampler(2)]]" in generated_code
    assert "tex.sample(s, uv, gradient2d(ddx, ddy))" in generated_code
    assert "tex.sample(s, direction, gradientcube(ddx, ddy))" in generated_code
    assert "tex.sample(s, volumeUv, gradient3d(ddx, ddy))" in generated_code
    assert "gradient2d(ddx3, ddy3)" not in generated_code


def test_metal_texture_sampling_capability_descriptors():
    codegen = MetalCodeGen()

    assert codegen.texture_sampling_capabilities("texture2d<float>") == {
        "texture_type": "texture2d<float>",
        "gather": True,
        "gather_offset": True,
        "sample_offset": True,
        "projected_offset": True,
        "compare_offset": False,
        "gather_compare_offset": False,
    }
    assert codegen.texture_sampling_capabilities("texturecube<float>") == {
        "texture_type": "texturecube<float>",
        "gather": True,
        "gather_offset": False,
        "sample_offset": False,
        "projected_offset": False,
        "compare_offset": False,
        "gather_compare_offset": False,
    }
    assert codegen.texture_sampling_capabilities("texture2d_array<float>[4]") == {
        "texture_type": "texture2d_array<float>",
        "gather": True,
        "gather_offset": True,
        "sample_offset": True,
        "projected_offset": True,
        "compare_offset": False,
        "gather_compare_offset": False,
    }
    assert codegen.texture_sampling_capabilities("depth2d<float>") == {
        "texture_type": "depth2d<float>",
        "gather": False,
        "gather_offset": False,
        "sample_offset": False,
        "projected_offset": False,
        "compare_offset": True,
        "gather_compare_offset": True,
    }
    assert codegen.texture_sampling_capabilities("texture3d<float>") == {
        "texture_type": "texture3d<float>",
        "gather": False,
        "gather_offset": False,
        "sample_offset": True,
        "projected_offset": True,
        "compare_offset": False,
        "gather_compare_offset": False,
    }


def test_metal_texture_dimension_descriptors():
    codegen = MetalCodeGen()

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
        "texture2d_array<float>[4]",
        coordinate_dimension=3,
        offset_dimension=2,
        sample_offset_dimension=2,
        texel_fetch_offset_dimension=2,
        gather_offset_dimension=2,
        gradient_dimension=2,
        query_lod_coordinate_dimension=3,
    )
    expect(
        "texture1d_array<float>",
        coordinate_dimension=2,
        texel_fetch_offset_dimension=1,
    )
    expect(
        "depth2d<float>",
        coordinate_dimension=2,
        compare_offset_dimension=2,
        compare_lod_offset_dimension=2,
        compare_grad_offset_dimension=2,
        gather_compare_offset_dimension=2,
        gradient_dimension=2,
        query_lod_coordinate_dimension=2,
    )
    expect(
        "texturecube_array<float>",
        gradient_dimension=3,
        query_lod_coordinate_dimension=4,
    )
    expect("texture2d_ms_array<float>", coordinate_dimension=3)
    expect(
        "texture3d<float>",
        coordinate_dimension=3,
        offset_dimension=3,
        sample_offset_dimension=3,
        texel_fetch_offset_dimension=3,
        gradient_dimension=3,
        query_lod_coordinate_dimension=3,
    )
    expect(
        "texture3d<float, access::read_write>",
        coordinate_dimension=3,
        texel_fetch_offset_dimension=3,
    )


def test_metal_texture_gather_offset_variants_use_metal_overloads():
    shader = """
    shader GatherOffsetVariants {
        sampler2D colorMap;
        sampler2DArray layerMap;
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler linearSampler;
        sampler cubeSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec3 direction @ TEXCOORD2;
            vec4 cubeLayer @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
            ivec2 offset0 @ TEXCOORD5;
            ivec2 offset1 @ TEXCOORD6;
            ivec2 offset2 @ TEXCOORD7;
            ivec2 offset3 @ TEXCOORD8;
            int component @ TEXCOORD9;
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

        vec4 gatherArrayOffsets(
            sampler2DArray layers,
            sampler s,
            vec3 uvLayer,
            ivec2 offsets[4]
        ) {
            return textureGatherOffsets(layers, s, uvLayer, offsets, 2);
        }

        vec4 gatherCubeOps(
            samplerCube cubeMap,
            samplerCubeArray cubeArray,
            sampler s,
            vec3 direction,
            vec4 cubeLayer
        ) {
            return textureGather(cubeMap, s, direction, 2)
                + textureGather(cubeArray, s, cubeLayer, 3);
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
                ) + gatherCubeOps(
                    cubeMap,
                    cubeArray,
                    cubeSampler,
                    input.direction,
                    input.cubeLayer
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture2d_array<float> layerMap [[texture(1)]]" in generated_code
    assert "texturecube<float> cubeMap [[texture(2)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(3)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert "sampler cubeSampler [[sampler(1)]]" in generated_code
    assert "float4 green = tex.gather(s, uv, int2(0), component::y);" in generated_code
    assert (
        "component == 0 ? tex.gather(s, uv, int2(0), component::x) : "
        "component == 1 ? tex.gather(s, uv, int2(0), component::y) : "
        "component == 2 ? tex.gather(s, uv, int2(0), component::z) : "
        "tex.gather(s, uv, int2(0), component::w)" in generated_code
    )
    assert (
        "float4 offsetGather = tex.gather(s, uv, offset, component::w);"
        in generated_code
    )
    assert (
        "component == 0 ? tex.gather(s, uv, offset, component::x) : "
        "component == 1 ? tex.gather(s, uv, offset, component::y) : "
        "component == 2 ? tex.gather(s, uv, offset, component::z) : "
        "tex.gather(s, uv, offset, component::w)" in generated_code
    )
    assert (
        "float4(layers.gather(s, uvLayer.xy, uint(uvLayer.z), offset0, component::x).x, "
        "layers.gather(s, uvLayer.xy, uint(uvLayer.z), offset1, component::x).y, "
        "layers.gather(s, uvLayer.xy, uint(uvLayer.z), offset2, component::x).z, "
        "layers.gather(s, uvLayer.xy, uint(uvLayer.z), offset3, component::x).w)"
        in generated_code
    )
    assert (
        "float4 gatherArrayOffsets(texture2d_array<float> layers, sampler s, float3 uvLayer, int2 offsets[4])"
        in generated_code
    )
    assert (
        "return float4(layers.gather(s, uvLayer.xy, uint(uvLayer.z), offsets[0], component::z).x, "
        "layers.gather(s, uvLayer.xy, uint(uvLayer.z), offsets[1], component::z).y, "
        "layers.gather(s, uvLayer.xy, uint(uvLayer.z), offsets[2], component::z).z, "
        "layers.gather(s, uvLayer.xy, uint(uvLayer.z), offsets[3], component::z).w);"
        in generated_code
    )
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code
    assert "cubeMap.gather(s, direction, component::z)" in generated_code
    assert (
        "cubeArray.gather(s, cubeLayer.xyz, uint(cubeLayer.w), component::w)"
        in generated_code
    )
    assert "cubeMap.gather(s, direction, int2(0)" not in generated_code
    assert (
        "cubeArray.gather(s, cubeLayer.xyz, uint(cubeLayer.w), int2(0)"
        not in generated_code
    )


def test_metal_cube_texture_gather_offsets_emit_diagnostics():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    offset_diagnostic = (
        "/* unsupported Metal texture gather: textureGatherOffset "
        "offsets require 2D or 2D-array textures */ float4(0.0)"
    )
    offsets_diagnostic = (
        "/* unsupported Metal texture gather: textureGatherOffsets "
        "offsets require 2D or 2D-array textures */ float4(0.0)"
    )
    assert "texturecube<float> cubeMap [[texture(0)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert "sampler cubeSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 gatherCube(texturecube<float> tex, sampler s, float3 direction, int2 offset, int2 offset0, int2 offset1, int2 offset2, int2 offset3, int component)"
        in generated_code
    )
    assert (
        "float4 gatherCubeArray(texturecube_array<float> tex, sampler s, float4 cubeLayer, int2 offset, int2 offset0, int2 offset1, int2 offset2, int2 offset3, int component)"
        in generated_code
    )
    assert generated_code.count(offset_diagnostic) == 4
    assert generated_code.count(offsets_diagnostic) == 4
    assert ".gather(" not in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code


def test_metal_unsupported_dimension_texture_gather_emits_diagnostics():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported Metal texture gather: textureGather requires 2D, "
        "2D-array, cube, or cube-array textures */ float4(0.0)"
    )
    assert "texture1d<float> lineMap [[texture(0)]]" in generated_code
    assert "texture3d<float> volumeMap [[texture(1)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 gatherLine(texture1d<float> tex, sampler s, float u, int component)"
        in generated_code
    )
    assert (
        "float4 gatherVolume(texture3d<float> tex, sampler s, float3 volumeUv, int component)"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 6
    assert ".gather(" not in generated_code
    assert "textureGather(" not in generated_code


def test_metal_texture_gather_offsets_mix_literal_and_dynamic_offsets():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d_array<float> layerMap [[texture(0)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 gatherOps(texture2d_array<float> layers, sampler s, float3 uvLayer, int2 dynamic0, int2 dynamic1, int component)"
        in generated_code
    )
    assert "textureGatherOffsets(" not in generated_code
    components = {
        0: "component::x",
        1: "component::y",
        2: "component::z",
        3: "component::w",
    }
    for component, component_option in components.items():
        assert (
            f"layers.gather(s, uvLayer.xy, uint(uvLayer.z), int2(-1, 0), {component_option}).x"
            in generated_code
        )
        assert (
            f"layers.gather(s, uvLayer.xy, uint(uvLayer.z), dynamic0, {component_option}).y"
            in generated_code
        )
        assert (
            f"layers.gather(s, uvLayer.xy, uint(uvLayer.z), int2(1, -1), {component_option}).z"
            in generated_code
        )
        assert (
            f"layers.gather(s, uvLayer.xy, uint(uvLayer.z), dynamic1, {component_option}).w"
            in generated_code
        )
        if component < 3:
            assert f"component == {component} ? float4(" in generated_code
    assert (
        "gatherOps(layerMap, linearSampler, input.uvLayer, input.dynamic0, input.dynamic1, input.component)"
        in generated_code
    )


def test_metal_direct_stage_gather_offsets_use_input_members():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "fragment float4 fragment_main(FSInput input [[stage_in]], texture2d<float> colorMap [[texture(0)]], texture2d_array<float> layerMap [[texture(1)]], sampler linearSampler [[sampler(0)]])"
        in generated_code
    )
    assert (
        "float4 dynamic = (input.component == 0 ? colorMap.gather(linearSampler, input.uv, int2(0), component::x) : "
        "input.component == 1 ? colorMap.gather(linearSampler, input.uv, int2(0), component::y) : "
        "input.component == 2 ? colorMap.gather(linearSampler, input.uv, int2(0), component::z) : "
        "colorMap.gather(linearSampler, input.uv, int2(0), component::w));"
        in generated_code
    )
    assert (
        "input.component == 0 ? colorMap.gather(linearSampler, input.uv, input.offset, component::x) : "
        "input.component == 1 ? colorMap.gather(linearSampler, input.uv, input.offset, component::y) : "
        "input.component == 2 ? colorMap.gather(linearSampler, input.uv, input.offset, component::z) : "
        "colorMap.gather(linearSampler, input.uv, input.offset, component::w)"
        in generated_code
    )
    for component_option in [
        "component::x",
        "component::y",
        "component::z",
        "component::w",
    ]:
        assert (
            f"layerMap.gather(linearSampler, input.uvLayer.xy, uint(input.uvLayer.z), input.offset0, {component_option}).x"
            in generated_code
        )
        assert (
            f"layerMap.gather(linearSampler, input.uvLayer.xy, uint(input.uvLayer.z), input.offset1, {component_option}).y"
            in generated_code
        )
        assert (
            f"layerMap.gather(linearSampler, input.uvLayer.xy, uint(input.uvLayer.z), input.offset2, {component_option}).z"
            in generated_code
        )
        assert (
            f"layerMap.gather(linearSampler, input.uvLayer.xy, uint(input.uvLayer.z), input.offset3, {component_option}).w"
            in generated_code
        )
    assert "textureGather(" not in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code


def test_metal_texture_sample_offset_variants_use_sample_offsets():
    shader = """
    shader TextureSampleOffsets {
        sampler2D colorMap;
        sampler2DArray layerMap;
        samplerCube cubeMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec3 direction @ TEXCOORD2;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
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
            vec4 arrayPlain = textureOffset(layers, s, uvLayer, offset);
            vec4 arrayLod = textureLodOffset(layers, s, uvLayer, lod, offset);
            vec4 arrayGrad = textureGradOffset(layers, s, uvLayer, ddx, ddy, offset);
            return plain + lodSample + gradSample + arrayPlain + arrayLod + arrayGrad;
        }

        vec4 unsupportedCubeOffset(
            samplerCube tex,
            sampler s,
            vec3 direction,
            ivec2 offset
        ) {
            return textureOffset(tex, s, direction, offset);
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
                ) + unsupportedCubeOffset(
                    cubeMap,
                    linearSampler,
                    input.direction,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture2d_array<float> layerMap [[texture(1)]]" in generated_code
    assert "texturecube<float> cubeMap [[texture(2)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 offsetOps(texture2d<float> tex, texture2d_array<float> layers, sampler s, float2 uv, float3 uvLayer, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "float4 plain = tex.sample(s, uv, offset);" in generated_code
    assert "float4 lodSample = tex.sample(s, uv, level(lod), offset);" in generated_code
    assert (
        "float4 gradSample = tex.sample(s, uv, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "float4 arrayPlain = layers.sample(s, uvLayer.xy, uint(uvLayer.z), offset);"
        in generated_code
    )
    assert (
        "float4 arrayLod = layers.sample(s, uvLayer.xy, uint(uvLayer.z), level(lod), offset);"
        in generated_code
    )
    assert (
        "float4 arrayGrad = layers.sample(s, uvLayer.xy, uint(uvLayer.z), gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "return /* unsupported Metal texture offset: textureOffset offsets require 2D, 2D-array, or 3D textures */ float4(0.0);"
        in generated_code
    )
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code


def test_metal_texture_3d_sample_offsets_use_sample_offsets():
    shader = """
    shader Texture3DSampleOffsets {
        sampler3D volumeMap;
        sampler linearSampler;

        struct FSInput {
            vec3 uvw @ TEXCOORD0;
            float lod;
            vec3 ddx @ TEXCOORD1;
            vec3 ddy @ TEXCOORD2;
            ivec3 offset @ TEXCOORD3;
        };

        vec4 offsetOps(
            sampler3D volume,
            sampler s,
            vec3 uvw,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec3 offset
        ) {
            vec4 plain = textureOffset(volume, s, uvw, offset);
            vec4 biased = textureOffset(volume, s, uvw, offset, 0.5);
            vec4 lodSample = textureLodOffset(volume, s, uvw, lod, offset);
            vec4 gradSample = textureGradOffset(volume, s, uvw, ddx, ddy, offset);
            return plain + biased + lodSample + gradSample;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return offsetOps(
                    volumeMap,
                    linearSampler,
                    input.uvw,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture3d<float> volumeMap [[texture(0)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 offsetOps(texture3d<float> volume, sampler s, float3 uvw, float lod, float3 ddx, float3 ddy, int3 offset)"
        in generated_code
    )
    assert "float4 plain = volume.sample(s, uvw, offset);" in generated_code
    assert "float4 biased = volume.sample(s, uvw, bias(0.5), offset);" in generated_code
    assert (
        "float4 lodSample = volume.sample(s, uvw, level(lod), offset);"
        in generated_code
    )
    assert (
        "float4 gradSample = volume.sample(s, uvw, gradient3d(ddx, ddy), offset);"
        in generated_code
    )
    assert "unsup" + "ported Metal texture offset" not in generated_code
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code


def test_metal_cube_texture_sample_offsets_emit_diagnostics():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texturecube<float> cubeMap [[texture(0)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert "depthcube<float> shadowMap [[texture(2)]]" in generated_code
    assert "depthcube_array<float> shadowArray [[texture(3)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        generated_code.count(
            "/* unsupported Metal texture offset: textureOffset offsets require 2D, 2D-array, or 3D textures */ float4(0.0)"
        )
        == 4
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture offset: textureLodOffset offsets require 2D, 2D-array, or 3D textures */ float4(0.0)"
        )
        == 4
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture offset: textureGradOffset offsets require 2D, 2D-array, or 3D textures */ float4(0.0)"
        )
        == 4
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture compare: textureCompareOffset offsets require 2D or 2D-array depth textures */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture compare: textureCompareLodOffset offsets require 2D or 2D-array depth textures */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture compare: textureCompareGradOffset offsets require 2D or 2D-array depth textures */ 0.0"
        )
        == 2
    )
    assert ".sample(" not in generated_code
    assert ".sample_compare(" not in generated_code
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code
    assert "textureCompareOffset(" not in generated_code
    assert "textureCompareLodOffset(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_metal_unsupported_cube_offsets_and_supported_projection_thread_samplers():
    shader = """
    shader UnsupportedImplicitCubeOffsetSamplerHelper {
        samplerCube cubeMap;
        sampler cubeMapSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeProj @ TEXCOORD1;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 helper(vec3 direction, float lod, vec3 ddx, vec3 ddy, ivec2 offset) {
            return textureOffset(cubeMap, direction, offset)
                + textureLodOffset(cubeMap, direction, lod, offset)
                + textureGradOffset(cubeMap, direction, ddx, ddy, offset);
        }

        vec4 projectedHelper(vec4 cubeProj, float lod, vec3 ddx, vec3 ddy, ivec2 offset) {
            return textureProj(cubeMap, cubeProj)
                + textureProjLod(cubeMap, cubeProj, lod)
                + textureProjGradOffset(cubeMap, cubeProj, ddx, ddy, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return helper(
                    input.direction,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + projectedHelper(
                    input.cubeProj,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "float4 helper(float3 direction, float lod, float3 ddx, float3 ddy, int2 offset, texturecube<float> cubeMap)"
        in generated_code
    )
    assert (
        "float4 projectedHelper(float4 cubeProj, float lod, float3 ddx, float3 ddy, int2 offset, texturecube<float> cubeMap, sampler cubeMapSampler)"
        in generated_code
    )
    assert "sampler cubeMapSampler [[sampler(0)]]" in generated_code
    assert (
        "helper(input.direction, input.lod, input.ddx, input.ddy, input.offset, cubeMap)"
        in generated_code
    )
    assert (
        "projectedHelper(input.cubeProj, input.lod, input.ddx, input.ddy, input.offset, cubeMap, cubeMapSampler)"
        in generated_code
    )
    assert (
        "helper(input.direction, input.lod, input.ddx, input.ddy, input.offset, cubeMap, cubeMapSampler)"
        not in generated_code
    )
    assert (
        "projectedHelper(input.cubeProj, input.lod, input.ddx, input.ddy, input.offset, cubeMap) +"
        not in generated_code
    )
    assert "cubeMap.sample(cubeMapSampler, cubeProj.xyz / cubeProj.w)" in generated_code
    assert (
        "cubeMap.sample(cubeMapSampler, cubeProj.xyz / cubeProj.w, level(lod))"
        in generated_code
    )
    assert generated_code.count("unsupported Metal texture offset") == 3
    assert generated_code.count("unsupported Metal projected texture") == 1


def test_metal_direct_stage_sample_offsets_and_texel_fetch_offset_use_input_members():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "fragment float4 fragment_main(FSInput input [[stage_in]], texture2d<float> colorMap [[texture(0)]], texture2d_array<float> layerMap [[texture(1)]], sampler linearSampler [[sampler(0)]])"
        in generated_code
    )
    assert (
        "float4 plain = colorMap.sample(linearSampler, input.uv, input.offset);"
        in generated_code
    )
    assert (
        "float4 lodSample = colorMap.sample(linearSampler, input.uv, level(input.lod), input.offset);"
        in generated_code
    )
    assert (
        "float4 gradSample = colorMap.sample(linearSampler, input.uv, gradient2d(input.ddx, input.ddy), input.offset);"
        in generated_code
    )
    assert (
        "float4 arrayPlain = layerMap.sample(linearSampler, input.uvLayer.xy, uint(input.uvLayer.z), input.offset);"
        in generated_code
    )
    assert (
        "float4 arrayLod = layerMap.sample(linearSampler, input.uvLayer.xy, uint(input.uvLayer.z), level(input.lod), input.offset);"
        in generated_code
    )
    assert (
        "float4 arrayGrad = layerMap.sample(linearSampler, input.uvLayer.xy, uint(input.uvLayer.z), gradient2d(input.ddx, input.ddy), input.offset);"
        in generated_code
    )
    assert (
        "float4 fetched = colorMap.read((input.pixel + input.offset), int(input.lod));"
        in generated_code
    )
    assert (
        "float4 fetchedLayer = layerMap.read((input.pixelLayer.xy + input.offset), uint(input.pixelLayer.z), int(input.lod));"
        in generated_code
    )
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code
    assert "texelFetchOffset(" not in generated_code


def test_metal_projected_texture_variants_use_sample_projection():
    shader = """
    shader TextureProjectionVariants {
        sampler2D colorMap;
        sampler3D volumeMap;
        samplerCube cubeMap;
        sampler linearSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 xyzq @ TEXCOORD2;
            vec3 direction @ TEXCOORD3;
            vec2 ddx @ TEXCOORD4;
            vec2 ddy @ TEXCOORD5;
            ivec2 offset @ TEXCOORD6;
            vec3 ddx3 @ TEXCOORD7;
            vec3 ddy3 @ TEXCOORD8;
            ivec3 offset3 @ TEXCOORD9;
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
            ivec2 offset,
            vec3 ddx3,
            vec3 ddy3,
            ivec3 offset3
        ) {
            vec4 projected = textureProj(tex, s, uvq);
            vec4 projectedBias = textureProj(tex, s, uvqw, 0.25);
            vec4 volumeProjected = textureProj(volume, s, xyzq);
            vec4 volumeProjectedOffset = textureProjOffset(volume, s, xyzq, offset3);
            vec4 volumeProjectedLodOffset = textureProjLodOffset(volume, s, xyzq, 3.0, offset3);
            vec4 volumeProjectedGradOffset = textureProjGradOffset(volume, s, xyzq, ddx3, ddy3, offset3);
            vec4 projectedOffset = textureProjOffset(tex, s, uvq, offset);
            vec4 projectedOffsetBias = textureProjOffset(tex, s, uvq, offset, 0.5);
            vec4 projectedLod = textureProjLod(tex, s, uvq, 2.0);
            vec4 projectedLodOffset = textureProjLodOffset(tex, s, uvq, 3.0, offset);
            vec4 projectedGrad = textureProjGrad(tex, s, uvq, ddx, ddy);
            vec4 projectedGradOffset = textureProjGradOffset(tex, s, uvq, ddx, ddy, offset);
            return projected
                + projectedBias
                + volumeProjected
                + volumeProjectedOffset
                + volumeProjectedLodOffset
                + volumeProjectedGradOffset
                + projectedOffset
                + projectedOffsetBias
                + projectedLod
                + projectedLodOffset
                + projectedGrad
                + projectedGradOffset;
        }

        vec4 unsupportedCubeProjection(samplerCube tex, sampler s, vec3 direction) {
            return textureProj(tex, s, direction);
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
                    input.offset,
                    input.ddx3,
                    input.ddy3,
                    input.offset3
                ) + unsupportedCubeProjection(
                    cubeMap,
                    linearSampler,
                    input.direction
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture3d<float> volumeMap [[texture(1)]]" in generated_code
    assert "texturecube<float> cubeMap [[texture(2)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 projectedOps(texture2d<float> tex, texture3d<float> volume, sampler s, float3 uvq, float4 uvqw, float4 xyzq, float2 ddx, float2 ddy, int2 offset, float3 ddx3, float3 ddy3, int3 offset3)"
        in generated_code
    )
    assert "float4 projected = tex.sample(s, uvq.xy / uvq.z);" in generated_code
    assert (
        "float4 projectedBias = tex.sample(s, uvqw.xy / uvqw.w, bias(0.25));"
        in generated_code
    )
    assert (
        "float4 volumeProjected = volume.sample(s, xyzq.xyz / xyzq.w);"
        in generated_code
    )
    assert (
        "float4 volumeProjectedOffset = volume.sample(s, xyzq.xyz / xyzq.w, offset3);"
        in generated_code
    )
    assert (
        "float4 volumeProjectedLodOffset = volume.sample(s, xyzq.xyz / xyzq.w, level(3.0), offset3);"
        in generated_code
    )
    assert (
        "float4 volumeProjectedGradOffset = volume.sample(s, xyzq.xyz / xyzq.w, gradient3d(ddx3, ddy3), offset3);"
        in generated_code
    )
    assert (
        "float4 projectedOffset = tex.sample(s, uvq.xy / uvq.z, offset);"
        in generated_code
    )
    assert (
        "float4 projectedOffsetBias = tex.sample(s, uvq.xy / uvq.z, bias(0.5), offset);"
        in generated_code
    )
    assert (
        "float4 projectedLod = tex.sample(s, uvq.xy / uvq.z, level(2.0));"
        in generated_code
    )
    assert (
        "float4 projectedLodOffset = tex.sample(s, uvq.xy / uvq.z, level(3.0), offset);"
        in generated_code
    )
    assert (
        "float4 projectedGrad = tex.sample(s, uvq.xy / uvq.z, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        "float4 projectedGradOffset = tex.sample(s, uvq.xy / uvq.z, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "return /* unsupported Metal projected texture: textureProj requires 1D, 2D, or 3D projection coordinates */ float4(0.0);"
        in generated_code
    )
    assert "textureProj(" not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLod(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGrad(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_metal_direct_projected_texture_stage_input_members():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture3d<float> volumeMap [[texture(1)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 projected = colorMap.sample(linearSampler, input.uvq.xy / input.uvq.z);"
        in generated_code
    )
    assert (
        "float4 projectedBias = colorMap.sample(linearSampler, input.uvqw.xy / input.uvqw.w, bias(0.25));"
        in generated_code
    )
    assert (
        "float4 volumeProjected = volumeMap.sample(linearSampler, input.xyzq.xyz / input.xyzq.w);"
        in generated_code
    )
    assert (
        "float4 projectedOffset = colorMap.sample(linearSampler, input.uvq.xy / input.uvq.z, input.offset);"
        in generated_code
    )
    assert (
        "float4 projectedLod = colorMap.sample(linearSampler, input.uvq.xy / input.uvq.z, level(input.lod));"
        in generated_code
    )
    assert (
        "float4 projectedLodOffset = colorMap.sample(linearSampler, input.uvq.xy / input.uvq.z, level(input.lod), input.offset);"
        in generated_code
    )
    assert (
        "float4 projectedGrad = colorMap.sample(linearSampler, input.uvq.xy / input.uvq.z, gradient2d(input.ddx, input.ddy));"
        in generated_code
    )
    assert (
        "float4 projectedGradOffset = colorMap.sample(linearSampler, input.uvq.xy / input.uvq.z, gradient2d(input.ddx, input.ddy), input.offset);"
        in generated_code
    )
    assert "unsupported Metal projected texture" not in generated_code
    assert "textureProj(" not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLod(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGrad(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_metal_direct_projected_array_texture_stage_input_members():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    projected_args = "input.uvLayerQ.xy / input.uvLayerQ.w, uint(input.uvLayerQ.z)"
    assert "texture2d_array<float> layerMap [[texture(0)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        f"float4 projected = layerMap.sample(linearSampler, {projected_args});"
        in generated_code
    )
    assert (
        f"float4 projectedOffset = layerMap.sample(linearSampler, {projected_args}, input.offset);"
        in generated_code
    )
    assert (
        f"float4 projectedLod = layerMap.sample(linearSampler, {projected_args}, level(input.lod));"
        in generated_code
    )
    assert (
        f"float4 projectedLodOffset = layerMap.sample(linearSampler, {projected_args}, level(input.lod), input.offset);"
        in generated_code
    )
    assert (
        f"float4 projectedGrad = layerMap.sample(linearSampler, {projected_args}, gradient2d(input.ddx, input.ddy));"
        in generated_code
    )
    assert (
        f"float4 projectedGradOffset = layerMap.sample(linearSampler, {projected_args}, gradient2d(input.ddx, input.ddy), input.offset);"
        in generated_code
    )
    assert "unsupported Metal projected texture" not in generated_code
    assert "textureProj(" not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLod(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGrad(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_metal_implicit_projected_array_texture_stage_input_members_use_default_sampler():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"
    projected_args = "input.uvLayerQ.xy / input.uvLayerQ.w, uint(input.uvLayerQ.z)"
    assert "texture2d_array<float> layerMap [[texture(0)]]" in generated_code
    assert "[[sampler(" not in generated_code
    assert (
        f"float4 projected = layerMap.sample({default_sampler}, {projected_args});"
        in generated_code
    )
    assert (
        f"float4 projectedOffset = layerMap.sample({default_sampler}, {projected_args}, input.offset);"
        in generated_code
    )
    assert (
        f"float4 projectedLod = layerMap.sample({default_sampler}, {projected_args}, level(input.lod));"
        in generated_code
    )
    assert (
        f"float4 projectedLodOffset = layerMap.sample({default_sampler}, {projected_args}, level(input.lod), input.offset);"
        in generated_code
    )
    assert (
        f"float4 projectedGrad = layerMap.sample({default_sampler}, {projected_args}, gradient2d(input.ddx, input.ddy));"
        in generated_code
    )
    assert (
        f"float4 projectedGradOffset = layerMap.sample({default_sampler}, {projected_args}, gradient2d(input.ddx, input.ddy), input.offset);"
        in generated_code
    )
    assert "unsupported Metal projected texture" not in generated_code
    assert "textureProj(" not in generated_code


def test_metal_projected_array_texture_resource_arrays_forward_samplers():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    projected_args = "uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z)"
    assert "array<texture2d_array<float>, 4> layerMaps [[texture(0)]]" in generated_code
    assert "array<sampler, 4> linearSamplers [[sampler(0)]]" in generated_code
    assert (
        "float4 projectedLeaf(array<texture2d_array<float>, 4> maps, array<sampler, 4> samplers, int layer, float4 uvLayerQ, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        f"float4 fixedProj = maps[2].sample(samplers[2], {projected_args});"
        in generated_code
    )
    assert (
        f"float4 dynamicOffset = maps[layer].sample(samplers[layer], {projected_args}, offset);"
        in generated_code
    )
    assert (
        f"float4 fixedLod = maps[1].sample(samplers[1], {projected_args}, level(lod));"
        in generated_code
    )
    assert (
        f"float4 dynamicLodOffset = maps[layer].sample(samplers[layer], {projected_args}, level(lod), offset);"
        in generated_code
    )
    assert (
        f"float4 fixedGrad = maps[3].sample(samplers[3], {projected_args}, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        f"float4 dynamicGradOffset = maps[layer].sample(samplers[layer], {projected_args}, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "projectedLeaf(layerMaps, linearSamplers, input.layer, input.uvLayerQ, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "unsupported Metal projected texture" not in generated_code
    assert "textureProj(" not in generated_code


def test_metal_implicit_projected_array_texture_resource_arrays_use_default_sampler():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"
    projected_args = "uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z)"
    assert "array<texture2d_array<float>, 4> layerMaps [[texture(0)]]" in generated_code
    assert "[[sampler(" not in generated_code
    assert (
        "float4 projectedLeaf(array<texture2d_array<float>, 4> maps, int layer, float4 uvLayerQ, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        f"float4 fixedProj = maps[2].sample({default_sampler}, {projected_args});"
        in generated_code
    )
    assert (
        f"float4 dynamicOffset = maps[layer].sample({default_sampler}, {projected_args}, offset);"
        in generated_code
    )
    assert (
        f"float4 fixedLod = maps[1].sample({default_sampler}, {projected_args}, level(lod));"
        in generated_code
    )
    assert (
        f"float4 dynamicLodOffset = maps[layer].sample({default_sampler}, {projected_args}, level(lod), offset);"
        in generated_code
    )
    assert (
        f"float4 fixedGrad = maps[3].sample({default_sampler}, {projected_args}, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        f"float4 dynamicGradOffset = maps[layer].sample({default_sampler}, {projected_args}, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "projectedLeaf(layerMaps, input.layer, input.uvLayerQ, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "unsupported Metal projected texture" not in generated_code
    assert "textureProj(" not in generated_code


def test_metal_implicit_projected_stage_input_members_use_default_sampler():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )
    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture3d<float> volumeMap [[texture(1)]]" in generated_code
    assert "depth2d<float> shadowMap [[texture(2)]]" in generated_code
    assert "depth2d_array<float> shadowArray [[texture(3)]]" in generated_code
    assert "[[sampler(" not in generated_code
    assert (
        f"float4 color = colorMap.sample({default_sampler}, input.uvq.xy / input.uvq.z);"
        in generated_code
    )
    assert (
        f"float4 colorLod = colorMap.sample({default_sampler}, input.uvqw.xy / input.uvqw.w, level(input.lod));"
        in generated_code
    )
    assert (
        f"float4 volume = volumeMap.sample({default_sampler}, input.xyzq.xyz / input.xyzq.w);"
        in generated_code
    )
    assert (
        f"float shadow = shadowMap.sample_compare({default_sampler}, input.uvq.xy / input.uvq.z, input.depth);"
        in generated_code
    )
    assert (
        f"float shadowGrad = shadowArray.sample_compare({default_sampler}, input.uvLayerQ.xy / input.uvLayerQ.w, uint(input.uvLayerQ.z), input.depth, gradient2d(input.ddx, input.ddy));"
        in generated_code
    )
    assert "unsupported Metal projected texture" not in generated_code
    assert "unsupported Metal texture compare" not in generated_code
    assert "textureProj(" not in generated_code
    assert "textureCompareProj(" not in generated_code


def test_metal_projected_shadow_compare_variants_use_sample_compare_projection():
    shader = """
    shader ProjectedShadowCompareVariants {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
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
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, s, uvLayerQ, depth);
            float offsetProjected = textureCompareProjOffset(tex, s, uvLayerQ, depth, offset);
            float lodProjected = textureCompareProjLod(tex, s, uvLayerQ, depth, lod);
            float lodOffsetProjected = textureCompareProjLodOffset(tex, s, uvLayerQ, depth, lod, offset);
            float gradProjected = textureCompareProjGrad(tex, s, uvLayerQ, depth, ddx, ddy);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, s, uvLayerQ, depth, ddx, ddy, offset);
            return projected + offsetProjected + lodProjected + lodOffsetProjected + gradProjected + gradOffsetProjected;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return vec4(projectedShadow(
                    shadowMap,
                    compareSampler,
                    input.uvq,
                    input.uvqw,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + projectedArrayShadow(
                    shadowArray,
                    compareSampler,
                    vec4(input.uvLayer, input.uvqw.w),
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert "depth2d_array<float> shadowArray [[texture(1)]]" in generated_code
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert (
        "float projectedShadow(depth2d<float> tex, sampler s, float3 uvq, float4 uvqw, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float projected = tex.sample_compare(s, uvq.xy / uvq.z, depth);"
        in generated_code
    )
    assert (
        "float projectedW = tex.sample_compare(s, uvqw.xy / uvqw.w, depth);"
        in generated_code
    )
    assert (
        "float offsetProjected = tex.sample_compare(s, uvq.xy / uvq.z, depth, offset);"
        in generated_code
    )
    assert (
        "float lodProjected = tex.sample_compare(s, uvq.xy / uvq.z, depth, level(lod));"
        in generated_code
    )
    assert (
        "float lodOffsetProjected = tex.sample_compare(s, uvq.xy / uvq.z, depth, level(lod), offset);"
        in generated_code
    )
    assert (
        "float gradProjected = tex.sample_compare(s, uvq.xy / uvq.z, depth, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        "float gradOffsetProjected = tex.sample_compare(s, uvq.xy / uvq.z, depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "float projectedArrayShadow(depth2d_array<float> tex, sampler s, float4 uvLayerQ, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float projected = tex.sample_compare(s, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth);"
        in generated_code
    )
    assert (
        "float offsetProjected = tex.sample_compare(s, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, offset);"
        in generated_code
    )
    assert (
        "float lodProjected = tex.sample_compare(s, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, level(lod));"
        in generated_code
    )
    assert (
        "float lodOffsetProjected = tex.sample_compare(s, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, level(lod), offset);"
        in generated_code
    )
    assert (
        "float gradProjected = tex.sample_compare(s, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        "float gradOffsetProjected = tex.sample_compare(s, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert "textureCompareProj(" not in generated_code


def test_metal_direct_projected_shadow_compare_stage_input_members():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert "depth2d_array<float> shadowArray [[texture(1)]]" in generated_code
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert (
        "float planar = shadowMap.sample_compare(compareSampler, input.uvq.xy / input.uvq.z, input.depth);"
        in generated_code
    )
    assert (
        "float planarOffset = shadowMap.sample_compare(compareSampler, input.uvqw.xy / input.uvqw.w, input.depth, input.offset);"
        in generated_code
    )
    assert (
        "float arrayGrad = shadowArray.sample_compare(compareSampler, input.uvLayerQ.xy / input.uvLayerQ.w, uint(input.uvLayerQ.z), input.depth, gradient2d(input.ddx, input.ddy));"
        in generated_code
    )
    assert "unsupported Metal texture compare" not in generated_code
    assert "textureCompareProj(" not in generated_code


def test_metal_projected_shadow_compare_resource_arrays_forward_samplers():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert (
        "array<depth2d_array<float>, 4> shadowArrays [[texture(4)]]" in generated_code
    )
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert (
        "float projectedLeaf(array<depth2d<float>, 4> shadowMaps, array<depth2d_array<float>, 4> shadowArrays, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "float planar = shadowMaps[layer].sample_compare(shadowSamplers[layer], uvq.xy / uvq.z, depth);"
        in generated_code
    )
    assert (
        "float planarOffset = shadowMaps[1].sample_compare(shadowSamplers[1], uvq.xy / uvq.z, depth, offset);"
        in generated_code
    )
    assert (
        "float planarLod = shadowMaps[2].sample_compare(shadowSamplers[2], uvq.xy / uvq.z, depth, level(lod));"
        in generated_code
    )
    assert (
        "float planarGradOffset = shadowMaps[layer].sample_compare(shadowSamplers[layer], uvq.xy / uvq.z, depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "float arrayProjected = shadowArrays[2].sample_compare(shadowSamplers[2], uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth);"
        in generated_code
    )
    assert (
        "float arrayOffset = shadowArrays[layer].sample_compare(shadowSamplers[layer], uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, offset);"
        in generated_code
    )
    assert (
        "float arrayGrad = shadowArrays[1].sample_compare(shadowSamplers[1], uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        "float projectedWrapper(array<depth2d<float>, 4> shadowMaps, array<depth2d_array<float>, 4> shadowArrays, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "projectedLeaf(shadowMaps, shadowArrays, shadowSamplers, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedWrapper(shadowMaps, shadowArrays, shadowSamplers, input.layer, input.uvq, input.uvLayerQ, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureCompareProj(" not in generated_code


def test_metal_implicit_projected_shadow_compare_resource_arrays_use_default_sampler():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))
    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert (
        "array<depth2d_array<float>, 4> shadowArrays [[texture(4)]]" in generated_code
    )
    assert "array<sampler" not in generated_code
    assert (
        "float projectedLeaf(array<depth2d<float>, 4> shadowMaps, array<depth2d_array<float>, 4> shadowArrays"
        in generated_code
    )
    assert (
        f"float planar = shadowMaps[layer].sample_compare({default_sampler}, uvq.xy / uvq.z, depth);"
        in generated_code
    )
    assert (
        f"float planarOffset = shadowMaps[1].sample_compare({default_sampler}, uvq.xy / uvq.z, depth, offset);"
        in generated_code
    )
    assert (
        f"float planarLod = shadowMaps[2].sample_compare({default_sampler}, uvq.xy / uvq.z, depth, level(lod));"
        in generated_code
    )
    assert (
        f"float planarGradOffset = shadowMaps[layer].sample_compare({default_sampler}, uvq.xy / uvq.z, depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        f"float arrayProjected = shadowArrays[2].sample_compare({default_sampler}, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth);"
        in generated_code
    )
    assert (
        f"float arrayOffset = shadowArrays[layer].sample_compare({default_sampler}, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, offset);"
        in generated_code
    )
    assert (
        f"float arrayGrad = shadowArrays[1].sample_compare({default_sampler}, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        "float projectedWrapper(array<depth2d<float>, 4> shadowMaps, array<depth2d_array<float>, 4> shadowArrays"
        in generated_code
    )
    assert (
        "projectedLeaf(shadowMaps, shadowArrays, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedWrapper(shadowMaps, shadowArrays, input.layer, input.uvq, input.uvLayerQ, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureCompareProj(" not in generated_code


def test_metal_unsized_projected_shadow_compare_arrays_infer_transitive_constant_size():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<depth2d<float>, 5> shadowMaps [[texture(0)]]" in generated_code
    assert (
        "array<depth2d_array<float>, 5> shadowArrays [[texture(5)]]" in generated_code
    )
    assert "depth2d<float> afterShadow [[texture(10)]]" in generated_code
    assert "array<sampler, 5> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(5)]]" in generated_code
    assert (
        "float shadowDeep(array<depth2d<float>, 5> shadowMaps, array<depth2d_array<float>, 5> shadowArrays, array<sampler, 5> shadowSamplers"
        in generated_code
    )
    assert (
        "float planarHigh = shadowMaps[LAYER].sample_compare(shadowSamplers[LAYER], uvq.xy / uvq.z, depth);"
        in generated_code
    )
    assert (
        "float planarLow = shadowMaps[1].sample_compare(shadowSamplers[1], uvq.xy / uvq.z, depth, offset);"
        in generated_code
    )
    assert (
        "float arrayHigh = shadowArrays[2 * 2].sample_compare(shadowSamplers[2 * 2], uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        "float arrayLow = shadowArrays[3].sample_compare(shadowSamplers[3], uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, offset);"
        in generated_code
    )
    assert (
        "float shadowMid(array<depth2d<float>, 5> shadowMaps, array<depth2d_array<float>, 5> shadowArrays, array<sampler, 5> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowDeep(shadowMaps, shadowArrays, shadowSamplers, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "shadowMid(shadowMaps, shadowArrays, shadowSamplers, input.uvq, input.uvLayerQ, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "float singleShadow = afterShadow.sample_compare(afterSampler, input.uvq.xy, input.depth);"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "textureCompareProj(" not in generated_code


def test_metal_projected_cube_shadow_compare_lowers_supported_forms():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "depthcube<float> cubeMap [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert (
        "float cubeProjected(depthcube<float> tex, sampler s, float4 cubeProj, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float cubeArrayProjected(depthcube_array<float> tex, sampler s, float4 cubeLayerProj, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float projected = tex.sample_compare(s, cubeProj.xyz / cubeProj.w, depth);"
        in generated_code
    )
    for func_name in {
        "textureCompareProjOffset",
        "textureCompareProjLodOffset",
        "textureCompareProjGradOffset",
    }:
        assert (
            generated_code.count(
                f"/* unsupported Metal texture compare: {func_name} offsets require 2D or 2D-array depth textures */ 0.0"
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
                f"/* unsupported Metal texture compare: {func_name} requires depth2d vec3/vec4 or depth2d_array vec4 projection coordinates */ 0.0"
            )
            == count
        )
    assert "textureCompareProj(" not in generated_code
    assert "textureCompareProjOffset(" not in generated_code
    assert "textureCompareProjLodOffset(" not in generated_code
    assert "textureCompareProjGradOffset(" not in generated_code
    assert ".sample_compare(s, cubeProj" in generated_code
    assert ".sample_compare(s, cubeLayerProj" not in generated_code


def test_metal_direct_projected_cube_texture_lowers_supported_color_forms():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texturecube<float> cubeMap [[texture(0)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 cubeProjected = cubeMap.sample(linearSampler, input.cubeProj.xyz / input.cubeProj.w);"
        in generated_code
    )
    assert (
        "float4 cubeLod = cubeMap.sample(linearSampler, input.cubeProj.xyz / input.cubeProj.w, level(input.lod));"
        in generated_code
    )
    assert (
        "float4 cubeGrad = cubeMap.sample(linearSampler, input.cubeProj.xyz / input.cubeProj.w, gradientcube(input.ddx, input.ddy));"
        in generated_code
    )
    assert (
        "float4 implicitCube = cubeMap.sample(sampler(mag_filter::linear, min_filter::linear), input.cubeProj.xyz / input.cubeProj.w);"
        in generated_code
    )
    assert (
        "float4 implicitCubeGrad = cubeMap.sample(sampler(mag_filter::linear, min_filter::linear), input.cubeProj.xyz / input.cubeProj.w, gradientcube(input.ddx, input.ddy));"
        in generated_code
    )
    assert (
        "/* unsupported Metal projected texture: textureProjLodOffset offsets require 2D, 2D-array, or 3D textures */ float4(0.0)"
        in generated_code
    )
    assert (
        "/* unsupported Metal projected texture: textureProjGradOffset offsets require 2D, 2D-array, or 3D textures */ float4(0.0)"
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
                f"/* unsupported Metal projected texture: {func_name} requires 1D, 2D, or 3D projection coordinates */ float4(0.0)"
            )
            == count
        )
    assert "textureProj(cubeMap" not in generated_code
    assert "textureProj(cubeArray" not in generated_code
    assert ".sample(linearSampler, input.cubeArrayProj" not in generated_code


def test_metal_projected_cube_texture_resource_arrays_lower_supported_color_forms():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "array<texturecube<float>, 4> cubeMaps [[texture(0)]]" in generated_code
    assert (
        "array<texturecube_array<float>, 4> cubeArrays [[texture(4)]]" in generated_code
    )
    assert "array<depthcube<float>, 4> shadowCubes [[texture(8)]]" in generated_code
    assert (
        "array<depthcube_array<float>, 4> shadowCubeArrays [[texture(12)]]"
        in generated_code
    )
    assert (
        "float4 projectedCube(array<texturecube<float>, 4> maps, int layer, float4 cubeProj, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float4 projectedCubeArray(array<texturecube_array<float>, 4> maps, int layer, float4 cubeArrayProj, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float4 projectedShadow(array<depthcube<float>, 4> maps, int layer, float4 cubeProj, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float4 projectedShadowArray(array<depthcube_array<float>, 4> maps, int layer, float4 cubeArrayProj, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float4 fixedProjected = maps[2].sample(sampler(mag_filter::linear, min_filter::linear), cubeProj.xyz / cubeProj.w);"
        in generated_code
    )
    assert (
        "float4 dynamicLodProjected = maps[layer].sample(sampler(mag_filter::linear, min_filter::linear), cubeProj.xyz / cubeProj.w, level(lod));"
        in generated_code
    )
    assert (
        "float4 globalProjected = cubeMaps[input.layer].sample(sampler(mag_filter::linear, min_filter::linear), input.cubeProj.xyz / input.cubeProj.w);"
        in generated_code
    )
    assert (
        "/* unsupported Metal projected texture: textureProjGradOffset offsets require 2D, 2D-array, or 3D textures */ float4(0.0)"
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
                f"/* unsupported Metal projected texture: {func_name} requires 1D, 2D, or 3D projection coordinates */ float4(0.0)"
            )
            == count
        )
    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"
    assert (
        f"float fixedProjected = maps[2].sample_compare({default_sampler}, cubeProj.xyz / cubeProj.w, depth);"
        in generated_code
    )
    assert (
        f"float dynamicLodProjected = maps[layer].sample_compare({default_sampler}, cubeProj.xyz / cubeProj.w, depth, level(lod));"
        in generated_code
    )
    assert (
        f"float globalShadow = shadowCubes[input.layer].sample_compare({default_sampler}, input.cubeProj.xyz / input.cubeProj.w, input.depth);"
        in generated_code
    )
    assert (
        "/* unsupported Metal texture compare: textureCompareProjGradOffset offsets require 2D or 2D-array depth textures */ 0.0"
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
                f"/* unsupported Metal texture compare: {func_name} requires depth2d vec3/vec4 or depth2d_array vec4 projection coordinates */ 0.0"
            )
            == count
        )
    assert (
        "projectedCube(cubeMaps, input.layer, input.cubeProj, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "projectedCubeArray(cubeArrays, input.layer, input.cubeArrayProj, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "projectedShadow(shadowCubes, input.layer, input.cubeProj, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "projectedShadowArray(shadowCubeArrays, input.layer, input.cubeArrayProj, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "[[sampler(" not in generated_code
    assert "textureProj(" not in generated_code
    assert "textureCompareProj(" not in generated_code


def test_metal_shadow_gather_compare_offsets_use_depth_overloads():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert "depth2d_array<float> shadowArray [[texture(1)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(2)]]" in generated_code
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 gatherShadow(depth2d<float> tex, sampler s, float2 uv, float depth, int2 offset)"
        in generated_code
    )
    assert "float4 gathered = tex.gather_compare(s, uv, depth);" in generated_code
    assert (
        "float4 offsetGathered = tex.gather_compare(s, uv, depth, offset);"
        in generated_code
    )
    assert (
        "float offsetCompared = tex.sample_compare(s, uv, depth, offset);"
        in generated_code
    )
    assert (
        "float4 gatherShadowArray(depth2d_array<float> tex, sampler s, float3 uvLayer, float depth, int2 offset)"
        in generated_code
    )
    assert "tex.gather_compare(s, uvLayer.xy, uint(uvLayer.z), depth)" in generated_code
    assert (
        "tex.gather_compare(s, uvLayer.xy, uint(uvLayer.z), depth, offset)"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, uvLayer.xy, uint(uvLayer.z), depth, offset)"
        in generated_code
    )
    assert (
        "float4 gatherCubeShadowArray(depthcube_array<float> tex, sampler s, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert (
        "return tex.gather_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth);"
        in generated_code
    )
    assert "textureGatherCompare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code
    assert "textureCompareOffset(" not in generated_code


def test_metal_cube_shadow_gather_compare_supports_cube_and_cube_array():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )
    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"

    assert "depthcube<float> cubeShadow [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(1)]]" in generated_code
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 gatherCubeShadow(depthcube<float> tex, sampler s, float3 direction, float depth)"
        in generated_code
    )
    assert "return tex.gather_compare(s, direction, depth);" in generated_code
    assert (
        "float4 gatherCubeArrayShadow(depthcube_array<float> tex, sampler s, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert (
        "return tex.gather_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth);"
        in generated_code
    )
    assert (
        "float4 implicitCubeShadow(depthcube<float> tex, float3 direction, float depth)"
        in generated_code
    )
    assert (
        f"return tex.gather_compare({default_sampler}, direction, depth);"
        in generated_code
    )
    assert (
        "float4 implicitCubeArrayShadow(depthcube_array<float> tex, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert (
        f"return tex.gather_compare({default_sampler}, cubeLayer.xyz, uint(cubeLayer.w), depth);"
        in generated_code
    )
    assert (
        "gatherCubeShadow(cubeShadow, compareSampler, input.direction, input.depth)"
        in generated_code
    )
    assert (
        "cubeShadowArray.gather_compare(compareSampler, input.cubeLayer.xyz, uint(input.cubeLayer.w), input.depth)"
        in generated_code
    )
    assert (
        f"cubeShadow.gather_compare({default_sampler}, input.direction, input.depth)"
        in generated_code
    )
    assert "unsupported Metal texture gather compare" not in generated_code
    assert "textureGatherCompare(" not in generated_code


def test_metal_implicit_shadow_gather_compare_offsets_cover_arrays_and_cube_arrays():
    shader = """
    shader ImplicitShadowGatherCompare {
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            ivec2 offset @ TEXCOORD2;
        };

        vec4 implicitArray(sampler2DArrayShadow tex, vec3 uvLayer, float depth, ivec2 offset) {
            return textureGatherCompare(tex, uvLayer, depth)
                + textureGatherCompareOffset(tex, uvLayer, depth, offset)
                + vec4(textureCompareOffset(tex, uvLayer, depth, offset));
        }

        vec4 implicitCubeArray(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth) {
            return textureGatherCompare(tex, cubeLayer, depth);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return implicitArray(shadowArray, input.uvLayer, input.depth, input.offset)
                    + implicitCubeArray(cubeShadowArray, input.cubeLayer, input.depth);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "depth2d_array<float> shadowArray [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(1)]]" in generated_code
    assert (
        "float4 implicitArray(depth2d_array<float> tex, float3 uvLayer, float depth, int2 offset)"
        in generated_code
    )
    assert (
        "tex.gather_compare(sampler(mag_filter::linear, min_filter::linear), uvLayer.xy, uint(uvLayer.z), depth)"
        in generated_code
    )
    assert (
        "tex.gather_compare(sampler(mag_filter::linear, min_filter::linear), uvLayer.xy, uint(uvLayer.z), depth, offset)"
        in generated_code
    )
    assert (
        "tex.sample_compare(sampler(mag_filter::linear, min_filter::linear), uvLayer.xy, uint(uvLayer.z), depth, offset)"
        in generated_code
    )
    assert (
        "float4 implicitCubeArray(depthcube_array<float> tex, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert (
        "return tex.gather_compare(sampler(mag_filter::linear, min_filter::linear), cubeLayer.xyz, uint(cubeLayer.w), depth);"
        in generated_code
    )
    assert (
        "implicitArray(shadowArray, input.uvLayer, input.depth, input.offset)"
        in generated_code
    )
    assert (
        "implicitCubeArray(cubeShadowArray, input.cubeLayer, input.depth)"
        in generated_code
    )
    assert "textureGatherCompare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code
    assert "textureCompareOffset(" not in generated_code


def test_metal_direct_shadow_gather_compare_stage_input_members():
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

    explicit_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(explicit_shader), "fragment"
    )
    implicit_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(implicit_shader), "fragment"
    )
    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"

    assert "depth2d<float> shadowMap [[texture(0)]]" in explicit_code
    assert "depth2d_array<float> shadowArray [[texture(1)]]" in explicit_code
    assert "depthcube_array<float> cubeShadowArray [[texture(2)]]" in explicit_code
    assert "sampler compareSampler [[sampler(0)]]" in explicit_code
    assert (
        "float4 planar = shadowMap.gather_compare(compareSampler, input.uv, input.depth);"
        in explicit_code
    )
    assert (
        "float4 planarOffset = shadowMap.gather_compare(compareSampler, input.uv, input.depth, input.offset);"
        in explicit_code
    )
    assert (
        "float4 arrayGather = shadowArray.gather_compare(compareSampler, input.uvLayer.xy, uint(input.uvLayer.z), input.depth);"
        in explicit_code
    )
    assert (
        "float4 arrayOffset = shadowArray.gather_compare(compareSampler, input.uvLayer.xy, uint(input.uvLayer.z), input.depth, input.offset);"
        in explicit_code
    )
    assert (
        "float4 cubeArrayGather = cubeShadowArray.gather_compare(compareSampler, input.cubeLayer.xyz, uint(input.cubeLayer.w), input.depth);"
        in explicit_code
    )
    assert "unsupported Metal texture gather compare" not in explicit_code
    assert "textureGatherCompare(" not in explicit_code

    assert "depth2d<float> shadowMap [[texture(0)]]" in implicit_code
    assert "depth2d_array<float> shadowArray [[texture(1)]]" in implicit_code
    assert "depthcube_array<float> cubeShadowArray [[texture(2)]]" in implicit_code
    assert "[[sampler(" not in implicit_code
    assert (
        f"float4 planar = shadowMap.gather_compare({default_sampler}, input.uv, input.depth);"
        in implicit_code
    )
    assert (
        f"float4 planarOffset = shadowMap.gather_compare({default_sampler}, input.uv, input.depth, input.offset);"
        in implicit_code
    )
    assert (
        f"float4 arrayGather = shadowArray.gather_compare({default_sampler}, input.uvLayer.xy, uint(input.uvLayer.z), input.depth);"
        in implicit_code
    )
    assert (
        f"float4 arrayOffset = shadowArray.gather_compare({default_sampler}, input.uvLayer.xy, uint(input.uvLayer.z), input.depth, input.offset);"
        in implicit_code
    )
    assert (
        f"float4 cubeArrayGather = cubeShadowArray.gather_compare({default_sampler}, input.cubeLayer.xyz, uint(input.cubeLayer.w), input.depth);"
        in implicit_code
    )
    assert "unsupported Metal texture gather compare" not in implicit_code
    assert "textureGatherCompare(" not in implicit_code


def test_metal_shadow_gather_compare_resource_arrays_forward_samplers():
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

    explicit_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(explicit_shader), "fragment"
    )
    implicit_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(implicit_shader), "fragment"
    )
    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in explicit_code
    assert "array<depth2d_array<float>, 4> shadowArrays [[texture(4)]]" in explicit_code
    assert "array<depthcube_array<float>, 4> cubeArrays [[texture(8)]]" in explicit_code
    assert "array<sampler, 4> compareSamplers [[sampler(0)]]" in explicit_code
    assert (
        "float4 gatherLayer(array<depth2d<float>, 4> maps, array<depth2d_array<float>, 4> arrays, array<depthcube_array<float>, 4> cubes, array<sampler, 4> samplers, int layer, float2 uv, float3 uvLayer, float4 cubeLayer, float depth, int2 offset)"
        in explicit_code
    )
    assert "maps[2].gather_compare(samplers[2], uv, depth)" in explicit_code
    assert (
        "maps[layer].gather_compare(samplers[layer], uv, depth, offset)"
        in explicit_code
    )
    assert (
        "arrays[1].gather_compare(samplers[1], uvLayer.xy, uint(uvLayer.z), depth)"
        in explicit_code
    )
    assert (
        "arrays[layer].gather_compare(samplers[layer], uvLayer.xy, uint(uvLayer.z), depth, offset)"
        in explicit_code
    )
    assert (
        "cubes[3].gather_compare(samplers[3], cubeLayer.xyz, uint(cubeLayer.w), depth)"
        in explicit_code
    )
    assert (
        "cubes[layer].gather_compare(samplers[layer], cubeLayer.xyz, uint(cubeLayer.w), depth)"
        in explicit_code
    )
    assert (
        "gatherLayer(shadowMaps, shadowArrays, cubeArrays, compareSamplers, input.layer, input.uv, input.uvLayer, input.cubeLayer, input.depth, input.offset)"
        in explicit_code
    )
    assert "textureGatherCompare" not in explicit_code

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in implicit_code
    assert "array<depth2d_array<float>, 4> shadowArrays [[texture(4)]]" in implicit_code
    assert "array<depthcube_array<float>, 4> cubeArrays [[texture(8)]]" in implicit_code
    assert "[[sampler(" not in implicit_code
    assert (
        "float4 gatherLayer(array<depth2d<float>, 4> maps, array<depth2d_array<float>, 4> arrays, array<depthcube_array<float>, 4> cubes, int layer, float2 uv, float3 uvLayer, float4 cubeLayer, float depth, int2 offset)"
        in implicit_code
    )
    assert f"maps[2].gather_compare({default_sampler}, uv, depth)" in implicit_code
    assert (
        f"maps[layer].gather_compare({default_sampler}, uv, depth, offset)"
        in implicit_code
    )
    assert (
        f"arrays[1].gather_compare({default_sampler}, uvLayer.xy, uint(uvLayer.z), depth)"
        in implicit_code
    )
    assert (
        f"arrays[layer].gather_compare({default_sampler}, uvLayer.xy, uint(uvLayer.z), depth, offset)"
        in implicit_code
    )
    assert (
        f"cubes[3].gather_compare({default_sampler}, cubeLayer.xyz, uint(cubeLayer.w), depth)"
        in implicit_code
    )
    assert (
        f"cubes[layer].gather_compare({default_sampler}, cubeLayer.xyz, uint(cubeLayer.w), depth)"
        in implicit_code
    )
    assert (
        "gatherLayer(shadowMaps, shadowArrays, cubeArrays, input.layer, input.uv, input.uvLayer, input.cubeLayer, input.depth, input.offset)"
        in implicit_code
    )
    assert "textureGatherCompare" not in implicit_code


def test_metal_unsupported_cube_shadow_gather_compare_offsets_emit_diagnostics():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported Metal texture gather compare: "
        "textureGatherCompareOffset offsets require 2D or 2D-array depth textures */ "
        "float4(0.0)"
    )
    assert "depthcube<float> cubeMap [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert "[[sampler(" not in generated_code
    assert "sampler(" not in generated_code
    assert (
        "float4 cubeOffset(depthcube<float> tex, float3 direction, float depth, int2 offset)"
        in generated_code
    )
    assert (
        "float4 cubeArrayOffset(depthcube_array<float> tex, float4 cubeLayer, float depth, int2 offset)"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 3
    assert (
        "cubeOffset(cubeMap, input.direction, input.depth, input.offset)"
        in generated_code
    )
    assert (
        "cubeArrayOffset(cubeArray, input.cubeLayer, input.depth, input.offset)"
        in generated_code
    )
    assert ".gather_compare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code


def test_metal_shadow_compare_lod_grad_use_depth_overloads():
    shader = """
    shader ShadowCompareLodGrad {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            vec3 cubeDdx @ TEXCOORD5;
            vec3 cubeDdy @ TEXCOORD6;
            ivec2 offset @ TEXCOORD7;
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
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float lodValue = textureCompareLod(tex, s, uvLayer, depth, lod);
            float lodOffsetValue = textureCompareLodOffset(tex, s, uvLayer, depth, lod, offset);
            float gradValue = textureCompareGrad(tex, s, uvLayer, depth, ddx, ddy);
            float gradOffsetValue = textureCompareGradOffset(tex, s, uvLayer, depth, ddx, ddy, offset);
            return lodValue + lodOffsetValue + gradValue + gradOffsetValue;
        }

        float compareCubeShadowArray(
            samplerCubeArrayShadow tex,
            sampler s,
            vec4 cubeLayer,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy
        ) {
            float lodValue = textureCompareLod(tex, s, cubeLayer, depth, lod);
            float gradValue = textureCompareGrad(tex, s, cubeLayer, depth, ddx, ddy);
            return lodValue + gradValue;
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
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float cubeArrayShadow = compareCubeShadowArray(
                    cubeShadowArray,
                    compareSampler,
                    input.cubeLayer,
                    input.depth,
                    input.lod,
                    input.cubeDdx,
                    input.cubeDdy
                );
                return vec4(shadow + arrayShadow + cubeArrayShadow);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert "depth2d_array<float> shadowArray [[texture(1)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(2)]]" in generated_code
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert (
        "float compareShadow(depth2d<float> tex, sampler s, float2 uv, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float lodValue = tex.sample_compare(s, uv, depth, level(lod));"
        in generated_code
    )
    assert (
        "float lodOffsetValue = tex.sample_compare(s, uv, depth, level(lod), offset);"
        in generated_code
    )
    assert (
        "float gradValue = tex.sample_compare(s, uv, depth, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        "float gradOffsetValue = tex.sample_compare(s, uv, depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "float compareShadowArray(depth2d_array<float> tex, sampler s, float3 uvLayer, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, uvLayer.xy, uint(uvLayer.z), depth, level(lod))"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, uvLayer.xy, uint(uvLayer.z), depth, level(lod), offset)"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, uvLayer.xy, uint(uvLayer.z), depth, gradient2d(ddx, ddy))"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, uvLayer.xy, uint(uvLayer.z), depth, gradient2d(ddx, ddy), offset)"
        in generated_code
    )
    assert (
        "float compareCubeShadowArray(depthcube_array<float> tex, sampler s, float4 cubeLayer, float depth, float lod, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth, level(lod))"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth, gradientcube(ddx, ddy))"
        in generated_code
    )
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareLodOffset(" not in generated_code
    assert "textureCompareGrad(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_metal_cube_shadow_compare_offsets_report_unsupported():
    shader = """
    shader CubeShadowCompareOffsetDiagnostics {
        samplerCubeShadow cubeMap;
        samplerCubeArrayShadow cubeArray;
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

        float cubeOffsets(
            samplerCubeShadow tex,
            sampler s,
            vec3 direction,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float compareOffset = textureCompareOffset(tex, s, direction, depth, offset);
            float lodOffset = textureCompareLodOffset(tex, s, direction, depth, lod, offset);
            float gradOffset = textureCompareGradOffset(tex, s, direction, depth, ddx, ddy, offset);
            return compareOffset + lodOffset + gradOffset;
        }

        float cubeArrayOffsets(
            samplerCubeArrayShadow tex,
            sampler s,
            vec4 cubeLayer,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float compareOffset = textureCompareOffset(tex, s, cubeLayer, depth, offset);
            float lodOffset = textureCompareLodOffset(tex, s, cubeLayer, depth, lod, offset);
            float gradOffset = textureCompareGradOffset(tex, s, cubeLayer, depth, ddx, ddy, offset);
            return compareOffset + lodOffset + gradOffset;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return cubeOffsets(
                    cubeMap,
                    compareSampler,
                    input.direction,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + cubeArrayOffsets(
                    cubeArray,
                    compareSampler,
                    input.cubeLayer,
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "depthcube<float> cubeMap [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert (
        "float cubeOffsets(depthcube<float> tex, sampler s, float3 direction, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float cubeArrayOffsets(depthcube_array<float> tex, sampler s, float4 cubeLayer, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture compare: textureCompareOffset offsets require 2D or 2D-array depth textures */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture compare: textureCompareLodOffset offsets require 2D or 2D-array depth textures */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture compare: textureCompareGradOffset offsets require 2D or 2D-array depth textures */ 0.0"
        )
        == 2
    )
    assert "textureCompareOffset(" not in generated_code
    assert "textureCompareLodOffset(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code
    assert ".sample_compare(s, direction, depth, offset)" not in generated_code
    assert (
        ".sample_compare(s, direction, depth, level(lod), offset)" not in generated_code
    )
    assert (
        ".sample_compare(s, direction, depth, gradientcube(ddx, ddy), offset)"
        not in generated_code
    )


def test_metal_nested_implicit_shadow_compare_lod_grad_uses_depth_overloads():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"
    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert (
        "float shadowOps(depth2d<float> tex, float2 uv, float depth, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        f"float2 lod = float2(tex.calculate_unclamped_lod({default_sampler}, uv), tex.calculate_clamped_lod({default_sampler}, uv));"
        in generated_code
    )
    assert (
        f"float cmp = tex.sample_compare({default_sampler}, uv, depth, level(lod.x));"
        in generated_code
    )
    assert (
        f"float grad = tex.sample_compare({default_sampler}, uv, depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "float wrappedShadow(depth2d<float> tex, float2 uv, float depth, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "return shadowOps(tex, uv, depth, ddx, ddy, offset);" in generated_code
    assert (
        "wrappedShadow(shadowMap, input.uv, input.depth, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureQueryLod(" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_metal_array_shadow_texture_resource_arrays_keep_compare_coordinates():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "array<depth2d_array<float>, 4> shadowArrays [[texture(0)]]" in generated_code
    )
    assert (
        "array<depthcube_array<float>, 4> cubeShadowArrays [[texture(4)]]"
        in generated_code
    )
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert (
        "float sampleArrayLayer(array<depth2d_array<float>, 4> shadowArrays, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowArrays[2].sample_compare(shadowSamplers[2], uvLayer.xy, uint(uvLayer.z), depth)"
        in generated_code
    )
    assert (
        "float sampleCubeLayer(array<depthcube_array<float>, 4> cubeShadowArrays, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "cubeShadowArrays[3].sample_compare(shadowSamplers[3], cubeLayer.xyz, uint(cubeLayer.w), depth)"
        in generated_code
    )
    assert (
        "sampleArrayLayer(shadowArrays, shadowSamplers, input.uvLayer, input.depth)"
        in generated_code
    )
    assert (
        "sampleCubeLayer(cubeShadowArrays, shadowSamplers, input.cubeLayer, input.depth)"
        in generated_code
    )
    assert "array<depth2d_array<float>, 1> shadowArrays" not in generated_code
    assert "textureCompare(" not in generated_code
    assert ".sample_compare(shadowSamplers[2], uvLayer, depth)" not in generated_code
    assert ".sample_compare(shadowSamplers[3], cubeLayer, depth)" not in generated_code


def test_metal_array_shadow_compare_lod_grad_resource_arrays():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert (
        "array<depth2d_array<float>, 4> shadowArrays [[texture(4)]]" in generated_code
    )
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 4> shadowMaps, array<depth2d_array<float>, 4> shadowArrays, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "float planarLod = shadowMaps[layer].sample_compare(shadowSamplers[layer], uv, depth, level(lod));"
        in generated_code
    )
    assert (
        "float planarGrad = shadowMaps[1].sample_compare(shadowSamplers[1], uv, depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "float arrayLod = shadowArrays[2].sample_compare(shadowSamplers[2], uvLayer.xy, uint(uvLayer.z), depth, level(lod));"
        in generated_code
    )
    assert (
        "float arrayGrad = shadowArrays[layer].sample_compare(shadowSamplers[layer], uvLayer.xy, uint(uvLayer.z), depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "shadowLayer(shadowMaps, shadowArrays, shadowSamplers, input.layer, input.uv, input.uvLayer, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_metal_cube_shadow_compare_lod_grad_resource_arrays():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<depthcube<float>, 4> cubeMaps [[texture(0)]]" in generated_code
    assert (
        "array<depthcube_array<float>, 4> cubeArrays [[texture(4)]]" in generated_code
    )
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert (
        "float cubeShadowLayer(array<depthcube<float>, 4> cubeMaps, array<depthcube_array<float>, 4> cubeArrays, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "float cubeLod = cubeMaps[layer].sample_compare(shadowSamplers[layer], direction, depth, level(lod));"
        in generated_code
    )
    assert (
        "float cubeGrad = cubeMaps[1].sample_compare(shadowSamplers[1], direction, depth, gradientcube(ddx, ddy));"
        in generated_code
    )
    assert (
        "float cubeArrayLod = cubeArrays[2].sample_compare(shadowSamplers[2], cubeLayer.xyz, uint(cubeLayer.w), depth, level(lod));"
        in generated_code
    )
    assert (
        "float cubeArrayGrad = cubeArrays[layer].sample_compare(shadowSamplers[layer], cubeLayer.xyz, uint(cubeLayer.w), depth, gradientcube(ddx, ddy));"
        in generated_code
    )
    assert (
        "cubeShadowLayer(cubeMaps, cubeArrays, shadowSamplers, input.layer, input.direction, input.cubeLayer, input.depth, input.lod, input.ddx, input.ddy)"
        in generated_code
    )
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGrad(" not in generated_code


def test_metal_array_shadow_texture_resource_arrays_reject_mismatched_fixed_helper_size():
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
        MetalCodeGen().generate(crosstl.translator.parse(shader))


def test_metal_texture_query_size_descriptors():
    codegen = MetalCodeGen()

    assert codegen.texture_query_size_descriptor("texture1d_array<float>") == {
        "return_type": "int2",
        "dimensions": (("get_width", True), ("get_array_size", False)),
    }
    assert codegen.texture_query_size_descriptor("texture2d_ms_array<float>") == {
        "return_type": "int3",
        "dimensions": (
            ("get_width", False),
            ("get_height", False),
            ("get_array_size", False),
        ),
    }
    assert codegen.texture_query_size_descriptor(
        "texture2d_array<uint, access::write>"
    ) == {
        "return_type": "int3",
        "dimensions": (
            ("get_width", False),
            ("get_height", False),
            ("get_array_size", False),
        ),
    }
    descriptor = codegen.texture_query_size_descriptor("texture3d<float>")
    assert (
        codegen.texture_query_size_descriptor_expression("tex", descriptor, "uint(lod)")
        == "int3(tex.get_width(uint(lod)), tex.get_height(uint(lod)), tex.get_depth(uint(lod)))"
    )
    assert codegen.texture_query_size_descriptor("samplerBuffer") is None


def test_metal_texture_query_resource_descriptors():
    codegen = MetalCodeGen()
    codegen.texture_variable_types = {
        "colorImage": "texture2d<float, access::read_write>",
        "colorMap": "texture2d<float>",
        "msTex": "texture2d_ms<float>",
        "msArray": "texture2d_ms_array<float>",
    }

    assert codegen.texture_query_resource_descriptor("colorImage") == {
        "texture_type": "texture2d<float, access::read_write>",
        "storage_image": True,
        "multisample": False,
        "size_descriptor": {
            "return_type": "int2",
            "dimensions": (("get_width", False), ("get_height", False)),
        },
    }
    assert codegen.texture_query_resource_descriptor("msArray") == {
        "texture_type": "texture2d_ms_array<float>",
        "storage_image": False,
        "multisample": True,
        "size_descriptor": {
            "return_type": "int3",
            "dimensions": (
                ("get_width", False),
                ("get_height", False),
                ("get_array_size", False),
            ),
        },
    }
    assert (
        codegen.texture_query_levels_expression("colorImage")
        == "/* unsupported Metal texture query: textureQueryLevels on texture2d<float, access::read_write> */ 0"
    )
    assert codegen.texture_query_levels_expression("msTex") == "1"
    assert (
        codegen.texture_query_levels_expression("colorMap")
        == "int(colorMap.get_num_mip_levels())"
    )
    assert (
        codegen.texture_samples_expression("colorMap")
        == "/* unsupported Metal texture samples query: requires multisample texture */ 0"
    )
    assert codegen.texture_samples_expression("msArray") == (
        "int(msArray.get_num_samples())"
    )
    assert codegen.texture_query_size_expression("colorImage") == (
        "int2(colorImage.get_width(), colorImage.get_height())"
    )


def test_metal_storage_image_size_queries_use_texture_dimensions():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "texture2d<float, access::read_write> colorImage [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture3d<float, access::read_write> volumeImage [[texture(1)]]"
        in generated_code
    )
    assert (
        "texture2d_array<float, access::read_write> layerImage [[texture(2)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> uintImage [[texture(3)]]" in generated_code
    )
    assert (
        "texture3d<int, access::read_write> intVolume [[texture(4)]]" in generated_code
    )
    assert (
        "int2 queryImage2D(texture2d<float, access::read_write> image)"
        in generated_code
    )
    assert (
        "int3 queryImageArray(texture2d_array<float, access::read_write> image)"
        in generated_code
    )
    assert (
        "return int2(image.get_width(), image.get_height()) + int2(image.get_width(), image.get_height());"
        in generated_code
    )
    assert (
        "return int3(image.get_width(), image.get_height(), image.get_depth()) + int3(image.get_width(), image.get_height(), image.get_depth());"
        in generated_code
    )
    assert (
        "return int3(image.get_width(), image.get_height(), image.get_array_size()) + int3(image.get_width(), image.get_height(), image.get_array_size());"
        in generated_code
    )
    assert (
        "int2 a = int2(colorImage.get_width(), colorImage.get_height()) + int2(colorImage.get_width(), colorImage.get_height());"
        in generated_code
    )
    assert (
        "int3 c = int3(layerImage.get_width(), layerImage.get_height(), layerImage.get_array_size()) + int3(layerImage.get_width(), layerImage.get_height(), layerImage.get_array_size());"
        in generated_code
    )
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "get_width(uint" not in generated_code


def test_metal_storage_image_levels_and_lod_queries_emit_diagnostics():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        generated_code.count(
            "/* unsupported Metal texture query: textureQueryLevels on texture2d<float, access::read_write> */ 0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture query: textureQueryLevels on texture3d<float, access::read_write> */ 0"
        )
        == 2
    )
    assert (
        "/* unsupported Metal texture query: textureQueryLevels on texture2d_array<float, access::read_write> */ 0"
        in generated_code
    )
    assert (
        "/* unsupported Metal texture query: textureQueryLevels on texture2d<uint, access::read_write> */ 0"
        in generated_code
    )
    assert (
        "/* unsupported Metal texture query: textureQueryLevels on texture3d<int, access::read_write> */ 0"
        in generated_code
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture query: textureQueryLod on texture2d<float, access::read_write> */ float2(0.0)"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture query: textureQueryLod on texture3d<float, access::read_write> */ float2(0.0)"
        )
        == 2
    )
    assert "textureQueryLevels(" not in generated_code
    assert "textureQueryLod(" not in generated_code
    assert "calculate_unclamped_lod" not in generated_code
    assert "calculate_clamped_lod" not in generated_code
    assert "sampler(" not in generated_code


def test_metal_texture_query_functions():
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
                ivec2 q = query2D(colorMap, linearSampler, input.uv);
                ivec3 qa = queryArray(layerMap);
                ivec2 qm = queryMs(msMap);
                return vec4(q.x + qa.z + qm.x);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture2d_array<float> layerMap [[texture(1)]]" in generated_code
    assert "texture2d_ms<float> msMap [[texture(2)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "int2 size = int2(tex.get_width(uint(1)), tex.get_height(uint(1)));"
        in generated_code
    )
    assert "int levels = int(tex.get_num_mip_levels());" in generated_code
    assert (
        "float2 lod = float2(tex.calculate_unclamped_lod(s, uv), tex.calculate_clamped_lod(s, uv));"
        in generated_code
    )
    assert (
        "return int3(tex.get_width(uint(0)), tex.get_height(uint(0)), tex.get_array_size());"
        in generated_code
    )
    assert "return int2(tex.get_width(), tex.get_height());" in generated_code


def test_metal_implicit_texture_query_lod_helper_threads_declared_sampler():
    shader = """
    shader ImplicitQueryLodDeclaredSampler {
        sampler2D colorMap;
        sampler colorMapSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        vec2 query(vec2 uv) {
            return textureQueryLod(colorMap, uv);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return vec4(query(input.uv), 0.0, 1.0);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "float2 query(float2 uv, texture2d<float> colorMap, sampler colorMapSampler)"
        in generated_code
    )
    assert "colorMap.calculate_unclamped_lod(colorMapSampler, uv)" in generated_code
    assert (
        "fragment float4 fragment_main(FSInput input [[stage_in]], texture2d<float> colorMap [[texture(0)]], sampler colorMapSampler [[sampler(0)]])"
        in generated_code
    )
    assert (
        "return float4(query(input.uv, colorMap, colorMapSampler), 0.0, 1.0);"
        in generated_code
    )


def test_metal_multisample_texture_samples_queries_use_get_num_samples():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d_ms<float> msTex [[texture(0)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(1)]]" in generated_code
    assert "array<texture2d_ms<float>, 4> msTextures [[texture(2)]]" in generated_code
    assert (
        "array<texture2d_ms_array<float>, 4> msArrays [[texture(6)]]" in generated_code
    )
    assert (
        "int querySamples(texture2d_ms<float> tex, texture2d_ms_array<float> texArray)"
        in generated_code
    )
    assert (
        "return int(tex.get_num_samples()) + int(texArray.get_num_samples());"
        in generated_code
    )
    assert (
        "int queryArraySamples(array<texture2d_ms<float>, 4> textures, array<texture2d_ms_array<float>, 4> arrays, int layer)"
        in generated_code
    )
    assert (
        "return int(textures[2].get_num_samples()) + int(arrays[layer].get_num_samples());"
        in generated_code
    )
    assert (
        "querySamples(msTex, msArray) + queryArraySamples(msTextures, msArrays, input.layer)"
        in generated_code
    )
    assert "textureSamples(" not in generated_code


def test_metal_multisample_image_samples_queries_use_get_num_samples():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d_ms<float> msTex [[texture(0)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(1)]]" in generated_code
    assert "array<texture2d_ms<float>, 4> msTextures [[texture(2)]]" in generated_code
    assert (
        "array<texture2d_ms_array<float>, 4> msArrays [[texture(6)]]" in generated_code
    )
    assert (
        "int querySamples(texture2d_ms<float> tex, texture2d_ms_array<float> texArray)"
        in generated_code
    )
    assert (
        "return int(tex.get_num_samples()) + int(texArray.get_num_samples());"
        in generated_code
    )
    assert (
        "int queryArraySamples(array<texture2d_ms<float>, 4> textures, array<texture2d_ms_array<float>, 4> arrays, int layer)"
        in generated_code
    )
    assert (
        "return int(textures[2].get_num_samples()) + int(arrays[layer].get_num_samples());"
        in generated_code
    )
    assert (
        "int(msTex.get_num_samples()) + int(msArray.get_num_samples())"
        in generated_code
    )
    assert "queryArraySamples(msTextures, msArrays, input.layer)" in generated_code
    assert generated_code.count(".get_num_samples()") == 6
    assert "imageSamples(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "unsupported Metal texture samples query" not in generated_code


def test_metal_non_multisample_texture_samples_emit_diagnostics():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported Metal texture samples query: requires multisample texture */ 0"
    )
    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture2d_array<float> layerMap [[texture(1)]]" in generated_code
    assert "texturecube<float> cubeMap [[texture(2)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(3)]]" in generated_code
    assert "array<texture2d<float>, 4> textures" in generated_code
    assert generated_code.count(diagnostic) == 10
    assert "textureSamples(" not in generated_code
    assert "get_num_samples" not in generated_code
    assert " sampler " not in generated_code


def test_metal_storage_image_samples_queries_emit_diagnostics():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported Metal texture samples query: requires multisample texture */ 0"
    )
    assert (
        "texture2d<float, access::read_write> colorImage [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture3d<float, access::read_write> volumeImage [[texture(1)]]"
        in generated_code
    )
    assert (
        "texture2d_array<float, access::read_write> layerImage [[texture(2)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> uintImage [[texture(3)]]" in generated_code
    )
    assert (
        "texture3d<int, access::read_write> intVolume [[texture(4)]]" in generated_code
    )
    assert generated_code.count(diagnostic) == 13
    assert "imageSamples(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "get_num_samples" not in generated_code
    assert " sampler " not in generated_code


def test_metal_storage_image_sampling_and_fetch_emit_diagnostics():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = "unsupported Metal storage image texture operation"
    assert (
        "texture2d<float, access::read_write> colorImage [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture3d<float, access::read_write> volumeImage [[texture(1)]]"
        in generated_code
    )
    assert (
        "texture2d_array<float, access::read_write> layerImage [[texture(2)]]"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 20
    for operation in {
        "texture",
        "textureLod",
        "textureGather",
        "texelFetch",
        "texelFetchOffset",
    }:
        assert f"{operation} on texture2d<float, access::read_write>" in generated_code
    assert ".sample(" not in generated_code
    assert ".read(" not in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGather(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "texelFetchOffset(" not in generated_code
    assert "sampler(mag_filter" not in generated_code
    assert " sampler " not in generated_code


def test_metal_storage_image_compare_calls_emit_diagnostics():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    comparison_diagnostic = "unsupported Metal storage image texture comparison"
    gather_diagnostic = "unsupported Metal storage image texture operation"
    assert (
        "texture2d<float, access::read_write> colorImage [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture3d<float, access::read_write> volumeImage [[texture(1)]]"
        in generated_code
    )
    assert (
        "texture2d_array<float, access::read_write> layerImage [[texture(2)]]"
        in generated_code
    )
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert generated_code.count(comparison_diagnostic) == 14
    assert generated_code.count(gather_diagnostic) == 4
    assert "sample_compare" not in generated_code
    assert "textureCompare(" not in generated_code
    assert "textureGatherCompare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code


def test_metal_multisample_texture_query_levels_use_single_level():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d_ms<float> msTex [[texture(0)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(1)]]" in generated_code
    assert "array<texture2d_ms<float>, 4> msTextures [[texture(2)]]" in generated_code
    assert (
        "array<texture2d_ms_array<float>, 4> msArrays [[texture(6)]]" in generated_code
    )
    assert "int levels2D(texture2d_ms<float> tex)" in generated_code
    assert "int levelsArray(texture2d_ms_array<float> tex)" in generated_code
    assert (
        "int levelsResourceArrays(array<texture2d_ms<float>, 4> textures, array<texture2d_ms_array<float>, 4> arrays, int layer)"
        in generated_code
    )
    assert generated_code.count("return 1;") == 2
    assert "return 1 + 1;" in generated_code
    assert "int c = 1;" in generated_code
    assert "int d = 1;" in generated_code
    assert "int e = 1;" in generated_code
    assert "int f = 1;" in generated_code
    assert "get_num_mip_levels" not in generated_code
    assert "textureQueryLevels(" not in generated_code


def test_metal_multisample_sampling_operations_emit_diagnostics():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d_ms<float> msTex [[texture(0)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(1)]]" in generated_code
    assert (
        "float4 invalid2D(texture2d_ms<float> tex, float2 uv, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float4 invalidArray(texture2d_ms_array<float> tex, float3 uvLayer, float2 ddx, float2 ddy, int2 offset)"
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
            f"unsupported Metal multisample texture call: {func_name} on texture2d_ms<float>"
            in generated_code
        )
        assert (
            f"unsupported Metal multisample texture call: {func_name} on texture2d_ms_array<float>"
            in generated_code
        )

    assert ".sample(" not in generated_code
    assert ".gather(" not in generated_code


def test_metal_multisample_compare_operations_emit_diagnostics():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d_ms<float> msTex [[texture(0)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(1)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code

    for func_name in {
        "textureCompare",
        "textureCompareOffset",
        "textureCompareLod",
        "textureCompareLodOffset",
        "textureCompareGrad",
        "textureCompareGradOffset",
    }:
        assert (
            f"unsupported Metal multisample texture comparison: {func_name} on texture2d_ms<float>"
            in generated_code
        )
        assert (
            f"unsupported Metal multisample texture comparison: {func_name} on texture2d_ms_array<float>"
            in generated_code
        )

    for func_name in {
        "textureCompareProj",
        "textureCompareProjLod",
        "textureCompareProjGradOffset",
    }:
        assert (
            f"unsupported Metal multisample texture comparison: {func_name} on texture2d_ms<float>"
            in generated_code
        )

    for func_name in {"textureGatherCompare", "textureGatherCompareOffset"}:
        assert (
            f"unsupported Metal multisample texture gather comparison: {func_name} on texture2d_ms<float>"
            in generated_code
        )
        assert (
            f"unsupported Metal multisample texture gather comparison: {func_name} on texture2d_ms_array<float>"
            in generated_code
        )

    assert ".sample_compare(" not in generated_code
    assert ".gather_compare(" not in generated_code
    assert "textureCompare(" not in generated_code
    assert "textureGatherCompare(" not in generated_code


def test_metal_multisample_compare_diagnostics_do_not_create_implicit_samplers():
    shader = """
    shader UnsupportedMultisampleCompareImplicitSampler {
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler msTexSampler;
        sampler msArraySampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            float depth;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float cmp = textureCompare(msTex, input.uv, input.depth);
                vec4 gathered = textureGatherCompare(msArray, input.uvLayer, input.depth);
                return vec4(cmp) + gathered;
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d_ms<float> msTex [[texture(0)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(1)]]" in generated_code
    assert "msTexSampler [[sampler" not in generated_code
    assert "msArraySampler [[sampler" not in generated_code
    assert "sampler(mag_filter" not in generated_code
    assert (
        "unsupported Metal multisample texture comparison: textureCompare on texture2d_ms<float>"
        in generated_code
    )
    assert (
        "unsupported Metal multisample texture gather comparison: textureGatherCompare on texture2d_ms_array<float>"
        in generated_code
    )
    assert ".sample_compare(" not in generated_code
    assert ".gather_compare(" not in generated_code


def test_metal_multisample_texture_query_lod_emits_diagnostics():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d_ms<float> msTex [[texture(0)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(1)]]" in generated_code
    assert "float2 query2D(texture2d_ms<float> tex, float2 uv)" in generated_code
    assert (
        "float2 queryArray(texture2d_ms_array<float> tex, float3 uvLayer)"
        in generated_code
    )
    assert "calculate_unclamped_lod" not in generated_code
    assert "calculate_clamped_lod" not in generated_code
    assert "textureQueryLod(" not in generated_code
    assert (
        generated_code.count(
            "unsupported Metal multisample texture query: textureQueryLod on texture2d_ms<float>"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported Metal multisample texture query: textureQueryLod on texture2d_ms_array<float>"
        )
        == 2
    )


def test_metal_multisample_texel_fetch_offsets_emit_diagnostics():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported Metal texel fetch offset: "
        "multisample textures do not support offsets */ float4(0.0)"
    )
    assert "texture2d_ms<float> msTex [[texture(0)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(1)]]" in generated_code
    assert (
        "float4 offset2D(texture2d_ms<float> tex, int2 pixel, int sampleIndex, int2 offset)"
        in generated_code
    )
    assert (
        "float4 offsetArray(texture2d_ms_array<float> tex, int3 pixelLayer, int sampleIndex, int2 offset)"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 4
    assert "texelFetchOffset(" not in generated_code
    assert ".read(" not in generated_code


def test_metal_cube_texel_fetches_emit_diagnostics():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texturecube<float> cubeMap [[texture(0)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert (
        "float4 fetchCube(texturecube<float> tex, int3 cubeCoord, int lod, int3 offset)"
        in generated_code
    )
    assert (
        "float4 fetchCubeArray(texturecube_array<float> tex, int4 cubeLayerCoord, int lod, int3 offset)"
        in generated_code
    )
    assert (
        generated_code.count(
            "unsupported Metal texel fetch: texelFetch on texturecube<float>"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported Metal texel fetch: texelFetchOffset on texturecube<float>"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported Metal texel fetch: texelFetch on texturecube_array<float>"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported Metal texel fetch: texelFetchOffset on texturecube_array<float>"
        )
        == 2
    )
    assert ".read(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "texelFetchOffset(" not in generated_code


def test_metal_array_shadow_texture_query_functions():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "depth2d_array<float> shadowArray [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(1)]]" in generated_code
    assert (
        "array<depth2d_array<float>, 4> shadowArrays [[texture(2)]]" in generated_code
    )
    assert (
        "array<depthcube_array<float>, 4> cubeShadowArrays [[texture(6)]]"
        in generated_code
    )
    assert "int3 query2DArrayShadow(depth2d_array<float> tex)" in generated_code
    assert "int3 queryCubeArrayShadow(depthcube_array<float> tex)" in generated_code
    assert (
        "int3 queryArrayElements(array<depth2d_array<float>, 4> shadowArrays, array<depthcube_array<float>, 4> cubeShadowArrays)"
        in generated_code
    )
    assert (
        "int3 size = int3(tex.get_width(uint(1)), tex.get_height(uint(1)), tex.get_array_size());"
        in generated_code
    )
    assert (
        "int3 size = int3(tex.get_width(uint(0)), tex.get_height(uint(0)), tex.get_array_size());"
        in generated_code
    )
    assert "int levels = int(tex.get_num_mip_levels());" in generated_code
    assert (
        "int3 arraySize = int3(shadowArrays[2].get_width(uint(1)), shadowArrays[2].get_height(uint(1)), shadowArrays[2].get_array_size());"
        in generated_code
    )
    assert (
        "int3 cubeSize = int3(cubeShadowArrays[3].get_width(uint(0)), cubeShadowArrays[3].get_height(uint(0)), cubeShadowArrays[3].get_array_size());"
        in generated_code
    )
    assert (
        "int arrayLevels = int(shadowArrays[2].get_num_mip_levels());" in generated_code
    )
    assert (
        "int cubeLevels = int(cubeShadowArrays[3].get_num_mip_levels());"
        in generated_code
    )
    assert "sample_compare" not in generated_code


def test_metal_direct_stage_texture_queries_mix_size_levels_and_lod():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )
    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture2d_array<float> layerMap [[texture(1)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(2)]]" in generated_code
    assert "depth2d<float> shadowMap [[texture(3)]]" in generated_code
    assert "depth2d_array<float> shadowArray [[texture(4)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(5)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "int2 colorSize = int2(colorMap.get_width(uint(1)), colorMap.get_height(uint(1)));"
        in generated_code
    )
    assert (
        "int3 layerSize = int3(layerMap.get_width(uint(2)), layerMap.get_height(uint(2)), layerMap.get_array_size());"
        in generated_code
    )
    assert (
        "int3 cubeSize = int3(cubeArray.get_width(uint(0)), cubeArray.get_height(uint(0)), cubeArray.get_array_size());"
        in generated_code
    )
    assert (
        "int2 shadowSize = int2(shadowMap.get_width(uint(0)), shadowMap.get_height(uint(0)));"
        in generated_code
    )
    assert (
        "int3 shadowArraySize = int3(shadowArray.get_width(uint(1)), shadowArray.get_height(uint(1)), shadowArray.get_array_size());"
        in generated_code
    )
    assert (
        "int3 cubeShadowSize = int3(cubeShadowArray.get_width(uint(0)), cubeShadowArray.get_height(uint(0)), cubeShadowArray.get_array_size());"
        in generated_code
    )
    assert "int colorLevels = int(colorMap.get_num_mip_levels());" in generated_code
    assert "int shadowLevels = int(shadowArray.get_num_mip_levels());" in generated_code
    assert (
        "float2 colorLod = float2(colorMap.calculate_unclamped_lod(linearSampler, input.uv), colorMap.calculate_clamped_lod(linearSampler, input.uv));"
        in generated_code
    )
    assert (
        "float2 layerLod = float2(layerMap.calculate_unclamped_lod(linearSampler, input.uvLayer.xy), layerMap.calculate_clamped_lod(linearSampler, input.uvLayer.xy));"
        in generated_code
    )
    assert (
        "float2 cubeLod = float2(cubeArray.calculate_unclamped_lod(linearSampler, input.cubeLayer.xyz), cubeArray.calculate_clamped_lod(linearSampler, input.cubeLayer.xyz));"
        in generated_code
    )
    assert (
        f"float2 implicitLayerLod = float2(layerMap.calculate_unclamped_lod({default_sampler}, input.uvLayer.xy), layerMap.calculate_clamped_lod({default_sampler}, input.uvLayer.xy));"
        in generated_code
    )
    assert (
        f"float2 implicitCubeShadowLod = float2(cubeShadowArray.calculate_unclamped_lod({default_sampler}, input.cubeLayer.xyz), cubeShadowArray.calculate_clamped_lod({default_sampler}, input.cubeLayer.xyz));"
        in generated_code
    )
    assert "textureQueryLod(" not in generated_code


def test_metal_array_texture_query_lod_uses_non_layer_coordinates():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "texture2d_array<float> layerMap [[texture(0)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert "array<texture2d_array<float>, 4> layerMaps [[texture(2)]]" in generated_code
    assert (
        "array<texturecube_array<float>, 4> cubeArrays [[texture(6)]]" in generated_code
    )
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert "array<sampler, 4> linearSamplers [[sampler(1)]]" in generated_code
    assert (
        "float2 queryArrayLod(texture2d_array<float> tex, sampler s, float3 uvLayer)"
        in generated_code
    )
    assert "tex.calculate_unclamped_lod(s, uvLayer.xy)" in generated_code
    assert "tex.calculate_clamped_lod(s, uvLayer.xy)" in generated_code
    assert (
        "float2 queryCubeArrayLod(texturecube_array<float> tex, sampler s, float4 cubeLayer)"
        in generated_code
    )
    assert "tex.calculate_unclamped_lod(s, cubeLayer.xyz)" in generated_code
    assert "tex.calculate_clamped_lod(s, cubeLayer.xyz)" in generated_code
    assert (
        "float2 queryArrayElementLod(array<texture2d_array<float>, 4> layerMaps, array<sampler, 4> linearSamplers, float3 uvLayer)"
        in generated_code
    )
    assert (
        "layerMaps[2].calculate_unclamped_lod(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "layerMaps[2].calculate_clamped_lod(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "float2 queryCubeArrayElementLod(array<texturecube_array<float>, 4> cubeArrays, array<sampler, 4> linearSamplers, float4 cubeLayer)"
        in generated_code
    )
    assert (
        "cubeArrays[3].calculate_unclamped_lod(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert (
        "cubeArrays[3].calculate_clamped_lod(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert "calculate_unclamped_lod(s, uvLayer)" not in generated_code
    assert "calculate_unclamped_lod(s, cubeLayer)" not in generated_code


def test_metal_nested_array_texture_query_lod_threads_resource_arrays():
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

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "array<texture2d_array<float>, 4> layerMaps [[texture(0)]]" in generated_code
    assert (
        "array<texturecube_array<float>, 4> cubeArrays [[texture(4)]]" in generated_code
    )
    assert "array<sampler, 4> linearSamplers [[sampler(0)]]" in generated_code
    assert "sampler cubeArraysSampler [[sampler(4)]]" in generated_code
    assert (
        "float2 explicitLeaf(float3 uvLayer, array<texture2d_array<float>, 4> layerMaps, array<sampler, 4> linearSamplers)"
        in generated_code
    )
    assert (
        "float2 explicitMid(float3 uvLayer, array<texture2d_array<float>, 4> layerMaps, array<sampler, 4> linearSamplers)"
        in generated_code
    )
    assert "return explicitLeaf(uvLayer, layerMaps, linearSamplers);" in generated_code
    assert (
        "layerMaps[2].calculate_unclamped_lod(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "float2 implicitLeaf(float4 cubeLayer, array<texturecube_array<float>, 4> cubeArrays, sampler cubeArraysSampler)"
        in generated_code
    )
    assert (
        "float2 implicitMid(float4 cubeLayer, array<texturecube_array<float>, 4> cubeArrays, sampler cubeArraysSampler)"
        in generated_code
    )
    assert (
        "return implicitLeaf(cubeLayer, cubeArrays, cubeArraysSampler);"
        in generated_code
    )
    assert (
        "cubeArrays[3].calculate_unclamped_lod(cubeArraysSampler, cubeLayer.xyz)"
        in generated_code
    )
    assert "explicitMid(input.uvLayer, layerMaps, linearSamplers)" in generated_code
    assert (
        "implicitMid(input.cubeLayer, cubeArrays, cubeArraysSampler)" in generated_code
    )
    assert "textureQueryLod(" not in generated_code


def test_metal_shadow_array_texture_query_lod_uses_non_layer_coordinates():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "depth2d_array<float> shadowArray [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(1)]]" in generated_code
    assert (
        "array<depth2d_array<float>, 4> shadowArrays [[texture(2)]]" in generated_code
    )
    assert (
        "array<depthcube_array<float>, 4> cubeShadowArrays [[texture(6)]]"
        in generated_code
    )
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert "array<sampler, 4> linearSamplers [[sampler(1)]]" in generated_code
    assert (
        "float2 queryArrayLod(depth2d_array<float> tex, sampler s, float3 uvLayer)"
        in generated_code
    )
    assert "tex.calculate_unclamped_lod(s, uvLayer.xy)" in generated_code
    assert "tex.calculate_clamped_lod(s, uvLayer.xy)" in generated_code
    assert (
        "float2 queryCubeArrayLod(depthcube_array<float> tex, sampler s, float4 cubeLayer)"
        in generated_code
    )
    assert "tex.calculate_unclamped_lod(s, cubeLayer.xyz)" in generated_code
    assert "tex.calculate_clamped_lod(s, cubeLayer.xyz)" in generated_code
    assert (
        "float2 queryArrayElementLod(array<depth2d_array<float>, 4> shadowArrays, array<sampler, 4> linearSamplers, float3 uvLayer)"
        in generated_code
    )
    assert (
        "shadowArrays[2].calculate_unclamped_lod(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "shadowArrays[2].calculate_clamped_lod(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "float2 queryCubeArrayElementLod(array<depthcube_array<float>, 4> cubeShadowArrays, array<sampler, 4> linearSamplers, float4 cubeLayer)"
        in generated_code
    )
    assert (
        "cubeShadowArrays[3].calculate_unclamped_lod(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert (
        "cubeShadowArrays[3].calculate_clamped_lod(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert "calculate_unclamped_lod(s, uvLayer)" not in generated_code
    assert "calculate_unclamped_lod(s, cubeLayer)" not in generated_code


def test_metal_mixed_cube_shadow_query_lod_and_compare_keep_coordinate_shapes():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "depthcube<float> cubeShadow [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(1)]]" in generated_code
    assert "sampler shadowSampler [[sampler(0)]]" in generated_code
    assert (
        "float inspectCubeShadow(depthcube<float> tex, sampler s, float3 direction, float depth, float lod, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "float2 lodValue = float2(tex.calculate_unclamped_lod(s, direction), tex.calculate_clamped_lod(s, direction));"
        in generated_code
    )
    assert "float cmp = tex.sample_compare(s, direction, depth);" in generated_code
    assert (
        "float cmpLod = tex.sample_compare(s, direction, depth, level(lod));"
        in generated_code
    )
    assert (
        "float grad = tex.sample_compare(s, direction, depth, gradientcube(ddx, ddy));"
        in generated_code
    )
    assert (
        "float inspectCubeArrayShadow(depthcube_array<float> tex, sampler s, float4 cubeLayer, float depth, float lod, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "float2 lodValue = float2(tex.calculate_unclamped_lod(s, cubeLayer.xyz), tex.calculate_clamped_lod(s, cubeLayer.xyz));"
        in generated_code
    )
    assert (
        "float cmp = tex.sample_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth);"
        in generated_code
    )
    assert (
        "float cmpLod = tex.sample_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth, level(lod));"
        in generated_code
    )
    assert (
        "float grad = tex.sample_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth, gradientcube(ddx, ddy));"
        in generated_code
    )
    assert (
        "inspectCubeShadow(cubeShadow, shadowSampler, input.direction, input.depth, input.lod, input.ddx, input.ddy)"
        in generated_code
    )
    assert (
        "inspectCubeArrayShadow(cubeShadowArray, shadowSampler, input.cubeLayer, input.depth, input.lod, input.ddx, input.ddy)"
        in generated_code
    )
    assert "textureQueryLod(" not in generated_code
    assert "textureCompare(" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGrad(" not in generated_code


def test_metal_implicit_shadow_array_texture_query_lod_uses_default_sampler():
    shader = """
    shader ImplicitShadowArrayTextureQueryLod {
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeShadowArrays[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
        };

        vec2 queryArrayLod(sampler2DArrayShadow tex, vec3 uvLayer) {
            return textureQueryLod(tex, uvLayer);
        }

        vec2 queryCubeArrayLod(samplerCubeArrayShadow tex, vec4 cubeLayer) {
            return textureQueryLod(tex, cubeLayer);
        }

        vec2 queryArrayElementLod(sampler2DArrayShadow shadowArrays[], vec3 uvLayer) {
            return textureQueryLod(shadowArrays[2], uvLayer);
        }

        vec2 queryCubeArrayElementLod(samplerCubeArrayShadow cubeShadowArrays[], vec4 cubeLayer) {
            return textureQueryLod(cubeShadowArrays[3], cubeLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = textureQueryLod(shadowArray, input.uvLayer);
                vec2 b = textureQueryLod(cubeShadowArray, input.cubeLayer);
                vec2 c = queryArrayLod(shadowArray, input.uvLayer);
                vec2 d = queryCubeArrayLod(cubeShadowArray, input.cubeLayer);
                vec2 e = queryArrayElementLod(shadowArrays, input.uvLayer);
                vec2 f = queryCubeArrayElementLod(cubeShadowArrays, input.cubeLayer);
                return vec4(a.x + b.y, c.x + d.y, e.x + f.y, 1.0);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "depth2d_array<float> shadowArray [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(1)]]" in generated_code
    assert (
        "array<depth2d_array<float>, 4> shadowArrays [[texture(2)]]" in generated_code
    )
    assert (
        "array<depthcube_array<float>, 4> cubeShadowArrays [[texture(6)]]"
        in generated_code
    )
    assert "sampler linearSampler" not in generated_code
    assert "linearSamplers" not in generated_code
    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"
    assert (
        f"shadowArray.calculate_unclamped_lod({default_sampler}, input.uvLayer.xy)"
        in generated_code
    )
    assert (
        f"cubeShadowArray.calculate_unclamped_lod({default_sampler}, input.cubeLayer.xyz)"
        in generated_code
    )
    assert (
        f"tex.calculate_unclamped_lod({default_sampler}, uvLayer.xy)" in generated_code
    )
    assert (
        f"tex.calculate_unclamped_lod({default_sampler}, cubeLayer.xyz)"
        in generated_code
    )
    assert (
        f"shadowArrays[2].calculate_unclamped_lod({default_sampler}, uvLayer.xy)"
        in generated_code
    )
    assert (
        f"cubeShadowArrays[3].calculate_unclamped_lod({default_sampler}, cubeLayer.xyz)"
        in generated_code
    )
    assert (
        "calculate_unclamped_lod(sampler(mag_filter::linear, min_filter::linear), uvLayer)"
        not in generated_code
    )
    assert (
        "calculate_unclamped_lod(sampler(mag_filter::linear, min_filter::linear), cubeLayer)"
        not in generated_code
    )


def test_metal_texture_operation_variants():
    shader = """
    shader TextureOps {
        sampler2D colorMap;

        struct VSOutput {
            vec2 uv;
            ivec2 pixel;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 lodColor = textureLod(colorMap, input.uv, 1.0);
                vec4 gradColor = textureGrad(colorMap, input.uv, vec2(0.1), vec2(0.2));
                vec4 fetched = texelFetch(colorMap, input.pixel, 0);
                vec4 gathered = textureGather(colorMap, input.uv);
                return lodColor + gradColor + fetched + gathered;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "colorMap.sample(" in generated_code
    assert "level(1.0)" in generated_code
    assert "gradient2d(float2(0.1), float2(0.2))" in generated_code
    assert "colorMap.read(input.pixel, 0)" in generated_code
    assert "colorMap.gather(" in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGrad(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "textureGather(" not in generated_code


def test_metal_texture_and_sampler_bindings_are_independent():
    shader = """
    shader ExplicitSampler {
        sampler2D colorMap;
        sampler colorMapSampler;

        struct VSOutput {
            vec2 uv;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return texture(colorMap, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "sampler colorMapSampler [[sampler(0)]]" in generated_code
    assert "colorMap.sample(colorMapSampler, input.uv)" in generated_code


def test_metal_explicit_sampler_argument():
    shader = """
    shader ExplicitSampler {
        sampler2D colorMap;
        sampler linearSampler;

        struct VSOutput {
            vec2 uv;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return texture(colorMap, linearSampler, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert "colorMap.sample(linearSampler, input.uv)" in generated_code


def test_metal_sampler_parameter_texture_call():
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
                return sampleColor(linearSampler, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleColor(sampler sampleState, float2 uv, texture2d<float> colorMap)"
        in generated_code
    )
    assert "colorMap.sample(sampleState, uv)" in generated_code
    assert "sampleColor(linearSampler, input.uv, colorMap)" in generated_code
    assert "sampleState [[stage_in]]" not in generated_code


def test_metal_texture_and_sampler_parameters():
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
                return sampleColor(colorMap, linearSampler, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleColor(texture2d<float> tex, sampler sampleState, float2 uv)"
        in generated_code
    )
    assert "tex.sample(sampleState, uv)" in generated_code
    assert "sampleColor(colorMap, linearSampler, input.uv)" in generated_code
    assert "tex [[stage_in]]" not in generated_code


def test_metal_struct_member_sampler_expression_is_explicit_sampler():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct SamplerPack" in generated_code
    assert "sampler samplers[2];" in generated_code
    assert (
        "float4 samplePacked(texture2d<float> tex, SamplerPack pack, int index, float2 uv)"
        in generated_code
    )
    assert "return tex.sample(pack.samplers[index], uv);" in generated_code
    assert "return samplePacked(colorMap, pack, 0, float2(0.5));" in generated_code
    assert "sampler(mag_filter::linear, min_filter::linear)" not in generated_code
    assert "bias(uv)" not in generated_code


def test_metal_implicit_sampler_for_texture_parameter():
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
                return sampleColor(colorMap, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "float4 sampleColor(texture2d<float> tex, float2 uv)" in generated_code
    assert (
        "tex.sample(sampler(mag_filter::linear, min_filter::linear), uv)"
        in generated_code
    )
    assert "sampleColor(colorMap, input.uv)" in generated_code


def test_metal_implicit_sampler_array_indexes_match_texture_array_elements():
    shader = """
    shader ImplicitSamplerArrayElements {
        sampler2D textures[4];
        sampler texturesSampler[4];

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleGlobal(int index, vec2 uv) {
            return texture(textures[index], uv);
        }

        vec4 sampleParam(sampler2D textures[4], sampler texturesSampler[4], int index, vec2 uv) {
            return texture(textures[index], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleGlobal(2, input.uv) + sampleParam(textures, texturesSampler, 1, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "array<sampler, 4> texturesSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleGlobal(int index, float2 uv, array<texture2d<float>, 4> textures, array<sampler, 4> texturesSampler)"
        in generated_code
    )
    assert (
        "float4 sampleParam(array<texture2d<float>, 4> textures, array<sampler, 4> texturesSampler, int index, float2 uv)"
        in generated_code
    )
    assert (
        generated_code.count(
            "return textures[index].sample(texturesSampler[index], uv);"
        )
        == 2
    )
    assert "textures[index].sample(texturesSampler, uv)" not in generated_code
    assert (
        "sampleGlobal(2, input.uv, textures, texturesSampler) + sampleParam(textures, texturesSampler, 1, input.uv)"
        in generated_code
    )


def test_metal_shadow_texture_compare():
    shader = """
    shader ShadowTexture {
        sampler2DShadow shadowMap;
        sampler shadowSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return textureCompare(shadowMap, shadowSampler, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert "sampler shadowSampler [[sampler(0)]]" in generated_code
    assert (
        "shadowMap.sample_compare(shadowSampler, input.uv, input.depth)"
        in generated_code
    )
    assert "textureCompare(" not in generated_code


def test_metal_shadow_texture_array_compare_with_indexed_sampler_array():
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
                return shadowLayer(shadowMaps, shadowSamplers, input.layer, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 4> shadowMaps, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[layer].sample_compare(shadowSamplers[layer], uv, depth)"
        in generated_code
    )
    assert "depth2d<float> shadowMaps[4] [[stage_in]]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_fixed_shadow_texture_and_sampler_arrays_keep_declared_size_with_constant_indices():
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int LAYER = 2;" in generated_code
    assert "array<depth2d<float>, 6> shadowMaps [[texture(0)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(6)]]" in generated_code
    assert "array<sampler, 6> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(6)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 6> shadowMaps, array<sampler, 6> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[LAYER].sample_compare(shadowSamplers[LAYER], uv, depth)"
        in generated_code
    )
    assert (
        "shadowMaps[1 + 2].sample_compare(shadowSamplers[1 + 2], uv, depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 4> shadowMaps" not in generated_code
    assert "depth2d<float> afterShadow [[texture(4)]]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_fixed_shadow_texture_and_sampler_arrays_resolve_constant_declared_size_for_bindings():
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int BASE_COUNT = 2;" in generated_code
    assert "constant int SHADOW_COUNT = BASE_COUNT * 3;" in generated_code
    assert (
        "array<depth2d<float>, SHADOW_COUNT> shadowMaps [[texture(0)]]"
        in generated_code
    )
    assert "depth2d<float> afterShadow [[texture(6)]]" in generated_code
    assert (
        "array<sampler, SHADOW_COUNT> shadowSamplers [[sampler(0)]]" in generated_code
    )
    assert "sampler afterSampler [[sampler(6)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, SHADOW_COUNT> shadowMaps, array<sampler, SHADOW_COUNT> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[2].sample_compare(shadowSamplers[2], uv, depth)" in generated_code
    )
    assert "depth2d<float> afterShadow [[texture(1)]]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_fixed_shadow_texture_and_sampler_arrays_resolve_inline_declared_size_expression_for_bindings():
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<depth2d<float>, 2 * 3> shadowMaps [[texture(0)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(6)]]" in generated_code
    assert "array<sampler, 2 * 3> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(6)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 2 * 3> shadowMaps, array<sampler, 2 * 3> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[2].sample_compare(shadowSamplers[2], uv, depth)" in generated_code
    )
    assert "depth2d<float> afterShadow [[texture(1)]]" not in generated_code
    assert "BinaryOpNode" not in generated_code
    assert "None" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_fixed_shadow_texture_arrays_preserve_parenthesized_and_unary_declared_sizes():
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, unaryShadowMaps, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "array<depth2d<float>, (2 + 1) * 2> shadowMaps [[texture(0)]]" in generated_code
    )
    assert "array<depth2d<float>, +6> unaryShadowMaps [[texture(6)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(12)]]" in generated_code
    assert "array<sampler, (2 + 1) * 2> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(6)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, (2 + 1) * 2> shadowMaps, array<sampler, (2 + 1) * 2> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[2].sample_compare(shadowSamplers[2], uv, depth)" in generated_code
    )
    assert (
        "unaryShadowMaps[2].sample_compare(shadowSamplers[2], uv, depth)"
        in generated_code
    )
    assert "depth2d<float> afterShadow [[texture(6)]]" not in generated_code
    assert "2 + 1 * 2" not in generated_code
    assert "BinaryOpNode" not in generated_code
    assert "None" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_unsized_shadow_texture_and_sampler_arrays_infer_helper_size():
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(4)]]" in generated_code
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(4)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 4> shadowMaps, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[3].sample_compare(shadowSamplers[3], uv, depth)" in generated_code
    )
    assert (
        "shadowMaps[1].sample_compare(shadowSamplers[1], uv, depth)" in generated_code
    )
    assert (
        "afterShadow.sample_compare(afterSampler, input.uv, input.depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "array<sampler, 1> shadowSamplers" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_unsized_shadow_texture_and_sampler_arrays_infer_transitive_helper_size():
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
                float arrayShadow = shadowMid(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "array<depth2d<float>, 5> shadowMaps [[texture(0)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(5)]]" in generated_code
    assert "array<sampler, 5> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(5)]]" in generated_code
    assert (
        "float shadowDeep(array<depth2d<float>, 5> shadowMaps, array<sampler, 5> shadowSamplers"
        in generated_code
    )
    assert (
        "float shadowMid(array<depth2d<float>, 5> shadowMaps, array<sampler, 5> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[4].sample_compare(shadowSamplers[4], uv, depth)" in generated_code
    )
    assert (
        "shadowMaps[1].sample_compare(shadowSamplers[1], uv, depth)" in generated_code
    )
    assert "shadowDeep(shadowMaps, shadowSamplers, uv, depth)" in generated_code
    assert (
        "shadowMid(shadowMaps, shadowSamplers, input.uv, input.depth)" in generated_code
    )
    assert (
        "afterShadow.sample_compare(afterSampler, input.uv, input.depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "array<sampler, 1> shadowSamplers" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_unsized_shadow_texture_and_sampler_arrays_preserve_dynamic_indexing():
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.layer, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(4)]]" in generated_code
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(4)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 4> shadowMaps, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[layer].sample_compare(shadowSamplers[layer], uv, depth)"
        in generated_code
    )
    assert (
        "shadowMaps[3].sample_compare(shadowSamplers[3], uv, depth)" in generated_code
    )
    assert (
        "shadowLayer(shadowMaps, shadowSamplers, input.layer, input.uv, input.depth)"
        in generated_code
    )
    assert (
        "afterShadow.sample_compare(afterSampler, input.uv, input.depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "array<sampler, 1> shadowSamplers" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_unsized_shadow_texture_and_sampler_arrays_ignore_unsupported_indices():
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.layer, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    dynamic_code = MetalCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = MetalCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "array<depth2d<float>, 1> shadowMaps [[texture(0)]]" in dynamic_code
    assert "depth2d<float> afterShadow [[texture(1)]]" in dynamic_code
    assert "array<sampler, 1> shadowSamplers [[sampler(0)]]" in dynamic_code
    assert "sampler afterSampler [[sampler(1)]]" in dynamic_code
    assert (
        "float shadowLayer(array<depth2d<float>, 1> shadowMaps, array<sampler, 1> shadowSamplers"
        in dynamic_code
    )
    assert (
        "shadowMaps[layer].sample_compare(shadowSamplers[layer], uv, depth)"
        in dynamic_code
    )
    assert "array<depth2d<float>, 2> shadowMaps" not in dynamic_code
    assert "depth2d<float> afterShadow [[texture(2)]]" not in dynamic_code

    assert "array<depth2d<float>, 1> shadowMaps [[texture(0)]]" in negative_code
    assert "depth2d<float> afterShadow [[texture(1)]]" in negative_code
    assert "array<sampler, 1> shadowSamplers [[sampler(0)]]" in negative_code
    assert "sampler afterSampler [[sampler(1)]]" in negative_code
    assert (
        "shadowMaps[-1].sample_compare(shadowSamplers[-1], uv, depth)" in negative_code
    )
    assert "array<depth2d<float>, 0> shadowMaps" not in negative_code
    assert "depth2d<float> afterShadow [[texture(0)]]" not in negative_code


def test_metal_unsized_shadow_texture_and_sampler_arrays_infer_constant_expression_size():
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<depth2d<float>, 5> shadowMaps [[texture(0)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(5)]]" in generated_code
    assert "array<sampler, 5> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(5)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 5> shadowMaps, array<sampler, 5> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[2 * 2].sample_compare(shadowSamplers[2 * 2], uv, depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_unsized_shadow_texture_and_sampler_arrays_infer_named_constant_size():
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int BASE = 2;" in generated_code
    assert "constant int LAYER = BASE * 2;" in generated_code
    assert "array<depth2d<float>, 5> shadowMaps [[texture(0)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(5)]]" in generated_code
    assert "array<sampler, 5> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(5)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 5> shadowMaps, array<sampler, 5> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[LAYER].sample_compare(shadowSamplers[LAYER], uv, depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_fixed_shadow_texture_array_rejects_mismatched_fixed_helper_size():
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
        MetalCodeGen().generate(crosstl.translator.parse(shader))


def test_metal_fixed_shadow_texture_array_widens_unsized_helper():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert (
        "float shadowUnsized(array<depth2d<float>, 4> shadowMaps, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[2].sample_compare(shadowSamplers[2], uv, depth)" in generated_code
    )
    assert (
        "shadowUnsized(shadowMaps, shadowSamplers, input.uv, input.depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_transitive_shadow_texture_array_shadowed_const_index_stays_dynamic():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int COUNT = 4;" in generated_code
    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert (
        "float leaf(array<depth2d<float>, 4> shadowMaps, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "float passThrough(array<depth2d<float>, 4> shadowMaps, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert generated_code.count("int COUNT = 0;") == 2
    assert (
        "shadowMaps[COUNT].sample_compare(shadowSamplers[COUNT], uv, depth)"
        in generated_code
    )
    assert "leaf(shadowMaps, shadowSamplers, uv, depth)" in generated_code
    assert (
        "passThrough(shadowMaps, shadowSamplers, input.uv, input.depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_transitive_shadow_texture_array_unshadowed_const_index_conflict_raises():
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
        MetalCodeGen().generate(crosstl.translator.parse(shader))


def test_metal_shadow_compare_sampler_parameter():
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
                return sampleShadow(shadowSampler, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert "sampler shadowSampler [[sampler(0)]]" in generated_code
    assert (
        "float sampleShadow(sampler compareSampler, float2 uv, float depth, depth2d<float> shadowMap)"
        in generated_code
    )
    assert "shadowMap.sample_compare(compareSampler, uv, depth)" in generated_code
    assert (
        "sampleShadow(shadowSampler, input.uv, input.depth, shadowMap)"
        in generated_code
    )
    assert "compareSampler [[stage_in]]" not in generated_code


def test_metal_shadow_texture_and_sampler_parameters():
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
                return sampleShadow(shadowMap, shadowSampler, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert "sampler shadowSampler [[sampler(0)]]" in generated_code
    assert (
        "float sampleShadow(depth2d<float> tex, sampler compareSampler, float2 uv, float depth)"
        in generated_code
    )
    assert "tex.sample_compare(compareSampler, uv, depth)" in generated_code
    assert (
        "sampleShadow(shadowMap, shadowSampler, input.uv, input.depth)"
        in generated_code
    )
    assert "tex [[stage_in]]" not in generated_code


def test_metal_implicit_sampler_for_shadow_texture_parameter():
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
                return sampleShadow(shadowMap, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert (
        "float sampleShadow(depth2d<float> tex, float2 uv, float depth)"
        in generated_code
    )
    assert (
        "tex.sample_compare(sampler(mag_filter::linear, min_filter::linear), uv, depth)"
        in generated_code
    )
    assert "sampleShadow(shadowMap, input.uv, input.depth)" in generated_code


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
    code_gen = MetalCodeGen()
    generated_code = code_gen.generate(ast)

    for expected in expected_outputs:
        assert expected in generated_code


def test_metal_sampler_1d_array_sampling_and_queries():
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
            ivec2 literalDims = textureSize(tex, 0);
            int levels = textureQueryLevels(tex);
            vec2 lodInfo = textureQueryLod(tex, samp, uvLayer);
            return texture(tex, samp, uvLayer)
                + textureLod(tex, samp, uvLayer, lod)
                + texelFetch(tex, pixelLayer, 0)
                + texelFetchOffset(tex, pixelLayer, 0, offset)
                + vec4(dims.x + literalDims.x, dims.y + literalDims.y, levels, lodInfo.x + lodInfo.y);
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
    generated_code = MetalCodeGen().generate(ast)

    assert "texture1d_array<float> lineArray [[texture(0)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLineArray(texture1d_array<float> tex, sampler samp, float2 uvLayer, int2 pixelLayer, int lod, int offset)"
        in generated_code
    )
    assert (
        "int2 dims = /* unsupported Metal texture size query: textureSize on "
        "texture1d_array<float> requires a compile-time literal mip level */ "
        "int2(0);" in generated_code
    )
    assert (
        "int2 literalDims = int2(tex.get_width(uint(0)), tex.get_array_size());"
        in generated_code
    )
    assert "int(tex.get_num_mip_levels())" in generated_code
    assert "tex.sample(samp, uvLayer.x, uint(uvLayer.y))" in generated_code
    assert (
        "/* unsupported Metal texture sampling: textureLod on "
        "texture1d_array<float> supports only implicit sampling */ float4(0.0)"
        in generated_code
    )
    assert (
        "float2 lodInfo = /* unsupported Metal texture query: "
        "textureQueryLod on texture1d_array<float> */ float2(0.0);" in generated_code
    )
    assert "tex.read(uint(pixelLayer.x), uint(pixelLayer.y), uint(0))" in generated_code
    assert (
        "tex.read(uint((pixelLayer.x + offset)), uint(pixelLayer.y), uint(0))"
        in generated_code
    )
    assert "tex.read(pixelLayer.xy" not in generated_code
    assert "tex.read((pixelLayer.xy" not in generated_code
    assert "calculate_unclamped_lod" not in generated_code
    assert (
        "tex.sample(samp, uvLayer.x, uint(uvLayer.y), level(lod))" not in generated_code
    )


def test_metal_sampler_1d_sampling_options_emit_target_diagnostics():
    shader = """
    shader OneDTextureOptions {
        sampler1D lineMap;
        sampler linearSampler;

        vec4 sampleLine(
            sampler1D tex,
            sampler samp,
            float u,
            vec2 uq,
            int pixel,
            int lod,
            int offset
        ) {
            return texture(tex, samp, u)
                + texture(tex, samp, u, 0.25)
                + textureLod(tex, samp, u, lod)
                + textureGrad(tex, samp, u, 0.1, 0.2)
                + textureProj(tex, samp, uq)
                + textureProj(tex, samp, uq, 0.25)
                + textureProjLod(tex, samp, uq, lod)
                + texelFetch(tex, pixel, 0)
                + texelFetch(tex, pixel, lod)
                + texelFetchOffset(tex, pixel, 0, offset)
                + textureOffset(tex, samp, u, offset);
        }

        fragment {
            vec4 main() @gl_FragColor {
                return sampleLine(lineMap, linearSampler, 0.5, vec2(0.5, 1.0), 4, 1, 1);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "texture1d<float> lineMap [[texture(0)]]" in generated_code
    assert "tex.sample(samp, u)" in generated_code
    assert "tex.sample(samp, uq.x / uq.y)" in generated_code
    assert "tex.read(uint(pixel), uint(0))" in generated_code
    assert "tex.read(uint((pixel + offset)), uint(0))" in generated_code
    assert (
        "/* unsupported Metal texture sampling: texture on texture1d<float> "
        "supports only implicit sampling */ float4(0.0)" in generated_code
    )
    assert (
        "/* unsupported Metal texture sampling: textureLod on texture1d<float> "
        "supports only implicit sampling */ float4(0.0)" in generated_code
    )
    assert (
        "/* unsupported Metal texture sampling: textureGrad on texture1d<float> "
        "supports only implicit sampling */ float4(0.0)" in generated_code
    )
    assert (
        "/* unsupported Metal projected texture: textureProj bias is not "
        "supported for Metal 1D textures */ float4(0.0)" in generated_code
    )
    assert (
        "/* unsupported Metal projected texture: textureProjLod explicit LOD is "
        "not supported for Metal 1D textures */ float4(0.0)" in generated_code
    )
    assert (
        "/* unsupported Metal texel fetch: texelFetch on texture1d<float> "
        "requires a compile-time literal mip level */ float4(0.0)" in generated_code
    )
    assert (
        "/* unsupported Metal texture offset: textureOffset offsets require 2D, "
        "2D-array, or 3D textures */ float4(0.0)" in generated_code
    )
    assert "level(lod)" not in generated_code
    assert "gradient2d" not in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGrad(" not in generated_code
    assert "textureOffset(" not in generated_code


def test_metal_image_1d_and_1d_array_storage_operations():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "texture1d<float, access::read_write> line [[texture(0)]]" in generated_code
    assert (
        "texture1d_array<float, access::read_write> layers [[texture(1)]]"
        in generated_code
    )
    assert (
        "texture1d<uint, access::read_write> counters [[texture(2)]]" in generated_code
    )
    assert (
        "texture1d_array<uint, access::read_write> layerCounters [[texture(3)]]"
        in generated_code
    )
    assert "int sizeLine = int(line.get_width());" in generated_code
    assert (
        "int2 sizeLayer = int2(layers.get_width(), layers.get_array_size());"
        in generated_code
    )
    assert "float oldValue = image.read(uint(x)).x;" in generated_code
    assert "image.write(float4(oldValue + value), uint(x));" in generated_code
    assert (
        "float4 oldValue = image.read(uint(coord.x), uint(coord.y));" in generated_code
    )
    assert (
        "image.write(oldValue + value, uint(coord.x), uint(coord.y));" in generated_code
    )
    assert "image.atomic_fetch_add(uint(x), value).x" in generated_code
    assert (
        "image.atomic_exchange(uint(coord.x), uint(coord.y), value).x" in generated_code
    )
    assert (
        "uint imageAtomicCompSwap_uimage1DArray(texture1d_array<uint, access::read_write> image, int2 coord, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "image.atomic_compare_exchange_weak(uint(coord.x), uint(coord.y), &original, value)"
        in generated_code
    )
    assert (
        "return imageAtomicCompSwap_uimage1DArray(image, coord, expected, value);"
        in generated_code
    )


def test_metal_multisample_storage_images_emit_read_textures_and_diagnostics():
    shader = """
    shader MetalMultisampleStorageImages {
        image2DMS colorImage @rgba16f;
        uimage2DMS counters @r32ui;
        image2DMSArray layered @rgba16f;

        vec4 touch(image2DMS image @rgba16f, uimage2DMS counterImage @r32ui, ivec2 pixel, int sampleIndex, vec4 value, uint count) {
            vec4 oldColor = imageLoad(image, pixel, sampleIndex);
            uint oldCount = imageLoad(counterImage, pixel, sampleIndex);
            imageStore(image, pixel, sampleIndex, oldColor + value);
            imageStore(counterImage, pixel, sampleIndex, oldCount + count);
            uint atomicOld = imageAtomicAdd(counterImage, pixel, sampleIndex, count);
            return oldColor + vec4(float(oldCount + atomicOld));
        }

        vec4 touchLayer(image2DMSArray image @rgba16f, ivec3 pixelLayer, int sampleIndex) {
            return imageLoad(image, pixelLayer, sampleIndex);
        }

        fragment {
            vec4 main() @ gl_FragColor {
                return touch(colorImage, counters, ivec2(0), 1, vec4(1.0), 2u)
                    + touchLayer(layered, ivec3(0, 1, 2), 3);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "texture2d_ms<float, access::read> colorImage [[texture(0)]]" in generated_code
    )
    assert "texture2d_ms<uint, access::read> counters [[texture(1)]]" in generated_code
    assert (
        "texture2d_ms_array<float, access::read> layered [[texture(2)]]"
        in generated_code
    )
    assert (
        "float4 touch(texture2d_ms<float, access::read> image, texture2d_ms<uint, access::read> counterImage"
        in generated_code
    )
    assert (
        "float4 oldColor = image.read(uint2(pixel), uint(sampleIndex));"
        in generated_code
    )
    assert (
        "uint oldCount = counterImage.read(uint2(pixel), uint(sampleIndex)).x;"
        in generated_code
    )
    assert (
        "return image.read(uint2(pixelLayer.xy), uint(pixelLayer.z), uint(sampleIndex));"
        in generated_code
    )
    assert (
        "unsupported Metal multisample image store: imageStore on texture2d_ms<float"
        in generated_code
    )
    assert (
        "unsupported Metal multisample image store: imageStore on texture2d_ms<uint"
        in generated_code
    )
    assert (
        "unsupported Metal multisample image atomic: imageAtomicAdd on texture2d_ms<uint"
        in generated_code
    )
    assert "texture2d_ms<float, access::read_write>" not in generated_code
    assert "texture2d_ms<uint, access::read_write>" not in generated_code
    assert "texture2d_ms_array<float, access::read_write>" not in generated_code
    assert ".write(" not in generated_code


def test_metal_multisample_storage_image_arrays_emit_read_textures_and_diagnostics():
    shader = """
    shader MetalMultisampleStorageImageArrays {
        image2DMSArray layered @rgba16f;
        uimage2DMSArray counters @r32ui;

        vec4 touchLayer(
            image2DMSArray image @rgba16f,
            uimage2DMSArray counterImage @r32ui,
            ivec3 pixelLayer,
            int sampleIndex,
            vec4 value,
            uint count
        ) {
            vec4 oldColor = imageLoad(image, pixelLayer, sampleIndex);
            uint oldCount = imageLoad(counterImage, pixelLayer, sampleIndex);
            imageStore(image, pixelLayer, sampleIndex, oldColor + value);
            imageStore(counterImage, pixelLayer, sampleIndex, oldCount + count);
            uint atomicOld = imageAtomicAdd(counterImage, pixelLayer, sampleIndex, count);
            return oldColor + vec4(float(oldCount + atomicOld));
        }

        fragment {
            vec4 main() @ gl_FragColor {
                return touchLayer(layered, counters, ivec3(0, 1, 2), 3, vec4(1.0), 2u);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "texture2d_ms_array<float, access::read> layered [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture2d_ms_array<uint, access::read> counters [[texture(1)]]"
        in generated_code
    )
    assert (
        "float4 touchLayer(texture2d_ms_array<float, access::read> image, "
        "texture2d_ms_array<uint, access::read> counterImage, int3 pixelLayer, "
        "int sampleIndex, float4 value, uint count)" in generated_code
    )
    assert (
        "float4 oldColor = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z), uint(sampleIndex));"
        in generated_code
    )
    assert (
        "uint oldCount = counterImage.read(uint2(pixelLayer.xy), uint(pixelLayer.z), uint(sampleIndex)).x;"
        in generated_code
    )
    diagnostic_prefix = "un" + "supported Metal multisample image"
    assert (
        f"{diagnostic_prefix} store: imageStore on texture2d_ms_array<float, access::read>"
        in generated_code
    )
    assert (
        f"{diagnostic_prefix} store: imageStore on texture2d_ms_array<uint, access::read>"
        in generated_code
    )
    assert (
        "uint atomicOld = /* "
        f"{diagnostic_prefix} atomic: imageAtomicAdd on "
        "texture2d_ms_array<uint, access::read> */ 0u;" in generated_code
    )
    assert "texture2d_ms_array<float, access::read_write>" not in generated_code
    assert "texture2d_ms_array<uint, access::read_write>" not in generated_code
    assert ".write(" not in generated_code
    assert "atomic_fetch_add" not in generated_code


def test_metal_storage_image_access_attributes_select_texture_access():
    shader = """
    shader StorageImageAccess {
        image2D readOnlyImage @readonly;
        image2D writeOnlyImage @writeonly;
        uimage2D readWriteImage @r32ui @readwrite;

        float readPixel(image2D image @readonly, ivec2 pixel) {
            return imageLoad(image, pixel).x;
        }

        void writePixel(image2D image @writeonly, ivec2 pixel, vec4 value) {
            imageStore(image, pixel, value);
        }

        compute {
            void main() {
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "texture2d<float, access::read> readOnlyImage [[texture(0)]]" in generated_code
    )
    assert (
        "texture2d<float, access::write> writeOnlyImage [[texture(1)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> readWriteImage [[texture(2)]]"
        in generated_code
    )
    assert (
        "float readPixel(texture2d<float, access::read> image, int2 pixel)"
        in generated_code
    )
    assert (
        "void writePixel(texture2d<float, access::write> image, int2 pixel, float4 value)"
        in generated_code
    )


def test_metal_storage_image_access_attributes_keep_default_image_helpers():
    shader = """
    shader StorageImageAccessDefaults {
        float readPixel(image2D image @readonly, ivec2 pixel) {
            return imageLoad(image, pixel);
        }

        float readLine(image1D image @readonly, int x) {
            return imageLoad(image, x);
        }

        float readLayer(image2DArray image @readonly, ivec3 pixelLayer) {
            return imageLoad(image, pixelLayer);
        }

        void writePixel(image2D image @writeonly, ivec2 pixel, float value) {
            imageStore(image, pixel, value);
        }

        compute {
            void main() {
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "float readPixel(texture2d<float, access::read> image, int2 pixel)"
        in generated_code
    )
    assert "return image.read(uint2(pixel)).x;" in generated_code
    assert (
        "float readLine(texture1d<float, access::read> image, int x)" in generated_code
    )
    assert "return image.read(uint(x)).x;" in generated_code
    assert (
        "float readLayer(texture2d_array<float, access::read> image, int3 pixelLayer)"
        in generated_code
    )
    assert (
        "return image.read(uint2(pixelLayer.xy), uint(pixelLayer.z)).x;"
        in generated_code
    )
    assert (
        "void writePixel(texture2d<float, access::write> image, int2 pixel, float value)"
        in generated_code
    )
    assert "image.write(float4(value), uint2(pixel));" in generated_code


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
            "requires read_write storage image access",
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
def test_metal_storage_image_access_rejects_invalid_operations(shader, match):
    ast = crosstl.translator.parse(shader)

    with pytest.raises(ValueError, match=match):
        MetalCodeGen().generate(ast)


def test_metal_storage_image_access_allows_compatible_helper_calls():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "float value = readPixel(source, int2(0, 0));" in generated_code
    assert "writePixel(target, int2(0, 0), float4(value));" in generated_code


def test_metal_storage_image_access_qualifiers_thread_through_helpers():
    shader = """
    shader StorageImageAccessQualifiedHelpers {
        image2D source @rgba32f @readonly;
        image2D target @rgba32f @writeonly;
        uimage2D counters @r32ui @readwrite;

        vec4 readSource(image2D image @rgba32f @readonly, ivec2 pixel) {
            return imageLoad(image, pixel);
        }

        void writeTarget(image2D image @rgba32f @writeonly, ivec2 pixel, vec4 value) {
            imageStore(image, pixel, value);
        }

        uint addCounter(uimage2D image @r32ui @readwrite, ivec2 pixel, uint value) {
            return imageAtomicAdd(image, pixel, value);
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                vec4 color = readSource(source, pixel);
                uint oldValue = addCounter(counters, pixel, 2u);
                writeTarget(target, pixel, color + vec4(float(oldValue)));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate_stage(ast, "compute")

    assert (
        "float4 readSource(texture2d<float, access::read> image, int2 pixel)"
        in generated_code
    )
    assert (
        "void writeTarget(texture2d<float, access::write> image, int2 pixel, float4 value)"
        in generated_code
    )
    assert (
        "uint addCounter(texture2d<uint, access::read_write> image, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "kernel void kernel_main(texture2d<float, access::read> source [[texture(0)]], texture2d<float, access::write> target [[texture(1)]], texture2d<uint, access::read_write> counters [[texture(2)]])"
        in generated_code
    )
    assert "float4 color = readSource(source, pixel);" in generated_code
    assert "uint oldValue = addCounter(counters, pixel, 2u);" in generated_code
    assert (
        "writeTarget(target, pixel, color + float4(float(oldValue)));" in generated_code
    )
    assert "source.write(" not in generated_code
    assert "target.read(" not in generated_code
    assert "counters.atomic_fetch_add(uint2(pixel), 2u).x" not in generated_code


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
            "function call 'addCounter' requires read_write storage image access",
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
def test_metal_storage_image_access_rejects_incompatible_helper_calls(shader, match):
    ast = crosstl.translator.parse(shader)

    with pytest.raises(ValueError, match=match):
        MetalCodeGen().generate(ast)


def test_metal_generic_image_memory_attributes_do_not_become_semantics():
    shader = """
    shader StorageImageMemoryAttrs {
        image2D coherentImage @coherent;
        uimage2D globalImage @r32ui @globallycoherent;

        float readPixel(image2D image @coherent, ivec2 pixel) {
            return imageLoad(image, pixel).x;
        }

        compute {
            void main() {
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "texture2d<float, access::read_write> coherentImage [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> globalImage [[texture(1)]]"
        in generated_code
    )
    assert (
        "float readPixel(texture2d<float, access::read_write> image, int2 pixel)"
        in generated_code
    )
    assert "[[coherent]]" not in generated_code
    assert "[[globallycoherent]]" not in generated_code


if __name__ == "__main__":
    pytest.main()
