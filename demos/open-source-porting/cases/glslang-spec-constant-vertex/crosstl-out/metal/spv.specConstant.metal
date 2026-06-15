
#include <metal_stdlib>
using namespace metal;

/* CrossGL fallback: Metal source output cannot preserve GLSL specialization constant id 16 for 'arraySize'; using the default literal. */
__attribute__((unused)) constant int arraySize = 5;
/* CrossGL fallback: Metal source output cannot preserve GLSL specialization constant id 17 for 'spBool'; using the default literal. */
__attribute__((unused)) constant bool spBool = true;
/* CrossGL fallback: Metal source output cannot preserve GLSL specialization constant id 18 for 'spFloat'; using the default literal. */
__attribute__((unused)) constant float spFloat = 3.14;
/* CrossGL fallback: Metal source output cannot preserve GLSL specialization constant id 19 for 'spDouble'; using the default literal. */
/* CrossGL fallback: Metal does not support double specialization constant 'spDouble'; lowered to float. */
__attribute__((unused)) constant float spDouble = 3.141592653589793;
/* CrossGL fallback: Metal source output cannot preserve GLSL specialization constant id 22 for 'scale'; using the default literal. */
__attribute__((unused)) constant uint scale = 2;
/* CrossGL fallback: Metal source output cannot preserve GLSL specialization constant id 116 for 'dupArraySize'; using the default literal. */
__attribute__((unused)) constant int dupArraySize = 12;
/* CrossGL fallback: Metal source output cannot preserve GLSL specialization constant id 117 for 'spDupBool'; using the default literal. */
__attribute__((unused)) constant bool spDupBool = true;
/* CrossGL fallback: Metal source output cannot preserve GLSL specialization constant id 118 for 'spDupFloat'; using the default literal. */
__attribute__((unused)) constant float spDupFloat = 3.14;
/* CrossGL fallback: Metal source output cannot preserve GLSL specialization constant id 119 for 'spDupDouble'; using the default literal. */
/* CrossGL fallback: Metal does not support double specialization constant 'spDupDouble'; lowered to float. */
__attribute__((unused)) constant float spDupDouble = 3.141592653589793;
/* CrossGL fallback: Metal source output cannot preserve GLSL specialization constant id 122 for 'dupScale'; using the default literal. */
__attribute__((unused)) constant uint dupScale = 2;

/* CrossGL fallback: GLSL builtin limit specialization gl_MaxImageUnits is not available in Metal; using the OpenGL minimum value. */
__attribute__((unused)) constant int gl_MaxImageUnits = 8;
struct VertexInput {
    float4 ucol_0 [[attribute(0)]];
    float4 ucol_1 [[attribute(1)]];
    float4 ucol_2 [[attribute(2)]];
    float4 ucol_3 [[attribute(3)]];
    float4 ucol_4 [[attribute(4)]];
    float4 dupUcol_0 [[attribute(5)]];
    float4 dupUcol_1 [[attribute(6)]];
    float4 dupUcol_2 [[attribute(7)]];
    float4 dupUcol_3 [[attribute(8)]];
    float4 dupUcol_4 [[attribute(9)]];
    float4 dupUcol_5 [[attribute(10)]];
    float4 dupUcol_6 [[attribute(11)]];
    float4 dupUcol_7 [[attribute(12)]];
    float4 dupUcol_8 [[attribute(13)]];
    float4 dupUcol_9 [[attribute(14)]];
    float4 dupUcol_10 [[attribute(15)]];
    float4 dupUcol_11 [[attribute(16)]];
};
struct VertexOutput {
    float4 color;
    int size;
    /* CrossGL fallback: Metal vertex entry points require a position output even when the GLSL source did not write gl_Position. */
    float4 __crossgl_position [[position]];
};
static inline float4 __attribute__((unused)) __crossgl_stage_io_get_VertexInput_dupUcol(VertexInput value, int index) {
    switch (index) {
    case 0: return value.dupUcol_0;
    case 1: return value.dupUcol_1;
    case 2: return value.dupUcol_2;
    case 3: return value.dupUcol_3;
    case 4: return value.dupUcol_4;
    case 5: return value.dupUcol_5;
    case 6: return value.dupUcol_6;
    case 7: return value.dupUcol_7;
    case 8: return value.dupUcol_8;
    case 9: return value.dupUcol_9;
    case 10: return value.dupUcol_10;
    case 11: return value.dupUcol_11;
    default: return float4(0);
    }
}

static inline float4 __attribute__((unused)) __crossgl_stage_io_get_VertexInput_ucol(VertexInput value, int index) {
    switch (index) {
    case 0: return value.ucol_0;
    case 1: return value.ucol_1;
    case 2: return value.ucol_2;
    case 3: return value.ucol_3;
    case 4: return value.ucol_4;
    default: return float4(0);
    }
}

void foo(thread VertexOutput& output, VertexInput input) {
    output.color += input.dupUcol_2;
    output.size += dupArraySize;
    if (spDupBool) {
        output.color *= dupScale;
    }
    output.color += float(spDupDouble / spDupFloat);
}

int builtin_spec_constant() {
    int result = gl_MaxImageUnits;
    return result;
}

// Vertex Shader
vertex VertexOutput vertex_main(VertexInput input [[stage_in]]) {
    VertexOutput output;
    output.color = input.ucol_2;
    output.size = arraySize;
    if (spBool) {
        output.color *= scale;
    }
    output.color += float(spDouble / spFloat);
    foo(output, input);
    VertexOutput _crossglReturn = output;
    _crossglReturn.__crossgl_position = float4(0.0, 0.0, 0.0, 1.0);
    return _crossglReturn;
}
