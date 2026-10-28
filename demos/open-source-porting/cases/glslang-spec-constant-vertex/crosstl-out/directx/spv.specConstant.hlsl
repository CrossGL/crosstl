
float __crossgl_uint_to_float1(uint value) {
    if (value == 0u) { return 0.0f; }
    uint exponent = uint(firstbithigh(value));
    uint significand;
    if (exponent <= 23u) {
        significand = value << (23u - exponent);
    } else {
        // Round the integer once, including a carry into the next exponent.
        uint shift = exponent - 23u;
        significand = value >> shift;
        uint remainder = value & ((1u << shift) - 1u);
        uint midpoint = 1u << (shift - 1u);
        if (remainder > midpoint || (remainder == midpoint && (significand & 1u) != 0u)) {
            significand += 1u;
        }
        if (significand == 0x1000000u) {
            significand >>= 1u;
            exponent += 1u;
        }
    }
    return asfloat(((exponent + 127u) << 23u) | (significand & 0x7fffffu));
}

static const int arraySize = 5;
static const bool spBool = true;
static const float spFloat = 3.14;
static const double spDouble = 3.141592653589793;
static const uint scale = 2u;
static const int dupArraySize = 12;
static const bool spDupBool = true;
static const float spDupFloat = 3.14;
static const double spDupDouble = 3.141592653589793;
static const uint dupScale = 2u;
static const int gl_MaxImageUnits = 8;

struct VertexInput {
    float4 ucol[5]: TEXCOORD0;
    float4 dupUcol[12]: TEXCOORD5;
};
struct VertexOutput {
    float4 color: TEXCOORD0;
    int size: TEXCOORD1;
};
void foo(float4 p[arraySize], VertexInput input, inout VertexOutput output);

int builtin_spec_constant();

void foo(float4 p[arraySize], VertexInput input, inout VertexOutput output) {
    output.color += input.dupUcol[2];
    output.size += dupArraySize;
    if (spDupBool) {
        output.color *= ((float4)(__crossgl_uint_to_float1(dupScale)));
    }
    output.color += float((spDupDouble / spDupFloat));
}

int builtin_spec_constant() {
    int result = gl_MaxImageUnits;
    return result;
}

// Vertex Shader
VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    output.color = input.ucol[2];
    output.size = arraySize;
    if (spBool) {
        output.color *= ((float4)(__crossgl_uint_to_float1(scale)));
    }
    output.color += float((spDouble / spFloat));
    foo(input.ucol, input, output);
    return output;
}
