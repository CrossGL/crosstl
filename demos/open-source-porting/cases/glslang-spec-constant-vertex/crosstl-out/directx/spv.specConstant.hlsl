
static const int arraySize = 5;
static const bool spBool = true;
static const float spFloat = 3.14;
static const double spDouble = 3.141592653589793;
static const uint scale = 2;
static const int dupArraySize = 12;
static const bool spDupBool = true;
static const float spDupFloat = 3.14;
static const double spDupDouble = 3.141592653589793;
static const uint dupScale = 2;

static const int gl_MaxImageUnits = 8;

struct VertexInput {
    float4 ucol[arraySize]: TEXCOORD0;
    float4 dupUcol[dupArraySize]: TEXCOORD5;
};
struct VertexOutput {
    float4 color: TEXCOORD0;
    int size: TEXCOORD1;
};
void foo(float4 p[arraySize], VertexInput input, inout VertexOutput output) {
    output.color += input.dupUcol[2];
    output.size += dupArraySize;
    if (spDupBool) {
        output.color *= dupScale;
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
        output.color *= scale;
    }
    output.color += float((spDouble / spFloat));
    foo(input.ucol, input, output);
    return output;
}
