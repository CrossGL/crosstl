
struct VertexInput {
    float4 position: POSITION;
};
struct VertexOutput {
    float4 gl_Position: SV_POSITION;
};
// Constant Buffers
cbuffer Uniforms : register(b0) {
    float4x4 modelViewProjectionMatrix;
    float timeValX;
    float timeValY;
    float2 mouse;
};
float4 rand(float2 A, float2 B, float2 C, float2 D);

float noise(float2 coord, float d);

float4 rand(float2 A, float2 B, float2 C, float2 D) {
    float2 s = float2(12.9898, 78.233);
    float4 tmp = float4(dot(A, s), dot(B, s), dot(C, s), dot(D, s));
    return ((frac((sin(tmp) * 43758.5453)) * 2.0) - 1.0);
}

float noise(float2 coord, float d) {
    float2 C[4];
    float d1 = (1.0 / d);
    C[0] = (floor((coord * d)) * d1);
    C[1] = (C[0] + float2(d1, 0.0));
    C[2] = (C[0] + float2(d1, d1));
    C[3] = (C[0] + float2(0.0, d1));
    float2 p = frac((coord * d));
    float2 q = (1.0 - p);
    float4 w = float4((q.x * q.y), (p.x * q.y), (p.x * p.y), (q.x * p.y));
    return dot(float4(rand(C[0], C[1], C[2], C[3])), w);
}

// Vertex Shader
VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    float4 pos = mul(modelViewProjectionMatrix, input.position);
    float noiseAmntX = noise(float2((-timeValX + (pos.x / 1000.0)), 100.0), 20.0);
    float noiseAmntY = noise(float2((timeValY + (pos.y / 1000.0)), (pos.x / 2000.0)), 20.0);
    float noiseB = noise(float2((timeValY * 0.25), (pos.y / 2000.0)), 20.0);
    float2 d = (float2(pos.x, pos.y) - mouse);
    float len = sqrt(((d.x * d.x) + (d.y * d.y)));
    if ((len < 300.0) && (len > 0.0)) {
        float pct = (len / 300.0);
        pct *= pct;
        pct = (1.0 - pct);
        d /= len;
        pos.x += ((d.x * pct) * 90.0);
        pos.y += ((d.y * pct) * 90.0);
    }
    pos.x += (noiseAmntX * 20.0);
    pos.y += (noiseAmntY * 10.0);
    output.gl_Position = pos;
    return output;
}
