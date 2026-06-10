
// Fragment Shader
float4 PSMain(float4 a : A): SV_TARGET {
    return -a.yxxx;
}
