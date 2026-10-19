
// Fragment Shader
float4 PSMain(float4 a : A): SV_TARGET {
    return asfloat(asuint(a.yxxx) ^ uint4(0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u));
}
