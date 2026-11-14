RWStructuredBuffer<float> o : register(u0);
float assignValue(float x) { return x * 2.0; }
[numthreads(64,1,1)] void main(uint3 id : SV_DispatchThreadID) { o[id.x] = assignValue(o[id.x]); }
