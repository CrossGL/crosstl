RWStructuredBuffer<float> o : register(u0);
[numthreads(64,1,1)] void main(uint3 id : SV_DispatchThreadID) { float a = o[id.x]; precise float r = a * a - a; o[id.x] = r; }
