// fmod(x, y) is the remainder x = i * y + f with i an integer, f the sign of x and |f| < |y| (HLSL intrinsic
// reference), which makes it exact. The floating-point % is defined when both operands have the same sign, so the
// kernel applies it to the magnitudes.
StructuredBuffer<float> xy : register(t0);
RWStructuredBuffer<float> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    float x = xy[2 * i];
    float y = xy[2 * i + 1];
    o[2 * i] = fmod(x, y);
    o[2 * i + 1] = abs(x) % abs(y);
}
