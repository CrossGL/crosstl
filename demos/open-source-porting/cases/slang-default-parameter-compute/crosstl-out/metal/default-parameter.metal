
#include <metal_stdlib>
using namespace metal;

int helper(int val, int a) {
    return val + a;
}

int test(int val) {
    return helper(val, 16) + helper(val, 256);
}

// Compute Shader
kernel void computeMain(uint3 dispatchThreadID [[thread_position_in_grid]], device int* outputBuffer [[buffer(0)]]) {
    int inVal = int(dispatchThreadID.x);
    int outVal = test(inVal);
    outputBuffer[dispatchThreadID.x] = outVal;
}
