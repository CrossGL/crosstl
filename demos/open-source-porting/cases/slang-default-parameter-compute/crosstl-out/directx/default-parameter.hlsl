
RWStructuredBuffer<int> outputBuffer : register(u0);
int helper(int val, int a) {
    return (val + a);
}

int test(int val) {
    return (helper(val, 16) + helper(val, 256));
}

// Compute Shader
[numthreads(4, 1, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    int inVal = int(dispatchThreadID.x);
    int outVal = test(inVal);
    outputBuffer[dispatchThreadID.x] = outVal;
}
