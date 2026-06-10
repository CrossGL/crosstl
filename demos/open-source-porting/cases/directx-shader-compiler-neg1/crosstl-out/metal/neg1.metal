
#include <metal_stdlib>
using namespace metal;

// Fragment Shader
fragment float4 fragment_main(float4 a [[A]]) {
    return -a.yxxx;
}
