// OpenCL C 7.4: sin is accurate to 4 ulp over the full range (full profile, no fast-relaxed-math).
__kernel void hazard(__global const float *inp, __global float *outp) {
    size_t i = get_global_id(0);
    outp[i] = sin(inp[i]);
}
