// OpenCL C 6.15.2: if one argument is a NaN, fmin returns the other argument.
__kernel void hazard(__global const float *inp, __global float *outp) {
    size_t i = get_global_id(0);
    outp[i] = fmin(inp[2 * i], inp[2 * i + 1]);
}
