// OpenCL C 6.15.2 round: halfway cases round away from zero, regardless of the current rounding direction.
__kernel void hazard(__global const float *inp, __global float *outp) {
    size_t i = get_global_id(0);
    outp[i] = round(inp[i]);
}
