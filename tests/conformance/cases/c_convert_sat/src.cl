// OpenCL C 6.4.3.3: convert_int_sat clamps to the int range and maps NaN to 0.
__kernel void hazard(__global const float *inp, __global int *outp) {
    size_t i = get_global_id(0);
    outp[i] = convert_int_sat(inp[i]);
}
