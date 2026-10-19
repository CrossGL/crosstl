// OpenCL C 6.4.3.3: convert_char_sat truncates, clamps to [-128, 127] and maps NaN to 0.
__kernel void hazard(__global const float *inp, __global int *outp) {
    size_t i = get_global_id(0);
    outp[i] = convert_char_sat(inp[i]);
}
