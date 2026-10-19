// OpenCL C 6.4.3.3: convert_uint_sat_rte rounds to nearest even, clamps to the uint range and maps NaN to 0.
__kernel void hazard(__global const float *inp, __global uint *outp) {
    size_t i = get_global_id(0);
    outp[i] = convert_uint_sat_rte(inp[i]);
}
