// OpenCL C 6.5.10: the shift count is masked to its log2(N) low bits.
__kernel void hazard(__global const uint *inp, __global uint *outp) {
    size_t i = get_global_id(0);
    uint x = inp[2 * i];
    uint s = inp[2 * i + 1];
    outp[i] = (x << s) ^ (x >> s);
}
