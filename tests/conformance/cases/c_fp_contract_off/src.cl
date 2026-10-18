// OpenCL C 6.15.2.1: with FP_CONTRACT OFF the implementation may not contract a*b - c.
#pragma OPENCL FP_CONTRACT OFF
__kernel void hazard(__global const float *inp, __global float *outp) {
    size_t i = get_global_id(0);
    float a = inp[3 * i];
    float b = inp[3 * i + 1];
    float c = inp[3 * i + 2];
    outp[i] = a * b - c;
}
