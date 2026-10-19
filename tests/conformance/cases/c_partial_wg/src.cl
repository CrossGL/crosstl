// OpenCL C 2.0: a non-uniform work-group size launches exactly the global size, so no guard is needed.
__kernel void hazard(__global const int *inp, __global int *outp) {
    size_t i = get_global_id(0);
    outp[i] = (int)i * 3 + inp[0];
}
