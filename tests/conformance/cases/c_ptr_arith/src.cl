// Pointer arithmetic on a global pointer.
__kernel void hazard(__global const int *inp, __global int *outp) {
    size_t i = get_global_id(0);
    __global const int *p = inp + 3;
    outp[i] = p[i];
}
