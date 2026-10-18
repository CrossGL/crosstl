// C99 6.3.1.8: the int operand converts to uint.
__kernel void hazard(__global const int *inp, __global int *outp) {
    size_t i = get_global_id(0);
    int x = inp[i];
    uint four = 4u;
    outp[i] = (x < four) ? 1 : 0;
}
