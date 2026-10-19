// C99 6.3.1.3: conversion to uchar is modulo 256.
__kernel void hazard(__global const int *inp, __global int *outp) {
    size_t i = get_global_id(0);
    uchar v = (uchar)inp[i];
    v = v + 1;
    outp[i] = (int)v;
}
