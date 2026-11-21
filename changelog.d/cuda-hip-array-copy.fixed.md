CUDA and HIP output copy whole-array assignments and initializers element by element, so nvcc and hipcc accept code that used to leave `b = a` or `int b[3] = a` unchanged and fail (#4757).
