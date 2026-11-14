CUDA and HIP output computes `isnan` and `isinf` of a vector component by
component, giving the `uchar2`, `uchar3` or `uchar4` used for `bvecN`. It used
to pass the vector to `isnan` or `isinf`, which nvcc and hipcc accept only for
scalars, and `any(isinf(v))` was refused (#4586).
