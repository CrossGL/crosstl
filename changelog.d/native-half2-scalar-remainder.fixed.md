CUDA and HIP output lowers `half2 % half` and `mod(half2, half)` through component-wise float remainder helpers instead of emitting `(v % s)` or `fmodf(v, s)`, which nvcc and hipcc reject (#5193).
