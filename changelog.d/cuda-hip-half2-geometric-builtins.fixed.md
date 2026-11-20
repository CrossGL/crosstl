CUDA and HIP output computes `length`, `normalize`, `distance` and `dot` on a
native `half2` (`f16vec2`, `vec2<f16>`) in `float` through generated helpers that
read the lanes with `__low2float` and `__high2float`. CrossTL used to refuse these
calls as undeclared functions (#3127).
