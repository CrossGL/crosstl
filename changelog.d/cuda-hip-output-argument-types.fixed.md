CUDA and HIP output refuse a call that passes an `out` or `inout` argument of another type, such as a `float`
variable passed to `out int`, with a typed error that names the argument. Both emitters declare such a parameter as
a C++ reference, which binds only an object of its own type, so the call did not compile.
