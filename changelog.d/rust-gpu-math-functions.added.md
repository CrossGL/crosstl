Rust output writes CrossGL atomics on storage buffers and workgroup memory as spirv_std atomics (`atomic_i_add`,
`atomic_f_add`, `atomic_s_min`, `atomic_compare_exchange`, `atomic_load`, `atomic_store`, ...) at the Device or
Workgroup scope, and lowers `cbrt`, `expm1`, `log1p`, `log10`, `rcp`, `copysign`, `signbit`, `pow` of a vector and a
scalar, `determinant`, `inverse`, `refract`, `select`, `length_squared`, `distance_squared`, `absdiff`, `sincos`,
GLSL's packing functions, `nan()`, `infinity()`, `epsilon()`, the subgroup barriers and `IsHelperLane()` to
num-traits, glam and spirv_std calls. `cbrt` keeps C's sign for negative operands, which rust-gpu's `Float::cbrt`
does not. Atomics on memory no other invocation shares, writes to read-only buffers and atomics SPIR-V lacks for the
location's type are refused by name.
