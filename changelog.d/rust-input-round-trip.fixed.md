Rust input writes CrossGL that every generator accepts for `T::default()` (an explicit zero constructor), `!` on
integers (`~`), vector `==` and `!=` (`all()` and `any()`), glam's `is_nan`/`is_finite` masks, field-less enums,
nested array types, `.len()` of a storage slice and typed `u32` and `f32` literals, and refuses rust-gpu execution
modes it cannot express instead of dropping them. Rust output types struct constructors and enum variants and reads a
storage buffer's length with `len()`.

Rust input also keeps Rust's results where CrossGL's namesake differs: `fract` truncates (`x - trunc(x)`) and
`fract_gl` floors, `rem_euclid` and `div_euclid` are Euclidean, glam's `x.smoothstep(e0, e1)` interpolates its
receiver, `i8`/`u8`/`i16`/`u16` arithmetic wraps in its type, and `let p = &mut place;` names the place instead of
copying it, so writes through `p` are kept. Rust output tries a switch's `default` arm after its cases.
