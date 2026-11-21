Rust input writes CrossGL that every generator accepts for `T::default()` (an explicit zero constructor), `!` on
integers (`~`), vector `==` and `!=` (`all()` and `any()`), glam's `is_nan`/`is_finite` masks, field-less enums,
nested array types, `.len()` of a storage slice and typed `u32` and `f32` literals, and refuses rust-gpu execution
modes it cannot express instead of dropping them. Rust output types struct constructors and enum variants and reads a
storage buffer's length with `len()`.
