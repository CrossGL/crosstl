GLSL output writes a floating literal where a `double` is required as a `double`
constant: `x * 0.1` with a `double` `x` gives `x * 0.1lf`, and `0.1f` gives the
`double` constant of its binary32 value. It used to write `double(0.1)`, which GLSL
reads as the `float` 0.1 widened (GLSL 4.60 section 4.1.4), so a compiler that follows
the specification multiplied by 0.100000001490116 and `double(1e-300)` was 0.
