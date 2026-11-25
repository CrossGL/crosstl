Metal and GLSL output no longer pass the leading type arguments of a generic
call such as `bump(vec2, w, w)` to the specialization: the call is written
`bump_vec2(w, w)`, as for `bump<vec2>(w, w)`. Both used to write
`bump_vec2(vec2, w, w)`, which their compilers rejected.
