CrossTL reads WGSL. `crosstl.translate("kernel.wgsl", backend=...)` and
`source_backend="wgsl"` import a WGSL module through a new front end in
`crosstl/backend/WGSL` (lexer with WGSL template-list discovery, parser for the
WGSL grammar, and an importer that types every expression as WGSL does). The
importer writes WGSL semantics CrossGL leaves open: zero-initialized `var`
declarations and workgroup variables, `loop`/`continuing`/`break if`, clauses
that never fall through, masked run-time shift amounts, integer division and
remainder by a run-time divisor with WGSL's results for zero and for
`i32` minimum by -1, `round` as round-half-even and `clamp` as
`min(max(e, low), high)`. Pointers become `inout` parameters and names for the
memory they point to; overrides become specialization constants. WESL modules
(`.wesl`) are still refused.
