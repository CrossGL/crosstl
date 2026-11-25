HLSL input reads `half` as DXC does: as `float`, a binary32 value, unless the
program is compiled with native 16-bit types (#4907). DXC maps `half` to `float`
without `-enable-16bit-types`; the importer used to give every `half` binary16,
so `half v = half(x); v * v` lost its low 13 bits after translation. A program
that names `float16_t`, `int16_t` or `uint16_t` compiles only with the option and
keeps binary16 `half`, and the source option `enable_16bit_types` states the mode
explicitly (`crosstl.translate(..., source_options={"enable_16bit_types": True})`),
which also defines DXC's `__HLSL_ENABLE_16_BIT` macro.
