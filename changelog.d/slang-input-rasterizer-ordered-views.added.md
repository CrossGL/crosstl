Slang input imports `RasterizerOrdered*` views as the HLSL importer does: the read-write resource with
`@rasterizer_ordered`, whose subscripts become `imageLoad` and `imageStore`. The importer used to keep the subscripts
of a rasterizer-ordered texture, which Metal and the other targets without texture subscripts cannot compile.
