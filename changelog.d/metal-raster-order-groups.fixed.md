Metal output keeps the ordering of rasterizer-ordered resources (CrossGL
`@rasterizer_ordered`, HLSL `RasterizerOrderedBuffer` and
`RasterizerOrderedTexture2D`). A fragment function now declares each such buffer
or texture argument with `[[raster_order_group(0)]]`; the output used to drop
the ordering, so overlapping fragments raced on the resource. Metal import reads
`[[raster_order_group(N)]]` on a buffer or texture argument as
`@rasterizer_ordered` instead of a name CrossGL does not define, which Metal
output refused and HLSL output declared as an unordered view.
