Metal output declares a CrossGL texture array of unknown length that a run-time
index reaches (`uniform sampler2D tex[];`, HLSL `Texture2D tex[]`) as an
argument buffer: `const device tex_slot* tex [[buffer(n)]]`, where each
`tex_slot` holds the element's texture and, for a combined sampler, its
sampler. The output used to size the array from the values its index could
take, which made the host's runtime descriptor array a fixed one, and refused
an index with no bound. An array that only constant indices reach keeps its
implicit size.
