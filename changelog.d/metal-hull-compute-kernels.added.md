Metal output writes HLSL-form tessellation control (hull) stages, from HLSL, Slang
or CrossGL, as Metal compute kernels. The kernel keeps the hull's name and runs
one thread per patch: it reads the input control points from a buffer, calls the
control-point function for every output control point and writes the results to
the buffer the post-tessellation vertex function reads as its control points,
then calls the patch-constant function and writes its `SV_TessFactor` and
`SV_InsideTessFactor` members as `MTLTriangleTessellationFactorsHalf` or
`MTLQuadTessellationFactorsHalf` records and its other members to a per-patch
buffer. Partitioning, winding and `maxtessfactor` are Metal pipeline state;
isoline domains and point or line output topologies are refused, and GLSL-form
tessellation control stages stay refused.
