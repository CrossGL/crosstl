Slang and HLSL output convert a product or an element index applied directly to
a ray-tracing transform call (`ObjectToWorld3x4() * p`, `p *
WorldToObject4x3()`, `ObjectToWorld3x4()[2][1]`) the same way as on a local of
the call's type, as `mul(...)` and a swapped index. The Slang generator did not
know the return types of `ObjectToWorld`, `ObjectToWorld3x4`,
`ObjectToWorld4x3`, `WorldToObject`, `WorldToObject3x4` and `WorldToObject4x3`,
and the HLSL generator did not know them for the unsuffixed `ObjectToWorld` and
`WorldToObject`, so slangc and DXC rejected the product and Slang read the
transposed element (#4558).
