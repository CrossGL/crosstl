HLSL input keeps `[[vk::location(N)]]` on stage input and output members and
parameters as `@location(N)` next to the semantic; it used to drop the attribute,
so a translated Vulkan stage could link its varyings at other locations. The
attribute is dropped on `SV_*` built-ins, which have no location, and refused on
`SV_Target<i>` at a location other than `i`. HLSL output keeps the semantic of a
member or parameter that has both and writes `[[vk::location(N)]]` inside
`#ifdef __spirv__`, so DXIL compiles without the ignored-attribute warning.
