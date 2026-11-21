SPIR-V output lowers `interpolateAtCentroid`, `interpolateAtSample` and `interpolateAtOffset` to the GLSL.std.450
interpolation instructions, reading the interpolant through the fragment input's own `Input` variable, including a
member of an entry point's struct parameter, an element of an input array or a component of an input vector. An
interpolant with no `Input` variable, such as a local value or a parameter the entry point writes, is refused.
