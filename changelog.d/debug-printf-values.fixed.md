A debug print whose value the program reads (`int r = printf(...)`) is refused with a typed
`<target>-debug-printf-value-unsupported` diagnostic. GLSL's `debugPrintfEXT`, SPIR-V's `NonSemantic.DebugPrintf`,
HLSL's, Slang's and Metal's prints return nothing, and C's `printf` result means something else in CUDA, HIP and
OpenCL C, so such a print was written into output that glslang, DXC or spirv-val rejected, or into code that read
another value. A CUDA, HIP or OpenCL C source translated back to its own language keeps it.
