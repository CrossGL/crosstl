HLSL output writes CrossGL's debug print as HLSL's `printf` intrinsic, which DXC
compiles to SPIR-V as `NonSemantic.DebugPrintf`; it used to be refused. DXIL has
no print operation, so DXC still rejects the call when it compiles to DXIL.
