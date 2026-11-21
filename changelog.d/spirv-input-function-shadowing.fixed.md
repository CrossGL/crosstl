SPIR-V input renames a parameter or local that shares its name with a function of the module, such as glslang's
`float twice;` in a shader that also calls `twice()`. C-family scoping let the local hide the function, so the
GLSL, HLSL and Metal translations were rejected.
