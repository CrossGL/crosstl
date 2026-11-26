GLSL ES output writes an integer literal used as a float in the float type, as in `x + 1.0`. It used to write `x + 1`,
which glslang rejected because GLSL ES converts nothing implicitly (#5058).
