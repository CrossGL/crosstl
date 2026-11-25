GLSL and WebGL output refuse a stage entry whose return value no GLSL stage output
receives, such as a closest-hit or compute entry's value. They used to
write the value as an expression statement before `return;`, so it was discarded.
A vertex entry that returns a call to a helper declared in its stage block now
declares the output structure the helper returns, which glslang used to reject.
