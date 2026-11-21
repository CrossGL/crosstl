GLSL and WebGL output rename program variables, functions and types that spell a
GLSL built-in function, as they rename GLSL keywords. A resource named `texture`
used to hide the built-in, so `texture(texture, uv)` failed in glslang with
"can't use function syntax on variable", and GLSL ES 3.00 rejected the
declaration itself. Locals, parameters, block members and user functions named
`mix`, `length`, `step` or another built-in are renamed the same way.
