GLSL and WebGL output rename program identifiers that spell a GLSL built-in
function. A resource named `texture` used to hide the built-in, so
`texture(texture, uv)` failed in glslang with "can't use function syntax on
variable", and GLSL ES 3.00 rejected the declaration itself. WebGL output
renames every such identifier. Desktop GLSL output renames user functions
(an overload such as `frexp(float, int)` beside `frexp(float, out int)` is an
error) and the variables, members and types whose built-in the program calls,
and keeps the others, such as a uniform `step` the host sets by name.
