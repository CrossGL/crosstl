GLSL output declares a prototype for a generic function that an
explicit-type-argument call such as `twice<float>(x)` reaches before the
generic's definition. glslang used to reject the call with "no matching
overloaded function found" (#4816).
