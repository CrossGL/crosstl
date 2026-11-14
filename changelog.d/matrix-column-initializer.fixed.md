HLSL and GLSL output accept a matrix aggregate initializer with one column vector per
column, as in ``mat2 m = {vec2(x, 1.0), vec2(2.0, x)}``, matching GLSL 4.60 §4.1.11.
The column type is used as the expected type for each element, so type conversions
(e.g. ``ivec2`` to ``vec2``) work correctly. A list that mixes vector and scalar elements
where the element count does not match the number of columns is refused with an
``element-count-mismatch`` error on both HLSL and GLSL targets. Fixes #3718.
