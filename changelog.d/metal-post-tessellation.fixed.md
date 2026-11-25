Metal output writes a CrossGL tessellation evaluation (domain) stage as a Metal
post-tessellation vertex function: `[[patch(triangle, N)]]` or
`[[patch(quad, N)]]`, with the control points as a `patch_control_point<T>`
member of its `[[stage_in]]` structure, the patch constants as the other
`[[attribute(n)]]` members, and the domain location as `[[position_in_patch]]`.
It used to refuse every domain stage. Metal import writes a `[[patch]]` vertex
function as a CrossGL `tessellation_evaluation` stage with `@domain` and an
`OutputPatch<T, N>` parameter instead of keeping `patch_control_point` and
`@patch` verbatim, so it now also translates to HLSL domain shaders.
