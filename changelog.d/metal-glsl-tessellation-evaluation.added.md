Metal output writes GLSL and SPIR-V tessellation evaluation shaders as Metal
post-tessellation vertex functions. Per-vertex input arrays and `gl_in[].gl_Position`
become `patch_control_point` attributes, `patch in` variables the patch's other
`[[attribute(n)]]` inputs, `gl_TessCoord` the `[[position_in_patch]]` input
(`vec3(coord, 0.0)` in a quad patch) and `gl_PrimitiveID` `[[patch_id]]`. Without
a tessellation control stage, which fixes the control-point count, the
`[[patch]]` attribute takes the draw's count. Isolines, `point_mode` and reads of
the tessellation levels are refused; Metal used to refuse every such stage.
