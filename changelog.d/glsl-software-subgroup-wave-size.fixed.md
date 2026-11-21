OpenGL GLSL output with `software_subgroup_width=32` accepts a compute entry
marked `WaveSize(32)`. The emulated subgroups have exactly 32 lanes, so the
output is the same as without the attribute: no `CROSSTL_REQUIRED_SUBGROUP_WIDTH`
define and no `gl_SubgroupSize` guard. Any other `WaveSize` still fails with
`hardware-contract-conflict`, as the DirectX generator does (#4707).
