WebGL 2 output now reports an unsupported lowering diagnostic naming the wave
or quad intrinsic, barrier, or built-in variable when a shader needs subgroup
operations. The GLSL importer renames user functions whose names CrossGL
reserves for these intrinsics, for example `subgroupAdd` to `subgroupAdd_` (#4843).
