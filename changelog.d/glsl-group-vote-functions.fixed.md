OpenGL output writes a stage whose only subgroup operations are boolean votes with `allInvocationsARB`,
`anyInvocationARB` and `allInvocationsEqualARB` under `GL_ARB_shader_group_vote`, or the core names from GLSL 4.60.
It used to require `GL_KHR_shader_subgroup_basic` and `GL_KHR_shader_subgroup_vote`, which OpenGL drivers such as
Mesa's llvmpipe reject (#3009).
