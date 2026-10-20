"""Types of the CrossGL built-in variables, in CrossGL (GLSL) spelling.

CrossGL spells its built-in variables as GLSL does, so their types are the ones GLSL 4.60 declares (section 7.1,
"Built-In Language Variables", and the subgroup variables of GL_KHR_shader_subgroup).  An emitter that infers the
type of ``let x = gl_GlobalInvocationID.x;`` maps the type below to its own spelling instead of guessing one.
"""

CROSSGL_BUILTIN_VARIABLE_TYPES = {
    # Compute (GLSL 4.60 section 7.1.3).
    "gl_GlobalInvocationID": "uvec3",
    "gl_LocalInvocationID": "uvec3",
    "gl_WorkGroupID": "uvec3",
    "gl_NumWorkGroups": "uvec3",
    "gl_WorkGroupSize": "uvec3",
    "gl_LocalInvocationIndex": "uint",
    # Vertex (section 7.1.1); the Vulkan spellings are GL_KHR_vulkan_glsl's.
    "gl_VertexID": "int",
    "gl_InstanceID": "int",
    "gl_VertexIndex": "int",
    "gl_InstanceIndex": "int",
    "gl_DrawID": "int",
    "gl_BaseVertex": "int",
    "gl_BaseInstance": "int",
    "gl_Position": "vec4",
    "gl_PointSize": "float",
    # Geometry and tessellation (sections 7.1.2 and 7.1.4).
    "gl_PrimitiveIDIn": "int",
    "gl_InvocationID": "int",
    "gl_PatchVerticesIn": "int",
    "gl_TessCoord": "vec3",
    # Fragment (section 7.1.5).
    "gl_FragCoord": "vec4",
    "gl_FrontFacing": "bool",
    "gl_PointCoord": "vec2",
    "gl_PrimitiveID": "int",
    "gl_SampleID": "int",
    "gl_SamplePosition": "vec2",
    "gl_Layer": "int",
    "gl_ViewportIndex": "int",
    "gl_HelperInvocation": "bool",
    "gl_FragDepth": "float",
    # Subgroups (GL_KHR_shader_subgroup).
    "gl_NumSubgroups": "uint",
    "gl_SubgroupID": "uint",
    "gl_SubgroupSize": "uint",
    "gl_SubgroupInvocationID": "uint",
}


def crossgl_builtin_variable_type(name):
    """The CrossGL type of built-in variable ``name``, or ``None`` if it is not one."""
    if not isinstance(name, str):
        return None
    return CROSSGL_BUILTIN_VARIABLE_TYPES.get(name)
