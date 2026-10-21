"""Built-in value attributes in GLSL, HLSL and WGSL spellings translate to the same target built-ins.

CrossGL accepts ``@gl_VertexID``, ``@SV_VertexID`` and ``@builtin(vertex_index)`` for one built-in. Each program below
is written in all three spellings; every spelling must give the same output for each target, and that output must
compile with the target's native compiler when it is installed.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator import parse
from crosstl.translator.semantics.builtin_values import (
    BuiltinValueUnsupportedError,
    canonicalize_builtin_values,
)

SPELLINGS = {
    "glsl": {
        "position": "@gl_Position",
        "vertex_index": "@gl_VertexID",
        "instance_index": "@gl_InstanceID",
        "front_facing": "@gl_FrontFacing",
        "frag_depth": "@gl_FragDepth",
        "sample_index": "@gl_SampleID",
        "global_invocation_id": "@gl_GlobalInvocationID",
        "local_invocation_id": "@gl_LocalInvocationID",
        "local_invocation_index": "@gl_LocalInvocationIndex",
        "workgroup_id": "@gl_WorkGroupID",
    },
    "hlsl": {
        "position": "@SV_Position",
        "vertex_index": "@SV_VertexID",
        "instance_index": "@SV_InstanceID",
        "front_facing": "@SV_IsFrontFace",
        "frag_depth": "@SV_Depth",
        "sample_index": "@SV_SampleIndex",
        "global_invocation_id": "@SV_DispatchThreadID",
        "local_invocation_id": "@SV_GroupThreadID",
        "local_invocation_index": "@SV_GroupIndex",
        "workgroup_id": "@SV_GroupID",
    },
}
SPELLINGS["wgsl"] = {name: f"@builtin({name})" for name in SPELLINGS["glsl"]}

GRAPHICS = """shader Builtins {{
struct VSOut {{ vec4 pos {position}; vec2 uv @location(0); }};
struct FSOut {{ vec4 color @gl_FragColor; float depth {frag_depth}; }};
vertex {{
VSOut main(uint vid {vertex_index}, uint iid {instance_index}) {{
    VSOut o;
    o.pos = vec4(float(vid), float(iid), 0.0, 1.0);
    o.uv = vec2(0.5);
    return o;
}}
}}
fragment {{
FSOut main(VSOut i, bool ff {front_facing}, uint sid {sample_index}) {{
    FSOut o;
    o.color = vec4(i.pos.xy, ff ? 1.0 : 0.0, float(sid));
    o.depth = i.uv.x;
    return o;
}}
}}
}}
"""

COMPUTE = """@compute
@workgroup_size(8, 1, 1)
fn k(@group(0) @binding(0) var<storage, read_write> o: array<u32>, gid: vec3<u32> {global_invocation_id},
     lid: vec3<u32> {local_invocation_id}, lidx: u32 {local_invocation_index}, wid: vec3<u32> {workgroup_id}) {{
    o[gid.x] = lid.x + lidx + wid.x;
}}
"""

# What each target must declare for the graphics program, whatever the spelling.
GRAPHICS_EXPECTED = {
    "directx": [
        ": SV_VertexID",
        ": SV_InstanceID",
        ": SV_IsFrontFace",
        ": SV_SampleIndex",
    ],
    "metal": [
        "[[vertex_id]]",
        "[[instance_id]]",
        "[[position]]",
        "[[front_facing]]",
        "[[sample_id]]",
        "[[depth(any)]]",
    ],
    "opengl": [
        "gl_Position = ",
        "gl_VertexID",
        "gl_InstanceID",
        "gl_FrontFacing",
        "gl_SampleID",
        "gl_FragCoord.xy",
        "gl_FragDepth = ",
    ],
    "vulkan": [
        "BuiltIn VertexIndex",
        "BuiltIn InstanceIndex",
        "BuiltIn Position",
        "BuiltIn FrontFacing",
        "BuiltIn SampleId",
        "BuiltIn FragCoord",
        "BuiltIn FragDepth",
    ],
}
COMPUTE_EXPECTED = {
    "directx": [
        ": SV_DispatchThreadID",
        ": SV_GroupThreadID",
        ": SV_GroupIndex",
        ": SV_GroupID",
        "[numthreads(",
    ],
    "metal": [
        "[[thread_position_in_grid]]",
        "[[thread_position_in_threadgroup]]",
        "[[thread_index_in_threadgroup]]",
        "[[threadgroup_position_in_grid]]",
    ],
    "opengl": [
        "gl_GlobalInvocationID",
        "gl_LocalInvocationID",
        "gl_LocalInvocationIndex",
        "gl_WorkGroupID",
    ],
    "vulkan": [
        "BuiltIn GlobalInvocationId",
        "BuiltIn LocalInvocationId",
        "BuiltIn LocalInvocationIndex",
        "BuiltIn WorkgroupId",
    ],
}
TARGETS = tuple(GRAPHICS_EXPECTED)


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


def _metal_compiler():
    if shutil.which("xcrun") is None:
        return None
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    return "xcrun" if found.returncode == 0 else None


_HLSL_ENTRIES = {
    "graphics": [("vs_6_0", "VSMain"), ("ps_6_0", "PSMain")],
    "vertex": [("vs_6_0", "VSMain")],
    "fragment": [("ps_6_0", "PSMain")],
    "compute": [("cs_6_0", "CSMain")],
}
_GLSL_STAGES = {
    "graphics": ("vert", "frag"),
    "vertex": ("vert",),
    "fragment": ("frag",),
    "compute": ("comp",),
}


def _validate(tmp_path, backend, code, program):
    """Compile ``code`` with the target's native compiler, when it is installed."""
    if backend == "directx":
        compiler = shutil.which("dxc")
        if compiler is None:
            return
        source = tmp_path / "out.hlsl"
        source.write_text(code)
        for profile, entry in _HLSL_ENTRIES[program]:
            _run([compiler, "-T", profile, "-E", entry, str(source)])
    elif backend == "metal":
        if _metal_compiler() is None:
            return
        source = tmp_path / "out.metal"
        source.write_text(code)
        _run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
            + ["-o", str(tmp_path / "out.air")]
        )
    elif backend == "opengl":
        if shutil.which("glslangValidator") is None:
            return
        # The generator writes both graphics stages behind the GL_*_SHADER macros glslang predefines per stage.
        for stage in _GLSL_STAGES[program]:
            _run(["glslangValidator", "--stdin", "-S", stage], input=code)
    elif backend == "vulkan":
        if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
            return
        asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        asm.write_text(code)
        _run(["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
        _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


@pytest.mark.parametrize("backend", TARGETS)
@pytest.mark.parametrize(
    "program, template, expected",
    [
        ("graphics", GRAPHICS, GRAPHICS_EXPECTED),
        ("compute", COMPUTE, COMPUTE_EXPECTED),
    ],
    ids=["graphics", "compute"],
)
def test_every_spelling_translates_to_the_target_builtins(
    tmp_path, backend, program, template, expected
):
    outputs = {
        spelling: _translate(tmp_path, template.format(**names), backend)
        for spelling, names in SPELLINGS.items()
    }
    for spelling, code in outputs.items():
        for needle in expected[backend]:
            assert needle in code, f"{spelling} spelling: {needle!r} missing"
    # The WGSL and GLSL spellings name the same built-ins, so they give the same program.
    assert outputs["wgsl"] == outputs["glsl"]
    if backend == "directx":
        # HLSL keeps a system-value semantic as written; semantics are case-insensitive.
        assert outputs["hlsl"].lower() == outputs["glsl"].lower()
    else:
        assert outputs["hlsl"] == outputs["glsl"]
    _validate(tmp_path, backend, outputs["wgsl"], program)


def test_hlsl_never_emits_builtin_attribute_names(tmp_path):
    hlsl = _translate(tmp_path, GRAPHICS.format(**SPELLINGS["wgsl"]), "directx")
    assert ": builtin" not in hlsl.lower()
    assert ": SV_POSITION" in hlsl and ": SV_DEPTH" in hlsl


def test_sv_position_makes_the_glsl_vertex_stage_write_gl_position(tmp_path):
    glsl = _translate(tmp_path, GRAPHICS.format(**SPELLINGS["hlsl"]), "opengl")
    assert "gl_Position = vec4(" in glsl
    assert "out vec4 pos" not in glsl


SAMPLE_MASK = """shader Mask {
struct FSOut { vec4 color @gl_FragColor; uint mask @builtin(sample_mask); };
fragment {
FSOut main(vec4 fc @builtin(position), uint inmask @builtin(sample_mask)) {
    FSOut o;
    o.color = fc;
    o.mask = inmask & 1u;
    return o;
}
}
}
"""


@pytest.mark.parametrize(
    "backend, needles",
    [
        (
            "opengl",
            ["uint(gl_SampleMaskIn[0])", "gl_SampleMask[0] = int((inmask & 1u));"],
        ),
        (
            "vulkan",
            [
                "%gl_SampleMaskIn BuiltIn SampleMask",
                "%gl_SampleMask BuiltIn SampleMask",
            ],
        ),
        ("directx", ["uint mask : SV_Coverage;", "uint inmask : SV_Coverage"]),
        ("metal", ["uint mask [[sample_mask]];", "uint inmask [[sample_mask]]"]),
    ],
)
def test_sample_mask_takes_the_name_of_its_direction(tmp_path, backend, needles):
    code = _translate(tmp_path, SAMPLE_MASK, backend)
    for needle in needles:
        assert needle in code
    _validate(tmp_path, backend, code, "fragment")


def _parsed(source):
    return parse(source)


def _attribute_names(node):
    return [attribute.name for attribute in node.attributes]


def test_rewrite_spells_each_builtin_for_the_target_family():
    source = GRAPHICS.format(**SPELLINGS["wgsl"])
    for target, position, vertex_index in (
        ("directx", "SV_POSITION", "SV_VertexID"),
        ("hlsl", "SV_POSITION", "SV_VertexID"),
        ("opengl", "gl_Position", "gl_VertexID"),
        ("vulkan", "gl_Position", "gl_VertexID"),
        ("metal", "gl_Position", "gl_VertexID"),
    ):
        program = _parsed(source)
        assert canonicalize_builtin_values(program, target) == 6
        vs_out = next(s for s in program.structs if s.name == "VSOut")
        assert _attribute_names(vs_out.members[0]) == [position]
        entry = program.stages[A.ShaderStage.VERTEX].entry_point
        assert _attribute_names(entry.parameters[0]) == [vertex_index]


def test_rewrite_keeps_native_spellings_and_other_targets():
    program = _parsed(GRAPHICS.format(**SPELLINGS["hlsl"]))
    assert canonicalize_builtin_values(program, "directx") == 0
    program = _parsed(GRAPHICS.format(**SPELLINGS["glsl"]))
    assert canonicalize_builtin_values(program, "opengl") == 0
    program = _parsed(GRAPHICS.format(**SPELLINGS["wgsl"]))
    assert canonicalize_builtin_values(program, "wgsl") == 0
    assert canonicalize_builtin_values(program, "cuda") == 0


def test_hlsl_spells_glsl_sample_index_as_a_system_value(tmp_path):
    hlsl = _translate(tmp_path, GRAPHICS.format(**SPELLINGS["glsl"]), "directx")
    assert ": SV_SampleIndex" in hlsl
    assert "gl_SampleID" not in hlsl


def test_unknown_builtin_name_is_refused(tmp_path):
    source = GRAPHICS.format(
        **{**SPELLINGS["wgsl"], "sample_index": "@builtin(sample_number)"}
    )
    for backend in TARGETS:
        with pytest.raises(BuiltinValueUnsupportedError, match="sample_number"):
            _translate(tmp_path, source, backend)


POINT_COORD = """shader Sprite {
struct FSOut { vec4 color @gl_FragColor; };
fragment {
FSOut main(vec2 pc %s) { FSOut o; o.color = vec4(pc, 0.0, 1.0); return o; }
}
}
"""


@pytest.mark.parametrize("spelling", ["@gl_PointCoord", "@builtin(point_coord)"])
def test_point_coord_is_refused_for_hlsl(tmp_path, spelling):
    # Direct3D 10 and later have no point sprites, so HLSL has no system value for the sprite coordinate.
    with pytest.raises(BuiltinValueUnsupportedError, match="no system value"):
        _translate(tmp_path, POINT_COORD % spelling, "directx")


@pytest.mark.parametrize("spelling", ["@gl_PointCoord", "@builtin(point_coord)"])
def test_point_coord_reaches_glsl(tmp_path, spelling):
    glsl = _translate(tmp_path, POINT_COORD % spelling, "opengl")
    assert "gl_PointCoord" in glsl
    _validate(tmp_path, "opengl", glsl, "fragment")


NUM_WORKGROUPS = """@compute
@workgroup_size(1, 1, 1)
fn k(@group(0) @binding(0) var<storage, read_write> o: array<u32>, gid: vec3<u32> @builtin(global_invocation_id),
     nwg: vec3<u32> %s) {
    o[gid.x] = nwg.x;
}
"""


@pytest.mark.parametrize("spelling", ["@gl_NumWorkGroups", "@builtin(num_workgroups)"])
@pytest.mark.parametrize(
    "backend, needle",
    [
        ("directx", "uint3 nwg = crossglNumWorkGroups;"),
        ("metal", "[[threadgroups_per_grid]]"),
        ("opengl", "gl_NumWorkGroups.x"),
        ("vulkan", "BuiltIn NumWorkgroups"),
    ],
)
def test_workgroup_count_reaches_every_target(tmp_path, spelling, backend, needle):
    code = _translate(tmp_path, NUM_WORKGROUPS % spelling, backend)
    assert needle in code
    _validate(tmp_path, backend, code, "compute")


def test_sample_mask_on_a_struct_read_and_written_is_refused(tmp_path):
    source = """shader Both {
struct Io { vec4 color @gl_FragColor; uint mask @builtin(sample_mask); };
fragment {
Io main(Io i) { return i; }
}
}
"""
    with pytest.raises(BuiltinValueUnsupportedError, match="gl_SampleMaskIn"):
        _translate(tmp_path, source, "opengl")


@pytest.mark.parametrize(
    "body, store",
    [
        ("return FSOut(fc, 3u);", "gl_SampleMask[0] = int(3u);"),
        ("return make(fc);", "gl_SampleMask[0] = int(crossgl_stage_return.mask);"),
        ("FSOut o = FSOut(fc, 3u); return o;", "gl_SampleMask[0] = int(3u);"),
    ],
    ids=["constructor", "call", "initializer"],
)
def test_glsl_stores_a_uint_sample_mask_as_int(tmp_path, body, store):
    # GLSL declares ``out int gl_SampleMask[]``; assigning a uint to it does not compile.
    source = """shader Mask {
struct FSOut { vec4 color @gl_FragColor; uint mask @SV_Coverage; };
FSOut make(vec4 c) { FSOut o; o.color = c; o.mask = 5u; return o; }
fragment {
FSOut main(vec4 fc @SV_Position) { %s }
}
}
""" % body
    glsl = _translate(tmp_path, source, "opengl")
    assert store in glsl
    _validate(tmp_path, "opengl", glsl, "fragment")


def test_spirv_sample_index_is_a_flat_builtin(tmp_path):
    spirv = _translate(tmp_path, GRAPHICS.format(**SPELLINGS["wgsl"]), "vulkan")
    assert "OpCapability SampleRateShading" in spirv
    assert "OpDecorate %gl_SampleID BuiltIn SampleId" in spirv
    assert spirv.count("OpDecorate %gl_SampleID Flat") == 1
    assert "CrossGL_fragment_input_sid" not in spirv


@pytest.mark.parametrize(
    "spelling", ["@gl_PrimitiveID", "@SV_PrimitiveID", "@builtin(primitive_index)"]
)
def test_spirv_fragment_primitive_id_declares_geometry(tmp_path, spelling):
    source = """shader Prim {
struct FSOut { vec4 color @gl_FragColor; };
fragment {
FSOut main(uint p %s) { FSOut o; o.color = vec4(float(p)); return o; }
}
}
""" % spelling
    spirv = _translate(tmp_path, source, "vulkan")
    assert "OpCapability Geometry" in spirv
    assert "OpDecorate %gl_PrimitiveID BuiltIn PrimitiveId" in spirv
    assert spirv.count("OpDecorate %gl_PrimitiveID Flat") == 1
    _validate(tmp_path, "vulkan", spirv, "fragment")


STAGE_VARIABLES = """shader Stage {{
    vertex {{
        int vertexId @input {vertex};
        vec3 inPos @input @location(0);
        vec4 outPos @output {position};
        void main() {{
            outPos = vec4(inPos * float(vertexId), 1.0);
            return;
        }}
    }}
}}
"""
STAGE_SPELLINGS = {
    "glsl": {"vertex": "@gl_VertexID", "position": "@gl_Position"},
    "hlsl": {"vertex": "@SV_VertexID", "position": "@SV_Position"},
    "wgsl": {"vertex": "@builtin(vertex_index)", "position": "@builtin(position)"},
}


@pytest.mark.parametrize(
    "backend, needles",
    [
        (
            "directx",
            ["uint cgl_vertexId_bits : SV_VertexID", "float4 position : SV_POSITION;"],
        ),
        (
            "metal",
            ["uint cgl_vertexId_bits [[vertex_id]]", "float4 position [[position]];"],
        ),
    ],
)
def test_stage_variables_in_every_spelling_become_entry_point_builtins(
    tmp_path, backend, needles
):
    """Stage-level built-in variables reach the HLSL and MSL entry point as built-ins whatever their spelling."""
    outputs = {
        spelling: _translate(tmp_path, STAGE_VARIABLES.format(**names), backend)
        for spelling, names in STAGE_SPELLINGS.items()
    }
    for spelling, code in outputs.items():
        for needle in needles:
            assert (
                needle.lower() in code.lower()
            ), f"{spelling} spelling: {needle!r} missing"
    assert outputs["wgsl"] == outputs["glsl"]
    _validate(tmp_path, backend, outputs["wgsl"], "vertex")
