"""SPIR-V output of ray payloads and callable data named by location.

GL_EXT_ray_tracing, and the GLSL and SPIR-V importers, name the payload of ``TraceRay`` and the callable data of
``CallShader`` by the location of a ``rayPayloadEXT`` or ``callableDataEXT`` variable. ``OpTraceRayKHR`` and
``OpExecuteCallableKHR`` take the variable itself, and the SPIR-V emitter refused the location ("must be an
addressable value"). It now passes the variable at that location.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

TRACE = """shader main {{
    accelerationStructureEXT scene @binding(1);
    layout(std430, binding = 0) buffer Output {{
    float outp[];
    }} output;
    vec4 near @location(0) @rayPayloadEXT;
    vec4 far @location({far_location}) @rayPayloadEXT;
    ray_generation {{
    void main() {{
        near = vec4(0.0);
        far = vec4(1.0);
        TraceRay(scene, gl_RayFlagsNoneEXT, 0xFF, 0u, 1u, 0u, vec3(0.0), 0.0, vec3(0.0, 0.0, 1.0), 100.0, {location});
        output.outp[0] = near.x + far.x;
    }}
    }}
}}
"""

CALL = """shader main {
    layout(std430, binding = 0) buffer Output {
    float outp[];
    } output;
    vec4 data @location(0) @callableDataEXT;
    ray_generation {
    void main() {
        data = vec4(1.0);
        CallShader(0u, 0);
        output.outp[0] = data.x;
    }
    }
}
"""

GLSL = """#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) rayPayloadEXT vec4 payload;
layout(binding = 1) uniform accelerationStructureEXT scene;
layout(std430, binding = 0) buffer Output { float outp[]; };
void main() {
    payload = vec4(0.0);
    traceRayEXT(scene, gl_RayFlagsNoneEXT, 0xFF, 0u, 1u, 0u, vec3(0.0), 0.0, vec3(0.0, 0.0, 1.0), 100.0, 0);
    outp[0] = payload.x;
}
"""


def _spirv(tmp_path, source, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    spirv = crosstl.translate(str(path), backend="vulkan")
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler and validator:
        text, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        text.write_text(spirv)
        environment = ["--target-env", "vulkan1.2"]
        assembled = subprocess.run(
            [assembler, *environment, str(text), "-o", str(binary)],
            capture_output=True,
            text=True,
        )
        assert assembled.returncode == 0, assembled.stderr
        validated = subprocess.run(
            [validator, *environment, str(binary)], capture_output=True, text=True
        )
        assert validated.returncode == 0, validated.stderr + spirv
    return spirv


def _operand_name(spirv, opcode):
    """The ``%name`` of the last operand of ``opcode``, through ``OpName`` when the ids are numbers.

    Without SPIRV-Tools the emitter writes numeric ids and names them with ``OpName`` only.
    """
    instruction = re.search(rf"^\s*{opcode}\b(.*)$", spirv, flags=re.M)
    assert instruction is not None, spirv
    operand = instruction.group(1).split()[-1]
    names = dict(re.findall(r'OpName (%\w+) "(\w+)"', spirv))
    return f"%{names[operand]}" if operand in names else operand


@pytest.mark.parametrize("location, name", [(0, "%near"), (1, "%far")])
def test_trace_ray_passes_the_payload_at_its_location(tmp_path, location, name):
    spirv = _spirv(tmp_path, TRACE.format(far_location=1, location=location))
    assert _operand_name(spirv, "OpTraceRayKHR") == name


def test_call_shader_passes_the_callable_data_at_its_location(tmp_path):
    spirv = _spirv(tmp_path, CALL)
    assert _operand_name(spirv, "OpExecuteCallableKHR") == "%data"


def test_glsl_trace_ray_reaches_spirv(tmp_path):
    spirv = _spirv(tmp_path, GLSL, ".rgen")
    assert _operand_name(spirv, "OpTraceRayKHR") == "%payload"


@pytest.mark.parametrize("location", [0, 7], ids=["shared", "unoccupied"])
def test_a_location_that_names_no_single_payload_is_refused(tmp_path, location):
    # Two payloads at location 0, and none at location 7.
    source = TRACE.format(far_location=0, location=location)
    with pytest.raises(SpirvLoweringRefused, match="payload argument"):
        _spirv(tmp_path, source)
