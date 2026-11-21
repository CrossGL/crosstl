"""Device address-space variables are refused for targets without program-scope device memory.

A CrossGL ``device`` variable is one object that every invocation shares. Only CUDA and HIP declare one
(``__device__``); the other generators wrote it as a uniform, a per-invocation copy or nothing (issue #4726).
"""

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import DeviceVariableUnsupportedError
from crosstl.translator.semantics.device_variables import refuse_device_variables

REFUSING_TARGETS = [
    "directx",
    "metal",
    "opengl",
    "webgl",
    "slang",
    "vulkan",
    "wgsl",
    "rust",
    "mojo",
]

SHADER_SCOPE = """
shader Probe {
    device float acc[64] = {1.0, 2.0, 4.0};
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            acc[i & 63u] = acc[i & 63u] * 2.0 + fin[i];
            outp[i] = acc[i & 63u];
        }
    }
}
"""

STAGE_SCOPE = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        device float acc[64];
        buffer float* fin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            acc[i & 63u] = acc[i & 63u] * 2.0 + fin[i];
            outp[i] = acc[i & 63u];
        }
    }
}
"""


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _write(tmp_path, source):
    path = tmp_path / "probe.cgl"
    path.write_text(source, encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("target", REFUSING_TARGETS)
@pytest.mark.parametrize("source", [SHADER_SCOPE, STAGE_SCOPE], ids=["shader", "stage"])
def test_device_variable_is_refused_by_name_and_target(tmp_path, source, target):
    with pytest.raises(DeviceVariableUnsupportedError) as raised:
        crosstl.translate(_write(tmp_path, source), backend=target)
    error = raised.value
    assert error.variable_name == "acc"
    assert error.target == target
    assert error.code == f"project.translate.{target}-device-variable-unsupported"
    assert "'acc'" in str(error)


@pytest.mark.parametrize("target", ["cuda", "hip"])
def test_cuda_and_hip_keep_the_device_variable(tmp_path, target):
    code = crosstl.translate(_write(tmp_path, SHADER_SCOPE), backend=target)
    assert "acc[64]" in code


@pytest.mark.parametrize("target", ["", "cgl", "crossgl", "cuda", "hip"])
def test_keeping_targets_pass_the_program_through(target):
    refuse_device_variables(_parse(SHADER_SCOPE), target)
    refuse_device_variables(_parse(STAGE_SCOPE), target)


@pytest.mark.parametrize(
    "declaration",
    [
        "var<device> acc: array<f32, 64>;",
        "device float acc;",
        "device vec4 acc;",
        "device mat4 acc;",
        "device Counters acc;",
        "device Counters acc[4];",
    ],
)
def test_every_plain_value_type_is_refused(declaration):
    program = _parse(f"""
shader Probe {{
    struct Counters {{
        uint hits;
        float total[2];
    }};
    {declaration}
    compute {{
        layout(local_size_x = 1) in;
        buffer float* outp;
        void main() {{
            outp[0] = 1.0;
        }}
    }}
}}
""")
    with pytest.raises(DeviceVariableUnsupportedError, match="'acc'"):
        refuse_device_variables(program, "metal")


@pytest.mark.parametrize(
    "declaration",
    [
        "device float* acc;",
        "float acc[64];",
        "static float acc[64];",
        "buffer float* acc;",
    ],
)
def test_pointers_and_non_device_variables_are_left_alone(declaration):
    program = _parse(f"""
shader Probe {{
    {declaration}
    compute {{
        layout(local_size_x = 1) in;
        buffer float* outp;
        void main() {{
            outp[0] = 1.0;
        }}
    }}
}}
""")
    refuse_device_variables(program, "metal")
