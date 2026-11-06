"""Function pointer template parameters bind their own explicit argument.

``template <typename T, short N, void (*load)(device const T *, short, thread float &), short M>`` names its
third parameter inside the declarator. The template parameter parser looked for the name at the end of the
declarator, filed the parameter as an anonymous constraint, and bound explicit arguments to the remaining
names by position: ``M`` took the function name and every later argument shifted. ggml's flash-attention
kernels pass their dequantizers this way, and their instantiations ended up dividing by a vector type.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.backend.Metal.template_function_parameters import (
    template_function_pointer_parameter,
)

KERNEL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    void scale_by_two(device const float *src, short i, thread float &out) { out = src[i] * 2.0f; }
    void scale_by_three(device const float *src, short i, thread float &out) { out = src[i] * 3.0f; }

    template <typename in_t, short step, void (*load)(device const in_t *, short, thread float &), short count>
    kernel void apply(device const in_t *src [[buffer(0)]],
                      device float *dst [[buffer(1)]],
                      uint gid [[thread_position_in_grid]])
    {
        float value;
        load(src + gid / count, short(gid % step), value);
        dst[gid] = value;
    }

    typedef decltype(apply<float, 2, scale_by_two, 4>) apply_t;

    template [[host_name("apply_two")]] kernel apply_t apply<float, 2, scale_by_two, 4>;
    template [[host_name("apply_three")]] kernel apply_t apply<float, 3, scale_by_three, 8>;
    """)


@pytest.mark.parametrize(
    ("declarator", "expected"),
    [
        (
            "void (*load)(device const T *, short, thread float &)",
            ("load", "void (*)(device const T *, short, thread float &)"),
        ),
        ("float (&pick)(float, float)", ("pick", "float (&)(float, float)")),
        ("void ( * f )( )", ("f", "void (*)()")),
        ("void (*)(int)", None),
        ("short N", None),
        ("metal::enable_if_t<(N > 2), bool>", None),
    ],
)
def test_function_pointer_declarators(declarator, expected):
    assert template_function_pointer_parameter(declarator) == expected


def test_function_pointer_parameter_keeps_its_position():
    parameters = MetalPreprocessor()._parse_template_parameter_list(
        "typename in_t, short step, "
        "void (*load)(device const in_t *, short, thread float &), short count = 4"
    )
    assert [parameter.name for parameter in parameters] == [
        "in_t",
        "step",
        "load",
        "count",
    ]
    load = parameters[2]
    assert not load.is_type_parameter and not load.is_constraint
    assert load.declared_type == "void (*)(device const in_t *, short, thread float &)"
    assert parameters[3].default == "4"


def _translate(tmp_path, backend):
    source = tmp_path / "apply.metal"
    source.write_text(KERNEL, encoding="utf-8")
    return crosstl.translate(str(source), backend=backend)


def test_instantiations_bind_arguments_after_the_function_pointer(tmp_path):
    code = _translate(tmp_path, "cgl")
    two = code[code.index("void apply_two(") :]
    two = two[: two.index("}")]
    three = code[code.index("void apply_three(") :]
    three = three[: three.index("}")]
    assert "scale_by_two(src + gid / 4, int16(gid % 2), value);" in two
    assert "scale_by_three(src + gid / 8, int16(gid % 3), value);" in three
    assert "void apply(" not in code


_VALIDATORS = {
    "opengl": (
        "apply.comp",
        [["glslangValidator", "-S", "comp", "apply.comp"]],
    ),
    "vulkan": (
        "apply.spvasm",
        [
            ["spirv-as", "--target-env", "vulkan1.1", "apply.spvasm", "-o", "a.spv"],
            ["spirv-val", "--target-env", "vulkan1.1", "a.spv"],
        ],
    ),
    "directx": (
        "apply.hlsl",
        [
            ["dxc", "-T", "cs_6_2", "-enable-16bit-types", "-E", entry, "apply.hlsl"]
            for entry in ("CSMain", "CSMain_2")
        ],
    ),
    "metal": (
        "apply.metal",
        [["xcrun", "-sdk", "macosx", "metal", "-c", "apply.metal", "-o", "a.air"]],
    ),
}


@pytest.mark.parametrize("backend", sorted(_VALIDATORS))
def test_instantiations_reach_every_target(tmp_path, backend):
    file_name, commands = _VALIDATORS[backend]
    code = _translate(tmp_path, backend)
    missing = [command[0] for command in commands if shutil.which(command[0]) is None]
    if missing:
        pytest.skip(f"{', '.join(missing)} not installed")
    (tmp_path / file_name).write_text(code, encoding="utf-8")
    for command in commands:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300, cwd=tmp_path
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
