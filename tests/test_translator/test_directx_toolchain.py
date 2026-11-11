import json
import shutil
import subprocess
import textwrap
from types import SimpleNamespace

import pytest

from crosstl.project.directx_toolchain import (
    directx_target_profiles_for_source,
    dxc_compiler_arguments_for_source,
    dxc_profile_for_source,
    hlsl_requires_native_16bit_types,
)
from tests.test_translator.test_project_translation import (
    assert_directx_compute_validates_if_available,
)

NATIVE_16_BIT_SOURCE = "RWStructuredBuffer<uint16_t> values;"
EXACT_WAVE_SIZE_SOURCE = "[WaveSize(32)]\n[numthreads(32, 2, 1)]\nvoid CSMain() {}"


@pytest.mark.parametrize(
    "source",
    (
        "float16_t value;",
        "int16_t2 value;",
        "uint16_t4 value;",
        "vector<float16_t, 3> value;",
    ),
)
def test_native_16_bit_scalar_and_vector_types_are_detected(source):
    assert hlsl_requires_native_16bit_types(source) is True


@pytest.mark.parametrize(
    "source",
    (
        "min16float value; min16int2 pair; min16uint4 lanes;",
        "float16_type value;",
        "my_int16_t value;",
        "uint16_t_values value;",
        "float16_/* do not join tokens across comments */t value;",
    ),
)
def test_non_native_or_embedded_type_names_are_not_detected(source):
    assert hlsl_requires_native_16bit_types(source) is False


@pytest.mark.parametrize(
    "source",
    (
        "// uint16_t4 ignored;\nfloat4 value;",
        "/* float16_t ignored;\nint16_t2 also_ignored; */ float value;",
        "float value; /* unterminated uint16_t4 comment",
        'string type_name = "uint16_t4"; float value;',
        r'string type_name = "escaped \" // int16_t2"; float value;',
    ),
)
def test_native_type_names_in_comments_and_literals_are_ignored(source):
    assert hlsl_requires_native_16bit_types(source) is False


@pytest.mark.parametrize(
    "source",
    (
        'string marker = "// uint16_t4"; int16_t value;',
        'string marker = "/* float16_t */"; uint16_t2 value;',
    ),
)
def test_comment_markers_in_literals_do_not_hide_following_code(source):
    assert hlsl_requires_native_16bit_types(source) is True


@pytest.mark.parametrize(
    ("profile", "expected"),
    (
        ("vs_6_0", "vs_6_2"),
        ("ps_6_1", "ps_6_2"),
        ("cs_6_2", "cs_6_2"),
        ("lib_6_3", "lib_6_3"),
        ("ms_6_6", "ms_6_6"),
        ("cs_6_10", "cs_6_10"),
        ("cs_7_0", "cs_7_0"),
    ),
)
def test_native_types_raise_profiles_to_6_2_without_lowering(profile, expected):
    assert dxc_profile_for_source(profile, NATIVE_16_BIT_SOURCE) == expected


def test_profiles_without_native_types_and_unrecognized_profiles_are_unchanged():
    assert dxc_profile_for_source("cs_6_0", "float value;") == "cs_6_0"
    assert dxc_profile_for_source("not-a-profile", NATIVE_16_BIT_SOURCE) == (
        "not-a-profile"
    )


@pytest.mark.parametrize(
    ("profile", "expected"),
    (
        ("cs_6_0", "cs_6_6"),
        ("cs_6_5", "cs_6_6"),
        ("cs_6_6", "cs_6_6"),
        ("lib_6_3", "lib_6_6"),
        ("cs_6_10", "cs_6_10"),
    ),
)
def test_exact_wave_size_raises_profiles_to_6_6_without_lowering(profile, expected):
    assert dxc_profile_for_source(profile, EXACT_WAVE_SIZE_SOURCE) == expected


def test_exact_wave_size_ignores_comments_and_literals():
    source = '// [WaveSize(32)]\nstring marker = "[WaveSize(32)]";'

    assert dxc_profile_for_source("cs_6_0", source) == "cs_6_0"
    assert directx_target_profiles_for_source(source) == (
        "directx-11",
        "directx-12",
    )


def test_exact_wave_size_requires_directx_12_without_native_16_bit_flag():
    assert dxc_compiler_arguments_for_source(EXACT_WAVE_SIZE_SOURCE) == ()
    assert directx_target_profiles_for_source(EXACT_WAVE_SIZE_SOURCE) == ("directx-12",)
    assert (
        dxc_profile_for_source(
            "cs_6_0", f"{NATIVE_16_BIT_SOURCE}\n{EXACT_WAVE_SIZE_SOURCE}"
        )
        == "cs_6_6"
    )


@pytest.mark.parametrize(
    "source",
    (
        "float16_t a; int16_t2 b; uint16_t4 c;",
        "uint16_t4 c; float16_t a; int16_t2 b;",
    ),
)
def test_native_types_have_deterministic_compiler_and_target_requirements(source):
    assert dxc_compiler_arguments_for_source(source) == ("-enable-16bit-types",)
    assert directx_target_profiles_for_source(source) == ("directx-12",)


def test_non_native_types_keep_default_compiler_and_target_requirements():
    source = "min16float4 value; // uint16_t is only documentation"

    assert dxc_compiler_arguments_for_source(source) == ()
    assert directx_target_profiles_for_source(source) == (
        "directx-11",
        "directx-12",
    )


@pytest.mark.parametrize(
    "source,profile,native16",
    [
        ("[numthreads(1, 1, 1)] void CSMain() {}", "cs_6_2", False),
        (NATIVE_16_BIT_SOURCE, "cs_6_2", True),
        (EXACT_WAVE_SIZE_SOURCE, "cs_6_6", False),
        (NATIVE_16_BIT_SOURCE + EXACT_WAVE_SIZE_SOURCE, "cs_6_6", True),
        ("// [WaveSize(32)]\nfloat value;", "cs_6_2", False),
    ],
)
def test_native_test_validator_uses_source_compiler_contract(
    tmp_path, monkeypatch, source, profile, native16
):
    from tests import runtime_helpers

    shader = tmp_path / "shader.hlsl"
    shader.write_text(source, encoding="utf-8")
    formatted = []
    commands = []

    def format_path(path):
        formatted.append(path)
        return "formatted:" + str(path)

    def bridge(command, source_path):
        assert source_path == shader
        commands.append(command)
        return ["compiler-bridge"]

    def run(command, directory, name, *, timeout):
        assert command == ["compiler-bridge"] and timeout == 300
        (directory / "kernel.dxil").write_bytes(b"compiled")
        (directory / f"{name}.json").write_text(
            json.dumps(dict(stdout="", stderr="")), encoding="utf-8"
        )

    monkeypatch.setattr(runtime_helpers, "dxc_file_path", format_path)
    monkeypatch.setattr(runtime_helpers, "dxc_long_path_command", bridge)
    monkeypatch.setattr(runtime_helpers, "run", run)
    runtime_helpers._validate_generated_artifact(shader, tmp_path, "directx")
    assert formatted == [shader, tmp_path / "kernel.dxil"]
    (command,) = commands
    assert command[command.index("-T") + 1] == profile
    assert ("-enable-16bit-types" in command) == native16
    assert "-WX" in command and command[command.index("-E") + 1] == "CSMain"
    assert "formatted:" + str(shader) in command
    assert command[-1] == "formatted:" + str(tmp_path / "kernel.dxil")


@pytest.mark.parametrize(
    "stage",
    (
        "raygeneration",
        "intersection",
        "anyhit",
        "closesthit",
        "miss",
        "callable",
        "mesh",
        "amplification",
    ),
)
def test_directx_12_only_stages_require_directx_12(stage):
    source = f'[shader("{stage}")]\nvoid Entry() {{}}'

    assert directx_target_profiles_for_source(source) == ("directx-12",)


@pytest.mark.parametrize(
    "call",
    (
        "WaveActiveSum(value)",
        "WaveReadLaneAt(value, 0u)",
        "WaveGetLaneIndex()",
        "WaveMultiPrefixSum(value, WaveMatch(value))",
        "QuadReadAcrossX(value)",
        "QuadAny(value != 0u)",
    ),
)
def test_wave_intrinsics_require_directx_12(call):
    source = f"uint value = 1u;\nuint result = {call};"

    assert directx_target_profiles_for_source(source) == ("directx-12",)


def test_wave_size_range_requires_directx_12():
    source = "[WaveSize(16, 64)]\n[numthreads(64, 1, 1)]\nvoid main() {}"

    assert directx_target_profiles_for_source(source) == ("directx-12",)


def test_directx_12_stage_and_wave_detection_ignores_comments_and_literals():
    source = (
        '// [shader("raygeneration")] WaveActiveSum(x)\n'
        '/* [shader("mesh")] */\n'
        'string marker = "[shader(\\"amplification\\")] QuadReadAcrossX(x)";\n'
        "uint my_WaveActiveSum(uint x) { return x; }\n"
        "float WaveHeight(float x) { return x * 0.5; }\n"
        "float QuadReadValue(float x) { return x; }\n"
        '[shader("compute")]\n'
        "[numthreads(1, 1, 1)]\n"
        "void main() {\n"
        "    uint y = my_WaveActiveSum(1u);\n"
        "    float h = WaveHeight(QuadReadValue(1.0));\n"
        "}"
    )

    assert directx_target_profiles_for_source(source) == (
        "directx-11",
        "directx-12",
    )


@pytest.mark.parametrize(
    "source,expected_profile,native16",
    [
        ("[numthreads(1, 1, 1)] void CSMain() {}", "cs_6_0", False),
        (NATIVE_16_BIT_SOURCE, "cs_6_2", True),
        (EXACT_WAVE_SIZE_SOURCE, "cs_6_6", False),
        (NATIVE_16_BIT_SOURCE + EXACT_WAVE_SIZE_SOURCE, "cs_6_6", True),
        ("// uint16_t in comment\nfloat value;", "cs_6_0", False),
    ],
)
def test_directx_compute_validation_helper_uses_source_compiler_contract(
    tmp_path, monkeypatch, source, expected_profile, native16
):
    commands = []

    def fake_run(command, *args, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        shutil, "which", lambda tool: "/bin/dxc" if tool == "dxc" else None
    )

    assert_directx_compute_validates_if_available(source, tmp_path)

    assert len(commands) == 1
    command = commands[0]
    assert command[0] == "/bin/dxc"
    assert command[command.index("-T") + 1] == expected_profile
    assert ("-enable-16bit-types" in command) is native16
    assert command[command.index("-E") + 1] == "CSMain"
    assert command[command.index("-Fo") + 1] == str(tmp_path / "shader.dxil")


def test_directx_compute_validation_helper_falls_back_to_glslang(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, *args, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_which(tool):
        if tool == "glslangValidator":
            return "/bin/glslangValidator"
        return None

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", fake_which)

    source = "[numthreads(1, 1, 1)] void CSMain() {}"
    assert_directx_compute_validates_if_available(source, tmp_path)

    assert len(commands) == 1
    command = commands[0]
    assert command[0] == "/bin/glslangValidator"
    assert command[command.index("-e") + 1] == "CSMain"
    assert command[command.index("-o") + 1] == str(tmp_path / "shader.spv")


def test_directx_compute_validation_helper_compiles_native_16bit_under_dxc(tmp_path):
    if not shutil.which("dxc"):
        pytest.skip("dxc is not installed")

    hlsl_16bit = textwrap.dedent("""
        RWStructuredBuffer<uint16_t> u16_buf : register(u0);
        RWStructuredBuffer<int16_t> s16_buf : register(u1);
        RWStructuredBuffer<float16_t> f16_buf : register(u2);
        [numthreads(1, 1, 1)]
        void CSMain(uint3 tid : SV_DispatchThreadID) {
            u16_buf[0] = uint16_t(1);
            s16_buf[0] = int16_t(-1);
            f16_buf[0] = float16_t(1.5);
        }
    """).strip()
    assert_directx_compute_validates_if_available(hlsl_16bit, tmp_path)
    output = tmp_path / "shader.dxil"
    assert output.is_file() and output.stat().st_size > 0


def test_directx_compute_validation_helper_compiles_ordinary_hlsl_under_dxc(tmp_path):
    if not shutil.which("dxc"):
        pytest.skip("dxc is not installed")

    hlsl_ordinary = textwrap.dedent("""
        RWStructuredBuffer<float> out_buf : register(u0);
        [numthreads(1, 1, 1)]
        void CSMain(uint3 tid : SV_DispatchThreadID) {
            out_buf[0] = 1.0f;
        }
    """).strip()
    assert_directx_compute_validates_if_available(hlsl_ordinary, tmp_path)
    output = tmp_path / "shader.dxil"
    assert output.is_file() and output.stat().st_size > 0


def test_directx_compute_validation_helper_preserves_compiler_diagnostics(tmp_path):
    if not shutil.which("dxc"):
        pytest.skip("dxc is not installed")

    bad_hlsl = "[numthreads(1, 1, 1)] void CSMain() { invalid_syntax_error; }"
    with pytest.raises(AssertionError) as exc_info:
        assert_directx_compute_validates_if_available(bad_hlsl, tmp_path)
    assert "error" in str(exc_info.value).lower()
