"""Project translation of kernels that read a workgroup size the source's host chooses at dispatch time.

A Metal kernel that reads ``[[threads_per_threadgroup]]`` gets that size from its host. Without a project workgroup
size, the DirectX, OpenGL and Vulkan generators give it the size the target's host sets: HLSL macros, GLSL
specialization ids or macros, and SPIR-V specialization constants. The read then sees the size the kernel runs with,
as it does on Metal. A CrossGL kernel has no host-chosen size, so reading its unconfigured size stays refused.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl.project as project_api
from crosstl.project import translate_project
from crosstl.translator.codegen.host_workgroup_size_ids import (
    HOST_WORKGROUP_SIZE_MACROS,
)

GLSLANG = shutil.which("glslangValidator")
SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")

READS = {
    "uint": "group_size",
    "uint3": "group_size.x + group_size.y + group_size.z",
}

# The read, lowered to the size each target's host sets.
LOWERED_READS = {
    "directx": (
        "uint3(CROSSGL_WORKGROUP_SIZE_X, CROSSGL_WORKGROUP_SIZE_Y, CROSSGL_WORKGROUP_SIZE_Z)"
    ),
    "opengl": "gl_WorkGroupSize",
    "vulkan": "BuiltIn WorkgroupSize",
}


def _write_kernel(repo, value_type):
    repo.mkdir()
    (repo / "group_size.metal").write_text(
        textwrap.dedent(f"""
            #include <metal_stdlib>
            using namespace metal;

            kernel void read_size(
                device uint* output [[buffer(0)]],
                uint3 lid [[thread_position_in_threadgroup]],
                {value_type} group_size [[threads_per_threadgroup]]) {{
                output[lid.x] = {READS[value_type]};
            }}
            """).lstrip(),
        encoding="utf-8",
    )


def _translate(repo, target, **config):
    return translate_project(
        project_api.ProjectConfig(
            root=repo, targets=[target], output_dir="out", **config
        ),
        format_output=False,
    ).to_json()


def _translated_artifact(payload, repo):
    assert payload["diagnostics"] == []
    assert payload["summary"]["translatedCount"] == 1
    (artifact,) = payload["artifacts"]
    assert artifact["status"] == "translated"
    return artifact, (repo / artifact["path"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("value_type", sorted(READS))
@pytest.mark.parametrize("target", ["directx", "opengl", "vulkan"])
def test_unconfigured_metal_size_read_takes_the_host_chosen_size(
    tmp_path, target, value_type
):
    repo = tmp_path / "repo"
    _write_kernel(repo, value_type)

    artifact, code = _translated_artifact(_translate(repo, target), repo)

    # No size is recorded, so nothing dispatches a placeholder.
    assert artifact.get("execution") is None
    assert LOWERED_READS[target] in code
    if target == "vulkan":
        assert len(re.findall(r"= OpSpecConstant ", code)) == 3
    else:
        for macro in HOST_WORKGROUP_SIZE_MACROS:
            assert macro in code
        assert "#error" in code


@pytest.mark.parametrize("target", ["directx", "opengl", "vulkan"])
def test_configured_size_still_fixes_a_metal_size_read(tmp_path, target):
    repo = tmp_path / "repo"
    _write_kernel(repo, "uint3")

    artifact, code = _translated_artifact(
        _translate(repo, target, workgroup_size=[64, 2, 1]), repo
    )

    assert artifact["execution"]["workgroupSize"] == [64, 2, 1]
    assert "CROSSGL_WORKGROUP_SIZE_X" not in code
    assert "OpSpecConstant " not in code


@pytest.mark.skipif(GLSLANG is None, reason="glslangValidator is not installed")
def test_glslang_compiles_both_host_chosen_paths_of_a_size_read(tmp_path):
    repo = tmp_path / "repo"
    _write_kernel(repo, "uint3")
    artifact, code = _translated_artifact(_translate(repo, "opengl"), repo)
    shader = repo / artifact["path"]

    def glslang(*arguments):
        return subprocess.run(
            [GLSLANG, *arguments, "-S", "comp", str(shader)],
            capture_output=True,
            text=True,
            check=False,
            cwd=tmp_path,
        )

    assert glslang().returncode != 0
    defines = [f"-D{macro}=8" for macro in HOST_WORKGROUP_SIZE_MACROS]
    source_path = glslang("-l", *defines)
    assert source_path.returncode == 0, source_path.stdout + source_path.stderr
    spirv = glslang("-V", "--target-env", "vulkan1.1", "-o", "kernel.spv")
    assert spirv.returncode == 0, spirv.stdout + spirv.stderr


@pytest.mark.skipif(
    SPIRV_AS is None or SPIRV_VAL is None, reason="SPIRV-Tools are not installed"
)
def test_spirv_val_accepts_the_host_chosen_size_read(tmp_path):
    repo = tmp_path / "repo"
    _write_kernel(repo, "uint3")
    artifact, _ = _translated_artifact(_translate(repo, "vulkan"), repo)
    binary = tmp_path / "kernel.spv"

    subprocess.run(
        [
            SPIRV_AS,
            "--target-env",
            "spv1.3",
            str(repo / artifact["path"]),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.1", str(binary)], check=True)
