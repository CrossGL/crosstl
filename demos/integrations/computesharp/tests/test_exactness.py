"""Check the arithmetic exactness analysis of translated artifacts.

The analyzers read the output of the actual target compilers: LLVM IR from
the Metal offline compiler and the NIR Mesa prints with ``MESA_GLSL=dump``.
The snippets below are reduced from that output. Each defect the check must
reject is introduced into an otherwise consistent report.
"""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from fixture_support import ROOT, load_module  # noqa: E402

EXACTNESS = load_module("computesharp_exactness", ROOT / "exactness.py")

FUSED_METAL_IR = """
define void @Execute(float addrspace(1)* %0) {
  %5 = fmul float %3, %4
  %6 = fadd contract float %5, %2
  %7 = tail call float @llvm.fmuladd.f32(float %3, float %4, float %6)
  %8 = tail call fast float @air.dot.v3f32(<3 x float> %a, <3 x float> %b)
  %9 = tail call float @air.mix.f32(float %7, float %8, float 5.000000e-01)
  %10 = fadd fast <2 x float> %c, %d
  ret void
}
declare float @llvm.fmuladd.f32(float, float, float)
"""

EXACT_METAL_IR = """
define void @Execute(float addrspace(1)* %0) {
  %5 = fmul float %3, %4
  %6 = fadd float %5, %2
  %7 = fsub <4 x float> %6, %5
  %8 = tail call float @air.dot.v3f32(<3 x float> <float 0.0, float 1.0, float 1.0>, <3 x float> <float 0.0, float 1.0, float 1.0>)
  %9 = tail call float @air.fast_sqrt.f32(float %8)
  %10 = mul i32 %a, %b
  ret void
}
declare float @air.dot.v3f32(<3 x float>, <3 x float>)
"""

NIR_DUMP = """
(declare (precise ) float p)
shader: MESA_SHADER_COMPUTE
name: GLSL2
impl main {
    block b0:   // preds:
    32     %0 = load_const (0x00000000)
    32    %31 = fmul! %14, %22
    32    %32 = fadd! %31, %30
    32x4  %33 = ffma %31, %32, %30
    32    %34 = fdot3 %33.xyz, %33.xyz
    32    %35 = iadd %0, %0
    32    %36 = fsat %34
}
"""


def test_metal_ir_counts_fused_calls_and_contracting_flags():
    assert EXACTNESS.analyze_metal_ir(FUSED_METAL_IR) == {
        "floatOperations": 3,
        "contractible": 5,
        "fusedCalls": 3,
    }
    assert EXACTNESS.analyze_metal_ir(EXACT_METAL_IR) == {
        "floatOperations": 3,
        "contractible": 0,
        "fusedCalls": 0,
    }


def test_nir_requires_exact_float_arithmetic():
    assert EXACTNESS.analyze_nir(NIR_DUMP) == {
        "floatOperations": 4,
        "contractible": 2,
        "nirShaders": 1,
    }
    exact = NIR_DUMP.replace("= ffma ", "= ffma! ").replace("= fdot3 ", "= fdot3! ")
    assert EXACTNESS.analyze_nir(exact)["contractible"] == 0


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def synthetic(tmp_path):
    manifest = {
        "shaders": {
            "a" * 64: {"artifact": "opengl/a.comp", "status": "translated"},
            "b" * 64: {"artifact": "opengl/b.comp", "status": "translated"},
            "c" * 64: {"artifact": "opengl/c.comp", "status": "compile-failed"},
        }
    }
    report = {
        "schemaVersion": EXACTNESS.SCHEMA_VERSION,
        "kind": EXACTNESS.KIND,
        "target": "opengl",
        "tool": {"driver": "llvmpipe"},
        "artifacts": {},
    }
    for index, (shader_id, shader) in enumerate(sorted(manifest["shaders"].items())):
        if shader["status"] != "translated":
            report["artifacts"][shader_id] = {
                "status": "not-compiled",
                "manifestStatus": shader["status"],
            }
            continue
        shader["artifactSha256"] = digest(shader_id)
        report["artifacts"][shader_id] = {
            "artifactSha256": digest(shader_id),
            "status": "exact",
            "floatOperations": 10 * index,
            "contractible": 0,
            "nirShaders": 1,
        }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    report["manifestSha256"] = EXACTNESS.sha256_file(manifest_path)
    return report, manifest, manifest_path


def test_a_consistent_report_verifies(tmp_path):
    report, manifest, manifest_path = synthetic(tmp_path)
    summary = EXACTNESS.verify(
        report, manifest, manifest_sha256=report["manifestSha256"]
    )
    assert summary == {
        "status": "passed",
        "target": "opengl",
        "tool": {"driver": "llvmpipe"},
        "exactArtifacts": 2,
        "notCompiled": 1,
        "floatOperations": 10,
        "contractible": 0,
    }
    report_path = tmp_path / "exactness.json"
    report_path.write_text(json.dumps(report))
    assert EXACTNESS.verify_file(report_path, manifest_path, target="opengl") == (
        summary
    )


def contractible(report):
    entry = report["artifacts"]["b" * 64]
    entry.update(status="contractible", contractible=3)


def missing(report):
    del report["artifacts"]["a" * 64]


def other_artifact(report):
    report["artifacts"]["a" * 64]["artifactSha256"] = "0" * 64


def compiled_failure_claimed_exact(report):
    report["artifacts"]["c" * 64] = {"status": "exact", "contractible": 0}


def unknown_shader(report):
    report["artifacts"]["d" * 64] = {"status": "exact", "contractible": 0}


def no_float_operations(report):
    for entry in report["artifacts"].values():
        if "floatOperations" in entry:
            entry["floatOperations"] = 0


def compile_failure(report):
    report["artifacts"]["a" * 64] = {"status": "compile-failed", "log": "error"}


def no_nir(report):
    report["artifacts"]["a" * 64]["status"] = "no-nir"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (contractible, "contractible"),
        (missing, "not analyzed"),
        (other_artifact, "the analyzed artifact differs"),
        (compiled_failure_claimed_exact, "the manifest records compile-failed"),
        (unknown_shader, "not in the manifest"),
        (no_float_operations, "no floating-point operation"),
        (compile_failure, "compile-failed"),
        (no_nir, "no-nir"),
    ],
)
def test_inconsistent_reports_are_rejected(tmp_path, mutate, message):
    report, manifest, _ = synthetic(tmp_path)
    mutated = copy.deepcopy(report)
    mutate(mutated)
    with pytest.raises(EXACTNESS.ExactnessError) as exc_info:
        EXACTNESS.verify(mutated, manifest, manifest_sha256=report["manifestSha256"])
    assert any(
        message in error for error in exc_info.value.errors
    ), exc_info.value.errors


def test_a_suite_without_float_arithmetic_states_it(tmp_path):
    report, manifest, manifest_path = synthetic(tmp_path)
    no_float_operations(report)
    summary = EXACTNESS.verify(
        report,
        manifest,
        manifest_sha256=report["manifestSha256"],
        require_float_operations=False,
    )
    assert summary["status"] == "passed"
    assert summary["floatOperations"] == 0
    report_path = tmp_path / "exactness.json"
    report_path.write_text(json.dumps(report))
    with pytest.raises(EXACTNESS.ExactnessError, match="no floating-point"):
        EXACTNESS.verify_file(report_path, manifest_path, target="opengl")
    # Contraction is still rejected when float arithmetic is not required.
    contractible(report)
    with pytest.raises(EXACTNESS.ExactnessError, match="contractible"):
        EXACTNESS.verify(
            report,
            manifest,
            manifest_sha256=report["manifestSha256"],
            require_float_operations=False,
        )


def test_reports_of_another_manifest_or_target_are_rejected(tmp_path):
    report, manifest, manifest_path = synthetic(tmp_path)
    with pytest.raises(EXACTNESS.ExactnessError, match="different manifest"):
        EXACTNESS.verify(report, manifest, manifest_sha256="0" * 64)
    report_path = tmp_path / "exactness.json"
    report_path.write_text(json.dumps(report))
    with pytest.raises(EXACTNESS.ExactnessError, match="not 'metal'"):
        EXACTNESS.verify_file(report_path, manifest_path, target="metal")
    with pytest.raises(EXACTNESS.ExactnessError, match="missing"):
        EXACTNESS.verify_file(tmp_path / "absent.json", manifest_path, target="opengl")


METAL_KERNEL = """
#include <metal_stdlib>
using namespace metal;
{pragma}
kernel void Execute(device float* values [[buffer(0)]],
                    uint id [[thread_position_in_grid]]) {{
    values[id] = values[id] * values[id + 1] + dot(float2(values[id]), float2(values[id + 2]));
}}
"""


def metal_compiler_available():
    if shutil.which("xcrun") is None:
        return False
    probe = subprocess.run(
        ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
    )
    return probe.returncode == 0


@pytest.mark.skipif(not metal_compiler_available(), reason="needs the Metal compiler")
def test_the_metal_compiler_output_distinguishes_contraction(tmp_path):
    artifacts = tmp_path / "artifacts"
    (artifacts / "metal").mkdir(parents=True)
    shaders = {}
    for name, pragma, dot in (
        ("fused", "", "dot(float2(values[id]), float2(values[id + 2]))"),
        ("exact", "#pragma clang fp contract(off)", "values[id + 2]"),
    ):
        source = METAL_KERNEL.format(pragma=pragma).replace(
            "dot(float2(values[id]), float2(values[id + 2]))", dot
        )
        path = artifacts / "metal" / f"{name}.metal"
        path.write_text(source)
        shaders[name * 8] = {
            "artifact": f"metal/{name}.metal",
            "artifactSha256": EXACTNESS.sha256_file(path),
            "status": "translated",
        }
    (artifacts / "metal" / "manifest.json").write_text(json.dumps({"shaders": shaders}))

    report = EXACTNESS.analyze("metal", artifacts)

    assert report["status"] == "failed"
    assert report["artifacts"]["exact" * 8]["status"] == "exact"
    assert report["artifacts"]["exact" * 8]["floatOperations"] >= 2
    fused = report["artifacts"]["fused" * 8]
    assert fused["status"] == "contractible"
    assert fused["fusedCalls"] >= 1
    assert report["errors"] == [f"{'fused' * 8}: contractible"]
