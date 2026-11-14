#!/usr/bin/env python3
"""Translate every example to every backend its category supports.

A translation passes when it writes non-trivial output with no leaked AST node
names, or, for a pair listed in EXPECTED_REJECTIONS, when the backend refuses
it. Any other outcome fails the run, including a listed pair that now
translates: delete its entry so the pair is tested from then on.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crosstl
from crosstl.translator.codegen.glsl_stage_entries import OpenGLRepeatedStageEntryError

EXAMPLES_ROOT = Path(__file__).resolve().parent
EXAMPLES_BY_CATEGORY = {
    "graphics": ["SimpleShader.cgl", "PerlinNoise.cgl", "ComplexShader.cgl"],
    "advanced": ["ArrayTest.cgl", "GenericPatternMatching.cgl"],
    "compute": ["ParticleSimulation.cgl"],
    "cross_platform": ["UniversalPBRShader.cgl"],
    "gpu_computing": ["MatrixMultiplication.cgl"],
}
BACKENDS = {
    "metal": ".metal",
    "directx": ".hlsl",
    "opengl": ".glsl",
    "webgl": ".webgl.glsl",
    "wgsl": ".wgsl",
    "vulkan": ".spvasm",
    "rust": ".rs",
    "mojo": ".mojo",
    "cuda": ".cu",
    "hip": ".hip",
    "slang": ".slang",
}
FORBIDDEN_OUTPUT_MARKERS = (
    "AssignmentNode(",
    "BinaryOpNode(",
    "FunctionNode(",
    "IdentifierNode(",
    "MemberAccessNode(",
    "VariableNode(",
)
BACKEND_COMPATIBILITY = {
    "graphics": [
        "metal",
        "directx",
        "opengl",
        "webgl",
        "wgsl",
        "vulkan",
        "rust",
        "mojo",
        "cuda",
        "hip",
        "slang",
    ],
    "advanced": [
        "metal",
        "directx",
        "opengl",
        "webgl",
        "wgsl",
        "vulkan",
        "rust",
        "mojo",
        "cuda",
        "hip",
        "slang",
    ],
    "compute": [
        "metal",
        "directx",
        "opengl",
        "webgl",
        "wgsl",
        "vulkan",
        "cuda",
        "hip",
    ],
    "cross_platform": [
        "metal",
        "directx",
        "opengl",
        "webgl",
        "wgsl",
        "vulkan",
        "rust",
        "mojo",
        "slang",
    ],
    "gpu_computing": ["cuda", "hip", "mojo", "rust", "webgl", "wgsl"],
}
EXPECTED_REJECTIONS = {
    (
        "advanced",
        "GenericPatternMatching",
        "vulkan",
    ): (
        "SPIR-V codegen intentionally rejects generic helpers that are not "
        "covered by concrete specializations."
    ),
    (
        "advanced",
        "GenericPatternMatching",
        "cuda",
    ): (
        "CUDA codegen intentionally rejects generic functions until all generic "
        "types are lowered to concrete device declarations."
    ),
    (
        "advanced",
        "GenericPatternMatching",
        "mojo",
    ): (
        "Mojo codegen intentionally rejects generic payload enum specializations "
        "until they can be lowered to concrete payload layouts."
    ),
    (
        "advanced",
        "GenericPatternMatching",
        "slang",
    ): (
        "Slang codegen intentionally rejects generic functions until all generic "
        "types are lowered to concrete shader declarations."
    ),
    (
        "advanced",
        "GenericPatternMatching",
        "wgsl",
    ): "WGSL codegen intentionally rejects generic structs with nested enum members.",
    (
        "compute",
        "ParticleSimulation",
        "webgl",
    ): (
        "WebGL codegen intentionally rejects compute stages; use WebGPU/WGSL for compute."
    ),
    (
        "gpu_computing",
        "MatrixMultiplication",
        "webgl",
    ): (
        "WebGL codegen intentionally rejects compute stages; use WebGPU/WGSL for compute."
    ),
    (
        "gpu_computing",
        "MatrixMultiplication",
        "wgsl",
    ): (
        "WGSL allows workgroup barriers only in uniform control flow (WGSL "
        "section 15.2), and the warp shuffle inside the divergent branch needs one."
    ),
    (
        "gpu_computing",
        "MatrixMultiplication",
        "mojo",
    ): (
        "The Mojo generator emits each compute invocation as its own function "
        "with no warp, so a warp shuffle has no other lane to read and is "
        "refused rather than replaced by a one-lane stand-in."
    ),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Directory for the generated files. Defaults to examples/output.",
    )
    return parser.parse_args(argv)


def validate_generated_output(output_file):
    if not output_file.exists() or output_file.stat().st_size <= 100:
        return "Output file too small"

    output_text = output_file.read_text(encoding="utf-8", errors="replace")
    for marker in FORBIDDEN_OUTPUT_MARKERS:
        if marker in output_text:
            return f"Output contains internal AST representation: {marker}"
    return None


def scheduled_cases():
    for category, examples in EXAMPLES_BY_CATEGORY.items():
        for example in examples:
            for backend in BACKEND_COMPATIBILITY[category]:
                yield category, Path(example).stem, backend


def translate_example(source, backend, output_file):
    """Translate ``source`` to ``output_file`` and return the files written.

    A GLSL compilation unit holds one entry point per stage (GLSL 4.60 section 6.1), so GLSL refuses an example
    with two compute kernels as a whole. Each kernel is then written to its own file, ``<name>.<entry><ext>``.
    """
    try:
        crosstl.translate(str(source), backend=backend, save_shader=str(output_file))
        return [output_file]
    except OpenGLRepeatedStageEntryError as refusal:
        entries = [entry for _stage, entry in refusal.stage_entries]
    written = []
    for entry, code in crosstl.translate_entries(str(source), backend, entries).items():
        entry_file = output_file.with_name(f"{source.stem}.{entry}{BACKENDS[backend]}")
        entry_file.write_text(code, encoding="utf-8")
        written.append(entry_file)
    return written


def run_case(category, name, backend, output_root):
    """Return None when the case behaves as expected, else the reason it failed."""
    source = EXAMPLES_ROOT / category / f"{name}.cgl"
    output_file = output_root / backend / category / f"{name}{BACKENDS[backend]}"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rejected = (category, name, backend) in EXPECTED_REJECTIONS
    try:
        written = translate_example(source, backend, output_file)
    except Exception as exc:
        return None if rejected else f"{type(exc).__name__}: {exc}"
    if rejected:
        return "translates now; delete its EXPECTED_REJECTIONS entry"
    for path in written:
        error = validate_generated_output(path)
        if error:
            return error
    return None


def main(argv=None):
    args = parse_args(argv)
    output_root = args.output_root or EXAMPLES_ROOT / "output"
    failures = []
    rejections = 0
    total = 0
    for category, name, backend in scheduled_cases():
        total += 1
        error = run_case(category, name, backend, output_root)
        if error:
            failures.append((category, name, backend, error))
            status = "FAILED"
        elif (category, name, backend) in EXPECTED_REJECTIONS:
            rejections += 1
            status = "rejected"
        else:
            status = "ok"
        print(f"{status:8} {backend:8} {category}/{name}")

    passed = total - rejections - len(failures)
    print(
        f"\n{total} translations: {passed} passed, {rejections} rejected as "
        f"expected, {len(failures)} failed"
    )
    for category, name, backend, error in failures:
        print(f"{category}/{name} -> {backend}: {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
