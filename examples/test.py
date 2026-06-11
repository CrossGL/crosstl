#!/usr/bin/env python3
"""CrossGL example conversion smoke test."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crosstl

EXAMPLES_ROOT = Path(__file__).resolve().parent
KNOWN_FAILURE_BUDGET = 2
MIN_SUCCESS_RATE = 90.0
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
EXAMPLE_BACKEND_SKIPS = {
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
        "cross_platform",
        "UniversalPBRShader",
        "wgsl",
    ): (
        "WGSL codegen intentionally rejects unsplit combined sampler2D resources "
        "until texture/sampler binding splitting is available for this example."
    ),
    (
        "gpu_computing",
        "MatrixMultiplication",
        "webgl",
    ): (
        "WebGL codegen intentionally rejects compute stages; use WebGPU/WGSL for compute."
    ),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path for a machine-readable test summary JSON file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional output directory for generated shader files. Defaults to "
            "examples/output."
        ),
    )
    return parser.parse_args(argv)


def stable_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def build_summary(
    total_tests,
    successful_tests,
    failed_tests,
    skipped_tests=None,
    consistency_summary=None,
):
    skipped_tests = skipped_tests or []
    success_rate = (successful_tests / total_tests) * 100 if total_tests else 0.0
    return {
        "schema_version": 1,
        "total": total_tests,
        "successful": successful_tests,
        "failed": len(failed_tests),
        "skipped": len(skipped_tests),
        "success_rate": round(success_rate, 1),
        "known_failure_budget": KNOWN_FAILURE_BUDGET,
        "minimum_success_rate": MIN_SUCCESS_RATE,
        "within_regression_budget": (
            len(failed_tests) <= KNOWN_FAILURE_BUDGET
            and round(success_rate, 1) >= MIN_SUCCESS_RATE
        ),
        "failures": [
            {
                "example": example,
                "backend": backend,
                "error": error,
            }
            for example, backend, error in failed_tests
        ],
        "skips": [
            {
                "example": example,
                "backend": backend,
                "reason": reason,
            }
            for example, backend, reason in skipped_tests
        ],
        "consistency": consistency_summary or {},
    }


def write_summary_json(summary, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(summary), encoding="utf-8")


def display_path(path):
    try:
        return path.relative_to(EXAMPLES_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def validate_generated_output(output_file):
    if not output_file.exists() or output_file.stat().st_size <= 100:
        return "Output file too small"

    output_text = output_file.read_text(encoding="utf-8", errors="replace")
    for marker in FORBIDDEN_OUTPUT_MARKERS:
        if marker in output_text:
            return f"Output contains internal AST representation: {marker}"
    return None


def main(argv=None):
    """Test comprehensive translation functionality across all example categories."""
    args = parse_args(argv)

    print("[CROSSGL] CrossGL Comprehensive Translation Test")
    print("=" * 60)

    output_root = args.output_root or EXAMPLES_ROOT / "output"
    for backend in BACKENDS:
        (output_root / backend).mkdir(parents=True, exist_ok=True)

    total_tests = 0
    successful_tests = 0
    failed_tests = []
    skipped_tests = []

    for category, examples in EXAMPLES_BY_CATEGORY.items():
        print(f"\n[TESTING] {category.upper()} examples:")
        print("-" * 40)

        for example in examples:
            example_path = EXAMPLES_ROOT / category / example
            example_name = Path(example).stem

            if not example_path.exists():
                print(f"[WARNING] Skipping {example} (not found)")
                continue

            print(f"\n[TRANSLATING] {example_name}:")

            # Get compatible backends for this category
            compatible_backends = BACKEND_COMPATIBILITY.get(
                category, list(BACKENDS.keys())
            )

            for backend in compatible_backends:
                if backend not in BACKENDS:
                    continue

                skip_reason = EXAMPLE_BACKEND_SKIPS.get(
                    (category, example_name, backend)
                )
                if skip_reason:
                    print(f"  [SKIPPED] {backend:8} -> {skip_reason}")
                    skipped_tests.append((example_name, backend, skip_reason))
                    continue

                total_tests += 1
                try:
                    backend_output_dir = output_root / backend / category
                    backend_output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = (
                        backend_output_dir / f"{example_name}{BACKENDS[backend]}"
                    )

                    result = crosstl.translate(
                        str(example_path),
                        backend=backend,
                        save_shader=str(output_file),
                    )

                    validation_error = validate_generated_output(output_file)
                    if validation_error is None:
                        print(f"  [SUCCESS] {backend:8} -> {display_path(output_file)}")
                        successful_tests += 1
                    else:
                        print(f"  [WARNING] {backend:8} -> {validation_error}")
                        failed_tests.append((example_name, backend, validation_error))

                except Exception as e:
                    print(f"  [ERROR] {backend:8} -> Error: {str(e)[:50]}...")
                    failed_tests.append((example_name, backend, str(e)))

    print("\n" + "=" * 60)
    print("[SUMMARY] TRANSLATION TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Skipped: {len(skipped_tests)}")
    summary = build_summary(total_tests, successful_tests, failed_tests, skipped_tests)
    print(f"Success rate: {summary['success_rate']:.1f}%")

    if failed_tests:
        print(f"\n[FAILED] Failed translations:")
        for example, backend, error in failed_tests:
            print(f"  - {example} -> {backend}: {error[:60]}...")

    print(f"\n[TESTING] Testing cross-backend consistency...")
    consistency_summary = test_cross_backend_consistency()
    summary = build_summary(
        total_tests,
        successful_tests,
        failed_tests,
        skipped_tests,
        consistency_summary,
    )
    if args.summary_json:
        write_summary_json(summary, args.summary_json)
        print(f"[SUMMARY_JSON] {args.summary_json}")

    print(f"\n[COMPLETE] Translation test complete!")
    print(
        f"[OUTPUT] Check output/ directory for organized results by backend and category."
    )
    if not summary["within_regression_budget"]:
        print(
            "[ERROR] Example regression budget exceeded: "
            f"failed={summary['failed']}, success_rate={summary['success_rate']:.1f}%",
            file=sys.stderr,
        )
        return 1
    return 0


def test_cross_backend_consistency():
    """Test that the same shader produces valid output across multiple backends."""
    test_shader = EXAMPLES_ROOT / "graphics" / "SimpleShader.cgl"
    test_shader_display = display_path(test_shader)
    if not test_shader.exists():
        print("[WARNING] SimpleShader.cgl not found for consistency test")
        return {
            "shader": test_shader_display,
            "backends": [],
            "outputs": {},
            "successful": 0,
            "total": 0,
        }

    consistency_backends = ["metal", "directx", "opengl", "webgl", "wgsl", "vulkan"]
    outputs = {}

    for backend in consistency_backends:
        try:
            result = crosstl.translate(str(test_shader), backend=backend)
            outputs[backend] = len(result) if result else 0
        except Exception:
            outputs[backend] = 0

    print(f"  Output sizes: {outputs}")
    non_zero_outputs = [k for k, v in outputs.items() if v > 100]
    print(
        f"  [SUCCESS] {len(non_zero_outputs)}/{len(consistency_backends)} backends produced substantial output"
    )
    return {
        "shader": test_shader_display,
        "backends": consistency_backends,
        "outputs": outputs,
        "successful": len(non_zero_outputs),
        "total": len(consistency_backends),
    }


if __name__ == "__main__":
    raise SystemExit(main())
