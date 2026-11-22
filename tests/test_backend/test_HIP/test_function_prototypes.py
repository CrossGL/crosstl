"""HIP function prototypes import as nothing: CrossGL declares each function once, at its definition.

The HIP importer wrote a body-less prototype such as ``__device__ float twice(float x);`` as a CrossGL function with
an empty body, so the program defined ``twice`` twice and hipcc rejected the round trip with ``redefinition of
'twice'``. A prototype now adds nothing to the CrossGL; its default arguments move to the definition, and the
generators declare a function ahead of an earlier caller where the target needs it.
"""

from __future__ import annotations

import re

import crosstl
from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser


def _crossgl(source):
    program = HipParser(HipLexer(source).tokenize()).parse()
    return HipToCrossGLConverter().generate(program)


def test_a_device_prototype_adds_no_function():
    crossgl = _crossgl("""
__device__ float twice(float x);
__global__ void probe(float* outp) { outp[0] = twice(outp[0]); }
__device__ float twice(float x) { return 2.0f * x; }
""")

    assert len(re.findall(r"f32 twice\(f32 x\)", crossgl)) == 1
    assert "return (2.0f * x);" in crossgl


def test_a_kernel_prototype_adds_no_entry_point():
    crossgl = _crossgl("""
__global__ void probe(float* outp);
__global__ void probe(float* outp) { outp[0] = 1.0f; }
""")

    assert crossgl.count("fn probe(") == 1


def test_an_empty_definition_is_kept():
    crossgl = _crossgl("""
__device__ void nop() {}
__global__ void probe(float* outp) { nop(); outp[0] = 1.0f; }
""")

    assert "void nop() {" in crossgl


def test_the_definition_after_its_caller_reaches_every_target(tmp_path):
    path = tmp_path / "kernel.hip"
    path.write_text(
        """
#include <hip/hip_runtime.h>
__device__ float twice(float x);
__global__ void probe(float* outp) { outp[0] = twice(outp[0]); }
__device__ float twice(float x) { return 2.0f * x; }
""",
        encoding="utf-8",
    )

    for backend in ("hip", "cuda", "metal", "directx"):
        output = crosstl.translate(str(path), backend=backend, format_output=False)
        assert len(re.findall(r"\btwice\s*\([^;{]*\)\s*\{", output)) == 1, backend
