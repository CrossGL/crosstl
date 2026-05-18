HIP Backend
===========

The HIP backend covers CrossGL-to-HIP generation and HIP source import for
AMD-oriented compute workflows. It is selected through the ``hip`` target and
source aliases.

Pipeline
--------

CrossGL output generation is implemented by
``crosstl.translator.codegen.hip_codegen.HipCodeGen``. The generator walks the
shared translator AST, emits HIP runtime includes, lowers CrossGL compute
builtins to HIP/CUDA-compatible grid and thread identifiers, maps CrossGL
scalar/vector/resource types to HIP types, and adds helper functions for
resource access and vector arithmetic when needed.

Reverse translation uses ``crosstl.backend.HIP.HipLexer.HipLexer`` and
``crosstl.backend.HIP.HipParser.HipParser`` to parse HIP source into the HIP
backend AST. ``crosstl.backend.HIP.HipCrossGLCodeGen`` then serializes that AST
back into CrossGL syntax.

Supported Surface
-----------------

The backend focuses on HIP compute code:

* HIP kernels declared with ``__global__`` plus device and host functions
* kernel launch expressions using ``<<<grid, block, ...>>>``
* HIP and CUDA-compatible builtins such as ``threadIdx``, ``blockIdx``,
  ``hipThreadIdx_x``, ``hipBlockIdx_x``, and synchronization functions
* shared, constant, and managed memory qualifiers
* HIP vector types, atomics, texture/surface handles, pointers, arrays, and
  initializer lists
* selected runtime calls surfaced as comments during reverse translation when no
  direct CrossGL statement exists

Implementation Notes
--------------------

HIP mirrors much of the CUDA backend but keeps its own lexer/parser and runtime
call handling. Prefer shared translator utilities for backend-neutral resource
queries, diagnostics, and vector arithmetic, but keep HIP runtime naming and
surface-object behavior local to ``HipCodeGen`` or ``HipToCrossGLConverter``.

When extending this backend, add focused tests under the HIP backend and
translator codegen test folders. If behavior intentionally diverges from CUDA,
document that difference here.
