CUDA Backend
============

The CUDA backend covers CrossGL-to-CUDA generation and CUDA source import for
compute-oriented workflows. It is selected through the ``cuda`` target and
source aliases.

Pipeline
--------

CrossGL output generation is implemented by
``crosstl.translator.codegen.cuda_codegen.CudaCodeGen``. The generator walks the
shared translator AST, emits CUDA runtime includes, lowers CrossGL compute
builtins to CUDA grid and thread identifiers, maps CrossGL scalar/vector types
to CUDA types, and inserts helper functions for resource queries and vector
arithmetic when required.

Reverse translation uses ``crosstl.backend.CUDA.CudaLexer.CudaLexer`` and
``crosstl.backend.CUDA.CudaParser.CudaParser`` to parse CUDA source into the
CUDA backend AST. ``crosstl.backend.CUDA.CudaCrossGLCodeGen`` then serializes
that AST back into CrossGL syntax.

Supported Surface
-----------------

The backend focuses on CUDA compute code:

* CUDA kernels declared with ``__global__`` plus device and host functions
* kernel launch expressions using ``<<<grid, block, ...>>>``
* CUDA builtins such as ``threadIdx``, ``blockIdx``, ``blockDim``, and
  ``gridDim`` mapped to CrossGL compute identifiers
* shared and constant memory declarations
* CUDA vector constructors, atomics, synchronization functions, pointers, and
  array declarators
* runtime calls surfaced as comments during reverse translation when there is no
  direct CrossGL statement equivalent

Implementation Notes
--------------------

CUDA codegen is visitor-based. Keep output-only behavior in
``CudaCodeGen`` and import-only behavior in the ``crosstl.backend.CUDA`` parser
and reverse converter. Shared resource-query, resource-diagnostic, and vector
arithmetic helpers live under ``crosstl.translator.codegen`` and should be used
when the rule also applies to HIP or another compute target.

When extending this backend, add focused tests under the CUDA backend and
translator codegen test folders. Document new public behavior here and keep
API-level details in class or method docstrings.
