DirectX and HLSL Backend
========================

The DirectX backend covers both CrossGL-to-HLSL generation and HLSL-to-CrossGL
import. It is selected through the ``directx``, ``hlsl``, or ``dx`` target and
source aliases, depending on whether the caller is emitting a target shader or
loading existing HLSL.

Pipeline
--------

CrossGL output generation starts from the shared translator AST and is handled
by ``crosstl.translator.codegen.directx_codegen.HLSLCodeGen``. The generator
normalizes stage names, validates resource declarations, maps CrossGL scalar,
vector, matrix, sampler, texture, image, and buffer types to HLSL, and emits
stage entry points such as ``VSMain``, ``PSMain``, and ``CSMain``.

Reverse translation uses ``crosstl.backend.DirectX.DirectxLexer.HLSLLexer`` and
``crosstl.backend.DirectX.DirectxParser.HLSLParser`` to parse HLSL into the
DirectX backend AST. ``crosstl.backend.DirectX.DirectxCrossGLCodeGen`` then
serializes that AST back into CrossGL syntax.

Supported Surface
-----------------

The backend is the primary path for DirectX shader integration:

* shader stages including vertex, fragment/pixel, compute, geometry,
  tessellation, mesh, amplification, and ray tracing stages
* HLSL semantics such as ``SV_POSITION``, ``SV_Target``, ``SV_VertexID``, and
  resource-related system values
* constant buffers, global resources, samplers, comparison samplers, texture
  operations, image atomics, and resource arrays
* preprocessor directives and include handling through the HLSL preprocessor
* HLSL-specific resource validation for duplicate constant-buffer names and
  resource/member shadowing

Implementation Notes
--------------------

The generator keeps per-function resource context while rendering a function so
that texture, sampler, and implicit sampler arguments can be emitted in the
shape expected by HLSL. Stage helpers live in ``stage_utils`` and resource array
size inference lives in ``resource_arrays``; keep backend-specific behavior in
the DirectX generator unless the rule is shared by multiple targets.

When extending this backend, add focused tests under the DirectX translator and
backend test folders. Prefer documenting new public behavior on this page and
API details in the relevant class or function docstrings.
