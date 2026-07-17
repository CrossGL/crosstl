DirectX and HLSL Backend
========================

The DirectX backend covers both CrossGL-to-HLSL generation and HLSL-to-CrossGL
import. It is selected through the ``directx``, ``hlsl``, or ``dx`` target and
source aliases, depending on whether the caller is emitting a target shader or
loading existing HLSL.

For project planning and target selection, ``dx11``, ``dx12``, ``d3d11``, and
``d3d12`` are accepted as target aliases for the same HLSL emitter. They
represent Direct3D deployment profiles, not separate DXBC or DXIL binary
artifact generators. Use the generated ``.hlsl`` output with the appropriate
Direct3D compiler/toolchain for the final runtime profile.

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
* Direct3D 11 and Direct3D 12 target profile aliases that resolve to portable
  HLSL output
* HLSL semantics such as ``SV_POSITION``, ``SV_Target``, ``SV_VertexID``, and
  resource-related system values
* constant buffers, global resources, samplers, comparison samplers, texture
  operations, image atomics, and resource arrays
* preprocessor directives and include handling through the HLSL preprocessor
* HLSL-specific resource validation for duplicate constant-buffer names and
  resource/member shadowing

16-bit Type Semantics
---------------------

Exact source ``half``, ``short``, and ``ushort`` types map to the native HLSL
types ``float16_t``, ``int16_t``, and ``uint16_t``, respectively. Generated
target metadata restricts artifacts that use these types to DirectX 12 and
shader model 6.2, and enables native 16-bit types with
``-enable-16bit-types``.

Explicit DirectX 11 materialization of an exact 16-bit type fails with an
actionable diagnostic instead of silently substituting a minimum-precision
type. Explicit ``min16float``, ``min16int``, and ``min16uint`` types retain
their HLSL minimum-precision semantics.

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

Keep profile-specific API packaging, root signature generation, bytecode
container output, and shader compiler invocation outside ``HLSLCodeGen`` until
the project has an explicit target-profile pipeline for those artifacts.
