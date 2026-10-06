GLSL Backend
============

The GLSL backend covers OpenGL-style shader generation and GLSL source import.
It is selected through the ``glsl``, ``opengl``, or ``ogl`` aliases in the
translator registry.

Pipeline
--------

CrossGL output generation is implemented by
``crosstl.translator.codegen.GLSL_codegen.GLSLCodeGen``. It emits GLSL from the
shared translator AST, including version/preprocessor lines, uniform blocks,
resource declarations, stage inputs and outputs, and ``main`` functions for
stage entry points.

Reverse translation uses ``crosstl.backend.GLSL.OpenglLexer.GLSLLexer`` and
``crosstl.backend.GLSL.OpenglParser.GLSLParser`` to parse GLSL source into the
OpenGL backend AST. ``crosstl.backend.GLSL.openglCrossglCodegen`` converts that
tree back into CrossGL syntax.

Supported Surface
-----------------

The backend is the primary path for OpenGL and Vulkan-style GLSL authoring:

* shader stages including vertex, fragment, compute, geometry, tessellation,
  mesh/task, and ray tracing-style qualifiers where represented in the AST
* ``#version`` and other preprocessor directives, precision statements, layout
  qualifiers, interface blocks, and uniform blocks
* sampler, image, texture, and atomic functions mapped to GLSL intrinsics
* built-in variables such as ``gl_Position``, ``gl_FragDepth``,
  ``gl_GlobalInvocationID``, and related compute identifiers
* GLSL cbuffer lowering to ``layout(std140, binding = N) uniform`` blocks

Vector equality and inequality use ``equal`` and ``notEqual``, preserving one
Boolean per component, including inside ``any`` and ``all`` or another vector
comparison. Scalar comparisons retain scalar operators. Implicit Boolean-vector
reduction to a scalar remains an error.

Half-Precision Values
---------------------

Scalar and two-, three-, and four-component half values use float32 GLSL
storage. Numeric narrowing rounds to binary16, nearest with ties to even,
before widening back to that storage representation. Casts, initializers,
assignments, helper arguments and returns retain this rounding boundary;
half-valued arithmetic is rounded before a later float conversion.

The lowering uses integer bit operations to preserve subnormal half values,
signed zero, overflow and infinity independently of packing-intrinsic
subnormal handling. NaNs remain NaNs; payload preservation is not promised.
Arguments are evaluated once. Double-to-half conversions, unsupported profiles,
unfolded global initializers, and updates whose evaluation order cannot be
preserved produce diagnostics. Finite scalar/vector literal constants are
rounded during generation and remain valid GLSL constant expressions.
This does not implement packed 16-bit buffers, bfloat16 semantics, or native
half-matrix arithmetic. Hosts must still use the reflected physical layout.

Required Linux CI executes all finite half values, both neighbors of every
finite half rounding midpoint, exceptional values and seeded float32 samples.
Separate checks cover assignment, helpers, structures, vectors and compound
updates. macOS CI runs the same data through original and translated Metal.
Retained artifacts include source, modules, input bytes, reference bytes,
output bytes and guards. These are numeric-conversion tests, not full MLX
runtime coverage.

Integer-To-Bfloat Conversion
----------------------------

Scalar signed and unsigned 32-bit and 64-bit integer conversions round directly
to bfloat precision, nearest with ties to even. The generated helper operates
on integer magnitudes and remainder bits before constructing the widened
float32 result. It does not introduce a float32 intermediate that could round
twice. Explicit source conversions through float32 remain distinct.

The 64-bit path requires ``GL_ARB_gpu_shader_int64`` and is emitted only when
used. It handles signed minimum values without signed negation overflow, and
retains rounding carry at the unsigned maximum. Helper arguments are evaluated
once; assignment, return and compound-assignment conversion boundaries are
preserved. Unsupported double and wide-integer vector conversions still produce
diagnostics. This support does not change the reflected buffer layout.

Linux native checks cover integer rounding midpoints, extrema and seeded input
values, with exact payloads and output guards. Original and generated Metal
provide separate controls. Mixed compound assignments also exercise locals,
members, indexed arrays and resource buffers with in-range results.

Software Subgroup Helpers
-------------------------

Explicit-width software subgroup operations can be reached through resolved
helper chains. Each call retains its source overload identity. Immutable scalar
arguments are propagated only when every reachable caller proves them uniform
across the workgroup. Supported helper branches, bounded loops and early exits
use those facts; output parameters, references, mutation and unknown calls do
not establish uniformity. Entry-point calls in uniform branches are supported;
lane-dependent branches, recursive call graphs and unproven early exits remain
errors. Const qualification alone does not prove invocation uniformity.
Raw hardware subgroup builtins are rejected in software mode; annotated
subgroup inputs use the configured logical width instead.

Generated collectives pair execution barriers with explicit shared-memory
ordering. This preserves scratch reuse across conditional helper calls on
the tested Mesa runtime. The required native vote gate covers repeated calls,
three wrapper levels, distinct even/odd-workgroup predicates, multiple logical
subgroups and unchanged input/output guards. These are collective semantics
checks, not complete MLX reduction or host-runtime coverage.

The required native helper-argument gate checks nested conditional calls,
workgroup-dependent bounds, repeated reductions and uniform loops on Windows,
Linux and macOS, with original Metal controls on macOS. Input preservation,
evaluation counters and output guards remain part of the numerical checks.

Workgroup pointer bounds use the configured software subgroup width when
proving logical subgroup indices and counts. Helper guards can narrow known
invocation components in branches and in the selected arm of a conditional
expression. Bounds include each caller's backing extent and pointer offset;
mutation, escaped references and unproven accesses remain diagnostic. These
proofs do not clamp indices or evaluate an inactive conditional read.
The native helper gate also checks subgroup-indexed and guarded shared reads
through nested helpers and offset views at workgroup widths 32, 64 and 128,
including original Metal controls and exact output guards.

Scalar float32, int32 and uint32 products combine adjacent lane pairs with
strides 1, 2, 4, 8 and 16. Integer products retain 32-bit wraparound; floating
products round at each multiplication. The native product gate retains
original Metal controls, finite rounding and overflow/underflow cases, NaNs,
infinities, signed zeros, repeated calls and guards. NaN payloads are not
required to match. This software order does not imply equivalence with every
possible native hardware reduction order or denormal mode.
Narrow, vector and wider products remain diagnostic; source-precision rounding
must be established separately from scratch storage width.

Pure subgroup-vote guards around scalar reduction returns are lowered to
converged collectives followed by value selection. A subgroup-wide vote is not
treated as workgroup-uniform; multiple logical subgroups can select different
results. Only safe by-value operands are speculated, and unproven effects or
participation remain diagnostic. Original Metal controls use fast math disabled,
matching the generated native runtime's NaN-sensitive compilation contract.

Uniform bounds-check exits also retain immutable local facts and individual
invocation components whose workgroup extent is one. Mutation, reference
escape, shadowing and unknown calls invalidate those facts. A uniform ``y``
component does not make ``x`` or the whole invocation vector uniform.

Native Attention Fixture
------------------------

The pinned MLX attention row-dot fixture translates all declared input types
and head dimensions with explicit software subgroups of width 32. Required
Linux CI validates GLSL with glslang and executes it through the OpenGL runtime;
it retains generated sources, validation modules, uploads and output readbacks.
The same cases have DirectX and original/generated Metal controls.

The fixture uses source-qualified index-range assertions derived from its
concrete buffer sizes. It rejects shapes outside signed 32-bit index bounds.
Because these GLSL entries use float32 storage, the host fixture expands
already-quantized half and bfloat16 values before upload. Source input bytes
remain available separately, and packing tests cover signed zero, fractional
values and small magnitudes. The uniform scalar uses its checked std140 layout.
This is bounded kernel execution evidence, not full MLX host integration or
unrestricted 64-bit buffer indexing.

Implementation Notes
--------------------

GLSL differs from HLSL and Metal because stage entry points lower to ``main``
and stage structs may need to be flattened into global ``in`` and ``out``
declarations. Keep flattening behavior near ``GLSLCodeGen`` stage helpers, and
use the shared codegen utilities only for backend-neutral operations such as
array declarators, resource array hints, and stage-name normalization.

When adding syntax support, update both source import and target generation when
the behavior is intended to be bidirectional. If support is intentionally
one-way, document that limitation here.
