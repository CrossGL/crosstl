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

Scalar and vector binary32-to-binary16 conversions use explicit round-to-nearest,
ties-to-even arithmetic on the representation bits. This preserves source
rounding at casts, initializers, stores, helper arguments and returns, including
compound assignments computed in binary32. The final bit reinterpretation keeps
native 16-bit storage; an ordinary HLSL narrowing cast would truncate instead.
Integer operands are converted through binary32, which exactly represents every
integer within the finite binary16 range. Binary16-to-binary32 widening decodes
the payload explicitly to retain subnormal values. Minimum-precision types and
explicit bit reinterpretations remain separate contracts.

The conversion preserves signed zeros, gradual underflow, overflow to infinity
and NaN classification, but does not promise NaN payload preservation for numeric
casts. Vector arguments are evaluated once. Double-to-half conversions fail
with a structured diagnostic rather than introducing double rounding. Native
half matrix arithmetic is outside this scalar/vector conversion contract.
Required Windows execution checks the same 258,052-value corpus as original and
generated Metal, with midpoint neighbors, both signs and boundary values, plus
conversion sites and side-effect counts. Compiler checks alone do not establish
numerical correctness.

Software Subgroup Reductions
----------------------------

The explicit ``software_subgroup_width=32`` target option lowers scalar
``WaveActiveSum``, ``WaveActiveMin`` and ``WaveActiveMax`` through groupshared
storage. It also supports ``WaveShuffleDown`` with
``relative_wave_shuffle_out_of_range="self"``. Each logical group contains 32
consecutive local invocations, independently of the device's physical wave
width. Multiple logical groups share an allocation but never read each other's
values. Every invocation participates in both barriers around a collective;
unproven divergent control flow remains a structured translation error.

The entry captures ``SV_GroupIndex`` in private per-invocation storage before
executing source statements. All dependent helpers use that captured identity,
including helpers with invocation-like parameter annotations. If the source
does not expose a complete local index, a collision-safe internal parameter is
added. Source-visible scalar or vector IDs retain their original values and
ordinary assignment semantics; modifying them cannot change subgroup partners
or shared-memory offsets.

Convergence analysis follows immutable integer and Boolean locals in lexical
scope, including nested counted-loop bounds derived from constant parameters.
Primitive constructors and unshadowed ``min``, ``max`` and ``clamp`` calls preserve
uniformity only when their operands are workgroup-uniform. Writes, reference
aliases, address escapes and possibly mutating calls invalidate these facts.
Lane-dependent bounds and unproven early exits remain translation errors.

Invocation-vector components are proven separately. A global or local ID
component is workgroup-uniform when its corresponding declared workgroup
dimension is one. For example, global ``y`` can select an entire ``(32, 1, 1)``
workgroup, but not a ``(32, 4, 1)`` workgroup. This fact does not make the whole
vector uniform and is not inferred from helper-parameter annotations. Mutation,
aliasing and lexical shadowing invalidate the component proof. Windows tests
exercise whole-workgroup early returns and row-dependent loops across multiple
three-dimensional dispatch shapes while checking inactive outputs and guards.

Payloads are limited to 32-bit ``float``, ``int`` and ``uint`` scalars. Floating
sums use a precise, increasing-lane-order fold; different native reduction
orders can round differently. Integer sums retain 32-bit wraparound. Floating
minimum and maximum ignore NaNs when a numeric lane exists and return NaN when
all lanes are NaN. Minimum chooses negative zero and maximum chooses positive
zero when both signs occur. NaN payloads are not a portable guarantee, and
floating arithmetic retains the selected DirectX toolchain's denormal behavior.

The Windows execution gate tests mixed reductions and shuffles, scratch reuse,
three workgroups with four logical subgroups each, integer overflow,
cancellation, NaNs, infinities and signed zeros. It retains compiler output,
shader and module hashes, input words and full output readbacks. This explicit
software path does not replace native-wave generation when the option is absent.

A separate required Windows gate translates eight pinned MLX gated-delta
backward configurations with software subgroups and dispatches the generated
DXIL using the upstream ``(32, 4, 1)`` workgroup shape. All six gradients are
checked against the same independent reference and tolerances used by the
Metal gate. Evidence includes all output words, guard regions, compiler output,
resource bindings and module identities. Read-only input bindings are uploaded
but not read back by the DirectX driver. These are direct kernel tests, not a
claim of complete MLX host integration or upstream-suite coverage.

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
