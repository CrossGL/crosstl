Metal Backend
=============

The Metal backend covers CrossGL-to-Metal Shading Language generation and
Metal-to-CrossGL source import. It is selected through the ``metal`` target and
source aliases.

Pipeline
--------

CrossGL output generation is implemented by
``crosstl.translator.codegen.metal_codegen.MetalCodeGen``. It emits Metal
Shading Language from the shared translator AST, adds ``metal_stdlib`` when
needed, maps CrossGL resources to Metal texture/sampler/buffer types, and
renders stage entry functions using Metal attributes.

Source ``round`` calls retain Metal's halfway-away-from-zero rule, rather than
inheriting a target's nearest-even or implementation-selected halfway rule.
Float and half scalars and vectors of up to four lanes use a portable binary32
helper, with the source return type restored before surrounding expressions.
Materialized bfloat wrappers keep their explicit narrow return. Arguments are
evaluated once. The helper uses positive zero below magnitude one half, matching
the original Metal native controls with and without fast math; infinities are
unchanged and NaNs remain NaNs. Integer-bit rounding avoids an extra floating
addition near the binary32 integer-precision boundary.

Saved CrossGL retains this source-specific helper. User-defined ``round``
overloads, Metal ``rint`` and HLSL source ``round`` keep their own semantics.
Unsupported source result types and runtime calls in global initializers emit
``project.translate.metal-round-unsupported`` instead of a target artifact.
Native regression checks cover halfway neighborhoods, all binary32 exponents,
scalar/vector calls and all half and bfloat bit patterns. They supplement,
rather than replace, numerical tests of complete translated kernels.

Reverse translation uses ``crosstl.backend.Metal.MetalLexer.MetalLexer`` and
``crosstl.backend.Metal.MetalParser.MetalParser`` to parse MSL into the Metal
backend AST. ``crosstl.backend.Metal.MetalCrossGLCodeGen`` then serializes that
AST back into CrossGL syntax.

Both directions preserve arithmetic expression grouping. In particular,
``a + (b + c)``, ``a * (b * c)`` and nested bitwise expressions retain their
right-hand parentheses instead of being flattened into left-associated
expressions. Floating-point rounding and mixed-width integer promotion make this
distinction observable. Native
regressions use strict compiler arithmetic settings; preserving the source tree
does not override a caller's choice to enable compiler fast-math reassociation.

Supported Surface
-----------------

Scalar ``as_type<T>`` imports preserve the explicit logical bit width.
The ``asfloat``, ``asint`` and ``asuint`` shorthand is used only when
both operands have the same known 32-bit component layout. Equal lane counts
do not justify narrowing a 64-bit result or widening a 16-bit payload.
Saved CrossGL retains the same bitcast contract.

Required native controls on Metal, DirectX and OpenGL cover signed and unsigned
16- and 64-bit payloads, every finite binary16 pattern, 32-bit positive controls,
single evaluation and output guards. OpenGL retains narrow integer source types
despite widened storage and normalizes signedness at the bitcast. For these
16- and 64-bit types, unsupported logical widths or shapes produce diagnostics
rather than numeric casts. Byte-sized explicit bitcasts remain incomplete on
DirectX and OpenGL; see `issue 2184 <https://github.com/CrossGL/crosstl/issues/2184>`_.
These controls do not establish arbitrary vector reshaping or NaN payload
preservation across widened floating-point storage.

The backend focuses on Apple GPU shader integration:

* vertex, fragment, compute, mesh/object/amplification, and ray tracing-style
  stage qualifiers represented in the parser AST
* Metal attributes such as ``[[vertex_id]]``, ``[[position]]``,
  ``[[color(N)]]``, thread and threadgroup identifiers, and resource bindings
* texture, sampler, image, constant-buffer, and threadgroup-style resource
  declarations
* scalar, vector, matrix, packed vector, SIMD vector, and half-precision type
  mappings
* Metal-specific handling for char-like types through ``CharTypeMapper``

Primitive ``using`` and ``typedef`` chains retain their declaration context,
including namespace imports, qualified names, declaration order and local
shadowing. Scalar ``half`` and ``bfloat`` aliases retain their logical types
through helper parameters, return values, constructors and buffer elements.
Read-only qualification is inherited through alias chains. Namespace-local
primitive aliases are resolved at use sites rather than emitted as conflicting
global typedefs. Cyclic, ambiguous, conflicting or missing primitive targets
produce ``project.translate.metal-scalar-alias-unresolved`` diagnostics with
source locations; no target artifact is emitted. Non-resource volatile scalar
aliases are also diagnosed because their value qualification is not yet
represented in the shared source. Pointer, aggregate and template aliases
continue to use their existing resolution paths.

Selecting one project entry retains scalar aliases used by its reference
parameters, reachable helper signatures, local types and module declarations.
Alias dependencies are followed transitively, including fields of retained
structs; unrelated aliases can still be removed. Type dependencies are distinct
from local value names, so a variable shadow does not discard a required type.
The native alias checks exercise device and constant references through packaged
Metal, OpenGL and DirectX dispatches.

Explicit ``auto*`` locals deduce their pointee type before source overload
selection. Resource offsets, address expressions, one-dimensional local arrays,
selected pointer-returning calls and pointer conditionals retain their source
types and read-only qualifiers. Missing or incompatible initializers produce
``project.translate.metal-auto-type-unresolved``. Qualifiers on the pointer
object itself, such as ``device auto* const``, also produce this diagnostic
until distinct pointer-level qualifiers can be represented; they are not moved
onto the pointee.

Deduction does not bypass target pointer restrictions. Non-escaping scalar and
fixed-array local aliases round-trip through Metal; unsupported private aliases
still produce the DirectX or OpenGL private-pointer diagnostic. Required native
controls on all three platforms cover device and threadgroup aliases, nonzero
offsets, single evaluation, unsigned wraparound, input preservation and output
guards. Metal additionally compares the private-alias controls with the original
source.

Conditional resource parameters preserve both their native binding and the
``function_constant(condition)`` annotation during Metal round trips. The
condition is distinct from the numeric id on a global function constant.
Project-configured and runtime-supplied Boolean combinations have native buffer
controls on Metal and OpenGL; DirectX requires concrete project values before
HLSL generation and executes those variants in its Windows job. The OpenGL
runtime loads specialized GLSL compilation results as validated SPIR-V binaries
without changing the packaged source identity.

The execution controls supply every reflected buffer, including conditionally
unused inputs. Omitting inactive bindings is not yet represented by the loader
descriptor. Aggregate constant references have compilation coverage; their
native execution still requires complete aggregate layout reflection and
packing. Neither limitation is bypassed by fabricating scalar metadata.

Helpers may advance or rebase their own constant-address-space pointer values.
These updates are distinct from stores through the pointer: pointee writes
remain unsupported, while offset updates survive the round trip. Native controls
compare original and generated Metal, including unchanged caller pointers.

Constant aggregate pointers use read-only structured resources in CrossGL, not
single-object constant buffers. Metal restores the original ``constant T*``
declaration and binding; HLSL uses an SRV and GLSL uses a read-only storage block.
Pointer-member access retains an explicit dereference, including computed
offsets, without repeating side effects. Constructor factories continue to
access their materialized ``this`` object by value.

Required native controls cover direct and indexed access, pointer arithmetic,
local aliases, helper calls and constructors, with unconditional and specialized
conditional bindings. The homogeneous two-member records use reflected layouts,
unsigned wraparound cases and output guards; Metal also executes the original
source. This does not establish general heterogeneous aggregate packing or
optional omission of inactive bindings.

Unnamed source parameters receive generated identifiers with ``@maybe_unused``
metadata in CrossGL. Synthesized receivers and source-used parameters whose uses
disappear during static-member lowering, constant specialization, type-only
evaluation or side-effect-free discard retain the same metadata. Metal marks
only those declarations unused, preserving strict warnings for genuinely unused
named parameters. Parameter types, arity, defaults
and argument evaluation are retained. Required native controls cover helpers,
constructors, template deduction, tag arguments, overloads and identifier
collisions, including discarded arguments with side effects.
Parameter-use controls also cover lexical shadowing, saved intermediates,
type aliases and retained compute-input bindings. Side-effectful discard
expressions are still evaluated.

Primary struct-template forward declarations retain their parameter signatures
and defaults for selecting a unique partial specialization. They do not invent
a definition for an incomplete primary. Static calls through constructor and
method aliases expose their concrete owner before member lowering; local alias
shadowing, alias chains and constructor initializer lists retain their scopes.
Ambiguous partial matches remain unresolved rather than selecting an arbitrary
body. Native controls cover side effects, unsigned wraparound and output guards
through project packages, with separate saved-intermediate compilation checks.

Instantiated member templates qualify implicit fields and explicit ``this``
receivers before resolving nested calls. Ordinary and template methods share
the same scope-aware lowering, preserving parameter and local shadowing,
const receivers and nested aggregate fields. Indexed aggregate arrays retain
their full receiver expressions, including multidimensional indices and their
side effects. Native controls check mutation, read-only access, single
evaluation, unsigned wraparound and output guards. Unresolved receivers produce
structured diagnostics instead of abandoning the member-lowering pass.
Reference-returning accessors still require identity-preserving lowering;
unsupported aliases are diagnosed rather than copied into temporary values.

Local ``thread auto&`` bindings to proven aggregate accessors capture each
integral argument with its declared parameter type and capture the resulting
storage indices at the binding. Later index changes do not redirect the alias,
and writes through multiple aliases remain visible in their original storage.
Nested value fields, per-iteration bindings, index side effects and narrow
parameter conversions have native controls. Deduced constness is retained.
Reference escapes, unsupported receiver storage and unproven alias uses remain
diagnostics; no value-copy fallback is used.

Explicit struct specializations compare primitive alias arguments by their
resolved type before materializing the primary template. Static member lookup
uses the same owner, including specialization declarations written through an
alias. Declaration context is retained when a local alias shadows an outer
name; const and pointer arguments remain distinct. Regression checks cover
forward declarations, direct translation and saved CrossGL intermediates, with
native value and output-guard checks in the existing platform jobs.

Materialized bfloat math wrappers retain their float computation and explicit
narrow return at each call. Nested calls, widening expressions and inferred
locals therefore observe the source rounding boundary. This applies to a
selected source wrapper, not to bare precise builtins whose result is float.
Native controls cover default, fast and precise wrappers and single evaluation
of arguments. The Metal target identifies whole-expression constructors from
the syntax tree before omitting a redundant expected-type conversion.

Narrow integer struct initializers convert expression carriers back to the declared
``char``, ``uchar``, ``short`` or ``ushort`` storage type. The same rule covers
vectors, aliases, materialized generic fields, nested aggregates and fixed arrays; omitted
fields are zero-initialized. It does not widen aggregate storage or change
arithmetic promotion. Native controls compare original and generated Metal over
all byte input values and 16-bit boundary and truncation cases, checking signedness,
field sizes, zero initialization, single evaluation and guarded outputs.

Member-template deduction retains the primary argument list of materialized
structs, including when an enabled partial specialization supplies the body.
Omitted arguments are expanded from that primary declaration's defaults before
matching pointer types. Named, anonymous, dependent and integral defaults retain
their positions; conflicting pointee types, address spaces and excess arguments
are rejected. Project translation retains concrete struct materialization even
when no free-function template is instantiated. Required native controls exercise
guarded integer writes through these wrappers on generated Metal, HLSL and GLSL,
with original Metal controls.
This does not establish arbitrary partial-specialization ordering or atomic
operations on every target.

Relaxed integer atomic stores retain their writable buffer or threadgroup
storage through ``atomicStore`` in the intermediate source. Metal emits native
``atomic_store_explicit``; HLSL and GLSL use atomic exchange with the old value
discarded. Signed and unsigned controls cover struct fields, pointer offsets,
helper calls, returned void calls, conditional execution and single evaluation
of indices and values. Nested fields and member arrays have separate compilation
coverage; their native-loader layout is not covered by these execution controls.
Other memory orders are diagnosed rather than weakened to relaxed ordering.
Unsupported store types fail translation instead of emitting a no-op.

Required native CI also exercises the unchanged pinned MLX integer
``scatter_axis`` replacement and sum specializations, including duplicate and
negative indices and output guard regions. This kernel-level proof does not
implement MLX's host ``ScatterAxis`` primitive or establish full-suite parity.

Implementation Notes
--------------------

Metal codegen carries more per-function state than the other shader backends
because cbuffer dependencies and global resource dependencies affect function
signatures. Keep new dependency analysis local to ``MetalCodeGen`` unless the
same rule is needed by another target.

Metal's attribute syntax is part of the public output surface, so tests should
assert generated attributes directly when extending stage input/output behavior.
