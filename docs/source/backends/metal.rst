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

Byte-valued struct initializers convert expression carriers back to the declared
``char`` or ``uchar`` storage type. The same rule covers byte vectors, aliases,
materialized generic fields, nested aggregates and fixed arrays; omitted byte
fields are zero-initialized. It does not widen aggregate storage or change
arithmetic promotion. Native controls compare original and generated Metal over
all byte input values, checking signedness, field sizes, zero initialization,
single evaluation and guarded outputs.

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
