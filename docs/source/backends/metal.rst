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

Reverse translation uses ``crosstl.backend.Metal.MetalLexer.MetalLexer`` and
``crosstl.backend.Metal.MetalParser.MetalParser`` to parse MSL into the Metal
backend AST. ``crosstl.backend.Metal.MetalCrossGLCodeGen`` then serializes that
AST back into CrossGL syntax.

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
