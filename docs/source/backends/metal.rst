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

Implementation Notes
--------------------

Metal codegen carries more per-function state than the other shader backends
because cbuffer dependencies and global resource dependencies affect function
signatures. Keep new dependency analysis local to ``MetalCodeGen`` unless the
same rule is needed by another target.

Metal's attribute syntax is part of the public output surface, so tests should
assert generated attributes directly when extending stage input/output behavior.
