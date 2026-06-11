Slang Backend
=============

The Slang backend covers CrossGL-to-Slang generation and Slang source import.
It is selected through the ``slang`` target and source aliases.

Pipeline
--------

CrossGL output generation is implemented by
``crosstl.translator.codegen.slang_codegen.SlangCodeGen``. The generator emits
Slang syntax from the shared translator AST, maps CrossGL types to Slang/HLSL
style types, preserves stage entry points, emits constant buffers and resources,
and generates helper functions for selected image and texture operations.

Reverse translation uses ``crosstl.backend.slang.SlangLexer.SlangLexer`` and
``crosstl.backend.slang.SlangParser.SlangParser`` to parse Slang source into the
Slang backend AST. ``crosstl.backend.slang.SlangCrossGLCodeGen`` then
serializes that AST back into CrossGL syntax.

Supported Surface
-----------------

The backend focuses on Slang shader authoring:

* imports, exports, typedefs, structs, constant buffers, globals, and functions
* vertex, fragment, and compute-style entry points represented in the AST
* Slang/HLSL scalar, vector, matrix, sampler, texture, image, and buffer types
* image load/store/atomic helpers and texture sampling/query helpers
* semantics and resource attributes that align with DirectX-style shader code

Implementation Notes
--------------------

Slang overlaps with the DirectX backend, but it has its own syntax surface and
helper generation. Keep Slang-specific resource helper names and type mappings
inside ``SlangCodeGen``. Promote a helper to shared code only when DirectX,
GLSL, or another backend needs exactly the same behavior.
