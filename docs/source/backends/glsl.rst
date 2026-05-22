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
