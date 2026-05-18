Mojo Backend
============

The Mojo backend covers CrossGL-to-Mojo generation and Mojo source import. It
is selected through the ``mojo`` target and source aliases.

Pipeline
--------

CrossGL output generation is implemented by
``crosstl.translator.codegen.mojo_codegen.MojoCodeGen``. The generator emits a
Mojo-like shader module, maps CrossGL scalar and aggregate types to Mojo types,
uses ``SIMD`` for vector storage, emits helper types for matrices, and generates
constructor/swizzle helpers when an expression needs them.

Reverse translation uses ``crosstl.backend.Mojo.MojoLexer.MojoLexer`` and
``crosstl.backend.Mojo.MojoParser.MojoParser`` to parse Mojo source into the
Mojo backend AST. ``crosstl.backend.Mojo.MojoCrossGLCodeGen`` then serializes
that AST back into CrossGL syntax.

Supported Surface
-----------------

The backend focuses on shader-like Mojo code:

* imports, structs, classes, constant buffers, functions, decorators, constants,
  and globals
* ``var``/``let``-style declarations, assignments, loops, conditionals,
  switches, returns, and expression statements
* ``SIMD`` vector forms, generated matrix helper structs, swizzles, vector
  constructors, and array literals
* shader decorators for vertex, fragment, and compute entry points
* fixed arrays through ``InlineArray`` and dynamic arrays through ``List``

Implementation Notes
--------------------

Mojo codegen carries helper-registration state because some CrossGL vector and
matrix operations require generated support functions. Keep helper generation
near ``MojoCodeGen`` so emitted code remains self-contained. Backend-neutral
array or expression utilities should only be shared after another target needs
the same behavior.
