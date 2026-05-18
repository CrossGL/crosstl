Rust Backend
============

The Rust backend covers CrossGL-to-Rust generation and Rust source import for
GPU-oriented Rust shader experiments. It is selected through the ``rust`` or
``rs`` target and source aliases.

Pipeline
--------

CrossGL output generation is implemented by
``crosstl.translator.codegen.rust_codegen.RustCodeGen``. The generator emits a
Rust-like shader module, maps CrossGL scalar, vector, matrix, sampler, and
constant-buffer types to Rust-style GPU helper types, and lowers shader
semantics to Rust attributes.

Reverse translation uses ``crosstl.backend.Rust.RustLexer.RustLexer`` and
``crosstl.backend.Rust.RustParser.RustParser`` to parse Rust source into the
Rust backend AST. ``crosstl.backend.Rust.RustCrossGLCodeGen`` then serializes
that AST back into CrossGL syntax.

Supported Surface
-----------------

The backend focuses on Rust syntax that is useful for shader-like programs:

* structs, functions, constants, statics, imports, traits, and impl blocks
* ``let`` bindings, loops, ``match`` statements, ranges, references, casts, and
  struct initialization expressions
* vector and matrix helper types such as ``Vec2``, ``Vec3``, ``Vec4``, ``Mat3``,
  and ``Mat4``
* shader attributes that identify vertex, fragment, and compute entry points
* fixed and dynamic array declarators mapped between CrossGL and Rust syntax

Implementation Notes
--------------------

Rust codegen is text-oriented and keeps type inference state for variables and
function returns while rendering expressions. Keep Rust-specific attribute and
type naming rules in ``RustCodeGen`` or the Rust reverse converter. Shared
array parsing utilities should stay in ``array_utils`` when the rule is not
specific to Rust syntax.
