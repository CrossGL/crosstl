Codegen Utilities
=================

Shared codegen utilities keep backend generators consistent without forcing all
backends into one inheritance hierarchy.

Resource queries
----------------

``ResourceQueryMixin`` detects resource query calls such as ``textureSize``,
``imageSize``, ``textureSamples``, ``imageSamples``, and
``textureQueryLevels``. For targets that need explicit metadata, it propagates
metadata requirements through function calls and generates helper functions for
dimension, mip, and sample-count queries.

Resource diagnostics
--------------------

``ResourceDiagnosticMixin`` centralizes fallback expressions for unsupported
resource operations. Backends use it to emit comments plus type-correct neutral
values for unsupported multisample, shadow, sampled-resource, query, and image
atomic calls.

Resource arrays
---------------

``collect_resource_array_size_hints`` infers minimum array sizes from literal
array indices and propagates requirements through function calls. It also
checks fixed sizes for conflicts. ``format_array_declarator`` handles C-style
declarators for fixed and dynamic resource arrays.

Array and AST helpers
---------------------

``array_utils`` handles array type strings, literal integer sizing, and struct
member type collection. ``ASTUtils`` converts AST type nodes to backend type
strings and exposes compatibility helpers for older AST shapes.

Vector arithmetic
-----------------

``VectorArithmeticMixin`` infers expression result types and lowers vector
binary operations into generated helper functions when a target lacks native
operator support for the specific vector shape.
