CrossGL AST
===========

The canonical AST in ``crosstl.translator.ast`` is the contract between the
CrossGL frontend and target code generators.

Program nodes
-------------

``ShaderNode`` is the root object. It groups shader stages, structs, functions,
global variables, constants, imports, preprocessors, and constant buffers.

``StageNode`` represents a pipeline stage such as vertex, fragment, compute,
mesh, task, or ray-tracing stages. Each stage points at an entry-point
``FunctionNode`` and can carry execution metadata such as workgroup size.

Type nodes
----------

The type system is represented with ``TypeNode`` subclasses:

.. list-table::
   :header-rows: 1

   * - Node
     - Purpose
   * - ``PrimitiveType``
     - Scalar values such as ``float``, ``int``, ``uint``, and ``bool``
   * - ``VectorType``
     - Fixed-width vector values such as ``vec3`` or ``uvec4``
   * - ``MatrixType``
     - Matrix values such as ``mat4`` and rectangular matrix forms
   * - ``ArrayType``
     - Static or dynamic arrays
   * - ``NamedType``
     - User-defined and backend resource types
   * - ``GenericType``
     - Generic type parameters and constraints

Declaration nodes
-----------------

Structs, enum variants, functions, parameters, variables, constants, generic
parameters, and attributes each have dedicated node classes. Code generators
use these nodes to preserve semantics such as bindings, qualifiers, generic
arguments, and stage annotations.

Statement and expression nodes
------------------------------

Statement nodes cover blocks, assignments, control flow, loops, matches,
switches, returns, breaks, and continues. Expression nodes cover literals,
identifiers, binary/unary operators, function calls, member and array access,
casts, constructors, lambdas, resource operations, wave intrinsics, ray tracing
operations, and mesh operations.

Compatibility helpers
---------------------

The module keeps aliases such as ``CbufferNode`` and ``VectorConstructorNode``
for older backend code. ``create_legacy_shader_node`` also provides a bridge
for tests and backends that still expect the older root shape.
