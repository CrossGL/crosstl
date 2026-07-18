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
   * - ``CooperativeMatrixType``
     - Matrix values distributed across a subgroup or other execution scope
   * - ``ArrayType``
     - Static or dynamic arrays
   * - ``NamedType``
     - User-defined and backend resource types
   * - ``GenericType``
     - Generic type parameters and constraints

Cooperative matrix fragment contracts
-------------------------------------

``CooperativeMatrixType`` keeps logical matrix metadata separate from the
lane-local fragment contract. The canonical CrossGL spelling accepts the
following ordered arguments::

   CooperativeMatrix<
       element_type,
       rows,
       columns,
       scope,
       use,
       memory_layout,
       fragment_layout,
       subgroup_size,
       elements_per_lane,
       fragment_provenance
   >

The first three arguments are required. Existing three- through six-argument
forms remain valid; omitted fragment fields are ``unspecified``. A concrete
``dense`` or ``metal_thread_elements`` layout must satisfy
``rows * columns == subgroup_size * elements_per_lane``. Other layout labels
may describe an opaque source profile and remain fail-closed until a target
implements that profile.

``fragment_provenance`` records how the frontend proved the fragment contract;
it is not a target capability or a substitute for a coordinate mapping. Metal
whole-fragment ``thread_elements()`` reference views use the
``metal_thread_elements`` layout and expand to ordered
``cooperative_matrix_element`` operations. Incompatible component types,
dependent widths, and conflicting fragment contracts produce structured
diagnostics instead of ordinary casts.

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
