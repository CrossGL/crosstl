"""Arrays of arrays of resources, flattened to one dimension for Vulkan.

GLSL (``GL_ARB_arrays_of_arrays``), HLSL and MSL declare arrays of arrays of samplers, textures and images, and
HLSL and CrossGL arrays of arrays of constant buffers (``ConstantBuffer<T> lights[2][4]``)::

    uniform sampler2D atlas[2][4];                 // GLSL
    Texture2D<float4> atlas[2][4] : register(t0);  // HLSL

A Vulkan SPIR-V module cannot: a ``UniformConstant`` variable is a sampler, image, sampled image or acceleration
structure, or a one-dimensional array of one (VUID-StandaloneSpirv-UniformConstant-04655), and the descriptor set
layout binding has one descriptor count. For SPIR-V targets :func:`flatten_descriptor_arrays` declares
``T atlas[A][B]`` as ``T atlas[A * B]`` and rewrites each access ``atlas[i][j]`` to ``atlas[int(i) * B + int(j)]``,
the row-major order DXC numbers the descriptors of a multi-dimensional resource array in, so the binding still holds
``A * B`` descriptors and every access selects the descriptor it selected before. Only the outermost dimension may
be unsized (a runtime descriptor array); the inner extents are copied into the index as written.

An access that does not index every dimension (a sub-array passed to a function, ``atlas[i].length()``), a use of the
whole array, and a parameter declared as an array of arrays of resources have no one-dimensional equivalent and are
refused with :class:`DescriptorArrayUnsupportedError`. A ``NonUniformResourceIndex`` or ``nonuniformEXT`` index
keeps the qualifier on the whole linear index, since the descriptor it selects still varies across the invocations.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .constant_buffers import constant_buffer_element
from .hub_types import call_name, iter_functions
from .opaque_structs import _is_opaque
from .rewrite import call, clone
from .scoped_walk import ScopedWalker, parameter_type, program_stages

__all__ = ["DescriptorArrayUnsupportedError", "flatten_descriptor_arrays"]

TARGETS = frozenset({"vulkan", "spirv"})
_NON_UNIFORM = frozenset({"NonUniformResourceIndex", "nonuniformEXT"})


class DescriptorArrayUnsupportedError(TranslationDiagnostic):
    """An array of arrays of resources is used in a way a one-dimensional descriptor array cannot express."""

    def __init__(self, target, name, detail, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.multidimensional-descriptor-array",)
        super().__init__(
            f"array of arrays of resources '{name}' {detail}: a {target} descriptor array has one dimension "
            "(VUID-StandaloneSpirv-UniformConstant-04655), so only accesses that index every dimension can be "
            "flattened",
            target=target,
            code=f"{CODE_PREFIX}{target}-multidimensional-descriptor-array-unsupported",
        )


def _dimensions(type_node):
    """``(element type, [outer size, ..., inner size])`` of a nested array type."""
    sizes = []
    while isinstance(type_node, A.ArrayType):
        sizes.append(type_node.size)
        type_node = type_node.element_type
    return type_node, sizes


def _is_descriptor_grid(type_node):
    base, sizes = _dimensions(type_node)
    return len(sizes) >= 2 and (
        _is_opaque(base) or constant_buffer_element(base) is not None
    )


def _flat_size(sizes):
    """The product of ``sizes``; ``None`` when the outermost is unsized."""
    if sizes[0] is None:
        return None
    total = clone(sizes[0]) if isinstance(sizes[0], A.ASTNode) else sizes[0]
    values = [getattr(size, "value", size) for size in sizes]
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        product = 1
        for value in values:
            product *= value
        return A.LiteralNode(product, A.PrimitiveType("int"))
    for size in sizes[1:]:
        total = A.BinaryOpNode(total, "*", clone(size))
    return total


class _Flattener(ScopedWalker):
    def __init__(self, program, target, grids):
        super().__init__(program)
        self.target = target
        # id(declared type) -> (name, [sizes outer to inner]) of each flattened declaration.
        self.grids = grids
        self.uses = {}
        self.consumed = set()

    def grid_of(self, name):
        declared = self.scope.lookup(name)
        return self.grids.get(id(declared)) if declared is not None else None

    def refuse(self, name, detail, node):
        raise DescriptorArrayUnsupportedError(
            self.target,
            name,
            detail,
            source_location=getattr(node, "source_location", None),
        )

    def visit(self, node):
        if isinstance(node, A.IdentifierNode):
            if self.grid_of(node.name) is not None:
                self.uses[id(node)] = node
            return
        if not isinstance(node, A.ArrayAccessNode):
            return
        indices = []
        current = node
        while isinstance(current, A.ArrayAccessNode):
            indices.append(current.index_expr)
            current = current.array_expr
        if not isinstance(current, A.IdentifierNode):
            return
        grid = self.grid_of(current.name)
        if grid is None:
            return
        _, sizes = grid
        indices.reverse()
        if len(indices) < len(sizes):
            # An inner node of a longer access chain; the outermost node of the chain rewrites it.
            return
        linear = call("int", [indices[0]])
        for size, index in zip(sizes[1:], indices[1:]):
            scaled = A.BinaryOpNode(linear, "*", clone(size))
            linear = A.BinaryOpNode(scaled, "+", call("int", [index]))
        qualifiers = [
            call_name(index)
            for index in indices
            if isinstance(index, A.FunctionCallNode)
            and call_name(index) in _NON_UNIFORM
        ]
        if qualifiers:
            linear = call(qualifiers[0], [linear])
        node.array_expr = node.array = current
        node.index_expr = node.index = linear
        self.consumed.add(id(current))

    def check_uses(self):
        for key in sorted(set(self.uses) - self.consumed):
            node = self.uses[key]
            self.refuse(
                node.name,
                "is used without indexing every dimension",
                node,
            )


def _declarations(program):
    yield from getattr(program, "global_variables", None) or []
    for stage in program_stages(program):
        yield from getattr(stage, "local_variables", None) or []


def flatten_descriptor_arrays(program, target) -> None:
    """Declare every array of arrays of resources of ``program`` with one dimension for a SPIR-V ``target``."""
    target = str(target or "").lower()
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return
    grids = {}
    for variable in _declarations(program):
        if isinstance(variable, A.VariableNode) and _is_descriptor_grid(
            variable.var_type
        ):
            _, sizes = _dimensions(variable.var_type)
            grids[id(variable.var_type)] = (variable.name, sizes)
    for function in iter_functions(program):
        for parameter in getattr(function, "parameters", None) or []:
            if _is_descriptor_grid(parameter_type(parameter)):
                raise DescriptorArrayUnsupportedError(
                    target,
                    getattr(parameter, "name", "?"),
                    f"is a parameter of '{function.name}'",
                    source_location=getattr(parameter, "source_location", None),
                )
    if not grids:
        return
    flattener = _Flattener(program, target, grids)
    flattener.run_program()
    flattener.check_uses()
    for variable in _declarations(program):
        if isinstance(variable, A.VariableNode) and id(variable.var_type) in grids:
            base, sizes = _dimensions(variable.var_type)
            flat = A.ArrayType(base, _flat_size(sizes))
            variable.var_type = flat
            if hasattr(variable, "vtype"):
                variable.vtype = flat
