"""Element types of storage-image parameters in HLSL and Slang output.

HLSL and Slang pass a resource only to a parameter of the same type, element
type included: DXC rejects a call that passes ``RWTexture2D<float>`` to a
``RWTexture2D<float4>`` parameter. A global storage image or texel buffer gets
its element type from its format, its ``@texel`` attribute, its use as an
image atomic's target or the four-component default, but a CrossGL parameter
such as ``image2D img`` carries none of them.

Globals and parameters joined by call sites therefore form one class, as in
``hlsl_atomic_resources``, and a parameter is declared with the type that the
globals of its class share. When those globals need different types, one HLSL
function cannot take them all, and the generator refuses the parameter rather
than emit a call DXC or slangc rejects.
"""

from __future__ import annotations

from .ast_walk import walk_ast_values
from .stage_utils import collect_stage_local_variables

__all__ = ["STORAGE_RESOURCE_TYPES", "StorageParameterTexels"]

_SHAPES = (
    "1D",
    "1DArray",
    "2D",
    "2DArray",
    "2DMS",
    "2DMSArray",
    "3D",
    "Cube",
    "CubeArray",
    "Buffer",
)
STORAGE_RESOURCE_TYPES = frozenset(
    f"{prefix}image{shape}" for prefix in ("", "i", "u") for shape in _SHAPES
)


def _parameters(function):
    return getattr(function, "parameters", getattr(function, "params", None)) or []


def _declared_type(node):
    return getattr(node, "param_type", getattr(node, "vtype", None))


def _expression_name(expr):
    """The variable an argument names, through any array subscripts, or ``None``."""
    while expr is not None and not isinstance(expr, str):
        name = getattr(expr, "name", None)
        if isinstance(name, str):
            return name
        if "ArrayAccess" not in type(expr).__name__:
            return None
        expr = getattr(expr, "array", getattr(expr, "array_expr", None))
    return expr


def _call_name(node):
    callee = getattr(node, "function", getattr(node, "name", None))
    return _expression_name(callee)


def _call_args(node):
    return getattr(node, "arguments", getattr(node, "args", None)) or []


class StorageParameterTexels:
    """Classes of storage resources joined by calls, and their globals."""

    def __init__(self, resource_base_type):
        self._resource_base_type = resource_base_type
        self._parent = {}
        self._parameter_keys = {}
        self._globals = {}

    def _is_tracked(self, vtype):
        return self._resource_base_type(vtype) in STORAGE_RESOURCE_TYPES

    def _find(self, key):
        root = self._parent.setdefault(key, key)
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def _union(self, first, second):
        first_root, second_root = self._find(first), self._find(second)
        if first_root != second_root:
            # Deterministic: the smaller key is the root.
            low, high = sorted((first_root, second_root))
            self._parent[high] = low

    def prepare(self, ast):
        """Join the storage resources passed to storage-resource parameters."""
        self._parent = {}
        self._parameter_keys = {}
        self._globals = {}
        functions = [
            node
            for node in walk_ast_values(ast)
            if hasattr(node, "body") and hasattr(node, "parameters")
        ]
        overloads = {}
        for function in functions:
            function_name = getattr(function, "name", None) or ""
            keys = []
            for parameter in _parameters(function):
                key = None
                if self._is_tracked(_declared_type(parameter)):
                    key = ("param", function_name, parameter.name)
                    self._parameter_keys[id(parameter)] = key
                    self._find(key)
                keys.append(key)
            overloads.setdefault(function_name, []).append(keys)
        declarations = list(getattr(ast, "global_variables", None) or [])
        declarations += collect_stage_local_variables(ast)
        for node in declarations:
            name = getattr(node, "name", None)
            if isinstance(name, str) and self._is_tracked(getattr(node, "vtype", None)):
                self._globals.setdefault(name, node)
        for function in functions:
            self._join_call_sites(function, overloads)

    def _join_call_sites(self, function, overloads):
        function_name = getattr(function, "name", None) or ""
        parameters = {parameter.name for parameter in _parameters(function)}
        for node in walk_ast_values(getattr(function, "body", None) or []):
            if not hasattr(node, "args") and not hasattr(node, "arguments"):
                continue
            args = _call_args(node)
            for keys in overloads.get(_call_name(node), ()):
                if len(keys) != len(args):
                    continue
                for parameter_key, argument in zip(keys, args):
                    name = _expression_name(argument)
                    if parameter_key is None or not name:
                        continue
                    if name in parameters:
                        self._union(parameter_key, ("param", function_name, name))
                    elif name in self._globals:
                        self._union(parameter_key, ("global", name))

    def copy_parameter(self, original, copied):
        """Give a copied parameter, such as a prototype's, its original's class."""
        key = self._parameter_keys.get(id(original))
        if key is not None:
            self._parameter_keys[id(copied)] = key

    def joined_globals(self, node):
        """The global resources a storage-resource parameter receives, by name."""
        key = self._parameter_keys.get(id(node))
        if key is None:
            return []
        root = self._find(key)
        return [
            self._globals[name]
            for name in sorted(self._globals)
            if ("global", name) in self._parent and self._find(("global", name)) == root
        ]

    def parameter_type(self, node, global_type):
        """The type the globals a parameter receives share, or ``None``.

        ``global_type`` maps a global declaration to its target type. Raises
        ``ValueError`` when the globals need different types.
        """
        declared = {}
        for global_node in self.joined_globals(node):
            declared.setdefault(global_type(global_node), []).append(global_node.name)
        if len(declared) <= 1:
            return next(iter(declared), None)
        received = "; ".join(
            f"{', '.join(names)} as {type_name}"
            for type_name, names in sorted(declared.items())
        )
        raise ValueError(
            f"storage image parameter '{node.name}' receives images of different "
            f"types ({received}); a resource parameter takes resources of one type"
        )
