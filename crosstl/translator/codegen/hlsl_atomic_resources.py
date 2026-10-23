"""Storage resources that HLSL image atomics reach.

HLSL ``Interlocked*`` functions need a scalar 32-bit integer element, so an
unformatted CrossGL integer storage image or texel buffer is declared with a
scalar element (``RWTexture2D<uint>``, ``RWBuffer<uint>``) when an image atomic
targets it, and with a four-component element otherwise (see
``storage_image_texels``).

A resource reaches an atomic either directly or through function parameters,
and HLSL passes a resource only to a parameter of the same type. Globals and
resource parameters joined by call sites therefore form one class: when an
atomic targets any member of a class, every member gets the scalar element.
"""

from .hlsl_texel_buffers import IMAGE_ATOMIC_INTRINSICS

__all__ = ["HLSLAtomicResources", "INTEGER_STORAGE_RESOURCE_TYPES"]

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
INTEGER_STORAGE_RESOURCE_TYPES = frozenset(
    f"{prefix}image{shape}" for prefix in ("i", "u") for shape in _SHAPES
)


def _call_name(codegen, node):
    callee = getattr(node, "function", getattr(node, "name", None))
    return codegen.expression_name(callee)


def _call_args(node):
    return getattr(node, "arguments", getattr(node, "args", None)) or []


def _parameters(function):
    return getattr(function, "parameters", getattr(function, "params", None)) or []


def _parameter_type(parameter):
    return getattr(parameter, "param_type", getattr(parameter, "vtype", None))


class HLSLAtomicResources:
    """Classes of integer storage resources joined by calls, and their atomics."""

    def __init__(self, codegen):
        self.codegen = codegen
        self._parent = {}
        self._atomic_roots = frozenset()
        self._parameter_keys = {}
        self._function_parameters = {}

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

    def _is_tracked(self, vtype):
        return self.codegen.resource_base_type(vtype) in INTEGER_STORAGE_RESOURCE_TYPES

    def _argument_key(self, function_name, parameters, argument):
        name = self.codegen.expression_name(argument)
        if not name:
            return None
        if name in parameters:
            return ("param", function_name, name)
        return ("global", name)

    def prepare(self, ast):
        """Join resources passed to resource parameters and mark atomic classes."""
        codegen = self.codegen
        self._parent = {}
        self._parameter_keys = {}
        self._function_parameters = {}
        functions = codegen.collect_functions(ast)
        tracked_parameters = {}
        for function in functions:
            function_name = getattr(function, "name", None) or ""
            names = []
            for parameter in _parameters(function):
                names.append(parameter.name)
                if self._is_tracked(_parameter_type(parameter)):
                    key = ("param", function_name, parameter.name)
                    self._parameter_keys[id(parameter)] = key
                    self._find(key)
            self._function_parameters.setdefault(function_name, set()).update(names)
            tracked_parameters.setdefault(function_name, []).append(
                [
                    (
                        ("param", function_name, parameter.name)
                        if self._is_tracked(_parameter_type(parameter))
                        else None
                    )
                    for parameter in _parameters(function)
                ]
            )

        atomic_keys = []
        for function in functions:
            function_name = getattr(function, "name", None) or ""
            parameters = self._function_parameters.get(function_name, set())
            for node in codegen.walk_ast(getattr(function, "body", None) or []):
                if not hasattr(node, "args") and not hasattr(node, "arguments"):
                    continue
                callee = _call_name(codegen, node)
                args = _call_args(node)
                if callee in IMAGE_ATOMIC_INTRINSICS and args:
                    key = self._argument_key(function_name, parameters, args[0])
                    if key is not None:
                        atomic_keys.append(key)
                    continue
                for overload in tracked_parameters.get(callee, ()):
                    if len(overload) != len(args):
                        continue
                    for parameter_key, argument in zip(overload, args):
                        if parameter_key is None:
                            continue
                        key = self._argument_key(function_name, parameters, argument)
                        if key is not None:
                            self._union(parameter_key, key)
        self._atomic_roots = frozenset(self._find(key) for key in atomic_keys)

    def _is_atomic_key(self, key):
        return key in self._parent and self._find(key) in self._atomic_roots

    def is_atomic_global(self, name):
        """Whether an image atomic reaches the global resource ``name``."""
        return self._is_atomic_key(("global", name))

    def is_atomic_declaration(self, node):
        """Whether an image atomic reaches the global or parameter ``node``."""
        name = getattr(node, "name", None)
        if not isinstance(name, str):
            return False
        key = self._parameter_keys.get(id(node))
        if key is None:
            function_name = getattr(self.codegen, "current_function_name", None) or ""
            if name in self._function_parameters.get(function_name, ()):
                key = ("param", function_name, name)
            else:
                key = ("global", name)
        return self._is_atomic_key(key)
