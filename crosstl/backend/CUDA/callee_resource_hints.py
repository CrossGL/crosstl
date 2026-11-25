"""Resource types that texture and surface objects take from the helper functions they are passed to.

A CUDA or HIP texture object (``cudaTextureObject_t``, ``hipTextureObject_t``) or surface object has no shape of its
own: the importers give it the CrossGL type of the fetches that read it (``tex1Dfetch<float>`` makes it a
``samplerBuffer``, ``surf2Dread`` an ``image2D``). A kernel that only passes the object to a ``__device__`` helper
never reads it itself, so its parameter kept the CUDA spelling, which no target declares, while the helper's
parameter took the fetch's type. Now a call to a helper gives each argument that names an object the type the
helper's parameter takes. The parameter types are found to a fixed point, so an object passed through several
helpers takes the type of the fetch at the end of the chain; two helpers that read it differently give it no type,
as two different fetches do.

Only functions with an object parameter can pass a type on, so no other function is walked.
"""

__all__ = ["CalleeResourceHintMixin"]

_OBJECT_TYPES = (
    "cudaTextureObject_t",
    "cudaSurfaceObject_t",
    "hipTextureObject_t",
    "hipSurfaceObject_t",
)


def _merge(table, key, resource_type):
    """Record ``resource_type`` for ``key``; two different types leave it none."""
    if resource_type is None:
        return
    table[key] = (
        resource_type if table.get(key, resource_type) == resource_type else None
    )


def _parameter(param):
    """``(name, type)`` of a parameter: a node in the CUDA AST, a ``{"name", "type"}`` dict in the HIP AST."""
    if isinstance(param, dict):
        return param.get("name"), param.get("type")
    return getattr(param, "name", None), getattr(param, "vtype", None)


def _has_object_parameter(function):
    return any(
        any(name in str(_parameter(param)[1] or "") for name in _OBJECT_TYPES)
        for param in getattr(function, "params", None) or []
    )


class CalleeResourceHintMixin:
    """Members of the CUDA and HIP converters that carry resource hints from a helper's parameters to its callers.

    The converter provides ``collect_resource_object_type_hints``, ``collect_declared_variable_names``,
    ``add_resource_object_type_hint``, ``get_resource_object_expression_name``, ``parse_cpp_template`` and
    ``is_user_defined_function``, and calls :meth:`collect_callee_resource_hints` for each call it collects hints
    from.
    """

    def collect_callee_parameter_resource_hints(self, functions):
        """``{function name: {parameter index: CrossGL type}}`` for the functions with an object parameter."""
        candidates = [
            function
            for function in functions
            if getattr(function, "name", None) and _has_object_parameter(function)
        ]
        self.callee_parameter_resource_hints = {}
        for _ in range(len(candidates) + 1):
            table = {}
            for function in candidates:
                hints = self.collect_resource_object_type_hints(
                    function, self.collect_declared_variable_names(function)
                )
                by_index = table.setdefault(function.name, {})
                for index, param in enumerate(function.params):
                    _merge(by_index, index, hints.get(_parameter(param)[0]))
            if table == self.callee_parameter_resource_hints:
                break
            self.callee_parameter_resource_hints = table
        return self.callee_parameter_resource_hints

    def collect_callee_resource_hints(self, hints, call):
        """Give each object argument of the helper call ``call`` the type the helper's parameter takes."""
        name = getattr(call, "name", None)
        if not isinstance(name, str):
            return
        base_name, _ = self.parse_cpp_template(name)
        if not self.is_user_defined_function(base_name):
            return
        table = getattr(self, "callee_parameter_resource_hints", None) or {}
        for index, resource_type in sorted((table.get(base_name) or {}).items()):
            args = getattr(call, "args", None) or []
            if resource_type is None or index >= len(args):
                continue
            self.add_resource_object_type_hint(
                hints,
                self.get_resource_object_expression_name(args[index]),
                resource_type,
            )
