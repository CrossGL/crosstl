"""Metal ``fast::`` and ``precise::`` math calls carried through CrossGL.

Metal sources compiled with ``-fno-fast-math`` (as MLX does) still select the
relaxed implementations through ``metal::fast::<op>``.  Dropping the namespace
turns those calls into the precise implementations, which changes results and
register pressure (and with it the pipeline's maximum threadgroup size).

The Metal frontend therefore writes ``fast::<op>(...)`` into CrossGL.  The
CrossGL parser normalizes such a call to the plain builtin ``<op>(...)`` and
records the namespace in the call's annotations, so every other backend sees
exactly the call it saw before.  Only the Metal backend reads the annotation
and emits ``metal::fast::<op>`` again.

``metal::precise::<op>`` calls that reach CrossGL as plain builtins travel the
same way with a ``precise`` annotation.  ``-fno-fast-math`` makes Metal's plain
binary32 overloads of these functions match the precise ones, but the binary16
overloads stay native half implementations (``tan(-0.0h)`` is ``+0``), so the
Metal backend spells the namespace again.  The OpenGL backend reads the
annotation for the odd hyperbolic functions, whose GLSL builtins drop the sign
of zero, and ``tanh`` of NaN, on Mesa.
"""

METAL_MATH_NAMESPACE_ANNOTATION = "metal.math_namespace"

# Metal fast:: functions whose CrossGL builtin has the same name and arity and
# which the Metal backend emits as the plain standard-library call.
METAL_FAST_MATH_FUNCTIONS = frozenset(
    {
        "cos",
        "exp",
        "exp2",
        "log",
        "log2",
        "pow",
        "rsqrt",
        "sin",
        "sqrt",
        "tan",
    }
)

# Metal precise:: functions the Metal frontend passes to CrossGL as the plain
# builtin of the same name and arity (the others lower to portable helpers).
METAL_PRECISE_MATH_FUNCTIONS = frozenset(
    {
        "asinh",
        "atanh",
        "cosh",
        "log10",
        "log2",
        "sinh",
        "tan",
        "tanh",
    }
)

_FAST_PREFIXES = ("metal::fast::", "fast::")
_PRECISE_PREFIXES = ("metal::precise::", "precise::")


def _namespaced_operation(name, prefixes, functions):
    text = str(name).lstrip(":")
    for prefix in prefixes:
        if text.startswith(prefix):
            operation = text[len(prefix) :]
            return operation if operation in functions else None
    return None


def metal_fast_math_operation(name):
    """Return ``op`` for a ``fast::op`` or ``metal::fast::op`` call name."""
    return _namespaced_operation(name, _FAST_PREFIXES, METAL_FAST_MATH_FUNCTIONS)


def metal_precise_math_operation(name):
    """Return ``op`` for a ``precise::op`` or ``metal::precise::op`` call name."""
    return _namespaced_operation(name, _PRECISE_PREFIXES, METAL_PRECISE_MATH_FUNCTIONS)


def metal_math_namespace(name):
    """Return ``(namespace, op)`` for a namespaced Metal math call, else None."""
    operation = metal_fast_math_operation(name)
    if operation is not None:
        return "fast", operation
    operation = metal_precise_math_operation(name)
    if operation is not None:
        return "precise", operation
    return None
