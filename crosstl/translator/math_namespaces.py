"""Metal ``fast::`` math calls carried through CrossGL.

Metal sources compiled with ``-fno-fast-math`` (as MLX does) still select the
relaxed implementations through ``metal::fast::<op>``.  Dropping the namespace
turns those calls into the precise implementations, which changes results and
register pressure (and with it the pipeline's maximum threadgroup size).

The Metal frontend therefore writes ``fast::<op>(...)`` into CrossGL.  The
CrossGL parser normalizes such a call to the plain builtin ``<op>(...)`` and
records the namespace in the call's annotations, so every other backend sees
exactly the call it saw before.  Only the Metal backend reads the annotation
and emits ``metal::fast::<op>`` again.
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

_FAST_PREFIXES = ("metal::fast::", "fast::")


def metal_fast_math_operation(name):
    """Return ``op`` for a ``fast::op`` or ``metal::fast::op`` call name."""
    text = str(name).lstrip(":")
    for prefix in _FAST_PREFIXES:
        if text.startswith(prefix):
            operation = text[len(prefix) :]
            if operation in METAL_FAST_MATH_FUNCTIONS:
                return operation
            return None
    return None
