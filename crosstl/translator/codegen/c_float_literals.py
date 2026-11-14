"""Spell CrossGL ``float`` literals for the CUDA and HIP emitters.

A CrossGL floating literal has type ``float`` whether or not the source wrote
the ``f`` suffix, as in GLSL. C++ gives an unsuffixed floating literal the type
``double``, so ``x * 0.5`` with a ``float`` ``x`` is evaluated in double
precision and ``fmod(x, 2.0)`` resolves to the host-only mixed-type
``std::fmod`` template. Writing the literal with the ``f`` suffix keeps the
CrossGL type.
"""

import math


def c_float_literal_text(value):
    """Return ``value`` as a C++ ``float`` literal, or None if it has no spelling.

    Python's ``repr`` gives the shortest decimal text that reads back as the same
    double, which is the text the CrossGL source wrote for every literal a
    ``float`` can hold. Non-finite values have no literal spelling and are left
    to the caller.
    """
    if not isinstance(value, float) or not math.isfinite(value):
        return None
    return f"{value!r}f"
