"""The binary32 C math functions of CUDA and HIP whose CrossGL spelling drops the ``f`` suffix.

CUDA Math API, single precision mathematical functions, and the HIP math API declare these ``float`` forms of the C
math library (C11 7.12). CrossGL spells each one, for every floating-point type, without the suffix, as GLSL, HLSL
and MSL do, so the importers write ``acosh(x)`` for ``acoshf(x)`` and every generator sees one name. Keeping the
suffixed name would hand the generators of other languages a call to a function their language does not declare.

The functions CrossGL carries under another name (``sinf`` and ``cosf`` keep the libm precision bound, ``roundf``
rounds halfway cases away from zero) and the ones with pointer results (``frexpf``, ``modff``, ``sincosf``,
:mod:`.pointer_math_import`) are mapped where those meanings are handled.
"""

LIBM_BINARY32_FUNCTIONS = {
    "asinhf": "asinh",
    "acoshf": "acosh",
    "atanhf": "atanh",
    "copysignf": "copysign",
    "cospif": "cospi",
    "sinpif": "sinpi",
    "exp10f": "exp10",
    "fdimf": "fdim",
    "ilogbf": "ilogb",
    "ldexpf": "ldexp",
    "log10f": "log10",
    "nextafterf": "nextafter",
}
