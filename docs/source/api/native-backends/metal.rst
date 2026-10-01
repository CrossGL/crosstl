Metal Source Backend
====================

Private Helpers
---------------

Class methods lowered into free functions retain private inline linkage.
Metal output marks private inline definitions as potentially unused, because
an entry-scoped translation may retain overloads that are not selected in that
unit. This does not change overload resolution, remove implementations or
disable compiler warnings. Exported functions and kernel entries remain
externally visible.

Materialized free-template operators retain inline linkage so identical
instantiations can occur in separate modules. Explicitly static operators and
operators from anonymous namespaces remain private; different private bodies
must not be merged during library linking. Native regressions compile and link
original and translated module pairs, then execute both entries and check their
individual results.

Precise Inverse Trigonometry
---------------------------

``metal::precise::asin`` and ``metal::precise::acos`` retain their precision
mode through portable float32 helpers instead of the target's ordinary
inverse-trigonometric intrinsic. Scalar and two-, three-, and four-component
vector calls are supported. Invalid domains produce NaNs; arcsine preserves
both zero signs. Default and fast namespace calls and user-defined overloads
keep their existing behavior. Unrepresentable precise signatures produce a
structured translation diagnostic.

The helpers share an attributed fdlibm rational approximation. Arcsine uses
that approximation near zero and a split-pi/2 reduction away from zero to avoid
cancellation. Native regression tests cover 4,394 inputs, ten scalar/vector
outputs per input, and argument evaluation counts. Required CI runs execute
generated HLSL on Windows, GLSL on Linux, and original/roundtrip Metal on macOS;
the float32 numerical gate is four ULPs, with exact zero signs and NaN checks.
This does not establish numerical parity for every precise math operation or
the complete MLX backend.

Generated support functions carry registered ``@source_license(fdlibm)``
metadata in the intermediate representation. License notices are emitted into
target source, including entry-scoped artifacts, rather than relying on source
comments surviving parsing.

Precise Sine and Cosine
----------------------

``metal::precise::sin`` and ``metal::precise::cos`` use portable binary32
helpers for scalar and two- to four-component vectors. Large finite arguments
are reduced with a 192-bit table and pairs of unsigned 32-bit words, retaining
the fractional remainder before conversion to float. No double-precision
shader capability is required. Small arguments use a split-pi/2 reduction or
the original value; sine preserves signed zero and subnormals. Nonfinite
arguments return a quiet NaN. Default/fast calls and user overloads retain
their existing behavior; unsupported precise operand types are diagnosed.

The reduction and polynomial coefficients are adapted from Arm
``optimized-routines`` revision ``375f32ed2f7098090f41795ad363822752a25a65``
(``math/sincosf.h`` and ``math/sincosf_data.c``). The MIT notice travels with
``@source_license(arm_optimized)`` metadata into generated target source.
The adaptation uses binary32 polynomial arithmetic instead of upstream
double arithmetic; it does not inherit the upstream implementation's error
bound. Required native tests enforce a four-ULP bound against an independent
160-digit reference, with exact zero signs, NaN classification, vector lanes,
single evaluation, large-exponent axis neighbors and buffer guards. They
execute HLSL on Windows, GLSL on Linux and original/generated Metal on macOS.
NaN payload identity, exception flags and other precise transcendental
operations are not covered by this contract.

.. automodule:: crosstl.backend.Metal

.. automodule:: crosstl.backend.Metal.MetalAst
   :members:

.. automodule:: crosstl.backend.Metal.MetalLexer
   :members:

.. automodule:: crosstl.backend.Metal.MetalParser
   :members:

.. automodule:: crosstl.backend.Metal.MetalCrossGLCodeGen
   :members:
