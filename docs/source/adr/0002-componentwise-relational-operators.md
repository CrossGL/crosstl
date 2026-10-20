# ADR 0002: CrossGL relational and equality operators are componentwise on vectors

- Status: accepted
- Date: 2026-10-19
- Scope: `crosstl/backend/GLSL/aggregate_equality.py`, `crosstl/translator/codegen/relational_builtins.py`,
  `crosstl/translator/codegen/spirv_boolean_reductions.py`, and the emitters that call them

## Context

The source languages disagree on what `a == b` means for two vectors:

| Language | `v == w` on `vec3`/`float3` | Componentwise form | Reference |
|---|---|---|---|
| GLSL, GLSL ES | one `bool`: every component equal | `equal(v, w)` → `bvec3`; `lessThan` ... `notEqual`, `not` | GLSL 4.60 §5.9, §8.7 |
| HLSL | `bool3` | the operator itself | HLSL operators reference, "Comparison operators" |
| MSL | `bool3` | the operator itself | MSL 4.1 §2.13, §6.4 |
| WGSL | `vec3<bool>` | the operator itself | WGSL §8.7 |
| SPIR-V | `OpFOrdEqual` etc. on vectors yields a bool vector | the instruction itself | SPIR-V 1.6 §3.52.15 |

GLSL's `<`, `<=`, `>` and `>=` take only scalars, and its `!` takes only a scalar `bool`, so on vectors the
disagreement is confined to `==` and `!=`; the other operators are simply not available there.

CrossGL had not decided which meaning it follows. The emitters had each picked one:

- the GLSL emitter lowered CrossGL `v == w` on vectors to `equal(v, w)`, so it already treated the operator as
  componentwise;
- the GLSL importer kept GLSL's `bool b = (v == w);` verbatim, which the HLSL emitter then emitted as
  `bool b = (v == w);` on `float3`, where DXC truncates the `bool3` to its first component: a silent change of
  meaning;
- CrossGL also accepts GLSL's built-in spellings (`lessThan`, `equal`, `not`, ...) because its built-ins follow
  GLSL's names, but only the GLSL emitter knew them. HLSL, MSL, WGSL and Slang emitted calls to functions that do
  not exist on those targets, and SPIR-V had neither them nor `any` and `all`;
- the GLSL emitter refused `!` on a vector, and MSL and WGSL reject `!` on a numeric vector although HLSL accepts it.

## Decision

CrossGL's `==`, `!=`, `<`, `<=`, `>` and `>=` are componentwise on vectors and yield a bool vector of the same width,
as in HLSL, MSL, WGSL and SPIR-V. `!` is componentwise on vectors; on a numeric vector it first converts each
component to bool (`x != 0`). Aggregate comparison is spelled `all(a == b)` and `any(a != b)`. On scalars, structures
and arrays the equality operators keep their whole-value meaning.

Four of the five source languages already have these semantics, so their importers need no change. The fifth, GLSL,
and the emitters are adapted as follows.

1. **GLSL importer** (`aggregate_equality.py`). Each GLSL `==` and `!=` is rewritten by operand type:
   - vectors: `all(a == b)` and `any(a != b)`;
   - matrices: the conjunction `all(a[0] == b[0]) && ...` (or the disjunction of `any(... != ...)`) over the
     columns. The operands are repeated, so an operand with side effects is refused with
     `GlslMatrixEqualityOperandError`;
   - scalars, structures and arrays: unchanged.

   Operand types come from `GlslExpressionTypes`, a small typer over the GLSL AST that knows variables, parameters,
   structure members, swizzles, subscripts, constructors, user function return types, built-in result types
   (`builtin_result_types.py`) and texel lookups. When neither operand's type is known the importer refuses with
   `GlslAggregateEqualityError` rather than guess scalar, because a wrong guess is the silent truncation above.

2. **GLSL's built-in spellings on other targets** (`relational_builtins.py`). Every emitter except GLSL calls
   `lower_relational_builtins` on the program first. It rewrites `lessThan(a, b)` ... `notEqual(a, b)` into the
   CrossGL operator they are defined as, and `not(b)` into `!b`. A user function with one of those names is not
   rewritten, and neither is a call with the wrong number of arguments; the emitter then rejects it as an unknown
   call.

3. **GLSL emitter.** CrossGL `!v` on a vector is emitted as `not(v)`, or `not(bvecN(v))` for a numeric vector.
   Vector `==` and `<` continue to be emitted as `equal` and `lessThan`.

4. **MSL and WGSL emitters.** `!` on a numeric vector converts the operand first: `!boolN(v)` and `!vecN<bool>(v)`.

5. **SPIR-V emitter** (`spirv_boolean_reductions.py`). `any` and `all` lower to `OpAny` and `OpAll`; a numeric
   vector operand is first compared with zero, and a scalar operand reduces to itself converted to `bool`. `bool`
   to floating-point conversion is added (`OpSelect` of `1.0` and `0.0`, GLSL 4.60 §5.4.1).

## Consequences

- Hub text read by HLSL, MSL, WGSL, Slang and SPIR-V means what it says on those targets; the GLSL importer's
  output means on every target what the GLSL source meant.
- A `.cgl` source that writes `bool b = (v == w);` on vectors initializes a `bool` from a bool vector. The GLSL
  emitter refuses it (`OpenGLScalarConversionError`). The HLSL, MSL and WGSL emitters still emit the initializer:
  DXC truncates it to the first component, and MSL and WGSL reject it. Making those emitters refuse implicit
  vector-to-scalar narrowing is follow-up work.
- GLSL sources whose equality operands cannot be typed are refused instead of being translated. The typer covers
  the constructs measured in the pinned GLSL corpus; gaps are closed by extending the typer, not by guessing.
- Structure and array equality keep their whole-value meaning in the hub, but HLSL, MSL and WGSL have no such
  operator. Lowering them to member-wise comparison is follow-up work; until then those emitters must refuse them.

Tests: `tests/test_translator/test_componentwise_relational_operators.py` covers the importer rewrite for vectors,
matrices, structures, conditions and untyped refusals, every emitter's lowering, and validation of the same program
with glslang, naga, spirv-val, `xcrun metal` and DXC where those tools are installed.
