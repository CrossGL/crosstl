# ADR 0001: Carry operation rules through CrossGL and check the rule table with SMT

- Status: accepted
- Date: 2026-10-18
- Scope: `crosstl/translator/semantics/`, `crosstl/_crosstl.py`, the project pipeline, `tests/conformance/rule_table_smt.py`

## Context

CrossGL records the syntax of an operation, not the rule it follows. `x + 1` imported from GLSL, HLSL or SPIR-V
wraps on signed overflow. The same text imported from Metal, CUDA or OpenCL C leaves overflow undefined. An emitter
that sees only the hub cannot tell the two apart, so it emits its own language's operator and inherits that
language's rule. The conformance harness measures the consequence:

| Case family | Source rule | Emitted | Target rule | Outcome at `c35dede8` (darwin-arm64 board) |
|---|---|---|---|---|
| `g_add_wrap`, `h_add_wrap`, `s_add_wrap`, `*_mul_wrap`, `*_loop_wrap` | wrap (GLSL 4.60 §4.1.3, ESSL 3.00 §4.1.3, SPIR-V §3.3.13 `OpIAdd` without `NoSignedWrap`, D3D11 functional spec §22.12.1 `iadd` and §22.12.6 `imul`) | `a + b` on `int` | undefined (MSL 4.1 §3.1: "The result of signed integer overflow is undefined") | diverged on Metal: Apple's compiler folds `x + 1 > x` to `true` |
| same | wrap | `a + b` on `int` | DXC emits `add nsw i32` (measured with DXC 1.9.2602 `-fcgl` and `-O3`; the `-O3` module folds `x + 1 > x` and stores the constant) | unsound on DirectX |
| same | wrap | `OpIAdd %int` | wrap by the SPIR-V specification, but MoltenVK (SPIRV-Cross) and KosmicKrisp re-translate it to MSL signed `+` | diverged on `vulkan-moltenvk` and `vulkan-kosmickrisp`, preserved on lavapipe |

The prototype in the paper's artifact (`crosstl-r.patch`, against `575bd374`) showed that carrying the rule in the hub
and discharging it once per target fixes these cells without per-emitter changes. It also showed its defects: a
flat, scope-less type environment; a crash on enumerations; an assumption comment placed before `#version`; no
project-pipeline hook; and an SMT checker that verified hand-written models of the polyfills rather than the
polyfills themselves.

## Decision

### 1. Hub intrinsics

The annotation pass rewrites each operation whose source rule is defined into a call of a reserved hub intrinsic.
The `__cgl_` prefix is reserved for them. The lowering pass replaces each intrinsic with one of:

- **native**: the target's own operation, when the target's rule refines the carried rule. The original node is
  restored, so output for such targets does not change.
- **polyfill**: a CrossGL expression template over the target's trusted base: unsigned 32-bit arithmetic, bit
  reinterpretation (`asint`, `asuint`), comparison and select.
- **refuse**: a `RuleInexpressibleError` with the stable code `project.translate.rule-inexpressible`.

Intrinsics, not node attributes, because a call survives every AST transformation that copies or rebuilds nodes, can
be written in `.cgl` text, and fails closed in any emitter that meets an unlowered one. The lowering pass also runs
for `.cgl` sources, so a hand-written `__cgl_wrap_add(a, b)` is honoured. `.cgl` output does not yet carry the
intrinsics, because the reverse code generators print CrossGL text directly; that is follow-up work.

Plain CrossGL operators carry no rule: signed overflow in a `.cgl` source is undefined. Every target's native operator
refines that rule, so `.cgl` sources translate as before.

Statement-shaped operations get statement intrinsics that keep the lvalue structure: `x += e` becomes
`__cgl_wrap_add_assign(x, e)`, and `++x` becomes `__cgl_wrap_preinc(x)`. Native lowering restores the original
statement. Polyfill lowering produces `x = asint(asuint(x) + asuint(e))`; it refuses when `x` has side effects, and
when an increment is used as a value, because the CrossGL parser and emitters do not yet handle assignment
expressions reliably. The 972 GLSL and HLSL files of the pinned corpus contain no increment used as a value.

### 2. Typing

Rules depend on operand types, so the passes need the hub's types. `crosstl/translator/semantics/hub_types.py`
infers scalar and vector types with lexical scopes: globals, buffer-block members, constants, structure members,
parameters, block-scoped locals and for-loop variables, swizzles, constructors, bit casts and user function return
types. It replaces the prototype's flat environment. It is the first piece of the shared typed IR (architecture
direction 3), and it is meant to replace the per-emitter `expression_result_type` functions.

An operation whose type cannot be inferred is left unchanged and counted in the `RuleReport` as unresolved. Once the
typing resolves every operation of the conformance cases and the pinned corpus, an unresolved operation that may
carry a rule becomes a refusal.

### 3. Proved absence of overflow

A for-loop counter `i` that is declared by the loop, compared as `i < e` (or `e > i`, or as a conjunct of the
condition), and written only by the loop's `++i` or `i++` update cannot overflow at the update. Its increment is left
as the target's native operation. The same holds for `--i` under `i > e`. Literal-only operations whose exact result
fits in 32 bits are also left native. Native and wrapping arithmetic agree on these operations, so this refines the
carried rule, and loop counters in the output stay readable.

### 4. Rule table format

The table is Python data in `crosstl/translator/semantics/rule_table.py`:

- `SOURCE_RULES[source]`: the rule families a source language defines, each with a citation.
- `TARGET_ENTRIES[(family, target)]`: a `TargetEntry(kind, reason, configuration)` where `kind` is `native`,
  `polyfill` or `refuse`. `reason` cites the target's specification, a measured compiler behaviour or a driver
  defect. `configuration` names a build setting the native entry depends on, for example Rust's
  `overflow-checks = false`.
- `POLYFILLS[intrinsic]`: CrossGL expression templates over the parameters `x0` and `x1`. Each parameter must occur
  exactly once, so substitution never duplicates an operand with side effects.
- `HELPERS[intrinsic]`: a polyfill that needs an operand more than once, as a `Helper(name, scalar, body)`. The
  lowering pass replaces the operation with a call of the reserved function `name` (`cgl_rule_` prefix), which
  evaluates each operand once, and injects the function's definition ahead of the program's functions
  (`crosstl/translator/semantics/helpers.py`). A vector operation calls a per-lane lift named `name` followed by the
  lane count. A program that defines a reserved helper name itself is refused.

Every (family, target) pair for every registered backend must have an entry; a missing entry is a test failure, not
a default.

Driver defects are table entries. The SPIR-V entry for `wrap` is `polyfill`, with the MoltenVK and KosmicKrisp
folding as its reason, although the SPIR-V specification makes `OpIAdd` wrap. Unsigned `OpIAdd` between bit casts is
the same operation in SPIR-V, and SPIRV-Cross prints it as unsigned MSL arithmetic.

The polyfill does not reach KosmicKrisp (Vulkan SDK 1.4.363, Mesa `269a955c89`). Its SPIR-V front end drops the bit
casts, because NIR's `iadd` has no signedness, and its MSL printer then types the addition from its signed uses. For
`g_add_wrap`, `MESA_KK_DEBUG=msl` shows `t28 = t24 + int(1); t29 = t24 < t28;`, which Apple's compiler folds. No
SPIR-V spelling of a 32-bit wrapping addition avoids this, so the defect needs a runtime-specific compile
configuration: a KosmicKrisp Vulkan profile whose `wrap` entry is `refuse`. Until that profile exists, the
`vulkan-kosmickrisp` cells of the wrap family stay `diverged` on the conformance board, and they are recorded as a
known driver defect rather than reclassified.

### 5. SMT check

`tests/conformance/rule_table_smt.py` checks every entry with Z3, in the style of Alive2. For each entry it asks for
an input on which the target construct does not refine the carried rule:

    find v, c :  defined_S(v)  and  not ( defined_T(v, c) and result_T(v, c) == rule_S(v) )

The checker does not use hand-written models of the polyfills. It parses the same template that the lowering pass
instantiates, with the CrossGL parser, and evaluates the resulting AST symbolically under the target's semantics of
the trusted base. A native entry is evaluated with the target's own operator semantics, where signed overflow is
undefined on Metal, DirectX (DXC `nsw`), CUDA and HIP, defined on GLSL, GLSL ES, WGSL and SPIR-V, and a trap or a
wrap on Rust depending on `overflow-checks`. Its results are 32-bit bit-vectors, quantified over all inputs.

Unsigned `/` and `%` are exact for a nonzero divisor. A zero divisor is unspecified on GLSL and GLSL ES (a fresh
value), gives the dividend and 0 on WGSL, and is undefined on every other target. A conditional is evaluated eagerly,
because SPIR-V `OpSelect` and WGSL `select` compute both arms, so a division by zero in the arm not taken still makes
the construct undefined. The solver runs `simplify`, `ctx-simplify`, `simplify` and then `smt`: contextual
simplification reduces a guarded `x0 / (x1 == 0 ? 1 : x1)` under `x1 != 0` to the carried rule's own `x0 / x1`,
which bit-blasting two 32-bit dividers does not prove within the timeout. Each tactic preserves equivalence.

A CI job runs the checker on every pull request and fails on any satisfiable query (an unsound entry), on any entry
it cannot model, and on any table pair without an entry.

### 6. Hooks

`apply_rule_carrying(ast, source, target)` runs in `crosstl.translate()` after the CrossGL AST is parsed and before
target code generation, for imported and `.cgl` sources. It also runs in the project pipeline's
`_crossgl_ast_for_project_target`, which the prototype missed. The passes are always on. There is no environment
switch.

## Rollout

1. This change: the framework, the `wrap` family (`+`, `-`, `*`, unary `-`, compound assignment, increment and
   decrement on 32-bit signed scalars and vectors), the SMT gate, and a Metal emitter fix: the argument of `as_type`
   no longer inherits the expected type of the result, which had turned `asint(asuint(v) + asuint(w))` on vectors
   back into signed arithmetic.
2. Families from the prototype, in this order, each with its conformance cells as the acceptance test. Done:
   Direct3D unsigned division by zero (`udiv_zero`, cell `h_udiv_zero`). Unsigned 32-bit `/`, `%`, `/=` and `%=` from
   HLSL whose divisor is not a nonzero literal become `__cgl_udiv_ones` and `__cgl_urem_ones`. Every target is a
   helper polyfill `x1 == 0u ? 0xFFFFFFFFu : x0 / (x1 == 0u ? 1u : x1)`, Direct3D included, because DXC lowers the
   operators to LLVM `udiv` and `urem`, which are undefined for a zero divisor. Remaining: shift-count masking, truncated signed remainder (moved from the SPIR-V
   emitter into the table), mixed-sign comparison, float-to-int with NaN and saturation, and `min`/`max`/`isnan` with
   NaN, including the configuration entries (Metal `math_mode(safe)`, HLSL `/Gis`, Vulkan float controls).
3. Narrow integers (S6) and 64-bit integers, which need the typed IR's storage legalization. Done for narrow
   registers: on Vulkan and WGSL, which have no 8- or 16-bit integer registers, `crosstl/translator/semantics/narrow.py`
   runs before annotation. It gives every narrow scalar, vector, array, parameter, return value and struct member a
   32-bit `int` and normalizes each write and each conversion to a narrow type modulo 2^N, with sign extension for
   signed types. Arithmetic results are normalized only where they keep a narrow type: C-family scalar operands are
   promoted to `int` first, but vector operands and GLSL, HLSL and SPIR-V explicit narrow types are not. Narrow host
   storage keeps the emitters' one-word-per-element layout. Reporting that layout in the reflection metadata is M3
   work. Measured on the darwin-arm64 board: the 18 `m_`, `c_` and `u_uchar_wrap` SPIR-V cells go from `diverged` to
   `preserved` on lavapipe, MoltenVK and KosmicKrisp, with no regression among the SPIR-V cells.
4. Unresolved rule-relevant operations become refusals.

## Consequences

- Measured with this change on the darwin-arm64 board (strict and relaxed): the 32 wrap-family cells on `metal-apple`
  and `vulkan-moltenvk` (`g_` and `h_` × `add`, `mul`, `loop`, and `s_` × `add`, `mul`) go from `diverged` to
  `preserved`, with no regression among the 892 cells. The `vulkan-kosmickrisp` add and loop cells stay `diverged`
  (section 4). DirectX execution needs WARP on Windows; the HLSL cells on macOS are compile-only.
- Output for GLSL, HLSL and SPIR-V sources to Metal, DirectX, CUDA, HIP, Slang and Vulkan uses unsigned arithmetic
  between bit casts for signed `+`, `-` and `*`, except for loop counters with a proved bound. Generated code is
  longer; the expected runtime cost, not yet measured, is the loss of `nsw`-based optimisations, which is the
  semantic difference.
- Translation from these sources refuses a signed increment that is used as a value, or a compound assignment to an
  lvalue with side effects, on targets that need the polyfill.
- Metal, CUDA, HIP and OpenCL C sources and `.cgl` sources are unaffected: their signed overflow is undefined.
- New rule families add table rows and templates, not emitter code, and the SMT gate covers them on arrival.
