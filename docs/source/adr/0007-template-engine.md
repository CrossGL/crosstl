# ADR 0007: MSL and HLSL templates go through a typed C++-subset front end in Python, checked against Clang

- Status: accepted
- Date: 2026-11-20
- Scope: the Metal and HLSL importers (`crosstl/backend/Metal/**`, `crosstl/backend/DirectX/**`), a new shared
  front-end package under `crosstl/translator/cxx/`, and the oracle tests under `tests/conformance/`

## Context

MSL is C++17 with exclusions (MSL 4.1 §1.5, §1.5.2 templates, §1.5.4 restrictions). HLSL 2021 adds function and
struct templates, operator overloading and member functions to HLSL (HLSL specification [Overload.Res.Sets]).
Real kernels depend on these features: the MLX, PyTorch MPS and candle sources are written as templates that
explicit instantiations turn into kernels.

The importers do not have a C++ semantic model. The Metal importer materializes templates textually: it rewrites
the source, substitutes template arguments into copies of declarations, mangles names and folds `if constexpr`
branches before parsing the result as a monomorphic program. That pipeline has grown to 201 modules and 82,257
lines under `crosstl/backend/Metal`. 89 of those modules (55,804 lines) take part in materialization, and
`crosstl/backend/Metal/preprocessing` alone is 30 modules and 28,723 lines. The HLSL parser mangles function and
struct templates by name (`function_templates.py`, `struct_templates.py`) and has no specialization model.

The contract asks for one semantic front end for both languages. It must cover templates with real
instantiation, including explicit and partial specialization and constrained overloads; constexpr evaluation;
overload resolution and the usual arithmetic conversions; namespaces; operator overloading; and member functions.
It also asks for this decision to be measured on corpus correctness, on latency for template-heavy kernels and on
the size of the result.

### Measurement setup

- CrossTL at `main` 1ebda13cf, Python 3.12, macOS 27.2 on an Apple M2 Max, corpus manifest c6c6f54e18bf.
- Corpus: `python -m tests.conformance.corpus.run --slices ggml-metal,pytorch-mps,candle-metal,apple-metal,mlx
  --targets metal --workers 6 --timeout 120 --no-gate`. Each Metal source is imported and emitted back as Metal,
  and every artifact is compiled with `xcrun -sdk macosx metal -c`.
- Apple compiler: `xcrun -sdk macosx metal -fsyntax-only` for front-end time, and `-ftime-trace` for
  instantiation counts.
- HLSL: DXC 1.9.0.5512 from Vulkan SDK 1.4.363.0, with `-HV 2021`.
- Upstream Clang: Homebrew LLVM (Clang 23). libclang wheel sizes come from PyPI (`libclang` 18.1.1).

### Corpus correctness at head

Metal to Metal, 240 units:

| Slice | Units | Usable | C++-semantic refusals | Other refusals | Timeouts (120 s) | Invalid or degraded |
|---|---:|---:|---:|---:|---:|---:|
| apple-metal | 89 | 64 | 9 | 15 | 0 | 1 |
| candle-metal | 16 | 10 | 5 | 0 | 0 | 1 |
| ggml-metal | 36 | 15 | 1 | 19 | 0 | 1 |
| mlx | 49 | 22 | 19 | 1 | 7 | 0 |
| pytorch-mps | 50 | 12 | 23 | 15 | 0 | 0 |
| all | 240 | 123 | 57 | 50 | 7 | 3 |

C++-semantic refusals are the diagnostics that a C++ semantic model would remove:

| Diagnostic (`project.translate.*`) | Cells |
|---|---:|
| `metal-template-specialization` | 18 |
| `metal-struct-method` | 17 |
| `metal-constexpr-branch-unresolved` | 8 |
| `metal-template-argument-unresolved` | 4 |
| `metal-struct-alias-unresolved` | 4 |
| `metal-local-type-unresolved` | 3 |
| `metal-source-overload-unresolved` | 2 |

Together with the 7 MLX timeouts, they account for 64 of the 117 cells that are not usable. The largest other
group is `pointer-reinterpret-unsupported` (20 cells, mostly ggml), which a front end does not change. One of the
invalid cells, candle `fill.metal`, is also a template defect: the materialized `fill_with<bfloat>` call no
longer matches any overload.

Two probes isolate the template machinery from the corpus:

- A Metal kernel template with a class template, a member function, an explicit specialization of a traits
  class, `if constexpr` and two explicit instantiations with `[[host_name]]`. CrossTL translates it to CrossGL,
  Metal, HLSL and SPIR-V. `xcrun metal` and DXC accept the outputs. GLSL refuses it with
  `OpenGLRepeatedStageEntryError`, which is correct, because a GLSL unit has one compute entry point.
- An HLSL 2021 compute shader with a struct template, a function template and the static member of an explicit
  specialization (`Traits<float>::is_float`). DXC compiles it to SPIR-V in 0.23 s. CrossTL refuses it on every
  target: the importer mangles the member to `Traits_float__is_float` and never declares it.

### Latency on template-heavy kernels

| MLX kernel | Function / class instantiations (Clang) | Apple front end | CrossTL, 6 workers, 120 s limit | CrossTL, 1 worker, 600 s limit |
|---|---:|---:|---:|---:|
| `quantized.metal` | 4,431 / 524 | 3.5 s | timeout | 468.2 s, then refused |
| `reduce.metal` | 3,005 / 534 | 2.7 s | 91.9 s, then refused | 122.4 s, then refused |
| `sort.metal` | 816 / 503 | 0.3 s | 70.6 s | 87.5 s |
| `softmax.metal` | 35 / 202 | 0.2 s | 3.9 s | 6.2 s |

CrossTL translates Metal to Metal. The six-worker numbers come from the corpus run; the single-worker run used the
corpus runner's worker for one unit at a time, on the same machine while other test jobs ran, so both columns
include contention. Both refusals are `metal-template-specialization`. The time goes into textual rewriting and
re-parsing, and it grows with the number of instantiations rather than with the size of the source.

`tests/conformance/corpus/budgets.json` sets the Metal p95 budget at 300 s, against 5 s for each other source
language. The p95 of this corpus run is 77.2 s.

### What the Clang route needs

- **MSL through upstream Clang.** Clang's C++17 mode rejects address-space qualified member functions and the
  `h` literal suffix. Clang's C++ for OpenCL 2021 mode (`-target spir64 -cl-std=clc++2021`) accepts a
  prototype kernel with address-space qualified methods, `half` literals, explicit specializations and explicit
  instantiations. It needs a shim for the whole Metal standard library, because the SDK headers cannot be
  redistributed, and for Metal attributes. A macro shim such as `#define buffer(n) ...` or `#define thread`
  also rewrites user identifiers called `buffer` or `thread`, so it is unsound without a Clang patch or plugin.
- **MSL through Apple's compiler.** `xcrun metal` is Clang, but it exists only on macOS and Windows, and it has
  no library interface. Its JSON AST dump of `quantized.metal` is 1,904,938,234 bytes.
- **HLSL.** DXC is a Clang fork with HLSL 2021 templates. Its AST dump contains the instantiated
  specializations. It has no Python interface and no PyPI package. Upstream Clang's HLSL mode needs `hlsl.h`,
  which the Homebrew build does not ship, and it is still incomplete upstream.
- **libclang.** The Python bindings come as native wheels of 20.3 to 26.5 MB per platform, and the last release
  is 18.1.1. Cursor traversal does not visit implicit template instantiations, so the bindings cannot return
  instantiated bodies. The only complete route is the JSON AST, at the size above.

## Options

**(a) Clang-based front end.** Clang's HLSL mode, and MSL through Clang C++ with shimmed Metal headers.

- Correctness: Clang is the reference semantics for C++, and DXC already parses the HLSL probe that CrossTL
  refuses.
- Latency: the Apple front end is 20 to 290 times faster than CrossTL on the MLX kernels above.
- Size and portability: a 20 to 26 MB native dependency per platform, or a JSON AST of up to about 2 GB per
  unit. There is no in-process route to instantiated bodies, and no upstream Clang mode that accepts MSL without
  an unsound shim. `pip install crosstl` would no longer translate MSL or HLSL on a host without the native
  dependency, and the output could depend on which Clang the host has.

**(b) Typed C++-subset front end in Python.** One parser, semantic analyser and instantiator for the C++ subset
that MSL 4.1 and HLSL 2021 share, with a language profile for each.

- Correctness: has to be built and tested; Clang does not vouch for it.
- Latency: instantiation is memoized on the canonical template arguments and runs on the typed AST, so no source
  text is rewritten or parsed again. The instantiation counts above (at most 4,431 functions and 534 classes per
  unit) bound the work.
- Size and portability: pure Python, identical output on every host, and it replaces the textual
  materialization layers instead of adding to them.

**(c) Hybrid.** The engine of option (b) translates, and Clang checks it.

- Correctness: Clang-family compilers act as oracles in the conformance tests. They report which
  specializations an entry point instantiates, the values of constant expressions and the selected overloads,
  and the tests compare these with the Python engine.
- Latency and size as in (b). Clang runs only in CI and in local conformance runs, never inside
  `crosstl.translate`.

A hybrid that uses Clang to translate when it is installed and Python otherwise was rejected: the output would
depend on the host, and both paths would need full maintenance.

## Decision

Option (c), as above: a typed C++-subset front end in Python is the only translation path for MSL and HLSL, and
Clang is its test oracle.

1. **Package.** `crosstl/translator/cxx/` holds the shared engine, in modules of at most 1,500 lines. It reads
   the output and line map of the conforming preprocessor and has declarations and a typed AST, name lookup
   (namespaces, class scope, `using`, argument-dependent lookup for operators), types and conversions, overload
   resolution, template argument deduction and partial ordering, an instantiator, and a constexpr evaluator.
   `profiles/msl.py` and `profiles/hlsl.py` hold the language differences: address spaces, attributes, built-in
   types and the standard library declarations. A test checks each profile's declarations against the
   inventories in `tests/conformance/inventory/data`.
2. **Semantics follow the C++ rules the languages inherit.** Deduction ([temp.deduct]), partial ordering of
   specializations ([temp.class.order], [temp.func.order]), overload resolution and conversion ranking
   ([over.match], [over.ics.rank]), the usual arithmetic conversions ([expr.arith.conv]) as amended by MSL 4.1
   §2 and the HLSL specification, and constant evaluation ([expr.const]). Where a language excludes a C++
   feature (MSL 4.1 §1.5.4), the profile refuses it with a typed diagnostic.
3. **Instantiation is semantic.** Each specialization is created once per canonical argument list, from the
   entry points and explicit instantiations outward, and is lowered to CrossGL as a monomorphic declaration with
   a deterministic name. No pass rewrites source text.
4. **Failure is typed.** Anything the engine cannot resolve raises a diagnostic with the source location and the
   rule it could not apply. Nothing falls back to the textual path once a construct has moved to the engine.
5. **The oracle.** A conformance suite compiles each probe and corpus unit with the Clang-family compiler of its
   language: `xcrun metal` on macOS and DXC on every platform. It compares the instantiated specializations,
   constant values and selected overloads with the engine's. A mismatch fails the test. Corpus units take this
   check in the nightly tier, because Clang's AST output is large.

## Order of work

1. Move every importer onto the conforming preprocessor in `crosstl/translator/preprocessor` first, because the
   engine consumes its output and line map. The HLSL importer already expands macros with it. The Metal importer
   (`crosstl/backend/Metal/preprocessor.py`, 3,927 lines) and the GLSL importer still preprocess on their own.
2. Build the engine and the HLSL profile, with HLSL 2021 templates as the first consumer, because HLSL has the
   smaller template surface and DXC runs on every CI platform. The HLSL probe above becomes its first test.
3. Add the MSL profile and move the Metal importer onto the engine one construct family at a time, in this
   order: class templates and member functions, explicit and partial specialization, `if constexpr` and
   constexpr evaluation, then overloads and aliases. Each move deletes the textual materialization modules it
   replaces in the same PR, and keeps the Metal round-trip corpus share and the MLX and ComputeSharp suites
   green.
4. When the last textual pass is gone, lower the Metal p95 latency budget to the value the corpus measures and
   record it in `budgets.json`.

## Consequences

- `crosstl.translate` stays pure Python and produces the same output on every host. Clang is a test dependency
  only.
- The textual materialization layers are removed, not wrapped. Until a construct family has moved, its current
  refusals stay in place, so no route gets weaker during the transition.
- The engine's correctness rests on the oracle tests and on the corpus. A construct that the oracle cannot check
  on a platform (for example `xcrun metal` off macOS) is checked by the macOS CI job.
- The 57 C++-semantic refusals, the 7 MLX timeouts and the candle `fill.metal` invalid output are the measured
  targets for M4. The HLSL probe and the MLX latency table are re-measured with each step.
