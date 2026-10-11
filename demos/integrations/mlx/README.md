# MLX Integration Demo

MLX is an external test corpus for CrossTL's repository translation and runtime
APIs. Its host adaptations, pinned contracts, execution evidence checks and tests
belong to this demo; the translator does not depend on MLX.

## Directory Guide

| Path | Purpose |
| --- | --- |
| `project.toml`, `run_porting.py` | Repository scan and translation harness |
| `contracts/` | Pinned discovery, artifact and dispatch contracts |
| `portable_host/` | DirectX/OpenGL host adapter and bounded execution checks |
| `run_metal_host.py`, [METAL_HOST.md](METAL_HOST.md) | Generated Metal host integration |
| `run_native_metal.py`, [NATIVE_METAL.md](NATIVE_METAL.md) | Unchanged upstream Metal reference |
| `tests/` | Harness, contract, discovery and CI tests |
| `tests/host/` | Portable host adapter tests |
| `tests/kernels/` | Pinned kernel translation and native execution tests |
| `tests/fixtures/` | MLX-specific reference programs |
| `fixtures/project_porting/` | Reduced source-specialization contracts |
| `fixtures/runtime_verification/` | Runtime inputs, expected values and artifact selectors |

Generic project and runtime tests also consume these reduced fixtures. Keeping
the inputs here preserves their MLX provenance without making the core fixture
tree project-specific. Moving a fixture does not change its shader or expected
numerical results.

Run the demo's local tests from the repository root:

```sh
python -m pytest -q -n auto demos/integrations/mlx/tests
```

Native execution remains opt-in locally and mandatory in its corresponding CI
job. The [Project Demo Testing workflow](../../../.github/workflows/demo-project-testing.yml)
contains the Windows DirectX, Linux OpenGL and macOS Metal jobs, including the
reference baseline and host integration checks. Each job retains its own pinned
source revision, required toolchains, numerical assertions and evidence uploads.
Complete DirectX corpus compilation uses pinned DXC on Ubuntu; Direct3D runtime
execution remains on Windows. Metal compilation and execution remain on macOS.
See the [CI coverage policy](../../../.github/TESTING.md) for the platform split.
Scheduled runs cover the existing corpus audits; changes to translation code,
tests, demo inputs or toolchain configuration run the complete workflow on pull
requests and main-branch pushes. Each job and matrix leg has its own queue.
A newer PR revision cancels the superseded workflow run, including active native
jobs, to free runners for the current code. Main, scheduled and manually
dispatched runs are not cancelled by this PR policy. No test matrix or numerical
gate is removed. A previous revision's successful proof does not satisfy the
current revision's required checks.
Core and demo tests also remain part of the complete test suite.

### Full-Entry GEMM Execution

`tests/kernels/test_current_gemm.py` translates the unchanged
`steel_gemm_fused_nn_float32_float32_bm32_bn32_bk16_wm2_wn2` entry at commit
`9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8` through the project API, package
verification and native Metal or DirectX loader. Both targets compare generated
execution with an exact matrix-product reference. Metal additionally compares
with the independently compiled upstream kernel; Windows does not claim to run
the original Metal control.
Fifteen cases cover scalar, partial-tile, aligned-tile and multi-tile matrices,
contiguous and two-dimensional broadcast batches, padded rows and batch strides,
and extra workgroups. Output padding and eight trailing guard values must remain
unchanged.

The six Boolean function constants are explicitly frozen for each of six
configurations: plain multiplication, addition of an output source, scaled
fused output, broadcast batches, broadcast batches with scaled fused output,
and fully aligned dimensions. The output-source cases use noncontiguous columns
and padded rows; the scaled cases use alpha 0.5 and beta -0.25. Broadcast inputs
use different batch axes for A and B, and a shared C for the fused variant.
The original Metal kernel receives matching native function constants.
The test uses the source-backed fragment mapping documented in
`contracts/cooperative-matrix-fragment-mapping.json`, a 32-by-2-by-2 workgroup,
and reflected parameter layouts. It does not edit upstream or generated kernels,
or substitute a reference shader. DirectX uses two source-backed workgroup
access assertions for the complete vector loads in `steel/gemm/loader.h`.
For this entry, 128 threads load four adjacent floats each into A's 32-by-20
and B's 16-by-36 shared arrays. The test enumerates every destination, proves
unique coverage of the live elements, and bounds the addresses to 0..635 and
0..571 respectively. Both this header and `steel/gemm/gemm.h` are hash-checked
against the pin before execution. No unchecked index ranges are supplied.
The original source is compiled with Metal 3.2 and the C++17/C++20 extension
warning exceptions from upstream's kernel build. Both original and generated
compilation disable fast math and treat remaining warnings as errors.

The existing macOS and Windows project-porting jobs run this required check and retain
translation reports, packages, compiler logs, input records, compiled modules,
readbacks and per-case parity results. Windows requires DXC compilation with
warnings as errors and Direct3D 12 execution. Quarter-integer inputs make these
products and sums exactly representable; this gate does not replace the separate
rounding-sensitive cooperative-matrix controls tracked by #2193.
Run it locally against a clean checkout
of the pinned kernel tree:

```sh
mkdir -p gemm-results
CROSTL_MLX_CURRENT_ROOT=/path/to/mlx \
CROSTL_REQUIRE_MLX_CURRENT_GEMM=1 \
python -m pytest -q -n auto \
  --basetemp=gemm-results/pytest --junitxml=gemm-results/results.xml \
  demos/integrations/mlx/tests/kernels/test_current_gemm.py
```

This proves one float32 NN entry and six function-constant configurations, not
the complete GEMM family or the upstream MLX suite. Transposed operands, other
element types and every mixed alignment combination are not covered. OpenGL
collective participation and live-offset bounds remain separate translation
gaps tracked by #2186 and #2179. This check does not establish full host-runtime
redirection for any backend. A configured gate is not proof of success until
its native job passes for the relevant revision.

### Resource-Index Controls

Guarded resource-index controls execute on original/generated Metal and on
Mesa 25.0.7 llvmpipe with exact results and output guards; the same 31 HLSL
artifacts compile with strict DXC settings. The local ARM64 Mesa 22.3.6 driver
returned incorrect results for a signed 64-bit conditional comparison. Updating
the validation environment resolves that case without changing generated GLSL.
This driver result is not evidence of complete MLX runtime coverage.

These controls cover scalar guards and integer fields of by-value records,
including nested records, copies, helper calls and preconditioned loops. Field
mutation and escaping references invalidate proofs; an unrelated field's bound
does not justify an access. Chained offsets cover reads, writes and atomic
destinations, including contended atomic updates. Finite nested value calls
preserve their bounds without authorizing recursive definitions or unsafe
intermediate arithmetic. The Metal round trip also preserves 8-bit and 16-bit
aggregate initializer conversions, including truncation and single evaluation.
Runtime-derived GEMM metadata still requires a checked source or host contract;
these controls do not supply arbitrary bounds for it.

The host adaptation documentation records upstream changes explicitly. A passing
kernel or bounded host check is not a claim that the entire upstream MLX suite
passes on a translated backend.

The selected tiled GEMM entry at pin `9c3d3557` now passes explicit `auto*`
deduction through the public project pipeline, without changing upstream
sources or bypassing specialization analysis. It now produces a Metal artifact:
`MMATile::frag_at(i, j)` aliases retain their original storage and binding-time
indices, including writes through `accum[k]`. The selected float32, non-transposed
entry now passes strict Metal compilation with the six Boolean function constants
set to false. Source-used parameter intent survives specialization and explicit
discard lowering ([#2164](https://github.com/CrossGL/crosstl/issues/2164)).
This is a compilation result, not numerical matrix execution or coverage of all
GEMM specializations. The constant
`GEMMParams` and `GEMMAddMMParams` pointers now retain their indirection, address
space and bindings ([#2160](https://github.com/CrossGL/crosstl/issues/2160)).
Conditional scalar-buffer parameters now retain their bindings and conditions
([#2153](https://github.com/CrossGL/crosstl/issues/2153)). The pointer-deduction
controls cover native Metal/OpenGL execution and strict HLSL compilation;
Windows execution is a separate required CI check. Pointer-object qualifiers
and foreign-target private aliases remain explicit diagnostics under
[#2152](https://github.com/CrossGL/crosstl/issues/2152) and
[#2055](https://github.com/CrossGL/crosstl/issues/2055). These results do not
establish matrix host dispatch or complete upstream-suite parity.
Mixed-access HLSL helper calls remain separately blocked by
[#2158](https://github.com/CrossGL/crosstl/issues/2158): a read-only view of a
writable buffer must preserve the physical resource class when passed to a
helper. The passing execution controls do not cover that unresolved combination.

Conditional-resource controls execute all four combinations of two Boolean
constants on Metal and OpenGL, with sparse binding indices, scalar constant
references and output guards. Metal results also match the original source.
OpenGL execution includes the specialized binary load path tracked in
[#2159](https://github.com/CrossGL/crosstl/issues/2159). Eight scalar/aggregate
HLSL variants have separate strict DXC compilation checks; Windows numerical
execution remains a required CI gate. These controls currently supply all
reflected bindings, including inactive inputs. Conditional aggregate references
compile, but their native packing remains under
[#1991](https://github.com/CrossGL/crosstl/issues/1991).

Constant aggregate pointer controls use read-only structured buffers on DirectX
and OpenGL, preserving element indexing instead of treating every pointer as a
single constant-buffer object. Twenty-four cases execute on Metal and OpenGL
through project packages and reflected two-member layouts. They cover offsets,
single evaluation, local aliases, helper calls, constructors, conditional
specializations, wraparound and output guards. Generated Metal matches original
Metal; all twenty-four HLSL variants compile with strict DXC checks. Windows
numerical execution is required separately. These homogeneous-record controls
do not establish heterogeneous aggregate packing, inactive-binding omission,
complete GEMM translation or matrix host dispatch.

Unnamed helper and constructor parameters retain their intent through saved
CrossGL and Metal round trips ([#2161](https://github.com/CrossGL/crosstl/issues/2161)).
Seven native controls on Metal and OpenGL verify argument side effects, defaults,
overload selection and output guards. Strict Metal compilation keeps unrelated
unused-parameter warnings enabled. These controls do not resolve the remaining
matrix helper materialization and host-dispatch requirements.

Parameter-use controls cover synthesized receivers, object-based static calls,
type-only expressions, explicit discards, local type aliases and compute-input
bindings. Thirteen cases execute on Metal and OpenGL with exact readback,
argument side-effect counts and output guards; Metal also executes the original
sources. Genuine unused named parameters remain warning-fatal. The same cases
are required in the Windows native job, separately from local HLSL compilation.

Partial-specialization static-owner controls cover primary forward declarations,
default arguments, scoped aliases, constructor initializers and method calls
([#2157](https://github.com/CrossGL/crosstl/issues/2157)). Thirteen cases execute
on Metal and OpenGL with exact readback, side-effect counts and output guards;
Metal also executes the original sources. All thirteen HLSL variants compile
with strict DXC checks, with Windows execution required separately. These controls
do not establish complete GEMM compilation or numerical matrix execution.

Aggregate-field receiver controls cover implicit fields, explicit `this`,
parameter and local shadowing, sibling templates, call operators, nested fields
and multidimensional arrays
([#2162](https://github.com/CrossGL/crosstl/issues/2162)). Thirteen cases execute on
Metal and OpenGL, including read-only receivers, argument/index side effects,
wraparound and output guards; Metal also executes the unchanged source controls.
All thirteen HLSL variants compile with strict DXC checks. Windows execution
remains a separate required gate in the existing native job.

Reference-capture controls cover mutable accessor aliases over nested value
fields ([#2163](https://github.com/CrossGL/crosstl/issues/2163)). Thirteen cases
execute on Metal and OpenGL, including changed indices, index side effects,
shared aliases, loop-local bindings, narrow parameter conversions, deduced
constness and output guards. Generated Metal matches the original sources, and
all thirteen HLSL variants compile with strict DXC checks; Windows execution
remains required separately. Unsupported escapes and alias uses are diagnosed.
These controls do not establish complete GEMM execution or upstream-suite parity.

Reduction reference review found an alias-specialization error
([#2100](https://github.com/CrossGL/crosstl/issues/2100)): `float16_t` selected
the primary `Limits` template instead of the explicit `half` specialization.
The generic translator correction restores the positive/negative infinity
initializers for `init_reduce_minfloat16` and `init_reduce_maxfloat16` at both
the historical `846d1762` and current `9c3d3557` pins. All twelve regenerated
Metal/HLSL/GLSL artifacts compile; Metal and OpenGL each pass 68 exact output
values and 64 guard values. Windows execution remains separate. This bounded
check does not accept the complete reduction family's artifact references or
establish its numerical coverage.

The same review exposed a remaining half-precision NaN predicate in HLSL
([#2008](https://github.com/CrossGL/crosstl/issues/2008)). Optimized DXC folded
`value != value` to false inside min/max reduction helpers. Binary16 payload
classification now preserves that predicate. Eight historical whole-array and
row artifacts differ only in the predicate; those and four current-pin artifacts
strictly compile with input-dependent votes retained in DXIL. Generic native
controls cover all 65,536 half payloads on Metal and OpenGL; Windows execution
remains a separate required check. No complete reduction reference set is
accepted from these results.

Eight byte-valued small-row artifacts exposed a writable-argument regression
([#2101](https://github.com/CrossGL/crosstl/issues/2101)). The HLSL correction
preserves the accumulator's storage location instead of narrowing it into a
temporary. All eight historical artifacts now pass strict DXC compilation;
their interfaces are unchanged and each body differs only at the writable
`total_val` arguments. The same eight entries at the current `9c3d3557` pin
also pass strict DXC compilation. This does not accept the full reduction
reference set or establish whole-kernel numerical parity. Upstream kernels are
unchanged.
Generic reference controls cover signed/unsigned overflow, aliases, nested
calls, scalar/vector values and indexed calls under conditionals and loops.

The subsequent complete historical HLSL reduction review covers all 2,396
entries and 39 shapes. Exact old sources were reproduced before comparing every
changed function and declaration: 2,302 artifacts change and 94 remain identical.
All 4,792 old/current strict DXC compiles pass, and a second independent compile
of every current artifact reproduces all 2,396 DXIL modules byte for byte.
The 9,216 materializations and 27,382 reflected resources remain covered.
Among them, 384 logical-half resources in 232 entries retain their two-byte
stride while using explicit unsigned-16 storage encoding. Other binding, access,
layout and entry-point fields agree; a non-required subgroup-size constant still
evaluates to 32. The recorded compiler identity is the pinned Linux DXC binary.
These are translation, interface and compiler checks, not whole-family numerical
execution or current-pin upstream-suite parity.

The separate Metal exponential review exposed numeric grouping loss in source
conversion and target generation
([#2104](https://github.com/CrossGL/crosstl/issues/2104)). Both stages now preserve
addition, multiplication and bitwise expression trees. Generic controls compare
cancellation, rounded products, signed zeros and mixed-width integer conversions
on native Metal and OpenGL, including saved CrossGL intermediates and output
guards. The exponential accuracy checks retain their existing tolerances.
The 15 affected Metal exponential references remain unchanged pending corpus
reconciliation; hosted Windows execution is verified separately.
For the current pinned, profiled bfloat Sigmoid case, only the Metal helper's
result parentheses change. Its reviewed artifact still matches original Metal
over all 65,282 non-NaN inputs, with eight output guards. The HLSL and GLSL
artifacts are byte-identical to their previous references; the OpenGL full-domain
execution also remains exact. The later precision-scope correction moves
`contract(off)` into the precise exponential body and removes the file-level
`contract(fast)` reset. Complete shader comparison leaves all other bytes and
the interface unchanged. Before accepting the new Metal fingerprint, the old
artifact, new artifact and unchanged upstream source each returned the same
65,282 results and eight guards, with read-only buffers unchanged. This review
does not accept the remaining exponential or complex-unary family references.
A subsequent unused-receiver annotation changes only the Metal source fingerprint.
The previous and annotated shaders compile to identical library bytes and both
match the unchanged upstream kernel across the same 65,282 inputs and eight
guards. Read-only buffers and the reflected interface remain unchanged; no
other corpus reference is updated by this review.

Cross-backend controls also found a separate Metal narrow-field reference
failure ([#2102](https://github.com/CrossGL/crosstl/issues/2102)): the field
retains byte storage but its writable helper parameter is widened. Unsigned
byte-vector compound assignment remains a distinct arithmetic limitation under
[#2023](https://github.com/CrossGL/crosstl/issues/2023). Passing DirectX/OpenGL
reference tests does not establish Metal round-trip support for those cases.
Shared source references also require alias-aware lowering
([#2103](https://github.com/CrossGL/crosstl/issues/2103)). The byte-argument fix
rejects potentially overlapping reference arguments rather than relying on
HLSL copy-in/copy-out; wider reference aliasing remains unresolved.

## Corpus Revisions

This directory contains the project-level MLX porting checks used by CrossTL.
The checks are pinned to MLX commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e` and exercise the Metal kernel tree
as a source repository, not as isolated parser snippets. This pinned revision is
an active repository-level verification target: configured coverage and expected
baselines are not, by themselves, evidence that every kernel translates or
passes a target validator.

The next corpus increment is separately pinned to
`9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8`. Its authoritative discovery census
is 49 Metal units, 17,832 entries, and zero discovery diagnostics. The compact
`contracts/arg_reduce.current-tree.translation.json` contract pins all 24
entries from `arg_reduce.metal` and all 72 deterministic Metal, OpenGL, and
DirectX artifacts. Required CI compiles every entry with the applicable native
validator. The common numerical subset is float32:
`argmin_float32` and `argmax_float32` run on Metal, Mesa EGL, and Direct3D 12
WARP over axis sizes 32 and 129, strides 1 and 2, ordinary values, and
NaN/Infinity inputs; macOS additionally compares against the upstream metallib.
All three backends also require all twelve signed/unsigned 16/32/64-bit arg-reduce
entries across 96 integer cases: axis sizes 31, 32, 33 and 129, both strides,
extrema, lowest-index ties and partial groups. Metal compares unchanged upstream
and generated kernels. DirectX uses exact-width software shuffles, including
resolved signed and unsigned overloads around two-word 64-bit payloads.
OpenGL widens 16-bit buffer elements to 32-bit storage; DirectX retains native
16-bit storage. Runtime inputs are checked against each reflected ABI.
This is a 24/17,832 deterministic translation and native-compiler increment with
14/17,832 common required numerical entries, not full-tree
coverage, upstream MLX test-suite execution, or MLX host-runtime redirection.

The current OpenGL identities include explicit signed 64-bit remainder,
source-typed half loads, and bfloat rounding at shuffle and sentinel boundaries.
All 24 bodies were reviewed against the retained sources and compiled with
`glslangValidator` and `spirv-val`. The 16 float32 and 96 integer cases are
required on Mesa EGL. An additional local review executed 80 half/bfloat cases with source-rounded
inputs, including rounding ties, subnormals, signed zeros, NaNs, and infinities;
these do not expand the contract's required CI runtime subset.

The Metal identities have a separate review of narrow aggregate storage and
byte-field initialization. All 24 generated entries compile with `-Werror`.
The eight changed bodies retain native byte/short fields; byte constructors
convert promoted values back to the declared field type. Source identity,
specialization counts, workgroup metadata and the 16 other bodies are unchanged.
A local execution review compares the four byte entries against the unchanged
upstream metallib and independent integer indices across 64 cases, including
extrema, ties, constant rows, tails and strided inputs. These cases supplement,
but do not expand, the required float32 CI runtime subset.

The four 16-bit Metal references additionally retain explicit `short`/`ushort`
conversions at aggregate construction. Removing only the two field conversions
in each entry reproduces its prior accepted source hash. The reviewed 32/64-bit
changes preserve typed extrema; recreating the preceding translator revision
reproduces every prior accepted hash. The 96 required integer cases above
validate these bodies against the unchanged upstream metallib, including
64-bit inputs above the 32-bit range. No upstream source adaptation is applied.

A subsequent unused-parameter annotation update changes all 24 Metal source
hashes without changing their compiled libraries. The review reproduces every
previous source identity and checks unchanged source, specialization and
workgroup contracts before accepting the updated references.

The DirectX identities have a separate complete-body review and strict DXC
compilation for all 24 entries. Changes preserve source-width index arithmetic,
byte conversions at aggregate and call boundaries, the negative-infinity
sentinel, and logical half loads from integer storage. Source hashes, template
materializations, resource registers, launch geometry, reduction comparisons
and tie-breaking remain unchanged. The 32-lane software subgroup path remains
required for the two float32 entries. The updated identities allow the existing
Windows numerical gate to run. Its 16 cases pass with the reviewed shaders;
an independent audit of retained uploads and outputs verifies all 32 reduction
indices, parameter bindings and executed module identities. This confirms the
existing float32 subset, not numerical coverage of all 24 scalar entries.

The four DirectX byte-entry references additionally include logical element
addressing for widened byte buffers. Their complete-body diffs replace packed
word extraction with one indexed resource read, retaining signed-byte extension.
Source hashes, template materializations, resource declarations and workgroup
metadata are unchanged. All four updated artifacts pass strict DXC compilation;
this reference review does not add native byte-entry numerical coverage.

Original Metal reference libraries use the unchanged upstream sources with
compiler warnings retained but not promoted to errors. The pinned headers use
C++17 constructs that some Metal 3.1 toolchains diagnose as extensions.
Generated Metal libraries still compile with `-Werror`; both paths retain
`-fno-fast-math` and the same numerical assertions. No upstream headers are
modified to accommodate the reference compiler.

A separate local FP8 review at `9c3d3557` compares all six scalar conversion
entries against unchanged upstream Metal: all 65,536 half and bfloat input
words, 1,024 float32 boundary inputs, and all 256 encodings for each decoder.
Generated Metal matches the original readback bytes. OpenGL matches the other
five entries, but its half encoder differs for all 1,023 negative NaNs:
the Metal reference returns `0x7e`, while OpenGL returns `0xfe`. Output guards
remain intact. Encoder comparisons are byte-exact; decoder comparisons retain
exact finite values and signed zeros but compare NaNs by classification.
An isolated half-to-float conversion followed by integer bit inspection
reproduces the source-profile gap tracked in
[crosstl#2081](https://github.com/CrossGL/crosstl/issues/2081). This is not a
lossless-copy defect or permission to canonicalize uploads. These local probes
do not establish Windows FP8 execution or expand the required CI runtime set.

The arithmetic correction for
[crosstl#2082](https://github.com/CrossGL/crosstl/issues/2082) retains bfloat
intermediates instead of inferring half precision from their storage width.
Mixed integer operands are converted before arithmetic; each narrow result
boundary remains observable. Nineteen reduced cases match unchanged original
Metal and generated Metal/OpenGL over 228 values per path, with exact output
bits and guards. Seventeen supported HLSL cases compile with warnings fatal
and pass Windows numerical execution. The two
bfloat-vector cases retain DirectX unsupported-type diagnostics rather than
claiming execution coverage.

The math-wrapper correction for
[crosstl#2084](https://github.com/CrossGL/crosstl/issues/2084) preserves float
computation and the source's bfloat return before subsequent arithmetic.
Nine reduced default, fast and precise wrapper cases match original Metal and
generated Metal/OpenGL over 108 values per path; all nine HLSL cases compile
with warnings fatal and pass Windows numerical execution. Independent review
of the [Windows artifacts](https://github.com/CrossGL/crosstl/actions/runs/37305074470/job/111746672534)
checks all 26 arithmetic/math cases against original Metal readbacks: 312
values, 52 guard words and fresh matching shader identities. They run within
the existing native conversion jobs.
Fresh current and historical `v_Sigmoidbfloat16bfloat16` entries compile to all
three targets. Local Metal execution matches the unchanged original kernel for
all 65,282 non-NaN bfloat inputs at each pin, including infinities and signed
zeros. Inputs and guards are checked; outputs are not preloaded with expected
values. The same OpenGL sweep has four differences with intact guards:
one exponential rounding midpoint
([crosstl#2085](https://github.com/CrossGL/crosstl/issues/2085)) and three
subnormal division results
([crosstl#2086](https://github.com/CrossGL/crosstl/issues/2086)). Independent
stage and division-only probes retain these failures. This does not establish
Windows Sigmoid parity or resolve the separate NaN policy.

The portable precise-exponential lowering and explicit
`binary32_division_profile = "rne-flush"` source option resolve those four
differences at the current pin. Generated Metal and OpenGL now match the
unchanged original Metal kernel for all 65,282 non-NaN bfloat inputs, with
guards intact. An independent Decimal reference, rounded at the source's
binary32 and bfloat boundaries, also matches every original readback. The
division profile is retained in the saved project configuration, report and
runtime-artifact provenance. The numerical test checks the generated source
hash and size against the reviewed artifact for each target before execution.

The existing native project job now checks that full domain and eight trailing
guards on each platform, instead of adding a separate runner. It retains 19
explicit original-Metal boundary results, including the exponential midpoint
and division underflow neighbors. Required primitive exponential tests also
exercise 16,132 binary32 inputs, scalar/vector calls and exact narrowing of
the computed float result against a 100-digit reference. The
[Windows native job at `c7579e41`](https://github.com/CrossGL/crosstl/actions/runs/37384697268/job/112014981317)
passes all 48 unary checks without skips. Independent retained-artifact review
verifies all 65,282 Sigmoid results against original Metal, eight output guards,
the saved division profile, and generated-source/DXIL identities. The repaired
half/bfloat exponential alias cases also pass: 90 result words and 64 guards
match an independent 120-digit reference within the existing two-ULP bound,
with single evaluation preserved. The same job passes the 34 selected upstream
host tests and 1,208 translated dispatches; independent review verifies 40,872
values on each CPU and DirectX path. This is not full MLX-suite parity, a
universal device-equivalence claim or a NaN payload policy. No upstream kernel
is changed.

Mixed integer-lvalue compound assignments retain both the integer-to-bfloat
conversion and the rounded arithmetic result before integer writeback
([crosstl#2083](https://github.com/CrossGL/crosstl/issues/2083)). Forty-eight
reduced cases cover signed and unsigned 32-bit integers, aliases, local and
resource arrays, and struct members. Original/generated Metal, generated
OpenGL and Windows DirectX match over 576 values per path. The Windows artifact
audit verifies all 48 cases, 60 output buffers and 120 guard words against the
unchanged original Metal controls, with fresh generated-source identities
([native Windows run](https://github.com/CrossGL/crosstl/actions/runs/37308563739/job/111758088868)).
The tests use existing native CI jobs. Unproven wide-integer conversions and operands
with observable side effects produce structured diagnostics; these checks do
not claim complete mixed compound-assignment support.

The Metal unary references have a complete review of all 877 artifacts:
538 are unchanged and 339 retain explicit byte conversions, Boolean promotions,
bfloat computation boundaries or the existing precise math implementations.
All 877 compile with warnings fatal. Their 3,363 reflected resources and 1,243
materializations are unchanged. The separate scalar contract and ArcCos loader
identity match the same reviewed artifacts; no source selection or tolerance
was changed.

A subsequent complete review recompiles all 877 retained outputs with warnings
fatal. Fifteen exponential entries now use the already reviewed precise-exp
helper with source expression grouping preserved; the other 862 are byte-for-byte
unchanged. All host interfaces and materialization counts remain unchanged.
The three affected scalar-subset identities match the complete family. This
reference update does not expand numerical tolerances or claim full-suite parity.

A local original/generated Metal comparison covers 69 changed operator/type
selections and 1,863,009 values per path. Fifty-three selections match exactly;
sixteen have floating-point bit differences. Sampled scalar float32 results
satisfy their existing four-ULP bounds. Required native precision checks also
verify exact narrowing of each computed float result near half/bfloat rounding
midpoints. They retain the original Metal controls separately, without claiming
bitwise transcendental parity. Complex differences and wider source/target
accuracy policy remain tracked in
[crosstl#2068](https://github.com/CrossGL/crosstl/issues/2068).
Whole-family DirectX/OpenGL reference reconciliation remains incomplete.
Required Square and ArcCos numerical cases remain separate and unchanged.

A separate local binary-math comparison at `9c3d3557` covers all 65,536 half
or bfloat input payloads against 12 fixed partners: signed zero, positive and
negative 0.5, 1, 2, 3, and infinity. Each selected `vv_` entry executes 786,432
pairs through unchanged original Metal, generated Metal and generated OpenGL.
The Metal round trip matches every comparison; OpenGL results are:

| Entry | Differing results | Output guards checked |
| --- | ---: | ---: |
| `vv_LogAddExpfloat16` | 1,123 | 112 |
| `vv_LogAddExpbfloat16` | 746 | 112 |
| `vv_Remainderfloat16` | 0 | 112 |
| `vv_Remainderbfloat16` | 271,292 | 112 |

Comparisons preserve finite bits, infinities and signed zeros; NaNs are compared
by classification, not payload. The remainder count includes signed-zero and
special-value differences, including 512 zero-to-infinity results. A reduced
float kernel confirms that translating `fmod` to GLSL `mod` loses source
semantics ([crosstl#2097](https://github.com/CrossGL/crosstl/issues/2097)).
LogAddExp also loses some nonzero results; its accuracy investigation remains
under [crosstl#2068](https://github.com/CrossGL/crosstl/issues/2068). A reduced
compensated-logarithm expression isolates lost binary32 rounding dependencies
in GLSL ([crosstl#2098](https://github.com/CrossGL/crosstl/issues/2098)).
The nested half/integer arithmetic gap in
[crosstl#1992](https://github.com/CrossGL/crosstl/issues/1992#issuecomment-6006266688)
now has source-width operand conversions and intermediate rounding in GLSL.
The generic regression executes all 65,536 half input encodings through nested
division, vector arithmetic, comparisons and conditional selection. Original
and generated Metal and Linux OpenGL each match 917,504 reference values,
including single-evaluation and unselected-branch controls; all 96 output
guard words pass. Metal uses `-fno-fast-math` for this rounding contract, and
NaNs are compared by classification, not payload. The checks run within the
existing native jobs without adding runners. A separate execution of the
unchanged historical `v_Sigmoidfloat16float16` entry at `846d1762` matches
original/generated Metal across all 63,490 non-NaN half inputs. Its generated
OpenGL still differs on four inputs; all guards pass. These remaining numerical
differences stay under #2068, and the historical GLSL reference is not accepted.
An independent HLSL conditional narrowing case fails strict DXC compilation
and remains tracked in [crosstl#2099](https://github.com/CrossGL/crosstl/issues/2099);
the GLSL correction does not establish support for that DirectX case.
All four binary-math
HLSL artifacts compile with strict DXC options and all four GLSL artifacts pass
glslangValidator and SPIR-V validation. These compiler results do not resolve
the numerical differences. The four-entry binary-math review adds no required CI case, changes
no reference identity or tolerance, and establishes no Windows numerical result.

The alias-bitcast defect in
[crosstl#2088](https://github.com/CrossGL/crosstl/issues/2088) is corrected by
resolving logical bfloat aliases before OpenGL bitcast lowering and resolving
elided local/namespace aliases in Metal bitcast type arguments. Generic translator
tests exercise every 16-bit payload through six paths, including qualified
aliases, helper returns, nested bitcasts and an operand evaluated exactly once.
Original/generated Metal and OpenGL execution preserve all 65,536 payloads,
including NaNs, without numerical tolerances. Separate exact widening controls
cover namespace aliases that conflict with an outer float alias. The required
[Windows checks](https://github.com/CrossGL/crosstl/actions/runs/37321634266/job/111801894608)
also pass: retained DXIL identities and readbacks verify the same exact alias
payloads and the precise-result narrowing controls above. These tests reuse
the existing native jobs. Precision tests retain explicit
input bitcast types to isolate arithmetic from alias resolution.

The complete unary Metal compile gate generates sources in five Ubuntu shards
and compiles all 877 verified artifacts in one macOS job. It retains every
entry, template-materialization and resource-interface check. Source reports
and JUnit results survive failed reference checks; missing, duplicate or
changed artifacts cannot pass the native consumer. This saves four macOS
runner starts without replacing any numerical execution gate.

The current pin adds cross-entropy, gated-delta forward and backward kernels
(including NAX variants), matrix-multiplication gather offsets, and attention
backward kernels. These files are included in discovery; they are not covered by
the arg-reduce translation and numerical checks above.

A separate native gate exercises all 15 discovered `Powercomplex64` entry
shapes at the same `9c3d3557` pin. Its 873 complex outputs cover scalar/vector
broadcasting, multidimensional grids, non-contiguous inputs, zero strides,
32/64-bit index variants, and partial final tiles in four-dimensional gathers.
Every shape retains its random cases and tests both signed-zero sides of the
negative-real branch cut with a half exponent, using the same error bound.
Inputs and readbacks are retained separately for all three datasets.
DirectX and OpenGL use the public runtime-package and native-loader APIs.
Metal executes the original source, generated kernels and public runtime package
for every dataset, retaining 873 comparisons on each path. Each output is
checked against an independently indexed CPU reference; required CI retains
the compiler logs, packages, bindings and numerical results. The OpenGL
index-range assertions are bounded fixture preconditions, not general runtime
bounds checks. Other binary operators and types, the complete upstream suite,
and MLX host-runtime redirection remain outside this proof.

The DirectX/OpenGL two- and three-dimensional cases also verify that truncated
stride buffers are rejected before dispatch. Their minimum lengths come from
proven constant-index accesses in the generated helpers, not from MLX-specific
buffer names. Valid workloads still run unchanged; this preflight guarantee does
not cover arbitrary dynamic indexing.

The existing native-host jobs also run real-valued power through public runtime
packages. Identity checks retain all 65,536 half and bfloat storage patterns and
5,632 binary32 inputs. A separate batch selects the explicit binary32 power
operand profile for 82,848 float32/bfloat operand pairs. Every non-NaN result,
signed zero and guard is checked exactly; NaNs use classification checks.
Those five cases retain 219,552 outputs and 40 guards per target. A third batch
selects `binary32_power_accuracy_profile = "portable-finite"` for 4,982 float32
pairs, including near-one bases with large exponents and seeded general inputs.
It compares normal finite results against a high-precision decimal reference
within 16 representable steps; guards remain exact. All six cases retain
unchanged original-source readbacks on Metal. Two-thread workgroups keep the complete
input sets within DirectX dispatch limits without adding inactive invocations.
No upstream source is modified. This covers the vector/vector entries' identity,
operand-domain behavior and selected finite accuracy, not result underflow,
all real-power layouts or full MLX host integration. The three-minute step uses
the existing native runners and preserves every earlier check and deadline.

## Scope

The general-gather jobs in [Project Demo Testing](../../../.github/workflows/demo-project-testing.yml)
uses the unchanged JIT template and indexing headers at
`9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8`. It executes 24 original/generated
Metal workloads through public project, package and native-dispatch APIs.
Coverage includes zero through two index buffers, scalar through three-dimensional
indices, negative and repeated indices, and dense, transposed, strided and
broadcast source/index layouts. Binary32 outputs are compared as exact storage
words, including trailing guards. No upstream source or test is modified.

The no-index specialization requires a valid zero-extent standard array.
CrossTL retains that native object and its layout instead of emitting an illegal
C-style array. Separate native controls exercise aliases, nested arrays,
packed/narrow elements, copies and neighboring fields. DirectX/OpenGL empty-array
representation remains unsupported ([#2042](https://github.com/CrossGL/crosstl/issues/2042)).
Private pointer-bearing aggregates use resource identities and signed offsets
on DirectX and OpenGL, with concrete buffer specialization on OpenGL
([#1544](https://github.com/CrossGL/crosstl/issues/1544)). Required Windows/Linux
gates execute the 20 indexed workloads. OpenGL uses explicit allocation-derived
index-range assertions; these bounded cases do not establish unrestricted
64-bit addressing. Returned void helper calls retain their computation and
argument effects ([#2043](https://github.com/CrossGL/crosstl/issues/2043)).
These are direct kernel proofs, not integration of general Gather into the MLX
host adapter or a passing complete upstream suite.

The same workflow requires pointer-offset execution controls: 13 cases on Metal
and 11 on DirectX/OpenGL, with exact outputs and neighboring guards. Metal also
executes the original source. Offset values, including struct members and
conditional scalar loads, do not change a pointer's address space. Both operand
orders preserve resource-backed aggregate pointers without repeating offset
side effects. Incompatible Metal helper arguments fail project translation
instead of replacing a reachable computation with zero
([#2053](https://github.com/CrossGL/crosstl/issues/2053)). DirectX/OpenGL still
reject same-buffer conditional pointers
([#2054](https://github.com/CrossGL/crosstl/issues/2054)) and local struct-pointer
aliases ([#2055](https://github.com/CrossGL/crosstl/issues/2055)); explicit negative
controls retain those diagnostics. These cases do not count as native passes.
Generated resource helpers use explicit 64-bit index arguments; native OpenGL
execution checks this contract independently of glslang acceptance
([#2056](https://github.com/CrossGL/crosstl/issues/2056)).

The [Metal host integration harness](METAL_HOST.md) redirects selected current-pinned
complex-power kernels through MLX's own runtime and runs its unchanged upstream
operations tests. Its dispatch trace distinguishes actual host execution from
compile-only coverage. Eight host layouts retain 402 complex readbacks across
ordinary inputs and both signed-zero branch cuts; unselected operations still
use upstream kernels.

The [portable host adapter](portable_host/README.md) connects MLX's C++ GPU
evaluation to native DirectX/OpenGL/generated Metal packages with the original
Metal and CUDA backends disabled. It covers five typed array-creation entries,
30 float32 unary entries, shared views, bit-exact 32-bit layout copies and 16
binary arithmetic entries, with 21 unchanged upstream tests and explicit
unsupported-operation failures.
This synchronous adapter does not yet
implement a complete MLX backend.

The current harness verifies:

- the [pinned native MLX Metal reference baseline](NATIVE_METAL.md) for exact
  upstream commit `4367c73b60541ddd5a266ce4644fd93d20223b6e`. On the Arm64
  `macos-26` runner, it compiles all 40 native Metal units (33 Metal 3.2 / 7
  Metal 4.0), runs
  776 Python tests with 44 skips, and passes 260/260 C++ cases and 3,490/3,490
  assertions. This native upstream reference does not establish
  translated-target correctness, runtime parity, or numerical parity;
- discovery of the MLX Metal kernel project surface under
  `mlx/backend/metal/kernels`;
- Metal-to-CrossGL-to-Metal translation of pinned `fence.metal`, including
  project artifact hashes, sizes, source maps, provenance, and native Metal
  compilation on macOS CI. The gate requires all three device-memory,
  sequentially consistent, system-scope atomic fences to survive the round
  trip without a weaker barrier fallback. Resource coherence and volatility
  preservation remain blocked by
  [#1660](https://github.com/CrossGL/crosstl/issues/1660), so this is not yet a
  complete semantic-equivalence claim;
- selected-entry Metal-to-CrossGL-to-Metal translation of all 877 discovered
  entries in ``unary.metal`` at the legacy reference revision
  `846d176227a0ac13d2667e58d2bb68b322109ab0`: 183 each for ``v_``, ``v2_``,
  ``gn1_``, and ``gn4large_``, plus 145 ``vn_`` entries. The exact contract
  spans 37 operators and 20 input/output type pairs across bfloat16, Boolean,
  complex64, float16, float32, signed and unsigned integers, FP8 encode/decode,
  and complex projection. Reachability pruning keeps one selected operator and
  kernel, materializes the required gather index helper only for ``gn*``, and
  reflects either three vector resources or five gather resources. Source
  ``constant`` shape/stride provenance, a read-only ``device`` dimension
  reference, and postfix output indexing survive the round trip. A host-owned
  ``[1, 1, 1]`` dispatch contract is preserved, and macOS CI compiles every
  deterministic artifact to non-empty AIR. This is complete discovered unary
  Metal compiler/reflection coverage, not Metal numerical execution. The reviewed
  `v_Squarefloat32float32` reference includes an unused-parameter annotation;
  removing only that annotation reproduces its prior accepted source hash, and
  both versions compile to byte-identical Metal libraries with warnings fatal
  and fast math disabled. This reference update does not approve unrelated
  pending corpus changes or establish current-pin runtime parity;
- selected-entry translation of all 877 discovered `unary.metal` entries
  at the same legacy reference revision to OpenGL. The schema-v2
  `contracts/unary.opengl-translation.json` contract pins every standalone
  `main` artifact across the same five shapes, 37 operators, 20 type pairs,
  1,243 materializations, and 3,363 reflected resources, totaling 4,780,991
  generated GLSL bytes. The references incorporate reviewed scalar/vector
  conversions and numeric helpers; every complete shader body and host
  interface is compared before accepting an identity change. Translation
  requires three explicit host/runtime
  index-range preconditions for `offset + i`, `out_idx++`, and `idx`; these are
  portability promises rather than inferred or runtime-enforced bounds. The
  lowering maps Metal `log10` through one-evaluation GLSL `log2`, preserves user
  overloads, and exposes the read-only `device const int& ndim` as a storage
  buffer whose scalar expression aliases element zero. Five required Linux CI
  shards compile every artifact for OpenGL/SPIR-V 1.3 with
  `glslangValidator` and validate every module with `spirv-val`. The gate
  requires 877 non-empty SPIR-V 1.3 modules. This is complete OpenGL
  translation, reflection, and native compiler coverage, not numerical
  execution or MLX host runtime redirection;
- selected-entry translation of all 877 discovered `unary.metal` entries at the
  legacy reference revision `846d176227a0ac13d2667e58d2bb68b322109ab0` to DirectX.
  The schema-v2
  `contracts/unary.directx-translation.json` contract pins every standalone
  `CSMain` artifact across the same five shapes, 37 operators, 20 type pairs,
  and 1,243 materializations, with 3,912 reflected HLSL resources: three per
  `v_` and `vn_` artifact, four per `v2_` artifact, and six per gather artifact.
  Work-per-thread and gather forms include the generated `CrossGLDispatchInfo`
  dispatch cbuffer at bindings `b3` and `b0`, respectively. DirectX bfloat
  lowering covers the
  complete unary intrinsic family, while `acosh`, `asinh`, and `atanh` decode
  through portable float helpers before exact bfloat reconstruction. Explicit
  contextual casts preserve float-to-native-16 return and initializer narrowing
  without DXC `-Wconversion` diagnostics under warnings-fatal compilation. The
  read-only source `device const int& ndim` becomes a `StructuredBuffer<int>`
  scalar view at element zero, and source `out_idx++` remains postfix in HLSL.
  The same three explicit host/runtime index-range preconditions are required.
  Five required Ubuntu CI shards compile every artifact with pinned Linux DXC,
  source-derived `-enable-16bit-types`, warnings fatal, profile `cs_6_2`, and
  entry point `CSMain`; the gate requires 877 non-empty DXIL modules. Together
  with the OpenGL proof this records complete discovered unary translation,
  reflection, and native compiler coverage at the recorded revisions, not
  numerical execution or MLX host runtime redirection. The earlier reference
  update compiled all 877 HLSL outputs and compared their complete
  bodies against hash-verified originals: 603 changed and 274 unchanged. The
  differences are accounted for by source-width arithmetic, byte and half
  conversions, Boolean promotion, sign preservation and precise math helpers.
  Checks with deliberately altered bindings, index steps, conversion widths
  and helper coefficients fail. A subsequent review recompiles all 34 Exp and
  Sigmoid entries: 15 scalar Exp and five bfloat Sigmoid bodies change, while
  the other 14 bodies are unchanged. The Exp changes preserve the source's
  precise exponential call and half/bfloat return boundaries; the Sigmoid
  changes preserve bfloat arithmetic instead of inferring a half temporary.
  Their complete body review rejects altered bindings, index steps, rounding
  constants and temporary types. A further review updates only the five
  unsigned 32-bit Sign layouts: the source changes retain a precise square-root
  helper in an unused complex overload, while warning-fatal DXC compilation
  produces identical binaries before and after the change. Current-pin Windows
  execution separately checks all five layouts, 3,303 output values and 40
  output guards through the native loader. The references now total 3,774,223
  bytes;
  source identities, classifications, resource bindings and dispatch contracts
  are unchanged. The required Windows Square and ArcCos numerical tests retain
  their existing shader identities, inputs and tolerances. Other historical
  HLSL families remain under review in
  [crosstl#2071](https://github.com/CrossGL/crosstl/issues/2071). Whole-family
  numerical parity is not claimed; bit-observable floating conversions also
  require the source-policy work tracked in
  [crosstl#2081](https://github.com/CrossGL/crosstl/issues/2081). The historical
  Sigmoid source uses the default exponential, unlike the current pin's
  precise call. This contract retains that distinction and does not enable
  the optional binary32 division profile. The separately configured current-pin
  full-domain Sigmoid test does not establish historical whole-family parity;
- selected-entry Metal-to-CrossGL-to-Metal translation of all 2,496
  discovered copy entries from `copy.metal` at the legacy reference revision
  `846d176227a0ac13d2667e58d2bb68b322109ab0`. The schema-v2
  `contracts/copy.metal-roundtrip.json` contract spans 30 shapes, 16 concrete
  templates, 13 source types, all 169 conversion pairs, 6,566 exact
  materializations, and 8,684 reflected resources. It records the conditional
  nested specialization on exactly 14 complex-to-Boolean entries, preserves
  explicit float and bfloat16 bit-pattern semantics, and validates registered
  complex representation shape before scalar projection. Twenty-four required
  Ubuntu shards each export 104 exact artifacts; one macOS consumer compiles
  all 2,496 sources with warnings fatal and requires a non-empty AIR object for
  every entry. This is complete copy translation,
  reflection, and native compiler coverage, not numerical execution or MLX host
  runtime redirection. The compiler-gated identity refresh retains source
  coverage, materializations and resource ABI. `artifactIdentityRefresh` retains
  the historical linkage-only audit separately from the earlier `proof` metadata.
  The subsequent complete review also accounts for byte conversions and typed
  vector initializers in 736 changed bodies, with 1,760 bodies unchanged. These
  reference updates do not establish coverage of a newer MLX revision;
- selected-entry translation of all 2,496 discovered historical
  `copy.metal` entries to OpenGL at revision
  `846d176227a0ac13d2667e58d2bb68b322109ab0`. The schema-v2
  `contracts/copy.opengl-translation.json` contract pins every standalone
  `main` artifact across all 30 shapes, 16 templates, 13 input/output types,
  169 conversion pairs, 6,566 materializations, and 8,684 reflected target
  resources. Registered `complex_t_float` representations project through the
  real field for the 150 complex-to-scalar entries only after exact ordered
  `real`/`imag` float-shape validation; malformed registered shapes continue to
  fail closed. Eight explicit host/runtime index-range preconditions bound
  `offset + i`, `src_idx`, `dst_idx`, `dst_idx + i`,
  `src_idx + src_offset`, `dst_idx + dst_offset`, `idx.x`, and `idx.y` to
  signed 32-bit OpenGL index space. These are portability promises, not
  inferred or runtime-enforced checks, and unproven wide indices remain
  rejected. Twenty-four required Linux CI shards retranslate every exact
  entry, verify deterministic identity, materialization, workgroup metadata,
  and reflected ABI, compile for OpenGL/SPIR-V 1.3 with
  `glslangValidator`, validate with `spirv-val`, and require 2,496 non-empty
  SPIR-V modules. Together with the Metal proof this closes complete copy
  translation, reflection, and native compiler coverage for those two targets.
  A complete source review accounts for narrow-conversion helpers in 308
  changed artifacts; 2,188 artifacts are byte-identical. All interfaces and
  indexing remain unchanged, and 61 fresh translation checks cover every
  changed conversion pair and shape. A subsequent full review preserves 2,482
  artifacts and updates only the 14 bfloat16-to-Boolean entries. Their sole
  change explicitly extracts the unsigned 16-bit word and promotes it to
  `int` before the unchanged sign-bit mask. Exact accepted-source reconstruction
  and 2,510 compiler/validator runs cover the whole family and both versions
  of those 14 entries. The compiled modules differ; this is not a claim of
  byte-identical compiler output. The conversion is checked over every bfloat16
  word, while all resource interfaces, materializations and index preconditions
  remain unchanged.
  Separately, `test_current_copy.py` executes current-pin contiguous and strided
  bfloat16-to-Boolean copies over all 65,536 bit patterns per entry, including
  signed zero, subnormals, infinities and NaNs. Generated Metal and OpenGL match
  unmodified original Metal across 131,072 computed values and 128 output guards
  per target. Inputs use lossless bit-pattern transport; output buffers start
  with the opposite expected value. The same tests are required in the existing
  Windows native job; local strict DXC checks pass, but Windows numerical
  execution of these additions remains unverified. The copy harness uses the
  existing host-runtime entry-name normalization for reflected HLSL constants,
  including exported names ending in an underscore. This changes test bindings,
  not upstream sources, generated shaders or numerical expectations.
  These two entries do not
  establish whole-family numerical parity or complete MLX host redirection;
- selected-entry translation of all 2,496 discovered historical
  `copy.metal` entries to DirectX at revision
  `846d176227a0ac13d2667e58d2bb68b322109ab0`. The compact schema-v2
  `contracts/copy.directx-translation.json` contract pins every standalone
  `CSMain` artifact across all 30 shapes, 16 templates, 13 input/output types,
  169 conversion pairs, and 6,566 exact materializations. Exact HLSL target
  reflection contains 10,036 resources; `g2`, `g2large`, `g3`, `g3large`,
  `gn2`, `gn4large`, `s2`, and `v2` receive generated
  `CrossGLDispatchInfo` metadata at their shape-specific `b0`, `b6`, or `b3`
  binding. Registered `complex_t_float` representations project through the
  real field for the 150 complex-to-scalar entries only after exact ordered
  `real`/`imag` float-shape validation, while malformed registered shapes fail
  closed. Unlike the OpenGL lowering, this native HLSL path introduces no
  additional source-scoped 32-bit index-range portability promise. Twenty-four
  required Ubuntu CI shards retranslate every exact entry, verify identity,
  materialization, workgroup metadata, and reflected ABI, and compile with
  checksum-pinned DXC using `-enable-16bit-types -WX -T cs_6_2 -E CSMain`,
  requiring 2,496 non-empty DXIL modules. Together with the Metal and OpenGL
  proofs this closes complete copy translation, reflection, and native
  compiler coverage on all three targets, not numerical execution or MLX host
  runtime redirection. A subsequent full-family reference review reconstructs
  every previously accepted source and verifies byte-identical strict DXIL
  and reflected interfaces for all 2,496 pairs, preserving 6,566
  materializations and 10,036 resources. The changes are unused structure
  fields and signed/unsigned casts of nonnegative 16-bit values before
  masking in bfloat-to-bool helpers. The review retains the historical source
  pin and all launch contracts. Compiler equivalence does not establish
  current-pin numerical execution or full upstream-suite parity;
- selected-entry Metal-to-CrossGL-to-Metal translation of all 4,122
  discovered current-pinned binary entries from ``binary.metal``. The complete
  contract spans 18 shapes and 11 concrete kernel templates, 24 operators, and
  25 input/output type pairs. It emits one kernel and one or two reachable
  materializations per artifact (6,026 exact materializations total), reflects
  each exact three- through seven-resource ABI, and preserves a host-owned
  ``[1, 1, 1]`` dispatch. Required macOS CI compiles every artifact with
  warnings fatal and requires 4,122 non-empty AIR outputs. This proves every
  discovered binary instantiation for Metal translation, reflection, and native
  compilation, not Metal numerical execution;
- selected-entry translation of all 4,122 discovered current-pinned
  `binary.metal` entries to OpenGL. The schema-v2
  `contracts/binary.opengl-translation.json` contract pins every standalone
  `main` artifact across all 18 shapes, 11 templates, 24 operators, 25 type
  pairs, 6,026 materializations, and 19,106 reflected resources, totaling
  16,534,612 generated GLSL bytes. Seven explicit
  host/runtime index-range preconditions bound `offset + i`, `a_idx`, `b_idx`,
  `out_idx`, `out_idx++`, `idx.x`, and `idx.y` to signed 32-bit OpenGL index
  space; they are portability promises, not inferred or runtime-enforced
  checks, and unproven wide indices still fail closed. Twenty-four required
  Linux CI shards retranslate every exact entry, verify deterministic identity,
  materialization, workgroup metadata, and reflected ABI, compile for
  OpenGL/SPIR-V 1.3 with `glslangValidator`, validate with `spirv-val`, and
  require 4,122 non-empty SPIR-V modules. The complete-family reference review
  remains open in [#2073](https://github.com/CrossGL/crosstl/issues/2073).
  Reviewed half-remainder entries have bounded numerical evidence described
  below, not whole-family parity or MLX host runtime redirection;
- selected-entry translation of all 4,122 discovered historical
  `binary.metal` entries to DirectX at revision
  `846d176227a0ac13d2667e58d2bb68b322109ab0`. The schema-v2
  `contracts/binary.directx-translation.json` contract pins every standalone
  `CSMain` artifact across all 18 shapes, 11 templates, 24 operators, 25 type
  pairs, and 6,026 materializations. Nine source shapes that consume
  `threads_per_grid` receive explicit `CrossGLDispatchInfo` interfaces, raising
  the exact aggregate reflection count to 21,248 resources across three-
  through eight-resource ABIs. Computed-result bfloat `ArcTan2`, `LogAddExp`,
  and `Power` paths expand to float and reconstruct with round-to-nearest-even;
  `Maximum` and `Minimum` expand only for comparison and return the selected
  original bfloat payload without requantization. All other unproven bfloat
  builtins remain fail-closed. The
  same seven explicit host/runtime index-range preconditions remain required.
  Twenty-four required Ubuntu CI shards retranslate every exact entry, verify
  deterministic identity, materialization, `CSMain` workgroup metadata, and
  reflected ABI, then compile with checksum-pinned DXC using native 16-bit
  types and warnings fatal. The gate requires 4,122 non-empty DXIL modules.
  The half minimum/maximum review updates 36 references across all 18 shapes.
  Complete before/after bodies, interfaces and materializations were checked;
  the only changes are a bit-preserving half-selection helper and its calls.
  Both versions compile with strict DXC, and independent rebuilds reproduce
  the current DXIL exactly. Two vector-vector artifacts match the shaders
  executed in Windows CI: 9,344 results and 16 guards agree bit for bit,
  including NaN payloads and signed zero. These samples do not establish
  whole-family numerical execution. Remaining binary reference reconciliation
  is tracked in [#2071](https://github.com/CrossGL/crosstl/issues/2071);
- selected-entry Metal-to-CrossGL-to-Metal translation of all 2,396
  discovered reduction entries from `reduce.metal` at historical revision
  `846d176227a0ac13d2667e58d2bb68b322109ab0`. The compact
  schema-v2 `contracts/reduce.metal-roundtrip.json` contract spans 39 exact ABI
  shapes, nine kernel templates, six operator families, 44 concrete operator
  types, and 13 input/output types. It pins every artifact identity plus exact
  materialization and resource digests while avoiding transient proof rows in
  git. The one- through twelve-resource interfaces contain 25,088 reflected
  resources in aggregate and retain a host-owned `[1, 1, 1]` workgroup
  contract. Twenty-four required Ubuntu shards cover 20 groups of 100 entries
  and four groups of 99. One macOS consumer compiles every exported artifact
  with warnings fatal and requires 2,396 non-empty AIR objects. Complete source
  and interface comparisons account for 558 updated references: source product
  grouping and half-precision infinity initializers. Native checks of the two
  half initializer kernels match unchanged upstream Metal and independent
  expected words, including output guards. This is complete reduce translation, reflection,
  and native compiler coverage, not numerical execution or MLX host runtime
  redirection. Subsequent reviews of all 400 entries in source shards 0 through 3
  update 393 identities for unused-parameter annotations and typed integer boundary
  constants. Every previous source is reconstructed to its accepted hash;
  previous and current sources compile and link to byte-identical Metal libraries
  with warnings fatal. The reviews preserve 1,534 materializations and 4,156
  resources, leaving the other 2,003 records unchanged. This compiler-equivalence
  check adds no numerical execution or current-pin coverage. Reference
  reconciliation for the remaining shards is still tracked by
  [#1966](https://github.com/CrossGL/crosstl/issues/1966);
- selected-entry Metal-to-CrossGL-to-Metal translation of all 2,052 host-named
  entries from `quantized.metal`. The compact schema-v2
  `contracts/quantized.metal-roundtrip.json` contract covers 38 normalized
  variants over 17 source templates for three data types, group sizes 32/64/128,
  and bit widths 2/3/4/5/6/8. It pins 14,904 exact materializations and
  40,559,670 generated Metal bytes without checking transient proof rows into
  git. Thirty deduplicated exact ABI contracts span four- through
  twenty-two-resource interfaces and contain 31,374 reflected resources in
  aggregate, all with host-owned `[1, 1, 1]` workgroups. The original terminal external
  proof compiled every entry twice with Metal 3.1 and warnings fatal, required
  byte-identical non-empty AIR, and is bound by the checked-in terminal,
  independent-audit, and adversarial identities. The `proof` section retains
  that original audit provenance; `entries` and `artifactContract` describe the
  reconciled output. Complete body and interface comparisons account for source
  grouping, explicit narrow conversions and infinity initializers. All 2,052
  current sources pass strict compilation and a separate native bundle rebuild.
  Twenty-four required Ubuntu CI
  shards rediscover the exact source entries, retranslate and reflect each one;
  twelve shards contain 86 entries and twelve contain 85. One dependent macOS
  job verifies every source identity and compiles all entries with Metal 3.1,
  warnings fatal and empty compiler streams. Missing, duplicate or changed
  sources fail before compilation, and the gate requires all 2,052 AIR outputs
  to be non-empty. This
  is complete quantized Metal translation, reflection, and native compiler
  coverage. The same macOS job additionally executes the 108 affine quantization
  and dequantization variants from the verified source bundle: three scalar
  types, three group sizes and six bit widths for both operations. Five groups
  per variant exercise positive and negative scales, exact dyadic values and
  zero ranges. Original upstream Metal and generated Metal must independently
  match expected packed bytes, scales and biases, with readonly inputs and
  output guards preserved. `prove_quantized_metal.py` retains native commands,
  modules, inputs, expected outputs and readbacks. The refreshed-source check
  covers all 108 variants and 40,320 scalar values per original/generated path.
  This bounded numerical check
  adds no macOS runner or repeated translation and does not establish numerical
  coverage of the remaining quantized families or MLX host runtime redirection;
- selected-entry translation of all 2,396 discovered `reduce.metal` entries at
  historical revision `846d176227a0ac13d2667e58d2bb68b322109ab0` to OpenGL. The compact schema-v2
  `contracts/reduce.opengl-translation.json` contract spans the same 39 exact
  ABI shapes, nine kernel templates, six operator families, 44 concrete
  operator types, and 13 input/output types. It pins 9,216 exact
  materializations, 32,115,641 generated GLSL bytes, and 25,088
  resources across scalar-layout-aware one- through twelve-resource target ABIs.
  Complete source comparisons and strict compiler checks cover every entry;
  85 fresh pipeline checks cover all reviewed edit patterns and ABI shapes.
  Resource declarations are unchanged. The 156 complex-buffer reference updates
  add explicit scalar-layout metadata rather than changing buffer storage.
  Six explicit host/runtime index-range preconditions bound the source pointer
  and output expressions required by wide-index shapes to signed 32-bit OpenGL
  index space; they are deployment promises rather than inferred or generated
  checks. Bounded unit-step loop-carried pointer-array initialization is proven
  one exact iteration at a time; a non-singleton per-invocation dynamic target
  remains a may-write and cannot establish full-array definite assignment,
  while shadowed loop-index bindings remain fail-closed. For-in range and
  scalar-count expressions are rendered in the outer lexical environment;
  Fixed-array iterable expressions and compile-time extents resolve before
  same-named pattern bindings enter scope. same-named patterns use
  independent controllers and shadow pointer,
  stage-builtin, and flattened stage-struct aliases in the loop body.
  For-in resource specialization resolves overloads from exact lexical
  pattern types. Dynamic for-in resource specialization resolves overloads
  from exact call-site lexical types. Null storage-pointer reachability and
  elision preserve exact lexical declaration identity. Null workgroup-pointer
  reachability and elision preserve exact lexical declaration identity.
  Nested resource-specialization discovery preserves deterministic lexical
  order across Python hash seeds. Workgroup-pointer bounds analysis visits
  every control-flow expression and preserves outer mutations across lexical
  blocks. loop-local fixed-array
  storage retains
  flattened stage-input struct
  declarations, and fixed-array patterns cannot inherit same-named outer
  scalar or vector-component bounds. Repeated loop bounds and generated loop
  controllers lose their proven
  intervals when exact scalar or vector-component dependencies can mutate.
  logical-offset mutation through resolved nested helpers is written back to
  callers. Direct
  element and addressed one-element forwarding through overload-resolved
  scalar or fixed-array helpers preserves that writeback;
  private scalar address views remain confined to their declaring lexical scopes;
  nonlocal scalar address views remain fail-closed;
  dynamic elements and unresolved, ambiguous, or recursive forwarding remain
  fail-closed. Fixed-array storage cannot escape through local private pointer or
  reference aliases; lexically shadowed arrays and condition-only reads are not
  misattributed, while residual private pointer or reference syntax remains
  fail-closed;
  unsupported wide
  indices, pointers, and recursion still fail closed. Twenty-four required
  Linux CI shards retranslate every exact entry, verify deterministic artifact,
  materialization, workgroup, and reflected ABI
  identity, compile for OpenGL/SPIR-V 1.3 with `glslangValidator`, validate with
  `spirv-val`, and require 2,396 non-empty SPIR-V modules. This is complete
  reduce OpenGL translation, reflection, and native compiler coverage, not
  numerical execution or MLX host runtime redirection;
- selected-entry translation of all 2,396 discovered `reduce.metal` entries to
  DirectX at the historical reference revision
  `846d176227a0ac13d2667e58d2bb68b322109ab0`. The compact schema-v2
  `contracts/reduce.directx-translation.json` contract pins every standalone
  `CSMain` HLSL artifact and its native DXIL identity across the same 39 exact
  ABI shapes, nine kernel templates, six operator families, 44 concrete
  operator types, 13 input/output types, and 9,216 exact materializations.
  Exact HLSL target reflection contains 27,382 resources across one- through
  thirteen-resource interfaces; the 37 shapes other than `init` and `all`
  receive generated `CrossGLDispatchInfo` metadata. Portable unshadowed `NAN`,
  explicit complex SIMD shuffle overloads, proven direct void tail recursion,
  bounded device/constant pointer-array offsets and writeback, logical bfloat
  pointee types over physical storage, and nested anonymous records lower only
  in their natively proven forms. Unsafe recursion, ambiguous pointer-array
  provenance, unresolved aggregate pointer members or arrays, and unsupported
  HIP record lifecycle or layout remain fail-closed. Unlike the OpenGL
  lowering, this native HLSL path introduces no additional source-scoped
  32-bit index-range portability promise. Twenty-four required Ubuntu CI
  shards retranslate every exact entry, verify deterministic identity,
  materialization, workgroup metadata, and reflected ABI, and compile with
  checksum-pinned DXC using `-enable-16bit-types -WX -T cs_6_2 -E CSMain`,
  requiring 2,396 non-empty DXIL modules. The reviewed references total
  36,957,938 HLSL bytes and 24,292,688 DXIL bytes. Source references preserve
  the original integer-product grouping. The subsequent half-selection review
  reproduces all 78 previous half minimum/maximum sources and DXIL identities,
  compares complete bodies and interfaces, and updates 76 changed references;
  the two initializer entries remain byte-identical. Only the bit-preserving
  selection helper and its calls change. Strict independent recompilation
  reproduces all 78 current modules. Shared Windows selection checks cover
  every binary16 encoding in both operand positions, with 853,840 exact values
  and 64 guards; they do not execute these complete reduction kernels.
  The remaining 2,318 reduction references retain their earlier independent
  compiler proof. This records complete
  translation, reflection and compiler coverage for this historical HLSL corpus,
  not numerical execution or MLX host runtime redirection. Other target reference
  updates remain separate;
- a checked-in reduced Metal fixture that mirrors MLX's reference-returning
  `frag_at` accessor over `val_frags[i * width + j]`. The fixture is translated
  to DirectX and OpenGL through the public `translate-project` CLI and retains
  three separate source contracts. The mutable scalar receiver is declared
  through a function-local `using` alias and assigned through `frag_at`. An
  implicit const scalar call is passed directly to a read-only helper. A second
  outer value owns a `float2`-backed `nestedTile`; its const `store` method binds
  `thread const auto& accum = nestedTile.frag_at(i, j)` and reads `accum[k]`.
  Each generated target must assign the scalar sentinel directly to the original
  `val_frags[...]` lvalue, read back that exact element, lower the implicit const
  call to its backing storage, and replace the nested accessor and `accum` alias
  with a read from `self.nestedTile.val_frags[...][k]`. A value-return helper,
  retained alias, or copied tile does not satisfy the proof. Windows CI requires
  DXC compilation; Linux CI requires `glslangValidator` compilation and
  `spirv-val` validation for OpenGL 4.5. The macOS matrix leg verifies the same
  proof set in both generated artifacts without requiring either target
  compiler. These checks do not execute an MLX runtime or establish numerical
  parity;
- a separate checked-in Metal fixture for the MLX `BaseMMAFrag::load` call
  shape in which a templated tile method passes `&(src[index])` to a templated
  fragment helper. Project translation must emit both DirectX and OpenGL
  artifacts with zero diagnostics. The materialized fragment helper must keep
  a pointer-backed source view and read it at `stride`; a scalar `float src`
  parameter is rejected. The DirectX proof requires a `StructuredBuffer<float>`
  source and preserves the addressed `src[index]` position as a composed
  `src_offset + index` view. The OpenGL proof requires the equivalent global
  storage-buffer plus `src_offset` form, carries `index` into that offset, and
  reads `src[src_offset + stride]`. Source-style unresolved member calls are
  rejected. This gate inspects generated structure;
  it does not require native target compilation, execute a shader, or claim
  runtime parity;
- target-separated DirectX and Vulkan project runs for the same 11-source
  reduced frontier: `arange.metal`, `arg_reduce.metal`, `binary_two.metal`,
  `layer_norm.metal`, `logsumexp.metal`, `random.metal`, `rms_norm.metal`,
  `rope.metal`, `scaled_dot_product_attention.metal`, `softmax.metal`, and
  `ternary.metal`. Vulkan must translate and structurally validate all 11.
  DirectX emits five aggregate artifacts whose entries do not require a
  runtime-selected workgroup size, two entry-scoped `layer_norm.metal`
  artifacts, two entry-scoped `logsumexp.metal` artifacts, and 12 entry-scoped
  `rms_norm.metal` artifacts selected by checked-in host dispatch contracts. It
  records exact expected failures for the other three sources. Each blocked
  report must retain
  the pinned total specialization
  count and exactly match its diagnostic entry names to the materialized host
  names, with no additional diagnostics. Separate configs prevent DirectX
  workgroup contracts from being silently ignored by Vulkan, where project
  workgroup rules are unsupported.
  This establishes target-specific structural and toolchain coverage, not
  semantic readiness or runtime parity;
- a separate project-level expected-failure check for pinned `fence.metal`
  across DirectX, OpenGL, and Vulkan. Each target must report its exact
  `project.translate.*-atomic-fence-unsupported` diagnostic, target-specific
  `*.atomic-thread-fence-contract-lowering` missing capability, and requested
  `mem_device`, `memory_order_seq_cst`, `thread_scope_system` contract without
  emitting a target file. The blocked contract is tracked by
  [#1537](https://github.com/CrossGL/crosstl/issues/1537);
- materialization evidence for all 24 host-named `arg_reduce.metal` compute
  entries within 39 total specializations. Vulkan emits the aggregate artifact.
  DirectX and OpenGL fail before emission with
  `project.translate.workgroup-size-entry-ambiguous` because pinned host
  dispatch uses runtime axis and pipeline-limit operands unavailable to source
  materialization;
- DirectX HLSL compiler checks with official DXC v1.9.2602.24 on Windows CI for
  the 21-artifact frontier representing eight pinned sources: `arange.metal`,
  `binary_two.metal`, two bounded `layer_norm.metal` entries, two bounded
  `logsumexp.metal` entries, `random.metal`, 12 test-derived `rms_norm.metal`
  entries, `rope.metal`, and `ternary.metal`. At the pinned revision the gate
  compiles 11, 225, 2, 2, 2, 12, 18, and 212 entries respectively, for 484
  generated compute entries in total. Each LayerNorm, LogSumExp, and RMSNorm
  artifact is emitted independently with its host-derived workgroup size and
  exact subgroup width; specialization constants are retained where required.
  The 16 normalization artifact references include explicit source-width index
  conversions and sign-preserving floating constants. The float16 RMSNorm
  artifact uses two-byte binary16 storage and explicit conversion at the source
  rounding boundary. Complete old/current body and interface comparisons and
  strict DXC compilation cover all 16 references; these updates do not expand
  the separate numerical execution claims below.
  The
  pinned rope translation supplies required function constant IDs through the
  quoted `"1"`, `"2"`, and `"3"` selectors in
  `[project.specialization_constants]` and materializes the concrete DirectX
  variant before compilation. Aggregate conditional lowering
  completed under [#1695](https://github.com/CrossGL/crosstl/issues/1695) admits
  every pinned `ternary.metal` entry to this compiler gate. Target-ABI overload
  identity [#1694](https://github.com/CrossGL/crosstl/issues/1694) and
  minimum-precision arithmetic widening
  [#1701](https://github.com/CrossGL/crosstl/issues/1701) admit all 225
  `binary_two.metal` entries. Exact-layout DirectX union lowering
  [#1728](https://github.com/CrossGL/crosstl/issues/1728) admits both
  `random.metal` entries; broader union layouts remain tracked by
  [#1696](https://github.com/CrossGL/crosstl/issues/1696), and runtime dispatch
  metadata remains tracked by
  [#1542](https://github.com/CrossGL/crosstl/issues/1542). Host dispatch contract
  import was completed under [#1793](https://github.com/CrossGL/crosstl/issues/1793).
  These are compilation claims only. The current-pin
  [random readiness audit](portable_host/README.md#random-generation-readiness)
  now passes its 20 bounded numerical cases on OpenGL, DirectX and generated
  Metal, including required Windows native execution. `RandomBits` host
  integration is not enabled.
  The three pending aggregate sources cover 76 compute entries. Those historical
  aggregate runs do not consume the later entry-scoped bounded contracts and
  remain asserted as failed artifacts; no placeholder workgroup size is
  restored. Separate bounded GEMV, arg-reduce, Softmax, and scaled-attention
  translation, package, and native-runtime proofs are documented below.
  `fence.metal` is
  excluded because its DirectX translation intentionally fails under
  [#1537](https://github.com/CrossGL/crosstl/issues/1537) before DXC. This gate
  establishes compiler acceptance only; it does not dispatch these kernels or
  establish numerical parity;
- a separate Windows CI Direct3D 12 execution proof for a checked-in,
  MLX-shaped Metal compute fixture with a file-scope immutable two-dimensional
  lookup table. The fixture is translated to HLSL during the test, compiled to
  DXIL with DXC, dispatched through the built-in DirectX runtime adapter and
  `compushady`, and read back as four exact unsigned values:
  `[5, 19, 11, 13]`. Because every result selects a table entry, this is a
  value-sensitive proof that the generated `static const` initializer survives;
  it is independent of the compiler-only frontier gate above;
- a second, independent Windows CI Direct3D 12 execution proof that translates
  the actual pinned `mlx/backend/metal/kernels/arange.metal` source with the MLX
  repository root as an include path. Project configuration selects the
  materialized `arangeuint32` entry and emits a standalone artifact at the
  deterministic `arange/arangeuint32.hlsl` path. The proof verifies the pinned
  source hash, entry-scoped provenance, source mapping, and the generated runtime
  artifact manifest before compiling `CSMain` to DXIL with DXC. It binds only
  the reflected `b0` start, `b1` step, and `u2` output resources and dispatches
  through the built-in Direct3D 12 adapter. Seven invocations use `start = 300`
  and `step = 17`; the required zero-tolerance readback is
  `[300, 317, 334, 351, 368, 385, 402]`;
- Direct3D 12 and OpenGL 4.3 native-loader execution of the actual pinned
  `mlx/backend/metal/kernels/binary.metal` source. The project request selects
  `ss_Addfloat32`, packages one standalone target entry, and verifies the exact
  upstream source hash, entry-scoped provenance, reflected resource bindings,
  and scalar buffer layouts. Four invocations read `1.5` and `2.25` from the
  two source buffers and must return `[3.75, 3.75, 3.75, 3.75]` from the
  translated kernel. Windows runs the HLSL artifact through Direct3D 12; Linux
  runs the GLSL artifact through a surfaceless Mesa OpenGL context. This proves
  one real binary operator entry and does not claim coverage of the other
  materialized binary variants or the upstream MLX host runtime;
- the `arangeuint32` and `ss_Addfloat32` native-loader checks above, together
  with the scalar copy, float32 dot-product, bounded GEMV, bounded Softmax,
  bounded scaled-attention, and unary Square and ArcCos checks described below,
  use the
  current corpus at commit
  `846d176227a0ac13d2667e58d2bb68b322109ab0`. The broader 40-unit frontier and
  its remaining proofs retain their recorded historical revision until each
  source contract is remeasured;
- Vulkan assembly and validator checks for the existing non-fence regression
  frontier when SPIR-V tools are available. Vulkan atomic-fence feature work is
  deferred; the separate `fence.metal` contract check prevents generated
  barriers from being mistaken for semantic support;
- entry-scoped OpenGL packaging for the materialized `arangeuint32` compute
  entry from `arange.metal`. Project configuration selects the source entry,
  emits it at the deterministic `arange/arangeuint32.glsl` path as OpenGL
  `main`, and records the source-to-target entry identity in the portability
  report. The standalone artifact exposes only the `start`, `step`, and `out`
  resources and preserves the source arithmetic without an MLX source rewrite;
- an eight-source OpenGL frontier containing `arg_reduce.metal`,
  `binary_two.metal`, `logsumexp.metal`, `rms_norm.metal`, `rope.metal`,
  `scaled_dot_product_attention.metal`, `softmax.metal`, and `ternary.metal`
  in two project runs. `binary_two.metal`, `rope.metal`, and `ternary.metal`
  must emit with zero diagnostics and compile for OpenGL/SPIR-V 1.3 before
  `spirv-val`. Their project configuration supplies 24 source-qualified
  index-range assertions, and the portability report must reproduce the exact
  assertion count and content. The other five sources must produce the same
  exact fail-closed workgroup-size diagnostic and no target file. This is native
  artifact validation only; the gate does not run the kernels or establish
  runtime parity;
- a separate bounded OpenGL dispatch-contract proof for the actual pinned
  `logsumexp.metal` source. The two float32 workloads emit standalone GLSL
  artifacts with workgroup sizes `[32, 1, 1]` and `[288, 1, 1]`, an exact
  subgroup width of 32, and preserved subgroup max and sum reductions. Linux CI
  compiles both artifacts to OpenGL SPIR-V 1.3 and validates both binaries. The
  runtime contract requires `GL_KHR_shader_subgroup` and rejects a device whose
  `GL_SUBGROUP_SIZE_KHR` value is not 32 before shader compilation or dispatch;
- full project materialization of pinned `gemv.metal` for DirectX. The gate
  requires 226 materialized specializations—224 host-named entries plus both
  reachable `elem_to_loc` helper overloads—no unsupported materializations or
  unresolved residue, no bare pure value-discard statements, one aggregate HLSL
  artifact, and exactly 224 host-named report execution entries joined by
  materialization identity. Every generated target entry and `numthreads`
  declaration must match its report contract. The emitted native 16-bit types
  require DXC to compile `CSMain`, `CSMain_85`, and `CSMain_113` under
  `cs_6_2` with `-enable-16bit-types` and zero diagnostics, then compile all
  224 functions in one `lib_6_6` invocation with the same flag, an exact
  export set, and exact profile-warning classification;
- full project materialization of pinned `gemv.metal` for OpenGL. The gate
  requires all 226 specializations materialized, all 224 entry-scoped GLSL
  artifacts emitted, exact workgroup and subgroup execution records, and the
  five explicit host index-range preconditions used for 64-bit source indices.
  Linux CI compiles every artifact with `glslangValidator` and validates every
  resulting SPIR-V 1.3 module with `spirv-val`. This historical aggregate gate
  does not establish runtime execution or numerical parity; a distinct bounded
  current-pinned native-runtime proof is documented below;
- on Linux CI, full project materialization and translation of `gemv.metal` to
  Vulkan produces 226 specializations and 224 `GLCompute` entry points. The
  generated artifact passes both `spirv-as` and `spirv-val` for `vulkan1.1`
  with zero semantic warnings and no known codegen fallbacks. This is structural
  validation, not numerical runtime parity;
- runtime artifact manifest, runtime-test manifest, and runtime-test plan
  generation for reduced `arange` readiness probes across DirectX, OpenGL, and
  Vulkan;
- reference runtime fixture execution reports for the reduced `arange`
  readiness probes, using supplied project test-runner adapters and
  deterministic expected-output checks;
- native runtime execution-readiness reports for the same reduced probes,
  using the built-in DirectX, OpenGL, and Vulkan native adapter contracts with
  missing runtime drivers reported as structured blockers;
- on Linux CI, native Vulkan execution and readback for the generated MLX
  `arange.metal` unsigned 32-bit, signed 32-bit, and floating-point entry points
  through the optional Vulkan compute runtime and Mesa Vulkan software driver.
- on Linux CI, native OpenGL 4.3 execution and readback for the selected
  `arangeuint32` artifact through ModernGL and Mesa EGL. Four invocations use
  `start = 300` and `step = 17`; the required zero-tolerance comparison is
  `[300, 317, 334, 351]`.
- on Linux CI, OpenGL SPIR-V specialization through PyOpenGL and Mesa EGL for a
  reduced generated compute artifact. The native adapter compiles the GLSL to
  SPIR-V, applies numeric constant ID 7, and requires exact readback for two
  independently selected unsigned values;
- on Windows CI, native Direct3D 12 execution and exact readback for the reduced
  file-scope immutable lookup fixture, one generated uint32 entry from pinned
  `arange.metal`, and one generated float addition entry from pinned
  `binary.metal`. Linux CI executes the same binary entry through OpenGL. None
  of these proofs executes the upstream MLX host runtime.

Pull requests run the 12-source pinned reduced scope: 11 non-fence frontier sources
and the explicitly blocked `fence.metal` contract source. They also run the
separate checked-in reference-accessor and template-member pointer fixtures;
those fixtures do not change the pinned MLX source count. Scheduled and
manually triggered CI also run the full-corpus artifact scout with finite Metal
template materialization budgets.
The generated full-corpus project config caps
`max_template_specializations` at 4096 and
`max_template_materialization_work` at 131072. Each artifact translation also
has a 120-second limit. A timed-out artifact is recorded with a structured
`project.translate.timeout` diagnostic, and the remaining canonical plan
continues. The outer 900-second command limit remains an aggregate CI boundary;
the durable checkpoint preserves every completed result before that boundary.
The scout discovers all 40 pinned MLX Metal kernel units and attempts 120
DirectX, OpenGL, and Vulkan artifacts.
Its fence-aware success condition is 117 translated artifacts and three expected
failed `fence.metal` records, one per target, with no fence target files emitted.
That condition is a gate expectation, not a claim that the pinned full corpus
currently satisfies it. Additional failures remain issue-backed scout results.
CI uploads generated portability reports, validation summaries embedded in
those reports, the durable full-corpus translation checkpoint, generated logs,
available generated artifacts, and a concise JSON summary. A subsequent
full-corpus invocation resumes verified completed jobs from a running or
interrupted checkpoint.
Because `binary_two.metal` was already in the DirectX/Vulkan frontier, its OpenGL
promotion does not change either reduced source count.
The same applies to `ternary.metal`: its OpenGL promotion expands the native
toolchain gate without changing the 11-source non-fence reduced frontier.

The checked-in historical full-corpus scout snapshot against MLX revision
`968d264f2903d578e699c4452a4dbf48633921aa`
scanned 40 Metal kernels and attempted 120 target artifacts across DirectX,
OpenGL, and Vulkan. It translated 24 artifacts and reported 96 structured
artifact failures behind tracked issues. The current materialization pass
rejects template-hostile targets when concrete variants are missing instead of
emitting generic artifacts, so full-corpus counts should not be treated as
runtime-complete coverage or as the current fence-aware expected baseline.

This is shader/kernel artifact coverage. It does not claim that the MLX host
runtime has been ported to Direct3D, OpenGL, or Vulkan. Running the upstream MLX
GPU unit tests against non-Metal targets requires runtime adapters, host-side
dispatch wiring, data layout validation, and backend-specific build integration.
The reduced harness now emits runtime readiness artifacts so those gaps are
visible in CI reports without claiming runtime parity. The current readiness
manifests consume reflected runtime artifact metadata, including entry points,
resource bindings, and dispatch geometry. Runtime-test plans now resolve
source-level fixture names against common generated resource aliases. The
reference-accessor fixture is not included in those runtime manifests; its
scope ends at generated-source inspection and native compiler validation. The
reduced arange fixtures select the translated artifact by source and target,
then select `CSMain`, `main`, or `arangeuint32` independently for DirectX,
OpenGL, or Vulkan dispatch. OpenGL project translation packages source entry
`arangeuint32` as a standalone `main` artifact with an entry-scoped reflected
interface; its unsigned fixture uses values above the 8-bit range to detect
entry drift. The general DirectX readiness probe still describes the aggregate
artifact's first `uint8` entry. The separate required Windows device proof
packages `arangeuint32` as a standalone `CSMain` artifact. Vulkan execution selects
`arangeuint32`, `arangeint32`, and
`arangefloat32` explicitly and uses the same wide unsigned probe.
Plans report remaining non-blocking platform, layout, and entry-point ownership
warnings. The reduced fixture execution
report exercises the project runner and adapter contract with reference
buffers. The native execution report attempts the built-in native adapter
contract separately. On Windows CI the generated DirectX frontier HLSL must
compile with DXC. Separate native tests translate and execute the reduced
immutable lookup fixture and the pinned source's generated uint32 arange entry
through Direct3D 12. The arange test is numerical evidence for that one source,
entry, dtype, dispatch shape, and fixture only; it does not turn the frontier
compiler gate into a general runtime-parity claim. The five aggregate DirectX
frontier artifacts carry exact per-source bfloat16 report evidence. All five
report `status=exact`, `approximationUsed=false`, a `uint-low-16-bits` register
representation, and round-to-nearest, ties-to-even conversion. All five
aggregate sources require native `uint16` storage declarations and report the
`directx.native-16bit-types` capability. The harness compares each artifact's
`bfloat16Lowering` and `requiredCapabilities` fields with this pinned contract
and fails closed if either field is missing or changes. Native-profile bfloat
helpers now use exact `uint16_t` boundaries, and the two selected
`random.metal` entries compile without the promotion warnings tracked by
[#1799](https://github.com/CrossGL/crosstl/issues/1799). DXC reports zero
warnings across all 484 entry-point runs in the 21-artifact emitted frontier.
The harness records this as a warning-clean contract and rejects any newly
observed warning. Contextual destination conversion under
[#1801](https://github.com/CrossGL/crosstl/issues/1801) is resolved for the
pinned frontier. The arange assignment is emitted as
`arangeint16_out[index] = int16_t((uint(arangeint16_start) +
(index * uint(arangeint16_step))));`. The rope assignments are emitted as
`index_1 = uint(((2 * pos.x) + (pos.y * stride)));` and
`index_1 = uint((pos.x + (pos.y * stride)));`. All 18 rope entries compile with
DXC profile `cs_6_2`, `-enable-16bit-types`, and `-WX`. Native 16-bit arithmetic
conversion under [#1802](https://github.com/CrossGL/crosstl/issues/1802) is also
resolved for the pinned frontier. The float16 assignment is emitted as
`arangefloat16_out[index] = (arangefloat16_start + (float16_t(index) *
arangefloat16_step));`. All 11 arange entries compile without the
destination-conversion warning previously tracked by #1801 and with DXC profile
`cs_6_2`, `-enable-16bit-types`, and `-WX`. This preserves the resolved int16
destination-conversion evidence. The five-source aggregate portion therefore
has compiler acceptance and a warning-clean diagnostic contract. This is
storage, conversion, report, and compiler evidence only; it does not execute a
bfloat16 workload or establish runtime or numerical parity. On macOS CI, the
generated `fence.metal` round-trip artifact must compile to AIR with the native
Metal compiler. This checks generated source and project metadata, not numerical
runtime parity or equivalent resource visibility. CrossGL/crosstl#1660 tracks
preservation of
the source `volatile coherent(system)` pointer contract. On
Linux CI the generated Vulkan
`arangeuint32`, `arangeint32`, and `arangefloat32` entry points must assemble,
load, dispatch, read back, and compare on the Vulkan compute runtime; other
unavailable native backends remain structured blockers until backend runtime
drivers are supplied by integration code. This still does not execute
the upstream MLX host runtime or the upstream MLX Python/C++ unit test suite on
non-Metal backends.

## Running Locally

Clone MLX and check out the pinned revision:

```bash
git clone https://github.com/ml-explore/mlx.git /tmp/mlx
git -C /tmp/mlx checkout 4367c73b60541ddd5a266ce4644fd93d20223b6e
```

Run the project-porting harness from the CrossTL repository:

```bash
python demos/integrations/mlx/run_porting.py --mlx-root /tmp/mlx
```

Run the full-corpus artifact scout:

```bash
python demos/integrations/mlx/run_porting.py \
  --mode full-corpus \
  --mlx-root /tmp/mlx \
  --summary /tmp/mlx/.crosstl-mlx-porting/full-corpus-summary.json
```

On Linux, install the OpenGL, SPIR-V, and Vulkan runtime dependencies to require
the clean OpenGL compiler gates, the pinned GEMV toolchain gate, and native
OpenGL and Vulkan execution of the generated MLX `arange` artifacts:

```bash
sudo apt-get update
sudo apt-get install -y glslang-tools libegl1 libgl1-mesa-dri libglx-mesa0 mesa-vulkan-drivers spirv-tools vulkan-tools
python -m pip install moderngl==5.12.0 PyOpenGL==3.1.10 vulkan==1.3.275.1
python demos/integrations/mlx/run_porting.py \
  --mlx-root /tmp/mlx \
  --require-opengl-frontier-toolchain \
  --require-opengl-gemv-toolchain \
  --require-opengl-native-runtime \
  --require-vulkan-gemv-toolchain \
  --require-vulkan-toolchain \
  --require-vulkan-native-runtime

python demos/integrations/mlx/prove_copy_opengl.py \
  --mlx-root /tmp/mlx \
  --work-dir .crosstl-mlx-porting/copy-opengl

python demos/integrations/mlx/prove_rms_norm_specialization.py \
  --mlx-root /tmp/mlx \
  --work-dir .crosstl-mlx-porting/rms-norm-specialization \
  --require-opengl-toolchain

python demos/integrations/mlx/prove_quantized_opengl.py \
  --mlx-root /tmp/mlx \
  --work-dir .crosstl-mlx-porting/quantized-opengl \
  --require-opengl-toolchain

python demos/integrations/mlx/prove_quantized_opengl.py \
  --mlx-root /tmp/mlx \
  --work-dir .crosstl-mlx-porting/quantized-gather-opengl \
  --entry-point affine_gather_qmv_fast_float_gs_32_b_2 \
  --require-opengl-toolchain
```

On Windows, install DXC to require DirectX HLSL validation for the reduced
frontier, the selected 256-point FFT entry, the pinned GEMV compiler frontier,
all three reference-accessor artifact proofs, the selected LayerNorm entries,
and the selected complex-copy entry:

```bash
python demos/integrations/mlx/run_porting.py \
  --mlx-root C:/path/to/mlx \
  --require-directx-toolchain \
  --require-directx-gemv-compiler-frontier

python demos/integrations/mlx/prove_layer_norm_directx.py \
  --mlx-root C:/path/to/mlx \
  --work-dir .crosstl-mlx-porting/layer-norm-directx \
  --require-directx-toolchain

python demos/integrations/mlx/prove_copy_directx.py \
  --mlx-root C:/path/to/mlx \
  --work-dir .crosstl-mlx-porting/copy-directx \
  --require-toolchain

python demos/integrations/mlx/prove_rms_norm_specialization.py \
  --mlx-root C:/path/to/mlx \
  --work-dir .crosstl-mlx-porting/rms-norm-specialization \
  --require-directx-toolchain
```

Install the DirectX runtime extra separately to execute the value-sensitive
lookup fixture through Direct3D 12:

```powershell
python -m pip install -e ".[directx-runtime]" pytest-xdist
$env:CROSTL_RUN_DIRECTX_LOOKUP_DEVICE_TEST = "1"
python -m pytest -q -n auto `
  tests/test_translator/test_native_runtime_drivers.py `
  -k "directx_compute_runtime_executes_mlx_file_scope_lookup_on_device"
```

With the pinned MLX checkout available, run the generated arange proof
separately:

```powershell
$env:CROSTL_MLX_ROOT = "C:/path/to/mlx"
$env:CROSTL_RUN_DIRECTX_MLX_ARANGE_DEVICE_TEST = "1"
python -m pytest -q -n auto `
  tests/test_translator/test_native_runtime_drivers.py `
  -k "directx_compute_runtime_executes_translated_pinned_mlx_arange_on_device"
```

The selected 256-point FFT proof uses the same pinned checkout and executes the
translated artifact through the native loader on Direct3D 12 WARP:

```powershell
$env:CROSTL_MLX_ROOT = "C:/path/to/mlx"
$env:CROSTL_REQUIRE_MLX_FFT_DIRECTX_NATIVE_LOADER = "1"
python -m pytest -q -n auto `
  demos/integrations/mlx/tests/kernels/test_fft_native_loader.py::test_pinned_mlx_fft_executes_through_directx_native_loader
```

On macOS, require native compilation of the generated Metal round-trip artifact:

```bash
python demos/integrations/mlx/run_porting.py \
  --mlx-root /tmp/mlx \
  --require-metal-toolchain
```

The harness writes reports, generated artifacts, and command logs under
`<mlx-root>/.crosstl-mlx-porting`.

## Corpus Baselines

The reduced compiler and runtime proofs remain pinned to MLX commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e`. Their checked-in source hashes,
dispatch contracts, generated artifacts, and numerical results describe that
exact reference revision and are not relabelled when the upstream corpus moves.

The scheduled full-corpus scout is pinned separately to current MLX commit
`846d176227a0ac13d2667e58d2bb68b322109ab0`. Entry discovery for that revision
records 42 Metal source units and 17,319 host-visible entries with no discovery
diagnostics. The scout plans 84 source-target coordinates across DirectX and
OpenGL, writes resumable progress and portability reports, and reports
unsupported constructs as structured failures. This corpus scan measures
translation coverage only; it does not claim MLX runtime integration or
numerical parity on either target.

## Quantized Wide OpenGL Checks

The Linux CI job `mlx-quantized-wide-opengl` translates 18 selected
`affine_qmv_wide` entries from the current pinned corpus. It covers float32,
float16, and bfloat16 inputs; 3-, 5-, and 6-bit weights; and two or four vectors
per threadgroup. Each case uses group size 128, eight reduction lanes, batch
mode 0, and the source-faithful `[32, 2, 1]` workgroup: two logical 32-lane
SIMD groups and 64 total invocations. The vector count controls input-vector
tiling and does not change the fixed subgroup count. These cases exercise the
local eight-element dequantization array, including zero-offset and nonzero
pointer updates.

The gate checks the pinned host-dispatch and kernel source identities, project
report, source and artifact hashes, preserved array parameters, and exact
execution metadata. It explicitly lowers the two logical SIMD groups in shared
memory with software subgroup width 32, without KHR hardware-subgroup metadata,
then requires GLSL compilation and SPIR-V validation. Missing tools or a missing
pinned checkout fail the required CI job. Generated shaders, portability reports,
SPIR-V modules, and JUnit results are uploaded for inspection. No upstream MLX
source changes or out-of-bounds recovery policy are used by these cases.

To run the gate locally with `glslangValidator` and `spirv-val` on `PATH`:

```bash
CROSTL_MLX_ROOT=/path/to/pinned/mlx \
CROSTL_REQUIRE_MLX_QUANTIZED_WIDE_OPENGL=1 \
python -m pytest -q -n auto \
  demos/integrations/mlx/tests/kernels/test_quantized_wide_opengl.py
```

This is selected-entry compiler coverage, not completion of all 2,052 quantized
entries. It does not execute those entries, establish numerical parity, redirect
the MLX host runtime, or run the upstream MLX test suite on OpenGL.

## Current Translator Gaps

The historical five-file DirectX frontier at `4367c73b` translates after
canonical lowering of materialized Metal `trunc` wrappers
([#2091](https://github.com/CrossGL/crosstl/issues/2091)). Reduced controls execute
on original/generated Metal and OpenGL with exact finite bits; the corresponding
Windows execution remains required. This does not establish full-tree parity.
The same whole-file configuration at `9c3d3557` currently rejects bfloat vector
construction in all five units with `project.translate.directx-bfloat16-unsupported`
for `bfloat16vec2`, under [#1488](https://github.com/CrossGL/crosstl/issues/1488).
Selected-entry contracts and this whole-file boundary are separate coverage.

The five-file configuration targeting Metal at `9c3d3557` translates `arange`,
`random`, `rope` and `ternary`; all four generated files compile with warnings
treated as errors. `binary_two` still rejects the standard-array value returned
by `DivMod::operator()` ([#2093](https://github.com/CrossGL/crosstl/issues/2093)).
This is separate from constructor bitcast typing and does not establish native
execution of these complete files or full-tree parity.

The current-pin gated-delta backward kernels still need OpenGL floating-point
atomic lowering ([#1986](https://github.com/CrossGL/crosstl/issues/1986)). Struct
selection now preserves anonymous type-parameter defaults and evaluates supported
`enable_if` partial specializations ([#1985](https://github.com/CrossGL/crosstl/issues/1985)).
The selected `seq_gated_delta_vjp_float_128_128_24_24_1` entry retains the floating
field in `mlx_atomic<float>`. Its generated Metal now compiles with warnings fatal;
the macOS host workflow requires compilation of both the original source and the
translated entry and retains their libraries, compiler logs and identities.
DirectX float buffer addition and exchange now have native execution coverage.
Eight gated-delta backward configurations have required Windows DirectX and
original/generated Metal numerical gates, checking all six gradients against an
independent reference. This covers selected head layouts, checkpoint intervals,
partial segments and float16/float32 storage, not complete MLX host integration.
The DirectX gated-delta and attention fixtures validate the generated binary16
storage metadata before binding physical `uint16_t` resources. They upload the
original half bytes without numeric conversion, retaining structured strides,
guards, tail padding and shared-allocation identities. Half-storage metadata on
float32 or bfloat16 resources is rejected. Numerical reference checks and
tolerances are unchanged.

The attention backward row-dot stage has required Windows, Linux and macOS
gates. They translate all 24 declared float32/float16/bfloat16 entries across
dimensions 64, 72, 80, 96, 128, 192, 256 and 512. Exact dyadic cases distinguish
storage interpretation, indexing and reduction behavior; additional fractional
cases compare against decoded-input float64 dot products at `rtol=atol=1e-5`.
Dispatches preserve the upstream `(32, 1, 1)` workgroup shape and add inactive
rows to test whole-workgroup early returns. An empty-length case must preserve
all output guards. Original and translated Metal read back every buffer;
DirectX and OpenGL read back the writable output. These checks cover the row-dot stage,
not the remaining attention-gradient stages or full upstream attention tests.

OpenGL uses software subgroups of width 32 and validates each generated shader
with glslang before native execution. The fixture records source-scoped index
bounds derived from its dimensions, lengths and six flattened heads. These are
bounded host preconditions, not a general guarantee for 64-bit MLX indices.
Generated GLSL exposes float32 input storage for this stage: the host fixture
decodes the already-quantized float16/bfloat16 source values before uploading
float32 buffers. It preserves the original input files separately from the
uploaded bytes and records both storage widths. The `qL` parameter uses a
checked 16-byte std140 block. No source kernels are rewritten, and this fixture
does not establish automatic layout conversion for a full MLX backend.

The attention tile-derivative stage has separate Windows and macOS gates for
all three declared storage types. Its 24 configurations cover causal masking,
fully masked rows, nonzero tile origins, partial workgroups, fractional inputs,
and the shared score/derivative allocation used by the upstream host. The
32-byte mixed integer/float parameter block is packed at checked field offsets.
Word-sized readback transport preserves native 16-bit buffer strides; an odd
element count has explicit tail padding after the output guard. Exact cases
compare bitwise and fractional cases use `rtol=atol=1e-5` against quantized
float64 references. Original/generated Metal also checks input preservation
and allocation identity. These are kernel-stage checks, not end-to-end
attention backward or upstream host redirection.

The DirectX alias cases use offset-zero views. Simultaneous SRV/UAV access to
one ranged allocation remains unsupported by the process-isolated worker
([#2080](https://github.com/CrossGL/crosstl/issues/2080)). It rejects that request
before submission rather than separating the aliased storage. This limitation
also affects float32; it is independent of binary16 storage encoding.

The attention reduction stage has a separate resident-sequence gate on Windows,
Linux and macOS. Each case executes a prefix of `set`, `add`, `add`, `set` with
distinct source allocations and shared accumulator/output allocations. The
accumulator is initialized once and is not read back or reuploaded between
dispatches. Prefix checks distinguish accumulation from reset behavior and
verify untouched rows, partial workgroups and buffer guards. Dyadic and
fractional inputs use the source storage precision; all output comparisons
are bitwise, including the float32 accumulator and rounded low-precision result.

Windows and original/generated Metal cover float32, float16 and bfloat16.
OpenGL covers float32 and float16, with explicit decoded float32 uploads and
source-scoped bounds derived from the fixture dimensions. OpenGL bfloat16
narrowing remains tracked in [#1488](https://github.com/CrossGL/crosstl/issues/1488).
Metal submits the sequence in one command buffer and verifies every allocation
after completion. Portable drivers perform only the final output readback;
Windows retains native two-byte source/output storage. These are maintained
reduction-stage tests, not a complete attention operation or automatic MLX
host-runtime redirection. No upstream kernels are modified.

The Linux gate executes the same eight derivative configurations for each of
float32 and float16. The 32-byte parameter block uses reflected mixed-field
layout metadata, preserved by the native runtime contract. The test uploads
already-quantized inputs into the generated float32 storage and compares actual
float32 readbacks against decoded source-type references; it does not round
outputs on the host to conceal a missing shader conversion. Original inputs,
uploads, generated modules, parameter layouts and guarded outputs are retained.
Source-qualified index bounds cover the dispatched tile. Actual shared buffer
allocation is tested alongside separate allocations. Bfloat16 derivatives remain
blocked by [issue #1488](https://github.com/CrossGL/crosstl/issues/1488); no complete
attention operation or automatic backend-wide repacking is claimed.

DirectX also requires a reduced native binary16 conversion gate. It shares
258,052 input words with the OpenGL and original/generated Metal controls,
including midpoint neighbors, subnormal boundaries, overflow and signed zeros.
Separate cases exercise implicit and explicit conversion, helper boundaries,
structure fields, vector components and single evaluation. Generated HLSL rounds
the representation bits to nearest-even before constructing a native half value;
the attention derivative buffers retain their two-byte element strides. This
addresses the truncating native-cast behavior tracked in
[issue #1993](https://github.com/CrossGL/crosstl/issues/1993), without changing the
24 derivative configurations or their tolerances. Passing the reduced gate is
not a substitute for passing those MLX cases or the full upstream suite.

Reduced Metal roundtrip tests execute float atomic addition and exchange on
scalar buffers and aggregate fields. They check 513 competing updates, returned
values, final storage and single evaluation of address/value operands. Separate
isolated cases preserve signed zeros, infinities, NaN bit patterns and subnormal
inputs against original-source Metal controls. The macOS workflow requires these
readbacks and retains the original/generated sources, compiled libraries and raw
buffer contents. Float min/max and bitwise atomics remain unsupported; local and
read-only storage are rejected. These tests do not establish portable float
atomic support or a complete gated-delta port.

Unknown predicates, unresolved constraint result types and multiple viable
constrained partials produce diagnostics instead of selecting a primary
declaration with a different layout. Ordering multiple viable partials remains
tracked in [#1987](https://github.com/CrossGL/crosstl/issues/1987). Native
regression tests require 194 exact readback words distinguishing fractional and
integer fields; the same tests run on Windows/DirectX, Linux/OpenGL and
macOS/Metal, with original-source controls on macOS. Parameter inference and named
Boolean free-function constraints have separate native checks. These reduced
tests do not establish complete gated-delta execution or full MLX backend support.

The latest full-corpus scout at MLX commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e` discovered 40 Metal units, 841
include dependencies, and 120 planned target artifacts. It emitted `arange.metal`
for DirectX, OpenGL, and Vulkan plus a Vulkan `arg_reduce.metal` artifact before
the 900-second limit expired without a canonical project report. That recorded
attempt predates durable progress files and therefore does not establish an
active full-corpus coordinate. Current runs write an atomic checkpoint containing
the completed, active, and pending job coordinates, accumulated diagnostics, and
partial artifact matrix. A later invocation resumes only artifacts whose source,
generated output, and source-remap identities still match. CrossGL/crosstl#1376
continues to track bounded materialization runtime.
[#1676](https://github.com/CrossGL/crosstl/issues/1676) remains the
repository-level acceptance target.

A bounded checkpoint probe against the same pin planned 80 DirectX and OpenGL
jobs. It preserved four completed records, identified `binary.metal`/DirectX as
active with 75 jobs pending, and validated the checkpoint after a bare process
interrupt. Resuming the probe retained those four records and returned directly
to the active coordinate. This verifies interruption recovery, not completion
of the full corpus or numerical parity.

Materialization charges configured work budgets to unique reachable entries,
helpers, struct specializations, and actual type-environment resolution. Exact
source-analysis snapshots and span indexes are reused with bounded retention;
the generated source remains unchanged. Artifact metadata keeps reachable
specializations, dependency-discovery work, and pruned eager candidates
separate.

A selected DirectX replay of `quantized.metal` now emits one artifact with zero
translation diagnostics for `affine_quantize_float_gs_32_b_2`. It materializes
five reachable specializations and three concrete records while pruning 110,861
unreachable candidates. The generated HLSL is 4,557 bytes with SHA-256
`9e7e4af1ceb66b2fa93e1029d370b67e91c2972c27c70bc8892c0866fb6b76b9`.
This path verifies the completed template-member and owner-dependent `constexpr`
work tracked by CrossGL/crosstl#1476 and CrossGL/crosstl#1672. After unreachable
materializations are pruned, this selected float specialization contains no live
native-width 16-bit types, and its report records `requiredCapabilities=[]`.
Native 16-bit HLSL support under
[#1799](https://github.com/CrossGL/crosstl/issues/1799) remains resolved and
validated elsewhere, including the pinned bfloat frontier, but is not required
by this selected specialization. Concrete `static_assert` evaluation under
[#1800](https://github.com/CrossGL/crosstl/issues/1800) is resolved for this
selected entry. Contextual narrowing under
[#1801](https://github.com/CrossGL/crosstl/issues/1801) remains recorded in the
broader DirectX toolchain evidence, but this selected `bits = 2` entry resolves
`OutType` to source `uint32_t` and generated HLSL `uint`. The output buffer's
source `uint8_t` element type is narrower than that accumulator. Its final
store retains this conversion before writing the expanded `uint` resource:
`out_[uint((out_index / uint64_t(writes_per_reduce)))] = (uint(output) & 255u);`.
The artifact contains
no remaining `static_assert`. Its ordinary type contract needs no native-16-bit
profile uplift, while the generated wave intrinsics keep the configured project
and compiler target scoped to `directx-12`. Official DXC validation with profile
`cs_6_0`, no `-enable-16bit-types`, and `-WX` passes.
The locally generated DXIL was nonempty; its byte size is not treated as a
cross-version compiler invariant. This evidence covers translation and compiler
acceptance only; it does not claim runtime execution or numerical parity.

The adjacent DirectX entry `affine_gather_qmv_fast_float_gs_32_b_2` also emits
one artifact with zero translation diagnostics. It materializes 10 reachable
specializations and eight concrete records while pruning 110,861 unreachable
candidates. The specialized `load_vector_float_float_16_2` helper retains the
caller's `thread U x_thread[values_per_thread]` storage with
`values_per_thread = 16`; HLSL represents it as an `inout float[16]` plus a
base offset and proves all four writes in each `i += 4` iteration. The
`qdot_float_16_2` helper retains the read-only `uint8_t` view over its
`uint32_t` storage resource. Each byte read selects the backing word and lane,
and the helper call composes the root word offset, byte-view offset, row offset,
and local alias offset without treating bytes as typed 32-bit elements.

The gather path also materializes the overloaded `elem_to_loc` helper as
`elem_to_loc_uint32_t`, preserving the source `uint32_t` index type. Its HLSL
call carries the independent shape and stride resource offsets, and no
unresolved `elem_to_loc(...)` call remains in the artifact.

The source `adjust_matrix_offsets` helper receives five storage pointers by
mutable reference. DirectX keeps each resource handle unchanged and passes its
logical offset as `inout int64_t`. The entry owns the offset variables, the
helper updates them, and the subsequent `qmv_fast_impl_float_32_2` call consumes
all five updated values. This preserves source pointer rebasing without passing
resource handles by reference or discarding writes to by-value offsets.

The pinned `gather_qmv` host dispatch in `mlx/backend/metal/quantized.cpp` sets
`bk = 32` and `MTL::Size group_dims(bk, 2, 1)`. The project rule therefore
emits `[numthreads(32, 2, 1)]`. The kernel's simdgroup indices require a
32-lane subgroup, so the generated HLSL also emits `[WaveSize(32)]` and requires
Shader Model 6.6. The resulting artifact is 16,461 bytes with SHA-256
`c3a0b1b98cd7bfe3619f5be64c0b041028c2dcf61836e4b7c2de831be61bc9d9`.
Windows CI compiles it with DXC profile `cs_6_6` and `-WX`. This is selected-entry
evidence for the fixed-array alias work tracked by
[#1497](https://github.com/CrossGL/crosstl/issues/1497) and the read-only storage
view work tracked by [#1546](https://github.com/CrossGL/crosstl/issues/1546),
the resource pointer-offset contract tracked by
[#1518](https://github.com/CrossGL/crosstl/issues/1518),
with the broader subgroup contract tracked by
[#1894](https://github.com/CrossGL/crosstl/issues/1894). Issue
[#1786](https://github.com/CrossGL/crosstl/issues/1786) established the exact
subgroup-width metadata and DirectX enforcement model. The remaining issues
cover broader cross-target, writable-view, alignment, subgroup fallback, and
runtime acceptance criteria. This proof does not dispatch the kernel through
Direct3D, run MLX tests, or claim numerical parity.

The current-corpus `affine_quantize_float_gs_32_b_2` entry now has a
selected OpenGL native-loader proof at commit
`846d176227a0ac13d2667e58d2bb68b322109ab0`. The project configuration supplies
three source-qualified index-range assertions for `in_index + i`, `gindex`, and
`out_index / writes_per_reduce`, each with inclusive bounds
`[0, 2147483647]`. These are explicit host/runtime portability preconditions
used to justify legal GLSL `uint` subscripts; they are not inferred or enforced
at runtime.

The target-scoped Metal option
`project.source_options.metal.target_options.opengl.software_subgroup_width = 32`
selects a fail-closed software implementation of this entry's scalar
`WaveActiveMin(float)`, `WaveActiveMax(float)`, and
`WaveShuffleDown(uint, int)` operations. This selected proof uses one compute
entry with local size `[32, 1, 1]`. The bounded mode can also partition a
concrete workgroup of at most 1,024 invocations into multiple 32-lane logical
subgroups when local size X is divisible by 32; unsupported operations, payload
types, helper ownership, raw subgroup builtins, or unproven lane-dependent
control flow still reject translation. The default hardware KHR subgroup path
is unchanged.
The software artifact uses collision-safe shared scratch storage and eight
barriers, emits `CROSSTL_SOFTWARE_SUBGROUP_WIDTH`, and emits no KHR subgroup
extension, `gl_Subgroup*` use, `CROSSTL_REQUIRED_SUBGROUP_WIDTH`, or hardware
subgroup-width runtime metadata.

The exact generated GLSL is 7,949 bytes with SHA-256
`61cc1cf6f33ecab9919191db3f68bf57267d549fb3d638d595d464e68d2f494c`.
Materialization records three concrete and eight reachable specializations, zero
dependency-discovery work, and 104,702 pruned candidates. Linux CI compiles it
for OpenGL/SPIR-V 1.3 with `glslangValidator`; `spirv-val` accepts the module,
whose disassembly contains eight control barriers and no group-nonuniform
instruction. The runtime package reflects `wBuffer` at binding 0 and the
`out_Buffer`, `scalesBuffer`, and `biasesBuffer` read-write resources at
bindings 1 through 3. Through surfaceless EGL and Mesa software OpenGL, one
32-thread dispatch transforms `[0, 1, 2, 3]` repeated eight times into eight
packed `uint32` values of `27`, scale `-1`, and bias `3`.
A second dispatch uses `[0, 0.5, 1.5, 3]` repeated eight times and requires
eight packed values of `47`, with the same scale and bias. This midpoint case
checks Metal's halfway-away rounding; the generated helper also preserves the
source's narrow shuffle operand conversion. The shader reference was reviewed
against original Metal endpoint and midpoint readbacks before being updated.

This is exact translation, packaging, native execution, and numerical parity
for these two bounded affine-quantize workloads. It does not redirect the
MLX host runtime, run MLX's test suite, cover other quantized entries, or turn
the constrained software subgroup mode into a general divergent-subgroup
implementation. [#1515](https://github.com/CrossGL/crosstl/issues/1515) records
the completed index-width normalization contract, while
[#1894](https://github.com/CrossGL/crosstl/issues/1894) continues to track
broader subgroup fallback coverage.

The historical `4367c73b60541ddd5a266ce4644fd93d20223b6e` corpus retains a
separate compile-only OpenGL gate for `affine_gather_qmv_fast_float_gs_32_b_2`.
It emits a 16,042-byte GLSL artifact with SHA-256
`652a1b94628622537d565ff5da0174bfb096b77653971f89a0614e2425548cff`, zero
project diagnostics, 10 reachable specializations, and eight concrete
materializations. The project report retains the pinned `32 x 2 x 1`
workgroup rule and four explicit index-range preconditions for the gather
resource lookups. The generated `load_vector` and `qdot` helpers preserve the
16-element private array. Reinterpreted storage pointers pass their root byte
base separately from their logical byte-view offset, so `qdot` composes the
incoming word offset, row offset, and loop index without capturing caller-local
expressions or scaling the word offset twice. Linux CI compiles the artifact for
OpenGL/SPIR-V 1.3 and validates the module with `spirv-val`. This advances the
cross-target contracts tracked by [#1497](https://github.com/CrossGL/crosstl/issues/1497),
[#1518](https://github.com/CrossGL/crosstl/issues/1518), and
[#1546](https://github.com/CrossGL/crosstl/issues/1546); their broader acceptance
criteria remain open. This remains compile-only evidence and does not claim
OpenGL dispatch, MLX test execution, or numerical parity.

CrossGL/crosstl#1659 is complete; resource-register relocation no longer blocks
the selected aggregate DirectX replay. The checked-in evidence also records
full-kernel Vulkan replays of `fp_quantized.metal` and `quantized_nax.metal` as
failed after selected materialization contracts. Only the explicitly documented
reduced quantized fixtures carry validator evidence.
CrossGL/crosstl#1660 tracks resource coherence and volatility qualifiers that
are currently absent from Metal round-trip artifacts.
Native toolchain validation now runs on the target operating systems, completing
the coverage tracked by CrossGL/crosstl#1312. The bounded Direct3D and OpenGL
compute runtime drivers tracked by CrossGL/crosstl#1472 and
CrossGL/crosstl#1516 are also complete. Packaged artifacts can now be dispatched
through the native-loader bridge; remaining source-specific execution gaps are
recorded with the individual proofs below. CrossGL/crosstl#1388 still tracks the
broader artifact execution metadata contract. OpenGL type aliases are inlined
before host-interface reflection and are not exposed as runtime resources.
CrossGL/crosstl#1471
tracks entry-point ownership for reflected constants in runtime reports.
CrossGL/crosstl#1474 is represented by exact per-artifact DirectX
bfloat16 storage and conversion evidence in the pinned project report. The
harness fails closed unless each emitted source retains the exact
`bfloat16Lowering` object and corresponding `requiredCapabilities` list; this
does not extend the bounded runtime proof to bfloat16 or claim numerical parity.
Every custom DXC invocation derives its effective profile and compiler
arguments from the emitted HLSL through `crosstl.project.directx_toolchain`.
Generated `float16_t`, `int16_t`, or `uint16_t` types select at least Shader
Model 6.2 and add `-enable-16bit-types`; ordinary HLSL retains its selected
profile and command. These checks do not prove Direct3D 10 or 11 compatibility;
CrossGL/crosstl#1670 tracks explicit target profiles, feature gates, and
compiler selection. CrossGL/crosstl#1669 tracks
the fixed arrays of resource aliases introduced by the pinned revision's wide
quantized matrix-vector helpers. CrossGL/crosstl#1671 originally tracked
workgroup backing provenance through nested FFT helper parameters. A dedicated
DirectX project replay now selects the forward complex 256-point entry
`fft_mem_256_float2_float2` from the complete pinned `fft.metal` source. The
host planner's `MTL::Size(1, threadgroup_batch_size, threads_per_fft)` dispatch
maps to `[numthreads(1, 1, 64)]`; the report and generated HLSL must retain that
axis order. The replay materializes 24 reachable template specializations and
21 of the 22 configured function constants. Function constant 3 is pruned
because its Rader transform branch is unreachable for this power-of-two plan.
The generated artifact contains no first-class workgroup pointer residue and is
locked by SHA-256 and byte count. Windows CI compiles it with DXC using
`cs_6_2`, `-enable-16bit-types`, and warnings as errors. It then packages and
reflects the artifact, requiring 8-byte `float2` strides for the input and
output resources and a 16-byte constant-buffer allocation for the generated
`uint3` dispatch input. The native-loader request derives that input from the
physical workgroup count before Direct3D 12 WARP dispatch and readback.

The runtime check supplies five inputs for this 256-point plan: an index-1
unit impulse, seeded float32 complex data, a complex constant, alternating real
values, and a complex impulse at the final element. It compares all outputs
with analytical references or a double-precision direct DFT, using `2e-4`
absolute and relative tolerance. Eight trailing scalar guards must remain
exactly unchanged after each dispatch. Each native test translates and packages
the source once, then executes all five inputs in the existing CI job. This
checks 2,560 result values and 40 guards for one bounded transform plan, not
general FFT coverage. It does not redirect the
MLX host runtime, run the MLX test suite, or establish broad FFT or backend
parity. The aggregate DirectX source materialization limit remains tracked by
[#1916](https://github.com/CrossGL/crosstl/issues/1916); broader runtime grid and
resource-layout contracts remain tracked by
[#1542](https://github.com/CrossGL/crosstl/issues/1542) and
[#1543](https://github.com/CrossGL/crosstl/issues/1543).

The current-corpus fixture selects commit
`846d176227a0ac13d2667e58d2bb68b322109ab0`, where MLX generalizes FFT
workgroup storage through `FFTIOTypeTraits` and adds function constant 22 for
the Bluestein twiddle-table path. Replaying the same 256-point entry with that
constant set to `false` materializes 37 specializations, records 42 reachable
specializations before pruning, and concretizes 21 of 23 configured function
constants. The source reaches two compatible `ReadWriter_float2_float2__load`
overload bodies. DirectX workgroup-pointer reachability now analyzes both
bodies, merges their forwarded calls and access ranges, and retains the
concrete `fft_mem_256_float2_float2_shared_in[256]` backing identity. Generated
helpers transport `crosstl_ptr_buf` as an integer offset instead of a bare
first-class workgroup pointer expression. The omitted default `twiddles =
nullptr` argument is carried into the CrossGL intermediate before target
lowering. Because the materialized `use_twiddle_table = false` call chain makes
every twiddle dereference unreachable, DirectX removes that unobserved resource
parameter through the forwarding chain rather than inventing a backing buffer.
A null pointer that can be observed or dereferenced still fails closed.

The current source emits a 176,506-byte HLSL artifact with SHA-256
`f8c8c4b18cabaa7f2997dcdb68b8a9f32645f274c271e454e44dfe463cd4a1ff`
and zero project diagnostics. All 20 native-16 ``power`` shift counts are
explicitly promoted to ``int`` before HLSL shifting, matching Metal/C++ integer
promotion rather than retaining minimum-precision count semantics. Windows CI
compiles it with DXC using `cs_6_2`,
`-enable-16bit-types`, and warnings as errors, then packages and dispatches it
through Direct3D 12 WARP. The same five-input workload, reflected
resource layouts, physical workgroup count, and `2e-4` output tolerances are
required for this exact current checkout. This lowering does not permit
arbitrary first-class workgroup pointers: roots without a concrete shared-array
identity or extent still fail closed, and
[#1518](https://github.com/CrossGL/crosstl/issues/1518) continues to track the
broader pointer-offset contract. The historical proof remains separately
recorded under its own commit provenance; it is no longer being used as a
substitute for current-corpus evidence. Its reviewed artifact is 145,277 bytes,
SHA-256 `d5fa1ae408154eae551c1ad20f02a81c47f95fe4be7df16714b4ff7bf4e8048b`.
Both artifact revisions preserve reflected bindings and dispatch dimensions;
native Windows numerical execution remains required before merging a reference
update. Successful DXC compilation alone does not satisfy that requirement.

[Windows run 37525218285](https://github.com/CrossGL/crosstl/actions/runs/37525218285/job/112481552970)
verifies both revised artifacts at `a688ad15`. The `mlx-fft-directx` artifact
(11443897833) retains all ten input fixtures, dispatch plans, compiler commands,
HLSL/DXIL snapshots and returned values. Independent DFT recomputation from the
saved inputs verifies 5,120 values and 80 exact guards across the two source
pins. Maximum absolute error is below `6.63e-5`, within the unchanged tolerances.
Outputs start at `-999.0`, not the expected answers. Source identities match
their contracts, and both compiled modules have SHA-256
`59c265eecbfff3b88b2746f535e055e1cf976fa0d76755d4eb7157a79c87f81b`.
This verifies these five controls for each historical pin; it does not establish
the `9c3d3557` corpus, arbitrary FFT plans or physical driver-upload bytes.

The current FFT source now also emits a 109,547-byte GLSL artifact with
SHA-256
`5c6fefea7315d7d091641d024aea27bbdcecf1f2f4b7a63a2db5af401e036512`
and 84 source-remap mappings. OpenGL keeps the 21 reachable function constants
as 21 deferred specialization constants and uses the same `[1, 1, 64]`
workgroup contract. A fifth current-source index assertion bounds
`batch_idx + index + r`. A sixth assertion bounds the resource helper's
`reference.offset + index` to `[0,255]` for this exact fixture: buffers are
unrebased, `batch_size=1`, and `min(2*lane + 128*e,254)+r` covers `[0,255]`
for `lane` in `[0,63]`, `e` in `[0,1]`, and `r` in `[0,1]`. This assertion
must not be reused for other batch sizes, dispatches, or rebased resources.
Generic specialization lowering now follows a
statically null storage pointer through direct forwarding calls and omits it
only when every reachable use is dead; observable null uses still fail closed.
The GLSL generator also decodes materialized Metal `vec<T,N>` constructor
names and emits every collision-safe overloaded resource-specialization body.
The generated module compiles with `glslangValidator`, passes `spirv-val`, and
has 19 control barriers and no group-nonuniform instructions.

Host reflection now records each runtime `vec2` SSBO element as `float32x2`
with 8-byte size, stride, and alignment under `std430`; the two scalar argument
blocks retain their 16-byte `std140` block identities. The resulting four-
resource native ABI has no blocked variants and publishes a ready deferred
SPIR-V request. Linux CI specializes and dispatches that request through a
surfaceless Mesa llvmpipe OpenGL context. All five inputs must match the
independent references within the required `2e-4` tolerances, with exact guard
preservation. The first execution must publish the verified compilation cache;
the remaining four must use it. This is one exact current-pinned plan and does not establish full
FFT, MLX host-runtime, or backend parity.

A dedicated project replay now translates the complete pinned `fft.metal`
source to one standalone OpenGL compute shader with a 4,096-specialization
limit and a 2,097,152-item materialization work budget. The report records 99
unique reachable template specializations, 22 function constants, no unsupported
materializations, and no project diagnostics. Four unsigned host-index
assertions and five entry-point-scoped workgroup access assertions make the
required dispatch preconditions explicit for memory extents from 256 through
4,096 elements. The Linux proof compiles the emitted GLSL to OpenGL SPIR-V 1.3
with `glslangValidator` and validates the binary with `spirv-val`. This proves
artifact construction and native toolchain acceptance; OpenGL runtime dispatch
and numerical parity with MLX's Metal backend remain outside this check.

The pinned `gemv.metal` DirectX compiler frontier verifies source SHA-256
`c34db77e61c1fea01f7f5d319a0bec1029a253e54d66bbce9009f32fe828ce9f` and
source size 5,383 bytes before translation. The project report must contain one
clean translated artifact, all 226 materializations with no unsupported
records, no unresolved materialization residue, and no standalone pure
value-discard statements such as `lid;`. Its generated SHA-256 and byte count
are checked against the emitted HLSL. Project configuration sets
`[project.workgroup_size_rules]` for `gemv.metal` to `[32, "BN", "BM"]`; the
report must retain the normalized `["32", "BN", "BM"]` rule. Exactly 224 of the
226 materializations must be host-named; the other two are signature-selected
`elem_to_loc<uint>` and `elem_to_loc<uint32_t>` helpers. Exactly 224 report
execution entries must join the host-named records by
`(hostName, materializedName)` identity. The artifact must expose the exact
target set `CSMain`, `CSMain_2`, ..., `CSMain_224`, independently of report or
materialization list order.

The resolved report sizes must be exactly `[32, 1, 1]`, `[32, 1, 4]`,
`[32, 1, 8]`, `[32, 2, 1]`, `[32, 4, 1]`, `[32, 8, 1]`, and `[32, 16, 1]`.
For every target entry, the emitted `numthreads` declaration must equal that
entry's report contract. This establishes exact workgroup-size specialization
for the generated aggregate artifact. DXC compiles representative scalar,
complex/Wave, and gather/constant-pointer paths (`CSMain`, `CSMain_85`, and
`CSMain_113`) with `cs_6_2` and `-enable-16bit-types`; all three invocations
must produce zero diagnostics. A second `lib_6_6` invocation retains
`-enable-16bit-types` while exporting and code-generating all 224 functions in
one DXIL library.

The library compile admits exactly 224 `numthreads ignored without accompanying
shader attribute` warnings caused by using a library profile. The gate derives
the expected warning source-line counts from the seven generated `numthreads`
forms and requires exact severity, message, source expression, and count
matches. Any unused-value warning, error, or other diagnostic fails the gate.
[#1786](https://github.com/CrossGL/crosstl/issues/1786) established the exact
wave-size specialization contract. This existing GEMV library frontier does not
apply that contract, so library compilation proves that DXC accepts and
code-generates every exported function but does not establish wave semantics,
runtime execution, numerical parity, or whole-kernel semantic validity.

The pinned `gemv.metal` OpenGL gate uses the same 4,096-specialization limit and
2,097,152-item materialization work budget. It requires SHA-256
`c34db77e61c1fea01f7f5d319a0bec1029a253e54d66bbce9009f32fe828ce9f`, one
source unit, all 226 specializations materialized without unsupported records,
and all 224 entry-scoped GLSL artifacts emitted without diagnostics. Every
artifact must retain its source entry identity, generated hash, source map,
source remap, resolved workgroup size, and exact subgroup-width contract.

The project configuration records `[32, "BN", "BM"]` as the workgroup-size
rule and 32 as the subgroup width. Generated shaders require
`GL_KHR_shader_subgroup_basic` and return before dispatch work when
`gl_SubgroupSize` differs from 32; the host must also query
`GL_SUBGROUP_SIZE_KHR` and reject a mismatch before dispatch. Five source-scoped
index-range assertions record the host preconditions that permit MLX's 64-bit
batch and gathered-matrix indices to narrow to OpenGL's 32-bit index domain.
These are explicit runtime preconditions, not inferred ranges, and this harness
does not enforce them at runtime.

Linux CI compiles all 224 artifacts for OpenGL SPIR-V 1.3 with
`glslangValidator` and validates every resulting SPIR-V 1.3 module with
`spirv-val`. [#1894](https://github.com/CrossGL/crosstl/issues/1894) tracks a
semantics-preserving fallback for devices without the required native subgroup
width. The gate proves complete translation and compiler acceptance under the
recorded project contracts; it does not establish runtime execution, host
integration, or numerical parity.

The checked
[`contracts/gemv.native-loader.dispatch.json`](contracts/gemv.native-loader.dispatch.json)
contract selects `gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0` at the
historical `846d176227a0ac13d2667e58d2bb68b322109ab0` revision. Its 6,981-byte
`gemv.metal` source has SHA-256
`0bd8bde0c867a17c345a3651f9f0a6c2909e0c74e76ea2a08f373fe4dcafaeda`.
The host-derived `gemv_axbpy` branch covers one contiguous float32 vector-matrix
product with `M=1`, `N=32`, `K=32`, no gathered or non-contiguous batch, and no
axpby bias path, matching
`python/tests/test_blas.py::TestBlas::test_matrix_vector`. It fixes parameters
`BM=1`, `BN=2`, `SM=8`, `SN=4`, `TM=4`, and `TN=4`, workgroup
`[32, 2, 1]`, subgroup width 32, and dispatch `[1, 1, 1]`. The normalized
contract identity is
`sha256:6b3bb18d130159f13874f06668b536fe4b9270ffbb2a1f44b6d9aac257aba7e4`;
its single variant and artifact identities are
`sha256:acaba2ec4813a364b06d95a5136bda80351797591a4d9f0b3d195f85da287fe3`
and
`sha256:34eab189b10cc699f06f4cbed04faae41a2658a2a3665a6866ed987f5946949a`.

Entry-scoped translation materializes only the selected GEMV and
`elem_to_loc_uint`, with no unsupported record or project diagnostic. The
8,382-byte HLSL has SHA-256
`6c9a9cff75874925dda1562ab18b512bb452ac1bdd5b45275b9590bd692f55af`,
retains `[numthreads(32, 2, 1)]` and `[WaveSize(32)]`, and passes official DXC
1.9.2602.24 under `cs_6_6`, `-enable-16bit-types`, and warnings as errors.
Direct3D does not guarantee that a multidimensional workgroup's flattened
`SV_GroupIndex` values are contiguous within each physical wave, so this entry
explicitly enables target-scoped 32-lane software subgroups. Logical subgroup
and lane IDs are `SV_GroupIndex / 32` and `SV_GroupIndex % 32`; a 64-float
`groupshared` array carries each shuffle, with two
`GroupMemoryBarrierWithGroupSync` calls. The source lane is validated before
addition, preventing unsigned-delta wrap, and an out-of-range shuffle returns
the calling invocation's value. The artifact contains no `WaveReadLaneAt`,
`WaveGetLaneIndex`, or physical-wave atomic allocator. `[WaveSize(32)]` remains
a source/reflection contract, not a dependency on physical lane topology.

The replaced 8,410-byte physical-wave artifact is retained as rejected
diagnostic evidence under SHA-256
`f8f1107d0de251fd300c7a16ce6638796bd08dd2eadd8f7959e37c78d0aa170d`.
Windows workflow run 33268998061, job 99143984804 mismatched all 32 outputs:
its reduction substituted logical lanes 5 through 8 with physical lanes 21
through 24, with maximum absolute error 1.90625. That exact signature rules out
a tolerance adjustment or merely guarding invalid high-lane reads.

The OpenGL target needs only the selected matrix-index assertion
`uint64(bm + tm) * marix_ld + out_col + tn` in the unsigned 32-bit range. Its
7,754-byte GLSL has SHA-256
`2a295b13be5c7bed11f01b86025dd7e0509c9003e515958fb1b24bc0b5ed07f1`
and partitions the 64-thread workgroup into two logical 32-lane subgroups with
a 64-float shared shuffle scratch array.

The reference update compares complete historical/current bodies and all 15
reflected resources per target. HLSL changes retain source-width matrix/stride
arithmetic and carry the logical invocation index through helper calls. Both
targets preserve the source's 16-bit shuffle offset conversion; OpenGL also
adds explicit shared-memory ordering and checks the source lane before addition.
Bindings, entry geometry, template materializations and upstream source hashes
are unchanged. Both historical references are reproduced exactly and compiled
before comparison; all four historical/current artifacts pass their validators.

Both target-specific software-subgroup analyses recognize either `value > 0`
or the integral-equivalent `value >= 1` as a canonical positive-to-zero
halving loop when paired with `/= 2` or `>>= 1`. That admits the source's
`sm >= 1; sm >>= 1` segmented shuffle reduction. DirectX additionally requires
one bounded compute entry, a concrete width-compatible workgroup, explicit
calling-invocation fallback, a supported scalar shuffle, an unambiguous helper
call graph, logical invocation identity, and statically uniform control flow;
violations fail before artifact emission. OpenGL retains rejection for wider
bounds, mutation, nontermination, escaping control flow, indirect calls, and
nested helper calls. `glslangValidator` and `spirv-val` accept the emitted
OpenGL module; SPIR-V contains three control barriers and no group-nonuniform
instruction or hardware-subgroup extension.

Both runtime packages expose the same 15 logical resources: matrix, vector,
bias placeholder, writable output, batch shape, three signed 64-bit batch
strides, and seven scalar argument blocks. A deterministic binary-fraction
workload uses `vector[row] = (row + 1) / 32` and
`matrix[row,column] = (row + column + 2) / 64`; output column `c` must equal
`5.5859375 + 0.2578125 * (c + 1)` for all 32 columns at `1e-5` absolute and
relative tolerance. Linux arm64 Mesa llvmpipe executes and reads back this
software-subgroup workload in required mode, and Windows CI requires the same
request through Direct3D 12 WARP.
The local OpenGL and hosted Windows readbacks match an independent
exact-rational calculation from the uploaded binary32 inputs for all 32 columns.
Windows run 37417907211, job 112120535710 retains the executed 7,108-byte DXIL
under SHA-256
`8df8a620174a9813e8f2a04437710f035b1ac369c3c2de5ae74e6cf051e6aecc`.
That job subsequently failed at the separate MXFP4 reference check; it is not a
passing whole-project run. Existing Windows/Linux
jobs retain the project report before identity checks, generated sources,
runtime package, compiler records, dispatch inputs, results and JUnit output,
including on failure. No extra runner or relaxed tolerance is required.

This bounded proof does not replace the historical 224-entry aggregate gates:
it adds numerical execution for one host-valid entry at `846d1762`, not the
current `9c3d3557` corpus. Gather, wide, batched, axpby, and the remaining host-named GEMV
entries, MLX host redirection, the full MLX test suite, and selected-entry Metal
compiler validation remain outside the claim. The separate native Metal
aggregate baseline continues to cover the existing round-trip boundary.

The checked
[`contracts/fp_quantized.native-loader.dispatch.json`](contracts/fp_quantized.native-loader.dispatch.json)
contract selects the historical `846d1762` entry
`mxfp4_quantize_dequantize_float_gs_32_b_4_hgs_false` from the 9,700-byte
`fp_quantized.metal` source with SHA-256
`ef4ba099710a63a0b5d27d3e5ce69a8528bee8f1757805aa606c8d8e43de18d4`.
It records the exact host branch from `quantized.cpp` and implementation in
`fp_quantized.h`: float32 MXFP4, group size 32, four payload bits, row-contiguous
layout, and no global scale. The host formula assigns one value per thread,
selects workgroup `[32, 1, 1]`, and dispatches one workgroup. The normalized
contract identity is
`sha256:5256e32b364ac303a6873f28b5ac3e9a1a811ac5c38bc41a977bce9191a025ed`;
its single variant and artifact identities are
`sha256:ebd6ab3f40f5839764f592943180ba64f11a66f10a5561b9468019032c04df8a`
and
`sha256:bde1bfa31c116a52a1dc3b6e546dfa2ee43dc968719393ba04f629b4e2d95319`.

Entry-scoped translation materializes only the selected specialization with
`T=float`, `group_size=32`, `bits=4`, and `has_global_scale=false`. The
9,816-byte HLSL has SHA-256
`ff64661e8c32779e3f397436b33f9effd042ea413678c7c1214d9ede7e656f81`,
retains `[numthreads(32, 1, 1)]` and `[WaveSize(32)]`, and passes DXC under
`cs_6_6`, `-enable-16bit-types`, and warnings as errors. Its compiled DXIL is
4,836 bytes. This artifact explicitly enables the DirectX-only
`project.source_options.metal.target_options.directx.widen_native_float16`
mode. Source `as_type<float16_t>(uint16_t)` reconstructs its exact payload as
float32 with integer IEEE-754 masks, and logical `float16_t` locals, function
parameters, and returns stay widened through the selected arithmetic path.
Optimized DXIL contains `uitofp i32` and `fmul float`, with no
`LegacyF16ToF32`, `half`, `fptrunc`, or `fpext`. Default DirectX native
binary16 lowering remains exact `asfloat16`/`asint16`/`asuint16` and is
unchanged unless this source-scoped option is enabled.

Four rejected Windows artifacts establish why every native-half boundary must
be absent. The 7,809-byte HLSL under SHA-256
`3591e38d20a612b4061fe3154ef0ea3deb035283294fbd27376ef90627569361`
produced numeric `uitofp i16 ... to half`; workflow run 33271117475, job
99149649480 collapsed all 28 nonzero values to signed zero on WARP. Exact
`bitcast i16 ... to half` produced same-size HLSL under SHA-256
`4e8044758d65b6b2c189092ce56fff3c5ba7948221883de490c1a4b9c5563352`,
but run 33272842347, job 99154326814 still consumed the intentionally
constructed binary16 subnormal in `fmul half` and produced the same 28 signed
zeros. Moving the multiply to float32 produced 7,909-byte HLSL under SHA-256
`938ca6fac1c47ea633453836b5d76833c294853bb92d6a410a2c4772dd7fa627`,
but `dx.op.legacyF16ToF32` yielded the identical result in run 33274360343,
job 99158370210. Integer decoding then produced 9,240-byte HLSL under SHA-256
`936088a24a6b575e50dc97e16a4c0dca63a76200ddd94d5211e4bf312fec1625`,
but its remaining `fptrunc float to half`, half sign operation, and
`fpext half to float` still returned the same 28 signed zeros in run
33275550062, job 99161501105. Removing every half instruction produced the
9,118-byte artifact under SHA-256
`7afdc612f9091ae47abca8c4fd9d2171e8ea42c6539e02a40bbad2de7d1a7c6a`,
but run 33277494856, job 99166677942 still returned the same signed zeros.
Its DXIL exposed the actual remaining defect: Metal/C++ scalar integer promotion
was missing, so `(uint16_t(bits) << 23)` became a 16-bit shift reduced to 7 and
masked with `0xffff`. The corrected HLSL emits
`int(uint16_t(bits)) << 23`; DXIL now shifts the 32-bit value by 23 before
`asfloat`, while retaining zero native-half instructions.

The 11,346-byte GLSL has SHA-256
`fa30bdc9d3983644c94683aa2556b6f896f730f1ae32e72ff0b7082f0e1bea3b`,
uses one explicit 32-lane software subgroup for `WaveActiveMax(float)`, and
passes `glslangValidator` and `spirv-val`. Its 15,296-byte SPIR-V has three
control barriers, no group-nonuniform instruction, and local size
`[32, 1, 1]`. Because GLSL widens source binary16 values to float32, the same
source bitcast preserves the low 16-bit payload through `unpackHalf2x16` rather
than incorrectly reinterpreting the widened integer as a 32-bit float; the
inverse form uses `packHalf2x16` and exact low-bit extraction.

The reference update compares complete historical/current bodies and all seven
reflected resources across both targets. HLSL preserves eight-bit assignments,
FP4 sign bits, and source-width index arithmetic. GLSL rounds the FP4 decoding
multiply at its source half-precision boundary, adds shared-memory ordering,
and preserves the scale selection through a bitwise select. All 16 FP4 payloads
decode to exact binary fractions at that rounding boundary. Both targets remove
unused FP8 E4M3 conversion functions; the selected four-bit path retains its FP4
conversion. Resource bindings, element widths, dispatch geometry, and template
materializations are unchanged. The non-required HLSL `sign_bit` metadata
retains the same 0/8 values with explicit byte conversion. Both historical
references reproduce exactly, and all four old/new artifacts pass compilation.

A subsequent template-scope correction removes only the unused
`integral_constant_res_t_res` structure, reducing the reachable record count
from nine to eight. The prior HLSL identity
`41852207113971342d1acbf07fe4066168601bb4db19479373a2a40c36347724`
and GLSL identity
`aba7ea0ab5256e12d1ce0893c15b9522aa34c2dda075a794896e2f1bef051868`
are retained here for comparison. Complete source comparisons and strict
recompilation confirm unchanged reflected interfaces and byte-identical DXIL
and SPIR-V. The selected specialization, workload, dispatch and numerical
assertions are unchanged; native execution remains required in the existing jobs.

The current references also include Metal-compatible halfway-away-from-zero
rounding. Compared with the preceding HLSL identity
`c25bb1bb9d47cbec9d94c732caf88b8e6ae1e7501744ce87f5371e1e63f29eb7`
and GLSL identity
`dc23d056d38464ba0fa1a25ef712789e47063532dfd78a2be41433fb83218886`,
the complete change adds the integer-bit rounding helper and replaces the scale
constructor's single `round(le)` call. All other generated source, reflected
bindings, dispatch geometry and template materialization are unchanged. Strict
compilation passes for all four old/current artifacts; compiled binaries differ
because this is a rounding correction, not a formatting change. Independent
rounding regressions compare generated Metal and OpenGL with the original Metal
source and a decimal oracle, including halfway neighbors and narrow inputs.
The native-loader workload and its zero-tolerance assertions are unchanged.

The scale conversion invokes the source `fp8_e8m0(float)` constructor factory
before the selected sibling float conversion operator; aggregate field
initialization would encode the scale incorrectly. Qualified `metal::round`
lowers to a helper preserving Metal halfway-away-from-zero rounding instead of
the target's native rounding rule. OpenGL lowers `metal::isfinite` to
a single-evaluation IEEE-754 float32 exponent-mask test and `signbit` to the
exact sign bit. The private `fp8_e4m3` scalar view is admitted only as a
read-only, exact one-member-layout projection. Unresolved constructor branches,
unsupported predicate types, writes, and receiver mutation remain fail-closed.

The reflected data ABI is float32 input at binding 0, the retained but
statically unread `global_scale` input at binding 1, and float32 read-write
output at binding 2. DirectX additionally reflects generated dispatch input at
`b0`; HLSL register namespaces make this legal beside input `t0`, while
`global_scale` uses `t1` and output uses `u2`. The real no-global-scale host
branch omits buffer 1. The generic reflected-resource loader instead supplies
one inert allocation because it cannot omit a declared resource, and the test
checks that the selected specialization contains no read of it.

The numerical request contains 32 exact FP4 E2M1 values from -6 through 6.
Its maximum absolute value and scale divisor are both exactly 6, so the encoded
MX scale is exactly 1 and quantize/dequantize must return every input bit-exactly
at zero absolute and relative tolerance. Windows CI requires that request
through Direct3D 12 WARP; Linux CI requires the software-subgroup artifact
through Mesa headless EGL. Both paths build the runtime artifact manifest,
package, loader manifest, reflected ABI descriptor, and one-workgroup dispatch
request before native execution.
The existing platform jobs retain source and package identities, compiler
records, uploaded inputs, readbacks, and JUnit output, including on failure.
The project report is saved before reference assertions. No additional runner
or tolerance change is introduced. Local Mesa execution of the updated GLSL
returns all 32 outputs bit-exactly; an independent readback audit derives the
FP4 alphabet and scale from the uploaded inputs rather than the saved expected
outputs. Updated Windows execution remains required.

This is one bounded float32 MXFP4/no-global-scale workload at `846d1762`, not
the current `9c3d3557` corpus. It does not cover
the remaining `fp_quantized` entries, global-scale variants, other group sizes,
bit widths or dtypes, MLX host redirection, selected-entry Metal compilation,
or the full MLX test suite. The separate native Metal aggregate baseline remains
the round-trip boundary.

Owner-dependent `constexpr` helper calls in quantized struct static members now
resolve for the selected pinned replay, completing CrossGL/crosstl#1672.
CrossGL/crosstl#1491 tracks remaining qualified-static-constant materialization
outside the compiler-validated DirectX frontier.
Built-in overloads are resolved alongside user-defined wrappers by source
signature.
Before native validation, the harness verifies
the four numeric-to-Boolean SIMD wrapper conversions and the signed 8-, 16-,
32-, and 64-bit `arange` arithmetic conversions in the generated artifact.
Ubuntu CI installs `glslangValidator` and runs it with an OpenGL/SPIR-V 1.3
target. It first compiles the focused scalar-conversion fixtures successfully,
then compiles the translated `arange.metal` artifact. Shuffle-and-fill wrappers
lower through backend-neutral subgroup semantics and preserve their explicit
fill value for lanes below the delta.
The reduced DirectX/Vulkan frontier and the eight-source OpenGL artifact gate
include `scaled_dot_product_attention.metal`. Function-local scalar
and vector aliases now retain lexical scope and resolve across declarations,
constructors, casts, and generic static-member owners. For the pinned attention
source this resolves 531 concrete uses across all 42 entries, including the
float accumulation type used by the half and bfloat input families. The full
source translates to DirectX, OpenGL, and Vulkan; local OpenGL and Vulkan native
validation passes, and official DXC v1.9.2602.24 compiles all 42 generated
DirectX compute entries. This is compiler validation, not Direct3D runtime
execution or numerical parity.
The project Vulkan artifact is warning-free because project preparation removes
unreachable generic declarations. Direct single-file translation still emits
five such warnings under
[#1568](https://github.com/CrossGL/crosstl/issues/1568). Qualified pointer and
array aliases remain tracked in
[#1567](https://github.com/CrossGL/crosstl/issues/1567).
`arg_reduce.metal` still materializes all 24 host-named entries. The historical
aggregate run emits Vulkan but continues to fail closed for DirectX and OpenGL
with `project.translate.workgroup-size-entry-ambiguous`, because one aggregate
artifact cannot infer the runtime-selected axis and pipeline limits. That
aggregate result no longer describes all available arg-reduce coverage.

The checked
[`contracts/arg_reduce.native-loader.dispatch.json`](contracts/arg_reduce.native-loader.dispatch.json)
contract selects historical pin `846d1762`'s `argmin_float32` and `argmax_float32` for
two axis-32 rows. It applies the host formula
`roundUp(min(ceilDiv(axisSize, 4), maxThreadsPerWorkgroup), simdWidth)`, fixes a
wave width of 32, emits `[32, 1, 1]`, and dispatches `[1, 2, 1]` workgroups.
Signature-aware helper materialization selects the scalar
`elem_to_loc<int64_t>` overload rather than the `uint3` overload. The HLSL
artifacts are 6,793 and 6,855 bytes with SHA-256
`6a2667147d9a6fb8260e3cff1e5fd4c87e97d647653bf1d7bc704ab719c72c91`
and `33b85b7e9ec1d21af96bc52a157c46039a174233f28e4e44cfbc59e5dcde6b75`;
official DXC 1.9.2602.24 accepts both under `cs_6_6`,
`-enable-16bit-types`, and warnings as errors. This proof explicitly sets
`project.source_options.metal.target_options.directx.relative_wave_shuffle_out_of_range`
to `"self"`: a relative shuffle whose source lane is outside the wave retains
the calling lane's value. Generated helpers select either the valid relative
lane or the calling lane before every unconditional `WaveReadLaneAt`, so the
second reduction cannot consume undefined high-lane state. The default DirectX
policy remains `"undefined"`, preserving existing artifacts unless a project
opts in.

The explicit OpenGL software-subgroup artifacts are 8,024 and 8,030 bytes with
SHA-256
`587b409127a5cec9711856acbdac69fbc05fe25fc8004f048de15057e5a9f59f`
and `009389698c327dde691ee87a4f34d5f80c7a33384e800d8d640f97a462ae1be6`.
They admit direct shuffle-helper calls only inside a canonical workgroup-uniform
halving loop (`offset > 0` with `/= constant >= 2` or `>>= constant >= 1`),
while lane-varying, nonterminating, escaping, mutated, indirect, and nested
forms remain rejected. Both modules pass `glslangValidator` and `spirv-val`,
contain five control barriers and four shared-memory fences, and contain no group-nonuniform SPIR-V
instruction.

The reference review compares complete generated bodies, reflected resources,
entry points, and materialized functions. HLSL changes make integer promotions
and 16-bit shuffle arguments explicit and preserve the negative-infinity bits.
The compiled differences are integer no-wrap flags and a nonnegative row-index
comparison, valid for these bounded rows and offsets. OpenGL also makes shared
memory ordering explicit, bounds shuffle offsets before addition, and implements
signed remainder with truncating division. Neither review substitutes for native
execution. With `CROSTL_KEEP_CORPUS_EVIDENCE=1`, failures retain the project
configuration, translation report, generated source, package, compiler records,
dispatch inputs, and any returned readbacks under `.crosstl-corpus-evidence`.

The native ABI reflects float32 input, uint32 output, int32 shape, int64 stride,
and uint64 size resources exactly: nine DirectX bindings include the generated
`CrossGLDispatchInfo`, while OpenGL has eight bindings. Linux arm64 llvmpipe
executed deterministic rows with repeated extrema and read back argmin indices
`[5, 7]` and argmax indices `[3, 2]`, proving lowest-index tie behavior.
The retained [Windows execution results](https://github.com/CrossGL/crosstl/actions/runs/37403863131/job/112076922079)
confirm the same indices. Independent inspection verifies the uploaded rows,
lowest-index ties, generated source and executed DXIL identities. The enclosing
job later failed at the separate GEMV reference check; it is not a passing
project-wide result.
Windows CI requires the same two workloads through Direct3D 12 WARP and Linux
CI requires them through surfaceless Mesa EGL. Other axes, dtypes, and the
remaining 22 host-named entries, MLX host redirection, and the full MLX test
suite remain outside this bounded claim. Entry-scoped Metal output still fails
explicitly at workgroup specialization with
`project.translate.workgroup-size-rule-unsupported-target`; the native macOS
aggregate baseline is separate and this proof does not claim an entry-scoped
Metal round trip. Project dispatch import was completed under
[#1793](https://github.com/CrossGL/crosstl/issues/1793), and remaining broad
entry packaging stays tracked in
[#1523](https://github.com/CrossGL/crosstl/issues/1523).

The checked
[`contracts/scaled_dot_product_attention.native-loader.dispatch.json`](contracts/scaled_dot_product_attention.native-loader.dispatch.json)
contract now selects current-pinned `sdpa_vector_float_64_64` for the bounded
one-pass vector path: batch 1, one query and KV head, query length 1, key length
4, query/value dimensions 64, scale 0.125, and no mask, causal mode, or sinks.
The host-derived contract fixes workgroup `[1024, 1, 1]`, subgroup width 32,
and one dispatched workgroup. Function constant IDs 20 through 25
(`has_mask`, `query_transposed`, `do_causal`, `bool_mask`, `float_mask`, and
`has_sinks`) are all false; ID 26 belongs to the two-pass path and is
intentionally absent. The dispatch content identity is
`97e5ebb69af8da3a0082776015787456f23b8bfdb0cff757f5364db2cfef8d2c`,
with variant ID
`sha256:8b2abb9f7179e051530697fb8d1956d0ff03a324e7acaa5fcdf4f4dd9f1befbb`
and artifact ID
`sha256:dd0138695bd82e1f8ea49bd667052b484420ee96cb2849c6eed20ba5eae39a89`.

The 8,841-byte HLSL artifact has SHA-256
`4f6bd4df5288687b239d09d546c903e7b4db3d14562b942f638fb160c486a46f`.
Official DXC 1.9.2602.24 accepts `CSMain` under `cs_6_6`,
`-enable-16bit-types`, and warnings as errors, producing 9,000 bytes of DXIL.
Its 32 physical waves receive unique workgroup-synchronized subgroup IDs; no
lane-varying `SV_GroupIndex / WaveGetLaneCount()` derivation remains. The
12,387-byte explicit software-subgroup GLSL artifact has SHA-256
`04f5c58fc3c4590c77583677f8edf261c1f5c9278a4163461c239c9e820ad9e8`.
It partitions 1,024 invocations into 32 logical subgroups and synchronizes the
source subgroup-ID-strided runtime loop across every workgroup round. Inactive
subgroups contribute typed collective identities, so all generated barriers
remain uniform. `glslangValidator` and `spirv-val` accept the resulting
OpenGL/SPIR-V 1.3 module; disassembly contains exactly nine
`OpControlBarrier` instructions, six `OpSpecConstantFalse` declarations,
local size `1024 1 1`, and no `OpGroupNonUniform` instruction.

The reference update compares complete old and current shader bodies and every
reflected resource. HLSL changes are explicit source integer promotions in key
and value offsets and sign-bit negation of the finite-limit constant. GLSL changes
add shared-memory ordering at subgroup barriers and preserve signed remainder.
Dispatch geometry, specializations and resource interfaces remain unchanged.
Both old references are reproduced from the earlier translator and strictly
compiled before comparison; fingerprints alone are not numerical evidence.

Both targets package complete native-loader requests. DirectX has 19 reflected
resources including generated `CrossGLDispatchInfo`; OpenGL has 18. Optional
`bmask`, `fmask`, mask-stride, and sinks bindings receive harmless placeholders.
The stored Boolean mask ABI is physically uint32 in both HLSL and GLSL block
storage. DirectX concretizes the six false constants. OpenGL retains them for a
verified deferred GLSL-to-SPIR-V specialization request and publishes the JSON
variant registry while explicitly omitting the unsupported native registry
header.

Windows CI requires the 64-value output to execute through Direct3D 12 WARP.
Linux CI binds PyOpenGL to surfaceless EGL, forces llvmpipe, performs deferred
SPIR-V specialization, dispatches through the OpenGL native loader, and compares
all values with a stable CPU scaled-attention reference at `2e-4` absolute and
relative tolerance. The current local Mesa run has maximum absolute error
`4.082320426146424e-08` and maximum relative error
`4.2163276126605175e-06`. An independent 100-digit reference using the uploaded
binary32 inputs gives maximum absolute error `4.688126085821141e-08` across
all 64 outputs. The retained [Windows execution results](https://github.com/CrossGL/crosstl/actions/runs/37403863131/job/112076922079)
also pass. Independent calculation from the uploaded binary32 inputs gives
maximum absolute error `4.700266193284516e-08` across all 64 values, with the
generated source, compiler output and executed DXIL identities verified.
This does not establish normalization execution: those later steps were skipped
after the GEMV reference failure in the same job.
Existing native jobs retain translation reports before identity checks,
packages, compiler records, inputs, results and JUnit reports on failure.
This is bounded evidence for one float32 one-pass
workload only. Masked, causal, sinks, two-pass and full-attention paths, other
dimensions and dtypes, the remaining host-named entries, MLX host redirection,
and the full MLX test suite remain outside the claim. The separate native Metal
baseline does not make this selected native-loader proof a Metal round trip.
The historical 42-entry aggregate DirectX/OpenGL result remains fail-closed
because it does not consume this entry-scoped contract; it no longer means that
no bounded attention dispatch, package, or numerical runtime proof exists.

The reduced reference-accessor fixture covers three non-template paths. The
mutable scalar call returns the direct `val_frags[i * width + j]` lvalue and
must retain storage identity through assignment and readback. The implicit const
scalar call must lower directly into its read-only helper argument. The nested
const path matches the pinned `BlockMMA`/`Ctile` shape with a reduced outer value,
a `float2` fragment tile, a `thread const auto&` alias, and an `accum[k]` read.
For DirectX and OpenGL, the last path must contain neither `frag_at` nor `accum`
and must read `self.nestedTile.val_frags[...][k]`. This proof does not cover
template-indexed or nested forwarding overloads, full `quantized.metal`
translation, shader execution, numerical parity, or upstream MLX host/runtime
integration.
The reduced template-member pointer fixture covers the next `BaseMMAFrag::load`
boundary independently. It requires materialization of the generic
`SrcPtrType` helper from `&(src[index])`, a pointer-backed helper parameter or
equivalent OpenGL buffer-offset view, and an indexed `src[stride]` read whose
base offset still contains the outer `index`. It rejects a scalarized `float`
parameter even when no source-style call remains. The proof ends at artifact
structure and does not establish target compiler acceptance, shader execution,
or numerical parity.
`binary_two.metal` now also belongs to the required OpenGL toolchain frontier.
CrossTL commit `db593d19b` specializes fixed-array helper views to their concrete
runtime storage resources while retaining fixed extents and offsets. For the
pinned source, project translation emits zero diagnostics, the generated GLSL
compiles for OpenGL/SPIR-V 1.3, and the resulting SPIR-V passes `spirv-val`. This
resolves [#1661](https://github.com/CrossGL/crosstl/issues/1661) for the pinned
frontier. It is artifact and toolchain evidence only; it does not establish
numerical or runtime parity.
The current clean OpenGL check explicitly selects
`binary16_remainder_profile = "binary32-quotient"` for the source half `fmod`
operations in `binary_two.metal`. This is a configuration adaptation; upstream
files are unchanged. The profile rounds the quotient, product and subtraction
separately in binary32 before converting to half. Native controls retain the
source's finite cancellation, signed-zero and exceptional-value behavior rather
than substituting exact mathematical remainder. The choice is recorded in the
project report and is not a universal Metal-device assumption. Unconfigured
half calls remain diagnostic, and bfloat comparison limits under #2000 remain
separate. These operation controls do not establish complete DivMod execution
or full MLX-suite parity.
The clean OpenGL frontier supplies 24 configured index-range assertions, all with
inclusive bounds `[0, 2147483647]`. The expressions are `offset + i`, `a_idx`,
`b_idx`, `out_idx`, `out_idx++`, `idx.x`, and `idx.y` for `binary_two.metal`;
`batch_idx * offset_stride`, `freq_stride * pos.x`, `in_index_1`, `in_index_2`,
`out_index_1`, and `out_index_2` for `rope.metal`; and `offset + i`, `a_idx`,
`b_idx`, `c_idx`, `bidx`, `cidx`, `out_idx`, `out_idx++`, `idx.x`, `idx.y`, and
`idx.z` for `ternary.metal`. These records are explicit MLX host/runtime
portability preconditions for OpenGL. They are not inferred guarantees, CrossTL
does not enforce them at runtime, and they do not establish runtime integration
or numerical parity.
The eight-source OpenGL/SPIR-V gate includes `rms_norm.metal`, `rope.metal`, and
`scaled_dot_product_attention.metal`. Their Metal function constants retain
their numeric identifiers as native GLSL specialization constants; the gate
compiles each generated module for OpenGL/SPIR-V 1.3 and validates the resulting
binary. Its reduced native specialization check establishes typed dispatch and
deterministic readback only, not numerical parity for those full aggregate
modules. The bounded `sdpa_vector_float_64_64` proof above is separate and does
compare every selected output with a CPU attention reference. For the pinned
DirectX rope check, project
configuration supplies IDs 1 through 3 and CrossTL materializes a concrete HLSL
variant before DXC.

The focused `prove_copy_opengl.py` gate translates the full upstream
`copy.metal` source pinned at commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e` under one selected entry-point
scope, `s_copycomplex64float32`. The source declares 2,496 entries and expands
to 2,497 preprocessed instantiations. Exactly one selected specialization is
materialized, while the current evidence prunes 69,915 candidate pairs. No
generated wrapper fallback is used.

The generated OpenGL artifact must lower the source
`static_cast<float>(src[0])` conversion to one evaluation of `(src[0]).real`.
Linux CI compiles the resulting GLSL to OpenGL/SPIR-V 1.3 and validates the
module with `spirv-val`. This bounded lowering follows the pinned
`complex64_t` conversion body; generalized user-defined conversion operators
remain tracked by [#1744](https://github.com/CrossGL/crosstl/issues/1744). This
proof does not claim shader execution, numerical parity, runtime integration,
or passage of the MLX test suite.

The companion `prove_copy_directx.py` gate selects
`s_copycomplex64bfloat16` from the same full pinned `copy.metal` source. It
requires exactly one reachable specialization, verifies that each generated
store evaluates the complex source once and projects `.real`, and requires the
exact round-to-nearest-ties-to-even bfloat16 helper. Windows CI compiles the
standalone HLSL entry with DXC using `cs_6_2`, `-enable-16bit-types`, and
warnings as errors. This proves selected-entry translation and native compiler
acceptance; it does not execute the shader, establish numerical parity, port
the MLX runtime dispatch path, or run the MLX test suite.

A separate native-loader check selects `v_copyfloat32float32` from the current
corpus at commit `846d176227a0ac13d2667e58d2bb68b322109ab0`. It materializes
`copy_v<float, float, 1>` and the reachable `cast_to<float, float>` helper while
pruning 62,424 unreachable candidate pairs. It packages one HLSL artifact and
one GLSL artifact with exact reflected scalar buffer layouts, dispatches eight
nonuniform float32 values through Direct3D 12 WARP on Windows and a headless
OpenGL software context on Linux, and requires exact readback on both targets.
The generated artifact hashes, byte counts, binding layouts, and 1-by-1-by-1
workgroup sizes are fixed in the test. This is bounded execution evidence for
one scalar copy entry. It does not cover the complex or bfloat16 entries,
redirect the MLX host runtime, or run the MLX test suite. Aggregate input layout
for `s_copycomplex64float32` remains part of the physical-resource contract
tracked by [#1543](https://github.com/CrossGL/crosstl/issues/1543).

The independent current-pinned complete copy Metal gate covers all 2,496
discovered entries from `copy.metal` at commit
`846d176227a0ac13d2667e58d2bb68b322109ab0`. It spans 30 shapes, 16 concrete
kernel templates, all 169 input/output type pairs over the 13 source types, and
six explicit conversion families. Every entry is translated under its own
entry scope through CrossGL, and a schema-v2 hash-pinned contract records the
exact artifact identity, source/default template provenance, and reflected ABI.
The 2,496 artifacts contain 6,566 exact materializations and 8,684 reflected
resources. Each artifact owns one `cast_to` specialization; the 14
complex-to-bool artifacts additionally own the nested `cast_to<bool, float>`
specialization, so the proof does not falsely assume one fixed specialization
count per shape.

The gate preserves MLX's explicit float and bfloat16 Boolean bit-pattern tests,
including the native-width `as_type<ushort>` bfloat path. Registered
`complex64_t` representation conversions validate the exact ordered
`real: float, imag: float` shape before projecting `.real`; malformed registered
representations fail closed and unregistered lookalikes are not projected.
Scalar/vector forms reflect three resources, generalized forms reflect exact
three- through eight-resource interfaces, and every artifact retains the
host-owned `[1, 1, 1]` workgroup contract. Required Ubuntu CI verifies and exports
the family in 24 disjoint 104-entry source shards. One dependent macOS job
verifies every exported identity against the complete contract, compiles every
artifact with `xcrun -sdk macosx metal -Werror -c`, and requires 2,496 non-empty AIR outputs.
Source reports, separate shard bundles and compiler results are retained.
This is complete discovered-copy translation, reflection, and native compiler
evidence, not Metal numerical execution, MLX host-runtime redirection, or MLX
test-suite parity. The older selected OpenGL, DirectX, and native-loader checks
remain separate bounded evidence.

The current copy references include a complete, compiler-gated review of byte
conversions and typed vector initialization. Of 2,496 freshly translated
artifacts, 1,760 are unchanged; reversing only the reviewed conversions and
`int2`/`long2` initializer syntax recovers every previous source body. All
2,496 current artifacts compile with warnings fatal, preserving the 6,566
materializations and 8,684 resources above. Changed indexing, byte widths,
address arithmetic, and loop bounds are rejected by negative review controls.
Historical proof and linkage-refresh metadata remain unchanged. This refresh
does not add numerical execution or newer upstream coverage; other families
remain under [#1966](https://github.com/CrossGL/crosstl/issues/1966).

The HLSL copy reference refresh compares all 2,496 bodies with their recorded
versions and compiles every entry with warnings fatal. The 1,820 changed bodies
retain source-width byte conversions, explicit integer arithmetic and half
rounding; all 6,566 materializations and 10,036 reflected resources are checked.
Half buffers use two-byte `uint16_t` storage with explicit binary16 metadata,
rather than a floating storage type. A subsequent 28-entry correction replaces
float32-mediated wide-integer-to-bfloat conversions with direct integer rounding
([#2089](https://github.com/CrossGL/crosstl/issues/2089)). Each changed body and
binding is reviewed and compiled with warnings fatal; four unchanged conversion
controls retain their previous identities. The generated source total is
4,859,340 bytes. Generic native tests cover 64,276 signed and unsigned inputs,
including every integer midpoint and adjacent value across the representable
exponents, limits, single evaluation and explicit float32-mediated controls.
Original and generated Metal results are verified independently; required
Windows execution remains separate from compiler acceptance. Nested bfloat
constructors now retain their result types before Metal bitcast validation
([#2090](https://github.com/CrossGL/crosstl/issues/2090)). Generic native controls
cover aliases, scalar and vector payload identity, direct wide-integer rounding
and saved intermediate source. Unknown or unequal storage widths remain errors.
These reduced controls do not establish full-corpus or host-runtime parity.

The current-pin native copy gate separately tests 11 float32 layouts and the
contiguous and strided half-copy entries at
`9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8`. Both half entries transport all
65,536 binary16 payloads, including signed zero, subnormals and NaNs, in bounded
dispatches with 32 untouched output guards. OpenGL uses the exact widened
binary32 representation; DirectX and Metal retain two-byte encoded storage.
The existing native jobs run these checks without additional platform runners.
The [Windows copy step](https://github.com/CrossGL/crosstl/actions/runs/37329793271/job/111829705702)
passes all 35 checks. Independent inspection confirms 131,072 half payloads,
128 guards and matching source/DXIL identities. The job later fails an unrelated
LogSumExp reference check, so this is not a passing-job or full-suite claim.
Copy evidence uploads include hidden generated-package directories as well as
the separately retained shaders, compiler results and readbacks.
Metal also executes the unchanged upstream entries and verifies read-only
inputs. Reports retain source identities, packages, compiler results and raw
readbacks. The HLSL corpus gate checks the physical `uint16_t` half-buffer layout
and logical binary16 metadata explicitly; this does not accept unreviewed
artifact identities or claim complete copy-conversion numerical coverage.

The focused `prove_layer_norm_directx.py` gate translates two host-selected
single-row entries from the pinned `layer_norm.metal` source: forward float32
with axis size 4099 and VJP float32 with axis size 8192 and `has_w=true`. It
reads the exact host dispatch formulas from the pinned `normalization.cpp` blob
and checks that both workloads are exercised by the pinned MLX fast-operation
tests. Those inputs derive workgroup sizes `[544, 1, 1]` and `[1024, 1, 1]`.
Looped entries remain outside this proof because their workgroup sizes depend on
the selected Metal pipeline's `maxTotalThreadsPerThreadgroup()` value.

Both selected source templates declare `SIMD_SIZE = 32`, consume the Metal lane
and simdgroup index builtins, and call the shared `simd_sum` reduction path. The
DirectX project variants therefore require an exact subgroup-width rule of 32.
Each standalone HLSL artifact must retain the rule provenance, emit exactly one
`[WaveSize(32)]` beside its host-derived `numthreads` contract, and compile with
DXC using `cs_6_6` and warnings as errors. Windows CI applies that gate to both
entries. This is source identity, host-dispatch, translation, reflection, and
native compiler evidence. It does not execute either kernel, compare numerical
results, port the MLX host runtime, or claim coverage of the MLX test suite.

The checked-in
[`contracts/layer_norm.dispatch.json`](contracts/layer_norm.dispatch.json)
fixture contains exactly two pinned single-row float32 LayerNorm records from
MLX commit `4367c73b60541ddd5a266ce4644fd93d20223b6e`. It captures each
record's entry point, workgroup size, subgroup width, specialization constants,
and dispatch geometry together with the source and host-dispatch provenance.
The finite records cover the forward axis-size-4099 workload and the VJP
axis-size-8192 workload described above. Only the VJP record applies function
constant `20` (`has_w=true`), matching the pinned host dispatch; the forward
record carries no function constant. These records do not prove runtime
execution, numerical parity, looped variants, or the full MLX test suite.

The project-porting harness consumes this contract against the unchanged pinned
`layer_norm.metal` source and requires two deterministic DirectX artifacts. It
checks source and contract identities, entry-scoped execution provenance,
host-derived workgroup sizes, function-constant values, exact subgroup-width
enforcement, generated hashes, and artifact paths. Windows CI compiles both
artifacts with DXC `cs_6_6` and warnings as errors. This extends the bounded
proof into the repository-level project report; it does not add runtime
execution or numerical parity for those historical axis-size-4099 forward and
axis-size-8192 VJP records.

The reviewed forward artifact is 7,749 bytes with SHA-256
`cbe455f2066a28047f0c29b577d0805af38c038d774f19208ee13b3c90a23c54`;
the VJP artifact is 9,616 bytes with SHA-256
`0bb98d28a12fb57d0ab5c33d97ebcffc141e2ed333f76c2329957a72353ff213`.
Their complete body changes preserve the source's precise reciprocal-square-root
and square-root operations, respectively. Resource interfaces, host dispatch
contracts and source-map origins are unchanged. Both pass strict DXC compilation;
the shared helpers have independent native arithmetic checks, but those checks
do not establish execution or numerical parity for these larger workloads.

A separate historical proof selects the forward `layer_normfloat32` entry at
commit `846d176227a0ac13d2667e58d2bb68b322109ab0` through
[`contracts/layer_norm.native-loader.dispatch.json`](contracts/layer_norm.native-loader.dispatch.json).
It records the exact host formula
`32 * ceil_div(ceil_div(axis_size, 8), 32)`, the upstream
`python/tests/test_fast.py::test_layer_norm` provenance, axis size 32, two rows,
a `[32, 1, 1]` workgroup, subgroup width 32, two dispatch workgroups, and no
function constants. An explicit source/entry-qualified workgroup-access
precondition bounds specialized helper views to indices 0 through 31; it is a
stated host portability contract, not an inferred or runtime-enforced bound.
Materialization emits the forward entry plus `initialize_buffer<1>` and
`threadgroup_sum<1>` from six reachable specializations while pruning 194
candidates.

The generated HLSL artifact is 7,586 bytes with SHA-256
`33852efe1e1c48a45ec3e25e4c9d271142e2ac8aecfbd01ae4dd0c5704464a5c`.
It retains `[WaveSize(32)]`, two `WaveActiveSum` calls, and compiles as
`cs_6_6` with `-enable-16bit-types`; Windows CI requires Direct3D 12 WARP
execution. The generated GLSL artifact is 8,403 bytes with SHA-256
`8baa04a8c4125c96f1c8f0ad42f86bfc7b2497ba5a59178d199a1a3fbe3d110e`.
An earlier reference review covered the four selected normalization loaders
(LayerNorm and RMSNorm, forward and VJP) at `846d1762`: source-width index
conversions in HLSL and shared-memory ordering in GLSL preserved the resource
interfaces. All eight artifacts passed strict compiler checks at that revision.
Linux OpenGL execution and an independent 100-digit calculation checked all 256
returned values within unchanged tolerances, with maximum absolute error below
`1.36e-7`. Windows execution remains required separately.

The latest HLSL reference adds thirteen unused placeholder fields in helper
structs. Removing those declarations reproduces the previous accepted source
hash, and both complete shaders compile to byte-identical DXIL with warnings
fatal. The numerical test, resource layout and dispatch requirements are
unchanged; no other normalization reference is updated by this review.

The forward LayerNorm references now retain the source's explicit precise
reciprocal-square-root helper. Complete body review found only that helper, its
declaration and one call substitution, with unchanged resource interfaces and
source-map origins. Previous and current OpenGL readbacks are identical for the
selected 64-value workload and differ from unchanged original Metal by at most
`7.46e-9`; original Metal output guards and read-only inputs remain intact.
The same helper passes independent replay of retained Windows execution, while
the complete updated HLSL kernel still requires its own Windows CI result.
The existing CI steps retain reports, requests, compiler
records, returned values and JUnit results on success or failure; these selected
cases do not establish whole-family or upstream-suite parity.

The LayerNorm GLSL software path admits a subgroup helper only when the
helper has one stable non-overloaded source identity and every call is direct,
unconditional, top-level, and made from the sole compute entry. Conditional,
nested, indirect, ambiguous, or potentially divergent helper use rejects
translation. Resource-specialized helper lookup may reuse a prevalidated
workgroup proof only when its base key is identical and its intervals cover the
requested narrower proof; incompatible or narrower assumptions remain
rejected.

The software GLSL contains the shared-memory sum helper and six barriers, but no
KHR subgroup extension, `gl_Subgroup*` use, `subgroupAdd`, or SPIR-V
`OpGroupNonUniform` instruction. `glslangValidator` and `spirv-val` accept it;
its disassembly contains six `OpControlBarrier` instructions. The native-loader
package reflects eight ready resources on both targets: float32 input, weight,
bias, and output arrays, plus 16-byte float32 epsilon and uint32 axis-size,
weight-stride, and bias-stride blocks. DirectX uses structured/constant-buffer
layouts and OpenGL uses std430/std140 layouts.

The deterministic workload uses two 32-element rows, 32 weights, 32 biases,
epsilon `1e-5`, and unit weight/bias strides. It compares 64 outputs with
`((x - mean) / sqrt(mean((x - mean)^2) + epsilon)) * weight + bias` at `5e-5`
absolute and relative tolerance. A local Linux arm64 llvmpipe EGL dispatch
passed with maximum absolute error below `8.88e-8`; Linux CI requires the same
Mesa numerical readback. This is bounded execution evidence for one float32
forward workload only. It does not redirect MLX host execution, run the MLX
test suite, or cover VJP, float16, bfloat16, looped entries, other axis sizes,
or the historical wider dispatch records above.

A sibling pinned-corpus proof selects `vjp_layer_normfloat32` at axis size 32
through
[`contracts/layer_norm_vjp.native-loader.dispatch.json`](contracts/layer_norm_vjp.native-loader.dispatch.json).
It fixes one float32 row, `has_w=true` through function constant ID `20`, a
`[32, 1, 1]` workgroup and 32-lane subgroup, one dispatch workgroup, and the
explicit helper-access interval 0 through 95 required by three SIMD-sized
threadgroup slices. The pinned
`python/tests/test_fast.py::test_layer_norm_grad` case supplies the upstream
shape and weighted-gradient provenance. Materialization selects
`vjp_layer_norm_single_row<float, 8>`, `initialize_buffer<3>`,
`threadgroup_sum<3>`, and `threadgroup_sum<1>` from seven reachable
specializations while pruning 194 candidates.

The generated HLSL is 9,140 bytes with SHA-256
`b684a98b4ef99d01084e8d0ea4e64d3075aa4e7fd75ef8becff2608b1c46cb86`.
It concretizes `has_w=true`, retains `[WaveSize(32)]` and four
`WaveActiveSum` calls, compiles as `cs_6_6` with `-enable-16bit-types`, and
must execute numerically through Direct3D 12 WARP on Windows CI. The generated
software-subgroup GLSL is 10,245 bytes with SHA-256
`9bfbaf8bd3e6bfff172ea8461a4f75d011e9310647b33fe933773e576298fac8`.
It retains deferred OpenGL specialization constant `20`, emits eight control
barriers with no hardware-subgroup extension or SPIR-V group-nonuniform
instruction, and passes `glslangValidator` plus `spirv-val`.

Both references retain the source's explicit `metal::precise::sqrt` through the
integer square-root helper. Relative to the previous artifacts, the only kernel
change is that helper, its declaration, and one call substitution. Resource
interfaces, specialization values and source-map origins are unchanged. The
previous and current OpenGL kernels produce identical readbacks for this
workload; an unchanged original Metal kernel, specialized with `has_w=true`,
also passes the same numerical limits. This local comparison does not replace
the required complete-kernel Windows execution.

The OpenGL ABI package keeps its exact JSON runtime variant registry ready and
actionable. Its generated native C++ registry header is deliberately marked
unavailable with reason `specialization-requires-deferred-compilation`, because
the GLSL source still requires specialization. The runtime derives a bounded
deferred-compilation request, compiles GLSL to SPIR-V, applies `has_w=true`
through the OpenGL SPIR-V specialization API, then dispatches the resulting
module. This is not an unavailable or blocked runtime variant.

The eight-resource native-loader ABI contains float32 `x`, `w`, `g`, `gx`, and
per-row `gw` buffers plus float32 epsilon, uint32 axis-size, and uint32
weight-stride scalar blocks. The deterministic request compares all 32 `gx`
and 32 `gw` values at `7e-5` absolute and relative tolerance. A local Linux
arm64 llvmpipe EGL run passed with maximum absolute errors below `4.82e-8` for
`gx` and `5.44e-8` for `gw`; Linux CI requires the same deferred SPIR-V Mesa
execution.

This proof is intentionally one row: the one-row boundary makes the per-row
`gw` temporary equal to the host's final reduced weight gradient. It does not
include the separate bias-gradient reduction, multi-row weight reduction,
`has_w=false`, float16 or bfloat16, looped entries, other axis sizes, MLX host
redirection, or the full MLX test suite.

Validate the fixture schema, provenance, deterministic identities, and bounded
evaluation with:

```bash
.venv/bin/python -m pytest -q -n auto \
  demos/integrations/mlx/tests/test_dispatch_contract_fixture.py
```

The checked-in
[`contracts/logsumexp.dispatch.json`](contracts/logsumexp.dispatch.json)
fixture captures two float32 block-reduction workloads from the pinned MLX
operation and gradient tests. The records use axis sizes 32 and 1025 to exercise
distinct results of the pinned host formula
`32 * ceil_div(ceil_div(axis_size, 4), 32)`: workgroup sizes `[32, 1, 1]` and
`[288, 1, 1]`. Both dispatch one output row, require a 32-lane subgroup, and
select `block_logsumexp_float32`. The looped path for axis sizes above 4096 and
the float16 and bfloat16 entries remain outside this bounded contract.

The companion
[`contracts/logsumexp.native-loader.dispatch.json`](contracts/logsumexp.native-loader.dispatch.json)
pins the same bounded workloads to historical corpus commit
`846d176227a0ac13d2667e58d2bb68b322109ab0`. The upstream kernel, host dispatch
formula, and referenced workload coverage are unchanged between the two
revisions, so the two evaluated variant sets are identical; only revision
provenance differs. The bounded Direct3D execution, OpenGL toolchain, and
bounded OpenGL software-runtime tests use this companion contract. The
historical contract remains attached to the recorded 40-unit frontier.

The repository harness evaluates both records against the unchanged pinned
`logsumexp.metal` source and emits one standalone DirectX artifact per workload.
It verifies source, contract, dispatch, template-materialization, and artifact
identities; checks the generated max and sum wave reductions, group barriers,
exponential, logarithm, output store, workgroup size, and `[WaveSize(32)]`; and
requires official DXC `cs_6_6` compilation with warnings as errors on Windows.
This establishes translation and native compiler acceptance for the two listed
test-derived dispatches.

The refreshed HLSL references retain the exact dispatch and resource interfaces.
Complete body comparison identifies only explicit unsigned 64-bit operand
promotions and sign-bit negation of the infinity and maximum-finite constants.
Both workgroup variants compile to byte-identical DXIL before and after these
source changes with the pinned DXC release and warnings fatal. Windows native
execution remains required; matching compiled modules is not a new execution
result or a full upstream-suite claim.

A subsequent review of these two historical `846d1762` HLSL references adds
thirteen private placeholder fields in otherwise empty helper structs. Removing
only those declarations reproduces both previously accepted source hashes.
The accepted and current sources again compile to byte-identical DXIL with
warnings fatal; entry points, resources and dispatch contracts are unchanged.
This review does not update other corpus references or change the required
Windows numerical test.

Both revision-specific contracts produce the same two dispatch variants as
standalone OpenGL artifacts.
Each generated GLSL file requires `GL_KHR_shader_subgroup_basic`, declares
`CROSSTL_REQUIRED_SUBGROUP_WIDTH` as 32, and checks `gl_SubgroupSize` before any
translated kernel work. The portability report records the matching
`GL_KHR_shader_subgroup` host requirement and `GL_SUBGROUP_SIZE_KHR` query, and
the runtime artifact manifest retains both the local size and exact subgroup
width. Linux CI requires `glslangValidator` and `spirv-val` to accept both
artifacts. The built-in Python runtime and generated C++ adapter reject a
reported width other than 32 before compiling the shader or allocating its
resources.

The default OpenGL path remains the guarded hardware-subgroup path described
above. A separately selected current-corpus axis-size-32 artifact opts into
``project.source_options.metal.target_options.opengl.software_subgroup_width =
32``. Because its complete workgroup is one 32-lane subgroup, CrossTL can prove
that ``simdgroup_index_in_threadgroup`` is zero and that
``thread_index_in_simdgroup`` is ``gl_LocalInvocationIndex``. It lowers the two
scalar ``WaveActiveMax(float)`` and ``WaveActiveSum(float)`` reductions through
shared-memory trees while continuing to reject direct raw subgroup builtins,
helper-contained operations, unsupported payloads, and unproven divergent
control flow. The default hardware artifacts and host width preflight are
unchanged.

The software artifact is 4,846 bytes with SHA-256
``2aeeafbf86fe61d3ddc8b9d4a6e23945ea8a34724f0689cd8a4149f0f1a8c7fc``.
It contains ``CROSSTL_SOFTWARE_SUBGROUP_WIDTH``, no KHR subgroup extension or
``gl_Subgroup*`` use, and no hardware subgroup execution metadata.
``glslangValidator`` and ``spirv-val`` accept its OpenGL/SPIR-V 1.3 module; the
SPIR-V contains ten control-barrier instructions and no group-nonuniform
instruction. Runtime reflection packages ``in_Buffer`` and ``out_Buffer`` as
std430 float32 resources at bindings 0 and 1, plus the 16-byte std140
axis-size block at binding 2, with one ready native load unit.
The reviewed reference adds six shared-memory barriers in the sum/max helpers;
all arithmetic, bindings and existing control barriers are unchanged. Original
Metal and both GLSL revisions were executed for axis sizes 1, 31, 32, 33, 64, 127
and 128 with eight input patterns per size. All 56 results per version match the
stable reference and original Metal within the existing tolerance; old/new GLSL
readbacks match exactly and all 28 suffix guards remain intact. This retained
reference review is separate from the single-workload CI proof below and uses
the historical ``846d1762`` pin, not the current integration pin.

Linux CI requires a surfaceless Mesa EGL dispatch of that software artifact.
The workload binds the 32 values ``(index - 16) / 8``, sets axis size 32,
dispatches one 32-thread workgroup, and compares the single readback with the
stable CPU LogSumExp reference ``3.9978051379373145`` at ``5e-5`` absolute and
relative tolerance. The Windows native-loader test exercises the same bounded
workload through the guarded HLSL artifact and Direct3D 12 WARP. This is exact
translation, packaging, and numerical execution coverage for one finite
current-corpus workload; it does not redirect the MLX runtime or run the MLX
test suite.

The axis-size-1025 record still produces a ``[288, 1, 1]`` artifact containing
nine logical 32-lane subgroups, but it remains outside this LogSumExp software
runtime package. The compiler now admits structurally constrained typed masked
sum, minimum, and maximum reductions, as exercised by the current-corpus
Softmax proof below; this LogSumExp workload has not been packaged or
numerically dispatched under that mode here. Its checked artifact therefore
retains the compatible width-32 hardware contract. Looped reductions, other
dtypes and shapes, and full-runtime parity remain tracked by
[#1894](https://github.com/CrossGL/crosstl/issues/1894).

Validate the LogSumExp contract schema, provenance, deterministic identities,
and bounded workload set with:

```bash
.venv/bin/python -m pytest -q -n auto \
  demos/integrations/mlx/tests/test_logsumexp_dispatch_contract_fixture.py
```

The checked-in
[`contracts/softmax.native-loader.dispatch.json`](contracts/softmax.native-loader.dispatch.json)
fixture selects current-pinned `block_softmax_float32` and records two
host-derived block workloads. Axis size 32 with two rows uses `[32, 1, 1]` and
dispatches two workgroups. Axis size 2049 with one row uses `[544, 1, 1]` and
dispatches one workgroup; 2049 is directly represented in the pinned upstream
`python/tests/test_ops.py::test_softmax` coverage. Both follow
`32 * ceilDiv(ceilDiv(axisSize, 4), 32)`, stay within the host's block-path
limit of 4096, require logical subgroup width 32, and package three scalar
resources: float32 input and output buffers plus the int32 axis-size block.

DirectX emits one guarded `CSMain` artifact for each workgroup size, retaining
`[WaveSize(32)]`. Official DXC 1.9.2602.24 accepts both under `cs_6_6` with
`-enable-16bit-types` and warnings as errors. Their SHA-256 values are
`8dd346e61bc18a553119caa1b409487f512e520f87cad9678bcd936bfe10f4d8`
and `0d3a924e407847c1cfc0825af8bbffeed69ea4558bc19160c53ea166d5715cc8`.
Both complete bodies were compared against hash-verified originals recovered
with translator revision `22a8f69f`. Their only differences are explicit
source-width index conversions and sign-bit negation of the negative sentinels.
The pinned compiler produces byte-identical DXIL for each old/new pair;
reflected resources, materialization and launch contracts are unchanged.
The current MLX pin produces the same shaders and compiled modules, with 142
pruned candidates rather than the historical pin's 131. This is compiler and
contract evidence, not a new Windows numerical result. Required native tests
retain their original workloads and tolerances. CI preserves their project
reports, packages, compiler records, dispatch inputs and returned values on
success or failure.
The 32-thread entry is provably one wave and keeps the zero-ID quotient fast
path; the 544-thread entry allocates one uniform ID per physical wave through a
workgroup-synchronized counter. The default OpenGL path remains a separate
guarded hardware-subgroup artifact.
The runtime proof opts into explicit software subgroups instead, producing a
5,755-byte axis-32 artifact with SHA-256
`c77ddf1ad3c364b6e1898232f7d4ce99e2d5f859cc515089368044eb80667573`
and a 7,374-byte axis-2049 artifact with SHA-256
`f0db9cf9b930224322cdd4aa2a01d6c910c1fa34a479705bd7b41de9247916be`.
Complete comparison against the recovered originals accounts for exactly six
shared-memory fences in the subgroup helpers. Computation, resource bindings,
launch geometry and control barriers are unchanged. The tests require all six
fences in both generated GLSL and validated SPIR-V.
Both pass `glslangValidator` and `spirv-val`, contain exactly 11
`OpControlBarrier` instructions, and contain no `OpGroupNonUniform` operation.

The 544-thread artifact partitions the workgroup into 17 independent logical
subgroups. Its subgroup-zero final reductions execute under a lane-dependent
condition, so CrossTL evaluates the branch prefix only for active lanes and
contributes typed identities from inactive lanes before calling the uniform
barriered helper: negative infinity for float maximum and zero for float sum.
The same fail-closed lifting supports 32-bit float, int, and uint sum, minimum,
and maximum identities. Conditional shuffle and structurally ambiguous masked
collectives remain rejected.

Windows CI requires both parameterized workloads to execute through the
Direct3D 12 WARP native loader. Linux CI requires both software artifacts to
execute through surfaceless Mesa EGL; a local Linux arm64 llvmpipe run passed
both workloads. Every output is compared with a stable float32 CPU Softmax
reference at `5e-5` absolute and relative tolerance. This is bounded evidence
for two block float32 workloads only: the looped path above axis 4096, float16
and bfloat16 variants, MLX host redirection, and the full MLX test suite remain
outside the claim. Selected entry-scoped Metal generation currently fails
explicitly with `project.translate.entry-point-target-unsupported`, so this
proof does not claim the Metal round trip.

The historical reduced-frontier aggregate can still list `softmax.metal` under
workgroup-size dispatch blockers because that aggregate run does not consume
this later current-corpus entry-scoped contract. That result describes the
aggregate pipeline coordinate; it does not mean that no bounded Softmax
translation, package, or native runtime integration exists.

The current-corpus dot-product proof selects
`dot_product_float32_it32_tg512_sg16` directly from `dot.metal`. Translation
materializes the concrete `dot_product<float, 32, 512, 16>` entry with a
512-thread workgroup and an exact subgroup width of 32. The generated artifacts
preserve the read-only `float4` storage views as four bit-preserving scalar
loads, both subgroup reductions, the 16-element workgroup reduction buffer, and
the final output store. Their portability report and runtime manifest retain
the source entry, target entry, workgroup size, subgroup requirement, and all
four reflected resource layouts. For DirectX, the 16 physical waves receive
unique, wave-uniform IDs through the synchronized allocator rather than a
lane-varying flattened-index quotient.

Windows CI compiles the HLSL artifact with DXC, packages its native-loader ABI,
and dispatches one workgroup through Direct3D 12 WARP. The bounded workload
computes the dot product of 1,024 float32 values containing `1.0` and `0.25` and
requires a readback of `256.0`. Linux CI independently preserves and compiles
the guarded 5,188-byte GLSL artifact with `glslangValidator`; its SHA-256 remains
`ef69a757339fe09897a38804c27be279a19a7db146e2e02f85f0349c59f3168d`, and it
continues to reject a device subgroup width other than 32 before translated
work.

The native DirectX case is not currently a passing portability claim: the
checkpoint run returned an incorrect value, tracked in
[#2120](https://github.com/CrossGL/crosstl/issues/2120). The original numerical
assertion remains required. Windows CI retains the translation report, package,
loader descriptor, input fixture, dispatch plan, availability details, DXC
command and compiled artifacts, and returned values under `mlx-dot-directx`,
including when validation fails. These records describe the native request and
readback; they do not establish the exact bytes uploaded by the driver.

A separate software artifact is 6,275 bytes with SHA-256
`a3c1958daa680419ce3f38559de1a6a2319a7abdac556a049632194c88223a32`.
It allocates 512 shared float elements and partitions them into sixteen
independent 32-lane logical subgroups. The first sum reduces within each
partition. For the second sum beneath `tid < 16`, inactive lanes contribute the
additive identity while all 512 invocations reach the helper barriers uniformly;
only the active lanes consume the subgroup-zero result. The artifact emits no
KHR subgroup extension, hardware `gl_Subgroup*` builtin, or group-nonuniform
SPIR-V operation. `glslangValidator` and `spirv-val` accept its SPIR-V 1.3
module, whose disassembly contains four control barriers and no group-nonuniform
instruction. Mesa llvmpipe executes the same workload through surfaceless EGL
and requires the same `256.0` readback at `1e-5` absolute and relative
tolerance.

macOS CI records that the equivalent selected Metal round trip still fails
closed at storage-backed vector pointer lowering under
[#1903](https://github.com/CrossGL/crosstl/issues/1903); it does not claim native
Metal compiler acceptance. These checks establish cross-target numerical
execution for one selected kernel workload and do not redirect MLX host
dispatch, cover the float16 or bfloat16 dot entries, or run the MLX test suite.

The current-corpus unary proofs select `v_Squarefloat32float32` and
`v_ArcCosfloat32float32` from the full include-expanded `unary.metal` source.
Each entry-scoped artifact emits exactly one
`unary_v<T=float, U=float, Op=..., N=1>` specialization. Concrete call-argument
typing keeps the out-of-line complex `ArcCos::operator()` body out of the float
artifact while preserving fail-closed handling when that complex overload is
reachable. Square retains `x * x`. ArcCos retains the source
`metal::precise::acos` contract through a portable float32 range-reduction
implementation with no-contraction qualifiers instead of relying on the target
intrinsic's unspecified accuracy. OpenGL materializes each precise helper result
through a collision-safe `precise` local, avoiding non-portable qualified
function return types while preserving SPIR-V `NoContraction`. The generated
artifacts also retain a one-thread workgroup, the source and output buffers, and
the size constant in their reflected runtime interfaces. Deterministic artifact
hashes and the pinned upstream source hash are checked before packaging.

The HLSL Square and ArcCos loader references and current-pin profiled bfloat
Sigmoid reference include private placeholder fields in otherwise empty structs.
Removing only those declarations reproduces each previously accepted checksum.
Strict DXC compilation of the old and new sources produces byte-identical
modules; computations, bindings and dispatch requirements are unchanged.
Only these reviewed references are updated. The required Windows numerical
tests, including Sigmoid's 65,282 non-NaN inputs and output guards, remain
unchanged; this comparison does not establish whole-corpus execution coverage.

The selected OpenGL references include shared union storage and bit-preserving
selection in the precise ArcCos helper. Both complete bodies were compared with
hash-verified original references and passed their existing Linux/Mesa compiler
and numerical tests without changing inputs or tolerances. This bounded loader
proof is separate from the complete historical OpenGL corpus review in
[crosstl#2073](https://github.com/CrossGL/crosstl/issues/2073).

The Square and ArcCos entries now also round-trip through Metal. Square is one
1,015-byte artifact with SHA-256
``244e34b7aa58b7abe7c3ff09f3f51f3aa283a42bf7585bf88200590767032495``;
ArcCos is one 3,131-byte artifact with SHA-256
``b1aef8dc745343835414e8a0fe98a463bc4ca5b6de3212f82a6f1e8f5a9d1351``.
Entry reachability retains only the selected ``struct Square`` or ``struct
ArcCos`` and its call helpers, with no unrelated unary struct, complex ArcCos
body, or illegal ``[[static]]`` member. The ArcCos artifact retains the
portable float32 range-reduction helper with function-local
``#pragma clang fp contract(off)`` and ``reassociate(off)`` directives. Neither
helper changes contraction settings in its callers. Bounded Metal source
reflection records each exact kernel plus read-only buffer 0, read-write buffer 1,
and read-only constant buffer 2.
Their ``[1, 1, 1]`` workgroup sizes are explicitly host-dispatch-owned because
MSL has no fixed source attribute equivalent to HLSL ``numthreads``. Both exact
artifacts compile with ``xcrun -sdk macosx metal -c`` on macOS CI.

The scalar float32 ArcCos reference was reviewed after moving those directives
inside the two helpers and removing file-level contraction resets. All
computations, resource interfaces and fourteen source-map origins are unchanged.
Previous and current sources compile to identical Metal libraries and produce
identical readbacks for 8,211 inputs with eight trailing guards. The unchanged
upstream kernel passes the same inputs within the existing absolute ``1e-6`` and
relative ``1e-5`` tolerances; its finite results differ by at most two ULPs.
This review changes only ``v_ArcCosfloat32float32``. It does not accept other
unary fingerprints or establish complex ArcCos equivalence.

The scalar ArcCos Metal reference also includes an unused-parameter annotation
on the stateless operator's ``self`` argument. Removing only that annotation
reproduces the previously accepted source checksum. Both versions compile with
warnings treated as errors to identical Metal libraries, retain identical
resource interfaces, and produce identical results for the same 8,211 inputs
and eight guards. The legacy unary manifest's byte total is calculated from
its recorded entry sizes; changing that total does not accept additional
artifact fingerprints. These controls use legacy revision ``846d1762`` and do
not replace validation of the active MLX revision or the full runtime port.

The five unsigned-32-bit Sign references were reviewed separately across ``v_``,
``v2_``, ``vn_``, ``gn1_`` and ``gn4large_`` dispatch shapes. Their only historical
source changes are a precise square-root helper and its call in the retained,
unused complex overload. Removing exactly that helper change recovers each
previous complete source hash. Reflected interfaces are unchanged; old and new
sources compile to byte-identical Metal libraries with matching output names.
The current ``9c3d3557`` pin changes the included complex implementation, but
each selected unsigned entry still compiles to the same library.

Original kernels at both pins and their translated counterparts return identical
results for 3,303 values and 40 output guards per path. Cases cover unsigned
boundaries, zeros, odd tails, two-dimensional dispatch and padded three-dimensional
strides; all read-only buffers remain unchanged. Reports retain file-level
provenance, not precise operation mappings. This review updates only the five
historical unsigned Sign identities and the corresponding scalar-subset entry;
complex Sign, other dtypes and DirectX references are not covered by it.

The same five OpenGL references have a separate full-source and native review.
The square-root helper change reconstructs each recorded previous source hash;
interfaces and template specializations are unchanged. Previous sources and
translations from both MLX revisions compile to byte-identical SPIR-V modules
with glslang and pass `spirv-val`. Linux software OpenGL execution returns the
same 3,303 integer results and 40 guards per path as the unchanged Metal kernels
at both revisions. No generated shader is patched before execution. Only these
five historical OpenGL fingerprints are updated; complex Sign and other unary
changes require their own review.

The pinned comparison-arithmetic CI step also exercises unsigned Sign across all
five layouts through reflected packages, with exact integer comparisons and
trailing guards. The Metal job additionally runs the unchanged upstream entries
and checks that their read-only buffers remain unchanged. This is bounded kernel
and host-binding coverage, not completion of the upstream MLX test suite.

The required family gate covers all 877 unary entries at the legacy reference
revision `846d176227a0ac13d2667e58d2bb68b322109ab0`, including 694 non-scalar
entries in addition to the 183
scalar ``v_`` entries. The shape split is exact: 183 ``v_`` kernels instantiate
``unary_v`` with explicit ``N=1``; 183 ``v2_`` kernels instantiate ``unary_v2``
with the source default ``N=WorkPerThread<T>::n``; 145 ``vn_`` kernels use that
same default on ``unary_v``; 183 ``gn1_`` kernels instantiate ``unary_g`` with
``N=1`` and ``IdxT=int``; and 183 ``gn4large_`` kernels instantiate
``unary_g`` with ``N=4`` and the source-default ``IdxT=int64_t``. Together they
classify 37 operators, 20 concrete input/output type pairs, and 16 semantic
families.

Every selected run traverses Metal to CrossGL to Metal with zero project
diagnostics, emits one exact operator implementation and kernel, prunes
non-selected operator bodies, and retains only reachable empty helper tags.
The three vector shapes materialize one kernel specialization. Each gather
shape additionally materializes exactly one call-site ``elem_to_loc``
specialization, for 1,243 specializations across 877 artifacts. ``v_``, ``v2_``,
and ``vn_`` reflect input, output, and size resources; ``gn1_`` and
``gn4large_`` reflect those data buffers plus constant shape/stride buffers and
the read-only device ``ndim`` binding.

The generic round-trip support resolves and elides chained source typedefs,
maps native bfloat types and result reconstruction, materializes constrained
free operators with source-compatible overload selection, infers concrete
aggregate aliases, recognizes branch-complete returns, and preserves narrow
``as_type`` storage. FP8 decode admits only the proven read-only immediate
thread-local view of a scalar parameter as a matching single-field aggregate;
local storage, multiple or mismatched fields, escaped pointers, and every other
unimplemented pointer reinterpretation remain fail-closed. The non-scalar
increment additionally preserves ``constant`` pointer provenance after portable
``StructuredBuffer`` lowering, keeps ``const device`` references read-only, and
retains postfix ``out_idx++`` semantics. Retained source union layout metadata
is reconstructed as a native MSL ``union`` rather than an ignored attribute,
and additive shift operands retain explicit grouping for warning-fatal native
compilation.

Every source entry, shape, template, operator, exact ``T``/``U`` pair, semantic
family, SHA-256, byte count, template-default provenance, materialization count,
and host resource contract is pinned by
[`contracts/unary.metal-roundtrip.json`](contracts/unary.metal-roundtrip.json)
and linked by hash from ``expected-gaps.json``. Native macOS CI invokes
``xcrun -sdk macosx metal -Werror -c`` for every entry and requires 877
non-empty AIR outputs with no warning exemption. This closes selected-entry
translation, reflection, and native compiler coverage for every discovered unary
instantiation. It remains a compiler proof, not Metal numerical execution,
DirectX/OpenGL whole-family coverage, MLX host runtime redirection, or an MLX
test-suite claim.

The complete binary family gate covers all 4,122 binary entries discovered at
the historical `846d1762` pin: fifteen 238-entry base shapes and three 184-entry
work-per-thread shapes. The 18 shapes and 11 concrete kernel templates span 24
operators and 25 concrete input/output type pairs across Boolean, signed and
unsigned integer, float16, float32, bfloat16, and complex64 values. Every
selected run emits one selected operator implementation and one kernel, prunes
non-selected operator bodies, and rejects residual template, ``decltype``,
call-operator, or unsupported-placeholder syntax.

Scalar-scalar artifacts materialize ``binary_ss`` and reflect three buffers.
Scalar/vector and vector/vector artifacts, including 2-D and source-default
work-per-thread forms, reflect those buffers plus ``size``. Fixed one-, two-,
and three-dimensional generalized artifacts add exact ``elem_to_loc_1``,
``elem_to_loc_2``, or ``elem_to_loc_3`` index helpers and two stride constants.
The ``gn2`` and ``gn4large`` forms add ``elem_to_loc_2_nd`` plus shape, two
stride, and rank constants. The resulting 4,122 artifacts contain 6,026 exact
materializations and 19,106 reflected resources across exact three-, four-,
five-, and seven-resource ABIs. Explicit and source-default ``N``/``IdxT``
provenance, call-site helper provenance, and a host-owned ``[1, 1, 1]``
workgroup contract are pinned per shape.

Generic repository-scale repairs behind this family map concrete 64-bit vectors
to native ``longN``/``ulongN`` types, rebind dependent free-operator calls only
to already materialized exact helpers, and select non-explicit constructors for
known contextual aggregate conversions while rejecting explicit-only and
ambiguous cases. Bfloat ``min``/``max`` arguments promote to float before typed
result reconstruction. Discarded type-constructor expressions retain argument
evaluation through an unambiguous ``(void)(...)`` form, and scalar Boolean
relational operands receive their C++ integral promotion explicitly. The
focused regressions retain fail-closed behavior outside those proven forms.

The binary reference artifacts also retain explicit byte narrowing after
arithmetic, Boolean integral promotions, and typed vector list initialization
for index helpers. These conversions preserve the source value types and
evaluation order; they do not change the selected operators or buffer interfaces.

The historical `846d1762` bfloat reference review compares all 4,122 complete
sources against the previous contract: 4,050 are unchanged and 72 change across
`ArcTan2`, `Power`, `Remainder` and `LogAddExp`, covering all 18 shapes. The changes
make float argument conversions and bfloat result boundaries explicit.
`LogAddExp` now selects the source's bfloat `log1p` overload, narrowing its argument
before the call. Kernel bodies outside those conversions, indexing, resource interfaces
and materialization contracts are unchanged. All 72 changed sources pass
strict native Metal compilation and reproduce through the public project API.

Local Metal 3.1 checks compare unchanged upstream kernels with those exact
reviewed sources. Each operator's vector/vector entry covers all 65,536 bfloat
encodings against 12 fixed partners: 786,432 pairs per operator, or 3,145,728
returned values per path. The reviewed output matches every non-NaN word and
NaN classification. The previous `LogAddExp` output differs in 35,900 cases;
the reviewed output has no differences. An additional 216 shape controls
exercise all 72 changed entries, including strided access, broadcast inputs
and partial work-per-thread tails. They verify 4,176 values and 1,728 output
guards per path, with readonly input and constant buffers unchanged. These
checks use `-fno-fast-math`; generated sources also require `-Werror`.
They do not establish every possible operand pair, other binary operators,
another device's floating-point policy or complete host-runtime parity.

Every entry identity, shape, template, operator, exact input/output pair,
semantic family, SHA-256, byte count, materialization contract, and host ABI is
pinned by
[`contracts/binary.metal-roundtrip.json`](contracts/binary.metal-roundtrip.json)
and linked by hash from ``expected-gaps.json``. The prior
[`contracts/binary.scalar-metal-roundtrip.json`](contracts/binary.scalar-metal-roundtrip.json)
remains an exact 238-entry ``ss_`` subset. Ubuntu runs the complete translation,
body, materialization and reflection checks across 24 disjoint shards. Each
shard exports the exact pinned sources and retains its reports and JUnit results.
One dependent macOS job verifies complete entry coverage and every source hash
against the checked-in contract before invoking
``xcrun -sdk macosx metal -Werror -c``. It requires 4,122 non-empty AIR outputs;
no source-warning exemption is used. Missing, duplicate or altered bundle entries
fail before compilation. Compiler commands, diagnostics and output identities are
retained even on failure. The local round-trip test still performs both phases.
This gate requires selected-entry translation, reflection and native compilation
for every discovered binary instantiation. The bfloat review above adds bounded
numerical evidence, not whole-family numerical parity, MLX host-runtime
redirection or an MLX test-suite claim.

The annotation-only reference review at the historical `846d1762` revision
covers 3,999 entries, including 231 scalar entries. Each accepted predecessor
was reconstructed to its exact recorded source hash and byte count. The only
source change is an unused-parameter annotation; both versions compile with
warnings treated as errors and link to byte-identical Metal libraries. Entry
selection, all 6,026 materializations and all 19,106 resource declarations
remain unchanged. This is compiler-equivalence evidence, not an additional
numerical or current-pin MLX test-suite result.

The 15 complex `LogAddExp` references have a separate arithmetic review.
Their precise square-root helper preserves the explicit `metal::precise::sqrt`
call in upstream complex `log1p`; the other square root remains ordinary,
as in the source. Accepted sources reconstruct to their recorded hashes,
and both accepted and current sources compile with `-Werror -fno-fast-math`.
Original MLX, accepted output and current output agree on 12,489 complex
results per path: 291 layout controls across all 15 entries and 12,198
vector/vector pairs covering the magnitude branch boundary, finite random
inputs, subnormals, signed zeros, infinities and NaNs. Every non-NaN word
matches exactly; NaN payloads are not compared. All 512 output guard words
per path and all read-only input and constant buffers remain unchanged.
The reflected interfaces and materialization contracts are unchanged.

[`test_current_complex_logaddexp.py`](tests/kernels/test_current_complex_logaddexp.py)
repeats these native comparisons after fresh project translation at the current
`9c3d3557` integration pin. It runs in the existing macOS binary-math step,
without adding runners or extending its deadline. DirectX and OpenGL skip
this original-Metal comparison; it does not establish their numerical parity,
complete complex input coverage, MLX host redirection or full upstream-suite
execution. The historical artifact contracts remain pinned to `846d1762`.

The 54 floating `ArcTan2` entries have also been reviewed with an explicit
source underflow policy, as described below. The final 54 floating `Power`
references now have a separate numerical review. Their accepted predecessors
reconstruct to the recorded source identities, and fresh project translation
reproduces every reviewed current artifact with unchanged interfaces and
materializations. All 108 old/new sources pass strict Metal compilation and
linking. The historical pin, 4,122 entries and all non-Power identities remain
unchanged by this final reference update.

The Power review executes original MLX, accepted output and updated output
across 162 layout controls and three vector stress domains. Each path checks
171,068 values per initialization, using two different finite output fills to
reject unwritten results, and 2,640 guards in total. Every read-only buffer
remains unchanged. Non-NaN words agree exactly, including zero signs; NaN
payloads are compared by classification. The identity cases include every
binary16 and bfloat16 storage word with exponent one. This is sampled argument
coverage, not exhaustive coverage of all base/exponent pairs.

Current-pin `test_current_real_power.py` separately checks 224,534 values and
48 guards on each generated Metal/OpenGL path, with original Metal source
controls, across its identity, operand-underflow and portable-finite profiles.
The existing oracles and precision bounds remain unchanged; the finite profile
is not claimed bit-identical to native power. Both local target runs pass eight
tests, and independent audits recheck the actual output words and artifact
identities. Original Metal input readbacks are verified; generated input
readbacks are not available through these executors. Those cases already run
in the existing native host matrix on all three targets. This local review
does not establish a Windows execution pass.

All known changed binary Metal references are now reviewed. Whole-family
numerical parity, complete host redirection and full upstream-suite execution
remain unproven. Other Metal corpus work remains tracked in
[#1966](https://github.com/CrossGL/crosstl/issues/1966).

The same selected-entry pipeline translates all 4,122 binary entries to
standalone OpenGL ``main`` artifacts. The schema-v2
[`contracts/binary.opengl-translation.json`](contracts/binary.opengl-translation.json)
contract preserves all 18 shapes, 11 templates, 24 operators, 25 type pairs,
and 6,026 exact materializations while pinning
16,534,612 generated GLSL bytes and 19,106 reflected
resources. Scalar artifacts expose three storage buffers; size-bounded vector
forms add an entry-scoped uniform block. One-dimensional generalized forms add
entry-scoped stride blocks, fixed two- and three-dimensional forms expose stride
storage buffers, and rank-generic forms expose shape and two stride buffers plus
an entry-scoped rank block.

OpenGL cannot implicitly preserve the source's runtime 64-bit buffer indices.
The complete contract therefore records explicit host/runtime bounds of
``[0, 2147483647]`` for ``offset + i``, ``a_idx``, ``b_idx``, ``out_idx``,
``out_idx++``, ``idx.x``, and ``idx.y``. These bounds are declared portability
preconditions, not inferred facts or generated runtime checks; omitting the
relevant proof keeps wide-index translation fail-closed. The generated target
retains exactly the selected operator implementation and one kernel while pruning
all unrelated operator bodies.

Required Linux CI partitions the family into 24 disjoint shards. Every shard
retranslates its exact entries, verifies artifact identity, source/default and
call-site materialization provenance, ``main`` workgroup metadata, and exact
three- through seven-resource host ABI, then compiles with
``glslangValidator --target-env opengl --target-env spirv1.3 -S comp`` and
validates with ``spirv-val --target-env spv1.3``. All 4,122 SPIR-V modules must
be non-empty. Remaining changed references are still under review in
[#2073](https://github.com/CrossGL/crosstl/issues/2073); the coverage definition
does not imply that every current generated reference has passed that review.

The 18 `Remainderfloat16` entries explicitly select
`binary16_remainder_profile = "binary32-quotient"`. This preserves the observed
Metal half-remainder arithmetic, including the source's subsequent sign
adjustment and half rounding. Other binary entries do not enable that profile.
All 18 complete old/new sources and resource interfaces were reviewed, and both
versions passed GLSL compilation and module validation before updating the
references. The generated source retains the arithmetic helpers' license notice.

Native checks against unchanged upstream Metal passed at both the contract's
`846d1762` revision and the goal's `9c3d3557` revision: 1,044 values across all
18 layouts, plus every binary16 input word against 12 fixed partners in the
vector-vector kernel, for 787,476 results and 624 guards per revision.
Inputs retain signed zero and NaN payload bits; result comparisons require exact
words except for NaN payloads, which use classification. This is bounded coverage,
not every possible input pair or whole-family parity.

`tests/kernels/test_current_half_remainder.py` makes 5,176 edge, seeded and
rounding-boundary pairs repeatable through the generated package loader on
each native target. The existing macOS job also runs unchanged upstream Metal;
Windows requires DXC/WARP and Linux requires GLSL/SPIR-V validation and Mesa
execution. Eight guards and bit-preserving input encodings are required. These
checks do not establish MLX host-runtime redirection or upstream test-suite parity.

The [Windows checkpoint run](https://github.com/CrossGL/crosstl/actions/runs/37541475685/job/112535784677)
at `35f5412f` passed all 101 binary-shape checks without skips. Independent replay
of artifact `11449688692` confirms the pinned half kernel's 5,176 results and
eight guards, using the saved encoded inputs and separately rounded reference.
Its HLSL SHA-256 is `f453f410146b187e758888ef83d0822a80d3f870d843ecee6dcaa2f3b9543f9b`;
the executed DXIL SHA-256 is `1ed6f3afeed8f074b3c30079acfa88699828c3db9f35a2b15b6ee693ff568206`.
Saved host inputs are verified; physical driver-upload bytes are not claimed.

Floating remainder has a separate numerical review at `9c3d3557`. Selecting
`binary32_remainder_profile = "flush-arithmetic-subnormals"` together with
`binary32_comparison_profile = "flush-subnormals"` preserves the characterized
source operation and its sign-adjustment comparisons. The bfloat standard-library
wrapper retains its float computation and narrow return. These are explicit
source execution policies, not a universal Metal-device guarantee.

Comparison and remainder policies alone are insufficient. Although the original
bfloat sweep matched all 786,432 pairs against 12 fixed partners, six of eight
additional cancellation pairs differed during the subsequent addition. Three
of 4,472 float32 pairs exposed the same independent operation defect, tracked in
[#2113](https://github.com/CrossGL/crosstl/issues/2113).

Adding `binary32_additive_profile = "rne-flush"` gives matching original Metal,
generated Metal and OpenGL results for 791,416 vector-vector values and 208
guards. This includes the original sweeps and 512 adjacent, opposite-signed
bfloat cancellation pairs. Raw inputs, readbacks and guards are retained without
normalization. Twelve selected project artifacts compile for Metal, OpenGL and
DirectX. These controls do not establish Windows execution or whole-family parity.

The 36 float32/bfloat remainder references select all three policies across the
18 access shapes. Complete historical and updated sources pass strict GLSL
compilation and module validation, with unchanged resource interfaces. At the
historical `846d1762` pin, native replay covers 133 cases, 792,992 values and
1,064 guards without differences. Only these reviewed references are updated;
the other changed binary references remain under review in #2073.

`tests/kernels/test_current_floating_remainder.py` makes 4,475 float32 and 4,984
bfloat pairs repeatable through generated packages at `9c3d3557`. The cases
include the cancellation regressions, signed zeros, subnormals and NaNs. Each
dtype has eight exact guards; finite results and zero signs must match exactly,
while NaNs compare by classification. The existing native jobs require Metal
execution plus unchanged upstream controls, DXC/WARP on Windows, and GLSL
validation plus Mesa execution on Linux. Reports, package source identities,
encoded inputs and readbacks are retained. This does not claim physical driver
upload-byte capture, exhaustive input pairs or upstream MLX test-suite parity.

`tests/kernels/test_current_integer_remainder.py` checks seven pinned
vector-vector kernels through generated packages. Boolean and signed/unsigned
8-bit cases cover every defined operand pair; signed/unsigned 16-bit and signed
32-/64-bit cases use boundaries and deterministic samples. Together they check
168,730 results and 56 output guards per native target. Division by zero and
signed 32-/64-bit minimum divided by minus one are outside the defined domain.
The corresponding 8-/16-bit division uses promoted integer operands and remains
covered. Expected results use an independent integer model with the source's
divisor-sign remainder convention, without tolerances or readback normalization.
macOS also executes unchanged upstream Metal and checks its input buffers.
The package fixture uses reflected physical storage; this is not a claim of
whole-family parity or physical driver-upload capture.
All seven integer entries share one translation and package build, avoiding six
repeated frontend passes per target. Each kernel retains its own compiled module,
descriptor, inputs, readback and original-source comparison. Failed batches retain
the package and per-kernel evidence; batching does not relax the 900-second limit.

The 123 affected integer remainder OpenGL references have complete historical
and updated source review, with unchanged resource interfaces. All 246 versions
pass strict compilation and module validation. At `846d1762`, original Metal
and generated OpenGL agree with an independent integer model across 376 cases,
175,867 results and 3,008 guards covering every affected access shape. The
change replaces signed `%` with explicit truncating division; source sign
adjustment and narrowing remain intact.

Minimum and maximum select `binary32_comparison_profile = "flush-subnormals"`
only for float32 and bfloat operands. This matches the characterized source
comparison while retaining the selected operand's original bits. Half comparisons
retain gradual subnormals. No remainder or additive policy is enabled for these
operators. Signed-zero ties select the second operand; NaNs preserve the selected
operand's sign and payload, including signaling NaNs.

All 108 affected extrema references have complete historical and updated source
review, unchanged resource interfaces and 216 strict compiler/validator checks.
At `846d1762`, original Metal and generated OpenGL match an independent reference
across 354 cases, 820,728 results and 2,832 guards. Every affected access shape is
covered. The vector sweeps include every half and bfloat storage word against
zero in both operand positions and against a permuted word; float32 uses boundary
and deterministic samples. Comparisons require exact words without NaN exceptions
or readback normalization. Saved input encodings, output bytes and unchanged input
buffers are independently replayed. This is not exhaustive pair coverage.

`tests/kernels/test_current_extrema.py` supplies six vector-vector kernels at
`9c3d3557` and checks 28,032 results and 48 guards through generated packages per
native target. Metal and OpenGL pass locally, with unchanged upstream Metal as
an additional control. All six HLSL packages pass strict DXC compilation;
Windows execution remains a required CI check, not an inferred result. The
fixture shares the existing native runners, evidence upload and 900-second
binary-math bound. Physical driver-upload bytes and full MLX runtime parity are
not claimed.

Addition/subtraction selects `binary32_additive_profile = "rne-flush"` for
float32 and bfloat operands only. Half results retain their explicit binary16
rounding. All 108 affected references have complete historical and updated
source review, unchanged resource interfaces and 216 strict compiler/validator
checks. At `846d1762`, 354 layout/domain cases compare 826,888 results and 2,832
guards with untouched Metal and generated OpenGL. The independent audit uses
decoded arithmetic and explicit rounding, checks unchanged input buffers, and
retains raw NaN payload differences.

`tests/kernels/test_current_additive.py` covers six vector-vector kernels at
`9c3d3557`: 34,192 results and 48 guards per target, including cancellation,
rounding midpoints, subnormal boundaries and seeded inputs. Finite results,
infinities, signed zeros and guards require exact words. NaNs are compared by
classification, with differing payload counts in the report; no readback is
normalized. Metal and OpenGL pass locally, and all six HLSL packages compile
with strict DXC. Windows execution is a separate required check. Bit-observable
numeric NaN conversions remain tracked in #2081. This does not establish
bitwise NaN or whole-family parity.

The binary step schedules floating remainder types and compatible translation batches
across its existing two workers. Original Metal library builds are shared per
source root and worker, but source dispatches and their input checks remain per
case. Explicit native-test selection avoids repeating portable tests already
covered on Ubuntu. Neither the native runner count nor the 900-second deadline
increases. The 15 complex-power layouts share setup in three translation/package
batches of five. Every layout retains its own native compiler check, descriptor
and all three datasets. Exact source-entry matching prevents selecting another
artifact from the same package.

The longest binary batches are listed first and subsequent tests are assigned
one at a time with xdist's load scheduler. This avoids leaving large batches
in one worker's queue while the other worker is idle; it does not add workers.

Division selects `binary32_division_profile = "rne-flush"` for float32/bfloat
operands; half division retains gradual subnormals and binary16 rounding. All
54 affected references have complete source-body/declaration review, unchanged
resource interfaces and 108 strict historical/generated compiler-validator
pairs. At `846d1762`, 177 layout/domain cases compare 427,580 values and 1,416
guards with untouched Metal and generated OpenGL. An independent decoded-value
audit checks physical inputs, readonly buffers and readbacks without rewriting
their values. The integer `sizeof` inference correction tracked in #2115 keeps
work-per-thread constants outside the floating arithmetic profile.

`tests/kernels/test_current_division.py` supplies three vector-vector kernels at
`9c3d3557`, covering 31,232 values and 24 guards per target. Metal and OpenGL
pass locally; all three HLSL packages compile with strict DXC. Native Windows
execution remains a separate requirement. Finite values, infinities, signed
zeros and guards use exact words; NaN payload differences are recorded rather
than normalized. This does not establish bitwise NaN or full runtime parity.

Half multiplication retains gradual subnormals and explicit nearest-even
binary16 rounding, without enabling a binary32 source profile. The 18 affected
OpenGL references have complete source review and 36 strict compiler/validator
checks. The only changes are the half-rounding helper and its use at the product
return boundary; all other source bytes and resource interfaces are unchanged.
At both `846d1762` and `9c3d3557`, unchanged Metal matches an independent integer
significand reference across 61 layout/domain cases, 203,284 results and 488
guards. Generated OpenGL matches the same cases. These include all 65,536 input
words against zero in both positions and against a deterministic permutation;
they do not cover every possible operand pair. Arithmetic NaNs are compared by
classification, while finite values, infinities, zero signs and guards are exact.

`tests/kernels/test_current_multiplication.py` supplies the current-pin
vector-vector half, float32 and bfloat kernels through generated packages on
each native target. Half retains its 5,632 input pairs and gradual underflow.
Float32 and bfloat each add 12,800 boundary, exponent and deterministic pairs,
with eight guards per kernel. Their native batch explicitly selects
`binary32_multiplication_profile = "rne-flush"`, flushing subnormal operands and
exact subnormal products before rounding, while retaining zero signs. The
independent integer reference applies binary32 rounding before bfloat narrowing.
Finite values, infinities, zero signs and guards must match exactly; arithmetic
NaNs compare by classification.

The profile is recorded in project reports, packages and per-kernel evidence.
It is selected only for this float32/bfloat multiplication batch, not half,
complex expressions or the complete corpus configuration. Existing artifact
fingerprints are unchanged. All three targets require strict compilation and
native readbacks; macOS also runs unchanged upstream controls. These cases share
the existing binary-math job and its 900-second bound, without adding a runner
or relaxing comparison rules. Selected vector kernels do not establish
whole-family, composed-expression or full runtime parity.

`tests/kernels/test_current_floating_binary.py` owns native execution for these
four fixture modules. The 18 kernels share five translation/package batches:
six half kernels without a binary32 profile, four comparison kernels, four
additive kernels, two division kernels and two multiplication kernels. Compared
with separate per-family batches, this avoids three repeated frontend and package
setup passes per target.
Every kernel retains its compiler check, descriptor, dispatch, readonly input
checks, result evidence and applicable unchanged Metal control. The inventory
test checks all 124,688 values and 144 guards against the original fixtures and
verifies each batch's settings against single-entry configuration. Comparison
policy is selected per operation: selection preserves exact NaN payloads even
when sharing a batch with arithmetic operations that permit payload differences.

`tests/kernels/test_current_atan2.py` executes the current-pin vector-vector
half, bfloat and float arctangent entries in one package batch. Its 31,232 pairs
and 24 guards include exponent boundaries, signed zeros, non-finite values and
deterministic random inputs. The characterized `flush-subnormals` profile is
explicit in the configuration and retained evidence. Results use an independent
high-precision reference with the source's float/half precision bounds; bfloat
checks apply float accuracy before narrowing. Axes, zero signs, output guards
and source-control read-only inputs remain exact. Widened OpenGL narrow results
must still be representable in their original storage type.

The existing 180-second arctangent step requires this test on each native target,
without another runner. Local original/generated Metal and OpenGL checks pass.
The [Windows arctangent step at `f5bd9c68`](https://github.com/CrossGL/crosstl/actions/runs/37659104143/job/112922487897)
passes in 124 seconds under the same limit. The generic source tests also cover
127 scaled copies of the normal/subnormal rounding midpoint and adjacent
operands. This is three-entry numerical evidence, not execution of every binary
layout or the full MLX suite.

Binary reference reconciliation remains open in #2073. The precise arctangent
fix (#2118) is reconciled across all 54 OpenGL and 54 DirectX variants, including
float32. Both corpus configurations explicitly select the characterized
`flush-subnormals` profile for arctangent only. Complete source and interface
review preserves indexing, bindings, dispatch geometry and materialization.
All recorded baseline identities are verified before accepting changed output:
the older OpenGL narrow-type baselines also account for the intervening
source-width rounding helpers. Both old and new artifacts pass strict native
compilation, and an independent build reproduces all 54 current DXIL modules.
Other binary entries retain their previous identities.

Additional local execution covers all 54 reviewed OpenGL arctangent entries
across 162 broadcast, strided and tail cases: 3,132 results and 1,296 guards per
path. Generated OpenGL and unchanged current-pin Metal pass the independent
precision-specific comparisons, with every read-only buffer unchanged.
Dispatch sizing follows the source's `WorkPerThread<T>` rule, including two
float32 values or four narrow values per thread where applicable. These layout
checks complement the larger vector-vector domain checks above; they do not
claim DirectX execution of every layout or full MLX-suite parity.

The historical Metal corpus and its scalar subset now select the same
`flush-subnormals` policy for `ArcTan2` only. This is a demo configuration
choice, not a change to the translator's default policy. Before that explicit
selection, the generated helper satisfied the preserve-subnormals model but
did not match the original Metal device's underflow behavior: the vector
controls had 208 bfloat and 247 float violations of the flush-profile checks.
No precision bound or numerical oracle was relaxed to reconcile those cases.
The selected policy is required in each project report and artifact's
provenance; unrelated operators retain their existing settings.

All 54 accepted Metal sources reconstruct to their exact recorded identities.
The accepted and updated sources pass 108 strict Metal compile/link checks,
with unchanged reflected interfaces, materializations and indexing bodies.
Local original-source, accepted-output and updated-output execution covers
165 workloads: the 162 layout controls above plus the three vector/vector
domains. Each path checks 34,364 results and 1,320 guards against the existing
precision-specific oracle, with every read-only buffer unchanged and finite
sentinels rejecting unwritten outputs. The observed maximum storage errors
are one half ULP, zero bfloat ULPs and two float ULPs for the updated output;
this is precision-bounded agreement, not bit identity for every finite result.

Fresh current-pin package execution separately repeats the existing 31,232-pair
test, including unchanged original Metal controls. CI already requires that
test in its bounded arctangent step; this review adds no runners or deadlines.
The artifact contracts retain their historical `846d1762` pin. These results
do not establish complete MLX host redirection or full upstream-suite parity.

Floating-power domain failures (#2119), narrow
log-add-exp accuracy (#2068), and float32/bfloat multiplication's source subnormal
policy (#2114) remain separate numerical work. Successful compilation alone
does not justify accepting those references.

The Windows run at `6f325314` exposed signaling-NaN differences in half minimum
selection (#2116). Later retained readbacks preserve all 9,344 half minimum/maximum
results and 16 guards exactly. At `90b6f596`, the complete required binary-math
step passes in 594 seconds under the unchanged 900-second limit. That step result
does not establish complete host-runtime redirection or full upstream-suite parity.

The native arithmetic step runs device execution, original-source controls and
Metal linkage checks only. Platform-independent configuration and generation
tests remain in the complete Ubuntu suite. The native selection is checked
against every opt-in device test in its modules; the 120-second bound and all
numerical cases remain unchanged. No additional native runners are introduced.

All 60 signed and unsigned 64-bit shift references also have complete source and
interface review. Their source changes only convert shift counts to the width
required by the GLSL target; operand widths and buffer layouts are unchanged.
Fresh translations reproduce the reviewed artifacts, and both historical and
updated sources pass strict compilation and module validation.

At the historical `846d1762` pin, untouched Metal and generated OpenGL agree with
an independent integer model across all 15 layouts: 184 cases, 7,332 values and
1,472 guards per path. Vector controls cover every shift count from 0 through 63.
Signed left-shift inputs stay within representable bounds; signed right-shift
cases include negative values. Results are compared as exact 64-bit words. This
does not claim behavior for invalid shift counts or full MLX runtime parity.

The DirectX sibling contract translates all 4,122 discovered entries to
standalone ``CSMain`` artifacts and pins their exact HLSL identities in
[`contracts/binary.directx-translation.json`](contracts/binary.directx-translation.json).
It preserves the same 18 shapes, 11 templates, 24 operators, 25 type pairs,
6,026 materializations, and seven explicit index-range preconditions. DirectX
resource namespaces retain the source buffer coordinates, while nine shapes
that consume ``threads_per_grid`` add ``CrossGLDispatchInfo``. The resulting
21,248 reflected DirectX resources span exact three-, four-, five-, six-, and
eight-resource target interfaces; the rank-generic forms expose shape and
stride buffers, an entry-scoped rank constant block, and generated dispatch
metadata.

Computed-result bfloat ``ArcTan2``, ``LogAddExp``, and ``Power`` paths expand
both operands to float, compute in float, and reconstruct the result in the
exact low-16-bit bfloat register representation with round-to-nearest ties-to-
even. ``Maximum`` and ``Minimum`` expand only for comparison and return the
selected original bfloat payload without requantization. The contract remains
explicit and fail-closed: unrelated bfloat builtins are not admitted, and
storage still requires Shader Model 6.2 native 16-bit types.

Required Ubuntu CI partitions the family into 24 disjoint shards. Every shard
retranslates its exact entries, verifies deterministic identity, source/default
and call-site materialization provenance, ``CSMain`` workgroup metadata, and
exact three- through eight-resource host ABI, then compiles with checksum-pinned
Linux DXC using ``-enable-16bit-types -WX -T cs_6_2 -E CSMain``. All 4,122 DXIL
modules must be non-empty. This closes discovered binary DirectX translation,
reflection, and native compiler coverage; it does not claim numerical
execution, MLX host-runtime redirection, or MLX test-suite parity.

These compiler-only shards do not require a Windows runner. Direct3D 12
execution and readback checks remain on Windows; compiling DXIL on Linux is
not a substitute for those runtime tests.

The current HLSL references include a complete comparison of all 4,122
historical and generated bodies, all 21,248 reflected resources, and strict
compilation of both sets. The review accounts for source-width indexing,
binary math and two-byte integer storage carrying IEEE binary16 values.
Bindings, dispatch contracts, loop bounds, source pins and specialization
counts remain unchanged. Of the generated artifacts, 2,435 change and 1,687
remain byte-identical. The contract retains the previous identity and review
scope; these compiler checks do not establish numerical execution of the
complete binary family.

The complete historical reduction gate covers all 2,396 host-named entries
from `reduce.metal` at commit
`846d176227a0ac13d2667e58d2bb68b322109ab0`. Three base forms cover
initialization, all-reduce, and simple row reduction. Six multidimensional
families each combine 32- and 64-bit index types with one, two, and five
logical dimensions, yielding 39 exact ABI shapes over nine source templates.
The family spans `And`, `Or`, `Sum`, `Prod`, `Min`, and `Max`, 44 concrete
operator types, and 13 source input and output types. Its compact schema-v2
contract records only deterministic classification, generated identity and
size, exact materialization identity/count, and exact resource identity/count
per entry; full transient translation reports and native AIR objects stay out
of git.

The one- through twelve-resource interfaces contain 25,088 reflected resources
in aggregate. Each shape pins normalized Metal resource types, exact binding
coordinates, access, and a host-owned `[1, 1, 1]` workgroup contract. Generic
lowering now distinguishes comparison operators inside non-type template
arguments from closing angle brackets, folds proven integral non-type struct
arguments for specialization selection without renaming source-spelled
materializations, normalizes visible aliases and raw scalar pointer spellings,
and admits pointer-plus/minus-integral built-in arithmetic before aggregate
free-operator dispatch. Concrete-helper deduplication ignores comments and
external whitespace but preserves exact preprocessing-token literal bytes.
Compile-time static members used by references, address-taking, receivers, or
unevaluated storage-sensitive expressions are hoisted into address-space-correct
Metal `constant` storage instead of being substituted as rvalues. Unknown
aliases, unresolved address provenance, non-integral pointer offsets, and
incompatible overloads continue to fail closed.

Reduction reference artifacts retain explicit byte-value conversions, internal
helper linkage and the source shuffle offset's 16-bit width. The reference
refresh preserves every entry, operator, materialization and resource interface;
the updated sources compile with warnings treated as errors. Compilation alone
does not establish numerical parity for the full reduction family.
The remaining 1,969 source identities were reconciled after reviewing 2,096
current artifacts against their exact accepted predecessors. Every pair has
identical reflected interfaces and produces a byte-identical Metal library;
the differences are unused-parameter annotations and explicitly typed integer
boundary constants. The other 300 entries retain their previously reviewed
sources and compiler evidence. All 9,216 materializations and 25,088 resources
are preserved. This completes the historical reference review, not numerical
qualification at the current integration pin.

Required Ubuntu CI partitions the sorted family into 24 disjoint shards: 20 with
100 entries and four with 99. Each shard retranslates its exact entries,
verifies deterministic artifact identity, materialization, workgroup metadata,
and reflected ABI, then exports its checked sources. One dependent macOS job
verifies the complete bundle against the pinned contract before invoking
`xcrun -sdk macosx metal -Werror -c` for every source and requiring a non-empty
AIR object. Both phases retain evidence; missing, duplicated or altered sources
fail before compilation. This removes 23 macOS jobs; the gate still requires all
2,396 AIR outputs. The default local test still translates and compiles each
selected entry.
This proves complete discovered-reduce translation, reflection, and
native compiler acceptance; it does not claim Metal numerical execution,
OpenGL or DirectX whole-family translation, MLX host-runtime redirection, or
MLX test-suite parity.

Windows CI compiles the selected HLSL entries with DXC and executes them through
the native loader on Direct3D 12 WARP. Linux CI compiles the GLSL entries with
`glslangValidator` and executes them through the same loader contract on a
surfaceless Mesa OpenGL context. The Square workload maps
`[-3.0, -1.5, 0.0, 2.0, 4.25]` to
`[9.0, 2.25, 0.0, 4.0, 18.0625]`. The ArcCos workload maps
`[-1.0, -0.5, 0.0, 0.5, 1.0]` to the corresponding five float32 arccos values.
Both platforms enforce explicit numerical tolerances. This is numerical
evidence for two selected unary specializations. This numerical evidence
does not cover the other unary operations or dtypes, redirect the MLX host
runtime, or run the MLX test suite.

The checked-in
[`contracts/rms_norm.dispatch.json`](contracts/rms_norm.dispatch.json) fixture
captures 12 distinct dispatch artifacts exercised by the pinned
`python/tests/test_fast.py::test_rms_norm` and `test_rms_norm_grad` workloads.
The forward records cover float32 workgroups of 32, 64, and 128 threads,
float16 and bfloat16 workgroups of 32 threads, and the 1024-thread looped path.
The VJP records cover 32- and 64-thread single-row paths plus the 1024-thread
looped path, each with both concrete values of function constant `20`
(`has_w`). Axis sizes 31, 32, and 33 share the same 32-thread artifact and are
recorded as covered inputs rather than duplicate artifacts.

The repository harness evaluates those finite records against the unchanged
pinned `rms_norm.metal` source and requires 12 deterministic DirectX artifacts.
Every artifact retains its workload inputs, dispatch workgroup count,
host-derived `numthreads` value, concrete function constants, and
`[WaveSize(32)]` enforcement. Windows CI compiles every artifact with official
DXC `cs_6_6`, `-enable-16bit-types`, and warnings as errors. This is complete
translation and compiler coverage for the listed unit-test dispatch variants;
the MLX runtime is not redirected to these artifacts, the kernels are not
executed, and numerical parity is not claimed.

All twelve compiler references preserve the source's precise reciprocal-square-root
operation. Complete old/current body review isolates one shared helper, its
declaration and one call substitution per artifact. Resource interfaces, dispatch
identities, specialization constants and source origins are unchanged. Both sets
pass strict DXC compilation. The helper matches the independently checked native
arithmetic implementation; this does not extend the bounded 32-element forward
execution proof to other dtypes, gradients or larger workloads.

Validate the RMSNorm contract schema, provenance, deterministic identities, and
bounded workload set with:

```bash
.venv/bin/python -m pytest -q -n auto \
  demos/integrations/mlx/tests/test_rms_norm_dispatch_contract_fixture.py
```

The focused `prove_rms_norm_specialization.py` gate fixes the project-level
RMSNorm specialization contract to the same upstream commit and to
`rms_norm.metal` SHA-256
`5d411a2350ba7ddf84eb35f9dcac7cde0d441bd55fa1e9e1ccc61d490d428dee`.
It translates the upstream source through `crosstl.project.translate_project`.
The source check also requires all four kernel templates to retain
`constexpr int SIMD_SIZE = 32`, both simdgroup lane/group builtins, and all 12
`simd_sum` calls. This is a semantic input contract: compiling a target shader
without an exact 32-lane subgroup guarantee is not sufficient evidence for
these reductions.
The pinned MLX host computes single-row workgroup width as
`32 * ceil_div(ceil_div(axis_size, 4), 32)` and uses the selected pipeline's
`maxTotalThreadsPerThreadgroup` for looped kernels. The proof materializes
`[32, 1, 1]` and `[64, 1, 1]` as representative upstream-valid results of
those host formulas. These two sizes deliberately do not claim complete axis,
device-limit, or runtime-selected workgroup coverage.

For DirectX, two named project variants combine those workgroup sizes with the
required `has_w` function constant through both selector forms:
`has_w=false` by name at `[32, 1, 1]` and `"20"=true` by numeric ID at
`[64, 1, 1]`. The gate verifies variant selector and workgroup provenance,
concrete specialization materialization, the pinned source hash, and the
generated `static const bool has_w` value. The DirectX project configuration
sets `subgroup_width_rules["mlx/backend/metal/kernels/rms_norm.metal"] = 32`.
Each HLSL library artifact must retain the exact subgroup-rule provenance and
Shader Model 6.6 enforcement metadata for all 12 pinned host-named entries,
emit one `[WaveSize(32)]` and one matching `numthreads` attribute per entry,
and retain the reflected workgroup contract. Windows CI uses two
warning-as-error DXC runs to compile one reflected representative entry from
each HLSL library with `cs_6_6`; native 16-bit artifacts additionally pass
`-enable-16bit-types`.

For OpenGL, the `workgroup_32` and `workgroup_64` variants leave `has_w`
deferred, retain `layout(constant_id = 20)`, and split each host-named entry
into a standalone `main` artifact. This existing RMSNorm proof deliberately
does not configure an OpenGL subgroup-width rule, so subgroup provenance and
enforcement metadata remain absent. Linux CI compiles all 24 GLSL artifacts to
OpenGL SPIR-V 1.3 and validates all 24 binaries with `spirv-val`; that result
does not establish the source's 32-lane simdgroup or `simd_sum` semantics. The
bounded LogSumExp proof above exercises the exact-width OpenGL contract without
inflating the RMSNorm claim.

The 24-artifact specialization proof above remains translation and native
compilation evidence only. A separate bounded proof selects the historical
`rmsfloat32` entry at commit
`846d176227a0ac13d2667e58d2bb68b322109ab0` through
[`contracts/rms_norm.native-loader.dispatch.json`](contracts/rms_norm.native-loader.dispatch.json).
That contract fixes a `[32, 1, 1]` workgroup, subgroup width 32, two dispatch
workgroups, and no function-constant value for the forward entry. Entry-scoped
runtime reflection now omits the unreachable VJP-only `has_w` constant while
preserving reachable constants, resolving
[#1795](https://github.com/CrossGL/crosstl/issues/1795). The selected template
materializes one `rms_single_row<float, RMS_N_READS>` specialization from four
reachable specializations while pruning 168 unrelated candidates.

The generated HLSL artifact is 5,544 bytes with SHA-256
`27146f3b1c16701885dd133b628c602027152453041badf9652450cb3e386c51`;
it retains `[WaveSize(32)]` and compiles as `cs_6_6` with
`-enable-16bit-types`. The generated GLSL
artifact is 6,882 bytes with SHA-256
`04b447795637318a75158fac8025e189a6374dc8a97311d49597dc9f04f76fad`.
Its explicit target-scoped 32-lane software subgroup lowers scalar
`WaveActiveSum`, emits six `OpControlBarrier` instructions and no
`OpGroupNonUniform` instruction, and passes `glslangValidator` plus
`spirv-val`. Default OpenGL generation remains on the hardware-subgroup path.

The updated forward references replace the target intrinsic with the source's
explicit precise reciprocal-square-root helper. Complete source comparison
isolates the helper, its declaration and one call substitution; resources and
source-map origins are unchanged. The new OpenGL readback matches unchanged
original Metal for all 64 selected results, with eight source-output guards and
unchanged read-only buffers. HLSL strictly compiles and uses the same helper as
the independently replayed Windows reciprocal-root test. Full updated-kernel
execution remains required on Windows; no tolerance or dispatch setting changes.

The six-buffer native-loader ABI packages float32 input, weights, and output,
plus 16-byte scalar blocks for epsilon, axis size, and weight stride. The
numerical workload runs two deterministic float32 rows of axis size 32 with 32
weights and compares 64 outputs against
`x * w * rsqrt(mean(x * x) + epsilon)` at `3e-5` absolute and relative
tolerance. CI requires the same package and readback on Direct3D 12 WARP and
Mesa software OpenGL. This is bounded execution evidence for one forward
float32 workload; it does not redirect the MLX host runtime, run the MLX test
suite, or cover complete RMSNorm runtime parity. In particular, that forward
proof does not cover VJP, looped, float16, or bfloat16 entries, other axis
sizes, or the remaining host/device dispatch space.

A sibling current-corpus VJP proof selects `vjp_rmsfloat32` through
[`contracts/rms_norm_vjp.native-loader.dispatch.json`](contracts/rms_norm_vjp.native-loader.dispatch.json).
It fixes one float32 row of axis size 32, `has_w=true` through function constant
ID `20`, a `[32, 1, 1]` workgroup and 32-lane subgroup, one dispatch workgroup,
and the two explicit 0-through-31 index-range assertions needed by the bounded
input and cotangent views. Materialization selects
`vjp_rms_single_row<float, RMS_N_READS>` from four reachable specializations
while pruning 168 unrelated candidates.

The generated HLSL is 8,863 bytes with SHA-256
`7076ecd04029ecdf6b266eef11b3bda85f261ba40767b16a11c7314a5857f846`.
It concretizes `has_w=true`, retains `[WaveSize(32)]` and four
`WaveActiveSum` calls, and compiles as `cs_6_6` with
`-enable-16bit-types`. The generated software-subgroup GLSL is 10,257 bytes
with SHA-256
`ac05c7d026a131807c9f0d4190642c0f8d015cd58ea4db8194ab470fb7c93ab7`.
It retains deferred OpenGL specialization constant `20`, emits six control
barriers with no hardware-subgroup extension or SPIR-V group-nonuniform
instruction, and passes `glslangValidator` plus `spirv-val`.

Complete source comparison isolates the precise reciprocal-square-root helper,
its declaration and one call substitution; resources and source-map origins are
unchanged. Previous and updated OpenGL readbacks are identical for all 64
gradient values. Comparison with unchanged original Metal gives a maximum
absolute difference below `5.97e-8`, with 16 source-output guards and unchanged
read-only buffers. Strict HLSL compilation and independently replayed Windows
helper tests pass; complete updated-kernel execution remains required on
Windows. No numerical tolerance or dispatch setting changes.

RMSNorm VJP exercises a canonical runtime row loop. The explicit software
subgroup path accepts that loop only after proving its initializer and bound
workgroup-uniform from `gl_WorkGroupID` and read-only scalar blocks; lane-varying
inputs, bound mutation, unresolved calls, and `break`, `continue`, or `return`
remain fail-closed. Default OpenGL generation remains on the hardware-subgroup
path.

The ten-resource native-loader ABI contains float32 `x`, `w`, `g`, `gx`, and
one-group `gw` buffers plus float32 epsilon and uint32 axis-size, weight-stride,
row-count, and rows-per-group scalar blocks. The deterministic request compares
all 32 `gx` and 32 `gw` values at `7e-5` absolute and relative tolerance. A
local Linux arm64 llvmpipe EGL run passed deferred GLSL-to-SPIR-V execution with
maximum absolute errors below `3.54e-8` for `gx` and `3.40e-8` for `gw`; CI
requires the same package and numerical readback through Direct3D 12 WARP and
Mesa software OpenGL.

This proof is intentionally one row and one group, so the kernel's group-local
`gw` result is also the final host-reduced weight gradient. It does not cover
multi-row weight reduction, `has_w=false`, float16 or bfloat16, looped entries,
other axis sizes, MLX host redirection, or the full MLX test suite.

`fence.metal` emits no DirectX, OpenGL, or Vulkan target artifact. The harness
requires the target-specific structured diagnostics and the exact requested
atomic-fence operands under #1537 instead of accepting generated barrier text as
semantic evidence.
Future scouts should add issue-backed blockers only when there are
concrete repros. Host runtime integration gaps should be handled in repository
integration code or downstream runtime adapters, not hidden as shader
translation successes.
The full GEMV Vulkan gate materializes all 226 source specializations and emits
224 `GLCompute` entry points. The generated artifact passes both `spirv-as` and
`spirv-val` for `vulkan1.1`, with zero semantic warnings and no known codegen
fallbacks. This is structural validation only: runtime integration is not
included, and the result does not establish numerical runtime parity.

Read-only scalar storage-pointer reinterpretation now has a shared AST contract
and target lowering for DirectX, OpenGL, and Vulkan. A 32-bit scalar storage
resource can be viewed through aligned 8-, 16-, or 32-bit scalar elements;
source pointer offsets are converted to bytes before target indexing, and the
generated OpenGL and Vulkan artifacts pass native validators. Writable views,
64-bit backing layouts, and incompatible address-space or alignment cases remain
explicit diagnostics under CrossGL/crosstl#1546. Metal `dispatch_bool` callbacks
with one integral-constant parameter now lower to a runtime branch whose two
callback bodies retain distinct compile-time `true` and `false` values. Nested
dispatches expand the full Cartesian specialization and reduced DirectX,
OpenGL, and Vulkan project fixtures pass their native validators. Other callback
helpers remain tracked in CrossGL/crosstl#1554. Concrete `const_for_loop`
callbacks now expand in source order when all three bounds are integral, the
callback has reference capture and one `auto` parameter, and its body has no
callback-local control transfer. Expansion is enabled only when the source
defines the recognized `integral_constant`, `Int`, recursive loop, and arithmetic
operator contracts; unrelated helpers with the same names remain opaque. Nested
loops preserve exact
`integral_constant<int, N>` argument types; unresolved or unsafe callbacks remain
opaque and fail through the existing structured materialization path. A reduced
Vulkan fixture preserves four stores at indices `0`, `1`, `4`, and `5`, then
passes `spirv-as` and `spirv-val`. OpenGL expected-type propagation for the same
aggregate call arguments remains tracked in CrossGL/crosstl#1559.

An isolated high-budget `quantized_nax.metal` Vulkan project run now expands the
concrete NAX tile callbacks, resolves conditional function-local dimensions, and
materializes `NAXTile<T, BR, BC>` as concrete 2-by-2 specializations. Explicit
member-template binding now preserves `float16_t` threadgroup arrays, and the
template-hostile project path initializes the verified compile-time callback
contracts before member lowering. Bounded, non-variadic namespace-scope alias
templates are now canonicalized after callback and member lowering, then their
backing struct templates are materialized once more. The high-budget report no
longer contains any `Int<...>` use or `using Int` declaration and has fallen from
111 unsupported records to zero. Proven function-local integral constants now
feed inferred and explicit member-template arguments with lexical shadowing and
concrete `sizeof` handling, so `BK_padded` and `BN_padded` no longer create
symbolic helper specializations. Free helper deduction now retains unnamed
parameters, recognizes empty braced type values, and applies the same proven
lexical constants, which materializes `tile_matmad_nax` with concrete tile types
and transpose values. A verified `dispatch_bool` helper whose remaining reachable
calls are lambdas is handed to the existing callback lowering; named functors and
altered helper contracts retain ordinary materialization. Verified
`const_for_loop` callbacks now lower bare callback returns to per-iteration
escapes and fold bare integral-constant parameters only when the source defines
the verified implicit value conversion. Materialization completes with 722
specializations and no unsupported records.

Concrete struct-owned `using` and `typedef` aliases are now resolved inside
C-style and named cast targets after owner materialization. The rewrite respects
qualified owners, lexical shadowing, concrete float and integer specializations,
and aliases whose targets already contain pointer qualifiers. Metal cast nodes
retain source qualifiers while exposing a canonical target type to the strict
CrossGL function-body parser. Reduced DirectX, OpenGL, and Vulkan project
fixtures pass their native validators.

Concrete struct-owned alias templates now resolve their declaring owner,
default arguments, dependent owner constants, and alias chains before member
template deduction. Namespace-qualified and nested same-named owners remain
distinct, and generic vector locals retain their concrete type instead of
borrowing a later same-named declaration. Reduced four-component DirectX,
OpenGL, and Vulkan fixtures pass native validation. This is a partial
implementation of CrossGL/crosstl#1490; dependent function-local aliases and
value expressions outside this contract remain tracked there.

The isolated high-budget `quantized_nax.metal` run still completes 722
specializations with no unsupported records and resolves the NAX fragment
aliases to concrete eight-lane float, half, and bfloat vectors. Metal reverse
translation now represents those local values as fixed aggregate wrappers with
explicit lane storage and element-wise helpers. Reduced DirectX, OpenGL, and
Vulkan fixtures preserve lane reads, writes, arithmetic, and mutable helper
parameters; their generated artifacts pass the available native validators.
The lowering rejects unsupported operators, member selections, mixed vector
shapes, and ABI-visible device or constant storage instead of changing the
source contract. Direct generic-vector canonicalization outside the Metal
frontend remains tracked in CrossGL/crosstl#1569.

Generic member calls now retain their receiver, method, and ordered type and
value arguments in the shared AST, including pointer-member calls and nested
generic types. Metal materialization resolves concrete template methods on
direct and nested struct-field receivers before target generation. Reduced
fixtures containing the five-argument `Atile.load` and `Btile.load` forms from
`fp_quantized.metal` pass the available DirectX, OpenGL, and Vulkan validators.
Calls that reach a target without a concrete specialization fail with a
structured diagnostic instead of losing the generic suffix or computation.

At pinned MLX commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e`, exact high-budget project
replays of the complete `fp_quantized.metal` source now advance past
`epilogue_op.apply` for both DirectX and OpenGL. The receiver declaration is
`thread const TransformNone_float_float& epilogue_op`. This frontier combines
helper array-decay deduction, specialized struct constexpr assertion evaluation,
lexical receiver alias resolution, statement-bounded member-template parsing,
concrete constructor preservation, line-wrapped qualified struct receiver
materialization, and contextual Metal method receiver resolution. CrossTL commit
`c7a3c61ad` resolves the contextual receiver on this path. Specialized struct
constexpr assertion evaluation resolves CrossGL/crosstl#1807.

Dependent helper deduction now resolves the function-local `BK_padded` and
`BN_padded` expressions together with the file-scope `SIMD_SIZE` constant before
specializing plain helper templates. Proven non-type arguments are serialized to
canonical values, so equivalent Boolean and integer spellings identify the same
concrete struct at the kernel call site and in the generated helper signature.
The exact DirectX and OpenGL runs each materialize 604 function
specializations with no unsupported template records. This advances the current
frontier through the applicable CrossGL/crosstl#1479 and CrossGL/crosstl#1490
contracts; both issues retain broader project-materialization scope.

Source-scoped project configuration now supplies concrete `true` values for
`align_M` (ID 200), `align_N` (ID 201), and `align_K` (ID 202) only to
`fp_quantized.metal`. Both target records preserve `project-source-pattern`
provenance. This advances the source-scoped configuration contract in
CrossGL/crosstl#1809 and the concrete function-constant contract in
CrossGL/crosstl#1538 without applying these identifiers to unrelated sources.

Both targets also advance through equivalent duplicate definitions of
`BaseMMAFrag_float_8_8::kFragRows` and through construction of
`QuantizedBlockLoader_float_32_32_36_1_64_16_4`. These paths exercise the
qualified static-constant contract in CrossGL/crosstl#1491 and the constructor
address-space provenance contract in CrossGL/crosstl#1810. Equivalent duplicate
owners now resolve `BaseMMAFrag_float_8_8::frag_type` to its concrete
two-component float vector, including component access at `k`; this resolves
CrossGL/crosstl#1811 for the pinned frontier. Constructor factories preserve the
`BlockLoader_float_16_32_36_1_64::src_ld` const-value initialization and lower
the partially initialized `MMATile_float_2_1_BaseMMAFrag_float_8_8::val_frags`
array through ordered element writes. These results advance the broader
constructor contracts in CrossGL/crosstl#1812 and CrossGL/crosstl#1813.

The complete materialized CrossGL intermediate reaches target generation. Strict
function-body parsing accepts the generic pointer reinterpretation in
`fp_qmv_wide_impl_bfloat16_t_16_4_2_16`,
`(vec<bfloat16_t, 4>*)(xv[v] + k0)`, including the generic pointee type. This
resolves CrossGL/crosstl#1814 and removes
`project.translate.crossgl-function-body-parse-failed` from both exact project
runs.

Source whole-fragment reads and writes through `thread_elements()` references
are now canonicalized to ordered `cooperative_matrix_element` operations. The
resulting cooperative-matrix contract records the `metal_thread_elements`
layout, a 32-lane subgroup, two elements per lane, and
`metal_thread_elements_reference_view` provenance. These fields survive into
both target diagnostics instead of being inferred again after source lowering.
Reduced read and write helpers compile with the native Xcode Metal compiler.
This resolves CrossGL/crosstl#1815 and CrossGL/crosstl#1816 for the pinned
frontier.

The checked-in
[`contracts/cooperative-matrix-fragment-mapping.json`](contracts/cooperative-matrix-fragment-mapping.json)
contract records the concrete `tile_4x4_row_pair` mapping used by this pinned
MLX source. In `mlx/backend/metal/kernels/steel/gemm/mma.h`,
`BaseMMAFrag<T, 8, 8>::get_coord` defines `qid = lane / 4`,
`fm = (qid & 4) + ((lane / 2) % 4)`, and
`fn = (qid & 2) * 2 + (lane % 2) * 2`. The two lane elements therefore map to
`(fm, fn)` and `(fm, fn + 1)`. The contract contains the resulting coordinates
for all 32 lanes and records `mlx_steel_BaseMMAFrag_get_coord` provenance. This
is source-specific evidence for the pinned MLX specialization; it is not a
universal layout claim for Metal cooperative matrices.

The materialized CrossGL intermediate contains 16 source
`CooperativeMatrixType` contract nodes. Before the current contract-flow change,
two nodes carried the complete 12-field contract. In the verified replay, all 16
carry the `metal_thread_elements` layout, subgroup size 32, two elements per
lane, `metal_thread_elements_reference_view` provenance, the
`tile_4x4_row_pair` mapping, and `mlx_steel_BaseMMAFrag_get_coord` mapping
provenance.

Parsing creates eight `CooperativeMatrixOpNode` operations: seven `element`
operations and one `multiply_accumulate` operation. Each element operation now
has scalar `expression_type` `float` and intentionally has no matrix
`result_type`. The multiply-accumulate operation has complete cooperative-matrix
`result_type` and `expression_type` contracts that preserve the accumulator and
destination representation. Shared expression result inference therefore
resolves CrossGL/crosstl#1610 for this contract without claiming that scalar
element expressions require matrix result types.

DirectX and OpenGL now provide an explicit opt-in lane-local cooperative-matrix
software-lowering foundation for the exact registered 8-by-8, 32-lane,
two-elements-per-lane mapping. Reduced target tests compile and validate type
representation, element access, negation, and element-wise addition,
subtraction, and multiplication. Cooperative-matrix load, store, multiply, and
multiply-accumulate operations remain fail closed. The default behavior also
remains fail closed, and the option is not wired through project profiles or
configuration. Full software fallback, target policy, runtime execution, and
numerical parity remain unimplemented. CrossGL/crosstl#1602 and
CrossGL/crosstl#1820 remain open for that work.

Exact high-budget `fp_quantized.metal` replays enable lane-local
cooperative-matrix lowering explicitly through both code-generation factory
paths, `crosstl.project.pipeline.get_codegen` and
`crosstl._crosstl.get_codegen`; this option is not yet available through project
configuration.

Three measured intermediate replays document translation progression and are
not current-boundary claims. In the first replay, DirectX ran for 292.953
seconds and OpenGL for 286.175 seconds before both reported
`project.translate.metal-local-type-unresolved` for local type `vec_w`, whose
extent remained `tn * bytes_per_pack`. In the second replay, DirectX ran for
286.540 seconds and OpenGL for 283.718 seconds. Both reached extent
`(2) * bytes_per_pack`, proving that `tn` had resolved to `2`. In the third
replay, DirectX ran for 418.540 seconds and OpenGL for 364.117 seconds; each
materialized 606 function specializations before branch-insensitive
private-pointer analysis reported a false `view-out-of-bounds` result for
`qouter_float_2_8_4.w`.

The implementation sequence establishes reusable contracts for function-local
struct hoisting, concrete `constexpr` local extents, and defaulted zero-argument
helper materialization. These contracts are backend-independent materialization
behavior rather than MLX-specific rewrites. The completed replay demonstrates
progression through all three contracts for this source specialization.
CrossGL/crosstl#1567 remains open globally because this source-specific evidence
does not establish its complete function-local typedef scope.

DirectX and OpenGL branch pruning, together with the project translation
regression, resolve CrossGL/crosstl#1829. The completed exact replay proves
progression past the earlier false `qouter_float_2_8_4.w` range result.

In the completed replay, DirectX ran for 429.507 seconds, materialized 606
function specializations with no unsupported specializations, and reported
`project.translate.directx-workgroup-pointer-unsupported`. The missing
capability is `directx.workgroup-pointer-lowering`; function
`BlockMMA_float_float_16_32_32_1_2_false_true_36_36__mma`, parameter `As`, stops
with reason `dynamic-control-flow-reassignment` and message `DirectX cannot
preserve workgroup pointer reassignment for 'As' across nested control flow`.
CrossGL/crosstl#1518 covers the required HLSL resource and `groupshared` alias
representation, reassignment and nested-alias semantics, and structured
rejection.

OpenGL ran for 417.522 seconds, materialized the same 606 function
specializations with no unsupported specializations, and reported
`project.translate.opengl-workgroup-pointer-unsupported`. The missing capability
is `opengl.workgroup-pointer-lowering`; parameter `dst_` stops with reason
`bare-pointer-expression` and message `OpenGL cannot emit a workgroup pointer as
a first-class value: dst_`. This target boundary spans two existing contracts:
CrossGL/crosstl#1544 covers pointer-bearing aggregate members and constructors,
including `QuantizedBlockLoader`, while CrossGL/crosstl#1671 covers concrete
workgroup backing provenance through helper parameters. It is not classified as
a shared DirectX/OpenGL boundary.

Each target report contains one failed artifact/provenance record, zero
translated artifacts, and one error; no target artifact was emitted. Native
validation was not attempted because there is no artifact. MLX host runtime
integration and execution were not attempted, and numerical parity was not
evaluated. CrossGL/crosstl#1546 remains open for the broader byte-address
provenance contract across pointer reinterpretation, but it is not the current
exact boundary. Earlier replays also established progression past
`qdot_float_16_4.x_thread` and its `unprovable-view-offset` boundary under
CrossGL/crosstl#1826, as well as the transitive local `constexpr`
materialization contract tracked by CrossGL/crosstl#1824.

A shared reduced CrossGL fixture exercises the resolved partition contract with
the required readback `[100, 101, 102, 103, 200, 201, 202, 203]`.
[GitHub Actions run 29641172600](https://github.com/CrossGL/crosstl/actions/runs/29641172600)
produced this exact readback on `windows-latest` through Direct3D and on
`ubuntu-latest` through OpenGL. The passing steps were
`Prove Direct3D private-pointer partition writeback` and
`Prove OpenGL private-pointer partition writeback`, respectively.

An independent Metal fixture, `private_pointer_word_view.metal`, initializes a
local struct with two 32-bit words, reads its eight bytes through a const
thread-local byte view, and computes the order-sensitive checksum
`sum(byte[index] * (index + 1))`. Its required readback is `[204]`. The Metal
source compiles locally as Metal 3.2 with Apple metal `32023.918`.
[GitHub Actions run 29649620337](https://github.com/CrossGL/crosstl/actions/runs/29649620337)
produced the exact `[204]` readback on `windows-latest` through Direct3D and on
`ubuntu-latest` through OpenGL. The passing steps were
`Prove Direct3D local-struct byte-view native readback` and
`Prove OpenGL local-struct byte-view native readback`, respectively. Both tests
required their native runtime and reported zero mismatches with zero absolute
and relative tolerance.

These reduced fixture-level proofs do not establish complete MLX artifact
translation, full MLX host runtime integration, full MLX test-suite execution,
or numerical parity for MLX workloads.

The previously recorded pinned Vulkan replays confirmed that both affected
kernels advanced past this contract without producing a full artifact.
`fp_quantized.metal` then stopped at
type inference for the reference-returning `frag_at(i, j)` argument tracked in
CrossGL/crosstl#1557. `quantized_nax.metal` next stops because the dependent
static owner of `mma` is absent, so its empty tag argument has no selected
parameter type. Dependent static-owner materialization remains tracked in
CrossGL/crosstl#1574. These results establish translation-frontier progress
only; they do not include runtime integration or numerical parity.

The previously recorded full pinned Vulkan run advanced beyond the
generic-vector-width diagnostic. The contextual initializer contract
implemented for CrossGL/crosstl#1573 now rejects the empty
`metal::bool_constant<...>{}` argument instead of inferring a zero-length array.
The selected parameter type is still
unavailable because the captured intermediate drops the dependent static owner
from `CTile::NAXFrag_t::mma`; CrossGL/crosstl#1574 tracks that remaining
materialization contract. The intermediate also retains unresolved
reference-returning `frag_at` calls, whose receiver identity remains tracked in
CrossGL/crosstl#1557. No full-kernel artifact or validator result is claimed.
Complete address-space, const, pointer-provenance, and
unresolved-alias diagnostic transport remains tracked in CrossGL/crosstl#1566.
Pointer-bearing aggregate propagation remains tracked in CrossGL/crosstl#1544,
and lowered receiver/reference semantics must satisfy CrossGL/crosstl#1557
before the kernel can be considered semantically ready.
Lazy logical and conditional evaluation in SPIR-V remains tracked in
CrossGL/crosstl#1560 for full-corpus semantic coverage.
Nested-return lowering in pointer-preserving SPIR-V inlining is covered by the
passing full GEMV Vulkan gate. Side-effectful compatibility arguments remain
rejected explicitly and tracked in CrossGL/crosstl#1562.

## Resolved Frontier Issues

The current reduced frontier no longer depends on the previously tracked issues:
CrossGL/crosstl#1672, CrossGL/crosstl#1659, CrossGL/crosstl#1516,
CrossGL/crosstl#1476, CrossGL/crosstl#1472, CrossGL/crosstl#1312,
CrossGL/crosstl#1661, CrossGL/crosstl#1573, CrossGL/crosstl#1555,
CrossGL/crosstl#1561,
CrossGL/crosstl#1551,
CrossGL/crosstl#1498,
CrossGL/crosstl#1394,
CrossGL/crosstl#1317,
CrossGL/crosstl#939, CrossGL/crosstl#940,
CrossGL/crosstl#941, CrossGL/crosstl#943, CrossGL/crosstl#944,
CrossGL/crosstl#945, and CrossGL/crosstl#946. CrossGL/crosstl#979,
CrossGL/crosstl#980,
CrossGL/crosstl#981, CrossGL/crosstl#982, CrossGL/crosstl#983,
CrossGL/crosstl#984, CrossGL/crosstl#985, CrossGL/crosstl#1001,
CrossGL/crosstl#1002, CrossGL/crosstl#1003, CrossGL/crosstl#1004,
CrossGL/crosstl#1006, CrossGL/crosstl#1007, CrossGL/crosstl#1012, and
CrossGL/crosstl#1013 are also covered by mainline fixes or superseded by the
current follow-up issue set. CrossGL/crosstl#1019, CrossGL/crosstl#1026,
CrossGL/crosstl#1028, CrossGL/crosstl#1029, CrossGL/crosstl#1030,
CrossGL/crosstl#1031, CrossGL/crosstl#1033, CrossGL/crosstl#1034,
CrossGL/crosstl#1035, and CrossGL/crosstl#1036 are closed by mainline fixes or
superseded by the current issue set and are no longer listed as active MLX
blockers. CrossGL/crosstl#1032, CrossGL/crosstl#1037, CrossGL/crosstl#1038,
CrossGL/crosstl#1039, CrossGL/crosstl#1068, CrossGL/crosstl#1104, and
CrossGL/crosstl#1105 are also closed or superseded by the current scout and
issue set. CrossGL/crosstl#1027 is no longer reported by the latest full-corpus
scout because the generated Metal quantization declarator now parses far enough
to reach target codegen. The current full-corpus scout no longer reports
runtime-adapter contracts, boolean SPIR-V interface lowering, or the previous
closed issue set as active missing capabilities. CrossGL/crosstl#1106,
CrossGL/crosstl#1107, CrossGL/crosstl#1110, CrossGL/crosstl#1111,
CrossGL/crosstl#1122, CrossGL/crosstl#1124, CrossGL/crosstl#1126, and
CrossGL/crosstl#1127 are also closed and are no longer tracked as active MLX
blockers. CrossGL/crosstl#852 is covered by the current OpenGL arange smoke
check. CrossGL/crosstl#1146 is resolved by bounded template replacement scans,
and CrossGL/crosstl#1184 is resolved by the latest mainline materialization
work. CrossGL/crosstl#1155 and CrossGL/crosstl#1160 are covered by the current
frontier after the SPIR-V project-artifact and multi-entry binding fixes.
CrossGL/crosstl#1203, CrossGL/crosstl#1204, and CrossGL/crosstl#1206 were
closed by the latest mainline helper-template, softmax parser, and SPIR-V
pointer-overload fixes. CrossGL/crosstl#1205, CrossGL/crosstl#1207,
CrossGL/crosstl#1218, and CrossGL/crosstl#1222 are also closed by the current
mainline OpenGL template, SIMD helper, steel attention diagnostic, and steel GEMM
materialization fixes. CrossGL/crosstl#1238, CrossGL/crosstl#1239, and
CrossGL/crosstl#1240 are closed by the assembled SPIR-V validation, complex
helper call, and fence initializer fixes. CrossGL/crosstl#1246,
CrossGL/crosstl#1248, CrossGL/crosstl#1249, CrossGL/crosstl#1250,
CrossGL/crosstl#1259, CrossGL/crosstl#1260, and CrossGL/crosstl#1261 are closed
by the current mainline access-chain index, materialization scalability,
templated functor, and Vulkan validation fixes. CrossGL/crosstl#1274 and
CrossGL/crosstl#1287 are closed by the current Vulkan complex helper validation
and full-corpus Metal template materialization fixes. CrossGL/crosstl#1329,
CrossGL/crosstl#1338, CrossGL/crosstl#1340, and CrossGL/crosstl#1346 are closed
by the current project-scale template and SPIR-V validation fixes.
CrossGL/crosstl#1355 is closed by the current OpenGL MLX template binding fix.
CrossGL/crosstl#1354 and CrossGL/crosstl#1362 are closed by the current
full-corpus materialization and Vulkan validation work. CrossGL/crosstl#1452,
CrossGL/crosstl#1453, and CrossGL/crosstl#1454 are covered by bounded template
materialization with source-located diagnostics for unsupported MLX reduction,
scan, and Steel specializations. CrossGL/crosstl#1392 is closed by fixture
resource binding through reflected backend aliases. CrossGL/crosstl#1500 is
covered by mapped-signature collision detection with overload-aware GLSL call
rewriting. CrossGL/crosstl#1502 is covered by contextual GLSL aggregate
construction for struct, fixed-array, vector, and matrix values.
CrossGL/crosstl#1503 is covered by explicit expected-type scalar coercion for
numeric-to-Boolean returns and signed mixed-width `arange` arithmetic.
CrossGL/crosstl#1661 is covered for pinned `binary_two.metal` by fixed-array
resource helper specialization in CrossTL commit `db593d19b` and the required
OpenGL/SPIR-V 1.3 compilation and validation gate.
CrossGL/crosstl#1807 is resolved for the pinned `fp_quantized.metal` frontier by
specialized struct constexpr assertion evaluation; contextual receiver
materialization remains tracked in CrossGL/crosstl#1479.
CrossGL/crosstl#1811 is resolved for the same frontier by equivalent duplicate
struct-alias resolution with concrete component typing and fail-closed conflict
diagnostics.
