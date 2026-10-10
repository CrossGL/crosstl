# CI Coverage

The CI suite separates portable Python compatibility from native GPU toolchain
and runtime checks.

Backend, translator, complete-suite, example and demo workflows replace a
superseded pull-request run when a new revision arrives. The concurrency key
includes the workflow name, event and PR number, so unrelated workflows and PRs
do not cancel each other. Main, scheduled and manually dispatched runs use their
unique run IDs and are not cancelled by this policy. Workflow-level keys are
distinct from native job-level queues. Project demo jobs retain their independent
queues, but a newer PR revision cancels the superseded run, including its active
native jobs, so stale revisions do not occupy Windows and macOS runners.
Cancelling an obsolete run is not a test pass; the replacement revision must
satisfy the same checks. This follows
[GitHub's concurrency policy](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).

| Workload | Ubuntu | Windows | macOS |
| --- | --- | --- | --- |
| Backend, translator and example tests | Python 3.8 through 3.13 | Python 3.13 | Python 3.13 |
| Complete core and demo unit suite | Required | Covered by focused platform jobs | Covered by focused platform jobs |
| Complete DirectX demo corpus compilation | Pinned Linux DXC | Native execution remains below | Not required |
| Direct3D execution and host integration | Not applicable | Required WARP and dispatch checks | Not applicable |
| OpenGL execution and host integration | Required Mesa/EGL checks | Platform-specific controls | Platform-specific controls |
| Metal compilation, reference and host integration | Not available | Not available | Required native checks |

The existing Windows project job also checks DXC inputs and outputs longer
than 260 characters, nested relative includes and retained error diagnostics.
Long-path commands run the installed DXC library in a separate process, using
an include handler that normalizes parent-relative paths before opening them.
Short-path commands continue to use the DXC executable. Generated artifacts
stay in their original locations; reports retain both their logical paths and
the actual compiler invocation. The bridge requires `dxcompiler.dll` and its
dependencies beside the selected `dxc.exe`, as provided by the DXC release.

The existing project-host arctangent step requires the generic two-argument
precision checks and three pinned binary entries on each native target. The
generic check retains 5,271 input pairs, including all 127 applicable scalings
of the normal/subnormal midpoint, adjacent operands and both signs. Each profile
checks 100,149 scalar/vector/broadcast angles, unchanged input/evaluation words
and eight guards. The pinned package batch adds 31,232 half, bfloat and float
results plus 24 guards, with original-source execution on Metal. No runner,
worker or deadline is added: the existing two-worker, 180-second step owns this
coverage. Native Windows execution cannot be replaced by Linux DXC compilation.

The existing 120-second arithmetic step also checks Metal power-function
domains on all three native targets, with an unchanged Metal source control.
Signed zeros, negative bases, integral-exponent parity, infinities, NaNs,
single evaluation, vector broadcasts and half narrowing are covered. Domain
classifications and guards are exact; finite controls use an independent decimal
reference. This does not establish full finite-domain accuracy or a common
subnormal policy. No additional runner or worker is required.

Every backend and code generator remains covered on all three operating
systems. Python-version compatibility is exercised fully on Ubuntu; repeating
those versions on Windows and macOS is not required. Windows and macOS each
run the complete backend directory in one job and the complete translator
directory in another, with two pytest workers per job. Ubuntu keeps its
component-level matrix. The version policy removes 198 duplicate jobs, and
grouping native-platform unit suites removes another 38 runner setups per
revision without removing a test suite. Failure reports retain individual
test names in JUnit even when suites share a runner.

The macOS translator and portable-host jobs use the `xcode-27` image and
explicitly select Xcode 27.1. Acquire/release atomic fences require Metal 4.1;
the default Xcode 26 toolchain on `macos-latest` cannot compile that language
mode. Both jobs check the selected compiler before running tests. This replaces
two existing runners, not additional matrix jobs, and does not change the
source fence order, numerical assertions or execution requirements. Other
macOS jobs retain their existing images and language requirements.

The open-source porting demo runs its portable tests and all-target reference
comparison once on Ubuntu, using two pytest workers. Windows and macOS retain
every target-specific regeneration, comparison and compiler check, without
repeating the other platforms' artifact checks. macOS uses Xcode's Metal compiler
and does not install the OpenGL and SPIR-V validators used by Ubuntu. The three
native jobs and their failure reports remain intact.

The demo's 77 unary, binary, copy and reduction DirectX compilation shards run
on Ubuntu. They retain every entry, strict compiler flags, artifact identity
checks and the same DXC release. Pinned installation verifies the Linux archive
checksum and does not fall back to another release. These jobs do not execute
Direct3D; WARP numerical execution and Windows host integration remain required.

If the required Windows collective/resource gate fails, its existing runner
also compares unchanged precise scalar and matrix workloads against the pinned
Agility SDK 1.619.6 (SDK 619). Archive and core checksums are verified before use;
the isolated dispatch process must report the selected core's digest. WARP
1.0.21 remains installed. The comparison retains a 180-second command deadline,
adds no runner, and cannot replace or excuse the required gate's failure.
Requested and observed runtime identities, compiled modules, inputs, expected
values and readbacks are retained in the portable-host evidence artifact.
Metal corpus compilation stays on macOS because it requires Apple's toolchain.

The complete unary, binary, copy, reduction and quantized Metal corpora separate portable translation
from native compilation. Unary uses five source-generation shards on Ubuntu;
each other family uses 24 shards,
preserving every body, materialization and resource-interface assertion. One
dependent macOS job per family verifies all artifact identities against the
checked-in contract and compiles every source with warnings fatal: 877 unary, 4,122 binary,
2,496 copy, 2,396 reduction and 2,052 quantized artifacts. Quantized compilation
retains the Metal 3.1 language standard and empty-compiler-stream requirement.
Shard downloads remain separate so duplicate
entries cannot overwrite each other. Missing, duplicated or changed sources fail
before any compilation; compiler failures retain diagnostics and output identities.
This removes 96 macOS jobs without sampling any corpus. Metal numerical and host
integration checks still run natively and are not replaced by this compiler job.
The quantized consumer also executes all 108 affine quantization and
dequantization variants from the same verified source bundle. Five groups per
variant cover both scale signs, dyadic values and zero ranges. Unchanged upstream
Metal and generated Metal must both match independently calculated packed bytes,
scales and biases; readonly inputs and output guards are checked exactly. The
15-minute native step retains commands, modules and readbacks in the existing
job artifact. It adds no runner and does not repeat translation on macOS.
The unary, copy, reduction and quantized source phases permit only the explicit Metal toolchain-unavailable
warning when that toolchain is absent, retain it in the report and reject all
other diagnostics. Their dependent macOS phases still require native compilation.
The unary source shards also retain their configuration, generated shaders,
reports and JUnit results when a reference check fails. The native consumer
cannot run until all source shards succeed; stale references remain failures.

The four DirectX corpus families retain each case's configuration, generated
shader and portability report before checking the recorded artifact identity.
When compilation is reached, its command, output and exit status are retained
alongside the compiled module. Every shard uploads this evidence and its JUnit
results on success or failure. A missing compiler record means compilation was
not reached; a generated shader alone is not proof that compilation passed.
Local runs clean up by default; set `CROSTL_KEEP_CORPUS_EVIDENCE=1` to retain the
same files under the upstream checkout's `.crosstl-corpus-evidence` directory.

The four OpenGL corpus families retain the same evidence in their existing
Ubuntu shards. Compiler and SPIR-V validator commands have separate records,
so a validation failure cannot overwrite a successful compilation record.
Reports are saved before reference checks, and missing or empty compiled modules
still fail. No runner, corpus sample or extra translation pass is added.

The existing Windows and Linux Softmax, arg-reduce, attention, GEMV, MXFP4 and
normalization loader steps retain their reports before artifact checks,
generated packages, native compiler commands and artifacts, input fixtures,
execution plans and returned values. The project
job's final upload includes these files and each step's JUnit results even
when a check fails. Missing native results indicate that dispatch was not
completed; retained expected values alone are not execution evidence.

FFT execution uses the same retained workspace mechanism. Each of its five
controls records inputs, expected outputs, guards and returned values separately;
DirectX also retains each dispatch plan and compiler invocation. OpenGL keeps
the deferred request and compiled cache, including the first publication and
subsequent cache hits. The existing native jobs upload these records immediately
after the FFT steps, including on failure, without adding runners or weakening
the numerical checks.

`demo-project-testing.yml` is the entry point for project integration checks.
Each native job keeps its source pin, numerical comparisons, guards, bounded execution
and retained evidence. Moving a compiler-only job must not disable an execution
gate or be presented as proof of runtime parity.

The existing comparison step also checks binary16 operand selection through
runtime packages. Two dispatch batches cover every binary16 word in both operand
positions, plus signed-zero, infinity and NaN boundary pairs. Scalar comparisons,
NaN early returns and two-, three- and four-component selections must preserve
the selected storage words exactly. Original Metal controls and output guards
remain required; no NaN normalization is permitted. The step keeps its two
workers and 180-second limit, with no additional runner.
Compiler regressions also check effectful half-valued conditional branches,
including nested scalar/vector results widened to float or double. Conversion
must happen once after selection without eagerly evaluating either branch.

The native host's binary32 arithmetic step selects device execution, original
source controls and Metal module-linkage tests. Configuration, reference-model
and generation-only checks remain in the complete Ubuntu suite. A workflow
regression test verifies that the selection includes every opt-in device test
in the arithmetic modules. All numerical inputs, profile variants and
guards remain required under the same 120-second bound, without new runners.

Multiplication profiles run in that same step. Exact integer references cover
subnormal inputs and outputs, normal-range rounding, signed zeros, overflow,
NaNs, literal zero products, scalar/vector forms and compound assignments.
Original Metal controls, input copies, output guards and single-evaluation
counters distinguish arithmetic policies from storage changes. These tests do
not infer a contraction policy for surrounding additions or subtractions.

The half-remainder profile runs in the existing arithmetic step on each native
target, with no additional runner. It checks 5,176 input pairs in scalar/vector
forms, exact guards and single-evaluation counters. The macOS job also executes
the unchanged Metal source. Source, compiled modules, inputs, expected values,
readbacks and device details use the existing arithmetic evidence upload.
The profile is opt-in and does not replace the unsupported-profile diagnostic
for unconfigured half remainder or imply full corpus parity.

The existing pinned binary-math steps also translate and execute MLX's actual
half-remainder vector kernel through the generated package loader. All three
native targets check 5,176 pairs and eight guards; macOS additionally executes
the untouched upstream kernel. Inputs retain their IEEE bit encodings and
comparisons permit only NaN payload differences. The cases share the existing
900-second bounds, runners and failure-artifact uploads. This supplements the
generic arithmetic controls; it does not replace them or claim whole-family parity.

The same pinned step includes Boolean and integer remainder packages. Seven
kernels cover all defined 8-bit operand pairs and boundary/seeded wider inputs,
checking 168,730 exact results and 56 guards per target. Zero divisors and signed
32-/64-bit quotient overflow are excluded explicitly. macOS runs unchanged
upstream controls; Windows requires DXC/WARP and Linux requires Mesa execution.

Floating binary cases include half, float32 and bfloat multiplication. The latter
two use an explicitly profiled batch with 25,600 results and 16 guards, independent
integer references and retained package provenance. Half keeps its gradual
underflow and existing cases. All 18 floating kernels execute in five compatible
batches under the same binary-math deadline, with no additional runner. The
profile does not change complete-corpus configurations or accepted fingerprints.
The existing runners, 900-second bound and failure-artifact upload are reused.
The seven integer entries share one translation and package build. Each entry
still has a separate compiler invocation, reflected descriptor, native dispatch
and readback. The entire package and completed case evidence are retained if a
later case fails.

Pinned minimum/maximum packages run in that same step: six half, bfloat and
float32 kernels check 28,032 exact selected-operand words and 48 guards per
target. Signed zeros, subnormals, infinities and signaling/quiet NaN payloads
are retained. Only float32/bfloat comparisons enable the characterized
flush-subnormals policy; half comparisons remain gradual. macOS also executes
unchanged source controls. There are no NaN comparison exceptions, new native
jobs or relaxed deadlines.

Pinned addition/subtraction adds six half, bfloat and float32 packages to the
same step: 34,192 results and 48 guards per target. Finite values, infinities,
signed zeros and guards require exact words; NaN payload differences are counted
separately. Float32/bfloat arithmetic uses the explicit round-to-nearest,
flush-subnormals source profile. Numeric NaN conversion policy remains tracked
in #2081; these tests do not establish bitwise NaN parity.

Pinned division adds three half, bfloat and float32 packages: 31,232 values and
24 guards per target, including all finite exponent ranges, rounding boundaries,
signed zeros and exceptional inputs. Float32/bfloat use the explicit division
profile; half retains gradual underflow. Finite words and guards are exact, and
NaN payload differences are reported separately. macOS also executes the
unchanged upstream source. Windows execution remains mandatory.

The binary step selects native test functions explicitly. Portable reflection,
configuration and mocked-dispatch tests remain in the Ubuntu complete suite.
Floating remainder types, the integer remainder batch and floating source-profile
batches are separate pytest cases,
scheduled across the existing two workers with one-test load-scheduling chunks.
The 16 floating entries share four translation and package builds: six half
kernels, four float32/bfloat comparison kernels, four float32/bfloat additive
kernels and two float32/bfloat division kernels. Half multiplication contributes
5,632 pairs and eight guards; the combined inventory retains all 99,088 values
and 128 guards. Comparison policy remains operation-specific, including exact
NaN payloads for selection inside the mixed half batch. Every entry keeps its own
artifact, exact load-unit selection, native compilation and readback; incompatible
profiles, duplicate entries and missing or blocked load units are rejected.
Each batch selects explicit entry names and one shared `[1, 1, 1]` workgroup rule;
separate per-entry rule validation remains tracked in #1970.
The unchanged
Metal reference library is built once per source root and worker, while every
case still runs its own original-source dispatch and checks its inputs. The
900-second bound, required native cases and evidence uploads are unchanged.
The 15 complex-power layouts are translated and packaged in three batches of
five. Each layout still has its own descriptor, native compilation, three
datasets and saved readbacks. Loader selection requires the exact source entry,
target and complete set of ready load units; it never defaults to the first
artifact in the batch. No additional native runners are used.

The same native-host jobs include a separate 180-second real-power step. Two
compatible translation batches cover exhaustive half/bfloat identity inputs,
binary32 identity controls and explicitly selected subnormal-operand semantics.
The five cases require 219,552 exact non-NaN outputs and 40 guards per target;
NaNs use classification checks. Metal also runs the unchanged source and
verifies its read-only inputs. Each case uses an exact two-thread workgroup
grid so all inputs fit DirectX's group-count limit; partial groups are rejected,
not rounded up or truncated. Reports and packages must retain the selected
operand policy. Remaining finite accuracy and output underflow are outside
these cases. This adds no runner and does not extend the earlier binary step.

The same arithmetic step verifies source-defined bitcast-name overloads beside
generated FMA, division and half-remainder helpers. Exact output words cover
scalar/vector overloads, aliases, namespace-qualified calls and both binary32
profiles. Original Metal is an independent control, and guards and side-effect
counters remain required. These checks add no platform jobs.

The native host job runs its full set of 17 portable contract-test modules on
Ubuntu only. Windows and macOS retain focused C callback, launch geometry,
library registration and memory-layout checks before their native workloads.
This avoids repeating package-generation and mocked execution tests on scarce
platform runners. All native compiler, device execution, adapted-library build
and unchanged upstream-test steps remain required on their respective systems;
the focused ABI checks do not replace them. Their JUnit report is retained in
the same native host evidence artifact. No additional runner is introduced.

Binary32 comparison profiles run once per native target in the existing
arithmetic step. They retain exact Boolean results, raw operand words, guards
and single-evaluation controls, including source functions named like bitcast
builtins. macOS additionally executes the unchanged source control. The two
explicit subnormal policies are tested independently of native Boolean casts
and other arithmetic profiles, without adding a runner or extending the bound.

Binary32 division source profiles run once per native target inside the existing
arithmetic step. Exact checks cover source operators, precise builtins, compound
writeback, narrow intermediates and guards. The project's Sigmoid boundary
regression stays under `demos/integrations/mlx/tests` and runs in the existing
pinned unary step. Neither adds a runner or duplicates the compiler-only corpus.
The Sigmoid check covers all 65,282 non-NaN bfloat inputs and eight output guards.
The same unary step requires native precise-logarithm checks for both explicit
subnormal profiles, scalar/vector results, narrow scalar promotion, operand
evaluation counts and guards. Its 28,024 binary32 inputs cover every exponent
and dense neighborhoods around one and the range-reduction boundaries. Ordinary
translation, metadata and compiler tests stay in the general suites rather than
being repeated in this native step; the existing 300-second bound is unchanged.
The same step requires correctly rounded precise square roots with preserved
and flushed subnormal inputs, vector and composed results, exhaustive non-NaN
half/bfloat scalar promotion, single evaluation and guards. Finite outputs are
bit-exact against an independent decimal reference, not an approximate tolerance.
Precise reciprocal square roots share this step and its unchanged deadline.
Their 21,026 binary32 inputs cover reciprocal output midpoints and both
subnormal policies; scalar/vector and nested results, exhaustive non-NaN
half/bfloat promotion, single evaluation and guards are checked independently.
The byte-conversion step also covers writable scalar/vector arguments, nested
calls, aliases, indexed locations and lazy conditional calls. Signed and unsigned
overflow checks retain output guards and original Metal comparisons. Separate
field-reference cases compile on DirectX/OpenGL and execute on their native
platforms; Metal field binding remains tracked in #2102. The vector-reference
control uses an explicit arithmetic conversion: it does not claim coverage of
the unsigned byte-vector compound-assignment failure tracked in #2023. These
cases reuse the existing three native jobs and their evidence uploads.
Distinct writable byte parameters have a native control. Potentially overlapping
byte arguments, including read-only aliases, must produce a structured
diagnostic rather than validating value-result code with different semantics.
Broader source-reference identity remains tracked in #2103.
The explicit division profile and precise-exponential lowering resolve the
previous OpenGL midpoint and underflow differences without relaxing comparisons.
The saved configuration and reviewed target identities are checked before
execution; required Windows evidence is still distinct from local Metal/OpenGL
results.

The existing Windows project job also checks native CBV/SRV/UAV allocation
ranges, shared read-only and disjoint writable views, and ordered allocation
reuse. The check is bounded and retains shader/module identities, physical
allocation addresses and full-allocation readback hashes. It adds no runner
and does not replace the public loader's numerical offset regressions.

The existing Windows storage gate also exercises explicitly encoded binary16
buffers through native loader descriptors. It checks all 65,536 payload words
in scalar, vector and homogeneous-struct integer storage, plus decoded arithmetic
and output guards. These handwritten ABI controls do not replace generated
half-copy tests. The same job also checks generated half arithmetic, assignment
results and constant-reference payloads using logical half values over integer
storage. Compiler inspection alone does not establish native correctness.
No runner is added and the existing execution bound is unchanged.

The same Windows, Metal and OpenGL storage jobs test direct 64-bit integer-to-bfloat
rounding over 64,276 signed and unsigned inputs. Every rounding midpoint and
its adjacent integer values, signed limits, carry into the next exponent,
single evaluation and explicit float32-mediated conversion are covered.
Original Metal and generated output must match an independent integer reference;
the corresponding DirectX execution is required on Windows, and GLSL execution
is required on Linux. OpenGL and Metal also cover wide compound assignments
through locals, members, arrays and resource buffers with in-range results.
Compiler-only checks remain distinct from these numerical results. No platform
job is added.

The existing self-comparison step checks both binary16 and binary32 private
scalar/vector predicates under optimized DXC. Native controls cover every
binary16 payload, lane-wise and aggregated results, unchanged input words and
output guards on each applicable target. These cases share the existing job
and deadline; no runner is added and repeated memory reads are not collapsed.

The existing Metal bfloat-vector job also checks nested constructor bitcasts.
It preserves all 65,536 payloads in scalar and two-, three- and four-lane values,
including aliases and single evaluation. Only defined lanes are inspected for
padded three-lane vectors. Separate conversion controls compare float32 and
wide-integer rounding with an independent reference. Unchanged source and
generated Metal must agree; unknown and unequal bitcast widths remain errors.

The Metal runtime reuses its unchanged host helper within each Python test
process. Source, compiler options, toolchain, SDK and architecture are part of
the cache identity; missing or modified executables are rebuilt. The native
package job checks reuse across independent runtime instances with real
dispatches. Shader compilation, device probing, numerical execution and
readback checks still run. This removes repeated Swift helper compilation
without sharing GPU state or adding a runner.

Concurrency is scoped to each job and matrix leg on a branch. Running native
work is allowed to finish, while only the newest waiting revision of that job
is retained. A long Metal corpus run therefore does not hold up an unrelated
DirectX or OpenGL check from the next revision. The workflow does not serialize
all platforms behind one workflow-wide queue.

Project demo checks run for changes to source, tests, demo inputs, support
contracts and build/toolchain configuration. Root documentation-only changes do
not launch the native demo matrices. Scheduled corpus audits remain enabled.
The Metal storage step has a 30-minute bound within an 80-minute job; the earlier
15-minute bound expired during passing cases as the required suite grew.

The full local command is `python -m pytest -q -n auto`. The all-file pre-commit
checks validate workflow contracts and generated support artifacts. Changes to
matrix coverage must update those contracts and this policy together.

The optional Metal discovery probe hides `xcrun` only when it confirms that the
Metal toolchain component is missing. A bounded lookup or probe timeout is
inconclusive and leaves tool discovery unchanged. Required native tests still
perform their own compiler and execution checks; a timeout is not a native pass.
The existing three-platform deferred-compilation jobs test this distinction.
The native Metal host job installs the missing component when needed, then
compiles and links the shared vector-add fixture before running native tests.
Compiler versions and the probe modules are retained with the host evidence;
an installation, compilation or linking failure stops the job without skipping
the required numerical checks.

The existing builtin-ownership jobs also execute scalar and vector truncation
with standard, fast and precise Metal namespace calls. They cover every half
and bfloat input payload, binary32 exponent boundaries, signed zeros, infinities,
NaN classification, source overloads and single evaluation. Exact finite result
bits and output guards are required; macOS compares unchanged source and generated
Metal. The same cases run on Linux OpenGL and Windows DirectX without new jobs.

The same jobs verify binary addition and multiplication grouping at vector widths
one through four. Cancellation and multiplication-rounding cases distinguish
right-associated expressions from left-associated ones; signed zeros and output
guards are checked bit for bit. Saved CrossGL and direct translation must agree.
Mixed signed/unsigned 32-bit and 64-bit bitwise cases check conversion order
before widening; expected high words distinguish the two associations.
Metal controls use `-fno-fast-math`, and DirectX uses `-Gis`, so compiler fast-math
reassociation does not hide a source-tree regression. Inputs, expected words,
native readbacks and generated artifact identities are retained with the existing
builtin-ownership evidence.

The existing gather/resource jobs run the outlined-functor cases in
`test_functor_member_runtime.py`. They verify selected and unused qualified
definitions with default arithmetic and the explicit additive profile on each
native target. Exact readbacks distinguish the qualified implementation from
its template fallback, retain input copies and check eight output guards.
macOS also executes the unchanged source. No runner or timeout is added.

The base-to-head coverage comparison accepts explicitly reviewed workflow moves
from `.github/ci-coverage-migrations.json`. Only workflow filenames are mapped;
job identities and coverage requirements are not removed. Each old job must
retain its timeout under the destination workflow, which must also retain every
positive permission and action-policy requirement. Missing jobs, duplicate job
destinations and weakened policies still fail. Applied mappings are recorded in
the comparison artifact. A mapping is inactive once its source is absent from
the base revision. Test, compiler and platform coverage are compared separately.
