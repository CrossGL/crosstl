# CI Coverage

The CI suite separates portable Python compatibility from native GPU toolchain
and runtime checks.

| Workload | Ubuntu | Windows | macOS |
| --- | --- | --- | --- |
| Backend, translator and example tests | Python 3.8 through 3.13 | Python 3.13 | Python 3.13 |
| Complete core and demo unit suite | Required | Covered by focused platform jobs | Covered by focused platform jobs |
| Complete DirectX demo corpus compilation | Pinned Linux DXC | Native execution remains below | Not required |
| Direct3D execution and host integration | Not applicable | Required WARP and dispatch checks | Not applicable |
| OpenGL execution and host integration | Required Mesa/EGL checks | Platform-specific controls | Platform-specific controls |
| Metal compilation, reference and host integration | Not available | Not available | Required native checks |

Every backend and code generator remains covered on all three operating
systems. Python-version compatibility is exercised fully on Ubuntu; repeating
those versions on Windows and macOS is not required. Windows and macOS each
run the complete backend directory in one job and the complete translator
directory in another, with two pytest workers per job. Ubuntu keeps its
component-level matrix. The version policy removes 198 duplicate jobs, and
grouping native-platform unit suites removes another 38 runner setups per
revision without removing a test suite. Failure reports retain individual
test names in JUnit even when suites share a runner.

The demo's 77 unary, binary, copy and reduction DirectX compilation shards run
on Ubuntu. They retain every entry, strict compiler flags, artifact identity
checks and the same DXC release. Pinned installation verifies the Linux archive
checksum and does not fall back to another release. These jobs do not execute
Direct3D; WARP numerical execution and Windows host integration remain required.
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

The existing Windows and Linux Softmax, arg-reduce, attention and normalization loader steps
retain their reports before artifact checks, generated packages, native compiler commands and
artifacts, input fixtures, execution plans and returned values. The project
job's final upload includes these files and each step's JUnit results even
when a check fails. Missing native results indicate that dispatch was not
completed; retained expected values alone are not execution evidence.

`demo-project-testing.yml` is the entry point for project integration checks.
Each native job keeps its source pin, numerical comparisons, guards, bounded execution
and retained evidence. Moving a compiler-only job must not disable an execution
gate or be presented as proof of runtime parity.

The native host job runs its full set of 17 portable contract-test modules on
Ubuntu only. Windows and macOS retain focused C callback, launch geometry,
library registration and memory-layout checks before their native workloads.
This avoids repeating package-generation and mocked execution tests on scarce
platform runners. All native compiler, device execution, adapted-library build
and unchanged upstream-test steps remain required on their respective systems;
the focused ABI checks do not replace them. Their JUnit report is retained in
the same native host evidence artifact. No additional runner is introduced.

Binary32 division source profiles run once per native target inside the existing
arithmetic step. Exact checks cover source operators, precise builtins, compound
writeback, narrow intermediates and guards. The project's Sigmoid boundary
regression stays under `demos/integrations/mlx/tests` and runs in the existing
pinned unary step. Neither adds a runner or duplicates the compiler-only corpus.
The Sigmoid check covers all 65,282 non-NaN bfloat inputs and eight output guards.
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

The same Windows and Metal storage jobs test direct 64-bit integer-to-bfloat
rounding over 64,276 signed and unsigned inputs. Every rounding midpoint and
its adjacent integer values, signed limits, carry into the next exponent,
single evaluation and explicit float32-mediated conversion are covered.
Original Metal and generated output must match an independent integer reference;
the corresponding DirectX execution is required on Windows. Compiler-only
checks remain distinct from these numerical results. No platform job is added.

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

The existing builtin-ownership jobs also execute scalar and vector truncation
with standard, fast and precise Metal namespace calls. They cover every half
and bfloat input payload, binary32 exponent boundaries, signed zeros, infinities,
NaN classification, source overloads and single evaluation. Exact finite result
bits and output guards are required; macOS compares unchanged source and generated
Metal. The same cases run on Linux OpenGL and Windows DirectX without new jobs.

The base-to-head coverage comparison accepts explicitly reviewed workflow moves
from `.github/ci-coverage-migrations.json`. Only workflow filenames are mapped;
job identities and coverage requirements are not removed. Each old job must
retain its timeout under the destination workflow, which must also retain every
positive permission and action-policy requirement. Missing jobs, duplicate job
destinations and weakened policies still fail. Applied mappings are recorded in
the comparison artifact. A mapping is inactive once its source is absent from
the base revision. Test, compiler and platform coverage are compared separately.
