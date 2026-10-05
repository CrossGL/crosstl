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

The complete binary, copy, reduction and quantized Metal corpora separate portable translation
from native compilation. Each family has 24 source-generation shards on Ubuntu,
preserving every body, materialization and resource-interface assertion. One
dependent macOS job per family verifies all artifact identities against the
checked-in contract and compiles every source with warnings fatal: 4,122 binary,
2,496 copy, 2,396 reduction and 2,052 quantized artifacts. Quantized compilation
retains the Metal 3.1 language standard and empty-compiler-stream requirement.
Shard downloads remain separate so duplicate
entries cannot overwrite each other. Missing, duplicated or changed sources fail
before any compilation; compiler failures retain diagnostics and output identities.
This removes 92 macOS jobs without sampling any corpus. Metal numerical and host
integration checks still run natively and are not replaced by this compiler job.
The quantized consumer also executes all 108 affine quantization and
dequantization variants from the same verified source bundle. Five groups per
variant cover both scale signs, dyadic values and zero ranges. Unchanged upstream
Metal and generated Metal must both match independently calculated packed bytes,
scales and biases; readonly inputs and output guards are checked exactly. The
15-minute native step retains commands, modules and readbacks in the existing
job artifact. It adds no runner and does not repeat translation on macOS.
The copy, reduction and quantized source phases permit only the explicit Metal toolchain-unavailable
warning when that toolchain is absent, retain it in the report and reject all
other diagnostics. Their dependent macOS phases still require native compilation.

The four DirectX corpus families retain each case's configuration, generated
shader and portability report before checking the recorded artifact identity.
When compilation is reached, its command, output and exit status are retained
alongside the compiled module. Every shard uploads this evidence and its JUnit
results on success or failure. A missing compiler record means compilation was
not reached; a generated shader alone is not proof that compilation passed.
Local runs clean up by default; set `CROSTL_KEEP_CORPUS_EVIDENCE=1` to retain the
same files under the upstream checkout's `.crosstl-corpus-evidence` directory.

`demo-project-testing.yml` is the entry point for project integration checks.
Each native job keeps its source pin, numerical comparisons, guards, bounded execution
and retained evidence. Moving a compiler-only job must not disable an execution
gate or be presented as proof of runtime parity.

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

The base-to-head coverage comparison accepts explicitly reviewed workflow moves
from `.github/ci-coverage-migrations.json`. Only workflow filenames are mapped;
job identities and coverage requirements are not removed. Each old job must
retain its timeout under the destination workflow, which must also retain every
positive permission and action-policy requirement. Missing jobs, duplicate job
destinations and weakened policies still fail. Applied mappings are recorded in
the comparison artifact. A mapping is inactive once its source is absent from
the base revision. Test, compiler and platform coverage are compared separately.
