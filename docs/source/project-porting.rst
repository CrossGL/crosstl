Project Porting
===============

CrossGL Translator includes a project-level orchestration layer for repositories
that contain shader or GPU source files across one or more supported source
backends. The project pipeline discovers translation units, invokes the existing
single-file translator for each unit, writes translated artifacts under a
separate output directory, and emits a machine-readable portability report.

Scope
-----

The project pipeline translates shader and kernel source artifacts. It does not
rewrite host runtime code, application build systems, resource binding setup, or
framework-specific backend integration. Those migration steps are reported as
manual follow-up work in the portability report.

Porting Workflow
----------------

Use the project pipeline as an audit-first migration workflow:

1. Start with a scan-only report to confirm discovery, source backend
   detection, configured targets, include directories, source overrides, and
   diagnostics before writing translated artifacts.
2. Add or refine ``crosstl.toml`` so repository-relative source roots,
   include/exclude patterns, source overrides, include directories, defines,
   named variants, optional entry-point selections, output directory, targets,
   and optional external corpus manifest are explicit.
3. Run ``translate-project`` into a separate output directory and keep the
   generated portability report with the translated artifacts.
4. Run ``validate-project`` on the generated report. Use the JSON output for
   automation, text output for local triage, or SARIF output for code-scanning
   systems.
5. Run ``inspect-report`` when the raw report is too large to review directly.
   The inspection output keeps bounded samples for diagnostics, failed
   artifacts, source maps, source remaps, validation records, external corpus
   entries, and migration actions.
6. Treat ``migration`` actions as manual host-integration work. They identify
   runtime API, resource binding, build-system, and backend framework review
   that remains outside shader/kernel source translation.

A typical first pass looks like:

.. code-block:: bash

   python -m crosstl scan /path/to/repo \
     --target metal \
     --target opengl \
     --output scan-report.json

   python -m crosstl translate-project /path/to/repo \
     --target metal \
     --target opengl \
     --output-dir crosstl-out \
     --report crosstl-out/portability-report.json

   python -m crosstl validate-project \
     crosstl-out/portability-report.json \
     --format text

   python -m crosstl inspect-report \
     crosstl-out/portability-report.json \
     --format text

The same project APIs are available to Python callers that need to integrate
with existing automation:

.. code-block:: python

   from pathlib import Path

   from crosstl.project import inspect_project_report, translate_project

   report_path = Path("crosstl-out/portability-report.json")
   report = translate_project(
       "/path/to/repo",
       targets=["metal", "opengl"],
       output_dir=report_path.parent,
       validate=True,
   )
   report.write_json(report_path)

   inspection = inspect_project_report(report_path)
   print(inspection["success"])

Use these report fields to decide the next action:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Report field
     - Triage use
   * - ``diagnosticCounts``, ``diagnosticsByCode``,
       ``diagnosticsByTarget``, ``diagnosticsBySourceBackend``, and
       ``diagnosticsByVariant``/``diagnosticsByCheckKind``
     - Separate configuration errors from source/backend translation failures,
       then group actionable diagnostics by target backend, source backend,
       named variant, and validation check kind before reviewing artifacts.
   * - ``missingCapabilityCounts``
     - Group unsupported source features, include resolution gaps, define
       forwarding gaps, artifact manifest issues, provenance issues, and
       optional toolchain validation gaps.
   * - ``artifactMatrix``
     - Confirm the expected unit, target, and named-variant artifact plan before
       translation, then identify missing or extra artifacts after translation.
   * - ``project.entryPointSelections`` and ``artifacts[].entryPoint``
     - Confirm that requested materialized source entries were each packaged
       under deterministic entry paths and identify their reflected target
       entries and stages.
   * - ``units[].entryDiscovery``
     - Review source-frontend discovery availability, host-visible entry names,
       stages, declaration provenance, and unresolved-name diagnostics before
       constructing an entry-scoped artifact plan.
   * - ``validation``
     - Check current source hashes and byte sizes, generated artifact hashes
       and byte sizes, source maps, source remaps, optional toolchain
       availability, and opt-in artifact or availability smoke test results
       after translation.
   * - ``externalCorpus``
     - Compare pinned reduced corpus entries with discovered units and emitted
       artifacts without treating the manifest as whole-repository semantic
       parity.
   * - ``migration``
     - Track manual runtime, binding, build-system, and backend integration
       follow-up work separately from translated shader/kernel artifacts.

Commands
--------

The legacy single-file command remains available:

.. code-block:: bash

   python -m crosstl examples/graphics/SimpleShader.cgl --backend metal

Legacy single-file options may appear before or after the input path.

The explicit single-file subcommand is equivalent:

.. code-block:: bash

   python -m crosstl translate examples/graphics/SimpleShader.cgl --backend metal

Both single-file forms also accept ``--source-backend``, repeatable
``--include-dir``, and repeatable ``--define`` overrides. Use them when a file
has a nonstandard extension or when the selected source frontend exposes
include-path and preprocessor define options. Use ``--output -`` to write the
translated source to stdout instead of creating an output file.

Scan a repository and print a JSON report:

.. code-block:: bash

   python -m crosstl scan /path/to/repo --target metal

Emit the same scan-only portability report with an explicit output path:

.. code-block:: bash

   python -m crosstl report /path/to/repo \
     --target metal \
     --output crosstl-out/portability-report.json

Scan and report commands exit nonzero when the generated report contains error
diagnostics, while still writing the JSON report to stdout or the requested
output file.
Use ``--output -`` on single-file translation, scan, report, validation, and
inspection commands, or ``--report -`` on ``translate-project``, when stdout
should be selected explicitly in scripts.

Translate every discovered unit to one or more targets:

.. code-block:: bash

   python -m crosstl translate-project /path/to/repo \
     --target metal \
     --target opengl \
     --output-dir crosstl-out \
     --report crosstl-out/portability-report.json \
     --run-toolchains

Project translation exits nonzero when the report contains failed artifacts or
error diagnostics.
``--validate`` records artifact existence, source and generated hash checks,
source-map and source-remap status, and configured toolchain availability
without invoking external compiler tools.

Embedded toolchain availability records name the configured validation hook
tools for each target; paths and availability remain environment-specific.
``--run-toolchains`` implies artifact validation and records any available
bounded toolchain smoke-check results in the generated portability report.
Smoke-check records and generated toolchain-failure diagnostics include a check
kind so report consumers can distinguish artifact checks from target tool
availability checks.
OpenGL smoke checks invoke ``glslangValidator`` with ``--stdin`` and an
explicit ``-S`` stage inferred from the artifact extension or common GLSL
builtins, defaulting to compute for generic ``.glsl`` outputs.
Vulkan smoke checks validate binary ``.spv`` artifacts with ``spirv-val`` and
assemble textual ``.spvasm`` artifacts with ``spirv-as -o`` pointed at the
platform null device.

Bounded Translation Concurrency
-------------------------------

Project translation is sequential by default. Use ``--workers`` to run
independent artifact jobs in isolated processes:

.. code-block:: bash

   python -m crosstl translate-project /path/to/repo \
     --target directx \
     --target opengl \
     --output-dir crosstl-out \
     --report crosstl-out/portability-report.json \
     --workers 2

Python callers use the corresponding ``max_workers`` argument to
``translate_project``. The value must be a positive integer; ``1`` retains the
sequential path.

At most ``N`` jobs are submitted at once. Each worker translates one planned
unit, target, variant, and selected entry combination in a separate process so
mutable frontend and generator state is not shared between concurrent jobs.
Before workers start, the pipeline traverses the canonical plan once to reject
output-path collisions. The execution pass then regenerates that plan lazily
and retains at most ``N`` scheduled requests instead of retaining every
unit, target, variant, and entry combination in memory.
The parent process consumes results in the deterministic project plan order,
publishes each staged artifact/source-remap pair, writes the corresponding
checkpoint completion, and assembles diagnostics, artifacts, and the artifact
matrix with the same ordering as a sequential run. Workers do not replace
published outputs directly.
Artifact validation and optional toolchain smoke checks run only after all
translation jobs have returned, avoiding nested toolchain concurrency.

Use ``--job-timeout-seconds`` to place a finite wall-clock limit on each
artifact job:

.. code-block:: bash

   python -m crosstl translate-project /path/to/repo \
     --target directx \
     --target opengl \
     --output-dir crosstl-out \
     --report crosstl-reports/portability-report.json \
     --checkpoint crosstl-reports/translation-checkpoint.json \
     --workers 2 \
     --job-timeout-seconds 300

Python callers use the corresponding ``job_timeout_seconds`` argument. The
value must be a positive finite number. A configured timeout uses process
isolation even when ``max_workers=1`` so the active translation can be stopped.
The budget starts when a job is submitted. A result that has already completed
is accepted even when its budget has elapsed before the coordinator reaches it.

When a job exceeds its budget, the coordinator terminates that worker-pool
generation, removes its private staging, and resubmits unaffected scheduled
jobs in canonical order. The timed-out coordinate is retained as a failed
artifact with a ``project.translate.timeout`` diagnostic and, when enabled, a
completed checkpoint record. Previously published output for that coordinate
is not replaced. Translation then continues so the final report describes the
rest of the repository instead of relying on an outer process timeout.
Changing the configured timeout changes checkpoint invocation identity and
therefore requires a new run rather than an incompatible resume.

An interruption stops further submission, cancels work that has not started,
and terminates pool processes that are still translating. Each concurrent run
adds a private token to its staging directories. After the pool settles, the
coordinator removes unconsumed results and any token-owned staging left by a
terminated process without touching another invocation's files. Previously
published outputs remain unchanged. Completed checkpoint records remain
available for a verified resume, and only coordinator-published results are
recorded as complete.
An ordinary worker failure raises ``ProjectTranslationWorkerError`` with the
source, target, output path, optional entry point, original exception type, and
original exception attached for programmatic inspection. An enabled checkpoint
records the same active coordinate and typed interruption message.

Spawned workers independently load installed backends and plugins. A backend
registered only in the current Python process is not inherited on every
platform; use ``max_workers=1`` or install that backend through the supported
plugin discovery mechanism.

Artifact Publication
--------------------

Each translation job writes its generated artifact and source-remap sidecar to
a temporary directory on the destination filesystem. The pipeline computes
the reported hashes, byte sizes, source maps, and placeholder diagnostics from
the staged files before publication. It then publishes the sidecar followed by
the artifact with atomic file replacement, treating the artifact replacement
as the pair's commit step.

If generation or publication fails, temporary files are removed and any
previously published artifact/remap pair is retained. If replacement has
started, the pipeline restores both prior files before reporting the job
failure. Consumers should still use the portability report as the authoritative
result of the latest run; a retained pair represents the last successful
translation, not the failed attempt.

Checkpointing Long Runs
-----------------------

``translate-project`` can persist progress independently of the final
portability report. Use a checkpoint for repository runs that may be interrupted
by a local process limit or CI timeout:

.. code-block:: bash

   python -m crosstl translate-project /path/to/repo \
     --target directx \
     --target opengl \
     --output-dir crosstl-out \
     --report crosstl-reports/portability-report.json \
     --checkpoint crosstl-reports/translation-checkpoint.json

The checkpoint is replaced atomically as work advances. It records the
``running``, ``interrupted``, or ``complete`` state; project and invocation
identity hashes; the deterministic job plan; completed, active, and pending
coordinates; scan and translation diagnostics; and a partial artifact matrix.
Generated source is not embedded in the checkpoint.

Resume an unfinished run with the same project configuration and translation
options:

.. code-block:: bash

   python -m crosstl translate-project /path/to/repo \
     --target directx \
     --target opengl \
     --output-dir crosstl-out \
     --report crosstl-reports/portability-report.json \
     --checkpoint crosstl-reports/translation-checkpoint.json \
     --resume

Before skipping a completed job, resume verifies the project identity, translator
implementation, complete job plan, current source identity, generated artifact
hash and size, and source remap hash and size. Stale, modified, missing, or
mismatched outputs stop the resume instead of being trusted. A checkpoint path
must be outside the artifact output directory and cannot replace the project
configuration or a registered source file.

The implementation identity hashes the installed Python and native worker
sources, including their package-relative paths. It does not require Git and is
independent of installation location, bytecode caches and source newline style.
Both released packages and editable installations use this identity; matching
version numbers alone do not establish compatibility. Source-backed packages are
required for checkpointing. Keep the installed sources unchanged during a run.
This fingerprint does not identify external compilers, dependencies or in-memory
runtime modifications.

Older checkpoints without an implementation identity, or checkpoints from a
different implementation, are rejected before completed artifacts are reused.
Restart without ``--resume`` and select a new checkpoint path after updating
CrossTL. Rejection does not overwrite the old checkpoint or its completed
artifacts.

The default writes each job transition. For projects with large report metadata,
``--checkpoint-interval-jobs N`` persists batches of completed jobs. A larger
interval reduces write overhead, but an ungraceful termination may cause up to
``N - 1`` already-emitted jobs to be translated again because only persisted
completions are trusted.

Python callers use the corresponding ``checkpoint_path``, ``resume``, and
``checkpoint_interval_jobs`` arguments to ``translate_project``. Worker count
does not affect checkpoint identity, but a configured per-job timeout does.

A progress checkpoint is not a final portability report. Only a ``complete``
checkpoint contains the canonical final report, and neither state establishes
host runtime integration or numerical parity.

Source Entry Discovery
----------------------

Project scans record an ``entryDiscovery`` object on every translation unit.
Its status distinguishes a frontend that supports discovery (``available``),
a frontend that does not yet expose discovery (``unavailable``), and a
frontend failure (``failed``). An available result contains ordered,
deduplicated entry records with the exported name, canonical stage, source
location, and declaration provenance. Discovery diagnostics are also promoted
to the project diagnostic stream under the ``source-entry-discovery`` check
kind.

Metal discovery expands configured includes, defines, and active
preprocessor branches without materializing template bodies or invoking a
target generator. It reports concrete entry functions and explicit
host-named template materializations. Commented examples, inactive branches,
ordinary helpers whose surrounding comments mention kernels, and unresolved
dynamic host names are not converted into entries. Locations currently use
the ``preprocessed-source`` coordinate space, which is recorded explicitly in
the report.

Comment and literal exclusions use the preprocessor's bounded interval index,
shared within a discovery call. Repeated candidate lookups do not rescan the
entire exclusion list. Each call creates its own preprocessor, so changed
includes, defines and source options cannot reuse indexes from an earlier
discovery. Entry metadata and unresolved-name diagnostics retain the same
half-open source-range semantics.

The public ``ProjectScan.discovered_entry_points()`` method returns a
repository-relative mapping compatible with ``ProjectConfig.entry_points``:

.. code-block:: python

   from dataclasses import replace

   from crosstl.project import scan_project, translate_project

   scan = scan_project("/path/to/repo")
   config = replace(
       scan.config,
       entry_points=scan.discovered_entry_points(),
       targets=("directx", "opengl"),
   )
   report = translate_project(config)

Discovery does not change translation behavior by itself. This explicit step
lets callers review or filter a potentially large entry set before scheduling
artifacts. Source frontends without a discovery provider report
``unavailable`` rather than returning an empty result that could be mistaken
for a source file with no entries.

For repositories where selected source files should expand automatically, add
repository-relative source patterns to ``translate_discovered_entry_points``:

.. code-block:: toml

   [project]
   translate_discovered_entry_points = [
     "kernels/arange.metal",
     "kernels/normalization/*.metal",
   ]

Only matching units whose discovery status is ``available`` and which contain
concrete entries are expanded. Each discovered entry uses the existing
entry-scoped artifact planner, output path, checkpoint coordinate, source map,
provenance, and target behavior. Source and entry ordering remain the ordering
recorded by the scan.

An explicit exact or glob selector in ``project.entry_points`` takes
precedence for every source it matches. Sources outside the configured
``translate_discovered_entry_points`` patterns retain aggregate translation.
This source-scoped contract avoids turning a repository-wide scan into an
unbounded artifact matrix; choose broad patterns only after reviewing the
discovered entry count in a scan report.

The same selection can be added for one invocation with a repeatable CLI
option:

.. code-block:: console

   python -m crosstl translate-project /path/to/repo \
     --translate-discovered-entry-points "kernels/arange.metal" \
     --target directx --target opengl

Invalid or unmatched patterns and matching sources with unavailable, failed,
or empty discovery produce structured configuration diagnostics. They are not
reported as successful per-entry expansion.

Entry discovery identifies shader and kernel declarations only. It does not
infer dispatch dimensions, resource bindings, host call sites, backend
initialization, or numerical parity.

Entry-Scoped Compute Artifacts
------------------------------

Repositories can request standalone artifacts for one or more materialized
source entries by adding a repository-relative selector table to
``crosstl.toml``:

.. code-block:: toml

   [project]
   include = ["kernels/arange.metal"]
   include_dirs = ["."]
   targets = ["directx", "metal", "opengl"]
   output_dir = "crosstl-out"

   [project.entry_points]
   "kernels/arange.metal" = ["arangeuint32", "arangec64"]

Each value may be one entry name or an ordered array of entry names. An array
creates one independently checkpointed artifact per entry in the declared
order. Empty arrays and duplicate names are rejected.

For DirectX compute output, each selected source entry is emitted as target
entry ``CSMain``. A source such as ``kernels/arange.metal`` produces
``crosstl-out/directx/kernels/arange/arangeuint32.hlsl`` and
``crosstl-out/directx/kernels/arange/arangec64.hlsl``. Each standalone HLSL
artifact retains only that entry's reachable helpers, resources, constants,
and execution contract. Explicit registers and spaces remain unchanged, while
runtime-loader metadata records the selected ``cs_6_0`` entry profile.

For OpenGL compute output, the same selection produces one ``.glsl`` artifact
per entry with target entry ``main``. For Metal compute output, it produces one
``.metal`` artifact per entry and retains only the selected kernel, reachable
helpers, referenced declarations, and recursively referenced struct families.
The emitted Metal entry name is recorded exactly rather than normalized to a
fixed ``main``. Runtime manifests reflect the standalone Metal source and keep
``buffer``, ``texture``, and ``sampler`` index spaces independent.

For all three targets, the portability report records every source entry,
target entry, and reflected stage; embedded validation records carry the same
identities. Runtime artifact manifests then reflect only the selected stage
interface from each standalone output.

Selection is exact after source materialization. Missing or ambiguous entries
fail with structured diagnostics and no target file. Targets that do not yet
implement standalone entry generation also fail explicitly instead of pruning
an aggregate artifact. When ``project.entry_points`` is absent, project
translation keeps the existing aggregate output path and behavior.

Entry selection scopes shader or kernel translation; it does not infer host
dispatch dimensions, runtime bindings, or backend integration. Record those
requirements through the corresponding dispatch and runtime contracts.

The next MLX kernel-tree increment is pinned independently at
``9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8``. Entry discovery must report
exactly 49 Metal units, 17,832 entries, and zero diagnostics. The compact
``arg_reduce.current-tree.translation.json`` contract pins all 24 discovered
``arg_reduce.metal`` entries and 72 deterministic Metal, OpenGL, and DirectX
artifacts. Required macOS, Linux, and Windows CI compiles all 24 entries with
the native Metal compiler, ``glslangValidator``/``spirv-val``, and DXC,
respectively. Numerical runtime parity remains an explicit representative
float32 subset: ``argmin_float32`` and ``argmax_float32`` must execute on Metal,
Mesa EGL, and Direct3D 12 WARP over two rows, axis sizes 32 and 129, strides 1
and 2, ordinary values, and NaN/Infinity values; the Metal path also executes
the exact upstream metallib for parity. This is a 24/17,832 deterministic
translation and native-compiler increment with 2/17,832 numerical runtime
coverage, not full-tree coverage. The contract explicitly records that the
upstream MLX test suite and MLX host-runtime redirection have not yet been
implemented. Strict JSON runtime requests encode non-finite float32 inputs with
the exact strings ``nan``, ``+infinity``, and ``-infinity`` before
deterministic IEEE-754 packing.

The current OpenGL identities were reviewed against retained sources after
signed 64-bit remainder and narrow-value lowering changed. All 24 entries
compile and validate, and the 16 required float32 execution cases pass on Mesa
EGL. A separate local review covers 80 half/bfloat cases with source-rounded
inputs; it does not expand the required CI runtime subset. A separate Metal
identity review covers native narrow aggregate fields and contextual byte-field
initializers. All 24 generated entries compile with ``-Werror``. The eight changed
bodies retain their source, specialization and execution metadata, while the
16 other bodies are unchanged. The four byte entries also match unchanged
upstream Metal execution and independent integer indices in 64 local cases
covering extrema, ties, constant rows, tails and strides. Required CI runtime
coverage remains the stated float32 subset.

All 24 DirectX identities have complete-body review and warning-fatal DXC
compilation. Reviewed changes make source-width index conversions and byte
conversion boundaries explicit, preserve the negative-infinity sentinel, and
decode logical half values from integer storage. Source hashes, template
materializations, resource registers, launch geometry, reduction comparisons
and tie-breaking are unchanged. The two float32 entries retain the required
32-lane software subgroup path. The existing Windows numerical gate remains
required and passes all 16 float32 cases with these reviewed shaders. A separate
audit verifies all 32 reduction indices against retained uploaded input words,
including strides, tails, ties and non-finite values, and checks parameter
bindings and executed module identities. This does not expand numerical
coverage to the other scalar entries or establish upstream-suite parity.

Constrained Metal free-function fallbacks are materialized only when their
recognized constraints select a unique implementation. A visible ordinary or
unconstrained-template overload with compatible argument count causes a
structured specialization diagnostic when precedence cannot be proven;
forward declarations participate even without a definition. General C++
conversion ranking and template partial ordering are not implemented by this
path. Use a distinct helper name or an explicit specialization to remove the
unresolved competition rather than relying on a guessed overload.
Generated helper names reserve existing source identifiers, including local
variables and parameters. A deterministic suffix avoids name collisions while
repeated calls to the same specialization reuse one helper.

Global explicit free-function specializations are selected after the constrained
primary is resolved; they are not competing overloads. Selection compares
canonical template arguments and concrete parameter signatures, including
visible aliases and deduced or defaulted arguments. A matching specialization
keeps its own body, while an unrelated specialization does not displace the
primary. Ambiguous canonical identities and unproven ordinary-overload
precedence still produce diagnostics. Namespaced explicit specializations remain
outside this global selection path. Required Metal, DirectX and OpenGL execution
tests distinguish primary addition from specialized subtraction and verify both
default and non-default template arguments.

Kernel translation omits an out-of-line call-operator definition only when its
unique unqualified owner has no references outside its own declaration and
call-operator definitions. This prevents unused, unmaterialized helper bodies
from blocking arithmetic-profile validation. Library-only sources, aliases,
template references, nested calls, exported definitions and uncertain owner
bindings retain their bodies. Source offsets and line breaks are preserved.
Required native checks distinguish the selected qualified implementation from
its template fallback; no reachable computation is replaced by a placeholder.

The two current-tree DirectX float32 entries explicitly select
``software_subgroup_width = 32``. Their shuffle helpers use shared storage and
workgroup barriers instead of hardware wave instructions. A private
per-invocation index initialized by the entry point preserves lane identity
through helper calls. Early returns are accepted only under proven
workgroup-uniform control flow; a subgroup-ID condition is uniform only when
the workgroup contains one logical subgroup. Writes, shadowing, and mutable
helper arguments invalidate the corresponding uniformity assumptions.
Other DirectX entries retain the native-wave compiler path. Both paths still
require DXC validation, and the numerical entries retain every existing case
and the required Windows WARP execution gate. Compilation alone does not
establish numerical parity or resolve a native-runtime timeout.

Explicit-width software subgroups on DirectX and OpenGL also support scalar
Boolean all/any votes. The native host workflow requires compiler validation and
device readbacks across multiple logical subgroups, repeated calls, single
operand evaluation and untouched output guards. macOS additionally executes
the original Metal source. Divergent exits before a later collective are
rejected rather than dropping the exit or forcing inactive lanes into the vote.
OpenGL resolves unconditional wrapper chains to their exact collective helper
overloads. Entry-point calls may occur in proven uniform loops or branches
controlled by immutable workgroup-uniform inputs. Recursive chains, ambiguous
root overloads, lane-dependent branches and escaped or mutated uniform inputs
remain diagnostic. The native vote gate exercises three wrapper levels and
different predicates in even and odd workgroups; shared scratch accesses use
explicit memory ordering as well as execution barriers. This does not establish
complete MLX reduction support: collective-result uniformity and full host
reduction dispatch remain separate work.

Scalar float32, int32 and uint32 products use an adjacent-pair reduction tree
with strides 1, 2, 4, 8 and 16 on both software backends. Integer products wrap
at 32 bits. The required native product gate checks repeated helper calls,
single operand evaluation, overflow, signed zeros, infinities, NaNs and
order-sensitive rounding against original Metal and independent references.
Input words, readbacks, output guards and validation modules are retained.
NaN payloads are not required to match. This defines the software reduction
order; it is not a claim of identical floating-point results for every possible
native hardware reduction order or denormal mode.
Narrow products remain diagnostic until their intermediate rounding is
preserved, even where a target normally uses a 32-bit carrier for that type.

Pure scalar helpers of the form ``if (subgroup_vote) return fallback; return
subgroup_reduce(value);`` can converge both collectives before selecting each
logical subgroup's result. This retains different subgroup votes within one
workgroup; it does not reclassify them as workgroup-uniform. The lowering requires
safe by-value operands and rejects memory loads, effectful calls, mutation and
unproven arithmetic. Direct votes, negated votes and immutable Boolean locals
are supported, including unused method receivers. Native-mode branches remain
unchanged. Required native tests compare generated outputs with independent
references and original Metal compiled without fast math, retaining exact
finite results, NaN classification, repeated calls and guards. Full MLX host
reduction planning and arbitrary collective control flow remain separate work.

The bounded Windows test also records live Direct3D 12 debug-layer messages
through ``tools/run_directx_diagnostics.py``. Each entry's JSON-lines log is
stored outside pytest's temporary directory and uploaded with the runtime
evidence, including on timeout. Missing debug-layer tooling is reported
explicitly and does not skip or replace the numerical test. The wrapper
preserves the wrapped Python module's arguments and exit status.
After device creation it records the loaded Direct3D and WARP DLL paths and
SHA-256 hashes, independently of the runtime installation log. The Windows
project-porting job pins WARP 1.0.21 with an archive checksum. Its
`release notes <https://www.nuget.org/packages/Microsoft.Direct3D.WARP/1.0.21>`_
describe revised uniform and divergent control-flow handling; numerical tests
remain the acceptance gate for the runtime update.
The current DirectX kernel test additionally saves the actual runtime-loaded
DXIL and the final packed register payloads immediately before each dispatch in
``native-dispatches``. This distinguishes the executed module and bindings from
the standalone compiler check's artifacts, including when dispatch never returns.

Before the full kernels, Windows also executes eight small translated reduction
checks: sparse-register metadata reads, signed 64-bit index division, helper-level
software shuffles, their combination with a bounded input loop, NaN-aware pair
comparisons, non-finite inputs, a second shared-memory reduction stage, and
private-array accumulation. Each runs in a separate process with a two-minute
deadline and exact readback comparisons.
The diagnostic logs, CrossGL input, generated HLSL, DXIL, and completed readbacks
are retained under ``mlx-current-results`` and uploaded before the full kernels
run. All eight checks are attempted and any failure fails the step. These isolate
runtime failures; they neither replace the MLX numerical cases nor establish
whole-kernel parity.

The current-pinned MLX integration exercises entry-scoped translation for all
877 discovered entries from the include-expanded ``unary.metal`` source. The
finite split is 183 each for ``v_``, ``v2_``, ``gn1_``, and ``gn4large_``, plus
145 ``vn_`` entries, spanning 37 operators, 20 concrete input/output type pairs,
and 16 semantic families. ``v_`` uses explicit ``N=1``; ``v2_`` and ``vn_``
retain ``N=WorkPerThread<T>::n`` from the source default; ``gn1_`` uses
``N=1, IdxT=int``; and ``gn4large_`` uses ``N=4`` plus the source-default
``IdxT=int64_t``.

Every independently translated artifact has one selected operator implementation
and one kernel. Vector shapes materialize one specialization and reflect input,
output, and size resources. Gather shapes additionally materialize the reachable
``elem_to_loc<int>`` or ``elem_to_loc<int64_t>`` helper and reflect constant
shape/stride buffers plus a read-only device dimension binding. The resulting
877 artifacts contain 1,243 exact materializations, preserve a host-owned
``[1, 1, 1]`` workgroup contract, and reject residual templates, ``decltype``,
call operators, unsupported placeholders, and non-selected operator bodies.

Required macOS CI compiles all 877 exact artifacts with
``xcrun -sdk macosx metal -Werror -c`` and requires a non-empty AIR output from
each with no warning exemption. The generic path resolves source typedef chains
and bfloat reconstruction, materializes source-compatible constrained free
operators, infers aggregate aliases, recognizes branch-complete returns,
preserves narrow ``as_type`` storage, and admits only a proven read-only
immediate scalar-parameter view as a matching thread-local single-field
aggregate. The non-scalar path also retains
constant-resource provenance, const-device references, postfix update position,
and native Metal union aliasing while ambiguous address spaces and unsupported
reinterpretation continue to fail closed. This complete discovered-unary
selected-entry proof does not claim Metal numerical execution, host-runtime
redirection, or the MLX test suite.

The same entry-scoped pipeline translates all 877 historical unary entries to
standalone OpenGL ``main`` artifacts at revision
``846d176227a0ac13d2667e58d2bb68b322109ab0``. The schema-v2
``unary.opengl-translation.json`` contract preserves the same five-shape,
37-operator, 20-type-pair classification and all 1,243 exact materializations,
while pinning 4,771,646 generated GLSL bytes and all 3,363 target-reflected
resources. Vector artifacts expose read-only input, read-write output, and an
entry-scoped size uniform block. Gather artifacts expose input, output, shape,
stride, and the read-only ``ndimBuffer`` storage resource; scalar uses of the
source ``device const int& ndim`` alias element zero without changing expression
typing.

OpenGL's 32-bit index profile cannot represent every source 64-bit index
implicitly. The complete contract therefore records explicit host/runtime
bounds of ``[0, 2147483647]`` for ``offset + i``, ``out_idx++``, and ``idx``.
These are declared portability preconditions, not inferred facts or generated
runtime checks; absent proof continues to fail closed. GLSL also has no native
``log10`` overload, so reachable Metal base-ten logarithms lower through a
single evaluation of ``log2(value) * 0.3010299956639812`` while user-defined
``log10`` functions remain ordinary calls.

Required CI partitions the family across five disjoint Linux shards. Each shard
retranslates its exact entries, verifies generated identity, materialization,
``main`` workgroup metadata, and reflected ABI, compiles with
``glslangValidator --target-env opengl --target-env spirv1.3 -S comp``, validates
with ``spirv-val --target-env spv1.3``, and requires every SPIR-V module to be
non-empty. This closes whole-family OpenGL translation, reflection, and native
compiler coverage; it does not claim OpenGL numerical execution, MLX host
runtime redirection, or MLX test-suite parity.

The same entry-scoped pipeline translates all 877 unary entries from the legacy
reference revision ``846d176227a0ac13d2667e58d2bb68b322109ab0`` to standalone
DirectX ``CSMain`` artifacts. The schema-v2
``unary.directx-translation.json`` contract retains the exact five-shape,
37-operator, 20-type-pair classification and 1,243 materializations. ``v_`` and
``vn_`` artifacts reflect input, output, and entry-scoped size resources;
``v2_`` artifacts additionally reflect the generated ``CrossGLDispatchInfo``
workgroup-count cbuffer at ``b3``. Gather artifacts reflect input, output,
shape, stride, and read-only ``ndim`` structured buffers plus their generated
``CrossGLDispatchInfo`` cbuffer at ``b0``, for 3,912 reflected HLSL resources
in total. Source scalar uses of
``device const int& ndim`` alias ``ndim[0]`` and source ``out_idx++`` remains a
postfix update.

The reviewed HLSL references total 3,766,443 bytes. The earlier complete review
compared all 877 artifacts against hash-verified originals and compiled them
with warnings fatal, updating 603 identities and retaining 274. A subsequent
review recompiles all 34 Exp and Sigmoid entries and updates 20 identities:
15 scalar Exp entries preserve precise exponential calls and five bfloat
Sigmoid entries preserve source arithmetic boundaries. The remaining 14
selected bodies are unchanged. Source pins, materialization, ABI and launch
contracts are unchanged. The separate required Windows Square and
ArcCos numerical tests keep their existing artifact identities and tolerances.
This compiler contract does not establish numerical parity for all 877 entries.
Historical Sigmoid uses the default exponential and no optional division
profile; the current-pin, explicitly profiled numerical proof is separate.

DirectX bfloat support covers every unary intrinsic required by the family.
The inverse-hyperbolic ``acosh``, ``asinh``, and ``atanh`` paths decode bfloat
to float, invoke their portable float helpers, and round back to the exact
bfloat payload representation. Explicit contextual HLSL constructors preserve
float-to-native-16 return and initializer narrowing without ``-Wconversion``
diagnostics when warnings are fatal. All artifacts derive
``-enable-16bit-types`` from retained native 16-bit declarations and target
shader model ``cs_6_2``. As with OpenGL, explicit host/runtime bounds of
``[0, 2147483647]`` for ``offset + i``, ``out_idx++``, and ``idx`` remain
portability preconditions rather than inferred or generated checks.

Canonical Metal ``fabs``, ``fmin``, and ``fmax`` calls lower to HLSL
``abs``, ``min``, and ``max``. Boolean scalar/vector ``select`` conditions
use HLSL's ``select(condition, trueValue, falseValue)`` ordering, preserving
evaluation of both value arguments. Source-defined helpers keep their own
semantics. Unresolved or non-Boolean conditions, invalid argument counts, and
shadowed target intrinsics produce source-located diagnostics; integer-mask
selection is not inferred from a Boolean conversion. The HLSL frontend normalizes
condition-first selection before target lowering and rejects three-argument
calls when a same-named source overload makes ownership unresolved. A bounded Windows test
requires DXC compilation and Direct3D readbacks for finite values, NaNs,
infinities, vector selection, and eager evaluation. This isolated arithmetic
check does not establish whole-MLX numerical or host-runtime coverage.

Canonical ``atan2(y, x)`` uses typed HLSL helpers to retain the sign of zero
when selecting a quadrant. The helpers also handle both-infinite operands,
signed axis angles, and NaNs explicitly. Finite angles use exponent-scaled
significands and a range-reduced polynomial instead of the target intrinsic;
``precise`` intermediates preserve the polynomial evaluation order. Float
scalar/vector operands are supported, with
explicit promotion and narrowing for half/minimum-precision forms and the
existing scalar bfloat decode/encode path. Arguments are evaluated once.
Unknown, mismatched or unsupported operand types and shadowed target intrinsics
fail closed. Source-defined overloads retain their behavior, and HLSL-to-HLSL
``atan2`` round trips remain native. This is not a general subnormal or
transcendental-accuracy contract.

A bounded Windows readback test checks 1,121 operand pairs through scalar and
vector calls, including raw-bit and float-upload echoes, signed zeros,
infinities, NaNs, extreme normal exponents, all quadrants and range-reduction
boundaries. Finite results must stay within the unchanged absolute error
bound of ``2e-6``; signed axis and infinite angles retain exact bit checks.
macOS executes the unchanged Metal source with fast math
disabled as a control; fast-math compilation may ignore signed zeros and
non-finite values. The pinned MLX complex-power test separately retains its
existing numerical reference and error bound; passing an isolated angular
test does not replace that end-to-end proof.

The frozen binary and unary HLSL contracts at MLX revision
``846d176227a0ac13d2667e58d2bb68b322109ab0`` retain their source pins,
entry classifications, materialization counts and interfaces. The angular
lowering changes 84 binary and 28 complex-unary artifact identities; each
changed entry is recompiled with DXC and warnings fatal before refreshing its
fingerprint. The other 4,887 identities remain unchanged. This historical
contract refresh is separate from current-revision MLX runtime coverage.

OpenGL also lowers canonical ``fabs``, ``fmin``, ``fmax``, and Boolean
``select`` for desktop GLSL 4.00 and later. Floating min/max helpers explicitly
return the numeric operand when the other operand is NaN, in either argument
order. Selection uses typed helper parameters and per-component conditions,
without numeric interpolation or short-circuiting the source value arguments.
Both values must have matching scalar/vector types; the condition may be a
scalar Boolean or a Boolean vector of the same width. Integer masks, mismatched
types, unresolved operands, and shadowed required target builtins produce
structured diagnostics. Generated helpers avoid source identifiers, and
source-defined functions retain their own behavior. Required Linux CI compiles
the generated GLSL, validates its SPIR-V, and executes the source through the
OpenGL runtime, retaining readbacks for finite values, NaNs, infinities, signed
zeros, mixed vector masks, and argument side effects. This is a focused operation
contract, not evidence that every MLX kernel or its host integration works.

Floating ``fmod`` is distinct from floor-based ``mod``. OpenGL uses integer
significand division for binary32 and binary64 scalar and vector operands,
including scalar broadcasting. This avoids floating quotient overflow and
preserves representable remainders and signed zero. Zero divisors, infinite
dividends and NaN operands produce NaN; payload identity is not promised.
Binary64 requires target 64-bit integer support as well as double precision.
Unsupported operand shapes and profiles produce structured diagnostics.
Unprofiled half and bfloat calls remain unsupported until their source-specific
exceptional-value rules can be represented; they are not silently widened.
Required Linux execution compares raw output words with an independent rational
oracle, including subnormals, extreme ratios, exceptional values and guards.
This does not establish parity with every source compiler's fast-math profile:
the MLX controls expose half signed-zero and infinite-divisor differences and
bfloat subnormal-policy differences tracked in
`issue #2097 <https://github.com/CrossGL/crosstl/issues/2097>`_.

Metal sources can explicitly select
``binary16_remainder_profile = "binary32-quotient"`` in their source options.
This profile widens half operands exactly, rounds division to binary32 with
ties to even, truncates the quotient toward zero, separately rounds its
product and the subtraction to binary32, and narrows the result to half.
It is not exact mathematical remainder: quotient cancellation can change
finite results, negative zero becomes positive zero for a finite nonzero divisor,
and an infinite divisor produces NaN. Integer arithmetic helpers enforce the
selected rounding steps without target contraction or reciprocal approximation.
Scalar and two-, three- and four-component forms support scalar broadcasting
and single argument evaluation. Other floating types and source-owned overloads
retain their own behavior.

Source-defined ``asfloat``, ``asint`` and ``asuint`` overloads receive distinct
internal names before target generation. Their source calls retain namespace
and overload identity while generated arithmetic helpers retain builtin bitcasts.

The option is explicit, not inferred from the target OS or selected for every
Metal source. Project reports retain ``binary16RemainderProfile`` and reject
provenance that disagrees with the resolved options. Native controls compare
5,176 input pairs, ten result forms per pair, side-effect counters and exact
guards against a separately evaluated rounding model. Unchanged-original
Metal is a required macOS control; this does not promise that all Metal devices
or compiler modes implement the same profile. Half-vector component bitcasts
use their logical 16-bit width rather than the widened GLSL representation.

Metal normalization retains qualified math builtin ownership when source helpers
share the same name. Source overloads receive deterministic, collision-safe
internal names before namespace qualification is removed. Explicit global calls,
namespace-local helpers, imported names, materialized templates, and source-owned
extensions in ``metal`` remain distinguishable from the builtin overloads.
Existing overload diagnostics reject unresolved bindings instead of guessing.
Required bounded CI on Windows, Linux, and macOS compiles and executes a reduced
ownership fixture, preserving generated source, native modules, and readbacks;
macOS also executes the original Metal source for comparison. These focused
checks cover the regression in
`issue #1947 <https://github.com/CrossGL/crosstl/issues/1947>`_, not general C++
namespace conformance or whole-repository runtime integration.

Required CI partitions the family across five disjoint Ubuntu shards. Each
shard retranslates its exact entries, verifies deterministic identity,
materialization, ``CSMain`` workgroup metadata, and reflected ABI, then compiles
with pinned DXC using ``-enable-16bit-types -WX -T cs_6_2 -E CSMain`` and
requires non-empty DXIL modules. Together, the DirectX and OpenGL contracts
close complete discovered-unary translation, reflection, and native compiler
coverage on both targets; they do not claim numerical execution, MLX host
runtime redirection, or MLX test-suite parity.

**Ordered loop updates.**

Metal and OpenGL emit comma-separated ``for`` updates individually in source
order, matching DirectX. Updates remain in the loop header so ``continue``
executes them and ``break`` skips them. Prefix/postfix increments, dependent
compound assignments, empty and single updates retain their loop-local scopes.
Required three-platform native tests check these effects and guarded outputs.
The pinned MLX ``gather_front<float, int, int, N>`` proof instantiates unchanged
upstream bodies for ``N`` equal to 1, 4 and 8. It checks negative and repeated
indices, empty slices, partial chunks and exact binary32 storage words, including
NaN payloads and signed zeros. The source wrapper only includes upstream headers
and declares template instantiations. This is kernel execution coverage, not
MLX Gather host integration or complete indexing support.

**Mixed-width integer arithmetic.**

The Metal frontend retains structured-buffer compound assignments as lvalues in
the intermediate representation, including side-effecting indices. It does not
expand them into a load/store pair that evaluates the index twice.

For built-in integer vector/scalar arithmetic and comparisons, the Metal
frontend explicitly converts the scalar to the vector element type before the
operation. Both operand orders, compound assignments and inferred local types
retain this source rule in saved CrossGL. Scalar pairs still use their usual
integer conversions, and shifts still promote their operands independently.
This does not change standalone CrossGL or native HLSL conversion rules.

Metal-to-HLSL lowering explicitly applies source integer conversions before arithmetic,
bitwise operations, comparisons and conditional selection. In particular,
``int64_t`` combined with ``uint`` uses signed 64-bit arithmetic, not HLSL's
implicit unsigned conversion. Vector/scalar pairs retain the vector element
type, and shifts preserve independent operand promotions, including compound
shifts with scalar counts and vector destinations. Compound assignments
convert before narrowing back to the destination. Before an indexed destination
is passed to a typed ``inout`` helper, effectful indices are captured in private
temporaries: DXC may otherwise evaluate a private-array index on both copy-in
and copy-out. Captures remain in the assignment expression, not at function
entry, preserving conditional selection, loop updates and returned values.
Nested arrays and structure-member arrays retain their destination identity.
A right operand that may modify a copied
destination receives a diagnostic rather than an unproven copy-in/copy-out
translation.

Native HLSL input retains its DXC integer conversion rules instead: the frontend
records explicit common operand types in CrossGL, including unsigned preference
across widths and scalar/vector conversions. Nested expressions, conditional
arms, aliases, uniquely resolved function results and compound assignments carry
those types through a saved intermediate file. Integer literal suffixes and
canonical 64-bit vector types are retained. Ambiguous wide return types and
unproven compound-assignment copy-in aliases remain diagnostic.

OpenGL signed remainder uses truncating division and multiplication rather than
relying on ``%`` for negative operands. Typed helpers evaluate operands once and
broadcast scalar operands to the result vector shape. Constant initializers
retain constant expressions. Unsupported side-effecting compound destinations
remain diagnostic. Zero divisors and signed-minimum divided by minus one are
outside the source-defined numerical contract.

OpenGL shifts convert 64-bit counts to matching signed or unsigned 32-bit scalar
or vector operands for native driver compatibility. This does not change the
promoted left operand or the result type. Compound shifts retain native indexed
lvalues and evaluate both the index and count once. Negative counts and counts
at least as large as the promoted left operand's bit width are outside the
source-defined contract; this lowering does not assign them portable semantics.

Required native CI checks scalar mixed-width arithmetic and scalar/vector signed
remainder on Windows, Linux and macOS, including original Metal controls,
unchanged inputs and guarded outputs. Wide shift counts cover both directions,
32-bit and 64-bit left operands, vector widths two through four, scalar
broadcasts and side-effecting binary and compound expressions on every target.
The original Metal shift controls use representable, nonnegative signed left-shift
operands and valid counts throughout. Vector/scalar arithmetic controls include
values outside the vector component range, both operand orders, comparisons,
inferred locals, nested operations and single-evaluation compound assignments.
The original Metal mixed-sign comparison controls disable only ``-Wsign-compare``
and record that flag; generated artifacts retain warning-fatal compilation.
Native HLSL controls additionally compare generated results on each target and
the original HLSL on Windows; optimized DXC checks cover direct and saved-CrossGL
round trips.
Twelve additional original/generated Metal and required Windows cases cover
eight compound operators on private arrays, nested arrays and structure-member
arrays in statement, expression, conditional and loop-update positions. Each
case checks modified elements, unchanged neighbors, evaluation counts, the
assignment result, input preservation and output guards. OpenGL retains an
explicit diagnostic for these unsupported effectful compound destinations.
These checks do not establish full MLX-suite parity.

**Precise scalar promotions.**

Metal's qualified precise sine, cosine, arctangent and inverse hyperbolic
cosine accept scalar ``half`` and ``bfloat`` operands by promotion to
``float``. Translation preserves that binary32 result, including inferred
``auto`` declarations, and evaluates each operand once. Narrow vectors still
require an explicit conversion to the corresponding float vector, matching
the source compiler's overload rules.

Native checks compare implicit and explicit promotions, output guards and
evaluation counts. Generated results retain the existing binary32 accuracy
and signed-zero requirements. The original Metal arctangent control uses its
separate documented zero/subnormal policy; that allowance does not apply to
translated output. This contract does not change unqualified half-math
overloads or establish full application parity.

**Binary32 negation.**

DirectX lowers unary minus on known ``float`` scalars and two- to four-lane
vectors by toggling the payload sign bit. This preserves subnormals, signed
zeros, infinities and NaN payloads without depending on the compiler's
floating-point denormal mode. Operand evaluation occurs once. Literal
constants retain constant-expression syntax; integer, narrow-float, double,
matrix and complex operations retain their separate lowering rules.

The existing native math job also checks scalar and vector negation, aliases,
structure members, array indexing and side-effecting helper calls against an
independent integer oracle. It retains raw input, expected and readback words,
compiler outputs, module hashes, evaluation counts and guards. This adds no
runner matrix and does not change the tolerance of the precise math checks.
Nested signs retain their grouping in Metal and HLSL instead of combining
into increment or decrement tokens. The same native job checks nested signs
on integer and binary32 scalars and vectors, genuine prefix/postfix updates,
numeric casts and side-effecting calls. It compares both results and final
operand values with independent word-level expectations; on macOS it also
executes the unchanged source as a control. OpenGL retains its existing
parenthesized expressions. These checks do not establish full application
parity or change the separate narrow-float and complex lowering rules.

**Fused arithmetic profiles.**

``crosstl.translator.fused_math`` provides an internal binary32 fused
multiply-add helper expressed with pairs of 32-bit unsigned words. It rounds
the exact product plus addend once, to nearest with ties to even, without
requiring 64-bit integers or double precision. It supports gradual underflow
or explicit signed-zero flushing of subnormal inputs and results before
rounding. NaNs are canonicalized; payloads and floating-point exception flags
are not represented. Generated artifacts retain the Berkeley SoftFloat license
for the adapted arithmetic.

Required Windows/DirectX, Linux/OpenGL and macOS/Metal checks execute both
policies against an independent integer oracle. Evidence includes the input
bits, expected bits, readbacks, generated source, native modules and runtime
diagnostics. The macOS source control separately compares default and precise
Metal ``fma`` against the flush-before-rounding policy. This is a selected
profile check, not a claim that every Metal device or compilation mode uses
that policy: the Metal language permits different rounding and subnormal
behavior.

Set ``source_options={"binary32_fma_profile": "rne-flush"}`` in the single-file
API, or select a profile in the project configuration:

.. code-block:: toml

   [project.source_options.metal]
   binary32_fma_profile = "rne-flush"

``rne-flush`` selects nearest-even rounding with signed-zero flushing of
subnormal operands and results before rounding. ``rne-gradual`` selects
nearest-even rounding with gradual underflow. The option lowers resolved
binary32 ``fma`` and ``precise::fma`` calls for scalars and two- to four-lane
vectors. Arguments are evaluated once. Helpers have translation-unit-private
Metal linkage so independently translated modules can share a library.
User-defined overloads, explicit ``fast::fma`` and non-binary32 operations
retain their existing lowering; this option does not configure other arithmetic.

The default is unset and preserves existing artifact identities. The selected
profile is an explicit source-environment assumption, not inferred from the
destination backend. Verify it against the original source environment before
enabling it for a repository: the
`Metal numerical contract <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>`_
allows multiple rounding and subnormal behaviors. Round-toward-zero arithmetic
is not implemented by these profiles. Existing corpus identities must be
regenerated and compiler-validated when a profile changes emitted code.

Required native CI tests source-call lowering as well as the integer helper;
macOS additionally links independently translated modules. This does not by
itself establish MLX host integration or complete upstream-suite parity.
The pinned ``d9add9d`` native package gate translates the unchanged
``v_Erffloat32float32`` and ``v_Expm1float32float32`` entries with ``rne-flush``.
It checks the upstream numerical regression inputs and a dense neighborhood
of zero against independent references, including signed zero, normal/subnormal
boundaries and Expm1 overflow. No upstream kernel is edited; this gate is
separate from running the upstream Python tests through an adapted MLX host.
`Issue #1962 <https://github.com/CrossGL/crosstl/issues/1962>`_ remains the
integration tracker for the MLX Expm1/Erf failures.

**Division arithmetic support.**

``crosstl.translator.division_math`` provides an internal binary32 division
helper using only 32-bit integer arithmetic. It rounds the exact quotient
once, to nearest with ties to even. It supports gradual underflow or signed-zero
flushing of subnormal operands and results before rounding, including values
just below the normal boundary that would otherwise round up to a normal value.
NaNs are canonicalized; payloads and exception flags are not represented.

The existing native arithmetic jobs compile and execute both policies on
DirectX, OpenGL and Metal against an independent exact-rational oracle.
Tests cover every exponent, signed zeros, subnormal and overflow boundaries,
infinities, NaNs, randomized operands and eight output guard words. Evidence
includes input and expected words, readbacks, generated source, native modules
and runtime diagnostics. The macOS control also executes unchanged Metal
division with ``-std=metal3.1 -fno-fast-math`` against the flush policy.

Select ``source_options={"binary32_division_profile": "rne-flush"}`` in the
single-file API, or use the project configuration:

.. code-block:: toml

   [project.source_options.metal]
   binary32_division_profile = "rne-flush"

``rne-gradual`` preserves subnormal operands and results; ``rne-flush`` flushes
them before rounding. The default is unset. Resolved binary32 division,
reciprocal expressions such as ``1.0f / x``, and default/precise ``divide`` calls
use the selected helper. Scalar and vector operands are evaluated once; bfloat
results retain their narrowing before later arithmetic. User-defined operators,
explicit fast builtins and other arithmetic types keep their separate rules.
Compound division supports plain local storage and direct buffer elements;
effectful operands, unsupported reference destinations and global constant
division produce a structured diagnostic instead of silently changing the
selected policy. Explicitly sequence effectful compound operands first.

Project reports and runtime manifests retain ``binary32DivisionProfile`` in
artifact provenance. Report validation compares it with the resolved source,
target and path-specific configuration. Saved CrossGL includes the arithmetic
implementation, so later target generation does not need the source option.
Native source-expression checks cover aliases, scalar/vector operations,
members, compound results, evaluation counts and narrow intermediates. Finite
results and guards are bit-exact; NaNs are compared by classification only after
the separate bfloat conversion and in the original Metal control.

The source control establishes behavior only for its tested device and compiler
settings, not a universal Metal contract. The pinned project demo tests Sigmoid
division boundaries on all three native targets. A full current-pin OpenGL
sweep now resolves the three subnormal-division differences but still has one
exponential midpoint difference tracked in
`issue #2085 <https://github.com/CrossGL/crosstl/issues/2085>`_. This is not complete
Sigmoid or upstream-suite parity. The remaining source-profile work is tracked
in `issue #2086 <https://github.com/CrossGL/crosstl/issues/2086>`_.

Metal sources can also select an explicit binary32 comparison policy:

.. code-block:: toml

   [project.source_options.metal]
   binary32_comparison_profile = "flush-subnormals"

``preserve-subnormals`` compares the represented binary32 values;
``flush-subnormals`` treats subnormal operands as signed zero for the comparison.
The default is unset. The six relational and equality operators use integer-word
ordering to preserve the selected behavior on every target, including signed
zero, infinities and unordered NaNs. Scalar and two-, three- and four-lane
operations evaluate each operand once. Existing source conversions are retained,
including bfloat narrowing before its binary32 comparison.

This policy does not modify stored operands, Boolean conversions, arithmetic,
half-only or binary64 comparisons. Source-defined operators retain their own
bodies. Unresolved operands and global constant comparisons that would require
a runtime helper produce ``project.translate.metal-comparison-profile-unsupported``.
Reports and packages retain ``binary32ComparisonProfile``; validation requires
it to match the resolved source, target and path-specific configuration. Saved
CrossGL retains the helper implementation without requiring the option again.
Native source controls establish behavior for their tested compiler and device,
not a universal Metal policy. Remainder and conversion subnormal behavior remain
separate contracts; selecting a comparison profile does not establish full
project numerical parity.

Precise two-argument arctangent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Metal ``precise::atan2`` retains a portable implementation for binary32 scalars
and two- to four-lane vectors, including scalar-to-vector broadcasts. Scalar half
and bfloat arguments promote to binary32, as required by the precise overload.
Each argument is evaluated once;
source-defined functions and default/fast-mode calls keep their separate paths.
Unsupported operand types, shapes and global runtime initializers produce
``project.translate.metal-precise-math-unsupported`` diagnostics.

The implementation handles signed-zero axes, infinities and NaN classification
explicitly, using integer-word division and the existing precise arctangent
range reduction for finite operands. Native checks use an independent decimal
reference and the six-ULP binary32 limit in Table 8.1 of the
`Metal Shading Language Specification
<https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>`_.
Axis results, zero signs, operand copies, evaluation counts and guards are exact.
This accuracy contract is not a claim of bit-identical transcendental results.
At the normal/subnormal midpoint, the arctangent correction breaks a division
rounding tie toward the smaller magnitude. Integer significand checks preserve
that boundary before applying the underflow profile. Native regression inputs
cover all 127 applicable exponent scalings, adjacent operands and both signs;
neighboring normal results must not be flushed.

The default preserves represented subnormals. A characterized source execution
can select an explicit policy:

.. code-block:: toml

   [project.source_options.metal]
   binary32_atan2_profile = "flush-subnormals"

``preserve-subnormals`` preserves operands and gradual result underflow.
``flush-subnormals`` replaces subnormal operands with signed zero and flushes
finite result underflow to positive zero. Exact signed-zero axes are unchanged.
The latter matches the tested original Metal implementation; it is not a
universal device assumption. Neither policy modifies stored inputs, normalizes
readbacks, nor selects a policy for other arithmetic or default/fast intrinsics.

Reports and runtime manifests retain ``binary32Atan2Profile`` when explicitly
selected. Validation checks it against resolved target/path source options, and
saved CrossGL retains the selected implementation. Generated Metal and OpenGL
tests cover both policies; the unchanged Metal control checks the characterized
flush policy with the same numerical assertions. DirectX native execution is a
separate required CI gate.

Generated Metal scopes contraction and reassociation directives to each precise
function body. It does not change the compiler's contraction setting for later
helpers or entry points. Native source/roundtrip checks exercise arithmetic
across function-return boundaries with contraction disabled, expression-local,
and unrestricted, while retaining the precise helper's numerical checks.

Precise logarithms
~~~~~~~~~~~~~~~~~~

Metal ``precise::log`` uses a portable binary32 implementation for scalars and
two- to four-lane vectors. Scalar half and bfloat arguments promote to binary32;
arguments are evaluated once. Default and fast calls, and source-defined
functions, keep their existing paths. Unsupported operand types and global
runtime initializers produce ``project.translate.metal-precise-math-unsupported``.

Integer normalization avoids arithmetic on subnormal operands. Range reduction
around one and an odd-series expansion preserve accuracy near one without
depending on the target's native logarithm. Native regression checks use a
120-digit decimal reference and the four-ULP binary32 limit in Table 8.1 of the
`Metal Shading Language Specification
<https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>`_.
The tests cover every binary32 exponent, dense neighborhoods around one and
the reduction boundaries, signed inputs, infinities, NaNs and narrow promotion.
Classifications, zero signs, input copies, evaluation counts and guards are
checked separately from finite numerical accuracy.

The default preserves represented subnormal operands. A characterized source
can select an explicit policy:

.. code-block:: toml

   [project.source_options.metal]
   binary32_log_profile = "flush-subnormals"

``preserve-subnormals`` computes a finite logarithm for positive subnormals and
NaN for negative subnormals. ``flush-subnormals`` treats either sign of a
subnormal operand as signed zero, yielding negative infinity. Both policies
leave stored inputs unchanged. The unchanged Metal source control checks the
flush policy for its tested compiler and device, not a universal Metal rule.
Neither policy changes other arithmetic, default/fast logarithms or readbacks.

Reports and runtime manifests retain an explicit ``binary32LogProfile``;
validation requires it to match resolved target/path source options. Saved
CrossGL retains the implementation without requiring the option again. Native
DirectX, OpenGL and Metal checks are part of the existing project-demo jobs.
This is a logarithm contract, not proof of complete project numerical parity.

Precise square roots
~~~~~~~~~~~~~~~~~~~~

Metal ``precise::sqrt`` uses a correctly rounded binary32 helper for scalar and
two- to four-lane calls. Scalar half and bfloat arguments promote to binary32;
default/fast calls and source-owned functions keep their existing behavior.
Unsupported operands and global runtime initializers produce
``project.translate.metal-precise-math-unsupported``.

The helper normalizes the operand using integer bits and computes a 24-bit
square root with a remainder for nearest rounding. It does not depend on native
approximate square roots, subnormal arithmetic, double precision or 64-bit
integers. Signed zeros and infinities are exact; negative nonzero operands and
NaNs produce NaN. Native tests compare against an independent 120-digit decimal
reference without a finite ULP tolerance, including squared rounding midpoints,
all binary32 exponents and composed square roots.

The default preserves subnormal operands. A characterized source can select:

.. code-block:: toml

   [project.source_options.metal]
   binary32_sqrt_profile = "flush-subnormals"

``flush-subnormals`` treats a subnormal operand as a zero with the same sign.
``preserve-subnormals`` returns a nonzero result for positive subnormals and NaN
for negative subnormals. Neither option changes stored inputs, other arithmetic
or readbacks. The unchanged Metal control characterizes the explicit flush
policy for its tested compiler and device, not every Metal implementation.

Reports, runtime manifests and packages retain ``binary32SqrtProfile`` and
validate it against resolved target/path options. Saved CrossGL retains the
implementation. Required native checks share the existing project-demo runners;
the contract does not establish complete project numerical parity.

Precise reciprocal square roots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Metal ``precise::rsqrt`` and ``metal::precise::rsqrt`` use correctly rounded
binary32 helpers for scalar and two- to four-lane calls. Scalar half and bfloat
arguments promote to binary32. Default/fast calls and source-owned functions
are unchanged; unsupported operands and global runtime initializers produce
``project.translate.metal-precise-math-unsupported``.

The helper retains both the division remainder and the integer square-root
remainder until final nearest-even rounding. It uses only bounded 32-bit integer
arithmetic, without target reciprocal-root approximations or an intermediate
rounded square root. Native tests use an independent 120-digit decimal reference,
including reciprocal output midpoints and nested calls. Signed zeros produce
signed infinities; positive infinity produces positive zero; negative nonzero
operands and NaNs produce NaN.

The default preserves subnormal operands. A characterized source may select
``binary32_rsqrt_profile = "flush-subnormals"`` in its Metal source options to
treat subnormal operands as signed zero. ``preserve-subnormals`` remains an
explicit alternative. The policy is independent of ``binary32_sqrt_profile``
and does not alter stored operands or readbacks. Reports and runtime packages
retain ``binary32RsqrtProfile`` and validate it against resolved target/path
options; saved CrossGL retains the implementation. Native source controls
characterize the selected policy on the tested Metal compiler and device.

Power-function domains
~~~~~~~~~~~~~~~~~~~~~~

Metal default and precise ``pow`` calls preserve signed-zero, negative-base,
infinity and NaN domain behavior before evaluating a positive finite base.
Integral-exponent parity is determined from binary32
bits, including the boundary above which all represented integers are even.
This avoids target-dependent negative-base behavior and out-of-range integer
conversions. Scalar and two- to four-lane calls evaluate each argument once;
scalar broadcasts, half return narrowing and materialized standard-library
bfloat wrappers retain their conversion boundaries. An exponent of exactly one
returns the base without arithmetic, preserving finite values, subnormals,
signed zeros and infinities exactly. By default, other calls with a subnormal
operand keep their existing target-native path. Source-defined functions
and explicit fast-mode calls keep their separate implementations.

Characterized sources may select an operand policy:

.. code-block:: toml

   [project.source_options.metal]
   binary32_power_operand_profile = "flush-subnormals"

This treats binary32 subnormal operands as signed zero after the exact
exponent-one shortcut, before applying the domain rules. It does not change
stored operands, flush normal inputs or select an output-underflow policy.
The default remains unset; preservation on arbitrary target-native paths is
not offered as a separate profile. Reports, runtime manifests and packages
retain ``binary32PowerOperandProfile`` and validate it against resolved
target/path options. Native source controls characterize the tested Metal
compiler and device, not a universal Metal policy.

Saved CrossGL retains these helpers. Global initializers requiring the runtime
helper produce ``project.translate.metal-precise-math-unsupported``. Native
regressions compare domain results exactly and selected ordinary finite results
against an independent decimal reference, including unchanged Metal source
execution. Binary32 controls use the specification's 16-ULP power bound; selected
half controls use a one-storage-ULP regression bound. Separate identity controls
require exact results across all binary32 exponent classes, boundary
significands and deterministic sampled inputs, with original Metal execution,
evaluation counters and buffer guards. NaN payload identity is not claimed.
Applications that require portable finite-result accuracy can opt in separately:

.. code-block:: toml

   [project.source_options.metal]
   binary32_power_accuracy_profile = "portable-finite"

The default remains target-native, preserving existing Metal round-trip results.
The profile is retained as ``binary32PowerAccuracyProfile`` in reports, runtime
manifests and packages, and validated against resolved target/path options.
For ordinary finite results, its licensed fdlibm-derived approximation retains
separate high and low logarithm and exponent products. This avoids the large
errors caused by multiplying a rounded target logarithm by a large exponent
when the base is close to one. The same scalar helper serves all vector lanes
and narrowing boundaries. Near-one inputs and seeded general inputs are checked
against a high-precision decimal reference in native execution. It is an
accuracy contract, not bit-exact reproduction of a source device's approximation.
In particular, original Metal execution on the characterized device returns
infinity for ``pow(0x3f7fffff, 0xce7fffff)`` (binary32 operand words), although the
mathematical result is finite. Extended-exponent approximation tests use the
decimal reference; the native source controls are recorded separately.

Range-boundary behavior remains target-native: the approximation is used only
when its logarithmic result is strictly between -125 and 127. The surrounding
binades, overflow and underflow retain the existing intrinsic, rather than
guessing a source-device flushing rule. This is not an output-underflow profile
or a full-domain accuracy guarantee. Cross-target result-underflow parity and
complete project numerical parity remain unresolved.

Binary32 remainder
~~~~~~~~~~~~~~~~~~

Binary32 ``fmod`` has a separate, opt-in operation policy:

.. code-block:: toml

   [project.source_options.metal]
   binary32_remainder_profile = "flush-arithmetic-subnormals"

``preserve-subnormals`` computes exact truncating remainder for the represented
operands, including subnormal results. ``flush-arithmetic-subnormals`` treats a
subnormal divisor as zero and flushes a computed subnormal remainder to signed
zero. Its no-division path still preserves the numerator when its magnitude is
less than a valid divisor. This distinction rules out blanket input or output
flushing. Both policies preserve zero signs and canonicalize arithmetic NaNs.
Stored operands and surrounding comparisons are unchanged.

The default is unset. The integer-significand implementation supports scalar
and two-, three- and four-lane binary32 calls, evaluating each operand once.
Materialized Metal standard-library bfloat wrappers keep their float computation
and narrow return. User-defined overloads, explicit fast calls, half-only and
binary64 calls retain their existing behavior. Global constant calls that would
require a runtime helper produce
``project.translate.metal-remainder-profile-unsupported``. Reports, runtime
manifests and packages retain ``binary32RemainderProfile`` and validate it against
the resolved configuration; saved CrossGL retains the implementation.

Native controls characterize the tested source compiler and device only. This
option does not select a universal Metal policy, change Boolean conversions or
establish whole-project numerical parity. Half remainder and binary32 comparisons
remain independently configured contracts.

Binary32 addition and subtraction can also select an explicit policy:

.. code-block:: toml

   [project.source_options.metal]
   binary32_additive_profile = "rne-flush"

``rne-gradual`` preserves subnormal operands and results; ``rne-flush`` treats
subnormal operands as signed zero and flushes subnormal results to signed zero.
Both round once to nearest, ties to even, and canonicalize arithmetic NaNs. The
default is unset. The implementation shares the integer-word exact-rounding
support with fused arithmetic, but the two policies remain independent. It does
not replace separate source operations with a fused expression.

Scalar and vector ``+`` and ``-`` evaluate operands once, retain source
conversions and preserve bfloat result narrowing. Proven side-effect-free
``+=`` and ``-=`` support writable thread storage, direct buffer elements and
indexed vector components. Wide vectors use per-lane lowering. Unresolved
types, runtime helpers in global constants and unproven compound assignments
produce ``project.translate.metal-additive-profile-unsupported``. Source-defined
operators keep their dispatch; arithmetic inside their bodies follows the
selected policy. Integer, half-only, binary64 and pointer arithmetic, increment,
decrement, unary negation and stored operand words are unchanged.

Reports, runtime manifests and packages retain ``binary32AdditiveProfile`` and
validate it against resolved source, target and path-specific options. Saved
CrossGL retains the implementation. Native controls characterize only the
tested source compiler and device; selecting this profile alone does not prove
whole-project numerical parity or select comparison and remainder policies.

Binary32 multiplication has a separate, opt-in policy:

.. code-block:: toml

   [project.source_options.metal]
   binary32_multiplication_profile = "rne-flush"

``rne-gradual`` preserves subnormal operands and results. ``rne-flush`` treats
subnormal operands as signed zero and flushes results whose exact magnitude is
below the smallest normal value before rounding. Both round to nearest, ties
to even, preserve the product's zero sign, and canonicalize arithmetic NaNs.
Integer-word multiplication prevents target constant-folding rules from
discarding the sign or exceptional result of a literal zero product.

Scalar and vector ``*`` preserve source conversions and bfloat result narrowing;
wide vectors lower per lane. Proven side-effect-free ``*=`` supports writable
thread storage, direct buffer elements and indexed vector components. Unknown
types, global constant expressions requiring runtime helpers and unproven
compound assignments produce
``project.translate.metal-multiplication-profile-unsupported``. Source-defined
operators retain their ownership. Integer, half-only and binary64 arithmetic,
stored operand words and the other arithmetic profiles are unchanged.

Reports, manifests and runtime packages retain ``binary32MultiplicationProfile``
and validate it against resolved source, target and path-specific options.
Saved CrossGL retains the implementation. This profile introduces an explicit
rounding boundary at each selected product; it does not reproduce a source
compiler's optional contraction of products with surrounding additions or
subtractions. Qualify the policy against the source build and device before
using it for a project. It does not establish whole-project numerical parity.

Metal source contraction controls are not yet preserved across translation.
Active ``#pragma clang fp contract(...)``, ``#pragma STDC FP_CONTRACT`` and
``#pragma OPENCL FP_CONTRACT`` directives, including their ``_Pragma`` forms,
produce ``project.translate.metal-contraction-unsupported``. The check runs
after conditional and macro expansion but before template materialization,
so moving or pruning a template cannot silently erase its source policy.
Comments, string literals, inactive branches and unused macro definitions do
not select a policy. Keep affected source on its original backend until the
directive's lexical semantics can be represented. Neither ``-fno-fast-math``
nor an individual arithmetic profile is a substitute for contraction control.
This diagnostic does not establish equivalence for directive-free source or
implement source compiler flags, scoped lowering or contraction provenance.

The legacy MLX copy reference at
``846d176227a0ac13d2667e58d2bb68b322109ab0`` proves all 2,496 discovered entries
from ``copy.metal`` through Metal-to-CrossGL-to-Metal translation. The family
covers 30 shapes, 16 concrete templates, 13 input and output types, and all 169
conversion pairs. Its schema-v2 contract pins every artifact and shape ABI,
including source/default parameter provenance. The 2,496 artifacts contain
6,566 exact materializations and 8,684 reflected resources. Every artifact has
one exact ``cast_to`` materialization; only the 14 complex-to-Boolean entries
add the nested ``cast_to<bool, float>`` body, so materialization accounting is
explicitly data-dependent rather than a false fixed per-shape count.

The generic lowering preserves MLX's float and bfloat16 Boolean bit tests,
recovers native ``ushort`` width after frontend ``uint16_t`` normalization, and
projects registered ``complex64_t`` values only after validating the ordered
``real``/``imag`` float representation. Wrong registered shapes fail closed and
unregistered lookalikes remain untouched. Reflected interfaces contain three to
eight exact resources and retain host-owned ``[1, 1, 1]`` workgroup metadata.
Required Ubuntu CI verifies and exports 24 disjoint 104-entry source shards.
One dependent macOS job verifies the complete source bundle and compiles all 2,496 exact
artifacts with ``xcrun -sdk macosx metal -Werror -c``, requiring a non-empty AIR
object for each. Reports, source bundles and compiler results are retained.
This proves translation, reflection, and native compiler
acceptance; it does not claim Metal numerical execution, MLX host-runtime
redirection, or MLX test-suite parity.

The compiler-gated identity refresh retains source coverage, materializations
and resource ABI. Its generated-source changes are limited to helper linkage
qualifiers. ``artifactIdentityRefresh`` records this audit separately from the
historical ``proof`` metadata; it does not establish coverage of a newer MLX
revision or add numerical execution claims.

A subsequent complete copy review accepts byte conversions and typed vector
initialization: 1,760 artifacts are unchanged, and reversing only the reviewed
conversions and ``int2``/``long2`` initializer syntax recovers all 736 changed
bodies. All 2,496 freshly generated artifacts compile with warnings fatal;
materializations, source/default provenance and resource ABI remain unchanged.
Negative controls reject changed indexing, byte widths, address arithmetic and
loop bounds. Historical proof and linkage-refresh metadata are preserved.
This is reference and compiler evidence, not an additional numerical-runtime
claim. Other families remain under
`issue #1966 <https://github.com/CrossGL/crosstl/issues/1966>`_.

The same selected-entry pipeline translates all 2,496 copy entries at historical
revision ``846d176227a0ac13d2667e58d2bb68b322109ab0`` to standalone OpenGL
``main`` artifacts. The schema-v2
``copy.opengl-translation.json`` contract preserves all 30 shapes, 16 concrete
kernel templates, 13 input/output types, all 169 conversion pairs, 6,566 exact
materializations, and 8,684 reflected target resources. Scalar and vector forms
expose source and destination storage buffers plus an entry-scoped size block;
fixed generalized forms add scalar stride blocks or stride storage buffers;
rank-generic forms expose shape and stride buffers plus an entry-scoped rank
block; dynamic forms add exact source and destination offset blocks.

The reference review covers all 2,496 complete sources and target interfaces.
Narrow-conversion helpers account for 308 changed artifacts; 2,188 remain
byte-identical. All indexing and interfaces are preserved. Sixty-one fresh
translation checks cover every changed conversion pair and shape, in addition
to complete compiler validation. This does not extend the historical revision
coverage or establish full numerical parity.

The generic registered-structure contract recognizes both canonical
``complex64_t`` and its emitted ``complex_t_float`` representation. The 150
complex-to-scalar entries project the real field only after validating the exact
ordered ``real``/``imag`` float shape, evaluate the source once, and retain
narrow target conversion semantics. A malformed registered representation
fails closed rather than emitting an invalid GLSL constructor.

OpenGL's 32-bit index profile cannot implicitly preserve every source 64-bit
buffer index. The complete contract therefore declares explicit host/runtime
bounds of ``[0, 2147483647]`` for ``offset + i``, ``src_idx``, ``dst_idx``,
``dst_idx + i``, ``src_idx + src_offset``, ``dst_idx + dst_offset``, ``idx.x``,
and ``idx.y``. These are portability preconditions rather than inferred facts
or generated checks; absent proof continues to fail closed.

Required Linux CI partitions the family into 24 disjoint 104-entry shards. Each
shard retranslates its entries, verifies deterministic artifact identity,
materialization provenance, standalone ``main`` workgroup metadata, and exact
target reflection, compiles with
``glslangValidator --target-env opengl --target-env spirv1.3 -S comp``, validates
with ``spirv-val --target-env spv1.3``, and requires a non-empty SPIR-V module.
Together with the Metal contract this closes complete discovered-copy
translation, reflection, and native compiler coverage for those two targets; it
does not claim numerical execution, MLX host-runtime redirection, or MLX
test-suite parity.

The DirectX path translates the same 2,496 entries to standalone ``CSMain``
artifacts. Its compact schema-v2 ``copy.directx-translation.json`` contract
preserves all 30 shapes, 16 concrete templates, 13 input/output types, all 169
conversion pairs, and 6,566 exact materializations. Exact target reflection
contains 10,036 resources. The ``g2``, ``g2large``, ``g3``, ``g3large``,
``gn2``, ``gn4large``, ``s2``, and ``v2`` shapes additionally expose generated
``CrossGLDispatchInfo`` metadata at their exact shape-specific ``b0``, ``b6``,
or ``b3`` binding.

The shared registered-structure conversion contract admits emitted
``complex_t_float`` values only after validating their ordered ``real`` and
``imag`` float fields. Exactly 150 complex-to-scalar entries project the real
field, malformed registered representations fail closed with structured
project diagnostics, and unregistered lookalikes remain untouched. Unlike the
OpenGL lowering, this native HLSL path introduces no additional source-scoped
32-bit index-range portability promise.

Required Ubuntu CI partitions the DirectX family into 24 disjoint 104-entry
shards. Each shard retranslates its exact entries, verifies deterministic HLSL
identity, materialization provenance, ``CSMain`` workgroup metadata, and exact
reflected ABI, then compiles with checksum-pinned DXC using
``-enable-16bit-types -WX -T cs_6_2 -E CSMain`` and requires a non-empty DXIL
module. Together with the Metal and OpenGL contracts this closes complete
discovered-copy translation, reflection, and native compiler coverage on all
three targets; it does not claim numerical execution, MLX host-runtime
redirection, or MLX test-suite parity.

The reviewed HLSL copy refresh at historical source revision ``846d1762``
compares all 2,496 bodies and compiles each one with warnings fatal. The 1,820
changed bodies preserve byte conversions, explicit integer arithmetic and
half rounding. All 6,566 materializations and 10,036 resource bindings are
checked; half resources use two-byte ``uint16_t`` storage with explicit
binary16 metadata. A subsequent 28-entry correction replaces float32-mediated
wide-integer conversions with direct integer rounding, with complete body and
binding comparison and warning-fatal compilation. Four unchanged controls retain
their identities. The source total is 4,859,340 bytes. Required Windows numerical
execution is tracked separately in
`#2089 <https://github.com/CrossGL/crosstl/issues/2089>`_; original and generated
Metal controls cover 64,276 integer inputs without changing tolerances. Nested bfloat
constructors in Metal bitcasts have a separate type-inference rejection in
`#2090 <https://github.com/CrossGL/crosstl/issues/2090>`_. These remain explicit
limitations while the compiler contracts preserve complete entry coverage.

The current-pinned MLX binary integration independently proves all 4,122
discovered entries from ``binary.metal``. Fifteen 238-entry base shapes and
three 184-entry work-per-thread shapes span 18 shapes, 11 concrete kernel
templates, 24 operators, and 25 input/output type pairs. Each selected artifact
contains one operator implementation and one kernel, rejects residual templates,
``decltype``, call operators, unsupported placeholders, and non-selected
operator bodies, and preserves explicit or source-default template provenance.

Scalar, size-bounded vector, fixed-dimensional generalized, and rank-generic
forms reflect exact three-, four-, five-, or seven-resource interfaces. The
generalized forms additionally materialize only their reachable
``elem_to_loc_1``, ``elem_to_loc_2``, ``elem_to_loc_3``, or
``elem_to_loc_2_nd`` helper for the exact 32- or 64-bit index type. In total,
4,122 artifacts contain 6,026 exact materializations and 19,106 reflected
resources while preserving a host-owned ``[1, 1, 1]`` workgroup contract. The
238-entry scalar contract remains an exact subset of the complete contract.

The generic path maps concrete signed and unsigned 64-bit vectors to native
Metal vector types, rewrites dependent free operators only to already-emitted
exact helpers, and applies known non-explicit contextual aggregate constructors
without admitting explicit-only or ambiguous conversions. Bfloat minimum and
maximum calls promote their arguments to float before typed reconstruction;
discarded type constructors remain evaluated through unambiguous void casts;
and scalar Boolean relational expressions receive explicit C++ integral
promotion. Focused tests retain conservative rejection boundaries for each
contextual operation.

Template declaration scanning distinguishes comparison and shift operator names
from template argument delimiters. Dependent struct references inside those
operator bodies remain in their enclosing template scope until specialization;
they are not emitted as concrete types with unresolved local constants.

A schema-v2 hash-pinned contract records every identity, shape, template,
classification, byte count, materialization, and resource ABI. Required Ubuntu
CI verifies translation, bodies and reflection across 24 disjoint shards and
exports the exact pinned sources. One dependent macOS job rejects missing,
duplicate or changed entries before compilation. It compiles all 4,122 exact artifacts
with ``xcrun -sdk macosx metal -Werror -c`` and requires non-empty AIR without a
warning exemption. Both phases retain failure evidence; the local round-trip
test still translates and compiles together. This closes discovered selected-entry translation,
reflection, and native compiler coverage. It does not claim Metal numerical
execution, host-runtime redirection, or MLX test-suite parity.

The same selected-entry pipeline translates all 4,122 binary entries to
standalone OpenGL ``main`` artifacts. The schema-v2
``binary.opengl-translation.json`` contract preserves all 18 shapes, 11 kernel
templates, 24 operators, 25 input/output type pairs, and 6,026 exact
materializations. It pins 16,276,504 generated GLSL bytes
and 19,106 reflected resources across exact three-, four-, five-, and
seven-resource target interfaces. Scalar forms expose three storage buffers;
size-bounded forms add an entry-scoped size uniform block; fixed generalized
forms expose scalar stride blocks or stride storage buffers; and rank-generic
forms expose shape and stride buffers plus an entry-scoped rank block.

OpenGL's 32-bit index profile cannot implicitly preserve the source's runtime
64-bit buffer indices. The complete contract therefore declares host/runtime
bounds of ``[0, 2147483647]`` for ``offset + i``, ``a_idx``, ``b_idx``,
``out_idx``, ``out_idx++``, ``idx.x``, and ``idx.y``. These are explicit
portability preconditions, not inferred facts or generated runtime checks;
unproven wide-index translation remains fail-closed.

Required Linux CI partitions the family into 24 disjoint shards. Each shard
retranslates its exact entries, checks deterministic identity, materialization
provenance, standalone ``main`` workgroup metadata, and target reflection,
compiles every artifact with
``glslangValidator --target-env opengl --target-env spirv1.3 -S comp``, validates
it with ``spirv-val --target-env spv1.3``, and requires a non-empty SPIR-V
module. This closes complete discovered-binary OpenGL translation, reflection,
and native compiler coverage; it does not claim numerical execution, MLX
host-runtime redirection, or MLX test-suite parity.

The same selected-entry pipeline translates all 4,122 binary entries to
standalone DirectX ``CSMain`` artifacts. The schema-v2
``binary.directx-translation.json`` contract preserves all 18 shapes, 11
kernel templates, 24 operators, 25 input/output type pairs, 6,026 exact
materializations, and the seven explicit index-range portability preconditions.
DirectX resource namespaces preserve source buffer coordinates. The nine shapes
that consume ``threads_per_grid`` add explicit ``CrossGLDispatchInfo``
reflection, producing 21,248 resources across exact three- through
eight-resource target interfaces.

Computed-result bfloat ``ArcTan2``, ``LogAddExp``, and ``Power`` paths expand
both operands to float, compute in float, and reconstruct the result with exact
round-to-nearest-ties-to-even. ``Maximum`` and ``Minimum`` also expand for the
comparison, but preserve and return the selected original low-16-bit bfloat
payload without requantization. Unproven bfloat builtins remain fail-closed,
and native 16-bit storage remains a Shader Model 6.2 requirement.

Required Ubuntu CI partitions the family into 24 disjoint shards. Each shard
retranslates its exact entries, checks deterministic identity, materialization
provenance, standalone ``CSMain`` workgroup metadata, and target reflection,
then compiles every artifact with checksum-pinned DXC using
``-enable-16bit-types -WX -T cs_6_2 -E CSMain`` and requires a non-empty DXIL
module. This closes complete discovered-binary DirectX translation, reflection,
and native compiler coverage; it does not claim numerical execution, MLX
host-runtime redirection, or MLX test-suite parity.

The binary HLSL identity review compares every historical/current body and all
21,248 reflected resources, with 8,244 strict compiler checks across both sets.
It accounts for source-width arithmetic and two-byte integer storage carrying
IEEE binary16 values without changing bindings, dispatch contracts, loop bounds,
source pins or specialization counts. The 2,435 changed artifacts and 1,687
byte-identical artifacts remain compiler evidence, not full-family numerical
execution evidence.

The current-pinned MLX reduction integration independently proves all 2,396
host-named entries from ``reduce.metal`` through Metal-to-CrossGL-to-Metal
translation. Three base forms plus six multidimensional families across two
index widths and three dimension counts produce 39 exact ABI shapes over nine
kernel templates. The family spans six operator families, 44 concrete operator
types, and 13 input/output types. Its compact schema-v2 contract stores only
per-entry classification, artifact identity and size, exact materialization
identity/count, and exact resource identity/count; transient proof rows,
translation reports, and AIR objects are not tracked.

The complete reflected interface contains 25,088 resources across exact one-
through twelve-resource ABIs and retains the host-owned ``[1, 1, 1]`` workgroup
contract. Comparison-aware template parsing preserves non-type expressions such
as ``(1 > 2)``; proven integral non-type arguments are folded for overload
selection without changing source-spelled specialization identity. Visible
aliases and raw pointer base spellings normalize only to proven concrete types,
and native pointer-plus/minus-integral expressions bypass aggregate free-
operator dispatch only when built-in semantics are type-proven. Concrete-helper
deduplication ignores comments and external whitespace while retaining exact
preprocessing-token literal bytes. Compile-time static members used by
references, address-taking, receivers, or unevaluated storage-sensitive
expressions are hoisted into address-space-correct Metal ``constant`` storage
instead of being substituted as rvalues. Ambiguous aliases, unresolved address
provenance, non-integral offsets, and incompatible overloads remain fail-closed.

Required Ubuntu CI uses 24 disjoint source shards: 20 contain 100 entries and four
contain 99. Every shard verifies deterministic identity, materialization,
workgroup metadata, and reflected ABI before exporting its checked sources.
One dependent macOS job verifies complete contract coverage and every source
identity before warning-fatal compilation with
``xcrun -sdk macosx metal -Werror -c``. All 2,396 AIR objects must be non-empty.
Missing, duplicated or altered sources fail before compilation; both phases
retain evidence. The default local test still translates and compiles each entry.
This closes complete discovered-reduce translation, reflection, and native
compiler coverage; it does not claim Metal numerical execution, DirectX or
OpenGL whole-family coverage, MLX host-runtime redirection, or MLX test-suite
parity.

The same selected-entry pipeline translates all 2,396 reduction entries to
standalone OpenGL ``main`` artifacts. The compact schema-v2
``reduce.opengl-translation.json`` contract preserves the 39 shapes, nine
kernel templates, six operator families, 44 concrete operator types, and 13
input/output types while pinning 9,216 exact materializations,
32,115,641 generated GLSL bytes, and 25,088 resources across
scalar-layout-aware one- through twelve-resource target ABIs.
All source bodies and resource interfaces have complete reference comparisons;
85 fresh pipeline checks cover every reviewed edit pattern and ABI shape.
The 156 complex-buffer metadata updates describe existing storage layouts,
not new resource declarations. These checks establish deterministic translation
and strict compilation, not whole-family numerical execution.

OpenGL's signed 32-bit logical buffer offsets require six explicit
source-expression portability preconditions: the one-, two-, and
five-dimensional ``LoopedElemToLoc`` source pointer expressions,
``inputs[i - 1] + reduction_size``, the long-column output expression, and
``out_idx``. Each is host-bounded to ``[0, 2147483647]``. These bounds are not
inferred and no runtime check is generated; missing, mismatched, negative, or
over-wide assertions remain fail-closed. Bounded unit-step loop-carried
pointer-array initialization is evaluated one exact iteration at a time. A
non-singleton per-invocation dynamic target remains a may-write and cannot
establish full-array definite assignment, while a shadowed loop-index binding
disables the serial-loop proof and remains fail-closed. For-in range and
scalar-count expressions are rendered in the outer lexical environment;
Fixed-array iterable expressions and compile-time extents resolve before
same-named pattern bindings enter scope. same-named patterns use independent
controllers and shadow pointer,
stage-builtin, and flattened stage-struct aliases in the loop body. For-in
resource specialization resolves overloads from exact lexical pattern types.
Dynamic for-in resource specialization resolves overloads from exact call-site
lexical types. Null storage-pointer reachability and elision preserve exact
lexical declaration identity. Null workgroup-pointer reachability and
elision preserve exact lexical declaration identity. Nested
resource-specialization discovery preserves deterministic lexical order across Python
hash seeds. Workgroup-pointer bounds analysis visits every control-flow
expression and preserves outer mutations across lexical blocks. loop-local
fixed-array
storage retains flattened
stage-input
struct declarations, and
fixed-array patterns cannot inherit same-named outer scalar or vector-component
bounds. Repeated loop bounds and generated loop controllers lose their proven intervals when
exact scalar or vector-component dependencies can mutate. logical-offset
mutation through resolved nested helpers is written back to callers. Direct
element and addressed one-element forwarding through overload-resolved scalar
or fixed-array helpers preserves that writeback;
private scalar address views remain confined to their declaring lexical scopes;
nonlocal scalar address views remain fail-closed;
dynamic elements and unresolved, ambiguous, or recursive forwarding remain
fail-closed. Fixed-array storage cannot escape through local private pointer or
reference aliases; lexically shadowed arrays and condition-only reads are not
misattributed, while residual private pointer or reference syntax remains
fail-closed.

Required Linux CI partitions the family into 24 disjoint shards: 20 contain
100 entries and four contain 99. Every shard verifies deterministic artifact,
materialization, workgroup, and scalar-layout-aware reflected ABI identity,
compiles with
``glslangValidator --target-env opengl --target-env spirv1.3 -S comp``, validates
with ``spirv-val --target-env spv1.3``, and requires a non-empty SPIR-V module
for every entry. This closes complete discovered-reduce OpenGL translation,
reflection, and native compiler coverage; it does not claim numerical
execution, DirectX whole-family translation, MLX host-runtime redirection, or
MLX test-suite parity.

The DirectX path translates the same 2,396 reduction entries to standalone
``CSMain`` artifacts. Its compact schema-v2
``reduce.directx-translation.json`` contract preserves all 39 shapes, nine
kernel templates, six operator families, 44 concrete operator types, 13
input/output types, and 9,216 exact materializations. Every entry pins both its
HLSL source and non-empty DXIL identity, byte count, materialization digest, and
resource digest. Exact target reflection contains 27,382 resources across
one- through thirteen-resource interfaces. The 37 shapes other than ``init``
and ``all`` additionally expose generated ``CrossGLDispatchInfo`` metadata;
the source data, shape, stride, rank, and reduction constants retain their
shape-specific coordinates and exact HLSL storage types.

The shared DirectX lowering emits a portable quiet binary32 value for
unshadowed ``NAN``, preserves explicit componentwise complex SIMD shuffle
overloads, and converts only proven direct void tail recursion to an iterative
loop. Device and constant pointer arrays lower to bounded resource offsets with
transitive whole-array and element writeback only after exact lexical overload,
alias, and backing-object provenance resolution. Logical bfloat pointee types
remain distinct from their physical ``uint16_t`` storage, and postfix
pointer-dereference typing removes exactly one pointer layer. Unresolved
aggregate pointer members or arrays, unsafe recursion, ambiguous pointer-array
aliasing or reinterpretation, and unsupported HIP record lifecycle or layout
remain fail-closed. Nested and function-local anonymous records are
materialized deterministically when their complete layout is representable.
Unlike the OpenGL lowering, the native HLSL path introduces no additional
source-scoped 32-bit index-range portability promise.

The pre-release native proof preserves one HLSL and one DXIL artifact for each
entry and independently recompiles all 2,396 HLSL artifacts with
checksum-pinned DXC, requiring byte-identical non-empty DXIL before accepting
the compact contract. Required Ubuntu CI partitions the contract into 24
disjoint shards: 20 contain 100 entries and four contain 99. Each shard
retranslates its exact entries, verifies deterministic HLSL identity,
materialization provenance, ``CSMain`` workgroup metadata, and exact reflected
ABI, then compiles with ``-enable-16bit-types -WX -T cs_6_2 -E CSMain`` and
requires a non-empty DXIL module. Together with the Metal and OpenGL contracts
this closes complete discovered-reduce translation, reflection, and native
compiler coverage on all three targets; it does not claim numerical execution,
MLX host-runtime redirection, or MLX test-suite parity.

OpenGL Pointer and Matrix Policies
----------------------------------

Private pointer helpers lower to fixed array parameters with separate element
offsets. Bounded cumulative pointer updates retain those offsets; proven-zero,
side-effect-free updates may be omitted only when their result is discarded.
Unknown rebasing, escaping pointer values, and unsupported aliasing remain
translation errors.

Two optional target policies are available through project source options:

.. code-block:: toml

   [project.source_options.metal.target_options.opengl]
   cooperative_matrix_software_lowering = true
   private_pointer_out_of_bounds_read = "error"

``cooperative_matrix_software_lowering`` defaults to ``false``. When enabled,
supported fragment operations lower to explicit scalar storage and subgroup
operations. Multiply-accumulate currently requires the float 8-by-8,
32-lane, two-elements-per-lane ``tile_4x4_row_pair`` contract, matching operand
dimensions, and an exact subgroup-width contract. Unsupported mappings are
rejected. This option does not remove the hardware subgroup requirement or
establish numerical equivalence for an entire kernel.

``private_pointer_out_of_bounds_read`` defaults to ``"error"``. An explicit
``"zero"`` policy guards eligible reads through const private pointers backed
by complete, statically sized local arrays, returning a typed zero outside the
array. It does not permit out-of-bounds writes or unresolved array slices.
The selected policy is retained in project source options; recovered artifacts
also contain ``CROSSTL_PRIVATE_POINTER_OOB_READ_ZERO``. This is an explicit
behavior choice for an otherwise invalid source access, not evidence that the
result matches the source runtime. Numerical validation remains necessary.

Metal sources may also opt into ``promote_derived_pointer_members = true`` under
``project.source_options.metal`` to separate supported derived pointer members
into backing objects and offsets during materialization. Unknown origins and
unsupported pointer mutations remain errors rather than guessed bindings.

The pinned MLX quantized-wide OpenGL gate covers 18 selected entries with strict
pointer bounds and no upstream edits. It binds the pinned host dispatch to its
source-faithful ``[32, 2, 1]`` workgroup for both two- and four-vector kernels:
two logical 32-lane software subgroups and 64 total invocations. The gate
validates source identities, reports, array preservation, logical subgroup
indexing, exact execution metadata, GLSL compilation, and SPIR-V modules on
Linux without claiming KHR hardware-subgroup metadata. It does not claim full
quantized coverage or MLX runtime integration.

OpenGL Software Subgroup Specialization
----------------------------------------

Metal compute sources that require fixed 32-lane SIMD semantics can opt into
a bounded OpenGL shared-memory lowering when the deployment device does not
provide the KHR subgroup extensions:

.. code-block:: toml

   [project.entry_points]
   "kernels/quantized.metal" = "affine_quantize_float_gs_32_b_2"

   [project.entry_workgroup_size_rules."kernels/quantized.metal"]
   affine_quantize_float_gs_32_b_2 = [32, 1, 1]

   [project.source_options.metal.target_options.opengl]
   software_subgroup_width = 32

This option is target-scoped and explicit; it does not change Metal parsing or
other target artifacts. The only accepted width is ``32``. The selected output
must contain exactly one compute entry with concrete positive local dimensions
and no more than 1,024 total invocations.
CrossTL partitions the linear invocation range into an exact compile-time count
of independent 32-lane software subgroups. Subgroup count, subgroup index,
subgroup width, and lane index lower respectively to the workgroup invocation
count divided by 32 and rounded up, ``gl_LocalInvocationIndex / 32``,
``CROSSTL_SOFTWARE_SUBGROUP_WIDTH``, and
``gl_LocalInvocationIndex % 32``. The one-subgroup case retains the simpler
``1u``, ``0u``, and ``gl_LocalInvocationIndex`` forms.

A fixed workgroup may end in a subgroup with fewer than 32 real invocations.
Reductions combine only those invocations, scratch storage retains the exact
workgroup size, and shuffle-down helpers use the calling value when their source
falls beyond the real lanes. Source shuffle results from inactive lanes are not
portable; numerical controls select only results from active source lanes.
Relative offsets are checked before adding the lane index, so unsigned overflow
cannot wrap an out-of-range shuffle into another lane's value.
Metal shuffle and broadcast lane parameters retain their source ``ushort``
conversion before target lowering, including arguments passed through helpers.
Canonical 32-bit wave arguments retain their own width. Unqualified source wave
calls respect namespace visibility, including local ``using namespace``
directives, before builtin argument conversion is applied.
Writes through an indexed destination do not invalidate the index's uniformity;
actual index mutations and divergent exits still prevent software lowering.
All real invocations must still reach the same workgroup barriers. This does not
provide dynamic nonuniform final dispatch groups: the runtime must preserve the
selected workgroup shape rather than round a source thread grid up silently.

Native loader requests can express an exact source thread grid with
``threadGridSize``, in addition to ``workgroupSize`` and ``workgroupCount``.
The group count must equal the component-wise ceiling of the thread grid divided
by the workgroup size. For example, ``threadGridSize = [1025, 1, 1]`` with
``workgroupSize = [1024, 1, 1]`` requires ``workgroupCount = [2, 1, 1]``.
When supplied, ``globalSize`` and ``gridSize`` must describe the exact thread
grid, not its padded bounds. Omitting ``threadGridSize`` retains full-group
dispatch and its existing dimension checks.

The Metal runtime implements this contract through native ``dispatchThreads``
after checking device support and grid limits. Original and generated Metal
controls check partial groups, source coordinates, subgroup sums and buffer
guards across one-, two- and three-dimensional grids. DirectX and OpenGL reject
this explicit contract with ``exact-thread-grid-unsupported`` until their
lowering can preserve partial-group identities and active lanes. This contract
does not yet enable the MLX small-row host plan on those targets.

The translator's ``plan_dispatch_regions`` and ``specialize_dispatch_region``
APIs provide the lowering needed for portable exact grids. The planner partitions
a grid into at most eight rectangular regions with uniform active workgroup
shapes. Specialization preserves source global IDs, workgroup IDs, grid extents
and group counts while leaving local IDs and subgroup operations tied to each
region's actual group shape. It operates on an independent, single-entry AST;
source kernels are not patched. Per-invocation coordinate captures remain private
to each invocation, including when read by helper functions.

Native tests execute these region artifacts through one shared-allocation
dispatch sequence on DirectX and OpenGL, with original and roundtrip Metal
controls. The caller must compile each specialization, submit its exact physical
counts and share allocations across the complete plan.

Project translation accepts a target-scoped ``dispatch_region`` source option
for DirectX and OpenGL. Its fields match ``DispatchRegion.to_json()``; the
configured ``workgroup_size`` must match the region's physical ``workgroupSize``.
For the final five invocations of a 37-thread grid with nominal width 32:

.. code-block:: toml

   [project]
   workgroup_size = [5, 1, 1]

   [project.source_options.metal.target_options.opengl.dispatch_region]
   threadGridSize = [37, 1, 1]
   sourceWorkgroupSize = [32, 1, 1]
   workgroupOffset = [1, 0, 0]
   workgroupCount = [1, 1, 1]
   workgroupSize = [5, 1, 1]

The region is lowered after entry selection and retained in artifact, package
and native-loader provenance as ``dispatchRegion``. The loader checks its
physical size against the reflected entry and rejects a different launch count
or a second thread-grid mapping. Each region needs its own output directory or
otherwise distinct artifact identity; a grid or offset change changes the
specialized program, even when the physical workgroup shape stays the same.

Region packages also record ``dispatchRegionProgram``: the selected source
entry, resolved CrossGL program hash (including included definitions), target,
lowering settings and installed translation-code identity. This is a consistency
record, not a cryptographic signature or a proof of numerical equivalence.

``select_native_loader_dispatch_regions`` accepts descriptor/package-root pairs
and orders the complete canonical plan for a requested exact grid and nominal
workgroup size. Missing, duplicate, unrelated or mixed-program regions are
rejected. Older packages without the program identity can still be loaded
individually, but cannot participate in automatic region selection.

``prepare_native_loader_dispatch_regions`` additionally verifies every artifact,
preflights its bindings and compiles through the native adapter before yielding
requests for ``dispatch_sequence``. Source allocations are shared, with uploads
only on the first region; derived launch uniforms remain region-local. Use the
context manager for the whole dispatch so compiled modules remain alive and are
cleaned up on success or failure.

On-demand specialization caching and MLX host dispatch integration are not yet
wired to these APIs. Single-request DirectX/OpenGL ``threadGridSize`` rejection
remains in place; use the complete region preparation API for an exact grid,
not an ordinary rounded dispatch.

The bounded mode supports scalar ``float``, ``int``, or ``uint`` sum, minimum,
maximum, and shuffle-down operations. Shared scratch spans the complete
workgroup, while every helper derives a subgroup-local lane and base so reads
and reductions cannot cross a 32-lane boundary. Subgroup operations normally
must execute in workgroup-uniform control flow. They may appear directly in the
entry or in a uniquely identified helper only when every helper call is direct,
unconditional, top-level, and owned by that sole entry. Compile-time branches,
canonical constant loops, and canonical runtime loops whose integer bounds are
proven workgroup-uniform remain valid.

One narrow lane-dependent reduction form is also supported: a direct
``WaveActiveSum``, ``WaveActiveMin``, or ``WaveActiveMax`` assignment in a
top-level entry-owned ``if`` with no ``else``. The condition must be
side-effect-free, the branch must contain exactly that one subgroup operation,
declarations and escaping control flow cannot precede it, and the payload and
target must be matching 32-bit numeric scalars. CrossTL evaluates the branch
prefix only for active lanes, contributes a typed identity from inactive lanes,
invokes the barriered subgroup helper uniformly across the workgroup, and
exposes the result only to active lanes. Sum uses zero. Minimum uses positive
infinity, ``INT_MAX``, or ``UINT_MAX``; maximum uses negative infinity,
``INT_MIN``, or unsigned zero. Conditional shuffle, nested or multi-operation
branches, and other unproven shapes continue to fail closed.

Direct source references to raw ``gl_Subgroup*`` or ``subgroup*`` builtins,
empty operation sets, unsupported payloads or operations, and unresolved helper
ownership also fail closed. Successful lowering emits barriered ``shared``
memory helpers and the marker ``CROSSTL_SOFTWARE_SUBGROUP_WIDTH``. It
deliberately emits no ``GL_KHR_shader_subgroup*`` extension,
``gl_Subgroup*`` use, ``CROSSTL_REQUIRED_SUBGROUP_WIDTH`` marker, or hardware
``subgroupWidth`` execution metadata. This keeps the software execution
contract distinct from the default hardware-subgroup path and its host
preflight. When the option is absent, KHR subgroup generation and exact-width
enforcement are unchanged.

The lowering establishes shader semantics only for this constrained contract.
It does not infer dispatch counts, prove arbitrary divergent control flow,
rewrite an application's loader, or claim parity for other entries. Invalid
requests fail with
``project.translate.opengl-software-subgroup-invalid`` and include the width,
workgroup size, operation, and reason available at the rejection site.

Project Index-Range Assertions
------------------------------

Some source index types cannot be represented directly by a target's legal
scalar index types. When the application already constrains an index at the
host or runtime boundary, record that precondition in ``crosstl.toml``:

.. code-block:: toml

   [[project.index_range_assertions]]
   source = "kernels/*.metal"
   function = "gather_values"
   expression = "element_index"
   minimum = 0
   maximum = 1023

Each assertion table has these fields:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``source``
     - Repository-relative source glob. The assertion is considered only for
       matching translation units; omitting it defaults to ``*``.
   * - ``function``
     - Optional exact source function name. When omitted, the assertion can
       apply in any function containing the matching expression.
   * - ``expression``
     - Source index expression covered by the assertion. Matching ignores
       whitespace but otherwise preserves the expression identity.
   * - ``minimum``
     - Inclusive integer lower bound for the expression.
   * - ``maximum``
     - Inclusive integer upper bound for the expression. It must not be less
       than ``minimum``.

Index-range assertions are explicit host/runtime portability preconditions.
CrossGL does not infer, emit, or enforce them at runtime. It uses an assertion
only to justify a semantics-preserving target index conversion when the full
asserted range is legal for the target representation and indexed extent. An
assertion does not clamp, wrap, or otherwise redefine out-of-range source
values; the application remains responsible for satisfying the precondition on
every execution.

Project Workgroup-Access Assertions
-----------------------------------

OpenGL cannot represent a source workgroup pointer directly. It specializes
pointer-free helpers against a concrete entry-owned ``shared`` array and must
prove that every composed access remains within that backing array. When the
source runtime already enforces an entry-specific absolute element range,
record that precondition explicitly:

.. code-block:: toml

   [[project.workgroup_access_assertions]]
   source = "kernels/fft.metal"
   entry_point = "fft_mem_256_*"
   function = "ReadWriter_*"
   parameter = "crosstl_ptr_buf"
   minimum = 0
   maximum = 255

Each assertion table has these fields:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``source``
     - Repository-relative source glob. Omitting it defaults to ``*``.
   * - ``entry_point``
     - Required source entry-point pattern. The assertion cannot cross entry
       ownership boundaries.
   * - ``function``
     - Helper-function pattern. Omitting it defaults to ``*``.
   * - ``parameter``
     - Workgroup-pointer parameter pattern. Omitting it defaults to ``*``.
   * - ``minimum``
     - Inclusive absolute element offset into the concrete backing array.
   * - ``maximum``
     - Inclusive absolute element offset. It must not be less than ``minimum``.

The assertion does not provide backing identity, extent, element type, or a
pointer offset. OpenGL must still derive those properties from the source call
graph and emits the original composed runtime offset expression. Matching
assertions are intersected with any statically derived access range; a
contradiction or an asserted range outside the concrete backing array fails
before artifact emission. Entries without a matching assertion continue to
require a complete static proof.

Workgroup-access assertions are host/runtime portability preconditions. CrossGL
records them in the project report but does not emit runtime checks or change
source indexing behavior. The application is responsible for satisfying every
assertion on each dispatch.

The portability report records the configured tables under
``project.workgroupAccessAssertions`` and their count under
``project.workgroupAccessAssertionCount``. Report consumers can therefore audit
the host/runtime assumptions used during translation alongside the generated
artifacts.

Exact DirectX bfloat16 Contract
-------------------------------

Exact DirectX bfloat16 lowering preserves bfloat16 storage and conversion
semantics instead of substituting IEEE half precision. When generated HLSL uses
native 16-bit storage, its artifact and runtime metadata advertise DirectX 12,
a minimum Shader Model of 6.2, and entry profiles such as ``cs_6_2``. DXC
validation and runtime loader commands for that HLSL require
``-enable-16bit-types``; application-specific compiler wrappers and build
commands must preserve the same option.

HLSL resolves retained scalar type aliases before selecting bfloat conversions.
Their logical bfloat type remains distinct from the physical ``uint`` register
payload, including initialization, helper parameters, returns and arithmetic.
The Metal frontend retains dependent primitive aliases and resolves namespace
ownership before emitting the intermediate source. Dependencies bind at their
declarations, so a later local alias cannot change a helper's parameter or return
type. Required native controls cover floating-point, integer, half and bfloat
chains, local shadowing and namespace-owned overloads, with unchanged Metal
sources as controls on macOS. Invalid primitive alias resolution produces
``project.translate.metal-scalar-alias-unresolved`` with a ``scalarAlias`` detail
record containing ``aliasName``, ``reason`` and, for cycles, ``dependencyChain``.
This does not establish arbitrary C++ name lookup or volatile scalar-value
support; unsupported volatile value aliases fail translation.

Successful artifacts that use this path record a ``bfloat16Lowering`` object
with ``status`` set to ``exact``, ``approximationUsed`` set to ``false``, and
the register, storage, and rounding representations used by the generated
HLSL. This keeps exact lowering distinct from a compile-only half-precision
substitution in machine-readable reports.

If an operation cannot be lowered with exact bfloat16 semantics, translation
fails closed with a structured
``project.translate.directx-bfloat16-unsupported`` diagnostic. Its
``details.bfloat16Lowering`` record identifies available context such as the
target profile, operation, source type, and reason instead of silently changing
precision or behavior.

These compiler requirements and diagnostics define an artifact contract only.
They do not imply automatic host runtime or backend integration: CrossGL does
not modify application loader code, configure a DirectX backend, bind
resources, or wire generated artifacts into a framework.

Fail-Closed Pointer Provenance
------------------------------

Targets that cannot represent a source pointer directly must prove its backing
storage and composed offset before emitting an artifact. For workgroup storage,
that proof includes the concrete backing declaration, entry-point ownership,
element extent and type, offset composition through helper calls, and the
affected materialization or specialization. Dynamic backing selection,
unresolved offsets, incompatible declarations, escaped identity, and
cross-entry ownership fail closed instead of producing target code with altered
aliasing or synchronization behavior.

When the target exception provides provenance, an OpenGL workgroup-pointer
diagnostic records the available ``function``, ``parameter``, ``backingName``,
``offsetExpression``, ``materializationName``, and ``reason`` values under
``details.workgroupPointer``. Unavailable values are omitted so consumers can
distinguish retained evidence from assumptions. The surrounding diagnostic
also identifies the source path, intended artifact, target, and missing target
capability.

This report contract localizes translation work and preserves actionable
evidence. It does not establish whole-repository semantic parity, rewrite host
runtime integration, or prove execution correctness for a framework or corpus.

Project scan, report, and translation commands also accept repeatable
``--source-root``, ``--include-dir``, ``--define``, and ``--source-override``
overrides. CLI source roots replace the configured source roots for that
command. CLI defines use ``NAME`` or ``NAME=VALUE`` syntax and override matching
names loaded from ``crosstl.toml``. CLI source overrides use ``PATTERN=BACKEND``
syntax and override matching source patterns loaded from ``crosstl.toml``.
These overrides are recorded in the emitted project report.
Scan, report, and translation commands accept repeatable ``--variant NAME``
selectors. ``crosstl.toml`` can set ``selected_variants = ["debug"]`` as the
default scoped variant list for project runs; explicit ``--variant`` arguments
override that configured default for the command. Scoped scan and report output
evaluates only the selected declared variants for variant-aware include and
define metadata, records the selected variant list, and does not claim omitted
variants as scanned.

Unsupported target backend names are reported as configuration diagnostics in
scan, report, and translation output. Translation still records per-artifact
failures for any artifact attempt that cannot be generated.

Validate artifacts referenced by a report:

.. code-block:: bash

   python -m crosstl validate-project crosstl-out/portability-report.json \
     --format text

Validation exits nonzero when the report metadata is malformed, artifact
records, source-map records, or preserved diagnostics are malformed, source-map
mapping lists are empty, file-granularity source maps do not contain one
file-level mapping, finer-grained mappings are not positive-length or fall
outside artifact-level file anchors, source-map, diagnostic location, or
diagnostic ``originalLocation`` spans are internally inconsistent, diagnostic
location file paths or artifact source paths are not repository-relative, project target
lists are not normalized and deduplicated, diagnostic or artifact targets are
not declared by the report, artifact sources are not declared translation units,
embedded validation records reference artifacts not declared by the report,
embedded toolchain runs reference failed report artifacts,
validation records contain duplicate identities or inconsistent status fields,
full reports with embedded validation artifacts omit validation summaries,
summarized embedded validation omits declared artifacts, embedded toolchain-run
coverage omits OK validation artifacts for available toolchains, failed
embedded toolchain runs omit matching diagnostics, external corpus entry presence,
discovery, or source-backend fields do not match the project root and declared
units, full reports omit units or skipped files that the current
project scan discovers, translated outputs are missing, artifact paths resolve
outside the repository, generated artifact hashes no longer match the files on
disk, source files with recorded hashes are missing or changed, or opt-in
toolchain smoke checks fail. The report file being validated is ignored during
that freshness scan, and files under the configured output directory remain
excluded from discovery.
Toolchain smoke checks only run for translated artifacts that still exist inside
the repository. Each smoke check is bounded by a short subprocess timeout, and
timeouts are reported as failed toolchain runs. Targets that need backend-
specific entry points, profiles, package metadata, or SDK context record bounded
tool availability runs instead of claiming artifact compilation. Validation
reports include
severity, diagnostic-code, and missing-capability rollups for generated and
preserved diagnostics, plus artifact target, artifact source-backend,
artifact variant, hash-status, source-size status, generated-size status,
source-map status, source-remap status, toolchain status, toolchain-run status,
toolchain-run target, toolchain-run source backend, diagnostic check kind,
toolchain-run check kind, toolchain-run tool, and toolchain-run variant rollups
for validation results.
The JSON validation report uses schema version 1 with a fixed top-level field
set so automation can detect contract drift. It includes compact project
context with the project root, output directory, configured targets, source
roots, include/exclude patterns, include directories, selected variants,
define/variant names without exposing raw define values, and the source report
hash used for validation provenance.
The default output is JSON; ``--format text`` prints a concise validation
summary with validation report identity metadata, source report hash, project
context, and the same rollups, and ``--format sarif`` emits validation
diagnostics as SARIF with project context and source report hash metadata in
invocation properties.

Inspect an existing report as a concise JSON, text, or SARIF summary:

.. code-block:: bash

   python -m crosstl inspect-report crosstl-out/portability-report.json \
     --format text \
     --max-diagnostics 20 \
     --max-failed-artifacts 20 \
     --max-source-map-artifacts 20 \
     --max-artifact-matrix-artifacts 20 \
     --max-artifact-provenance-artifacts 20 \
     --max-define-processing-artifacts 20 \
     --max-include-path-processing-artifacts 20 \
     --max-include-dependencies 20 \
     --max-skipped-sources 20 \
     --max-validation-artifacts 20 \
     --max-toolchain-runs 20 \
     --max-migration-actions 20 \
     --max-runtime-references 20 \
     --max-external-corpus-entries 20

Report inspection includes inspection identity, SARIF invocation metadata, and
source report schema/kind metadata, source report hash metadata,
source report generation metadata,
validation status,
invalid/unavailable report status, project counts, project configuration path,
project root, output directory, configuration counts, normalized source-root,
include-pattern, exclude-pattern, and include-directory lists, runtime-reference
rollups and bounded runtime-reference samples, failed artifacts
with variant labels when present, diagnostic code and missing-capability rollups,
validation diagnostic-code, missing-capability, artifact target,
artifact source-backend, artifact variant, hash-status, source-map status,
source-remap status, toolchain status, and toolchain-run target,
source-backend, check-kind, tool, and variant rollups, report
source-backend, source override mappings, file-extension, and artifact
target rollups, source-map count, granularity, target, and source-backend
rollups, source-remap count, mapping-count, granularity, target, and
source-backend rollups,
artifact matrix completion counts, matrix source provenance, target and
variant completion rollups, sampled missing and extra artifact identities,
bounded validation artifact and validation toolchain-run samples with
truncation counts, failed validation metadata on artifact provenance samples,
include-directory status counts, inactive source-root and include-directory
record details, diagnostics, configurable diagnostic and failed-artifact
truncation counts, external corpus rollups, sampled missing and
present-but-undiscovered external corpus entries with retained provenance
metadata and configurable sample limits, and migration actions.
Inspection sample-limit options accept non-negative integer counts and default
to ``20`` for each sampled report section.
The JSON inspection report uses schema version 1 with a fixed top-level field
set so automation can detect contract drift while optional report sections
remain present with ``available: false`` until their source report data exists.
Migration action inspection is bounded and records truncation counts for large
reports.
``--format sarif`` emits the inspection diagnostics as SARIF for
code-scanning workflows. SARIF invocation properties include the source report
path, source report hash, report identity metadata, project root, output
directory, configured targets, source roots, include/exclude patterns, include
directories, and selected variants. SARIF locations include line and column
metadata and positive-length character spans when diagnostics carry source
offsets.

Build a metadata-only runtime integration plan from a portability report:

.. code-block:: bash

   python -m crosstl plan-runtime crosstl-out/portability-report.json \
     --format text \
     --max-runtime-references 20

Runtime planning emits a ``crosstl-runtime-integration-plan`` JSON document
with source report hash metadata, validation diagnostics from the source
report, project target summaries, runtime-reference rollups and bounded
samples, per-target compiler runtime-plan request commands, and manual actions
for runtime references found in host or build files. The compiler request
entries point at the metadata-only ``runtime-loader-plan-v1`` contract request
in the compiler repository. This is planning evidence only: it does not import
compiler internals, execute device code, or rewrite host application code.

Build a runtime artifact manifest for downstream host or package tooling:

.. code-block:: bash

   python -m crosstl runtime-manifest crosstl-out/portability-report.json \
     --format text

Runtime artifact manifests emit a ``crosstl-runtime-artifact-manifest`` JSON
document from a validated portability report. The manifest lists translated
artifacts by target with source/backend/variant identity, generated artifact
hash and byte-size metadata, source-map anchors, optional compiler
``source-remap`` sidecars, and the runtime planning contract summary required
by downstream packaging or host integration tooling. Invalid source reports
produce diagnostic-only failed manifests. The manifest is a handoff contract;
it does not generate runtime framework code, execute device code, or rewrite
host application code.

Validation, runtime planning and manifest construction each read the report once.
Within a call, the unit inventory and configuration-diagnostic checks share one
current project scan, and the manifest reuses that validation for its runtime
plan. The reported hash identifies the exact report bytes that were validated.
No scan or validation result is retained across calls; subsequent calls recheck
sources, includes, configuration and generated artifacts. Keep project files
unchanged during validation and packaging: this is not an atomic filesystem
snapshot.

Within one artifact-manifest build, variants of the same source/backend/target
reuse source-wide reflection, including unavailable results. Generated artifacts
are still validated and reflected individually, and entry-specific execution
and specialization metadata is merged separately. Source results are not cached
across manifest builds.

Build a backend-neutral runtime binding manifest for host integrations:

.. code-block:: bash

   python -m crosstl runtime-binding-manifest crosstl-out/portability-report.json \
     --output crosstl-out/runtime-bindings.json

``translate-project`` can write the same binding manifest beside the
portability report in one run:

.. code-block:: bash

   python -m crosstl translate-project /path/to/repo \
     --target cgl \
     --output-dir crosstl-out \
     --report crosstl-out/portability-report.json \
     --runtime-binding-manifest crosstl-out/runtime-bindings.json

Runtime binding manifests emit a ``crosstl-runtime-binding-manifest`` JSON
document derived from the validated portability report and runtime artifact
metadata. Each entry is backend-neutral and includes ``sourceFile``,
``sourceBackend``, ``targetBackend``, ``artifactPath``, ``entryPoint``,
``resourceBindings``, ``bufferMutability``, ``scalarConstants``,
``specializationConstants``, ``dispatchDimensions``, ``sourceProvenance``, and
``validation``. Resource bindings include set/binding coordinates, access, and
derived mutability.
Dispatch dimensions record reflected workgroup size data when available while
leaving workgroup, global, and grid counts unset for host code to provide.
Reflection, runtime artifact manifests, and runtime binding manifests keep
function and specialization constants in dedicated ``specializationConstants``
records with their own counts. They are not reported as ``resources`` or
``resourceBindings``, nor as ordinary ``constants`` or ``scalarConstants``.

Build a deterministic runtime handoff package from a runtime artifact manifest:

.. code-block:: bash

   python -m crosstl package-runtime crosstl-out/runtime-manifest.json \
     --package-dir crosstl-runtime-package \
     --format text

Runtime packages emit a ``crosstl-runtime-package`` JSON report and write a
package manifest, translated artifacts, source-remap sidecars, and a short
integration guide into the package directory. Packaging revalidates artifact
hash and byte-size metadata before copying files so stale generated outputs are
reported as structured diagnostics instead of hidden. The package is a handoff
artifact for host or build-system tooling; it does not rewrite host application
code, execute device code, generate runtime framework code, or install target
SDKs.

Inspect a runtime handoff package before host binding:

.. code-block:: bash

   python -m crosstl inspect-runtime-package \
     crosstl-runtime-package/runtime-package.json \
     --format text

Runtime package inspections emit a ``crosstl-runtime-package-inspection`` JSON
document with ready and failed host-binding records. The inspection is read-only
and verifies copied packaged artifacts and source-remap sidecars against the
package manifest's recorded paths, hashes, and byte sizes. Missing, stale, or
malformed package contents are reported as structured diagnostics before host
loader or build-system tooling consumes the handoff package. Inspection
preserves the ``runtime-loader-plan-v1`` summary linkage and does not rewrite
host application code, execute device code, generate runtime framework code, or
install target SDKs.

Build a host binding plan from a runtime package manifest:

.. code-block:: bash

   python -m crosstl plan-host-bindings \
     crosstl-runtime-package/runtime-package.json \
     --format text

Host binding plans emit a ``crosstl-runtime-host-binding-plan`` JSON document
with per-target packaged artifact paths, package-inspection readiness metadata,
``bind-runtime-artifact`` actions for host loader or build-system tooling, and
``review-runtime-references`` actions when the source repository contained
runtime API references. The planner reuses runtime package inspection and only
emits bind actions for ready package records; missing or stale package artifacts
remain diagnostics instead of host-integration work items. The plan preserves the
``runtime-loader-plan-v1`` summary linkage from earlier reports. It is an action
plan only; it does not rewrite host application code, execute device code,
generate runtime framework code, or install target SDKs.

Build a target-scoped runtime adapter plan from a runtime package manifest:

.. code-block:: bash

   python -m crosstl plan-runtime-adapters \
     crosstl-runtime-package/runtime-package.json \
     --format text

Runtime adapter plans emit a ``crosstl-runtime-adapter-plan`` JSON document
from the same package handoff metadata used by package inspection. The plan
lists ready package bindings by target with ``adapterKind``, ``artifactFormat``,
``requiredTools``, ``hostResponsibilities``, source-remap handoff paths,
parser-derived ``hostInterface`` entry point and resource summaries where the
packaged artifact frontend is available, and ``wire-runtime-adapter`` actions
for host loader or build-system tooling. When host interface metadata is
unavailable or not ready, the plan emits ``resolve-host-interface-metadata``
actions so host and build tooling can provide reflection or backend-specific
binding metadata before wiring the adapter. Source targets with registered
frontends can contribute parser-derived interface summaries; formats that need
compiled reflection, such as SPIR-V handoff artifacts without reflected entry
point/resource data, remain explicit follow-up actions. The plan also carries
through package inspection diagnostics and
``review-runtime-references`` actions when the source repository contained
runtime API references. The plan is a target-scoped integration contract; it
does not rewrite host application code, execute device code, generate runtime
framework code, or install target SDKs.

Materialize runtime adapter descriptor files from a runtime package manifest:

.. code-block:: bash

   python -m crosstl materialize-runtime-adapters \
     crosstl-runtime-package/runtime-package.json \
     --adapter-dir crosstl-runtime-adapters \
     --format text

Runtime adapter descriptor packages emit a
``crosstl-runtime-adapter-package`` JSON document and write a deterministic
``runtime-adapters.json`` manifest, an ``ADAPTERS.md`` summary, and one
``adapters/<target>/*.adapter.json`` descriptor per ready or blocked runtime
adapter plan record. Each descriptor preserves the packaged artifact path,
target adapter identity, source-remap handoff path, host-interface metadata,
required tools, host responsibilities, and validation readiness for downstream
host loader or build-system tooling. The descriptor package is metadata only:
it does not rewrite host application code, execute device code, generate
runtime framework code, or install target SDKs.

Runtime Adapter Execution Contracts
-----------------------------------

Runtime fixture execution uses a backend-agnostic adapter contract carried on
each ``RuntimeExecutionRequest`` as ``adapter_contract``. The contract can be
loaded from a fixture's ``runtimeAdapter`` object and merged with manifest
metadata already recorded on a translated artifact. This keeps execution
fixtures stable across downstream runtimes while letting package inspection
provide reflected entry points, resource bindings, and dispatch workgroup
sizes when they are available.

The contract fields are intentionally limited to kernel execution metadata:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Purpose
   * - ``entryPoints``
     - Names, stages, execution config, optional parameter records, and
       workgroup-size metadata for callable translated kernels or shaders.
   * - ``resourceBindings``
     - Backend-neutral resource names, kinds, types, set/binding numbers,
       access modes, and optional fixture value names that an adapter maps to
       runtime buffers, textures, samplers, or parameter blocks.
   * - ``specializationConstants`` / ``functionConstants``
     - Specialization or function constant identifiers, dtypes, values,
       defaults, and required flags needed before launching the entry point.
   * - ``dispatch``
     - Entry point, workgroup size, workgroup count, global size, or grid size
       for compute-style launches. Fixture dispatch counts can augment
       artifact-manifest workgroup sizes.
   * - ``validationHooks``
     - Expected pre-run, runtime, post-run, or comparison checks that a
       downstream executor should perform or report as skipped/unavailable.

Runtime planning scopes artifact execution metadata to the selected compiled
entry before merging the fixture's requested adapter contract. When both the
selected entry and requested dispatch provide a concrete workgroup size, the
values must match. A mismatch produces
``project.runtime-verification.workgroup-size-mismatch`` during runtime setup,
records both sizes and the selected-entry provenance, and leaves the test case
unplanned. When only one side provides the size, the planner carries that value
into the merged dispatch, so either missing side of the runtime contract can be
completed from the other. This completion does not hide a disagreement when
both sides are present.

Downstream runtimes implement the ``RuntimeAdapter`` protocol or subclass
``RuntimeExecutor`` and receive the merged contract in ``run(request)``. For
example, an MLX validation adapter can consume translated Metal artifacts and
map neutral fixture buffers and function constants to MLX runtime objects, but
the fixture contract remains expressed in terms of entry points, bindings,
constants, dispatch geometry, and validation hooks rather than MLX APIs.

Saved project test-runner plans can execute deterministic runtime fixtures when
callers supply adapter implementations explicitly:

.. code-block:: bash

   python -m crosstl execute-test-runner \
     crosstl-out/project-test-runner-plan.json \
     --runtime-executor native-vulkan=tools.runtime.vulkan:VulkanRuntimeAdapter \
     --output crosstl-out/project-test-runner-report.json

``--runtime-executor`` is repeatable and uses
``EXECUTOR=MODULE:OBJECT``. ``MODULE`` may be a dotted Python module name or a
``.py`` file path. ``OBJECT`` may be an adapter instance, adapter class, or
factory returning an object with ``run(request)`` or the parity-adapter methods
``prepare_buffers(state)``, ``dispatch(state, buffers)``, and
``collect_outputs(state, result)``.

For the built-in DirectX, OpenGL, and Vulkan native parity adapters, callers can
also pass ``--native-runtime-adapter TARGET`` or
``--native-runtime-adapter TARGET=MODULE:OBJECT``. The optional object is the
backend runtime driver consumed by the native adapter after artifact validation.
Use ``--no-native-runtime-validation`` only when the caller has already handled
toolchain validation or is running a controlled test fixture.

CrossTL includes optional ``DirectXComputeRuntime``, ``OpenGLComputeRuntime``,
and ``VulkanComputeRuntime`` reference drivers for bounded compute fixtures.
They import target dependencies lazily and report structured unavailability or
setup failures when the required API, loader, device, or resource contract is
not available. Their supported resource shapes are intentionally narrower than
the translated shader languages; a successful translation does not imply that
one of these reference drivers can execute the complete host workload.

Native Dispatch Limits
----------------------

The Python native drivers reject invalid launch dimensions before submitting
device work. Counts and local sizes must contain one to three positive integers;
zero, negative, Boolean and fractional values are not clamped or coerced.
Dimension fields must be sequences, not scalar values, mappings, sets, strings
or byte strings. False-valued malformed metadata cannot select a fallback or
bypass local-size checks. The existing empty-sequence/``None`` convention for
omitted dimensions is preserved. Derived group counts use exact integer ceiling
division. Malformed dimensions report ``dispatch-dimensions-invalid``, identifying
the field and, for a sequence, the failing node before any device work.

DirectX checks Direct3D 12 compute limits. OpenGL queries the current context's
per-axis group-count and local-size limits and maximum invocations per group.
Vulkan queries the selected physical device before creating a logical device.
The Metal worker checks device local-size limits and the compiled pipeline's
maximum threads per group. OpenGL and DirectX preflight every request in a
sequence before allocating resources or executing the first node.

An exceeded limit reports ``dispatch-limit-exceeded`` with ``field``,
``requested``, ``maximum``, the requested geometry and the applicable ``limits``.
Per-axis failures include ``axis`` (zero-based); aggregate invocation failures
do not. Sequence diagnostics include ``nodeIndex``. Metal returns the detailed
worker record under ``dispatchValidation`` while preserving its process logs.
Missing or invalid capability information is an error, not an assumed limit.
These checks do not tile an oversized workload, infer missing local sizes from
shader binaries, or replace compiler validation of the actual shader layout.

OpenGL also checks API error state after setup, resource binding, submission,
synchronization and readback. A non-success state reports ``opengl-api-error``
with ``phase`` and ``glError``; inability to read that state is a failure too.
Unchanged output after a rejected submission cannot be reported as successful
execution. These Python-driver checks do not add limit validation to generated
C++ loader adapters or establish complete host-runtime integration.

Byte Field Updates
------------------

Byte-valued variables and fields retain their signed or unsigned eight-bit range
after compound assignment and prefix/postfix updates on Metal, DirectX and
OpenGL. This includes nested fields rooted in a stable local variable. Postfix
expressions return the value before the update, including at overflow boundaries.
Updates whose owner requires evaluating an array index or function call remain
outside this contract; unsupported forms produce a diagnostic instead of
duplicating an observable evaluation.

Lossless Float Buffer Storage
-----------------------------

Typed float32 buffers can opt into ``encoding: "ieee754-binary32"`` in the
Python native dispatch API and runtime-verification fixtures. Each ``values``
element is then an unsigned 32-bit storage word, not a numeric float:

.. code-block:: json

   {
     "dtype": "float32",
     "shape": [3],
     "encoding": "ieee754-binary32",
     "values": [2143363909, 1065353216, 2147483648]
   }

These words represent a noncanonical quiet NaN, 1.0 and negative zero. Native
packing and readback retain their bits without converting through host floats.
The reflected buffer remains float32, so kernels still perform floating-point
arithmetic. Encoding does not bypass physical-layout, size, binding or allocation
checks. Unknown encodings, incompatible dtypes, Boolean words, fractional words
and words outside the unsigned 32-bit range are rejected.

Upload and readback choose their encodings independently, including an
initialized read-write buffer. An output request may omit ``values`` and select
encoded readback using only dtype, shape and encoding. Returned words describe
actual native storage; no original input values are substituted after execution.
Encoded expected outputs require exact word equality and matching encoding,
regardless of numeric tolerance settings. This checks storage preservation, not
a requirement that arithmetic NaN results preserve an operand's payload.

Without ``encoding``, the existing numeric buffer and non-finite token contracts
remain unchanged. Native regression gates exercise generated Metal, DirectX and
OpenGL partial updates, with original Metal controls and shared-allocation
DirectX/OpenGL sequences. Metal's process-isolated runtime has no sequence API.
The representation does not cover double storage or generated C++ adapter
serialization.

Bfloat16 Storage
~~~~~~~~~~~~~~~~

Native Metal ``bfloat`` buffers use ``dtype: "bfloat16"`` and
``encoding: "bfloat16-bits"``. Uploads and readbacks contain unsigned 16-bit
storage words, not host floating-point conversions. An explicit encoding is
required so bfloat16 cannot be mistaken for IEEE binary16 or float32. Exact-word
comparisons retain signed zeros, subnormals and NaN payloads.

DirectX's bfloat lowering exposes ``uint16_t`` buffers. Supply ``dtype: "uint16"``
and unsigned integer values to that physical interface. Explicit HLSL
``int16_t`` and ``uint16_t`` buffers, and Metal ``short`` and ``ushort`` buffers,
retain two-byte integer storage. DXC requires native 16-bit types to be enabled.
OpenGL's widened bfloat lowering exposes float32 storage; exact bfloat carriers
are binary32 words with the original word in the upper 16 bits. Neither target
accepts a logical bfloat16 payload in place of its reflected physical layout.

Required native copy controls cover all 65,536 bfloat storage patterns and
output guards, with original Metal execution as a round-trip control. Storage
support does not establish bfloat arithmetic, conversion rounding or complete
MLX host integration. Padded vectors and unsupported allocation views remain
subject to the native loader's existing restrictions.
Metal and OpenGL controls additionally cover two- and four-lane vectors and
homogeneous structures. DirectX vector and aggregate lowering remain tracked
separately in issue #1488.

OpenGL rounds float32-to-bfloat numeric conversions to nearest, with ties to
even, using integer operations on the binary32 representation. Constructors,
assignments, function arguments and returns, scalar/vector arithmetic, and
vector components retain bfloat precision before widening to float32. Existing
bfloat values are not requantized during identity copies or vector repacking.
The helper quiets NaNs while retaining their sign and upper payload bits;
same-type copies preserve every payload bit instead. Native Metal can
canonicalize NaNs during numeric conversion. Its conversion controls require
exact agreement between original and generated Metal, and separately check
NaN classification, sign and bfloat representability. They do not establish
cross-target NaN payload identity. All finite conversion results and identity
copies retain exact bitwise comparisons. Runtime scalar 32-bit integer conversions use
integer rounding in OpenGL and DirectX to avoid an intermediate float32 rounding
step. OpenGL runtime conversions from double and wider integers that could
introduce double rounding fail with a structured diagnostic. This does not
establish support for every bfloat operation in an MLX host backend.
Scalar integer constant initializers also round directly to bfloat, without
an intermediate float32 value. Constructor, cast, alias and implicit initializer
forms preserve the difference between ``bfloat(16842753u)`` (16908288) and
``bfloat(float(16842753u))`` (16777216). Native controls retain signed zero,
midpoint neighbors, ties of both parities, exponent carry and 32-bit extrema.
Unsupported constant expressions, including unproven unsigned negation,
narrowing integer casts and Metal literals whose wider source type was lost,
remain diagnostic rather than silently using a rounded float32 initializer.
Metal integer literal width preservation is tracked in issue #2070.

Binary16 Storage
~~~~~~~~~~~~~~~~

Metal ``half`` and DirectX ``float16_t`` buffers can use ``dtype: "float16"``
with ``encoding: "ieee754-binary16"``. Values are unsigned 16-bit words; for
example, ``[0, 32768, 1, 32257]`` represents positive zero, negative zero,
the smallest positive subnormal and a quiet NaN with a payload. Uploads and
readbacks use two bytes per scalar, without conversion through host floats.
The same exact-word comparison rules apply as for binary32 storage.

Reflection retains the actual element size, stride and alignment through
packaging and native-loader descriptors. Scalar, naturally aligned vectors,
homogeneous structures and constant-buffer values retain their physical
layouts. Padded layouts that cannot be represented by a tightly packed value
array remain unsupported. Ambiguous HLSL ``half`` declarations are not assumed
to have native 16-bit storage; explicit ``float16_t`` is required.

Without an encoding, finite numeric values are rounded to binary16 on upload;
overflow is rejected. Explicit non-finite input tokens remain available.
Use encoded values when signed-zero or NaN payload identity must be checked.
Boolean words, negative words, values above 65535, inconsistent encodings,
misaligned views and truncated allocations fail before dispatch.
Metal supports aligned offset views. The DirectX Python driver realizes aligned
structured-buffer ranges with native descriptors; this does not change the
shader's storage type or repair values after readback.

Metal device references to supported scalar and naturally aligned two- or
four-component vector values retain ``storageLayout: "metal-buffer"`` with
``runtimeSized: false``. Their ``blockSizeBytes`` and
``minimumBindingSizeBytes`` both describe one physical value. Const references
remain read-only; writable references support initialized readback through the
native loader. Offset views must satisfy the same alignment and minimum-size
checks as other buffers. Pointer arrays and constant-address-space references
retain their existing layouts. This layout contract does not establish support
for unresolved aliases, padded aggregates or every source reference-lowering
case.

Writable entry references remain bound storage rather than value-result entry
parameters. Ordinary helper references retain their existing direction
qualifiers. OpenGL distinguishes value updates through an entry reference from
pointer-offset updates, including when both occur in the same function.
Required native tests cover initialized signed 32-bit scalar writes on all
three targets, including two- and four-component vector writes.
The tests retain buffer guards, aligned offsets and disjoint writable views of
one allocation; Metal also executes the unchanged original source. DirectX
promotes read-only device vector references to structured buffers and constant
references to constant blocks, retaining access modes, component types and
widths in the package interface. Buffer-reference expressions address the
first value, not a mutable pointer offset. Local values and helper parameters
that shadow a global buffer retain their arithmetic updates. Unresolved aliases,
padded source layouts and overlapping writable views are not established by
this coverage.

An HLSL artifact may explicitly distinguish logical binary16 values from native
unsigned 16-bit storage. Its first line contains a versioned resource contract,
serialized by ``crosstl.translator.resource_storage.resource_storage_header``:

.. code-block:: hlsl

   // crosstl-resource-storage: {"schemaVersion":1,"resources":{"values":{"logicalElementType":"float16","encoding":"ieee754-binary16"}}}
   StructuredBuffer<uint16_t> values : register(t0);

Reflection retains ``elementType: "uint16"`` and the actual ``physicalType``;
it adds ``storageEncoding`` containing the logical type and encoding above.
The loader accepts ``float16`` values for that declared representation without
changing their bytes. Shader loads decode with ``asfloat16`` and stores encode
with ``asuint16``; arithmetic remains floating-point arithmetic. This contract
supports tightly packed scalar, vector and homogeneous-struct structured
buffers, and one reflected scalar or vector in a fixed constant buffer.
Textures, padded structures and widened storage are not covered by this codec.
Unknown encodings, duplicate or misplaced headers, undeclared resources and
incompatible physical layouts fail before dispatch. The comment declares an
ABI convention, not proof of the shader's implementation or numerical behavior.

The HLSL generator uses integer storage for native half structured buffers and
promoted entry-point scalar constant references. Local variables, shared memory,
function values and arithmetic retain their logical half types. Structures use
separate physical declarations and member-wise conversion helpers, including
nested structures and fixed arrays; the native loader still requires a
representable reflected layout. Existing user-declared constant buffers and
typed textures are not rewritten. Side-effecting compound-assignment targets
and shadowed bitcast intrinsics produce diagnostics rather than ambiguous code.

This lowering avoids typed ``float16_t`` resource loads, which can quiet
signaling NaNs on the pinned Windows runtime. The required Windows storage gate
passes with the integer representation. An independent readback audit verifies
49 generated cases: 19 copy forms, eight buffer forms, eight exhaustive-test
dispatches covering all 65,536 binary16 words in two storage forms, six
arithmetic/update cases and eight constant-reference payloads. It checks guards,
offset-allocation bytes, freshly regenerated shader
identities and compiled-module identities. Handwritten ABI controls remain
separate from this generated-code evidence. This establishes exact transport for
the tested resource forms, not numerical parity for arbitrary half arithmetic or
complete host-runtime integration.

OpenGL's widened half lowering continues to expose float32 physical storage.
A binary16 payload cannot bind that storage. Conversion tests use the reported
physical representation and verify the half-rounded result; they do not claim
native two-byte OpenGL storage. Required native tests cover copy payloads,
conversion rounding and output guards, but do not by themselves establish
complete MLX half-precision operation coverage.

Metal roundtrip and OpenGL same-type half copies preserve their logical
representation, including NaN payloads, through buffer loads, local storage,
helper arguments and returns,
vector components and homogeneous structures. OpenGL uses exact widened
binary32 words for this contract; callers must supply the representation of
the logical binary16 value. Repacking existing half components does not round
them again. Actual conversions from float32 and half arithmetic still apply
the binary16 rounding contract. Required native copy controls cover every
binary16 word as well as pointer and vector access paths.

Rounding a mathematical result to binary16 does not establish bitwise parity
with every native half-precision intrinsic. The sine control currently matches
the unchanged source in the Metal round trip but differs from the mathematical
reference used by OpenGL. Its evidence retains both comparisons; it is not a
cross-backend parity claim. Compiler/profile accuracy remains tracked in
`issue #2068 <https://github.com/CrossGL/crosstl/issues/2068>`_.

This contract applies to the Python native runtime drivers. The generated C++
DirectX adapter still requires four-byte-multiple structured-buffer strides;
its two-byte view support and encoded-value serialization remain separate work.

Shared Native Allocation Views
------------------------------

A runtime fixture value can include an optional ``allocation`` object to keep
native allocation identity separate from the reflected resource name and
binding coordinate. The allocation object has the following fields:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Purpose
   * - ``id``
     - Required, stable allocation identity. Values attached to separate
       bindings refer to one allocation only when this value is identical.
   * - ``byteOffset``
     - Byte offset of the typed resource view. The default is ``0``.
   * - ``byteLength``
     - Bounded view length in bytes. When omitted, runtime planning derives it
       from the fixture shape and exact scalar layout when possible.
   * - ``allocationByteLength``
     - Total allocation size in bytes. When omitted, runtime planning derives
       it from the greatest known end offset among views with the same ``id``.

The view remains typed by the surrounding fixture value's ``kind``, ``dtype``,
``shape``, and physical layout metadata. Its access mode and binding coordinates
remain part of the corresponding ``resourceBindings`` entry. The allocation
object does not replace those contracts or permit a fixture to reinterpret an
incompatible physical layout.

For example, separate reflected input and output bindings can intentionally
refer to the same full allocation while retaining distinct binding coordinates:

.. code-block:: json

   {
     "inputs": [
       {
         "name": "source",
         "kind": "buffer",
         "dtype": "float32",
         "shape": [2],
         "values": [1.0, 2.0],
         "allocation": {
           "id": "working-set",
           "byteOffset": 0,
           "byteLength": 8,
           "allocationByteLength": 8
         }
       }
     ],
     "expectedOutputs": [
       {
         "name": "destination",
         "kind": "buffer",
         "dtype": "float32",
         "shape": [2],
         "values": [2.0, 4.0],
         "allocation": {
           "id": "working-set",
           "byteOffset": 0,
           "byteLength": 8,
           "allocationByteLength": 8
         }
       }
     ],
     "runtimeAdapter": {
       "resourceBindings": [
         {
           "name": "source",
           "kind": "buffer",
           "binding": 0,
           "access": "read"
         },
         {
           "name": "destination",
           "kind": "buffer",
           "binding": 1,
           "access": "write"
         }
       ]
     }
   }

Runtime planning preserves the explicit allocation ID on each bound resource
and validates the complete group before adapter execution. It rejects malformed
ranges, conflicting total sizes, out-of-bounds or misaligned views, explicit
input/output views that disagree for one binding, incompatible overlapping
physical layouts, and overlapping writable views without a synchronization
plan. Validation diagnostics identify the allocation and affected binding
coordinates or byte ranges. Driver-specific failures also identify the
applicable target constraint. A plan containing these errors is not dispatched.

The DirectX and OpenGL reference drivers group compatible bindings by allocation
ID, combine non-conflicting fixture uploads, create one physical device
allocation for the group, and bind that allocation at each reflected coordinate.
Conflicting upload bytes fail setup instead of causing an implicit conversion or
per-binding allocation. DirectX requires one dtype and stride across the group
and rejects simultaneous aliases of a constant buffer within one dispatch.
An ordered DirectX sequence may reuse an immutable constant allocation across
nodes when every view has the same scalar layout and extent at a 256-byte-aligned offset.
Conflicting uploads, different allocation sizes, and mixing constant buffers with
SRV/UAV resources remain errors. Source constants are uploaded once; derived
region constants retain distinct allocations. OpenGL supports bounded
uniform-block and storage-buffer ranges, subject to the offset-alignment limits
reported by the active context. It rejects mixed uniform/storage groups,
incompatible overlapping scalar layouts, and overlapping writable ranges.

DirectX requests containing allocation subranges use a process-isolated D3D12
worker. MSVC with the Windows SDK is required to build this worker; unchanged
builds are reused within the Python process. The selected adapter must match
the caller's device. Each allocation ID creates one physical buffer, while
CBV addresses and structured SRV/UAV descriptors preserve the requested offset
and extent. Ordered dispatches keep that allocation alive and synchronize queue
completion before the next dispatch. Output decoding selects the requested
range from a complete allocation readback, retaining its hash and GPU address
in runtime evidence. Execution is bounded to 120 seconds, with a 256 MiB limit
per allocation and 512 MiB per request. No shader rewriting, host evaluation or
independent zero-offset copies substitute for ranged binding.

Ranged execution supports shared read-only views, disjoint UAV views and
cross-dispatch read/write reuse. Simultaneous SRV/CBV reads and UAV writes to one
allocation within a dispatch remain unsupported by the worker's classic
resource-state model and fail with ``allocation-state-incompatible``. This
limitation is not evidence that an equivalent enhanced-barrier implementation
is impossible. Windows CI requires native offset, shared-allocation and
ordered-reuse controls, including complete allocation guards.

The built-in Vulkan driver does not currently realize shared allocation IDs or
bounded allocation views, and no native shared-allocation support is claimed for
Metal, WebGL, WGSL, CUDA, HIP, Mojo, Rust, or Slang targets. Runtime-plan
serialization on those targets is not evidence that one physical allocation was
reused. Target-specific synchronization, resource-state transitions, allocation
lifetime, and framework memory planning remain host-runtime responsibilities.

The ``allocation`` field is optional for backward compatibility. When it is
absent, runtime planning assigns a deterministic per-binding allocation ID, so
existing independent bindings remain independent. A single initialized
``read_write`` binding continues to use one allocation for upload and readback.
Aliasing between separate bindings is never inferred from equal values or
similar names; it requires an explicit shared ``id``.

The translator stops at this contract boundary. Full framework rewrites,
non-kernel host API ports, application command scheduling, target SDK
installation, build-system migration, memory lifetime policy, and production
runtime framework generation remain downstream integration work.

Runtime Execution Graphs
------------------------

The runtime execution graph API represents a bounded multi-operation workload
without embedding target runtime calls in the project report. Use
``parse_runtime_execution_graph`` to load a versioned graph,
``validate_runtime_execution_graph`` to obtain structured diagnostics,
``inspect_runtime_graph_package`` to verify packaged artifact and interface
references, and ``execute_runtime_graph`` to run a supported native graph. The
types and functions are available from ``crosstl.project``.

A graph declares resources separately from operations. Resource records carry
their role, kind, exact physical layout, optional allocation view, and bounded
lifetime. Nodes use stable IDs and explicit ``dependsOn`` edges and have one of
the following operation records:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Node kind
     - Contract
   * - ``dispatch``
     - Selects one translated artifact and entry point, maps named bindings to
       graph resources with explicit access, and records dispatch geometry and
       constants.
   * - ``copy``
     - Describes bounded source and destination byte ranges between compatible
       resources.
   * - ``fill``
     - Describes a bounded byte range and scalar fill value for one resource.
   * - ``barrier``
     - Makes an explicit write-to-read or write-to-write visibility transition
       for named resources between dependency-ordered operations.

Dispatch, copy, and fill nodes may also carry bounded repeat or condition
records. Validation rejects unbounded controls, dependency cycles, missing
resource or artifact references, unsafe access ordering, missing visibility
barriers, incompatible layouts or ranges, and temporary-resource lifetimes that
do not cover their producers and consumers. Failures are returned as structured
diagnostics with graph, node, resource, path, and missing-capability context;
unsupported constructs are not silently removed.

Package inspection validates the graph before reading package metadata. For
each dispatch it resolves the artifact selector deterministically, requires one
ready packaged artifact, checks the requested entry point, and verifies named
binding presence, uniqueness, and access compatibility against the reflected
host interface. Inspection is read-only and records explicitly that device
execution was not performed.

The DirectX and OpenGL reference runtimes currently execute dependency-ordered
``dispatch`` and ``barrier`` paths. All requests are validated before device
work begins, resources are allocated once for the graph sequence, and a
payload-free temporary resource can retain one physical allocation from its
producer through its consumer. Intermediate results remain on the device and
output readback is deferred until the sequence completes. The native proofs
execute a two-stage reduction: an input of shape ``[4]`` containing
``[0, 1, 7, 42]`` produces two partial sums in a temporary of shape ``[2]``,
then an output of shape ``[1]`` containing ``[50]``. Direct3D 12 and OpenGL 4.3
both verify the exact result.

``copy`` and ``fill`` nodes and bounded control records are part of the
serializable and validated graph model, but the native executor currently
returns explicit unsupported-capability diagnostics for them. Vulkan native
graph execution is not supported, and no native graph execution is claimed for
Metal, WebGL, WGSL, CUDA, HIP, Mojo, Rust, or Slang. The backend-neutral graph
contract can still be parsed and validated independently of those targets.

The pinned MLX revision
``4367c73b60541ddd5a266ce4644fd93d20223b6e`` is a corpus and CI reference for
the runtime-porting work, not a special case in the graph schema or executor.
This capability does not rewrite host applications or runtime frameworks and
does not claim that the complete MLX runtime or upstream test suite has been
ported.

Project Runtime Fixture Manifest Generation
-------------------------------------------

Curated repository fixture metadata can be converted into a standard
``crosstl-project-runtime-test-manifest`` document without adding a native
runtime adapter. This lets project ports describe parity cases as deterministic
inputs, expected outputs, tolerances, artifact selectors, runtime adapter
contracts, resource bindings, function or specialization constants, and dispatch
geometry, then reuse the existing runtime test manifest parser, planner, and
report writer.

Build a project runtime test manifest from a translated artifact report or
runtime artifact manifest plus curated fixture metadata:

.. code-block:: bash

   python -m crosstl runtime-test-manifest \
     crosstl-out/runtime-manifest.json \
     fixtures/runtime-fixtures.json \
     --format text

The fixture metadata convention is a small repository-agnostic input document:

.. code-block:: json

   {
     "kind": "crosstl-project-runtime-fixture-metadata",
     "fixtures": [
       {
         "id": "reduced-binary-add-f32",
         "selector": {
           "source": "mlx/backend/metal/kernels/binary.metal",
           "target": "metal",
           "path": "out/metal/reduced_binary_add.metal"
         },
         "inputs": [{"name": "lhs", "values": [1.0, 2.0]}],
         "expectedOutputs": [{"name": "out", "values": [3.0, 4.0]}],
         "runtimeAdapter": {
           "entryPoints": [{"name": "binary_add", "stage": "compute"}],
           "resourceBindings": [
             {"name": "lhs", "kind": "buffer", "binding": 0, "value": "lhs"},
             {"name": "out", "kind": "buffer", "binding": 1, "value": "out"}
           ],
           "functionConstants": [
             {"name": "element_count", "id": 0, "value": 2}
           ],
           "dispatch": {"entryPoint": "binary_add", "globalSize": [2, 1, 1]}
         }
       }
     ]
   }

The generator validates each fixture selector against the translated artifacts.
Incomplete fixture data, duplicate ids, missing expected outputs, unresolved
artifacts, and ambiguous selectors are emitted as structured diagnostics on the
generated manifest. Valid fixture records remain in the manifest so
``plan_runtime_test_manifest`` and ``verify_runtime_test_manifest`` can apply
the same adapter dependency checks and runtime planning used by hand-authored
manifests.

Generated test records also include ``metadata.runtimeMetadata``. When the
selected artifact carries ``runtimeDataStatus`` from a runtime artifact
manifest, that status is preserved; otherwise the generator derives readiness
from the merged runtime adapter contract. The manifest summary includes
``runtimeMetadataStatusCounts`` so downstream tooling can separate incomplete
fixture data from incomplete artifact metadata before attempting native runtime
execution.

Build a runtime loader manifest from a runtime package manifest:

.. code-block:: bash

   python -m crosstl runtime-loader-manifest \
     crosstl-runtime-package/runtime-package.json \
     --format text

Runtime loader manifests emit a ``crosstl-runtime-loader-manifest`` JSON
document derived from runtime adapter planning. The manifest groups per-target
load units with package-relative artifact paths, adapter kind, artifact format,
source-remap handoff paths, parser-derived ``hostInterface`` metadata when
available, required target tools, host responsibilities, ordered loader steps,
and blockers that must be resolved before a host loader or build-system adapter
can consume the artifact safely. Unavailable interface reflection remains an
explicit ``resolve-host-interface-metadata`` blocker instead of being hidden or
treated as generated host code. The manifest carries package inspection
diagnostics and runtime-reference review actions forward, and it remains a
metadata contract only: it does not rewrite host application code, execute
device code, generate runtime framework code, or install target SDKs.

Native Metal Package Execution
------------------------------

``MetalRuntimeParityAdapter`` and ``MetalComputeRuntime`` execute compute
artifacts through the public package, loader descriptor and dispatch-request
APIs on macOS 13 or newer. They require Xcode's Metal and Swift tools and an
available Metal device. No additional Python GPU binding is required. The
runtime compiles an identity-checked source snapshot with warnings fatal and
fast math disabled, links a Metal library, and runs a shipped Swift worker.

Requests require explicit or reflected ``workgroupSize`` and
``workgroupCount``; the runtime does not guess a group size from a maximum-thread
attribute. Buffer arguments share one index namespace, from 0 through 30, in
resource set zero. Scalar buffers, tightly packed two-/four-component 32-bit
vectors and flat homogeneous scalar structs retain their exact physical layout.
``constant T*`` parameters are read-only runtime buffers; supported
``constant T&`` scalar/vector parameters are fixed-size values. Metal ``long``
and ``ulong`` scalar storage uses signed/unsigned 64-bit host values. Scalar
``bool`` buffers and constants use one-byte storage with Boolean host values;
integer values are not implicitly converted. Readback rejects bytes other than
zero and one. Padded vectors, half storage, Boolean vectors, nested or mixed
structs, textures, samplers and dynamic threadgroup arguments are not supported
by this buffer runtime.

Dispatch values describe physical storage, not a target-independent tensor
encoding. DirectX and OpenGL Boolean buffers retain their reflected
``uint32`` representation with four-byte elements. Callers must use that
representation rather than upload Metal's one-byte Boolean payloads. The
loader rejects mismatched widths. Required native checks exercise translated
comparison and mask kernels, guarded odd-sized buffers, read-modify-write
values and offset views; macOS also executes the original Metal source.
This storage support does not by itself implement MLX Boolean host operations.

The worker specializes Boolean, float32, int32 and uint32 function constants
by numeric ID, verifies required compiled buffer arguments and alignment,
checks device and pipeline threadgroup limits, dispatches three-dimensional
threadgroups, synchronizes and reads the requested byte views. Compatible
aliased views share one allocation; contradictory overlapping initial bytes
are rejected. The default combined buffer limit is 256 MiB, configurable with
``MetalComputeRuntime(max_buffer_bytes=...)``. This is not dynamic shader
memory-safety analysis.

Compilation, probing and execution each have a default 120-second deadline,
configurable with ``MetalRuntimeParityAdapter(timeout_seconds=...)``. A timeout
terminates and reaps the worker process group and reports a structured failure;
it never substitutes host computation. The default runtime shares a verified
helper executable within the Python process when the helper source, compiler
options, toolchain, SDK and architecture are unchanged. Every probe and dispatch
still starts a separate bounded process; shader compilation, GPU state and
readbacks are not cached. ``close()`` releases instance-owned resources without
invalidating another instance's shared helper. Shared executables are removed
at process exit; custom command runners or tool resolvers retain private helpers
that ``close()`` removes. Device name,
compiled-library hash, execution width and dispatch geometry accompany readback
evidence. The implementation follows Metal's
`compiled binding reflection
<https://developer.apple.com/documentation/metal/mtlcomputepipelinereflection>`_.

Required macOS CI covers package execution, sparse bindings, multidimensional
indexing, function constants and invalid contracts. The current MLX binary-shape
gate additionally runs all 15 selected complex-power shapes through this public
runtime, alongside original-source and roundtrip Metal comparisons. This adds
a native reference path; it does not provide persistent MLX streams, automatic
host-code redirection, a generated Metal C++ loader, full upstream-suite parity,
or the remaining runtime adapters tracked by issue #1424.

A separate `MLX Metal host integration harness
<https://github.com/CrossGL/crosstl/blob/main/demos/integrations/mlx/METAL_HOST.md>`_
builds the pinned upstream runtime with a documented per-entry library resolver.
It runs the unchanged upstream operations module with selected translated
complex-power entries and records dispatches from MLX's own command encoder.
All 15 independently emitted entries must compile and link into one Metal
library before host execution. Source ``static`` and anonymous-namespace helpers
retain internal linkage, including materialized template helpers. Inline
definitions and implicit template instantiations retain repeatable linkage, and
generated constructors and lowered member helpers remain artifact-private.
Required native tests also link unrelated modules with identical private helper
names, verify their distinct results and exercise exported visible callables.
Declaration-only external functions remain a separate limitation: the Metal
frontend currently drops those declarations, so a translated caller cannot link
against a definition in another source file. This is tracked in
`issue #1978 <https://github.com/CrossGL/crosstl/issues/1978>`_; the selected MLX
library proof does not exercise or establish that capability.
Eight array layouts execute three datasets with 402 retained complex readbacks,
including both signed-zero branch cuts. Per-dataset traces must account for each
selected entry exactly once, and the verifier recomputes references independently
of the workload's success flags. Original MLX JIT signed-zero behavior is documented
separately and is not used as the numerical reference for these additional cases.
Other operations remain on the original backend; this is partial, explicit host
integration, not automatic C++ runtime translation or full upstream-suite parity.

The `portable MLX host adapter
<https://github.com/CrossGL/crosstl/blob/main/demos/integrations/mlx/portable_host/README.md>`_
instead builds MLX with its original Metal and CUDA backends disabled. Its
synchronous callback connects ``Arange`` for five scalar types and 30 float32 unary entries to public
DirectX/OpenGL/Metal runtime packages. Shared-buffer view primitives reuse upstream
shape, stride and ownership logic, including broadcasts, transposes, splits and
non-copying reshapes. Layout-changing contiguous conversions, reshapes, flattening
and unflattening dispatch the unchanged uint32 general-copy specialization for
float32/int32/uint32 storage. Bounds-checked, rebased source spans preserve negative
strides; integer-word transport preserves NaN payloads and subnormals. Six binary
primitives dispatch 16 unchanged entries: addition, subtraction, multiplication,
minimum and maximum for float32/int32/uint32, and float32 division. Translated
copies materialize non-row-contiguous and broadcast inputs before arithmetic;
no CPU elementwise fallback is used. Unsupported casts remain explicit errors. Other
primitives retain upstream unsupported-GPU errors.
Linux/OpenGL, Windows/Direct3D 12 and macOS/generated Metal CI build the adapted
library, require 21 unchanged upstream tests against CPU and translated GPU execution, and check
20 array-creation records plus 130 unary records against independent references,
including 8,193 consecutive binary32 inputs at and above one for inverse
hyperbolic cosine at unchanged upstream tolerances.
The verifier reconstructs the five allowed source adaptations from the pin,
requires exact bytes before and after execution, and rejects unrelated tracked
changes. Forty additional records cover view layouts, chained unary dispatch,
source preservation and exact 64-bit values. The verifier independently checks
complete numerical records, another 33 copying/source-preservation records with
1,716 exact storage words, 134 binary records with 5,274 outputs, the initial 360
nonempty dispatches, all 52 entries, and eleven required
rejection cases before publishing a schema-version-2 summary. Copy checks include
invalid allocation bounds, unsupported storage widths and excessive sizes. Each
copy dispatch retains its geometry and checks a 128-byte destination guard.
Binary checks require preserved operand words, exact integer results, zero signs,
nonfinite classification and ``rtol=2e-6, atol=1e-6`` for finite float32 outputs.
Minimum/maximum ties select the second operand as specified by the pinned source.
Binary dispatches also retain a checked 128-byte destination guard. Required native
conditional-selection controls cover operand bits and lazy branch evaluation.
Unary execution is limited to contiguous float32 inputs and 65,535 stored
elements. Source
checks do not attest to a separately supplied binary; CI retains its build log.
The Metal path compiles the generated package with warnings fatal and fast math
disabled; the original MLX Metal backend is unavailable in this build. All three
targets use explicit one-thread-per-workgroup dispatch. This is selected
host-staging coverage, not a complete translated MLX backend.

Exact Scalar Physical Resource Layouts
--------------------------------------

Source reflection records a ``scalarLayout`` only when the target-language
resource has an exact physical representation covered by the project runtime
contract. HLSL reflection supports ``StructuredBuffer`` and
``RWStructuredBuffer`` resources whose element type is a scalar or up-to-four
component vector of ``float``, ``int``, or ``uint``, plus scalar ``int64_t``
and ``uint64_t`` elements. Single-member ``cbuffer`` declarations support the
same types. GLSL reflection supports explicit ``std430`` buffer blocks
containing one scalar runtime-array member and explicit ``std140`` uniform
blocks containing one scalar member of type ``float``, ``int``, ``uint``,
``int64_t``, or ``uint64_t``.

The reflected layout records ``physicalType``, ``elementType``,
``elementSizeBytes``, ``elementStrideBytes``, ``alignmentBytes``,
``memberOffsetBytes``, ``storageLayout``, and ``runtimeSized``. It also records
``memberName`` when the source block provides one and ``blockSizeBytes`` for a
fixed scalar block. These fields are preserved through runtime packages and
native loader ABI descriptors. Descriptor-to-request validation rejects
missing, incomplete, or mismatched layouts instead of inferring a host ABI.

Native runtime allocation consumes the same physical contract. DirectX
constant-buffer views require a fixed HLSL scalar block and allocate at least
the reflected block size, rounded to the 256-byte API alignment. OpenGL
uniform buffers require a fixed ``std140`` scalar block and zero-pad the upload
to ``blockSizeBytes``; ``std430`` scalar runtime arrays retain their logical
payload size for storage-buffer readback.

Standard GLSL ``vec``, ``ivec``, ``uvec``, and ``bvec`` block members now
receive exact component widths, vector widths, and ``std140``/``std430``
alignment metadata; ``i64vec`` and ``u64vec`` use the supported 64-bit physical
tables. Runtime ``vec2`` and ``vec4`` storage arrays remain tightly packed,
while ``vec3`` records its logical element size and padded array stride
separately. The current native loader rejects padded storage vectors rather
than uploading a falsely tight layout. GLSL ``dvec`` values, HLSL 64-bit
vectors, matrices, fixed arrays, unsupported aggregate shapes, narrow or
floating-point scalar widths, implicit GLSL block layouts, arbitrary member
offsets, and multi-member HLSL blocks do not receive usable loader metadata. Those
shapes remain unresolved or fail closed when a native loader request requires
a physical layout. Native requests range-check signed and unsigned 64-bit
values and preserve them with little-endian 8-byte packing; 64-bit
specialization constants remain intentionally unsupported.

Flat homogeneous structs are supported as HLSL structured-buffer elements and
GLSL ``std430`` storage-array elements. Each may contain 1-64 members of one
supported scalar type: ``float``, ``int``, ``uint``, ``int64_t``, or ``uint64_t``.
The layout retains the actual struct name as ``physicalType``, a
``componentCount``, and ordered ``structMembers`` containing each member's
``name``, scalar ``physicalType``, and ``offsetBytes``. Structs are not relabeled
as native vectors: two float members have an 8-byte element size and 4-byte
alignment on these storage paths. Allocation sizing divides the flattened scalar
count by the member count before applying the element stride.

Dispatch validation requires unique member names, exact homogeneous types and
offsets, tight stride, complete elements, and the matching target storage class.
Nested structs, mixed scalar types, arrays, padded records, explicit member
qualifiers and duplicate declarations remain unsupported on these storage paths.
No MLX-specific type-name mapping is used.

Fixed OpenGL uniform blocks may contain mixed supported scalar/vector members.
Their ``scalarLayout`` retains the block name as ``physicalType`` and ordered
``blockMembers`` with names, actual GLSL types, component types, vector widths,
byte offsets, sizes and alignments. Packing follows the `OpenGL std140 rules
<https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf>`_ (section 7.6.2.2).
For example, ``int, vec2, vec3, float, uint`` members occupy offsets
``0, 8, 16, 28, 32`` in a 48-byte block. This is the generated target layout,
not an assumption about the original language's struct ABI.

``payloadEncoding: uint32-le-words`` identifies byte transport, not homogeneous
shader data. Inputs provide the complete little-endian block as uint32 words,
including internal and trailing padding; no numerical conversion of mixed
fields is performed. Reflection, package descriptors, contract comparison and
dispatch retain the same member records. Both loader and driver independently
validate the supplied layout against standard packing before upload. Missing,
overlapping, misaligned or contradictory records and wrong payload lengths are
rejected. Arrays, matrices, nested types and explicit member qualifiers remain
unsupported and produce incomplete-reflection diagnostics. Required Linux CI
executes translated mixed scalar and padded scalar/vector controls through the
public package and native-loader APIs.

For a single reflected HLSL or GLSL entry, ``minimumBindingSizeBytes`` records
the minimum buffer footprint proven by constant-index accesses in its mandatory
straight-line prefix. Analysis uses the existing target parsers, follows unique
helper definitions, and propagates buffer identity and bounded integer offsets.
Known single-argument HLSL and GLSL bitcasts retain accesses in their operands,
including HLSL's 16-bit storage bitcasts. Their result bits are not treated as
known integer offsets. User-defined overloads still follow the helper rules;
output-parameter overloads such as three-argument HLSL ``asuint`` stop analysis.
It stops at unresolved calls, ambiguous overloads, recursion and dynamic control
flow. Unresolved preprocessing disables this analysis; array parameter extents
alone do not establish required sizes. Missing metadata means unknown, not zero.
This is a lower bound from proven accesses, not a complete dynamic bounds check.

Packaging and native loader descriptors preserve this requirement. Runtime
preflight checks both the bound view and the provided values against the binding's
physical layout. A larger backing allocation or weaker value metadata cannot
make a short view valid. Larger usable buffers remain accepted. Malformed or
overflowing minimum byte counts and undersized views produce structured errors
before native execution. Required native CI covers a general helper-based
buffer access and rejects truncated stride buffers for the current MLX two- and
three-dimensional complex-power entries before executing the unchanged valid
numerical workloads.

The storage rules follow `DXC buffer packing
<https://github.com/microsoft/DirectXShaderCompiler/wiki/Buffer-Packing>`_ and
the `GLSL buffer layout specification
<https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html#uniform-and-shader-storage-block-layout-qualifiers>`_.

Required native CI exercises a reduced two-field transform and the unmodified
``g1_Powercomplex64`` entry from MLX commit
``9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8``. The latter runs 256 finite and
zero-base cases through translated HLSL/GLSL runtime packages; macOS compiles
and executes both original and roundtrip Metal. The numerical bound is
``5e-5 * max(1, abs(reference))`` for the complex absolute error. Reports retain
every input, output, error and bound, and buffers use nonzero sentinels to detect
missed writes. This proves the selected kernel and layout path, not full binary
family coverage, upstream MLX-suite execution, or MLX host-runtime redirection.

The separate binary-shape gate discovers and executes every one of the 15
``Powercomplex64`` entry shapes at that pin. Its 873 complex outputs exercise
scalar/vector broadcasting, two- and three-dimensional dispatch, non-contiguous
storage, zero strides, 32/64-bit index paths and partial final tiles in
four-dimensional gathers. Each shape retains its random workload and adds two
branch-cut workloads: negative-real bases with positive or negative imaginary
zero, raised to the power one half. Each dataset retains its own inputs and
readback evidence, without weakening the complex-error bound above.
Expected input locations are enumerated from logical
coordinates and strides independently of the shader's index helpers. Metal
executes original and translated kernels and separately consumes the public
runtime package, as DirectX and OpenGL do. All three Metal paths execute each
dataset, retaining 873 comparisons per path. Package construction happens once
per entry; each dataset still receives fresh buffers, artifact identity checks,
native compilation and its own device/library/dispatch evidence. DirectX also
compiles with warnings fatal, and OpenGL validates a generated SPIR-V 1.3 module
before native GLSL execution.

The gate rejects a changed source census, missing native tools, missing entry
outputs and numerical mismatches. OpenGL's explicit ``[0, 511]`` index-range
assertions apply only to these bounded workloads; they are not inferred bounds
for arbitrary tensors. A 900-second outer deadline and always-uploaded artifacts
preserve failures without converting them to optional skips. This extends
access-shape coverage for one operator and type, not every binary operation or
the upstream MLX test suite.

At pinned MLX commit ``4367c73b60541ddd5a266ce4644fd93d20223b6e``, the
``arangeuint32`` entry from ``arange.metal`` is translated to DirectX and
OpenGL, reflected, packaged, converted through the public native loader bridge,
and executed in Windows Direct3D and Linux EGL CI. With ``start = 3``,
``step = 2``, and four invocations, both readbacks are exactly
``[3, 5, 7, 9]``. This is an end-to-end proof for one scalar kernel contract;
it is not a claim of vector or aggregate layout support, full MLX runtime
integration, or MLX test-suite parity.

At current pinned MLX commit
``846d176227a0ac13d2667e58d2bb68b322109ab0``, the selected
``fft_mem_256_float2_float2`` entry also passes an entry-scoped OpenGL proof.
Six index assertions and one 256-element workgroup-access assertion
bound its host contract. The resource-helper bound applies only to one
unrebased 256-point transform with batch size one and grid ``[1,1,64]``;
it must not be reused for arbitrary tensors. Translation materializes 37 specializations from 42
reachable records, prunes 2,120 candidates, and preserves 21 reachable
function constants for deferred specialization. The deterministic GLSL is
109,547 bytes with SHA-256
``5c6fefea7315d7d091641d024aea27bbdcecf1f2f4b7a63a2db5af401e036512``;
``glslangValidator`` and ``spirv-val`` accept it, and its SPIR-V has 19 control
barriers with no group-nonuniform instruction.

The reflected ABI has two ``std430`` float32 ``vec2`` arrays at 8-byte size,
stride, and alignment plus two 16-byte ``std140`` integer blocks. The runtime
variant registry has no blocked keys and produces a verified deferred SPIR-V
request for workgroup size ``[1, 1, 64]``. Linux Mesa llvmpipe executes five
inputs: an index-1 unit impulse, seeded float32 complex data, a complex
constant, alternating real values, and a final-element complex impulse.
Across these dispatches, 2,560 scalar outputs must match analytical references
or a double-precision direct DFT at ``2e-4`` absolute and relative tolerance;
40 trailing scalar guards must remain exactly unchanged. Translation and
packaging run once; the first dispatch publishes the verified compilation cache
and the remaining four must hit it. The same controls are required in the
existing Windows native-loader jobs, without adding runners. Changed HLSL
references still require Windows numerical execution before merge.
This is one selected current-pinned plan; it does
not redirect the MLX host runtime, cover other FFT plans or dtypes, prove a
Metal round trip, or establish full backend parity.

At the same current pin, a bounded GEMV proof selects
``gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0`` for one contiguous
float32 vector-matrix product with ``M=1``, ``N=32``, and ``K=32``. The
host-derived contract fixes workgroup ``[32, 2, 1]``, subgroup width 32, and
one dispatched workgroup. Entry-scoped translation materializes the selected
GEMV and ``elem_to_loc_uint`` only. Its 8,188-byte HLSL has SHA-256
``f300bbea75b2ed9e47c29313a56f882ed848cbb93858f1347fbc97a60e167223``
and passes official DXC 1.9.2602.24 under ``cs_6_6``,
``-enable-16bit-types``, and warnings as errors. DirectX explicitly enables a
32-lane target-scoped software subgroup because physical waves need not contain
contiguous flattened ``SV_GroupIndex`` values. Logical subgroup and lane IDs
are ``SV_GroupIndex / 32`` and ``SV_GroupIndex % 32``. A 64-float
``groupshared`` array carries each shuffle between two
``GroupMemoryBarrierWithGroupSync`` calls; the source lane is validated before
addition so an extreme unsigned delta cannot wrap the scratch index, and an
out-of-range source returns the calling invocation's value. The HLSL contains
no ``WaveReadLaneAt``, ``WaveGetLaneIndex``, or physical-wave atomic allocator.
``[WaveSize(32)]`` remains the source/reflection contract without making the
reduction depend on physical lane topology.

The superseded 8,410-byte physical-wave artifact remains recorded as rejected
evidence under SHA-256
``f8f1107d0de251fd300c7a16ce6638796bd08dd2eadd8f7959e37c78d0aa170d``.
Windows workflow run 33268998061, job 99143984804 mismatched every output: the
reduction replaced logical lanes 5 through 8 with physical lanes 21 through
24, reaching maximum absolute error 1.90625. This exact substitution rules out
a tolerance adjustment or merely guarding invalid high-lane reads.

OpenGL uses two logical 32-lane software subgroups in the 64-thread workgroup.
Both target-specific fail-closed analyses admit the source's
``sm >= 1; sm >>= 1`` loop as the integral-equivalent positive-to-zero form of
``sm > 0``. DirectX also requires one bounded compute entry, concrete
width-compatible dimensions, explicit calling-invocation fallback, a supported
scalar shuffle, unambiguous helper identity, logical invocation identity, and
statically uniform control flow before artifact emission. OpenGL retains
rejection for wider bounds, mutation, nontermination, escaping control flow,
and indirect or nested helper calls. The 7,705-byte GLSL has SHA-256
``f5ef8900ee65d63a6df2818ef111f56b4f269c6366c82d82a9d97c967042f562``;
``glslangValidator`` and ``spirv-val`` accept it, and its SPIR-V contains three
control barriers with no group-nonuniform instruction.

The reflected DirectX and OpenGL ABIs each contain 15 resources, including
signed 64-bit batch strides and seven scalar argument blocks. A deterministic
binary-fraction workload compares all 32 output columns at ``1e-5`` absolute
and relative tolerance. Linux Mesa llvmpipe executes the software-subgroup
artifact in required mode, and Windows CI requires the same workload through
Direct3D 12 WARP. This proof covers one host-valid current-corpus entry; gather,
wide, batched, axpby, and remaining GEMV entries, MLX host redirection, selected
Metal compilation, and the full MLX suite remain outside the claim. It does
not change the separate historical 224-entry aggregate compiler gates.

The same current pin has a bounded MXFP4 quantize/dequantize contract for
``mxfp4_quantize_dequantize_float_gs_32_b_4_hgs_false``. Host provenance from
``quantized.cpp`` and ``fp_quantized.h`` fixes float32 input, group size 32,
four payload bits, no global scale, workgroup ``[32, 1, 1]``, and one dispatched
workgroup. Entry-scoped translation materializes only that specialization. The
9,123-byte HLSL has SHA-256
``3fe38e171ba8c8ea1adfc8efad20b242ca02dd05e1a5a53a9b9d1e18459d8c7d`` and
passes DXC under ``cs_6_6``, ``-enable-16bit-types``, and warnings as errors.
Its 4,716-byte DXIL uses the explicit DirectX-only
``project.source_options.metal.target_options.directx.widen_native_float16``
mode. Source ``as_type<float16_t>(uint16_t)`` reconstructs its payload directly
as float32 with integer IEEE-754 masks, while logical ``float16_t`` locals,
parameters, and returns stay widened through arithmetic, sign application, and
the consuming conversion. DXIL contains ``uitofp i32`` and ``fmul float`` but
no ``LegacyF16ToF32``, ``half``, ``fptrunc``, or ``fpext``. The default native
binary16 contract remains exact ``asfloat16``/``asint16``/``asuint16`` and is
unchanged when this target-scoped option is absent.

Four Windows failures bound this contract. The 7,809-byte HLSL under SHA-256
``3591e38d20a612b4061fe3154ef0ea3deb035283294fbd27376ef90627569361``
produced numeric ``uitofp i16 ... to half``; workflow run 33271117475, job
99149649480 collapsed all 28 nonzero values to signed zero on WARP. Exact
``bitcast i16 ... to half`` produced same-size HLSL under SHA-256
``4e8044758d65b6b2c189092ce56fff3c5ba7948221883de490c1a4b9c5563352``,
but run 33272842347, job 99154326814 consumed the constructed subnormal in
``fmul half`` and produced the same result. Moving arithmetic to float32
produced 7,909-byte HLSL under SHA-256
``938ca6fac1c47ea633453836b5d76833c294853bb92d6a410a2c4772dd7fa627``;
``dx.op.legacyF16ToF32`` still failed in run 33274360343, job 99158370210.
Integer decoding produced 9,240-byte HLSL under SHA-256
``936088a24a6b575e50dc97e16a4c0dca63a76200ddd94d5211e4bf312fec1625``,
but the remaining ``fptrunc float to half``, half sign operation, and
``fpext half to float`` returned the identical 28 signed zeros in run
33275550062, job 99161501105. Removing every half instruction produced a
9,118-byte artifact under SHA-256
``7afdc612f9091ae47abca8c4fd9d2171e8ea42c6539e02a40bbad2de7d1a7c6a``,
but run 33277494856, job 99166677942 still returned the same signed zeros.
DXIL then exposed the actual remaining defect: absent Metal/C++ scalar integer
promotion reduced ``uint16_t(bits) << 23`` to a 16-bit shift by 7 plus an
``0xffff`` mask. The corrected HLSL emits ``int(uint16_t(bits)) << 23`` and
DXIL shifts the 32-bit value by 23 before ``asfloat``, with no native-half
operation anywhere in the selected path.

The 9,571-byte explicit-software-subgroup GLSL has SHA-256
``cbbe989c40317c04ffe915f1f314f55db8896edfd38f04ad4b8882be53b2a4da``;
``glslangValidator`` and ``spirv-val`` accept its three-barrier SPIR-V, which
contains no group-nonuniform instruction. GLSL widens binary16 values to
float32, so the same source bitcast preserves the exact low 16-bit payload
through ``unpackHalf2x16`` rather than a float32 bitcast; inverse forms use
``packHalf2x16`` with exact low-bit extraction.

Exact scale semantics require the source ``fp8_e8m0(float)`` constructor
factory before conversion back to float. Qualified ``metal::round`` maps to the
portable math intrinsic, while OpenGL ``metal::isfinite`` and ``signbit`` use
single-evaluation IEEE-754 bit tests. Read-only private scalar-to-struct views
are accepted only for exact one-member layouts. Unsupported predicate types,
receiver mutation, writes, and unresolved constructor branches remain
fail-closed.

The reflected data ABI is float32 input at binding 0, an inert declared
``global_scale`` input at binding 1, and float32 read-write output at binding 2.
DirectX also uses ``b0`` for generated dispatch input, legally distinct from
``t0`` in the HLSL register namespaces. The real host omits ``global_scale``
for this specialization; the generic loader allocates it because reflection
retains the declaration, and the selected code contains no read. A 32-element
workload uses only exactly representable FP4 E2M1 values with maximum magnitude
6. The scale divisor is 6 and the MX scale is exactly 1, so Windows WARP and
Linux Mesa required-mode tests demand bit-exact float32 readback at zero
tolerance. Other quantized entries and parameter families, MLX host redirection,
selected-entry Metal compilation, and full-suite parity remain outside this
bounded proof.

At current pinned MLX commit
``846d176227a0ac13d2667e58d2bb68b322109ab0``, a bounded arg-reduce proof
selects ``argmin_float32`` and ``argmax_float32`` for two axis-32 rows. The
checked host dispatch formula produces workgroups ``[32, 1, 1]`` and dispatch
``[1, 2, 1]`` with subgroup width 32. Signature-aware source instantiation
materializes the scalar ``elem_to_loc<int64_t>`` helper rather than its
``uint3`` overload. DirectX generation explicitly sets
``project.source_options.metal.target_options.directx.relative_wave_shuffle_out_of_range``
to ``"self"`` for these artifacts. Relative source lanes outside the wave then
retain the calling lane's value; generated helpers select a proven in-range
source before an unconditional ``WaveReadLaneAt``. This deterministically
refines source undefined behavior without changing in-range shuffle semantics.
The default policy remains ``"undefined"`` so unrelated project artifacts are
unchanged. The generated HLSL artifacts pass official DXC 1.9.2602.24 under
``cs_6_6`` with ``-enable-16bit-types`` and warnings as errors. OpenGL uses the
explicit software subgroup and admits direct shuffle-helper calls only inside
a proven canonical workgroup-uniform halving loop. Both GLSL modules pass
``glslangValidator`` and ``spirv-val``, contain five control barriers, and
contain no group-nonuniform SPIR-V instruction.

The reflected runtime contract includes exact signed and unsigned 64-bit
scalar resources: float32 input, uint32 output, int32 shape, int64 stride, and
uint64 size buffers or scalar blocks. Linux arm64 Mesa EGL executes rows with
tied extrema and reads back argmin indices ``[5, 7]`` and argmax indices
``[3, 2]``, proving lowest-index tie behavior; Windows CI requires the same
workloads through Direct3D 12 WARP. This bounded proof does not unblock the
24-entry aggregate DirectX/OpenGL artifact, which remains fail-closed with
``project.translate.workgroup-size-entry-ambiguous``. Other axis sizes,
dtypes, and entries, MLX host redirection, and the full MLX test suite remain
outside the claim. Entry-scoped Metal also remains unavailable because the
per-entry workgroup specialization rule fails explicitly with
``project.translate.workgroup-size-rule-unsupported-target``; this proof does
not claim a Metal round trip.

The same current pin now has a bounded one-pass scaled-attention proof for
``sdpa_vector_float_64_64``. The checked host contract selects batch/head/query
counts of one, key length 4, dimensions ``D=64`` and ``V=64``, scale ``0.125``,
and no mask, causal mode, or sinks. It fixes ``[1024, 1, 1]`` with 32 logical
subgroups and one dispatched workgroup. Function constants 20 through 25 are
all false; two-pass-only ID 26 is not part of this artifact. The exact HLSL is
8,721 bytes with SHA-256
``003c8b9e85bad7363bae2e3d80380d979cbe0b8988d0d98751131c3acfbff6b6``
and passes DXC 1.9.2602.24 under ``cs_6_6``, ``-enable-16bit-types``, and
warnings as errors. Its 32 physical waves receive unique, wave-uniform IDs
through the synchronized allocator; no lane-varying flattened-index quotient
remains.

OpenGL uses an explicit 32-lane software artifact across the full 1,024-thread
workgroup. The subgroup-ID-strided runtime loop is synchronized round by round;
inactive subgroups supply typed reduction identities so all barriers remain
uniform. Its 12,089-byte GLSL has SHA-256
``9b7cb7dc9a76b9fb93c30fd93d13ad639f5493f60fd97b965514db0fe6b4840b``.
The validated SPIR-V has nine control barriers, six false specialization
constants, local size ``1024 1 1``, and no group-nonuniform instruction.

The native-loader ABIs contain 19 DirectX and 18 OpenGL resources. Deferred
optional resources receive placeholders, including uint32 physical storage for
``bmask``. DirectX concretizes the six constants and executes through WARP;
OpenGL builds a verified deferred compilation request, specializes the six
constants, and executes through Mesa surfaceless EGL. Both compare all 64
outputs with a stable CPU reference at ``2e-4`` absolute and relative
tolerance. The local Mesa proof measured maximum absolute error
``4.082320426146424e-08`` and maximum relative error
``4.2163276126605175e-06``. Masked, causal, sinks, two-pass and full-attention
paths, other dimensions and dtypes, MLX host redirection, selected Metal
round-trip validation, and the full MLX suite remain outside this bounded
claim. The historical 42-entry aggregate DirectX/OpenGL run remains fail-closed
because it does not consume this entry-scoped contract; it is not evidence that
a bounded attention runtime proof is absent.

At current pinned MLX commit
``846d176227a0ac13d2667e58d2bb68b322109ab0``, a bounded LayerNorm VJP proof
selects ``vjp_layer_normfloat32`` for one axis-32 row with function constant
``has_w=true``, workgroup and subgroup width 32, and eight reflected resources.
DirectX materializes the constant and executes the generated HLSL through
Direct3D 12 WARP. OpenGL retains constant ID 20 in GLSL, lowers the reductions
to an explicit software subgroup, derives a verified deferred-compilation
request, compiles to SPIR-V, specializes ``has_w=true``, and executes through
Mesa EGL. Both paths compare all 32 input-gradient and 32 per-row
weight-gradient values. The one-row boundary means the per-row weight-gradient
temporary is also the final host-reduced result. This proof excludes the
separate bias-gradient dispatch, multi-row weight reduction, other dtypes and
axis sizes, MLX host redirection, and the full MLX test suite.

The same current pin also has a bounded RMSNorm VJP proof for
``vjp_rmsfloat32`` with one axis-32 row, ``has_w=true``, one 32-thread
workgroup, and ten reflected resources. DirectX concretizes the function
constant and executes HLSL through WARP. OpenGL retains constant ID 20, lowers
four reductions through the explicit software subgroup, and admits the
kernel's runtime row loop only after proving its initializer and bound
workgroup-uniform. The deferred path compiles GLSL to SPIR-V, specializes the
constant, verifies the interface, and executes through Mesa EGL. Both targets
compare all 32 input-gradient and 32 one-group weight-gradient values. The
one-row/one-group boundary makes the group-local weight gradient equal to the
final host reduction; multi-row reduction, other dtypes and axes, MLX host
redirection, and the full MLX test suite remain outside this proof.

The current pin additionally has a bounded Softmax proof for
``block_softmax_float32``. Its checked dispatch contract applies the pinned
host formula ``32 * ceilDiv(ceilDiv(axisSize, 4), 32)`` to two rows of axis 32
and one row of axis 2049, producing workgroups ``[32, 1, 1]`` and
``[544, 1, 1]``. Both DirectX artifacts retain ``WaveSize(32)``. OpenGL keeps
the default guarded hardware-subgroup artifacts and separately packages
explicit 32-lane software-subgroup artifacts; the wide artifact partitions 544
invocations into 17 logical subgroups and uses typed inactive-lane identities
for masked maximum and sum reductions. Official DXC accepts both HLSL artifacts
under ``cs_6_6`` with ``-enable-16bit-types`` and warnings as errors. The
32-thread entry is provably one wave and retains the zero-ID quotient fast path;
the 544-thread entry allocates one uniform ID for each physical wave through a
workgroup-synchronized counter. Both software modules pass
``glslangValidator`` and ``spirv-val``, contain 11 control barriers, and contain
no group-nonuniform SPIR-V instruction.

Windows CI requires both workloads to execute through Direct3D 12 WARP, and
Linux CI requires both to execute through surfaceless Mesa EGL. Every output is
compared with a stable float32 CPU Softmax reference at ``5e-5`` absolute and
relative tolerance. This evidence covers only the two finite float32 block
workloads: axis sizes above 4096, half and bfloat16 entries, MLX host
redirection, and the full MLX test suite remain outside the claim. Selected
entry-scoped Metal generation also remains unavailable and fails explicitly
with ``project.translate.entry-point-target-unsupported``; this bounded proof
does not claim a Metal round trip.

Build a versioned native loader ABI descriptor and optional C declarations for
one ready load unit:

.. code-block:: bash

   python -m crosstl native-loader-abi \
     crosstl-runtime-package/runtime-loader-manifest.json \
     --load-unit copy:directx \
     --output copy.directx.abi.json \
     --declarations-output copy.directx.abi.h \
     --execution-output copy.directx.native-loader-execution.h \
     --target-adapter-output copy.directx.native-loader-adapter.hpp

``native-loader-abi`` selects exactly one ready load unit. ``--load-unit`` is
optional only when the input manifest contains one unit. The command emits a
schema-v1 ``crosstl-native-loader-abi-descriptor`` JSON document containing
the target entry point, artifact identity and hash, source identity and remap,
binding namespaces and coordinates, access modes, scalar layout,
specialization constants, and provenance. A blocked unit, incomplete host
interface, malformed artifact identity, ambiguous selection, or invalid
binding coordinate produces a structured diagnostic instead of declarations.

``--declarations-output`` writes deterministic C declarations for the same
descriptor. The header contains a guarded, versioned ABI type contract and
immutable unit and binding descriptor data. The declarations compile as C11
and C++17 and can be included for more than one load unit without redefining
the shared ABI types. They describe what downstream host integration must
load and bind; they do not execute the unit.

``--execution-output`` writes a deterministic, allocation-free C11/C++17
execution wrapper for the selected unit. The header includes the declaration
contract and adds request, adapter-callback, result, and structured-error
types plus a unit-specific ``*_execute`` function. Before consulting the
adapter, that function validates the ABI version, target, exact binding
identity and coordinates, access modes, specialization identities and types,
payload presence, and dispatch geometry.

After validation, the wrapper loads the artifact, applies specialization
values to that artifact, creates the pipeline, binds resources in descriptor
order, dispatches, synchronizes, and reads back writable resources. It then
releases resources in reverse binding order, destroys the pipeline, and
unloads the artifact on both success and failure paths. The wrapper uses
fixed-size stack storage derived from the descriptor and performs no heap
allocation.

The caller owns the execution request, binding and specialization payloads,
readback destinations, and adapter context for the duration of the call. The
adapter owns each native artifact, pipeline, and resource handle that its
callbacks return; the wrapper passes those handles to the corresponding
unload, destroy, and release callbacks. A nonzero callback return value is
preserved as ``adapter_status`` with the failing phase and binding or
specialization index. ``error`` records the primary execution failure, while
``cleanup_error`` records the first cleanup failure without replacing an
earlier primary failure. A cleanup failure becomes the primary error only
when all preceding execution phases succeeded.

``--target-adapter-output`` writes the deterministic C++17 reference adapter
for the selected target. Include the unit execution header before this adapter
header. The adapter fills the shared callback table rather than replacing the
unit-specific validation wrapper, so request validation, execution order,
structured errors, and cleanup behavior remain defined by the common ABI.
Direct3D 12 and OpenGL compute have reference adapters. A target without a
reference adapter produces a structured
``project.native-loader-target-adapter.target-unsupported`` diagnostic.

Build descriptors and declarations for every ready unit in one operation:

.. code-block:: bash

   python -m crosstl native-loader-abi-package \
     crosstl-runtime-package/runtime-loader-manifest.json \
     crosstl-native-loader-abi

``native-loader-abi-package`` validates every unit and generates every
available target adapter before writing output. It emits target-scoped
descriptor, declaration, and execution header files, one adapter header per
supported target, plus ``native-loader-abi-package.json`` and prints that
package manifest as JSON.
Each packaged unit records ``executionABIPath`` and ``executionABIHash``;
the generated-file inventory identifies execution headers as
``native-loader-execution-abi`` and reference adapters as
``native-loader-target-adapter``. The schema-v3 ``targetAdapters`` array records
the target, availability, generated path, and SHA-256 hash. Targets without a
reference implementation remain in that array with
``reason: target-adapter-unavailable`` rather than being silently omitted.

Schema-v3 packages also publish ``runtime/runtime-variant-registry.json`` and
record its file hash, registry identity, and exact variant count in
``runtimeVariantRegistry``. When that registry is ready, the package verifies
and copies every referenced translated artifact, recording each as
``runtime-target-artifact``; descriptor byte sizes and hashes remain available
for independent verification. A metadata-only loader manifest that lacks the
provenance required for exact lookup remains valid, but marks the registry and
native header unavailable instead of claiming dispatch readiness.

When every registry target has a reference adapter, the package also emits
``native-runtime-variant-registry.hpp``. This deterministic C++17 header maps
canonical runtime variant keys to unit execution wrappers and retains the exact
target, entry point, workgroup size, subgroup width, and specialization
payloads selected during packaging. Packages containing targets without a
reference adapter keep the JSON registry but mark the native header unavailable.
A ready OpenGL GLSL-source variant that still carries specialization constants
also keeps its JSON registry ready, copied artifact, and exact lookup key, but
marks the native header unavailable with reason
``specialization-requires-deferred-compilation``. The C++ header cannot express
that compile-then-specialize step; callers instead derive and execute the
verified deferred SPIR-V compilation request. Other registry-generation errors
remain fatal and are not converted into this fallback.

The same operations are available through the public project API:

.. code-block:: python

   from crosstl.project import (
       NATIVE_LOADER_ABI_PACKAGE_KIND,
       NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
       NATIVE_LOADER_ABI_PACKAGE_VERSION,
       NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH,
       NATIVE_RUNTIME_VARIANT_REGISTRY_PATH,
       NATIVE_LOADER_TARGET_ADAPTER_KIND,
       NATIVE_LOADER_TARGET_ADAPTER_VERSION,
       NativeRuntimeVariantRegistryError,
       build_native_loader_abi_descriptor,
       build_native_loader_abi_package,
       build_runtime_variant_dispatch_request,
       generate_native_loader_declarations,
       generate_native_loader_execution_abi,
       generate_native_loader_target_adapter,
       generate_native_runtime_variant_registry,
       native_loader_target_adapter_targets,
   )

   descriptor = build_native_loader_abi_descriptor(
       loader_manifest,
       load_unit_id="copy:directx",
   )
   declarations = generate_native_loader_declarations(descriptor)
   execution_abi = generate_native_loader_execution_abi(descriptor)
   target_adapter = generate_native_loader_target_adapter(
       descriptor["target"]
   )

   package = build_native_loader_abi_package(
       "crosstl-runtime-package/runtime-loader-manifest.json",
       "crosstl-native-loader-abi",
   )

``build_native_loader_abi_descriptor`` validates and normalizes one loader
unit before returning deterministic JSON-compatible metadata.
``generate_native_loader_declarations`` validates that descriptor before
rendering the C representation. ``generate_native_loader_execution_abi``
validates the same descriptor before rendering the executable callback
wrapper. ``generate_native_loader_target_adapter`` renders the target callback
implementation for a supported canonical target, while
``native_loader_target_adapter_targets`` reports the available target names.
``build_native_loader_abi_package`` validates all load units and adapter output
before writing, then emits one descriptor, declaration header, and execution
header per unit, one reference adapter per supported target, and a
deterministic package manifest. The manifest records content hashes, source
loader-manifest identity, generated paths, target, adapter, and unit counts,
and uses the exported
``NATIVE_LOADER_ABI_PACKAGE_KIND``, ``NATIVE_LOADER_ABI_PACKAGE_VERSION``, and
``NATIVE_LOADER_ABI_PACKAGE_MANIFEST`` constants for its kind, schema version,
and file name. A blocked, incomplete, or duplicate unit prevents package
publication rather than producing a partially described package.

The generated Direct3D adapter owns its device, queue, fence, pipelines, and
resource allocations within an explicit caller-created context. The generated
OpenGL adapter consumes a caller-owned current desktop context and an explicit
function table; it does not create a window-system context or choose EGL, GLX,
WGL, or another loader. Both adapters reject unsupported artifact, resource,
layout, and capability contracts through nonzero adapter statuses. They are
reference implementations for one unit execution lifecycle, not repository
schedulers or host-runtime rewrites. Host application rewriting, primitive
selection, graph policy, full MLX runtime integration, and MLX test-suite
parity are not provided or claimed.

The Direct3D 12 adapter accepts packaged HLSL compute source and DXIL
containers. HLSL source compilation uses the DXC API with shader model 6.2 and
native 16-bit types enabled. Hosts that compile HLSL source need the official
``dxcapi.h`` header; ``dxcompiler.dll`` and ``dxil.dll`` must be discoverable
at execution time. DXIL-only hosts do not need the DXC API, but cannot apply
source specializations. Generated HLSL source specialization requires both the
reflected constant name and numeric ID so the adapter can replace the exact
CrossTL fallback declaration before passing a numeric definition to DXC.
Structured-buffer SRV and UAV bindings and constant-buffer CBV bindings are
supported when the descriptor includes a compatible scalar layout. Other
resource kinds fail closed.

The OpenGL adapter accepts GLSL compute source with entry point ``main`` and
OpenGL SPIR-V compute binaries. GLSL source can run on a current OpenGL 4.3
desktop context but does not support specialization. SPIR-V specialization
requires OpenGL 4.6 or caller-confirmed ``GL_ARB_gl_spirv`` support and the
matching ``glShaderBinary`` and ``glSpecializeShader`` entry points. The
adapter supports set-zero shader-storage and uniform-buffer bindings; nonzero
sets and texture, image, sampler, scalar-uniform, and shared-allocation
contracts fail closed.

Native CI translates reduced source fixtures through CrossTL, verifies
packaged artifact hashes, binds distinct input and output buffers, applies one
specialization, dispatches on Direct3D 12 and surfaceless OpenGL, and compares
deterministic readback with the expected values. These checks prove the
generated adapter lifecycle for the reduced contracts. They do not establish
semantic parity for every translated kernel or execute an upstream
repository's complete test suite.

For a complete DirectX or OpenGL compute descriptor, the public project API can
construct and preflight the backend-neutral runtime request consumed by the
native parity adapters:

.. code-block:: python

   import json
   from pathlib import Path

   from crosstl.project import build_native_loader_dispatch_request

   package_root = Path("crosstl-native-loader-abi")
   descriptor = json.loads(
       (package_root / "descriptors/directx/copy.abi.json").read_text()
   )
   request = build_native_loader_dispatch_request(
       descriptor,
       package_root,
       input_values={
           "input_values": {
               "dtype": "float32",
               "shape": [4],
               "values": [1.0, 2.0, 3.0, 4.0],
           },
           # The descriptor reflects this binding as read_write.
           "output_values": {
               "dtype": "float32",
               "shape": [4],
               "values": [1.0, 2.0, 3.0, 4.0],
           },
       },
       output_values={
           "output_values": {
               "dtype": "float32",
               "shape": [4],
               "values": [2.0, 4.0, 6.0, 8.0],
           }
       },
       dispatch_geometry={"workgroupCount": [1, 1, 1]},
       specialization_values={3: 4},
       expected_target="directx",
   )

``build_native_loader_dispatch_request`` supports compute-stage DirectX HLSL
and OpenGL GLSL artifacts. It validates the descriptor, requires exact
reflected binding names, verifies the artifact size and SHA-256 digest inside
the package root, validates specialization values and dispatch geometry, and
returns a preflighted ``RuntimeExecutionRequest``. Buffer bindings require a
complete, tightly packed 32-bit scalar layout; missing or ambiguous physical
layout metadata is a structured error rather than an inferred ABI.

The returned request also carries a frozen ``RuntimeArtifactIdentity`` copied
from the validated descriptor's byte size, SHA-256 digest, and unit ID. Native
execution uses this pinned record instead of rereading identity fields from the
request's public artifact mapping. Later mutation of that mapping therefore
cannot replace the expected identity used at execution. The record pins
identity metadata only; it does not freeze other artifact metadata or the file
at the artifact path.

An exact binding name may appear in both ``input_values`` and ``output_values``
only when the descriptor reflects that resource as ``read_write``. In the
example above, the input payload initializes one native allocation and the
expected output contract marks that same allocation for readback and
comparison. Request construction requires the two roles to agree on dtype and
shape and validates both against the complete reflected physical scalar layout.
An overlap for another access mode, or an incompatible contract, produces a
structured diagnostic.

This API prepares one request and its native resource contract. Repository
scheduling, host application rewriting, and full MLX test-suite parity remain
outside its scope. Generated descriptors for the exact HLSL and GLSL scalar
forms described above carry the complete physical layout into request
construction; unsupported resource shapes remain rejected rather than
inferred. The pinned MLX ``arangeuint32`` proof establishes native DirectX and
OpenGL execution for that one contract, not full MLX runtime integration or
numerical parity across the MLX test suite.

Build a deterministic runtime variant registry from either a ready runtime
package or loader manifest:

.. code-block:: bash

   crosstl runtime-variant-registry \
     crosstl-runtime-package/runtime-loader-manifest.json \
     --output runtime-variant-registry.json

Runtime variant registries emit a schema-v1
``crosstl-runtime-variant-registry`` JSON document whose runtime variant key
schema is version 2. Each ``variants`` entry is indexed by a canonical
``crosstl-rvk2:`` key: URL-safe base64 without padding over canonical JSON
containing the source unit and source entry, target and target profile, the
selected binding-interface entry point's execution identity, type and value
template arguments, specialization constant IDs and values, and defines. The
``execution`` identity contains ``workgroupSize`` and ``subgroupWidth``; each
field remains ``null`` when the selected entry does not provide an exact value.
Unselected entry points and project-level aggregate metadata do not affect the
key. Key fields are sorted before encoding, registry records and target
summaries are ordered by key, and ``registryHash`` covers the key schema and
records. Equivalent input records therefore produce the same registry
regardless of package or loader record order.

Each registry record preserves source and target names separately and maps the
exact key to the target artifact path, format, hash and byte size, target entry
point, binding resources and ordinary constants, pipeline specialization
constants, and translation and source provenance. Inputs use closed package
and loader field sets. Malformed schemas fail before records are emitted;
duplicate keys and keys with conflicting artifacts or metadata are diagnosed
and rejected. Package inspection hash or size failures remain explicit
``stale`` records, and loader blockers remain ``blocked`` records. Both are
listed as available exact keys but have ``lookup.eligible`` set to false.

The public ``build_runtime_variant_registry`` API builds the document,
``encode_runtime_variant_key`` and ``decode_runtime_variant_key`` expose the
key contract, and ``lookup_runtime_variant`` performs exact lookup with the
available keys included in not-found diagnostics. Lookup validates the closed
registry schema, ``registryHash``, canonical key-to-record identity, and record
eligibility before returning a ready artifact. Modified or malformed registry
records fail as invalid rather than participating in selection. When the
non-execution identity matches but the requested execution identity does not,
the diagnostic reports ``requestedExecution`` and the exact
``availableExecutionAlternatives`` with their keys, status, workgroup size,
and subgroup width. There is no fallback to one of those alternatives. Legacy
``crosstl-rvk1:`` keys are rejected with guidance to regenerate both the key
and registry. This remains deterministic selection and packaging metadata;
target compilation, deferred compilation, host runtime dispatch, device
execution, and numerical parity are not established by the registry.

Exact native variant dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A schema-v3 native loader package binds the exact JSON registry to the
descriptors, execution wrappers, translated artifacts, and optional native
registry header by SHA-256 identity. Build a preflighted request from one
canonical key:

.. code-block:: python

   import json
   from pathlib import Path

   from crosstl.project import build_runtime_variant_dispatch_request

   package_root = Path("crosstl-native-loader-abi")
   package = json.loads(
       (package_root / "native-loader-abi-package.json").read_text()
   )
   registry = json.loads(
       (package_root / package["runtimeVariantRegistry"]["path"]).read_text()
   )
   key = registry["lookup"]["readyKeys"][0]

   request = build_runtime_variant_dispatch_request(
       registry,
       key,
       package_root,
       input_values,
       output_values,
       {"workgroupCount": [1, 1, 1]},
   )

``build_runtime_variant_dispatch_request`` performs exact lookup and rejects a
registry that does not belong to the package. It verifies the packaged registry
hash and, when a native header is published, the native header hash. It also
checks descriptor size and hash, source and target provenance, artifact
identity, entry point, workgroup size, specialization identity and value, and
translated artifact bytes before delegating to
``build_native_loader_dispatch_request``. Runtime callers provide resource
values and dispatch counts but cannot replace the selected workgroup size or
specialization values.

Exact variant selection extends the request's frozen artifact identity with the
selected variant name and canonical variant key. Native execution retains the
pinned descriptor size, hash, and artifact ID plus those variant fields even if
the request's public artifact mapping is later mutated.

At native runtime setup, the DirectX and OpenGL adapters read the selected
translated artifact once and compute its byte size and SHA-256 digest. When the
request records both values, a mismatch produces a structured setup diagnostic
with the expected and observed identities before target validation, compilation,
or runtime loading. Partial identity metadata and malformed fields fail closed.
An artifact without either recorded field is still captured, but the result is
reported as ``not-recorded`` rather than verified. Artifact read and snapshot
materialization failures also produce structured setup diagnostics. The
captured bytes are materialized in an adapter-owned temporary directory, and
subsequent source validation and target compilation use that snapshot instead
of reopening the original artifact path. Source-format loading reads the
snapshot, while compiled modules are produced from it. Replacing the original
file after capture therefore cannot change the bytes consumed by that
execution.

This snapshot isolates one execution from later changes to the original path;
it is not a filesystem security boundary. It does not prevent another process
with access to the temporary directory from changing the snapshot, attest
compiler output, or verify artifacts consumed by a caller-supplied runtime
outside these DirectX and OpenGL adapters.

The generated C++17 header exposes
``crosstl_native_runtime_variant_lookup``,
``crosstl_native_runtime_variant_make_request``, and
``crosstl_native_runtime_variant_execute``. Execution accepts only a pointer
returned by the generated registry, checks ABI and target identity, and compares
the exact specialization payloads before invoking the selected unit wrapper.
DirectX HLSL and OpenGL SPIR-V specializations are supported according to the
target adapter contracts; precompiled DXIL specialization and GLSL source
specialization remain fail-closed.

This bridge selects and prepares one native compute dispatch from an existing
ready variant. It does not synthesize a missing variant, perform deferred target
compilation, rewrite a repository's host runtime, schedule multiple kernels,
translate framework control flow, or establish full MLX test-suite parity.

Bounded deferred native compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Repositories that select from a finite set of source variants can compile one
fully resolved variant after packaging without accepting arbitrary source
generation or compiler arguments. This path is defined by the schema-v1
``crosstl-native-deferred-compilation-request`` contract and the public project
API:

.. code-block:: python

   from crosstl.project import (
       build_native_deferred_compilation_dispatch_request,
       build_native_deferred_compilation_request,
       compile_native_deferred_compilation_request,
       execute_native_deferred_compilation_request,
   )

   request = build_native_deferred_compilation_request(
       source,
       includes,
       target,
       variant,
       expected_loader_descriptor,
   )
   compilation = compile_native_deferred_compilation_request(
       request,
       package_root,
       cache_root,
   )
   dispatch = build_native_deferred_compilation_dispatch_request(
       compilation,
       input_values,
       output_values,
       {"workgroupCount": [1, 1, 1]},
   )

An exact ready runtime variant can form the same closed request directly from a
schema-v3 native loader package:

.. code-block:: python

   from crosstl.project import (
       build_runtime_variant_deferred_compilation_request,
       execute_native_deferred_compilation_request,
   )

   request = build_runtime_variant_deferred_compilation_request(
       registry,
       key,
       package_root,
   )
   result = execute_native_deferred_compilation_request(
       request,
       package_root,
       cache_root,
       input_values,
       output_values,
       {"globalSize": [count, 1, 1]},
   )

``build_runtime_variant_deferred_compilation_request`` performs the same exact
lookup and package-to-registry identity checks as native variant dispatch. It
then verifies the selected source artifact bytes, loader descriptor size and
digest, source and target identity, entry point, execution configuration, and
specialization interface before deriving the request. The canonical variant
key, type and value arguments, compile definitions, specialization values, and
execution identity are copied from the selected registry record rather than
accepted from the runtime caller.

This bridge accepts DirectX HLSL source and OpenGL GLSL source records. Binary
DXIL or SPIR-V records remain ahead-of-time artifacts and cannot be treated as
compiler source. Native CI selects the exact record, compiles it with DXC or
``glslangValidator``, and dispatches the resulting binary on Direct3D 12 or a
software OpenGL device. The current package bridge emits an empty include list,
so the selected target source must be self-contained. A source artifact with
literal includes fails during verified materialization until the native loader
package carries a complete include closure.

``source`` and every ``includes`` record carry a portable package path, source
format, byte size, and SHA-256 digest. ``target`` fixes the backend, profile,
compute entry point, and binary output format. ``variant`` carries the canonical
runtime variant key, finite type and value arguments, compile definitions,
specialization values, workgroup size, and optional subgroup width. The request
also pins the expected native loader ABI descriptor by path, size, and digest.
A canonical ``requestHash`` covers the complete closed-schema document.
Unknown fields, arbitrary compiler flags, unresolved or non-finite values,
source/target format mismatches, and inconsistent variant keys fail contract
validation.

Before a compiler is consulted,
``materialize_native_deferred_compilation_inputs`` verifies the source, every
include, and the loader descriptor against their recorded size and digest. It
rejects symbolic links, path escapes, portable case collisions, non-regular
files, undeclared or unreachable includes, ambiguous angle includes, and
dynamic include operands. Only a complete literal include closure is copied
into an isolated source tree; the compiler receives include directories derived
from that verified closure.

Compilation then validates the loader descriptor against the request, including
target, stage, entry point, source identity, execution configuration, and exact
specialization identity and value rules. It separately reflects the source and
compares entry-point execution, bindings, scalar layout, constants, and the
OpenGL specialization interface with the descriptor. Any drift fails before
target compilation or dispatch. Source reflection currently evaluates the
primary translation unit. Include files remain byte-verified compiler inputs,
but declarations visible only after preprocessing an include are not folded
into host-interface reflection in this path and therefore cannot satisfy the
interface check.

DirectX requests compile packaged HLSL compute source to DXIL with ``dxc``.
OpenGL requests compile packaged GLSL compute source to SPIR-V with
``glslangValidator``. Compiler commands are derived from the validated request;
callers cannot append arbitrary flags. The schema-v1 compilation result records
the tool name, reported version, executable hash, probe and compile commands,
source and include provenance, compiler diagnostics, and the compiled output's
format, size, and SHA-256 digest.

Successful outputs use a deterministic cache key derived from the complete
request hash and the toolchain name, version, and executable hash. Cache entries
retain the expected interface identity and output identity, are published
atomically, and are revalidated on lookup. The toolchain executable is also
rechecked before a cached or newly compiled result is returned. Compiler
failures, missing or malformed output, interface drift, and partial cache
entries are never published as successful cache entries.

``build_native_deferred_compilation_dispatch_request`` converts a successful
compiled result into the same preflighted ``RuntimeExecutionRequest`` used by
the native loader path. The result retains the normalized source request;
dispatch revalidates its hash and rejects target, specialization, or descriptor
provenance drift. It preserves exact binding, specialization, execution, and
compiled-artifact identity. ``execute_native_deferred_compilation_request``
performs compilation or exact cache reuse and then dispatches through the
DirectX or OpenGL native runtime adapter. Callers still supply resource values
and dispatch counts. An injected OpenGL context remains caller-owned when
``OpenGLComputeRuntime`` is constructed with ``release_context=False``.

This is a bounded compute compilation and dispatch contract. It does not
generate arbitrary source, rewrite host applications or runtime frameworks,
choose repository scheduling policy, or provide framework-specific runtime
integration. It does not claim complete MLX kernel coverage, MLX runtime
integration, or parity with the MLX test suite. The implemented scope is
tracked in `GitHub issue #1854
<https://github.com/CrossGL/crosstl/issues/1854>`_.

Build deterministic host loader scaffold metadata from a runtime loader
manifest:

.. code-block:: bash

   python -m crosstl scaffold-host-loaders \
     crosstl-runtime-package/runtime-loader-manifest.json \
     --scaffold-dir crosstl-host-loaders \
     --format text

Host loader scaffolds emit a ``crosstl-runtime-host-loader-scaffolds`` JSON
document and write a small metadata bundle with ``host-loader-scaffolds.json``,
``HOST_LOADERS.md``, and one target-scoped ``*.loader.json`` file for each
ready load unit. The scaffold files preserve the loader manifest's artifact
paths, source-remap handoff paths, host interface metadata, required tools,
host responsibilities, and ordered load steps so host loader or build-system
tooling can consume the package contract deterministically. Load units with
unresolved blockers, such as missing host interface reflection metadata, remain
listed in the scaffold report and guide but do not get target loader files.
The bundle is metadata for integration tooling only: it does not rewrite host
application code, execute device code, generate runtime framework code, or
install target SDKs.

Inspect host loader scaffold files before runtime tooling consumes them:

.. code-block:: bash

   python -m crosstl inspect-host-loader-scaffolds \
     crosstl-host-loaders/host-loader-scaffolds.json \
     --format text

Host loader scaffold inspections emit a
``crosstl-runtime-host-loader-scaffolds-inspection`` JSON document that verifies
the scaffold manifest, integration guide, and target-scoped loader metadata
files are present, readable, and consistent with the scaffold manifest. Ready
loader metadata files are parsed and checked for matching scaffold identity,
target, adapter kind, package path, and status. Blocked load units remain
explicitly blocked without requiring target loader files. Missing, malformed,
or mismatched scaffold files are reported as structured diagnostics before host
loader or build-system tooling consumes the metadata. Inspection remains
read-only and does not rewrite host application code, execute device code,
generate runtime framework code, or install target SDKs.

Build a read-only host loader consumption plan from scaffold metadata:

.. code-block:: bash

   python -m crosstl plan-host-loader-consumption \
     crosstl-host-loaders/host-loader-scaffolds.json \
     --format text

Host loader consumption plans emit a
``crosstl-runtime-host-loader-consumption-plan`` JSON document. Planning runs
scaffold inspection first, reads only ready target-scoped host loader unit JSON
files, carries required tools and host responsibilities forward, and promotes
ordered ``loadSteps`` into actionable records for host build or runtime
integration tooling. Blocked scaffold records remain actionable
``resolve-loader-scaffold-blockers`` entries, and failed scaffold inspection
diagnostics are reported without reading unsafe unit files. The plan remains
metadata only: it does not rewrite host application code, execute device code,
generate runtime framework code, or install target SDKs.

Write a deterministic host integration handoff bundle from a consumption plan:

.. code-block:: bash

   python -m crosstl host-integration-handoff \
     crosstl-host-loaders/host-loader-consumption-plan.json \
     --handoff-dir crosstl-host-integration \
     --format text

Host integration handoff bundles emit a
``crosstl-runtime-host-integration-handoff`` JSON report and write
``host-integration.json``, ``HOST_INTEGRATION.md``, and one
``targets/*.integration.json`` file per target. The bundle is designed as a
stable handoff for build-system and runtime integration tools: it preserves the
validated loader units, promoted actions, required tools, host responsibilities,
package paths, scaffold files, and blocked-unit records from the consumption
plan. It remains a metadata bundle only and does not rewrite host application
code, execute device code, generate runtime framework code, or install target
SDKs.

Inspect host integration handoff files before downstream tooling consumes them:

.. code-block:: bash

   python -m crosstl inspect-host-integration-handoff \
     crosstl-host-integration/host-integration.json \
     --format text

Host integration handoff inspections emit a
``crosstl-runtime-host-integration-handoff-inspection`` JSON document that
verifies the handoff manifest, guide, and per-target
``targets/*.integration.json`` files are present, readable, and consistent with
the handoff manifest. Target files are parsed for matching kind, target, status,
loader-unit counts, and action counts. Missing, malformed, wrong-kind, or
mismatched handoff files are reported as structured diagnostics. Inspection is
read-only and remains bundle-local: it does not rewrite host application code,
execute device code, generate runtime framework code, install target SDKs, or
re-run host integration.

Build a read-only host integration execution plan from an inspected handoff:

.. code-block:: bash

   python -m crosstl plan-host-integration-execution \
     crosstl-host-integration/host-integration.json \
     --host-root . \
     --format text

Host integration execution plans emit a
``crosstl-runtime-host-integration-execution-plan`` JSON document. Planning
runs handoff inspection first, records optional host-root readiness, and
normalizes per-target handoff actions into stable phase-ordered execution
steps. The step phases cover tool preparation, loader consumption, artifact
loading, host responsibility handling, blocker resolution, and other host
actions. Plans carry required tools, host responsibilities, package paths,
scaffold files, target status, and structured diagnostics so downstream host
or build-system tooling can decide what to run next. Plans also include a
``deviceExecution`` readiness block that declares the adapter-backed dispatch
inputs a target will need, including the runtime package root, runtime adapter
descriptor root, and an external target runtime runner. The plan remains
metadata only: it does not rewrite host application code, execute device code,
generate runtime framework code, or install target SDKs.

Execute deterministic host integration checks from a saved execution plan:

.. code-block:: bash

   python -m crosstl execute-host-integration \
     crosstl-host-integration-execution-plan.json \
     --host-root . \
     --scaffold-root crosstl-host-loader-scaffolds \
     --package-root crosstl-runtime-package \
     --adapter-root crosstl-runtime-package/runtime-adapters \
     --format text

Host integration execution results emit a
``crosstl-runtime-host-integration-execution-result`` JSON document. Execution
validates the plan, revalidates the planned host root or an explicit
``--host-root`` override, records scaffold-root and package-root readiness,
verifies generated loader scaffold files when ``--scaffold-root`` is provided,
checks package artifact and source-remap paths when ``--package-root`` is
provided, verifies runtime adapter descriptor manifests and descriptor files
when ``--adapter-root`` is provided, checks required host tools on ``PATH``, and
reports project-specific host responsibilities as skipped actionable steps.
Blocked plan steps remain blocked in the result, and missing files, stale
descriptor hashes, or invalid paths are emitted as structured diagnostics.
The result ``deviceExecution`` block reports whether each target has a ready
runtime package and verified runtime adapter descriptor for an external runner.
Pass ``--runner-manifest`` with a
``crosstl-runtime-device-runner-manifest`` JSON file to record target runner
readiness alongside package and adapter readiness. Runner manifests list
target-specific runner ids, statuses, optional capabilities, and optional
commands; they are validated as readiness metadata only. This command still
does not rewrite host application code, dispatch device work, generate runtime
framework code, or install target SDKs.

Diagnostics with ``originalLocation`` keep the generated or validation
location as the primary SARIF location and attach the original source span as a
related location. Remapped diagnostics also expose sanitized
``diagnosticLocation`` and ``originalLocation`` SARIF properties for tools that
filter or group results without walking related locations.
Inspection exits nonzero when validation finds report errors.

Configuration
-------------

When present, ``crosstl.toml`` is loaded from the repository root. The initial
configuration contract is intentionally small:

.. code-block:: toml

   [project]
   source_roots = ["shaders", "kernels"]
   include = ["**/*"]
   exclude = ["third_party/**", "build/**"]
   targets = ["metal", "opengl"]
   output_dir = "crosstl-out"
   include_dirs = ["shaders/include"]
   external_corpus_manifest = "external-corpus.json"
   workgroup_size = [32, 8, 4]

   [project.sources]
   "legacy/**/*.shader" = "cgl"

   [project.defines]
   USE_FAST_PATH = "1"

   [project.workgroup_size_rules]
   "kernels/gemv.metal" = ["32", "BN", "BM"]

   [project.entry_workgroup_size_rules."kernels/gemv.metal"]
   "gemv_wide*" = ["32", "k_lanes / 8", "1"]

   [project.subgroup_width_rules]
   "kernels/wave.metal" = "WIDTH"

   [project.source_options.metal]
   max_template_specializations = 2048
   max_template_materialization_work = 131072

   [project.source_options.metal.source_patterns."kernels/scan*.metal"]
   max_template_specializations = 4096

   [project.source_options.metal.source_patterns."kernels/proven-layout.metal"]
   cooperative_matrix_fragment_mapping = "tile_4x4_row_pair"
   cooperative_matrix_fragment_mapping_provenance = "project_source_contract"

   [project.source_options.metal.target_options.opengl]
   max_template_materialization_work = 65536

   [project.source_options.metal.target_options.opengl.source_patterns."kernels/gemv.metal"]
   max_template_materialization_work = 131072

   [project.variants.debug]
   USE_FAST_PATH = "0"
   workgroup_size = [32, 4, 8]

   [project.specialization_constants]
   useFastPath = true
   "2" = 16

   [project.source_specialization_constants."kernels/*.metal"]
   "2" = 32

   [project.source_specialization_constants."kernels/gemv.metal"]
   "2" = 16

   [project.variants.debug.specialization_constants]
   useFastPath = false
   "2" = 8

An explicit ``--config`` path may be absolute or repository-relative. Relative
config paths are resolved against the repository root passed to the command, not
against the shell's current working directory. When ``--config`` is provided,
the referenced file must exist; otherwise the command exits with an error
instead of silently falling back to default scan settings.

Function and specialization constant values are configured under
``[project.specialization_constants]``. Each key selects a source declaration
by its exact name or by a quoted, non-negative numeric ID such as ``"2"``.
Configured values are checked against the declaration's scalar source type. If
both selectors address the same declaration, their values must agree; a
name/id conflict fails the artifact instead of choosing one silently.

Repository-wide numeric IDs and source names do not need to be unique. Use
``[project.source_specialization_constants."<repo-relative-pattern>"]`` to
override specialization selectors only for matching translation units. For
example, unrelated sources can configure the same numeric ID as a Boolean in
one table and an integer in another without offering either value to the other
source.

Source patterns use normalized repository-relative paths and merge per
selector. ``project.specialization_constants`` provides the base values. Every
matching source table can add selectors; for a selector present in more than
one table, an exact normalized path wins over a glob. Otherwise, fewer wildcard
operators take precedence, followed by the longer normalized pattern. Patterns
with equal specificity may provide the same scalar value; lexical pattern order
then selects one stable provenance path without changing the value. Equal-
specificity patterns that provide different values fail closed with a
``ValueError`` identifying the source, selector, matching patterns, and values.
An exact table only overrides selectors it contains, so less-specific matching
tables can still supply other selectors.

Source ``function_constant`` and ``constant_id`` identifiers use C-family
integral literal rules. Decimal, leading-zero octal, hexadecimal, and binary
forms are accepted, together with apostrophe digit separators and the standard
``u``/``l``/``ll`` integer suffix combinations supported by the source
frontend. IDs are normalized before duplicate checks, configuration lookup,
reflection, or target emission. Metal ``function_constant`` indices use the
native inclusive range ``0`` through ``65535``; GLSL ``constant_id`` values use
``0`` through ``2147483647``. Consequently, ``1``, ``01``, and ``0x1`` all
select the canonical configuration key ``"1"`` and collide if used by separate
declarations. A non-canonical source form is retained as ``idSpelling`` beside
the numeric ``id`` in artifact specialization records. Invalid digits, negative
values, constant expressions, and values outside the applicable source range
produce ``project.translate.specialization-constant-id-invalid`` with the source
span, original spelling, and a structured reason.

``[project.variants.<name>.specialization_constants]`` applies after the
project-level and matching source tables and overrides the same selector for
that named variant.
Artifact ``specializationConstants`` records retain the effective value and
``valueProvenance`` with the project or variant configuration path, selector,
selector kind, and variant name. Values selected from a source table use
``project-source-pattern`` provenance with the normalized ``sourcePattern`` and
the complete configuration path. The report's
``project.sourceSpecializationConstants``,
``project.sourceSpecializationPatternCount``, and
``project.sourceSpecializationConstantCounts`` fields preserve the normalized
configuration for deterministic report validation. Using a name in one table
and the corresponding ID in another still creates two matches, so different
values fail as a conflicting contract rather than relying on table precedence.

A declaration without a source initializer is required; one with an initializer
has a source default. Explicit project or variant values override source
defaults. Targets with native specialization retain required declarations for
the host runtime to provide, while targets without it must receive a concrete
configured value or a materializable source default.

OpenGL defers specialization natively as
``layout(constant_id = N) const ...`` and does not lower these declarations to
uniforms or resources. A required declaration without a source default receives
only an encoding initializer needed for valid GLSL; its report record remains
``required`` and the host must provide the value before execution. DirectX
instead materializes a concrete CrossGL variant before HLSL generation. It uses
the selected variant override, project value, or source default and fails closed
without emitting HLSL when a required value is missing, conflicting, or
incompatible with the source type.

``project.workgroup_size`` defines one concrete local workgroup size as exactly
three positive integers in X, Y, Z order. A named variant can override it with
``project.variants.<name>.workgroup_size``. Workgroup-size entries are execution
metadata, not preprocessor defines. They therefore do not enter the artifact
define map, while each named size still produces its own variant path and
deterministic execution identity.

For DirectX and OpenGL translation from Metal, a concrete
``project.workgroup_size`` or
``project.variants.<name>.workgroup_size`` can specialize every compute entry
produced by deterministic, host-named template materialization. This requires
a complete one-to-one join between emitted compute stages and materialization
records using their stable ``hostName`` and ``materializedName`` identities.
Every joined entry receives the configured size for that project variant.
DirectX preserves the entries in one HLSL library artifact, while OpenGL emits
one standalone ``main`` artifact per entry. Their execution records retain the
source, materialized, and target entry identities together with
``project-config`` or ``project-variant`` provenance.

``[project.workgroup_size_rules]`` defines repository-relative, source-specific
workgroup sizes for materialized compute entries. Each key is a source path
pattern and each value contains three integral expressions in X, Y, Z order.
An exact path takes precedence over a glob; otherwise the most specific matching
pattern is selected. Expressions may use the concrete parameters recorded for
each host-named template materialization and the integer operators supported by
the Metal constant-expression evaluator. They are evaluated with signed 64-bit
intermediate bounds. Calls, casts, member access, unknown identifiers,
non-integral parameters or literals, unsigned width-dependent arithmetic,
overflow, division by zero, and non-positive results fail closed with a
structured workgroup-rule diagnostic.

Rule evaluation joins emitted compute entries to materialization records by the
stable ``hostName`` and ``materializedName`` identities. Record order is not
significant. Every host-named record must be matched exactly once; missing,
duplicate, conflicting, or unmatched records fail the artifact. Helper
materializations without ``hostName`` remain provenance records and are not
reported as runnable entries. The source is materialized and parsed once per
target, after which a distinct size is applied to each matched compute stage.

``[project.entry_workgroup_size_rules.<source-pattern>]`` handles source files
that contain template families with different dispatch formulas. Its keys are
host entry-point patterns and its values use the same three-expression format.
The most specific matching entry pattern overrides
``[project.workgroup_size_rules]`` for that entry; the source-wide rule remains
the fallback for entries without an override. Without a source-wide fallback,
every host-named materialization must match an entry rule. Every configured
entry pattern must match at least one host-named materialization. Missing entry
coverage and stale patterns fail closed with
``project.translate.workgroup-size-entry-rule-unmatched``.

Metal also consumes source-wide and entry-specific workgroup-size rules, but as
host-dispatch contracts rather than source specialization. MSL does not encode
a fixed ``numthreads`` or ``local_size`` attribute: the generated kernel keeps
its exact emitted entry name, ``execution.entryPoints`` records each evaluated
size, and report validation verifies that the reflected compute-entry identities
match the contract. DirectX and OpenGL continue to encode the same canonical
sizes in HLSL and GLSL respectively. This distinction prevents a host dispatch
requirement from being mistaken for Metal source metadata.

For DirectX and OpenGL project translation, a consumed Metal
``[[threads_per_threadgroup]]`` parameter requires this concrete configuration
or equivalent concrete source execution metadata. Translation emits
``[numthreads(x, y, z)]`` and the matching OpenGL local-size declaration from
the same canonical value. A scalar source parameter observes ``.x``, a
two-component parameter observes ``.xy``, and a three-component parameter
retains all components. Missing, malformed, non-positive, target-limit-
exceeding, or conflicting values fail the artifact with an
``execution-specialization`` diagnostic instead of using a default local size.

DirectX can package several materialized compute entries in one HLSL artifact.
Each exported entry receives its own ``numthreads`` declaration and retains its
source, materialized, and target entry identities. Standard OpenGL GLSL exposes
one runnable ``main`` entry, so project translation emits one independently
runnable artifact per source entry. Each OpenGL artifact records only its own
entry in ``execution`` and maps that entry to ``main``. Its source-wide template
materialization metadata retains the complete host identity set so report
validation and artifact-matrix inspection reject a missing split artifact.
Helper wrappers are not presented as runnable OpenGL entries.

The fixed ``project.workgroup_size`` contract remains available for genuinely
single-entry artifacts, source metadata proving a shared size, and the complete
host-named materialization join described above. Merely configuring one size
does not prove an ordinary multi-entry aggregate safe: sources without that
deterministic materialization identity remain ambiguous and fail closed instead
of applying the value to every entry. Missing, duplicate, conflicting, or
unmatched host records also fail closed. A multi-entry OpenGL source is still
packaged as separate runnable artifacts even when every entry uses the same
size, and translation fails if the artifact model cannot represent that split.

Targets outside DirectX, Metal, and OpenGL reject a matching workgroup-size
rule before source materialization or target generation. Metal accepts the rule
as host-dispatch metadata; DirectX and OpenGL additionally specialize target
source dimensions. A failed artifact and structured
``execution-specialization`` diagnostic retain the selected rule, target, and
supported target set so the configuration cannot be silently ignored.

Successful artifact records include an ``execution`` object with the canonical
``workgroupSize``, affected ``sourceEntryPoints``, configuration or source
``provenance``, and a SHA-256 ``identity``. Report validation recomputes that
identity, and runtime artifact manifests preserve the execution object alongside
reflected target dispatch metadata. Workgroup size is independent of subgroup
width; the project pipeline does not infer or record a subgroup requirement from
any workgroup dimension.

Rule-based artifacts instead record an ``execution.entryPoints`` array. Each
entry includes the source, materialized, and target entry names, evaluated
dimensions, exact expression rule, concrete parameter values and provenance,
the joined materialization identity, and a deterministic SHA-256 identity. The
aggregate execution identity covers the complete entry array and rule
provenance. Entry-specific rules additionally retain the selected entry pattern
and its nested configuration path. Report validation selects each source and
entry pattern again, re-evaluates every expression, verifies the materialization
join and hashes, and checks the generated target entry metadata. These records
describe shader or kernel translation and dispatch requirements; they do not
rewrite framework runtime code or establish numerical runtime parity.

``[project.subgroup_width_rules]`` defines repository-relative,
source-specific exact subgroup widths for materialized compute entries. Each
key is a source path pattern and each value is one bounded integral expression.
Pattern selection, materialization joins, expression syntax, parameter
provenance, and signed 64-bit evaluation follow the per-entry workgroup rule
contract. The expression must resolve independently for every host-named
materialized entry; unknown or non-integral parameters, invalid arithmetic,
non-positive results, missing materializations, and ambiguous joins fail the
artifact with a structured ``execution-specialization`` diagnostic.

DirectX currently enforces this contract for exact widths ``4``, ``8``,
``16``, ``32``, ``64``, and ``128``. Every generated target entry receives one
single-value ``[WaveSize(width)]`` attribute, and its execution metadata records
a ``cs_6_6`` profile requirement. Report validation re-evaluates the expression
against the recorded template materialization, verifies deterministic entry and
execution identities, and checks the generated ``WaveSize`` and shader-profile
contract. A subgroup-width rule can accompany a per-entry workgroup-size rule;
both must resolve to the same materialized entry identities.

OpenGL accepts exact widths ``1``, ``2``, ``4``, ``8``, ``16``, ``32``, ``64``,
and ``128`` through a device-compatibility contract. Generated GLSL requires
``GL_KHR_shader_subgroup_basic``, declares
``CROSSTL_REQUIRED_SUBGROUP_WIDTH``, and guards the compute entry before any
translated work. Execution metadata requires the host extension
``GL_KHR_shader_subgroup`` and the ``GL_SUBGROUP_SIZE_KHR`` query. The built-in
Python and generated C++ OpenGL adapters compare that query with the artifact
contract before shader compilation, resource allocation, or dispatch, and
report a structured mismatch instead of running on an incompatible device.
Report validation verifies the extension, marker, guard, execution metadata,
and deterministic identities. The shader guard is a defensive fallback; hosts
must honor the recorded pre-dispatch check.

Every other target currently fails closed before generation with
``project.translate.subgroup-width-enforcement-unsupported`` and reason
``target-not-supported``. These failures record the missing
``execution.subgroup-width-specialization`` capability, rule provenance, and
the supported target set without emitting a misleading target artifact.

Subgroup-width specialization establishes a compiler-facing shader contract
only. It does not dispatch device work, verify hardware support, integrate a
host runtime, or establish numerical parity. Workgroup dimensions also remain
independent and do not imply a subgroup width.

For example, the host code at pinned MLX commit
``4367c73b60541ddd5a266ce4644fd93d20223b6e`` selects GEMV tile parameters per
entry and dispatches ``(32, BN, BM)``. That is evidence for distinct per-entry
workgroup variants. The leading ``32`` remains the X workgroup dimension and is
not evidence of a required subgroup width. This repository example does not
change the backend-neutral configuration contract.

The pinned MLX project-porting gate applies this contract to
``mlx/backend/metal/kernels/rms_norm.metal`` at commit
``4367c73b60541ddd5a266ce4644fd93d20223b6e``. Its DirectX project declares two
named variants, selecting ``has_w=false`` by declaration name and ``"20"=true``
by numeric ID. The gate checks report provenance and concrete materialization,
then compiles a reflected compute entry from each generated HLSL artifact with
DXC on Windows. Its current OpenGL proof leaves the subgroup rule unconfigured,
checks deferred ``layout(constant_id = 20)`` emission, and validates generated
OpenGL SPIR-V on Linux. The separate bounded LogSumExp proof exercises the
OpenGL exact-width contract. These checks prove translation and native compilation only; they do not claim
RMSNorm numerical runtime parity or full MLX test-suite support. Numerical
execution also requires host dispatch values to match each compiled artifact's
workgroup-size and subgroup-width contracts.

A separate current-corpus proof pins ``rms_norm.metal`` at commit
``846d176227a0ac13d2667e58d2bb68b322109ab0`` and selects the forward
``rmsfloat32`` entry for an axis-size-32, two-row workload. Entry-scoped runtime
reflection excludes the unreachable VJP-only function constant ``has_w`` while
preserving constants used by selected entries. The proof packages the same six
resources for HLSL and GLSL, requires ``WaveSize(32)`` on DirectX, and uses the
explicit target-scoped 32-lane software subgroup on OpenGL. Windows CI executes
the package with Direct3D 12 WARP; Linux CI validates the GLSL through
``glslangValidator`` and ``spirv-val`` and executes it with surfaceless Mesa
EGL. Both compare 64 float32 outputs against the independent RMSNorm formula at
``3e-5`` absolute and relative tolerance. This evidence covers only the
selected forward float32 workload; VJP, looped, half-precision, other axis-size,
host-runtime redirection, and full MLX test-suite coverage remain outside its
claim.

``source_roots`` limits discovery to selected directories. ``include`` and
``exclude`` use shell-style patterns against repository-relative paths. Project
reports include order-preserving source-root status records and status counts
so active, missing, non-directory, outside-project, and scan-visible roots can
be triaged without re-running discovery. Missing source roots, source roots
that resolve to files or other non-directory paths, and roots that resolve
outside the repository are reported as scan or configuration diagnostics.
Include, exclude, and source override patterns must also be
repository-relative; absolute patterns or patterns containing parent-directory
segments are reported as configuration diagnostics and skipped. Source
overrides allow extensionless or non-standard files to be assigned to a
registered source backend. Override patterns are also considered during default
discovery, so override-only files do not require broad include globs. CLI source
roots replace the configured source roots before scan, report, or translation.
CLI source overrides are merged with this configuration before scan, report, or
translation.
Known override backend aliases are canonicalized in reports; invalid override
backend names are reported as configuration diagnostics.
Explicit broad include patterns may also match compiled shader artifacts or
known source formats that CrossTL cannot parse yet. Project scans keep those
files in the skipped-file rollups and emit structured diagnostics with the
same specific guidance as single-file translation, while continuing to discover
supported translation units in the repository.
Include directories, defines, and named
variants are recorded in project reports. Source frontend options can also be
set under ``[project.source_options.<source-backend>]`` and are forwarded only
to source frontend and reverse-codegen callables that expose matching keyword
options. Metal source imports
support ``max_template_specializations`` as the project-specific unique concrete
helper specialization cap and ``max_template_materialization_work`` as the
project template materialization work budget. Materialization work is charged
to the reachable concrete graph and the type information actually resolved. It
counts each unique reachable source entry, concrete helper specialization, and
concrete struct specialization once, together with uncached function and type-
environment resolution and concrete struct-field type resolution performed for
that graph. Shared transitive helpers and repeated occurrences of the same
concrete signature are therefore deduplicated. The budget does not precharge a
whole-source ``source instantiations x template declarations`` Cartesian
estimate, and repeated scans of progressively expanded source text are not work
items merely because the text was scanned again.

Metal cooperative-matrix fragment mappings are opt-in. Configure
``cooperative_matrix_fragment_mapping`` together with
``cooperative_matrix_fragment_mapping_provenance`` only for source files whose
lane-coordinate contract has been established independently. The built-in
``tile_4x4_row_pair`` profile is exact for an 8x8 matrix distributed over 32
lanes with two adjacent row elements per lane. The Metal
``thread_elements()`` identity and matching cardinality do not select that
profile automatically. Unknown, incomplete, or shape-incompatible profiles
fail before target emission, and selected profile metadata is retained in
project diagnostics.

Direct ``translate()`` calls from Metal to template-hostile targets use the
same reachable-specialization preparation as one-unit project translation.
Explicit instantiations, ``host_name`` attributes, template defaults, include
paths, defines, and materialization budgets are therefore applied before
DirectX or OpenGL code generation. If a reachable declaration still requires
template arguments, direct translation raises a ``ValueError`` carrying the
``project.translate.template-materialization-unsupported`` diagnostic code,
missing-capability list, materialization metadata, and source location instead
of returning an artifact with unresolved target resource types. Metal,
CrossGL, and already-preprocessed source paths retain their existing behavior.
This contract does not infer variants for which the source supplies no concrete
evidence, and it is not a full-corpus or runtime-parity claim.

Concrete struct-member array accesses participate in free-function and member
template deduction. C-style dimensions and standard ``array<T, N>`` layers are
consumed one index at a time; struct-scoped aliases are resolved before indexing.
Selecting a pointer element preserves its Metal address space and qualifiers,
while a subsequent index selects its pointee. Native package tests cover integer
and fractional values, free and member calls, index side effects, and output
guards on the three generated backends. Source deduction is separate from the
target representation of pointer-bearing aggregates described below.

Addresses taken through concrete Metal pointer members retain the pointee's
address space and read-only qualification, including scoped aliases, nested
owners and fixed pointer arrays. A const owner or pointer slot does not make a
mutable pointee read-only. Metal round-trip tests compile and execute the original
and generated kernels across constant, device, thread and threadgroup storage,
checking overload selection, index side effects, writes and output guards.
Incompatible address spaces and removal of pointee constness remain errors.
DirectX selected compute entries can carry constant/device buffer references in
private aggregates as binding identities and signed element offsets. Concrete
resource arguments are forwarded through helpers; loads and stores select a
bound resource explicitly instead of creating arrays of HLSL resource objects.
Nested aggregates, fixed pointer arrays, value copies, mutable reference parameters,
rebasing and concrete template owners retain their backing buffers. Required
Windows numerical controls use the same inputs and guarded outputs as the
original/generated Metal controls. Unknown pointer escapes, incompatible access
contracts, local reference aliases, reference returns, pointer identity operations,
compound pointee updates and external
aggregate buffer layouts remain diagnostic. This representation does not yet
cover thread/threadgroup pointer members, and it
does not change the source buffer ABI to store these private handles.
OpenGL specializes the same private handles against concrete storage buffers;
final subscripts require proven bounds or explicit allocation-derived range
assertions. Unsupported aggregate escapes remain diagnostic on both targets.
Two- and four-component vectors of 32-bit floating-point, signed integer and
unsigned integer elements retain their component types and widths in these
private references. Offsets count complete vectors, not scalar components;
guarded native controls exercise rebasing, buffer selection and untouched output
ranges. Three-component, narrow, boolean and 64-bit typed vector pointees remain
diagnostic because this path does not establish their storage layouts.
An explicit null storage-pointer argument can be omitted from a private helper
only when its formal is unused or forwarded through a proven acyclic chain of
unused formals. Literal branches may establish that proof; runtime conditions
cannot. Entry bindings, helper execution and all remaining argument evaluations
are retained. Every affected call must supply a null literal or an unshadowed,
type-compatible pointer parameter. Pointer reads, identity operations, recursion,
ambiguous overloads and side-effecting pointer arguments remain diagnostic.
This does not represent null handles or support null-initialized aggregate fields.

DirectX contextual vector initializers retain the result width of supported
componentwise unary builtins on 32-bit float scalars and vectors, including
nested calls. Declared function return types take precedence over builtin
inference. Unknown calls, wrong argument counts and overfilled vectors retain
their aggregate diagnostics; inferred types do not duplicate expression
evaluation or establish numerical parity for the enclosing kernel.

Concrete Metal ``vec<T, N>`` constructors preserve scalar splats, copied vectors
and mixed component arguments for float, half and integer widths two through
four. Global and local aliases resolve before constructor emission. Named
``static_cast`` and functional vector conversions select the declared conversion
operator by destination type and receiver qualifiers, retaining its body and
single evaluation. Incompatible or ambiguous receivers produce structured
diagnostics. Required native controls include mutable receivers, const overloads,
materialized template owners, side-effecting temporary construction and output
guards; macOS also executes the original source. This bounded contract does not
establish arbitrary user-defined conversion chains or bfloat16 parity.

Metal round trips additionally preserve native two-, three- and four-lane bfloat
vectors, including scalar alias chains, vector aliases and declared conversions
on materialized template owners. Required macOS execution compares original and
generated kernels using exact integer readbacks: ties-to-even conversion,
signed zero, overflow, infinity, raw 16-bit payload copies, indexed/swizzled
components, vector size and output guards. Raw copies include NaN payloads;
this is not a claim about every arithmetic operation or NaN conversion rule.
The unchanged pinned general-gather wrapper executes as original and generated
Metal in a required macOS gate. Six specializations cover zero, one or two index
buffers and scalar through three-dimensional indices across dense, transposed,
strided and broadcast layouts. The 24 workloads retain negative and repeated
indices, exact binary32 storage payloads and trailing output guards. Separate
required Windows and Linux gates execute the same twenty indexed workloads
through DirectX and OpenGL packages. The four zero-index workloads retain their
Metal gate; their empty pointer-array representation remains unsupported on
the foreign targets. Foreign-target bfloat parity remains unfinished. Compiler success is
not a substitute for the required native readback evidence.

The portable MLX host adapter additionally connects ``Gather::eval_gpu`` to
on-demand packages built from that unchanged wrapper. It validates index values,
allocation spans, shapes, strides and dispatch dimensions before submission.
Six source storage types and signed/unsigned 32-bit and 64-bit indices are
supported within the documented rank and allocation bounds. The required host
gate compares 42 indexing workloads and unchanged upstream ``test_take`` on
separate CPU and native paths, retaining uploaded storage, module identities,
native readbacks and output guards. ``GatherAxis::eval_gpu`` uses the unchanged
axis-gather template with validated source/index contiguity and upstream grid
geometry. A separate required host step compares 60 workloads and unchanged
``test_take_along_axis`` on each target, including noncontiguous indices and exact
64-bit storage. Larger allocations, additional storage types and full
indexing/autodiff coverage remain separate work; this is not full upstream-suite parity. See
``demos/integrations/mlx/portable_host/README.md`` for the exact host contract.

Zero-extent Metal standard arrays retain native array objects rather than
illegal C-style zero-length arrays. Layout calculations retain the element
storage required by Metal, including packed/narrow elements, nested arrays and
concrete structs. Native controls verify logical size zero, alignment,
initialization, independent copies, adjacent fields and pointer-element storage
qualification. DirectX/OpenGL report an unsupported-representation diagnostic
for these objects until their value and layout contracts are implemented.
This does not make C-style zero-length arrays valid or define portable pointer
object sizes.

During project translation, Metal template-member inference preserves a
generic pointer template parameter as a pointer rather than reducing it to its
pointee type. For a parameter such as ``Pointer src``, a bare tracked pointer,
legal pointer-plus-integral or pointer-minus-integral expression, or the address
of a directly subscripted pointer or array element binds ``Pointer`` to the
complete pointer type. Pointer identity, the proven Metal address space, and
``const``/``volatile`` qualification are retained. Legal offset forms include
``ptr + offset``, ``offset + ptr``, and ``ptr - offset`` when ``offset`` has a
known integral type.

This does not change deduction for a parameter declared as ``device U*`` or
``threadgroup U*``: after pointer compatibility is established, ``U`` is still
deduced as the pointee type. Inference remains conservative. An unknown base or
address space, a non-integral offset, pointer-pointer arithmetic, an offset
minus a pointer, or address-taking outside a proven ``&base[index]`` shape
fails closed with ``project.translate.metal-struct-method``. Addressed-element
indices must also be known integral expressions without unsupported calls,
assignments, side effects, nested subscripts, or ambiguous expression forms.

Concrete pointee comparisons resolve visible non-template ``using`` and
``typedef`` chains at the method declaration and call site before deciding
whether two pointer types are compatible. Declaration order, nested shadowing,
and sibling scopes are preserved. Forward references, alias cycles, and chains
whose equivalence cannot be proved remain failed bindings; they are not treated
as matching merely because their unresolved spelling is the same.

For DirectX, a supported storage-pointer helper parameter is emitted as a
``StructuredBuffer`` or ``RWStructuredBuffer`` resource together with a signed
element offset. Passing ``&buffer[index]``, a previously rebased alias, or an
alias forwarded through another supported helper composes that offset without
emitting HLSL pointer syntax or mutating the resource handle. The generated
helper applies the offset to every indexed load or store. Element-type
changes, insufficient read or write access, pointer-to-pointer parameters, and
arguments without a concrete structured-buffer root fail with
``project.translate.directx-resource-pointer-parameter-unsupported`` rather
than producing an invalid call.

A local read-only ``auto*`` alias whose initializer resolves to a concrete
``StructuredBuffer`` or ``RWStructuredBuffer`` root deduces its element type
from that backing resource before DirectX alias lowering. Direct and nested
aliases retain the accumulated signed element offset and read-only contract.
Explicit pointee types remain authoritative: incompatible declared element
types are rejected rather than being replaced with the backing type.

Metal reverse translation preserves conditional and assignment expressions as
lower-precedence operands when serializing binary and postfix expressions. For
resource-backed pointer aliases, a dynamic offset such as
``buffer + (enabled ? first : second)`` therefore retains the conditional as
the offset expression before DirectX or OpenGL resource-plus-offset lowering.
This guarantees expression-tree preservation for the translated artifact; it
does not provide host dispatch, resource binding, or numerical runtime parity.

The contract applies generally to Metal sources handled by project translation.
The pinned MLX ``BaseMMAFrag::load(&(src[index]))`` call shape is a focused
acceptance example for retaining the qualified pointer through nested template-
member materialization; it is not evidence of runtime parity or completion of
the pinned or full MLX corpus.

Metal struct-method lowering preserves direct mutable and read-only reference
accessors when the returned lvalue is receiver-owned scalar storage or an
exactly indexed fixed-array element. Simple receiver declarations can use a
lexically visible ``using`` or ``typedef`` chain that resolves to the concrete
struct; alias scope and declaration order are honored before the accessor is
rewritten to original storage. A proven ``thread const`` receiver selects one
matching const accessor, including implicit calls from a const struct method. A
local ``thread const auto&`` binding may also read through an accessor on a
nested value member when the member path contains no pointer, reference, or
array traversal. The binding is replaced with the original fixed-array storage
only when every use is an indexed read and the accessor arguments cannot change
through a reference, member mutation, or subsequent call during the binding's
lifetime.

Constant-address-space receivers, unresolved aliases, pointer-member
receivers, ambiguous overloads, mutable or escaping local reference bindings,
side-effectful or unstable indices, non-indexed alias uses, and indirect storage
remain fail-closed with
``project.translate.metal-struct-method`` rather than being converted to
value-returning helpers.

Non-entry Metal ``const`` reference parameters retain their input-only contract
in the shared representation. DirectX and OpenGL receive value inputs, mutable
references remain ``inout``, and Metal round-trip generation reconstructs a
``const`` address-space reference. Stage-entry buffer references continue
through the resource binding path instead of being rewritten as helper values.

The default materialization work budget is derived from the active template
specialization limit, so larger finite source-instantiated kernels can complete
without raising the unique helper specialization cap. Use
``[project.source_options.metal.source_patterns."<repo-relative-glob>"]`` to
raise or lower Metal budgets for matching sources. Use
``[project.source_options.metal.target_options.<target>]`` and its nested
``source_patterns`` table to override budgets only for one target, such as
OpenGL. Both limits remain fail-closed. If the next unique entry, helper, struct
specialization, or required type-environment resolution would cross a configured
limit, translation fails without emitting the target artifact. The structured
diagnostic identifies the concrete item or resolution that crossed the limit
and reports the requested count, active limit, configuration field that set it,
source location, and suggested remediation. Project reports include
order-preserving include-directory status records and status counts so missing,
non-directory, outside-project, and frontend-visible active include directories
can be triaged without re-running discovery. Missing include directories,
include entries that resolve to files or other non-directory paths, and include
directories that resolve outside the repository are reported as non-blocking
configuration diagnostics so reports retain portability and provenance context.
Existing include directories that remain inside the repository, plus configured
defines, are passed to source frontends that expose preprocessor options. CLI
include and define overrides are merged with this configuration before scan,
report, or translation. Translation artifacts record ``defineProcessing``
metadata so reports distinguish define maps that were forwarded to the source
lexer from define maps that were not requested or could not be consumed by that
frontend. Report inspection samples include effective define names and
deterministic define fingerprints without exposing define values.
When configured defines cannot be forwarded, translation reports emit a
non-blocking warning diagnostic with a missing-capability rollup so the
limitation appears in validation and inspection summaries.
Scan reports also emit diagnostics for active ``#error`` and ``#warning``
directives after evaluating project and selected variant conditionals. Active
``#error`` directives are reported as errors, while active ``#warning``
directives are reported as warnings.
Scan reports also emit non-blocking warning diagnostics when active
``#define`` or ``#undef`` directives in translation units or resolved include
files shadow configured project or selected variant define names. The
diagnostics identify the source location and define name without reporting
configured define values.
Summary, inspection payloads, and text reports also include define-processing
rollups by target, source backend, and named variant when variants are
configured, so target-specific, frontend-specific, and variant-specific
preprocessing gaps are visible without reading every artifact record. Report
inspection also includes sampled artifact
define-processing metadata with status, frontend support, and define counts,
but not define values, so artifact-level preprocessing state can be triaged
without exposing configuration values.
Define-processing inspection summaries also include redacted project define
names, deterministic define fingerprints, selected variant names, and
per-variant define records without exposing configured define values. Text
issue lines for unsupported define forwarding include define names and the same
fingerprint so the affected define set can be identified without revealing
values.
They also record ``includePathProcessing`` metadata so active include-directory
paths can be distinguished from include paths that were not requested or could
not be consumed by the selected source frontend. Include-path processing
warnings are also emitted when active include paths cannot be forwarded, so
the report diagnostics identify affected source frontends without failing the
batch translation.
Summary, inspection payloads, and text reports also roll up by target, source
backend, and named variant when variants are configured. Report inspection
includes sampled artifacts whose active include paths could not be forwarded, so
the affected source, target, and frontend are visible without reading every
artifact record. It also includes sampled artifact include-path processing
metadata with status, frontend support, and include path counts, so
artifact-level include forwarding state can be triaged from the report summary.
Inspection summaries also include configured include-directory status records
plus frontend-visible and inactive directory counts, so report consumers can
distinguish directories that reached the frontend from missing, non-directory,
or outside-project entries. Text issue lines for unsupported include-path
forwarding name the frontend-visible configured include directories so the
affected configuration is visible next to the artifact identity.
During scan, project reports also record ``#include`` directives discovered in
translation units. Each dependency record keeps the include target, local,
system, or dynamic kind, line and column, and a status of ``resolved``,
``missing``, ``system``, ``dynamic``, or ``outside-project``. Resolved
dependencies record the repository-relative resolved path and whether the match
came from the source directory or a configured include directory. A directive
that uses one project define, such as ``#include PROJECT_HEADER``, is resolved
when that define's value is a quoted or angle-bracket include target; the
dependency keeps ``resolvedFromDefine`` so the report remains actionable.
When named variants are configured, include discovery evaluates those
define-backed include targets with the same base-plus-variant define maps used
for translation, and variant-scoped dependency records keep ``variant``.
Scan-time include discovery also honors simple ``#if``, ``#ifdef``,
``#ifndef``, ``#elif``, ``#else``, and ``#endif`` branches using the same
project and variant define maps. Supported ``#if`` expressions include
``defined`` checks, boolean operators, parentheses, integer and boolean define
values, and simple integer comparisons. Unsupported conditional expressions
remain open so discovery does not hide possible dependencies.
Resolved include files are scanned recursively for additional dependencies.
Nested dependency records keep ``source`` when the directive came from a
resolved include file rather than the root translation unit, so diagnostics and
inspection output can point to the include file that introduced the dependency.
If a resolved nested include cannot be read, the already discovered dependency
is kept and project scan emits ``project.scan.include-read-failed`` with
``include.resolution`` capability metadata.
Unresolved system includes are recorded without warning because they often
refer to SDK or toolchain headers. Missing local includes, dynamic include
expressions, and include paths that resolve outside the repository emit
structured ``include.resolution`` diagnostics. When the failed include came
from a project or selected variant define, diagnostics identify that define
and the variant context when applicable.
Report inspection samples resolved include dependencies, unresolved system
include dependencies, and include issues, including the source location,
source backend, include kind, unit source hash and byte size, resolved path,
resolved include hash and byte size, and resolution source where available.
Define-backed include samples also retain the project define name that supplied
the include target and the variant name when the dependency came from a named
variant define map.
``output_dir`` must resolve inside the repository root; paths that escape the
repository are reported as configuration diagnostics and artifacts are not
written. When named variants are configured, project translation emits one
artifact attempt per variant and passes base defines merged with the variant's
define overrides to the source frontend. Variant artifacts are written under a
variant path segment inside each target output directory, and the original
variant name plus applied define map are recorded on the artifact and variant
name is recorded on validation records. ``--variant NAME`` can be repeated to
scope scan, report, or translation runs to declared variants; when no explicit
``--variant`` arguments are provided, ``selected_variants`` in ``crosstl.toml``
sets the default scoped variant list. Scoped reports declare only the selected
variants, de-duplicate repeated selections before planning, and do not claim
omitted variants as scanned or attempted.
CrossGL source translation applies object-like define expansion and conditional
branch selection for ``#if``/``#ifdef``/``#ifndef``/``#elif``/``#else``/``#endif``
when defines are provided. Project translation also passes selected variant
define maps into native source frontends that expose define options; current
project coverage includes OpenGL/GLSL and Vulkan angle include expansion,
DirectX/HLSL, Metal/MSL, Slang, and CUDA/HIP local header expansion, CUDA/HIP
runtime system include preservation, conditional branches, and project include
directories through those paths. Other native preprocessor behavior remains
backend-dependent.

Configuration scalar values and define/source-override maps are type checked
when ``crosstl.toml`` is loaded. Define names, source override patterns, source
override backend names, named variants, and variant define names must be
non-empty strings. Malformed values are rejected before scan or translation so
reports do not silently stringify invalid project metadata.

``external_corpus_manifest`` points at an optional repository-relative JSON
manifest of pinned upstream shader or GPU-source reductions. When configured,
the manifest path must be a non-empty string. Project reports use the manifest
for coverage accounting only: they record declared paths, present and missing
entries, discovered translation units, source-backend and target rollups, valid
and invalid manifest-entry counts, and artifact outcomes for entries included
in the project run. CrossTL does not clone upstream repositories, run native
build systems, or claim whole corpus semantic parity from this manifest.
The bundled support manifest is a reduced, fixture-backed coverage manifest
with one pinned entry per registered source backend; those entries support
provenance and accounting checks rather than corpus-wide semantic parity
claims.
Malformed manifest entries are reported as configuration diagnostics and
skipped from retained corpus entries. Duplicate manifest paths or explicit
entry ids are also reported and skipped so generated reports do not inflate
corpus coverage. The summary still records how many manifest entries were
skipped. Inspection samples for missing and present-but-undiscovered entries
retain repository, commit, and source URL metadata when the manifest provides
those provenance fields.

Project reports include configured define, variant, and specialization constant
selectors and values, and artifact records include the applied define map used
for that translation attempt. Review reports before sharing them outside the
repository if those values include private build metadata. Compact inspection
summaries list
configured define names, deterministic define fingerprints, variant names,
per-variant define counts, variant define names, and deterministic per-variant
define fingerprints without printing define values.

Bounded Host Dispatch Contract Import
-------------------------------------

Projects can import versioned JSON host dispatch contracts through
``project.dispatch_contracts`` in ``crosstl.toml``:

.. code-block:: toml

   [project]
   dispatch_contracts = [
     "contracts/layer-norm.dispatch.json",
     "contracts/copy.dispatch.json",
   ]

The ``scan``, ``report``, and ``translate-project`` commands also accept a
repeatable ``--dispatch-contract PATH`` option. Command-line imports augment
the configured contract list for that invocation:

.. code-block:: bash

   python -m crosstl scan /path/to/repo \
     --dispatch-contract contracts/layer-norm.dispatch.json

   python -m crosstl report /path/to/repo \
     --dispatch-contract contracts/layer-norm.dispatch.json \
     --dispatch-contract contracts/copy.dispatch.json \
     --output crosstl-out/portability-report.json

   python -m crosstl translate-project /path/to/repo \
     --dispatch-contract contracts/layer-norm.dispatch.json \
     --output-dir crosstl-out

Contract expressions are evaluated over finite declared domains in a
deterministic order. The resulting project metadata includes:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Report field
     - Contents
   * - ``dispatchContractFiles``
     - Ordered configured and command-line contract paths.
   * - ``dispatchContractCount``
     - Number of imported contract manifests.
   * - ``dispatchVariantCount``
     - Total number of deterministically evaluated dispatch variants.
   * - ``dispatchContracts``
     - Embedded normalized manifests, content identities, provenance, and
       evaluated variant records.

The embedded manifests and evaluations are machine-readable and self-contained
for project report validation, including deterministic replay without the
external contract files. During scanning, evaluated records are converted into
a deterministic source-scoped artifact plan. Compile-equivalent records share
one artifact job while their distinct dispatch geometries remain available as
dispatch variants. A contract that names an undiscovered source, an unknown
entry point, or a conflicting artifact identity fails before target emission.

``translate-project`` applies each planned job only to its referenced source
unit. The job selects the source entry point and carries its workgroup size,
required subgroup width, specialization constants, stable artifact identity,
and contract provenance into target generation. Unreferenced source units retain
their ordinary project configuration. Generated reports expose this contract
through:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Report field
     - Contents
   * - ``project.dispatchArtifactCount``
     - Number of source-scoped compile jobs in the deterministic plan.
   * - ``project.dispatchArtifactPlan``
     - Planned artifacts, dispatch variants, source units, stable identities,
       entry points, execution specialization, and provenance.
   * - ``artifacts[*].dispatchArtifact``
     - The exact planned job applied to an emitted or failed artifact record.
   * - ``artifacts[*].execution.provenance``
     - A closed reference back to the matching artifact-plan record.
   * - ``artifactMatrix.variantMode``
     - ``source-scoped`` when dispatch-derived and ordinary artifacts coexist.

Report validation deterministically rebuilds the plan from the embedded
contracts and discovered units, rejects unknown or tampered dispatch variants,
and checks emitted execution metadata against the planned entry point and
specialization. DirectX and OpenGL can emit source-scoped artifacts when their
target contracts are representable. Requirements such as an exact subgroup
width still fail closed on a target that cannot enforce them.

Named project variants cannot yet be composed with dispatch-derived variants;
that work remains tracked in `GitHub issue #1798
<https://github.com/CrossGL/crosstl/issues/1798>`_. Imported contracts do not
execute the host dispatch path, allocate or bind runtime resources, or establish
numerical parity. Those responsibilities remain with repository integration and
runtime adapters.

Report Shape
------------

Project reports are JSON documents with:

- top-level metadata: report schema version, report kind, generation timestamp,
  and generator name/pipeline/package-version fields.
- ``project`` metadata: root, config path, optional config hash, source roots,
  source-root status records and status counts, include/exclude
  patterns, targets, output directory, source override map, include
  directories, include-directory status records and status counts, define and
  variant maps, project and per-variant specialization constant maps, project
  and per-variant workgroup sizes,
  per-variant define and specialization constant counts, and counts for source
  roots, include patterns, exclude patterns, source overrides, include
  directories, defines, variants, and project specialization constants.
- ``summary``: total unit/artifact/diagnostic/source-map counts plus rollups by
  unit source backend, unit file extension, skipped reason, skipped file
  extension, unit source override, skipped source override, artifact source
  backend, variant, target backend, source-map granularity, source-map target,
  source-map source backend, source-map variant, source-remap mapping count,
  source-remap granularity, source-remap target, source-remap source backend,
  source-remap variant,
  include dependency kind,
  include dependency status, include dependency source backend, include
  dependency source-backend status, include dependency resolution source,
  include dependency variant, artifact provenance pipeline, intermediate,
  source backend plus intermediate, target plus intermediate, variant plus
  intermediate, diagnostic severity (``diagnosticCounts``), diagnostic code
  (``diagnosticsByCode``), diagnostic target backend
  (``diagnosticsByTarget``), diagnostic source backend
  (``diagnosticsBySourceBackend``), diagnostic variant
  (``diagnosticsByVariant``), diagnostic check kind
  (``diagnosticsByCheckKind``), and missing capability
  (``missingCapabilityCounts``).
- ``units``: discovered translation units with stable repository-relative POSIX
  paths,
  source backend names, path-derived extensions, source hashes, source byte
  sizes, and source overrides. Units that contain ``#include`` directives also
  include ``includeDependencies`` records for project-level include triage. Include scans
  ignore directives inside C-style block comments while still recognizing active
  directives after same-line block comments. Resolved include dependencies
  record repository-relative include paths, resolution source, SHA-256 hashes,
  and byte sizes so report validation can detect include file content or size
  drift after scan.
  Full report validation also re-scans current source files and rejects missing
  or extra include dependency records. Recursive include scans stop at include
  cycles and
  emit ``project.scan.include-cycle`` diagnostics with ``include.resolution``
  missing-capability rollups while preserving the dependency that closes the
  cycle for triage.
- ``skipped``: stable repository-relative POSIX paths for files intentionally
  left untranslated with
  reason codes and source override metadata when an override selected an
  unsupported source backend. Known unsupported source or binary artifact
  extensions are recorded with ``unsupported-extension`` and a matching scan
  diagnostic so broad repository scans remain auditable. Full reports require
  skipped source override metadata to match the configured source override map.
- ``artifacts``: attempted outputs with stable repository-relative POSIX source
  and output paths, source backend, target, applied define map, optional variant
  name, target/variant-scoped output path with the target backend suffix,
  status, source hash, source byte size, generated artifact hash, generated
  artifact byte size, pipeline provenance, and file-span source-map anchors for
  successful translations. Full reports require every artifact to carry a source
  hash and source byte size,
  artifact source hashes to match their declared translation-unit source
  hashes, artifact source byte sizes to match their declared translation-unit
  source byte sizes,
  artifact output paths to match the target/variant directory plus the
  source-relative path with the target backend suffix, artifact source paths
  to match declared translation units, unit source backend names to be
  registered canonical source backend names, unit source override metadata to
  match the configured source override map, and artifact source backend names
  to match those units. Full reports with translated or failed artifacts must
  include the expected artifact matrix for each declared translation unit,
  target, and configured variant. Full reports also require artifact define
  maps to match the project-level defines merged with the artifact variant's
  define overrides, and require ``defineProcessing`` metadata to match the
  artifact define map, registered source frontend support, and summary rollups
  including named-variant rollups. Full reports also require
  ``includePathProcessing`` metadata to match active include-directory records,
  registered source frontend support, and summary rollups including
  named-variant rollups.
  Artifacts with function or specialization constant declarations also carry
  dedicated ``specializationConstants`` records for identity, required/default
  state, effective values, and value provenance, plus
  ``specializationMaterialization`` metadata that distinguishes native deferred
  specialization from a concrete CrossGL variant.
  Artifacts with a concrete workgroup-size contract carry an ``execution``
  record with canonical dimensions, source entry points, provenance, and a
  deterministic identity. Full report validation rejects malformed dimensions,
  unknown variant provenance, or an identity that does not match the artifact
  source, target, variant, entries, and dimensions.
  Successful artifact records in full reports must include file-level
  source-map anchors. Generated CrossGL artifacts also include a
  compiler-compatible ``source-remap`` sidecar with a file-level
  generated/original mapping for compiler ``--source-remap`` consumers. The
  source-map and source-remap ``offset``, ``length``, and ``endOffset`` fields
  are UTF-8 byte offsets. Source maps use a closed schema with
  ``file``, ``line``, ``column``, ``offset``, ``length``, ``endLine``,
  ``endColumn``, and ``endOffset`` span fields. ``mappingGranularity`` may be
  ``file``, ``line``, ``statement``, or ``token``. File-granularity source maps
  must contain one mapping that exactly matches the artifact-level source and
  generated anchors. Finer-grained source maps keep those artifact-level
  anchors as file spans and may include one or more positive-length mappings
  whose source and generated files match the anchors. Line-preserving source and
  generated artifacts include line-granularity mappings with UTF-8 byte offsets.
  The report records the sidecar path, per-artifact mapping count, aggregate
  source-remap mapping count, hash, generated-file identity, summary rollups by
  target and source backend, and bounded inspection
  samples for source-map and source-remap artifacts with declared target,
  source and generated hash, and byte-size metadata. Validation checks that
  artifact-level source-map spans still cover the current source and generated
  files, recomputes line-preserving mappings, requires source-remap metadata
  mapping counts to match source-map mappings, and checks that compiler
  source-remap sidecars use the closed schema-1 field set. The project pipeline
  emits line-granularity
  source maps only when the generated artifact preserves the same logical lines
  after newline normalization, allowing a final-newline-only difference;
  translated artifacts keep file-granularity source maps until backend pipelines
  expose generated line, statement, or token provenance.
  Token similarity does not establish source origin. Generated helpers, reordered
  declarations, and lowered expressions therefore do not acquire line mappings
  from matching identifiers or comments. Validation rejects legacy heuristic
  line maps even when their hashes and sidecars are internally consistent;
  regenerate those reports to record file-level provenance.
  File-granularity mappings establish an artifact's source association, not a
  byte-for-byte or line-for-line correspondence. A schema-1 source-remap sidecar
  does not carry this distinction by itself: consumers must consult the report's
  ``mappingGranularity`` and must not interpolate diagnostic locations through
  file-level mappings. Compiler-side handling of coarse and synthetic provenance
  remains a separate integration requirement.
  Artifact provenance records the
  ``single-file-translate`` pipeline and uses ``crossgl`` as the intermediate
  marker only when both source and target backends route through the CrossGL
  bridge. Report summaries and inspections include provenance rollups by
  pipeline, intermediate marker, source backend with intermediate marker,
  target with intermediate marker, and variant with intermediate marker, plus
  bounded artifact provenance samples for direct and bridge artifacts.
  Inspection samples include failed validation
  status, existence, hash, source-map, and source-remap status metadata when the
  validated artifact no longer matches the report.
  Metal artifacts that are materialized before translation can include
  ``templateMaterialization`` metadata. That metadata records whether
  materialization succeeded, configured template parameters, unsupported
  templates with missing parameter names, and concrete specializations with the
  original template name, materialized function name, parameter map,
  specialization source, and optional ``hostName`` for source-instantiated
  kernels. Source-instantiated Metal artifacts can also include an ``accounting``
  object. ``reachableSpecializationCount`` counts unique concrete function and
  struct specializations selected for the artifact,
  ``dependencyDiscoveryWorkCount`` counts uncached type-environment and type
  resolution charged separately from those specializations, and
  ``prunedCandidateCount`` records source-instantiation/template-declaration
  pairs from the former eager candidate space that were not selected. The same
  accounting object is included in a materialization-work budget diagnostic
  when all three counts are available. Report validation rejects missing,
  negative, boolean, or unknown accounting fields.
  Full reports require failed artifacts to carry an actionable error string and
  reject failed artifacts that claim
  generated hashes or source-map records. Full reports also reject translated
  artifacts that carry error metadata. Invalid project output directories are
  recorded as failed artifacts without writing files.
- ``artifactMatrix``: scan and translation metadata with expected, emitted,
  translated, failed, missing, extra, and completion counts plus target,
  source-backend, and variant completion rollups for the unit, target, and
  variant matrix.
  Scan-only reports include the expected artifact plan with zero emitted,
  translated, and failed artifacts so automation can review planned outputs
  before artifact generation. Report inspection also includes sampled missing
  and extra artifact identities from report-provided or translation
  artifact-derived matrix metadata, and text inspection identifies which matrix
  source was used, so incomplete batch outputs are visible without opening
  every artifact record.
- ``externalCorpus``: optional manifest-backed corpus accounting with declared
  entries, present/missing and discovered-unit status, source-backend and target
  rollups, valid/invalid manifest-entry counts, and translated/failed artifact
  outcome counts for manifest entries. Validation checks entry presence against
  the project root, checks discovered/source-backend fields against declared
  translation units, and rejects missing or inconsistent retained-entry and
  manifest-entry summary counts.
- ``diagnostics``: structured diagnostics using severity, code, message,
  location, optional ``originalLocation``, target, source-backend, variant, and
  missing-capability fields aligned with the compiler diagnostic contract.
  ``location`` identifies the report or generated artifact span that produced
  the diagnostic, while ``originalLocation`` preserves the original repository
  source span when diagnostics are remapped through generated artifacts.
  Project-level include and define forwarding limitations are warnings, not
  translation failures. Scan-time ``#define`` and ``#undef`` directives in
  translation units or resolved include files that shadow active project or
  selected variant define names are also reported as warnings; directives inside
  C-style block comments are ignored.
- ``validation``: report contract checks, generated timestamp and generator
  metadata checks, report inspection summaries, failed
  source artifact checks, project metadata, target normalization, and config
  count checks including compact variant-name and variant-define-count
  inspection summaries, unit and skipped record shape checks, artifact record
  shape checks, source and generated hash checks, duplicate artifact identity
  checks, required
  source/generated hash, source-size, generated-size, and
  source-map/source-remap status fields for summarized validation artifacts,
  aggregate validation artifact and
  validation status summary counts, direct validation report project context,
  source report hash metadata,
  artifact target, source backend, variant, hash-status, source-size status,
  generated-size status,
  source-map status, source-remap status,
  toolchain status, toolchain-run status rollups, toolchain-run check-kind
  metadata, toolchain-run tool rollups, and a closed standalone validation-report
  field set,
  failed-artifact text with
  source-backend context plus non-OK hash, source-size, generated-size,
  source-map, and source-remap statuses, bounded
  validation artifact samples with source-backend context, bounded validation
  toolchain-run inspection samples, source-root and
  include-directory status record and count consistency checks, config hash
  shape and current-file checks, unit source hash and byte-size shape and
  current-file checks, full-report artifact matrix coverage and artifact define
  map checks, artifact define-processing metadata and status/target/source-backend/variant
  rollup checks,
  artifact include-path processing metadata and
  status/target/source-backend/variant rollup checks,
  artifact matrix emitted/translated/failed/missing/extra/completion count and
  target/variant rollup checks,
  full-report source-map granularity, target, source-backend, and variant
  rollup checks,
  source-remap granularity, target, source-backend, and variant rollup checks,
  source hash and source byte-size checks, source-size validation status checks,
  generated artifact byte-size checks,
  failed artifact error metadata checks, translated artifact error metadata
  rejection, required artifact
  provenance and provenance value checks, artifact provenance source-backend,
  target, and variant rollup checks,
  failed artifact generated metadata rejection, required translated artifact
  source maps, required CrossGL artifact source remaps, source-map
  record shape, non-empty mapping list, file-level mapping cardinality,
  positive-length finer-grained mappings, finer-grained mapping containment
  within artifact-level anchors, span consistency, anchor consistency, current
  file-level source-map span coverage,
  source-remap metadata shape, mapping-count consistency, sidecar hash and byte
  size, closed compiler sidecar field sets, and sidecar content checks, external
  corpus record, per-entry artifact count, required manifest-entry accounting,
  and summary checks, summary consistency checks, migration action shape,
  rollup, and target declaration checks,
  preserved diagnostic shape, repository-relative file path, location and
  ``originalLocation`` span consistency, target declaration checks, diagnostic
  severity rollup checks, scan-scope
  count consistency, diagnostic check-kind rollup consistency, validation
  toolchain status consistency checks, validation artifact and toolchain run
  record shape and duplicate identity checks, validation artifact coverage,
  required validation summary records for embedded validation artifacts,
  embedded toolchain-run coverage for available toolchains,
  failed embedded toolchain-run diagnostics,
  toolchain-run target, source-backend, check-kind, tool, and variant rollups,
  toolchain target coverage and status consistency checks,
  include dependency record shape and include dependency summary consistency,
  current include dependency status, resolved-path, resolved-hash,
  resolved-size, source-backend status rollup, resolution-source checks, and
  project-define include provenance checks, current include-scan diagnostic
  presence checks, resolved and unresolved include inspection samples,
  artifact source, source-backend,
  target, variant, and source-relative output layout declaration checks,
  current project scan coverage for omitted unit and skipped-source records,
  translated artifact existence checks, escaped output directory and
  artifact-path checks, source artifact existence and hash mismatch checks,
  generated artifact hash and byte-size mismatch checks, optional external
  toolchain availability, and opt-in toolchain smoke results including bounded
  timeout failures.
- ``migration``: actionable manual follow-up work outside shader/kernel
  translation. The report records documented non-goals for runtime API
  migration, build-system rewrites, and backend framework integration. Each
  action has a documented kind, severity, message, and target list, plus
  action count and kind, severity, target, and runtime-reference rollups. Scan-only reports
  include supported requested targets when translation units are present.
  Translation reports scope ``manual-runtime-integration`` to targets that
  produced translated artifacts, covering host runtime API, resource binding,
  build script, and backend integration review. Runtime actions can include
  ``runtimeReferences`` entries for detected host or build files, with
  repository-relative path, line, column, backend, kind, and symbol metadata.
  Reports also include runtime-reference count, backend, kind, and path rollups
  so inspection tools can summarize host integration evidence without parsing
  each action. These references are evidence for follow-up integration work;
  they are not host-code rewrites. Reports with unresolved system include
  dependencies also emit ``manual-include-resolution`` actions so target SDK or
  toolchain header assumptions remain visible without claiming automatic header
  rewriting.
