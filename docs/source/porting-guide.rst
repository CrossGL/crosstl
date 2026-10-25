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
   systems. It checks each target's artifacts with that target's reference
   tool when the tool is on ``PATH``; ``python -m crosstl doctor`` lists the
   tools it looks for and which of them it finds, and ``--format json`` gives
   the same list to scripts.
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

Set ``[project].freeze_specialization_constants = true`` (or
``ProjectConfig(freeze_specialization_constants=True, ...)``) to make configured
Boolean specialization values immutable for a Metal, DirectX or OpenGL artifact.
The default is false. Source declarations without configured values retain their
existing deferred/default behavior. Non-Boolean frozen values and other targets
are rejected explicitly.

Frozen values are materialized before target capability checks. The shared pass
discards only branches proven inactive by Boolean literals, frozen names, logical
operators and Boolean equality; it preserves lexical blocks and leaves general
arithmetic and unknown conditions intact. A same-named local disables that name's
folding for the enclosing function. Entry selection then removes helpers that are
no longer reachable. This is not authorization to narrow a live wide index.

The generated source contains a versioned frozen-value header. Reflection,
package inspection and native-loader descriptors retain it; package inspection
rejects conflicting or missing metadata. The verified native dispatch API rejects
conflicting overrides and does not send frozen values to a backend specialization
API. Executable C callback generation currently rejects these descriptors because
those callbacks do not yet enforce immutable specialization values. Ordinary
descriptor declarations and the verified Python native dispatch path remain
available. A frozen variant proves only its selected configuration, not every
runtime specialization of the original source.

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
When entries are selected explicitly or through discovery, pattern coverage is
checked against the complete selection for that source, not separately against
each split artifact. Each artifact still evaluates only its own matching rule
using its materialized template parameters. Rules that match no selected entry
remain errors, including rules for entries excluded from the selection.

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
initializer evaluation order is preserved when aggregate fields have side effects.
Private captures evaluate each field once, at the original expression, including
nested array fields, conditional branches and loop updates. Required
Windows numerical controls use the same inputs and guarded outputs as the
original/generated Metal controls. Unknown pointer escapes, incompatible access
contracts, local reference aliases, reference returns, pointer identity operations,
compound pointee updates and external
aggregate buffer layouts remain diagnostic. This representation does not change
the source buffer ABI to store these private handles; thread-local pointer
members remain unsupported.
OpenGL specializes the same private handles against concrete storage buffers;
final subscripts require proven bounds or explicit allocation-derived range
assertions. Unsupported aggregate escapes remain diagnostic on both targets.
Bounds for integer members survive nested value construction, copies, assignments
and straight-line by-value helper returns, including resource-handle rebasing.
Finite nested helper calls retain their argument bounds; they are not recursive
function definitions. Value summaries ignore unused storage pointer parameters, but
reject pointer-dependent results, reference/output parameters and side-effecting
arguments. Chained bounded offsets retain their proofs for reads, writes and
atomic destinations.
Storage specializations distinguish the member bounds at each call site;
an unbounded call cannot reuse a proof from a bounded call. Unknown mutations,
recursive or ambiguous constructors, unmodeled control flow and arithmetic
outside the source integer domain do not establish a range. These proofs permit
representable OpenGL indices, not arbitrary buffer-length or host-dispatch claims.
Source guards also establish bounds for scalars and integer fields of local or
by-value records, including nested records, inside conditional branches and the
bodies of preconditioned loops. Field proofs retain their owning record: writes
to unrelated fields preserve them, while writes to the field or its parent
invalidate them. Pointer-backed, reference and escaped records do not acquire
new field guard proofs. Comparison promotions and intermediate arithmetic
must preserve the source values; mixed signed/unsigned comparisons cannot imply
nonnegativity when they wrap negative operands. Each loop iteration uses its
condition, not just its initializer. Mutation invalidates earlier bounds, escaped
locals do not acquire new guard proofs, and a do-while condition does not constrain
the first iteration. These checks establish index representability, not the
length of a runtime allocation.
Fixed workgroup arrays of 32-bit integer or floating-point elements, including
two- and four-component vectors, can also back these references. Their
allocations retain distinct identities when
source names are shadowed; copies, rebasing and helper calls preserve the shared
storage rather than copying array contents. Access helpers select only compatible
address spaces. Initializers, unsupported layouts, qualified allocations and
dynamic entry-point workgroup bindings remain diagnostic. OpenGL index conversion
still requires a range proof: bounded project assertions can target the generated
``crosstl_workgroup_load_<element>`` and ``crosstl_workgroup_store_<element>``
helpers' ``reference.offset + index`` expressions. Four-lane native controls
exercise barriers, cross-lane reads, multiple allocations and untouched guards.
Vector controls cover rebased shared arrays across three workgroups with exact
output and shared-storage guards. Three-component layouts remain rejected.
Rewritten resource aggregates currently retain file-level source provenance;
fine-grained diagnostic locations within generated helpers are not available.
General typed reinterpretation through aggregate-held references remains unsupported;
the immediate byte-record copy contract below is a separate supported operation.
Calls within the selected resource-helper graph are bound to source overloads
before private handles replace pointers. Selection uses lexical declarations,
array and vector components, scalar arithmetic conversions, and exact aggregate
or cooperative-matrix contracts. A candidate must be no worse for every argument
and better for at least one; unresolved types, ambiguous conversions and unproven
reference bindings remain diagnostic. Selected overloads receive distinct symbols
so target type erasure cannot change the call. Native controls cover scalar and
aggregate arguments, pointer access modes, loop-local shadows and single argument
evaluation. This does not establish support for arbitrary pointer reinterpretation
inside aggregate helpers or for a complete matrix kernel.
Direct storage-buffer vector views in OpenGL also accept indexed addresses such
as ``reinterpret_cast<const device uint2*>(&input[base])[index]``. The scalar
base offset and subsequent vector index retain their distinct byte strides.
Native controls check two- and four-lane 32-bit integer and floating-point views
at nonzero offsets against exact output words and allocation guards. This does
not extend vector reinterpretation through aggregate-held pointers or the Metal
round-trip target.
DirectX also supports immediate storage-to-workgroup copies through aligned
single-byte-array record views when both backings have matching 32-bit scalar
layouts. The copy retains the record's byte extent, captures the addresses once,
and proves the full destination span through helper forwarding and bounded
loops. Required native controls cover integer and floating-point backing words,
nonzero offsets, pointer rebasing, and allocation guards.
OpenGL direct byte-record copies prove the complete source span before narrowing
to signed target indices, including the last expanded element. Bounded offsets
are retained through pointer-helper specialization; mixed calling contexts get
distinct specializations, and pointer mutation invalidates affected bounds.
Unproven or overflowing spans are diagnostic. A shared destination's extent does
not prove a source-buffer range. Native controls include guarded source offsets,
64-bit offset expressions and an extra workgroup that must leave outputs intact.
Immediate byte-record copies through aggregate-held pointers are supported on
DirectX and OpenGL with explicit ``workgroup_access_assertions``. The assertion
selects the source entry, function and destination member path (for example
``cursor.dst``) and covers all absolute destination elements, not only the first
element of each copy. Its inclusive range must contain the complete record and
fit every compatible shared allocation that can reach the destination. A
conservative allocation-origin analysis follows constructors, aggregate fields,
pointer returns, pointer arrays, copies and mutable argument writebacks before
pointer types are replaced by handles. Different source aggregate types remain
distinct; instances of the same type join their possible origins. Unresolved
pointer producers retain all compatible allocations rather than excluding them.
Private handles retain their initialization and do not become shared variables
merely because their pointees use workgroup storage. Missing, conflicting and out-of-bounds
contracts remain diagnostic. Assertions are caller-supplied preconditions; they
do not insert runtime checks or establish that arbitrary input values satisfy
the contract. Source offsets remain signed and wide; OpenGL still requires its
ordinary index-narrowing proofs. This does not establish full matrix-kernel
translation, numerical parity or host-runtime integration.
Metal-imported thread and threadgroup pointer returns preserve their address space
and pointee constness in the intermediate type and Metal round-trip output.
This type preservation does not establish general private-pointer lowering for
DirectX or OpenGL.
Static constants do not contribute instance fields or initializer arguments,
including in nested aggregates, arrays and byte-copy record layouts. Supported
numeric static member reads preserve the constant's declared type and evaluate
the object expression once, including calls and indexed receivers with side
effects. Uninitialized or mutable static state and resource-bearing static
members remain diagnostic; they are not replaced with default values.

HLSL writable arguments capture nonliteral array indices at the call expression
before copy-in and copy-out, including indexed aggregate receivers. The captured
location is reused for writeback; index side effects are not repeated, and captures
remain inside their original conditional or loop evaluation. DirectX and OpenGL
specialize uniquely resolved helpers when multiple source reference arguments
denote the same stable location. The specialized helper uses one shared parameter,
including nested helper calls, so reads observe preceding writes through any alias.
Readonly aliases may share a writable reference; ordinary by-value and target
value-result parameters retain their original semantics. Calls requiring this
specialization reject side-effecting sibling arguments rather than changing their
evaluation. This does not establish general alias analysis for different index
expressions, overlapping aggregate/component views or unresolved overloads.
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

Metal import restores variable types, qualifiers and resource-name bindings
when leaving a nested block, branch, loop body or switch. A local scalar or array
that shadows a buffer parameter cannot change the type of a later access to
the enclosing parameter. The local binding still governs accesses inside its
scope; invalid scalar subscripts remain errors. Compiler and native controls
cover nested shadows, loop-local declarations and helper returns on generated
Metal, HLSL and GLSL. Shared declarations across separate switch cases remain a
distinct target-emission limitation tracked in
`issue #2232 <https://github.com/CrossGL/crosstl/issues/2232>`_.

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

Implementation Notes
--------------------

:doc:`project-porting` describes the individual generated helpers, the source
and target constructs each one covers, and the bounded proofs that exercise
them.

.. toctree::
   :maxdepth: 1

   project-porting
