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
     - Confirm that a requested materialized source entry was packaged under its
       deterministic entry path and identify the reflected target entry and
       stage.
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

Entry-Scoped Compute Artifacts
------------------------------

Repositories can request a standalone artifact for one materialized source
entry by adding a repository-relative selector table to ``crosstl.toml``:

.. code-block:: toml

   [project]
   include = ["kernels/arange.metal"]
   include_dirs = ["."]
   targets = ["directx", "opengl"]
   output_dir = "crosstl-out"

   [project.entry_points]
   "kernels/arange.metal" = "arangeuint32"

For DirectX compute output, the selected source entry is emitted as target entry
``CSMain``. A source such as ``kernels/arange.metal`` produces
``crosstl-out/directx/kernels/arange/arangeuint32.hlsl``. The standalone HLSL
retains only the selected entry's reachable helpers, resources, constants, and
execution contract. Explicit registers and spaces remain unchanged, while
runtime-loader metadata records the selected ``cs_6_0`` entry profile.

For OpenGL compute output, the same selection produces
``crosstl-out/opengl/kernels/arange/arangeuint32.glsl`` with target entry
``main``. For both targets, the portability report records the source entry,
target entry, and reflected stage on the artifact; embedded validation records
carry the same identity. Runtime artifact manifests then reflect only the
selected stage interface from that standalone output.

Selection is exact after source materialization. Missing or ambiguous entries
fail with structured diagnostics and no target file. Targets that do not yet
implement standalone entry generation also fail explicitly instead of pruning
an aggregate artifact. When ``project.entry_points`` is absent, project
translation keeps the existing aggregate output path and behavior.

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

The portability report records the configured tables under
``project.indexRangeAssertions`` and their count under
``project.indexRangeAssertionCount``. Report consumers can therefore audit the
host/runtime assumptions used during translation alongside the generated
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

CrossTL includes ``crosstl.project.native_runtime_drivers:VulkanComputeRuntime``
as an optional reference driver for simple Vulkan compute fixtures. The driver
imports the Python ``vulkan`` binding lazily, reports structured unavailability
when the binding, loader, or a compute-capable device is unavailable, and
currently supports storage-buffer fixtures with 32-bit scalar element types and
a single descriptor set. DirectX and OpenGL native dispatch drivers remain
downstream or follow-up integration work.

The translator stops at this contract boundary. Full framework rewrites,
non-kernel host API ports, application command scheduling, target SDK
installation, build-system migration, memory lifetime policy, and production
runtime framework generation remain downstream integration work.

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

   [project.subgroup_width_rules]
   "kernels/wave.metal" = "WIDTH"

   [project.source_options.metal]
   max_template_specializations = 2048
   max_template_materialization_work = 131072

   [project.source_options.metal.source_patterns."kernels/scan*.metal"]
   max_template_specializations = 4096

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
project-level table and overrides the same selector for that named variant.
Artifact ``specializationConstants`` records retain the effective value and
``valueProvenance`` with the project or variant configuration path, selector,
selector kind, and variant name. Using a name in one table and the corresponding
ID in another still creates two matches, so different values fail as a
conflicting contract rather than relying on table precedence.

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

Targets that do not implement per-entry workgroup specialization reject a
matching rule before source materialization or target generation. The failed
artifact and structured ``execution-specialization`` diagnostic retain the
selected rule, target, and supported target set so the configuration cannot be
silently ignored.

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
provenance. Report validation re-evaluates every expression, verifies the
materialization join and hashes, and checks the generated target entry metadata.
These records describe shader or kernel translation and dispatch requirements;
they do not rewrite framework runtime code or establish numerical runtime
parity.

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

OpenGL cannot enforce an exact subgroup width through this project contract, so
a matching rule fails before GLSL generation with
``project.translate.subgroup-width-enforcement-unsupported`` and reason
``opengl-enforcement-unavailable``. Every other target currently fails closed
with the same diagnostic and reason ``target-not-supported``. These failures
record the missing ``execution.subgroup-width-specialization`` capability, rule
provenance, and the supported target set without emitting a misleading target
artifact.

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
DXC on Windows. Its unconfigured OpenGL project checks deferred
``layout(constant_id = 20)`` emission and validates generated OpenGL SPIR-V on
Linux. This proves translation and native compilation only; it does not claim
RMSNorm numerical runtime parity or full MLX test-suite support. Numerical
execution also requires host dispatch values to match each compiled artifact's
workgroup-size contract.

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
to source lexers that expose matching keyword options. Metal source imports
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
