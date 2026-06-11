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
   named variants, output directory, targets, and optional external corpus
   manifest are explicit.
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

Downstream runtimes implement the ``RuntimeAdapter`` protocol or subclass
``RuntimeExecutor`` and receive the merged contract in ``run(request)``. For
example, an MLX validation adapter can consume translated Metal artifacts and
map neutral fixture buffers and function constants to MLX runtime objects, but
the fixture contract remains expressed in terms of entry points, bindings,
constants, dispatch geometry, and validation hooks rather than MLX APIs.

The translator stops at this contract boundary. Full framework rewrites,
non-kernel host API ports, application command scheduling, target SDK
installation, build-system migration, memory lifetime policy, and production
runtime framework generation remain downstream integration work.

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
or build-system tooling can decide what to run next. The plan remains metadata
only: it does not rewrite host application code, execute device code, generate
runtime framework code, or install target SDKs.

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

   [project.sources]
   "legacy/**/*.shader" = "cgl"

   [project.defines]
   USE_FAST_PATH = "1"

   [project.source_options.metal]
   max_template_specializations = 2048

   [project.variants.debug]
   USE_FAST_PATH = "0"

An explicit ``--config`` path may be absolute or repository-relative. Relative
config paths are resolved against the repository root passed to the command, not
against the shell's current working directory. When ``--config`` is provided,
the referenced file must exist; otherwise the command exits with an error
instead of silently falling back to default scan settings.

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
support ``max_template_specializations`` for project-specific explicit helper
materialization budgets. Project reports include
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

Project reports include configured define and variant names and values, and
artifact records include the applied define map used for that translation
attempt. Review reports before sharing them outside the repository if those
values include private build metadata. Compact inspection summaries list
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
  variant maps, per-variant define counts, and counts for source roots,
  include patterns, exclude patterns, source overrides, include directories,
  defines, and variants.
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
