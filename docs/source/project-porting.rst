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

   python -m crosstl._crosstl scan /path/to/repo \
     --target metal \
     --target opengl \
     --output scan-report.json

   python -m crosstl._crosstl translate-project /path/to/repo \
     --target metal \
     --target opengl \
     --output-dir crosstl-out \
     --report crosstl-out/portability-report.json

   python -m crosstl._crosstl validate-project \
     crosstl-out/portability-report.json \
     --format text

   python -m crosstl._crosstl inspect-report \
     crosstl-out/portability-report.json \
     --format text

Use these report fields to decide the next action:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Report field
     - Triage use
   * - ``diagnosticCounts`` and ``diagnosticsByCode``
     - Separate configuration errors from source/backend translation failures
       before reviewing artifacts.
   * - ``missingCapabilityCounts``
     - Group unsupported source features, include resolution gaps, define
       forwarding gaps, artifact manifest issues, provenance issues, and
       optional toolchain validation gaps.
   * - ``artifactMatrix``
     - Confirm the expected unit, target, and named-variant artifact plan before
       translation, then identify missing or extra artifacts after translation.
   * - ``validation``
     - Check current source hashes, generated artifact hashes, source maps,
       source remaps, optional toolchain availability, and opt-in artifact or
       availability smoke test results after translation.
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

   python -m crosstl._crosstl examples/graphics/SimpleShader.cgl --backend metal

The explicit single-file subcommand is equivalent:

.. code-block:: bash

   python -m crosstl._crosstl translate examples/graphics/SimpleShader.cgl --backend metal

Both single-file forms also accept ``--source-backend``, repeatable
``--include-dir``, and repeatable ``--define`` overrides. Use them when a file
has a nonstandard extension or when the selected source frontend exposes
include-path and preprocessor define options.

Scan a repository and print a JSON report:

.. code-block:: bash

   python -m crosstl._crosstl scan /path/to/repo --target metal

Emit the same scan-only portability report with an explicit output path:

.. code-block:: bash

   python -m crosstl._crosstl report /path/to/repo \
     --target metal \
     --output crosstl-out/portability-report.json

Scan and report commands exit nonzero when the generated report contains error
diagnostics, while still writing the JSON report to stdout or the requested
output file.

Translate every discovered unit to one or more targets:

.. code-block:: bash

   python -m crosstl._crosstl translate-project /path/to/repo \
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
``--run-toolchains`` implies artifact validation and records any available
bounded toolchain smoke-check results in the generated portability report.
Smoke-check records include a check kind so report consumers can distinguish
artifact checks from target tool availability checks.

Project scan, report, and translation commands also accept repeatable
``--source-root``, ``--include-dir``, ``--define``, and ``--source-override``
overrides. CLI source roots replace the configured source roots for that
command. CLI defines use ``NAME`` or ``NAME=VALUE`` syntax and override matching
names loaded from ``crosstl.toml``. CLI source overrides use ``PATTERN=BACKEND``
syntax and override matching source patterns loaded from ``crosstl.toml``.
These overrides are recorded in the emitted project report.
Scan, report, and translation commands accept repeatable ``--variant NAME``
selectors. Scoped scan and report output evaluates only the selected declared
variants for variant-aware include and define metadata, records the selected
variant list, and does not claim omitted variants as scanned.

Unsupported target backend names are reported as configuration diagnostics in
scan, report, and translation output. Translation still records per-artifact
failures for any artifact attempt that cannot be generated.

Validate artifacts referenced by a report:

.. code-block:: bash

   python -m crosstl._crosstl validate-project crosstl-out/portability-report.json \
     --format text

Validation exits nonzero when the report metadata is malformed, artifact
records, source-map records, or preserved diagnostics are malformed, source-map
mapping lists are empty, file-granularity source maps do not contain one
file-level mapping, finer-grained mappings are not positive-length or fall
outside artifact-level file anchors, source-map or diagnostic location spans
are internally inconsistent, diagnostic location
file paths or artifact source paths are not repository-relative, project target
lists are not normalized and deduplicated, diagnostic or artifact targets are
not declared by the report, artifact sources are not declared translation units,
embedded validation records reference artifacts not declared by the report,
embedded toolchain runs reference failed report artifacts,
validation records contain duplicate identities or inconsistent status fields,
summarized embedded validation omits declared artifacts, external corpus entry
presence, discovery, or source-backend fields do not match the project root and
declared units, translated outputs are missing, artifact paths resolve outside
the repository, generated artifact hashes no longer match the files on disk,
source files with recorded hashes are missing or changed, or opt-in toolchain
smoke checks fail.
Toolchain smoke checks only run for translated artifacts that still exist inside
the repository. Each smoke check is bounded by a short subprocess timeout, and
timeouts are reported as failed toolchain runs. DirectX and Metal hooks currently
record target tool availability because full compiler invocation needs backend-
specific entry point and profile selection. Validation reports include
severity, diagnostic-code, and missing-capability rollups for generated and
preserved diagnostics, plus artifact target, artifact source-backend,
artifact variant, hash-status, source-map status, source-remap status,
toolchain status, toolchain-run status, toolchain-run target, toolchain-run
source backend, toolchain-run check kind, and toolchain-run variant rollups for
validation results.
The JSON validation report uses schema version 1 with a fixed top-level field
set so automation can detect contract drift.
The default output is JSON; ``--format text``
prints a concise validation summary with the same rollups, and ``--format
sarif`` emits validation diagnostics as SARIF.

Inspect an existing report as a concise JSON, text, or SARIF summary:

.. code-block:: bash

   python -m crosstl._crosstl inspect-report crosstl-out/portability-report.json \
     --format text \
     --max-diagnostics 20 \
     --max-failed-artifacts 20 \
     --max-source-map-artifacts 20 \
     --max-artifact-matrix-artifacts 20 \
     --max-artifact-provenance-artifacts 20 \
     --max-define-processing-artifacts 20 \
     --max-include-path-processing-artifacts 20 \
     --max-include-dependencies 20 \
     --max-validation-artifacts 20 \
     --max-toolchain-runs 20 \
     --max-migration-actions 20 \
     --max-external-corpus-entries 20

Report inspection includes validation status, invalid/unavailable report status,
project counts, project configuration path and counts, failed artifacts with
variant labels when present, diagnostic code and missing-capability rollups,
validation diagnostic-code, missing-capability, artifact target,
artifact source-backend, artifact variant, hash-status, source-map status,
source-remap status, toolchain status, and toolchain-run rollups, report
source-backend, file-extension, and artifact
target rollups, source-map count, granularity, target, and source-backend
rollups, artifact matrix completion counts, target and variant completion
rollups, sampled missing and extra artifact identities,
bounded validation artifact and validation toolchain-run samples with
truncation counts,
include-directory status counts, inactive source-root and include-directory
record details, diagnostics, configurable diagnostic and failed-artifact
truncation counts, external corpus rollups, sampled missing and
present-but-undiscovered external corpus entries with configurable sample
limits, and migration actions.
Inspection sample-limit options accept non-negative integer counts and default
to ``20`` for each sampled report section.
The JSON inspection report uses schema version 1 with a fixed top-level field
set so automation can detect contract drift while optional report sections
remain data-dependent.
Migration action inspection is bounded and records truncation counts for large
reports.
``--format sarif`` emits the inspection diagnostics as SARIF for
code-scanning workflows. Inspection exits nonzero when validation finds report
errors.

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
Invalid override backend names are reported as configuration diagnostics.
Explicit broad include patterns may also match compiled shader artifacts or
known source formats that CrossTL cannot parse yet. Project scans keep those
files in the skipped-file rollups and emit structured diagnostics with the
same specific guidance as single-file translation, while continuing to discover
supported translation units in the repository.
Include directories, defines, and named
variants are recorded in project reports. Project reports include
order-preserving include-directory status records and status counts so missing,
non-directory, outside-project, and frontend-visible active include directories
can be triaged without re-running discovery. Missing include directories,
include entries that resolve to files or other non-directory paths, and include
directories that resolve outside the repository are reported as non-blocking
configuration diagnostics so reports retain portability and provenance context.
Existing include directories that remain inside the repository, plus configured
defines, are passed to source frontends that expose preprocessor options. CLI
include and define overrides are merged with this configuration before scan,
report, or translation. Translation artifacts record ``defineProcessing`` metadata so
reports distinguish define maps that were forwarded to the source lexer from
define maps that were not requested or could not be consumed by that frontend.
When configured defines cannot be forwarded, translation reports emit a
non-blocking warning diagnostic with a missing-capability rollup so the
limitation appears in validation and inspection summaries.
Summary, inspection payloads, and text reports also include define-processing
rollups by source backend and by named variant when variants are configured, so
frontend-specific and variant-specific preprocessing gaps are visible without
reading every artifact record. Report inspection also includes sampled artifact
define-processing metadata with status, frontend support, and define counts,
but not define values, so artifact-level preprocessing state can be triaged
without exposing configuration values.
They also record ``includePathProcessing`` metadata so active include-directory
paths can be distinguished from include paths that were not requested or could
not be consumed by the selected source frontend. Include-path processing
warnings are also emitted when active include paths cannot be forwarded, so
the report diagnostics identify affected source frontends without failing the
batch translation.
summaries and text reports also roll up by source backend and by named variant
when variants are configured. Report inspection includes sampled artifacts
whose active include paths could not be forwarded, so the affected source,
target, and frontend are visible without reading every artifact record. It also
includes sampled artifact include-path processing metadata with status,
frontend support, and include path counts, so artifact-level include forwarding
state can be triaged from the report summary.
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
project and variant define maps; unsupported conditional expressions remain
open so discovery does not hide possible dependencies.
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
from a project define, diagnostics identify that define.
Report inspection samples resolved include dependencies, unresolved system
include dependencies, and include issues, including the source location,
source backend, include kind,
resolved path, and resolution source where available. Define-backed include
samples also retain the project define name that supplied the include target and
the variant name when the dependency came from a named variant define map.
``output_dir`` must resolve inside the repository root; paths that escape the
repository are reported as configuration diagnostics and artifacts are not
written. When named variants are configured, project translation emits one
artifact attempt per variant and passes base defines merged with the variant's
define overrides to the source frontend. Variant artifacts are written under a
variant path segment inside each target output directory, and the original
variant name plus applied define map are recorded on the artifact and variant
name is recorded on validation records. ``--variant NAME`` can be repeated to
scope scan, report, or translation runs to declared variants; scoped reports
declare only the selected variants, de-duplicate repeated selections before
planning, and do not claim omitted variants as scanned or attempted.
CrossGL source translation applies object-like define expansion and conditional
branch selection for ``#if``/``#ifdef``/``#ifndef``/``#elif``/``#else``/``#endif``
when defines are provided. Project translation also passes selected variant
define maps into native source frontends that expose define options; current
project coverage includes OpenGL/GLSL conditional branches, CUDA/HIP runtime
system include preservation, CUDA/HIP local header expansion, and project
include directories through that path. Other native preprocessor behavior
remains backend-dependent.

Configuration scalar values and define/source-override maps are type checked
when ``crosstl.toml`` is loaded. Define names, source override patterns, named
variants, and variant define names must be non-empty strings. Malformed values
are rejected before scan or translation so reports do not silently stringify
invalid project metadata.

``external_corpus_manifest`` points at an optional repository-relative JSON
manifest of pinned upstream shader or GPU-source reductions. When configured,
the manifest path must be a non-empty string. Project reports use the manifest
for coverage accounting only: they record declared paths, present and missing
entries, discovered translation units, source-backend and target rollups, valid
and invalid manifest-entry counts, and artifact outcomes for entries included
in the project run. CrossTL does not clone upstream repositories, run native
build systems, or claim whole corpus semantic parity from this manifest.
Malformed manifest entries are reported as configuration diagnostics and
skipped from retained corpus entries. Duplicate manifest paths or explicit
entry ids are also reported and skipped so generated reports do not inflate
corpus coverage. The summary still records how many manifest entries were
skipped.

Project reports include configured define and variant names and values, and
artifact records include the applied define map used for that translation
attempt. Review reports before sharing them outside the repository if those
values include private build metadata. Compact inspection summaries list
configured variant names and per-variant define counts without printing define
names or values.

Report Shape
------------

Project reports are JSON documents with:

- top-level metadata: report schema version, report kind, generation timestamp,
  and generator name/pipeline/package-version fields.
- ``project`` metadata: root, config path, source roots, source-root status
  records and status counts, include/exclude
  patterns, targets, output directory, source override map, include
  directories, include-directory status records and status counts, define and
  variant maps, per-variant define counts, and counts for source roots,
  include patterns, exclude patterns, source overrides, include directories,
  defines, and variants.
- ``summary``: total unit/artifact/diagnostic/source-map counts plus rollups by
  unit source backend, unit file extension, skipped reason, skipped file
  extension, unit source override, skipped source override, artifact source
  backend, variant, target backend, source-map granularity, source-map target,
  source-map source backend, source-map variant, source-remap target,
  source-remap source backend, source-remap variant, include dependency kind,
  include dependency status, include dependency source backend, include
  dependency source-backend status, include dependency resolution source,
  include dependency variant, diagnostic code (``diagnosticsByCode``), and
  missing capability (``missingCapabilityCounts``).
- ``units``: discovered translation units with stable repository-relative POSIX
  paths,
  source backend names, path-derived extensions, source hashes, and source
  overrides. Units that contain ``#include`` directives also include
  ``includeDependencies`` records for project-level include triage. Resolved
  include dependencies record repository-relative include paths, resolution
  source, and SHA-256 hashes so report validation can detect include file
  content drift after scan. Full report validation also re-scans current source
  files and rejects missing or extra include dependency records. Recursive include
  scans stop at include cycles and
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
  status, source hash, generated artifact hash,
  pipeline provenance, and file-span source-map anchors for successful
  translations. Full reports require every artifact to carry a source hash,
  artifact source hashes to match their declared translation-unit source
  hashes,
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
  The report records the sidecar path, hash, generated-file identity, summary
  rollups by target and source backend, and bounded inspection samples for
  source-map and source-remap artifacts. Validation checks that artifact-level
  source-map spans still cover the current source and generated files, recomputes
  line-preserving mappings, and checks that compiler source-remap sidecars use
  the closed schema-1 field set. The project pipeline emits line-granularity
  source maps only when the generated artifact is line-preserving after newline
  normalization; translated artifacts keep file-granularity source maps until
  backend pipelines expose generated line, statement, or token provenance.
  Artifact provenance records the
  ``single-file-translate`` pipeline and uses ``crossgl`` as the intermediate
  marker only when both source and target backends route through the CrossGL
  bridge. Report summaries and inspections include provenance rollups by
  pipeline, intermediate marker, and source backend with intermediate marker,
  plus bounded artifact provenance samples for direct and bridge artifacts.
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
  artifact-derived matrix metadata, so incomplete batch outputs are visible
  without opening every artifact record.
- ``externalCorpus``: optional manifest-backed corpus accounting with declared
  entries, present/missing and discovered-unit status, source-backend and target
  rollups, valid/invalid manifest-entry counts, and translated/failed artifact
  outcome counts for manifest entries. Validation checks entry presence against
  the project root, checks discovered/source-backend fields against declared
  translation units, and rejects missing or inconsistent retained-entry and
  manifest-entry summary counts.
- ``diagnostics``: structured diagnostics using severity, code, message,
  location, target, and missing-capability fields compatible with the compiler
  diagnostic contract. Project-level include and define forwarding limitations
  are warnings, not translation failures.
- ``validation``: report contract checks, generated timestamp and generator
  metadata checks, report inspection summaries, failed
  source artifact checks, project metadata, target normalization, and config
  count checks including compact variant-name and variant-define-count
  inspection summaries, unit and skipped record shape checks, artifact record
  shape checks, source and generated hash checks, duplicate artifact identity
  checks, required
  source/generated hash and source-map/source-remap status fields for
  summarized validation artifacts, aggregate validation artifact and validation
  status summary counts, direct validation report artifact target, source
  backend, variant, hash-status, source-map status, source-remap status,
  toolchain status, toolchain-run status rollups, toolchain-run check-kind
  metadata, and a closed standalone validation-report field set,
  failed-artifact text with
  non-OK hash, source-map, and source-remap statuses, bounded validation artifact
  and validation toolchain-run inspection samples, source-root and
  include-directory status record and count consistency checks, unit source
  hash shape and current-file
  checks, full-report artifact matrix coverage and artifact define map checks,
  artifact define-processing metadata and status/source-backend/variant rollup
  checks,
  artifact include-path processing metadata and status/source-backend/variant
  rollup checks,
  artifact matrix emitted/translated/failed/missing/extra/completion count and
  target/variant rollup checks,
  full-report source-map granularity, target, source-backend, and variant
  rollup checks,
  source-remap target, source-backend, and variant rollup checks,
  source hash checks, failed artifact error metadata checks, translated artifact
  error metadata rejection, required artifact provenance and provenance value
  checks, failed artifact generated metadata rejection, required translated
  artifact source maps, required CrossGL artifact source remaps, source-map
  record shape, non-empty mapping list, file-level mapping cardinality,
  positive-length finer-grained mappings, finer-grained mapping containment
  within artifact-level anchors, span consistency, anchor consistency, current
  file-level source-map span coverage,
  source-remap metadata shape, sidecar hash, closed compiler sidecar field
  sets, and sidecar content checks, external
  corpus record, per-entry artifact count, required manifest-entry accounting,
  and summary checks, summary consistency checks, migration action shape,
  rollup, and target declaration checks,
  preserved diagnostic shape, repository-relative file path, span consistency,
  target declaration checks, scan-scope count consistency, validation
  toolchain status consistency checks, validation artifact and toolchain run
  record shape and duplicate identity checks, validation artifact coverage,
  toolchain-run target, source-backend, and variant rollups, toolchain target
  coverage and status consistency checks,
  include dependency record shape and include dependency summary consistency,
  current include dependency status, resolved-path, resolved-hash,
  source-backend status rollup, resolution-source checks, and project-define
  include provenance checks, resolved and unresolved include inspection samples,
  artifact source, source-backend,
  target, variant, and source-relative output layout declaration checks,
  translated artifact existence checks, escaped output directory and
  artifact-path checks, source artifact existence and hash mismatch checks,
  generated artifact hash mismatch checks, optional external toolchain
  availability, and opt-in toolchain smoke results including bounded timeout
  failures.
- ``migration``: actionable manual follow-up work outside shader/kernel
  translation. The report records documented non-goals for runtime API
  migration, build-system rewrites, and backend framework integration. Each
  action has a documented kind, severity, message, and target list, plus
  action count and kind, severity, and target rollups. Scan-only reports
  include supported requested targets when translation units are present.
  Translation reports scope ``manual-runtime-integration`` to targets that
  produced translated artifacts, covering host runtime API, resource binding,
  build script, and backend integration review.
