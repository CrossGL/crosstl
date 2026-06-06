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
     --report crosstl-out/portability-report.json

Project translation exits nonzero when the report contains failed artifacts or
error diagnostics.

Project scan, report, and translation commands also accept repeatable
``--include-dir``, ``--define``, and ``--source-override`` overrides. CLI
defines use ``NAME`` or ``NAME=VALUE`` syntax and override matching names loaded
from ``crosstl.toml``. CLI source overrides use ``PATTERN=BACKEND`` syntax and
override matching source patterns loaded from ``crosstl.toml``. These overrides
are recorded in the emitted project report.

Unsupported target backend names are reported as configuration diagnostics in
scan, report, and translation output. Translation still records per-artifact
failures for any artifact attempt that cannot be generated.

Validate artifacts referenced by a report:

.. code-block:: bash

   python -m crosstl._crosstl validate-project crosstl-out/portability-report.json \
     --format text

Validation exits nonzero when the report metadata is malformed, artifact
records, source-map records, or preserved diagnostics are malformed, source-map
mapping lists are empty or do not contain one file-level mapping, source-map or
diagnostic location spans are internally inconsistent, diagnostic location file
paths or artifact source paths are not repository-relative, project target lists
are not normalized and deduplicated, diagnostic or artifact targets are not
declared by the report, artifact sources are not declared translation units,
embedded validation records reference artifacts not declared by the report,
validation records contain duplicate identities or inconsistent status fields,
summarized embedded validation omits declared artifacts, external corpus entry
presence, discovery, or source-backend fields do not match the project root and
declared units, translated outputs are missing, artifact paths resolve outside
the repository, generated artifact hashes no longer match the files on disk,
source files with recorded hashes are missing or changed, or opt-in toolchain
smoke checks fail.
Toolchain smoke checks only run for translated artifacts that still exist inside
the repository. Each smoke check is bounded by a short subprocess timeout, and
timeouts are reported as failed toolchain runs. Validation reports include
severity, diagnostic-code, and missing-capability rollups for generated and
preserved diagnostics, plus artifact target, hash-status, toolchain status,
and toolchain-run status rollups for validation results. The default output is
JSON; ``--format text`` prints a concise validation summary with the same
rollups, and ``--format sarif`` emits validation diagnostics as SARIF.

Inspect an existing report as a concise JSON, text, or SARIF summary:

.. code-block:: bash

   python -m crosstl._crosstl inspect-report crosstl-out/portability-report.json \
     --format text \
     --max-diagnostics 20 \
     --max-failed-artifacts 20

Report inspection includes validation status, invalid/unavailable report status,
project counts, project configuration path and counts, failed artifacts with
variant labels when present, diagnostic code and missing-capability rollups,
validation diagnostic-code,
missing-capability, and artifact target and variant rollups, report
source-backend, file-extension, and artifact target rollups, source-map count,
granularity, target, and source-backend rollups, artifact matrix completion
counts plus sampled missing and extra artifact identities,
include-directory status counts, inactive source-root and include-directory
record details, diagnostics, configurable diagnostic and failed-artifact
truncation counts, external corpus rollups, sampled missing and
present-but-undiscovered external corpus entries, and migration actions.
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
against the shell's current working directory.

``source_roots`` limits discovery to selected directories. ``include`` and
``exclude`` use shell-style patterns against repository-relative paths. Project
reports include order-preserving source-root status records and status counts
so active, missing, non-directory, outside-project, and scan-visible roots can
be triaged without re-running discovery. Missing source roots, source roots
that resolve to files or other non-directory paths, and roots that resolve
outside the repository are reported as scan or configuration diagnostics.
Include and source override patterns must also be
repository-relative; absolute patterns or patterns containing parent-directory
segments are reported as configuration diagnostics and skipped. Source overrides
allow extensionless or non-standard files to be assigned to a registered source
backend. Override patterns are also considered during default discovery, so
override-only files do not require broad include globs. CLI source overrides are
merged with this configuration before scan or translation. Invalid override
backend names are reported as configuration diagnostics.
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
include and define overrides are merged with this configuration before scan or
translation.
During scan, project reports also record ``#include`` directives discovered in
translation units. Each dependency record keeps the include target, local,
system, or dynamic kind, line and column, and a status of ``resolved``,
``missing``, ``system``, ``dynamic``, or ``outside-project``. Resolved
dependencies record the repository-relative resolved path and whether the match
came from the source directory or a configured include directory. Unresolved
system includes are recorded without warning because they often refer to SDK or
toolchain headers. Missing local includes, dynamic include expressions, and
include paths that resolve outside the repository emit structured
``include.resolution`` diagnostics.
``output_dir`` must resolve inside the repository root; paths that escape the
repository are reported as configuration diagnostics and artifacts are not
written. When named variants are configured, project translation emits one
artifact attempt per variant and passes base defines merged with the variant's
define overrides to the source frontend. Variant artifacts are written under a
variant path segment inside each target output directory, and the original
variant name plus applied define map are recorded on the artifact and variant
name is recorded on validation records. CrossGL source translation applies
object-like define expansion and conditional branch selection for
``#if``/``#ifdef``/``#ifndef``/``#elif``/``#else``/``#endif`` when defines are
provided. Native preprocessor behavior remains backend-dependent.

Configuration scalar values and define/source-override maps are type checked
when ``crosstl.toml`` is loaded. Define names, source override patterns, named
variants, and variant define names must be non-empty strings. Malformed values
are rejected before scan or translation so reports do not silently stringify
invalid project metadata.

``external_corpus_manifest`` points at an optional repository-relative JSON
manifest of pinned upstream shader or GPU-source reductions. Project reports use
the manifest for coverage accounting only: they record declared paths, present
and missing entries, discovered translation units, source-backend and target
rollups, valid and invalid manifest-entry counts, and artifact outcomes for
entries included in the project run. CrossTL does not clone upstream
repositories, run native build systems, or claim whole corpus semantic parity
from this manifest. Malformed manifest entries are reported as configuration
diagnostics and skipped from retained corpus entries. Duplicate manifest paths
or explicit entry ids are also reported and skipped so generated reports do not
inflate corpus coverage. The summary still records how many manifest entries
were skipped.

Project reports include configured define and variant names and values, and
artifact records include the applied define map used for that translation
attempt. Review reports before sharing them outside the repository if those
values include private build metadata. Compact inspection summaries list
configured variant names without printing define names or values.

Report Shape
------------

Project reports are JSON documents with:

- top-level metadata: report schema version, report kind, generation timestamp,
  and generator name/pipeline/package-version fields.
- ``project`` metadata: root, config path, source roots, source-root status
  records and status counts, include/exclude
  patterns, targets, output directory, source override map, include
  directories, include-directory status records and status counts, define and
  variant maps, and counts for source roots, include patterns, exclude
  patterns, source overrides, include directories, defines, and variants.
- ``summary``: total unit/artifact/diagnostic/source-map counts plus rollups by
  unit source backend, unit file extension, skipped reason, skipped file
  extension, unit source override, skipped source override, artifact source
  backend, variant, target backend, source-map granularity, source-map target,
  source-map source backend, include dependency kind, include dependency
  status, diagnostic code (``diagnosticsByCode``), and missing capability
  (``missingCapabilityCounts``).
- ``units``: discovered translation units with repository-relative paths,
  source backend names, path-derived extensions, source hashes, and source
  overrides. Units that contain ``#include`` directives also include
  ``includeDependencies`` records for project-level include triage. Resolved
  include dependencies record repository-relative include paths, resolution
  source, and SHA-256 hashes so report validation can detect include file
  content drift after scan.
- ``skipped``: repository-relative files intentionally left untranslated with
  reason codes and source override metadata when an override selected an
  unsupported source backend. Full reports require skipped source override
  metadata to match the configured source override map.
- ``artifacts``: attempted outputs with source path, source backend, target,
  applied define map, optional variant name, target/variant-scoped output path
  with the target backend suffix, status, source hash, generated artifact hash,
  pipeline provenance, and file-granularity source-map anchors for successful
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
  define overrides.
  Successful artifact records in full reports must include file-level
  source-map anchors. Generated CrossGL artifacts also include a
  compiler-compatible ``source-remap`` sidecar with a file-level
  generated/original mapping for compiler ``--source-remap`` consumers. The
  report records the sidecar path, hash, generated-file identity, and summary
  rollups by target and source backend. The project pipeline does not claim
  fine-grained source-map coverage yet.
  Artifact provenance records the
  ``single-file-translate`` pipeline and uses ``crossgl`` as the intermediate
  marker only when both source and target backends route through the CrossGL
  bridge. Full reports require failed artifacts to carry an actionable error
  string and reject failed artifacts that claim generated hashes or source-map
  records. Full reports also reject translated artifacts that carry error
  metadata. Invalid project output directories are recorded as failed artifacts
  without writing files.
- ``artifactMatrix``: translation-report metadata with expected, emitted,
  translated, failed, missing, extra, and completion counts for the unit, target,
  and variant matrix. Scan-only reports omit this object because they
  intentionally contain no translated or failed artifacts. Report inspection
  also includes sampled missing and extra artifact identities when report
  metadata is sufficient, so incomplete batch outputs are visible without
  opening every artifact record.
- ``externalCorpus``: optional manifest-backed corpus accounting with declared
  entries, present/missing and discovered-unit status, source-backend and target
  rollups, valid/invalid manifest-entry counts, and translated/failed artifact
  outcome counts for manifest entries. Validation checks entry presence against
  the project root, checks discovered/source-backend fields against declared
  translation units, and rejects inconsistent retained-entry and manifest-entry
  summary counts when those fields are present.
- ``diagnostics``: structured diagnostics using severity, code, message,
  location, target, and missing-capability fields compatible with the compiler
  diagnostic contract.
- ``validation``: report contract checks, generated timestamp and generator
  metadata checks, report inspection summaries, failed
  source artifact checks, project metadata, target normalization, and config
  count checks including compact variant-name inspection summaries, unit and
  skipped record shape checks, artifact record shape checks, source and
  generated hash checks, duplicate artifact identity checks, required
  source/generated hash status fields for summarized validation artifacts,
  aggregate validation artifact and hash-status summary counts, direct
  validation report artifact target, variant, hash-status, toolchain status,
  and toolchain-run status rollups, source-root and include-directory status
  record and count consistency checks, unit source hash shape and current-file
  checks, full-report artifact matrix coverage and artifact define map checks,
  artifact matrix emitted/translated/failed/missing/extra/completion count checks,
  full-report source-map granularity, target, and source-backend rollup checks,
  source-remap target and source-backend rollup checks,
  source hash checks, failed artifact error metadata checks, translated artifact
  error metadata rejection, required artifact provenance and provenance value
  checks, failed artifact generated metadata rejection, required translated
  artifact source maps, required CrossGL artifact source remaps, source-map
  record shape, non-empty mapping list, single file-level mapping, span
  consistency, anchor consistency, source-remap metadata shape, sidecar hash,
  and sidecar content checks, external
  corpus record, per-entry artifact count, and summary checks, summary
  consistency checks, migration action shape and target declaration checks,
  preserved diagnostic shape, repository-relative file path, span consistency,
  target declaration checks, scan-scope count consistency, validation
  toolchain status consistency checks, validation artifact and toolchain run
  record shape and duplicate identity checks, validation artifact and
  toolchain target coverage and status consistency checks,
  include dependency record shape and include dependency summary consistency,
  current include dependency status, resolved-path, resolved-hash, and
  resolution-source checks,
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
  action has a documented kind, severity, message, and target list. Scan-only
  reports include supported requested targets when translation units are
  present. Translation reports scope ``manual-runtime-integration`` to targets
  that produced translated artifacts, covering host runtime API, resource
  binding, build script, and backend integration review.
