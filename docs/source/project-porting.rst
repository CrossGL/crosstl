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

Scan a repository and print a JSON report:

.. code-block:: bash

   python -m crosstl._crosstl scan /path/to/repo --target metal

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

   python -m crosstl._crosstl validate-project crosstl-out/portability-report.json

Validation exits nonzero when the report metadata is malformed, artifact
records, source-map records, or preserved diagnostics are malformed, artifact
targets are not declared by the report, translated outputs are missing, artifact
paths resolve outside the repository, or opt-in toolchain smoke checks fail.
Toolchain smoke checks only run for translated artifacts that still exist inside
the repository.

Inspect an existing report as a concise JSON or text summary:

.. code-block:: bash

   python -m crosstl._crosstl inspect-report crosstl-out/portability-report.json \
     --format text

Report inspection includes validation status, project counts, failed artifacts,
diagnostics, external corpus rollups, and migration actions. It exits nonzero
when validation finds report errors.

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

``source_roots`` limits discovery to selected directories. ``include`` and
``exclude`` use shell-style patterns against repository-relative paths. Missing
source roots and roots that resolve outside the repository are reported as scan
diagnostics. Include and source override patterns must also be
repository-relative; absolute patterns or patterns containing parent-directory
segments are reported as configuration diagnostics and skipped. Source overrides
allow extensionless or non-standard files to be assigned to a registered source
backend. Override patterns are also considered during default discovery, so
override-only files do not require broad include globs. CLI source overrides are
merged with this configuration before scan or translation. Invalid override
backend names are reported as configuration diagnostics.
Include directories, defines, and named
variants are recorded in project reports. Missing include directories are
reported as configuration diagnostics. Include directories that resolve outside
the repository are reported as non-blocking configuration diagnostics so reports
retain portability and provenance context. Include directories and defines are
passed to source frontends that expose preprocessor options. CLI include and
define overrides are merged with this configuration before scan or translation.
``output_dir`` must resolve inside the repository root; paths that escape the
repository are reported as configuration diagnostics and artifacts are not
written. When named variants are configured, project translation emits one
artifact attempt per variant and passes base defines merged with the variant's
define overrides to the source frontend. Variant artifacts are written under a
variant path segment inside each target output directory, and the original
variant name is recorded on the artifact and validation records. Native
preprocessor behavior remains backend-dependent.

Configuration scalar values and define/source-override maps are type checked
when ``crosstl.toml`` is loaded. Malformed values are rejected before scan or
translation so reports do not silently stringify invalid project metadata.

``external_corpus_manifest`` points at an optional repository-relative JSON
manifest of pinned upstream shader or GPU-source reductions. Project reports use
the manifest for coverage accounting only: they record declared paths, present
and missing entries, discovered translation units, source-backend and target
rollups, and artifact outcomes for entries included in the project run. CrossTL
does not clone upstream repositories, run native build systems, or claim whole
corpus semantic parity from this manifest. Malformed manifest entries are
reported as configuration diagnostics and skipped before corpus accounting so
generated project reports remain contract-valid.

Project reports include configured define and variant names and values. Review
reports before sharing them outside the repository if those values include
private build metadata.

Report Shape
------------

Project reports are JSON documents with:

- top-level metadata: report schema version, report kind, generation timestamp,
  and generator name/pipeline/package-version fields.
- ``project`` metadata: root, config path, source roots, include/exclude
  patterns, targets, output directory, source override map, include
  directories, define and variant maps, and source override/define/variant
  counts.
- ``summary``: total unit/artifact/diagnostic/source-map counts plus rollups by
  source backend and target backend.
- ``units``: discovered translation units with repository-relative paths,
  source backend names, and source overrides.
- ``artifacts``: attempted outputs with source path, source backend, target,
  optional variant name, output path, status, source hash, pipeline provenance,
  and file-granularity source-map anchors for successful translations. Invalid
  project output directories are recorded as failed artifacts without writing
  files.
- ``externalCorpus``: optional manifest-backed corpus accounting with declared
  entries, present/missing and discovered-unit status, source-backend and target
  rollups, and translated/failed artifact outcomes for manifest entries.
- ``diagnostics``: structured diagnostics using severity, code, message,
  location, target, and missing-capability fields compatible with the compiler
  diagnostic contract.
- ``validation``: report contract checks, generated timestamp and generator
  metadata checks, report inspection summaries, failed
  source artifact checks, project metadata and config count checks, unit and
  skipped record shape checks, artifact record shape checks, source hash and
  provenance checks, source-map record shape and anchor consistency checks,
  external corpus record and summary checks, summary consistency checks,
  migration action shape checks, preserved diagnostic shape checks, validation
  result and toolchain run record shape checks, artifact target and variant
  declaration checks, translated artifact existence checks, escaped
  artifact-path checks, optional external toolchain availability, and opt-in
  toolchain smoke results.
- ``migration``: actionable manual follow-up work outside shader/kernel
  translation. Each action has a documented kind, severity, message, and target
  list; the current project pipeline emits ``manual-runtime-integration`` for
  host runtime API, resource binding, build script, and backend integration
  review.
