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
override-only files do not require broad include globs. Invalid override backend
names are reported as configuration diagnostics.
Include directories, defines, and
variants are recorded in project reports. Missing include directories are
reported as configuration diagnostics. Include directories that resolve outside
the repository are reported as non-blocking configuration diagnostics so reports
retain portability and provenance context. Include directories and defines are
passed to source frontends that expose preprocessor options. ``output_dir`` must
resolve inside the repository root; paths that escape the repository are reported
as configuration diagnostics and artifacts are not written. Named variant
expansion through every native preprocessor remains a tracked project-porting
capability, and reports emit structured warnings when variants are present but
not yet expanded.

Project reports include configured define and variant names and values. Review
reports before sharing them outside the repository if those values include
private build metadata.

Report Shape
------------

Project reports are JSON documents with:

- ``project`` metadata: root, config path, source roots, include/exclude
  patterns, targets, output directory, include directories, define and variant
  maps, and define/variant counts.
- ``summary``: total unit/artifact/diagnostic/source-map counts plus rollups by
  source backend and target backend.
- ``units``: discovered translation units with repository-relative paths,
  source backend names, and source overrides.
- ``artifacts``: attempted outputs with source path, source backend, target,
  output path, status, source hash, pipeline provenance, and file-granularity
  source-map anchors for successful translations. Invalid project output
  directories are recorded as failed artifacts without writing files.
- ``diagnostics``: structured diagnostics using severity, code, message,
  location, target, and missing-capability fields compatible with the compiler
  diagnostic contract.
- ``validation``: report contract checks, failed source artifact checks,
  project metadata checks, artifact record shape checks, source hash and
  provenance checks, source-map record shape checks, preserved diagnostic shape
  checks, artifact target declaration checks, translated artifact existence
  checks, escaped artifact-path checks, optional external toolchain
  availability, and opt-in toolchain smoke results.
- ``migration``: actionable manual follow-up work outside shader/kernel
  translation.
