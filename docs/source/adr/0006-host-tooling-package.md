# ADR 0006: Move the runtime and host tooling into a separate `crosstl-host` distribution

- Status: accepted; implemented in stages (stage 1 lands with this record)
- Date: 2026-11-20
- Scope: `crosstl/_crosstl.py` (the command line), `crosstl/project/`, `pyproject.toml`, and the import boundary of
  `crosstl/translator/` and `crosstl/backend/`

## Context

The `crosstl` distribution ships two products in one package:

- **The translator.** `crosstl.translate`, `translate_with_metadata` and `translate_entries`, the front ends in
  `crosstl/backend/` (201,103 lines in 425 modules at `1ebda13c`) and the IR, semantics and code generators in
  `crosstl/translator/` (442,920 lines in 829 modules). Its contract is per-artifact: valid target source plus
  reflection metadata for every artifact (bindings, constants, workgroup size, buffer layouts).
- **Runtime and host tooling.** `crosstl/project/` (107,660 lines in 47 modules, plus a Swift Metal worker and a C++
  Direct3D 12 worker) plans how a host application loads, binds, packages, dispatches and verifies translated
  artifacts: runtime manifests, binding manifests, native loader ABIs, loader scaffolds, runtime packages, variant
  registries, host integration hand-offs, and the native runtime drivers that execute them.

The command line exposes both through 29 subcommands of `crosstl`. Six concern translation:

| Translation | Purpose |
|---|---|
| `translate` | one file to one target |
| `scan` | find translatable sources in a repository |
| `translate-project` | translate a repository and write the portability report |
| `validate-project` | check a portability report |
| `inspect-report` | summarise a portability report |
| `doctor` | list the native validators CrossTL can find |

The other 23 concern the host: `plan-runtime`, `runtime-manifest`, `runtime-test-manifest`, `test-runner-plan`,
`inspect-test-runner-plan`, `execute-test-runner`, `runtime-binding-manifest`, `package-runtime`,
`inspect-runtime-package`, `plan-host-bindings`, `plan-runtime-adapters`, `materialize-runtime-adapters`,
`runtime-loader-manifest`, `native-loader-abi`, `native-loader-abi-package`, `runtime-variant-registry`,
`scaffold-host-loaders`, `inspect-host-loader-scaffolds`, `plan-host-loader-consumption`,
`host-integration-handoff`, `inspect-host-integration-handoff`, `plan-host-integration-execution` and
`execute-host-integration`.

Measured facts that bear on the decision:

- **The dependency already points one way.** No module under `crosstl/translator/` or `crosstl/backend/` imports
  `crosstl.project` or the command-line module; three mention `crosstl.project` in docstrings only. The host side
  consumes the translator's output and metadata, never the reverse.
- **The split inside `crosstl/project/` is not yet clean.** `pipeline.py` (59,338 lines) defines the translation
  entry points `scan_project` (line 9,845) and `translate_project` (29,132) next to host planning such as
  `plan_runtime_integration` (33,037) and `build_runtime_package` (35,420). It imports 11 sibling modules; the
  other 34 modules (40,468 lines) are not imported by `pipeline.py`, and 7 of them import it.
- **The host tooling has public users.** `docs/source/project-porting.rst` and `docs/source/porting-guide.rst`
  document `from crosstl.project import ...`. In the repository, 173 Python files under
  `demos/integrations/mlx/`, 1 under `demos/integrations/computesharp/` and 326 under `tests/` import
  `crosstl.project`.
- **The command line is part of the stable API.** The M6 milestone of the core contract requires `crosstl.translate`
  unchanged and subcommands deprecated rather than removed.

The two products differ in what they need. The translator needs pure-Python code, deterministic output and native
validators in CI. The host tooling needs native runtimes (Metal, Direct3D 12, OpenGL, Vulkan), platform workers
and macOS and Windows runners, and it changes at the pace of the host integrations that drive it.

## Decision

### 1. Two distributions from this repository

- `crosstl` keeps the translator: the public functions in `crosstl/__init__.py`, `crosstl/translator/`,
  `crosstl/backend/`, the six translation subcommands, project scanning, translation, report validation and
  inspection, and the reflection metadata that every artifact carries. Reflection is part of the translation
  contract (core completion criterion 5), so it stays.
- A new distribution, `crosstl-host` (import package `crosstl_host`), takes the host tooling: the 23 host
  subcommands, host planning, runtime packaging, loader ABIs and scaffolds, runtime drivers and native workers.
  It depends on `crosstl` and reads only its public functions and its metadata. `crosstl[host]` is an extra that
  installs it.
- Both are built from this repository, so a translator fix and the host change it needs still land in one pull
  request and are tested together by the merge queue.

### 2. Deprecate, do not remove

- **Command line.** `crosstl <host subcommand> ...` keeps working. It prints one deprecation line to standard error
  naming `crosstl-host`, then runs the same command from `crosstl_host` with the same arguments, output and exit
  status. Without `crosstl-host` installed, it exits with status 2 and says which package to install. The
  subcommands stay in `crosstl --help` under a "deprecated" heading.
- **Python.** `crosstl.project` keeps re-exporting the moved names through a module-level `__getattr__` that issues
  a `DeprecationWarning` naming the new module.
- Removal is a separate, later decision, taken no earlier than two minor releases after `crosstl-host` is first
  published. This record does not schedule it.

### 3. Stages

Each stage is one pull request, leaves every suite green, and changes no output:

1. **Boundary check** (with this record). `tests/conformance/test_package_boundary.py` fails when a module under
   `crosstl/translator/` or `crosstl/backend/` imports `crosstl.project`, the command-line module or the
   `crosstl` package root, directly, relatively or through `importlib`.
2. **Split `pipeline.py`** into translation modules and host modules, and extend the boundary check so that the
   translation modules of `crosstl/project/` do not import the host modules.
3. **Move the 23 host subcommands** out of `crosstl/_crosstl.py` into one command-line module of the host side,
   behind the same parser. Tests pin each subcommand's help text and exit status before and after.
4. **Create `crosstl-host`** in `host/` with its own `pyproject.toml`; move the host modules and workers; add the
   forwarding subcommands and the deprecated re-exports; and move the host CI jobs to a workflow of its own.
5. **Migrate the in-repository users** to `crosstl_host`. Demos under `demos/integrations/` are migrated by their
   owners, who are notified through issues; the deprecated re-exports keep their suites green until then.

Publishing `crosstl-host` to a package index is a maintainer action and is outside these stages.

## Alternatives considered

- **Keep one package.** Simplest, and no user changes anything. Rejected: the translator's install, review surface
  and CI keep carrying runtime drivers and platform workers that translation never uses.
- **One distribution with an optional extra only.** `crosstl[host]` would add the runtime dependencies but install
  the same code. Rejected: it separates dependencies but not code, tests or ownership.
- **A separate repository.** Gives the strongest separation. Rejected: translator fixes that the host tooling needs
  would span two repositories and two merge queues, the corpus suites under `demos/integrations/` exercise both
  halves together, and the agents working on this repository do not create repositories or change their settings.

## Consequences

- The `crosstl` package keeps the translation API and the six translation subcommands exactly as they are.
- Users of the host subcommands and of `crosstl.project`'s host functions see deprecation notices and keep working
  output until removal is decided.
- The boundary check makes the translator's independence from host tooling a tested property, so stages 2 to 5 can
  land in any order after it without the translator acquiring a host dependency.
- Stage 2 is required in any case: `pipeline.py` is the largest module in the package, and the split also serves
  the size limits of criterion 6.
