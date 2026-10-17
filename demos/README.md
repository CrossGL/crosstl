# Translation Demos

CrossTL translates shader and compute programs independently of any particular
application. These demos exercise that translation against real source projects
and document the host changes needed to execute generated artifacts.

| Demo | Scope |
| --- | --- |
| [Open-source porting](open-source-porting/README.md) | Pinned shader corpora and target-toolchain checks |
| [MLX integration](integrations/mlx/README.md) | Repository discovery, kernel translation and native host integration |
| [Shader showcase](readme/showcase_pbr.cgl) | CrossGL graphics example used in the repository README |

Project-specific adapters, fixtures and tests live beside their demo. Reusable
parsers, code generators, package contracts and runtime APIs belong in `crosstl/`,
with independent regression tests in `tests/`. Demo packages are not included in
the installed translator distribution.

The complete local test command includes both sets:

```sh
python -m pytest -q -n auto
```

Native demo jobs retain generated sources, compiler diagnostics and execution
evidence. See each demo's README for its pinned revisions, verified scope and
remaining limitations; successful translation alone does not establish runtime
parity.
