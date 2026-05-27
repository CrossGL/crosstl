# CrossGL support matrix

This directory is the source of truth for backend capability tracking.

The matrix is intentionally data-driven:

- `backends.json` lists every built-in backend, the repo files that implement it,
  test locations, and official documentation sources.
- `features.json` lists the capabilities CrossGL wants to support and the current
  status per backend.
- `generated/support-matrix.json` and `docs/source/support-matrix.rst` are generated
  by `tools/support_matrix.py`.
- `generated/graphics-backend-roadmap.json` is a focused generated view for the
  DirectX, OpenGL, and Metal roadmap.
- The generated JSON also records test counts and stable sampled unsupported
  markers from implementation paths for each backend.

Status values are conservative. Do not mark a feature `supported` unless there is
implementation and test evidence. Use `partial`, `diagnostic`, `unsupported`,
`validated_rejection`, or `unknown` when the behavior is incomplete, deliberately
rejected, or not audited yet.

Run:

```sh
python tools/support_matrix.py update
python tools/support_matrix.py check
python tools/support_matrix.py audit
python tools/support_matrix.py audit --backend directx --backend opengl --backend metal
python tools/support_matrix.py audit --backend directx --status partial --output /tmp/directx-partial.json
```

The nightly CI job can also fetch official backend documentation URLs and report
hash/status changes without making the generated matrix depend on fragile live
scraping.

Process rules:

- Treat `unknown` as a work item, not as implicit support.
- Prefer evidence strings that point to a focused test or validation branch.
- Do not edit generated files directly; update the catalog and rerun the tool.
- Documentation probes are advisory. A changed upstream document should start a
  reviewed catalog update, not an automatic status change.
