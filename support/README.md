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
- `tools/support_signals.py` generates transient extraction reports from backend
  docs, relevant same-site docs links, implementation files, and tests. CI uses
  those signals to find stale support claims, discover documented API candidates
  missing from the catalog, and enrich managed GitHub issues without
  hand-written per-backend evidence files.
- The generated JSON also records test counts and stable sampled unsupported
  markers from implementation paths for each backend.
- Project-porting support rows track repository scan, batch translation,
  project configuration, artifact reports, provenance, validation hooks, and
  structured diagnostics. The implementation emits scan and portability reports
  through `python -m crosstl._crosstl scan`, `report`, and `translate-project`,
  validates reports through `validate-project`, and summarizes reports through
  `inspect-report`.

Status values are conservative. Do not mark a feature `supported` unless there is
implementation and test evidence. Use `partial`, `diagnostic`, `unsupported`,
`validated_rejection`, or `unknown` when the behavior is incomplete, deliberately
rejected, or not audited yet.

Generated backlog rows and managed GitHub sub-issues are reserved for actionable
coverage gaps: `partial`, `unsupported`, and `unknown`. Evidence-backed
`diagnostic` and `validated_rejection` rows stay visible in the matrix counts,
but they are treated as closed-loop behavior rather than open implementation
work.

Run:

```sh
python tools/support_matrix.py update
python tools/support_matrix.py check
python tools/support_matrix.py audit
python tools/support_matrix.py audit --backend directx --backend opengl --backend metal
python tools/support_matrix.py audit --backend directx --status partial --output /tmp/directx-partial.json
python tools/support_signals.py docs --output support/generated/backend-docs-report.json
python tools/support_signals.py extract --docs-report support/generated/backend-docs-report.json --output support/generated/support-signals.json
python -m crosstl._crosstl scan /path/to/repo --target metal --output /tmp/crosstl-project-scan.json
```

The hourly issue-sync CI fetches official backend documentation URLs, crawls a
bounded set of relevant same-site docs links, extracts API-like candidates, and
cross-checks them against implementation/test identifiers. The checked-in matrix
keeps reviewed support statuses conservative; generated support-signal issues
track missing or uncataloged candidates so coverage gaps do not depend on manual
lookup.

Process rules:

- Treat `unknown` as a work item, not as implicit support.
- Prefer evidence strings that point to a focused test or validation branch.
- Do not edit generated files directly; update the catalog and rerun the tool.
- Documentation probes are advisory. A changed upstream document should start a
  reviewed catalog update, not an automatic status change.
