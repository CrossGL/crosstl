# CI Coverage

The CI suite separates portable Python compatibility from native GPU toolchain
and runtime checks.

| Workload | Ubuntu | Windows | macOS |
| --- | --- | --- | --- |
| Backend, translator and example tests | Python 3.8 through 3.13 | Python 3.13 | Python 3.13 |
| Complete core and demo unit suite | Required | Covered by focused platform jobs | Covered by focused platform jobs |
| Complete DirectX demo corpus compilation | Pinned Linux DXC | Native execution remains below | Not required |
| Direct3D execution and host integration | Not applicable | Required WARP and dispatch checks | Not applicable |
| OpenGL execution and host integration | Required Mesa/EGL checks | Platform-specific controls | Platform-specific controls |
| Metal compilation, reference and host integration | Not available | Not available | Required native checks |

Every backend and code generator remains in the unit matrices on all three
operating systems. Python-version compatibility is exercised fully on Ubuntu;
repeating those versions on Windows and macOS is not required. This removes 198
duplicate platform/version jobs per revision without removing a backend suite.

The demo's 77 unary, binary, copy and reduction DirectX compilation shards run
on Ubuntu. They retain every entry, strict compiler flags, artifact identity
checks and the same DXC release. Pinned installation verifies the Linux archive
checksum and does not fall back to another release. These jobs do not execute
Direct3D; WARP numerical execution and Windows host integration remain required.
Metal corpus compilation stays on macOS because it requires Apple's toolchain.

`demo-project-testing.yml` is the entry point for project integration checks.
Each native job keeps its source pin, numerical comparisons, guards, bounded execution
and retained evidence. Moving a compiler-only job must not disable an execution
gate or be presented as proof of runtime parity.

Project demo checks run for changes to source, tests, demo inputs, support
contracts and build/toolchain configuration. Root documentation-only changes do
not launch the native demo matrices. Scheduled corpus audits remain enabled.
The Metal storage step has a 30-minute bound within an 80-minute job; the earlier
15-minute bound expired during passing cases as the required suite grew.

The full local command is `python -m pytest -q -n auto`. The all-file pre-commit
checks validate workflow contracts and generated support artifacts. Changes to
matrix coverage must update those contracts and this policy together.
