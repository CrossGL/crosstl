# Changelog

All notable changes to CrossTL are documented in this file.

## [Unreleased]

### Added

- Target-only WebGL/WebGL2 GLSL ES backend support.
- Target-only WebGPU/WGSL backend support for vertex, fragment, and compute output.
- DirectX target profile aliases for `dx11`, `dx12`, `d3d11`, and `d3d12`, all resolving to HLSL source output.

### Improved

- Split target backend registration from native source frontend availability in the backend registry and support matrix.
- Added target alias and target profile metadata to backend support inventory.
- Documented target-only source rejection for WebGL and WGSL outputs.

## [2.0.0] - 2026-06-01

### Breaking Changes

- **CrossGL language expanded**: New constructs (generics, traits, algebraic enums, pattern matching, do-while, double-precision types) mean `.cgl` files written for 2.0 may not parse under 1.x.
- **AST restructured**: `ASTNode` gained `child_nodes()`, `walk()`, and `bind_parent_links()` methods. `ShaderNode.stages` is now a `StageMap` wrapper instead of a raw `dict`.
- **Internal imports changed**: Direct imports like `from crosstl.translator.codegen import GLSL_codegen` are no longer supported. Use the registry (`crosstl.translator.codegen.registry`) or the public `crosstl.translate()` API.
- **Codegen `generate()` contract changed**: Backend code generators now expect the expanded AST structure with new node types.

### Added

- **Validation layer** (`crosstl/translator/validation.py`): 2700+ lines of texture intrinsic, stage, and resource validation.
- **Language specification extractor** (`crosstl/translator/language_spec.py`): Machine-readable snapshot of the CrossGL language surface.
- **Plugin/registry system**: Backends self-register via `BackendSpec`/`SourceSpec`, enabling third-party extensions.
- **Shared codegen utilities**: `enum_utils`, `match_utils`, `generic_function_utils`, `generic_struct_utils`, `vector_arithmetic`, `image_access_contracts`, `resource_arrays`, `resource_query`, `stage_utils`.
- **New language features**: Generics (`generic<T>`), traits, algebraic enums with data, pattern matching (`match`/`case`), guard clauses, do-while loops, double-precision types (`dvec2-4`, `dmat2-4`).
- **Stage coverage**: Full support for compute, geometry, tessellation, mesh/task, and ray-tracing stages across applicable backends.
- **Support matrix tooling**: Automated feature tracking (`support/features.json`), CI-driven matrix validation, and issue synchronization.
- **Documentation**: Sphinx-based docs with per-backend API reference, architecture guide, and support matrix.

### Improved

- All 9 backends (GLSL, HLSL, Metal, SPIR-V, CUDA, HIP, Mojo, Rust, Slang) massively expanded — 10-20x more generated code surface handling new language features.
- CI expanded to cross-platform (Linux/Windows/macOS), multi-Python (3.8-3.13), per-backend test matrices with JUnit XML output.
- Test suite grew to 7,983 passing tests (from ~200 at 1.0.0).

### Fixed

- Fragment output name handling in GLSL codegen.
- Vector constructor declarations in CUDA/HIP parsers.
- Switch-default ordering preservation.
- Texture comparison function translation across backends.

### Removed

- `build/` directory (stale artifact).
- `.agent-coordination/` directory (obsolete automation scaffolding).
- `experimental/` backend directory.
- `pytest` removed from runtime `install_requires` (it was incorrectly listed as a runtime dependency).

---

## [1.1.0] - 2025-12-15

### Added

- Backend registry and source registry for plugin-style backend loading.
- Slang backend (lexer, parser, codegen, preprocessor).
- Mojo and Rust backends.

### Improved

- Refactored shader translation pipeline.
- Backend integration consolidated.

---

## [1.0.0] - 2025-10-01

Initial release with:
- CrossGL intermediate representation language.
- Translation targets: Metal, DirectX (HLSL), OpenGL (GLSL), Vulkan (SPIR-V), CUDA, HIP.
- Bidirectional translation (native → CrossGL and CrossGL → native).
- Basic vertex, fragment, and compute shader stage support.
