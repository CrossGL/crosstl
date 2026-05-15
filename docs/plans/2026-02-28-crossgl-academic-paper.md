# CrossGL Academic Paper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write a complete academic paper about CrossGL's hub-and-spoke compiler infrastructure for the conference submission, using the Springer `svproc` LaTeX template.

**Architecture:** The paper will be written as a single LaTeX file (`paper/template/template.tex`) replacing the sample content. It uses the `svproc` document class with `url`, `graphicx`, `float`, `listings`, and `xcolor` packages. The paper follows a systems/compiler paper structure: motivation, language design, architecture, code generation, evaluation, and discussion. All references use `\thebibliography` (inline, no `.bib` file). Double-blind: no author info in initial submission.

**Tech Stack:** LaTeX (Springer svproc class), pdflatex for compilation

---

### Task 1: Set Up Document Skeleton and Packages

**Files:**
- Modify: `paper/template/template.tex`

**Step 1: Replace the entire template content with the paper skeleton**

Replace the full contents of `paper/template/template.tex` with the document skeleton that includes:
- `svproc` document class
- Required packages: `url`, `graphicx`, `float`, `listings` (for code), `xcolor` (for syntax highlighting), `booktabs` (for tables), `amsmath`, `amssymb`
- `lstdefinelanguage` for CrossGL syntax highlighting
- Double-blind compliant title block (no author names, use `Anonymous Author(s)`)
- All section headings as empty stubs
- Empty `thebibliography` block

The skeleton should have these sections:
```latex
\title{CrossGL: A Hub-and-Spoke Compiler Infrastructure for Universal Shader Translation Across Heterogeneous GPU Architectures}
\begin{abstract} ... \end{abstract}
\section{Introduction}
\section{Background and Related Work}
  \subsection{GPU Programming Landscape}
  \subsection{Existing Translation Tools}
  \subsection{Intermediate Representations for Shaders}
\section{The CrossGL Language}
  \subsection{Language Design Goals}
  \subsection{Syntax and Type System}
  \subsection{Shader Stage Model}
\section{System Architecture}
  \subsection{Hub-and-Spoke Translation Model}
  \subsection{Dual-AST Design}
  \subsection{Translation Pipeline}
  \subsection{Plugin Architecture}
\section{Code Generation}
  \subsection{Type and Semantic Mapping}
  \subsection{Per-Backend Strategies}
  \subsection{SPIR-V Assembly Generation}
\section{Evaluation}
  \subsection{Translation Correctness}
  \subsection{Language Feature Coverage}
  \subsection{Case Study: Universal PBR Shader}
  \subsection{Complexity Analysis}
\section{Discussion}
  \subsection{Limitations}
  \subsection{Future Work}
\section{Conclusion}
```

**Step 2: Compile to verify skeleton builds**

Run: `cd paper/template && pdflatex template.tex`
Expected: Successful compilation with warnings about empty references, produces `template.pdf`

**Step 3: Commit**

```bash
git add paper/template/template.tex
git commit -m "paper: set up LaTeX skeleton with section structure"
```

---

### Task 2: Write the Abstract (150-250 words)

**Files:**
- Modify: `paper/template/template.tex` (the `\begin{abstract}...\end{abstract}` block)

**Step 1: Write the abstract**

The abstract must cover:
1. **Problem** (2-3 sentences): GPU ecosystem fragmentation forces developers to maintain separate shader codebases for each platform (GLSL, HLSL, Metal, SPIR-V, CUDA, HIP). Existing tools only handle subsets or require vendor lock-in. This creates O(N^2) translation complexity.
2. **Approach** (2-3 sentences): We present CrossGL, a hub-and-spoke compiler infrastructure centered on a purpose-built intermediate shader language. The architecture reduces N-to-N translation to O(N) by routing all translations through a universal AST derived from the CrossGL language.
3. **Results** (2-3 sentences): CrossGL supports bidirectional translation across 10 languages, covers 16 pipeline stages including modern ray tracing and mesh shaders, and passes 321+ test cases. The system achieves correct translation of complex real-world shaders including a full PBR rendering pipeline.
4. **Keywords**: shader translation, cross-platform GPU computing, intermediate representation, compiler infrastructure, heterogeneous architectures

Target: 200 words exactly. No references, no abbreviations without definition.

**Step 2: Compile and check**

Run: `cd paper/template && pdflatex template.tex`
Expected: PASS, abstract renders correctly

**Step 3: Commit**

```bash
git add paper/template/template.tex
git commit -m "paper: write abstract"
```

---

### Task 3: Write the Introduction (~2 pages)

**Files:**
- Modify: `paper/template/template.tex` (`\section{Introduction}`)

**Step 1: Write the introduction**

Structure (5-6 paragraphs):

**Paragraph 1 — The problem:** The modern GPU computing landscape is deeply fragmented. OpenGL uses GLSL, DirectX uses HLSL, Apple requires Metal Shading Language, Vulkan uses SPIR-V, and GPU compute adds CUDA (NVIDIA), HIP (AMD), and emerging languages like Mojo and Slang. Developers targeting multiple platforms must maintain separate shader codebases, leading to duplicated effort, inconsistent behavior, and portability barriers. Cite: Khronos SPIR-V spec [1], Microsoft HLSL docs [2], Apple Metal spec [3].

**Paragraph 2 — Why existing solutions fall short:** Tools like SPIRV-Cross [4], Google's Tint [5], Mozilla's Naga [6], and Microsoft's DXC [7] address subsets of the problem but none provide a universal solution. SPIRV-Cross only converts FROM SPIR-V. Tint is WebGPU-focused. Naga handles a limited set. DXC is HLSL-centric. Each requires pairwise converters, giving O(N^2) complexity for N languages.

**Paragraph 3 — Our approach:** We introduce CrossGL, a compiler infrastructure that uses a hub-and-spoke model with a purpose-built intermediate language at the center. Any source language is first translated to CrossGL, then from CrossGL to any target — requiring only 2N converters instead of N(N-1).

**Paragraph 4 — Key design decisions:** The CrossGL language was designed to be a superset of shader concepts: it supports 16 pipeline stages (including ray tracing and mesh shaders), a rich type system with generics, and Rust-inspired pattern matching. The dual-AST architecture separates backend-specific representations from the universal translation AST.

**Paragraph 5 — Contributions (bulleted list):**
- The CrossGL intermediate shader language supporting 16 pipeline stages and modern GPU features
- A hub-and-spoke translation architecture reducing O(N^2) to O(N) cross-platform shader translation
- Bidirectional translation across 10 languages: GLSL, HLSL, Metal, SPIR-V, CUDA, HIP, Rust, Mojo, Slang, and CrossGL itself
- An extensible plugin architecture enabling new backend addition without core modifications
- An open-source implementation with 321+ passing tests, published on PyPI

**Paragraph 6 — Paper organization:** Brief roadmap of remaining sections.

References needed: [1]-[7] as listed above.

**Step 2: Compile and verify**

Run: `cd paper/template && pdflatex template.tex`
Expected: PASS, ~2 pages of introduction

**Step 3: Commit**

```bash
git add paper/template/template.tex
git commit -m "paper: write introduction section"
```

---

### Task 4: Write Background and Related Work (~2 pages)

**Files:**
- Modify: `paper/template/template.tex` (`\section{Background and Related Work}`)

**Step 1: Write the three subsections**

**§2.1 GPU Programming Landscape** (~0.5 pages):
- Brief taxonomy of GPU shading languages: GLSL (OpenGL/Vulkan), HLSL (DirectX), Metal (Apple), SPIR-V (Vulkan binary), CUDA/HIP (compute), Slang (Shader Slang research language), Mojo (Modular), Rust GPU (rust-gpu)
- Table 1: Comparison of languages — columns: Language, Platform, Domain (graphics/compute/both), Pipeline Stages, Type System features
- Cite each language spec: [1]-[3], [8]-[13]

**§2.2 Existing Translation Tools** (~1 page):
- **SPIRV-Cross** [4]: Converts SPIR-V to GLSL/HLSL/Metal. Limitation: unidirectional, SPIR-V-centric, no CUDA/HIP/Mojo/Rust support
- **Google Tint** [5]: WGSL compiler for WebGPU ecosystem. Limitation: WebGPU-focused, not general purpose
- **Mozilla Naga** [6]: Shader translation in wgpu. Limitation: Limited language set (WGSL, GLSL, SPIR-V, Metal)
- **Microsoft DXC** [7]: HLSL compiler to DXIL/SPIR-V. Limitation: HLSL-centric, DirectX ecosystem
- **Shader Slang** [14]: Research language from NVIDIA. Offers high-level abstractions but is a new language, not a translator
- Table 2: Feature comparison matrix — rows: tools, columns: supported source languages, target languages, bidirectional?, pipeline stages, extensible?
- Key gap: No tool provides universal N-to-N translation with an open intermediate language

**§2.3 Intermediate Representations for Shaders** (~0.5 pages):
- SPIR-V as a binary IR [1] — designed for Vulkan, not human-readable, not bidirectional
- LLVM IR [15] — too low-level for shader semantics (pipeline stages, semantics lost)
- MLIR [16] — promising but no shader dialect yet
- CrossGL fills the gap: a human-readable, high-level IR preserving shader semantics
- Cite compiler textbook for IR theory [17]

**Step 2: Compile and verify**

Run: `cd paper/template && pdflatex template.tex`
Expected: PASS, tables render, references resolve

**Step 3: Commit**

```bash
git add paper/template/template.tex
git commit -m "paper: write background and related work section"
```

---

### Task 5: Write The CrossGL Language Section (~2 pages)

**Files:**
- Modify: `paper/template/template.tex` (`\section{The CrossGL Language}`)

**Step 1: Write the three subsections**

**§3.1 Language Design Goals** (~0.5 pages):
- Universal expressiveness: must capture concepts from ALL target languages
- Semantic preservation: pipeline stages, vertex attributes, shader semantics must survive translation
- Human readability: unlike SPIR-V, CrossGL is designed for developers to read and write
- Extensibility: new features (ray tracing, mesh shaders) should be addable without breaking existing shaders
- Minimalism: YAGNI — only include constructs that map to real GPU operations

**§3.2 Syntax and Type System** (~1 page):
- Code listing: `SimpleShader.cgl` as Figure 1 (Listing 1) — shows the `shader` block, struct definitions, vertex/fragment stages
- Type system description with table:
  - Primitive types: `i8`-`i64`, `u8`-`u64`, `f16`-`f64`, `bool`
  - Vector types: `vec2`-`vec4`, `ivec2`-`ivec4`, `uvec2`-`uvec4`, `bvec2`-`bvec4`
  - Matrix types: `mat2`-`mat4`, `mat2x3`, etc.
  - Texture/sampler types: `sampler2D`, `samplerCube`, etc.
  - Generic types: `generic<T: Numeric>`
  - Arrays, pointers, references
- Operators and control flow: standard C-like plus `match` (Rust-inspired), `for-in`, `let`/`let mut`
- Attribute system: `@semantic` annotations for shader I/O semantics

**§3.3 Shader Stage Model** (~0.5 pages):
- Table 3: All 16 supported pipeline stages organized by category:
  - Traditional: vertex, fragment, geometry, tessellation_control, tessellation_evaluation, compute
  - Modern: mesh, task/amplification, object
  - Ray Tracing: ray_generation, ray_intersection, ray_closest_hit, ray_miss, ray_any_hit, ray_callable
- Execution models: graphics, compute, ray_tracing, general_purpose
- Code listing: compute shader example showing `layout()` and workgroup syntax

**Step 2: Compile and verify**

Run: `cd paper/template && pdflatex template.tex`
Expected: PASS, code listings render with syntax highlighting

**Step 3: Commit**

```bash
git add paper/template/template.tex
git commit -m "paper: write CrossGL language section"
```

---

### Task 6: Write System Architecture Section (~3 pages)

**Files:**
- Modify: `paper/template/template.tex` (`\section{System Architecture}`)

**Step 1: Write the four subsections**

**§4.1 Hub-and-Spoke Translation Model** (~1 page):
- Figure 2: Hub-and-spoke diagram showing CrossGL at center with spokes to all 10 languages. Label forward (CGL->target) and reverse (source->CGL) directions.
- Complexity argument: N languages require N forward + N reverse = 2N converters vs N(N-1) pairwise
- Three translation modes:
  1. Forward: `.cgl` -> Lexer -> Parser -> Universal AST -> CodeGen -> target
  2. Reverse: `.hlsl`/`.metal`/etc. -> Backend Lexer -> Backend Parser -> Backend AST -> Reverse CodeGen -> `.cgl`
  3. Cross: Source -> Reverse -> CGL -> Forward -> Target (two-hop)
- Code snippet from `_crosstl.py` showing the dispatch logic

**§4.2 Dual-AST Design** (~0.75 pages):
- Two distinct AST layers and why:
  1. **Universal AST** (`translator/ast.py`): 60+ node types, rich type system nodes, GPU-specific nodes (TextureOp, AtomicOp, WaveOp, RayTracingOp, MeshOp), visitor pattern, annotations
  2. **Backend ASTs** (`backend/common_ast.py` + per-backend extensions): simpler, native-language-oriented, used by reverse parsers
- Why two layers: backend parsers need to faithfully represent source semantics in native terms; the universal AST provides the normalized form for code generation
- Figure 3: Class hierarchy diagram of key AST node categories

**§4.3 Translation Pipeline** (~0.75 pages):
- Figure 4: Pipeline flow diagram for forward translation:
  `Source text -> Lexer (regex-based, 220+ token types) -> Token stream -> Recursive descent parser (12-level precedence climbing) -> Universal AST (ShaderNode) -> Backend CodeGen -> Target text`
- Lexer: combined regex pattern, keyword post-lookup, token caching, error reporting with line/column
- Parser: hand-written recursive descent (not parser-generator), 2233 lines, speculative parsing for ambiguous constructs, operator precedence climbing (12 levels from assignment to primary)
- Both registries: `SourceRegistry` (extension-based dispatch) and `BackendRegistry` (name/alias dispatch)

**§4.4 Plugin Architecture** (~0.5 pages):
- Dynamic backend discovery: scans `backend/*/` for `backend_spec.py` and `source_spec.py`
- Lazy loading: backend modules imported only when needed
- `BackendSpec` dataclass: name, aliases, extensions, codegen factory
- How to add a new backend: drop files into `backend/NewLang/`, implement lexer/parser/AST/codegen, register via spec file

**Step 2: Compile and verify**

Run: `cd paper/template && pdflatex template.tex`
Expected: PASS, figures referenced correctly

**Step 3: Commit**

```bash
git add paper/template/template.tex
git commit -m "paper: write system architecture section"
```

---

### Task 7: Write Code Generation Section (~2 pages)

**Files:**
- Modify: `paper/template/template.tex` (`\section{Code Generation}`)

**Step 1: Write the three subsections**

**§5.1 Type and Semantic Mapping** (~0.75 pages):
- Table 4: Type mapping across backends (CrossGL -> GLSL, HLSL, Metal, SPIR-V, CUDA, Rust, Mojo)
  - Rows: vec2, vec3, vec4, mat4, sampler2D, int, uint, float, bool
- Table 5: Semantic mapping (CrossGL -> GLSL, HLSL, Metal)
  - Rows: gl_Position, gl_FragColor, gl_VertexID, gl_GlobalInvocationID
- Table 6: Function mapping
  - Rows: lerp, frac, ddx, rsqrt, tex2D
- The `ASTUtils` centralized mapper for consistent type resolution

**§5.2 Per-Backend Strategies** (~0.75 pages):
- Common pattern: `generate()` -> iterate AST -> `generate_function()` -> `generate_statement()` -> `generate_expression()` with `map_type()`, `map_semantic()`, `map_operator()`
- Backend-specific challenges:
  - GLSL: `layout(location=N)` qualifiers, `#version` directives, `void main()` entry points
  - HLSL: Semantic annotations (`SV_POSITION`), constant buffers, register bindings
  - Metal: Argument buffers, `[[attribute(N)]]`, `device`/`constant` address spaces, `kernel` functions
  - CUDA/HIP: `__global__`, `__device__`, `__shared__` qualifiers, thread indexing (`threadIdx`, `blockIdx`)
  - Rust: Ownership model mapping, `()` return type, `i32`/`f32` primitives
  - Mojo: `fn` syntax, `Float32`/`Int32` types, `None` return type

**§5.3 SPIR-V Assembly Generation** (~0.5 pages):
- Generates textual SPIR-V assembly (not binary) — 1603 lines of specialized codegen
- ID allocation system with bound tracking
- Type registry: primitive, vector, matrix, struct, pointer, function, array types all tracked by ID
- Structured control flow via `OpSelectionMerge`/`OpLoopMerge`
- GLSL.std.450 extended instruction set integration
- Optional `spirv-tools` integration for validation and formatting

**Step 2: Compile and verify**

Run: `cd paper/template && pdflatex template.tex`
Expected: PASS, tables render correctly

**Step 3: Commit**

```bash
git add paper/template/template.tex
git commit -m "paper: write code generation section"
```

---

### Task 8: Write Evaluation Section (~3 pages)

**Files:**
- Modify: `paper/template/template.tex` (`\section{Evaluation}`)

**Step 1: Write the four subsections**

**§6.1 Translation Correctness** (~0.75 pages):
- Test suite: 321+ passing tests across 54 test files
- Test categories: lexer tests, parser tests, codegen tests (forward), reverse codegen tests (backward)
- CI matrix: Python 3.8-3.13 x Ubuntu/Windows/macOS
- Table 7: Test pass rates per backend
- Methodology: each test provides CrossGL input, translates to target, and validates output structure/syntax

**§6.2 Language Feature Coverage** (~0.75 pages):
- Table 8: Feature coverage matrix — rows: language features (structs, functions, control flow, textures, compute, ray tracing, generics, pattern matching); columns: backends
- Discussion of coverage gaps and why (e.g., Rust doesn't have native shader stages, Mojo lacks ray tracing intrinsics)

**§6.3 Case Study: Universal PBR Shader** (~1 page):
- Real-world shader: `UniversalPBRShader.cgl` — 489 lines, Cook-Torrance BRDF, cascaded shadow mapping, IBL, parallax mapping, tone mapping
- Successfully translated to all 9 target backends
- Code listing: Fragment of PBR shader in CrossGL and corresponding HLSL/Metal output (side by side or sequential)
- Lines of code comparison table (Table 9): CrossGL source vs generated output per backend
- Qualitative analysis: readability, correctness, idiomatic output

**§6.4 Complexity Analysis** (~0.5 pages):
- Converter complexity: 2N = 20 converters for 10 languages, vs N(N-1) = 90 pairwise
- Lines of code: total codebase ~20K lines Python
- Per-backend effort: average ~500-1000 lines for lexer+parser+AST+codegen per new backend
- Adding an 11th language: 2 new converters vs 20 new pairwise converters

**Step 2: Compile and verify**

Run: `cd paper/template && pdflatex template.tex`
Expected: PASS, all tables and listings render

**Step 3: Commit**

```bash
git add paper/template/template.tex
git commit -m "paper: write evaluation section"
```

---

### Task 9: Write Discussion and Conclusion (~2 pages)

**Files:**
- Modify: `paper/template/template.tex` (`\section{Discussion}` and `\section{Conclusion}`)

**Step 1: Write discussion subsections**

**§7.1 Limitations** (~0.75 pages):
- No optimization passes: translation is direct AST-to-text, no dead code elimination, constant folding, or loop optimization
- Semantic gaps: some language-specific features don't map cleanly (e.g., Rust ownership semantics, CUDA warp-level primitives beyond basic wave ops)
- No binary SPIR-V output (textual assembly only — requires external `spirv-as`)
- Two-hop translation may introduce artifacts compared to direct pairwise translation
- Performance: no runtime benchmarks of generated shaders (correctness focus only)

**§7.2 Future Work** (~0.75 pages):
- Optimization passes over the universal AST (dead code elimination, constant propagation)
- Binary SPIR-V emission and DXIL support
- ML kernel specialization: extending CrossGL for tensor operations, automatic differentiation constructs
- Formal verification of translation equivalence
- IDE integration: language server protocol (LSP) for CrossGL
- WebGPU/WGSL backend addition
- Community-contributed backends via the plugin architecture

**Step 2: Write conclusion** (~0.5 pages):
- Restate the problem and solution
- Summarize key contributions (mirror the introduction's bullet points)
- Emphasize the open-source availability (PyPI, Zenodo DOI)
- Forward-looking statement about heterogeneous computing future

**Step 3: Compile and verify**

Run: `cd paper/template && pdflatex template.tex`
Expected: PASS

**Step 4: Commit**

```bash
git add paper/template/template.tex
git commit -m "paper: write discussion and conclusion sections"
```

---

### Task 10: Write References (20+ entries)

**Files:**
- Modify: `paper/template/template.tex` (`\begin{thebibliography}...\end{thebibliography}`)

**Step 1: Write all bibliography entries**

Minimum 20 references. Each entry must have: authors, title, venue/publisher, year, DOI where available.

Required references:
1. Khronos Group: SPIR-V Specification, v1.6. (2022)
2. Microsoft: HLSL Programming Guide. Microsoft Docs (2023)
3. Apple: Metal Shading Language Specification, v3.1. Apple Inc. (2023)
4. Kronos Group: SPIRV-Cross. GitHub repository (2023)
5. Google: Tint — WebGPU Shader Compiler. (2023)
6. Mozilla: Naga — Shader Translation Infrastructure. (2023)
7. Microsoft: DirectX Shader Compiler (DXC). GitHub repository (2023)
8. NVIDIA: CUDA Programming Guide, v12.0. (2023)
9. AMD: HIP Programming Guide. (2023)
10. Modular: Mojo Programming Language. (2023)
11. Embark Studios: rust-gpu — Making Rust a first-class language for GPU shaders. (2023)
12. Yong He et al.: Slang: Language Mechanisms for Extensible Real-Time Shading Systems. ACM TOG 37(4) (2018)
13. Kessenich, J., Sellers, G., Shreiner, D.: OpenGL Programming Guide, 9th ed. Addison-Wesley (2016)
14. Aho, A.V., Lam, M.S., Sethi, R., Ullman, J.D.: Compilers: Principles, Techniques, and Tools, 2nd ed. Pearson (2006)
15. Lattner, C., Adve, V.: LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation. CGO (2004)
16. Lattner, C. et al.: MLIR: Scaling Compiler Infrastructure for Domain Specific Computation. CGO (2021)
17. Mark, W.R., Glanville, R.S., Akeley, K., Kilgard, M.J.: Cg: A System for Programming Graphics Hardware in a C-like Language. ACM TOG 22(3) (2003)
18. Tatarchuk, N. et al.: Real-time hair rendering on the GPU. ACM SIGGRAPH Courses (2009)
19. Cook, R.L., Torrance, K.E.: A Reflectance Model for Computer Graphics. ACM TOG 1(1) (1982)
20. Pharr, M., Jakob, W., Humphreys, G.: Physically Based Rendering, 4th ed. MIT Press (2023)
21. McCool, M., Reinders, J., Robison, A.: Structured Parallel Programming. Morgan Kaufmann (2012)
22. Nickolls, J., Buck, I., Garland, M., Skadron, K.: Scalable Parallel Programming with CUDA. ACM Queue 6(2) (2008)
23. Sellers, G., Kessenich, J.: Vulkan Programming Guide. Addison-Wesley (2017)

**Step 2: Verify all references are cited in text**

Scan through all sections to ensure every `\cite{...}` has a corresponding `\bibitem{...}` and every `\bibitem` is cited at least once.

**Step 3: Compile and verify**

Run: `cd paper/template && pdflatex template.tex`
Expected: PASS, no undefined reference warnings

**Step 4: Commit**

```bash
git add paper/template/template.tex
git commit -m "paper: add all references"
```

---

### Task 11: Add Figures and Tables, Final Polish

**Files:**
- Modify: `paper/template/template.tex` (throughout)

**Step 1: Ensure all figures are described**

Since this is a LaTeX-only submission, complex diagrams should use ASCII/textual descriptions in `figure` environments or be described as TikZ diagrams if possible. Key figures needed:
- Figure 1: CrossGL code listing (SimpleShader.cgl) — use `lstlisting`
- Figure 2: Hub-and-spoke architecture diagram — use a simple textual/TikZ representation
- Figure 3: Translation pipeline flow — use textual description in figure

All tables (Tables 1-9) should already be in place from previous tasks.

**Step 2: Final polish pass**

- Ensure consistent terminology throughout (CrossGL, not CrossTL/crosstl in running text)
- Check all cross-references (`\ref`, `\cite`) resolve
- Verify abstract is 150-250 words
- Verify total is under 18 pages of main text
- Remove any placeholder text
- Ensure double-blind compliance (no author names, no "our university", no self-identifying URLs)

**Step 3: Final compile**

Run: `cd paper/template && pdflatex template.tex && pdflatex template.tex` (twice for references)
Expected: PASS, no warnings except possibly font substitution

**Step 4: Commit**

```bash
git add paper/template/template.tex
git commit -m "paper: final polish, figures, and formatting"
```

---

## Reference Information for All Tasks

**Key codebase files to reference for content accuracy:**
- `crosstl/translator/ast.py` — Universal AST (60+ node types)
- `crosstl/translator/lexer.py` — CrossGL lexer (220+ token types)
- `crosstl/translator/parser.py` — Recursive descent parser (2233 lines)
- `crosstl/translator/codegen/*.py` — All 9 code generators
- `crosstl/translator/codegen/ast_utils.py` — Type mapping utilities
- `crosstl/translator/codegen/registry.py` — Backend registry
- `crosstl/translator/source_registry.py` — Source language registry
- `crosstl/translator/plugin_loader.py` — Plugin discovery
- `crosstl/backend/common_ast.py` — Backend common AST (527 lines)
- `crosstl/backend/*/` — All 9 backend implementations
- `crosstl/_crosstl.py` — Main entry point and translation dispatch
- `crosstl/formatter.py` — External tool integration
- `examples/graphics/SimpleShader.cgl` — Simple shader example
- `examples/cross_platform/UniversalPBRShader.cgl` — Complex PBR shader
- `examples/advanced/GenericPatternMatching.cgl` — Generics example
- `tests/` — 54 test files, 321+ passing tests
