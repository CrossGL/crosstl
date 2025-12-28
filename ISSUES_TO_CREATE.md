# CrossTL Missing Features - Issues to Create

This document lists all missing features and improvements needed across all backends and the frontend (CrossGL).

## Summary

CrossTL supports 9 backends: DirectX, Metal, CUDA, HIP, GLSL, SPIRV, Rust, Mojo, and Slang.

**Current Status:**
- ✅ All backends have Parser and Lexer implementations
- ✅ All backends have Backend→CrossGL code generation
- ✅ All backends have CrossGL→Backend code generation  
- ✅ 508 tests passing
- ⚠️ Some features are incomplete across certain backends
- ⚠️ Test coverage varies significantly across backends

---

## Backend-Specific Missing Features

### 1. Metal Backend

**Missing Control Flow:**
- [ ] **Issue:** Add `while` loop support for Metal backend
  - **Description:** Metal backend parser and codegen need to support `while` loops
  - **Impact:** Prevents translation of code using while loops to/from Metal
  - **Components affected:** MetalParser.py, MetalCrossGLCodeGen.py, metal_codegen.py
  - **Priority:** Medium
  - **Labels:** enhancement, metal, control-flow

**Missing Features:**
- [ ] **Issue:** Add `cbuffer` support for Metal backend codegen
  - **Description:** Metal backend needs complete constant buffer (cbuffer) support in both directions
  - **Impact:** Limits ability to translate constant buffer declarations
  - **Components affected:** MetalCrossGLCodeGen.py, metal_codegen.py
  - **Priority:** High
  - **Labels:** enhancement, metal, buffers

- [ ] **Issue:** Add `const` declaration support for Metal backend
  - **Description:** Add support for const variable declarations in Metal
  - **Impact:** Missing const variables in generated Metal code
  - **Components affected:** MetalParser.py, MetalCrossGLCodeGen.py
  - **Priority:** Low
  - **Labels:** enhancement, metal

**Test Coverage:**
- [ ] **Issue:** Increase Metal backend test coverage
  - **Description:** Metal has 28 backend tests and 16 frontend tests. Add more comprehensive tests
  - **Suggested additions:**
    - Complex shader tests
    - Edge case handling
    - Error condition tests
  - **Priority:** Medium
  - **Labels:** testing, metal

---

### 2. CUDA Backend

**Test Coverage (Critical):**
- [ ] **Issue:** Add comprehensive CUDA backend tests
  - **Description:** CUDA backend currently has 0 backend tests (no test_parser.py, test_lexer.py, or test_codegen.py). Need to create comprehensive test suite
  - **Components needed:**
    - tests/test_backend/test_CUDA/test_parser.py
    - tests/test_backend/test_CUDA/test_lexer.py  
    - tests/test_backend/test_CUDA/test_codegen.py
  - **Priority:** High
  - **Labels:** testing, cuda, critical

- [ ] **Issue:** Add comprehensive CUDA frontend tests
  - **Description:** CUDA has 0 frontend codegen tests. Add tests to test_translator/test_codegen/test_CUDA_codegen.py
  - **Priority:** High
  - **Labels:** testing, cuda, frontend

**Missing Features:**
- [ ] **Issue:** Complete `switch` statement support for CUDA backend
  - **Description:** CUDA parser has partial switch support, needs full implementation in codegen
  - **Impact:** Switch statements may not translate correctly
  - **Components affected:** CudaParser.py, CudaCrossGLCodeGen.py
  - **Priority:** Medium
  - **Labels:** enhancement, cuda, control-flow

- [ ] **Issue:** Add `const` declaration support for CUDA backend
  - **Description:** Add const variable declaration support
  - **Components affected:** CudaParser.py, CudaCrossGLCodeGen.py
  - **Priority:** Low
  - **Labels:** enhancement, cuda

---

### 3. HIP Backend

**Test Coverage (Critical):**
- [ ] **Issue:** Add comprehensive HIP backend tests
  - **Description:** HIP backend currently has 0 backend tests. Need to create comprehensive test suite
  - **Components needed:**
    - tests/test_backend/test_HIP/test_parser.py
    - tests/test_backend/test_HIP/test_lexer.py
    - tests/test_backend/test_HIP/test_codegen.py
  - **Priority:** High
  - **Labels:** testing, hip, critical

**Missing Features:**
- [ ] **Issue:** Add `switch` statement support for HIP backend
  - **Description:** HIP backend needs switch/case statement support
  - **Impact:** Cannot translate switch statements to/from HIP
  - **Components affected:** HipParser.py, HipCrossGLCodeGen.py, hip_codegen.py
  - **Priority:** Medium
  - **Labels:** enhancement, hip, control-flow

- [ ] **Issue:** Add `const` declaration support for HIP backend
  - **Description:** Add const variable declaration support for HIP
  - **Components affected:** HipParser.py, HipCrossGLCodeGen.py
  - **Priority:** Low
  - **Labels:** enhancement, hip

- [ ] **Issue:** Complete `enum` support for HIP backend
  - **Description:** HIP has partial enum support, needs full implementation in codegen
  - **Components affected:** HipParser.py, HipCrossGLCodeGen.py, hip_codegen.py
  - **Priority:** Medium
  - **Labels:** enhancement, hip

- [ ] **Issue:** Complete `cbuffer` support for HIP backend
  - **Description:** HIP needs full constant buffer support
  - **Components affected:** HipCrossGLCodeGen.py, hip_codegen.py
  - **Priority:** Medium
  - **Labels:** enhancement, hip, buffers

---

### 4. GLSL Backend

**Test Coverage:**
- [ ] **Issue:** Add comprehensive GLSL backend tests  
  - **Description:** GLSL backend test files exist but may need expansion
  - **Current:** Has test_parser.py, test_lexer.py, test_codegen.py
  - **Priority:** Medium
  - **Labels:** testing, glsl

- [ ] **Issue:** Add comprehensive GLSL frontend tests
  - **Description:** Add tests to test_translator/test_codegen/test_GLSL_codegen.py
  - **Priority:** High
  - **Labels:** testing, glsl, frontend

**Missing Features:**
- [ ] **Issue:** Complete `cbuffer` support for GLSL backend
  - **Description:** GLSL needs full constant buffer/uniform buffer support
  - **Components affected:** OpenglParser.py, openglCrossglCodegen.py, GLSL_codegen.py
  - **Priority:** High
  - **Labels:** enhancement, glsl, buffers

- [ ] **Issue:** Add `const` declaration support for GLSL backend
  - **Description:** Complete const variable support
  - **Components affected:** OpenglParser.py, openglCrossglCodegen.py
  - **Priority:** Low
  - **Labels:** enhancement, glsl

---

### 5. SPIRV Backend

**Test Coverage (Critical):**
- [ ] **Issue:** Add comprehensive SPIRV backend tests
  - **Description:** SPIRV backend currently has 0 backend tests despite having a complete implementation
  - **Components needed:**
    - tests/test_backend/test_SPIRV/test_parser.py
    - tests/test_backend/test_SPIRV/test_lexer.py
    - tests/test_backend/test_SPIRV/test_codegen.py
  - **Priority:** High
  - **Labels:** testing, spirv, critical

- [ ] **Issue:** Add comprehensive SPIRV frontend tests
  - **Description:** Add tests to test_translator/test_codegen/test_SPIRV_codegen.py
  - **Priority:** High
  - **Labels:** testing, spirv, frontend

**Missing Features:**
- [ ] **Issue:** Add `cbuffer` support for SPIRV backend
  - **Description:** SPIRV needs constant buffer support
  - **Components affected:** VulkanParser.py, VulkanCrossGLCodeGen.py, SPIRV_codegen.py
  - **Priority:** High
  - **Labels:** enhancement, spirv, buffers

- [ ] **Issue:** Add `const` declaration support for SPIRV backend
  - **Description:** Add const variable declaration support
  - **Components affected:** VulkanParser.py, VulkanCrossGLCodeGen.py
  - **Priority:** Low
  - **Labels:** enhancement, spirv

---

### 6. Rust Backend

**Missing Features:**
- [ ] **Issue:** Add `switch` statement support for Rust backend
  - **Description:** Rust backend should translate switch to match expressions
  - **Impact:** Cannot translate switch statements to Rust match
  - **Components affected:** RustParser.py, RustCrossGLCodeGen.py, rust_codegen.py
  - **Priority:** Medium
  - **Labels:** enhancement, rust, control-flow

- [ ] **Issue:** Complete `cbuffer` support for Rust backend
  - **Description:** Rust backend needs full constant buffer translation
  - **Components affected:** RustCrossGLCodeGen.py, rust_codegen.py
  - **Priority:** Medium
  - **Labels:** enhancement, rust, buffers

**Test Coverage:**
- [ ] **Issue:** Expand Rust backend test coverage
  - **Description:** Rust has good coverage (76 backend, 22 frontend) but could add more edge cases
  - **Priority:** Low
  - **Labels:** testing, rust

---

### 7. Mojo Backend

**Missing Features:**
- [ ] **Issue:** Complete `cbuffer` support for Mojo backend
  - **Description:** Mojo backend needs full constant buffer support
  - **Components affected:** MojoCrossGLCodeGen.py, mojo_codegen.py
  - **Priority:** Medium
  - **Labels:** enhancement, mojo, buffers

**Test Coverage:**
- [ ] **Issue:** Expand Mojo backend test coverage
  - **Description:** Mojo has good coverage (61 backend, 19 frontend) but could benefit from more tests
  - **Priority:** Low
  - **Labels:** testing, mojo

---

### 8. Slang Backend

**Test Coverage:**
- [ ] **Issue:** Add comprehensive Slang frontend tests
  - **Description:** Slang only has 2 frontend tests, needs significant expansion
  - **Priority:** High
  - **Labels:** testing, slang, frontend, critical

**Missing Features:**
- [ ] **Issue:** Add `while` loop support for Slang backend
  - **Description:** Slang backend needs while loop support
  - **Components affected:** SlangParser.py, SlangCrossGLCodeGen.py, slang_codegen.py
  - **Priority:** Medium
  - **Labels:** enhancement, slang, control-flow

- [ ] **Issue:** Add `switch` statement support for Slang backend
  - **Description:** Slang backend needs switch/case statement support
  - **Components affected:** SlangParser.py, SlangCrossGLCodeGen.py, slang_codegen.py
  - **Priority:** Medium
  - **Labels:** enhancement, slang, control-flow

- [ ] **Issue:** Add `const` declaration support for Slang backend
  - **Description:** Add const variable declaration support
  - **Components affected:** SlangParser.py, SlangCrossGLCodeGen.py
  - **Priority:** Low
  - **Labels:** enhancement, slang

---

### 9. DirectX Backend

**Missing Features:**
- [ ] **Issue:** Add `const` declaration support for DirectX backend
  - **Description:** DirectX backend needs const variable declaration support
  - **Components affected:** DirectxParser.py, DirectxCrossGLCodeGen.py
  - **Priority:** Low
  - **Labels:** enhancement, directx

**Status:**
- DirectX has the most complete implementation with 52 backend tests and 15 frontend tests
- Good baseline for other backends to follow

---

## Cross-Backend Missing Features

### Advanced Language Features

These features are missing or incomplete across ALL backends:

- [ ] **Issue:** Add `enum` support across all backends
  - **Description:** Only HIP has partial enum support. Need comprehensive enum support for all backends
  - **Backends affected:** DirectX, Metal, CUDA, GLSL, SPIRV, Rust, Mojo, Slang
  - **Priority:** High
  - **Labels:** enhancement, cross-backend, language-feature

- [ ] **Issue:** Add `template`/`generics` support across all backends
  - **Description:** No backend currently supports templates or generics
  - **Impact:** Cannot translate generic/template code
  - **Backends affected:** All
  - **Priority:** High
  - **Labels:** enhancement, cross-backend, language-feature

- [ ] **Issue:** Add `class` support for relevant backends
  - **Description:** Only HIP and Mojo have partial class support. Need comprehensive class support where applicable
  - **Backends affected:** CUDA, HIP, Mojo, Rust (structs with impl), Slang
  - **Priority:** Medium
  - **Labels:** enhancement, cross-backend, language-feature

- [ ] **Issue:** Add `typedef`/type alias support across backends
  - **Description:** Type aliases are not consistently supported
  - **Backends affected:** All
  - **Priority:** Medium
  - **Labels:** enhancement, cross-backend, language-feature

- [ ] **Issue:** Add `namespace`/`module` support across backends
  - **Description:** Namespace/module support is incomplete
  - **Backends affected:** Metal, CUDA, HIP, Rust, Mojo, Slang
  - **Priority:** Medium
  - **Labels:** enhancement, cross-backend, language-feature

---

## Frontend (CrossGL) Missing Features

The CrossGL language and translator need these improvements:

- [ ] **Issue:** Add comprehensive `enum` support to CrossGL frontend
  - **Description:** CrossGL parser and AST need enum support
  - **Components affected:** translator/parser.py, translator/ast.py, translator/lexer.py
  - **Priority:** High
  - **Labels:** enhancement, frontend, language-feature

- [ ] **Issue:** Add `template`/`generics` support to CrossGL frontend
  - **Description:** Add generic/template support to CrossGL language
  - **Components affected:** translator/parser.py, translator/ast.py, translator/lexer.py
  - **Priority:** High
  - **Labels:** enhancement, frontend, language-feature

- [ ] **Issue:** Add `class` support to CrossGL frontend
  - **Description:** Add class/object support to CrossGL
  - **Components affected:** translator/parser.py, translator/ast.py, translator/lexer.py
  - **Priority:** Medium
  - **Labels:** enhancement, frontend, language-feature

- [ ] **Issue:** Add `namespace`/`module` support to CrossGL frontend
  - **Description:** Add namespace/module system to CrossGL
  - **Components affected:** translator/parser.py, translator/ast.py, translator/lexer.py
  - **Priority:** Medium
  - **Labels:** enhancement, frontend, language-feature

- [ ] **Issue:** Add `typedef`/type alias support to CrossGL frontend
  - **Description:** Add type alias support to CrossGL
  - **Components affected:** translator/parser.py, translator/ast.py, translator/lexer.py
  - **Priority:** Medium
  - **Labels:** enhancement, frontend, language-feature

- [ ] **Issue:** Add `do-while` loop support to CrossGL frontend
  - **Description:** CrossGL should support do-while loops
  - **Components affected:** translator/parser.py, translator/ast.py
  - **Priority:** Low
  - **Labels:** enhancement, frontend, control-flow

- [ ] **Issue:** Enhance `const` support in CrossGL frontend
  - **Description:** Improve const variable declaration support
  - **Components affected:** translator/parser.py, translator/ast.py
  - **Priority:** Low
  - **Labels:** enhancement, frontend

---

## Testing Infrastructure

- [ ] **Issue:** Create comprehensive test suite template
  - **Description:** Create a standard template for backend tests that can be adapted for each backend
  - **Benefits:** Ensures consistent test coverage across all backends
  - **Priority:** High
  - **Labels:** testing, infrastructure

- [ ] **Issue:** Add integration tests for cross-backend translation
  - **Description:** Test translating from one backend to another (e.g., HLSL → CrossGL → Metal)
  - **Priority:** Medium
  - **Labels:** testing, integration

- [ ] **Issue:** Add performance benchmarks
  - **Description:** Create benchmarks to measure translation performance and correctness
  - **Priority:** Low
  - **Labels:** testing, performance

---

## Documentation

- [ ] **Issue:** Document language feature support matrix
  - **Description:** Create and maintain a feature support matrix showing which features are supported by each backend
  - **Location:** docs/feature-matrix.md
  - **Priority:** High
  - **Labels:** documentation

- [ ] **Issue:** Create backend-specific documentation
  - **Description:** Document backend-specific quirks, limitations, and best practices
  - **Priority:** Medium
  - **Labels:** documentation

- [ ] **Issue:** Add migration guides
  - **Description:** Create guides for migrating from one backend to another
  - **Priority:** Low
  - **Labels:** documentation

---

## Priority Summary

### Critical (Must Do)
1. Add comprehensive tests for CUDA backend (0 tests currently)
2. Add comprehensive tests for HIP backend (0 tests currently)  
3. Add comprehensive tests for SPIRV backend (0 tests currently)
4. Add comprehensive frontend tests for GLSL (0 tests currently)
5. Add comprehensive frontend tests for Slang (only 2 tests)

### High Priority
6. Add enum support across all backends and frontend
7. Add template/generics support across all backends and frontend
8. Complete cbuffer support for all backends
9. Create comprehensive test suite template
10. Document feature support matrix

### Medium Priority
11. Add switch statement support where missing
12. Add while loop support where missing
13. Add class support for relevant backends
14. Increase test coverage for all backends

### Low Priority
15. Add const declaration support across backends
16. Add typedef/type alias support
17. Documentation improvements

---

## Total Issues to Create: 60+

This represents a comprehensive roadmap for making CrossTL feature-complete across all backends.
