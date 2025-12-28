# CrossTL Feature Support Matrix

This document provides a comprehensive overview of language feature support across all CrossTL backends.

**Last Updated:** December 2024

---

## Backend Overview

CrossTL supports 9 backends with bidirectional translation:

| Backend | Parser | Lexer | Backend→CrossGL | CrossGL→Backend | Backend Tests | Frontend Tests |
|---------|--------|-------|-----------------|-----------------|---------------|----------------|
| **DirectX** | ✅ | ✅ | ✅ | ✅ | 52 | 15 |
| **Metal** | ✅ | ✅ | ✅ | ✅ | 28 | 16 |
| **CUDA** | ✅ | ✅ | ✅ | ✅ | 0 ⚠️ | 0 ⚠️ |
| **HIP** | ✅ | ✅ | ✅ | ✅ | 0 ⚠️ | 16 |
| **GLSL** | ✅ | ✅ | ✅ | ✅ | 31 | 0 ⚠️ |
| **SPIRV** | ✅ | ✅ | ✅ | ✅ | 0 ⚠️ | 0 ⚠️ |
| **Rust** | ✅ | ✅ | ✅ | ✅ | 76 | 22 |
| **Mojo** | ✅ | ✅ | ✅ | ✅ | 61 | 19 |
| **Slang** | ✅ | ✅ | ✅ | ✅ | 18 | 2 ⚠️ |

**Total Tests:** 508 (all passing ✅)

---

## Core Language Features

### Data Structures

| Feature | DirectX | Metal | CUDA | HIP | GLSL | SPIRV | Rust | Mojo | Slang |
|---------|---------|-------|------|-----|------|-------|------|------|-------|
| **Struct** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Enum** | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Class** | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | 🟡 | ❌ |
| **Union** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Array** | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |

**Legend:**
- ✅✅✅ = Full support (Parser + Backend→CGL + CGL→Backend)
- 🟡 = Partial support
- ❌ = Not supported

---

### Control Flow

| Feature | DirectX | Metal | CUDA | HIP | GLSL | SPIRV | Rust | Mojo | Slang |
|---------|---------|-------|------|-----|------|-------|------|------|-------|
| **if/else** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **for loop** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **while loop** | ✅✅ | ❌ | ✅✅ | ✅✅ | 🟡 | ✅✅ | ✅✅ | ✅✅ | ❌ |
| **do-while** | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **switch/case** | ✅✅ | ✅✅ | 🟡 | ❌ | ✅✅ | ✅✅ | ❌ | ✅✅ | ❌ |
| **break** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **continue** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **return** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |

---

### Functions

| Feature | DirectX | Metal | CUDA | HIP | GLSL | SPIRV | Rust | Mojo | Slang |
|---------|---------|-------|------|-----|------|-------|------|------|-------|
| **Function Declaration** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Function Call** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Recursion** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Template/Generics** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Overloading** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

---

### Operators

| Feature | DirectX | Metal | CUDA | HIP | GLSL | SPIRV | Rust | Mojo | Slang |
|---------|---------|-------|------|-----|------|-------|------|------|-------|
| **Arithmetic (+, -, *, /)** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Modulo (%)** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Comparison (<, >, ==, !=)** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Logical (&&, \|\|, !)** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Bitwise (&, \|, ^, ~)** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Shift (<<, >>)** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Assignment (=)** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Compound (+=, -=, etc.)** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Ternary (? :)** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Member Access (.)** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Array Access ([])** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |

---

### Variables & Constants

| Feature | DirectX | Metal | CUDA | HIP | GLSL | SPIRV | Rust | Mojo | Slang |
|---------|---------|-------|------|-----|------|-------|------|------|-------|
| **Variable Declaration** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **const** | ❌ | 🟡 | 🟡 | ❌ | 🟡 | ❌ | 🟡 | 🟡 | ❌ |
| **static** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ |
| **let (Rust/Mojo)** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅✅ | ✅✅ | ❌ |
| **var (Mojo)** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅✅ | ❌ |

---

### Shader-Specific Features

| Feature | DirectX | Metal | CUDA | HIP | GLSL | SPIRV | Rust | Mojo | Slang |
|---------|---------|-------|------|-----|------|-------|------|------|-------|
| **cbuffer/Uniform Buffer** | ✅✅✅ | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | 🟡 | 🟡 | ✅✅✅ |
| **Uniform Variables** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Texture Sampling** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Vertex Shader** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Fragment/Pixel Shader** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Compute Shader** | ✅ | ✅ | ✅✅✅ | ✅✅✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Geometry Shader** | 🟡 | ❌ | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | 🟡 |
| **Tessellation Shader** | 🟡 | ❌ | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ |

---

### Type System

| Feature | DirectX | Metal | CUDA | HIP | GLSL | SPIRV | Rust | Mojo | Slang |
|---------|---------|-------|------|-----|------|-------|------|------|-------|
| **Basic Types (int, float)** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Vector Types (vec2-4)** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Matrix Types** | ✅✅ | ✅✅ | ✅ | ✅ | ✅✅ | ✅✅ | ✅ | ✅ | ✅✅ |
| **bool** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **double** | ✅✅✅ | ✅✅ | ✅✅✅ | ✅✅✅ | ✅✅ | ✅✅ | ✅✅✅ | ✅✅✅ | ✅✅ |
| **half/float16** | ✅✅ | ✅✅✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅✅ |
| **uint** | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **Typedef/Type Alias** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Type Casting** | ✅✅ | ✅✅ | ✅✅ | ✅✅ | ✅✅ | ✅✅ | ✅✅✅ | ✅✅ | ✅✅ |

---

### Advanced Features

| Feature | DirectX | Metal | CUDA | HIP | GLSL | SPIRV | Rust | Mojo | Slang |
|---------|---------|-------|------|-----|------|-------|------|------|-------|
| **Pointers** | ❌ | 🟡 | 🟡 | 🟡 | ❌ | ❌ | 🟡 | 🟡 | ❌ |
| **References** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ |
| **Namespace/Module** | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | 🟡 | ❌ |
| **Attributes/Annotations** | 🟡 | 🟡 | ❌ | ❌ | 🟡 | 🟡 | ✅✅ | ✅✅ | ❌ |
| **Macros/Preprocessor** | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ❌ | 🟡 |
| **Import/Include** | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
| **Comments** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Backend-Specific Notes

### DirectX (HLSL)
- **Strengths:** Most complete implementation, excellent test coverage
- **Limitations:** Missing const declarations, no enum support
- **Unique Features:** cbuffer support is excellent

### Metal
- **Strengths:** Good coverage for Apple ecosystem
- **Limitations:** No while loops, incomplete cbuffer support
- **Unique Features:** Excellent half-precision float support

### CUDA
- **Strengths:** Strong compute shader support
- **Limitations:** **Critical: No tests!** Incomplete switch support
- **Unique Features:** CUDA-specific kernel features

### HIP
- **Strengths:** AMD GPU compute support, partial enum support
- **Limitations:** **Critical: No backend tests!** Missing switch statements
- **Unique Features:** Similar to CUDA but for AMD

### GLSL (OpenGL)
- **Strengths:** Wide compatibility, comprehensive parser
- **Limitations:** **No frontend tests**, incomplete cbuffer support
- **Unique Features:** OpenGL-specific semantics

### SPIRV (Vulkan)
- **Strengths:** Modern graphics API support
- **Limitations:** **Critical: No tests!** Missing cbuffer support
- **Unique Features:** Descriptor sets, layout qualifiers

### Rust
- **Strengths:** Best test coverage (76+22), strong type system support
- **Limitations:** Missing switch (should use match), incomplete cbuffer
- **Unique Features:** Ownership/borrowing concepts, attributes

### Mojo
- **Strengths:** Excellent test coverage, modern language features
- **Limitations:** Incomplete cbuffer support
- **Unique Features:** Python compatibility, AI/ML optimizations

### Slang
- **Strengths:** Modern real-time graphics
- **Limitations:** **Critical: Only 2 frontend tests!** Missing while/switch
- **Unique Features:** Slang-specific shading language features

---

## Frontend (CrossGL) Language Support

The CrossGL intermediate language currently supports:

✅ **Fully Supported:**
- Structs
- Functions (declaration, call, return)
- Control flow (if/else, for loops)
- All operators (arithmetic, logical, bitwise, comparison)
- Variable declarations and assignments
- Arrays and array access
- Member access
- Vector and matrix types
- Basic type system
- Shader stages (vertex, fragment, compute)

🟡 **Partially Supported:**
- Constant buffers (cbuffer)
- While loops
- Switch statements
- Const declarations
- Attributes/Annotations

❌ **Not Supported:**
- Enums
- Classes/Objects
- Templates/Generics
- Namespaces/Modules
- Type aliases (typedef)
- Advanced pointer operations
- References
- Function overloading

---

## Test Coverage Analysis

### Backends with Excellent Coverage (50+ tests)
- Rust: 98 tests total
- Mojo: 80 tests total
- DirectX: 67 tests total

### Backends Needing More Tests
- **CUDA: 0 tests** ⚠️ CRITICAL
- **SPIRV: 0 tests** ⚠️ CRITICAL
- **GLSL: 31 backend tests, 0 frontend tests** ⚠️
- **Slang: 18 backend tests, 2 frontend tests** ⚠️

### Total Test Count: 508 tests (all passing ✅)

---

## Recommendations

### Immediate Priorities
1. ✅ Add comprehensive tests for CUDA backend
2. ✅ Add comprehensive tests for HIP backend  
3. ✅ Add comprehensive tests for SPIRV backend
4. ✅ Add frontend tests for GLSL
5. ✅ Add frontend tests for Slang

### Short-term Goals
6. ✅ Implement enum support across all backends
7. ✅ Complete cbuffer support for all backends
8. ✅ Add template/generics support to frontend and backends
9. ✅ Standardize while loop support
10. ✅ Standardize switch statement support

### Long-term Goals
11. ✅ Add class/object support
12. ✅ Implement namespace/module system
13. ✅ Add typedef/type alias support
14. ✅ Improve geometry and tessellation shader support
15. ✅ Add function overloading

---

## Contributing

When contributing new features:
1. Add support to CrossGL frontend first (parser, AST, lexer)
2. Implement backend→CrossGL translation
3. Implement CrossGL→backend translation
4. Add comprehensive tests (parser, lexer, codegen)
5. Update this feature matrix
6. Update documentation

---

**Note:** This matrix is based on analysis of the codebase as of December 2024. Features marked with 🟡 may have partial or incomplete implementations. For the most up-to-date information, please check the latest test results and source code.
