# ADR 0005: CrossGL storage buffers without a declared layout use std430

- Status: accepted
- Date: 2026-10-27
- Scope: `crosstl/translator/semantics/default_buffer_layouts.py`,
  `crosstl/translator/semantics/structured_layouts.py`, `crosstl/translator/codegen/metal_hlsl_resources.py`,
  `crosstl/translator/codegen/metal_hlsl_buffer_layout.py`, and every importer that marks `@buffer_layout`

## Context

A storage buffer's element layout is part of a program's interface: the host writes the bytes and the kernel reads
them. The languages lay out the same element type differently:

| Element | std430 (GLSL, SPIR-V, WGSL) | HLSL structured buffer | MSL `device` buffer |
|---|---|---|---|
| `struct { vec3 a; float b; }` | `b` at 12, 16 bytes | `b` at 12, 16 bytes | `b` at 16, 32 bytes |
| `struct { float a; vec3 b; }` | `b` at 16, 32 bytes | `b` at 4, 16 bytes | `b` at 16, 32 bytes |
| `vec3 v[]` | 16-byte stride | 12-byte stride | 16-byte stride |

References: GLSL 4.60 §7.6.2.2, D3D11.3 functional specification §5.3.7, MSL 3.2 §2.2 and table 2.4.

The importers already mark a buffer whose source layout is not std430 (`@buffer_layout(hlsl_structured)` from HLSL,
CUDA and Metal packed vectors, `@buffer_layout(metal)` from Metal), and the generators reproduce a marked layout.
A buffer written in CrossGL itself, or imported from GLSL, carried no mark, and each generator declared the element
type unchanged, so the Metal output of `struct { vec3 a; float b; }` read `b` four bytes later than every other
target and the host.

## Decision

A CrossGL storage buffer (`buffer T*`, `StructuredBuffer<T>`, `RWStructuredBuffer<T>`, a bound runtime array) that
carries no `@buffer_layout` attribute holds its elements at std430 offsets and strides. `@buffer_layout(std430)`
states the same thing explicitly.

`mark_default_buffer_layouts` runs in rule carrying, before `legalize_structured_layouts`. For each target whose
own buffer rules differ from std430 it adds `@buffer_layout(std430)` to every unmarked buffer whose element type
that target lays out differently, and the target's existing machinery for marked layouts reproduces it or raises a
typed diagnostic. The Metal generator declares such elements with packed vectors and explicit padding at their
std430 offsets, as it does for HLSL structured buffers.

Booleans are excluded: each target stores a `bool` in its own physical size, which the project package reports to
the host, rather than in std430's 32 bits.

## Consequences

- The GLSL, SPIR-V and WGSL output is unchanged; Metal output changes only for element types whose MSL layout
  differs from std430, which were wrong before.
- HLSL structured buffers align vectors to their scalar, so for the HLSL targets the marker also adds
  `@buffer_layout(std430)` where the HLSL layout differs (`struct { float a; vec3 b; }`, `vec3 v[]`, `mat3`), and
  `legalize_structured_layouts` stores such buffers as 32-bit words read and written at their std430 offsets. A
  stage's own buffers are lowered too, and their helper functions belong to that stage.
- An importer that produces a buffer with a non-std430 layout must mark it; an unmarked buffer is a std430 claim.
