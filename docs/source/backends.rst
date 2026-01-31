Supported Backends
==================

CrossTL supports translation to and from multiple shader languages.

DirectX (HLSL)
--------------

The DirectX backend generates HLSL (High-Level Shading Language) code
compatible with DirectX 11 and 12.

.. automodule:: crosstl.translator.codegen.directx_codegen
   :members:
   :undoc-members:
   :show-inheritance:

Metal
-----

The Metal backend generates Apple Metal Shading Language code for macOS,
iOS, and tvOS platforms.

.. automodule:: crosstl.translator.codegen.metal_codegen
   :members:
   :undoc-members:
   :show-inheritance:

OpenGL (GLSL)
-------------

The OpenGL backend generates GLSL (OpenGL Shading Language) code compatible
with OpenGL 3.3+ and OpenGL ES 3.0+.

.. automodule:: crosstl.translator.codegen.GLSL_codegen
   :members:
   :undoc-members:
   :show-inheritance:

Vulkan (SPIR-V)
---------------

The Vulkan backend generates SPIR-V assembly code for Vulkan applications.

.. automodule:: crosstl.translator.codegen.SPIRV_codegen
   :members:
   :undoc-members:
   :show-inheritance:

CUDA
----

The CUDA backend generates NVIDIA CUDA code for GPU computing.

.. automodule:: crosstl.translator.codegen.cuda_codegen
   :members:
   :undoc-members:
   :show-inheritance:

HIP
---

The HIP backend generates AMD HIP code for GPU computing.

.. automodule:: crosstl.translator.codegen.hip_codegen
   :members:
   :undoc-members:
   :show-inheritance:

Mojo
----

The Mojo backend generates Mojo code for high-performance computing.

.. automodule:: crosstl.translator.codegen.mojo_codegen
   :members:
   :undoc-members:
   :show-inheritance:

Rust
----

The Rust backend generates Rust GPU code.

.. automodule:: crosstl.translator.codegen.rust_codegen
   :members:
   :undoc-members:
   :show-inheritance:

Slang
-----

The Slang backend generates Slang shading language code.

.. automodule:: crosstl.translator.codegen.slang_codegen
   :members:
   :undoc-members:
   :show-inheritance:
