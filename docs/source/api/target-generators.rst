Target Generators
=================

The target generator classes expose a common public entry point:
``generate(ast_node)`` accepts a CrossGL AST and returns source text or assembly
for the selected backend.

GLSL
----

.. autoclass:: crosstl.translator.codegen.GLSL_codegen.GLSLCodeGen
   :members: generate

WebGL / GLSL ES
---------------

.. autoclass:: crosstl.translator.codegen.webgl_codegen.WebGLCodeGen
   :members: generate

WebGPU / WGSL
-------------

.. autoclass:: crosstl.translator.codegen.wgsl_codegen.WGSLCodeGen
   :members: generate

DirectX / HLSL
--------------

The ``directx`` generator emits HLSL source. Target profile aliases such as
``dx11``, ``dx12``, ``d3d11``, and ``d3d12`` resolve to this generator so
project configs can state their intended Direct3D deployment profile while
still producing portable ``.hlsl`` source.

.. autoclass:: crosstl.translator.codegen.directx_codegen.HLSLCodeGen
   :members: generate

Metal
-----

.. autoclass:: crosstl.translator.codegen.metal_codegen.MetalCodeGen
   :members: generate

Vulkan SPIR-V
-------------

.. autoclass:: crosstl.translator.codegen.SPIRV_codegen.VulkanSPIRVCodeGen
   :members: generate

CUDA
----

.. autoclass:: crosstl.translator.codegen.cuda_codegen.CudaCodeGen
   :members: generate

HIP
---

.. autoclass:: crosstl.translator.codegen.hip_codegen.HipCodeGen
   :members: generate

Mojo
----

.. autoclass:: crosstl.translator.codegen.mojo_codegen.MojoCodeGen
   :members: generate

Rust
----

.. autoclass:: crosstl.translator.codegen.rust_codegen.RustCodeGen
   :members: generate

Slang
-----

.. autoclass:: crosstl.translator.codegen.slang_codegen.SlangCodeGen
   :members: generate
