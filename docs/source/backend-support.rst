Backend Support
===============

CrossGL Translator keeps source imports and target code generators in separate
registries. A language can be available as an input source, an output target,
or both.

Built-in targets
----------------

.. list-table::
   :header-rows: 1

   * - Target
     - Backend name
     - Aliases
     - Target aliases
     - Target profiles
     - Output extension
   * - DirectX / HLSL
     - ``directx``
     - ``hlsl``, ``dx``
     - ``dx11``, ``dx12``, ``d3d11``, ``d3d12``
     - ``directx-11``, ``directx-12``
     - ``.hlsl``
   * - OpenGL / GLSL
     - ``opengl``
     - ``glsl``, ``ogl``
     - n/a
     - n/a
     - ``.glsl``
   * - WebGL / GLSL ES
     - ``webgl``
     - ``webgl2``, ``essl``, ``glsl-es``
     - n/a
     - n/a
     - ``.webgl.glsl``
   * - WebGPU / WGSL
     - ``wgsl``
     - ``webgpu``
     - n/a
     - n/a
     - ``.wgsl``
   * - Metal
     - ``metal``
     - ``metal``, ``msl``
     - n/a
     - n/a
     - ``.metal``
   * - Vulkan SPIR-V
     - ``vulkan``
     - ``spirv``, ``spv``
     - n/a
     - n/a
     - ``.spvasm``
   * - CUDA
     - ``cuda``
     - ``cu``
     - n/a
     - n/a
     - ``.cu``
   * - HIP
     - ``hip``
     - ``hip``
     - n/a
     - n/a
     - ``.hip``
   * - Mojo
     - ``mojo``
     - ``mojo``
     - n/a
     - n/a
     - ``.mojo``
   * - Rust
     - ``rust``
     - ``rust``, ``rs``
     - n/a
     - n/a
     - ``.rs``
   * - Slang
     - ``slang``
     - ``slang``
     - n/a
     - n/a
     - ``.slang``

Built-in sources
----------------

.. list-table::
   :header-rows: 1

   * - Source
     - Registry name
     - Extensions
     - Reverse CrossGL generation
   * - CrossGL
     - ``cgl``
     - ``.cgl``
     - Native source format
   * - DirectX / HLSL
     - ``directx``
     - ``.hlsl``
     - Yes
   * - OpenGL / GLSL
     - ``opengl``
     - ``.glsl``
     - Yes
   * - Metal
     - ``metal``
     - ``.metal``
     - Yes
   * - Vulkan SPIR-V
     - ``vulkan``
     - ``.spvasm``
     - Yes
   * - CUDA
     - ``cuda``
     - ``.cu``, ``.cuh``, ``.cuda``
     - Yes
   * - HIP
     - ``hip``
     - ``.hip``
     - Yes
   * - Mojo
     - ``mojo``
     - ``.mojo``
     - Yes
   * - Rust
     - ``rust``
     - ``.rs``, ``.rust``
     - Yes
   * - Slang
     - ``slang``
     - ``.slang``
     - Yes

Target-only sources
-------------------

Some targets are intentionally output-only while their native source frontends
are still under development.

.. list-table::
   :header-rows: 1

   * - Target
     - Output extension
     - Source input behavior
   * - WebGL / GLSL ES
     - ``.webgl.glsl``
     - Rejected as target-only output; use ``.glsl`` or ``.cgl`` as input.
   * - WebGPU / WGSL
     - ``.wgsl``
     - Rejected until a dedicated WGSL/WESL source frontend lands.

Runtime discovery
-----------------

The public helpers ``crosstl.supported_backends()`` and
``crosstl.supported_sources()`` return the names registered in the current
process. Plugin modules under ``crosstl.backend`` can add more source or target
entries at runtime.
