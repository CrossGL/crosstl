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
     - Output extension
   * - DirectX / HLSL
     - ``directx``
     - ``hlsl``, ``dx``
     - ``.hlsl``
   * - OpenGL / GLSL
     - ``opengl``
     - ``glsl``, ``ogl``
     - ``.glsl``
   * - Metal
     - ``metal``
     - ``metal``
     - ``.metal``
   * - Vulkan SPIR-V
     - ``vulkan``
     - ``spirv``, ``spv``
     - ``.spirv``
   * - CUDA
     - ``cuda``
     - ``cu``
     - ``.cu``
   * - HIP
     - ``hip``
     - ``hip``
     - ``.hip``
   * - Mojo
     - ``mojo``
     - ``mojo``
     - ``.mojo``
   * - Rust
     - ``rust``
     - ``rust``, ``rs``
     - ``.rs``
   * - Slang
     - ``slang``
     - ``slang``
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
     - ``.spv``, ``.spirv``
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

Runtime discovery
-----------------

The public helpers ``crosstl.supported_backends()`` and
``crosstl.supported_sources()`` return the names registered in the current
process. Plugin modules under ``crosstl.backend`` can add more source or target
entries at runtime.
