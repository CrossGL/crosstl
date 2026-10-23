Backend Support
===============

CrossGL Translator keeps source imports and target code generators in separate
registries. A language can be available as an input source, an output target,
or both. The tables below list what is registered; they do not say how much of
each language translates correctly.

Measured coverage
-----------------

CI measures that on every push to ``main`` and nightly. Each feature in a
pinned upstream feature list gets a generated probe, CrossTL translates it, and
the language's reference compiler or validator checks the result. Metal, GLSL and
SPIR-V compute probes also run, on the GPU or on Mesa's software drivers, where
the translation must compute what the native probe computes.
`COVERAGE.md <https://github.com/CrossGL/crosstl/blob/coverage/COVERAGE.md>`_
lists the score and every gap for each language and direction, and each row
with gaps has an open issue labelled ``coverage``.

.. image:: https://raw.githubusercontent.com/CrossGL/crosstl/coverage/coverage.svg
   :alt: CrossTL coverage per language and direction

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

Every source below except CrossGL has a reverse generator that emits CrossGL,
so its programs can be translated through CrossGL to the targets above.

.. list-table::
   :header-rows: 1

   * - Source
     - Registry name
     - Extensions
   * - CrossGL
     - ``cgl``
     - ``.cgl``
   * - DirectX / HLSL
     - ``directx``
     - ``.hlsl``, ``.hlsli``, ``.fx``, ``.fxh``
   * - OpenGL / GLSL
     - ``opengl``
     - ``.glsl`` and stage extensions such as ``.vert``, ``.frag``,
       ``.comp``, ``.geom``, ``.tesc``, ``.tese``, ``.mesh``, ``.task`` and
       ``.rgen``
   * - Metal
     - ``metal``
     - ``.metal``, ``.msl``
   * - Vulkan SPIR-V
     - ``vulkan``
     - ``.spvasm``
   * - CUDA
     - ``cuda``
     - ``.cu``, ``.cuh``, ``.cuda``
   * - HIP
     - ``hip``
     - ``.hip``
   * - OpenCL C
     - ``opencl``
     - ``.cl``, ``.opencl``
   * - Mojo
     - ``mojo``
     - ``.mojo``
   * - Rust
     - ``rust``
     - ``.rs``, ``.rust``
   * - Slang
     - ``slang``
     - ``.slang``, ``.slangh``

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
