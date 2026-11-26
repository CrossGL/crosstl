Quickstart
==========

The commands on this page work straight after ``pip install crosstl`` in an
empty directory. CI runs every one of them from a clean install on Linux,
macOS and Windows with each supported Python version
(``tools/install_smoke.py``).

Install
-------

CrossTL needs Python 3.11 or later and nothing else:

.. code-block:: bash

   pip install crosstl
   crosstl doctor

Translating needs no other tool. Validating the generated code needs the
target's own compiler or validator, such as glslang, DXC, spirv-tools, naga or
the Metal compiler, which CrossTL does not install. ``crosstl doctor`` lists
the tools it looks for and which of them are on ``PATH``. To work on CrossTL
itself, install a checkout in editable mode with ``pip install -e .``
instead.

Translate a native shader
-------------------------

Save this Metal kernel as ``shader.metal``:

.. code-block:: cpp

   #include <metal_stdlib>
   using namespace metal;

   kernel void add(device float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
       a[i] = a[i] + b[i];
   }

Translate it from the command line. The source language comes from the input
file's extension and the target from the output file's:

.. code-block:: bash

   crosstl shader.metal -o shader.hlsl
   crosstl shader.metal -o shader.comp
   crosstl shader.metal -o shader.wgsl
   crosstl shader.metal --backend vulkan -o -

``--backend`` chooses the target explicitly, and ``-o -`` writes to standard
output. Without ``--backend``, ``-o -`` writes the CrossGL that the shader
imports to. An input or output file name whose extension names no language
CrossTL reads or writes, such as ``shader.spv``, is refused with the list of
extensions it accepts, and nothing is written.

The same translations from Python:

.. code-block:: python

   import crosstl

   hlsl = crosstl.translate("shader.metal", backend="directx")
   crosstl.translate("shader.metal", backend="wgsl", save_shader="kernel.wgsl")

   # The CrossGL program the Metal kernel imports to.
   crossgl = crosstl.translate("shader.metal", backend="cgl")

``crosstl.supported_sources()`` and ``crosstl.supported_backends()`` list the
languages CrossTL reads and writes; :doc:`backend-support` describes them, and
each language has a page under :doc:`languages/index` with its measured
coverage and every known gap.

Write CrossGL
-------------

Every translation goes through CrossGL, and one CrossGL source generates every
target. Save this compute kernel as ``saxpy.cgl``:

.. code-block:: glsl

   shader Saxpy {
       compute {
           layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
           buffer float* x;
           buffer float* y;
           uniform float a;

           void main() {
               uint i = gl_GlobalInvocationID.x;
               y[i] = a * x[i] + y[i];
           }
       }
   }

Then generate the targets you need:

.. code-block:: python

   import crosstl

   metal = crosstl.translate("saxpy.cgl", backend="metal")
   crosstl.translate("saxpy.cgl", backend="vulkan", save_shader="saxpy.spvasm")

.. code-block:: bash

   crosstl saxpy.cgl -o saxpy.metal
   python -m crosstl translate saxpy.cgl --backend directx --output -

Porting a repository
--------------------

To translate a whole repository, scan it first, then translate, validate and
inspect the project report:

.. code-block:: bash

   python -m crosstl scan /path/to/repo --target metal --output scan-report.json
   python -m crosstl translate-project /path/to/repo --target metal \
     --output-dir crosstl-out --report crosstl-out/portability-report.json
   python -m crosstl validate-project crosstl-out/portability-report.json --format text
   python -m crosstl inspect-report crosstl-out/portability-report.json --format text

Project reports cover shader and kernel source translation; host runtime API
migration and build-system changes remain manual work listed in the report.
See :doc:`porting-guide` for the workflow and the report contract.

Example sources
---------------

A checkout's ``examples/`` directory holds larger CrossGL programs.
``python examples/test.py`` translates each of them to the targets that
support its category and writes the results to ``examples/output/``.

.. list-table::
   :header-rows: 1

   * - Path
     - What it covers
   * - ``examples/graphics/SimpleShader.cgl``
     - Basic vertex/fragment translation
   * - ``examples/graphics/ComplexShader.cgl``
     - Larger shader structure and helper functions
   * - ``examples/compute/ParticleSimulation.cgl``
     - Compute-stage resource and dispatch patterns
   * - ``examples/gpu_computing/MatrixMultiplication.cgl``
     - Parallel compute workload
   * - ``examples/cross_platform/UniversalPBRShader.cgl``
     - Portable physically based rendering shader
   * - ``examples/advanced/GenericPatternMatching.cgl``
     - Pattern matching and generic language constructs
