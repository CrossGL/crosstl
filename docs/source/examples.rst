Examples
========

The repository includes CrossGL examples under ``examples/`` and generated
reference outputs under ``examples/output/``.

Translate from Python
---------------------

.. code-block:: python

   import crosstl

   metal = crosstl.translate(
       "examples/graphics/SimpleShader.cgl",
       backend="metal",
       save_shader="SimpleShader.metal",
   )

Translate from the command line
-------------------------------

.. code-block:: bash

   python -m crosstl._crosstl examples/graphics/SimpleShader.cgl --backend directx

Translate native shader input
-----------------------------

Native source inputs are selected by file extension. When a reverse generator
exists, the translator can import the native shader and emit CrossGL or another
target.

.. code-block:: python

   import crosstl

   crossgl = crosstl.translate("shader.hlsl", backend="cgl")
   metal = crosstl.translate("shader.hlsl", backend="metal")

Useful example files
--------------------

.. list-table::
   :header-rows: 1

   * - Category
     - Path
     - What it covers
   * - Graphics
     - ``examples/graphics/SimpleShader.cgl``
     - Basic vertex/fragment translation
   * - Graphics
     - ``examples/graphics/ComplexShader.cgl``
     - Larger shader structure and helper functions
   * - Compute
     - ``examples/compute/ParticleSimulation.cgl``
     - Compute-stage resource and dispatch patterns
   * - GPU computing
     - ``examples/gpu_computing/MatrixMultiplication.cgl``
     - Parallel compute workload
   * - Cross-platform
     - ``examples/cross_platform/UniversalPBRShader.cgl``
     - Portable physically based rendering shader
   * - Advanced language features
     - ``examples/advanced/GenericPatternMatching.cgl``
     - Pattern matching and generic language constructs
