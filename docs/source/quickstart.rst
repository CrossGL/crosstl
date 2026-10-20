Quickstart
==========

Install the package from PyPI, or in editable mode from a checkout:

.. code-block:: bash

   pip install crosstl
   pip install -e .

Translate a CrossGL shader from Python. ``save_shader`` is optional:

.. code-block:: python

   import crosstl

   metal = crosstl.translate(
       "examples/graphics/SimpleShader.cgl",
       backend="metal",
       save_shader="SimpleShader.metal",
   )

Or from the command line:

.. code-block:: bash

   python -m crosstl translate examples/graphics/SimpleShader.cgl --backend directx

Native source inputs are selected by file extension. When the source frontend
has a reverse generator, the translator imports the native shader into CrossGL
and can emit CrossGL or any other target:

.. code-block:: python

   crossgl = crosstl.translate("shader.hlsl", backend="cgl")
   metal = crosstl.translate("shader.hlsl", backend="metal")

``crosstl.supported_sources()`` and ``crosstl.supported_backends()`` list the
registered inputs and targets; :doc:`backend-support` describes them.

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
See :doc:`project-porting` for the workflow and the report contract.

Example sources
---------------

The ``examples/`` directory holds CrossGL programs. ``python examples/test.py``
translates each of them to the targets that support its category and writes
the results to ``examples/output/``.

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
