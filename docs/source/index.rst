CrossTL Documentation
=====================

CrossTL is a universal shader language translator that enables developers to write
shaders once and deploy them across multiple graphics APIs and platforms.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api/index
   backends
   contributing

Features
--------

* **Universal Translation**: Write shaders in CrossGL and translate to HLSL, Metal,
  GLSL, SPIR-V, CUDA, HIP, Mojo, Rust, and Slang.
* **Reverse Translation**: Convert existing shaders back to CrossGL for
  cross-platform portability.
* **Extensible Architecture**: Plugin-based backend system allows adding new
  target languages.
* **Code Formatting**: Built-in formatting support using clang-format and
  SPIR-V tools.

Quick Start
-----------

Installation::

    pip install crosstl

Basic usage::

    import crosstl

    # Translate a CrossGL shader to Metal
    metal_code = crosstl.translate("shader.cgl", backend="metal")

    # Translate HLSL to CrossGL
    cgl_code = crosstl.translate("shader.hlsl", backend="cgl")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
