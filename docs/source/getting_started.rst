Getting Started
===============

Installation
------------

CrossTL can be installed via pip::

    pip install crosstl

For development, clone the repository and install in editable mode::

    git clone https://github.com/CrossGL/crosstl.git
    cd crosstl
    pip install -e .

Basic Usage
-----------

Translating Shaders
~~~~~~~~~~~~~~~~~~~

The main entry point for CrossTL is the ``translate`` function:

.. code-block:: python

    import crosstl

    # Translate a CrossGL shader to HLSL
    hlsl_code = crosstl.translate(
        "shader.cgl",
        backend="directx",
        save_shader="output.hlsl"
    )

    # Translate to Metal
    metal_code = crosstl.translate("shader.cgl", backend="metal")

    # Translate to GLSL
    glsl_code = crosstl.translate("shader.cgl", backend="opengl")

Supported Backends
~~~~~~~~~~~~~~~~~~

CrossTL supports the following target backends:

* ``directx`` / ``hlsl`` - DirectX HLSL
* ``metal`` - Apple Metal Shading Language
* ``opengl`` / ``glsl`` - OpenGL GLSL
* ``vulkan`` / ``spirv`` - Vulkan SPIR-V
* ``cuda`` - NVIDIA CUDA
* ``hip`` - AMD HIP
* ``mojo`` - Mojo
* ``rust`` - Rust GPU
* ``slang`` - Slang

Reverse Translation
~~~~~~~~~~~~~~~~~~~

CrossTL also supports reverse translation from existing shader languages back
to CrossGL:

.. code-block:: python

    import crosstl

    # Convert HLSL to CrossGL
    cgl_code = crosstl.translate("shader.hlsl", backend="cgl")

    # Convert Metal to CrossGL
    cgl_code = crosstl.translate("shader.metal", backend="cgl")

Command-Line Interface
----------------------

CrossTL provides a command-line interface for shader translation::

    # Translate to Metal
    python -m crosstl shader.cgl --backend metal --output shader.metal

    # Translate to HLSL without formatting
    python -m crosstl shader.cgl -b directx -o shader.hlsl --no-format

CrossGL Language
----------------

CrossGL is a unified shader language designed for cross-platform development.
Here's an example vertex and fragment shader:

.. code-block:: text

    shader main {
        vertex {
            input vec3 position;
            input vec2 texCoord;
            output vec2 vTexCoord;

            fn main() {
                vTexCoord = texCoord;
                gl_Position = vec4(position, 1.0);
            }
        }

        fragment {
            input vec2 vTexCoord;
            output vec4 fragColor;

            fn main() {
                fragColor = vec4(vTexCoord, 0.0, 1.0);
            }
        }
    }
