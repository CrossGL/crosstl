WebGL Backend
=============

The WebGL backend emits WebGL 2.0 compatible GLSL ES from CrossGL. It is
selected through the ``webgl``, ``webgl2``, ``essl``, or ``glsl-es`` target
aliases.

Pipeline
--------

CrossGL output generation is implemented by
``crosstl.translator.codegen.webgl_codegen.WebGLCodeGen``. The generator reuses
the OpenGL GLSL emitter, forces ``#version 300 es``, inserts default precision
qualifiers when they are absent, and rejects shader stages that WebGL cannot
represent.

WebGL source
------------

WebGL 2 shaders are GLSL ES 3.00, a profile of GLSL that the GLSL frontend
reads: their ``#version 300 es`` line defines ``GL_ES`` for the preprocessor
and stays in the imported CrossGL. The source names ``webgl``, ``webgl2``,
``essl`` and ``glsl-es`` select that frontend::

    crosstl.translate("shader.frag", backend="metal", source_backend="webgl")

``.webgl.glsl`` is the extension CrossTL writes WebGL output under, so the
source registry does not read a file with it as source unless the source
backend is named, and a project scan does not import translated output again.

Supported Surface
-----------------

The first WebGL target surface focuses on portable graphics shaders:

* vertex and fragment shader generation
* GLSL ES 3.00 version and precision handling
* stage parameters, return semantics, and struct member semantics inherited
  from the GLSL emitter
* mixed graphics and compute inputs emit the supported vertex/fragment stages
  when present while pure compute inputs still report deterministic diagnostics
* deterministic diagnostics for compute, geometry, tessellation, mesh/task, and
  ray-tracing stages

Implementation Notes
--------------------

Keep WebGL-specific restrictions in ``WebGLCodeGen`` rather than adding WebGL
conditionals throughout ``GLSLCodeGen``. When a capability becomes safe for
WebGL, promote it with focused WebGL tests in the same change.
