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

WebGL is a target-only backend. Native WebGL source files such as
``.webgl.glsl`` are intentionally rejected by the source registry so they do
not fall through to OpenGL source parsing by accident.

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
WebGL, promote it by adding focused WebGL tests and updating the support matrix
metadata in the same change.
