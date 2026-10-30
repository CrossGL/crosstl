WGSL Backend
============

The WGSL backend emits WebGPU shader source from CrossGL. It is selected
through the ``wgsl`` or ``webgpu`` target aliases.

Pipeline
--------

CrossGL output generation is implemented by
``crosstl.translator.codegen.wgsl_codegen.WGSLCodeGen``. The generator maps the
shared CrossGL AST to WGSL syntax, including ``fn`` signatures, WGSL scalar and
vector types, entry point attributes, struct member attributes, and compute
``@workgroup_size`` metadata.

WGSL is a target-only backend. ``.wgsl`` and ``.wesl`` files remain rejected as
source inputs until the project has a dedicated WGSL frontend.

Supported Surface
-----------------

The initial WGSL target surface covers common portable shader output:

* vertex, fragment, and compute entry point generation
* scalar, vector, matrix, array, struct, function, and local variable syntax
* generic functions, monomorphized into one concrete WGSL function per
  instantiation the module calls; calls whose type arguments cannot be
  deduced, ambiguous generic overloads and generic resource parameters are
  refused
* stage input/output semantics mapped to ``@location`` and ``@builtin``
  attributes; unbound vertex and fragment entry parameters take the lowest
  free ``@location``, integer user IO is marked ``@interpolate(flat)``, and a
  ``gl_*``/``SV_*`` system semantic without a WGSL built-in value (for example
  ``SV_PrimitiveID`` or ``SV_ClipDistance0``) is refused with
  ``project.translate.wgsl-stage-io-unsupported`` instead of becoming a user
  location
* explicit WGSL ``@location``, ``@builtin``, ``@interpolate``, and
  ``@invariant`` attributes preserved from CrossGL declarations
* compute workgroup size metadata from CrossGL layout qualifiers
* direct compute builtin references such as ``gl_GlobalInvocationID`` injected
  as WGSL entry point parameters
* compute ``barrier()`` calls lowered to ``workgroupBarrier()``
* deterministic diagnostics for non-WebGPU stages and pointer types that do not
  have a safe WGSL lowering yet
* deterministic diagnostics for CrossGL combined texture/sampler resources,
  and texture sampling calls that need explicit WebGPU binding lowering before
  they can emit valid WGSL
* typed diagnostics (``project.translate.wgsl-numeric-type-unsupported``) for
  64-bit floating-point, 64-bit integer and bfloat16 types, which WGSL cannot
  represent; they are never narrowed to ``f32`` or ``i32``
* ``texture(sampler, coords, bias)`` lowered to ``textureSampleBias`` with the
  texture's split sampler; a bias on a 1D texture, which has no
  ``textureSampleBias`` overload, or on an element of a sampled texture array
  selected at run time is refused with
  ``project.translate.wgsl-texture-bias-unsupported``, and a bias in a compute
  stage with ``project.translate.wgsl-implicit-lod-stage-unsupported``
* typed diagnostics (``project.translate.wgsl-integer-texture-sample-unsupported``)
  for ``texture``, ``textureLod``, ``textureGrad``, ``textureOffset`` and their
  offset forms on ``isampler*`` and ``usampler*`` textures, which WGSL cannot
  sample; ``texelFetch`` and ``textureSize`` on them become ``textureLoad`` and
  ``textureDimensions``
* typed diagnostics (``project.translate.wgsl-cube-texture-offset-unsupported``)
  for ``textureOffset``, ``textureLodOffset``, ``textureGradOffset``,
  ``textureCompareOffset`` and ``textureCompareLodOffset`` on cube array
  textures, for which WGSL has no offset lookup

Implementation Notes
--------------------

WGSL differs from the C-like shader targets because stage interfaces are part
of entry point signatures and struct field attributes, not global ``in`` and
``out`` declarations. Keep WGSL-specific type, semantic, and binding decisions
inside ``WGSLCodeGen`` until another backend needs the exact same behavior.
