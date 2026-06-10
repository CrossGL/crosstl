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
* stage input/output semantics mapped to ``@location`` and ``@builtin``
  attributes
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

Implementation Notes
--------------------

WGSL differs from the C-like shader targets because stage interfaces are part
of entry point signatures and struct field attributes, not global ``in`` and
``out`` declarations. Keep WGSL-specific type, semantic, and binding decisions
inside ``WGSLCodeGen`` until another backend needs the exact same behavior.
