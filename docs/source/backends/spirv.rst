Vulkan SPIR-V Backend
=====================

The Vulkan SPIR-V backend covers CrossGL-to-SPIR-V text generation and Vulkan
SPIR-V-style source import. It is selected through the canonical ``vulkan``
target/source name or the ``spirv`` and ``spv`` aliases.

Pipeline
--------

CrossGL output generation is implemented by
``crosstl.translator.codegen.SPIRV_codegen.VulkanSPIRVCodeGen``. The generator
builds a SPIR-V module from the shared translator AST, assigns result ids,
registers types, constants, variables, functions, entry points, capabilities,
decorations, and ordered instruction sections.

Reverse translation uses ``crosstl.backend.SPIRV.VulkanLexer.VulkanLexer`` and
``crosstl.backend.SPIRV.VulkanParser.VulkanParser`` to parse Vulkan/SPIR-V style
source into the Vulkan backend AST. ``crosstl.backend.SPIRV.VulkanCrossGLCodeGen``
then serializes that AST back into CrossGL syntax.

Supported Surface
-----------------

The backend focuses on Vulkan shader module structure:

* shader stages and execution models for vertex, fragment, compute, and related
  entry-point forms
* SPIR-V type, pointer, array, image, sampler, sampled-image, and function type
  registration
* descriptor sets, bindings, push constants, layouts, input/output variables,
  and storage classes
* arithmetic, logical, selection, loop, function-call, texture, image, and
  resource-query instructions
* deterministic module ordering for capabilities, extensions, imports,
  decorations, types, constants, globals, functions, and entry points

Implementation Notes
--------------------

SPIR-V generation is stateful: ids, type caches, decorations, and control-flow
blocks are all part of one module build. Keep module-level state transitions in
``VulkanSPIRVCodeGen`` and avoid sharing mutable SPIR-V state with text
backends. Tests should assert both generated instructions and stable ordering
when changing this backend.
