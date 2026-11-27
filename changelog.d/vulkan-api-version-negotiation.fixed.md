Use `vkEnumerateInstanceVersion` to negotiate Vulkan 1.1 and validate SPIR-V
module version before dispatch.

- `VulkanComputeRuntime._create_instance` now calls `vkEnumerateInstanceVersion`
  when available; it requests Vulkan 1.1 when the loader reports at least 1.1,
  and falls back to 1.0 otherwise.
- `_VulkanDispatchContext._create_device_context` stores the effective API version
  as the minimum of the requested instance version and the physical device's
  `VkPhysicalDeviceProperties.apiVersion`.
- `_VulkanDispatchContext._create_shader_module` validates the SPIR-V module version
  header against the effective device API version before calling
  `vkCreateShaderModule`, raising `RuntimeAdapterSetupError` with
  `reasonKind: spirv-version-incompatible` on mismatch.
