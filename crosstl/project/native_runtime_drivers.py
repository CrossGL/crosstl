"""Optional native runtime drivers for project runtime verification."""

from __future__ import annotations

import ctypes
import importlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeAdapterDispatchError,
    RuntimeAdapterSetupError,
    RuntimeExecutionRequest,
    RuntimeExecutorAvailability,
    RuntimeExecutorUnavailable,
)


@dataclass(frozen=True)
class _PreparedVulkanBuffer:
    name: str
    set_index: int
    binding_index: int
    resource_kind: str | None
    dtype: str
    shape: tuple[int, ...]
    source: str | None
    output_name: str | None
    payload: bytes

    @property
    def size(self) -> int:
        return len(self.payload)


@dataclass
class _VulkanBufferResource:
    prepared: _PreparedVulkanBuffer
    buffer: Any
    memory: Any


class VulkanComputeRuntime:
    """Reference Vulkan compute runtime for simple storage-buffer fixtures.

    The driver is optional. It imports the Python ``vulkan`` binding lazily and
    reports structured unavailability when the binding, Vulkan loader, or a
    compute-capable device is unavailable.
    """

    name = "vulkan-compute-runtime"

    def __init__(
        self,
        *,
        module_loader: Any | None = None,
        application_name: str = "CrossTL Runtime Verification",
        timeout_ns: int = 10_000_000_000,
    ):
        self._module_loader = module_loader or importlib.import_module
        self.application_name = application_name
        self.timeout_ns = int(timeout_ns)

    def is_available(
        self,
        adapter: Any,
        request: RuntimeExecutionRequest,
    ) -> RuntimeExecutorAvailability:
        _ = adapter, request
        try:
            vk = self._load_vulkan()
        except RuntimeExecutorUnavailable as exc:
            return RuntimeExecutorAvailability(
                False,
                reason=str(exc),
                details={
                    "reasonKind": "dependency-unavailable",
                    "target": "vulkan",
                    "missingPythonModules": ["vulkan"],
                },
            )

        instance = None
        try:
            instance = self._create_instance(vk)
            physical_device, queue_family = self._select_compute_device(vk, instance)
        except Exception as exc:  # pragma: no cover - depends on local Vulkan loader
            return RuntimeExecutorAvailability(
                False,
                reason=f"Vulkan runtime is unavailable: {exc}",
                details={
                    "reasonKind": "vulkan-runtime-unavailable",
                    "target": "vulkan",
                    "error": str(exc),
                },
            )
        finally:
            if instance is not None:
                self._destroy_instance(vk, instance)

        return RuntimeExecutorAvailability(
            True,
            details={
                "reasonKind": "available",
                "target": "vulkan",
                "runtime": self.name,
                "physicalDevice": str(physical_device),
                "queueFamilyIndex": queue_family,
            },
        )

    def load_artifact(self, adapter: Any, state: Any, module_path: Path) -> bytes:
        _ = adapter, state
        try:
            code = Path(module_path).read_bytes()
        except OSError as exc:
            raise RuntimeAdapterSetupError(
                f"Vulkan runtime artifact could not be read: {exc}",
                details={"target": "vulkan", "modulePath": str(module_path)},
            ) from exc
        if len(code) % 4:
            raise RuntimeAdapterSetupError(
                "Vulkan runtime artifacts must contain word-aligned SPIR-V bytes.",
                details={
                    "target": "vulkan",
                    "modulePath": str(module_path),
                    "byteLength": len(code),
                },
            )
        return code

    def dispatch(
        self,
        adapter: Any,
        state: Any,
        request: NativeRuntimeDispatchRequest,
    ) -> dict[str, Mapping[str, Any]]:
        _ = adapter, state
        vk = self._load_vulkan()
        shader_code = self._shader_code(request)
        prepared_buffers = _prepare_vulkan_buffers(request.buffers)
        workgroup_count = _workgroup_count(request)
        if not prepared_buffers:
            raise RuntimeExecutorUnavailable(
                "Vulkan compute runtime requires at least one storage buffer."
            )

        context = _VulkanDispatchContext(
            vk=vk,
            runtime=self,
            shader_code=shader_code,
            entry_point=request.entry_point or "main",
            buffers=prepared_buffers,
            workgroup_count=workgroup_count,
        )
        return context.run()

    def _load_vulkan(self) -> Any:
        try:
            return self._module_loader("vulkan")
        except Exception as exc:
            raise RuntimeExecutorUnavailable(
                "Python Vulkan bindings are unavailable; install the optional "
                "'vulkan' package to run Vulkan runtime fixtures."
            ) from exc

    def _shader_code(self, request: NativeRuntimeDispatchRequest) -> bytes:
        if isinstance(request.loaded_artifact, bytes):
            return request.loaded_artifact
        if isinstance(request.loaded_artifact, bytearray):
            return bytes(request.loaded_artifact)
        return self.load_artifact(None, None, request.module_path)

    def _create_instance(self, vk: Any) -> Any:
        application_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=self.application_name,
            applicationVersion=1,
            pEngineName="CrossTL",
            engineVersion=1,
            apiVersion=getattr(vk, "VK_API_VERSION_1_1", vk.VK_API_VERSION_1_0),
        )
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=application_info,
        )
        return vk.vkCreateInstance(create_info, None)

    def _destroy_instance(self, vk: Any, instance: Any) -> None:
        destroy = getattr(vk, "vkDestroyInstance", None)
        if callable(destroy):
            destroy(instance, None)

    def _select_compute_device(self, vk: Any, instance: Any) -> tuple[Any, int]:
        devices = vk.vkEnumeratePhysicalDevices(instance)
        for physical_device in devices:
            families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
            for index, family in enumerate(families):
                if int(family.queueFlags) & int(vk.VK_QUEUE_COMPUTE_BIT):
                    return physical_device, index
        raise RuntimeExecutorUnavailable(
            "No Vulkan physical device exposes a compute queue family."
        )


class _VulkanDispatchContext:
    def __init__(
        self,
        *,
        vk: Any,
        runtime: VulkanComputeRuntime,
        shader_code: bytes,
        entry_point: str,
        buffers: Sequence[_PreparedVulkanBuffer],
        workgroup_count: tuple[int, int, int],
    ):
        self.vk = vk
        self.runtime = runtime
        self.shader_code = shader_code
        self.entry_point = entry_point
        self.buffers = tuple(buffers)
        self.workgroup_count = workgroup_count
        self.instance = None
        self.physical_device = None
        self.queue_family = None
        self.device = None
        self.queue = None
        self.shader_module = None
        self.descriptor_set_layout = None
        self.pipeline_layout = None
        self.pipeline = None
        self.descriptor_pool = None
        self.descriptor_set = None
        self.command_pool = None
        self.command_buffer = None
        self.fence = None
        self.resources: list[_VulkanBufferResource] = []

    def run(self) -> dict[str, Mapping[str, Any]]:
        try:
            self._create_device_context()
            self._create_shader_module()
            self._create_buffers()
            self._create_descriptors()
            self._create_pipeline()
            self._record_and_submit()
            return self._read_outputs()
        except RuntimeExecutorUnavailable:
            raise
        except RuntimeAdapterDispatchError:
            raise
        except Exception as exc:  # pragma: no cover - depends on local Vulkan driver
            raise RuntimeAdapterDispatchError(
                f"Vulkan compute dispatch failed: {exc}",
                details={"target": "vulkan", "runtime": self.runtime.name},
            ) from exc
        finally:
            self._cleanup()

    def _create_device_context(self) -> None:
        vk = self.vk
        self.instance = self.runtime._create_instance(vk)
        self.physical_device, self.queue_family = self.runtime._select_compute_device(
            vk, self.instance
        )
        priority = [1.0]
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.queue_family,
            queueCount=1,
            pQueuePriorities=priority,
        )
        device_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
        )
        self.device = vk.vkCreateDevice(self.physical_device, device_info, None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.queue_family, 0)

    def _create_shader_module(self) -> None:
        vk = self.vk
        create_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(self.shader_code),
            pCode=self.shader_code,
        )
        self.shader_module = vk.vkCreateShaderModule(self.device, create_info, None)

    def _create_buffers(self) -> None:
        for prepared in self.buffers:
            self.resources.append(self._create_buffer(prepared))

    def _create_buffer(self, prepared: _PreparedVulkanBuffer) -> _VulkanBufferResource:
        vk = self.vk
        create_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=max(prepared.size, 1),
            usage=_vulkan_buffer_usage(vk, prepared.resource_kind),
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buffer_handle = vk.vkCreateBuffer(self.device, create_info, None)
        requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer_handle)
        memory_type = self._find_memory_type(
            int(requirements.memoryTypeBits),
            int(vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            | int(vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        )
        allocation = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=requirements.size,
            memoryTypeIndex=memory_type,
        )
        memory = vk.vkAllocateMemory(self.device, allocation, None)
        vk.vkBindBufferMemory(self.device, buffer_handle, memory, 0)
        self._write_memory(memory, prepared.payload)
        return _VulkanBufferResource(
            prepared=prepared, buffer=buffer_handle, memory=memory
        )

    def _find_memory_type(self, type_bits: int, required_flags: int) -> int:
        properties = self.vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        for index in range(int(properties.memoryTypeCount)):
            if not (type_bits & (1 << index)):
                continue
            flags = int(properties.memoryTypes[index].propertyFlags)
            if flags & required_flags == required_flags:
                return index
        raise RuntimeExecutorUnavailable(
            "No host-visible Vulkan memory type is available for runtime buffers."
        )

    def _write_memory(self, memory: Any, payload: bytes) -> None:
        if not payload:
            return
        pointer = self.vk.vkMapMemory(self.device, memory, 0, len(payload), 0)
        try:
            _write_mapped_memory(pointer, payload)
        finally:
            self.vk.vkUnmapMemory(self.device, memory)

    def _read_memory(self, memory: Any, size: int) -> bytes:
        pointer = self.vk.vkMapMemory(self.device, memory, 0, size, 0)
        try:
            return _read_mapped_memory(pointer, size)
        finally:
            self.vk.vkUnmapMemory(self.device, memory)

    def _create_descriptors(self) -> None:
        set_indices = {buffer.set_index for buffer in self.buffers}
        if set_indices != {0}:
            raise RuntimeExecutorUnavailable(
                "Vulkan compute runtime currently supports a single descriptor set 0."
            )
        vk = self.vk
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=buffer.binding_index,
                descriptorType=_vulkan_descriptor_type(vk, buffer.resource_kind),
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for buffer in self.buffers
        ]
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        self.descriptor_set_layout = vk.vkCreateDescriptorSetLayout(
            self.device, layout_info, None
        )
        descriptor_counts: dict[Any, int] = {}
        for buffer in self.buffers:
            descriptor_type = _vulkan_descriptor_type(vk, buffer.resource_kind)
            descriptor_counts[descriptor_type] = (
                descriptor_counts.get(descriptor_type, 0) + 1
            )
        pool_sizes = [
            vk.VkDescriptorPoolSize(type=descriptor_type, descriptorCount=count)
            for descriptor_type, count in descriptor_counts.items()
        ]
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=1,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        self.descriptor_pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)
        allocate_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.descriptor_set_layout],
        )
        self.descriptor_set = vk.vkAllocateDescriptorSets(self.device, allocate_info)[0]
        writes = []
        for resource in self.resources:
            buffer_info = vk.VkDescriptorBufferInfo(
                buffer=resource.buffer,
                offset=0,
                range=max(resource.prepared.size, 1),
            )
            writes.append(
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=self.descriptor_set,
                    dstBinding=resource.prepared.binding_index,
                    descriptorCount=1,
                    descriptorType=_vulkan_descriptor_type(
                        vk, resource.prepared.resource_kind
                    ),
                    pBufferInfo=[buffer_info],
                )
            )
        vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)

    def _create_pipeline(self) -> None:
        vk = self.vk
        layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.descriptor_set_layout],
        )
        self.pipeline_layout = vk.vkCreatePipelineLayout(self.device, layout_info, None)
        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.shader_module,
            pName=self.entry_point,
        )
        pipeline_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=self.pipeline_layout,
        )
        pipelines = vk.vkCreateComputePipelines(
            self.device,
            getattr(vk, "VK_NULL_HANDLE", None),
            1,
            [pipeline_info],
            None,
        )
        self.pipeline = _first_vulkan_handle(pipelines)

    def _record_and_submit(self) -> None:
        vk = self.vk
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.queue_family,
        )
        self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)
        allocate_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self.command_buffer = vk.vkAllocateCommandBuffers(self.device, allocate_info)[0]
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        )
        vk.vkBeginCommandBuffer(self.command_buffer, begin_info)
        vk.vkCmdBindPipeline(
            self.command_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline,
        )
        vk.vkCmdBindDescriptorSets(
            self.command_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_layout,
            0,
            1,
            [self.descriptor_set],
            0,
            None,
        )
        vk.vkCmdDispatch(self.command_buffer, *self.workgroup_count)
        vk.vkEndCommandBuffer(self.command_buffer)
        fence_info = vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        self.fence = vk.vkCreateFence(self.device, fence_info, None)
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffer],
        )
        vk.vkQueueSubmit(self.queue, 1, [submit_info], self.fence)
        vk.vkWaitForFences(self.device, 1, [self.fence], True, self.runtime.timeout_ns)

    def _read_outputs(self) -> dict[str, Mapping[str, Any]]:
        outputs: dict[str, Mapping[str, Any]] = {}
        for resource in self.resources:
            prepared = resource.prepared
            if prepared.source != "expectedOutput":
                continue
            payload = self._read_memory(resource.memory, prepared.size)
            outputs[prepared.output_name or prepared.name] = {
                "dtype": prepared.dtype,
                "shape": list(prepared.shape),
                "values": _unpack_values(payload, prepared.dtype),
            }
        return outputs

    def _cleanup(self) -> None:
        vk = self.vk
        device = self.device
        if device is not None:
            wait_idle = getattr(vk, "vkDeviceWaitIdle", None)
            if callable(wait_idle):
                try:
                    wait_idle(device)
                except Exception:
                    pass
        _vk_destroy(vk, "vkDestroyFence", device, self.fence)
        _vk_destroy(vk, "vkDestroyCommandPool", device, self.command_pool)
        _vk_destroy(vk, "vkDestroyPipeline", device, self.pipeline)
        _vk_destroy(vk, "vkDestroyPipelineLayout", device, self.pipeline_layout)
        _vk_destroy(vk, "vkDestroyDescriptorPool", device, self.descriptor_pool)
        _vk_destroy(
            vk,
            "vkDestroyDescriptorSetLayout",
            device,
            self.descriptor_set_layout,
        )
        _vk_destroy(vk, "vkDestroyShaderModule", device, self.shader_module)
        for resource in reversed(self.resources):
            _vk_destroy(vk, "vkDestroyBuffer", device, resource.buffer)
            free_memory = getattr(vk, "vkFreeMemory", None)
            if (
                callable(free_memory)
                and device is not None
                and resource.memory is not None
            ):
                free_memory(device, resource.memory, None)
        _vk_destroy(vk, "vkDestroyDevice", None, device)
        if self.instance is not None:
            self.runtime._destroy_instance(vk, self.instance)


def _vk_destroy(vk: Any, name: str, owner: Any, handle: Any) -> None:
    if handle is None:
        return
    destroy = getattr(vk, name, None)
    if not callable(destroy):
        return
    if owner is None:
        destroy(handle, None)
    else:
        destroy(owner, handle, None)


def _first_vulkan_handle(handles: Any) -> Any:
    if isinstance(handles, (str, bytes, bytearray)):
        return handles
    try:
        return handles[0]
    except (IndexError, TypeError, KeyError):
        return handles


def _write_mapped_memory(mapped: Any, payload: bytes) -> None:
    try:
        view = memoryview(mapped)
    except TypeError:
        ctypes.memmove(mapped, payload, len(payload))
        return

    try:
        byte_view = view.cast("B") if view.format != "B" else view
        byte_view[: len(payload)] = payload
    finally:
        view.release()


def _read_mapped_memory(mapped: Any, size: int) -> bytes:
    try:
        view = memoryview(mapped)
    except TypeError:
        return ctypes.string_at(mapped, size)

    try:
        byte_view = view.cast("B") if view.format != "B" else view
        return byte_view[:size].tobytes()
    finally:
        view.release()


def _vulkan_descriptor_type(vk: Any, resource_kind: str | None) -> Any:
    if resource_kind in {"constant-buffer", "uniform"}:
        return vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    return vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER


def _vulkan_buffer_usage(vk: Any, resource_kind: str | None) -> Any:
    if resource_kind in {"constant-buffer", "uniform"}:
        return vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    return vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT


def _prepare_vulkan_buffers(
    bindings: Mapping[str, NativeRuntimeBufferBinding],
) -> tuple[_PreparedVulkanBuffer, ...]:
    prepared = []
    seen_bindings: set[tuple[int, int]] = set()
    for name, binding in bindings.items():
        resource = binding.binding
        if resource.kind not in (
            None,
            "buffer",
            "storage-buffer",
            "constant-buffer",
            "uniform",
        ):
            raise RuntimeExecutorUnavailable(
                f"Vulkan compute runtime supports buffer resources only: {name}."
            )
        if resource.binding is None:
            raise RuntimeExecutorUnavailable(
                f"Vulkan compute runtime requires an explicit binding for {name}."
            )
        set_index = _int_field(resource.set, default=0)
        binding_index = _int_field(resource.binding)
        descriptor = (set_index, binding_index)
        if descriptor in seen_bindings:
            raise RuntimeExecutorUnavailable(
                "Vulkan compute runtime requires unique set/binding pairs."
            )
        seen_bindings.add(descriptor)
        dtype = _normalize_dtype(binding.dtype)
        shape = tuple(int(value) for value in binding.shape)
        element_count = (
            math.prod(shape) if shape else len(_flatten_values(binding.value))
        )
        if binding.source == "expectedOutput":
            payload = b"\x00" * (element_count * _dtype_size(dtype))
        else:
            payload = _pack_values(binding.value, dtype, expected_count=element_count)
        prepared.append(
            _PreparedVulkanBuffer(
                name=name,
                set_index=set_index,
                binding_index=binding_index,
                resource_kind=resource.kind,
                dtype=dtype,
                shape=shape,
                source=binding.source,
                output_name=_runtime_value_name(binding),
                payload=payload,
            )
        )
    return tuple(
        sorted(prepared, key=lambda item: (item.set_index, item.binding_index))
    )


def _runtime_value_name(binding: NativeRuntimeBufferBinding) -> str | None:
    value_name = binding.metadata.get("runtimeValueName")
    if isinstance(value_name, str) and value_name.strip():
        return value_name.strip()
    return None


def _workgroup_count(request: NativeRuntimeDispatchRequest) -> tuple[int, int, int]:
    dispatch = request.dispatch
    if dispatch is None:
        raise RuntimeExecutorUnavailable(
            "Vulkan compute runtime requires dispatch geometry."
        )
    if dispatch.workgroup_count:
        values = tuple(int(value) for value in dispatch.workgroup_count)
    elif dispatch.global_size and dispatch.workgroup_size:
        values = tuple(
            max(1, math.ceil(int(global_value) / int(local_value)))
            for global_value, local_value in zip(
                dispatch.global_size,
                dispatch.workgroup_size,
            )
        )
    else:
        raise RuntimeExecutorUnavailable(
            "Vulkan compute runtime requires workgroupCount or globalSize/workgroupSize."
        )
    return _pad3(values, field_name="workgroupCount")


def _pad3(values: Sequence[int], *, field_name: str) -> tuple[int, int, int]:
    if len(values) > 3:
        raise RuntimeExecutorUnavailable(
            f"Vulkan compute runtime {field_name} must have at most three dimensions."
        )
    padded = tuple(max(1, int(value)) for value in values) + (1,) * (3 - len(values))
    return padded[:3]


def _int_field(value: Any, *, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise RuntimeExecutorUnavailable("Expected an integer field.")
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeExecutorUnavailable(
            f"Expected an integer field, got {value!r}."
        ) from exc


def _normalize_dtype(dtype: str | None) -> str:
    aliases = {
        "float": "float32",
        "f32": "float32",
        "float32_t": "float32",
        "uint": "uint32",
        "u32": "uint32",
        "int": "int32",
        "i32": "int32",
    }
    normalized = str(dtype or "").strip().lower()
    value = aliases.get(normalized, normalized)
    if value not in {"float32", "uint32", "int32"}:
        raise RuntimeExecutorUnavailable(
            f"Vulkan compute runtime supports float32, uint32, and int32 buffers, not {dtype!r}."
        )
    return value


def _dtype_format(dtype: str) -> str:
    return {"float32": "f", "uint32": "I", "int32": "i"}[dtype]


def _dtype_size(dtype: str) -> int:
    return struct.calcsize("<" + _dtype_format(dtype))


def _flatten_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    if isinstance(value, Sequence):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_values(item))
        return flattened
    return [value]


def _pack_values(value: Any, dtype: str, *, expected_count: int) -> bytes:
    values = _flatten_values(value)
    if len(values) != expected_count:
        raise RuntimeExecutorUnavailable(
            "Vulkan compute runtime buffer value count does not match shape."
        )
    return struct.pack("<" + _dtype_format(dtype) * expected_count, *values)


def _unpack_values(payload: bytes, dtype: str) -> list[Any]:
    size = _dtype_size(dtype)
    if len(payload) % size:
        raise RuntimeAdapterDispatchError(
            "Vulkan runtime output byte length is not aligned to the dtype size.",
            details={"target": "vulkan", "dtype": dtype, "byteLength": len(payload)},
        )
    count = len(payload) // size
    if count == 0:
        return []
    return list(struct.unpack("<" + _dtype_format(dtype) * count, payload))
