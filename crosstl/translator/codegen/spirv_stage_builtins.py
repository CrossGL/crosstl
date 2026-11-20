"""Stage built-ins and execution modes in the SPIR-V generator: stage built-in info and capabilities, interface variables, fragment depth, workgroup size built-ins, descriptor types, specialization-sized arrays, function qualifiers, local sizes, stage attributes, layout directions, mesh output limits, geometry output modes and tessellation modes."""

import re
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

from ..host_workgroup_size import require_static_workgroup_size
from ..matrix_shape import crossgl_matrix_shape
from ..refusals import VulkanRefusal
from ..stage_utils import (
    STAGE_QUALIFIER_NAMES,
    normalize_stage_name,
)
from .array_utils import evaluate_literal_int_expression
from .specialized_workgroup_size import has_specialized_workgroup_size
from .spirv_entry_workgroup_sizes import (
    effective_workgroup_config,
    entry_workgroup_size_value,
)
from .spirv_extension_builtins import (
    extension_builtin_info,
    require_extension_builtin_capability,
)
from .spirv_per_vertex_builtins import per_vertex_builtin_info
from .spirv_ray_builtins import ray_builtin_info
from .spirv_shading_rate import (
    require_shading_rate_capability,
    shading_rate_builtin_info,
)
from .spirv_specialized_workgroup_size import workgroup_size_components
from .spirv_tessellation_domain import (
    evaluation_takes_glsl_defaults,
    refuse_program_without_domain,
)
from .spirv_values import (
    SpirvId,
    SpirvType,
    UnsupportedSPIRVFeatureError,
)


class SpirvStageBuiltinMixin:
    """Stage builtin members of ``VulkanSPIRVCodeGen``."""

    def stage_builtin_info(self, name: str):
        name = self.spirv_builtin_alias(name)
        # Before the graphics tables: gl_PrimitiveID and gl_InstanceID name other built-ins in ray stages.
        ray_builtin = ray_builtin_info(self, name)
        if ray_builtin is not None:
            return ray_builtin
        per_vertex = per_vertex_builtin_info(self, name)
        if per_vertex is not None:
            return per_vertex
        shading_rate = shading_rate_builtin_info(name)
        if shading_rate is not None:
            return shading_rate
        extension_builtin = extension_builtin_info(name)
        if extension_builtin is not None:
            return extension_builtin
        builtins = {
            "gl_VertexID": (
                "uint",
                "VertexIndex",
                "Input",
                {"Vertex"},
            ),
            "gl_InstanceID": (
                "uint",
                "InstanceIndex",
                "Input",
                {"Vertex"},
            ),
            # Base-inclusive; lower_spirv_instance_ids spells gl_InstanceID as this minus BaseInstance.
            "gl_InstanceIndex": (
                "uint",
                "InstanceIndex",
                "Input",
                {"Vertex"},
            ),
            "gl_FragCoord": (
                "vec4",
                "FragCoord",
                "Input",
                {"Fragment"},
            ),
            "gl_FrontFacing": (
                "bool",
                "FrontFacing",
                "Input",
                {"Fragment"},
            ),
            "gl_PointCoord": (
                "vec2",
                "PointCoord",
                "Input",
                {"Fragment"},
            ),
            "gl_FragSizeEXT": (
                "uvec2",
                "FragSizeEXT",
                "Input",
                {"Fragment"},
            ),
            "gl_FragInvocationCountEXT": (
                "uint",
                "FragInvocationCountEXT",
                "Input",
                {"Fragment"},
            ),
            "gl_SampleID": (
                "int",
                "SampleId",
                "Input",
                {"Fragment"},
            ),
            "gl_SamplePosition": (
                "vec2",
                "SamplePosition",
                "Input",
                {"Fragment"},
            ),
            "gl_SampleMaskIn": (
                "int[1]",
                "SampleMask",
                "Input",
                {"Fragment"},
            ),
            "gl_SampleMask": (
                "int[1]",
                "SampleMask",
                "Output",
                {"Fragment"},
            ),
            "gl_BaryCoordEXT": (
                "vec3",
                "BaryCoordKHR",
                "Input",
                {"Fragment"},
            ),
            "gl_BaryCoordNoPerspEXT": (
                "vec3",
                "BaryCoordNoPerspKHR",
                "Input",
                {"Fragment"},
            ),
            "gl_Position": (
                "vec4",
                "Position",
                "Output",
                {"Vertex", "TessellationEvaluation", "Geometry"},
            ),
            "gl_PointSize": (
                "float",
                "PointSize",
                "Output",
                {"Vertex", "TessellationEvaluation", "Geometry"},
            ),
            "gl_FragDepth": (
                "float",
                "FragDepth",
                "Output",
                {"Fragment"},
            ),
            "gl_FragStencilRefEXT": (
                "int",
                "FragStencilRefEXT",
                "Output",
                {"Fragment"},
            ),
            "gl_InvocationID": (
                "int",
                "InvocationId",
                "Input",
                {"Geometry", "TessellationControl"},
            ),
            "gl_PrimitiveID": (
                "int",
                "PrimitiveId",
                "Input",
                {
                    "Fragment",
                    "Geometry",
                    "TessellationControl",
                    "TessellationEvaluation",
                },
            ),
            "gl_ViewIndex": (
                "int",
                "ViewIndex",
                "Input",
                {
                    "Vertex",
                    "Fragment",
                    "Geometry",
                    "TessellationControl",
                    "TessellationEvaluation",
                },
            ),
            "gl_TessCoord": (
                "vec3",
                "TessCoord",
                "Input",
                {"TessellationEvaluation"},
            ),
            "gl_PatchVerticesIn": (
                "int",
                "PatchVertices",
                "Input",
                {"TessellationControl", "TessellationEvaluation"},
            ),
            "gl_TessLevelOuter": (
                "float[4]",
                "TessLevelOuter",
                {
                    "TessellationControl": "Output",
                    "TessellationEvaluation": "Input",
                },
                {"TessellationControl", "TessellationEvaluation"},
            ),
            "gl_TessLevelInner": (
                "float[2]",
                "TessLevelInner",
                {
                    "TessellationControl": "Output",
                    "TessellationEvaluation": "Input",
                },
                {"TessellationControl", "TessellationEvaluation"},
            ),
            "gl_LaunchIDEXT": (
                "uvec3",
                "LaunchIdKHR",
                "Input",
                {
                    "RayGenerationKHR",
                    "IntersectionKHR",
                    "AnyHitKHR",
                    "ClosestHitKHR",
                    "MissKHR",
                    "CallableKHR",
                },
            ),
            "gl_LaunchSizeEXT": (
                "uvec3",
                "LaunchSizeKHR",
                "Input",
                {
                    "RayGenerationKHR",
                    "IntersectionKHR",
                    "AnyHitKHR",
                    "ClosestHitKHR",
                    "MissKHR",
                    "CallableKHR",
                },
            ),
            "gl_HitTEXT": (
                "float",
                "RayTmaxKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR"},
            ),
            "gl_HitKindEXT": (
                "uint",
                "HitKindKHR",
                "Input",
                {"AnyHitKHR", "ClosestHitKHR"},
            ),
            "gl_WorldRayOriginEXT": (
                "vec3",
                "WorldRayOriginKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR", "MissKHR"},
            ),
            "gl_WorldRayDirectionEXT": (
                "vec3",
                "WorldRayDirectionKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR", "MissKHR"},
            ),
            "gl_ObjectRayOriginEXT": (
                "vec3",
                "ObjectRayOriginKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR"},
            ),
            "gl_ObjectRayDirectionEXT": (
                "vec3",
                "ObjectRayDirectionKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR"},
            ),
            "gl_RayTminEXT": (
                "float",
                "RayTminKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR", "MissKHR"},
            ),
            "gl_InstanceCustomIndexEXT": (
                "int",
                "InstanceCustomIndexKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR"},
            ),
            "gl_GeometryIndexEXT": (
                "int",
                "RayGeometryIndexKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR"},
            ),
        }
        return builtins.get(name)

    def stage_builtin_execution_label(self, execution_models: set) -> str:
        order = [
            "Vertex",
            "Fragment",
            "GLCompute",
            "Geometry",
            "TessellationControl",
            "TessellationEvaluation",
            "MeshEXT",
            "TaskEXT",
            "RayGenerationKHR",
            "IntersectionKHR",
            "AnyHitKHR",
            "ClosestHitKHR",
            "MissKHR",
            "CallableKHR",
        ]
        models = [model for model in order if model in execution_models]
        models.extend(sorted(execution_models - set(models)))
        return ", ".join(models)

    def resolve_stage_builtin_storage_class(
        self, name: str, storage_spec, execution_models: set
    ) -> Optional[str]:
        execution_model = self.current_stage_builtin_execution_model()

        if execution_model is not None and execution_model not in execution_models:
            self.refuse_lowering(
                f"SPIR-V builtin {name} is only valid in "
                f"{self.stage_builtin_execution_label(execution_models)} stages"
            )
            return None

        if not isinstance(storage_spec, dict):
            return storage_spec

        if execution_model is None:
            self.refuse_lowering(
                f"SPIR-V builtin {name} requires a single "
                "tessellation execution model to choose Input or Output storage"
            )
            return None

        return storage_spec.get(execution_model)

    def current_stage_builtin_execution_model(self) -> Optional[str]:
        execution_model = self.current_execution_model
        if execution_model is not None:
            return execution_model
        function_models = self.execution_models_for_function(self.current_function_node)
        if len(function_models) == 1:
            return next(iter(function_models))
        return None

    def stage_builtin_cache_key(
        self, name: str, storage_spec, storage_class: str
    ) -> str:
        if isinstance(storage_spec, dict):
            return f"{name}::{storage_class}"
        return name

    def mark_function_interface_variable(self, variable: SpirvId):
        if self.current_function_id is None:
            return

        variables = self.function_interface_variables.setdefault(
            self.current_function_id, []
        )
        if all(existing.id != variable.id for existing in variables):
            variables.append(variable)

        if self.current_function_name is not None:
            named_variables = self.function_interface_variables_by_name.setdefault(
                self.current_function_name, []
            )
            if all(existing.id != variable.id for existing in named_variables):
                named_variables.append(variable)

    def mark_interface_variable_if_needed(self, variable: Optional[SpirvId]):
        if variable is None:
            return

        for interface_variable in self.inputs + self.outputs:
            if interface_variable.id == variable.id:
                self.mark_function_interface_variable(interface_variable)
                return
        if (
            self.include_resource_interface_variables
            and variable.type.storage_class in {
                "StorageBuffer",
                "Uniform",
                "UniformConstant",
            }
            and any(
                global_variable.id == variable.id
                for global_variable in (
                    *self.global_variables.values(),
                    *self.cbuffer_variables.values(),
                )
            )
        ):
            self.mark_function_interface_variable(variable)

    def merge_function_interface_variables_from_callee(
        self, function_name: str, function_id: Optional[int] = None
    ):
        if function_id is not None:
            for variable in self.function_interface_variables.get(function_id, []):
                self.mark_function_interface_variable(variable)
            if function_id in self.function_interface_variables:
                return

        for variable in self.function_interface_variables_by_name.get(
            function_name, []
        ):
            self.mark_function_interface_variable(variable)

    def mark_builtin_interface_variable(self, variable: SpirvId):
        if variable.id in self.builtin_interface_variable_ids:
            self.mark_function_interface_variable(variable)

    def mark_fragment_depth_replacing_if_needed(
        self, builtin_name: str, storage_class: str
    ):
        if (
            builtin_name != "FragDepth"
            or storage_class != "Output"
            or self.current_execution_model != "Fragment"
            or self.current_function_id is None
        ):
            return
        self.fragment_depth_replacing_function_ids.add(self.current_function_id)

    def mark_fragment_stencil_ref_replacing_if_needed(
        self, builtin_name: str, storage_class: str
    ):
        if (
            builtin_name != "FragStencilRefEXT"
            or storage_class != "Output"
            or self.current_execution_model != "Fragment"
            or self.current_function_id is None
        ):
            return
        self.fragment_stencil_ref_replacing_function_ids.add(self.current_function_id)

    def mark_fragment_builtin_execution_mode_if_needed(
        self, builtin_name: str, storage_class: str
    ):
        self.mark_fragment_depth_replacing_if_needed(builtin_name, storage_class)
        self.mark_fragment_stencil_ref_replacing_if_needed(builtin_name, storage_class)

    def require_stage_builtin_capabilities(self, builtin_name: str):
        if require_shading_rate_capability(self, builtin_name):
            return
        if require_extension_builtin_capability(self, builtin_name):
            return
        if builtin_name in {"BaryCoordKHR", "BaryCoordNoPerspKHR"}:
            self.require_capability("FragmentBarycentricKHR")
            self.require_extension("SPV_KHR_fragment_shader_barycentric")
        elif builtin_name in {"FragSizeEXT", "FragInvocationCountEXT"}:
            self.require_capability("FragmentDensityEXT")
            self.require_extension("SPV_EXT_fragment_invocation_density")
        elif builtin_name in {"SampleId", "SamplePosition"}:
            self.require_capability("SampleRateShading")
        elif builtin_name == "FragStencilRefEXT":
            self.require_capability("StencilExportEXT")
            self.require_extension("SPV_EXT_shader_stencil_export")
        elif (
            builtin_name == "PrimitiveId"
            and self.current_stage_builtin_execution_model() == "Fragment"
        ):
            # SPIR-V section 3.21: PrimitiveId requires Geometry, Tessellation, ray tracing or mesh shading; a
            # fragment shader reading it declares Geometry, as glslang does.
            self.require_capability("Geometry")
        elif builtin_name == "ViewIndex":
            # SPV_KHR_multiview: core in SPIR-V 1.3, an extension before it.
            self.require_capability("MultiView")
            self.require_extension("SPV_KHR_multiview")
        elif builtin_name in {
            "Layer",
            "ViewportIndex",
        } and self.current_stage_builtin_execution_model() in {
            "Vertex",
            "TessellationEvaluation",
        }:
            # SPV_EXT_shader_viewport_index_layer lets these stages write both, as glslang declares them for
            # GL_ARB_shader_viewport_layer_array (SPIR-V section 3.21, BuiltIn).
            self.require_capability("ShaderViewportIndexLayerEXT")
            self.require_extension("SPV_EXT_shader_viewport_index_layer")
        elif builtin_name == "ViewportIndex":
            self.require_capability("MultiViewport")
        elif (
            builtin_name in {"Layer", "PrimitiveId"}
            and self.current_stage_builtin_execution_model() == "Fragment"
        ):
            # A fragment stage reads both through the Geometry capability (SPIR-V section 3.2.20, BuiltIn).
            self.require_capability("Geometry")

    def ensure_builtin_variable(self, name: str) -> Optional[SpirvId]:
        builtin = self.ensure_compute_builtin(name)
        if builtin is not None:
            return builtin
        return self.ensure_stage_builtin(name)

    def ensure_compute_builtin(self, name: str) -> Optional[SpirvId]:
        info = self.compute_builtin_info(name)
        if info is None:
            return None

        cache_name = self.spirv_builtin_cache_name(name)
        if cache_name in self.global_variables:
            builtin_id = self.global_variables[cache_name]
            self.mark_builtin_interface_variable(builtin_id)
            return builtin_id

        type_name, builtin_name, storage_class = info
        if storage_class == "Constant" and not self.module_has_one_workgroup_size:
            # A module-wide WorkgroupSize built-in would resize every entry point; read the entry's own size.
            return entry_workgroup_size_value(
                self, name, self.map_crossgl_type(type_name)
            )
        if builtin_name.startswith("Subgroup") or builtin_name == "NumSubgroups":
            self.require_group_non_uniform(
                "GroupNonUniformBallot" if builtin_name.endswith("Mask") else None
            )
        type_id = self.map_crossgl_type(type_name)
        if storage_class == "Constant":
            builtin_id = self.register_workgroup_size_builtin(
                name, type_id, builtin_name
            )
        else:
            builtin_id = self.register_builtin_input(name, type_id, builtin_name)

        self.global_variables[cache_name] = builtin_id
        self.mark_builtin_interface_variable(builtin_id)
        return builtin_id

    def ensure_stage_builtin(self, name: str) -> Optional[SpirvId]:
        info = self.stage_builtin_info(name)
        if info is None:
            return None

        type_name, builtin_name, storage_spec, execution_models = info
        storage_class = self.resolve_stage_builtin_storage_class(
            name, storage_spec, execution_models
        )
        if storage_class is None:
            return None

        cache_key = self.stage_builtin_cache_key(
            self.spirv_builtin_cache_name(name), storage_spec, storage_class
        )
        if cache_key in self.global_variables:
            builtin_id = self.global_variables[cache_key]
            self.mark_fragment_builtin_execution_mode_if_needed(
                builtin_name, storage_class
            )
            self.mark_builtin_interface_variable(builtin_id)
            return builtin_id

        type_id = self.map_crossgl_type(type_name)
        if storage_class not in {"Input", "Output"}:
            return None

        if self.current_stage_builtin_execution_model() in {
            "TessellationControl",
            "TessellationEvaluation",
        }:
            self.require_capability("Tessellation")

        builtin_id = self.register_builtin_variable(
            name, type_id, builtin_name, storage_class
        )
        self.global_variables[cache_key] = builtin_id
        self.mark_builtin_interface_variable(builtin_id)
        return builtin_id

    def register_builtin_input(
        self, name: str, type_id: SpirvId, builtin_name: str
    ) -> SpirvId:
        return self.register_builtin_variable(name, type_id, builtin_name, "Input")

    def register_builtin_variable(
        self, name: str, type_id: SpirvId, builtin_name: str, storage_class: str
    ) -> SpirvId:
        self.require_stage_builtin_capabilities(builtin_name)
        ptr_type = self.register_pointer_type(type_id, storage_class)
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpVariable %{ptr_type.id} {storage_class}")
        self.emit(f'OpName %{id_value} "{name}"')
        self.decorations.append(f"OpDecorate %{id_value} BuiltIn {builtin_name}")
        if builtin_name in {"TessLevelOuter", "TessLevelInner"}:
            self.decorations.append(f"OpDecorate %{id_value} Patch")
        if (
            storage_class == "Input"
            and self.current_stage_builtin_execution_model() == "Fragment"
            and self.interface_type_requires_flat(type_id)
        ):
            # Vulkan VUID-StandaloneSpirv-Flat-04744 covers built-in inputs too (gl_FragSizeEXT, gl_SampleID).
            self.decorations.append(f"OpDecorate %{id_value} Flat")
        self.mark_fragment_builtin_execution_mode_if_needed(builtin_name, storage_class)

        spirv_id = SpirvId(id_value, ptr_type.type, name)
        self.variable_value_types[id_value] = type_id
        self.builtin_names_by_variable_id[id_value] = builtin_name
        if storage_class == "Input":
            self.inputs.append(spirv_id)
            self.readonly_builtin_pointer_names[id_value] = name
        elif storage_class == "Output":
            self.outputs.append(spirv_id)
        self.builtin_interface_variable_ids.add(id_value)
        return spirv_id

    def register_workgroup_size_builtin(
        self, name: str, type_id: SpirvId, builtin_name: str
    ) -> SpirvId:
        uint_type = self.register_primitive_type("uint")
        components, specialized = workgroup_size_components(
            self, self.current_stage, uint_type
        )

        id_value = self.get_id()
        component_list = " ".join(f"%{component.id}" for component in components)
        opcode = "OpSpecConstantComposite" if specialized else "OpConstantComposite"
        self.emit(f"%{id_value} = {opcode} %{type_id.id} {component_list}")
        self.emit(f'OpName %{id_value} "{name}"')
        self.decorations.append(f"OpDecorate %{id_value} BuiltIn {builtin_name}")

        spirv_id = SpirvId(id_value, type_id.type, name)
        self.value_types[id_value] = type_id
        return spirv_id

    def process_dotted_builtin(self, name: str) -> Optional[SpirvId]:
        if "." not in name:
            return None

        base_name, member_name = name.rsplit(".", 1)
        builtin = self.ensure_builtin_variable(base_name)
        if builtin is None:
            return None

        base_value = self.get_variable_value(builtin)
        member_info = self.vector_member_info(base_value.type.base_type, member_name)
        if member_info is None:
            swizzle_info = self.vector_swizzle_info(
                base_value.type.base_type, member_name
            )
            if swizzle_info is None:
                return None

            indices, _, result_type = swizzle_info
            return self.vector_shuffle(base_value, result_type, indices)

        member_index, member_type = member_info
        return self.composite_extract(base_value, member_type, member_index)

    def register_array_type(
        self, element_type: SpirvId, size: Optional[int] = None
    ) -> SpirvId:
        """Create and register an array type."""
        self.reject_nonpositive_array_length(size)
        key = (element_type.id, size)
        if key in self.array_types:
            return self.array_types[key]

        id_value = self.get_id()

        if size is not None:
            size_const = self.array_length_constant(size)
            self.emit(f"%{id_value} = OpTypeArray %{element_type.id} %{size_const.id}")
        else:
            self.emit(f"%{id_value} = OpTypeRuntimeArray %{element_type.id}")
            if self.is_descriptor_type_id(element_type.id):
                # An unsized array of images, samplers or acceleration structures is a runtime descriptor array.
                self.require_capability("RuntimeDescriptorArray")
                self.require_extension("SPV_EXT_descriptor_indexing")

        type_name = f"array_{element_type.type.base_type}_{size if size else 'rt'}"
        spirv_type = SpirvType(type_name)
        spirv_id = SpirvId(id_value, spirv_type, type_name)
        self.array_types[key] = spirv_id
        if element_type.id in self.cooperative_matrix_containing_type_ids:
            self.cooperative_matrix_containing_type_ids.add(spirv_id.id)
        return spirv_id

    def is_descriptor_type_id(self, type_id) -> bool:
        """Whether ``type_id`` is an image, sampler, sampled image or acceleration structure type."""
        if type_id in self.sampled_image_type_ids.values():
            return True
        return any(
            getattr(registered, "id", None) == type_id
            for registered in (
                *self.resource_types.values(),
                *self.resource_image_types.values(),
            )
        )

    def register_layout_array_type(
        self, element_type: SpirvId, size: Optional[int], layout: str
    ) -> SpirvId:
        """Create a layout-specific array clone for distinct ArrayStride values."""
        self.reject_nonpositive_array_length(size)
        key = (element_type.id, size, layout)
        if key in self.layout_array_types:
            return self.layout_array_types[key]

        id_value = self.get_id()
        if size is not None:
            size_const = self.array_length_constant(size)
            self.emit(f"%{id_value} = OpTypeArray %{element_type.id} %{size_const.id}")
        else:
            self.emit(f"%{id_value} = OpTypeRuntimeArray %{element_type.id}")

        type_name = (
            f"array_{element_type.type.base_type}_{size if size else 'rt'}_{layout}"
        )
        spirv_id = SpirvId(id_value, SpirvType(type_name), type_name)
        self.layout_array_types[key] = spirv_id
        if element_type.id in self.cooperative_matrix_containing_type_ids:
            self.cooperative_matrix_containing_type_ids.add(spirv_id.id)
        return spirv_id

    def reject_nonpositive_array_length(self, size) -> None:
        """Reject fixed array extents that SPIR-V cannot represent."""
        if size is None:
            return
        specialization_length = self.named_specialization_array_length(size)
        if specialization_length is not None:
            resolved_size = self.specialization_constant_default_values.get(
                specialization_length.id
            )
        else:
            resolved_size = evaluate_literal_int_expression(
                size, self.active_literal_int_constants()
            )
        if not isinstance(resolved_size, int) or resolved_size >= 1:
            return
        raise UnsupportedSPIRVFeatureError(
            "non-positive fixed array extent",
            f"SPIR-V fixed arrays require an extent of at least one; got " f"'{size}'.",
            missing_capabilities=("spirv.nonpositive_array_extent",),
        )

    def named_specialization_array_length(self, name: str) -> Optional[SpirvId]:
        """Return an integer specialization constant usable as an array length."""
        if not isinstance(name, str):
            return None
        normalized_name = name.strip()
        if normalized_name in self.current_function_literal_int_constant_shadows:
            return None
        value = self.named_constants.get(normalized_name)
        if value is None or not self.is_specialization_constant_value(value):
            return None
        value_type = self.value_types.get(value.id)
        if value_type is None:
            return None
        primitive_type = self.normalize_primitive_name(value_type.type.base_type)
        return value if primitive_type in self.INTEGER_TYPE_NAMES else None

    def array_length_default_value(self, size, fallback: int = 1) -> int:
        """Return a concrete default used only for static layout bookkeeping."""
        if isinstance(size, int):
            return size
        specialization_length = self.named_specialization_array_length(size)
        if specialization_length is not None:
            value = self.specialization_constant_default_values.get(
                specialization_length.id
            )
            if isinstance(value, int):
                return value
            raise UnsupportedSPIRVFeatureError(
                "specialization array default evaluation",
                "SPIR-V array layout requires a concrete default for "
                f"specialization-dependent extent '{size}'",
                missing_capabilities=("spirv.specialization_array_default",),
            )
        literal_size = evaluate_literal_int_expression(
            size, self.active_literal_int_constants()
        )
        return fallback if literal_size is None else literal_size

    def array_length_constant(self, size) -> SpirvId:
        """Resolve a literal or named specialization-dependent array length."""
        specialization_length = self.named_specialization_array_length(size)
        if specialization_length is not None:
            return specialization_length
        return self.register_constant(size, self.register_primitive_type("int"))

    def type_contains_specialization_sized_array(
        self, type_id: Optional[SpirvId], seen: Optional[set] = None
    ) -> bool:
        """Return whether an aggregate contains a specialization-sized array."""
        if type_id is None:
            return False
        if seen is None:
            seen = set()
        if type_id.id in seen:
            return False
        seen.add(type_id.id)

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            if self.named_specialization_array_length(size) is not None:
                return True
            return self.type_contains_specialization_sized_array(element_type, seen)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is None:
            return False
        return any(
            self.type_contains_specialization_sized_array(member_type, seen)
            for member_type, _ in struct_members
        )

    def register_undefined_value(self, type_id: SpirvId) -> SpirvId:
        """Return a module-scope undefined value for an aggregate type."""
        type_id = self.ensure_registered_type(type_id)
        if type_id.id in self.undefined_values:
            return self.undefined_values[type_id.id]

        id_value = self.get_id()
        self.emit(f"%{id_value} = OpUndef %{type_id.id}")
        undefined = SpirvId(id_value, type_id.type)
        self.value_types[id_value] = type_id
        self.undefined_values[type_id.id] = undefined
        return undefined

    def determine_array_element_type(self, array_id: "SpirvId") -> Optional["SpirvId"]:
        """Determine the element type of an array based on its SpirvId.

        Args:
            array_id: The SpirvId of the array

        Returns:
            SpirvId of the element type, or None if it cannot be determined
        """
        if (
            not array_id
            or not hasattr(array_id, "type")
            or not hasattr(array_id.type, "base_type")
        ):
            return None

        array_type = array_id.type.base_type

        for (element_type_id, _), arr_type_id in self.array_types.items():
            if arr_type_id.type.base_type == array_type:
                return self.find_registered_type_by_id(element_type_id)

        if array_type.startswith("ptr_"):
            base_type = array_type.replace("ptr_", "", 1)
            for (element_type_id, _), arr_type_id in self.array_types.items():
                if arr_type_id.type.base_type == base_type:
                    return self.find_registered_type_by_id(element_type_id)

            match = re.search(r"array_([^_]+)_", base_type)
            if match:
                element_type_name = match.group(1)

                for type_dict in [
                    self.primitive_types,
                    self.vector_types,
                    self.matrix_types,
                ]:
                    for type_id in type_dict.values():
                        if type_id.type.base_type == element_type_name:
                            return type_id

        # A matrix value's subscript is a column; a vector value's is a component.
        matrix_shape = crossgl_matrix_shape(array_type)
        if matrix_shape is not None:
            scalar, _columns, rows = matrix_shape
            return self.register_vector_type(self.register_primitive_type(scalar), rows)
        vector_info = self.vector_component_type_and_count(array_type)
        if vector_info is not None:
            return self.register_primitive_type(vector_info[0])

        # Last resort: Try to parse from type name
        for type_dict in [
            self.primitive_types,
            self.vector_types,
            self.matrix_types,
        ]:
            for type_id in type_dict.values():
                if type_id.type.base_type in array_type:
                    return type_id

        return self.primitive_types["float"]

    def get_function_qualifier(self, func) -> Optional[str]:
        """Return the shader-stage qualifier from old or new function AST shapes."""
        if hasattr(func, "qualifiers") and func.qualifiers:
            return normalize_stage_name(func.qualifiers[0]) if func.qualifiers else None
        if hasattr(func, "qualifier"):
            qualifier = normalize_stage_name(func.qualifier)
            if qualifier in STAGE_QUALIFIER_NAMES:
                return qualifier

        for attr in getattr(func, "attributes", []) or []:
            attr_name = normalize_stage_name(getattr(attr, "name", attr))
            if attr_name in STAGE_QUALIFIER_NAMES:
                return attr_name
        return None

    def stage_key(self, stage_type) -> str:
        """Normalize a stage enum or string to a registry key."""
        if hasattr(stage_type, "value"):
            return stage_type.value
        return str(stage_type).split(".")[-1].lower()

    def spirv_execution_model(self, stage_name: Optional[str]) -> str:
        """Map a CrossGL stage name to a SPIR-V execution model."""
        stage_map = {
            "vertex": "Vertex",
            "fragment": "Fragment",
            "compute": "GLCompute",
            "geometry": "Geometry",
            "tessellation_control": "TessellationControl",
            "tessellation_evaluation": "TessellationEvaluation",
            "mesh": "MeshEXT",
            "task": "TaskEXT",
            "object": "TaskEXT",
            "amplification": "TaskEXT",
            "ray_generation": "RayGenerationKHR",
            "ray_intersection": "IntersectionKHR",
            "ray_closest_hit": "ClosestHitKHR",
            "ray_miss": "MissKHR",
            "ray_any_hit": "AnyHitKHR",
            "ray_callable": "CallableKHR",
        }
        return stage_map.get(stage_name or "fragment", "Fragment")

    def compute_local_size(self, stage) -> Tuple[int, int, int]:
        """Return compute workgroup dimensions from a stage execution config."""
        if stage is None:
            config = self.module_compute_entry_config or {}
        else:
            config = getattr(stage, "execution_config", {}) or {}
        config = effective_workgroup_config(self, config)
        for key in ("local_size", "workgroup_size", "numthreads"):
            value = config.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                return tuple(
                    self.positive_local_size_dimension(
                        value[index], key, "xyz"[index], stage
                    )
                    for index in range(3)
                )

        if not has_specialized_workgroup_size(config):
            # The pipeline sets a specialized size, so only the source's host leaves it unknown.
            require_static_workgroup_size(config)
        return (
            self.positive_local_size_dimension(
                config.get("local_size_x", 1), "local_size_x", "x", stage
            ),
            self.positive_local_size_dimension(
                config.get("local_size_y", 1), "local_size_y", "y", stage
            ),
            self.positive_local_size_dimension(
                config.get("local_size_z", 1), "local_size_z", "z", stage
            ),
        )

    def positive_local_size_dimension(
        self, value, source: str, axis: str, stage
    ) -> int:
        """Coerce a LocalSize dimension to a valid positive SPIR-V literal."""
        invalid = isinstance(value, bool)
        if invalid:
            dimension = None
        else:
            try:
                dimension = int(value)
            except (TypeError, ValueError):
                # A named compile-time constant (`local_size_x = TILE_SIZE`) sizes the workgroup too.
                dimension = evaluate_literal_int_expression(
                    value, self.literal_int_constants
                )

        if dimension is not None and dimension > 0:
            return dimension

        warning_key = (id(stage), source, axis)
        if warning_key not in self.local_size_warning_keys:
            self.local_size_warning_keys.add(warning_key)
            self.refuse_lowering(
                "SPIR-V LocalSize "
                f"{axis} dimension from {source} must be a positive integer "
                "literal"
            )
        return 1

    def stage_attribute_value(self, stage, attribute_name: str):
        """Return the first argument for a stage entry-point attribute."""
        entry_point = getattr(stage, "entry_point", stage)
        for attr in getattr(entry_point, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != attribute_name:
                continue
            arguments = getattr(attr, "arguments", None)
            if arguments is None:
                arguments = getattr(attr, "args", [])
            if arguments:
                return arguments[0]
        return None

    def stage_has_attribute(self, stage, attribute_name: str) -> bool:
        """Return whether a stage entry point has a flag-style attribute."""
        entry_point = getattr(stage, "entry_point", stage)
        for attr in getattr(entry_point, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() == attribute_name:
                return True
        return False

    def stage_attribute_flag_names(self, stage) -> List[str]:
        """Return normalized no-argument stage entry-point attribute names."""
        entry_point = getattr(stage, "entry_point", stage)
        flags = []
        for attr in getattr(entry_point, "attributes", []) or []:
            arguments = getattr(attr, "arguments", None)
            if arguments is None:
                arguments = getattr(attr, "args", [])
            if not arguments:
                flags.append(str(getattr(attr, "name", "")).lower())
        return flags

    def stage_layout_value(self, stage, attribute_name: str):
        """Return the first argument for a stage layout qualifier."""
        for layout in getattr(stage, "layout_qualifiers", []) or []:
            for entry in getattr(layout, "entries", []) or []:
                if str(getattr(entry, "name", "")).lower() != attribute_name:
                    continue
                arguments = getattr(entry, "arguments", None)
                if arguments is None:
                    arguments = getattr(entry, "args", [])
                if arguments:
                    return arguments[0]
        return None

    def stage_layout_direction_value(self, stage, direction: str, attribute_names):
        """Return a stage layout value matching a direction and name set."""
        names = {str(name).lower() for name in attribute_names}
        for layout in getattr(stage, "layout_qualifiers", []) or []:
            if getattr(layout, "direction", None) != direction:
                continue
            for entry in getattr(layout, "entries", []) or []:
                if str(getattr(entry, "name", "")).lower() not in names:
                    continue
                arguments = getattr(entry, "arguments", None)
                if arguments is None:
                    arguments = getattr(entry, "args", [])
                if arguments:
                    return arguments[0]
        return None

    def stage_layout_flag_names(self, stage, direction: Optional[str] = None):
        """Return normalized no-argument stage layout entries."""
        flags = []
        for layout in getattr(stage, "layout_qualifiers", []) or []:
            if (
                direction is not None
                and getattr(layout, "direction", None) != direction
            ):
                continue
            for entry in getattr(layout, "entries", []) or []:
                arguments = getattr(entry, "arguments", None)
                if arguments is None:
                    arguments = getattr(entry, "args", [])
                if not arguments:
                    flags.append(str(getattr(entry, "name", "")).lower())
        return flags

    def normalized_stage_value(self, value) -> Optional[str]:
        """Return a normalized string for stage metadata values."""
        text = self.attribute_value_to_string(value)
        if text is None:
            return None
        return str(text).strip().strip('"').lower()

    def positive_stage_int_value(self, value, context: str, default: int = 1) -> int:
        """Return a positive integer stage metadata value or a safe default."""
        literal = self.literal_int_argument(value)
        if literal is not None and literal > 0:
            return literal
        self.refuse_lowering(f"SPIR-V {context} must be a positive integer literal")
        return default

    def stage_positive_int_metadata(
        self,
        stage,
        context: str,
        attribute_names=(),
        layout_direction: Optional[str] = None,
        layout_names=(),
        default: int = 1,
    ) -> int:
        """Return positive integer metadata from attributes or layouts."""
        for attribute_name in attribute_names:
            value = self.stage_attribute_value(stage, attribute_name)
            if value is not None:
                return self.positive_stage_int_value(value, context, default)

        if layout_direction is not None and layout_names:
            value = self.stage_layout_direction_value(
                stage, layout_direction, layout_names
            )
            if value is not None:
                return self.positive_stage_int_value(value, context, default)

        return default

    def mesh_stage_limit(self, stage, attribute_name: str) -> Optional[int]:
        """Return a literal mesh output limit from function or layout metadata."""
        for value in (
            self.stage_attribute_value(stage, attribute_name),
            self.stage_layout_value(stage, attribute_name),
        ):
            limit = self.literal_int_argument(value)
            if limit is not None:
                return max(0, limit)
        return None

    def mesh_output_array_limits(self, stage) -> Dict[str, int]:
        """The vertex and primitive limits that a mesh entry's output arrays declare.

        ``SetMeshOutputCounts`` may take counts computed at run time, so its arguments bound
        ``OutputVertices`` and ``OutputPrimitivesEXT`` only when they are constants; the ``@vertices``,
        ``@primitives`` and ``@indices`` array sizes are the limits the entry declares, as HLSL's are.
        """
        limits: Dict[str, int] = {}
        entry = getattr(stage, "entry_point", None)
        for param in getattr(entry, "parameters", getattr(entry, "params", [])) or []:
            info = self.mesh_output_parameter_info(param, "MeshEXT")
            if info is None:
                continue
            count = info.get("count")
            if not isinstance(count, int) or isinstance(count, bool):
                continue
            key = "vertices" if info.get("role") == "vertices" else "primitives"
            limits[key] = max(count, limits.get(key, 0))
        return limits

    def mesh_stage_topology_mode(self, stage) -> str:
        """Return the SPIR-V mesh output-topology execution mode."""
        topology = self.stage_attribute_value(stage, "outputtopology")
        topology_name = self.attribute_value_to_string(topology)
        if topology_name is None:
            for layout in getattr(stage, "layout_qualifiers", []) or []:
                if getattr(layout, "direction", None) != "out":
                    continue
                for entry in getattr(layout, "entries", []) or []:
                    entry_name = str(getattr(entry, "name", "")).lower()
                    if entry_name in {"point", "points", "line", "lines"}:
                        topology_name = entry_name
                        break
                    if entry_name in {"triangle", "triangles"}:
                        topology_name = entry_name
                        break
                if topology_name is not None:
                    break

        topology_modes = {
            "point": "OutputPoints",
            "points": "OutputPoints",
            "line": "OutputLinesEXT",
            "lines": "OutputLinesEXT",
            "triangle": "OutputTrianglesEXT",
            "triangles": "OutputTrianglesEXT",
        }
        normalized = str(topology_name or "triangle").lower()
        if normalized not in topology_modes:
            raise VulkanRefusal(
                "SPIR-V mesh stage outputtopology must be point, line, or "
                f"triangle: {topology_name}"
            )
        return topology_modes[normalized]

    def geometry_input_mode(self, stage) -> str:
        """Return the SPIR-V geometry input primitive execution mode."""
        mode_map = {
            "point": "InputPoints",
            "points": "InputPoints",
            "line": "InputLines",
            "lines": "InputLines",
            "line_adjacency": "InputLinesAdjacency",
            "lines_adjacency": "InputLinesAdjacency",
            "lineadj": "InputLinesAdjacency",
            "triangle": "Triangles",
            "triangles": "Triangles",
            "tri": "Triangles",
            "triangle_adjacency": "InputTrianglesAdjacency",
            "triangles_adjacency": "InputTrianglesAdjacency",
            "triangleadj": "InputTrianglesAdjacency",
        }
        for flag in self.stage_attribute_flag_names(stage):
            if flag in mode_map:
                return mode_map[flag]
        for flag in self.stage_layout_flag_names(stage, "in"):
            if flag in mode_map:
                return mode_map[flag]
        # A Geometry entry point declares exactly one input primitive mode (SPIR-V specification, Execution
        # Mode); writing InputPoints for a stage that states none would read its vertices as points.
        self.refuse_lowering(
            "SPIR-V geometry stage states no input primitive: it needs points, lines, lines_adjacency, "
            "triangles or triangles_adjacency as a GLSL input layout or a stage attribute"
        )
        return "InputPoints"

    def geometry_output_mode(self, stage) -> str:
        """Return the SPIR-V geometry output primitive execution mode."""
        mode_map = {
            "point": "OutputPoints",
            "points": "OutputPoints",
            "line": "OutputLineStrip",
            "lines": "OutputLineStrip",
            "line_strip": "OutputLineStrip",
            "triangle": "OutputTriangleStrip",
            "triangles": "OutputTriangleStrip",
            "tri": "OutputTriangleStrip",
            "triangle_strip": "OutputTriangleStrip",
        }
        value = self.stage_attribute_value(stage, "outputtopology")
        normalized = self.normalized_stage_value(value)
        if normalized in mode_map:
            return mode_map[normalized]
        if normalized is not None:
            self.refuse_lowering(
                "SPIR-V geometry outputtopology must be point, "
                f"line_strip, or triangle_strip: {normalized}"
            )
        for flag in self.stage_layout_flag_names(stage, "out"):
            if flag in mode_map:
                return mode_map[flag]
        # A Geometry entry point declares exactly one output primitive mode (SPIR-V specification, Execution
        # Mode); writing OutputPoints for a stage that states none would draw its strips as points.
        self.refuse_lowering(
            "SPIR-V geometry stage states no output primitive: it needs points, line_strip or triangle_strip "
            "as a GLSL output layout or an outputtopology attribute"
        )
        return "OutputPoints"

    def emit_geometry_execution_modes(self, function_id: SpirvId, stage):
        """Emit geometry capability and execution modes."""
        self.require_capability("Geometry")
        vertex_count_names = ("max_vertices", "maxvertexcount")
        max_vertices = self.stage_positive_int_metadata(
            stage,
            "geometry OutputVertices",
            attribute_names=vertex_count_names,
            layout_direction="out",
            layout_names=vertex_count_names,
        )
        if all(
            self.stage_attribute_value(stage, name) is None
            for name in vertex_count_names
        ) and (
            self.stage_layout_direction_value(stage, "out", vertex_count_names) is None
        ):
            # OutputVertices bounds the vertices the stage emits, and a vertex past it is discarded; a stage that
            # states no bound cannot be given one.
            self.refuse_lowering(
                "SPIR-V geometry stage states no max_vertices (or maxvertexcount), which OutputVertices needs"
            )
        invocations = self.stage_positive_int_metadata(
            stage,
            "geometry Invocations",
            attribute_names=("invocations",),
            layout_direction="in",
            layout_names=("invocations",),
        )
        self.emit(
            f"OpExecutionMode %{function_id.id} {self.geometry_input_mode(stage)}"
        )
        self.emit(
            f"OpExecutionMode %{function_id.id} {self.geometry_output_mode(stage)}"
        )
        self.emit(f"OpExecutionMode %{function_id.id} OutputVertices {max_vertices}")
        if invocations != 1:
            self.emit(f"OpExecutionMode %{function_id.id} Invocations {invocations}")

    def tessellation_control_output_vertices(self, stage) -> int:
        """Return tessellation-control output control-point count."""
        names = ("outputcontrolpoints", "vertices")
        if all(self.stage_attribute_value(stage, name) is None for name in names) and (
            self.stage_layout_direction_value(stage, "out", names) is None
        ):
            # The output patch size sizes gl_out and OutputVertices. GLSL requires layout(vertices) (4.60
            # 4.4.2.1) and HLSL outputcontrolpoints, so a stage that states none cannot be given one.
            self.refuse_lowering(
                "SPIR-V tessellation control stage states no output patch size (layout(vertices) or "
                "outputcontrolpoints), which OutputVertices needs"
            )
        return self.stage_positive_int_metadata(
            stage,
            "tessellation_control OutputVertices",
            attribute_names=names,
            layout_direction="out",
            layout_names=("vertices", "outputcontrolpoints"),
        )

    def tessellation_metadata_sources(self, stage):
        """Return evaluation metadata sources, preferring evaluation metadata."""
        sources = [stage]
        if (
            self.tessellation_control_stage is not None
            and self.tessellation_control_stage is not stage
        ):
            sources.append(self.tessellation_control_stage)
        return sources

    def tessellation_domain_mode(self, stage) -> Optional[str]:
        """Return the tessellation-evaluation domain execution mode."""
        mode_map = {
            "tri": "Triangles",
            "triangle": "Triangles",
            "triangles": "Triangles",
            "triangle_cw": "Triangles",
            "triangle_ccw": "Triangles",
            "quad": "Quads",
            "quads": "Quads",
            "isoline": "Isolines",
            "isolines": "Isolines",
        }
        for source in self.tessellation_metadata_sources(stage):
            for attribute_name in ("domain", "outputtopology"):
                value = self.stage_attribute_value(source, attribute_name)
                normalized = self.normalized_stage_value(value)
                if normalized in mode_map:
                    return mode_map[normalized]
                if attribute_name == "domain" and normalized is not None:
                    self.refuse_lowering(
                        "SPIR-V tessellation domain must be "
                        f"triangle, quads, or isolines: {normalized}"
                    )
            for flag in self.stage_layout_flag_names(source, "in"):
                if flag in mode_map:
                    return mode_map[flag]
        # Left to a control stage outside the program; see spirv_tessellation_domain.
        return None

    def tessellation_spacing_mode(self, stage) -> Optional[str]:
        """Return the tessellation-evaluation spacing execution mode."""
        mode_map = {
            "integer": "SpacingEqual",
            "equal": "SpacingEqual",
            "equal_spacing": "SpacingEqual",
            "fractional_even": "SpacingFractionalEven",
            "fractional_even_spacing": "SpacingFractionalEven",
            "fractional_odd": "SpacingFractionalOdd",
            "fractional_odd_spacing": "SpacingFractionalOdd",
        }
        for source in self.tessellation_metadata_sources(stage):
            value = self.stage_attribute_value(source, "partitioning")
            normalized = self.normalized_stage_value(value)
            if normalized in mode_map:
                return mode_map[normalized]
            if normalized is not None:
                self.refuse_lowering(
                    "SPIR-V tessellation partitioning must be "
                    "integer, fractional_even, or fractional_odd: "
                    f"{normalized}"
                )
            for flag in self.stage_layout_flag_names(source, "in"):
                if flag in mode_map:
                    return mode_map[flag]
        return "SpacingEqual" if evaluation_takes_glsl_defaults(self, stage) else None

    def tessellation_vertex_order_mode(self, stage) -> Optional[str]:
        """Return the tessellation-evaluation vertex-order execution mode."""
        mode_map = {
            "cw": "VertexOrderCw",
            "triangle_cw": "VertexOrderCw",
            "ccw": "VertexOrderCcw",
            "triangle_ccw": "VertexOrderCcw",
        }
        for source in self.tessellation_metadata_sources(stage):
            value = self.stage_attribute_value(source, "outputtopology")
            normalized = self.normalized_stage_value(value)
            if normalized in mode_map:
                return mode_map[normalized]
            for flag in self.stage_attribute_flag_names(source):
                if flag in mode_map:
                    return mode_map[flag]
            for flag in self.stage_layout_flag_names(source, "in"):
                if flag in mode_map:
                    return mode_map[flag]
        return "VertexOrderCcw" if evaluation_takes_glsl_defaults(self, stage) else None

    def tessellation_modes_follow_hlsl(self, stage) -> bool:
        """Whether ``stage`` declares its domain as an HLSL-style ``@domain`` attribute.

        HLSL declares the spacing and winding on the hull shader only, so an HLSL-style stage that names neither
        leaves them to the control stage, as DXC does. A GLSL evaluation shader that omits them means
        ``equal_spacing`` and ``ccw`` (GLSL 4.60 section 4.4.1.2), which SPIR-V must state.
        """
        return self.stage_attribute_value(stage, "domain") is not None

    def tessellation_point_mode(self, stage) -> bool:
        """Return whether tessellation evaluation should emit PointMode."""
        for source in self.tessellation_metadata_sources(stage):
            topology = self.normalized_stage_value(
                self.stage_attribute_value(source, "outputtopology")
            )
            if topology in {"point", "points"}:
                return True
            if self.stage_has_attribute(source, "point_mode"):
                return True
            if "point_mode" in self.stage_layout_flag_names(source, "in"):
                return True
        return False

    def tessellation_control_stated_modes(self, stage) -> List[str]:
        """Return the spacing, winding and point mode a control stage without ``@domain`` states itself.

        SPIR-V allows these modes on either tessellation stage; the evaluation stage then states the domain
        and anything else, so nothing is defaulted here.
        """
        modes = []
        if self.stage_attribute_value(stage, "partitioning") is not None:
            modes.append(self.tessellation_spacing_mode(stage))
        topology = self.normalized_stage_value(
            self.stage_attribute_value(stage, "outputtopology")
        )
        if topology in {"cw", "ccw", "triangle_cw", "triangle_ccw"}:
            modes.append(self.tessellation_vertex_order_mode(stage))
        if self.tessellation_point_mode(stage):
            modes.append("PointMode")
        return modes

    def emit_tessellation_execution_modes(
        self, execution_model: str, function_id: SpirvId, stage
    ):
        """Emit tessellation capability and execution modes."""
        self.require_capability("Tessellation")
        if execution_model == "TessellationControl":
            self.emit(
                f"OpExecutionMode %{function_id.id} OutputVertices "
                f"{self.tessellation_control_output_vertices(stage)}"
            )
            if not self.tessellation_modes_follow_hlsl(stage):
                for mode in self.tessellation_control_stated_modes(stage):
                    self.emit(f"OpExecutionMode %{function_id.id} {mode}")
                return
            # An HLSL hull declares the domain, spacing and winding; DXC states them on the control stage too.

        modes = [
            self.tessellation_domain_mode(stage),
            self.tessellation_spacing_mode(stage),
            self.tessellation_vertex_order_mode(stage),
        ]
        refuse_program_without_domain(self, stage, modes[0])
        if self.tessellation_point_mode(stage):
            modes.append("PointMode")
        for mode in modes:
            if mode is not None:
                self.emit(f"OpExecutionMode %{function_id.id} {mode}")

    def mesh_stage_output_limits(self, function_id: SpirvId, stage) -> Tuple[int, int]:
        """Return OutputVertices and OutputPrimitivesEXT execution-mode limits."""
        observed_vertices, observed_primitives = (
            self.mesh_output_counts_by_function.get(function_id.id, (None, None))
        )
        return self.mesh_stage_limits_with_observed_counts(
            stage, observed_vertices, observed_primitives
        )
