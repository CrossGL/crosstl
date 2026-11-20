"""SPIR-V assembly modules in the SPIR-V parser: decoration groups, metadata-only and linkage surfaces, entry points and workgroup sizes."""


class VulkanAssemblyModuleMixin:
    """Assembly module members of ``VulkanParser``."""

    def spirv_apply_decoration_groups(
        self, decorations, member_decorations, group_decorates, group_member_decorates
    ):
        for group_id, targets in group_decorates:
            group_decorations = decorations.get(group_id, [])
            if not group_decorations:
                continue
            for target in targets:
                decorations.setdefault(target, []).extend(
                    (decoration, list(operands))
                    for decoration, operands in group_decorations
                )

        for group_id, target_members in group_member_decorates:
            group_decorations = decorations.get(group_id, [])
            if not group_decorations:
                continue
            for index in range(0, len(target_members) - 1, 2):
                target = target_members[index]
                member = target_members[index + 1]
                member_decorations.setdefault(target, []).extend(
                    (member, decoration, list(operands))
                    for decoration, operands in group_decorations
                )

    def spirv_assembly_has_metadata_only_surface(self, types):
        metadata_type_kinds = {
            "acceleration_structure",
            "array",
            "cooperative_matrix",
            "image",
            "named_barrier",
            "pointer",
            "ray_query",
            "runtime_array",
            "sampled_image",
            "sampler",
        }
        return any(
            type_info.get("kind") in metadata_type_kinds for type_info in types.values()
        )

    def spirv_assembly_has_linkage_metadata_surface(
        self, capabilities, types, extended_instruction_imports
    ):
        if "Linkage" not in capabilities:
            return False
        return bool(types or extended_instruction_imports)

    def spirv_assembly_has_entry_point_metadata_surface(self, entry_points):
        return bool(entry_points)

    def spirv_entry_workgroup_size(self, function_id, function_entry_points):
        workgroup_sizes = getattr(self, "spirv_workgroup_sizes", None)
        for entry_point in function_entry_points:
            if workgroup_sizes is None:
                break
            if entry_point["execution_model"] in self.SPIRV_WORKGROUP_EXECUTION_MODELS:
                return workgroup_sizes.entry_dimensions(
                    function_id, entry_point["name"]
                )
        return None
