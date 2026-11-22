"""Resource registers, spaces and Vulkan bindings in the Slang generator, plus constant buffers and global-variable signatures."""

from ..refusals import SlangRefusal
from .array_utils import evaluate_literal_int_expression
from .push_constant_blocks import (
    slang_push_constant_cbuffer,
    slang_push_constant_declaration,
)
from .slang_declaration_modifiers import (
    slang_member_modifier_prefix,
    slang_packoffset_suffix,
)
from .slang_hlsl_cbuffer_layout import slang_hlsl_cbuffer_packoffsets


class SlangResourceBindingsMixin:
    """Resource-binding members of ``SlangCodeGen``."""

    def binding_index_value(self, value, prefixes=()):
        if hasattr(value, "value") and value.value is not None:
            raw_value = value.value
        elif hasattr(value, "name") and value.name is not None:
            raw_value = value.name
        else:
            raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        for prefix in prefixes:
            if raw_value.startswith(prefix) and raw_value[len(prefix) :].isdigit():
                return int(raw_value[len(prefix) :])
        return None

    def binding_expr_value(self, value, prefixes=()):
        text = self.attribute_value_to_string(value)
        if text is None:
            return None
        text = str(text).strip()
        lower_text = text.lower()
        for prefix in prefixes:
            if lower_text.startswith(prefix) and text[len(prefix) :]:
                return text[len(prefix) :]
        return text

    def register_space_index(self, value):
        if hasattr(value, "value") and value.value is not None:
            raw_value = value.value
        elif hasattr(value, "name") and value.name is not None:
            raw_value = value.name
        else:
            raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        if raw_value.startswith("space") and raw_value[5:].isdigit():
            return int(raw_value[5:])
        return None

    def register_space_expr(self, value):
        text = self.attribute_value_to_string(value)
        if text is None:
            return None
        text = str(text).strip()
        lower_text = text.lower()
        if lower_text.startswith("space") and text[5:]:
            return text[5:]
        return text

    def explicit_slang_resource_binding(self, node):
        binding = None
        binding_expr = None
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = getattr(attr, "arguments", []) or []
            if attr_name != "binding" or not arguments:
                continue
            binding = self.binding_index_value(arguments[0])
            binding_expr = self.binding_expr_value(arguments[0])
            break
        return binding, binding_expr

    def explicit_slang_resource_set(self, node):
        descriptor_set = None
        descriptor_set_expr = None
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = getattr(attr, "arguments", []) or []
            if attr_name not in {"set", "group", "space"} or not arguments:
                continue
            descriptor_set = self.binding_index_value(arguments[0])
            descriptor_set_expr = self.binding_expr_value(arguments[0])
            break
        return descriptor_set, descriptor_set_expr

    def explicit_slang_register(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = getattr(attr, "arguments", []) or []
            if attr_name != "register" or not arguments:
                continue
            register_arg = self.attribute_value_to_string(arguments[0])
            if register_arg is None:
                return None, None, None, None, None
            register_arg = str(register_arg).strip()
            register_prefix = ""
            for char in register_arg:
                if not char.isalpha():
                    break
                register_prefix += char
            register_prefix = register_prefix.lower() or None
            binding = self.binding_index_value(arguments[0], (register_prefix or "",))
            binding_expr = self.binding_expr_value(
                arguments[0], (register_prefix or "",)
            )
            space = None
            space_expr = None
            for argument in arguments[1:]:
                space = self.register_space_index(argument)
                space_expr = self.register_space_expr(argument)
                if space_expr is not None:
                    break
            return register_prefix, binding, binding_expr, space, space_expr
        return None, None, None, None, None

    def slang_register_prefix_for_type(self, type_name, node=None, forced_prefix=None):
        if forced_prefix:
            return forced_prefix
        mapped_type = self.map_resource_type_with_format(type_name, node)
        base_type = self.resource_base_type(mapped_type)
        if not isinstance(base_type, str):
            return None
        if base_type in {"SamplerState", "SamplerComparisonState"}:
            return "s"
        if base_type.startswith(
            (
                "RWTexture",
                "RWBuffer",
                "RWStructuredBuffer",
                "AppendStructuredBuffer",
                "ConsumeStructuredBuffer",
                "RWByteAddressBuffer",
            )
        ):
            return "u"
        if base_type.startswith(
            (
                "Sampler",
                "Texture",
                "StructuredBuffer",
                "ByteAddressBuffer",
                "RaytracingAccelerationStructure",
            )
        ):
            return "t"
        if base_type.startswith("ConstantBuffer"):
            return "b"
        return None

    def slang_resource_array_count(self, node, type_name):
        if self.slang_resource_array_is_unbounded(node, type_name):
            return None
        count = self.slang_resource_array_count_from_type_node(
            getattr(node, "var_type", None)
        )
        if count is None:
            count = self.slang_resource_array_count_from_type_node(
                getattr(node, "param_type", None)
            )
        if count is None:
            count = self.slang_resource_array_count_from_type_name(type_name)
        return max(count or 1, 1)

    def slang_resource_array_is_unbounded(self, node, type_name):
        for type_node in (
            getattr(node, "var_type", None),
            getattr(node, "param_type", None),
        ):
            if self.slang_resource_array_is_unbounded_from_type_node(type_node):
                return True
        return self.slang_resource_array_is_unbounded_from_type_name(type_name)

    def slang_resource_array_is_unbounded_from_type_node(self, type_node):
        current = type_node
        while current is not None and current.__class__.__name__ == "ArrayType":
            if getattr(current, "size", None) is None:
                return True
            current = getattr(current, "element_type", None)
        return False

    def slang_resource_array_is_unbounded_from_type_name(self, type_name):
        if not isinstance(type_name, str) or "[" not in type_name:
            return False

        index = 0
        while True:
            start = type_name.find("[", index)
            if start < 0:
                return False
            end = type_name.find("]", start + 1)
            if end < 0:
                return False
            if not type_name[start + 1 : end].strip():
                return True
            index = end + 1

    def slang_resource_array_count_from_type_node(self, type_node):
        if type_node is None:
            return None
        if type_node.__class__.__name__ != "ArrayType":
            return None

        total = 1
        current = type_node
        saw_array = False
        while current is not None and current.__class__.__name__ == "ArrayType":
            saw_array = True
            size = evaluate_literal_int_expression(getattr(current, "size", None))
            if size is None or size <= 0:
                return None
            total *= size
            current = getattr(current, "element_type", None)
        return total if saw_array else None

    def slang_resource_array_count_from_type_name(self, type_name):
        if not isinstance(type_name, str) or "[" not in type_name:
            return None

        total = 1
        index = 0
        saw_array = False
        while True:
            start = type_name.find("[", index)
            if start < 0:
                break
            end = type_name.find("]", start + 1)
            if end < 0:
                return None
            saw_array = True
            size_text = type_name[start + 1 : end].strip()
            if not size_text or not size_text.isdigit():
                return None
            size = int(size_text)
            if size <= 0:
                return None
            total *= size
            index = end + 1
        return total if saw_array else None

    def slang_resource_register_space_key(self, descriptor_set, register_space):
        if descriptor_set is not None:
            return descriptor_set
        if register_space is not None:
            return register_space
        return 0

    def slang_source_bound_sampler_uses_remappable_register(
        self, register_prefix, explicit_binding, register_binding, resource_count
    ):
        return (
            register_prefix == "s"
            and explicit_binding is not None
            and register_binding is None
            and resource_count is not None
        )

    def reserve_explicit_slang_resource_declarations(self, node):
        for declaration, type_name, forced_prefix in self.slang_resource_declarations(
            node
        ):
            self.reserve_explicit_slang_resource_declaration(
                declaration, type_name, forced_prefix
            )

    def slang_resource_declarations(self, node):
        declarations = []

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return

            for global_var in getattr(current, "global_variables", []) or []:
                type_name = self.slang_resource_declaration_type(global_var)
                declarations.append((global_var, type_name, None))

            for cbuffer in getattr(current, "cbuffers", []) or []:
                declarations.append((cbuffer, "ConstantBuffer", "b"))

            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage_type, stage in stages.items():
                    stage_name = self.get_stage_name(stage_type)
                    for local_var in self.slang_stage_global_local_variables(
                        stage_name, getattr(stage, "local_variables", [])
                    ):
                        type_name = self.slang_resource_declaration_type(local_var)
                        declarations.append((local_var, type_name, None))

        collect(node)
        return declarations

    def reserve_explicit_slang_resource_declaration(
        self, node, type_name, forced_register_prefix=None
    ):
        explicit_binding, _explicit_binding_expr = self.explicit_slang_resource_binding(
            node
        )
        explicit_set, _explicit_set_expr = self.explicit_slang_resource_set(node)
        (
            register_prefix,
            register_binding,
            _register_binding_expr,
            register_space,
            _register_space_expr,
        ) = self.explicit_slang_register(node)
        self.validate_slang_resource_binding_consistency(
            node, explicit_binding, register_binding, explicit_set, register_space
        )
        binding = explicit_binding if explicit_binding is not None else register_binding
        resource_count = self.slang_resource_array_count(node, type_name)

        prefix = self.slang_register_prefix_for_type(
            type_name, node, register_prefix or forced_register_prefix
        )
        space_key = self.slang_resource_register_space_key(explicit_set, register_space)
        if self.slang_source_bound_sampler_uses_remappable_register(
            prefix, explicit_binding, register_binding, resource_count
        ):
            descriptor_set = (
                explicit_set if explicit_set is not None else register_space
            )
            descriptor_set = descriptor_set or 0
            self.reserve_slang_vk_binding_range(
                descriptor_set,
                explicit_binding,
                resource_count,
                self.slang_resource_name(node),
            )
            assigned_binding = self.next_available_slang_resource_register(
                prefix, space_key, resource_count, preferred=explicit_binding
            )
            if assigned_binding is None:
                return
            self.reserve_slang_resource_register_range(
                prefix,
                assigned_binding,
                resource_count,
                self.slang_resource_name(node),
                space_key,
            )
            self.advance_slang_resource_register(
                prefix, space_key, assigned_binding, resource_count
            )
            self.slang_source_sampler_register_assignments[id(node)] = (
                assigned_binding,
                space_key,
            )
            return

        if binding is not None and prefix is not None:
            self.reserve_slang_resource_register_range(
                prefix,
                binding,
                resource_count,
                self.slang_resource_name(node),
                space_key,
            )
        if explicit_binding is not None:
            descriptor_set = (
                explicit_set if explicit_set is not None else register_space
            )
            descriptor_set = descriptor_set or 0
            self.reserve_slang_vk_binding_range(
                descriptor_set,
                explicit_binding,
                resource_count,
                self.slang_resource_name(node),
            )

    def slang_resource_name(self, node):
        return getattr(node, "name", getattr(node, "variable_name", "<anonymous>"))

    def next_available_slang_resource_register(
        self, register_prefix, space, count, preferred=None
    ):
        ranges = self.slang_used_resource_registers.get((register_prefix, space), [])
        candidates = []
        if preferred is not None:
            candidates.append(preferred)
        candidates.append(
            self.slang_resource_register_cursors.get((register_prefix, space), 0)
        )

        for candidate in candidates:
            binding = candidate
            while True:
                end = None if count is None else binding + max(count, 1) - 1
                conflict_end = None
                for used_start, used_end, _used_name in ranges:
                    if not self.slang_resource_register_ranges_overlap(
                        binding, end, used_start, used_end
                    ):
                        continue
                    if used_end is None:
                        return None
                    conflict_end = (
                        used_end
                        if conflict_end is None
                        else max(conflict_end, used_end)
                    )
                if conflict_end is None:
                    return binding
                binding = conflict_end + 1

    def assigned_slang_source_sampler_register(self, node, space_key):
        assignment = self.slang_source_sampler_register_assignments.get(id(node))
        if assignment is None:
            return None
        binding, assignment_space = assignment
        if assignment_space != space_key:
            return None
        return binding

    def next_available_slang_resource_register_space(
        self, register_prefix, start_space, count
    ):
        space = start_space if isinstance(start_space, int) else 0
        while True:
            binding = self.next_available_slang_resource_register(
                register_prefix, space, count
            )
            if binding is not None:
                return space, binding
            space += 1

    def next_available_slang_vk_binding(self, descriptor_set, count, preferred=None):
        count = 1
        ranges = self.slang_used_vk_bindings.get(descriptor_set, [])
        candidates = []
        if preferred is not None:
            candidates.append(preferred)
        candidates.append(self.slang_vk_binding_cursors.get(descriptor_set, 0))

        for candidate in candidates:
            binding = candidate
            while True:
                end = None if count is None else binding + max(count, 1) - 1
                conflict_end = None
                for used_start, used_end, _used_name in ranges:
                    if not self.slang_resource_register_ranges_overlap(
                        binding, end, used_start, used_end
                    ):
                        continue
                    if used_end is None:
                        break
                    conflict_end = (
                        used_end
                        if conflict_end is None
                        else max(conflict_end, used_end)
                    )
                else:
                    if conflict_end is None:
                        return binding
                    binding = conflict_end + 1
                    continue
                break
        return None

    def advance_slang_resource_register(self, register_prefix, space, start, count):
        count = 1 if count is None else max(count, 1)
        key = (register_prefix, space)
        self.slang_resource_register_cursors[key] = max(
            self.slang_resource_register_cursors.get(key, 0), start + count
        )

    def advance_slang_vk_binding(self, descriptor_set, start, count):
        count = 1
        self.slang_vk_binding_cursors[descriptor_set] = max(
            self.slang_vk_binding_cursors.get(descriptor_set, 0),
            start + count,
        )

    def reserve_slang_resource_register_range(
        self, register_prefix, start, count, name, space=0
    ):
        end = None if count is None else start + max(count, 1) - 1
        namespace = (register_prefix, space)
        ranges = self.slang_used_resource_registers.setdefault(namespace, [])
        for used_start, used_end, used_name in ranges:
            if not self.slang_resource_register_ranges_overlap(
                start, end, used_start, used_end
            ):
                continue
            if used_start == start and used_end == end and used_name == name:
                return
            raise SlangRefusal(
                f"Conflicting Slang resource binding for '{name}': "
                f"{self.slang_resource_range_label(register_prefix, start, end, space)} "
                f"overlaps '{used_name}' "
                f"{self.slang_resource_range_label(register_prefix, used_start, used_end, space)}"
            )
        ranges.append((start, end, name))

    def reserve_slang_vk_binding_range(self, descriptor_set, start, count, name):
        count = 1
        end = start + count - 1
        ranges = self.slang_used_vk_bindings.setdefault(descriptor_set, [])
        for used_start, used_end, used_name in ranges:
            if not self.slang_resource_register_ranges_overlap(
                start, end, used_start, used_end
            ):
                continue
            if used_start == start and used_end == end and used_name == name:
                return
            raise SlangRefusal(
                f"Conflicting Slang Vulkan resource binding for '{name}': "
                f"{self.slang_vk_binding_range_label(start, end, descriptor_set)} "
                f"overlaps '{used_name}' "
                f"{self.slang_vk_binding_range_label(used_start, used_end, descriptor_set)}"
            )
        ranges.append((start, end, name))

    def slang_vk_binding_range_label(self, start, end, descriptor_set):
        if end is None:
            label = f"binding {start}+"
        elif start == end:
            label = f"binding {start}"
        else:
            label = f"bindings {start}-{end}"
        return f"{label}, set {descriptor_set}"

    def slang_resource_register_ranges_overlap(self, start, end, used_start, used_end):
        end_value = float("inf") if end is None else end
        used_end_value = float("inf") if used_end is None else used_end
        return start <= used_end_value and used_start <= end_value

    def slang_resource_range_label(self, register_prefix, start, end, space=0):
        if end is None:
            label = f"{register_prefix}{start}+"
        elif start == end:
            label = f"{register_prefix}{start}"
        else:
            label = f"{register_prefix}{start}-{register_prefix}{end}"
        if space:
            return f"{label}, space{space}"
        return label

    def validate_slang_resource_binding_consistency(
        self, node, explicit_binding, register_binding, explicit_set, register_space
    ):
        name = getattr(node, "name", getattr(node, "variable_name", "<anonymous>"))
        if (
            explicit_binding is not None
            and register_binding is not None
            and explicit_binding != register_binding
        ):
            raise SlangRefusal(
                "Conflicting Slang resource binding metadata for "
                f"'{name}': binding {explicit_binding} does not match "
                f"register binding {register_binding}"
            )
        if (
            explicit_set is not None
            and register_space is not None
            and explicit_set != register_space
        ):
            raise SlangRefusal(
                "Conflicting Slang resource set metadata for "
                f"'{name}': set {explicit_set} does not match "
                f"register space{register_space}"
            )

    def slang_resource_binding_decorations(
        self, node, type_name, forced_register_prefix=None, auto_assign=False
    ):
        explicit_binding, explicit_binding_expr = self.explicit_slang_resource_binding(
            node
        )
        explicit_set, explicit_set_expr = self.explicit_slang_resource_set(node)
        (
            register_prefix,
            register_binding,
            register_binding_expr,
            register_space,
            register_space_expr,
        ) = self.explicit_slang_register(node)
        self.validate_slang_resource_binding_consistency(
            node, explicit_binding, register_binding, explicit_set, register_space
        )

        if register_prefix is None:
            register_prefix = self.slang_register_prefix_for_type(
                type_name, node, forced_register_prefix
            )
        if register_prefix is None:
            return "", ""

        descriptor_set = explicit_set if explicit_set is not None else register_space
        descriptor_set_expr = explicit_set_expr or register_space_expr
        has_explicit_vk_set = explicit_set is not None or register_space is not None
        space_key = self.slang_resource_register_space_key(
            descriptor_set, register_space
        )
        resource_count = self.slang_resource_array_count(node, type_name)
        source_bound_sampler_register = (
            self.slang_source_bound_sampler_uses_remappable_register(
                register_prefix, explicit_binding, register_binding, resource_count
            )
        )

        assigned_register = None
        if source_bound_sampler_register:
            assigned_register = self.assigned_slang_source_sampler_register(
                node, space_key
            )
            if assigned_register is not None:
                register_binding = assigned_register
                register_binding_expr = str(assigned_register)

        if (
            register_binding is None
            and explicit_binding is not None
            and not source_bound_sampler_register
        ):
            register_binding = explicit_binding
            register_binding_expr = explicit_binding_expr

        if register_binding is None and register_binding_expr is None and auto_assign:
            preferred_register_binding = (
                explicit_binding if source_bound_sampler_register else None
            )
            register_binding = self.next_available_slang_resource_register(
                register_prefix,
                space_key,
                resource_count,
                preferred=preferred_register_binding,
            )
            if register_binding is None:
                if descriptor_set is not None or descriptor_set_expr is not None:
                    raise SlangRefusal(
                        "Unable to assign Slang resource binding for "
                        f"'{self.slang_resource_name(node)}' in space "
                        f"{descriptor_set_expr or descriptor_set}: "
                        "unbounded resource array occupies the remaining range"
                    )
                space_key, register_binding = (
                    self.next_available_slang_resource_register_space(
                        register_prefix, space_key, resource_count
                    )
                )
                if space_key:
                    descriptor_set = space_key
                    descriptor_set_expr = str(space_key)
            register_binding_expr = str(register_binding)

        if descriptor_set_expr is None and descriptor_set is not None:
            descriptor_set_expr = str(descriptor_set)

        vk_binding = explicit_binding
        vk_binding_expr = explicit_binding_expr
        vk_descriptor_set = descriptor_set if descriptor_set is not None else 0
        vk_descriptor_set_expr = descriptor_set_expr or str(vk_descriptor_set)
        if (
            not auto_assign
            and vk_binding is None
            and vk_binding_expr is None
            and register_binding_expr is not None
        ):
            vk_binding = register_binding if isinstance(register_binding, int) else None
            vk_binding_expr = register_binding_expr
        if vk_binding is None and vk_binding_expr is None and auto_assign:
            preferred_vk_binding = (
                register_binding if isinstance(register_binding, int) else None
            )
            vk_binding = self.next_available_slang_vk_binding(
                vk_descriptor_set,
                resource_count,
                preferred=preferred_vk_binding,
            )
            if vk_binding is None:
                if has_explicit_vk_set:
                    raise SlangRefusal(
                        "Unable to assign Slang Vulkan resource binding for "
                        f"'{self.slang_resource_name(node)}' in set "
                        f"{descriptor_set_expr or descriptor_set}: "
                        "unbounded resource array occupies the remaining range"
                    )
                vk_descriptor_set = 0
                while vk_binding is None:
                    vk_descriptor_set += 1
                    vk_binding = self.next_available_slang_vk_binding(
                        vk_descriptor_set,
                        resource_count,
                        preferred=preferred_vk_binding,
                    )
                vk_descriptor_set_expr = str(vk_descriptor_set)
            vk_binding_expr = str(vk_binding)

        prefix = ""
        if vk_binding_expr is not None:
            prefix = f"[[vk::binding({vk_binding_expr}, {vk_descriptor_set_expr})]] "

        if register_binding is None and explicit_binding is not None:
            register_binding = explicit_binding
        if register_binding is None and register_binding_expr is not None:
            register_binding = register_binding_expr
        register_space_suffix = ""
        if descriptor_set is not None:
            register_space_suffix = f", space{descriptor_set}"
        elif register_space_expr is not None and register_space_expr.isdigit():
            register_space_suffix = f", space{register_space_expr}"

        suffix = ""
        if register_prefix and register_binding is not None:
            suffix = (
                f" : register({register_prefix}{register_binding}"
                f"{register_space_suffix})"
            )
        if register_binding is not None and isinstance(register_binding, int):
            self.reserve_slang_resource_register_range(
                register_prefix,
                register_binding,
                resource_count,
                self.slang_resource_name(node),
                space_key,
            )
            self.advance_slang_resource_register(
                register_prefix, space_key, register_binding, resource_count
            )
        if auto_assign and vk_binding is not None:
            self.reserve_slang_vk_binding_range(
                vk_descriptor_set,
                vk_binding,
                resource_count,
                self.slang_resource_name(node),
            )
            self.advance_slang_vk_binding(
                vk_descriptor_set,
                vk_binding,
                resource_count,
            )
        return prefix, suffix

    def apply_slang_resource_binding_decorations(
        self,
        declaration,
        node,
        type_name,
        forced_register_prefix=None,
        auto_assign=False,
    ):
        push_constant = slang_push_constant_declaration(node, declaration)
        if push_constant is not None:
            return push_constant
        prefix, suffix = self.slang_resource_binding_decorations(
            node, type_name, forced_register_prefix, auto_assign
        )
        return f"{prefix}{declaration}{suffix}"

    def generate_cbuffer(self, node):
        name = getattr(node, "name", None)
        if not name or not hasattr(node, "members"):
            return ""
        push_constant = slang_push_constant_cbuffer(self, node)
        if push_constant is not None:
            return push_constant
        declaration = self.apply_slang_resource_binding_decorations(
            f"cbuffer {name}",
            node,
            "ConstantBuffer",
            forced_register_prefix="b",
            auto_assign=True,
        )
        result = f"{declaration} {{\n"
        hlsl_packoffsets = slang_hlsl_cbuffer_packoffsets(self, node) or {}
        for member in node.members:
            if hasattr(member, "member_type"):
                member_type_name = self.convert_type_node_to_string(member.member_type)
                # A block's members are module-scope names; their types pick mul() for matrix products.
                self.variable_types[member.name] = member_type_name
                member_type = self.convert_type(member_type_name)
            else:
                member_type = self.convert_type(getattr(member, "vtype", "float"))
            modifiers = slang_member_modifier_prefix(
                self,
                member,
                member_type,
                f"constant buffer member '{name}.{member.name}'",
            )
            declaration = self.format_declaration(member_type, member.name)
            placement = hlsl_packoffsets.get(member.name) or slang_packoffset_suffix(
                self, member
            )
            result += f"    {modifiers}{declaration}{placement};\n"
        result += "};"
        return result

    def slang_attribute_signature(self, node):
        attributes = []
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", ""))
            arguments = tuple(
                self.attribute_value_to_string(argument)
                for argument in getattr(attr, "arguments", []) or []
            )
            attributes.append((attr_name, arguments))
        return tuple(attributes)

    def canonical_slang_compile_time_type_text(self, type_text):
        type_text = str(type_text)
        for name, value in sorted(self.literal_int_constants.items()):
            type_text = type_text.replace(f"[{name}]", f"[{value}]")
        return type_text

    def slang_global_variable_signature(self, node, declaration_type):
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        return (
            self.canonical_slang_compile_time_type_text(
                self.type_name_string(declaration_type)
            ),
            tuple(
                str(qualifier) for qualifier in getattr(node, "qualifiers", []) or []
            ),
            self.slang_attribute_signature(node),
            self.semantic_from_node(node),
            repr(initial_value) if initial_value is not None else None,
        )

    def skip_duplicate_slang_global_variable(self, node, declaration_type):
        name = getattr(node, "name", getattr(node, "variable_name", None))
        if not name:
            return False

        signature = self.slang_global_variable_signature(node, declaration_type)
        existing_signature = self.slang_global_declaration_signatures.get(name)
        if existing_signature is None:
            self.slang_global_declaration_signatures[name] = signature
            return False
        if existing_signature == signature:
            return True
        raise SlangRefusal(
            f"Duplicate Slang global declaration with incompatible definitions: {name}"
        )
