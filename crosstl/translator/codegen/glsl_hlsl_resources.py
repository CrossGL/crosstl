"""GLSL emission of HLSL resource semantics.

The members here map the HLSL register model onto GLSL: register-class
binding assignment, binding records for each emitted resource and combined
sampler, resource-specialized functions, and storage images declared without a
format, which take the format of the bound image unit.
"""

import re

from ..ast import FunctionCallNode


class GLSLHlslResourceMixin:
    """HLSL resource and binding members of ``GLSLCodeGen``."""

    def glsl_has_resource_function_specializations(self, source_name):
        return bool(source_name) and any(
            getattr(func, "_glsl_resource_source_name", None) == source_name
            for func in self.glsl_resource_function_specializations.values()
        )

    def glsl_function_has_structured_buffer_parameter(self, func):
        for parameter in getattr(func, "parameters", None) or []:
            parameter_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            if parameter_type is not None and self.is_structured_buffer_type(
                parameter_type
            ):
                return True
        return False

    HLSL_REGISTER_CLASS_ORDER = {"b": 0, "t": 1, "u": 2, "s": 3}

    def hlsl_register_binding(self, node):
        """Return ``(class, index)`` when ``node`` is bound only by an HLSL register.

        HLSL register classes (``b``, ``t``, ``u`` and ``s``) are separate binding
        namespaces, while OpenGL shares one shader-storage binding table between
        read-only (``t``) and read-write (``u``) structured buffers. Register
        numbers are therefore source bindings rather than OpenGL binding points;
        explicit ``@binding``, descriptor sets and register spaces keep their
        exact target meaning.
        """
        selected = None
        for attr in getattr(node, "attributes", None) or []:
            attr_name = str(getattr(attr, "name", "") or "").lower()
            arguments = getattr(attr, "arguments", []) or []
            if not arguments:
                continue
            if attr_name in {"binding", "buffer", "texture", "sampler", "set", "space"}:
                return None
            if attr_name != "register":
                continue
            if len(arguments) > 1:
                return None
            raw_value = str(self.attribute_value_to_string(arguments[0]) or "")
            raw_value = raw_value.strip().lower()
            register_class, digits = raw_value[:1], raw_value[1:]
            if register_class not in self.HLSL_REGISTER_CLASS_ORDER:
                return None
            if not digits.isdigit():
                return None
            selected = (register_class, int(digits))
        return selected

    def reserve_deferred_hlsl_register_bindings(self, deferred, used_bindings):
        """Assign OpenGL binding points to resources bound only by HLSL registers.

        Exact bindings are reserved first. Each register-bound resource then keeps
        its register number when that binding point is free in its OpenGL table
        and otherwise takes the next free one. Resources are visited in register
        class order (b, t, u, s), then register number, so the mapping does not
        depend on declaration order. Overlapping registers of one class remain
        errors.
        """
        source_ranges = {}
        ordered = sorted(
            deferred,
            key=lambda entry: (
                self.HLSL_REGISTER_CLASS_ORDER[entry[2][0]],
                entry[2][1],
                entry[5],
            ),
        )
        for node, namespace, register, count, name, _order in ordered:
            register_class, register_index = register
            self.reserve_resource_binding_range(
                source_ranges,
                "HLSL",
                f"register {register_class}",
                register_index,
                count,
                name,
            )
            target = self.next_available_resource_binding_from(
                used_bindings, namespace, register_index, count
            )
            self.reserve_resource_binding_range(
                used_bindings, "OpenGL", namespace, target, count, name
            )
            self.glsl_register_target_bindings[id(node)] = target
            self.glsl_hlsl_register_by_id[id(node)] = (
                f"{register_class}{register_index}"
            )

    GLSL_BINDING_RECORD_NAMESPACES = {
        "buffer binding": "shader-storage-buffer",
        "uniform buffer binding": "uniform-buffer",
        "texture binding": "texture",
        "image binding": "image",
        "sampler binding": "sampler",
    }

    def glsl_source_register(self, node):
        """The HLSL register (``t0``) that bound ``node`` in the source, if any."""
        planned = (getattr(self, "glsl_hlsl_register_by_id", None) or {}).get(id(node))
        if planned is not None:
            return planned
        register = self.hlsl_register_binding(node)
        if register is not None:
            return f"{register[0]}{register[1]}"
        return None

    def record_glsl_resource_binding(
        self,
        node,
        source_name,
        glsl_name,
        kind,
        binding_namespace,
        binding,
        count=1,
        of=None,
    ):
        if binding is None:
            return
        record = {
            "name": source_name,
            "glslName": glsl_name,
            "kind": kind,
            "namespace": self.GLSL_BINDING_RECORD_NAMESPACES.get(
                binding_namespace, binding_namespace
            ),
            "binding": int(binding),
            "count": max(count, 1) if isinstance(count, int) else 1,
            "register": self.glsl_source_register(node),
        }
        if of is not None:
            record["of"] = of
        self.glsl_resource_binding_records.append(record)

    def record_glsl_combined_sampler_use(self, texture_arg, sampler_arg):
        texture_name = self.expression_name(texture_arg)
        sampler_name = self.expression_name(sampler_arg)
        if not texture_name or not sampler_name:
            return
        entry = (str(texture_name), str(sampler_name))
        if entry not in self.glsl_combined_sampler_uses:
            self.glsl_combined_sampler_uses.append(entry)

    def resource_binding_records(self):
        """Describe the OpenGL binding point assigned to each resource.

        Valid after ``generate``. Each record names the source resource, its
        GLSL identifier, its OpenGL binding table (``shader-storage-buffer``,
        ``uniform-buffer``, ``texture``, ``image`` or ``sampler``) and binding
        point, the binding count for arrays and the HLSL register that bound it
        when the source used one. OpenGL combines textures and samplers in
        texture units, so a source sampler that is combined with a texture is
        reported once per texture unit it must be bound to, with ``texture``
        naming that texture.
        """
        records = [dict(record) for record in self.glsl_resource_binding_records]
        textures = {}
        for record in records:
            if record["kind"] == "texture":
                textures.setdefault(record["glslName"], record)
                textures.setdefault(record["name"], record)
        sampler_registers = getattr(self, "glsl_sampler_registers", {}) or {}
        bound_units = set()
        unresolved_samplers = []
        for texture_name, sampler_name in self.glsl_combined_sampler_uses:
            texture = textures.get(texture_name)
            if texture is None:
                # A texture parameter of a helper: the source sampler state applies
                # to the global textures that reach it.
                if sampler_name not in unresolved_samplers:
                    unresolved_samplers.append(sampler_name)
                continue
            key = (texture["binding"], sampler_name)
            if key in bound_units:
                continue
            bound_units.add(key)
            records.append(
                self.glsl_combined_sampler_record(
                    sampler_name, sampler_registers, texture
                )
            )
        for sampler_name in unresolved_samplers:
            for texture in [r for r in records if r["kind"] == "texture"]:
                key = (texture["binding"], sampler_name)
                if key in bound_units:
                    continue
                bound_units.add(key)
                records.append(
                    self.glsl_combined_sampler_record(
                        sampler_name, sampler_registers, texture
                    )
                )
        return records

    def glsl_combined_sampler_record(self, sampler_name, sampler_registers, texture):
        source = sampler_registers.get(sampler_name, (sampler_name, None))
        return {
            "name": source[0],
            "glslName": sampler_name,
            "kind": "sampler",
            "namespace": "texture",
            "binding": texture["binding"],
            "count": texture["count"],
            "register": source[1],
            "texture": texture["name"],
        }

    GLSL_UNKNOWN_IMAGE_FORMAT = "unknown"

    GLSL_FORMATTED_IMAGE_LOAD_EXTENSION = "GL_EXT_shader_image_load_formatted"

    def glsl_unknown_image_format_qualifier(self, vtype, node):
        """Format qualifier of a storage image whose format the runtime chooses.

        Sources such as HLSL bind typed storage textures of any compatible format
        to one declaration, so the GLSL image is declared without a format and
        takes the format of the bound image unit. Loads from such images need
        ``GL_EXT_shader_image_load_formatted``; images that are only written are
        declared ``writeonly`` and need no extension. Atomic operations require a
        single-channel 32-bit format, which is the only format they accept.
        """
        names = {self.resource_node_name(node)}
        if node is not None and getattr(node, "name", None):
            names.add(self.glsl_global_identifier_name(node.name))
        usages = set()
        for name in names:
            usages |= self.glsl_image_usages.get(name, set())
        self.glsl_unknown_format_images.update(name for name in names if name)
        if "atomic" in usages:
            base_type = str(self.resource_base_type(vtype) or "")
            if base_type.startswith("iimage"):
                return "r32i"
            if base_type.startswith("uimage"):
                return "r32ui"
            return "r32f"
        if usages & {"load", "call"}:
            self.glsl_formatted_image_load_names.update(names)
        else:
            self.glsl_write_only_unknown_format_images.update(names)
        return None

    def collect_glsl_image_usages(self, ast):
        """Map resource names to how image intrinsics and calls use them."""
        usages = {}
        user_functions = {
            getattr(function, "name", None)
            for function in getattr(ast, "functions", []) or []
        }
        for node in self.walk_ast(ast):
            if not isinstance(node, FunctionCallNode):
                continue
            name = self.function_call_name(node)
            arguments = list(getattr(node, "args", None) or [])
            if not name or not arguments:
                continue
            if name in user_functions:
                for argument in arguments:
                    argument_name = self.expression_name(argument)
                    if argument_name:
                        usages.setdefault(argument_name, set()).add("call")
                continue
            target = self.expression_name(arguments[0])
            if not target:
                continue
            if name == "imageLoad":
                usages.setdefault(target, set()).add("load")
            elif name.startswith("imageAtomic"):
                usages.setdefault(target, set()).add("atomic")
            elif name == "imageStore":
                usages.setdefault(target, set()).add("store")
        return usages

    def glsl_unknown_format_image(self, image_arg):
        name = self.expression_name(image_arg)
        if not name:
            return False
        alias = self.current_resource_aliases.get(name)
        if alias is not None:
            name = alias.get("expression", name)
        return name in self.glsl_unknown_format_images

    def ensure_glsl_formatted_image_load_support(self, code):
        if not self.glsl_formatted_image_load_names:
            return code
        extension = self.GLSL_FORMATTED_IMAGE_LOAD_EXTENSION
        if re.search(rf"^\s*#extension\s+{extension}\b", code, flags=re.MULTILINE):
            return code
        lines = code.splitlines(keepends=True)
        version_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip().startswith("#version")
            ),
            -1,
        )
        lines.insert(version_index + 1, f"#extension {extension} : require\n")
        self.current_glsl_extensions.add(extension)
        return "".join(lines)
