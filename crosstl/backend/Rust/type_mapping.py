"""Rust types, including rust-gpu resources and images, mapped to CrossGL types and declarators."""

import re

from .RustAst import AttributeNode


class RustTypeMappingMixin:
    """Type-mapping members of ``RustToCrossGLConverter``."""

    def map_type(self, rust_type):
        """Map a Rust type name to the closest CrossGL type name."""
        if not rust_type:
            return "void"

        const_generic_type = self.normalize_const_generic_argument(rust_type)
        if const_generic_type != rust_type:
            return self.map_type(const_generic_type)

        type_macro = self.map_type_macro(rust_type)
        if type_macro is not None:
            return type_macro

        higher_ranked_type = self.strip_higher_ranked_type(rust_type)
        if higher_ranked_type != rust_type:
            return self.map_type(higher_ranked_type)

        lifetime_stripped_type = self.strip_lifetime_type_syntax(rust_type)
        if lifetime_stripped_type != rust_type:
            return self.map_type(lifetime_stripped_type)

        referenced_type = self.strip_reference_type(rust_type)
        if referenced_type != rust_type:
            return self.map_type(referenced_type)

        raw_pointer_type = self.strip_raw_pointer_type(rust_type)
        if raw_pointer_type != rust_type:
            return f"ptr<{self.map_type(raw_pointer_type)}>"

        tuple_type = self.map_tuple_type(rust_type)
        if tuple_type is not None:
            return tuple_type

        trait_object_type = self.strip_trait_object_type(rust_type)
        if trait_object_type != rust_type:
            return self.map_type(trait_object_type)

        function_pointer_type = self.map_function_pointer_type(rust_type)
        if function_pointer_type is not None:
            return function_pointer_type

        callable_type = self.map_callable_trait_type(rust_type)
        if callable_type is not None:
            return callable_type

        rust_array_type = self.normalize_rust_array_generic_argument(rust_type)
        if rust_array_type is not None:
            return self.map_type(rust_array_type)

        expanded_type = self.resolve_imported_module_path(rust_type)
        if expanded_type != rust_type:
            return self.map_type(expanded_type)

        byte_addressable_type = self.map_byte_addressable_buffer_type(rust_type)
        if byte_addressable_type is not None:
            return byte_addressable_type

        array_parts = self.split_array_type(rust_type)
        if array_parts:
            base_type, array_suffix = array_parts
            return (
                f"{self.map_type(base_type)}"
                f"{self.normalize_array_suffix_for_crossgl(array_suffix)}"
            )

        resolved_alias = self.resolve_type_alias(rust_type)
        if resolved_alias is not None:
            return resolved_alias

        imported_alias = self.resolve_imported_type_alias(rust_type)
        if imported_alias is not None:
            return imported_alias

        builtin_type = self.map_builtin_type(rust_type)
        if builtin_type is not None:
            return builtin_type

        crossgl_type = self.format_unmapped_crossgl_type(rust_type)
        if crossgl_type is not None:
            return crossgl_type

        generic_type = self.format_unmapped_generic_type(rust_type)
        if generic_type is not None:
            return generic_type

        return rust_type

    def map_type_macro(self, rust_type):
        if not isinstance(rust_type, str):
            return None

        match = re.match(
            r"^(?:(?:[A-Za-z_][A-Za-z0-9_]*)::)*comptime_type!\((?P<body>.*)\)$",
            rust_type.strip(),
        )
        if match is None:
            return None

        body = match.group("body").strip()
        if not body:
            return None
        return self.map_type(body)

    def map_function_return_type(self, rust_type):
        tuple_type = self.map_tuple_type(rust_type)
        if tuple_type is not None:
            return tuple_type
        return self.map_type(rust_type)

    def map_tuple_type(self, rust_type):
        tuple_elements = self.split_tuple_type(rust_type)
        if tuple_elements is None:
            return None
        if not tuple_elements:
            return "void"
        mapped_elements = [self.map_type(element) for element in tuple_elements]
        return f"Tuple<{', '.join(mapped_elements)}>"

    def map_callable_trait_type(self, rust_type):
        if not isinstance(rust_type, str):
            return None

        stripped = rust_type.strip()
        match = re.match(r"^(Fn(?:Mut|Once)?)(?:\((.*)\))?(?:\s*->\s*(.+))?$", stripped)
        if not match:
            return None

        trait_name, params, return_type = match.groups()
        parts = [trait_name]
        if params:
            for param in self.split_generic_arguments(params):
                parts.append(self.map_type(param))
        if return_type:
            parts.append("to")
            parts.append(self.map_type(return_type))
        elif params is not None:
            parts.append("to")
            parts.append("void")

        sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", "_".join(parts)).strip("_")
        if not sanitized:
            return trait_name
        if sanitized[0].isdigit():
            sanitized = f"_{sanitized}"
        return sanitized

    def map_function_pointer_type(self, rust_type):
        if not isinstance(rust_type, str):
            return None
        if re.match(
            r"^(?:unsafe\s+)?(?:extern(?:\s*\"[^\"]+\")?\s*)?fn\s*\(",
            rust_type.strip(),
        ):
            return "auto"
        return None

    def format_unmapped_crossgl_type(self, rust_type):
        if not isinstance(rust_type, str) or "::" not in rust_type:
            return None

        generic = self.parse_generic_type(rust_type)
        if generic is not None:
            base_name, args = generic
            mapped_args = [self.map_type(arg) for arg in args]
            return (
                f"{self.crossgl_type_path_identifier(base_name)}"
                f"<{', '.join(mapped_args)}>"
            )

        return self.crossgl_type_path_identifier(rust_type)

    def format_unmapped_generic_type(self, rust_type):
        generic = self.parse_generic_type(rust_type)
        if generic is None:
            return None

        base_name, args = generic
        normalized_args = [
            self.normalize_unmapped_generic_argument(arg) for arg in args
        ]
        if normalized_args == args:
            return None
        return f"{base_name}<{', '.join(normalized_args)}>"

    def normalize_unmapped_generic_argument(self, arg):
        normalized = self.normalize_lifetime_generic_argument(arg)
        normalized = self.normalize_const_generic_argument(normalized)
        function_pointer_type = self.map_function_pointer_type(normalized)
        if function_pointer_type is not None:
            return function_pointer_type

        raw_pointer_type = self.strip_raw_pointer_type(normalized)
        if raw_pointer_type != normalized:
            return f"ptr<{self.map_type(raw_pointer_type)}>"
        return normalized

    def normalize_const_generic_argument(self, arg):
        if not isinstance(arg, str):
            return arg

        stripped = arg.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return arg
        if not self.outer_braces_wrap_type_argument(stripped):
            return arg

        inner = stripped[1:-1].strip()
        return inner or arg

    def outer_braces_wrap_type_argument(self, arg):
        depth = 0
        for index, char in enumerate(arg):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and index != len(arg) - 1:
                    return False
            if depth < 0:
                return False
        return depth == 0

    def strip_raw_pointer_type(self, rust_type):
        if not isinstance(rust_type, str):
            return rust_type

        match = re.match(r"^\*(?:const|mut)\s*(.+)$", rust_type)
        if match:
            return match.group(1).strip()

        return rust_type

    def crossgl_type_path_identifier(self, type_name):
        return self.crossgl_identifier(type_name.replace("::", "_"))

    def resolve_imported_type_alias(self, rust_type):
        generic = self.parse_generic_type(rust_type)
        if generic is not None:
            base_name, args = generic
            target = self.imported_type_aliases.get(base_name)
            if target is None:
                return None

            target_type = f"{target}<{', '.join(args)}>"
            return self.resolve_imported_target_type(rust_type, target_type)

        target = self.imported_type_aliases.get(rust_type)
        if target is None:
            return None

        return self.resolve_imported_target_type(rust_type, target)

    def resolve_imported_target_type(self, rust_type, target_type):
        if target_type == rust_type:
            return None

        mapped = self.map_type(target_type)
        if mapped == target_type:
            return None
        if mapped == self.format_unmapped_crossgl_type(target_type):
            return rust_type
        return mapped

    def resolve_imported_module_path(self, path):
        if not isinstance(path, str) or "::" not in path:
            return path

        resolved = path
        seen = set()
        while "::" in resolved:
            root, remainder = resolved.split("::", 1)
            target = self.imported_module_aliases.get(root)
            if target is None or root in seen:
                return resolved

            seen.add(root)
            resolved = f"{target}::{remainder}"

        return resolved

    def map_builtin_type(self, rust_type):
        image_macro_type = self.map_image_macro_type(rust_type)
        if image_macro_type is not None:
            return image_macro_type

        byte_addressable_type = self.map_byte_addressable_buffer_type(rust_type)
        if byte_addressable_type is not None:
            return byte_addressable_type

        for type_name in self.type_lookup_names(rust_type):
            mapped = self.type_map.get(type_name)
            if mapped is not None:
                return mapped

        generic = self.parse_generic_type(rust_type)
        if generic is None:
            return None

        base_name, args = generic
        for base_candidate in self.type_lookup_names(base_name):
            resource_type = self.map_resource_generic_type(base_candidate, args)
            if resource_type is not None:
                return resource_type

            candidate = f"{base_candidate}<{', '.join(args)}>"
            mapped = self.type_map.get(candidate)
            if mapped is not None:
                return mapped

            if base_candidate in ["Vec2", "Vec3", "Vec4"]:
                return self.type_map.get(base_candidate)

        return None

    def map_resource_generic_type(self, base_name, args):
        if not args:
            return None

        if base_name == "SampledImage":
            return self.map_sampled_image_generic_type(args[0])

        if base_name == "RuntimeArray":
            return f"{self.map_type(args[0])}[]"

        if base_name == "TypedBuffer":
            return self.format_typed_buffer_resource_type(args[0])

        if "ByteAddressableBuffer" in self.type_lookup_names(base_name):
            return self.map_byte_addressable_buffer_generic_argument(args[0])

        if base_name in {"Input", "Output"}:
            return self.map_type(args[0])

        sampled_texture_map = {
            "Texture1D": "sampler1D",
            "Texture1DArray": "sampler1DArray",
            "Texture2D": "sampler2D",
            "Texture2DArray": "sampler2DArray",
            "Texture2DMS": "sampler2DMS",
            "Texture2DMSArray": "sampler2DMSArray",
            "Texture3D": "sampler3D",
            "TextureCube": "samplerCube",
            "TextureCubeArray": "samplerCubeArray",
        }
        if base_name in sampled_texture_map:
            return sampled_texture_map[base_name]

        depth_texture_map = {
            "DepthTexture2D": "sampler2DShadow",
            "DepthTexture2DArray": "sampler2DArrayShadow",
            "DepthTextureCube": "samplerCubeShadow",
            "DepthTextureCubeArray": "samplerCubeArrayShadow",
        }
        if base_name in depth_texture_map:
            return depth_texture_map[base_name]

        buffer_map = {
            "Buffer": "StructuredBuffer",
            "RwBuffer": "RWStructuredBuffer",
            "AppendBuffer": "AppendStructuredBuffer",
            "ConsumeBuffer": "ConsumeStructuredBuffer",
        }
        if base_name in buffer_map:
            return f"{buffer_map[base_name]}<{self.map_type(args[0])}>"

        image_suffixes = {
            "Image1D": "1D",
            "Image1DArray": "1DArray",
            "Image2D": "2D",
            "Image3D": "3D",
            "ImageCube": "Cube",
            "Image2DArray": "2DArray",
            "Image2DMS": "2DMS",
            "Image2DMSArray": "2DMSArray",
        }
        suffix = image_suffixes.get(base_name)
        if suffix is None:
            return None

        element_type = self.map_type(args[0])
        if element_type.startswith("u"):
            return f"uimage{suffix}"
        if element_type.startswith("i"):
            return f"iimage{suffix}"
        return f"image{suffix}"

    def map_typed_buffer_parameter_type(self, rust_type):
        if not isinstance(rust_type, str):
            return None

        rust_type = self.strip_lifetime_type_syntax(rust_type.strip())
        if rust_type.startswith("&mut "):
            return self.map_typed_buffer_reference_type(rust_type[5:].strip(), True)
        if rust_type.startswith("&"):
            return self.map_typed_buffer_reference_type(rust_type[1:].strip(), False)
        return None

    def map_byte_addressable_buffer_parameter_type(self, rust_type):
        if not isinstance(rust_type, str):
            return None

        rust_type = self.strip_lifetime_type_syntax(rust_type.strip())
        if rust_type.startswith("&mut "):
            return self.map_byte_addressable_buffer_type(
                rust_type[5:].strip(),
                writable=True,
            )
        if rust_type.startswith("&"):
            return self.map_byte_addressable_buffer_type(
                rust_type[1:].strip(),
                writable=False,
            )
        return None

    def map_rust_gpu_storage_buffer_parameter_type(self, rust_type, attributes):
        if not isinstance(rust_type, str):
            return None
        if not self.has_rust_gpu_storage_buffer_attribute(attributes):
            return None

        rust_type = self.strip_lifetime_type_syntax(rust_type.strip())
        writable = False
        if rust_type.startswith("&mut "):
            writable = True
            rust_type = rust_type[5:].strip()
        elif rust_type.startswith("&"):
            rust_type = rust_type[1:].strip()
        else:
            return None

        element_type = self.rust_gpu_storage_buffer_element_type(rust_type)
        if element_type is None:
            return None

        buffer_kind = "RWStructuredBuffer" if writable else "StructuredBuffer"
        return self.format_typed_buffer_resource_type(element_type, buffer_kind)

    def has_rust_gpu_storage_buffer_attribute(self, attributes):
        for attr in self.effective_attributes(attributes):
            if not isinstance(attr, AttributeNode) or attr.name != "spirv":
                continue
            if "storage_buffer" in getattr(attr, "args", []):
                return True
        return False

    def rust_gpu_storage_buffer_element_type(self, rust_type):
        stripped = rust_type.strip()
        if stripped.endswith("[]"):
            element_type = stripped[:-2].strip()
            return element_type or None
        if "[" in stripped and stripped.endswith("]"):
            base_type, _array_size = stripped.rsplit("[", 1)
            if _array_size[:-1].strip():
                return base_type.strip() or None
        if stripped.startswith("[") and stripped.endswith("]"):
            body = stripped[1:-1].strip()
            if ";" in body:
                body = body.split(";", 1)[0].strip()
            return body or None

        generic = self.parse_generic_type(stripped)
        if generic is not None:
            base_name, args = generic
            if base_name in {"RuntimeArray", "Slice"} and args:
                return args[0]

        return None

    def map_byte_addressable_buffer_type(self, rust_type, writable=False):
        if not isinstance(rust_type, str):
            return None

        rust_type = self.strip_lifetime_type_syntax(rust_type.strip())
        generic = self.parse_generic_type(rust_type)
        if generic is not None:
            base_name, args = generic
            if "ByteAddressableBuffer" not in self.type_lookup_names(base_name):
                return None
            return self.map_byte_addressable_buffer_generic_argument(
                args[0],
                writable,
            )

        if "ByteAddressableBuffer" not in self.type_lookup_names(rust_type):
            return None
        return "RWByteAddressBuffer" if writable else "ByteAddressBuffer"

    def map_byte_addressable_buffer_generic_argument(
        self,
        argument_type,
        writable=False,
    ):
        argument_type = self.strip_lifetime_type_syntax(argument_type.strip())
        compact_argument = re.sub(r"\s+", "", argument_type)

        if compact_argument.startswith("&mut"):
            return "RWByteAddressBuffer"
        if compact_argument.startswith("&"):
            return "ByteAddressBuffer"
        return "RWByteAddressBuffer" if writable else "ByteAddressBuffer"

    def map_typed_buffer_reference_type(self, rust_type, writable):
        mapped_type = self.map_typed_buffer_type(rust_type, writable=writable)
        if mapped_type is not None:
            return mapped_type
        return None

    def map_typed_buffer_type(self, rust_type, writable=False):
        generic = self.parse_generic_type(rust_type)
        if generic is None:
            return None

        base_name, args = generic
        if "TypedBuffer" not in self.type_lookup_names(base_name):
            return None

        buffer_kind = "RWStructuredBuffer" if writable else "StructuredBuffer"
        return self.format_typed_buffer_resource_type(args[0], buffer_kind)

    def format_typed_buffer_resource_type(
        self,
        element_type,
        buffer_kind="StructuredBuffer",
    ):
        mapped_element_type = self.map_typed_buffer_element_type(element_type)
        return f"{buffer_kind}<{mapped_element_type}>"

    def map_typed_buffer_element_type(self, element_type):
        element_type = element_type.strip()
        if element_type.startswith("[") and element_type.endswith("]"):
            body = element_type[1:-1].strip()
            if ";" in body:
                inner_type, size = body.split(";", 1)
                return f"{self.map_type(inner_type.strip())}[{size.strip()}]"
            return self.map_type(body)
        return self.map_type(element_type)

    def map_sampled_image_generic_type(self, image_type):
        image_config = self.parse_image_macro_type(image_type)
        if image_config is not None:
            suffix = self.image_macro_dimension_suffix(image_config)
            if suffix is None:
                return None

            prefix = self.image_macro_sampler_prefix(image_config)
            if image_config["depth"] is True:
                return f"{prefix}{suffix}Shadow"
            return f"{prefix}{suffix}"

        mapped_image_type = self.map_type(image_type)
        sampler_type = self.sampled_image_type_from_mapped_image(mapped_image_type)
        if sampler_type is not None:
            return sampler_type

        return None

    def sampled_image_type_from_mapped_image(self, mapped_image_type):
        if not isinstance(mapped_image_type, str):
            return None

        if mapped_image_type.startswith("sampler"):
            return mapped_image_type
        if mapped_image_type.startswith("uimage"):
            return f"usampler{mapped_image_type[len('uimage') :]}"
        if mapped_image_type.startswith("iimage"):
            return f"isampler{mapped_image_type[len('iimage') :]}"
        if mapped_image_type.startswith("image"):
            return f"sampler{mapped_image_type[len('image') :]}"
        return None

    def map_image_macro_type(self, rust_type):
        image_config = self.parse_image_macro_type(rust_type)
        if image_config is None:
            return None

        subpass_type = self.map_image_macro_subpass_type(image_config)
        if subpass_type is not None:
            return subpass_type

        suffix = self.image_macro_dimension_suffix(image_config)
        if suffix is None:
            return None

        if image_config["sampled"] is True:
            prefix = self.image_macro_sampler_prefix(image_config)
            if image_config["depth"] is True:
                return f"{prefix}{suffix}Shadow"
            return f"{prefix}{suffix}"

        prefix = self.image_macro_storage_prefix(image_config)
        return f"{prefix}{suffix}"

    def parse_image_macro_type(self, rust_type):
        if not isinstance(rust_type, str):
            return None

        match = re.match(
            r"^(?:(?:[A-Za-z_][A-Za-z0-9_]*)::)*Image!\((?P<args>.*)\)$",
            rust_type,
        )
        if match is None:
            return None

        args = self.split_generic_arguments(match.group("args"))
        if not args:
            return None

        config = {
            "dimension": args[0].replace(" ", ""),
            "type": None,
            "format": None,
            "sampled": None,
            "multisampled": False,
            "arrayed": False,
            "depth": None,
        }

        for arg in args[1:]:
            key, value = self.split_image_macro_argument(arg)
            if key in {"type", "format"}:
                config[key] = value
            elif key in {"sampled", "multisampled", "arrayed", "depth"}:
                config[key] = self.parse_image_macro_bool(value)

        return config

    def split_image_macro_argument(self, arg):
        if "=" not in arg:
            return arg.strip(), "true"

        key, value = arg.split("=", 1)
        return key.strip(), value.strip()

    def parse_image_macro_bool(self, value):
        value = (value or "").strip().lower()
        if value == "true":
            return True
        if value == "false":
            return False
        return None

    def image_macro_dimension_suffix(self, config):
        dimension = config["dimension"].lower()
        if dimension == "buffer":
            if (
                config["multisampled"] is True
                or config["arrayed"] is True
                or config["depth"] is True
            ):
                return None
            return "Buffer"

        suffix_map = {
            "1d": "1D",
            "2d": "2D",
            "rect": "2DRect",
            "3d": "3D",
            "cube": "Cube",
        }
        suffix = suffix_map.get(dimension)
        if suffix is None:
            return None

        if config["multisampled"] is True:
            if suffix != "2D":
                return None
            suffix = f"{suffix}MS"

        if config["arrayed"] is True:
            suffix = f"{suffix}Array"

        return suffix

    def map_image_macro_subpass_type(self, config):
        if config["dimension"].lower() != "subpass":
            return None

        family = self.image_macro_numeric_family(config)
        if family == "u":
            subpass_type = "usubpassInput"
        elif family == "i":
            subpass_type = "isubpassInput"
        else:
            subpass_type = "subpassInput"

        if config["multisampled"] is True:
            subpass_type += "MS"
        if config["arrayed"] is True:
            subpass_type += "Array"
        return subpass_type

    def image_macro_sampler_prefix(self, config):
        if config["depth"] is True:
            return "sampler"

        family = self.image_macro_numeric_family(config)
        if family == "u":
            return "usampler"
        if family == "i":
            return "isampler"
        return "sampler"

    def image_macro_storage_prefix(self, config):
        family = self.image_macro_numeric_family(config)
        if family == "u":
            return "uimage"
        if family == "i":
            return "iimage"
        return "image"

    def image_macro_numeric_family(self, config):
        sample_type = config.get("type")
        if sample_type:
            sample_type = sample_type.rsplit("::", 1)[-1].strip().lower()
            if sample_type.startswith("u"):
                return "u"
            if sample_type.startswith("i"):
                return "i"
            return "f"

        image_format = config.get("format")
        if image_format:
            image_format = image_format.rsplit("::", 1)[-1].strip().lower()
            image_format = image_format.replace("_", "")
            if image_format.endswith("ui"):
                return "u"
            if image_format.endswith("i") and not image_format.endswith("ui"):
                return "i"

        return "f"

    def resolve_type_alias_target(self, rust_type):
        generic = self.parse_generic_type(rust_type)
        if generic is not None:
            base_name, args = generic
            for alias_name in self.type_lookup_names(base_name):
                alias = self.type_aliases.get(alias_name)
                if alias is None:
                    continue
                return self.apply_alias_type_arguments(alias, args)
            return None

        for alias_name in self.type_lookup_names(rust_type):
            alias = self.type_aliases.get(alias_name)
            if alias is not None and not getattr(alias, "generics", None):
                return alias.alias_type
        return None

    def resolve_type_alias(self, rust_type):
        for alias_name in self.type_lookup_names(rust_type):
            alias = self.type_aliases.get(alias_name)
            if alias is None:
                continue
            if getattr(alias, "generics", None):
                return None
            return alias.name

        generic = self.parse_generic_type(rust_type)
        if generic is None:
            return None

        base_name, args = generic
        alias = None
        for alias_name in self.type_lookup_names(base_name):
            alias = self.type_aliases.get(alias_name)
            if alias is not None:
                break
        if alias is None:
            return None

        generics = getattr(alias, "generics", []) or []
        if not generics or len(generics) != len(args):
            return None

        if self.alias_target_base_matches(alias, base_name):
            return None

        substituted = self.substitute_type_parameters(alias.alias_type, generics, args)
        if substituted == rust_type:
            return None
        return self.map_type(substituted)

    def alias_target_base_matches(self, alias, base_name):
        target = getattr(alias, "alias_type", None)
        target_generic = self.parse_generic_type(target)
        if target_generic is None:
            return target == base_name

        target_base, _ = target_generic
        return target_base == base_name

    def type_lookup_names(self, type_name):
        names = [type_name]
        if "::" in type_name:
            names.append(type_name.rsplit("::", 1)[-1])
        return names

    def parse_generic_type(self, rust_type):
        if not rust_type or "<" not in rust_type or not rust_type.endswith(">"):
            return None

        base_name, remainder = rust_type.split("<", 1)
        base_name = base_name.strip()
        if not base_name:
            return None

        args_text = remainder[:-1]
        args = self.split_generic_arguments(args_text)
        if not args:
            return None
        return base_name, args

    def split_generic_arguments(self, args_text):
        args = []
        current = []
        depth = 0

        for char in args_text:
            if char == "," and depth == 0:
                arg = "".join(current).strip()
                if arg:
                    args.append(arg)
                current = []
                continue

            if char in "<[({":
                depth += 1
            elif char in ">])}":
                depth = max(0, depth - 1)

            current.append(char)

        arg = "".join(current).strip()
        if arg:
            args.append(arg)
        return args

    def strip_higher_ranked_type(self, type_name):
        if not isinstance(type_name, str):
            return type_name

        stripped = type_name.strip()
        if not stripped.startswith("for<"):
            return type_name

        depth = 0
        for index, char in enumerate(stripped[3:], start=3):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
                if depth == 0:
                    return stripped[index + 1 :].strip()

        return type_name

    def strip_lifetime_type_syntax(self, type_name):
        if not isinstance(type_name, str) or "'" not in type_name:
            return type_name

        generic = self.parse_generic_type(type_name)
        if generic is not None:
            base_name, args = generic
            stripped_args = []
            changed = False
            for arg in args:
                if self.is_lifetime_type_argument(arg):
                    changed = True
                    continue
                stripped_arg = self.strip_lifetime_type_syntax(arg)
                stripped_arg = self.normalize_lifetime_generic_argument(stripped_arg)
                changed = changed or stripped_arg != arg
                stripped_args.append(stripped_arg)

            if changed:
                if not stripped_args:
                    return base_name
                return f"{base_name}<{', '.join(stripped_args)}>"

        return self.strip_reference_lifetime(type_name)

    def normalize_lifetime_generic_argument(self, type_name):
        """Make lifetime-bearing generic args parseable after lifetime erasure."""
        if not isinstance(type_name, str):
            return type_name

        normalized = self.strip_reference_lifetime(type_name.strip())
        if self.is_lifetime_type_argument(normalized):
            return ""

        previous = None
        while normalized != previous:
            previous = normalized
            normalized = self.strip_reference_type(normalized)

        trait_object_type = self.strip_trait_object_type(normalized)
        if trait_object_type != normalized:
            return self.normalize_lifetime_generic_argument(trait_object_type)

        array_parts = self.split_array_type(normalized)
        if array_parts is not None:
            base_type, array_suffix = array_parts
            normalized_base = self.normalize_lifetime_generic_argument(base_type)
            return f"{normalized_base}{array_suffix}"

        array_type = self.normalize_rust_array_generic_argument(normalized)
        if array_type is not None:
            return array_type

        tuple_elements = self.split_tuple_type(normalized)
        if tuple_elements is not None:
            if not tuple_elements:
                return "void"
            elements = [
                self.normalize_lifetime_generic_argument(element)
                for element in tuple_elements
            ]
            return f"Tuple<{', '.join(elements)}>"

        generic = self.parse_generic_type(normalized)
        if generic is None:
            return normalized

        base_name, args = generic
        normalized_args = []
        for arg in args:
            if self.is_lifetime_type_argument(arg):
                continue
            normalized_arg = self.normalize_lifetime_generic_argument(arg)
            if normalized_arg:
                normalized_args.append(normalized_arg)
        if not normalized_args:
            return base_name
        return f"{base_name}<{', '.join(normalized_args)}>"

    def normalize_rust_array_generic_argument(self, type_name):
        if not isinstance(type_name, str):
            return None

        text = type_name.strip()
        if not (text.startswith("[") and text.endswith("]")):
            return None

        inner = text[1:-1].strip()
        if not inner:
            return None

        element_type, size = self.split_rust_array_generic_inner(inner)
        normalized_element = self.normalize_lifetime_generic_argument(element_type)
        suffix = f"[{size.strip()}]" if size is not None and size.strip() else "[]"
        return f"{normalized_element}{suffix}"

    def split_rust_array_generic_inner(self, inner):
        depth = 0
        for index, char in enumerate(inner):
            if char == ";" and depth == 0:
                return inner[:index].strip(), inner[index + 1 :].strip()
            if char in "<[({":
                depth += 1
            elif char in ">])}":
                depth = max(0, depth - 1)
        return inner.strip(), None

    def is_lifetime_type_argument(self, type_name):
        return bool(re.fullmatch(r"'[A-Za-z_][A-Za-z0-9_]*", type_name.strip()))

    def strip_reference_lifetime(self, type_name):
        return re.sub(
            r"(&\s*)'[A-Za-z_][A-Za-z0-9_]*\s*",
            r"\1",
            type_name,
        )

    def substitute_type_parameters(self, alias_type, generics, args):
        substitutions = {
            self.generic_parameter_name(generic): arg
            for generic, arg in zip(generics, args)
        }
        result = alias_type

        for name, replacement in substitutions.items():
            result = self.replace_type_identifier(result, name, replacement)

        return result

    def generic_parameter_name(self, generic):
        return generic.split(":", 1)[0].strip()

    def replace_type_identifier(self, text, name, replacement):
        result = []
        index = 0
        name_len = len(name)

        while index < len(text):
            if (
                text.startswith(name, index)
                and self.is_type_identifier_boundary(text, index - 1)
                and self.is_type_identifier_boundary(text, index + name_len)
            ):
                result.append(replacement)
                index += name_len
            else:
                result.append(text[index])
                index += 1

        return "".join(result)

    def is_type_identifier_boundary(self, text, index):
        if index < 0 or index >= len(text):
            return True
        return not (text[index].isalnum() or text[index] == "_")

    def strip_reference_type(self, type_name):
        match = re.match(r"^&\s*mut\s*(.+)$", type_name)
        if match:
            return match.group(1).strip()
        match = re.match(r"^&\s*(.+)$", type_name)
        if match:
            return match.group(1).strip()
        return type_name

    def strip_trait_object_type(self, type_name):
        for prefix in ("impl ", "dyn "):
            if type_name.startswith(prefix):
                return self.strip_trait_bound_type(type_name[len(prefix) :].strip())
        return type_name

    def strip_trait_bound_type(self, type_name):
        for bound in self.split_top_level_type_bounds(type_name):
            bound = bound.strip()
            if bound and not self.is_lifetime_type_argument(bound):
                return self.strip_associated_type_constraints(bound)
        return type_name

    def split_top_level_type_bounds(self, type_name):
        bounds = []
        current = []
        depth = 0

        for char in type_name:
            if char == "+" and depth == 0:
                bound = "".join(current).strip()
                if bound:
                    bounds.append(bound)
                current = []
                continue

            if char in "<[({":
                depth += 1
            elif char in ">])}":
                depth = max(0, depth - 1)

            current.append(char)

        bound = "".join(current).strip()
        if bound:
            bounds.append(bound)
        return bounds

    def strip_associated_type_constraints(self, type_name):
        generic = self.parse_generic_type(type_name)
        if generic is None:
            return type_name

        base_name, args = generic
        kept_args = [
            arg
            for arg in args
            if "=" not in arg and not self.is_lifetime_type_argument(arg)
        ]
        if not kept_args:
            return base_name
        return f"{base_name}<{', '.join(kept_args)}>"

    def split_array_type(self, type_name):
        if not type_name or "[" not in type_name:
            return None

        array_index = self.find_top_level_array_suffix(type_name)
        if array_index is None:
            return None

        base_type = type_name[:array_index]
        array_suffix = type_name[len(base_type) :]
        if not base_type or not array_suffix:
            return None
        return base_type, array_suffix

    def find_top_level_array_suffix(self, type_name):
        depth = 0
        for index, char in enumerate(type_name):
            if char == "[" and depth == 0:
                return index
            if char in "<([{":
                depth += 1
            elif char in ">)]}":
                depth = max(0, depth - 1)
        return None

    def normalize_array_suffix_for_crossgl(self, array_suffix):
        return re.sub(
            r"\s*as\s+[ui](?:8|16|32|64|128|size)\b",
            "",
            array_suffix,
        )

    def format_typed_declarator(self, type_name, name):
        mapped_type = self.map_type(type_name)
        array_parts = self.split_array_type(mapped_type)
        if array_parts:
            base_type, array_suffix = array_parts
            return f"{base_type} {name}{array_suffix}"
        return f"{mapped_type} {name}"

    def format_global_declarator(self, type_name, name):
        mapped_type = self.map_global_declaration_type(type_name)
        array_parts = self.split_array_type(mapped_type)
        if array_parts:
            base_type, array_suffix = array_parts
            return f"{base_type} {name}{array_suffix}"
        return f"{mapped_type} {name}"

    def map_global_declaration_type(self, type_name):
        mapped_type = self.map_type(type_name)
        array_parts = self.split_array_type(mapped_type)
        if not array_parts:
            return mapped_type

        base_type, array_suffix = array_parts
        tuple_type = self.map_tuple_type(base_type)
        if tuple_type is None:
            return mapped_type
        return f"{tuple_type}{array_suffix}"

    def expand_repeated_array_literal(self, element, size):
        try:
            count = int(self.generate_expression(size))
        except (TypeError, ValueError):
            return None

        if count < 0:
            return None

        value = self.generate_expression(element)
        return "{" + ", ".join(value for _ in range(count)) + "}"
