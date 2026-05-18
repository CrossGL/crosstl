"""Helpers for recognizing resource query expressions in CrossGL AST nodes."""

from ..ast import ArrayAccessNode, FunctionCallNode


class ResourceQueryMixin:
    """Helpers for resource query metadata propagation and code generation."""

    query_function_names = {
        "textureSize",
        "imageSize",
        "textureSamples",
        "imageSamples",
        "textureQueryLevels",
    }

    def collect_resource_query_requirements(self, node):
        """Collect global and function-parameter resources needing metadata."""
        functions = self.query_collect_functions(node)
        functions_by_name = {getattr(func, "name", None): func for func in functions}
        functions_by_name = {
            name: func for name, func in functions_by_name.items() if name
        }
        param_names = {
            func_name: {
                getattr(param, "name", None)
                for param in getattr(func, "parameters", getattr(func, "params", []))
            }
            for func_name, func in functions_by_name.items()
        }
        param_names = {
            func_name: {name for name in names if name}
            for func_name, names in param_names.items()
        }

        global_query_names = set()
        function_param_query_names = {
            func_name: set() for func_name in functions_by_name
        }

        def mark_resource_name(func_name, resource_name):
            """Record a resource as global or function-parameter metadata."""
            if not resource_name:
                return False
            if resource_name in param_names.get(func_name, set()):
                before = len(function_param_query_names[func_name])
                function_param_query_names[func_name].add(resource_name)
                return len(function_param_query_names[func_name]) != before
            before = len(global_query_names)
            global_query_names.add(resource_name)
            return len(global_query_names) != before

        for func_name, func in functions_by_name.items():
            for call in self.query_walk_nodes(getattr(func, "body", [])):
                if not isinstance(call, FunctionCallNode):
                    continue
                func_call_name = self.raw_function_call_name(call)
                raw_args = getattr(call, "arguments", getattr(call, "args", []))
                if func_call_name in self.query_function_names and raw_args:
                    mark_resource_name(func_name, self.get_expression_name(raw_args[0]))

        changed = True
        while changed:
            changed = False
            for caller_name, caller in functions_by_name.items():
                caller_params = param_names.get(caller_name, set())
                for call in self.query_walk_nodes(getattr(caller, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.raw_function_call_name(call)
                    callee = functions_by_name.get(callee_name)
                    if callee is None:
                        continue

                    callee_required = function_param_query_names.get(callee_name, set())
                    if not callee_required:
                        continue

                    callee_params = getattr(
                        callee, "parameters", getattr(callee, "params", [])
                    )
                    raw_args = getattr(call, "arguments", getattr(call, "args", []))
                    for index, param in enumerate(callee_params):
                        if index >= len(raw_args):
                            continue
                        param_name = getattr(param, "name", None)
                        if param_name not in callee_required:
                            continue

                        arg_name = self.get_expression_name(raw_args[index])
                        if not arg_name:
                            continue
                        if arg_name in caller_params:
                            before = len(function_param_query_names[caller_name])
                            function_param_query_names[caller_name].add(arg_name)
                            changed = (
                                changed
                                or len(function_param_query_names[caller_name])
                                != before
                            )
                        else:
                            before = len(global_query_names)
                            global_query_names.add(arg_name)
                            changed = changed or len(global_query_names) != before

        return (
            global_query_names,
            {
                func_name: names
                for func_name, names in function_param_query_names.items()
                if names
            },
        )

    def collect_resource_query_names(self, node):
        """Collect resource names used directly in resource query calls."""
        query_names = set()
        visited = set()

        def visit_node(current):
            """Visit one AST value while collecting direct resource queries."""
            if current is None or isinstance(current, (str, int, float, bool)):
                return
            if isinstance(current, (list, tuple, set)):
                for item in current:
                    visit_node(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    visit_node(item)
                return

            current_id = id(current)
            if current_id in visited:
                return
            visited.add(current_id)

            if isinstance(current, FunctionCallNode):
                func_name = self.raw_function_call_name(current)
                raw_args = getattr(current, "arguments", getattr(current, "args", []))
                if (
                    func_name
                    in {
                        "textureSize",
                        "imageSize",
                        "textureSamples",
                        "imageSamples",
                        "textureQueryLevels",
                    }
                    and raw_args
                ):
                    resource_name = self.get_expression_name(raw_args[0])
                    if resource_name:
                        query_names.add(resource_name)

            if hasattr(current, "__dict__"):
                for key, value in vars(current).items():
                    if key in {"parent", "annotations"}:
                        continue
                    visit_node(value)

        visit_node(node)
        return query_names

    def query_collect_functions(self, root):
        """Collect function-like nodes from an AST subtree."""
        functions = []
        for node in self.query_walk_nodes(root):
            if hasattr(node, "body") and hasattr(node, "parameters"):
                functions.append(node)
        return functions

    def query_walk_nodes(self, root):
        """Yield AST nodes recursively while avoiding parent/annotation cycles."""
        visited = set()

        def walk(value):
            """Yield one AST value and recurse into child values."""
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for item in value.values():
                    yield from walk(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    yield from walk(item)
                return

            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)
            yield value

            if hasattr(value, "__dict__"):
                for key, child in vars(value).items():
                    if key in {"parent", "annotations"}:
                        continue
                    yield from walk(child)

        yield from walk(root)

    def raw_function_call_name(self, node):
        """Return a function call name without backend formatting."""
        func_expr = getattr(node, "function", getattr(node, "name", None))
        if hasattr(func_expr, "name"):
            return func_expr.name
        if isinstance(func_expr, str):
            return func_expr
        return None

    def get_parameter_type(self, param):
        """Return a parameter type from current or legacy AST shapes."""
        if hasattr(param, "param_type"):
            return param.param_type
        if hasattr(param, "vtype"):
            return param.vtype
        return "void"

    def get_variable_node_type(self, node):
        """Return a variable type from current or legacy AST shapes."""
        if hasattr(node, "var_type"):
            return node.var_type
        if hasattr(node, "vtype"):
            return node.vtype
        return None

    def query_metadata_name(self, resource_name):
        """Return the generated metadata variable name for a resource."""
        return f"{resource_name}_metadata"

    def query_type_name(self, type_name):
        """Return a string type name suitable for resource query decisions."""
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            return self.convert_type_node_to_string(type_name)
        return str(type_name)

    def query_array_suffix(self, type_name):
        """Return the array suffix for a resource type, if present."""
        type_name = self.query_type_name(type_name)
        if "[" not in type_name or "]" not in type_name:
            return ""
        return type_name[type_name.find("[") :]

    def query_metadata_type(self, type_name):
        """Return the metadata type used for a resource type."""
        return f"CglResourceQueryInfo{self.query_array_suffix(type_name)}"

    def query_metadata_expression(self, resource_expr):
        """Return the metadata expression paired with a resource expression."""
        if isinstance(resource_expr, ArrayAccessNode):
            base_expr = self.query_metadata_expression(resource_expr.array_expr)
            if base_expr is None:
                return None
            return f"{base_expr}[{self.visit(resource_expr.index_expr)}]"

        resource_name = self.get_expression_name(resource_expr)
        if resource_name:
            return self.query_metadata_name(resource_name)
        return None

    def require_helper_function(self, name, body):
        """Register a helper function body if it has not already been added."""
        self.helper_functions.setdefault(name, body)

    def needs_query_metadata(self, name, type_name):
        """Return whether a resource declaration needs query metadata."""
        current_function = getattr(self, "current_function_name", None)
        if current_function:
            query_params = getattr(self, "query_metadata_function_params", {})
            if name not in query_params.get(current_function, set()):
                return False
        elif name not in self.query_resource_names:
            return False
        resource_type = self.resource_base_type(self.query_type_name(type_name))
        if self.dimension_query_spec(resource_type) is None:
            return False
        self.resource_query_info_required = True
        return True

    def query_metadata_parameter(self, name, type_name):
        """Return an extra metadata parameter declaration when required."""
        if not self.needs_query_metadata(name, type_name):
            return None
        return self.format_typed_declarator(
            self.query_metadata_type(type_name), self.query_metadata_name(name)
        )

    def query_metadata_declaration(self, name, type_name):
        """Return a metadata variable declaration when required."""
        if self.needs_query_metadata(name, type_name):
            declaration = self.format_typed_declarator(
                self.query_metadata_type(type_name),
                self.query_metadata_name(name),
                dynamic_array_as_pointer=False,
            )
            return f"{declaration} = {{}}"
        return None

    def query_metadata_call_arguments(self, func_name, raw_args, args):
        """Expand call arguments with resource metadata arguments."""
        query_params = getattr(self, "query_metadata_function_params", {}).get(
            func_name, set()
        )
        if not query_params:
            return args

        functions_by_name = getattr(self, "query_functions_by_name", {})
        callee = functions_by_name.get(func_name)
        if callee is None:
            return args

        expanded_args = []
        params = getattr(callee, "parameters", getattr(callee, "params", []))
        for index, arg in enumerate(args):
            expanded_args.append(arg)
            if index >= len(params) or index >= len(raw_args):
                continue
            param_name = getattr(params[index], "name", None)
            if param_name not in query_params:
                continue
            metadata_arg = self.query_metadata_expression(raw_args[index])
            if metadata_arg:
                expanded_args.append(metadata_arg)
        return expanded_args

    def query_return_type(self, dimensions):
        """Return an integer vector type for a dimension query result."""
        if len(dimensions) == 1:
            return "int"
        return f"int{len(dimensions)}"

    def query_constructor(self, return_type, values):
        """Return a constructor expression for a query result."""
        if return_type == "int":
            return values[0]
        return f"make_{return_type}({', '.join(values)})"

    def query_dimension_expression(self, dimension, mip_arg):
        """Return a metadata field expression for one queried dimension."""
        value = f"info.{dimension}"
        if mip_arg is not None and dimension in {"width", "height", "depth"}:
            return f"cgl_lod_extent({value}, {mip_arg})"
        return value

    def query_helper_prefix(self):
        """Return helper code shared by mip-aware dimension queries."""
        return (
            "__device__ inline int cgl_lod_extent(int extent, int mipLevel)\n"
            "{\n"
            "    int shifted = extent >> mipLevel;\n"
            "    return shifted > 1 ? shifted : 1;\n"
            "}"
        )

    def dimension_query_spec(self, type_name):
        """Return dimension/mip/sample metadata for a queryable resource type."""
        specs = {
            "sampler1D": (("width",), True, False),
            "sampler2D": (("width", "height"), True, False),
            "sampler2DShadow": (("width", "height"), True, False),
            "sampler2DArray": (("width", "height", "elements"), True, False),
            "sampler2DArrayShadow": (
                ("width", "height", "elements"),
                True,
                False,
            ),
            "sampler3D": (("width", "height", "depth"), True, False),
            "samplerCube": (("width", "height"), True, False),
            "samplerCubeShadow": (("width", "height"), True, False),
            "samplerCubeArray": (("width", "height", "elements"), True, False),
            "samplerCubeArrayShadow": (
                ("width", "height", "elements"),
                True,
                False,
            ),
            "sampler2DMS": (("width", "height"), False, True),
            "sampler2DMSArray": (("width", "height", "elements"), False, True),
            "image2D": (("width", "height"), False, False),
            "iimage2D": (("width", "height"), False, False),
            "uimage2D": (("width", "height"), False, False),
            "image3D": (("width", "height", "depth"), False, False),
            "iimage3D": (("width", "height", "depth"), False, False),
            "uimage3D": (("width", "height", "depth"), False, False),
            "imageCube": (("width", "height"), False, False),
            "image2DArray": (("width", "height", "elements"), False, False),
            "iimage2DArray": (("width", "height", "elements"), False, False),
            "uimage2DArray": (("width", "height", "elements"), False, False),
            "image2DMS": (("width", "height"), False, True),
            "iimage2DMS": (("width", "height"), False, True),
            "uimage2DMS": (("width", "height"), False, True),
            "image2DMSArray": (("width", "height", "elements"), False, True),
            "iimage2DMSArray": (("width", "height", "elements"), False, True),
            "uimage2DMSArray": (("width", "height", "elements"), False, True),
        }
        spec = specs.get(type_name)
        if spec is None:
            return None
        dimensions, mip, samples = spec
        return {"dimensions": dimensions, "mip": mip, "samples": samples}

    def build_dimension_query_helper(self, helper_name, spec):
        """Build helper code for a texture/image dimension query."""
        return_type = self.query_return_type(spec["dimensions"])
        params = "CglResourceQueryInfo info"
        mip_arg = None
        if spec["mip"]:
            params += ", int mipLevel"
            mip_arg = "mipLevel"

        values = [
            self.query_dimension_expression(dimension, mip_arg)
            for dimension in spec["dimensions"]
        ]
        return_value = self.query_constructor(return_type, values)
        return (
            f"__device__ inline {return_type} {helper_name}({params})\n"
            "{\n"
            f"    return {return_value};\n"
            "}"
        )

    def build_sample_count_query_helper(self, helper_name):
        """Build helper code for a sample-count query."""
        return (
            f"__device__ inline int {helper_name}(CglResourceQueryInfo info)\n"
            "{\n"
            "    return info.samples;\n"
            "}"
        )

    def build_texture_query_levels_helper(self, helper_name, spec):
        """Build helper code for texture mip-level queries."""
        if not spec["mip"]:
            return (
                f"__device__ inline int {helper_name}(CglResourceQueryInfo info)\n"
                "{\n"
                "    return 1;\n"
                "}"
            )
        return (
            f"__device__ inline int {helper_name}(CglResourceQueryInfo info)\n"
            "{\n"
            "    return info.levels > 0 ? info.levels : 1;\n"
            "}"
        )

    def ensure_query_prefix_helper(self):
        """Ensure shared query helper code has been registered."""
        self.require_helper_function("cgl_lod_extent", self.query_helper_prefix())

    def is_sampled_resource_type(self, type_name):
        """Return whether a resource type is sampler-backed."""
        return isinstance(type_name, str) and type_name.startswith("sampler")

    def generate_dimension_query(self, func_name, raw_args, args):
        """Generate a helper call for a dimension query, if supported."""
        if not raw_args:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        spec = self.dimension_query_spec(resource_type)
        metadata_expr = self.query_metadata_expression(raw_args[0])
        if spec is None or metadata_expr is None:
            return None

        self.resource_query_info_required = True
        helper_name = f"cgl_{func_name}_{resource_type}"
        if spec["mip"]:
            self.ensure_query_prefix_helper()
        self.require_helper_function(
            helper_name, self.build_dimension_query_helper(helper_name, spec)
        )

        if spec["mip"]:
            lod = args[1] if len(args) > 1 else "0"
            return f"{helper_name}({metadata_expr}, {lod})"
        return f"{helper_name}({metadata_expr})"

    def generate_sample_count_query(self, func_name, raw_args, args):
        """Generate a helper call for a sample-count query, if supported."""
        if not raw_args:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        spec = self.dimension_query_spec(resource_type)
        metadata_expr = self.query_metadata_expression(raw_args[0])
        if spec is None or not spec["samples"] or metadata_expr is None:
            return None

        self.resource_query_info_required = True
        helper_name = f"cgl_{func_name}_{resource_type}"
        self.require_helper_function(
            helper_name, self.build_sample_count_query_helper(helper_name)
        )
        return f"{helper_name}({metadata_expr})"

    def generate_texture_query_levels(self, raw_args):
        """Generate a helper call for ``textureQueryLevels``, if supported."""
        if not raw_args:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if not self.is_sampled_resource_type(resource_type):
            return None
        spec = self.dimension_query_spec(resource_type)
        metadata_expr = self.query_metadata_expression(raw_args[0])
        if spec is None or metadata_expr is None:
            return None

        self.resource_query_info_required = True
        helper_name = f"cgl_textureQueryLevels_{resource_type}"
        self.require_helper_function(
            helper_name, self.build_texture_query_levels_helper(helper_name, spec)
        )
        return f"{helper_name}({metadata_expr})"
