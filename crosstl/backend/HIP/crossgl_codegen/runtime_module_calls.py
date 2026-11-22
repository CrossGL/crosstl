"""HIP host occupancy, function, module, texture reference, library, kernel and linker calls in the HIP to CrossGL converter, described as comments."""


class HipRuntimeModuleCallMixin:
    """Runtime module call members of ``HipToCrossGLConverter``."""

    def describe_hip_module_call(self, node, name, args):
        """Comments for an occupancy, function attribute or module call; None for any other name."""
        if name in {
            "hipOccupancyMaxPotentialBlockSize",
            "hipOccupancyMaxPotentialBlockSizeVariableSMem",
            "hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags",
        }:
            if len(args) >= 5:
                grid_output = self.format_runtime_raw_output_target(node.args[0])
                block_output = self.format_runtime_raw_output_target(node.args[1])
                grid_output_name = self.get_runtime_pointer_target_name(node.args[0])
                block_output_name = self.get_runtime_pointer_target_name(node.args[1])
                query_kind = {
                    "hipOccupancyMaxPotentialBlockSize": "maxPotentialBlockSize",
                    "hipOccupancyMaxPotentialBlockSizeVariableSMem": (
                        "maxPotentialBlockSizeVariableSMem"
                    ),
                    "hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags": (
                        "maxPotentialBlockSizeVariableSMemWithFlags"
                    ),
                }[name]
                query_operands = [args[2], args[3], args[4]]
                if len(args) >= 6:
                    query_operands.append(args[5])
                query_operands_text = ", ".join(query_operands)
                if grid_output_name is not None:
                    self.register_device_query_source(
                        grid_output_name,
                        f"occupancy.{query_kind}.grid({query_operands_text})",
                    )
                if block_output_name is not None:
                    self.register_device_query_source(
                        block_output_name,
                        f"occupancy.{query_kind}.block({query_operands_text})",
                    )
                comment = (
                    f"// HIP occupancy max potential block size: "
                    f"grid output: {grid_output}, block output: {block_output}, "
                    f"kernel: {args[2]}, dynamic shared memory: {args[3]}, "
                    f"block size limit: {args[4]}"
                )
                if len(args) >= 6:
                    comment += f", flags: {args[5]}"
                return [comment]
        elif name in {
            "hipOccupancyMaxActiveBlocksPerMultiprocessor",
            "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        }:
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                query_kind = (
                    "maxActiveBlocksPerMultiprocessorWithFlags"
                    if name.endswith("WithFlags")
                    else "maxActiveBlocksPerMultiprocessor"
                )
                query_operands = [args[1], args[2], args[3]]
                if len(args) >= 5:
                    query_operands.append(args[4])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"occupancy.{query_kind}({', '.join(query_operands)})",
                    )
                comment = (
                    f"// HIP occupancy active blocks per multiprocessor: "
                    f"output: {output}, kernel: {args[1]}, "
                    f"block size: {args[2]}, dynamic shared memory: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", flags: {args[4]}"
                return [comment]
        elif name == "hipGetFuncBySymbol":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP get function by symbol: output: {output}, symbol: {args[1]}"
                ]
        elif name == "hipFuncGetAttribute":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"function.attribute({args[1]}, {args[2]})",
                    )
                return [
                    f"// HIP function get attribute: output: {output}, "
                    f"attribute: {args[1]}, function: {args[2]}"
                ]
        elif name == "hipFuncGetAttributes":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_member_query_sources(
                        output_name,
                        {
                            member: f"function.attributes.{member}({args[1]})"
                            for member in self.HIP_FUNCTION_ATTRIBUTE_MEMBERS
                        },
                    )
                return [
                    f"// HIP function get attributes: output: {output}, "
                    f"function: {args[1]}"
                ]
        elif name == "hipFuncSetAttribute":
            if len(args) >= 3:
                return [
                    f"// HIP function set attribute: function: {args[0]}, "
                    f"attribute: {args[1]}, value: {args[2]}"
                ]
        elif name == "hipFuncSetCacheConfig":
            if len(args) >= 2:
                return [
                    f"// HIP function set cache config: function: {args[0]}, "
                    f"config: {args[1]}"
                ]
        elif name == "hipFuncSetSharedMemConfig":
            if len(args) >= 2:
                return [
                    f"// HIP function set shared memory config: function: {args[0]}, "
                    f"config: {args[1]}"
                ]
        elif name in {"hipModuleLoad", "hipModuleLoadData"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                source_label = "file" if name == "hipModuleLoad" else "image"
                return [
                    f"// HIP module load: output: {output}, "
                    f"{source_label}: {args[1]}"
                ]
        elif name == "hipModuleLoadDataEx":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP module load data ex: output: {output}, image: {args[1]}, "
                    f"options: {args[2]}, option keys: {args[3]}, "
                    f"option values: {args[4]}"
                ]
        elif name == "hipModuleLoadFatBinary":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP module load fat binary: output: {output}, "
                    f"fat binary: {args[1]}"
                ]
        elif name == "hipModuleUnload":
            if args:
                return [f"// HIP module unload: {args[0]}"]
        elif name == "hipModuleGetFunction":
            if len(args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP module get function: output: {output}, "
                    f"module: {args[1]}, name: {args[2]}"
                ]
        elif name == "hipModuleGetFunctionCount":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"module.functionCount({args[1]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP module get function count: output: {output}, "
                    f"module: {args[1]}"
                ]
        elif name == "hipModuleGetGlobal":
            if len(args) >= 4:
                pointer_output = self.format_runtime_pointer_target(node.args[0])
                size_output = self.format_runtime_pointer_target(node.args[1])
                size_output_name = self.get_runtime_pointer_target_name(node.args[1])
                self.clear_lvalue_metadata_source(node.args[0])
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"module.global.size({args[2]}, {args[3]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[1])
                return [
                    f"// HIP module get global: pointer output: {pointer_output}, "
                    f"size output: {size_output}, module: {args[2]}, name: {args[3]}"
                ]
        elif name == "hipModuleGetTexRef":
            if len(args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP module get texture reference: output: {output}, "
                    f"module: {args[1]}, name: {args[2]}"
                ]
        return None

    def describe_hip_texture_reference_call(self, node, name, args):
        """Comments for a texture binding or texture reference call; None for any other name."""
        if name == "hipBindTexture":
            if len(args) >= 5:
                offset = self.format_runtime_raw_output_target(node.args[0])
                return [
                    "// HIP texture reference bind: "
                    f"offset output: {offset}, texture: {args[1]}, "
                    f"pointer: {args[2]}, desc: {args[3]}, bytes: {args[4]}"
                ]
        elif name == "hipBindTexture2D":
            if len(args) >= 7:
                offset = self.format_runtime_raw_output_target(node.args[0])
                return [
                    "// HIP texture reference bind 2D: "
                    f"offset output: {offset}, texture: {args[1]}, "
                    f"pointer: {args[2]}, desc: {args[3]}, width: {args[4]}, "
                    f"height: {args[5]}, pitch: {args[6]}"
                ]
        elif name == "hipBindTextureToArray":
            if len(args) >= 3:
                return [
                    "// HIP texture reference bind array: "
                    f"texture: {args[0]}, array: {args[1]}, desc: {args[2]}"
                ]
        elif name == "hipBindTextureToMipmappedArray":
            if len(args) >= 3:
                return [
                    "// HIP texture reference bind mipmapped array: "
                    f"texture: {args[0]}, mipmapped array: {args[1]}, "
                    f"desc: {args[2]}"
                ]
        elif name == "hipGetTextureReference":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP get texture reference: output: {output}, "
                    f"symbol: {args[1]}"
                ]
        elif name == "hipGetTextureAlignmentOffset":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"textureReference.alignmentOffset({args[1]})",
                    )
                return [
                    "// HIP texture alignment offset query: "
                    f"output: {output}, texture: {args[1]}"
                ]
        elif name == "hipUnbindTexture":
            if args:
                return [f"// HIP texture reference unbind: {args[0]}"]
        elif name == "hipTexRefGetAddress":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    "// HIP texture reference get address: "
                    f"output: {output}, texture: {args[1]}"
                ]
        elif name == "hipTexRefGetAddressMode":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"textureReference.addressMode({args[1]}, {args[2]})",
                    )
                return [
                    "// HIP texture reference get address mode: "
                    f"output: {output}, texture: {args[1]}, dim: {args[2]}"
                ]
        elif name in {
            "hipTexRefGetArray",
            "hipTexRefGetBorderColor",
            "hipTexRefGetFilterMode",
            "hipTexRefGetFlags",
            "hipTexRefGetMaxAnisotropy",
            "hipTexRefGetMipmapFilterMode",
            "hipTexRefGetMipmapLevelBias",
            "hipTexRefGetMipMappedArray",
            "hipTexRefGetMipmappedArray",
        }:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                labels = {
                    "hipTexRefGetArray": "array",
                    "hipTexRefGetBorderColor": "border color",
                    "hipTexRefGetFilterMode": "filter mode",
                    "hipTexRefGetFlags": "flags",
                    "hipTexRefGetMaxAnisotropy": "max anisotropy",
                    "hipTexRefGetMipmapFilterMode": "mipmap filter mode",
                    "hipTexRefGetMipmapLevelBias": "mipmap level bias",
                    "hipTexRefGetMipMappedArray": "mipmapped array",
                    "hipTexRefGetMipmappedArray": "mipmapped array",
                }
                query_names = {
                    "hipTexRefGetFilterMode": "filterMode",
                    "hipTexRefGetFlags": "flags",
                    "hipTexRefGetMaxAnisotropy": "maxAnisotropy",
                    "hipTexRefGetMipmapFilterMode": "mipmapFilterMode",
                    "hipTexRefGetMipmapLevelBias": "mipmapLevelBias",
                }
                query_name = query_names.get(name)
                if output_name is not None and query_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"textureReference.{query_name}({args[1]})",
                    )
                elif output_name is not None and name == "hipTexRefGetBorderColor":
                    self.register_member_query_sources(
                        output_name,
                        {
                            f"[{index}]": (
                                f"textureReference.borderColor[{index}]({args[1]})"
                            )
                            for index in range(4)
                        },
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP texture reference get {labels[name]}: "
                    f"output: {output}, texture: {args[1]}"
                ]
        elif name == "hipTexRefGetFormat":
            if len(args) >= 3:
                format_output = self.format_runtime_raw_output_target(node.args[0])
                channels_output = self.format_runtime_raw_output_target(node.args[1])
                format_output_name = self.get_runtime_pointer_target_name(node.args[0])
                channels_output_name = self.get_runtime_pointer_target_name(
                    node.args[1]
                )
                if format_output_name is not None:
                    self.register_device_query_source(
                        format_output_name,
                        f"textureReference.format({args[2]})",
                    )
                if channels_output_name is not None:
                    self.register_device_query_source(
                        channels_output_name,
                        f"textureReference.channelCount({args[2]})",
                    )
                return [
                    "// HIP texture reference get format: "
                    f"format output: {format_output}, "
                    f"channels output: {channels_output}, texture: {args[2]}"
                ]
        elif name == "hipTexRefGetMipmapLevelClamp":
            if len(args) >= 3:
                min_output = self.format_runtime_raw_output_target(node.args[0])
                max_output = self.format_runtime_raw_output_target(node.args[1])
                min_output_name = self.get_runtime_pointer_target_name(node.args[0])
                max_output_name = self.get_runtime_pointer_target_name(node.args[1])
                if min_output_name is not None:
                    self.register_device_query_source(
                        min_output_name,
                        f"textureReference.mipmapLevelClamp.min({args[2]})",
                    )
                if max_output_name is not None:
                    self.register_device_query_source(
                        max_output_name,
                        f"textureReference.mipmapLevelClamp.max({args[2]})",
                    )
                return [
                    "// HIP texture reference get mipmap level clamp: "
                    f"min output: {min_output}, max output: {max_output}, "
                    f"texture: {args[2]}"
                ]
        elif name == "hipTexRefSetAddress":
            if len(args) >= 4:
                offset = self.format_runtime_raw_output_target(node.args[0])
                return [
                    "// HIP texture reference set address: "
                    f"offset output: {offset}, texture: {args[1]}, "
                    f"pointer: {args[2]}, bytes: {args[3]}"
                ]
        elif name == "hipTexRefSetAddress2D":
            if len(args) >= 4:
                return [
                    "// HIP texture reference set address 2D: "
                    f"texture: {args[0]}, desc: {args[1]}, pointer: {args[2]}, "
                    f"pitch: {args[3]}"
                ]
        elif name in {
            "hipTexRefSetAddressMode",
            "hipTexRefSetArray",
            "hipTexRefSetBorderColor",
            "hipTexRefSetFilterMode",
            "hipTexRefSetFlags",
            "hipTexRefSetFormat",
            "hipTexRefSetMaxAnisotropy",
            "hipTexRefSetMipmapFilterMode",
            "hipTexRefSetMipmapLevelBias",
            "hipTexRefSetMipmapLevelClamp",
            "hipTexRefSetMipmappedArray",
        }:
            texture = args[0] if args else "<missing>"
            if name == "hipTexRefSetAddressMode" and len(args) >= 3:
                return [
                    "// HIP texture reference set address mode: "
                    f"texture: {texture}, dim: {args[1]}, mode: {args[2]}"
                ]
            if name == "hipTexRefSetArray" and len(args) >= 3:
                return [
                    "// HIP texture reference set array: "
                    f"texture: {texture}, array: {args[1]}, flags: {args[2]}"
                ]
            if name == "hipTexRefSetBorderColor" and len(args) >= 2:
                return [
                    "// HIP texture reference set border color: "
                    f"texture: {texture}, color: {args[1]}"
                ]
            if name == "hipTexRefSetFilterMode" and len(args) >= 2:
                return [
                    "// HIP texture reference set filter mode: "
                    f"texture: {texture}, mode: {args[1]}"
                ]
            if name == "hipTexRefSetFlags" and len(args) >= 2:
                return [
                    "// HIP texture reference set flags: "
                    f"texture: {texture}, flags: {args[1]}"
                ]
            if name == "hipTexRefSetFormat" and len(args) >= 3:
                return [
                    "// HIP texture reference set format: "
                    f"texture: {texture}, format: {args[1]}, "
                    f"components: {args[2]}"
                ]
            if name == "hipTexRefSetMaxAnisotropy" and len(args) >= 2:
                return [
                    "// HIP texture reference set max anisotropy: "
                    f"texture: {texture}, value: {args[1]}"
                ]
            if name == "hipTexRefSetMipmapFilterMode" and len(args) >= 2:
                return [
                    "// HIP texture reference set mipmap filter mode: "
                    f"texture: {texture}, mode: {args[1]}"
                ]
            if name == "hipTexRefSetMipmapLevelBias" and len(args) >= 2:
                return [
                    "// HIP texture reference set mipmap level bias: "
                    f"texture: {texture}, bias: {args[1]}"
                ]
            if name == "hipTexRefSetMipmapLevelClamp" and len(args) >= 3:
                return [
                    "// HIP texture reference set mipmap level clamp: "
                    f"texture: {texture}, min: {args[1]}, max: {args[2]}"
                ]
            if name == "hipTexRefSetMipmappedArray" and len(args) >= 3:
                return [
                    "// HIP texture reference set mipmapped array: "
                    f"texture: {texture}, mipmapped array: {args[1]}, "
                    f"flags: {args[2]}"
                ]
        return None

    def describe_hip_library_call(self, node, name, args):
        """Comments for a driver entry point, library, kernel, linker, address range handle or module launch call; None for any other name."""
        if name == "hipGetDriverEntryPoint":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[1])
                status_output = self.format_runtime_raw_output_target(node.args[3])
                return [
                    f"// HIP get driver entry point: symbol: {args[0]}, "
                    f"output: {output}, flags: {args[2]}, "
                    f"status output: {status_output}"
                ]
        elif name in {"hipLibraryLoadData", "hipLibraryLoadFromFile"}:
            if len(args) >= 8:
                output = self.format_runtime_raw_output_target(node.args[0])
                source_label = "file" if name == "hipLibraryLoadFromFile" else "code"
                return [
                    f"// HIP library load: output: {output}, "
                    f"{source_label}: {args[1]}, jit options: {args[2]}, "
                    f"jit option values: {args[3]}, jit option count: {args[4]}, "
                    f"library options: {args[5]}, "
                    f"library option values: {args[6]}, "
                    f"library option count: {args[7]}"
                ]
        elif name == "hipLibraryUnload":
            if args:
                return [f"// HIP library unload: {args[0]}"]
        elif name == "hipLibraryGetKernel":
            if len(args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP library get kernel: output: {output}, "
                    f"library: {args[1]}, name: {args[2]}"
                ]
        elif name == "hipLibraryGetKernelCount":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"library.kernelCount({args[1]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP library get kernel count: output: {output}, "
                    f"library: {args[1]}"
                ]
        elif name == "hipLibraryEnumerateKernels":
            if len(args) >= 3:
                self.clear_lvalue_metadata_source(node.args[0])
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// HIP library enumerate kernels: output: {output}, "
                    f"max kernels: {args[1]}, library: {args[2]}"
                ]
        elif name == "hipKernelGetLibrary":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP kernel get library: output: {output}, "
                    f"kernel: {args[1]}"
                ]
        elif name == "hipKernelGetName":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [f"// HIP kernel get name: output: {output}, kernel: {args[1]}"]
        elif name == "hipKernelGetParamInfo":
            if len(args) >= 4:
                offset_output = self.format_runtime_pointer_target(node.args[2])
                size_output = self.format_runtime_pointer_target(node.args[3])
                offset_output_name = self.get_runtime_pointer_target_name(node.args[2])
                size_output_name = self.get_runtime_pointer_target_name(node.args[3])
                if offset_output_name is not None:
                    self.register_device_query_source(
                        offset_output_name,
                        f"kernel.param.offset({args[0]}, {args[1]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[2])
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"kernel.param.size({args[0]}, {args[1]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[3])
                return [
                    f"// HIP kernel get parameter info: kernel: {args[0]}, "
                    f"param index: {args[1]}, offset output: {offset_output}, "
                    f"size output: {size_output}"
                ]
        elif name == "hipKernelGetFunction":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP kernel get function: output: {output}, "
                    f"kernel: {args[1]}"
                ]
        elif name == "hipKernelGetAttribute":
            if len(args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"kernel.attribute({args[1]}, {args[2]}, {args[3]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP kernel get attribute: output: {output}, "
                    f"attribute: {args[1]}, kernel: {args[2]}, device: {args[3]}"
                ]
        elif name == "hipKernelSetAttribute":
            if len(args) >= 4:
                return [
                    f"// HIP kernel set attribute: attribute: {args[0]}, "
                    f"value: {args[1]}, kernel: {args[2]}, device: {args[3]}"
                ]
        elif name == "hipLinkCreate":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[3])
                return [
                    f"// HIP link create: options: {args[0]}, "
                    f"option keys: {args[1]}, option values: {args[2]}, "
                    f"state output: {output}"
                ]
        elif name == "hipLinkAddFile":
            if len(args) >= 6:
                return [
                    f"// HIP link add file: state: {args[0]}, input type: {args[1]}, "
                    f"path: {args[2]}, options: {args[3]}, option keys: {args[4]}, "
                    f"option values: {args[5]}"
                ]
        elif name == "hipLinkAddData":
            if len(args) >= 8:
                return [
                    f"// HIP link add data: state: {args[0]}, input type: {args[1]}, "
                    f"image: {args[2]}, bytes: {args[3]}, name: {args[4]}, "
                    f"options: {args[5]}, option keys: {args[6]}, "
                    f"option values: {args[7]}"
                ]
        elif name == "hipLinkComplete":
            if len(args) >= 3:
                self.clear_lvalue_metadata_source(node.args[1])
                binary_output = self.format_runtime_pointer_target(node.args[1])
                self.clear_lvalue_metadata_source(node.args[2])
                size_output = self.format_runtime_pointer_target(node.args[2])
                size_output_name = self.get_runtime_pointer_target_name(node.args[2])
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"link.complete.size({args[0]})",
                    )
                return [
                    f"// HIP link complete: state: {args[0]}, "
                    f"binary output: {binary_output}, size output: {size_output}"
                ]
        elif name == "hipLinkDestroy":
            if args:
                return [f"// HIP link destroy: state: {args[0]}"]
        elif name == "hipMemGetHandleForAddressRange":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory get handle for address range: output: {output}, "
                    f"device pointer: {args[1]}, bytes: {args[2]}, "
                    f"handle type: {args[3]}, flags: {args[4]}"
                ]
        elif name == "hipModuleLaunchKernel":
            if len(args) >= 11:
                return [
                    f"// HIP module launch kernel: function: {args[0]}, "
                    f"grid: ({args[1]}, {args[2]}, {args[3]}), "
                    f"block: ({args[4]}, {args[5]}, {args[6]}), "
                    f"shared memory: {args[7]}, stream: {args[8]}, "
                    f"params: {args[9]}, extra: {args[10]}"
                ]
        elif name in {"hipExtModuleLaunchKernel", "hipHccModuleLaunchKernel"}:
            if len(args) >= 13:
                label = "extended" if name == "hipExtModuleLaunchKernel" else "HCC"
                comment = (
                    f"// HIP {label} module launch kernel: function: {args[0]}, "
                    f"global work size: ({args[1]}, {args[2]}, {args[3]}), "
                    f"local work size: ({args[4]}, {args[5]}, {args[6]}), "
                    f"shared memory: {args[7]}, stream: {args[8]}, "
                    f"params: {args[9]}, extra: {args[10]}, "
                    f"start event: {args[11]}, stop event: {args[12]}"
                )
                if len(args) >= 14:
                    comment += f", flags: {args[13]}"
                return [comment]
        elif name == "hipLaunchCooperativeKernel":
            if len(args) >= 6:
                return [
                    f"// HIP cooperative kernel launch: function: {args[0]}, "
                    f"grid: {args[1]}, block: {args[2]}, params: {args[3]}, "
                    f"shared memory: {args[4]}, stream: {args[5]}"
                ]
        elif name == "hipLaunchCooperativeKernelMultiDevice":
            if len(args) >= 3:
                return [
                    f"// HIP cooperative multi-device launch: params: {args[0]}, "
                    f"devices: {args[1]}, flags: {args[2]}"
                ]
        elif name == "hipModuleLaunchCooperativeKernel":
            if len(args) >= 10:
                return [
                    f"// HIP module cooperative kernel launch: function: {args[0]}, "
                    f"grid: ({args[1]}, {args[2]}, {args[3]}), "
                    f"block: ({args[4]}, {args[5]}, {args[6]}), "
                    f"shared memory: {args[7]}, stream: {args[8]}, "
                    f"params: {args[9]}"
                ]
        elif name == "hipModuleLaunchCooperativeKernelMultiDevice":
            if len(args) >= 3:
                return [
                    f"// HIP module cooperative multi-device launch: "
                    f"params: {args[0]}, devices: {args[1]}, flags: {args[2]}"
                ]
        return None
