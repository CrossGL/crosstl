using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace ComputeSharp.Portable.Native;

/// <summary>Binding kinds of <c>cs_binding</c> (see native/csbackend.h).</summary>
internal enum NativeBindingKind : uint
{
    Buffer = 1,
    Texture = 2,
    Image = 3,
    Sampler = 4,
    ConstantBytes = 5,
    UniformBuffer = 6,
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDeviceInfo
{
    public uint AbiVersion;
    public uint IsHardware;
    public uint SupportsFp64;
    public uint UnifiedMemory;
    public ulong DedicatedMemory;
    public ulong SharedMemory;
    public uint ComputeUnits;
    public uint WavefrontSize;
    public uint MaxThreadsPerGroup;
    public uint MaxGroupSizeX;
    public uint MaxGroupSizeY;
    public uint MaxGroupSizeZ;
    public uint Reserved;
    public fixed byte Name[256];
    public fixed byte Api[64];
    public fixed byte Driver[512];
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeBinding
{
    public NativeBindingKind Kind;
    public uint Slot;
    public uint Format;
    public uint Access;
    public IntPtr Object;
    public ulong Offset;
    public ulong Size;
    public void* Data;
}

/// <summary>
/// P/Invoke declarations for the native backend library. The library name is resolved
/// to the Metal or OpenGL implementation selected by <see cref="BackendSelection"/>.
/// </summary>
internal static unsafe partial class CsBackend
{
    public const string LibraryName = "csbackend";
    public const uint AbiVersion = 1;
    public const uint Texture1D = 1;
    public const uint Texture2D = 2;
    public const uint Texture3D = 3;
    public const uint UsageShaderRead = 1;
    public const uint UsageShaderWrite = 2;
    public const uint FilterLinear = 1;
    public const uint AddressMirror = 2;

    static CsBackend()
    {
        NativeLibrary.SetDllImportResolver(typeof(CsBackend).Assembly, Resolve);
    }

    /// <summary>Forces the resolver registration before the first native call.</summary>
    public static void EnsureResolver()
    {
    }

    private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != LibraryName)
        {
            return IntPtr.Zero;
        }

        string path = BackendSelection.NativeLibraryPath;
        if (!File.Exists(path))
        {
            throw new DllNotFoundException($"The {BackendSelection.Backend} backend library was not found at '{path}'.");
        }

        return NativeLibrary.Load(path);
    }

    [LibraryImport(LibraryName, EntryPoint = "cs_abi_version")]
    public static partial uint AbiVersionOf();

    [LibraryImport(LibraryName, EntryPoint = "cs_device_create")]
    public static partial int DeviceCreate(out IntPtr device, NativeDeviceInfo* info, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_device_destroy")]
    public static partial void DeviceDestroy(IntPtr device);

    [LibraryImport(LibraryName, EntryPoint = "cs_format_supported")]
    public static partial int FormatSupported(IntPtr device, uint dimension, uint format, uint usage);

    [LibraryImport(LibraryName, EntryPoint = "cs_buffer_create")]
    public static partial int BufferCreate(IntPtr device, ulong size, out IntPtr buffer, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_buffer_destroy")]
    public static partial void BufferDestroy(IntPtr device, IntPtr buffer);

    [LibraryImport(LibraryName, EntryPoint = "cs_buffer_write")]
    public static partial int BufferWrite(IntPtr device, IntPtr buffer, ulong offset, void* data, ulong size, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_buffer_read")]
    public static partial int BufferRead(IntPtr device, IntPtr buffer, ulong offset, void* data, ulong size, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_texture_create")]
    public static partial int TextureCreate(IntPtr device, uint dimension, uint format, uint width, uint height, uint depth, uint usage, out IntPtr texture, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_texture_destroy")]
    public static partial void TextureDestroy(IntPtr device, IntPtr texture);

    [LibraryImport(LibraryName, EntryPoint = "cs_texture_write")]
    public static partial int TextureWrite(IntPtr device, IntPtr texture, uint x, uint y, uint z, uint width, uint height, uint depth, void* data, ulong rowPitch, ulong slicePitch, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_texture_read")]
    public static partial int TextureRead(IntPtr device, IntPtr texture, uint x, uint y, uint z, uint width, uint height, uint depth, void* data, ulong rowPitch, ulong slicePitch, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_sampler_create")]
    public static partial int SamplerCreate(IntPtr device, uint filter, uint addressMode, out IntPtr sampler, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_sampler_destroy")]
    public static partial void SamplerDestroy(IntPtr device, IntPtr sampler);

    [LibraryImport(LibraryName, EntryPoint = "cs_pipeline_create")]
    public static partial int PipelineCreate(IntPtr device, byte* source, byte* entryPoint, uint threadsX, uint threadsY, uint threadsZ, out IntPtr pipeline, byte* log, int logCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_pipeline_destroy")]
    public static partial void PipelineDestroy(IntPtr device, IntPtr pipeline);

    [LibraryImport(LibraryName, EntryPoint = "cs_list_create")]
    public static partial int ListCreate(IntPtr device, out IntPtr list, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_list_dispatch")]
    public static partial int ListDispatch(IntPtr list, IntPtr pipeline, uint groupsX, uint groupsY, uint groupsZ, NativeBinding* bindings, uint bindingCount, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_list_barrier")]
    public static partial int ListBarrier(IntPtr list, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_list_copy_buffer")]
    public static partial int ListCopyBuffer(IntPtr list, IntPtr source, ulong sourceOffset, IntPtr destination, ulong destinationOffset, ulong size, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_list_fill_buffer")]
    public static partial int ListFillBuffer(IntPtr list, IntPtr buffer, ulong offset, ulong size, uint value, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_list_copy_buffer_to_texture")]
    public static partial int ListCopyBufferToTexture(IntPtr list, IntPtr buffer, ulong offset, ulong rowPitch, ulong slicePitch, IntPtr texture, uint x, uint y, uint z, uint width, uint height, uint depth, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_list_copy_texture")]
    public static partial int ListCopyTexture(IntPtr list, IntPtr source, uint sourceX, uint sourceY, uint sourceZ, IntPtr destination, uint destinationX, uint destinationY, uint destinationZ, uint width, uint height, uint depth, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_list_submit")]
    public static partial int ListSubmit(IntPtr list, byte* error, int errorCapacity);

    [LibraryImport(LibraryName, EntryPoint = "cs_list_destroy")]
    public static partial void ListDestroy(IntPtr list);
}
