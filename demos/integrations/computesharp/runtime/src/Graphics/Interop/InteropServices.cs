using System;
using System.Runtime.InteropServices;
using ComputeSharp.Resources;

namespace ComputeSharp.Interop;

/// <summary>
/// Direct3D 12 interop. Translated backends have no Direct3D 12 objects; these methods throw
/// <see cref="PlatformNotSupportedException"/> and the Try variants return E_NOINTERFACE. The
/// tests that exercise them are classified as Windows-specific and are not run on these targets.
/// </summary>
public static unsafe class InteropServices
{
    private const int E_NOINTERFACE = unchecked((int)0x80004002);

    /// <summary>Gets the underlying Direct3D 12 device (not available on translated backends).</summary>
    public static void GetID3D12Device(GraphicsDevice device, Guid* riid, void** ppvObject) => throw Unsupported();

    /// <summary>Gets the underlying Direct3D 12 resource (not available on translated backends).</summary>
    public static void GetID3D12Resource<T>(Buffer<T> buffer, Guid* riid, void** ppvObject)
        where T : unmanaged => throw Unsupported();

    /// <summary>Gets the underlying Direct3D 12 resource (not available on translated backends).</summary>
    public static void GetID3D12Resource<T>(Texture1D<T> texture, Guid* riid, void** ppvObject)
        where T : unmanaged => throw Unsupported();

    /// <summary>Gets the underlying Direct3D 12 resource (not available on translated backends).</summary>
    public static void GetID3D12Resource<T>(Texture2D<T> texture, Guid* riid, void** ppvObject)
        where T : unmanaged => throw Unsupported();

    /// <summary>Gets the underlying Direct3D 12 resource (not available on translated backends).</summary>
    public static void GetID3D12Resource<T>(Texture3D<T> texture, Guid* riid, void** ppvObject)
        where T : unmanaged => throw Unsupported();

    /// <summary>Gets the underlying Direct3D 12 resource (not available on translated backends).</summary>
    public static void GetID3D12Resource<T>(TransferBuffer<T> buffer, Guid* riid, void** ppvObject)
        where T : unmanaged => throw Unsupported();

    /// <summary>Gets the underlying Direct3D 12 resource (not available on translated backends).</summary>
    public static void GetID3D12Resource<T>(TransferTexture2D<T> texture, Guid* riid, void** ppvObject)
        where T : unmanaged => throw Unsupported();

    /// <summary>Gets the underlying Direct3D 12 resource (not available on translated backends).</summary>
    public static void GetID3D12Resource<T>(TransferTexture3D<T> texture, Guid* riid, void** ppvObject)
        where T : unmanaged => throw Unsupported();

    /// <summary>Tries to get the underlying Direct3D 12 device.</summary>
    public static int TryGetID3D12Device(GraphicsDevice device, Guid* riid, void** ppvObject) => NoInterface(ppvObject);

    /// <summary>Tries to get the underlying Direct3D 12 resource.</summary>
    public static int TryGetID3D12Resource<T>(Buffer<T> buffer, Guid* riid, void** ppvObject)
        where T : unmanaged => NoInterface(ppvObject);

    /// <summary>Tries to get the underlying Direct3D 12 resource.</summary>
    public static int TryGetID3D12Resource<T>(Texture1D<T> texture, Guid* riid, void** ppvObject)
        where T : unmanaged => NoInterface(ppvObject);

    /// <summary>Tries to get the underlying Direct3D 12 resource.</summary>
    public static int TryGetID3D12Resource<T>(Texture2D<T> texture, Guid* riid, void** ppvObject)
        where T : unmanaged => NoInterface(ppvObject);

    /// <summary>Tries to get the underlying Direct3D 12 resource.</summary>
    public static int TryGetID3D12Resource<T>(Texture3D<T> texture, Guid* riid, void** ppvObject)
        where T : unmanaged => NoInterface(ppvObject);

    /// <summary>Tries to get the underlying Direct3D 12 resource.</summary>
    public static int TryGetID3D12Resource<T>(TransferBuffer<T> buffer, Guid* riid, void** ppvObject)
        where T : unmanaged => NoInterface(ppvObject);

    /// <summary>Tries to get the underlying Direct3D 12 resource.</summary>
    public static int TryGetID3D12Resource<T>(TransferTexture2D<T> texture, Guid* riid, void** ppvObject)
        where T : unmanaged => NoInterface(ppvObject);

    /// <summary>Tries to get the underlying Direct3D 12 resource.</summary>
    public static int TryGetID3D12Resource<T>(TransferTexture3D<T> texture, Guid* riid, void** ppvObject)
        where T : unmanaged => NoInterface(ppvObject);

    private static PlatformNotSupportedException Unsupported()
    {
        return new PlatformNotSupportedException("Direct3D 12 interop is not available on the translated Metal/OpenGL backends.");
    }

    private static int NoInterface(void** ppvObject)
    {
        if (ppvObject is not null)
        {
            *ppvObject = null;
        }

        return E_NOINTERFACE;
    }
}

/// <summary>Configures the Direct3D 12 memory allocator (not available on translated backends).</summary>
public static class AllocationServices
{
    /// <summary>Configures a D3D12MA allocator factory. Translated backends have no Direct3D 12 allocator.</summary>
    public static void ConfigureAllocatorFactory(ICustomQueryInterface allocatorFactory)
    {
        ArgumentNullException.ThrowIfNull(allocatorFactory);

        throw new PlatformNotSupportedException("The D3D12 memory allocator is not available on the translated Metal/OpenGL backends.");
    }
}
