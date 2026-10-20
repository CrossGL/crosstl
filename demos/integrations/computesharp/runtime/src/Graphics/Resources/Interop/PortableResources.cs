using System;
using ComputeSharp.Interop;

namespace ComputeSharp.Resources.Interop
{
    /// <summary>The kind of native object a shader resource binds.</summary>
    internal enum PortableResourceKind
    {
        ConstantBuffer,
        StructuredBuffer,
        Texture,
        ReadWriteTexture,
    }

    /// <summary>A validated native view of a resource bound to a dispatch.</summary>
    internal readonly struct PortableResourceView
    {
        public PortableResourceKind Kind { get; init; }

        public IntPtr Handle { get; init; }

        /// <summary>Gets the DXGI format of textures, or zero for buffers.</summary>
        public uint Format { get; init; }

        /// <summary>Gets the texture dimension (1, 2 or 3), or zero for buffers.</summary>
        public uint Dimension { get; init; }

        /// <summary>Gets the element count of buffers.</summary>
        public int Length { get; init; }

        /// <summary>Gets the size in bytes of buffers.</summary>
        public long SizeInBytes { get; init; }

        public long ResourceId { get; init; }

        public string Description { get; init; }
    }

    /// <summary>A resource that can be bound to a translated shader.</summary>
    internal interface IPortableReadOnlyResource
    {
        PortableResourceView ValidateAndGetView(GraphicsDevice device, out ReferenceTracker.Lease lease);
    }

    /// <summary>A writable texture that supports clears and state transitions.</summary>
    internal interface IPortableReadWriteResource : IPortableReadOnlyResource
    {
        PortableResourceView ValidateAndGetViewForClear(GraphicsDevice device, out bool isNormalized, out ReferenceTracker.Lease lease);

        void ValidateAndTransition(GraphicsDevice device, ResourceState resourceState, out ReferenceTracker.Lease lease);
    }
}

namespace ComputeSharp.Portable
{
    /// <summary>Direct3D 12 limits that the upstream API validates against.</summary>
    internal static class D3D12Limits
    {
        public const int ConstantBufferElementCount = 4096;
        public const int ConstantBufferDataPlacementAlignment = 256;
        public const int ConstantBufferPartialUpdateAlignment = 16;
        public const int Texture1DDimension = 16384;
        public const int Texture2DDimension = 16384;
        public const int Texture3DDimension = 2048;
        public const int MaxThreadGroupsPerDimension = 65535;
        public const int TextureDataPitchAlignment = 256;

        public static int AlignPitch(long rowSizeInBytes)
        {
            return checked((int)((rowSizeInBytes + TextureDataPitchAlignment - 1) & ~(long)(TextureDataPitchAlignment - 1)));
        }
    }
}
