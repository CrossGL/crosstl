using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ComputeSharp.Graphics.Resources.Enums;
using ComputeSharp.Interop;
using ComputeSharp.Portable;
using ComputeSharp.Portable.Native;

#pragma warning disable CA1063

namespace ComputeSharp.Resources
{
    /// <summary>A <see langword="class"/> representing a typed 1D texture in host memory used for transfers.</summary>
    public abstract unsafe partial class TransferTexture1D<T> : IReferenceTrackedObject, IGraphicsResource, IMemoryOwner<T>
        where T : unmanaged
    {
        private ReferenceTracker referenceTracker;
        private T* mappedData;

        private protected TransferTexture1D(GraphicsDevice device, int width, ResourceType resourceType, AllocationMode allocationMode)
        {
            using ReferenceTracker.Lease _0 = ReferenceTracker.Create(this, out this.referenceTracker);

            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, D3D12Limits.Texture1DDimension);

            using ReferenceTracker.Lease _1 = device.GetReferenceTracker().GetLease();

            device.ThrowIfDeviceLost();

            if (!device.IsTextureSupported<T>(CsBackend.Texture1D, CsBackend.UsageShaderRead))
            {
                UnsupportedTextureTypeException.ThrowForTexture1D<T>();
            }

            GraphicsDevice = device;
            Width = width;

            nuint size = (nuint)width * (nuint)sizeof(T);

            this.mappedData = (T*)NativeMemory.AlignedAlloc(size, 512);
            NativeMemory.Clear(this.mappedData, size);
        }

        /// <summary>Gets the <see cref="ComputeSharp.GraphicsDevice"/> associated with the current instance.</summary>
        public GraphicsDevice GraphicsDevice { get; }

        /// <summary>Gets the width of the current texture.</summary>
        public int Width { get; }

        internal T* MappedData => this.mappedData;

        /// <inheritdoc/>
        public Memory<T> Memory
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                using ReferenceTracker.Lease _0 = GetReferenceTracker().GetLease();

                return new MemoryManager(this).Memory;
            }
        }

        /// <summary>Gets a <see cref="Span{T}"/> over the texture data.</summary>
        public Span<T> Span
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                using ReferenceTracker.Lease _0 = GetReferenceTracker().GetLease();

                return new(this.mappedData, Width);
            }
        }

        void IReferenceTrackedObject.DangerousOnDispose()
        {
            NativeMemory.AlignedFree(this.mappedData);

            this.mappedData = null;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void ThrowIfDeviceMismatch(GraphicsDevice device)
        {
            if (GraphicsDevice != device)
            {
                GraphicsDeviceMismatchException.Throw(this, device);
            }
        }

        private sealed class MemoryManager(TransferTexture1D<T> buffer) : MemoryManager<T>
        {
            private readonly TransferTexture1D<T> buffer = buffer;

            public override Memory<T> Memory
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => CreateMemory(this.buffer.Width);
            }

            public override Span<T> GetSpan()
            {
                return this.buffer.Span;
            }

            public override MemoryHandle Pin(int elementIndex = 0)
            {
                default(ArgumentOutOfRangeException).ThrowIfNotZero(elementIndex);

                using ReferenceTracker.Lease _0 = this.buffer.GetReferenceTracker().GetLease();

                return new(this.buffer.mappedData);
            }

            public override void Unpin()
            {
            }

            protected override void Dispose(bool disposing)
            {
            }
        }
    }

    /// <summary>A <see langword="class"/> representing a typed 2D texture in host memory used for transfers.</summary>
    public abstract unsafe partial class TransferTexture2D<T> : IReferenceTrackedObject, IGraphicsResource
        where T : unmanaged
    {
        private ReferenceTracker referenceTracker;
        private T* mappedData;

        private protected TransferTexture2D(GraphicsDevice device, int width, int height, ResourceType resourceType, AllocationMode allocationMode)
        {
            using ReferenceTracker.Lease _0 = ReferenceTracker.Create(this, out this.referenceTracker);

            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, D3D12Limits.Texture2DDimension);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, D3D12Limits.Texture2DDimension);

            using ReferenceTracker.Lease _1 = device.GetReferenceTracker().GetLease();

            device.ThrowIfDeviceLost();

            if (!device.IsTextureSupported<T>(CsBackend.Texture2D, CsBackend.UsageShaderRead))
            {
                UnsupportedTextureTypeException.ThrowForTexture2D<T>();
            }

            GraphicsDevice = device;
            Width = width;
            Height = height;

            // Match the Direct3D 12 copyable footprint, which pads rows to 256 bytes.
            RowPitch = D3D12Limits.AlignPitch((long)width * sizeof(T));

            nuint size = (nuint)RowPitch * (nuint)height;

            this.mappedData = (T*)NativeMemory.AlignedAlloc(size, 512);
            NativeMemory.Clear(this.mappedData, size);
        }

        /// <summary>Gets the <see cref="ComputeSharp.GraphicsDevice"/> associated with the current instance.</summary>
        public GraphicsDevice GraphicsDevice { get; }

        /// <summary>Gets the width of the current texture.</summary>
        public int Width { get; }

        /// <summary>Gets the height of the current texture.</summary>
        public int Height { get; }

        internal int RowPitch { get; }

        internal T* MappedData => this.mappedData;

        /// <summary>Gets the <see cref="TextureView2D{T}"/> instance that maps the current texture data.</summary>
        public TextureView2D<T> View
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                using ReferenceTracker.Lease _0 = GetReferenceTracker().GetLease();

                return new(this.mappedData, Width, Height, RowPitch);
            }
        }

        void IReferenceTrackedObject.DangerousOnDispose()
        {
            NativeMemory.AlignedFree(this.mappedData);

            this.mappedData = null;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void ThrowIfDeviceMismatch(GraphicsDevice device)
        {
            if (GraphicsDevice != device)
            {
                GraphicsDeviceMismatchException.Throw(this, device);
            }
        }
    }

    /// <summary>A <see langword="class"/> representing a typed 3D texture in host memory used for transfers.</summary>
    public abstract unsafe partial class TransferTexture3D<T> : IReferenceTrackedObject, IGraphicsResource
        where T : unmanaged
    {
        private ReferenceTracker referenceTracker;
        private T* mappedData;

        private protected TransferTexture3D(GraphicsDevice device, int width, int height, int depth, ResourceType resourceType, AllocationMode allocationMode)
        {
            using ReferenceTracker.Lease _0 = ReferenceTracker.Create(this, out this.referenceTracker);

            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, D3D12Limits.Texture3DDimension);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, D3D12Limits.Texture3DDimension);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(depth, 1, D3D12Limits.Texture3DDimension);

            using ReferenceTracker.Lease _1 = device.GetReferenceTracker().GetLease();

            device.ThrowIfDeviceLost();

            if (!device.IsTextureSupported<T>(CsBackend.Texture3D, CsBackend.UsageShaderRead))
            {
                UnsupportedTextureTypeException.ThrowForTexture3D<T>();
            }

            GraphicsDevice = device;
            Width = width;
            Height = height;
            Depth = depth;
            RowPitch = D3D12Limits.AlignPitch((long)width * sizeof(T));

            nuint size = (nuint)RowPitch * (nuint)height * (nuint)depth;

            this.mappedData = (T*)NativeMemory.AlignedAlloc(size, 512);
            NativeMemory.Clear(this.mappedData, size);
        }

        /// <summary>Gets the <see cref="ComputeSharp.GraphicsDevice"/> associated with the current instance.</summary>
        public GraphicsDevice GraphicsDevice { get; }

        /// <summary>Gets the width of the current texture.</summary>
        public int Width { get; }

        /// <summary>Gets the height of the current texture.</summary>
        public int Height { get; }

        /// <summary>Gets the depth of the current texture.</summary>
        public int Depth { get; }

        internal int RowPitch { get; }

        internal T* MappedData => this.mappedData;

        /// <summary>Gets the <see cref="TextureView3D{T}"/> instance that maps the current texture data.</summary>
        public TextureView3D<T> View
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                using ReferenceTracker.Lease _0 = GetReferenceTracker().GetLease();

                return new(this.mappedData, Width, Height, Depth, RowPitch);
            }
        }

        void IReferenceTrackedObject.DangerousOnDispose()
        {
            NativeMemory.AlignedFree(this.mappedData);

            this.mappedData = null;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void ThrowIfDeviceMismatch(GraphicsDevice device)
        {
            if (GraphicsDevice != device)
            {
                GraphicsDeviceMismatchException.Throw(this, device);
            }
        }
    }
}
