using System;
using System.Runtime.CompilerServices;
using ComputeSharp.Graphics.Helpers;
using ComputeSharp.Graphics.Resources.Enums;
using ComputeSharp.Interop;
using ComputeSharp.Portable;
using ComputeSharp.Portable.Diagnostics;
using ComputeSharp.Portable.Native;
using ComputeSharp.Resources.Interop;

#pragma warning disable CA1063

namespace ComputeSharp.Portable
{
    /// <summary>A native texture and its host transfer operations.</summary>
    internal sealed unsafe class NativeTexture
    {
        private readonly GraphicsDevice device;
        private IntPtr handle;

        public NativeTexture(GraphicsDevice device, uint dimension, uint format, int width, int height, int depth, bool readWrite, int elementSize)
        {
            this.device = device;
            Dimension = dimension;
            Format = format;
            ElementSize = elementSize;
            ResourceId = EvidenceLog.NextResourceId();
            uint usage = CsBackend.UsageShaderRead | (readWrite ? CsBackend.UsageShaderWrite : 0);
            this.handle = device.Native.CreateTexture(dimension, format, (uint)width, (uint)height, (uint)depth, usage);
        }

        public IntPtr Handle => this.handle;

        public uint Dimension { get; }

        public uint Format { get; }

        public int ElementSize { get; }

        public long ResourceId { get; }

        public void Read(int x, int y, int z, int width, int height, int depth, void* destination, long rowPitch, long slicePitch, string description)
        {
            this.device.Native.ReadTexture(this.handle, (uint)x, (uint)y, (uint)z, (uint)width, (uint)height, (uint)depth, (IntPtr)destination, (ulong)rowPitch, (ulong)slicePitch);
            if (EvidenceLog.Enabled)
            {
                long bytes = (slicePitch * (depth - 1)) + (rowPitch * (height - 1)) + ((long)width * ElementSize);
                EvidenceLog.TextureReadback(description, ResourceId, (IntPtr)destination, bytes, Format, width, height, depth, rowPitch);
            }
        }

        public void Write(int x, int y, int z, int width, int height, int depth, void* source, long rowPitch, long slicePitch)
        {
            this.device.Native.WriteTexture(this.handle, (uint)x, (uint)y, (uint)z, (uint)width, (uint)height, (uint)depth, (IntPtr)source, (ulong)rowPitch, (ulong)slicePitch);
        }

        public void CopyTo(NativeTexture destination, int sourceX, int sourceY, int sourceZ, int destinationX, int destinationY, int destinationZ, int width, int height, int depth)
        {
            using NativeCommandList list = new(this.device.Native);

            list.CopyTexture(this.handle, (uint)sourceX, (uint)sourceY, (uint)sourceZ, destination.handle, (uint)destinationX, (uint)destinationY, (uint)destinationZ, (uint)width, (uint)height, (uint)depth);
            list.Submit();
        }

        public void Destroy()
        {
            this.device.Native.DestroyTexture(this.handle);
            this.handle = IntPtr.Zero;
        }
    }

    /// <summary>The tracked state of a texture, mirroring the Direct3D 12 resource states the upstream API exposes.</summary>
    internal enum TextureState
    {
        Common,
        ReadOnly,
        UnorderedAccess,
    }
}

namespace ComputeSharp.Resources
{
    /// <summary>A <see langword="class"/> representing a typed 1D texture stored in device memory.</summary>
    public abstract unsafe partial class Texture1D<T> : IReferenceTrackedObject, IGraphicsResource, IPortableReadWriteResource
        where T : unmanaged
    {
        private ReferenceTracker referenceTracker;
        private readonly NativeTexture texture;
        private readonly bool isReadWrite;
        private TextureState state;

        private protected Texture1D(GraphicsDevice device, int width, ResourceType resourceType, AllocationMode allocationMode)
        {
            using ReferenceTracker.Lease _0 = ReferenceTracker.Create(this, out this.referenceTracker);

            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, D3D12Limits.Texture1DDimension);

            using ReferenceTracker.Lease _1 = device.GetReferenceTracker().GetLease();

            device.ThrowIfDeviceLost();

            this.isReadWrite = resourceType == ResourceType.ReadWrite;
            uint usage = CsBackend.UsageShaderRead | (this.isReadWrite ? CsBackend.UsageShaderWrite : 0);
            if (!device.IsTextureSupported<T>(CsBackend.Texture1D, usage))
            {
                UnsupportedTextureTypeException.ThrowForTexture1D<T>();
            }

            GraphicsDevice = device;
            Width = width;
            this.state = this.isReadWrite ? TextureState.UnorderedAccess : TextureState.Common;
            this.texture = new NativeTexture(device, CsBackend.Texture1D, DXGIFormatHelper.GetForType<T>(), width, 1, 1, this.isReadWrite, sizeof(T));
        }

        /// <summary>Gets the <see cref="ComputeSharp.GraphicsDevice"/> associated with the current instance.</summary>
        public GraphicsDevice GraphicsDevice { get; }

        /// <summary>Gets the width of the current texture.</summary>
        public int Width { get; }

        internal NativeTexture NativeTexture => this.texture;

        internal void CopyTo(ref T destination, int size, int sourceOffsetX, int width)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetX, 0, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, Width);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetX + width, Width, nameof(sourceOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfLessThan(size, width);

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            fixed (void* destinationPointer = &destination)
            {
                this.texture.Read(sourceOffsetX, 0, 0, width, 1, 1, destinationPointer, (long)width * sizeof(T), (long)width * sizeof(T), ToString() ?? GetType().Name);
            }
        }

        internal void CopyTo(Texture1D<T> destination, int sourceOffsetX, int destinationOffsetX, int width)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetX, 0, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetX, 0, destination.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, destination.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(destinationOffsetX + width, 1, destination.Width, nameof(destinationOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetX + width, Width, nameof(sourceOffsetX));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _2 = destination.GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            destination.ThrowIfDeviceMismatch(GraphicsDevice);

            this.texture.CopyTo(destination.texture, sourceOffsetX, 0, 0, destinationOffsetX, 0, 0, width, 1, 1);
        }

        internal void CopyTo(ReadBackTexture1D<T> destination, int sourceOffsetX, int destinationOffsetX, int width)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetX, 0, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetX, 0, destination.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, destination.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(destinationOffsetX + width, 1, destination.Width, nameof(destinationOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetX + width, Width, nameof(sourceOffsetX));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _2 = destination.GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            destination.ThrowIfDeviceMismatch(GraphicsDevice);

            this.texture.Read(sourceOffsetX, 0, 0, width, 1, 1, destination.MappedData + destinationOffsetX, (long)width * sizeof(T), (long)width * sizeof(T), ToString() ?? GetType().Name);
        }

        internal void CopyFrom(ref T source, int size, int destinationOffsetX, int width)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetX, 0, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, Width);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffsetX + width, Width, nameof(destinationOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfLessThan(size, width);

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            fixed (void* sourcePointer = &source)
            {
                this.texture.Write(destinationOffsetX, 0, 0, width, 1, 1, sourcePointer, (long)width * sizeof(T), (long)width * sizeof(T));
            }
        }

        internal void CopyFrom(UploadTexture1D<T> source, int sourceOffsetX, int destinationOffsetX, int width)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetX, 0, source.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetX, 0, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, source.Width);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetX + width, source.Width, nameof(sourceOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffsetX + width, Width, nameof(destinationOffsetX));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _2 = source.GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            source.ThrowIfDeviceMismatch(GraphicsDevice);

            this.texture.Write(destinationOffsetX, 0, 0, width, 1, 1, source.MappedData + sourceOffsetX, (long)width * sizeof(T), (long)width * sizeof(T));
        }

        protected virtual partial void DangerousOnDispose()
        {
            // The texture is null when the constructor threw before allocating it.
            this.texture?.Destroy();
        }

        /// <summary>Throws if the texture is not in a readonly state.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected void ThrowIfIsNotInReadOnlyState()
        {
            if (this.state == TextureState.UnorderedAccess)
            {
                static void Throw()
                {
                    throw new InvalidOperationException(
                        "The texture is not currently in readonly mode. This API can only be used when creating a compute graph with " +
                        "the ComputeContext type, and after having used ComputeContext.Transition() to change the state of the texture.");
                }

                Throw();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void ThrowIfDeviceMismatch(GraphicsDevice device)
        {
            if (GraphicsDevice != device)
            {
                GraphicsDeviceMismatchException.Throw(this, device);
            }
        }

        internal PortableResourceView GetView(PortableResourceKind kind)
        {
            return new PortableResourceView
            {
                Kind = kind,
                Handle = this.texture.Handle,
                Format = this.texture.Format,
                Dimension = this.texture.Dimension,
                ResourceId = this.texture.ResourceId,
                Description = ToString() ?? GetType().Name,
            };
        }

        PortableResourceView IPortableReadOnlyResource.ValidateAndGetView(GraphicsDevice device, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            ThrowIfDeviceMismatch(device);

            return GetView(this.isReadWrite ? PortableResourceKind.ReadWriteTexture : PortableResourceKind.Texture);
        }

        PortableResourceView IPortableReadWriteResource.ValidateAndGetViewForClear(GraphicsDevice device, out bool isNormalized, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            ThrowIfDeviceMismatch(device);

            isNormalized = DXGIFormatHelper.IsNormalizedType<T>();

            return GetView(PortableResourceKind.ReadWriteTexture);
        }

        void IPortableReadWriteResource.ValidateAndTransition(GraphicsDevice device, ResourceState resourceState, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            ThrowIfDeviceMismatch(device);

            this.state = TextureStateHelper.FromResourceState(resourceState);
        }
    }

    /// <summary>A <see langword="class"/> representing a typed 2D texture stored in device memory.</summary>
    public abstract unsafe partial class Texture2D<T> : IReferenceTrackedObject, IGraphicsResource, IPortableReadWriteResource
        where T : unmanaged
    {
        private ReferenceTracker referenceTracker;
        private readonly NativeTexture texture;
        private readonly bool isReadWrite;
        private TextureState state;

        private protected Texture2D(GraphicsDevice device, int width, int height, ResourceType resourceType, AllocationMode allocationMode)
        {
            using ReferenceTracker.Lease _0 = ReferenceTracker.Create(this, out this.referenceTracker);

            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, D3D12Limits.Texture2DDimension);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, D3D12Limits.Texture2DDimension);

            using ReferenceTracker.Lease _1 = device.GetReferenceTracker().GetLease();

            device.ThrowIfDeviceLost();

            this.isReadWrite = resourceType == ResourceType.ReadWrite;
            uint usage = CsBackend.UsageShaderRead | (this.isReadWrite ? CsBackend.UsageShaderWrite : 0);
            if (!device.IsTextureSupported<T>(CsBackend.Texture2D, usage))
            {
                UnsupportedTextureTypeException.ThrowForTexture2D<T>();
            }

            GraphicsDevice = device;
            Width = width;
            Height = height;
            this.state = this.isReadWrite ? TextureState.UnorderedAccess : TextureState.Common;
            this.texture = new NativeTexture(device, CsBackend.Texture2D, DXGIFormatHelper.GetForType<T>(), width, height, 1, this.isReadWrite, sizeof(T));
        }

        /// <summary>Gets the <see cref="ComputeSharp.GraphicsDevice"/> associated with the current instance.</summary>
        public GraphicsDevice GraphicsDevice { get; }

        /// <summary>Gets the width of the current texture.</summary>
        public int Width { get; }

        /// <summary>Gets the height of the current texture.</summary>
        public int Height { get; }

        internal NativeTexture NativeTexture => this.texture;

        internal void CopyTo(ref T destination, int size, int sourceOffsetX, int sourceOffsetY, int width, int height)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetX, 0, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetY, 0, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, Height);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetX + width, Width, nameof(sourceOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetY + height, Height, nameof(sourceOffsetY));
            default(ArgumentOutOfRangeException).ThrowIfLessThan(size, (long)width * height);

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            long rowPitch = (long)width * sizeof(T);

            fixed (void* destinationPointer = &destination)
            {
                this.texture.Read(sourceOffsetX, sourceOffsetY, 0, width, height, 1, destinationPointer, rowPitch, rowPitch * height, ToString() ?? GetType().Name);
            }
        }

        internal void CopyTo(Texture2D<T> destination, int sourceOffsetX, int sourceOffsetY, int destinationOffsetX, int destinationOffsetY, int width, int height)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetX, 0, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetY, 0, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetX, 0, destination.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetY, 0, destination.Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, destination.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, destination.Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(destinationOffsetX + width, 1, destination.Width, nameof(destinationOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(destinationOffsetY + height, 1, destination.Height, nameof(destinationOffsetY));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetX + width, Width, nameof(sourceOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetY + height, Height, nameof(sourceOffsetY));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _2 = destination.GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            destination.ThrowIfDeviceMismatch(GraphicsDevice);

            this.texture.CopyTo(destination.texture, sourceOffsetX, sourceOffsetY, 0, destinationOffsetX, destinationOffsetY, 0, width, height, 1);
        }

        internal void CopyTo(ReadBackTexture2D<T> destination, int sourceOffsetX, int sourceOffsetY, int destinationOffsetX, int destinationOffsetY, int width, int height)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetX, 0, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetY, 0, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetX, 0, destination.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetY, 0, destination.Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, destination.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, destination.Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(destinationOffsetX + width, 1, destination.Width, nameof(destinationOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(destinationOffsetY + height, 1, destination.Height, nameof(destinationOffsetY));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetX + width, Width, nameof(sourceOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetY + height, Height, nameof(sourceOffsetY));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _2 = destination.GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            destination.ThrowIfDeviceMismatch(GraphicsDevice);

            byte* target = (byte*)destination.MappedData + ((long)destinationOffsetY * destination.RowPitch) + ((long)destinationOffsetX * sizeof(T));

            this.texture.Read(sourceOffsetX, sourceOffsetY, 0, width, height, 1, target, destination.RowPitch, (long)destination.RowPitch * destination.Height, ToString() ?? GetType().Name);
        }

        internal void CopyFrom(ref T source, int size, int destinationOffsetX, int destinationOffsetY, int width, int height)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetX, 0, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetY, 0, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, Height);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffsetX + width, Width, nameof(destinationOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffsetY + height, Height, nameof(destinationOffsetY));
            default(ArgumentOutOfRangeException).ThrowIfLessThan(size, (long)width * height);

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            long rowPitch = (long)width * sizeof(T);

            fixed (void* sourcePointer = &source)
            {
                this.texture.Write(destinationOffsetX, destinationOffsetY, 0, width, height, 1, sourcePointer, rowPitch, rowPitch * height);
            }
        }

        internal void CopyFrom(UploadTexture2D<T> source, int sourceOffsetX, int sourceOffsetY, int destinationOffsetX, int destinationOffsetY, int width, int height)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetX, 0, source.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetY, 0, source.Height);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetX, 0, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetY, 0, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, source.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, source.Height);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetX + width, source.Width, nameof(sourceOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetY + height, source.Height, nameof(sourceOffsetY));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffsetX + width, Width, nameof(destinationOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffsetY + height, Height, nameof(destinationOffsetY));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _2 = source.GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            source.ThrowIfDeviceMismatch(GraphicsDevice);

            byte* origin = (byte*)source.MappedData + ((long)sourceOffsetY * source.RowPitch) + ((long)sourceOffsetX * sizeof(T));

            this.texture.Write(destinationOffsetX, destinationOffsetY, 0, width, height, 1, origin, source.RowPitch, (long)source.RowPitch * source.Height);
        }

        protected virtual partial void DangerousOnDispose()
        {
            // The texture is null when the constructor threw before allocating it.
            this.texture?.Destroy();
        }

        /// <summary>Throws if the texture is not in a readonly state.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected void ThrowIfIsNotInReadOnlyState()
        {
            if (this.state == TextureState.UnorderedAccess)
            {
                static void Throw()
                {
                    throw new InvalidOperationException(
                        "The texture is not currently in readonly mode. This API can only be used when creating a compute graph with " +
                        "the ComputeContext type, and after having used ComputeContext.Transition() to change the state of the texture.");
                }

                Throw();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void ThrowIfDeviceMismatch(GraphicsDevice device)
        {
            if (GraphicsDevice != device)
            {
                GraphicsDeviceMismatchException.Throw(this, device);
            }
        }

        internal PortableResourceView GetView(PortableResourceKind kind)
        {
            return new PortableResourceView
            {
                Kind = kind,
                Handle = this.texture.Handle,
                Format = this.texture.Format,
                Dimension = this.texture.Dimension,
                ResourceId = this.texture.ResourceId,
                Description = ToString() ?? GetType().Name,
            };
        }

        PortableResourceView IPortableReadOnlyResource.ValidateAndGetView(GraphicsDevice device, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            ThrowIfDeviceMismatch(device);

            return GetView(this.isReadWrite ? PortableResourceKind.ReadWriteTexture : PortableResourceKind.Texture);
        }

        PortableResourceView IPortableReadWriteResource.ValidateAndGetViewForClear(GraphicsDevice device, out bool isNormalized, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            ThrowIfDeviceMismatch(device);

            isNormalized = DXGIFormatHelper.IsNormalizedType<T>();

            return GetView(PortableResourceKind.ReadWriteTexture);
        }

        void IPortableReadWriteResource.ValidateAndTransition(GraphicsDevice device, ResourceState resourceState, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            ThrowIfDeviceMismatch(device);

            this.state = TextureStateHelper.FromResourceState(resourceState);
        }
    }

    /// <summary>A <see langword="class"/> representing a typed 3D texture stored in device memory.</summary>
    public abstract unsafe partial class Texture3D<T> : IReferenceTrackedObject, IGraphicsResource, IPortableReadWriteResource
        where T : unmanaged
    {
        private ReferenceTracker referenceTracker;
        private readonly NativeTexture texture;
        private readonly bool isReadWrite;
        private TextureState state;

        private protected Texture3D(GraphicsDevice device, int width, int height, int depth, ResourceType resourceType, AllocationMode allocationMode)
        {
            using ReferenceTracker.Lease _0 = ReferenceTracker.Create(this, out this.referenceTracker);

            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, D3D12Limits.Texture3DDimension);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, D3D12Limits.Texture3DDimension);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(depth, 1, D3D12Limits.Texture3DDimension);

            using ReferenceTracker.Lease _1 = device.GetReferenceTracker().GetLease();

            device.ThrowIfDeviceLost();

            this.isReadWrite = resourceType == ResourceType.ReadWrite;
            uint usage = CsBackend.UsageShaderRead | (this.isReadWrite ? CsBackend.UsageShaderWrite : 0);
            if (!device.IsTextureSupported<T>(CsBackend.Texture3D, usage))
            {
                UnsupportedTextureTypeException.ThrowForTexture3D<T>();
            }

            GraphicsDevice = device;
            Width = width;
            Height = height;
            Depth = depth;
            this.state = this.isReadWrite ? TextureState.UnorderedAccess : TextureState.Common;
            this.texture = new NativeTexture(device, CsBackend.Texture3D, DXGIFormatHelper.GetForType<T>(), width, height, depth, this.isReadWrite, sizeof(T));
        }

        /// <summary>Gets the <see cref="ComputeSharp.GraphicsDevice"/> associated with the current instance.</summary>
        public GraphicsDevice GraphicsDevice { get; }

        /// <summary>Gets the width of the current texture.</summary>
        public int Width { get; }

        /// <summary>Gets the height of the current texture.</summary>
        public int Height { get; }

        /// <summary>Gets the depth of the current texture.</summary>
        public int Depth { get; }

        internal NativeTexture NativeTexture => this.texture;

        internal void CopyTo(ref T destination, int size, int sourceOffsetX, int sourceOffsetY, int sourceOffsetZ, int width, int height, int depth)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetX, 0, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetY, 0, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetZ, 0, Depth);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(depth, 1, Depth);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetX + width, Width, nameof(sourceOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetY + height, Height, nameof(sourceOffsetY));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetZ + depth, Depth, nameof(sourceOffsetZ));
            default(ArgumentOutOfRangeException).ThrowIfLessThan(size, (long)width * height * depth);

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            long rowPitch = (long)width * sizeof(T);

            fixed (void* destinationPointer = &destination)
            {
                this.texture.Read(sourceOffsetX, sourceOffsetY, sourceOffsetZ, width, height, depth, destinationPointer, rowPitch, rowPitch * height, ToString() ?? GetType().Name);
            }
        }

        internal void CopyTo(Texture3D<T> destination, int sourceOffsetX, int sourceOffsetY, int sourceOffsetZ, int destinationOffsetX, int destinationOffsetY, int destinationOffsetZ, int width, int height, int depth)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetX, 0, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetY, 0, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetZ, 0, Depth);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetX, 0, destination.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetY, 0, destination.Height);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetZ, 0, destination.Depth);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(depth, 1, Depth);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, destination.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, destination.Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(depth, 1, destination.Depth);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(destinationOffsetX + width, 1, destination.Width, nameof(destinationOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(destinationOffsetY + height, 1, destination.Height, nameof(destinationOffsetY));
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(destinationOffsetZ + depth, 1, destination.Depth, nameof(destinationOffsetZ));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetX + width, Width, nameof(sourceOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetY + height, Height, nameof(sourceOffsetY));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetZ + depth, Depth, nameof(sourceOffsetZ));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _2 = destination.GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            destination.ThrowIfDeviceMismatch(GraphicsDevice);

            this.texture.CopyTo(destination.texture, sourceOffsetX, sourceOffsetY, sourceOffsetZ, destinationOffsetX, destinationOffsetY, destinationOffsetZ, width, height, depth);
        }

        internal void CopyTo(ReadBackTexture3D<T> destination, int sourceOffsetX, int sourceOffsetY, int sourceOffsetZ, int destinationOffsetX, int destinationOffsetY, int destinationOffsetZ, int width, int height, int depth)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetX, 0, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetY, 0, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetZ, 0, Depth);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetX, 0, destination.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetY, 0, destination.Height);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetZ, 0, destination.Depth);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(depth, 1, Depth);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, destination.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, destination.Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(depth, 1, destination.Depth);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(destinationOffsetX + width, 1, destination.Width, nameof(destinationOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(destinationOffsetY + height, 1, destination.Height, nameof(destinationOffsetY));
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(destinationOffsetZ + depth, 1, destination.Depth, nameof(destinationOffsetZ));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetX + width, Width, nameof(sourceOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetY + height, Height, nameof(sourceOffsetY));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetZ + depth, Depth, nameof(sourceOffsetZ));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _2 = destination.GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            destination.ThrowIfDeviceMismatch(GraphicsDevice);

            long slicePitch = (long)destination.RowPitch * destination.Height;
            byte* target = (byte*)destination.MappedData + ((long)destinationOffsetZ * slicePitch) + ((long)destinationOffsetY * destination.RowPitch) + ((long)destinationOffsetX * sizeof(T));

            this.texture.Read(sourceOffsetX, sourceOffsetY, sourceOffsetZ, width, height, depth, target, destination.RowPitch, slicePitch, ToString() ?? GetType().Name);
        }

        internal void CopyFrom(ref T source, int size, int destinationOffsetX, int destinationOffsetY, int destinationOffsetZ, int width, int height, int depth)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetX, 0, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetY, 0, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetZ, 0, Depth);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(depth, 1, Depth);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffsetX + width, Width, nameof(destinationOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffsetY + height, Height, nameof(destinationOffsetY));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffsetZ + depth, Depth, nameof(destinationOffsetZ));
            default(ArgumentOutOfRangeException).ThrowIfLessThan(size, (long)width * height * depth);

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            long rowPitch = (long)width * sizeof(T);

            fixed (void* sourcePointer = &source)
            {
                this.texture.Write(destinationOffsetX, destinationOffsetY, destinationOffsetZ, width, height, depth, sourcePointer, rowPitch, rowPitch * height);
            }
        }

        internal void CopyFrom(UploadTexture3D<T> source, int sourceOffsetX, int sourceOffsetY, int sourceOffsetZ, int destinationOffsetX, int destinationOffsetY, int destinationOffsetZ, int width, int height, int depth)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetX, 0, source.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetY, 0, source.Height);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffsetZ, 0, source.Depth);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetX, 0, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetY, 0, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffsetZ, 0, Depth);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(depth, 1, Depth);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(width, 1, source.Width);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(height, 1, source.Height);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(depth, 1, source.Depth);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetX + width, source.Width, nameof(sourceOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetY + height, source.Height, nameof(sourceOffsetY));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffsetZ + depth, source.Depth, nameof(sourceOffsetZ));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffsetX + width, Width, nameof(destinationOffsetX));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffsetY + height, Height, nameof(destinationOffsetY));
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffsetZ + depth, Depth, nameof(destinationOffsetZ));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _2 = source.GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            source.ThrowIfDeviceMismatch(GraphicsDevice);

            long slicePitch = (long)source.RowPitch * source.Height;
            byte* origin = (byte*)source.MappedData + ((long)sourceOffsetZ * slicePitch) + ((long)sourceOffsetY * source.RowPitch) + ((long)sourceOffsetX * sizeof(T));

            this.texture.Write(destinationOffsetX, destinationOffsetY, destinationOffsetZ, width, height, depth, origin, source.RowPitch, slicePitch);
        }

        protected virtual partial void DangerousOnDispose()
        {
            // The texture is null when the constructor threw before allocating it.
            this.texture?.Destroy();
        }

        /// <summary>Throws if the texture is not in a readonly state.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected void ThrowIfIsNotInReadOnlyState()
        {
            if (this.state == TextureState.UnorderedAccess)
            {
                static void Throw()
                {
                    throw new InvalidOperationException(
                        "The texture is not currently in readonly mode. This API can only be used when creating a compute graph with " +
                        "the ComputeContext type, and after having used ComputeContext.Transition() to change the state of the texture.");
                }

                Throw();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void ThrowIfDeviceMismatch(GraphicsDevice device)
        {
            if (GraphicsDevice != device)
            {
                GraphicsDeviceMismatchException.Throw(this, device);
            }
        }

        internal PortableResourceView GetView(PortableResourceKind kind)
        {
            return new PortableResourceView
            {
                Kind = kind,
                Handle = this.texture.Handle,
                Format = this.texture.Format,
                Dimension = this.texture.Dimension,
                ResourceId = this.texture.ResourceId,
                Description = ToString() ?? GetType().Name,
            };
        }

        PortableResourceView IPortableReadOnlyResource.ValidateAndGetView(GraphicsDevice device, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            ThrowIfDeviceMismatch(device);

            return GetView(this.isReadWrite ? PortableResourceKind.ReadWriteTexture : PortableResourceKind.Texture);
        }

        PortableResourceView IPortableReadWriteResource.ValidateAndGetViewForClear(GraphicsDevice device, out bool isNormalized, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            ThrowIfDeviceMismatch(device);

            isNormalized = DXGIFormatHelper.IsNormalizedType<T>();

            return GetView(PortableResourceKind.ReadWriteTexture);
        }

        void IPortableReadWriteResource.ValidateAndTransition(GraphicsDevice device, ResourceState resourceState, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            ThrowIfDeviceMismatch(device);

            this.state = TextureStateHelper.FromResourceState(resourceState);
        }
    }
}

namespace ComputeSharp.Portable
{
    internal static class TextureStateHelper
    {
        public static TextureState FromResourceState(ResourceState resourceState)
        {
            return resourceState switch
            {
                ResourceState.ReadOnly => TextureState.ReadOnly,
                ResourceState.ReadWrite => TextureState.UnorderedAccess,
                _ => throw new ArgumentException("Invalid resource state.", nameof(resourceState))
            };
        }
    }
}
