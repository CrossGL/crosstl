using System;
using System.Buffers;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ComputeSharp.Graphics.Helpers;
using ComputeSharp.Graphics.Resources.Enums;
using ComputeSharp.Interop;
using ComputeSharp.Portable;
using ComputeSharp.Portable.Diagnostics;
using ComputeSharp.Portable.Native;
using ComputeSharp.Resources.Debug;
using ComputeSharp.Resources.Interop;

#pragma warning disable CA1063

namespace ComputeSharp.Resources
{
    /// <summary>A <see langword="class"/> representing a typed buffer stored in device memory.</summary>
    public abstract unsafe partial class Buffer<T> : IReferenceTrackedObject, IGraphicsResource, IPortableReadOnlyResource
        where T : unmanaged
    {
        private ReferenceTracker referenceTracker;
        private IntPtr nativeBuffer;
        private readonly ResourceType resourceType;

        /// <summary>The size in bytes of the usable buffer data.</summary>
        protected readonly nint SizeInBytes;

        [RequiresUnreferencedCode("This method reads type info of all fields of the resource element type (recursively).")]
        private protected Buffer(GraphicsDevice device, int length, uint elementSizeInBytes, ResourceType resourceType, AllocationMode allocationMode)
        {
            using ReferenceTracker.Lease _0 = ReferenceTracker.Create(this, out this.referenceTracker);

            if (resourceType == ResourceType.Constant)
            {
                default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(length, 1, D3D12Limits.ConstantBufferElementCount);
            }
            else
            {
                default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(length, 1, (uint.MaxValue / elementSizeInBytes) & ~255);
            }

            using ReferenceTracker.Lease _1 = device.GetReferenceTracker().GetLease();

            device.ThrowIfDeviceLost();

            if (TypeInfo<T>.IsDoubleOrContainsDoubles && !device.IsDoublePrecisionSupportAvailable())
            {
                UnsupportedDoubleOperationException.Throw<T>();
            }

            nint usableSizeInBytes = checked((nint)(length * elementSizeInBytes));
            nint effectiveSizeInBytes = resourceType == ResourceType.Constant
                ? AlignmentHelper.Pad(usableSizeInBytes, D3D12Limits.ConstantBufferDataPlacementAlignment)
                : usableSizeInBytes;

            this.SizeInBytes = usableSizeInBytes;
            this.resourceType = resourceType;
            ResourceId = EvidenceLog.NextResourceId();
            GraphicsDevice = device;
            Length = length;
            this.nativeBuffer = device.Native.CreateBuffer((ulong)effectiveSizeInBytes);
        }

        /// <summary>Gets the <see cref="ComputeSharp.GraphicsDevice"/> associated with the current instance.</summary>
        public GraphicsDevice GraphicsDevice { get; }

        /// <summary>Gets the length of the current buffer.</summary>
        public int Length { get; }

        internal long ResourceId { get; }

        internal bool IsPaddingPresent
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => this.SizeInBytes > (nint)Length * sizeof(T);
        }

        internal IntPtr NativeBuffer => this.nativeBuffer;

        internal abstract void CopyTo(ref T destination, int sourceOffset, int count);

        internal abstract void CopyTo(Buffer<T> destination, int sourceOffset, int destinationOffset, int count);

        /// <summary>Copies through a temporary host buffer.</summary>
        protected void CopyToWithCpuBuffer(Buffer<T> destination, int sourceOffset, int destinationOffset, int count)
        {
            T[] array = ArrayPool<T>.Shared.Rent(count);

            try
            {
                ref T r0 = ref MemoryMarshal.GetArrayDataReference(array);

                CopyTo(ref r0, sourceOffset, count);

                destination.CopyFrom(ref r0, destinationOffset, count);
            }
            finally
            {
                ArrayPool<T>.Shared.Return(array);
            }
        }

        internal abstract void CopyFrom(ref T source, int destinationOffset, int count);

        void IReferenceTrackedObject.DangerousOnDispose()
        {
            if (GraphicsDevice is GraphicsDevice device && this.nativeBuffer != IntPtr.Zero)
            {
                device.Native.DestroyBuffer(this.nativeBuffer);
            }

            this.nativeBuffer = IntPtr.Zero;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void ThrowIfDeviceMismatch(GraphicsDevice device)
        {
            if (GraphicsDevice != device)
            {
                GraphicsDeviceMismatchException.Throw(this, device);
            }
        }

        /// <summary>Validates the device and returns the native buffer for a GPU clear.</summary>
        internal IntPtr ValidateAndGetNativeBufferForClear(GraphicsDevice device, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            ThrowIfDeviceMismatch(device);

            return this.nativeBuffer;
        }

        /// <summary>Validates the device and returns the native buffer for a barrier.</summary>
        internal IntPtr ValidateAndGetNativeBuffer(GraphicsDevice device, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            ThrowIfDeviceMismatch(device);

            return this.nativeBuffer;
        }

        PortableResourceView IPortableReadOnlyResource.ValidateAndGetView(GraphicsDevice device, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            ThrowIfDeviceMismatch(device);

            return new PortableResourceView
            {
                Kind = this.resourceType == ResourceType.Constant ? PortableResourceKind.ConstantBuffer : PortableResourceKind.StructuredBuffer,
                Handle = this.nativeBuffer,
                Length = Length,
                SizeInBytes = this.resourceType == ResourceType.Constant
                    ? AlignmentHelper.Pad(this.SizeInBytes, D3D12Limits.ConstantBufferDataPlacementAlignment)
                    : this.SizeInBytes,
                ResourceId = ResourceId,
                Description = ToString() ?? GetType().Name,
            };
        }

        /// <summary>Reads device bytes into host memory and records the readback.</summary>
        internal void ReadNative(nint byteOffset, void* destination, nint byteLength, bool recordReadback = true)
        {
            GraphicsDevice.Native.ReadBuffer(this.nativeBuffer, (ulong)byteOffset, (IntPtr)destination, (ulong)byteLength);
            if (recordReadback)
            {
                EvidenceLog.Readback(ToString() ?? GetType().Name, ResourceId, (IntPtr)destination, byteLength);
            }
        }

        /// <summary>Writes host bytes into device memory.</summary>
        internal void WriteNative(nint byteOffset, void* source, nint byteLength)
        {
            GraphicsDevice.Native.WriteBuffer(this.nativeBuffer, (ulong)byteOffset, (IntPtr)source, (ulong)byteLength);
        }

        /// <summary>Copies device bytes between buffers on the GPU.</summary>
        internal void CopyNative(nint sourceByteOffset, Buffer<T> destination, nint destinationByteOffset, nint byteLength)
        {
            using NativeCommandList list = new(GraphicsDevice.Native);

            list.CopyBuffer(this.nativeBuffer, (ulong)sourceByteOffset, destination.nativeBuffer, (ulong)destinationByteOffset, (ulong)byteLength);
            list.Submit();
        }
    }

    /// <summary>A <see langword="class"/> representing a typed structured buffer stored in device memory.</summary>
    public abstract class StructuredBuffer<T> : Buffer<T>
        where T : unmanaged
    {
        [RequiresUnreferencedCode("This method reads type info of all fields of the resource element type (recursively).")]
        private protected unsafe StructuredBuffer(GraphicsDevice device, int length, ResourceType resourceType, AllocationMode allocationMode)
            : base(device, length, (uint)sizeof(T), resourceType, allocationMode)
        {
        }

        internal override unsafe void CopyTo(ref T destination, int sourceOffset, int count)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(count, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffset, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffset + count, Length, nameof(sourceOffset));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            fixed (void* destinationPointer = &destination)
            {
                ReadNative((nint)sourceOffset * sizeof(T), destinationPointer, (nint)count * sizeof(T));
            }
        }

        internal override unsafe void CopyTo(Buffer<T> destination, int sourceOffset, int destinationOffset, int count)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(count, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(count, 0, destination.Length);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffset, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffset + count, Length, nameof(sourceOffset));
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffset, 0, destination.Length);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffset + count, destination.Length, nameof(destinationOffset));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _2 = destination.GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            destination.ThrowIfDeviceMismatch(GraphicsDevice);

            if (!destination.IsPaddingPresent)
            {
                CopyNative((nint)sourceOffset * sizeof(T), destination, (nint)destinationOffset * sizeof(T), (nint)count * sizeof(T));
            }
            else
            {
                CopyToWithCpuBuffer(destination, sourceOffset, destinationOffset, count);
            }
        }

        internal unsafe void CopyTo(ReadBackBuffer<T> destination, int sourceOffset, int destinationOffset, int count)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(count, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(count, 0, destination.Length);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffset, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffset + count, Length, nameof(sourceOffset));
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffset, 0, destination.Length);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffset + count, destination.Length, nameof(destinationOffset));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _2 = destination.GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            destination.ThrowIfDeviceMismatch(GraphicsDevice);

            ReadNative((nint)sourceOffset * sizeof(T), destination.MappedData + destinationOffset, (nint)count * sizeof(T));
        }

        internal override unsafe void CopyFrom(ref T source, int offset, int length)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(length, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(offset, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(offset + length, Length, nameof(offset));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            fixed (void* sourcePointer = &source)
            {
                WriteNative((nint)offset * sizeof(T), sourcePointer, (nint)length * sizeof(T));
            }
        }

        internal unsafe void CopyFrom(UploadBuffer<T> source, int sourceOffset, int destinationOffset, int count)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(count, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(count, 0, source.Length);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffset, 0, source.Length);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffset + count, source.Length, nameof(sourceOffset));
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffset, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffset + count, Length, nameof(destinationOffset));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _2 = source.GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            source.ThrowIfDeviceMismatch(GraphicsDevice);

            WriteNative((nint)destinationOffset * sizeof(T), source.MappedData + sourceOffset, (nint)count * sizeof(T));
        }
    }

    /// <summary>A <see langword="class"/> representing a typed buffer in host memory used for transfers.</summary>
    public abstract unsafe partial class TransferBuffer<T> : IReferenceTrackedObject, IGraphicsResource, IMemoryOwner<T>
        where T : unmanaged
    {
        private ReferenceTracker referenceTracker;
        private T* mappedData;

        [RequiresUnreferencedCode("This method reads type info of all fields of the resource element type (recursively).")]
        private protected TransferBuffer(GraphicsDevice device, int length, ResourceType resourceType, AllocationMode allocationMode)
        {
            using ReferenceTracker.Lease _0 = ReferenceTracker.Create(this, out this.referenceTracker);

            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(length, 1, (uint.MaxValue / (uint)sizeof(T)) & ~255);

            using ReferenceTracker.Lease _1 = device.GetReferenceTracker().GetLease();

            device.ThrowIfDeviceLost();

            if (TypeInfo<T>.IsDoubleOrContainsDoubles && !device.IsDoublePrecisionSupportAvailable())
            {
                UnsupportedDoubleOperationException.Throw<T>();
            }

            GraphicsDevice = device;
            Length = length;

            nuint sizeInBytes = (nuint)length * (nuint)sizeof(T);

            // Transfer resources are host memory; Direct3D 12 upload and readback heaps are zero initialized.
            this.mappedData = (T*)NativeMemory.AlignedAlloc(sizeInBytes == 0 ? 1 : sizeInBytes, 256);
            NativeMemory.Clear(this.mappedData, sizeInBytes);
        }

        /// <summary>Gets the <see cref="ComputeSharp.GraphicsDevice"/> associated with the current instance.</summary>
        public GraphicsDevice GraphicsDevice { get; }

        /// <summary>Gets the length of the current buffer.</summary>
        public int Length { get; }

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

        /// <summary>Gets a <see cref="Span{T}"/> over the buffer data.</summary>
        public Span<T> Span
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                using ReferenceTracker.Lease _0 = GetReferenceTracker().GetLease();

                return new(this.mappedData, Length);
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

        private sealed class MemoryManager(TransferBuffer<T> buffer) : MemoryManager<T>
        {
            private readonly TransferBuffer<T> buffer = buffer;

            public override Memory<T> Memory
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => CreateMemory(this.buffer.Length);
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
}

namespace ComputeSharp
{
    /// <summary>A <see langword="class"/> representing a typed read-only constant buffer stored in device memory.</summary>
    [DebuggerTypeProxy(typeof(BufferDebugView<>))]
    [DebuggerDisplay("{ToString(),raw}")]
    public sealed class ConstantBuffer<T> : Resources.Buffer<T>
        where T : unmanaged
    {
        private const int ElementAlignment = D3D12Limits.ConstantBufferPartialUpdateAlignment;

        [RequiresUnreferencedCode("This method reads type info of all fields of the resource element type (recursively).")]
        internal ConstantBuffer(GraphicsDevice device, int length, AllocationMode allocationMode)
            : base(device, length, (uint)GetPaddedSize(), ResourceType.Constant, allocationMode)
        {
        }

        /// <summary>Gets a single <typeparamref name="T"/> value from the current constant buffer.</summary>
        public ref readonly T this[int i] => throw new InvalidExecutionContextException($"{typeof(ConstantBuffer<T>)}[{typeof(int)}]");

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe int GetPaddedSize()
        {
            return AlignmentHelper.Pad(sizeof(T), ElementAlignment);
        }

        internal override unsafe void CopyTo(ref T destination, int offset, int length)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(length, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(offset, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(offset + length, Length, nameof(offset));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            nint pitch = GetPaddedSize();
            nint bytes = pitch * length;
            byte[] staging = ArrayPool<byte>.Shared.Rent((int)Math.Max(bytes, 1));

            try
            {
                fixed (byte* stagingPointer = staging)
                fixed (void* destinationPointer = &destination)
                {
                    ReadNative(pitch * offset, stagingPointer, bytes);

                    MemoryHelper.Copy<T>(
                        source: stagingPointer,
                        destination: destinationPointer,
                        sourceElementOffset: 0,
                        destinationElementOffset: 0,
                        sourceElementPitchInBytes: (uint)pitch,
                        destinationElementPitchInBytes: (uint)sizeof(T),
                        count: (uint)length);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(staging);
            }
        }

        internal override unsafe void CopyTo(Resources.Buffer<T> destination, int sourceOffset, int destinationOffset, int length)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(length, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(length, 0, destination.Length);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(sourceOffset, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(sourceOffset + length, Length, nameof(sourceOffset));
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(destinationOffset, 0, destination.Length);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(destinationOffset + length, destination.Length, nameof(destinationOffset));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _2 = destination.GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            destination.ThrowIfDeviceMismatch(GraphicsDevice);

            if (destination is ConstantBuffer<T>)
            {
                nint pitch = GetPaddedSize();

                CopyNative(pitch * sourceOffset, destination, pitch * destinationOffset, pitch * length);
            }
            else
            {
                CopyToWithCpuBuffer(destination, sourceOffset, destinationOffset, length);
            }
        }

        internal override unsafe void CopyFrom(ref T source, int offset, int length)
        {
            default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(length, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfNotInRange(offset, 0, Length);
            default(ArgumentOutOfRangeException).ThrowIfGreaterThan(offset + length, Length, nameof(offset));

            using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
            using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

            GraphicsDevice.ThrowIfDeviceLost();

            nint pitch = GetPaddedSize();
            nint bytes = pitch * length;
            byte[] staging = ArrayPool<byte>.Shared.Rent((int)Math.Max(bytes, 1));

            try
            {
                fixed (byte* stagingPointer = staging)
                fixed (void* sourcePointer = &source)
                {
                    // Preserve the padding bytes between elements, as a mapped Direct3D 12 upload would.
                    ReadNative(pitch * offset, stagingPointer, bytes, recordReadback: false);

                    MemoryHelper.Copy<T>(
                        source: sourcePointer,
                        destination: stagingPointer,
                        sourceElementOffset: 0,
                        destinationElementOffset: 0,
                        sourceElementPitchInBytes: (uint)sizeof(T),
                        destinationElementPitchInBytes: (uint)pitch,
                        count: (uint)length);

                    WriteNative(pitch * offset, stagingPointer, bytes);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(staging);
            }
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"ComputeSharp.ConstantBuffer<{typeof(T)}>[{Length}]";
        }
    }
}
