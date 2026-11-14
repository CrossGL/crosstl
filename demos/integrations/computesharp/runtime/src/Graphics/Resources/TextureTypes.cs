// Portable leaf texture types. The public surface matches the pinned upstream
// ReadOnlyTexture*/ReadWriteTexture* types; only the native resource handling differs.
using System.Diagnostics;
using System.Runtime.CompilerServices;
using ComputeSharp.Graphics.Resources.Enums;
using ComputeSharp.Interop;
using ComputeSharp.Resources;
using ComputeSharp.Resources.Debug;
using ComputeSharp.Resources.Interop;

#pragma warning disable IDE0022

namespace ComputeSharp;

/// <summary>A <see langword="class"/> representing a typed readonly 1D texture stored in device memory.</summary>
[DebuggerTypeProxy(typeof(Texture1DDebugView<>))]
[DebuggerDisplay("{ToString(),raw}")]
public sealed class ReadOnlyTexture1D<T> : Texture1D<T>
    where T : unmanaged
{
    internal ReadOnlyTexture1D(GraphicsDevice device, int width, AllocationMode allocationMode)
        : base(device, width, ResourceType.ReadOnly, allocationMode)
    {
    }

    /// <summary>Gets a single <typeparamref name="T"/> value from the current texture.</summary>
    public ref readonly T this[int x] => throw new InvalidExecutionContextException($"{typeof(ReadOnlyTexture1D<T>)}[{typeof(int)}]");

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"ComputeSharp.ReadOnlyTexture1D<{typeof(T)}>[{Width}]";
    }
}

/// <summary>A <see langword="class"/> representing a typed readonly 1D texture with a normalized pixel view.</summary>
[DebuggerTypeProxy(typeof(Texture1DDebugView<>))]
[DebuggerDisplay("{ToString(),raw}")]
public sealed class ReadOnlyTexture1D<T, TPixel> : Texture1D<T>, IReadOnlyNormalizedTexture1D<TPixel>
    where T : unmanaged, IPixel<T, TPixel>
    where TPixel : unmanaged
{
    internal ReadOnlyTexture1D(GraphicsDevice device, int width, AllocationMode allocationMode)
        : base(device, width, ResourceType.ReadOnly, allocationMode)
    {
    }

    /// <summary>Gets a single <typeparamref name="TPixel"/> value from the current texture.</summary>
    public ref readonly TPixel this[int x] => throw new InvalidExecutionContextException($"{typeof(ReadOnlyTexture1D<T, TPixel>)}[{typeof(int)}]");

    /// <summary>Samples the current texture at a specified location.</summary>
    public ref readonly TPixel Sample(float u) => throw new InvalidExecutionContextException($"{typeof(ReadOnlyTexture1D<T, TPixel>)}.{nameof(Sample)}({typeof(float)})");

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"ComputeSharp.ReadOnlyTexture1D<{typeof(T)}, {typeof(TPixel)}>[{Width}]";
    }
}

/// <summary>A <see langword="class"/> representing a typed readwrite 1D texture stored in device memory.</summary>
[DebuggerTypeProxy(typeof(Texture1DDebugView<>))]
[DebuggerDisplay("{ToString(),raw}")]
public sealed partial class ReadWriteTexture1D<T> : Texture1D<T>
    where T : unmanaged
{
    internal ReadWriteTexture1D(GraphicsDevice device, int width, AllocationMode allocationMode)
        : base(device, width, ResourceType.ReadWrite, allocationMode)
    {
    }

    /// <summary>Gets a single <typeparamref name="T"/> value from the current texture.</summary>
    public ref T this[int x] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture1D<T>)}[{typeof(int)}]");

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"ComputeSharp.ReadWriteTexture1D<{typeof(T)}>[{Width}]";
    }
}

/// <summary>A <see langword="class"/> representing a typed readwrite 1D texture with a normalized pixel view.</summary>
[DebuggerTypeProxy(typeof(Texture1DDebugView<>))]
[DebuggerDisplay("{ToString(),raw}")]
public sealed partial class ReadWriteTexture1D<T, TPixel> : Texture1D<T>, IReadWriteNormalizedTexture1D<TPixel>
    where T : unmanaged, IPixel<T, TPixel>
    where TPixel : unmanaged
{
    internal ReadWriteTexture1D(GraphicsDevice device, int width, AllocationMode allocationMode)
        : base(device, width, ResourceType.ReadWrite, allocationMode)
    {
    }

    /// <summary>Gets a single <typeparamref name="TPixel"/> value from the current texture.</summary>
    public ref TPixel this[int x] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture2D<T, TPixel>)}[{typeof(int)}]");

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"ComputeSharp.ReadWriteTexture1D<{typeof(T)}, {typeof(TPixel)}>[{Width}]";
    }
}

partial class ReadWriteTexture1D<T>
{
    private ReadOnly? readOnlyWrapper;

    /// <summary>Gets a readonly view of the current texture, usable after a transition to the readonly state.</summary>
    public IReadOnlyTexture1D<T> AsReadOnly()
    {
        using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
        using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

        GraphicsDevice.ThrowIfDeviceLost();

        ThrowIfIsNotInReadOnlyState();

        ReadOnly? readOnlyWrapper = this.readOnlyWrapper;

        if (readOnlyWrapper is not null)
        {
            return readOnlyWrapper;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static ReadOnly InitializeWrapper(ReadWriteTexture1D<T> texture)
        {
            return texture.readOnlyWrapper = new(texture);
        }

        return InitializeWrapper(this);
    }

    /// <inheritdoc/>
    protected override void DangerousOnDispose()
    {
        base.DangerousOnDispose();

        this.readOnlyWrapper?.Dispose();
    }

    private sealed class ReadOnly : ReferenceTrackedObject, IReadOnlyTexture1D<T>, IPortableReadOnlyResource
    {
        private readonly ReadWriteTexture1D<T> owner;

        public ReadOnly(ReadWriteTexture1D<T> owner)
        {
            using ReferenceTracker.Lease _0 = GetReferenceTracker().GetLease();

            owner.GetReferenceTracker().DangerousAddRef();

            this.owner = owner;
        }

        /// <inheritdoc/>
        public ref readonly T this[int x] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture1D<T>.ReadOnly)}[{typeof(int)}]");

        /// <inheritdoc/>
        public ref readonly T Sample(float u) => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture1D<T>.ReadOnly)}.{nameof(Sample)}({typeof(float)})");

        /// <inheritdoc/>
        public int Width => this.owner.Width;

        /// <inheritdoc/>
        public GraphicsDevice GraphicsDevice => this.owner.GraphicsDevice;

        PortableResourceView IPortableReadOnlyResource.ValidateAndGetView(GraphicsDevice device, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            using ReferenceTracker.Lease _1 = this.owner.GetReferenceTracker().GetLease();

            this.owner.ThrowIfDeviceMismatch(device);

            return this.owner.GetView(PortableResourceKind.Texture);
        }

        protected override void DangerousOnDispose()
        {
            this.owner.GetReferenceTracker().DangerousRelease();
        }
    }
}

partial class ReadWriteTexture1D<T, TPixel>
{
    private ReadOnly? readOnlyWrapper;

    /// <summary>Gets a readonly view of the current texture, usable after a transition to the readonly state.</summary>
    internal IReadOnlyNormalizedTexture1D<TPixel> AsReadOnly()
    {
        using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
        using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

        GraphicsDevice.ThrowIfDeviceLost();

        ThrowIfIsNotInReadOnlyState();

        ReadOnly? readOnlyWrapper = this.readOnlyWrapper;

        if (readOnlyWrapper is not null)
        {
            return readOnlyWrapper;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static ReadOnly InitializeWrapper(ReadWriteTexture1D<T, TPixel> texture)
        {
            return texture.readOnlyWrapper = new(texture);
        }

        return InitializeWrapper(this);
    }

    /// <inheritdoc/>
    protected override void DangerousOnDispose()
    {
        base.DangerousOnDispose();

        this.readOnlyWrapper?.Dispose();
    }

    private sealed class ReadOnly : ReferenceTrackedObject, IReadOnlyNormalizedTexture1D<TPixel>, IPortableReadOnlyResource
    {
        private readonly ReadWriteTexture1D<T, TPixel> owner;

        public ReadOnly(ReadWriteTexture1D<T, TPixel> owner)
        {
            using ReferenceTracker.Lease _0 = GetReferenceTracker().GetLease();

            owner.GetReferenceTracker().DangerousAddRef();

            this.owner = owner;
        }

        /// <inheritdoc/>
        public ref readonly TPixel this[int x] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture1D<T, TPixel>.ReadOnly)}[{typeof(int)}]");

        /// <inheritdoc/>
        public ref readonly TPixel Sample(float u) => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture1D<T, TPixel>.ReadOnly)}.{nameof(Sample)}({typeof(float)})");

        /// <inheritdoc/>
        public int Width => this.owner.Width;

        /// <inheritdoc/>
        public GraphicsDevice GraphicsDevice => this.owner.GraphicsDevice;

        PortableResourceView IPortableReadOnlyResource.ValidateAndGetView(GraphicsDevice device, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            using ReferenceTracker.Lease _1 = this.owner.GetReferenceTracker().GetLease();

            this.owner.ThrowIfDeviceMismatch(device);

            return this.owner.GetView(PortableResourceKind.Texture);
        }

        protected override void DangerousOnDispose()
        {
            this.owner.GetReferenceTracker().DangerousRelease();
        }
    }
}

/// <summary>A <see langword="class"/> representing a typed readonly 2D texture stored in device memory.</summary>
[DebuggerTypeProxy(typeof(Texture2DDebugView<>))]
[DebuggerDisplay("{ToString(),raw}")]
public sealed class ReadOnlyTexture2D<T> : Texture2D<T>
    where T : unmanaged
{
    internal ReadOnlyTexture2D(GraphicsDevice device, int width, int height, AllocationMode allocationMode)
        : base(device, width, height, ResourceType.ReadOnly, allocationMode)
    {
    }

    /// <summary>Gets a single <typeparamref name="T"/> value from the current texture.</summary>
    public ref readonly T this[int x, int y] => throw new InvalidExecutionContextException($"{typeof(ReadOnlyTexture2D<T>)}[{typeof(int)}, {typeof(int)}]");

    /// <summary>Gets a single <typeparamref name="T"/> value from the current texture.</summary>
    public ref readonly T this[Int2 xy] => throw new InvalidExecutionContextException($"{typeof(ReadOnlyTexture2D<T>)}[{typeof(Int2)}]");

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"ComputeSharp.ReadOnlyTexture2D<{typeof(T)}>[{Width}, {Height}]";
    }
}

/// <summary>A <see langword="class"/> representing a typed readonly 2D texture with a normalized pixel view.</summary>
[DebuggerTypeProxy(typeof(Texture2DDebugView<>))]
[DebuggerDisplay("{ToString(),raw}")]
public sealed class ReadOnlyTexture2D<T, TPixel> : Texture2D<T>, IReadOnlyNormalizedTexture2D<TPixel>
    where T : unmanaged, IPixel<T, TPixel>
    where TPixel : unmanaged
{
    internal ReadOnlyTexture2D(GraphicsDevice device, int width, int height, AllocationMode allocationMode)
        : base(device, width, height, ResourceType.ReadOnly, allocationMode)
    {
    }

    /// <summary>Gets a single <typeparamref name="TPixel"/> value from the current texture.</summary>
    public ref readonly TPixel this[int x, int y] => throw new InvalidExecutionContextException($"{typeof(ReadOnlyTexture2D<T, TPixel>)}[{typeof(int)}, {typeof(int)}]");

    /// <summary>Gets a single <typeparamref name="TPixel"/> value from the current texture.</summary>
    public ref readonly TPixel this[Int2 xy] => throw new InvalidExecutionContextException($"{typeof(ReadOnlyTexture2D<T, TPixel>)}[{typeof(Int2)}]");

    /// <summary>Samples the current texture at a specified location.</summary>
    public ref readonly TPixel Sample(float u, float v) => throw new InvalidExecutionContextException($"{typeof(ReadOnlyTexture2D<T, TPixel>)}.{nameof(Sample)}({typeof(float)}, {typeof(float)})");

    /// <summary>Samples the current texture at a specified location.</summary>
    public ref readonly TPixel Sample(Float2 uv) => throw new InvalidExecutionContextException($"{typeof(ReadOnlyTexture2D<T, TPixel>)}.{nameof(Sample)}({typeof(Float2)})");

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"ComputeSharp.ReadOnlyTexture2D<{typeof(T)}, {typeof(TPixel)}>[{Width}, {Height}]";
    }
}

/// <summary>A <see langword="class"/> representing a typed readwrite 2D texture stored in device memory.</summary>
[DebuggerTypeProxy(typeof(Texture2DDebugView<>))]
[DebuggerDisplay("{ToString(),raw}")]
public sealed partial class ReadWriteTexture2D<T> : Texture2D<T>
    where T : unmanaged
{
    internal ReadWriteTexture2D(GraphicsDevice device, int width, int height, AllocationMode allocationMode)
        : base(device, width, height, ResourceType.ReadWrite, allocationMode)
    {
    }

    /// <summary>Gets a single <typeparamref name="T"/> value from the current texture.</summary>
    public ref T this[int x, int y] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture2D<T>)}[{typeof(int)}, {typeof(int)}]");

    /// <summary>Gets a single <typeparamref name="T"/> value from the current texture.</summary>
    public ref T this[Int2 xy] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture2D<T>)}[{typeof(Int2)}]");

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"ComputeSharp.ReadWriteTexture2D<{typeof(T)}>[{Width}, {Height}]";
    }
}

/// <summary>A <see langword="class"/> representing a typed readwrite 2D texture with a normalized pixel view.</summary>
[DebuggerTypeProxy(typeof(Texture2DDebugView<>))]
[DebuggerDisplay("{ToString(),raw}")]
public sealed partial class ReadWriteTexture2D<T, TPixel> : Texture2D<T>, IReadWriteNormalizedTexture2D<TPixel>
    where T : unmanaged, IPixel<T, TPixel>
    where TPixel : unmanaged
{
    internal ReadWriteTexture2D(GraphicsDevice device, int width, int height, AllocationMode allocationMode)
        : base(device, width, height, ResourceType.ReadWrite, allocationMode)
    {
    }

    /// <summary>Gets a single <typeparamref name="TPixel"/> value from the current texture.</summary>
    public ref TPixel this[int x, int y] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture2D<T, TPixel>)}[{typeof(int)}, {typeof(int)}]");

    /// <summary>Gets a single <typeparamref name="TPixel"/> value from the current texture.</summary>
    public ref TPixel this[Int2 xy] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture2D<T, TPixel>)}[{typeof(Int2)}]");

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"ComputeSharp.ReadWriteTexture2D<{typeof(T)}, {typeof(TPixel)}>[{Width}, {Height}]";
    }
}

partial class ReadWriteTexture2D<T>
{
    private ReadOnly? readOnlyWrapper;

    /// <summary>Gets a readonly view of the current texture, usable after a transition to the readonly state.</summary>
    public IReadOnlyTexture2D<T> AsReadOnly()
    {
        using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
        using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

        GraphicsDevice.ThrowIfDeviceLost();

        ThrowIfIsNotInReadOnlyState();

        ReadOnly? readOnlyWrapper = this.readOnlyWrapper;

        if (readOnlyWrapper is not null)
        {
            return readOnlyWrapper;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static ReadOnly InitializeWrapper(ReadWriteTexture2D<T> texture)
        {
            return texture.readOnlyWrapper = new(texture);
        }

        return InitializeWrapper(this);
    }

    /// <inheritdoc/>
    protected override void DangerousOnDispose()
    {
        base.DangerousOnDispose();

        this.readOnlyWrapper?.Dispose();
    }

    private sealed class ReadOnly : ReferenceTrackedObject, IReadOnlyTexture2D<T>, IPortableReadOnlyResource
    {
        private readonly ReadWriteTexture2D<T> owner;

        public ReadOnly(ReadWriteTexture2D<T> owner)
        {
            using ReferenceTracker.Lease _0 = GetReferenceTracker().GetLease();

            owner.GetReferenceTracker().DangerousAddRef();

            this.owner = owner;
        }

        /// <inheritdoc/>
        public ref readonly T this[int x, int y] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture2D<T>.ReadOnly)}[{typeof(int)}, {typeof(int)}]");

        /// <inheritdoc/>
        public ref readonly T this[Int2 xy] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture2D<T>.ReadOnly)}[{typeof(Int2)}]");

        /// <inheritdoc/>
        public ref readonly T Sample(float u, float v) => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture2D<T>.ReadOnly)}.{nameof(Sample)}({typeof(float)}, {typeof(float)})");

        /// <inheritdoc/>
        public ref readonly T Sample(Float2 uv) => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture2D<T>.ReadOnly)}.{nameof(Sample)}({typeof(Float2)})");

        /// <inheritdoc/>
        public int Width => this.owner.Width;

        /// <inheritdoc/>
        public int Height => this.owner.Height;

        /// <inheritdoc/>
        public GraphicsDevice GraphicsDevice => this.owner.GraphicsDevice;

        PortableResourceView IPortableReadOnlyResource.ValidateAndGetView(GraphicsDevice device, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            using ReferenceTracker.Lease _1 = this.owner.GetReferenceTracker().GetLease();

            this.owner.ThrowIfDeviceMismatch(device);

            return this.owner.GetView(PortableResourceKind.Texture);
        }

        protected override void DangerousOnDispose()
        {
            this.owner.GetReferenceTracker().DangerousRelease();
        }
    }
}

partial class ReadWriteTexture2D<T, TPixel>
{
    private ReadOnly? readOnlyWrapper;

    /// <summary>Gets a readonly view of the current texture, usable after a transition to the readonly state.</summary>
    internal IReadOnlyNormalizedTexture2D<TPixel> AsReadOnly()
    {
        using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
        using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

        GraphicsDevice.ThrowIfDeviceLost();

        ThrowIfIsNotInReadOnlyState();

        ReadOnly? readOnlyWrapper = this.readOnlyWrapper;

        if (readOnlyWrapper is not null)
        {
            return readOnlyWrapper;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static ReadOnly InitializeWrapper(ReadWriteTexture2D<T, TPixel> texture)
        {
            return texture.readOnlyWrapper = new(texture);
        }

        return InitializeWrapper(this);
    }

    /// <inheritdoc/>
    protected override void DangerousOnDispose()
    {
        base.DangerousOnDispose();

        this.readOnlyWrapper?.Dispose();
    }

    private sealed class ReadOnly : ReferenceTrackedObject, IReadOnlyNormalizedTexture2D<TPixel>, IPortableReadOnlyResource
    {
        private readonly ReadWriteTexture2D<T, TPixel> owner;

        public ReadOnly(ReadWriteTexture2D<T, TPixel> owner)
        {
            using ReferenceTracker.Lease _0 = GetReferenceTracker().GetLease();

            owner.GetReferenceTracker().DangerousAddRef();

            this.owner = owner;
        }

        /// <inheritdoc/>
        public ref readonly TPixel this[int x, int y] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture2D<T, TPixel>.ReadOnly)}[{typeof(int)}, {typeof(int)}]");

        /// <inheritdoc/>
        public ref readonly TPixel this[Int2 xy] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture2D<T, TPixel>.ReadOnly)}[{typeof(Int2)}]");

        /// <inheritdoc/>
        public ref readonly TPixel Sample(float u, float v) => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture2D<T, TPixel>.ReadOnly)}.{nameof(Sample)}({typeof(float)}, {typeof(float)})");

        /// <inheritdoc/>
        public ref readonly TPixel Sample(Float2 uv) => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture2D<T, TPixel>.ReadOnly)}.{nameof(Sample)}({typeof(Float2)})");

        /// <inheritdoc/>
        public int Width => this.owner.Width;

        /// <inheritdoc/>
        public int Height => this.owner.Height;

        /// <inheritdoc/>
        public GraphicsDevice GraphicsDevice => this.owner.GraphicsDevice;

        PortableResourceView IPortableReadOnlyResource.ValidateAndGetView(GraphicsDevice device, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            using ReferenceTracker.Lease _1 = this.owner.GetReferenceTracker().GetLease();

            this.owner.ThrowIfDeviceMismatch(device);

            return this.owner.GetView(PortableResourceKind.Texture);
        }

        protected override void DangerousOnDispose()
        {
            this.owner.GetReferenceTracker().DangerousRelease();
        }
    }
}

/// <summary>A <see langword="class"/> representing a typed readonly 3D texture stored in device memory.</summary>
[DebuggerTypeProxy(typeof(Texture3DDebugView<>))]
[DebuggerDisplay("{ToString(),raw}")]
public sealed class ReadOnlyTexture3D<T> : Texture3D<T>
    where T : unmanaged
{
    internal ReadOnlyTexture3D(GraphicsDevice device, int width, int height, int depth, AllocationMode allocationMode)
        : base(device, width, height, depth, ResourceType.ReadOnly, allocationMode)
    {
    }

    /// <summary>Gets a single <typeparamref name="T"/> value from the current texture.</summary>
    public ref readonly T this[int x, int y, int z] => throw new InvalidExecutionContextException($"{typeof(ReadOnlyTexture3D<T>)}[{typeof(int)}, {typeof(int)}, {typeof(int)}]");

    /// <summary>Gets a single <typeparamref name="T"/> value from the current texture.</summary>
    public ref readonly T this[Int3 xyz] => throw new InvalidExecutionContextException($"{typeof(ReadOnlyTexture3D<T>)}[{typeof(Int3)}]");

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"ComputeSharp.ReadOnlyTexture3D<{typeof(T)}>[{Width}, {Height}, {Depth}]";
    }
}

/// <summary>A <see langword="class"/> representing a typed readonly 3D texture with a normalized pixel view.</summary>
[DebuggerTypeProxy(typeof(Texture3DDebugView<>))]
[DebuggerDisplay("{ToString(),raw}")]
public sealed class ReadOnlyTexture3D<T, TPixel> : Texture3D<T>, IReadOnlyNormalizedTexture3D<TPixel>
    where T : unmanaged, IPixel<T, TPixel>
    where TPixel : unmanaged
{
    internal ReadOnlyTexture3D(GraphicsDevice device, int width, int height, int depth, AllocationMode allocationMode)
        : base(device, width, height, depth, ResourceType.ReadOnly, allocationMode)
    {
    }

    /// <summary>Gets a single <typeparamref name="TPixel"/> value from the current texture.</summary>
    public ref readonly TPixel this[int x, int y, int z] => throw new InvalidExecutionContextException($"{typeof(ReadOnlyTexture3D<T, TPixel>)}[{typeof(int)}, {typeof(int)}, {typeof(int)}]");

    /// <summary>Gets a single <typeparamref name="TPixel"/> value from the current texture.</summary>
    public ref readonly TPixel this[Int3 xyz] => throw new InvalidExecutionContextException($"{typeof(ReadOnlyTexture3D<T, TPixel>)}[{typeof(Int3)}]");

    /// <summary>Samples the current texture at a specified location.</summary>
    public ref readonly TPixel Sample(float u, float v, float w) => throw new InvalidExecutionContextException($"{typeof(ReadOnlyTexture3D<T, TPixel>)}.{nameof(Sample)}({typeof(float)}, {typeof(float)}, {typeof(float)})");

    /// <summary>Samples the current texture at a specified location.</summary>
    public ref readonly TPixel Sample(Float3 uvw) => throw new InvalidExecutionContextException($"{typeof(ReadOnlyTexture3D<T, TPixel>)}.{nameof(Sample)}({typeof(Float3)})");

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"ComputeSharp.ReadOnlyTexture3D<{typeof(T)}, {typeof(TPixel)}>[{Width}, {Height}, {Depth}]";
    }
}

/// <summary>A <see langword="class"/> representing a typed readwrite 3D texture stored in device memory.</summary>
[DebuggerTypeProxy(typeof(Texture3DDebugView<>))]
[DebuggerDisplay("{ToString(),raw}")]
public sealed partial class ReadWriteTexture3D<T> : Texture3D<T>
    where T : unmanaged
{
    internal ReadWriteTexture3D(GraphicsDevice device, int width, int height, int depth, AllocationMode allocationMode)
        : base(device, width, height, depth, ResourceType.ReadWrite, allocationMode)
    {
    }

    /// <summary>Gets a single <typeparamref name="T"/> value from the current texture.</summary>
    public ref T this[int x, int y, int z] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture3D<T>)}[{typeof(int)}, {typeof(int)}, {typeof(int)}]");

    /// <summary>Gets a single <typeparamref name="T"/> value from the current texture.</summary>
    public ref T this[Int3 xyz] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture3D<T>)}[{typeof(Int3)}]");

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"ComputeSharp.ReadWriteTexture3D<{typeof(T)}>[{Width}, {Height}, {Depth}]";
    }
}

/// <summary>A <see langword="class"/> representing a typed readwrite 3D texture with a normalized pixel view.</summary>
[DebuggerTypeProxy(typeof(Texture3DDebugView<>))]
[DebuggerDisplay("{ToString(),raw}")]
public sealed partial class ReadWriteTexture3D<T, TPixel> : Texture3D<T>, IReadWriteNormalizedTexture3D<TPixel>
    where T : unmanaged, IPixel<T, TPixel>
    where TPixel : unmanaged
{
    internal ReadWriteTexture3D(GraphicsDevice device, int width, int height, int depth, AllocationMode allocationMode)
        : base(device, width, height, depth, ResourceType.ReadWrite, allocationMode)
    {
    }

    /// <summary>Gets a single <typeparamref name="TPixel"/> value from the current texture.</summary>
    public ref TPixel this[int x, int y, int z] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture3D<T, TPixel>)}[{typeof(int)}, {typeof(int)}, {typeof(int)}]");

    /// <summary>Gets a single <typeparamref name="TPixel"/> value from the current texture.</summary>
    public ref TPixel this[Int3 xyz] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture3D<T, TPixel>)}[{typeof(Int3)}]");

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"ComputeSharp.ReadWriteTexture3D<{typeof(T)}, {typeof(TPixel)}>[{Width}, {Height}, {Depth}]";
    }
}

partial class ReadWriteTexture3D<T>
{
    private ReadOnly? readOnlyWrapper;

    /// <summary>Gets a readonly view of the current texture, usable after a transition to the readonly state.</summary>
    public IReadOnlyTexture3D<T> AsReadOnly()
    {
        using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
        using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

        GraphicsDevice.ThrowIfDeviceLost();

        ThrowIfIsNotInReadOnlyState();

        ReadOnly? readOnlyWrapper = this.readOnlyWrapper;

        if (readOnlyWrapper is not null)
        {
            return readOnlyWrapper;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static ReadOnly InitializeWrapper(ReadWriteTexture3D<T> texture)
        {
            return texture.readOnlyWrapper = new(texture);
        }

        return InitializeWrapper(this);
    }

    /// <inheritdoc/>
    protected override void DangerousOnDispose()
    {
        base.DangerousOnDispose();

        this.readOnlyWrapper?.Dispose();
    }

    private sealed class ReadOnly : ReferenceTrackedObject, IReadOnlyTexture3D<T>, IPortableReadOnlyResource
    {
        private readonly ReadWriteTexture3D<T> owner;

        public ReadOnly(ReadWriteTexture3D<T> owner)
        {
            using ReferenceTracker.Lease _0 = GetReferenceTracker().GetLease();

            owner.GetReferenceTracker().DangerousAddRef();

            this.owner = owner;
        }

        /// <inheritdoc/>
        public ref readonly T this[int x, int y, int z] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture3D<T>.ReadOnly)}[{typeof(int)}, {typeof(int)}, {typeof(int)}]");

        /// <inheritdoc/>
        public ref readonly T this[Int3 xyz] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture3D<T>.ReadOnly)}[{typeof(Int3)}]");

        /// <inheritdoc/>
        public ref readonly T Sample(float u, float v, float w) => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture3D<T>.ReadOnly)}.{nameof(Sample)}({typeof(float)}, {typeof(float)}, {typeof(float)})");

        /// <inheritdoc/>
        public ref readonly T Sample(Float3 uvw) => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture3D<T>.ReadOnly)}.{nameof(Sample)}({typeof(Float3)})");

        /// <inheritdoc/>
        public int Width => this.owner.Width;

        /// <inheritdoc/>
        public int Height => this.owner.Height;

        /// <inheritdoc/>
        public int Depth => this.owner.Depth;

        /// <inheritdoc/>
        public GraphicsDevice GraphicsDevice => this.owner.GraphicsDevice;

        PortableResourceView IPortableReadOnlyResource.ValidateAndGetView(GraphicsDevice device, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            using ReferenceTracker.Lease _1 = this.owner.GetReferenceTracker().GetLease();

            this.owner.ThrowIfDeviceMismatch(device);

            return this.owner.GetView(PortableResourceKind.Texture);
        }

        protected override void DangerousOnDispose()
        {
            this.owner.GetReferenceTracker().DangerousRelease();
        }
    }
}

partial class ReadWriteTexture3D<T, TPixel>
{
    private ReadOnly? readOnlyWrapper;

    /// <summary>Gets a readonly view of the current texture, usable after a transition to the readonly state.</summary>
    public IReadOnlyNormalizedTexture3D<TPixel> AsReadOnly()
    {
        using ReferenceTracker.Lease _0 = GraphicsDevice.GetReferenceTracker().GetLease();
        using ReferenceTracker.Lease _1 = GetReferenceTracker().GetLease();

        GraphicsDevice.ThrowIfDeviceLost();

        ThrowIfIsNotInReadOnlyState();

        ReadOnly? readOnlyWrapper = this.readOnlyWrapper;

        if (readOnlyWrapper is not null)
        {
            return readOnlyWrapper;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static ReadOnly InitializeWrapper(ReadWriteTexture3D<T, TPixel> texture)
        {
            return texture.readOnlyWrapper = new(texture);
        }

        return InitializeWrapper(this);
    }

    /// <inheritdoc/>
    protected override void DangerousOnDispose()
    {
        base.DangerousOnDispose();

        this.readOnlyWrapper?.Dispose();
    }

    private sealed class ReadOnly : ReferenceTrackedObject, IReadOnlyNormalizedTexture3D<TPixel>, IPortableReadOnlyResource
    {
        private readonly ReadWriteTexture3D<T, TPixel> owner;

        public ReadOnly(ReadWriteTexture3D<T, TPixel> owner)
        {
            using ReferenceTracker.Lease _0 = GetReferenceTracker().GetLease();

            owner.GetReferenceTracker().DangerousAddRef();

            this.owner = owner;
        }

        /// <inheritdoc/>
        public ref readonly TPixel this[int x, int y, int z] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture3D<T, TPixel>.ReadOnly)}[{typeof(int)}, {typeof(int)}, {typeof(int)}]");

        /// <inheritdoc/>
        public ref readonly TPixel this[Int3 xyz] => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture3D<T, TPixel>.ReadOnly)}[{typeof(Int3)}]");

        /// <inheritdoc/>
        public ref readonly TPixel Sample(float u, float v, float w) => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture3D<T, TPixel>.ReadOnly)}.{nameof(Sample)}({typeof(float)}, {typeof(float)}, {typeof(float)})");

        /// <inheritdoc/>
        public ref readonly TPixel Sample(Float3 uvw) => throw new InvalidExecutionContextException($"{typeof(ReadWriteTexture3D<T, TPixel>.ReadOnly)}.{nameof(Sample)}({typeof(Float3)})");

        /// <inheritdoc/>
        public int Width => this.owner.Width;

        /// <inheritdoc/>
        public int Height => this.owner.Height;

        /// <inheritdoc/>
        public int Depth => this.owner.Depth;

        /// <inheritdoc/>
        public GraphicsDevice GraphicsDevice => this.owner.GraphicsDevice;

        PortableResourceView IPortableReadOnlyResource.ValidateAndGetView(GraphicsDevice device, out ReferenceTracker.Lease lease)
        {
            lease = GetReferenceTracker().GetLease();

            using ReferenceTracker.Lease _1 = this.owner.GetReferenceTracker().GetLease();

            this.owner.ThrowIfDeviceMismatch(device);

            return this.owner.GetView(PortableResourceKind.Texture);
        }

        protected override void DangerousOnDispose()
        {
            this.owner.GetReferenceTracker().DangerousRelease();
        }
    }
}
