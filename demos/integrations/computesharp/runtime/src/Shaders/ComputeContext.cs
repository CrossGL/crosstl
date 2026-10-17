using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using ComputeSharp.Descriptors;
using ComputeSharp.Interop;
using ComputeSharp.Portable;
using ComputeSharp.Portable.Native;
using ComputeSharp.Portable.Shaders;
using ComputeSharp.Resources.Interop;
using ComputeSharp.Shaders.Loading;

namespace ComputeSharp;

/// <summary>
/// A context to batch dispatches, barriers, clears and state transitions on a translated
/// device. Operations execute in order when the context is disposed.
/// </summary>
public struct ComputeContext : IDisposable, IAsyncDisposable
{
    private GraphicsDevice? device;
    private NativeCommandList? commandList;

    internal ComputeContext(GraphicsDevice device)
    {
        this.device = device;
        this.commandList = null;

        device.GetReferenceTracker().DangerousAddRef();
    }

    /// <summary>Gets the <see cref="ComputeSharp.GraphicsDevice"/> instance for the current context.</summary>
    public readonly GraphicsDevice GraphicsDevice
    {
        get
        {
            default(InvalidOperationException).ThrowIf(this.device is null);

            return this.device;
        }
    }

    /// <summary>Inserts a resource barrier; like the original runtime, it requires previously recorded work.</summary>
    internal readonly void Barrier()
    {
        default(InvalidOperationException).ThrowIf(this.device is null);
        default(InvalidOperationException).ThrowIf(this.commandList is null);

        this.commandList!.Barrier();
    }

    /// <summary>Records a state transition, which orders later work after earlier work on the resource.</summary>
    internal readonly void Transition()
    {
        default(InvalidOperationException).ThrowIf(this.device is null);

        GetCommandList().Barrier();
    }

    internal readonly void ClearBuffer(IntPtr buffer, long sizeInBytes)
    {
        default(InvalidOperationException).ThrowIf(this.device is null);

        GetCommandList().FillBuffer(buffer, 0, (ulong)sizeInBytes, 0);
    }

    internal readonly void UploadTexture(PortableResourceView view, int width, int height, int depth, ReadOnlySpan<byte> data, long rowPitch, long slicePitch)
    {
        default(InvalidOperationException).ThrowIf(this.device is null);

        GetCommandList().UploadTexture(view.Handle, 0, 0, 0, (uint)width, (uint)height, (uint)depth, data, (ulong)rowPitch, (ulong)slicePitch);
    }

    internal readonly void Run<T>(int x, in T shader)
        where T : struct, IComputeShader, IComputeShaderDescriptor<T>
    {
        Run(x, 1, 1, in shader);
    }

    internal readonly void Run<T>(int x, int y, in T shader)
        where T : struct, IComputeShader, IComputeShaderDescriptor<T>
    {
        Run(x, y, 1, in shader);
    }

    internal readonly void Run<T>(int x, int y, int z, in T shader)
        where T : struct, IComputeShader, IComputeShaderDescriptor<T>
    {
        default(InvalidOperationException).ThrowIf(this.device is null);
        default(ArgumentOutOfRangeException).ThrowIfNegativeOrZero(x);
        default(ArgumentOutOfRangeException).ThrowIfNegativeOrZero(y);
        default(ArgumentOutOfRangeException).ThrowIfNegativeOrZero(z);

        int groupsX = Math.DivRem(x, T.ThreadsX, out int modX) + (modX == 0 ? 0 : 1);
        int groupsY = Math.DivRem(y, T.ThreadsY, out int modY) + (modY == 0 ? 0 : 1);
        int groupsZ = Math.DivRem(z, T.ThreadsZ, out int modZ) + (modZ == 0 ? 0 : 1);

        default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(groupsX, 1, D3D12Limits.MaxThreadGroupsPerDimension);
        default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(groupsY, 1, D3D12Limits.MaxThreadGroupsPerDimension, nameof(groupsX));
        default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(groupsZ, 1, D3D12Limits.MaxThreadGroupsPerDimension, nameof(groupsX));

        PipelineData pipelineData = PipelineDataLoader<T>.GetPipelineData(this.device);
        NativeCommandList commandList = GetCommandList();
        DispatchBuilder builder = new(this.device, T.ResourceDescriptorRanges.Length, rangeOffset: 0);

        PortableConstantBufferLoader constantBufferLoader = new(builder);
        T.LoadConstantBuffer(in shader, ref constantBufferLoader, x, y, z);

        PortableGraphicsResourceLoader graphicsResourceLoader = new(builder);
        T.LoadGraphicsResources(in shader, ref graphicsResourceLoader);

        DispatchRecorder.Record(commandList, pipelineData, builder, groupsX, groupsY, groupsZ, x, y, z);
    }

    internal readonly void Run<T, TPixel>(IReadWriteNormalizedTexture2D<TPixel> texture, in T shader)
        where T : struct, IComputeShader<TPixel>, IComputeShaderDescriptor<T>
        where TPixel : unmanaged
    {
        default(InvalidOperationException).ThrowIf(this.device is null);

        int x = texture.Width;
        int y = texture.Height;
        int groupsX = Math.DivRem(x, T.ThreadsX, out int modX) + (modX == 0 ? 0 : 1);
        int groupsY = Math.DivRem(y, T.ThreadsY, out int modY) + (modY == 0 ? 0 : 1);

        default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(groupsX, 1, D3D12Limits.MaxThreadGroupsPerDimension);
        default(ArgumentOutOfRangeException).ThrowIfNotBetweenOrEqual(groupsY, 1, D3D12Limits.MaxThreadGroupsPerDimension, nameof(groupsX));

        PipelineData pipelineData = PipelineDataLoader<T>.GetPipelineData(this.device);
        NativeCommandList commandList = GetCommandList();

        // The implicit output texture takes the first descriptor range; the shader's own
        // resources follow it, as in the upstream root signature.
        DispatchBuilder builder = new(this.device, T.ResourceDescriptorRanges.Length, rangeOffset: 1);

        PortableConstantBufferLoader constantBufferLoader = new(builder);
        T.LoadConstantBuffer(in shader, ref constantBufferLoader, x, y, 1);

        PortableGraphicsResourceLoader graphicsResourceLoader = new(builder);
        T.LoadGraphicsResources(in shader, ref graphicsResourceLoader);

        PortableResourceView output = ((IPortableReadOnlyResource)texture).ValidateAndGetView(this.device, out ReferenceTracker.Lease lease);
        lease.Dispose();
        builder.SetView(0, output);

        DispatchRecorder.Record(commandList, pipelineData, builder, groupsX, groupsY, 1, x, y, 1);
    }

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Dispose()
    {
        default(InvalidOperationException).ThrowIf(this.device is null);

        GraphicsDevice device = this.device;

        this.device = null;

        NativeCommandList? commandList = this.commandList;

        this.commandList = null;

        try
        {
            if (commandList is not null)
            {
                try
                {
                    commandList.Submit();
                }
                finally
                {
                    commandList.Dispose();
                }
            }
        }
        finally
        {
            device.GetReferenceTracker().DangerousRelease();
        }
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync()
    {
        // Translated submissions complete synchronously; the returned task is already completed.
        Dispose();

        return ValueTask.CompletedTask;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    [UnscopedRef]
    private readonly NativeCommandList GetCommandList()
    {
        ref NativeCommandList? commandList = ref Unsafe.AsRef(in this.commandList);

        return commandList ??= new NativeCommandList(this.device!.Native);
    }
}
