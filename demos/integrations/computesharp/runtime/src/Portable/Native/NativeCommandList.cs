using System;
using System.Collections.Generic;

namespace ComputeSharp.Portable.Native;

/// <summary>
/// An ordered list of native operations that executes on <see cref="Submit"/>.
/// Every <see cref="ComputeContext"/> and every synchronous copy uses one list.
/// </summary>
internal sealed unsafe class NativeCommandList : IDisposable
{
    private const int ErrorCapacity = 8192;

    private readonly NativeDevice device;
    private readonly List<IntPtr> temporaryBuffers = [];
    private IntPtr list;
    private bool submitted;

    public NativeCommandList(NativeDevice device)
    {
        this.device = device;
        this.list = device.CreateList();
    }

    /// <summary>Gets the number of dispatches recorded so far.</summary>
    public int DispatchCount { get; private set; }

    /// <summary>Gets the dispatch sequence numbers recorded in this list.</summary>
    public List<long> DispatchSequence { get; } = [];

    public void Dispatch(IntPtr pipeline, uint groupsX, uint groupsY, uint groupsZ, NativeBinding[] bindings, int count)
    {
        ThrowIfSubmitted();
        DispatchCount++;
        if (BackendSelection.Fault == "omit-dispatch")
        {
            return;
        }

        IntPtr list = this.list;
        this.device.Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            fixed (NativeBinding* pointer = bindings)
            {
                NativeDevice.Check("dispatch", CsBackend.ListDispatch(list, pipeline, groupsX, groupsY, groupsZ, pointer, (uint)count, error, ErrorCapacity), error);
            }
        });
    }

    public void Barrier()
    {
        ThrowIfSubmitted();
        IntPtr list = this.list;
        this.device.Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            NativeDevice.Check("barrier", CsBackend.ListBarrier(list, error, ErrorCapacity), error);
        });
    }

    public void CopyBuffer(IntPtr source, ulong sourceOffset, IntPtr destination, ulong destinationOffset, ulong size)
    {
        ThrowIfSubmitted();
        IntPtr list = this.list;
        this.device.Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            NativeDevice.Check("buffer copy", CsBackend.ListCopyBuffer(list, source, sourceOffset, destination, destinationOffset, size, error, ErrorCapacity), error);
        });
    }

    public void FillBuffer(IntPtr buffer, ulong offset, ulong size, byte value)
    {
        ThrowIfSubmitted();
        IntPtr list = this.list;
        this.device.Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            NativeDevice.Check("buffer fill", CsBackend.ListFillBuffer(list, buffer, offset, size, value, error, ErrorCapacity), error);
        });
    }

    public void CopyTexture(IntPtr source, uint sourceX, uint sourceY, uint sourceZ, IntPtr destination, uint destinationX, uint destinationY, uint destinationZ, uint width, uint height, uint depth)
    {
        ThrowIfSubmitted();
        IntPtr list = this.list;
        this.device.Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            NativeDevice.Check("texture copy", CsBackend.ListCopyTexture(list, source, sourceX, sourceY, sourceZ, destination, destinationX, destinationY, destinationZ, width, height, depth, error, ErrorCapacity), error);
        });
    }

    /// <summary>
    /// Records a copy of host data into a texture region. The data is staged in a temporary
    /// device buffer that lives until the list is disposed, so the copy keeps command order.
    /// </summary>
    public void UploadTexture(IntPtr texture, uint x, uint y, uint z, uint width, uint height, uint depth, ReadOnlySpan<byte> data, ulong rowPitch, ulong slicePitch)
    {
        ThrowIfSubmitted();
        IntPtr staging = this.device.CreateBuffer((ulong)data.Length);
        this.temporaryBuffers.Add(staging);
        fixed (byte* pointer = data)
        {
            this.device.WriteBuffer(staging, 0, (IntPtr)pointer, (ulong)data.Length);
        }

        IntPtr list = this.list;
        this.device.Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            NativeDevice.Check("texture upload", CsBackend.ListCopyBufferToTexture(list, staging, 0, rowPitch, slicePitch, texture, x, y, z, width, height, depth, error, ErrorCapacity), error);
        });
    }

    /// <summary>Executes the recorded operations and waits for their completion.</summary>
    public void Submit()
    {
        ThrowIfSubmitted();
        this.submitted = true;
        IntPtr list = this.list;
        this.device.Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            NativeDevice.Check("submission", CsBackend.ListSubmit(list, error, ErrorCapacity), error);
        });
    }

    public void Dispose()
    {
        IntPtr list = this.list;
        this.list = IntPtr.Zero;
        if (list != IntPtr.Zero)
        {
            this.device.Invoke(() => CsBackend.ListDestroy(list));
        }

        foreach (IntPtr buffer in this.temporaryBuffers)
        {
            this.device.DestroyBuffer(buffer);
        }

        this.temporaryBuffers.Clear();
    }

    private void ThrowIfSubmitted()
    {
        if (this.submitted || this.list == IntPtr.Zero)
        {
            throw new InvalidOperationException("The command list has already been submitted.");
        }
    }
}
