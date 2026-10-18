using System;
using System.Collections.Generic;

namespace ComputeSharp.Portable.Native;

/// <summary>
/// An ordered list of native operations that executes on <see cref="Submit"/>.
/// Every <see cref="ComputeContext"/> and every synchronous copy uses one list.
/// Buffer offsets are relative to the visible region of each buffer; the list adds the
/// guard offset and verifies the guards of every buffer it used after the submission.
/// </summary>
internal sealed unsafe class NativeCommandList : IDisposable
{
    private const int ErrorCapacity = 8192;

    private readonly NativeDevice device;
    private readonly List<IntPtr> temporaryBuffers = [];
    private readonly HashSet<IntPtr> usedBuffers = [];
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

        NativeBinding[] native = new NativeBinding[count];
        int guardTarget = -1;
        for (int i = 0; i < count; i++)
        {
            NativeBinding binding = bindings[i];
            if (binding.Kind is NativeBindingKind.Buffer or NativeBindingKind.UniformBuffer)
            {
                ulong size = binding.Size != 0 ? binding.Size : this.device.BufferSize(binding.Object) - binding.Offset;
                binding.Offset = this.device.NativeOffset(binding.Object, binding.Offset, size, "buffer binding");
                binding.Size = size;
                this.usedBuffers.Add(binding.Object);
                if (guardTarget < 0 && binding.Kind == NativeBindingKind.Buffer && (binding.Access & CsBackend.UsageShaderWrite) != 0)
                {
                    guardTarget = i;
                }
            }

            native[i] = binding;
        }

        IntPtr list = this.list;
        this.device.Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            fixed (NativeBinding* pointer = native)
            {
                NativeDevice.Check("dispatch", CsBackend.ListDispatch(list, pipeline, groupsX, groupsY, groupsZ, pointer, (uint)count, error, ErrorCapacity), error);
            }
        });

        if (BackendSelection.Fault == "corrupt-guard" && guardTarget >= 0)
        {
            // Negative control: a device-side write of four bytes just past the visible
            // region of the first writable buffer, as an out-of-bounds store would do.
            IntPtr buffer = native[guardTarget].Object;
            ulong end = (this.device.BufferSize(buffer) + 3) & ~3UL;
            ulong offset = BufferGuards.Bytes + end;
            this.device.Invoke(() =>
            {
                byte* error = stackalloc byte[ErrorCapacity];
                error[0] = 0;
                NativeDevice.Check("buffer fill", CsBackend.ListFillBuffer(list, buffer, offset, 4, 0x3C, error, ErrorCapacity), error);
            });
        }
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
        ulong nativeSource = this.device.NativeOffset(source, sourceOffset, size, "buffer copy source");
        ulong nativeDestination = this.device.NativeOffset(destination, destinationOffset, size, "buffer copy destination");
        this.usedBuffers.Add(source);
        this.usedBuffers.Add(destination);
        IntPtr list = this.list;
        this.device.Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            NativeDevice.Check("buffer copy", CsBackend.ListCopyBuffer(list, source, nativeSource, destination, nativeDestination, size, error, ErrorCapacity), error);
        });
    }

    public void FillBuffer(IntPtr buffer, ulong offset, ulong size, byte value)
    {
        ThrowIfSubmitted();
        ulong nativeOffset = this.device.NativeOffset(buffer, offset, size, "buffer fill");
        this.usedBuffers.Add(buffer);
        IntPtr list = this.list;
        this.device.Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            NativeDevice.Check("buffer fill", CsBackend.ListFillBuffer(list, buffer, nativeOffset, size, value, error, ErrorCapacity), error);
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

        ulong nativeOffset = this.device.NativeOffset(staging, 0, (ulong)data.Length, "texture upload");
        this.usedBuffers.Add(staging);
        IntPtr list = this.list;
        this.device.Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            NativeDevice.Check("texture upload", CsBackend.ListCopyBufferToTexture(list, staging, nativeOffset, rowPitch, slicePitch, texture, x, y, z, width, height, depth, error, ErrorCapacity), error);
        });
    }

    /// <summary>
    /// Executes the recorded operations, waits for their completion and verifies the
    /// guard regions of every buffer the list used.
    /// </summary>
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

        foreach (IntPtr buffer in this.usedBuffers)
        {
            if (this.device.IsLiveBuffer(buffer))
            {
                this.device.VerifyGuards(buffer);
            }
        }
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
        this.usedBuffers.Clear();
    }

    private void ThrowIfSubmitted()
    {
        if (this.submitted || this.list == IntPtr.Zero)
        {
            throw new InvalidOperationException("The command list has already been submitted.");
        }
    }
}
