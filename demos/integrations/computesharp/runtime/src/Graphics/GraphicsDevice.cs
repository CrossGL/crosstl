using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using ComputeSharp.Graphics.Helpers;
using ComputeSharp.Interop;
using ComputeSharp.Portable.Diagnostics;
using ComputeSharp.Portable.Native;
using ComputeSharp.Portable.Shaders;
using ComputeSharp.Shaders.Loading;

namespace ComputeSharp;

/// <summary>
/// A <see langword="class"/> that represents a translated compute device (Metal or OpenGL).
/// </summary>
[DebuggerDisplay("{ToString(),raw}")]
public sealed unsafe partial class GraphicsDevice : IReferenceTrackedObject
{
    private ReferenceTracker referenceTracker;
    private readonly List<PipelineData> cachedPipelineData;
    private readonly Lock pipelineLock = new();
    private IntPtr staticSampler;
    private int deviceLost;

    /// <summary>
    /// Occurs when the device is lost. Translated backends do not report device removal,
    /// so failures surface as exceptions from the operation that observed them.
    /// </summary>
    public event EventHandler<DeviceLostEventArgs>? DeviceLost;

    internal GraphicsDevice(NativeDevice native)
    {
        using ReferenceTracker.Lease _0 = ReferenceTracker.Create(this, out this.referenceTracker);

        Native = native;
        ShaderLibrary = TranslatedShaderLibrary.Load();
        NativeDeviceInfo info = native.Info;

        Name = native.Name;
        Luid = Luid.FromValue(DeviceHelper.StableLuid(native.Name, native.Driver));
        DedicatedMemorySize = (nuint)info.DedicatedMemory;
        SharedMemorySize = (nuint)info.SharedMemory;
        IsHardwareAccelerated = info.IsHardware != 0;
        ComputeUnits = info.ComputeUnits;
        WavefrontSize = info.WavefrontSize;
        this.cachedPipelineData = [];
        this.staticSampler = native.CreateSampler(CsBackend.FilterLinear, CsBackend.AddressMirror);

        EvidenceLog.Write("device", json =>
        {
            json.WriteString("name", Name);
            json.WriteString("api", native.Api);
            json.WriteString("driver", native.Driver);
            json.WriteBoolean("hardware", IsHardwareAccelerated);
            json.WriteBoolean("fp64", info.SupportsFp64 != 0);
            json.WriteNumber("computeUnits", ComputeUnits);
            json.WriteNumber("wavefrontSize", WavefrontSize);
            json.WriteString("manifest", ShaderLibrary.ManifestPath);
            json.WriteString("translatorCommit", ShaderLibrary.TranslatorCommit);
            json.WriteNumber("translatedShaders", ShaderLibrary.Count);
        });
    }

    /// <summary>Gets the locally unique identifier for the current device.</summary>
    public Luid Luid { get; }

    /// <summary>Gets the name of the current <see cref="GraphicsDevice"/> instance.</summary>
    public string Name { get; }

    /// <summary>Gets the number of bytes of dedicated video memory that are not shared with the CPU.</summary>
    public nuint DedicatedMemorySize { get; }

    /// <summary>Gets the number of bytes of shared system memory.</summary>
    public nuint SharedMemorySize { get; }

    /// <summary>Gets whether or not the current device has hardware acceleration.</summary>
    public bool IsHardwareAccelerated { get; }

    /// <summary>Gets the number of total lanes on the current device.</summary>
    public uint ComputeUnits { get; }

    /// <summary>Gets the number of lanes in a SIMD wave on the current device.</summary>
    public uint WavefrontSize { get; }

    internal NativeDevice Native { get; }

    internal TranslatedShaderLibrary ShaderLibrary { get; }

    internal IntPtr StaticSampler => this.staticSampler;

    /// <summary>Gets whether translated memory is coherent with the host after a submission completes.</summary>
    internal bool IsCacheCoherentUMA => true;

    /// <summary>Checks whether the current device supports double precision floating point operations in shaders.</summary>
    public bool IsDoublePrecisionSupportAvailable()
    {
        using ReferenceTracker.Lease _0 = GetReferenceTracker().GetLease();

        ThrowIfDeviceLost();

        return Native.Info.SupportsFp64 != 0;
    }

    /// <summary>Checks whether the current device supports the creation of <see cref="ReadOnlyTexture1D{T}"/> resources for a specified type <typeparamref name="T"/>.</summary>
    public bool IsReadOnlyTexture1DSupportedForType<T>()
        where T : unmanaged
    {
        return IsTextureSupported<T>(CsBackend.Texture1D, CsBackend.UsageShaderRead);
    }

    /// <summary>Checks whether the current device supports the creation of <see cref="ReadWriteTexture1D{T}"/> resources for a specified type <typeparamref name="T"/>.</summary>
    public bool IsReadWriteTexture1DSupportedForType<T>()
        where T : unmanaged
    {
        return IsTextureSupported<T>(CsBackend.Texture1D, CsBackend.UsageShaderRead | CsBackend.UsageShaderWrite);
    }

    /// <summary>Checks whether the current device supports the creation of <see cref="ReadOnlyTexture2D{T}"/> resources for a specified type <typeparamref name="T"/>.</summary>
    public bool IsReadOnlyTexture2DSupportedForType<T>()
        where T : unmanaged
    {
        return IsTextureSupported<T>(CsBackend.Texture2D, CsBackend.UsageShaderRead);
    }

    /// <summary>Checks whether the current device supports the creation of <see cref="ReadWriteTexture2D{T}"/> resources for a specified type <typeparamref name="T"/>.</summary>
    public bool IsReadWriteTexture2DSupportedForType<T>()
        where T : unmanaged
    {
        return IsTextureSupported<T>(CsBackend.Texture2D, CsBackend.UsageShaderRead | CsBackend.UsageShaderWrite);
    }

    /// <summary>Checks whether the current device supports the creation of <see cref="ReadOnlyTexture3D{T}"/> resources for a specified type <typeparamref name="T"/>.</summary>
    public bool IsReadOnlyTexture3DSupportedForType<T>()
        where T : unmanaged
    {
        return IsTextureSupported<T>(CsBackend.Texture3D, CsBackend.UsageShaderRead);
    }

    /// <summary>Checks whether the current device supports the creation of <see cref="ReadWriteTexture3D{T}"/> resources for a specified type <typeparamref name="T"/>.</summary>
    public bool IsReadWriteTexture3DSupportedForType<T>()
        where T : unmanaged
    {
        return IsTextureSupported<T>(CsBackend.Texture3D, CsBackend.UsageShaderRead | CsBackend.UsageShaderWrite);
    }

    internal bool IsTextureSupported<T>(uint dimension, uint usage)
        where T : unmanaged
    {
        using ReferenceTracker.Lease _0 = GetReferenceTracker().GetLease();

        ThrowIfDeviceLost();

        return Native.IsFormatSupported(dimension, DXGIFormatHelper.GetForType<T>(), usage);
    }

    internal void RegisterPipelineData(PipelineData pipelineData)
    {
        lock (this.pipelineLock)
        {
            this.cachedPipelineData.Add(pipelineData);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void ThrowIfDeviceLost()
    {
        if (Volatile.Read(ref this.deviceLost) != 0)
        {
            static void Throw(object self) => throw new InvalidOperationException($"""The device "{self}" has been lost and can no longer be used.""");

            Throw(this);
        }
    }

    /// <summary>Marks the device as lost after a backend failure that leaves it unusable.</summary>
    internal void MarkDeviceLost(DeviceLostReason reason)
    {
        if (Interlocked.Exchange(ref this.deviceLost, 1) == 0)
        {
            _ = ThreadPool.QueueUserWorkItem(static state =>
            {
                (GraphicsDevice device, DeviceLostReason reason) = ((GraphicsDevice, DeviceLostReason))state!;
                device.DeviceLost?.Invoke(device, new DeviceLostEventArgs(reason));
            }, (this, reason));
        }
    }

    void IReferenceTrackedObject.DangerousOnDispose()
    {
        DeviceHelper.NotifyDisposedDevice(this);

        // Fields are null when the constructor threw before initializing them.
        if (this.cachedPipelineData is not null)
        {
            foreach (PipelineData pipelineData in this.cachedPipelineData)
            {
                pipelineData.Dispose();
            }
        }

        if (Native is not null)
        {
            Native.DestroySampler(Interlocked.Exchange(ref this.staticSampler, IntPtr.Zero));
            Native.Dispose();
        }
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"[{Luid}] {Name}";
    }
}
