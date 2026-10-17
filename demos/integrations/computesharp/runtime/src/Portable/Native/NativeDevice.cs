using System;
using System.Collections.Concurrent;
using System.IO;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace ComputeSharp.Portable.Native;

/// <summary>Selects the translated backend and the files it loads.</summary>
internal static class BackendSelection
{
    /// <summary>Gets the selected backend: <c>metal</c> or <c>opengl</c>.</summary>
    public static string Backend { get; } = ResolveBackend();

    /// <summary>Gets the path of the native backend library.</summary>
    public static string NativeLibraryPath { get; } = ResolveNativeLibraryPath();

    /// <summary>Gets the root directory of translated shader artifacts, if configured.</summary>
    public static string? ArtifactRoot { get; } = Environment.GetEnvironmentVariable("CROSSTL_COMPUTESHARP_ARTIFACTS");

    /// <summary>Gets the dispatch evidence log path, if configured.</summary>
    public static string? EvidencePath { get; } = Environment.GetEnvironmentVariable("CROSSTL_COMPUTESHARP_EVIDENCE");

    /// <summary>Gets the negative-control fault to inject, if any.</summary>
    public static string? Fault { get; } = NormalizeFault(Environment.GetEnvironmentVariable("CROSSTL_COMPUTESHARP_FAULT"));

    /// <summary>Gets whether the backend requires all native calls on one thread.</summary>
    public static bool RequiresDedicatedThread => Backend == "opengl";

    private static string ResolveBackend()
    {
        string? configured = Environment.GetEnvironmentVariable("CROSSTL_COMPUTESHARP_BACKEND");
        if (!string.IsNullOrWhiteSpace(configured))
        {
            string normalized = configured.Trim().ToLowerInvariant();
            return normalized switch
            {
                "metal" => "metal",
                "opengl" or "gl" => "opengl",
                _ => throw new InvalidOperationException($"Unsupported CROSSTL_COMPUTESHARP_BACKEND value '{configured}'.")
            };
        }

        return OperatingSystem.IsMacOS() ? "metal" : "opengl";
    }

    private static string ResolveNativeLibraryPath()
    {
        string? configured = Environment.GetEnvironmentVariable("CROSSTL_COMPUTESHARP_NATIVE_LIBRARY");
        if (!string.IsNullOrWhiteSpace(configured))
        {
            return configured;
        }

        string fileName = Backend == "metal" ? "libcsbackend_metal.dylib" : "libcsbackend_gl.so";
        return Path.Combine(AppContext.BaseDirectory, fileName);
    }

    private static string? NormalizeFault(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        string fault = value.Trim().ToLowerInvariant();
        return fault switch
        {
            "omit-dispatch" or "swap-bindings" or "perturb-constants" or "corrupt-readback" => fault,
            _ => throw new InvalidOperationException($"Unsupported CROSSTL_COMPUTESHARP_FAULT value '{value}'.")
        };
    }
}

/// <summary>Raised when a native backend call fails.</summary>
internal sealed class NativeBackendException(string operation, int status, string message)
    : InvalidOperationException($"{BackendSelection.Backend} backend {operation} failed ({status}): {message}")
{
    public int Status { get; } = status;
}

/// <summary>
/// Runs native backend calls. OpenGL contexts are bound to one thread, so every call
/// on that backend is marshalled to a dedicated owner thread; Metal calls run inline.
/// </summary>
internal sealed class NativeExecutor : IDisposable
{
    private readonly BlockingCollection<WorkItem>? queue;
    private readonly Thread? thread;

    public NativeExecutor(bool dedicatedThread)
    {
        if (!dedicatedThread)
        {
            return;
        }

        this.queue = [];
        this.thread = new Thread(Run) { IsBackground = true, Name = "ComputeSharp portable backend" };
        this.thread.Start();
    }

    public void Invoke(Action action)
    {
        if (this.thread is null || Thread.CurrentThread == this.thread)
        {
            action();
            return;
        }

        using WorkItem item = new(action);
        this.queue!.Add(item);
        item.Completed.Wait();
        item.Exception?.Throw();
    }

    public T Invoke<T>(Func<T> function)
    {
        T result = default!;
        Action action = () => { result = function(); };
        Invoke(action);
        return result;
    }

    private void Run()
    {
        foreach (WorkItem item in this.queue!.GetConsumingEnumerable())
        {
            try
            {
                item.Action();
            }
            catch (Exception exception)
            {
                item.Exception = ExceptionDispatchInfo.Capture(exception);
            }
            finally
            {
                item.Completed.Set();
            }
        }
    }

    public void Dispose()
    {
        this.queue?.CompleteAdding();
    }

    private sealed class WorkItem(Action action) : IDisposable
    {
        public Action Action { get; } = action;

        public ManualResetEventSlim Completed { get; } = new(false);

        public ExceptionDispatchInfo? Exception { get; set; }

        public void Dispose() => Completed.Dispose();
    }
}

/// <summary>A native backend device and its resource operations.</summary>
internal sealed unsafe class NativeDevice : IDisposable
{
    private const int ErrorCapacity = 8192;

    private readonly NativeExecutor executor;
    private IntPtr device;

    private NativeDevice(NativeExecutor executor, IntPtr device, NativeDeviceInfo info)
    {
        this.executor = executor;
        this.device = device;
        Info = info;
        Name = ReadText(info.Name, 256);
        Api = ReadText(info.Api, 64);
        Driver = ReadText(info.Driver, 512);
    }

    public NativeDeviceInfo Info { get; }

    public string Name { get; }

    public string Api { get; }

    public string Driver { get; }

    public IntPtr Handle => this.device;

    public static NativeDevice Create()
    {
        CsBackend.EnsureResolver();
        NativeExecutor executor = new(BackendSelection.RequiresDedicatedThread);
        try
        {
            (IntPtr handle, NativeDeviceInfo info) = executor.Invoke(() =>
            {
                uint version = CsBackend.AbiVersionOf();
                if (version != CsBackend.AbiVersion)
                {
                    throw new InvalidOperationException($"Native backend ABI {version} does not match the runtime ABI {CsBackend.AbiVersion}.");
                }

                NativeDeviceInfo info = default;
                byte* error = stackalloc byte[ErrorCapacity];
                error[0] = 0;
                int status = CsBackend.DeviceCreate(out IntPtr handle, &info, error, ErrorCapacity);
                Check("device creation", status, error);
                return (handle, info);
            });
            return new NativeDevice(executor, handle, info);
        }
        catch
        {
            executor.Dispose();
            throw;
        }
    }

    public void Invoke(Action action) => this.executor.Invoke(action);

    public T Invoke<T>(Func<T> function) => this.executor.Invoke(function);

    public bool IsFormatSupported(uint dimension, uint format, uint usage)
    {
        IntPtr device = this.device;
        return Invoke(() => CsBackend.FormatSupported(device, dimension, format, usage) != 0);
    }

    public IntPtr CreateBuffer(ulong size)
    {
        IntPtr device = this.device;
        return Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            Check("buffer creation", CsBackend.BufferCreate(device, size, out IntPtr buffer, error, ErrorCapacity), error);
            return buffer;
        });
    }

    public void DestroyBuffer(IntPtr buffer)
    {
        IntPtr device = this.device;
        if (buffer != IntPtr.Zero && device != IntPtr.Zero)
        {
            Invoke(() => CsBackend.BufferDestroy(device, buffer));
        }
    }

    public void WriteBuffer(IntPtr buffer, ulong offset, IntPtr data, ulong size)
    {
        IntPtr device = this.device;
        Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            Check("buffer write", CsBackend.BufferWrite(device, buffer, offset, (void*)data, size, error, ErrorCapacity), error);
        });
    }

    public void ReadBuffer(IntPtr buffer, ulong offset, IntPtr data, ulong size)
    {
        IntPtr device = this.device;
        Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            Check("buffer read", CsBackend.BufferRead(device, buffer, offset, (void*)data, size, error, ErrorCapacity), error);
        });
        if (BackendSelection.Fault == "corrupt-readback" && size > 0)
        {
            ((byte*)data)[0] ^= 0x5A;
        }
    }

    public IntPtr CreateTexture(uint dimension, uint format, uint width, uint height, uint depth, uint usage)
    {
        IntPtr device = this.device;
        return Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            Check("texture creation", CsBackend.TextureCreate(device, dimension, format, width, height, depth, usage, out IntPtr texture, error, ErrorCapacity), error);
            return texture;
        });
    }

    public void DestroyTexture(IntPtr texture)
    {
        IntPtr device = this.device;
        if (texture != IntPtr.Zero && device != IntPtr.Zero)
        {
            Invoke(() => CsBackend.TextureDestroy(device, texture));
        }
    }

    public void WriteTexture(IntPtr texture, uint x, uint y, uint z, uint width, uint height, uint depth, IntPtr data, ulong rowPitch, ulong slicePitch)
    {
        IntPtr device = this.device;
        Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            Check("texture write", CsBackend.TextureWrite(device, texture, x, y, z, width, height, depth, (void*)data, rowPitch, slicePitch, error, ErrorCapacity), error);
        });
    }

    public void ReadTexture(IntPtr texture, uint x, uint y, uint z, uint width, uint height, uint depth, IntPtr data, ulong rowPitch, ulong slicePitch)
    {
        IntPtr device = this.device;
        Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            Check("texture read", CsBackend.TextureRead(device, texture, x, y, z, width, height, depth, (void*)data, rowPitch, slicePitch, error, ErrorCapacity), error);
        });
        if (BackendSelection.Fault == "corrupt-readback" && width > 0 && height > 0 && depth > 0)
        {
            ((byte*)data)[0] ^= 0x5A;
        }
    }

    public IntPtr CreateSampler(uint filter, uint addressMode)
    {
        IntPtr device = this.device;
        return Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            Check("sampler creation", CsBackend.SamplerCreate(device, filter, addressMode, out IntPtr sampler, error, ErrorCapacity), error);
            return sampler;
        });
    }

    public void DestroySampler(IntPtr sampler)
    {
        IntPtr device = this.device;
        if (sampler != IntPtr.Zero && device != IntPtr.Zero)
        {
            Invoke(() => CsBackend.SamplerDestroy(device, sampler));
        }
    }

    public IntPtr CreatePipeline(string source, string entryPoint, int threadsX, int threadsY, int threadsZ, out string log)
    {
        IntPtr device = this.device;
        string compilerLog = string.Empty;
        IntPtr pipeline = Invoke(() =>
        {
            byte[] sourceBytes = Utf8z(source);
            byte[] entryBytes = Utf8z(entryPoint);
            byte* buffer = stackalloc byte[ErrorCapacity];
            buffer[0] = 0;
            fixed (byte* sourcePointer = sourceBytes)
            fixed (byte* entryPointer = entryBytes)
            {
                int status = CsBackend.PipelineCreate(device, sourcePointer, entryPointer, (uint)threadsX, (uint)threadsY, (uint)threadsZ, out IntPtr handle, buffer, ErrorCapacity);
                compilerLog = Marshal.PtrToStringUTF8((IntPtr)buffer) ?? string.Empty;
                if (status != 0)
                {
                    throw new NativeBackendException("pipeline creation", status, compilerLog);
                }

                return handle;
            }
        });
        log = compilerLog;
        return pipeline;
    }

    public void DestroyPipeline(IntPtr pipeline)
    {
        IntPtr device = this.device;
        if (pipeline != IntPtr.Zero && device != IntPtr.Zero)
        {
            Invoke(() => CsBackend.PipelineDestroy(device, pipeline));
        }
    }

    public IntPtr CreateList()
    {
        IntPtr device = this.device;
        return Invoke(() =>
        {
            byte* error = stackalloc byte[ErrorCapacity];
            error[0] = 0;
            Check("command list creation", CsBackend.ListCreate(device, out IntPtr list, error, ErrorCapacity), error);
            return list;
        });
    }

    public void Dispose()
    {
        IntPtr device = Interlocked.Exchange(ref this.device, IntPtr.Zero);
        if (device != IntPtr.Zero)
        {
            this.executor.Invoke(() => CsBackend.DeviceDestroy(device));
        }

        this.executor.Dispose();
    }

    internal static void Check(string operation, int status, byte* error)
    {
        if (status != 0)
        {
            throw new NativeBackendException(operation, status, Marshal.PtrToStringUTF8((IntPtr)error) ?? string.Empty);
        }
    }

    private static string ReadText(byte* text, int capacity)
    {
        int length = 0;
        while (length < capacity && text[length] != 0)
        {
            length++;
        }

        return Encoding.UTF8.GetString(text, length);
    }

    private static byte[] Utf8z(string text)
    {
        int count = Encoding.UTF8.GetByteCount(text);
        byte[] bytes = new byte[count + 1];
        _ = Encoding.UTF8.GetBytes(text, 0, text.Length, bytes, 0);
        return bytes;
    }
}
