using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading;
using ComputeSharp.Portable.Native;

namespace ComputeSharp.Portable.Diagnostics;

/// <summary>
/// Writes JSON Lines evidence of device creation, capability queries, resource allocation
/// and release, pipeline compilation, dispatches and readbacks when
/// <c>CROSSTL_COMPUTESHARP_EVIDENCE</c> is set. Records carry the test case that issued them
/// so results can be traced to the frozen case inventory.
/// </summary>
internal static class EvidenceLog
{
    private static readonly object Gate = new();
    private static readonly StreamWriter? Writer = Open();
    private static readonly ConcurrentDictionary<MethodBase, string?> TestMethods = new();

    // A test project that is a program (no MSTest methods) is one case: its entry point.
    // The host names the program assembly in CROSSTL_COMPUTESHARP_PROGRAM, so work issued
    // by a test host's own entry point is never attributed to a case.
    private static readonly MethodInfo? ProgramEntryPoint = ResolveProgramEntryPoint();
    private static long sequence;
    private static long resourceIds;

    public static bool Enabled => Writer is not null;

    /// <summary>Allocates a process-unique resource identifier.</summary>
    public static long NextResourceId() => Interlocked.Increment(ref resourceIds);

    /// <summary>Allocates a process-unique operation sequence number.</summary>
    public static long NextSequence() => Interlocked.Increment(ref sequence);

    /// <summary>Gets the test case currently executing, if the host published one.</summary>
    public static string? CurrentCase()
    {
        if (AppContext.GetData("ComputeSharp.Portable.CurrentCase") is string published && published.Length > 0)
        {
            return published;
        }

        return CurrentTestMethod();
    }

    /// <summary>Writes one record; <paramref name="attribute"/> selects whether it names the current test case.</summary>
    public static void Write(string kind, Action<Utf8JsonWriter> body, bool attribute = true)
    {
        if (Writer is null)
        {
            return;
        }

        using MemoryStream stream = new();
        using (Utf8JsonWriter json = new(stream))
        {
            json.WriteStartObject();
            json.WriteString("record", kind);
            json.WriteNumber("time", Stopwatch.GetTimestamp() * 1.0 / Stopwatch.Frequency);
            json.WriteString("backend", BackendSelection.Backend);
            if (BackendSelection.Fault is string fault)
            {
                json.WriteString("fault", fault);
            }

            string? testCase = attribute ? CurrentCase() : null;
            if (testCase is not null)
            {
                json.WriteString("case", testCase);
            }

            body(json);
            json.WriteEndObject();
        }

        string line = Encoding.UTF8.GetString(stream.ToArray());
        lock (Gate)
        {
            Writer.WriteLine(line);
            Writer.Flush();
        }
    }

    /// <summary>
    /// Records the allocation of a resource. <paramref name="memory"/> is <c>device</c> for
    /// allocations made through the native backend and <c>host</c> for transfer resources,
    /// which the portable runtime keeps in host memory.
    /// </summary>
    public static void Resource(long resourceId, Type type, string memory, long bytes, Action<Utf8JsonWriter> shape)
    {
        if (Writer is null)
        {
            return;
        }

        Write("resource", json =>
        {
            json.WriteNumber("resourceId", resourceId);
            json.WriteString("type", TypeName(type));
            json.WriteString("memory", memory);
            json.WriteNumber("bytes", bytes);
            shape(json);
        });
    }

    /// <summary>
    /// Records that the memory of a resource was released. Finalizers can release resources
    /// while another case runs, so the record names no case.
    /// </summary>
    public static void Release(long resourceId)
    {
        if (Writer is null)
        {
            return;
        }

        Write("release", json => json.WriteNumber("resourceId", resourceId), attribute: false);
    }

    /// <summary>Records the answer of a device capability query made through the public API.</summary>
    public static void Capability(string capability, bool available, Action<Utf8JsonWriter>? details = null)
    {
        if (Writer is null)
        {
            return;
        }

        Write("capability", json =>
        {
            json.WriteString("capability", capability);
            json.WriteBoolean("available", available);
            details?.Invoke(json);
        });
    }

    /// <summary>Formats a type as its namespace-qualified name with C#-style type arguments.</summary>
    public static string TypeName(Type type)
    {
        if (!type.IsGenericType)
        {
            return type.FullName ?? type.Name;
        }

        string name = type.GetGenericTypeDefinition().FullName ?? type.Name;
        int tick = name.IndexOf('`');
        if (tick >= 0)
        {
            name = name[..tick];
        }

        StringBuilder builder = new(name);
        builder.Append('<');
        Type[] arguments = type.GetGenericArguments();
        for (int i = 0; i < arguments.Length; i++)
        {
            if (i > 0)
            {
                builder.Append(", ");
            }

            builder.Append(TypeName(arguments[i]));
        }

        return builder.Append('>').ToString();
    }

    public static string Hex(ReadOnlySpan<byte> data)
    {
        return Convert.ToHexString(data).ToLowerInvariant();
    }

    public static string Sha256(ReadOnlySpan<byte> data)
    {
        return Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant();
    }

    public static unsafe void Readback(string resource, long resourceId, IntPtr data, long bytes)
    {
        if (Writer is null)
        {
            return;
        }

        ReadOnlySpan<byte> span = new((void*)data, checked((int)bytes));
        string hash = Sha256(span);
        Write("readback", json =>
        {
            json.WriteString("resource", resource);
            json.WriteNumber("resourceId", resourceId);
            json.WriteNumber("bytes", bytes);
            json.WriteString("sha256", hash);
        });
    }

    /// <summary>Records a texture readback with its region and, when selected, captures its pixels.</summary>
    public static unsafe void TextureReadback(string resource, long resourceId, IntPtr data, long bytes, uint format, int width, int height, int depth, long rowPitch)
    {
        if (Writer is null)
        {
            return;
        }

        ReadOnlySpan<byte> span = new((void*)data, checked((int)bytes));
        string hash = Sha256(span);
        string? capture = ReadbackCapture.TryWrite(CurrentCase(), hash, span, format, width, height, depth, rowPitch);
        Write("readback", json =>
        {
            json.WriteString("resource", resource);
            json.WriteNumber("resourceId", resourceId);
            json.WriteNumber("bytes", bytes);
            json.WriteString("sha256", hash);
            json.WriteNumber("format", format);
            json.WriteNumber("width", width);
            json.WriteNumber("height", height);
            json.WriteNumber("depth", depth);
            json.WriteNumber("rowPitch", rowPitch);
            if (capture is not null)
            {
                json.WriteString("capture", capture);
            }
        });
    }

    private static StreamWriter? Open()
    {
        string? path = BackendSelection.EvidencePath;
        if (string.IsNullOrWhiteSpace(path))
        {
            return null;
        }

        string? directory = Path.GetDirectoryName(Path.GetFullPath(path));
        if (directory is not null)
        {
            Directory.CreateDirectory(directory);
        }

        FileStream stream = new(path, FileMode.Append, FileAccess.Write, FileShare.ReadWrite);
        return new StreamWriter(stream, new UTF8Encoding(false));
    }

    private static MethodInfo? ResolveProgramEntryPoint()
    {
        string? program = Environment.GetEnvironmentVariable("CROSSTL_COMPUTESHARP_PROGRAM");
        Assembly? entry = Assembly.GetEntryAssembly();
        if (string.IsNullOrEmpty(program) || entry is null || entry.GetName().Name != program)
        {
            return null;
        }

        return entry.EntryPoint;
    }

    private static string? CurrentTestMethod()
    {
        StackTrace trace = new(false);
        for (int i = 0; i < trace.FrameCount; i++)
        {
            MethodBase? method = trace.GetFrame(i)?.GetMethod();
            if (method is null)
            {
                continue;
            }

            string? name = TestMethods.GetOrAdd(method, static method =>
            {
                foreach (object attribute in method.GetCustomAttributes(inherit: true))
                {
                    for (Type? type = attribute.GetType(); type is not null; type = type.BaseType)
                    {
                        if (type.Name == "TestMethodAttribute")
                        {
                            return $"{method.DeclaringType?.FullName}.{method.Name}";
                        }
                    }
                }

                if (ProgramEntryPoint is { } entry &&
                    method.MetadataToken == entry.MetadataToken &&
                    method.Module == entry.Module)
                {
                    return $"{method.DeclaringType?.FullName}.{method.Name}";
                }

                return null;
            });
            if (name is not null)
            {
                return name;
            }
        }

        return null;
    }
}
