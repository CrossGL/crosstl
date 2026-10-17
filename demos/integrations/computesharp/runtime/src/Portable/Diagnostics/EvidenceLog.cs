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
/// Writes JSON Lines evidence of device creation, pipeline compilation, dispatches and
/// readbacks when <c>CROSSTL_COMPUTESHARP_EVIDENCE</c> is set. Records carry the test
/// case that issued them so results can be traced to the frozen case inventory.
/// </summary>
internal static class EvidenceLog
{
    private static readonly object Gate = new();
    private static readonly StreamWriter? Writer = Open();
    private static readonly ConcurrentDictionary<MethodBase, string?> TestMethods = new();
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

    public static void Write(string kind, Action<Utf8JsonWriter> body)
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

            string? testCase = CurrentCase();
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
