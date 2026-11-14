using System.Collections;
using System.Globalization;
using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

internal static class Program
{
    private const string TestNamespace = "Microsoft.VisualStudio.TestTools.UnitTesting.";
    private const BindingFlags StaticProperties = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true
    };

    public static int Main(string[] args)
    {
        // A host-side test project (for example the source generator tests) compiles no shader
        // into its assembly; only such a project may be exported without HLSL descriptors.
        bool hostOnly = args.Length == 3 && args[2] == "--host-only";
        if (args.Length != 2 && !hostOnly)
        {
            Console.Error.WriteLine("Usage: Discovery <test-assembly.dll> <new-output-directory> [--host-only]");
            return 2;
        }

        string output = Path.GetFullPath(args[1]);
        string temporary = output + ".partial-" + Guid.NewGuid().ToString("N");
        try
        {
            if (Path.Exists(output))
                throw new InvalidOperationException("Output already exists: " + output);
            string summary = Export(Path.GetFullPath(args[0]), temporary, hostOnly);
            Directory.Move(temporary, output);
            Console.WriteLine(summary);
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            if (Directory.Exists(temporary))
                Directory.Delete(temporary, recursive: true);
            return 1;
        }
    }

    // Writes the discovery report and the generated HLSL of one test assembly into a new
    // directory and returns a one-line summary of the export.
    private static string Export(string assemblyPath, string temporary, bool hostOnly)
    {
        string assemblyHash = Hash(File.ReadAllBytes(assemblyPath));
        AssemblyDependencyResolver resolver = new(assemblyPath);
        AssemblyLoadContext.Default.Resolving += (_, name) =>
        {
            string? dependency = resolver.ResolveAssemblyToPath(name);
            return dependency is null ? null : AssemblyLoadContext.Default.LoadFromAssemblyPath(dependency);
        };
        Assembly assembly = AssemblyLoadContext.Default.LoadFromAssemblyPath(assemblyPath);
        using SourceLocator locator = new(assemblyPath);
        Type[] types = assembly.GetTypes().OrderBy(TypeName, StringComparer.Ordinal).ToArray();
        CaseCollector collector = new(locator);
        foreach (Type type in types.Where(t => HasAttribute(t, TestNamespace + "TestClassAttribute")))
        {
            foreach (MethodInfo method in type.GetMethods(BindingFlags.Public | BindingFlags.NonPublic |
                         BindingFlags.Instance | BindingFlags.Static).OrderBy(m => m.Name, StringComparer.Ordinal))
                collector.AddTestMethod(type, method);
        }
        // A test project that is a program has no MSTest methods: its one case is the
        // entry point, which throws when the program's own check fails.
        string projectKind = "mstest";
        if (collector.MethodCount == 0 && assembly.EntryPoint is MethodInfo entryPoint)
        {
            projectKind = "program";
            collector.AddEntryPoint(entryPoint);
        }
        if (collector.Cases.Count == 0)
            throw new InvalidOperationException("No test cases discovered");

        List<object> shaders = ExportShaders(types, temporary);
        if (shaders.Count == 0 && !hostOnly)
            throw new InvalidOperationException("No generated HLSL descriptors discovered");
        if (shaders.Count != 0 && hostOnly)
            throw new InvalidOperationException("A host-only test project contains generated HLSL descriptors");
        if (Hash(File.ReadAllBytes(assemblyPath)) != assemblyHash)
            throw new InvalidOperationException("Test assembly changed during export");
        object report = new
        {
            schemaVersion = 1,
            kind = "computesharp.discovery",
            assembly = new { name = assembly.GetName().Name, sha256 = assemblyHash },
            projectKind,
            testMethodCount = collector.MethodCount,
            caseCount = collector.Cases.Count,
            shaderCount = shaders.Count,
            coverageFrozen = false,
            nativeExecutionQualified = false,
            methods = collector.Methods,
            tests = collector.Cases,
            shaders
        };
        File.WriteAllText(Path.Combine(temporary, "discovery.json"), Serialize(report, hostOnly) + "\n", new UTF8Encoding(false));
        return $"Exported {collector.Cases.Count} cases, {collector.MethodCount} methods and {shaders.Count} HLSL descriptors";
    }

    // Writes the HLSL of every generated compute shader descriptor and returns its metadata.
    private static List<object> ExportShaders(Type[] types, string temporary)
    {
        Directory.CreateDirectory(Path.Combine(temporary, "hlsl"));
        List<object> shaders = [];
        foreach (Type type in types.Where(t => t.GetInterfaces().Any(i => i.IsGenericType &&
                     i.GetGenericTypeDefinition().FullName == "ComputeSharp.Descriptors.IComputeShaderDescriptor`1")))
        {
            string source = (string)Property(type, "HlslSource");
            if (string.IsNullOrWhiteSpace(source))
                throw new InvalidOperationException("Empty HLSL source: " + TypeName(type));
            string id = Hash(Encoding.UTF8.GetBytes(TypeName(type)));
            string path = "hlsl/" + id + ".hlsl";
            byte[] bytes = Encoding.UTF8.GetBytes(source);
            File.WriteAllBytes(Path.Combine(temporary, "hlsl", id + ".hlsl"), bytes);
            object ranges = Property(type, "ResourceDescriptorRanges");
            object rangeArray = ranges.GetType().GetMethod("ToArray", Type.EmptyTypes)!.Invoke(ranges, null)!;
            shaders.Add(new
            {
                id,
                type = TypeName(type),
                path,
                sha256 = Hash(bytes),
                threads = new[] { (int)Property(type, "ThreadsX"), (int)Property(type, "ThreadsY"), (int)Property(type, "ThreadsZ") },
                constantBufferSize = (int)Property(type, "ConstantBufferSize"),
                staticSamplerRequired = (bool)Property(type, "IsStaticSamplerRequired"),
                resourceDescriptorRanges = JsonSerializer.SerializeToNode(rangeArray, rangeArray.GetType(), JsonOptions)
            });
        }
        return shaders;
    }

    // Collects the test methods and the executed cases (one per data row) of a test assembly.
    private sealed class CaseCollector(SourceLocator locator)
    {
        private readonly HashSet<string> identities = [];
        private readonly Dictionary<string, int> occurrences = [];

        public List<object> Cases { get; } = [];

        public List<object> Methods { get; } = [];

        public int MethodCount { get; private set; }

        public void AddTestMethod(Type type, MethodInfo method)
        {
            object[] attributes = method.GetCustomAttributes(inherit: true).ToArray();
            if (!attributes.Any(a => DerivesFrom(a.GetType(), TestNamespace + "TestMethodAttribute")))
                return;
            string methodName = TypeName(type) + "." + method.Name;
            string signature = AddMethod(methodName, method);
            bool ignored = HasAttribute(type, TestNamespace + "IgnoreAttribute") || HasAttribute(method, TestNamespace + "IgnoreAttribute");
            object[] sources = attributes.Where(a => DataInterface(a.GetType()) is not null).ToArray();
            if (sources.Length == 0)
            {
                if (method.GetParameters().Length != 0)
                    throw new InvalidOperationException("Missing test data: " + signature);
                AddCase(methodName, signature, [], method.Name, ignored);
            }
            foreach (object source in sources)
            {
                Type dataInterface = DataInterface(source.GetType())!;
                object? data = dataInterface.GetMethod("GetData")!.Invoke(source, [method]);
                if (data is not IEnumerable rows)
                    throw new InvalidOperationException("Invalid data source: " + signature);
                int rowCount = 0;
                foreach (object? row in rows)
                {
                    if (row is not object?[] arguments || arguments.Length != method.GetParameters().Length)
                        throw new InvalidOperationException("Test data arity differs: " + signature);
                    string displayName = (string?)dataInterface.GetMethod("GetDisplayName")!.Invoke(source, [method, DisplayArguments(source, arguments)]) ?? method.Name;
                    AddCase(methodName, signature, arguments, displayName, ignored);
                    rowCount++;
                }
                if (rowCount == 0)
                    throw new InvalidOperationException("Empty data source: " + signature);
            }
        }

        public void AddEntryPoint(MethodInfo entryPoint)
        {
            Type owner = entryPoint.DeclaringType ?? throw new InvalidOperationException("Entry point has no declaring type");
            string methodName = TypeName(owner) + "." + entryPoint.Name;
            string signature = AddMethod(methodName, entryPoint);
            string id = Hash(Encoding.UTF8.GetBytes(signature + "\n[]\n1"));
            Cases.Add(new { id, method = methodName, signature, displayName = entryPoint.Name, occurrence = 1, arguments = Array.Empty<JsonObject>(), ignored = false });
        }

        private string AddMethod(string methodName, MethodInfo method)
        {
            MethodCount++;
            string signature = methodName + "(" + string.Join(",", method.GetParameters().Select(p => TypeName(p.ParameterType))) + ")";
            (string path, int line) = locator.Locate(method);
            Methods.Add(new { method = methodName, signature, source = new { path, line } });
            return signature;
        }

        private void AddCase(string methodName, string signature, object?[] arguments, string displayName, bool ignored)
        {
            JsonObject[] values = arguments.Select(Argument).ToArray();
            string row = signature + "\n" + JsonSerializer.Serialize(values);
            // MSTest executes repeated data rows separately under the same display name,
            // so each repetition is a distinct case identified by its ordinal.
            int occurrence = occurrences[row] = occurrences.GetValueOrDefault(row) + 1;
            string id = Hash(Encoding.UTF8.GetBytes(row + "\n" + occurrence.ToString(CultureInfo.InvariantCulture)));
            if (!identities.Add(id))
                throw new InvalidOperationException("Duplicate test case: " + signature);
            Cases.Add(new { id, method = methodName, signature, displayName, occurrence, arguments = values, ignored });
        }
    }

    private static bool HasAttribute(MemberInfo member, string name) =>
        member.GetCustomAttributes(inherit: true).Any(a => DerivesFrom(a.GetType(), name));

    // The report of a host-only project states it, so a lock frozen from a shader-bearing
    // project cannot be satisfied by an export that skipped the descriptor check.
    private static string Serialize(object report, bool hostOnly)
    {
        if (!hostOnly)
            return JsonSerializer.Serialize(report, JsonOptions);
        JsonObject node = JsonSerializer.SerializeToNode(report, JsonOptions)!.AsObject();
        node["hostOnly"] = true;
        return node.ToJsonString(JsonOptions);
    }

    private static bool DerivesFrom(Type? type, string name) =>
        type is not null && (type.FullName == name || DerivesFrom(type.BaseType, name));

    private static Type? DataInterface(Type type) =>
        type.GetInterfaces().SingleOrDefault(i => i.FullName == TestNamespace + "ITestDataSource");

    // The MSTest adapter names a DataRow case without an explicit display name with its string
    // arguments in quotes, which DataRowAttribute.GetDisplayName itself omits. Other data sources
    // name their cases as GetDisplayName does.
    private static object?[] DisplayArguments(object source, object?[] arguments)
    {
        if (source.GetType().FullName != TestNamespace + "DataRowAttribute" ||
            source.GetType().GetProperty("DisplayName")?.GetValue(source) is not null)
            return arguments;
        return arguments.Select(a => a is string text ? (object)("\"" + text + "\"") : a).ToArray();
    }

    private static object Property(Type type, string name) =>
        type.GetProperties(StaticProperties).Single(p => p.Name == name || p.Name.EndsWith("." + name, StringComparison.Ordinal)).GetValue(null)
        ?? throw new InvalidOperationException("Missing descriptor property: " + name);

    private static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    private static string TypeName(Type type) => type.IsArray ? TypeName(type.GetElementType()!) + "[" + new string(',', type.GetArrayRank() - 1) + "]" :
        type.IsGenericType ? type.GetGenericTypeDefinition().FullName! + "<" + string.Join(",", type.GetGenericArguments().Select(TypeName)) + ">" : type.FullName ?? type.Name;

    private static JsonObject Argument(object? value)
    {
        if (value is null)
            return new() { ["kind"] = "null" };
        if (value is Type type)
            return new() { ["kind"] = "type", ["name"] = TypeName(type) };
        if (value is Enum enumeration)
            return new() { ["kind"] = "enum", ["type"] = TypeName(value.GetType()), ["name"] = enumeration.ToString(), ["value"] = enumeration.ToString("D") };
        if (value is float single)
            return new() { ["kind"] = "System.Single", ["bits"] = BitConverter.SingleToUInt32Bits(single).ToString("x8", CultureInfo.InvariantCulture) };
        if (value is double number)
            return new() { ["kind"] = "System.Double", ["bits"] = BitConverter.DoubleToUInt64Bits(number).ToString("x16", CultureInfo.InvariantCulture) };
        if (value is string text)
            return new() { ["kind"] = "System.String", ["value"] = text };
        if (value is bool flag)
            return new() { ["kind"] = "System.Boolean", ["value"] = flag };
        if (value is char character)
            return new() { ["kind"] = "System.Char", ["value"] = (int)character };
        if (value is sbyte or byte or short or ushort or int or uint or long or ulong)
            return new() { ["kind"] = TypeName(value.GetType()), ["value"] = ((IFormattable)value).ToString(null, CultureInfo.InvariantCulture) };
        if (value is Array array && array.Rank == 1)
            return new() { ["kind"] = "array", ["type"] = TypeName(array.GetType()), ["items"] = new JsonArray(array.Cast<object?>().Select(v => (JsonNode)Argument(v)).ToArray()) };
        throw new InvalidOperationException("Unsupported test argument type: " + TypeName(value.GetType()));
    }
}

/// <summary>Resolves test methods to repository-relative source locations through the portable PDB.</summary>
internal sealed class SourceLocator : IDisposable
{
    private static readonly System.Text.RegularExpressions.Regex MappedRoot = new(@"^/_\d*/", System.Text.RegularExpressions.RegexOptions.CultureInvariant);
    private readonly PEReader peReader;
    private readonly MetadataReaderProvider provider;
    private readonly MetadataReader pdb;

    public SourceLocator(string assemblyPath)
    {
        peReader = new PEReader(File.OpenRead(assemblyPath));
        DebugDirectoryEntry embedded = peReader.ReadDebugDirectory().FirstOrDefault(e => e.Type == DebugDirectoryEntryType.EmbeddedPortablePdb);
        if (embedded.DataSize != 0)
        {
            provider = peReader.ReadEmbeddedPortablePdbDebugDirectoryData(embedded);
        }
        else
        {
            string pdbPath = Path.ChangeExtension(assemblyPath, ".pdb");
            if (!File.Exists(pdbPath))
                throw new InvalidOperationException("No portable PDB for source locations: " + assemblyPath);
            provider = MetadataReaderProvider.FromPortablePdbStream(File.OpenRead(pdbPath));
        }
        pdb = provider.GetMetadataReader();
    }

    public (string Path, int Line) Locate(MethodInfo method)
    {
        MethodBase body = method;
        if (method.GetCustomAttribute<System.Runtime.CompilerServices.AsyncStateMachineAttribute>() is { } async)
            body = async.StateMachineType.GetMethod("MoveNext", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)!;
        MethodDefinitionHandle handle = MetadataTokens.MethodDefinitionHandle(body.MetadataToken);
        MethodDebugInformation information = pdb.GetMethodDebugInformation(handle.ToDebugInformationHandle());
        foreach (SequencePoint point in information.GetSequencePoints())
        {
            if (point.IsHidden)
                continue;
            string document = pdb.GetString(pdb.GetDocument(point.Document).Name).Replace('\\', '/');
            return (Relative(document), point.StartLine);
        }
        throw new InvalidOperationException("No source location for test method: " + method.DeclaringType?.FullName + "." + method.Name);
    }

    private static string Relative(string document)
    {
        System.Text.RegularExpressions.Match mapped = MappedRoot.Match(document);
        if (mapped.Success)
            return document[mapped.Length..];
        for (DirectoryInfo? directory = new FileInfo(document).Directory; directory is not null; directory = directory.Parent)
        {
            if (Path.Exists(Path.Combine(directory.FullName, ".git")))
                return Path.GetRelativePath(directory.FullName, document).Replace('\\', '/');
        }
        throw new InvalidOperationException("Source document is outside a repository: " + document);
    }

    public void Dispose()
    {
        provider.Dispose();
        peReader.Dispose();
    }
}
