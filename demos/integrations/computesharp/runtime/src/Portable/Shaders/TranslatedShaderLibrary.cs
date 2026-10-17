using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using ComputeSharp.Portable.Native;

namespace ComputeSharp.Portable.Shaders;

/// <summary>A target binding reported by CrossTL for one translated resource.</summary>
internal sealed record TranslatedBinding(
    string? Name,
    string Kind,
    string Namespace,
    int Binding,
    int Count,
    string? Register,
    string? Of,
    string? Type,
    string? Format);

/// <summary>One CrossTL artifact for a generated HLSL source.</summary>
internal sealed class TranslatedShader
{
    private string? source;

    public required string HlslSha256 { get; init; }

    public required string Id { get; init; }

    public required string Type { get; init; }

    public required string Status { get; init; }

    public string? Error { get; init; }

    public string? ArtifactPath { get; init; }

    public string? ArtifactSha256 { get; init; }

    public string? EntryPoint { get; init; }

    public IReadOnlyList<TranslatedBinding> Bindings { get; init; } = [];

    /// <summary>Gets the artifact text after verifying its recorded hash.</summary>
    public string LoadSource()
    {
        if (this.source is not null)
        {
            return this.source;
        }

        if (Status != "translated" || ArtifactPath is null)
        {
            throw new InvalidOperationException(
                $"CrossTL did not produce a {BackendSelection.Backend} artifact for shader {Type} ({HlslSha256}): {Status} {Error}");
        }

        byte[] bytes = File.ReadAllBytes(ArtifactPath);
        string hash = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        if (hash != ArtifactSha256)
        {
            throw new InvalidOperationException($"Artifact {ArtifactPath} hash {hash} does not match the manifest hash {ArtifactSha256}.");
        }

        return this.source = Encoding.UTF8.GetString(bytes);
    }
}

/// <summary>The translated shader manifest for the selected backend.</summary>
internal sealed class TranslatedShaderLibrary
{
    private readonly Dictionary<string, TranslatedShader> shaders;

    private TranslatedShaderLibrary(string manifestPath, string target, string translatorCommit, Dictionary<string, TranslatedShader> shaders)
    {
        ManifestPath = manifestPath;
        Target = target;
        TranslatorCommit = translatorCommit;
        this.shaders = shaders;
    }

    public string ManifestPath { get; }

    public string Target { get; }

    public string TranslatorCommit { get; }

    public int Count => this.shaders.Count;

    public static TranslatedShaderLibrary Load()
    {
        string root = BackendSelection.ArtifactRoot
            ?? throw new InvalidOperationException("CROSSTL_COMPUTESHARP_ARTIFACTS must point to the CrossTL artifact directory.");
        string target = BackendSelection.Backend;
        string manifestPath = Path.Combine(root, target, "manifest.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(manifestPath));
        JsonElement manifest = document.RootElement;
        if (manifest.GetProperty("target").GetString() != target)
        {
            throw new InvalidOperationException($"Manifest {manifestPath} is not a {target} manifest.");
        }

        string commit = manifest.TryGetProperty("translator", out JsonElement translator) &&
            translator.TryGetProperty("commit", out JsonElement commitElement) &&
            commitElement.ValueKind == JsonValueKind.String
            ? commitElement.GetString()!
            : "unknown";

        Dictionary<string, TranslatedShader> shaders = new(StringComparer.Ordinal);
        foreach (JsonProperty property in manifest.GetProperty("shaders").EnumerateObject())
        {
            JsonElement entry = property.Value;
            List<TranslatedBinding> bindings = [];
            if (entry.TryGetProperty("bindings", out JsonElement bindingArray))
            {
                foreach (JsonElement binding in bindingArray.EnumerateArray())
                {
                    bindings.Add(new TranslatedBinding(
                        Name: OptionalString(binding, "name"),
                        Kind: OptionalString(binding, "kind") ?? "unknown",
                        Namespace: OptionalString(binding, "namespace") ?? "unknown",
                        Binding: binding.GetProperty("binding").GetInt32(),
                        Count: binding.TryGetProperty("count", out JsonElement count) ? count.GetInt32() : 1,
                        Register: OptionalString(binding, "register"),
                        Of: OptionalString(binding, "of"),
                        Type: OptionalString(binding, "type"),
                        Format: OptionalString(binding, "format")));
                }
            }

            string? artifact = OptionalString(entry, "artifact");
            shaders[property.Name] = new TranslatedShader
            {
                HlslSha256 = property.Name,
                Id = OptionalString(entry, "id") ?? property.Name,
                Type = OptionalString(entry, "type") ?? "unknown",
                Status = OptionalString(entry, "status") ?? "unknown",
                Error = OptionalString(entry, "error") ?? OptionalValidationLog(entry),
                ArtifactPath = artifact is null ? null : Path.Combine(root, artifact),
                ArtifactSha256 = OptionalString(entry, "artifactSha256"),
                EntryPoint = OptionalString(entry, "entryPoint"),
                Bindings = bindings,
            };
        }

        return new TranslatedShaderLibrary(manifestPath, target, commit, shaders);
    }

    public TranslatedShader Find(string hlslSha256, string typeName)
    {
        if (this.shaders.TryGetValue(hlslSha256, out TranslatedShader? shader))
        {
            return shader;
        }

        throw new InvalidOperationException(
            $"No CrossTL {Target} artifact exists for shader {typeName} with HLSL hash {hlslSha256} in {ManifestPath}.");
    }

    public static string HashHlsl(string hlslSource)
    {
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(hlslSource))).ToLowerInvariant();
    }

    private static string? OptionalString(JsonElement element, string name)
    {
        return element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
    }

    private static string? OptionalValidationLog(JsonElement entry)
    {
        return entry.TryGetProperty("validation", out JsonElement validation) ? OptionalString(validation, "log") : null;
    }
}
