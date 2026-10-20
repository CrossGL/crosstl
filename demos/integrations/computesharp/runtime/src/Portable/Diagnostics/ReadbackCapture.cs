using System;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using ImageSharpBgra32 = SixLabors.ImageSharp.PixelFormats.Bgra32;
using ImageSharpRgba32 = SixLabors.ImageSharp.PixelFormats.Rgba32;

namespace ComputeSharp.Portable.Diagnostics;

/// <summary>
/// Saves texture readbacks of selected test cases as lossless PNG files so their
/// pixels can be compared outside the test process. <c>CROSSTL_COMPUTESHARP_CAPTURE</c>
/// names the output directory and <c>CROSSTL_COMPUTESHARP_CAPTURE_CASES</c> lists
/// case-name prefixes separated by semicolons. Files are named by the SHA-256 of
/// the readback bytes recorded in the evidence, so identical outputs are stored once.
/// Only two-dimensional 8-bit RGBA and BGRA readbacks are captured.
/// </summary>
internal static class ReadbackCapture
{
    // DXGI_FORMAT_R8G8B8A8_UNORM and DXGI_FORMAT_B8G8R8A8_UNORM.
    private const uint Rgba8 = 28;
    private const uint Bgra8 = 87;

    private static readonly string? OutputDirectory = ResolveDirectory();
    private static readonly string[] CasePrefixes = (Environment.GetEnvironmentVariable("CROSSTL_COMPUTESHARP_CAPTURE_CASES") ?? string.Empty)
        .Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    /// <summary>Writes the readback if it is selected and returns the captured file name.</summary>
    public static string? TryWrite(string? testCase, string sha256, ReadOnlySpan<byte> data, uint format, int width, int height, int depth, long rowPitch)
    {
        if (OutputDirectory is null || testCase is null || depth != 1 || width < 1 || height < 1 || (format != Rgba8 && format != Bgra8))
        {
            return null;
        }

        bool selected = false;
        foreach (string prefix in CasePrefixes)
        {
            selected |= testCase.StartsWith(prefix, StringComparison.Ordinal);
        }

        if (!selected)
        {
            return null;
        }

        int rowBytes = checked(width * 4);
        if (rowPitch < rowBytes || data.Length < checked((rowPitch * (height - 1)) + rowBytes))
        {
            throw new InvalidOperationException($"Readback of {width}x{height} pixels does not fit {data.Length} bytes with row pitch {rowPitch}.");
        }

        string name = sha256 + ".png";
        string path = Path.Combine(OutputDirectory, name);
        if (File.Exists(path))
        {
            return name;
        }

        byte[] pixels = new byte[rowBytes * height];
        for (int y = 0; y < height; y++)
        {
            data.Slice(checked((int)(rowPitch * y)), rowBytes).CopyTo(pixels.AsSpan(rowBytes * y, rowBytes));
        }

        string temporary = Path.Combine(OutputDirectory, $"{name}.{Environment.ProcessId}.{Environment.CurrentManagedThreadId}.tmp");
        PngEncoder encoder = new() { ColorType = PngColorType.RgbWithAlpha, BitDepth = PngBitDepth.Bit8 };
        if (format == Rgba8)
        {
            using Image<ImageSharpRgba32> image = Image.LoadPixelData<ImageSharpRgba32>(pixels, width, height);
            image.SaveAsPng(temporary, encoder);
        }
        else
        {
            using Image<ImageSharpBgra32> image = Image.LoadPixelData<ImageSharpBgra32>(pixels, width, height);
            image.SaveAsPng(temporary, encoder);
        }

        File.Move(temporary, path, overwrite: true);
        return name;
    }

    private static string? ResolveDirectory()
    {
        string? directory = Environment.GetEnvironmentVariable("CROSSTL_COMPUTESHARP_CAPTURE");
        if (string.IsNullOrWhiteSpace(directory))
        {
            return null;
        }

        directory = Path.GetFullPath(directory);
        Directory.CreateDirectory(directory);
        return directory;
    }
}
