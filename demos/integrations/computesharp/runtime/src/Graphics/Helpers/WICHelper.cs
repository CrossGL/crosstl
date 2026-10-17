using System;
using System.Buffers;
using System.IO;
using System.Runtime.InteropServices;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Formats.Bmp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.Formats.Tiff;
using SixLabors.ImageSharp.PixelFormats;
using ImageSharpBgra32 = SixLabors.ImageSharp.PixelFormats.Bgra32;
using ImageSharpRgba32 = SixLabors.ImageSharp.PixelFormats.Rgba32;
using ImageSharpRgba64 = SixLabors.ImageSharp.PixelFormats.Rgba64;

namespace ComputeSharp.Graphics.Helpers;

/// <summary>
/// Image loading and saving for textures. The original runtime uses Windows Imaging Component;
/// off Windows this uses SixLabors.ImageSharp (the version the upstream tests pin) with the
/// same supported element types: <see cref="Bgra32"/>, <see cref="Rgba32"/>, <see cref="Rgba64"/>,
/// <see cref="R8"/> and <see cref="R16"/>. JPEG XR (WMP) and DDS containers are not available.
/// </summary>
internal sealed unsafe class WICHelper
{
    private WICHelper()
    {
    }

    public static WICHelper Instance { get; } = new();

    public UploadTexture2D<T> LoadTexture<T>(GraphicsDevice device, ReadOnlySpan<char> filename)
        where T : unmanaged
    {
        using FileStream stream = File.OpenRead(filename.ToString());

        return LoadTexture<T>(device, stream);
    }

    public UploadTexture2D<T> LoadTexture<T>(GraphicsDevice device, ReadOnlySpan<byte> span)
        where T : unmanaged
    {
        using MemoryStream stream = new(span.ToArray(), writable: false);

        return LoadTexture<T>(device, stream);
    }

    public UploadTexture2D<T> LoadTexture<T>(GraphicsDevice device, Stream stream)
        where T : unmanaged
    {
        ValidateType<T>();

        using Image image = Decode<T>(stream);

        UploadTexture2D<T> upload = device.AllocateUploadTexture2D<T>(image.Width, image.Height);

        try
        {
            CopyPixels(image, upload.View);
        }
        catch
        {
            upload.Dispose();
            throw;
        }

        return upload;
    }

    public void LoadTexture<T>(TextureView2D<T> texture, ReadOnlySpan<char> filename)
        where T : unmanaged
    {
        using FileStream stream = File.OpenRead(filename.ToString());

        LoadTexture(texture, stream);
    }

    public void LoadTexture<T>(TextureView2D<T> texture, ReadOnlySpan<byte> span)
        where T : unmanaged
    {
        using MemoryStream stream = new(span.ToArray(), writable: false);

        LoadTexture(texture, stream);
    }

    public void LoadTexture<T>(TextureView2D<T> texture, Stream stream)
        where T : unmanaged
    {
        ValidateType<T>();

        using Image image = Decode<T>(stream);

        default(ArgumentException).ThrowIf(texture.Width != image.Width, nameof(texture.Width));
        default(ArgumentException).ThrowIf(texture.Height != image.Height, nameof(texture.Height));

        CopyPixels(image, texture);
    }

    public void SaveTexture<T>(TextureView2D<T> texture, ReadOnlySpan<char> filename)
        where T : unmanaged
    {
        IImageEncoder encoder = EncoderForFilename(filename);
        using Image image = ToImage(texture);
        using FileStream stream = File.Create(filename.ToString());

        image.Save(stream, encoder);
    }

    public void SaveTexture<T>(TextureView2D<T> texture, Stream stream, ImageFormat format)
        where T : unmanaged
    {
        IImageEncoder encoder = EncoderForFormat(format);
        using Image image = ToImage(texture);

        image.Save(stream, encoder);
    }

    public void SaveTexture<T>(TextureView2D<T> texture, IBufferWriter<byte> writer, ImageFormat format)
        where T : unmanaged
    {
        using MemoryStream stream = new();

        SaveTexture(texture, stream, format);

        stream.GetBuffer().AsSpan(0, (int)stream.Length).CopyTo(writer.GetSpan((int)stream.Length));
        writer.Advance((int)stream.Length);
    }

    private static void ValidateType<T>()
        where T : unmanaged
    {
        if (typeof(T) != typeof(Bgra32) &&
            typeof(T) != typeof(Rgba32) &&
            typeof(T) != typeof(Rgba64) &&
            typeof(T) != typeof(R8) &&
            typeof(T) != typeof(R16))
        {
            default(ArgumentException).Throw(nameof(T));
        }
    }

    private static Image Decode<T>(Stream stream)
        where T : unmanaged
    {
        if (typeof(T) == typeof(Bgra32)) return Image.Load<ImageSharpBgra32>(stream);
        if (typeof(T) == typeof(Rgba32)) return Image.Load<ImageSharpRgba32>(stream);
        if (typeof(T) == typeof(Rgba64)) return Image.Load<ImageSharpRgba64>(stream);
        if (typeof(T) == typeof(R8)) return Image.Load<L8>(stream);
        if (typeof(T) == typeof(R16)) return Image.Load<L16>(stream);

        throw new ArgumentException($"Unsupported image element type {typeof(T)}.", nameof(T));
    }

    private static void CopyPixels<T>(Image image, TextureView2D<T> texture)
        where T : unmanaged
    {
        switch (image)
        {
            case Image<ImageSharpBgra32> bgra: CopyPixels(bgra, texture); break;
            case Image<ImageSharpRgba32> rgba: CopyPixels(rgba, texture); break;
            case Image<ImageSharpRgba64> rgba64: CopyPixels(rgba64, texture); break;
            case Image<L8> l8: CopyPixels(l8, texture); break;
            case Image<L16> l16: CopyPixels(l16, texture); break;
            default: throw new InvalidOperationException("Unexpected decoded image type.");
        }
    }

    private static void CopyPixels<TImage, T>(Image<TImage> image, TextureView2D<T> texture)
        where TImage : unmanaged, IPixel<TImage>
        where T : unmanaged
    {
        if (sizeof(TImage) != sizeof(T))
        {
            throw new InvalidOperationException($"Pixel size mismatch between {typeof(TImage)} and {typeof(T)}.");
        }

        T* data = texture.DangerousGetAddressAndByteStride(out int strideInBytes);
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<TImage> row = accessor.GetRowSpan(y);
                MemoryMarshal.AsBytes(row).CopyTo(new Span<byte>((byte*)data + ((long)y * strideInBytes), row.Length * sizeof(TImage)));
            }
        });
    }

    private static Image ToImage<T>(TextureView2D<T> texture)
        where T : unmanaged
    {
        if (typeof(T) == typeof(Bgra32)) return ToImage<ImageSharpBgra32, T>(texture);
        if (typeof(T) == typeof(Rgba32)) return ToImage<ImageSharpRgba32, T>(texture);
        if (typeof(T) == typeof(Rgba64)) return ToImage<ImageSharpRgba64, T>(texture);
        if (typeof(T) == typeof(R8)) return ToImage<L8, T>(texture);
        if (typeof(T) == typeof(R16)) return ToImage<L16, T>(texture);

        throw new ArgumentException($"Unsupported image element type {typeof(T)}.", nameof(T));
    }

    private static Image<TImage> ToImage<TImage, T>(TextureView2D<T> texture)
        where TImage : unmanaged, IPixel<TImage>
        where T : unmanaged
    {
        Image<TImage> image = new(texture.Width, texture.Height);
        T* data = texture.DangerousGetAddressAndByteStride(out int strideInBytes);
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<TImage> row = accessor.GetRowSpan(y);
                new ReadOnlySpan<byte>((byte*)data + ((long)y * strideInBytes), row.Length * sizeof(TImage)).CopyTo(MemoryMarshal.AsBytes(row));
            }
        });

        return image;
    }

    private static IImageEncoder EncoderForFilename(ReadOnlySpan<char> filename)
    {
        string extension = Path.GetExtension(filename).ToString().ToLowerInvariant();

        return extension switch
        {
            ".dib" or ".rle" or ".bmp" => new BmpEncoder(),
            ".png" => new PngEncoder(),
            ".jpe" or ".jfif" or ".exif" or ".jpg" or ".jpeg" => new JpegEncoder(),
            ".tif" or ".tiff" => new TiffEncoder(),
            ".jxr" or ".hdp" or ".wdp" or ".wmp" or ".dds" => throw new NotSupportedException($"The {extension} container is not available off Windows."),
            _ => throw new ArgumentException("Unsupported image file extension.", nameof(filename))
        };
    }

    private static IImageEncoder EncoderForFormat(ImageFormat format)
    {
        return format switch
        {
            ImageFormat.Bmp => new BmpEncoder(),
            ImageFormat.Png => new PngEncoder(),
            ImageFormat.Jpeg => new JpegEncoder(),
            ImageFormat.Tiff => new TiffEncoder(),
            ImageFormat.Wmp or ImageFormat.Dds => throw new NotSupportedException($"The {format} container is not available off Windows."),
            _ => throw new ArgumentException("Unsupported image format.", nameof(format))
        };
    }
}
