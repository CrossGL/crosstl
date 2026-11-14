// Portable ComputeContext extensions. The overload set matches the pinned upstream
// ComputeContextExtensions; operations are recorded into the context's native command list.
using System;
using ComputeSharp.Descriptors;
using ComputeSharp.Graphics.Helpers;
using ComputeSharp.Interop;
using ComputeSharp.Portable;
using ComputeSharp.Resources;
using ComputeSharp.Resources.Interop;

namespace ComputeSharp;

/// <summary>A <see langword="class"/> with extension methods for <see cref="ComputeContext"/>.</summary>
public static class ComputeContextExtensions
{

    /// <summary>Inserts a barrier for the input buffer.</summary>
    public static void Barrier<T>(this ref readonly ComputeContext context, ReadWriteBuffer<T> buffer)
        where T : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(buffer);

        _ = buffer.ValidateAndGetNativeBuffer(context.GraphicsDevice, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Barrier();
        }
    }

    /// <summary>Inserts a barrier for the input resource.</summary>
    public static void Barrier<T>(this ref readonly ComputeContext context, ReadWriteTexture1D<T> texture)
        where T : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        _ = ((IPortableReadOnlyResource)texture).ValidateAndGetView(context.GraphicsDevice, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Barrier();
        }
    }

    /// <summary>Inserts a barrier for the input resource.</summary>
    public static void Barrier<T>(this ref readonly ComputeContext context, ReadWriteTexture2D<T> texture)
        where T : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        _ = ((IPortableReadOnlyResource)texture).ValidateAndGetView(context.GraphicsDevice, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Barrier();
        }
    }

    /// <summary>Inserts a barrier for the input resource.</summary>
    public static void Barrier<T>(this ref readonly ComputeContext context, ReadWriteTexture3D<T> texture)
        where T : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        _ = ((IPortableReadOnlyResource)texture).ValidateAndGetView(context.GraphicsDevice, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Barrier();
        }
    }

    /// <summary>Inserts a barrier for the input resource.</summary>
    public static void Barrier<T, TPixel>(this ref readonly ComputeContext context, ReadWriteTexture1D<T, TPixel> texture)
        where T : unmanaged, IPixel<T, TPixel>
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        _ = ((IPortableReadOnlyResource)texture).ValidateAndGetView(context.GraphicsDevice, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Barrier();
        }
    }

    /// <summary>Inserts a barrier for the input resource.</summary>
    public static void Barrier<T, TPixel>(this ref readonly ComputeContext context, ReadWriteTexture2D<T, TPixel> texture)
        where T : unmanaged, IPixel<T, TPixel>
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        _ = ((IPortableReadOnlyResource)texture).ValidateAndGetView(context.GraphicsDevice, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Barrier();
        }
    }

    /// <summary>Inserts a barrier for the input resource.</summary>
    public static void Barrier<T, TPixel>(this ref readonly ComputeContext context, ReadWriteTexture3D<T, TPixel> texture)
        where T : unmanaged, IPixel<T, TPixel>
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        _ = ((IPortableReadOnlyResource)texture).ValidateAndGetView(context.GraphicsDevice, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Barrier();
        }
    }

    /// <summary>Inserts a barrier for the input resource.</summary>
    public static void Barrier<TPixel>(this ref readonly ComputeContext context, IReadWriteNormalizedTexture1D<TPixel> texture)
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        _ = ((IPortableReadOnlyResource)texture).ValidateAndGetView(context.GraphicsDevice, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Barrier();
        }
    }

    /// <summary>Inserts a barrier for the input resource.</summary>
    public static void Barrier<TPixel>(this ref readonly ComputeContext context, IReadWriteNormalizedTexture2D<TPixel> texture)
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        _ = ((IPortableReadOnlyResource)texture).ValidateAndGetView(context.GraphicsDevice, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Barrier();
        }
    }

    /// <summary>Inserts a barrier for the input resource.</summary>
    public static void Barrier<TPixel>(this ref readonly ComputeContext context, IReadWriteNormalizedTexture3D<TPixel> texture)
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        _ = ((IPortableReadOnlyResource)texture).ValidateAndGetView(context.GraphicsDevice, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Barrier();
        }
    }

    /// <summary>Clears a buffer, setting all its bytes to zero.</summary>
    public static void Clear<T>(this ref readonly ComputeContext context, ReadWriteBuffer<T> buffer)
        where T : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(buffer);

        IntPtr nativeBuffer = buffer.ValidateAndGetNativeBufferForClear(context.GraphicsDevice, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.ClearBuffer(nativeBuffer, TextureEncoding.BufferSize(buffer));
        }
    }

    /// <summary>Clears a texture, setting all its texels to zero.</summary>
    public static void Clear<T>(this ref readonly ComputeContext context, ReadWriteTexture1D<T> texture)
        where T : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        PortableResourceView view = ((IPortableReadWriteResource)texture).ValidateAndGetViewForClear(context.GraphicsDevice, out _, out ReferenceTracker.Lease lease);

        using (lease)
        {
            TextureEncoding.Clear(in context, view, texture.Width, 1, 1);
        }
    }

    /// <summary>Clears a texture, setting all its texels to zero.</summary>
    public static void Clear<T>(this ref readonly ComputeContext context, ReadWriteTexture2D<T> texture)
        where T : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        PortableResourceView view = ((IPortableReadWriteResource)texture).ValidateAndGetViewForClear(context.GraphicsDevice, out _, out ReferenceTracker.Lease lease);

        using (lease)
        {
            TextureEncoding.Clear(in context, view, texture.Width, texture.Height, 1);
        }
    }

    /// <summary>Clears a texture, setting all its texels to zero.</summary>
    public static void Clear<T>(this ref readonly ComputeContext context, ReadWriteTexture3D<T> texture)
        where T : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        PortableResourceView view = ((IPortableReadWriteResource)texture).ValidateAndGetViewForClear(context.GraphicsDevice, out _, out ReferenceTracker.Lease lease);

        using (lease)
        {
            TextureEncoding.Clear(in context, view, texture.Width, texture.Height, texture.Depth);
        }
    }

    /// <summary>Clears a texture, setting all its texels to zero.</summary>
    public static void Clear<T, TPixel>(this ref readonly ComputeContext context, ReadWriteTexture1D<T, TPixel> texture)
        where T : unmanaged, IPixel<T, TPixel>
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        PortableResourceView view = ((IPortableReadWriteResource)texture).ValidateAndGetViewForClear(context.GraphicsDevice, out _, out ReferenceTracker.Lease lease);

        using (lease)
        {
            TextureEncoding.Clear(in context, view, texture.Width, 1, 1);
        }
    }

    /// <summary>Clears a texture, setting all its texels to zero.</summary>
    public static void Clear<T, TPixel>(this ref readonly ComputeContext context, ReadWriteTexture2D<T, TPixel> texture)
        where T : unmanaged, IPixel<T, TPixel>
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        PortableResourceView view = ((IPortableReadWriteResource)texture).ValidateAndGetViewForClear(context.GraphicsDevice, out _, out ReferenceTracker.Lease lease);

        using (lease)
        {
            TextureEncoding.Clear(in context, view, texture.Width, texture.Height, 1);
        }
    }

    /// <summary>Clears a texture, setting all its texels to zero.</summary>
    public static void Clear<T, TPixel>(this ref readonly ComputeContext context, ReadWriteTexture3D<T, TPixel> texture)
        where T : unmanaged, IPixel<T, TPixel>
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        PortableResourceView view = ((IPortableReadWriteResource)texture).ValidateAndGetViewForClear(context.GraphicsDevice, out _, out ReferenceTracker.Lease lease);

        using (lease)
        {
            TextureEncoding.Clear(in context, view, texture.Width, texture.Height, texture.Depth);
        }
    }

    /// <summary>Clears a texture, setting all its texels to zero.</summary>
    public static void Clear<TPixel>(this ref readonly ComputeContext context, IReadWriteNormalizedTexture1D<TPixel> texture)
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        PortableResourceView view = ((IPortableReadWriteResource)texture).ValidateAndGetViewForClear(context.GraphicsDevice, out _, out ReferenceTracker.Lease lease);

        using (lease)
        {
            TextureEncoding.Clear(in context, view, texture.Width, 1, 1);
        }
    }

    /// <summary>Clears a texture, setting all its texels to zero.</summary>
    public static void Clear<TPixel>(this ref readonly ComputeContext context, IReadWriteNormalizedTexture2D<TPixel> texture)
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        PortableResourceView view = ((IPortableReadWriteResource)texture).ValidateAndGetViewForClear(context.GraphicsDevice, out _, out ReferenceTracker.Lease lease);

        using (lease)
        {
            TextureEncoding.Clear(in context, view, texture.Width, texture.Height, 1);
        }
    }

    /// <summary>Clears a texture, setting all its texels to zero.</summary>
    public static void Clear<TPixel>(this ref readonly ComputeContext context, IReadWriteNormalizedTexture3D<TPixel> texture)
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        PortableResourceView view = ((IPortableReadWriteResource)texture).ValidateAndGetViewForClear(context.GraphicsDevice, out _, out ReferenceTracker.Lease lease);

        using (lease)
        {
            TextureEncoding.Clear(in context, view, texture.Width, texture.Height, texture.Depth);
        }
    }

    /// <summary>Fills a texture with a specified value.</summary>
    public static void Fill<T, TPixel>(this ref readonly ComputeContext context, ReadWriteTexture1D<T, TPixel> texture, T value)
        where T : unmanaged, IPixel<T, TPixel>
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        PortableResourceView view = ((IPortableReadWriteResource)texture).ValidateAndGetViewForClear(context.GraphicsDevice, out _, out ReferenceTracker.Lease lease);

        using (lease)
        {
            TextureEncoding.Fill(in context, view, texture.Width, 1, 1, DXGIFormatHelper.ExtendToNormalizedValue(value.ToPixel()));
        }
    }

    /// <summary>Fills a texture with a specified value.</summary>
    public static void Fill<T, TPixel>(this ref readonly ComputeContext context, ReadWriteTexture2D<T, TPixel> texture, T value)
        where T : unmanaged, IPixel<T, TPixel>
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        PortableResourceView view = ((IPortableReadWriteResource)texture).ValidateAndGetViewForClear(context.GraphicsDevice, out _, out ReferenceTracker.Lease lease);

        using (lease)
        {
            TextureEncoding.Fill(in context, view, texture.Width, texture.Height, 1, DXGIFormatHelper.ExtendToNormalizedValue(value.ToPixel()));
        }
    }

    /// <summary>Fills a texture with a specified value.</summary>
    public static void Fill<T, TPixel>(this ref readonly ComputeContext context, ReadWriteTexture3D<T, TPixel> texture, T value)
        where T : unmanaged, IPixel<T, TPixel>
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        PortableResourceView view = ((IPortableReadWriteResource)texture).ValidateAndGetViewForClear(context.GraphicsDevice, out _, out ReferenceTracker.Lease lease);

        using (lease)
        {
            TextureEncoding.Fill(in context, view, texture.Width, texture.Height, texture.Depth, DXGIFormatHelper.ExtendToNormalizedValue(value.ToPixel()));
        }
    }

    /// <summary>Fills a texture with a specified value.</summary>
    public static void Fill<TPixel>(this ref readonly ComputeContext context, IReadWriteNormalizedTexture1D<TPixel> texture, TPixel value)
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        PortableResourceView view = ((IPortableReadWriteResource)texture).ValidateAndGetViewForClear(context.GraphicsDevice, out _, out ReferenceTracker.Lease lease);

        using (lease)
        {
            TextureEncoding.Fill(in context, view, texture.Width, 1, 1, DXGIFormatHelper.ExtendToNormalizedValue(value));
        }
    }

    /// <summary>Fills a texture with a specified value.</summary>
    public static void Fill<TPixel>(this ref readonly ComputeContext context, IReadWriteNormalizedTexture2D<TPixel> texture, TPixel value)
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        PortableResourceView view = ((IPortableReadWriteResource)texture).ValidateAndGetViewForClear(context.GraphicsDevice, out _, out ReferenceTracker.Lease lease);

        using (lease)
        {
            TextureEncoding.Fill(in context, view, texture.Width, texture.Height, 1, DXGIFormatHelper.ExtendToNormalizedValue(value));
        }
    }

    /// <summary>Fills a texture with a specified value.</summary>
    public static void Fill<TPixel>(this ref readonly ComputeContext context, IReadWriteNormalizedTexture3D<TPixel> texture, TPixel value)
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        PortableResourceView view = ((IPortableReadWriteResource)texture).ValidateAndGetViewForClear(context.GraphicsDevice, out _, out ReferenceTracker.Lease lease);

        using (lease)
        {
            TextureEncoding.Fill(in context, view, texture.Width, texture.Height, texture.Depth, DXGIFormatHelper.ExtendToNormalizedValue(value));
        }
    }

    /// <summary>Runs the input shader with the specified parameters.</summary>
    public static void For<T>(this ref readonly ComputeContext context, int x, in T shader)
        where T : struct, IComputeShader, IComputeShaderDescriptor<T>
    {
        context.Run(x, in shader);
    }

    /// <summary>Runs the input shader with the specified parameters.</summary>
    public static void For<T>(this ref readonly ComputeContext context, int x, int y, in T shader)
        where T : struct, IComputeShader, IComputeShaderDescriptor<T>
    {
        context.Run(x, y, in shader);
    }

    /// <summary>Runs the input shader with the specified parameters.</summary>
    public static void For<T>(this ref readonly ComputeContext context, int x, int y, int z, in T shader)
        where T : struct, IComputeShader, IComputeShaderDescriptor<T>
    {
        context.Run(x, y, z, in shader);
    }

    /// <summary>Runs the input pixel shader over a target texture.</summary>
    public static void ForEach<T, TPixel>(this ref readonly ComputeContext context, IReadWriteNormalizedTexture2D<TPixel> texture)
        where T : struct, IComputeShader<TPixel>, IComputeShaderDescriptor<T>
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        context.Run(texture, default(T));
    }

    /// <summary>Runs the input pixel shader over a target texture.</summary>
    public static void ForEach<T, TPixel>(this ref readonly ComputeContext context, IReadWriteNormalizedTexture2D<TPixel> texture, in T shader)
        where T : struct, IComputeShader<TPixel>, IComputeShaderDescriptor<T>
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        context.Run(texture, in shader);
    }

    /// <summary>Transitions a texture to a specified state.</summary>
    public static void Transition(this ref readonly ComputeContext context, ReadWriteTexture1D<float> texture, ResourceState resourceState)
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        ((IPortableReadWriteResource)texture).ValidateAndTransition(context.GraphicsDevice, resourceState, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Transition();
        }
    }

    /// <summary>Transitions a texture to a specified state.</summary>
    public static void Transition(this ref readonly ComputeContext context, ReadWriteTexture1D<Float2> texture, ResourceState resourceState)
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        ((IPortableReadWriteResource)texture).ValidateAndTransition(context.GraphicsDevice, resourceState, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Transition();
        }
    }

    /// <summary>Transitions a texture to a specified state.</summary>
    public static void Transition(this ref readonly ComputeContext context, ReadWriteTexture1D<Float3> texture, ResourceState resourceState)
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        ((IPortableReadWriteResource)texture).ValidateAndTransition(context.GraphicsDevice, resourceState, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Transition();
        }
    }

    /// <summary>Transitions a texture to a specified state.</summary>
    public static void Transition(this ref readonly ComputeContext context, ReadWriteTexture1D<Float4> texture, ResourceState resourceState)
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        ((IPortableReadWriteResource)texture).ValidateAndTransition(context.GraphicsDevice, resourceState, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Transition();
        }
    }

    /// <summary>Transitions a texture to a specified state.</summary>
    public static void Transition<T, TPixel>(this ref readonly ComputeContext context, ReadWriteTexture1D<T, TPixel> texture, ResourceState resourceState)
        where T : unmanaged, IPixel<T, TPixel>
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        ((IPortableReadWriteResource)texture).ValidateAndTransition(context.GraphicsDevice, resourceState, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Transition();
        }
    }

    /// <summary>Transitions a texture to a specified state.</summary>
    public static void Transition(this ref readonly ComputeContext context, ReadWriteTexture2D<float> texture, ResourceState resourceState)
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        ((IPortableReadWriteResource)texture).ValidateAndTransition(context.GraphicsDevice, resourceState, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Transition();
        }
    }

    /// <summary>Transitions a texture to a specified state.</summary>
    public static void Transition(this ref readonly ComputeContext context, ReadWriteTexture2D<Float2> texture, ResourceState resourceState)
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        ((IPortableReadWriteResource)texture).ValidateAndTransition(context.GraphicsDevice, resourceState, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Transition();
        }
    }

    /// <summary>Transitions a texture to a specified state.</summary>
    public static void Transition(this ref readonly ComputeContext context, ReadWriteTexture2D<Float3> texture, ResourceState resourceState)
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        ((IPortableReadWriteResource)texture).ValidateAndTransition(context.GraphicsDevice, resourceState, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Transition();
        }
    }

    /// <summary>Transitions a texture to a specified state.</summary>
    public static void Transition(this ref readonly ComputeContext context, ReadWriteTexture2D<Float4> texture, ResourceState resourceState)
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        ((IPortableReadWriteResource)texture).ValidateAndTransition(context.GraphicsDevice, resourceState, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Transition();
        }
    }

    /// <summary>Transitions a texture to a specified state.</summary>
    public static void Transition<T, TPixel>(this ref readonly ComputeContext context, ReadWriteTexture2D<T, TPixel> texture, ResourceState resourceState)
        where T : unmanaged, IPixel<T, TPixel>
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        ((IPortableReadWriteResource)texture).ValidateAndTransition(context.GraphicsDevice, resourceState, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Transition();
        }
    }

    /// <summary>Transitions a texture to a specified state.</summary>
    public static void Transition(this ref readonly ComputeContext context, ReadWriteTexture3D<float> texture, ResourceState resourceState)
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        ((IPortableReadWriteResource)texture).ValidateAndTransition(context.GraphicsDevice, resourceState, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Transition();
        }
    }

    /// <summary>Transitions a texture to a specified state.</summary>
    public static void Transition(this ref readonly ComputeContext context, ReadWriteTexture3D<Float2> texture, ResourceState resourceState)
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        ((IPortableReadWriteResource)texture).ValidateAndTransition(context.GraphicsDevice, resourceState, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Transition();
        }
    }

    /// <summary>Transitions a texture to a specified state.</summary>
    public static void Transition(this ref readonly ComputeContext context, ReadWriteTexture3D<Float3> texture, ResourceState resourceState)
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        ((IPortableReadWriteResource)texture).ValidateAndTransition(context.GraphicsDevice, resourceState, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Transition();
        }
    }

    /// <summary>Transitions a texture to a specified state.</summary>
    public static void Transition(this ref readonly ComputeContext context, ReadWriteTexture3D<Float4> texture, ResourceState resourceState)
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        ((IPortableReadWriteResource)texture).ValidateAndTransition(context.GraphicsDevice, resourceState, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Transition();
        }
    }

    /// <summary>Transitions a texture to a specified state.</summary>
    public static void Transition<T, TPixel>(this ref readonly ComputeContext context, ReadWriteTexture3D<T, TPixel> texture, ResourceState resourceState)
        where T : unmanaged, IPixel<T, TPixel>
        where TPixel : unmanaged
    {
        default(ArgumentNullException).ThrowIfNull(texture);

        ((IPortableReadWriteResource)texture).ValidateAndTransition(context.GraphicsDevice, resourceState, out ReferenceTracker.Lease lease);

        using (lease)
        {
            context.Transition();
        }
    }
}
