"""Graphics built-ins, interpolation qualifiers and execution modes of the vertex, fragment, geometry, tessellation,
mesh, task and ray-tracing stages."""

from .base import c

GRAPHICS = (
    c("graphics", "fragment-coordinate", "BuiltIn.FragCoord", stage="fragment"),
    c("graphics", "front-facing", "BuiltIn.FrontFacing", stage="fragment"),
    c("graphics", "fragment-depth", "BuiltIn.FragDepth", stage="fragment"),
    c("graphics", "flat-interpolation", "Decoration.Flat", stage="fragment"),
    c(
        "graphics",
        "noperspective-interpolation",
        "Decoration.NoPerspective",
        stage="fragment",
    ),
    c(
        "graphics",
        "centroid-interpolation",
        "Decoration.Centroid",
        stage="fragment",
    ),
    c(
        "graphics",
        "sample-interpolation",
        "Decoration.Sample",
        "Capability.SampleRateShading",
        stage="fragment",
    ),
    c(
        "graphics",
        "multiple-render-targets",
        "Decoration.Location",
        stage="fragment",
    ),
    c("graphics", "vertex-index", "BuiltIn.VertexIndex", stage="vertex"),
    c("graphics", "instance-index", "BuiltIn.InstanceIndex", stage="vertex"),
    c("graphics", "point-size", "BuiltIn.PointSize", stage="vertex"),
    c(
        "graphics",
        "clip-distance",
        "BuiltIn.ClipDistance",
        "Capability.ClipDistance",
        stage="vertex",
    ),
    c("graphics", "invariant-position", "Decoration.Invariant", stage="vertex"),
    c(
        "graphics",
        "draw-index",
        "BuiltIn.DrawIndex",
        "Capability.DrawParameters",
        stage="vertex",
    ),
    c(
        "graphics",
        "base-vertex",
        "BuiltIn.BaseVertex",
        "Capability.DrawParameters",
        stage="vertex",
    ),
    c(
        "graphics",
        "base-instance",
        "BuiltIn.BaseInstance",
        "Capability.DrawParameters",
        stage="vertex",
    ),
    c(
        "graphics",
        "cull-distance",
        "BuiltIn.CullDistance",
        "Capability.CullDistance",
        stage="vertex",
    ),
    c("graphics", "primitive-id", "BuiltIn.PrimitiveId", stage="fragment"),
    c(
        "graphics",
        "sample-index",
        "BuiltIn.SampleId",
        "Capability.SampleRateShading",
        stage="fragment",
    ),
    c("graphics", "sample-mask-input", "BuiltIn.SampleMask", stage="fragment"),
    c("graphics", "sample-mask", "BuiltIn.SampleMask", stage="fragment"),
    c(
        "graphics",
        "helper-invocation",
        "BuiltIn.HelperInvocation",
        stage="fragment",
    ),
    c(
        "graphics",
        "early-fragment-tests",
        "ExecutionMode.EarlyFragmentTests",
        stage="fragment",
    ),
    *(
        c(
            "graphics",
            name,
            "ExecutionMode.PixelInterlockOrderedEXT",
            "Capability.FragmentShaderPixelInterlockEXT",
            "OpBeginInvocationInterlockEXT",
            stage="fragment",
        )
        for name in ("rasterizer-ordered-image", "rasterizer-ordered-buffer")
    ),
    c("graphics", "layer", "BuiltIn.Layer", stage="fragment"),
    c("graphics", "viewport-index", "BuiltIn.ViewportIndex", stage="fragment"),
    c("graphics", "point-coordinate", "BuiltIn.PointCoord", stage="fragment"),
    c(
        "graphics",
        "conservative-depth",
        "BuiltIn.FragDepth",
        "ExecutionMode.DepthGreater",
        stage="fragment",
    ),
    c(
        "graphics",
        "barycentric-coordinate",
        "BuiltIn.BaryCoordKHR",
        "Capability.FragmentBarycentricKHR",
        stage="fragment",
    ),
    c(
        "graphics",
        "shading-rate",
        "BuiltIn.ShadingRateKHR",
        "Capability.FragmentShadingRateKHR",
        stage="fragment",
    ),
    c(
        "graphics",
        "view-index",
        "BuiltIn.ViewIndex",
        "Capability.MultiView",
        stage="fragment",
    ),
    c(
        "graphics",
        "stencil-export",
        "BuiltIn.FragStencilRefEXT",
        "Capability.StencilExportEXT",
        stage="fragment",
    ),
    *(
        c("graphics", name, f"ExecutionMode.{mode}", stage="geometry")
        for name, mode in (
            ("geometry-input-lines", "InputLines"),
            ("geometry-input-triangles", "Triangles"),
            ("geometry-input-lines-adjacency", "InputLinesAdjacency"),
            ("geometry-input-triangles-adjacency", "InputTrianglesAdjacency"),
            ("geometry-output-line-strip", "OutputLineStrip"),
            ("geometry-output-triangle-strip", "OutputTriangleStrip"),
        )
    ),
    c(
        "graphics",
        "geometry-invocations",
        "ExecutionMode.Invocations",
        "BuiltIn.InvocationId",
        stage="geometry",
    ),
    c(
        "graphics",
        "geometry-streams",
        "Decoration.Stream",
        "Capability.GeometryStreams",
        stage="geometry",
    ),
    *(
        c("graphics", name, *anchors, stage="tessellation-evaluation")
        for name, *anchors in (
            ("tessellation-domain-quads", "ExecutionMode.Quads"),
            ("tessellation-domain-isolines", "ExecutionMode.Isolines"),
            (
                "tessellation-level-input",
                "BuiltIn.TessLevelOuter",
                "BuiltIn.TessLevelInner",
            ),
            ("tessellation-patch-input", "Decoration.Patch"),
            (
                "tessellation-spacing-fractional-odd",
                "ExecutionMode.SpacingFractionalOdd",
            ),
            (
                "tessellation-spacing-fractional-even",
                "ExecutionMode.SpacingFractionalEven",
            ),
            ("tessellation-winding-cw", "ExecutionMode.VertexOrderCw"),
            ("tessellation-point-mode", "ExecutionMode.PointMode"),
        )
    ),
    *(
        c("graphics", name, anchor, stage="tessellation-control")
        for name, anchor in (
            (
                "tessellation-control-spacing-fractional-odd",
                "ExecutionMode.SpacingFractionalOdd",
            ),
            (
                "tessellation-control-spacing-fractional-even",
                "ExecutionMode.SpacingFractionalEven",
            ),
            ("tessellation-control-winding-ccw", "ExecutionMode.VertexOrderCcw"),
            ("tessellation-control-point-mode", "ExecutionMode.PointMode"),
            ("tessellation-patch-output", "Decoration.Patch"),
            ("tessellation-patch-vertices", "BuiltIn.PatchVertices"),
        )
    ),
    *(
        c("graphics", name, *anchors, stage="mesh")
        for name, *anchors in (
            (
                "mesh-output-points",
                "ExecutionMode.OutputPoints",
                "BuiltIn.PrimitivePointIndicesEXT",
            ),
            (
                "mesh-output-lines",
                "ExecutionMode.OutputLinesEXT",
                "BuiltIn.PrimitiveLineIndicesEXT",
            ),
            ("mesh-per-primitive-output", "Decoration.PerPrimitiveEXT"),
            ("mesh-cull-primitive", "BuiltIn.CullPrimitiveEXT"),
            ("mesh-primitive-id", "BuiltIn.PrimitiveId"),
            ("mesh-layer", "BuiltIn.Layer"),
            ("mesh-viewport-index", "BuiltIn.ViewportIndex"),
            ("mesh-task-payload", "StorageClass.TaskPayloadWorkgroupEXT"),
            ("mesh-workgroup-id", "BuiltIn.WorkgroupId"),
            ("mesh-local-invocation-index", "BuiltIn.LocalInvocationIndex"),
        )
    ),
    *(
        c("graphics", name, *anchors, stage="task")
        for name, *anchors in (
            (
                "task-payload-array",
                "StorageClass.TaskPayloadWorkgroupEXT",
                "OpTypeArray",
            ),
            ("task-workgroup-id", "BuiltIn.WorkgroupId"),
            (
                "task-local-invocation-index",
                "BuiltIn.LocalInvocationIndex",
                "OpTypeArray",
            ),
        )
    ),
    *(
        c("graphics", name, *anchors, stage=stage)
        for name, stage, *anchors in (
            (
                "ray-payload",
                "ray-generation",
                "StorageClass.RayPayloadKHR",
                "OpTraceRayKHR",
            ),
            (
                "ray-callable-data",
                "ray-generation",
                "StorageClass.CallableDataKHR",
                "OpExecuteCallableKHR",
            ),
            (
                "ray-reported-attributes",
                "intersection",
                "StorageClass.HitAttributeKHR",
                "OpReportIntersectionKHR",
            ),
            (
                "ray-custom-attributes",
                "closest-hit",
                "StorageClass.HitAttributeKHR",
                "OpTypeStruct",
            ),
            ("ray-shader-record", "miss", "StorageClass.ShaderRecordBufferKHR"),
        )
    ),
)
