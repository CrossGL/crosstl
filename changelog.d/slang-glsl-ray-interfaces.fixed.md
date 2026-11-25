Slang output now writes GLSL and SPIR-V ray tracing stages in the form Slang declares: the payload, hit attributes and
callable data become `static` variables that the entry point receives as parameters and copies back, `TraceRay` takes
a `RayDesc` and the payload, `CallShader` the callable data and `ReportHit` the hit attributes. slangc rejected the
module-scope interface variables CrossTL wrote before (a non-`static` global is a uniform) and `ReportHit` without
its attribute argument; `ReportHit(t, kind)` without a hit attribute variable now reports placeholder attributes of
the type the hit stages declare. CrossGL ray stages without such variables keep their Slang forms. What has no DXR form, such as `gl_ObjectToWorldEXT`, is refused for Slang.
