Mojo output refuses `DispatchMesh` with a `mojo-lowering-unsupported` diagnostic, since Mojo's gpu package has no
task stage and no operation that launches mesh workgroups. It used to call a generated helper whose body was `pass`,
so the task shader launched no mesh workgroups (#4392).
