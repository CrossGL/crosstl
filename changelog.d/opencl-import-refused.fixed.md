OpenCL C input reports a construct it cannot import, such as inline `asm`, with `OpenCLImportRefused`, the
`project.translate.opencl-import-unsupported` code and an `OpenCL C` message. It used to raise `HipImportRefused` and
describe the construct as HIP (#4699).
