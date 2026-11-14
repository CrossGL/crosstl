// Print the threadgroup memory slots of a compute pipeline, one "name index" line per argument.
//
// usage: metal_threadgroup_bindings <library.metallib> <kernel>
//
// The host sizes threadgroup memory with setThreadgroupMemoryLength(_:index:), so these indices
// are the kernel's threadgroup ABI. Lines are sorted by name for a stable comparison.
import Foundation
import Metal

let arguments = CommandLine.arguments
guard arguments.count == 3 else {
  FileHandle.standardError.write("usage: metal_threadgroup_bindings <library> <kernel>\n".data(using: .utf8)!)
  exit(2)
}
guard let device = MTLCreateSystemDefaultDevice() else {
  FileHandle.standardError.write("no Metal device\n".data(using: .utf8)!)
  exit(3)
}
do {
  let library = try device.makeLibrary(URL: URL(fileURLWithPath: arguments[1]))
  guard let function = library.makeFunction(name: arguments[2]) else {
    FileHandle.standardError.write("kernel \(arguments[2]) not found\n".data(using: .utf8)!)
    exit(4)
  }
  var reflection: MTLComputePipelineReflection?
  _ = try device.makeComputePipelineState(
    function: function, options: [.bindingInfo], reflection: &reflection)
  let slots = (reflection?.bindings ?? [])
    .filter { $0.type == .threadgroupMemory }
    .map { "\($0.name) \($0.index)" }
    .sorted()
  for slot in slots {
    print(slot)
  }
} catch {
  FileHandle.standardError.write("\(error)\n".data(using: .utf8)!)
  exit(5)
}
