import Foundation
import Metal

enum ExecutionError: Error {
    case unavailable(String)
}

func run() throws {
    guard (4...5).contains(CommandLine.arguments.count),
          let count = Int(CommandLine.arguments[3]), count > 0,
          let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        throw ExecutionError.unavailable("Expected metallib, entry, count and a Metal device")
    }
    let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
    guard let function = library.makeFunction(name: CommandLine.arguments[2]) else {
        throw ExecutionError.unavailable("Kernel entry point is missing")
    }
    let pipeline: MTLComputePipelineState
    if CommandLine.arguments.count == 5 {
        let linked = MTLLinkedFunctions()
        linked.functions = try CommandLine.arguments[4].split(separator: ",").map { name in
            guard let callable = library.makeFunction(name: String(name)) else {
                throw ExecutionError.unavailable("Visible function is missing: \(name)")
            }
            return callable
        }
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.linkedFunctions = linked
        pipeline = try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
    } else {
        pipeline = try device.makeComputePipelineState(function: function)
    }
    guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride,
                                        options: .storageModeShared),
          let command = queue.makeCommandBuffer(),
          let encoder = command.makeComputeCommandEncoder() else {
        throw ExecutionError.unavailable("Metal allocation failed")
    }
    buffer.contents().initializeMemory(as: Float.self, repeating: .nan, count: count)
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(buffer, offset: 0, index: 0)
    encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
        throw ExecutionError.unavailable("Metal dispatch failed: \(String(describing: command.error))")
    }
    let values = Array(UnsafeBufferPointer(
        start: buffer.contents().assumingMemoryBound(to: Float.self), count: count
    ))
    let output = try JSONSerialization.data(withJSONObject: [
        "values": values, "device": device.name,
    ], options: [.sortedKeys])
    FileHandle.standardOutput.write(output)
}

do {
    try run()
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
