import Foundation
import Metal

struct Request: Decodable {
    let left: [Float]
    let right: [Float]
}

enum ExecutionError: Error {
    case unavailable(String)
}

func run() throws {
    guard CommandLine.arguments.count == 4 else {
        throw ExecutionError.unavailable("Expected metallib, entry and request paths")
    }
    let request = try JSONDecoder().decode(Request.self, from: Data(contentsOf:
        URL(fileURLWithPath: CommandLine.arguments[3])))
    guard !request.left.isEmpty, request.left.count % 2 == 0,
          request.left.count == request.right.count,
          let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        throw ExecutionError.unavailable("Expected complex pairs and a Metal device")
    }
    let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
    guard let function = library.makeFunction(name: CommandLine.arguments[2]) else {
        throw ExecutionError.unavailable("Kernel entry point is missing")
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    func buffer<T>(_ values: [T]) throws -> MTLBuffer {
        let result = values.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        }
        guard let result else { throw ExecutionError.unavailable("Allocation failed") }
        return result
    }
    let buffers = try [buffer(request.left), buffer(request.right),
        buffer([Float](repeating: -1234, count: request.left.count)),
        buffer([Int64(1)]), buffer([Int64(1)])]
    guard let command = queue.makeCommandBuffer(),
          let encoder = command.makeComputeCommandEncoder() else {
        throw ExecutionError.unavailable("Command allocation failed")
    }
    encoder.setComputePipelineState(pipeline)
    for (index, allocation) in buffers.enumerated() {
        encoder.setBuffer(allocation, offset: 0, index: index)
    }
    encoder.dispatchThreadgroups(MTLSize(width: request.left.count / 2, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
        throw ExecutionError.unavailable("Dispatch failed: \(String(describing: command.error))")
    }
    let values = Array(UnsafeBufferPointer(
        start: buffers[2].contents().assumingMemoryBound(to: Float.self), count: request.left.count))
    let output = try JSONSerialization.data(withJSONObject: ["values": values, "device": device.name],
                                            options: [.sortedKeys])
    FileHandle.standardOutput.write(output)
}

do { try run() } catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
