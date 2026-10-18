import Foundation
import Metal

struct Request: Decodable {
    let inputBits: [UInt32]
    let rows: Int
    let axisSize: UInt64
    let axisStride: Int64
    let rowStride: Int64
}

enum ExecutionError: Error {
    case unavailable(String)
}

func run() throws {
    guard CommandLine.arguments.count == 4 else {
        throw ExecutionError.unavailable("Expected metallib, entry point, and request paths")
    }
    let request = try JSONDecoder().decode(
        Request.self,
        from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[3]))
    )
    guard request.rows > 0, !request.inputBits.isEmpty,
          let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        throw ExecutionError.unavailable("A Metal device and nonempty inputs are required")
    }
    let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
    guard let function = library.makeFunction(name: CommandLine.arguments[2]) else {
        throw ExecutionError.unavailable("Kernel entry point is missing")
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    guard pipeline.threadExecutionWidth == 32 else {
        throw ExecutionError.unavailable("This dispatch requires 32-lane SIMD groups")
    }

    func buffer<T>(_ values: [T]) throws -> MTLBuffer {
        let allocation = values.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        }
        guard let allocation else {
            throw ExecutionError.unavailable("Metal buffer allocation failed")
        }
        return allocation
    }
    let buffers = try [
        buffer(request.inputBits),
        buffer([UInt32](repeating: UInt32.max, count: request.rows)),
        buffer([Int32(request.rows)]),
        buffer([request.rowStride]),
        buffer([Int64(1)]),
        buffer([UInt64(1)]),
        buffer([request.axisStride]),
        buffer([request.axisSize]),
    ]
    guard let command = queue.makeCommandBuffer(),
          let encoder = command.makeComputeCommandEncoder() else {
        throw ExecutionError.unavailable("Metal command allocation failed")
    }
    encoder.setComputePipelineState(pipeline)
    for (index, allocation) in buffers.enumerated() {
        encoder.setBuffer(allocation, offset: 0, index: index)
    }
    encoder.dispatchThreadgroups(
        MTLSize(width: 1, height: request.rows, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
        throw ExecutionError.unavailable("Metal dispatch failed: \(String(describing: command.error))")
    }
    let indices = Array(UnsafeBufferPointer(
        start: buffers[1].contents().assumingMemoryBound(to: UInt32.self),
        count: request.rows
    ))
    let output = try JSONSerialization.data(withJSONObject: [
        "indices": indices,
        "threadExecutionWidth": pipeline.threadExecutionWidth,
    ], options: [.sortedKeys])
    FileHandle.standardOutput.write(output)
}

do {
    try run()
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
