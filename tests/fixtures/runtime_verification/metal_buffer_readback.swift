import Foundation
import Metal

struct Request: Decodable {
    let buffers: [[UInt8]]
    let grid: [Int]
    let outputIndex: Int
    let outputCount: Int
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
    guard request.grid.count == 3, request.grid.allSatisfy({ $0 > 0 }),
          request.buffers.allSatisfy({ !$0.isEmpty }),
          request.buffers.indices.contains(request.outputIndex), request.outputCount > 0,
          request.outputCount <= request.buffers[request.outputIndex].count / MemoryLayout<Float>.stride,
          let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        throw ExecutionError.unavailable("Invalid buffer request or unavailable Metal device")
    }
    let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
    guard let function = library.makeFunction(name: CommandLine.arguments[2]) else {
        throw ExecutionError.unavailable("Kernel entry point is missing")
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    let buffers = try request.buffers.map { bytes -> MTLBuffer in
        let allocation = bytes.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        }
        guard let allocation else { throw ExecutionError.unavailable("Allocation failed") }
        return allocation
    }
    guard let command = queue.makeCommandBuffer(),
          let encoder = command.makeComputeCommandEncoder() else {
        throw ExecutionError.unavailable("Command allocation failed")
    }
    encoder.setComputePipelineState(pipeline)
    for (index, allocation) in buffers.enumerated() {
        encoder.setBuffer(allocation, offset: 0, index: index)
    }
    encoder.dispatchThreadgroups(MTLSize(width: request.grid[0], height: request.grid[1],
                                        depth: request.grid[2]),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
        throw ExecutionError.unavailable("Dispatch failed: \(String(describing: command.error))")
    }
    let values = Array(UnsafeBufferPointer(
        start: buffers[request.outputIndex].contents().assumingMemoryBound(to: Float.self),
        count: request.outputCount))
    let output = try JSONSerialization.data(withJSONObject: ["values": values, "device": device.name],
                                            options: [.sortedKeys])
    FileHandle.standardOutput.write(output)
}

do { try run() } catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
