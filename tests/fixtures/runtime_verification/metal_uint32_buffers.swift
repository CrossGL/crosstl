import Foundation
import Metal

struct Request: Decodable {
    let inputs: [[UInt32]]
    let outputCount: Int
    let threadCount: Int
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
    guard !request.inputs.isEmpty, request.inputs.allSatisfy({ !$0.isEmpty }),
          request.outputCount > 0, request.threadCount > 0,
          let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        throw ExecutionError.unavailable("Expected nonempty buffers and a Metal device")
    }
    let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
    guard let function = library.makeFunction(name: CommandLine.arguments[2]) else {
        throw ExecutionError.unavailable("Kernel entry point is missing")
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    func buffer(_ values: [UInt32]) throws -> MTLBuffer {
        let result = values.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        }
        guard let result else { throw ExecutionError.unavailable("Allocation failed") }
        return result
    }
    let buffers = try (request.inputs + [[UInt32](repeating: 0xdeadbeef,
        count: request.outputCount)]).map(buffer)
    guard let command = queue.makeCommandBuffer(),
          let encoder = command.makeComputeCommandEncoder() else {
        throw ExecutionError.unavailable("Command allocation failed")
    }
    encoder.setComputePipelineState(pipeline)
    for (index, allocation) in buffers.enumerated() {
        encoder.setBuffer(allocation, offset: 0, index: index)
    }
    encoder.dispatchThreadgroups(MTLSize(width: request.threadCount, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
        throw ExecutionError.unavailable("Dispatch failed: \(String(describing: command.error))")
    }
    let readbacks = buffers.map { allocation in
        Array(UnsafeBufferPointer(
            start: allocation.contents().assumingMemoryBound(to: UInt32.self),
            count: allocation.length / MemoryLayout<UInt32>.stride))
    }
    let output = try JSONSerialization.data(withJSONObject: ["values": readbacks.last!,
        "buffers": readbacks, "device": device.name],
                                            options: [.sortedKeys])
    FileHandle.standardOutput.write(output)
}

do { try run() } catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
