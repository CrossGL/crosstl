import Foundation
import Metal

struct Request: Decodable {
    let buffers: [String]
    let allocationIds: [Int]?
    let workgroupCount: [Int]
    let workgroupSize: [Int]
    let simdWidth: Int
}

enum ExecutionError: Error {
    case unavailable(String)
}

func run() throws {
    let args = CommandLine.arguments
    guard args.count == 5 else {
        throw ExecutionError.unavailable("Expected metallib, entry, request and output paths")
    }
    let request = try JSONDecoder().decode(Request.self, from: Data(contentsOf:
        URL(fileURLWithPath: args[3])))
    guard request.workgroupCount.count == 3, request.workgroupSize.count == 3,
          request.workgroupCount.allSatisfy({ $0 > 0 }),
          request.workgroupSize.allSatisfy({ $0 > 0 }), !request.buffers.isEmpty,
          let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw ExecutionError.unavailable("Invalid dispatch geometry or unavailable Metal device")
    }
    let library = try device.makeLibrary(URL: URL(fileURLWithPath: args[1]))
    guard let function = library.makeFunction(name: args[2]) else {
        throw ExecutionError.unavailable("Kernel entry point is missing")
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    guard pipeline.threadExecutionWidth == request.simdWidth,
          request.workgroupSize.reduce(1, *) <= pipeline.maxTotalThreadsPerThreadgroup else {
        throw ExecutionError.unavailable("Unsupported SIMD width or workgroup size")
    }
    let allocationIds = request.allocationIds ?? Array(request.buffers.indices)
    guard allocationIds.count == request.buffers.count else {
        throw ExecutionError.unavailable("Allocation identity count does not match bindings")
    }
    var allocations: [Int: MTLBuffer] = [:]
    var initialPayloads: [Int: Data] = [:]
    var buffers: [MTLBuffer] = []
    for (index, filename) in request.buffers.enumerated() {
        let bytes = try Data(contentsOf: URL(fileURLWithPath: filename))
        guard !bytes.isEmpty else { throw ExecutionError.unavailable("Empty buffer") }
        let identity = allocationIds[index]
        if let allocation = allocations[identity] {
            guard initialPayloads[identity] == bytes else {
                throw ExecutionError.unavailable("Conflicting shared-allocation payloads")
            }
            buffers.append(allocation)
            continue
        }
        let allocation = bytes.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        }
        guard let allocation else { throw ExecutionError.unavailable("Allocation failed") }
        allocations[identity] = allocation
        initialPayloads[identity] = bytes
        buffers.append(allocation)
    }
    guard let command = queue.makeCommandBuffer(),
          let encoder = command.makeComputeCommandEncoder() else {
        throw ExecutionError.unavailable("Command allocation failed")
    }
    encoder.setComputePipelineState(pipeline)
    for (index, buffer) in buffers.enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
    }
    let groups = request.workgroupCount
    let threads = request.workgroupSize
    encoder.dispatchThreadgroups(MTLSize(width: groups[0], height: groups[1], depth: groups[2]),
        threadsPerThreadgroup: MTLSize(width: threads[0], height: threads[1], depth: threads[2]))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
        throw ExecutionError.unavailable("Dispatch failed: \(String(describing: command.error))")
    }
    let output = URL(fileURLWithPath: args[4])
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    for (index, buffer) in buffers.enumerated() {
        try Data(bytes: buffer.contents(), count: buffer.length)
            .write(to: output.appendingPathComponent("buffer-\(index).bin"))
    }
    let result = try JSONSerialization.data(withJSONObject: ["device": device.name,
        "simdWidth": pipeline.threadExecutionWidth, "bufferBytes": buffers.map { $0.length },
        "allocationIds": allocationIds, "uniqueAllocations": allocations.count],
        options: [.sortedKeys])
    FileHandle.standardOutput.write(result)
}

do { try run() } catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
