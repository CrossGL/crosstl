import Foundation
import Metal

struct Dispatch: Decodable {
    let library: String
    let entry: String
    let bindings: [Int]
    let workgroupCount: [Int]
    let workgroupSize: [Int]
    let simdWidth: Int
}

struct SequenceRequest: Decodable {
    let allocations: [String]
    let dispatches: [Dispatch]
}

enum SequenceError: Error {
    case invalid(String)
}

func run() throws {
    let args = CommandLine.arguments
    guard args.count == 3 else {
        throw SequenceError.invalid("Expected request and output paths")
    }
    let request = try JSONDecoder().decode(SequenceRequest.self,
        from: Data(contentsOf: URL(fileURLWithPath: args[1])))
    guard !request.allocations.isEmpty, !request.dispatches.isEmpty else {
        throw SequenceError.invalid("Empty dispatch sequence")
    }
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw SequenceError.invalid("Metal device unavailable")
    }
    var pipelines: [MTLComputePipelineState] = []
    for node in request.dispatches {
        guard !node.bindings.isEmpty,
              node.bindings.allSatisfy({ request.allocations.indices.contains($0) }),
              node.workgroupCount.count == 3, node.workgroupSize.count == 3,
              node.workgroupCount.allSatisfy({ $0 > 0 && $0 <= 65535 }),
              node.workgroupSize.allSatisfy({ $0 > 0 && $0 <= 1024 }) else {
            throw SequenceError.invalid("Invalid geometry or allocation binding")
        }
        let library = try device.makeLibrary(URL: URL(fileURLWithPath: node.library))
        guard let function = library.makeFunction(name: node.entry) else {
            throw SequenceError.invalid("Missing entry point")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        guard pipeline.threadExecutionWidth == node.simdWidth,
              node.workgroupSize.reduce(1, *) <= pipeline.maxTotalThreadsPerThreadgroup else {
            throw SequenceError.invalid("Unsupported SIMD width or workgroup size")
        }
        pipelines.append(pipeline)
    }
    var allocations: [MTLBuffer] = []
    for filename in request.allocations {
        let bytes = try Data(contentsOf: URL(fileURLWithPath: filename))
        guard !bytes.isEmpty else { throw SequenceError.invalid("Empty allocation") }
        let buffer = bytes.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        }
        guard let buffer else { throw SequenceError.invalid("Buffer allocation failed") }
        allocations.append(buffer)
    }
    guard let command = queue.makeCommandBuffer() else {
        throw SequenceError.invalid("Command allocation failed")
    }
    // Encoder boundaries synchronize tracked resources without a host transfer.
    for (node, pipeline) in zip(request.dispatches, pipelines) {
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw SequenceError.invalid("Encoder allocation failed")
        }
        encoder.setComputePipelineState(pipeline)
        for (slot, identity) in node.bindings.enumerated() {
            encoder.setBuffer(allocations[identity], offset: 0, index: slot)
        }
        let groups = node.workgroupCount
        let threads = node.workgroupSize
        encoder.dispatchThreadgroups(MTLSize(width: groups[0], height: groups[1], depth: groups[2]),
            threadsPerThreadgroup: MTLSize(width: threads[0], height: threads[1], depth: threads[2]))
        encoder.endEncoding()
    }
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
        throw SequenceError.invalid("Dispatch failed: \(String(describing: command.error))")
    }
    let output = URL(fileURLWithPath: args[2])
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    for (index, buffer) in allocations.enumerated() {
        try Data(bytes: buffer.contents(), count: buffer.length)
            .write(to: output.appendingPathComponent("allocation-\(index).bin"))
    }
    let result = try JSONSerialization.data(withJSONObject: ["device": device.name,
        "dispatches": request.dispatches.count, "readbackPasses": 1,
        "uniqueAllocations": allocations.count, "bindings": request.dispatches.map { $0.bindings }],
        options: [.sortedKeys])
    FileHandle.standardOutput.write(result)
}

do { try run() } catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
