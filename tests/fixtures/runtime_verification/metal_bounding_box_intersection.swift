// Traces rays through one bounding box whose intersection function is a generated Metal intersection function.
//
// Usage: swift metal_bounding_box_intersection.swift SOURCE FUNCTION OUTPUT_FLOATS TMIN:TMAX [TMIN:TMAX ...]
//
// SOURCE is Metal source that defines FUNCTION as an [[intersection(bounding_box)]] function. The harness appends a
// kernel that traces one ray per TMIN:TMAX pair from (0, 0, -5) along +z through the box (-1, -1, -1)..(1, 1, 1),
// whose intersection function table holds FUNCTION, and binds a zeroed buffer of OUTPUT_FLOATS floats to the
// function's [[buffer(0)]] when OUTPUT_FLOATS is positive. It prints one JSON object: "hits" holds, per ray, whether
// the intersector returned a bounding-box hit and its distance; "output" holds the buffer's floats.

import Foundation
import Metal

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

let arguments = CommandLine.arguments
guard arguments.count >= 5, let outputFloats = Int(arguments[3]) else {
    fail("usage: metal_bounding_box_intersection.swift SOURCE FUNCTION OUTPUT_FLOATS TMIN:TMAX ...")
}
let functionName = arguments[2]
var intervals: [Float] = []
for pair in arguments[4...] {
    let parts = pair.split(separator: ":").compactMap { Float($0) }
    guard parts.count == 2 else { fail("bad interval \(pair)") }
    intervals += parts
}
let rayCount = intervals.count / 2

guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
guard device.supportsRaytracing else { fail("device does not support ray tracing") }
guard let generated = try? String(contentsOfFile: arguments[1], encoding: .utf8) else {
    fail("cannot read \(arguments[1])")
}

let kernelSource = """

kernel void crossgl_trace_rays(
    primitive_acceleration_structure crossgl_scene [[buffer(0)]],
    metal::raytracing::intersection_function_table<> crossgl_table [[buffer(1)]],
    device float* crossgl_hits [[buffer(2)]],
    constant float2* crossgl_intervals [[buffer(3)]],
    uint crossgl_index [[thread_position_in_grid]]) {
    metal::raytracing::ray crossgl_ray;
    crossgl_ray.origin = float3(0.0, 0.0, -5.0);
    crossgl_ray.direction = float3(0.0, 0.0, 1.0);
    crossgl_ray.min_distance = crossgl_intervals[crossgl_index].x;
    crossgl_ray.max_distance = crossgl_intervals[crossgl_index].y;
    metal::raytracing::intersector<> crossgl_intersector;
    metal::raytracing::intersection_result<> crossgl_result =
        crossgl_intersector.intersect(crossgl_ray, crossgl_scene, crossgl_table);
    bool crossgl_hit = crossgl_result.type == metal::raytracing::intersection_type::bounding_box;
    crossgl_hits[2 * crossgl_index] = crossgl_hit ? 1.0 : 0.0;
    crossgl_hits[2 * crossgl_index + 1] = crossgl_hit ? crossgl_result.distance : -1.0;
}
"""

let library: MTLLibrary
do {
    let options = MTLCompileOptions()
    library = try device.makeLibrary(source: generated + kernelSource, options: options)
} catch {
    fail("compile failed: \(error)")
}
guard let intersectionFunction = library.makeFunction(name: functionName) else {
    fail("no function \(functionName)")
}
guard let kernel = library.makeFunction(name: "crossgl_trace_rays") else { fail("no kernel") }

let linked = MTLLinkedFunctions()
linked.functions = [intersectionFunction]
let pipelineDescriptor = MTLComputePipelineDescriptor()
pipelineDescriptor.computeFunction = kernel
pipelineDescriptor.linkedFunctions = linked
let pipeline: MTLComputePipelineState
do {
    pipeline = try device.makeComputePipelineState(
        descriptor: pipelineDescriptor, options: [], reflection: nil)
} catch {
    fail("pipeline failed: \(error)")
}

let tableDescriptor = MTLIntersectionFunctionTableDescriptor()
tableDescriptor.functionCount = 1
guard let table = pipeline.makeIntersectionFunctionTable(descriptor: tableDescriptor),
    let handle = pipeline.functionHandle(function: intersectionFunction)
else { fail("no intersection function table") }
table.setFunction(handle, index: 0)

let outputBuffer = device.makeBuffer(
    length: max(outputFloats, 1) * MemoryLayout<Float>.stride, options: .storageModeShared)!
memset(outputBuffer.contents(), 0, outputBuffer.length)
if outputFloats > 0 {
    table.setBuffer(outputBuffer, offset: 0, index: 0)
}

// MTLAxisAlignedBoundingBox is two packed float3 values.
let box: [Float] = [-1, -1, -1, 1, 1, 1]
let boxBuffer = device.makeBuffer(
    bytes: box, length: box.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
let geometry = MTLAccelerationStructureBoundingBoxGeometryDescriptor()
geometry.boundingBoxBuffer = boxBuffer
geometry.boundingBoxCount = 1
geometry.boundingBoxStride = 24
geometry.intersectionFunctionTableOffset = 0
let structureDescriptor = MTLPrimitiveAccelerationStructureDescriptor()
structureDescriptor.geometryDescriptors = [geometry]
let sizes = device.accelerationStructureSizes(descriptor: structureDescriptor)
guard let scene = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
    let scratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 16), options: .storageModePrivate),
    let queue = device.makeCommandQueue()
else { fail("cannot allocate the acceleration structure") }

let buildCommands = queue.makeCommandBuffer()!
let builder = buildCommands.makeAccelerationStructureCommandEncoder()!
builder.build(
    accelerationStructure: scene, descriptor: structureDescriptor, scratchBuffer: scratch, scratchBufferOffset: 0)
builder.endEncoding()
buildCommands.commit()
buildCommands.waitUntilCompleted()

let hitsBuffer = device.makeBuffer(
    length: 2 * rayCount * MemoryLayout<Float>.stride, options: .storageModeShared)!
let intervalBuffer = device.makeBuffer(
    bytes: intervals, length: intervals.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
let commands = queue.makeCommandBuffer()!
let encoder = commands.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)
encoder.setAccelerationStructure(scene, bufferIndex: 0)
encoder.setIntersectionFunctionTable(table, bufferIndex: 1)
encoder.setBuffer(hitsBuffer, offset: 0, index: 2)
encoder.setBuffer(intervalBuffer, offset: 0, index: 3)
encoder.useResource(scene, usage: .read)
encoder.useResource(boxBuffer, usage: .read)
encoder.useResource(outputBuffer, usage: [.read, .write])
encoder.dispatchThreads(
    MTLSize(width: rayCount, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
encoder.endEncoding()
commands.commit()
commands.waitUntilCompleted()
if let error = commands.error { fail("dispatch failed: \(error)") }

let hitValues = Array(UnsafeBufferPointer(
    start: hitsBuffer.contents().bindMemory(to: Float.self, capacity: 2 * rayCount), count: 2 * rayCount))
let outputValues = Array(UnsafeBufferPointer(
    start: outputBuffer.contents().bindMemory(to: Float.self, capacity: outputFloats), count: outputFloats))
var hits: [[String: Any]] = []
for index in 0..<rayCount {
    hits.append(["hit": hitValues[2 * index] == 1.0, "distance": Double(hitValues[2 * index + 1])])
}
let report: [String: Any] = ["hits": hits, "output": outputValues.map { Double($0) }]
let data = try! JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
