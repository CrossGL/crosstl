import Foundation
import Metal

struct Upload: Decodable {
  let offset: Int
  let data: Data
}
struct Allocation: Decodable {
  let id: String
  let length: Int
  let uploads: [Upload]
}
struct Buffer: Decodable {
  let index: Int
  let allocation: String
  let offset: Int
  let length: Int
  let alignment: Int
  let output: String?
}
struct Constant: Decodable {
  let id: Int
  let dtype: String
  let data: Data
}
struct Request: Decodable {
  let library: String
  let entryPoint: String
  let workgroupCount: [Int]
  let workgroupSize: [Int]
  let threadGridSize: [Int]?
  let allocations: [Allocation]
  let buffers: [Buffer]
  let constants: [Constant]
}
struct RuntimeFailure: Error, CustomStringConvertible {
  let description: String
  init(_ message: String) { description = message }
}
struct DispatchLimitFailure: Error, CustomStringConvertible {
  let field: String
  let axis: Int?
  let requested: Int
  let maximum: Int
  let counts: [Int]
  let size: [Int]
  let maximumSize: [Int]
  let maximumInvocations: Int

  var description: String { "Unsupported group dimension or invocation count: \(field)." }
  var details: [String: Any] {
    var result: [String: Any] = [
      "target": "metal", "reasonKind": "dispatch-limit-exceeded", "field": field,
      "requested": requested, "maximum": maximum,
      "workgroupCount": counts, "workgroupSize": size,
      "limits": [
        "maxWorkgroupCount": [Int](repeating: Int(UInt32.max), count: 3),
        "maxWorkgroupSize": maximumSize, "maxWorkgroupInvocations": maximumInvocations,
      ],
    ]
    if let axis = axis { result["axis"] = axis }
    return result
  }
}

func require(_ condition: Bool, _ message: String) throws {
  if !condition { throw RuntimeFailure(message) }
}

func emit(_ value: [String: Any]) throws {
  let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
  FileHandle.standardOutput.write(data)
  FileHandle.standardOutput.write(Data([10]))
}

@available(macOS 13.0, *)
func run() throws {
  let device = MTLCreateSystemDefaultDevice()
  if CommandLine.arguments.dropFirst().first == "--probe" {
    try emit(["available": device != nil, "device": device?.name ?? ""])
    return
  }
  guard let device = device else { throw RuntimeFailure("No Metal device is available.") }
  let request = try JSONDecoder().decode(
    Request.self, from: FileHandle.standardInput.readDataToEndOfFile())
  let library = try device.makeLibrary(URL: URL(fileURLWithPath: request.library))
  let constants = MTLFunctionConstantValues()
  var constantIds = Set<Int>()
  for constant in request.constants {
    let type: MTLDataType
    let length: Int
    switch constant.dtype {
    case "bool":
      type = .bool
      length = 1
    case "float32":
      type = .float
      length = 4
    case "int32":
      type = .int
      length = 4
    case "uint32":
      type = .uint
      length = 4
    default: throw RuntimeFailure("Unsupported function constant type.")
    }
    try require(
      constant.id >= 0 && constantIds.insert(constant.id).inserted && constant.data.count == length,
      "Invalid function constant.")
    constant.data.withUnsafeBytes { bytes in
      constants.setConstantValue(bytes.baseAddress!, type: type, index: constant.id)
    }
  }
  let function: MTLFunction
  if request.constants.isEmpty {
    guard let selected = library.makeFunction(name: request.entryPoint) else {
      throw RuntimeFailure("Metal entry point is missing.")
    }
    function = selected
  } else {
    function = try library.makeFunction(name: request.entryPoint, constantValues: constants)
  }
  var reflection: MTLComputePipelineReflection?
  let pipeline = try device.makeComputePipelineState(
    function: function, options: [.bindingInfo, .bufferTypeInfo], reflection: &reflection)
  guard let reflection = reflection else {
    throw RuntimeFailure("Metal binding reflection is unavailable.")
  }
  for argument in reflection.bindings where argument.isUsed && argument.isArgument {
    guard argument.type == .buffer, let reflected = argument as? MTLBufferBinding else {
      throw RuntimeFailure("Metal runtime supports buffer arguments only.")
    }
    let matches = request.buffers.filter { $0.index == argument.index }
    try require(matches.count == 1, "Required Metal buffer binding is missing or ambiguous.")
    let binding = matches[0]
    try require(
      reflected.bufferAlignment > 0 && binding.offset % reflected.bufferAlignment == 0,
      "Buffer offset conflicts with compiled Metal alignment.")
    try require(
      binding.length >= reflected.bufferDataSize,
      "Buffer view is smaller than its compiled Metal element.")
  }
  try require(
    request.workgroupCount.count == 3 && request.workgroupSize.count == 3,
    "Dispatch requires three dimensions.")
  let maximum = device.maxThreadsPerThreadgroup
  let limits = [maximum.width, maximum.height, maximum.depth]
  func validateLimit(_ value: Int, _ maximum: Int, field: String, axis: Int? = nil) throws {
    if value > maximum {
      throw DispatchLimitFailure(
        field: field, axis: axis, requested: value, maximum: maximum,
        counts: request.workgroupCount, size: request.workgroupSize,
        maximumSize: limits, maximumInvocations: pipeline.maxTotalThreadsPerThreadgroup)
    }
  }
  for axis in 0..<3 {
    try require(
      request.workgroupCount[axis] > 0,
      "Invalid group count.")
    try validateLimit(
      request.workgroupCount[axis], Int(UInt32.max), field: "workgroupCount", axis: axis)
    try require(
      request.workgroupSize[axis] > 0,
      "Unsupported group dimension.")
    try validateLimit(request.workgroupSize[axis], limits[axis], field: "workgroupSize", axis: axis)
  }
  var total = 1
  for dimension in request.workgroupSize {
    let product = total.multipliedReportingOverflow(by: dimension)
    try require(!product.overflow, "Group size overflow.")
    total = product.partialValue
  }
  try validateLimit(total, pipeline.maxTotalThreadsPerThreadgroup, field: "workgroupInvocations")
  if let grid = request.threadGridSize {
    try require(grid.count == 3, "Thread grid requires three dimensions.")
    var supportsNonuniform = device.supportsFamily(.apple4)
    if #unavailable(macOS 27.0) {
      supportsNonuniform = supportsNonuniform || device.supportsFamily(.mac2)
    }
    try require(supportsNonuniform, "Device does not support nonuniform threadgroup sizes.")
    for axis in 0..<3 {
      try require(grid[axis] > 0, "Invalid thread grid extent.")
      try validateLimit(grid[axis], Int(UInt32.max), field: "threadGridSize", axis: axis)
      let count = 1 + (grid[axis] - 1) / request.workgroupSize[axis]
      try require(count == request.workgroupCount[axis], "Thread grid conflicts with group count.")
    }
  }
  var allocations: [String: MTLBuffer] = [:]
  for allocation in request.allocations {
    try require(
      allocations[allocation.id] == nil && allocation.length > 0
        && allocation.length <= device.maxBufferLength, "Invalid Metal allocation.")
    guard let buffer = device.makeBuffer(length: allocation.length, options: .storageModeShared)
    else {
      throw RuntimeFailure("Metal buffer allocation failed.")
    }
    memset(buffer.contents(), 0, allocation.length)
    for upload in allocation.uploads {
      try require(
        upload.offset >= 0 && upload.offset <= allocation.length
          && upload.data.count <= allocation.length - upload.offset, "Upload exceeds allocation.")
      upload.data.withUnsafeBytes { bytes in
        if let pointer = bytes.baseAddress {
          buffer.contents().advanced(by: upload.offset).copyMemory(
            from: pointer, byteCount: bytes.count)
        }
      }
    }
    allocations[allocation.id] = buffer
  }
  guard let queue = device.makeCommandQueue(), let command = queue.makeCommandBuffer(),
    let encoder = command.makeComputeCommandEncoder()
  else {
    throw RuntimeFailure("Metal command setup failed.")
  }
  encoder.setComputePipelineState(pipeline)
  var indices = Set<Int>()
  for binding in request.buffers {
    guard let allocation = allocations[binding.allocation] else {
      throw RuntimeFailure("Missing buffer allocation.")
    }
    try require(
      binding.index >= 0 && binding.index < 31 && indices.insert(binding.index).inserted,
      "Invalid buffer index.")
    try require(
      binding.alignment > 0 && binding.offset >= 0 && binding.offset % binding.alignment == 0,
      "Misaligned buffer offset.")
    try require(
      binding.offset <= allocation.length && binding.length > 0
        && binding.length <= allocation.length - binding.offset, "Buffer view exceeds allocation.")
    encoder.setBuffer(allocation, offset: binding.offset, index: binding.index)
  }
  let groups = request.workgroupCount
  let size = request.workgroupSize
  let groupSize = MTLSize(width: size[0], height: size[1], depth: size[2])
  if let grid = request.threadGridSize {
    encoder.dispatchThreads(
      MTLSize(width: grid[0], height: grid[1], depth: grid[2]),
      threadsPerThreadgroup: groupSize)
  } else {
    encoder.dispatchThreadgroups(
      MTLSize(width: groups[0], height: groups[1], depth: groups[2]),
      threadsPerThreadgroup: groupSize)
  }
  encoder.endEncoding()
  command.commit()
  command.waitUntilCompleted()
  if let error = command.error { throw error }
  try require(command.status == .completed, "Metal command did not complete.")
  var outputs: [String: String] = [:]
  for binding in request.buffers {
    if let name = binding.output, let allocation = allocations[binding.allocation] {
      try require(outputs[name] == nil, "Duplicate readback name.")
      outputs[name] = Data(
        bytes: allocation.contents().advanced(by: binding.offset), count: binding.length
      ).base64EncodedString()
    }
  }
  try emit([
    "device": device.name, "outputs": outputs,
    "threadExecutionWidth": pipeline.threadExecutionWidth,
  ])
}

do {
  if #available(macOS 13.0, *) {
    try run()
  } else {
    throw RuntimeFailure("Metal runtime requires macOS 13 or newer.")
  }
} catch {
  if let failure = error as? DispatchLimitFailure {
    try? emit(["error": failure.details])
  }
  FileHandle.standardError.write(Data("\(error)\n".utf8))
  exit(1)
}
