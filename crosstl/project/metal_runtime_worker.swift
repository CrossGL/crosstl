import Foundation
import Metal

// An upload carries its bytes inline as base64 `data`, or, in a served
// request, as `size` bytes of the payload that follows the request's line.
struct Upload: Decodable {
  let offset: Int
  let data: Data?
  let size: Int?
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
  let binaryBytes: Int?
}
// The framing of a served request, read before the rest of its JSON.
struct Frame: Decodable {
  let binaryBytes: Int?
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

// A long-lived worker keeps its device, queue, libraries and pipelines; every
// request still gets fresh zeroed allocations and its own command buffer.
@available(macOS 13.0, *)
final class Session {
  // Bounds the libraries and pipelines one worker retains.
  static let cacheLimit = 256
  let device: MTLDevice
  let queue: MTLCommandQueue
  private var libraries: [String: MTLLibrary] = [:]
  private var pipelines: [String: (MTLComputePipelineState, MTLComputePipelineReflection)] = [:]

  init(device: MTLDevice) throws {
    guard let queue = device.makeCommandQueue() else {
      throw RuntimeFailure("Metal command setup failed.")
    }
    self.device = device
    self.queue = queue
  }

  // A library is reused only while its file keeps the same identity.
  func library(_ path: String) throws -> (MTLLibrary, String) {
    var info = stat()
    try require(stat(path, &info) == 0, "Metal library is unavailable.")
    let modified = info.st_mtimespec
    let key = "\(info.st_dev):\(info.st_ino):\(info.st_size):\(modified.tv_sec):\(modified.tv_nsec):\(path)"
    if let cached = libraries[key] { return (cached, key) }
    let library = try device.makeLibrary(URL: URL(fileURLWithPath: path))
    if libraries.count >= Session.cacheLimit {
      libraries.removeAll()
      pipelines.removeAll()
    }
    libraries[key] = library
    return (library, key)
  }

  func pipeline(_ request: Request) throws -> (
    MTLComputePipelineState, MTLComputePipelineReflection
  ) {
    let (library, libraryKey) = try library(request.library)
    var key = libraryKey + "\u{0}" + request.entryPoint
    for constant in request.constants {
      key += "\u{0}\(constant.id):\(constant.dtype):\(constant.data.base64EncodedString())"
    }
    if let cached = pipelines[key] { return cached }
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
        constant.id >= 0 && constantIds.insert(constant.id).inserted
          && constant.data.count == length,
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
    if pipelines.count >= Session.cacheLimit { pipelines.removeAll() }
    pipelines[key] = (pipeline, reflection)
    return (pipeline, reflection)
  }
}

func copyBytes(_ data: Data, from start: Int, count: Int, to destination: UnsafeMutableRawPointer) {
  data.withUnsafeBytes { bytes in
    if count > 0, let base = bytes.baseAddress {
      destination.copyMemory(from: base.advanced(by: start), byteCount: count)
    }
  }
}

// Executes one request; `payload` holds the served bytes its uploads name.
// Returns the reply and each readback's bytes in binding order.
@available(macOS 13.0, *)
func execute(_ request: Request, _ session: Session, _ payload: Data?) throws -> (
  [String: Any], [(String, Data)]
) {
  let device = session.device
  let (pipeline, reflection) = try session.pipeline(request)
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
  var cursor = 0
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
      let source: Data
      let start: Int
      let count: Int
      if let data = upload.data {
        try require(upload.size == nil, "Upload has inline and served bytes.")
        (source, start, count) = (data, 0, data.count)
      } else if let payload = payload, let size = upload.size {
        try require(size >= 0 && size <= payload.count - cursor, "Upload exceeds its payload.")
        (source, start, count) = (payload, cursor, size)
        cursor += size
      } else {
        throw RuntimeFailure("Upload has no bytes.")
      }
      try require(
        upload.offset >= 0 && upload.offset <= allocation.length
          && count <= allocation.length - upload.offset, "Upload exceeds allocation.")
      copyBytes(source, from: start, count: count, to: buffer.contents().advanced(by: upload.offset))
    }
    allocations[allocation.id] = buffer
  }
  try require(cursor == (payload?.count ?? 0), "Request payload has unclaimed bytes.")
  guard let command = session.queue.makeCommandBuffer(),
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
  var outputs: [(String, Data)] = []
  var names = Set<String>()
  for binding in request.buffers {
    if let name = binding.output, let allocation = allocations[binding.allocation] {
      try require(names.insert(name).inserted, "Duplicate readback name.")
      outputs.append(
        (name, Data(bytes: allocation.contents().advanced(by: binding.offset), count: binding.length))
      )
    }
  }
  return (["device": device.name, "threadExecutionWidth": pipeline.threadExecutionWidth], outputs)
}

func inlineOutputs(_ reply: [String: Any], _ outputs: [(String, Data)]) -> [String: Any] {
  var result = reply
  var encoded: [String: String] = [:]
  for (name, data) in outputs { encoded[name] = data.base64EncodedString() }
  result["outputs"] = encoded
  return result
}

// Reads the bytes that follow a served request's line, or nil at a short read.
func readPayload(_ count: Int) -> Data? {
  var data = Data(count: count)
  let read = data.withUnsafeMutableBytes { bytes in
    count == 0 ? 0 : fread(bytes.baseAddress, 1, count, stdin)
  }
  return read == count ? data : nil
}

func writeReply(_ status: String, _ reply: [String: Any], _ payload: [Data]?) throws {
  let data = try JSONSerialization.data(withJSONObject: reply, options: [.sortedKeys])
  let size = payload?.reduce(0) { $0 + $1.count }
  let prefix = size.map { "\(status)\($0):" } ?? status
  FileHandle.standardOutput.write(Data(prefix.utf8) + data + Data([10]))
  for chunk in payload ?? [] where !chunk.isEmpty {
    FileHandle.standardOutput.write(chunk)
  }
}

// Serve mode reads one JSON request per line until end of input. Each reply
// is one line: "+" and the response, or "-" and {"failure", "error"?}. A
// request whose JSON states `binaryBytes` is followed by that many bytes, which
// its uploads take in order; its reply is "+<size>:" and the response, whose
// `outputs` name each readback's [offset, length] in the <size> bytes that
// follow the line. A request whose framing cannot be read ends serving, since
// the bytes after it cannot be skipped.
@available(macOS 13.0, *)
func serve(_ session: Session) throws {
  setvbuf(stdin, nil, _IOFBF, 1 << 20)
  while let line = readLine(strippingNewline: true) {
    let text = Data(line.utf8)
    guard let frame = try? JSONDecoder().decode(Frame.self, from: text),
      (frame.binaryBytes ?? 0) >= 0,
      let payload = readPayload(frame.binaryBytes ?? 0)
    else {
      try writeReply("-", ["failure": "Unreadable request framing.", "fatal": true], nil)
      return
    }
    let served = frame.binaryBytes != nil
    do {
      let request = try JSONDecoder().decode(Request.self, from: text)
      let (reply, outputs) = try autoreleasepool {
        try execute(request, session, served ? payload : nil)
      }
      if !served {
        try writeReply("+", inlineOutputs(reply, outputs), nil)
        continue
      }
      var extents: [String: [Int]] = [:]
      var offset = 0
      for (name, data) in outputs {
        extents[name] = [offset, data.count]
        offset += data.count
      }
      var result = reply
      result["outputs"] = extents
      try writeReply("+", result, outputs.map { $0.1 })
    } catch let failure as DispatchLimitFailure {
      try writeReply("-", ["failure": failure.description, "error": failure.details], nil)
    } catch {
      try writeReply("-", ["failure": "\(error)"], nil)
    }
  }
}

@available(macOS 13.0, *)
func run() throws {
  let device = MTLCreateSystemDefaultDevice()
  let mode = CommandLine.arguments.dropFirst().first
  if mode == "--probe" {
    try emit(["available": device != nil, "device": device?.name ?? ""])
    return
  }
  guard let device = device else { throw RuntimeFailure("No Metal device is available.") }
  let session = try Session(device: device)
  if mode == "--serve" {
    try serve(session)
    return
  }
  let request = try JSONDecoder().decode(
    Request.self, from: FileHandle.standardInput.readDataToEndOfFile())
  let (reply, outputs) = try execute(request, session, nil)
  try emit(inlineOutputs(reply, outputs))
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
