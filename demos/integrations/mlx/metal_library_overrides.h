#pragma once

#include <Metal/Metal.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace crosstl::metal_overrides {

struct Configuration {
  std::filesystem::path root;
  std::filesystem::path trace;
  std::unordered_set<std::string> libraries;

  Configuration() {
    const char* directory = std::getenv("CROSTL_METAL_LIBRARY_OVERRIDES");
    const char* log = std::getenv("CROSTL_METAL_LIBRARY_TRACE");
    if (!directory && !log) {
      return;
    }
    if (!directory || !log) {
      throw std::runtime_error("Metal overrides require a directory and trace.");
    }
    root = directory;
    trace = log;
    if (!root.is_absolute() || !trace.is_absolute()) {
      throw std::runtime_error("Metal override paths must be absolute.");
    }
    std::ifstream manifest(root / "libraries.txt");
    if (!manifest) {
      throw std::runtime_error("Cannot read required Metal library manifest.");
    }
    std::string name;
    while (std::getline(manifest, name)) {
      if (name.empty() ||
          name.find_first_not_of(
              "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") !=
              std::string::npos ||
          !libraries.insert(name).second) {
        throw std::runtime_error("Invalid or duplicate Metal library name.");
      }
    }
    if (!manifest.eof() || libraries.empty()) {
      throw std::runtime_error("Metal override manifest is empty or unreadable.");
    }
  }
};

inline const Configuration& configuration() {
  static const Configuration value;
  return value;
}

inline void record(const std::string& event) {
  static std::mutex mutex;
  std::lock_guard lock(mutex);
  std::ofstream output(configuration().trace, std::ios::app);
  output << event << '\n';
  output.flush();
  if (!output) {
    throw std::runtime_error("Cannot write Metal override execution trace.");
  }
}

inline NS::SharedPtr<MTL::Library> load_library(
    MTL::Device* device, const std::string& name) {
  const auto& config = configuration();
  if (!config.libraries.contains(name)) {
    return {};
  }
  static std::mutex mutex;
  static std::unordered_map<
      MTL::Device*, std::unordered_map<std::string, NS::SharedPtr<MTL::Library>>>
      libraries;
  std::lock_guard lock(mutex);
  auto& device_libraries = libraries[device];
  if (auto found = device_libraries.find(name); found != device_libraries.end()) {
    return found->second;
  }
  const auto path = config.root / (name + ".metallib");
  NS::Error* error = nullptr;
  auto library = NS::TransferPtr(device->newLibrary(
      NS::String::string(path.c_str(), NS::UTF8StringEncoding), &error));
  if (!library) {
    throw std::runtime_error(
        "Cannot load required translated Metal library " + name + ": " +
        (error ? error->localizedDescription()->utf8String() : "unknown error"));
  }
  const auto label = "crosstl:" + name;
  library->setLabel(NS::String::string(label.c_str(), NS::UTF8StringEncoding));
  record("library\t" + name);
  device_libraries.emplace(name, library);
  return library;
}

inline std::mutex pipeline_mutex;
inline std::unordered_map<const MTL::ComputePipelineState*, std::string> pipelines;

inline MTL::ComputePipelineState* register_pipeline(
    MTL::ComputePipelineState* pipeline,
    MTL::Library* library,
    const std::string& entry) {
  auto label = library->label();
  std::lock_guard lock(pipeline_mutex);
  if (label && std::string(label->utf8String()).starts_with("crosstl:")) {
    pipelines[pipeline] = entry;
  } else {
    pipelines.erase(pipeline);
  }
  return pipeline;
}

inline std::string pipeline_entry(const MTL::ComputePipelineState* pipeline) {
  std::lock_guard lock(pipeline_mutex);
  auto entry = pipelines.find(pipeline);
  return entry == pipelines.end() ? std::string{} : entry->second;
}

inline void dispatch(
    const std::string& entry,
    const char* kind,
    MTL::Size grid,
    MTL::Size group) {
  if (entry.empty()) {
    return;
  }
  auto size = [](MTL::Size value) {
    return std::to_string(value.width) + "," + std::to_string(value.height) +
        "," + std::to_string(value.depth);
  };
  record("dispatch\t" + entry + "\t" + kind + "\t" + size(grid) + "\t" +
         size(group));
}

} // namespace crosstl::metal_overrides
