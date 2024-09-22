
#include <mignificient/orchestrator/device.hpp>

#include <fstream>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace mignificient { namespace orchestrator {

  SharingModel sharing_model(const std::string& val)
  {
    static std::map<std::string, SharingModel> mapping = {
      {"sequential", SharingModel::SEQUENTIAL},
      {"overlap_cpu", SharingModel::OVERLAP_CPU},
      {"overlap_cpu_memcpy", SharingModel::OVERLAP_CPU_MEMCPY},
      {"full_overlap", SharingModel::FULL_OVERLAP}
    };

    auto it = mapping.find(val);
    if(it != mapping.end()) {
      return (*it).second;
    } else {
      throw std::runtime_error{"Wrong SharingModel value"};
    }
  }

  GPUDevice::GPUDevice(const Json::Value& gpu, SharingModel sharing_model):
    _uuid(gpu["uuid"].asString()),
    _memory(gpu["memory"].asFloat())
  {
    const Json::Value& instances = gpu["instances"];
    for (const auto& instance : instances) {
      _mig_instances.emplace_back(
        instance["uuid"].asString(),
        instance["memory"].asFloat(),
        instance["instance_size"].asString(),
        sharing_model
      );
    }

    // Artificially add a full device
    if(_mig_instances.size() == 0) {
      _mig_instances.emplace_back(
        _uuid,
        _memory,
        "7g",
        sharing_model
      );
    }
  }

  GPUManager::GPUManager(const std::string& devices_data_path, SharingModel sharing_model)
  {
    std::ifstream device_file{devices_data_path};

    Json::CharReaderBuilder builder;
    JSONCPP_STRING errs;
    Json::Value device_data;
    if (!Json::parseFromStream(builder, device_file, &device_data, &errs)) {
      spdlog::error("Failed to parse JSON config! {}", errs);
      throw std::runtime_error("Failed to parse JSON");
    }

    const Json::Value& gpus = device_data["gpus"];
    for (const auto& gpu : gpus) {
      _devices.emplace_back(gpu, sharing_model);
    }
  }

}}
