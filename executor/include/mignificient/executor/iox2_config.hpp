#ifndef __MIGNIFICIENT_EXECUTOR_IOX2_CONFIG_HPP__
#define __MIGNIFICIENT_EXECUTOR_IOX2_CONFIG_HPP__

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <iox2/iceoryx2.hpp>

namespace mignificient {

  // Env var with a client's iceoryx2 root directory, set by the orchestrator for the
  // gpuless server and the executor (gpuless/src/iox2_config.hpp reads the same one).
  constexpr const char* IOX2_ROOT_ENV = "MIGNIFICIENT_IOX2_ROOT";
  // Env var with the iceoryx2 config file (orchestrator's ipc.iceoryx2-config);
  constexpr const char* IOX2_CONFIG_ENV = "MIGNIFICIENT_IOX2_CONFIG";

  // The iceoryx2 config from $MIGNIFICIENT_IOX2_CONFIG (else the global one) with global.root-path = root;
  // root == nullptr keeps it unchanged.
  // With the file-backed iceoryx2 (external/iceoryx2-file-backend.patch) all of a client's
  // iceoryx2 state lives under its root.
  inline iox2::Config iox2_config(const char* root)
  {
    auto cfg = iox2::Config::global_config().to_owned();
    if(const char* file = std::getenv(IOX2_CONFIG_ENV)) {
      auto str = iox2::bb::StaticString<iox2::bb::platform::IOX2_MAX_PATH_LENGTH>::from_utf8_null_terminated_unchecked(file);
      if(!str.has_value()) {
        throw std::runtime_error(std::string{"iceoryx2 config path too long: "} + file);
      }
      auto path = iox2::bb::FilePath::create(str.value());
      if(!path.has_value()) {
        throw std::runtime_error(std::string{"invalid iceoryx2 config path: "} + file);
      }
      auto loaded = iox2::Config::from_file(path.value());
      if(!loaded.has_value()) {
        throw std::runtime_error(std::string{"cannot load iceoryx2 config "} + file);
      }
      cfg = std::move(loaded.value());
    }
    if(root) {
      auto str = iox2::bb::StaticString<iox2::bb::platform::IOX2_MAX_PATH_LENGTH>::from_utf8_null_terminated_unchecked(root);
      if(!str.has_value()) {
        throw std::runtime_error(std::string{"iceoryx2 root path too long: "} + root);
      }
      auto path = iox2::bb::Path::create(str.value());
      if(!path.has_value()) {
        throw std::runtime_error(std::string{"invalid iceoryx2 root path: "} + root);
      }
      cfg.global().set_root_path(path.value());
    }
    return cfg;
  }

}

#endif
