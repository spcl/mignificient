#ifndef __MIGNIFICIENT_EXECUTOR_IOX2_CONFIG_HPP__
#define __MIGNIFICIENT_EXECUTOR_IOX2_CONFIG_HPP__

#include <stdexcept>
#include <string>

#include <iox2/iceoryx2.hpp>

namespace mignificient {

  // Env var with a client's iceoryx2 root directory, set by the orchestrator for the
  // gpuless server and the executor (gpuless/src/iox2_config.hpp reads the same one).
  constexpr const char* IOX2_ROOT_ENV = "MIGNIFICIENT_IOX2_ROOT";

  // The global iceoryx2 config with global.root-path = root; root == nullptr keeps it unchanged.
  // With the file-backed iceoryx2 (external/iceoryx2-file-backend.patch) all of a client's
  // iceoryx2 state lives under its root.
  inline iox2::Config iox2_config(const char* root)
  {
    auto cfg = iox2::Config::global_config().to_owned();
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
