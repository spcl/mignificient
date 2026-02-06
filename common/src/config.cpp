#include <mignificient/ipc/config.hpp>
#include <cstdlib>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace mignificient { namespace ipc {

  IPCBackend IPCConfig::convert_ipc_backend(std::string_view value)
  {
    if (value == "iceoryx1") {
      return IPCBackend::ICEORYX_V1;
    } else if (value == "iceoryx2") {
      return IPCBackend::ICEORYX_V2;
    } else {
      spdlog::error("Invalid IPC backend: {} (must be 'iceoryx1' or 'iceoryx2')", value);
      abort();
    }
  }

  IPCConfig IPCConfig::from_json(const Json::Value& config) {

    IPCConfig ipc_config;

    if (!config.isMember("ipc")) {
      // No IPC config, use defaults
      return ipc_config;
    }

    const Json::Value& ipc = config["ipc"];

    if (ipc.isMember("backend")) {
      std::string backend_str = ipc["backend"].asString();
      ipc_config.backend = convert_ipc_backend(backend_str);
    }

    // Environment variable override for backend
    const char* env_backend = std::getenv("MIGNIFICIENT_IPC_BACKEND");
    if (env_backend != nullptr) {
      std::string backend_str(env_backend);
      ipc_config.backend = convert_ipc_backend(backend_str);
    }

    // Parse buffer configurations
    if (ipc.isMember("buffer-config")) {

      const Json::Value& buffer_config = ipc["buffer-config"];

      for (const auto& component : buffer_config.getMemberNames()) {
        const Json::Value& comp_config = buffer_config[component];

        BufferConfig buf_cfg;

        if (comp_config.isMember("invocation-size")) {
            buf_cfg.request_size = comp_config["invocation-size"].asUInt64();
        } else if (comp_config.isMember("request-size")) {
            buf_cfg.request_size = comp_config["request-size"].asUInt64();
        }

        if (comp_config.isMember("result-size")) {
            buf_cfg.response_size = comp_config["result-size"].asUInt64();
        } else if (comp_config.isMember("response-size")) {
            buf_cfg.response_size = comp_config["response-size"].asUInt64();
        }

        if (comp_config.isMember("queue-capacity")) {
            buf_cfg.queue_capacity = comp_config["queue-capacity"].asUInt64();
        }

        if (comp_config.isMember("history-size")) {
            buf_cfg.history_size = comp_config["history-size"].asUInt64();
        }

        ipc_config.buffer_configs[component] = buf_cfg;
      }
    }

    if (ipc.isMember("polling")) {
      const Json::Value& polling = ipc["polling"];

      if (polling.isMember("mode")) {
        std::string mode_str = polling["mode"].asString();
        if (mode_str == "wait") {
            ipc_config.polling_mode = PollingMode::WAIT;
        } else if (mode_str == "poll") {
            ipc_config.polling_mode = PollingMode::POLL;
        } else {
            throw std::runtime_error("Invalid polling mode: " + mode_str + " (must be 'wait' or 'poll')");
        }
      }

      if (polling.isMember("poll-interval-us")) {
        ipc_config.poll_interval_us = polling["poll-interval-us"].asUInt();
      }
    }

    const char* env_gpuless_req = std::getenv("MIGNIFICIENT_GPULESS_REQUEST_SIZE");
    const char* env_gpuless_resp = std::getenv("MIGNIFICIENT_GPULESS_RESPONSE_SIZE");

    if (env_gpuless_req != nullptr) {
      size_t size = std::stoull(env_gpuless_req);
      if (ipc_config.buffer_configs.count("gpuless-server") > 0) {
        ipc_config.buffer_configs["gpuless-server"].request_size = size;
      }
    }

    if (env_gpuless_resp != nullptr) {
      size_t size = std::stoull(env_gpuless_resp);
      if (ipc_config.buffer_configs.count("gpuless-executor") > 0) {
        ipc_config.buffer_configs["gpuless-executor"].response_size = size;
      }
    }

    return ipc_config;
  }

}} // namespace mignificient::ipc
