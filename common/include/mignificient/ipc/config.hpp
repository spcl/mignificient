#ifndef MIGNIFICIENT_IPC_CONFIG_HPP
#define MIGNIFICIENT_IPC_CONFIG_HPP

#include <cstddef>
#include <string>
#include <unordered_map>
#include <json/value.h>

namespace mignificient { namespace ipc {

enum class IPCBackend {
    ICEORYX_V1,
    ICEORYX_V2
};

enum class PollingMode {
    WAIT,  // Blocking wait on waitset
    POLL   // Active polling (non-blocking)
};

struct BufferConfig {
    size_t request_size;      // Size of request messages in bytes
    size_t response_size;     // Size of response messages in bytes
    size_t queue_capacity;    // Maximum number of messages in queue
    size_t history_size;      // History size for publisher (0 = no history)

    BufferConfig():
      request_size(1048576),
      response_size(5242880),
      queue_capacity(10),
      history_size(0)
    {}

    BufferConfig(size_t req_size, size_t resp_size, size_t capacity, size_t history = 0):
      request_size(req_size),
      response_size(resp_size),
      queue_capacity(capacity),
      history_size(history)
    {}
};

struct IPCConfig {
    IPCBackend backend;
    std::unordered_map<std::string, BufferConfig> buffer_configs;
    PollingMode polling_mode;
    uint32_t poll_interval_us;

    static IPCBackend convert_ipc_backend(std::string_view value);

    IPCConfig():
      backend(IPCBackend::ICEORYX_V1),
      polling_mode(PollingMode::WAIT),
      poll_interval_us(100)
    {
      // Default buffer configurations
      buffer_configs["orchestrator-executor"] = BufferConfig(1048576, 5242880, 10);
      buffer_configs["orchestrator-gpuless"] = BufferConfig(32, 32, 5);
      buffer_configs["gpuless-executor"] = BufferConfig(52428800, 52428800, 5);
    }

    static IPCConfig from_json(const Json::Value& config);

    static std::string backend_string(IPCBackend backend)
    {
      return backend == IPCBackend::ICEORYX_V1 ? "iceoryx1" : "iceoryx2";
    }

    static std::string polling_mode_string(PollingMode polling_mode)
    {
      return polling_mode == PollingMode::WAIT ? "wait" : "poll";
    }
};

}} // namespace mignificient::ipc

#endif // MIGNIFICIENT_IPC_CONFIG_HPP
