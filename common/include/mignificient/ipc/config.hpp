#ifndef MIGNIFICIENT_IPC_CONFIG_HPP
#define MIGNIFICIENT_IPC_CONFIG_HPP

#include <cstddef>
#include <cstdio>
#include <functional>
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
    // iceoryx2: each client gets its own iceoryx2 root directory under this base (client_root()).
    std::string client_root_base;
    // iceoryx2 config file every process loads (ipc.iceoryx2-config); empty: iceoryx2's default lookup.
    std::string iox2_config_file;
    // The client's event sockets live in its root, and a unix socket path has at most 107 bytes:
    // <root>/iox2_<u128>.event leaves 56 for the root, so 39 for the base.
    static constexpr size_t MAX_CLIENT_ROOT_BASE_LEN = 39;

    static IPCBackend convert_ipc_backend(std::string_view value);

    IPCConfig():
      backend(IPCBackend::ICEORYX_V1),
      polling_mode(PollingMode::WAIT),
      poll_interval_us(100),
      client_root_base("/dev/shm/mignificient")
    {
      // Default buffer configurations
      buffer_configs["orchestrator-executor"] = BufferConfig(1048576, 5242880, 10);
      buffer_configs["orchestrator-gpuless"] = BufferConfig(32, 32, 5);
      buffer_configs["gpuless-executor"] = BufferConfig(52428800, 52428800, 5);
    }

    static IPCConfig from_json(const Json::Value& config);

    // <base>/<16 hex digits of a hash of the client id>. Not the id itself: with user and function
    // names in it, the socket paths would cap the names at a few dozen characters.
    std::string client_root(const std::string& client_id) const
    {
      char name[17];
      snprintf(name, sizeof(name), "%016zx", std::hash<std::string>{}(client_id));
      return client_root_base + "/" + name;
    }

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
