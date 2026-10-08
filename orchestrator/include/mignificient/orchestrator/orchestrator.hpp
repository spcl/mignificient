#ifndef __MIGNIFICIENT_ORCHESTRATOR_ORCHESTRATOR_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_ORCHESTRATOR_HPP__

#include <mignificient/orchestrator/users.hpp>
#include <mignificient/ipc/config.hpp>
#include <mignificient/ipc/types.hpp>
#include <optional>
#include <queue>
#include <unordered_map>

#include <iceoryx_hoofs/posix_wrapper/signal_handler.hpp>
#include <iceoryx_posh/popo/listener.hpp>
#include <iceoryx_posh/popo/user_trigger.hpp>
#include <json/value.h>

#include <mignificient/orchestrator/client.hpp>
#include <mignificient/orchestrator/container_worker.hpp>
#include <mignificient/orchestrator/device.hpp>
#include <mignificient/orchestrator/invocation.hpp>
#include <mignificient/orchestrator/http.hpp>
#include <unordered_set>

#ifdef MIGNIFICIENT_WITH_ICEORYX2
#include <iox2/iceoryx2.hpp>
#endif

namespace mignificient { namespace orchestrator {

  struct Orchestrator {

    static void init(const Json::Value& config);
    Orchestrator(const Json::Value& config, const std::string& device_db_path);

    Client* client(int id);
    void add_client();

    void run();
    void event_loop();

    // Get IPC configuration (for use by Client and other components)
    static const ipc::IPCConfig& ipc_config() { return _ipc_config; }
    static ipc::IPCBackend ipc_backend() { return _ipc_config.backend; }

  private:

    int _client_id = 0;
    int _invoc_id = 0;
    int _timeout_check_interval_ms = 100;

    void _check_timeouts();
    void _check_oom();
    // Drains container starts and tears down clients that failed to start.
    void _check_startup();
    // Drops the client's IPC event attachments before it is killed.
    void _detach_client(Client* client);
    void _handle_admin_request(AdminRequest&& req);

    std::unordered_map<int, Client> clients;

    const void* last_message;

    static bool _quit;
    static iox::popo::WaitSet<>* _waitset_ptr;
    static std::shared_ptr<HTTPServer> _http_server;
    static ipc::IPCConfig _ipc_config;
    static void _sigHandler(int sig);

    std::optional<iox::posix::SignalGuard> sigint;
    std::optional<iox::posix::SignalGuard> sigterm;

    std::unordered_map<int, Context> _client_contexts;
    std::unordered_map<int, Context> _server_contexts;
    Context _http_context;

    // executor.startup-timeout-ms: executor and gpuless server must register within it.
    std::chrono::milliseconds _startup_timeout;
    ContainerWorker _container_worker;

    GPUManager _gpu_manager;

    // iceoryx1: V1 HTTP handler (used as WaitSet callback)
    std::optional<iox::popo::WaitSet<>> _waitset;
    std::optional<HTTPTriggerV1> _http_trigger_v1;

    static void _handle_http_v1(iox::popo::UserTrigger*, Orchestrator* this_ptr);

    void _event_loop_v1();

#ifdef MIGNIFICIENT_WITH_ICEORYX2

    std::optional<iox2::WaitSet<iox2::ServiceType::Ipc>> _waitset_v2;

    std::optional<HTTPTriggerV2> _http_trigger_v2;

    std::map<
      iox2::WaitSetAttachmentId<iox2::ServiceType::Ipc>,
      std::tuple<Client*, bool>
    > _waitset_mappings_v2;

    std::optional<iox2::WaitSetGuard<iox2::ServiceType::Ipc>> _timeout_interval_guard;

    void _event_loop_v2();
    void _handle_http_v2();
#endif

    // Last member, destroyed first: clients hold guards attached to the waitsets above.
    Users _users;
  };

}}

#endif
