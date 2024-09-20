#ifndef __MIGNIFICIENT_ORCHESTRATOR_ORCHESTRATOR_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_ORCHESTRATOR_HPP__

#include <optional>
#include <queue>
#include <unordered_map>

#include <iceoryx_hoofs/posix_wrapper/signal_handler.hpp>
#include <iceoryx_posh/popo/listener.hpp>
#include <iceoryx_posh/popo/user_trigger.hpp>
#include <json/value.h>

#include <mignificient/orchestrator/client.hpp>
#include <mignificient/orchestrator/invocation.hpp>
#include <mignificient/orchestrator/http.hpp>
#include <unordered_set>

namespace mignificient { namespace orchestrator {

  struct Orchestrator {

    static void init(const Json::Value& config);
    Orchestrator(const Json::Value& value);

    Client* client(int id);
    void add_client();

    void run();
    void event_loop();

  private:

    int _client_id = 0;
    int _invoc_id = 0;

    std::unordered_map<int, Client> clients;

    const void* last_message;
    iox::popo::WaitSet<> _waitset;

    static bool _quit;
    static iox::popo::WaitSet<>* _waitset_ptr;
    static std::shared_ptr<HTTPServer> _http_server;
    static void _sigHandler(int sig);

    std::optional<iox::posix::SignalGuard> sigint;
    std::optional<iox::posix::SignalGuard> sigterm;

    std::unordered_map<int, Context> _client_contexts;
    std::unordered_map<int, Context> _server_contexts;
    Context _http_context;

    HTTPTrigger _http_trigger;

  };

}}

#endif
