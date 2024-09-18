#ifndef __MIGNIFICIENT_ORCHESTRATOR_ORCHESTRATOR_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_ORCHESTRATOR_HPP__

#include <optional>
#include <unordered_map>

#include <iceoryx_hoofs/posix_wrapper/signal_handler.hpp>
#include <iceoryx_posh/popo/wait_set.hpp>
#include <json/value.h>

#include <mignificient/orchestrator/client.hpp>
#include <mignificient/orchestrator/http.hpp>

namespace mignificient {
namespace orchestrator {

struct Orchestrator {

  Orchestrator(const Json::Value &value);

  Client *client(int id);
  void add_client();

  void run();
  void event_loop();

  static void init(const Json::Value& config);

private:
  int _client_id = 0;
  int _invoc_id = 0;

  std::unordered_map<int, Client> clients;

  const void *last_message;
  iox::popo::WaitSet<> _waitset;

  static bool _quit;
  static iox::popo::WaitSet<> *_waitset_ptr;
  static std::shared_ptr<HTTPServer> _http_server;
  static void _sigHandler(int sig);

  std::optional<iox::posix::SignalGuard> sigint;
  std::optional<iox::posix::SignalGuard> sigterm;
};

} // namespace orchestrator
} // namespace mignificient

#endif
