#ifndef __MIGNIFICIENT_ORCHESTRATOR_USERS_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_USERS_HPP__

#include <iceoryx_posh/popo/subscriber.hpp>
#include <iceoryx_posh/popo/publisher.hpp>

#include <mignificient/executor/executor.hpp>
#include <mignificient/orchestrator/client.hpp>
#include <mignificient/orchestrator/event.hpp>
#include <mignificient/orchestrator/invocation.hpp>

namespace mignificient { namespace orchestrator {

  class Users {
  public:

      void process_invocations(const std::vector<ActiveInvocation>& invocations)
      {
          for (const auto& invocation : invocations) {
              process_invocation(invocation);
          }
      }

  private:

    int _index = 0;
    std::unordered_map<std::string, std::vector<std::unique_ptr<Client>>> _gpu_clients;

    void process_invocation(const ActiveInvocation& invocation)
    {

      const std::string& username = invocation.user();
      const std::string& fname = invocation.function_name();

      Client* selected_client = nullptr;

      auto it = _gpu_clients.find(username);
      if (it != _gpu_clients.end()) {

          for (auto& client : it->second) {

              if (client->fname() == fname) {
                  if (!client->isBusy()) {
                      selected_client = client.get();
                      spdlog::error("Using an existing client {} for user {}", selected_client->id(), username);
                      break;
                  }
              }
          }
      }

      if (!selected_client) {

        std::string client_id = unique_client_name(username, fname);
        _gpu_clients[username].push_back(std::make_unique<Client>(client_id, fname));
        auto& new_client = _gpu_clients[username].back();

        spdlog::error("Allocate a new client {} for user {}", client_id, username);

        //auto gpulessServer = std::make_shared<GpulessServer>(/* parameters */);
        //auto executor = std::make_shared<Executor>(/* parameters */);

        //new_client.setGpulessServer(gpulessServer);
        //new_client.setExecutor(executor);

        selected_client = new_client.get();
      }
    }

    std::string unique_client_name(const std::string& username, const std::string &fname)
    {
      return fmt::format("{}-{}-{}", username, fname, _index++);
    }
  };

}}

#endif
