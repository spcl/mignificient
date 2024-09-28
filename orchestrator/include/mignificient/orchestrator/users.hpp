#ifndef __MIGNIFICIENT_ORCHESTRATOR_USERS_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_USERS_HPP__

#include <iceoryx_posh/popo/subscriber.hpp>
#include <iceoryx_posh/popo/publisher.hpp>

#include <mignificient/executor/executor.hpp>
#include <mignificient/orchestrator/client.hpp>
#include <mignificient/orchestrator/device.hpp>
#include <mignificient/orchestrator/event.hpp>
#include <mignificient/orchestrator/executor.hpp>
#include <mignificient/orchestrator/invocation.hpp>
#include <stdexcept>

namespace mignificient { namespace orchestrator {

  class Users {
  public:

    Users(GPUManager& gpu_manager, const Json::Value& config):
      _config(config),
      _gpu_manager(gpu_manager)
    {

    }

    std::tuple<Client*, bool> process_invocation(std::unique_ptr<ActiveInvocation> && invocation)
    {
      const std::string& username = invocation->user();
      const std::string& fname = invocation->function_name();
      float required_memory = invocation->gpu_memory();

      Client* selected_client = nullptr;
      GPUInstance* selected_gpu = nullptr;
      bool new_client_created = false;

      // Check for existing idle container
      auto it = _gpu_clients.find(username);
      if (it != _gpu_clients.end()) {
        for (auto& client : it->second) {
          if (client->fname() == fname && !client->is_busy()) {

            selected_client = client.get();
            selected_gpu = client->gpu_instance();
            if (!selected_gpu->is_busy()) {
              // Ideal case: idle container on idle GPU
              spdlog::error("Using an existing client {} for user {}", selected_client->id(), username);
              break;
            } else {
              selected_client = nullptr;
            }

          }
        }
      }

      if(!selected_client) {

        // No idle container found, look for an idle GPU
        selected_gpu = _gpu_manager.get_free_gpu();

        if (!selected_gpu) {
          // No idle GPU, find the least busy one
          selected_gpu = _gpu_manager.get_least_busy_gpu();
        }

        if(selected_gpu && selected_gpu->has_enough_memory(required_memory)) {

          // Create a new client
          std::string client_id = unique_client_name(username, fname);
          _gpu_clients[username].push_back(std::make_unique<Client>(client_id, fname));
          selected_client = _gpu_clients[username].back().get();
          new_client_created = true;

          spdlog::error("Allocate a new client {} for user {}", client_id, username);

          GPUlessServer gpuless_server;
          gpuless_server.start(client_id, *selected_gpu, _config["poll-gpuless-sleep"].asBool(), _config["bare-metal-executor"]);

          auto executor = std::make_unique<BareMetalExecutorCpp>(client_id, fname, invocation->function_path(), invocation->gpu_memory(), *selected_gpu, _config["bare-metal-executor"]);
          executor->start(_config["poll-sleep"].asBool());

          selected_client->set_gpuless_server(std::move(gpuless_server), selected_gpu);
          selected_gpu->add_executor(executor.get());
          selected_client->set_executor(std::move(executor));

        } else {

          // No GPU with enough memory
          spdlog::error("Rejected invocation for function {} due to insufficient GPU memory", fname);
          invocation->failure("Not enough GPUs!");
          return std::make_tuple(nullptr, false);

        }

      }

      // Add invocation to the selected client/GPU
      auto* invoc_ptr = invocation.get();
      selected_client->add_invocation(std::move(invocation));
      selected_gpu->add_invocation(selected_client, invoc_ptr);

      spdlog::info("Processed invocation for function {} on client {}", fname, selected_client->id());
      return std::make_tuple(new_client_created ? selected_client : nullptr, true);
    }

  private:

    const Json::Value& _config;
    GPUManager& _gpu_manager;

    int _index = 0;
    std::unordered_map<std::string, std::vector<std::unique_ptr<Client>>> _gpu_clients;


    std::string unique_client_name(const std::string& username, const std::string &fname)
    {
      return fmt::format("{}-{}-{}", username, fname, _index++);
    }
  };

}}

#endif
