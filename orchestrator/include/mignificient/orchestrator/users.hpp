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

    Users(GPUManager& gpu_manager, const Json::Value& config, const ipc::IPCConfig& ipc_config):
      _config(config),
      _gpu_manager(gpu_manager),
      _ipc_config(ipc_config)
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

      /**
       * (A) Idle container on idle GPU -> schedule on it.
       *
       * Busy containers on idle GPUs or idle containers on busy GPUs?
       * (B) Then we add a new client IF there is an idle GPU.
       * (C) If not, then we select a client with least busy GPU.
       *
       * There is no container for this function?
       * (D) Add a new on idle or least busy GPU.
       * (E) No GPU can support this? Reject.
       */

      // Check for existing idle container
      // Ideal case: idle container on idle GPU
      auto it = _gpu_clients.find(username);
      size_t min_pending = std::numeric_limits<size_t>::max();
      bool fully_idle = false;

      if (it != _gpu_clients.end()) {
        for (auto& client : it->second) {
          if (client->fname() == fname) {

            selected_gpu = client->gpu_instance();

            if (!client->is_busy() && !selected_gpu->is_busy()) {
              // Variant (A) - we have an idle container on idle GPU
              spdlog::error("Using an existing client {} for user {}", client->id(), username);
              selected_client = client.get();
              fully_idle = true;
              break;
            } else {

              if (selected_gpu->pending_invocations() < min_pending) {
                // Variant (B) - we have an busy container/GPU, found minimal
                selected_client = client.get();
                min_pending = selected_gpu->pending_invocations();
              }

            }
          }
        }
        ++it;
      }

      // Variant A - jump forward to the end.

      // We have a client but not fully idle? Variant B, C
      if(selected_client && !fully_idle) {

        // No idle container found, look for an idle GPU
        auto idle_gpu = _gpu_manager.get_free_gpu(required_memory);

        // Variant B - allocate on this GPU
        if(idle_gpu) {

          selected_gpu = idle_gpu;
          selected_client = allocate(username, fname, invocation.get(), selected_gpu);
          new_client_created = true;

        }
        // Variant C - we continue with selected least busy GPU

      // No client? Variant D, E
      } else if (!selected_client){

        selected_gpu = _gpu_manager.get_free_gpu(required_memory);

        // Found idle GPU? Didn't find idle GPU 
        if (!selected_gpu) {
          // No idle GPU, find the least busy one
          selected_gpu = _gpu_manager.get_least_busy_gpu(required_memory);
        }

        // Variant D - allocate on least busy GPU
        if(selected_gpu) {

          selected_client = allocate(username, fname, invocation.get(), selected_gpu);
          new_client_created = true;

        } else {

          // Variant (E). No GPU with enough memory
          spdlog::error("Rejected invocation for function {} due to insufficient GPU memory", fname);
          invocation->failure("Not enough GPUs!");
          return std::make_tuple(nullptr, false);

        }

      }

      // Add invocation to the selected client/GPU
      auto* invoc_ptr = invocation.get();
      selected_client->add_invocation(std::move(invocation));
      selected_gpu->add_invocation(selected_client, invoc_ptr);

      SPDLOG_DEBUG("Processed invocation for function {} on client {}", fname, selected_client->id());
      return std::make_tuple(new_client_created ? selected_client : nullptr, true);
    }

    template<typename F>
    void apply_clients(F func)
    {
      for(auto& [username, clients] : _gpu_clients) {
        for(auto& client : clients) {
          func(client.get());
        }
      }
    }

  private:

    Client* allocate(const std::string& username, const std::string& fname, ActiveInvocation* invocation, GPUInstance* selected_gpu)
    {
      // Create a new client with configured buffer sizes
      std::string client_id = unique_client_name(username, fname);
      const std::string& fhandler = invocation->function_handler();

      ipc::BufferConfig executor_buf;
      ipc::BufferConfig gpuless_buf;
      auto it_exec = _ipc_config.buffer_configs.find("orchestrator-executor");
      if (it_exec != _ipc_config.buffer_configs.end()) executor_buf = it_exec->second;
      auto it_gpuless = _ipc_config.buffer_configs.find("orchestrator-gpuless");
      if (it_gpuless != _ipc_config.buffer_configs.end()) gpuless_buf = it_gpuless->second;

      _gpu_clients[username].push_back(std::make_unique<Client>(_ipc_config.backend, client_id, fname, executor_buf, gpuless_buf));
      auto selected_client = _gpu_clients[username].back().get();

      SPDLOG_DEBUG("Allocate a new client {} for user {}", client_id, username);

      int executor_cpu_idx = -1;
      int gpuless_cpu_idx = -1;
      if(_config["cpu-bind-executor"].asBool()) {
        executor_cpu_idx = _cpu_index++;

        if(_config["cpu-bind-gpuless"].asBool()) {
          if(_config["cpu-bind-gpuless-separate"].asBool()) {
            gpuless_cpu_idx = _cpu_index++;
          } else {
            gpuless_cpu_idx = executor_cpu_idx;
          }
        }

        SPDLOG_DEBUG("Binding executor {} to CPU {}, Gpuless server bound to CPU {}", client_id, executor_cpu_idx, gpuless_cpu_idx);

      }

      GPUlessServer gpuless_server;
      gpuless_server.start(
        _ipc_config, client_id, *selected_gpu,
        _config["poll-gpuless-sleep"].asBool(),
        _config["use-vmm"].asBool(),
        _config["bare-metal-executor"], gpuless_cpu_idx
      );

      std::unique_ptr<Executor> executor;
      if(invocation->language() == Language::CPP) {

        auto exec = std::make_unique<BareMetalExecutorCpp>(
          _ipc_config, client_id, fname, fhandler, invocation->function_path(),
          invocation->gpu_memory(), *selected_gpu, _config["bare-metal-executor"],
          invocation->ld_preload()
        );
        exec->start(_config["poll-sleep"].asBool(), executor_cpu_idx);

        executor = std::move(exec);
      } else {

        auto exec = std::make_unique<BareMetalExecutorPython>(
          _ipc_config, client_id, fname, fhandler, invocation->function_path(),
          invocation->cuda_binary(), invocation->cubin_analysis(),
          invocation->gpu_memory(), *selected_gpu,
          _config["bare-metal-executor"],
          invocation->ld_preload()
        );
        exec->start(_config["poll-sleep"].asBool(), executor_cpu_idx);

        executor = std::move(exec);

      }

      selected_client->set_gpuless_server(std::move(gpuless_server), selected_gpu);
      selected_gpu->add_executor(executor.get());
      selected_client->set_executor(std::move(executor));

      return selected_client;
    }

    const Json::Value& _config;
    GPUManager& _gpu_manager;
    const ipc::IPCConfig& _ipc_config;

    int _index = 0;
    // TODO: this might require extension to support platforms where hyperthreads have consecutive IDs
    int _cpu_index = 2;
    std::unordered_map<std::string, std::vector<std::unique_ptr<Client>>> _gpu_clients;


    std::string unique_client_name(const std::string& username, const std::string &fname)
    {
      return fmt::format("{}-{}-{}", username, fname, _index++);
    }
  };

}}

#endif
