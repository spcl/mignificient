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

namespace mignificient { namespace orchestrator {

  class Users {
  public:

    Users(GPUManager& gpu_manager, const Json::Value& config, const ipc::IPCConfig& ipc_config, ContainerWorker& container_worker):
      _config(config),
      _gpu_manager(gpu_manager),
      _ipc_config(ipc_config),
      _container_worker(container_worker),
      _cpu_cap(config.get("cpu-cap", false).asBool()),
      _cgroup_root(config.get("cgroup-root", "").asString())
    {
      if(!_cpu_cap) {
        spdlog::info("CPU cap disabled (executor.cpu-cap): requests' cpu-cores are ignored");
      } else if(_cgroup_root.empty()) {
        spdlog::info("CPU cap enabled for containers only: executor.cgroup-root is empty, bare-metal clients with cpu-cores fail to start");
      } else if(!CpuCgroup::prepare_root(_cgroup_root)) {
        spdlog::critical("executor.cgroup-root {} is not usable (run tools/setup-cgroup.sh)", _cgroup_root);
        std::exit(EXIT_FAILURE);
      }
    }

    std::tuple<Client*, bool> process_invocation(std::unique_ptr<ActiveInvocation> && invocation)
    {
      const std::string& username = invocation->user();
      const std::string& fname = invocation->function_name();
      const std::string& fhandler = invocation->function_handler();
      float required_memory = invocation->gpu_memory();

      Client* selected_client = nullptr;
      GPUInstance* selected_gpu = nullptr;
      bool new_client_created = false;

      /**
       * (A) Idle container on idle GPU -> schedule on it.
       * (A') Lukewarm container -> swap-in first, then execute.
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
      Client* lukewarm_client = nullptr;

      if (it != _gpu_clients.end()) {
        for (auto& client : it->second) {
          // Client identity = function name + handler
          if (client->fname() == fname && client->handler() == fhandler) {

            if (client->function_path() != invocation->function_path() || client->language() != invocation->language()) {
              spdlog::error("Rejected invocation for function {} handler {}: different function-path/language", fname, fhandler);
              invocation->respond_bad_request(fmt::format(
                "function '{}' (handler '{}') is already registered with a different function-path/language", fname, fhandler));
              return std::make_tuple(nullptr, false);
            }

            GPUInstance* client_gpu = client->gpu_instance();

            if (!client->is_busy() && !client->is_lukewarm() && !client_gpu->is_busy()) {
              // Variant (A) - we have an idle container on idle GPU
              spdlog::info("Using an existing client {} for user {}", client->id(), username);
              selected_client = client.get();
              selected_gpu = client_gpu;
              fully_idle = true;
              break;
            } else if (client->is_lukewarm() && !lukewarm_client) {
              // Variant (A') - lukewarm container found
              lukewarm_client = client.get();
            } else {

              if (client_gpu->pending_invocations() < min_pending) {
                // Variant (B) - we have an busy container/GPU, found minimal
                selected_client = client.get();
                selected_gpu = client_gpu;
                min_pending = client_gpu->pending_invocations();
              }

            }
          }
        }
      }

      // Variant (A') - prefer lukewarm over allocating new cold container
      if(!selected_client && !fully_idle && lukewarm_client) {
        spdlog::info("Using lukewarm client {} for user {}, triggering swap-in", lukewarm_client->id(), username);
        selected_client = lukewarm_client;
        selected_gpu = lukewarm_client->gpu_instance();
        log_ignored_executor(*invocation, selected_client);

        auto* invoc_ptr = invocation.get();
        // Store the invocation for later
        selected_client->add_invocation(std::move(invocation));
        // GPU needs to also know the invocation to know the uuid later
        selected_gpu->add_pending_invocation(selected_client, invoc_ptr);

        // Trigger swap-in; when SWAP_IN_CONFIRM arrives, scheduling will continue
        selected_client->set_swap_in_for_invocation(true);
        selected_client->swap_in();

        return std::make_tuple(nullptr, true);
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

      if(!new_client_created) {
        log_ignored_executor(*invocation, selected_client);
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

    Client* find_client(const std::string& user, const std::string& id)
    {
      auto it = _gpu_clients.find(user);
      if (it == _gpu_clients.end())
        return nullptr;
      for (auto& client : it->second) {
        if (client->id() == id)
          return client.get();
      }
      return nullptr;
    }

    void remove_client(const std::string& user, Client* target)
    {
      auto it = _gpu_clients.find(user);
      if (it == _gpu_clients.end())
        return;

      auto& clients = it->second;
      for (auto cit = clients.begin(); cit != clients.end(); ++cit) {
        if (cit->get() == target) {
          clients.erase(cit);
          return;
        }
      }
    }

    template<typename F>
    Json::Value list_containers(F status_fn)
    {
      Json::Value result(Json::objectValue);
      for (auto& [username, clients] : _gpu_clients) {
        Json::Value user_containers(Json::arrayValue);
        for (auto& client : clients) {
          Json::Value entry;
          entry["id"] = client->id();
          entry["function"] = client->fname();
          entry["status"] = client->status_string();
          entry["allocated"] = client->is_active();
          user_containers.append(entry);
        }
        result[username] = user_containers;
      }
      return result;
    }

    template<typename F>
    void check_timeouts(F on_timeout)
    {
      for (auto& [username, clients] : _gpu_clients) {
        for (auto it = clients.begin(); it != clients.end(); ) {
          if ((*it)->check_timeout()) {
            on_timeout(it->get());
            it = clients.erase(it);
          } else {
            ++it;
          }
        }
      }
    }

    // Unregistered clients that can't start anymore: on_fail(client, reason), then removed.
    template<typename F>
    void check_startup(std::chrono::milliseconds timeout, F on_fail)
    {
      for (auto& [username, clients] : _gpu_clients) {
        for (auto it = clients.begin(); it != clients.end(); ) {
          std::optional<std::string> reason;
          if ((reason = (*it)->startup_failure(timeout))) {
            on_fail(it->get(), *reason);
            it = clients.erase(it);
          } else {
            ++it;
          }
        }
      }
    }

    template<typename F>
    void check_oom(F on_oom)
    {
      for (auto& [username, clients] : _gpu_clients) {
        for (auto it = clients.begin(); it != clients.end(); ) {
          if ((*it)->is_oom_detected()) {
            on_oom(it->get());
            it = clients.erase(it);
          } else {
            ++it;
          }
        }
      }
    }

  private:

    static void log_ignored_executor(const ActiveInvocation& invocation, const Client* client)
    {
      if(invocation.executor() && (*invocation.executor() == "container") != client->executor_ptr()->is_container()) {
        spdlog::info("Warm client {}: ignoring requested executor '{}'", client->id(), *invocation.executor());
      }
    }

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

      _gpu_clients[username].push_back(std::make_unique<Client>(
        _ipc_config.backend, client_id, fname, executor_buf, gpuless_buf, _ipc_config.client_root(client_id)
      ));
      auto selected_client = _gpu_clients[username].back().get();
      selected_client->set_function_config(fhandler, invocation->function_path(), invocation->language());

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

      auto spawn_time = std::chrono::high_resolution_clock::now();

      // The request's "executor" picks the executor kind of a new client; config is the default.
      bool use_container = invocation->executor().value_or(_config["type"].asString()) == "container";
      spdlog::info("Allocate client {} with {} executor{}", client_id, use_container ? "container" : "bare-metal",
                   _ipc_config.backend == ipc::IPCBackend::ICEORYX_V2 ? ", iceoryx2 directory " + _ipc_config.client_root(client_id) : "");
      float cpu_cores = _cpu_cap ? invocation->cpu_cores() : 0;

      // GPUless server always runs bare-metal on the host
      GPUlessServer gpuless_server;
      bool started = gpuless_server.start(
        _ipc_config, client_id, *selected_gpu,
        _config["poll-gpuless-sleep"].asBool(),
        _config["use-vmm"].asBool(),
        _config["bare-metal-executor"],
        invocation->gpu_memory(),
        gpuless_cpu_idx
      );

      std::string container_runtime = _config.isMember("container-runtime") ? _config["container-runtime"].asString() : "docker";

      std::unique_ptr<Executor> executor;
      if(use_container) {

        if(invocation->language() == Language::CPP) {

          auto exec = std::make_unique<DockerContainerExecutorCpp>(
            _ipc_config, client_id, fname, fhandler, invocation->function_path(),
            invocation->gpu_memory(), *selected_gpu, _config["container-executor"],
            invocation->ld_preload(), invocation->code_package(), _container_worker, container_runtime
          );
          exec->set_cpu_cores(cpu_cores);
          started = exec->start(_config["poll-sleep"].asBool(), executor_cpu_idx) && started;

          executor = std::move(exec);
        } else {

          auto exec = std::make_unique<DockerContainerExecutorPython>(
            _ipc_config, client_id, fname, fhandler, invocation->function_path(),
            invocation->cuda_binary(), invocation->cubin_analysis(),
            invocation->gpu_memory(), *selected_gpu, _config["container-executor"],
            invocation->ld_preload(), invocation->code_package(), _container_worker, container_runtime
          );
          exec->set_cpu_cores(cpu_cores);
          started = exec->start(_config["poll-sleep"].asBool(), executor_cpu_idx) && started;

          executor = std::move(exec);
        }

      } else {

        if(invocation->language() == Language::CPP) {

          auto exec = std::make_unique<BareMetalExecutorCpp>(
            _ipc_config, client_id, fname, fhandler, invocation->function_path(),
            invocation->gpu_memory(), *selected_gpu, _config["bare-metal-executor"],
            invocation->ld_preload()
          );
          exec->set_cpu_cores(cpu_cores);
          started = exec->start(_config["poll-sleep"].asBool(), executor_cpu_idx) && started;

          executor = std::move(exec);
        } else {

          auto exec = std::make_unique<BareMetalExecutorPython>(
            _ipc_config, client_id, fname, fhandler, invocation->function_path(),
            invocation->cuda_binary(), invocation->cubin_analysis(),
            invocation->gpu_memory(), *selected_gpu,
            _config["bare-metal-executor"],
            invocation->ld_preload(), invocation->code_package()
          );
          exec->set_cpu_cores(cpu_cores);
          started = exec->start(_config["poll-sleep"].asBool(), executor_cpu_idx) && started;

          executor = std::move(exec);
        }

      }

      // CPU cap, bare-metal: one cgroup with the executor and (cgroup-include-gpuless) its gpuless server. Containers
      // get --cpus; their gpuless server stays outside.
      // processes are moved in right after the spawn, so their first instructions run uncapped;
      // clone3(CLONE_INTO_CGROUP) if that ever matters.
      if(started && !use_container && cpu_cores > 0 && _cgroup_root.empty()) {
        spdlog::critical("Client {} requests cpu-cores {} but executor.cgroup-root is empty (run tools/setup-cgroup.sh)", client_id, cpu_cores);
        started = false;
      } else if(started && !use_container && cpu_cores > 0) {
        CpuCgroup cgroup;
        bool capped = cgroup.create(_cgroup_root, fmt::format("{}-{}", getpid(), client_id), cpu_cores) &&
          cgroup.add(executor->pid()) &&
          (!_config.get("cgroup-include-gpuless", true).asBool() || cgroup.add(gpuless_server.pid()));
        if(!capped) {
          started = false;
        }
        selected_client->set_cgroup(std::move(cgroup));
      }

      if(!started) {
        // Torn down by check_startup on the next tick.
        selected_client->set_startup_error("executor failed to start: spawn failed");
      }
      selected_client->set_spawn_time(spawn_time);
      selected_client->set_gpuless_server(std::move(gpuless_server), selected_gpu);
      selected_gpu->add_executor(executor.get());
      selected_client->set_executor(std::move(executor));

      return selected_client;
    }

    const Json::Value& _config;
    GPUManager& _gpu_manager;
    const ipc::IPCConfig& _ipc_config;
    ContainerWorker& _container_worker;

    // executor.cpu-cap: false ignores the requests' cpu-cores.
    bool _cpu_cap;
    // Parent of the per-client CPU-cap cgroups (bare-metal); required for bare-metal clients with cpu-cores.
    std::string _cgroup_root;

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
