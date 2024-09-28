#ifndef __MIGNIFICIENT_ORCHESTRATOR_DEVICE_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_DEVICE_HPP__

#include <stdexcept>
#include <string>
#include <queue>
#include <vector>
#include <memory>

#include <drogon/HttpResponse.h>
#include <json/json.h>

#include <mignificient/orchestrator/client.hpp>
#include <mignificient/orchestrator/executor.hpp>
#include <mignificient/orchestrator/invocation.hpp>

namespace mignificient { namespace orchestrator {

  enum class SharingModel
  {
    SEQUENTIAL,
    OVERLAP_CPU,
    OVERLAP_CPU_MEMCPY,
    FULL_OVERLAP
  };

  SharingModel sharing_model(const std::string& val);

  class GPUInstance {
  public:

    GPUInstance(const std::string& uuid, float memory, const std::string& instance_size, SharingModel model)
        : _uuid(uuid),
          _memory(memory),
          _instance_size(instance_size),
          _sharing_model(model)
    {}

    const std::string& uuid() const
    {
      return _uuid;
    }

    float memory() const
    {
      return _memory;
    }

    const std::string& compute_units() const
    {
      return _instance_size;
    }

    SharingModel getModel() const
    {
      return _sharing_model;
    }

    void add_invocation(Client* client, ActiveInvocation* invocation)
    {
      if(is_busy() || !client->is_active()) {

        spdlog::info("[GPUInstance {}] Add invocation with id {} for client {}", _uuid, invocation->uuid(), client->id());
        _pending_invocations.emplace(invocation, client);

        schedule_next();

      } else {

        _current_invocation = std::make_tuple(invocation, client);
        spdlog::info("[GPUInstance {}] Start a new active invocation with id {} for client {}", _uuid, invocation->uuid(), client->id());
        client->send_request();
        client->activate_kernels();
      }
    }

    bool is_busy() const
    {
      return std::get<0>(_current_invocation) != nullptr;
    }

    void schedule_next()
    {
      /***
       * (1) Case 1: no active invocation, put the next one. When? Just registered or finished.
       * (2) There is an active invocation, we try to move forward the next pending one.
       */

      // FIXME: Not tested with scheduling more than 2 invocations
      auto [invocation, client] = _pending_invocations.front();
      spdlog::error("Attempting to schedule {}, client is active? {}", invocation->uuid(), client->is_active());
      if(!client->is_active()) {
        return;
      }

      if(is_busy()) {

        auto status = client->status();

        /**
         * SEQUENTIAL: do nothing, wait for current to finish
         * OVERLAP_CPU: start function, block device
         * OVERLAP_CPU_MEMORY: start function, block device
         * FULL_OVERLAP: start function, block device
         */
        if(status == ClientStatus::NOT_ACTIVE && _sharing_model != SharingModel::SEQUENTIAL) {
          spdlog::info("[GPUInstance {}] Start CPU invocation with id {} for client {}", _uuid, invocation->uuid(), client->id());
          client->send_request();
        }

        if(status == ClientStatus::CPU_ONLY && _sharing_model == SharingModel::OVERLAP_CPU_MEMCPY) {
          spdlog::info("[GPUInstance {}] Active memcpy for invocation with id {} for client {}", _uuid, invocation->uuid(), client->id());
          client->activate_memcpy();
        } else if(status == ClientStatus::MEMCPY && _sharing_model == SharingModel::FULL_OVERLAP) {
          spdlog::info("[GPUInstance {}] Active full execution for invocation with id {} for client {}", _uuid, invocation->uuid(), client->id());
          client->activate_kernels();
        }

      } else {

        spdlog::info("[GPUInstance {}] Start a new invocation with id {} for client {}", _uuid, invocation->uuid(), client->id());
        _pending_invocations.pop();

        spdlog::error("new invoc {}", fmt::ptr(client));
        _current_invocation = std::make_tuple(invocation, client);

        client->send_request();
        client->activate_kernels();

        // Check if the next one can be scheduled
        if(_sharing_model != SharingModel::SEQUENTIAL && !_pending_invocations.empty()) {
          schedule_next();
        }
      }
    }

    void finish_current_invocation()
    {
      auto [invoc, client] = _current_invocation;
        spdlog::error("finish invoc {}", fmt::ptr(client));
      spdlog::info("[GPUInstance] Finished invocation with id {} for client {}", invoc->uuid(), client->id());

      _current_invocation = std::make_tuple(nullptr, nullptr);
      if(!_pending_invocations.empty()) {
        schedule_next();
      }
    }

    void yield_current_invocation()
    {
      auto [invoc, client] = _current_invocation;
      spdlog::info("[GPUInstance] Yielded invocation with id {} for client {}", invoc->uuid(), client->id());
      if(!_pending_invocations.empty()) {
        schedule_next();
      }
    }

    void add_executor(Executor* executor)
    {
      _used_memory += executor->gpu_memory();
      _executors[executor->pid()] = executor;
    }

    void close_executor(pid_t pid)
    {
      auto memory = _executors[pid]->gpu_memory();
      spdlog::info("Closing down executor PID {}, freeing up {} MB", pid, memory);
      _executors.erase(pid);
      _used_memory -= memory;
    }

    bool has_enough_memory(float alloc) const
    {
      return _memory - _used_memory >= alloc;
    }

    size_t pending_invocations() const
    {
      return _pending_invocations.size();
    }

  private:
    std::string _uuid;

    float _memory;

    float _used_memory = 0;

    /**
      * NVIDIA MIG has sizes: 1g.5GB, 1g, 2g, 3g, 4g, 7g
      */
    std::string _instance_size;

    SharingModel _sharing_model;

    typedef std::tuple<ActiveInvocation*, Client*> invoc_t;

    std::queue<invoc_t> _pending_invocations;

    invoc_t _current_invocation;

    std::unordered_map<pid_t, Executor*> _executors;
  };

  class GPUDevice {
  public:

    GPUDevice(const Json::Value& gpu, SharingModel sharing_model);

    const std::string& uuid() const
    {
      return _uuid;
    }

    float memory() const
    {
      return _memory;
    }

    std::vector<GPUInstance>& instances()
    {
      return _mig_instances;
    }

  private:
    std::string _uuid;
    float _memory;

    std::vector<GPUInstance> _mig_instances;
  };

  class GPUManager {
  public:

    GPUManager(const std::string& devices_data_path, SharingModel sharing_model);

    GPUInstance* get_free_gpu()
    {
      if (!_idle_gpus.empty()) {
          auto* gpu = _idle_gpus.front();
          _idle_gpus.pop();
          return gpu;
      }
      return nullptr;
    }

    void return_gpu(GPUInstance* gpu) {
      _idle_gpus.push(gpu);
    }

    GPUInstance* get_least_busy_gpu()
    {
      GPUInstance* least_busy = nullptr;
      size_t min_pending = std::numeric_limits<size_t>::max();
      for (auto& device : _devices) {
        for (auto& instance : device.instances()) {
          if (instance.pending_invocations() < min_pending) {
              least_busy = &instance;
              min_pending = instance.pending_invocations();
          }
        }
      }
      return least_busy;
    }

  private:
    std::vector<GPUDevice> _devices;
    std::queue<GPUInstance*> _idle_gpus;
  };

}}

#endif
