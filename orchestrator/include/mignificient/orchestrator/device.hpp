#ifndef __MIGNIFICIENT_ORCHESTRATOR_DEVICE_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_DEVICE_HPP__

#include <stdexcept>
#include <string>
#include <queue>
#include <vector>
#include <memory>

#include <drogon/HttpResponse.h>
#include <json/json.h>

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

    void add_invocation(std::shared_ptr<ActiveInvocation> invocation)
    {
      _pending_invocations.push(std::move(invocation));
    }

    void finished_current_invocation()
    {
      // finish current one
      // move to another one
      throw std::runtime_error("unimplemented");
    }

    void add_executor(std::shared_ptr<Executor> executor)
    {
      _executors[executor->pid()] = executor;
    }

    void close_executor(pid_t pid)
    {
      _executors.erase(pid);
    }

  private:
    std::string _uuid;

    float _memory;

    /**
      * NVIDIA MIG has sizes: 1g.5GB, 1g, 2g, 3g, 4g, 7g
      */
    std::string _instance_size;

    SharingModel _sharing_model;

    std::queue<std::shared_ptr<ActiveInvocation>> _pending_invocations;

    std::shared_ptr<ActiveInvocation> _current_invocation;

    std::unordered_map<pid_t, std::shared_ptr<Executor>> _executors;
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

  private:
    std::string _uuid;
    float _memory;

    std::vector<GPUInstance> _mig_instances;
  };

  class GPUManager {
  public:

    GPUManager(const std::string& devices_data_path, SharingModel sharing_model);

  private:
    std::vector<GPUDevice> _devices;
  };

}}

#endif
