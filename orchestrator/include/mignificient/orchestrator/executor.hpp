#ifndef __MIGNIFICIENT_ORCHESTRATOR_EXECUTOR_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_EXECUTOR_HPP__

#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <string>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/core.h>
#include <json/value.h>

#include <mignificient/ipc/config.hpp>

extern "C" char **environ;

namespace mignificient { namespace orchestrator {

  class GPUInstance;

  enum class GPUlessMessage {

    LOCK_DEVICE = 0,
    BASIC_EXEC = 1,
    MEMCPY_ONLY = 2,
    FULL_EXEC = 3,
    SWAP_OFF = 4,
    SWAP_IN = 5,

    REGISTER = 10,
    SWAP_OFF_CONFIRM = 11
  };

  class GPUlessServer {
  public:

    bool start(const ipc::IPCConfig& ipc_config, const std::string& user_id, GPUInstance& instance, bool poll_sleep, bool use_vmm, const Json::Value& config, int cpu_idx = -1);

  private:
    pid_t _pid;
  };

  /**
   * posix_spawn does not preserve existing envs when
   * adding new ones. We need to append new ones to the existing.
   */
  struct Environment
  {
    static Environment& instance()
    {
      static Environment env;
      return env;
    }

    void add(char* env)
    {
      if(global_size + current_idx < envs.size()) {
        envs[global_size + current_idx++] = env;
      } else {
        envs.push_back(env);
        current_idx++;
      }
    }

    void restart()
    {
      current_idx = 0;
    }

    char** data()
    {
      return envs.data();
    }

    const std::vector<char*>& vector()
    {
      return envs;
    }

  private:

    Environment()
    {
      for (char **env = environ; *env != nullptr; env++) {
        envs.push_back(*env);
      }

      global_size = envs.size();
      current_idx = 0;
    }

    size_t global_size;
    size_t current_idx;
    std::vector<char*> envs;
  };

  class Executor {
  public:
      Executor(const ipc::IPCConfig& ipc_config, const std::string& user, const std::string& function, float gpu_memory, GPUInstance& device, const std::optional<std::string>& ld_preload):
        _ipc_config(ipc_config),
        _user(user),
        _gpu_memory(gpu_memory),
        _ld_preload(ld_preload),
        _pid(0),
        _function(function),
        _device(device)
      {}

      virtual ~Executor() = default;

      const std::optional<std::string>& ld_preload() const
      {
        return _ld_preload;
      }

      const std::string& user() const
      {
        return _user;
      }

      float gpu_memory() const
      {
        return _gpu_memory;
      }

      pid_t pid() const
      {
        return _pid;
      }


  protected:
      void _configure_backends(Environment& env);
      std::vector<std::string> temporary_envs;

      const ipc::IPCConfig& _ipc_config;
      std::string _user;
      std::optional<std::string> _ld_preload;
      float _gpu_memory;
      pid_t _pid;
      std::string _function;
      GPUInstance& _device;
  };

  class BareMetalExecutorCpp : public Executor {
  public:
    using Executor::Executor;

    BareMetalExecutorCpp(
      const ipc::IPCConfig& ipc_config, const std::string& user_id, const std::string& function,
      const std::string& function_path,
      float gpu_memory, GPUInstance& device, const Json::Value& config,
      std::optional<std::string> ld_preload
    ):
      Executor(ipc_config, user_id, function, gpu_memory, device, ld_preload),
      _function_path(function_path),
      _cpp_executor(config["cpp"].asString()),
      _gpuless_lib(config["gpuless-lib"].asString())
    {}

    bool start(bool poll_sleep, int cpu_idx = -1);

  private:
    std::string _cpp_executor;
    std::string _function_path;
    std::string _gpuless_lib;
  };

  class BareMetalExecutorPython : public Executor {
  public:
    using Executor::Executor;

    BareMetalExecutorPython(
        const ipc::IPCConfig& ipc_config,
        const std::string& user_id,
        const std::string& function,
        const std::string& function_path,
        const std::string& cuda_binary,
        const std::string& cubin_analysis,
        float gpu_memory,
        GPUInstance& device,
        const Json::Value& config,
        std::optional<std::string> ld_preload
    ):
      Executor(ipc_config, user_id, function, gpu_memory, device, ld_preload),
      _function_path(function_path),
      _cuda_binary(cuda_binary),
      _cubin_analysis(cubin_analysis),
      _python_interpreter(config["python"][0].asString()),
      _python_executor(config["python"][1].asString()),
      _python_path(config["pythonpath"].asString()),
      _gpuless_lib(config["gpuless-lib"].asString())
    {}

    bool start(bool poll_sleep, int cpu_idx = -1);

  private:
    std::string _function_path;
    std::string _cuda_binary;
    std::string _cubin_analysis;
    std::string _python_interpreter;
    std::string _python_executor;
    std::string _python_path;
    std::string _gpuless_lib;
  };

  class SarusContainerExecutorCpp : public Executor {
  public:
      using Executor::Executor;
    SarusContainerExecutorCpp(
        const ipc::IPCConfig& ipc_config,
        const std::string& user_id,
        const std::string& function,
        const std::string& function_path,
        float gpu_memory, GPUInstance& device,
        const Json::Value& config,
        const std::optional<std::string>& ld_preload
    ):
      Executor(ipc_config, user_id, function, gpu_memory, device, ld_preload),
      _function_path(function_path),
      _cpp_executor(config["cpp"].asString()),
      _gpuless_lib(config["gpuless-lib"].asString())
    {}

    bool start(bool poll_sleep, int cpu_idx = -1);

  private:
    std::string _cpp_executor;
    std::string _function_path;
    std::string _gpuless_lib;
  };

}}

#endif
