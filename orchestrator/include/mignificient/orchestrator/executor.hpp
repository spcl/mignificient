#ifndef __MIGNIFICIENT_ORCHESTRATOR_EXECUTOR_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_EXECUTOR_HPP__

#include <array>
#include <csignal>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <sys/wait.h>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/core.h>
#include <json/value.h>

#include <mignificient/ipc/config.hpp>
#include <mignificient/orchestrator/container_worker.hpp>

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
    SWAP_OFF_CONFIRM = 11,
    SWAP_IN_CONFIRM = 12,

    OUT_OF_MEMORY = 13,
    INVOCATION_FINISH = 14
  };

  class GPUlessServer {
  public:

    bool start(const ipc::IPCConfig& ipc_config, const std::string& user_id, GPUInstance& instance, bool poll_sleep, bool use_vmm, const Json::Value& config, float max_memory, int cpu_idx = -1);

    pid_t pid() const { return _pid; }

    // Reaps the server if it has exited; the pid is then forgotten, so it's never signalled.
    bool exited()
    {
      if(_pid > 0 && waitpid(_pid, nullptr, WNOHANG) == _pid) {
        _pid = -1;
        return true;
      }
      return false;
    }

    void stop()
    {
      if(_pid > 0) {
        kill(_pid, SIGKILL);
        waitpid(_pid, nullptr, 0);
        _pid = -1;
      }
    }

  private:
    pid_t _pid = -1;
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
      Executor(const ipc::IPCConfig& ipc_config, const std::string& user, const std::string& function, const std::string& function_handler, float gpu_memory, GPUInstance& device, const std::optional<std::string>& ld_preload):
        _ipc_config(ipc_config),
        _user(user),
        _gpu_memory(gpu_memory),
        _ld_preload(ld_preload),
        _pid(-1),
        _function(function),
        _function_handler(function_handler),
        _device(device)
      {}

      virtual ~Executor() = default;

      void stop()
      {
        if(_worker) {
          // Container: the worker kills it, or cancels a start that hasn't finished.
          if(!_container_id.empty()) {
            _worker->stop(std::exchange(_container_id, ""));
          } else {
            _worker->cancel(_user);
          }
        }
        // Never signal pid <= 0: kill(0) / kill(-1) hit our own process group / every process.
        else if(_pid > 0) {
          kill(_pid, SIGKILL);
          waitpid(_pid, nullptr, 0);
          _pid = -1;
        }
      }

      // Bare metal: reaps the process if it has exited. Containers report exits through the worker.
      bool exited()
      {
        if(_pid > 0 && waitpid(_pid, nullptr, WNOHANG) == _pid) {
          _pid = -1;
          return true;
        }
        return false;
      }

      // The executor registered: a container no longer needs the early-death watch.
      void registered()
      {
        if(_worker) {
          _worker->unwatch(_user);
        }
      }

      bool is_container() const
      {
        return _worker != nullptr;
      }

      void set_container_id(const std::string& id)
      {
        _container_id = id;
      }

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

      // The client's own iceoryx2 directory (iceoryx2 backend only).
      std::optional<std::string> _iox2_root() const
      {
        if(_ipc_config.backend != ipc::IPCBackend::ICEORYX_V2) {
          return std::nullopt;
        }
        return _ipc_config.client_root(_user);
      }
      std::vector<std::string> temporary_envs;

      const ipc::IPCConfig& _ipc_config;
      std::string _user;
      std::optional<std::string> _ld_preload;
      float _gpu_memory;
      pid_t _pid;
      std::string _function;
      std::string _function_handler;
      GPUInstance& _device;
      // Set for container executors; `_user` is the client id.
      ContainerWorker* _worker = nullptr;
      std::string _container_id;
  };

  class BareMetalExecutorCpp : public Executor {
  public:
    using Executor::Executor;

    BareMetalExecutorCpp(
      const ipc::IPCConfig& ipc_config, const std::string& user_id, const std::string& function,
      const std::string& function_handler, const std::string& function_path,
      float gpu_memory, GPUInstance& device, const Json::Value& config,
      std::optional<std::string> ld_preload
    ):
      Executor(ipc_config, user_id, function, function_handler, gpu_memory, device, ld_preload),
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

  /**
   * Argv + environment of an executor process. The bare-metal launcher spawns it,
   * the container launcher passes it to `<runtime> run`.
   */
  struct LaunchSpec {
    std::vector<std::string> argv;
    std::vector<std::pair<std::string, std::string>> env;
  };

  class ExecutorPython : public Executor {
  public:

    ExecutorPython(
        const ipc::IPCConfig& ipc_config,
        const std::string& user_id,
        const std::string& function,
        const std::string& function_handler,
        const std::string& function_path,
        const std::string& cuda_binary,
        const std::string& cubin_analysis,
        float gpu_memory,
        GPUInstance& device,
        const Json::Value& config,
        const std::optional<std::string>& ld_preload,
        const std::optional<std::string>& code_package
    ):
      Executor(ipc_config, user_id, function, function_handler, gpu_memory, device, ld_preload),
      _function_path(function_path),
      _cuda_binary(cuda_binary),
      _cubin_analysis(cubin_analysis),
      _python_interpreter(config["python"][0].asString()),
      _python_executor(config["python"][1].asString()),
      _python_path(config["pythonpath"].asString()),
      _gpuless_lib(config["gpuless-lib"].asString()),
      _code_package(code_package)
    {}

    /**
     * The single launch description of a Python executor, for both launchers.
     * Interpreter: <code-package>/env/bin/python, or the configured one without a package.
     */
    LaunchSpec python_launch(bool poll_sleep, int cpu_idx) const;

  protected:
    std::string _function_path;
    std::string _cuda_binary;
    std::string _cubin_analysis;
    std::string _python_interpreter;
    std::string _python_executor;
    std::string _python_path;
    std::string _gpuless_lib;
    std::optional<std::string> _code_package;
  };

  class BareMetalExecutorPython : public ExecutorPython {
  public:
    using ExecutorPython::ExecutorPython;

    bool start(bool poll_sleep, int cpu_idx = -1);
  };

  class DockerContainerExecutorCpp : public Executor {
  public:
    using Executor::Executor;

    DockerContainerExecutorCpp(
        const ipc::IPCConfig& ipc_config,
        const std::string& user_id,
        const std::string& function,
        const std::string& function_handler,
        const std::string& function_path,
        float gpu_memory, GPUInstance& device,
        const Json::Value& config,
        const std::optional<std::string>& ld_preload,
        const std::optional<std::string>& code_package,
        ContainerWorker& worker,
        const std::string& container_runtime = "docker"
    ):
      Executor(ipc_config, user_id, function, function_handler, gpu_memory, device, ld_preload),
      _function_path(function_path),
      _cpp_executor(config["cpp"].asString()),
      _gpuless_lib(config["gpuless-lib"].asString()),
      _image(config["image"].asString()),
      _code_package(code_package),
      _container_runtime(container_runtime)
    {
      _worker = &worker;
    }

    // Queues `<runtime> run` on the worker; the container id arrives through drain().
    bool start(bool poll_sleep, int cpu_idx = -1);

  private:
    std::string _cpp_executor;
    std::string _function_path;
    std::string _gpuless_lib;
    std::string _image;
    std::optional<std::string> _code_package;
    std::string _container_runtime;
  };

  class DockerContainerExecutorPython : public ExecutorPython {
  public:

    DockerContainerExecutorPython(
        const ipc::IPCConfig& ipc_config,
        const std::string& user_id,
        const std::string& function,
        const std::string& function_handler,
        const std::string& function_path,
        const std::string& cuda_binary,
        const std::string& cubin_analysis,
        float gpu_memory, GPUInstance& device,
        const Json::Value& config,
        const std::optional<std::string>& ld_preload,
        const std::optional<std::string>& code_package,
        ContainerWorker& worker,
        const std::string& container_runtime = "docker"
    ):
      ExecutorPython(
        ipc_config, user_id, function, function_handler, function_path, cuda_binary, cubin_analysis,
        gpu_memory, device, config, ld_preload, code_package
      ),
      _image(config["image"].asString()),
      _container_runtime(container_runtime)
    {
      _worker = &worker;
    }

    // Queues `<runtime> run` on the worker; the container id arrives through drain().
    bool start(bool poll_sleep, int cpu_idx = -1);

  private:
    std::string _image;
    std::string _container_runtime;
  };

}}

#endif
