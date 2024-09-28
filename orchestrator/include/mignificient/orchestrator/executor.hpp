#ifndef __MIGNIFICIENT_ORCHESTRATOR_EXECUTOR_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_EXECUTOR_HPP__

#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <string>

#include <fcntl.h>
#include <spawn.h>
#include <unistd.h>
#include <sched.h>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/core.h>
#include <jsoncpp/json/value.h>

extern char **environ;

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

    bool start(const std::string& user_id, GPUInstance& instance, bool poll_sleep, const Json::Value& config);

  private:
    pid_t _pid;
  };

  enum class Language {
    CPP,
    PYTHON
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
      Executor(const std::string& user, const std::string& function, float gpu_memory, GPUInstance& device):
        _user(user),
        _gpu_memory(gpu_memory),
        _pid(0),
        _function(function),
        _device(device)
      {}

      virtual ~Executor() = default;

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
      std::string _user;
      float _gpu_memory;
      pid_t _pid;
      std::string _function;
      GPUInstance& _device;
  };

  class BareMetalExecutorCpp : public Executor {
  public:
    using Executor::Executor;

    BareMetalExecutorCpp(const std::string& user_id, const std::string& function, const std::string& function_path, float gpu_memory, GPUInstance& device, const Json::Value& config):
      Executor(user_id, function, gpu_memory, device),
      _function_path(function_path),
      _cpp_executor(config["cpp"].asString()),
      _gpuless_lib(config["gpuless-lib"].asString())
    {}

    bool start(bool poll_sleep)
    {

      char* argv[] = {const_cast<char*>(_cpp_executor.c_str()), NULL};

      std::string poll_type = fmt::format("POLL_TYPE={}", poll_sleep ? "wait" : "poll");
      std::string fname = fmt::format("FUNCTION_NAME={}", _function);
      std::string cbinary = fmt::format("CUDA_BINARY={}", _function_path);
      std::string ffile = fmt::format("FUNCTION_FILE={}", _function_path);
      std::string preload = fmt::format("LD_PRELOAD={}", _gpuless_lib);
      std::string exec_type = "EXECUTOR_TYPE=shmem";
      std::string container_name = fmt::format("CONTAINER_NAME={}", _user);

      auto& envs = Environment::instance();
      envs.restart();
      envs.add(const_cast<char*>(poll_type.c_str()));
      envs.add(const_cast<char*>(fname.c_str()));
      envs.add(const_cast<char*>(cbinary.c_str()));
      envs.add(const_cast<char*>(ffile.c_str()));
      envs.add(const_cast<char*>(preload.c_str()));
      envs.add(const_cast<char*>(exec_type.c_str()));
      envs.add(const_cast<char*>(container_name.c_str()));
      envs.add(nullptr);

      for(char* ptr : envs.vector()) {

        if(ptr)
          spdlog::error(ptr);
      }

      posix_spawnattr_t attr;
      posix_spawn_file_actions_t file_actions;

      std::string log_name = fmt::format("output_executor_{}.log", _user);
      int log_fd = open(log_name.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (log_fd == -1) {
          perror("open");
          exit(1);
      }

      posix_spawnattr_init(&attr);
      posix_spawn_file_actions_init(&file_actions);

      posix_spawn_file_actions_adddup2(&file_actions, log_fd, STDOUT_FILENO);
      posix_spawn_file_actions_adddup2(&file_actions, log_fd, STDERR_FILENO);

      int status = posix_spawnp(&_pid, argv[0], &file_actions, &attr, argv, envs.data());

      if (status == 0) {
        spdlog::info("Child process spawned successfully, PID: %d\n", _pid);
      } else {
        spdlog::error("posix_spawn failed: %s\n", strerror(status));
        return false;
      }

      // Clean up
      posix_spawnattr_destroy(&attr);
      posix_spawn_file_actions_destroy(&file_actions);

      return true;
    }

  private:
    std::string _cpp_executor;
    std::string _function_path;
    std::string _gpuless_lib;
  };

  class SarusContainerExecutor : public Executor {
  public:
      using Executor::Executor;
  };

}}

#endif
