#include <mignificient/orchestrator/executor.hpp>

#include <fcntl.h>
#include <spawn.h>
#include <unistd.h>
#include <sched.h>

#include <mignificient/orchestrator/device.hpp>

namespace mignificient { namespace orchestrator {

  bool GPUlessServer::start(const std::string& user_id, GPUInstance& instance, bool poll_sleep, const Json::Value& config)
  {
    std::string gpuless_mgr = config["gpuless-exec"].asString();
    std::string app_name = fmt::format("server-{}", user_id);
    std::string poll_type = poll_sleep ? "wait" : "poll";
    char* argv[] = {
      const_cast<char*>(gpuless_mgr.c_str()),
      const_cast<char*>(instance.uuid().c_str()),
      const_cast<char*>("shmem"),
      const_cast<char*>(app_name.c_str()),
      const_cast<char*>(poll_type.c_str()),
      const_cast<char*>(user_id.c_str()),
      NULL
    };

    posix_spawnattr_t attr;
    posix_spawn_file_actions_t file_actions;

    std::string log_name = fmt::format("output_gpuless_{}.log", user_id);
    int log_fd = open(log_name.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (log_fd == -1) {
        perror("open");
        exit(1);
    }

    posix_spawnattr_init(&attr);
    posix_spawn_file_actions_init(&file_actions);

    posix_spawn_file_actions_adddup2(&file_actions, log_fd, STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&file_actions, log_fd, STDERR_FILENO);

    int status = posix_spawnp(&_pid, argv[0], &file_actions, &attr, argv, nullptr);

    if (status == 0) {
      spdlog::info("Child process spawned successfully, PID: {}", _pid);
    } else {
      spdlog::error("posix_spawn failed: {}", strerror(status));
      return false;
    }

    // Clean up
    posix_spawnattr_destroy(&attr);
    posix_spawn_file_actions_destroy(&file_actions);

    return true;
  }

  bool BareMetalExecutorCpp::start(bool poll_sleep)
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
      spdlog::info("Child process spawned successfully, PID: {}", _pid);
    } else {
      spdlog::error("posix_spawn failed: %s\n", strerror(status));
      return false;
    }

    // Clean up
    posix_spawnattr_destroy(&attr);
    posix_spawn_file_actions_destroy(&file_actions);

    return true;
  }

}}
