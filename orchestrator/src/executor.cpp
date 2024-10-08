#include <mignificient/orchestrator/executor.hpp>

#include <fcntl.h>
#include <spawn.h>
#include <unistd.h>
#include <sched.h>

#include <mignificient/orchestrator/device.hpp>

namespace mignificient { namespace orchestrator {

  bool GPUlessServer::start(const std::string& user_id, GPUInstance& instance, bool poll_sleep, const Json::Value& config, int cpu_idx)
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

    std::string cpu_idx_str = fmt::format("CPU_BIND_IDX={}", cpu_idx);
    std::vector<char*> envs;
    if(cpu_idx != -1) {
      envs.emplace_back(const_cast<char*>(cpu_idx_str.c_str()));
    }
    envs.emplace_back(nullptr);

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

    int status = posix_spawnp(&_pid, argv[0], &file_actions, &attr, argv, envs.data());

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

  bool BareMetalExecutorCpp::start(bool poll_sleep, int cpu_idx)
  {

    char* argv[] = {const_cast<char*>(_cpp_executor.c_str()), NULL};

    std::string poll_type = fmt::format("POLL_TYPE={}", poll_sleep ? "wait" : "poll");
    std::string fname = fmt::format("FUNCTION_NAME={}", _function);
    std::string cbinary = fmt::format("CUDA_BINARY={}", _function_path);
    std::string ffile = fmt::format("FUNCTION_FILE={}", _function_path);
    std::string preload = fmt::format("LD_PRELOAD={}", _gpuless_lib);
    std::string exec_type = "EXECUTOR_TYPE=shmem";
    std::string container_name = fmt::format("CONTAINER_NAME={}", _user);
    std::string cpu_idx_str = fmt::format("CPU_BIND_IDX={}", cpu_idx);

    std::string gpuless_elf_path = fmt::format("GPULESS_ELF_DEFINITION={}.txt", _function_path);

    auto& envs = Environment::instance();
    envs.restart();
    envs.add(const_cast<char*>(poll_type.c_str()));
    envs.add(const_cast<char*>(fname.c_str()));
    envs.add(const_cast<char*>(cbinary.c_str()));
    envs.add(const_cast<char*>(ffile.c_str()));
    envs.add(const_cast<char*>(preload.c_str()));
    envs.add(const_cast<char*>(exec_type.c_str()));
    envs.add(const_cast<char*>(container_name.c_str()));
    envs.add(const_cast<char*>(gpuless_elf_path.c_str()));
    if(cpu_idx != -1) {
      envs.add(const_cast<char*>(cpu_idx_str.c_str()));
    }
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

  bool BareMetalExecutorPython::start(bool poll_sleep, int cpu_idx)
  {
    char* argv[] = {
      const_cast<char*>(_python_interpreter.c_str()),
      const_cast<char*>(_python_executor.c_str()),
      nullptr
    };

    std::string poll_type = fmt::format("POLL_TYPE={}", poll_sleep ? "wait" : "poll");
    std::string fname = fmt::format("FUNCTION_NAME={}", _function);
    std::string cbinary = fmt::format("CUDA_BINARY={}", _function_path);
    std::string ffile = fmt::format("FUNCTION_FILE={}", _function_path);
    std::string preload = fmt::format("LD_PRELOAD={}", _gpuless_lib);
    std::string pythonpath = fmt::format("PYTHONPATH={}", _python_path);
    std::string exec_type = "EXECUTOR_TYPE=shmem";
    std::string container_name = fmt::format("CONTAINER_NAME={}", _user);
    std::string cpu_idx_str = fmt::format("CPU_BIND_IDX={}", cpu_idx);

    auto& envs = Environment::instance();
    envs.restart();
    envs.add(const_cast<char*>(poll_type.c_str()));
    envs.add(const_cast<char*>(fname.c_str()));
    envs.add(const_cast<char*>(cbinary.c_str()));
    envs.add(const_cast<char*>(ffile.c_str()));
    envs.add(const_cast<char*>(preload.c_str()));
    envs.add(const_cast<char*>(exec_type.c_str()));
    envs.add(const_cast<char*>(container_name.c_str()));
    envs.add(const_cast<char*>(pythonpath.c_str()));
    if(cpu_idx != -1) {
      envs.add(const_cast<char*>(cpu_idx_str.c_str()));
    }
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

  bool SarusContainerExecutorCpp::start(bool poll_sleep, int cpu_idx)
  {
    //sarus run -t -e POLL_TYPE=wait -e EXECUTOR_TYPE=shmem -e MANAGER_IP=148.187.105.35 -e CUDA_BINARY=/artifact/benchmarks/microbenchmark/latency/latency_size.so -e FUNCTION_NAME=function -e CONTAINER_NAME=client_0 -e FUNCTION_FILE=/artifact/benchmarks/microbenchmark/latency/latency_size.so --mount type=bind,source=/tmp,target=/tmp --mount type=bind,source=/scratch/mcopik/gpus/mignificient-artifact,target=/artifact spcleth/mignificient:executor-sarus bash -c "LD_LIBRARY_PATH=/usr/local/cuda-11.6/compat/ LD_PRELOAD=/build/gpuless/libgpuless.so /build/executor/bin/executor_cpp"

    // TODO: do we need -t here?
    std::vector<std::string> argv{
      "sarus", "run", "-t"
    };

    argv.emplace_back("-e");
    argv.emplace_back(fmt::format("POLL_TYPE={}", poll_sleep ? "wait" : "poll"));
    argv.emplace_back("EXECUTOR_TYPE=shmem");
    argv.emplace_back(fmt::format("CONTAINER_NAME={}", _user));
    //argv.emplace_back(fmt::format("FUNCTION_NAME={}", _function));
    //argv.emplace_back(fmt::format("CUDA_BINARY={}", _function_path));
    //argv.emplace_back(fmt::format("FUNCTION_FILE={}", _function_path));

    auto& envs = Environment::instance();
    envs.restart();
    //envs.add(const_cast<char*>(poll_type.c_str()));
    //envs.add(const_cast<char*>(fname.c_str()));
    //envs.add(const_cast<char*>(cbinary.c_str()));
    //envs.add(const_cast<char*>(ffile.c_str()));
    //envs.add(const_cast<char*>(preload.c_str()));
    //envs.add(const_cast<char*>(exec_type.c_str()));
    //envs.add(const_cast<char*>(container_name.c_str()));
    //envs.add(nullptr);

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

    //int status = posix_spawnp(&_pid, argv[0], &file_actions, &attr, argv, envs.data());

    //if (status == 0) {
    //  spdlog::info("Child process spawned successfully, PID: {}", _pid);
    //} else {
    //  spdlog::error("posix_spawn failed: %s\n", strerror(status));
    //  return false;
    //}

    //// Clean up
    //posix_spawnattr_destroy(&attr);
    //posix_spawn_file_actions_destroy(&file_actions);

    return true;
  }

}}
