#include <mignificient/orchestrator/executor.hpp>

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <spawn.h>
#include <unistd.h>
#include <sched.h>

#include <mignificient/orchestrator/device.hpp>

namespace mignificient { namespace orchestrator {

  bool GPUlessServer::start(
    const ipc::IPCConfig& ipc_config, const std::string& user_id, GPUInstance& instance,
    bool poll_sleep, bool use_vmm,
    const Json::Value& config, float gpu_memory, int cpu_idx
  )
  {
    std::string gpuless_mgr = config["gpuless-exec"].asString();
    std::string app_name = fmt::format("server-{}", user_id);
    std::string poll_type = poll_sleep ? "wait" : "poll";
    std::string use_vmm_flag{use_vmm ? "1" : "0"};
    char* argv[] = {
      const_cast<char*>(gpuless_mgr.c_str()),
      const_cast<char*>(instance.uuid().c_str()),
      const_cast<char*>("shmem"),
      const_cast<char*>(app_name.c_str()),
      const_cast<char*>(poll_type.c_str()),
      const_cast<char*>(user_id.c_str()),
      const_cast<char*>(use_vmm_flag.c_str()),
      NULL
    };

    std::string cpu_idx_str = fmt::format("CPU_BIND_IDX={}", cpu_idx);
    std::string ipc_backend = fmt::format("IPC_BACKEND={}", ipc::IPCConfig::backend_string(ipc_config.backend));
    std::string memory_limit = fmt::format("MIGNIFICIENT_MAX_GPU_MEMORY={}", std::to_string(gpu_memory));

    std::vector<char*> envs;
    envs.emplace_back(const_cast<char*>(ipc_backend.c_str()));
    envs.emplace_back(const_cast<char*>(memory_limit.c_str()));
    if(cpu_idx != -1) {
      envs.emplace_back(const_cast<char*>(cpu_idx_str.c_str()));
    }

#ifdef MIGNIFICIENT_WITH_ICEORYX2
    std::string req_size = fmt::format("GPULESS_REQUEST_SIZE={}", ipc_config.buffer_configs.at("gpuless-executor").request_size);
    std::string resp_size = fmt::format("GPULESS_RESPONSE_SIZE={}", ipc_config.buffer_configs.at("gpuless-executor").response_size);
    std::string queue_cap = fmt::format("GPULESS_QUEUE_CAPACITY={}", ipc_config.buffer_configs.at("gpuless-executor").queue_capacity);
    envs.emplace_back(const_cast<char*>(req_size.c_str()));
    envs.emplace_back(const_cast<char*>(resp_size.c_str()));
    envs.emplace_back(const_cast<char*>(queue_cap.c_str()));
#endif

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

  void Executor::_configure_backends(Environment& envs)
  {
    temporary_envs.clear();

    temporary_envs.push_back(fmt::format("IPC_BACKEND={}", ipc::IPCConfig::backend_string(_ipc_config.backend)));
    envs.add(const_cast<char*>(temporary_envs.back().c_str()));

#ifdef MIGNIFICIENT_WITH_ICEORYX2
    temporary_envs.push_back(fmt::format("GPULESS_REQUEST_SIZE={}", _ipc_config.buffer_configs.at("gpuless-executor").request_size));
    envs.add(const_cast<char*>(temporary_envs.back().c_str()));
    temporary_envs.push_back(fmt::format("GPULESS_RESPONSE_SIZE={}", _ipc_config.buffer_configs.at("gpuless-executor").response_size));
    envs.add(const_cast<char*>(temporary_envs.back().c_str()));
    temporary_envs.push_back(fmt::format("GPULESS_QUEUE_CAPACITY={}", _ipc_config.buffer_configs.at("gpuless-executor").queue_capacity));
    envs.add(const_cast<char*>(temporary_envs.back().c_str()));
#endif
  }

  bool BareMetalExecutorCpp::start(bool poll_sleep, int cpu_idx)
  {
    char* argv[] = {const_cast<char*>(_cpp_executor.c_str()), NULL};

    std::string poll_type = fmt::format("POLL_TYPE={}", poll_sleep ? "wait" : "poll");
    std::string fname = fmt::format("FUNCTION_HANDLER={}", _function_handler);
    std::string cbinary = fmt::format("CUDA_BINARY={}", _function_path);
    std::string ffile = fmt::format("FUNCTION_FILE={}", _function_path);

    std::string preload;
    if(_ld_preload.has_value()) {
      preload = fmt::format("LD_PRELOAD={}:{}", _ld_preload.value(), _gpuless_lib);
    } else {
      preload = fmt::format("LD_PRELOAD={}", _gpuless_lib);
    }

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

    _configure_backends(envs);

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
    std::string fname = fmt::format("FUNCTION_HANDLER={}", _function_handler);
    std::string cbinary = fmt::format("CUDA_BINARY={}", _cuda_binary);
    std::string ffile = fmt::format("FUNCTION_FILE={}", _function_path);
    std::string pythonpath = fmt::format("PYTHONPATH={}", _python_path);
    std::string exec_type = "EXECUTOR_TYPE=shmem";
    std::string container_name = fmt::format("CONTAINER_NAME={}", _user);
    std::string cpu_idx_str = fmt::format("CPU_BIND_IDX={}", cpu_idx);
    std::string gpuless_elf_path = fmt::format("GPULESS_ELF_DEFINITION={}", _cubin_analysis);

    std::string preload;
    if(_ld_preload.has_value()) {
      preload = fmt::format("LD_PRELOAD={}:{}", _ld_preload.value(), _gpuless_lib);
    } else {
      preload = fmt::format("LD_PRELOAD={}", _gpuless_lib);
    }

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
    envs.add(const_cast<char*>(gpuless_elf_path.c_str()));

    _configure_backends(envs);

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

  static std::string _build_docker_command(
    const std::string& container_runtime,
    const std::string& image,
    const std::optional<std::string>& code_package,
    const std::vector<std::pair<std::string, std::string>>& env_vars,
    const std::vector<std::string>& cmd
  )
  {
    std::string command = container_runtime + " run -d --sysctl net.core.rmem_default=2097152 --sysctl net.core.rmem_max=2097152 --user 1000:1000 -v /opt/miniconda3/envs/cuda_116_pytorch/:/code2 --mount type=bind,source=/dev/shm,target=/dev/shm --mount type=bind,source=/home/mcopik/.config/iceoryx2,target=/etc/iceoryx2 --mount type=bind,source=/tmp/iceoryx2,target=/tmp/iceoryx2 ";

    if(code_package.has_value()) {
      if(container_runtime == "sarus") {
        command += fmt::format(" --mount type=bind,source={},destination=/code", code_package.value());
      } else {
        command += fmt::format(" -v {}:/code", code_package.value());
      }
    }


    for(const auto& [key, val] : env_vars) {
      command += fmt::format(" -e {}={}", key, val);
    }

    command += " " + image;

    for(const auto& c : cmd) {
      command += " " + c;
    }

    return command;
  }

  static std::string _capture_container_id(const std::string& command)
  {
    std::array<char, 128> buffer;
    std::string result;
    FILE* pipe = popen(command.c_str(), "r");
    if(!pipe) {
      spdlog::error("Failed to run container command: {}", command);
      return "";
    }
    while(fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
      result += buffer.data();
    }
    int status = pclose(pipe);
    if(status != 0) {
      spdlog::error("Container command failed with status {}: {}", status, command);
      return "";
    }
    // Trim trailing whitespace/newline
    while(!result.empty() && (result.back() == '\n' || result.back() == '\r' || result.back() == ' ')) {
      result.pop_back();
    }
    return result;
  }

  bool DockerContainerExecutorCpp::start(bool poll_sleep, int cpu_idx)
  {
    std::string preload;
    if(_ld_preload.has_value()) {
      preload = fmt::format("{}:{}", _ld_preload.value(), _gpuless_lib);
    } else {
      preload = _gpuless_lib;
    }

    // Determine function file path: if code_package is provided, map to /code/<basename>
    std::string function_file = _function_path;
    std::string cuda_binary = _function_path;
    std::string gpuless_elf = _function_path + ".txt";
    if(_code_package.has_value()) {
      // Extract basename from function_path
      auto pos = _function_path.find_last_of('/');
      std::string basename = (pos != std::string::npos) ? _function_path.substr(pos + 1) : _function_path;
      function_file = "/code/" + basename;
      cuda_binary = "/code/" + basename;
      gpuless_elf = "/code/" + basename + ".txt";
    }

    std::vector<std::pair<std::string, std::string>> env_vars = {
      {"POLL_TYPE", poll_sleep ? "wait" : "poll"},
      {"EXECUTOR_TYPE", "shmem"},
      {"CONTAINER_NAME", _user},
      {"FUNCTION_HANDLER", _function_handler},
      {"FUNCTION_FILE", function_file},
      {"CUDA_BINARY", cuda_binary},
      {"LD_PRELOAD", preload},
      {"IPC_BACKEND", ipc::IPCConfig::backend_string(_ipc_config.backend)},
      {"MIGNIFICIENT_MAX_GPU_MEMORY", std::to_string(_gpu_memory)},
      {"GPULESS_ELF_DEFINITION", gpuless_elf}
    };

#ifdef MIGNIFICIENT_WITH_ICEORYX2
    env_vars.emplace_back("GPULESS_REQUEST_SIZE", std::to_string(_ipc_config.buffer_configs.at("gpuless-executor").request_size));
    env_vars.emplace_back("GPULESS_RESPONSE_SIZE", std::to_string(_ipc_config.buffer_configs.at("gpuless-executor").response_size));
    env_vars.emplace_back("GPULESS_QUEUE_CAPACITY", std::to_string(_ipc_config.buffer_configs.at("gpuless-executor").queue_capacity));
#endif

    std::string command = _build_docker_command(
      _container_runtime, _image, _code_package, env_vars,
      {_cpp_executor}
    );

    spdlog::info("Starting container executor: {}", command);

    _container_id = _capture_container_id(command);
    if(_container_id.empty()) {
      spdlog::error("Failed to start container for user {}", _user);
      return false;
    }

    spdlog::info("Container started with ID: {} for user {}", _container_id, _user);
    return true;
  }

  void DockerContainerExecutorCpp::stop()
  {
    if(!_container_id.empty()) {
      std::string command = fmt::format("{} kill {}", _container_runtime, _container_id);
      spdlog::info("Stopping container: {}", command);
      int ret = system(command.c_str());
      if(ret != 0) {
        spdlog::warn("Failed to kill container {}", _container_id);
      }
      _container_id.clear();
    }
  }

  bool DockerContainerExecutorPython::start(bool poll_sleep, int cpu_idx)
  {
    std::string preload;
    if(_ld_preload.has_value()) {
      preload = fmt::format("{}:{}", _ld_preload.value(), _gpuless_lib);
    } else {
      preload = _gpuless_lib;
    }

    std::string function_file = _function_path;
    std::string cuda_binary = _cuda_binary;
    std::string gpuless_elf = _cubin_analysis;
    if(_code_package.has_value()) {
      auto pos = _function_path.find_last_of('/');
      std::string basename = (pos != std::string::npos) ? _function_path.substr(pos + 1) : _function_path;
      function_file = "/code/" + basename;

      pos = _cuda_binary.find_last_of('/');
      basename = (pos != std::string::npos) ? _cuda_binary.substr(pos + 1) : _cuda_binary;
      cuda_binary = "/code/" + basename;

      pos = _cubin_analysis.find_last_of('/');
      basename = (pos != std::string::npos) ? _cubin_analysis.substr(pos + 1) : _cubin_analysis;
      gpuless_elf = "/code/" + basename;
    }

    std::vector<std::pair<std::string, std::string>> env_vars = {
      {"POLL_TYPE", poll_sleep ? "wait" : "poll"},
      {"EXECUTOR_TYPE", "shmem"},
      {"CONTAINER_NAME", _user},
      {"FUNCTION_HANDLER", _function_handler},
      {"FUNCTION_FILE", function_file},
      //{"PYTHONPATH", "/code/.python_packages:/opt/mignificient/build/executor"},
      {"PYTHONPATH", "/code2/lib/python3.9/site-packages/:/opt/mignificient/build/executor"},
      {"CUDA_BINARY", cuda_binary},
      //{"LD_PRELOAD", preload},
      {"LD_PRELOAD", "/opt/mignificient/build/gpuless/libgpuless.so"},
      {"IPC_BACKEND", ipc::IPCConfig::backend_string(_ipc_config.backend)},
      {"MIGNIFICIENT_MAX_GPU_MEMORY", std::to_string(_gpu_memory)},
      {"GPULESS_ELF_DEFINITION", gpuless_elf}
    };

#ifdef MIGNIFICIENT_WITH_ICEORYX2
    env_vars.emplace_back("GPULESS_REQUEST_SIZE", std::to_string(_ipc_config.buffer_configs.at("gpuless-executor").request_size));
    env_vars.emplace_back("GPULESS_RESPONSE_SIZE", std::to_string(_ipc_config.buffer_configs.at("gpuless-executor").response_size));
    env_vars.emplace_back("GPULESS_QUEUE_CAPACITY", std::to_string(_ipc_config.buffer_configs.at("gpuless-executor").queue_capacity));
#endif

    std::string command = _build_docker_command(
      _container_runtime, _image, _code_package, env_vars,
      {_python_interpreter, _python_executor}
    );

    spdlog::info("Starting container executor (Python): {}", command);

    _container_id = _capture_container_id(command);
    if(_container_id.empty()) {
      spdlog::error("Failed to start Python container for user {}", _user);
      return false;
    }

    spdlog::info("Container started with ID: {} for user {}", _container_id, _user);
    return true;
  }

  void DockerContainerExecutorPython::stop()
  {
    if(!_container_id.empty()) {
      std::string command = fmt::format("{} kill {}", _container_runtime, _container_id);
      spdlog::info("Stopping container: {}", command);
      int ret = system(command.c_str());
      if(ret != 0) {
        spdlog::warn("Failed to kill container {}", _container_id);
      }
      _container_id.clear();
    }
  }

}}
