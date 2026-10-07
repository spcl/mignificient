#include <mignificient/orchestrator/executor.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string_view>
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

  LaunchSpec ExecutorPython::python_launch(bool poll_sleep, int cpu_idx) const
  {
    LaunchSpec spec;
    spec.argv = {
      _code_package.has_value() ? _code_package.value() + "/env/bin/python" : _python_interpreter,
      _python_executor
    };

    std::string preload = _ld_preload.has_value() ? fmt::format("{}:{}", _ld_preload.value(), _gpuless_lib) : _gpuless_lib;

    spec.env = {
      {"POLL_TYPE", poll_sleep ? "wait" : "poll"},
      {"EXECUTOR_TYPE", "shmem"},
      {"CONTAINER_NAME", _user},
      {"FUNCTION_HANDLER", _function_handler},
      {"FUNCTION_FILE", _function_path},
      {"CUDA_BINARY", _cuda_binary},
      {"GPULESS_ELF_DEFINITION", _cubin_analysis},
      {"PYTHONPATH", _python_path},
      {"LD_PRELOAD", preload},
      {"IPC_BACKEND", ipc::IPCConfig::backend_string(_ipc_config.backend)},
      {"MIGNIFICIENT_MAX_GPU_MEMORY", std::to_string(_gpu_memory)}
    };

#ifdef MIGNIFICIENT_WITH_ICEORYX2
    spec.env.emplace_back("GPULESS_REQUEST_SIZE", std::to_string(_ipc_config.buffer_configs.at("gpuless-executor").request_size));
    spec.env.emplace_back("GPULESS_RESPONSE_SIZE", std::to_string(_ipc_config.buffer_configs.at("gpuless-executor").response_size));
    spec.env.emplace_back("GPULESS_QUEUE_CAPACITY", std::to_string(_ipc_config.buffer_configs.at("gpuless-executor").queue_capacity));
#endif

    if(cpu_idx != -1) {
      spec.env.emplace_back("CPU_BIND_IDX", std::to_string(cpu_idx));
    }

    // Models live in the package: TORCH_HOME=<pkg>/<torch-home> (read-only in containers).
    if(_code_package.has_value()) {
      Json::Value meta;
      std::ifstream meta_file{*_code_package + "/package.json"};
      if(Json::Reader{}.parse(meta_file, meta) && meta["torch-home"].isString()) {
        std::filesystem::path rel{meta["torch-home"].asString()};
        if(rel.is_relative() && !rel.empty() && std::none_of(rel.begin(), rel.end(), [](const auto& c) { return c == ".."; })) {
          spec.env.emplace_back("TORCH_HOME", *_code_package + "/" + rel.string());
        } else {
          spdlog::warn("Ignoring invalid torch-home '{}' in {}/package.json", rel.string(), *_code_package);
        }
      }
    }

    return spec;
  }

  bool BareMetalExecutorPython::start(bool poll_sleep, int cpu_idx)
  {
    LaunchSpec spec = python_launch(poll_sleep, cpu_idx);

    // Inherit the orchestrator's environment, minus the variables the spec sets.
    std::vector<std::string> env_strings;
    for(char** env = environ; *env != nullptr; env++) {
      std::string_view entry{*env};
      std::string_view key = entry.substr(0, entry.find('='));
      bool overridden = std::any_of(spec.env.begin(), spec.env.end(), [&](const auto& kv) { return kv.first == key; });
      if(!overridden) {
        env_strings.emplace_back(entry);
      }
    }
    for(const auto& [key, val] : spec.env) {
      env_strings.push_back(key + "=" + val);
    }

    std::vector<char*> argv, envp;
    for(auto& a : spec.argv) argv.push_back(a.data());
    argv.push_back(nullptr);
    for(auto& e : env_strings) envp.push_back(e.data());
    envp.push_back(nullptr);

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

    int status = posix_spawnp(&_pid, argv[0], &file_actions, &attr, argv.data(), envp.data());

    posix_spawnattr_destroy(&attr);
    posix_spawn_file_actions_destroy(&file_actions);
    close(log_fd);

    if (status == 0) {
      spdlog::info("Child process spawned successfully, PID: {}, interpreter {}", _pid, spec.argv[0]);
    } else {
      spdlog::error("posix_spawn failed: {}", strerror(status));
      return false;
    }

    return true;
  }

  // iceoryx2's default root; bind-mounted at the same path into executor containers.
  static constexpr const char* ICEORYX2_ROOT = "/tmp/iceoryx2";

  static std::string _shell_quote(const std::string& arg)
  {
    std::string out = "'";
    for(char c : arg) {
      out += (c == '\'') ? std::string{"'\\''"} : std::string(1, c);
    }
    return out + "'";
  }

  /**
   * `<runtime> run -d --rm --user <uid>:<gid> [-v <pkg>:<pkg>:ro] <ipc mounts> [--label] -e ... <image> <argv>`
   * The code package is mounted read-only at its host path, so host paths stay valid inside.
   * Returns the container id, or "" on failure.
   */
  static std::string _start_container(
    const std::string& container_runtime,
    const std::string& image,
    const std::optional<std::string>& code_package,
    const std::vector<std::pair<std::string, std::string>>& env_vars,
    const std::vector<std::string>& cmd
  )
  {
    std::error_code ec;
    // A bind mount fails if its source is missing.
    std::filesystem::create_directories(ICEORYX2_ROOT, ec);

    std::vector<std::string> args = {
      container_runtime, "run", "-d", "--rm",
      "--user", fmt::format("{}:{}", getuid(), getgid())
    };

    // iceoryx2 events are unix datagrams; a fresh container netns has max_dgram_qlen=10, so bursts of
    // notifications overflow (FailedToDeliverSignal, lost wakeups). Use the host's value.
    static const long dgram_qlen = [] {
      long v = 512;
      if(FILE* f = fopen("/proc/sys/net/unix/max_dgram_qlen", "r")) {
        long x;
        if(fscanf(f, "%ld", &x) == 1 && x > 0) v = x;
        fclose(f);
      }
      return v;
    }();
    args.insert(args.end(), {"--sysctl", fmt::format("net.unix.max_dgram_qlen={}", dgram_qlen)});

    if(code_package.has_value()) {
      args.insert(args.end(), {"-v", fmt::format("{0}:{0}:ro", code_package.value())});
    }

    args.insert(args.end(), {"--mount", "type=bind,source=/dev/shm,target=/dev/shm"});
    args.insert(args.end(), {"--mount", fmt::format("type=bind,source={0},target={0}", ICEORYX2_ROOT)});

    const char* home = getenv("HOME");
    if(home && std::filesystem::is_directory(fmt::format("{}/.config/iceoryx2", home), ec)) {
      args.insert(args.end(), {"--mount", fmt::format("type=bind,source={}/.config/iceoryx2,target=/etc/iceoryx2", home)});
    }

    // Lets the test harness find and clean up containers of one test run.
    if(const char* test_id = getenv("MIGNIFICIENT_TEST_ID")) {
      args.insert(args.end(), {"--label", fmt::format("mignificient.test={}", test_id)});
    }

    for(const auto& [key, val] : env_vars) {
      args.insert(args.end(), {"-e", fmt::format("{}={}", key, val)});
    }

    args.push_back(image);
    args.insert(args.end(), cmd.begin(), cmd.end());

    std::string command;
    for(const auto& arg : args) {
      command += (command.empty() ? "" : " ") + _shell_quote(arg);
    }

    spdlog::info("Starting container executor: {}", command);

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

    std::vector<std::pair<std::string, std::string>> env_vars = {
      {"POLL_TYPE", poll_sleep ? "wait" : "poll"},
      {"EXECUTOR_TYPE", "shmem"},
      {"CONTAINER_NAME", _user},
      {"FUNCTION_HANDLER", _function_handler},
      {"FUNCTION_FILE", _function_path},
      {"CUDA_BINARY", _function_path},
      {"LD_PRELOAD", preload},
      {"IPC_BACKEND", ipc::IPCConfig::backend_string(_ipc_config.backend)},
      {"MIGNIFICIENT_MAX_GPU_MEMORY", std::to_string(_gpu_memory)},
      {"GPULESS_ELF_DEFINITION", _function_path + ".txt"}
    };

#ifdef MIGNIFICIENT_WITH_ICEORYX2
    env_vars.emplace_back("GPULESS_REQUEST_SIZE", std::to_string(_ipc_config.buffer_configs.at("gpuless-executor").request_size));
    env_vars.emplace_back("GPULESS_RESPONSE_SIZE", std::to_string(_ipc_config.buffer_configs.at("gpuless-executor").response_size));
    env_vars.emplace_back("GPULESS_QUEUE_CAPACITY", std::to_string(_ipc_config.buffer_configs.at("gpuless-executor").queue_capacity));
#endif

    if(cpu_idx != -1) {
      env_vars.emplace_back("CPU_BIND_IDX", std::to_string(cpu_idx));
    }

    _container_id = _start_container(_container_runtime, _image, _code_package, env_vars, {_cpp_executor});
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
    LaunchSpec spec = python_launch(poll_sleep, cpu_idx);

    _container_id = _start_container(_container_runtime, _image, _code_package, spec.env, spec.argv);
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
