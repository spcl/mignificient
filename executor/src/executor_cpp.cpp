
#include <chrono>
#include <string>

#include <dlfcn.h>
#include <sys/prctl.h>

#include <spdlog/spdlog.h>

#include <mignificient/executor/executor.hpp>

typedef int (*fptr)(mignificient::Invocation);

fptr load_function()
{
  std::string function_file{std::getenv("FUNCTION_FILE")};
  std::string function_name{std::getenv("FUNCTION_NAME")};

  void *handle = dlopen(function_file.c_str(), RTLD_NOW);
  if (handle == nullptr)
  {
      spdlog::error("Couldn't load the function file {}, error: {}!", function_file, dlerror());
      exit(EXIT_FAILURE);
  }

  fptr func = (fptr)dlsym(handle, function_name.c_str());
  if (!func)
  {
      spdlog::error("Couldn't load the function {}, error: {}!", function_name, dlerror());
      exit(EXIT_FAILURE);
  }

  return func;
}


int main(int argc, char **argv)
{
  std::string container_name{std::getenv("CONTAINER_NAME")};
  std::string ipc_backend{std::getenv("IPC_BACKEND")};

  mignificient::executor::Runtime runtime{mignificient::ipc::IPCConfig::convert_ipc_backend(ipc_backend), container_name};
  runtime.register_runtime();

  // Get killed on parent's death
  prctl(PR_SET_PDEATHSIG, SIGHUP);

  fptr func = nullptr;

  int cpu = sched_getcpu();
  spdlog::info("Running on CPU: {}", cpu);

  while(true) {

    auto invocation_data = runtime.loop_wait();

    if(invocation_data.size == 0) {
      std::cerr << "Empty payload, quit" << std::endl;
      break;
    }

    if(!func) {
      func = load_function();
    }

    std::string_view input{reinterpret_cast<const char*>(invocation_data.data), invocation_data.size};
    spdlog::info("Invoke, data size {}, input string {}", invocation_data.size, input);

    auto start = std::chrono::high_resolution_clock::now();

    int size = func({runtime, std::move(invocation_data), runtime.result()});

    auto end = std::chrono::high_resolution_clock::now();

    //runtime.gpu_yield();

    runtime.finish(size);
    spdlog::info("Finished invocation {}", std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() / 1000.0);

  }

  // FIXME
  //dlclose(handle);

  return 0;
}
