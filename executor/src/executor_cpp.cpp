
#include <chrono>
#include <string>

#include <dlfcn.h>
#include <sys/prctl.h>

#include <spdlog/spdlog.h>

#include <mignificient/executor/executor.hpp>

int main(int argc, char **argv)
{

  std::string function_file{std::getenv("FUNCTION_FILE")};
  std::string function_name{std::getenv("FUNCTION_NAME")};
  std::string container_name{std::getenv("CONTAINER_NAME")};

  mignificient::executor::Runtime runtime{container_name};
  runtime.register_runtime();

  prctl(PR_SET_PDEATHSIG, SIGHUP);

  typedef int (*fptr)(mignificient::Invocation);
  fptr func;

  void *handle = dlopen(function_file.c_str(), RTLD_NOW);
  if (handle == nullptr)
  {
      spdlog::error("Couldn't load the function file {}, error: {}!", function_file, dlerror());
      exit(EXIT_FAILURE);
  }

  func = (fptr)dlsym(handle, function_name.c_str());
  if (!func)
  {
      spdlog::error("Couldn't load the function {}, error: {}!", function_name, dlerror());
      exit(EXIT_FAILURE);
  }

  while(true) {

    auto invocation_data = runtime.loop_wait();

    if(invocation_data.size == 0) {
      std::cerr << "Empty payload, quit" << std::endl;
      break;
    }

    std::string_view input{reinterpret_cast<const char*>(invocation_data.data), invocation_data.size};
    spdlog::info("Invoke, data size {}, input string {}", invocation_data.size, input);

    auto start = std::chrono::high_resolution_clock::now();

    int size = func({runtime, std::move(invocation_data), runtime.result()});

    auto end = std::chrono::high_resolution_clock::now();

    //runtime.gpu_yield();

    //runtime.result().size = 10;

    runtime.finish(size);
    spdlog::info("Finished invocation ");

  }

  dlclose(handle);

  return 0;
}
