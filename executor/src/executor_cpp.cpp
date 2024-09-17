
#include <dlfcn.h>
#include <spdlog/spdlog.h>
#include <string>

#include <mignificient/executor/executor.hpp>

int main(int argc, char **argv) {

  std::string function_file{std::getenv("FUNCTION_FILE")};
  std::string function_name{std::getenv("FUNCTION_NAME")};
  std::string container_name{std::getenv("CONTAINER_NAME")};

  mignificient::executor::Runtime runtime{container_name};

  typedef size_t (*fptr)(mignificient::Invocation);
  fptr func;

  void *handle = dlopen(function_file.c_str(), RTLD_NOW);
  if (handle == nullptr)
  {
      std::cout << dlerror() << std::endl;
      exit(EXIT_FAILURE);
  }

  func = (fptr)dlsym(handle, function_name.c_str());
  if (!func)
  {
      std::cout << dlerror() << std::endl;
      exit(EXIT_FAILURE);
  }

  while(true) {

    auto invocation_data = runtime.loop_wait();

    if(invocation_data.size == 0) {
      std::cerr << "Empty payload, quit" << std::endl;
      break;
    }
    spdlog::info("Invoke, data size {}, first element {}", invocation_data.size, invocation_data.data[0]);

    size_t size = func({runtime, std::move(invocation_data), runtime.result()});

    runtime.gpu_yield();

    //runtime.result().size = 10;
    std::string_view res{"{ \"test\": 42 }"};

    std::copy_n(res.data(), res.length(), reinterpret_cast<char*>(runtime.result().data));
    //runtime.result().size = res.length();

    runtime.finish(res.length());

  }

  dlclose(handle);

  return 0;
}
