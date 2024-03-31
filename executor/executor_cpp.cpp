
#include <spdlog/spdlog.h>
#include <string>

#include "executor.hpp"

int main(int argc, char **argv) {

  std::string function_file{std::getenv("FUNCTION_FILE")};
  std::string function_name{std::getenv("FUNCTION_NAME")};
  std::string container_name{std::getenv("CONTAINER_NAME")};

  mignificient::executor::Runtime runtime{container_name};

  while(true) {

    auto invocation_data = runtime.loop_wait();

    if(invocation_data.size == 0) {
      std::cerr << "Empty payload, quit" << std::endl;
      break;
    }

    spdlog::info("Invoke, data size {}, first element {}", invocation_data.size, invocation_data.data[0]);

    runtime.result().size = 10;
    std::string_view res{"{ \"test\": 42 }"};

    std::copy_n(res.data(), res.length(), reinterpret_cast<char*>(runtime.result().data.data()));
    runtime.result().size = res.length();

    runtime.finish();

  }

  return 0;
}
