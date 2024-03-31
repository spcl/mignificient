
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

    std::cerr << "invoke with payload size " << invocation_data.size << std::endl;

  }

  return 0;
}
