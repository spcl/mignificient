
#include "executor.hpp"
#include "function.hpp"

namespace mignificient {

  void Invocation::gpu_yield()
  {
    runtime.gpu_yield();
  }

}
