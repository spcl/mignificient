
#include <mignificient/executor/executor.hpp>
#include <mignificient/executor/function.hpp>

namespace mignificient {

  void Invocation::gpu_yield()
  {
    runtime.gpu_yield();
  }

}
