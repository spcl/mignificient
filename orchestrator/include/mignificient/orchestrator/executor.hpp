#ifndef __MIGNIFICIENT_ORCHESTRATOR_EXECUTOR_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_EXECUTOR_HPP__

#include <memory>
#include <string>

#include <sched.h>

namespace mignificient { namespace orchestrator {

  class GPUInstance;

  class GPUlessServer {

  };

  class Executor {
  public:
      Executor(const std::string& user, const std::string& function, GPUInstance& device):
        _user(user),
        _pid(0),
        _function(function),
        _device(device)
      {}

      virtual ~Executor() = default;

      void start()
      {
        // spawn!
      }

      const std::string& user() const
      {
        return _user;
      }

      pid_t pid() const
      {
        return _pid;
      }

  private:
      std::string _user;
      pid_t _pid;
      std::string _function;
      GPUInstance& _device;
  };

  class BareMetalExecutor : public Executor {
  public:
      using Executor::Executor;
  };

  class SarusContainerExecutor : public Executor {
  public:
      using Executor::Executor;
  };

}}

#endif
