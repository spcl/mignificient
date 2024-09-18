#ifndef __MIGNIFICIENT_ORCHESTRATOR_EXECUTOR_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_EXECUTOR_HPP__

#include <memory>
#include <string>

#include <sched.h>

#include <mignificient/orchestrator/device.hpp>

namespace mignificient { namespace orchestrator {

  class GPUlessServer {

  };

  class Executor {
  public:
      Executor(std::string userName, pid_t pid, std::string functionName, std::shared_ptr<GPUDevice> boundDevice)
          : userName_(std::move(userName)), pid_(pid), functionName_(std::move(functionName)), boundDevice_(std::move(boundDevice)) {}

      virtual ~Executor() = default;

      const std::string& getUserName() const { return userName_; }

      pid_t getPID() const { return pid_; }

      const std::string& getFunctionName() const { return functionName_; }

      std::shared_ptr<GPUDevice> getBoundDevice() const { return boundDevice_; }

  private:
      std::string _userName;
      pid_t _pid;
      std::string _functionName;
      std::shared_ptr<GPUDevice> _boundDevice;
  };

  class BareMetalExecutor : public Executor {
  public:
      using Executor::Executor;
  };

  class SarusContainerExecutor : public Executor {
  public:
      using Executor::ActiveExecutor;
  };

}}

#endif
