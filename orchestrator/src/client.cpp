#include <mignificient/orchestrator/client.hpp>

#include <mignificient/orchestrator/device.hpp>

namespace mignificient { namespace orchestrator {

  void Client::finished(std::string_view response)
  {

    if(_finished_invocation) {
      gpu_instance()->finish_current_invocation(_finished_invocation.get());
      _finished_invocation->respond(response);
      _finished_invocation = nullptr;
    } else {
      gpu_instance()->finish_current_invocation(_active_invocation.get());
      _active_invocation->respond(response);
      _active_invocation = nullptr;
    }

    _status = ClientStatus::NOT_ACTIVE;
  }

  void Client::yield()
  {
    gpu_instance()->yield_current_invocation();

    _status = ClientStatus::NOT_ACTIVE;
  }

}}
