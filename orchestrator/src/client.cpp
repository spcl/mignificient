#include <mignificient/orchestrator/client.hpp>

#include <mignificient/orchestrator/device.hpp>

namespace mignificient { namespace orchestrator {

  void Client::finished(std::string_view response)
  {
    _status = ClientStatus::NOT_ACTIVE;

    // FIXME: Is the finished_invocation really necessary? When can this happen in practice?
    if(_finished_invocation) {
      auto tmp = std::move(_finished_invocation);
      _finished_invocation = nullptr;
      gpu_instance()->finish_current_invocation(tmp.get());
      tmp->respond(response);
    } else {
      auto tmp = std::move(_active_invocation);
      _active_invocation = nullptr;
      gpu_instance()->finish_current_invocation(tmp.get());
      tmp->respond(response);
    }
  }

  void Client::yield()
  {
    gpu_instance()->yield_current_invocation();

    _status = ClientStatus::NOT_ACTIVE;
  }

}}
