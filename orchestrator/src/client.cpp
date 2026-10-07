#include <chrono>

#include <mignificient/orchestrator/client.hpp>

#include <mignificient/orchestrator/device.hpp>
#include <mignificient/orchestrator/orchestrator.hpp>

namespace mignificient { namespace orchestrator {

#ifdef MIGNIFICIENT_WITH_ICEORYX2
  CommunicationIceoryxV2::CommunicationIceoryxV2(const std::string& id)
  {
    auto& node = Orchestrator::iceoryx_node_v2();
    {
      auto exec_send_service = node.service_builder(
          iox2::ServiceName::create(fmt::format("{}.Orchestrator.Client.Send", id).c_str()).value())
      .publish_subscribe<mignificient::executor::Invocation>()
      .max_publishers(1)
      .max_subscribers(1)
      .open_or_create().value();

      auto pub_result = exec_send_service.publisher_builder().create();
      if (pub_result.has_value()) {
        client_send = std::move(pub_result.value());
      }

      auto exec_pub_service = node.service_builder(
          iox2::ServiceName::create(fmt::format("{}.Orchestrator.Client.Recv", id).c_str()).value())
      .publish_subscribe<mignificient::executor::InvocationResult>()
      .max_publishers(1)
      .max_subscribers(1)
      .open_or_create().value();

      auto sub_result = exec_pub_service.subscriber_builder().create();
      if (sub_result.has_value()) {
        client_recv = std::move(sub_result.value());
      }

      {
        auto exec_event_service = node.service_builder(
            iox2::ServiceName::create(fmt::format("{}.Orchestrator.Client.Notify", id).c_str()).value())
        .event().open_or_create();
        if (exec_event_service.has_value()) {
          client_event_notify = std::move(exec_event_service.value());
        }
      }

      {
        auto exec_event_service = node.service_builder(
            iox2::ServiceName::create(fmt::format("{}.Orchestrator.Client.Listen", id).c_str()).value())
        .event().open_or_create();
        if (exec_event_service.has_value()) {
          client_event_listen = std::move(exec_event_service.value());
        }
      }

      client_listener = client_event_listen->listener_builder().create().value();
      client_notifier = client_event_notify->notifier_builder().create().value();
      client_payload = client_send.value().loan_uninit().value();
    }

    {
      {
        auto exec_event_service = node.service_builder(
            iox2::ServiceName::create(fmt::format("{}.Orchestrator.Gpuless.Notify", id).c_str()).value())
        .event().open_or_create();
        if (exec_event_service.has_value()) {
          gpuless_event_notify = std::move(exec_event_service.value());
        }
      }
      {
        auto exec_event_service = node.service_builder(
            iox2::ServiceName::create(fmt::format("{}.Orchestrator.Gpuless.Listen", id).c_str()).value())
        .event().open_or_create();
        if (exec_event_service.has_value()) {
          gpuless_event_listen = std::move(exec_event_service.value());
        }
      }

      gpuless_listener = gpuless_event_listen->listener_builder().create().value();
      gpuless_notifier = gpuless_event_notify->notifier_builder().create().value();

      {
        auto swap_result_service = node.service_builder(
            iox2::ServiceName::create(fmt::format("{}.Orchestrator.Gpuless.SwapResult", id).c_str()).value())
        .publish_subscribe<mignificient::executor::SwapResult>()
        .max_publishers(1)
        .max_subscribers(1)
        .open_or_create().value();

        gpuless_swap_recv = std::move(swap_result_service.subscriber_builder().create().value());
      }
    }

  }
#endif

  void Client::finished(std::string_view response, int32_t status)
  {
    _status = ClientStatus::NOT_ACTIVE;

    // FIXME: Is the finished_invocation really necessary? When can this happen in practice?
    if(_finished_invocation) {
      auto tmp = std::move(_finished_invocation);
      _finished_invocation = nullptr;
      gpu_instance()->finish_current_invocation(tmp.get());
      if(status != 0) tmp->respond_error(status); else tmp->respond(response);
    } else {
      auto tmp = std::move(_active_invocation);
      _active_invocation = nullptr;
      gpu_instance()->finish_current_invocation(tmp.get());
      if(status != 0) tmp->respond_error(status); else tmp->respond(response);
    }
  }

  void Client::yield()
  {
    gpu_instance()->yield_current_invocation();

    _status = ClientStatus::NOT_ACTIVE;
  }

  void Client::oom_kill()
  {
    auto kill_start = std::chrono::high_resolution_clock::now();

    _status = ClientStatus::NOT_ACTIVE;

    // Kill executor (process or container)
    _executor->stop();

    // Gpuless should be exiting on its own; give it a moment, then force kill
    pid_t gpuless_pid = _gpuless_server.pid();
    if (gpuless_pid > 0 && waitpid(gpuless_pid, nullptr, WNOHANG) == 0) {
      // Not yet exited, force kill
      kill(gpuless_pid, SIGKILL);
      waitpid(gpuless_pid, nullptr, 0);
    }
    gpu_instance()->remove_pending_invocations(this);

    auto kill_end = std::chrono::high_resolution_clock::now();
    double kill_time_us = std::chrono::duration<double, std::micro>(kill_end - kill_start).count();
    spdlog::info("[KillStats] oom_kill for {}: {:.1f} us ({:.3f} ms)",
                 _id, kill_time_us, kill_time_us / 1000.0);

    // Respond with OOM error to active invocation
    if (_active_invocation) {
      _active_invocation->respond_oom();
      auto tmp = std::move(_active_invocation);
      _active_invocation = nullptr;
      gpu_instance()->finish_current_invocation(tmp.get());
    }

    // Respond with OOM error to finished invocation waiting for HTTP reply
    if (_finished_invocation) {
      _finished_invocation->respond_oom();
      _finished_invocation = nullptr;
    }

    // Drain pending invocations with OOM error
    while (!_pending_invocations.empty()) {
      auto inv = std::move(_pending_invocations.front());
      _pending_invocations.pop();
      inv->respond_oom();
    }

    // Unregister executor from GPU instance
    gpu_instance()->close_executor(_executor.get());
  }

  void Client::timeout_kill()
  {
    _kill("timeout_kill", [](ActiveInvocation& inv) { inv.respond_timeout(); });
  }

  void Client::fail_pending(const std::string& reason)
  {
    _kill("fail_pending", [&](ActiveInvocation& inv) { inv.failure(reason); });
  }

  std::optional<std::string> Client::startup_failure(std::chrono::milliseconds timeout)
  {
    if(_startup_error) {
      return _startup_error;
    }
    if(!_gpuless_active && _gpuless_server.exited()) {
      return "executor failed to start: gpuless server exited before registering";
    }
    if(!_executor_active && _executor->exited()) {
      return "executor failed to start: executor exited before registering";
    }
    if(std::chrono::high_resolution_clock::now() - _spawn_time > timeout) {
      return fmt::format(
        "executor failed to start: {} not registered after {} ms",
        !_executor_active ? "executor" : "gpuless server", timeout.count()
      );
    }
    return std::nullopt;
  }

  void Client::_kill(const char* what, const std::function<void(ActiveInvocation&)>& reply)
  {
    auto kill_start = std::chrono::high_resolution_clock::now();

    _status = ClientStatus::NOT_ACTIVE;

    // Kill gpuless server (always bare-metal) and executor
    pid_t gpuless_pid = _gpuless_server.pid();
    if (gpuless_pid > 0) {
      kill(gpuless_pid, SIGKILL);
      waitpid(gpuless_pid, nullptr, 0);
    }
    _executor->stop();
    gpu_instance()->remove_pending_invocations(this);

    auto kill_end = std::chrono::high_resolution_clock::now();
    double kill_time_us = std::chrono::duration<double, std::micro>(kill_end - kill_start).count();
    spdlog::info("[KillStats] {} for {}: {:.1f} us ({:.3f} ms)",
                 what, _id, kill_time_us, kill_time_us / 1000.0);

    // Reply to the active invocation
    if (_active_invocation) {
      reply(*_active_invocation);
      auto tmp = std::move(_active_invocation);
      _active_invocation = nullptr;
      gpu_instance()->finish_current_invocation(tmp.get());
    }

    // Reply to the finished invocation waiting for HTTP reply
    if (_finished_invocation) {
      reply(*_finished_invocation);
      _finished_invocation = nullptr;
    }

    // Drain pending invocations
    // TODO: in future, we might want to allocate a new container for them
    while (!_pending_invocations.empty()) {
      auto inv = std::move(_pending_invocations.front());
      _pending_invocations.pop();
      reply(*inv);
    }

    // Unregister executor from GPU instance
    gpu_instance()->close_executor(_executor.get());

    // Our queued invocations may have blocked others on this GPU.
    if (!gpu_instance()->is_busy() && gpu_instance()->pending_invocations() > 0) {
      gpu_instance()->schedule_next();
    }
  }

}}
