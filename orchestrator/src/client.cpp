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

  void Client::oom_kill()
  {
    auto kill_start = std::chrono::high_resolution_clock::now();

    _status = ClientStatus::NOT_ACTIVE;

    // Kill executor
    kill(_executor->pid(), SIGKILL);

    // Gpuless should be exiting on its own; give it a moment, then force kill
    int status;
    pid_t result = waitpid(_gpuless_server.pid(), &status, WNOHANG);
    if (result == 0) {
      // Not yet exited, force kill
      kill(_gpuless_server.pid(), SIGKILL);
      waitpid(_gpuless_server.pid(), nullptr, 0);
    }
    waitpid(_executor->pid(), nullptr, 0);

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
    gpu_instance()->close_executor(_executor->pid());
  }

  void Client::timeout_kill()
  {
    auto kill_start = std::chrono::high_resolution_clock::now();

    _status = ClientStatus::NOT_ACTIVE;

    // Kill gpuless server and executor processes
    kill(_gpuless_server.pid(), SIGKILL);
    kill(_executor->pid(), SIGKILL);
    waitpid(_gpuless_server.pid(), nullptr, 0);
    waitpid(_executor->pid(), nullptr, 0);

    auto kill_end = std::chrono::high_resolution_clock::now();
    double kill_time_us = std::chrono::duration<double, std::micro>(kill_end - kill_start).count();
    spdlog::info("[KillStats] timeout_kill for {}: {:.1f} us ({:.3f} ms)",
                 _id, kill_time_us, kill_time_us / 1000.0);

    // Respond with timeout error to active invocation
    if (_active_invocation) {
      _active_invocation->respond_timeout();
      auto tmp = std::move(_active_invocation);
      _active_invocation = nullptr;
      gpu_instance()->finish_current_invocation(tmp.get());
    }

    // Respond with timeout error to finished invocation waiting for HTTP reply
    if (_finished_invocation) {
      _finished_invocation->respond_timeout();
      _finished_invocation = nullptr;
    }

    // Drain pending invocations with timeout error
    // TODO: in future, we might want to allocate a new container for them
    while (!_pending_invocations.empty()) {
      auto inv = std::move(_pending_invocations.front());
      _pending_invocations.pop();
      inv->respond_timeout();
    }

    // Unregister executor from GPU instance
    gpu_instance()->close_executor(_executor->pid());
  }

}}
