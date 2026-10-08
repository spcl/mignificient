#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <sys/stat.h>

#include <mignificient/orchestrator/client.hpp>
#ifdef MIGNIFICIENT_WITH_ICEORYX2
#include <mignificient/executor/iox2_config.hpp>
#endif

#include <mignificient/orchestrator/device.hpp>
#include <mignificient/orchestrator/orchestrator.hpp>

namespace mignificient { namespace orchestrator {

#ifdef MIGNIFICIENT_WITH_ICEORYX2
  CommunicationIceoryxV2::CommunicationIceoryxV2(const std::string& id, const std::string& root)
  {
    auto node_result = iox2::NodeBuilder().config(iox2_config(root.c_str())).create<iox2::ServiceType::Ipc>();
    if(!node_result.has_value()) {
      spdlog::error("Failed to create iceoryx2 node for client {} in {}: {}", id, root, static_cast<uint64_t>(node_result.error()));
      abort();
    }
    this->node = std::move(node_result.value());
    auto& node = this->node.value();
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

  void Client::_create_iox2_root()
  {
    std::error_code ec;
    // Left over by an orchestrator that didn't shut down cleanly (client ids restart at 0).
    if(std::filesystem::exists(_iox2_root, ec)) {
      spdlog::warn("Removing stale iceoryx2 directory {}", _iox2_root);
      std::filesystem::remove_all(_iox2_root, ec);
    }
    // Fatal like the node creation right after it: iceoryx2 would otherwise create the directory itself,
    // with its default permissions.
    if(mkdir(_iox2_root.c_str(), 0700) != 0) {
      spdlog::critical("Failed to create iceoryx2 directory {}: {}", _iox2_root, strerror(errno));
      abort();
    }
  }

  Client::~Client()
  {
    // A no-op after a kill; on orchestrator shutdown it stops the processes and containers still running
    // (container stops are queued to the worker, which runs them before it exits).
    _gpuless_server.stop();
    if(_executor) {
      _executor->stop();
    }
    _cgroup.remove();
#ifdef MIGNIFICIENT_WITH_ICEORYX2
    if(_comm_v2) {
      // Close our ports and node first, then drop everything the client's processes left behind.
      _comm_v2.reset();
      std::error_code ec;
      std::filesystem::remove_all(_iox2_root, ec);
      if(ec) {
        spdlog::error("Failed to remove iceoryx2 directory {}: {}", _iox2_root, ec.message());
      }
    }
#endif
  }

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
    if(gpu_instance()->yield_current_invocation(this)) {
      _status = ClientStatus::NOT_ACTIVE;
    }
  }

  void Client::oom_kill()
  {
    // Same as the other kills (container executors are stopped through the worker), with an OOM reply.
    _kill("oom_kill", [](ActiveInvocation& inv) { inv.respond_oom(); });
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
    // After registration too: a crashed server (e.g. SIGKILL) leaves its iceoryx2 ports behind, and the
    // executor would wait for answers until the function timeout.
    if(_gpuless_server.exited()) {
      if(!_gpuless_active) {
        return "executor failed to start: gpuless server exited before registering";
      }
      if(_gpuless_server.crashed()) {
        return "gpuless server crashed";
      }
    }
    if(_executor->exited()) {
      return _executor_active ? "executor exited" : "executor failed to start: executor exited before registering";
    }
    if(!is_active() && std::chrono::high_resolution_clock::now() - _spawn_time > timeout) {
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
    _gpuless_server.stop();
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
