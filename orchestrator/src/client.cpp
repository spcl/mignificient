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

}}
