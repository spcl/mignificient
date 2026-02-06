#ifndef __MIGNIFICIENT_ORCHESTRATOR_CLIENT_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_CLIENT_HPP__

#include <optional>
#include <queue>

#include <iceoryx_posh/popo/subscriber.hpp>
#include <iceoryx_posh/popo/publisher.hpp>
#include <iceoryx_posh/popo/untyped_publisher.hpp>

#include <mignificient/executor/executor.hpp>
#include <mignificient/orchestrator/event.hpp>
#include <mignificient/orchestrator/executor.hpp>
#include <mignificient/orchestrator/invocation.hpp>
#include <mignificient/ipc/config.hpp>
#include <mignificient/ipc/types.hpp>

#ifdef MIGNIFICIENT_WITH_ICEORYX2
#include <iox2/iceoryx2.hpp>
#include <iox2/service_type.hpp>
#include <iox2/waitset.hpp>
#endif

namespace mignificient { namespace orchestrator {

  enum class ClientStatus
  {
    NOT_ACTIVE,
    CPU_ONLY,
    MEMCPY,
    FULL
  };

  struct CommunicationIceoryxV1 {

    iox::popo::Publisher<mignificient::executor::Invocation> _send;
    iox::popo::Subscriber<mignificient::executor::InvocationResult> _recv;
    iox::popo::Publisher<int> _gpuless_send;
    iox::popo::Subscriber<int> _gpuless_recv;
    iox::popo::Sample<int, iox::mepoo::NoUserHeader> _gpuless_payload;
    iox::popo::Sample<mignificient::executor::Invocation, iox::mepoo::NoUserHeader> _payload;

    CommunicationIceoryxV1(const std::string& id):
      _send(
        iox::capro::ServiceDescription{
          iox::RuntimeName_t{iox::TruncateToCapacity_t{}, id.c_str()},
          "Orchestrator",
          "Receive"
        }
      ),
      _recv(
        iox::capro::ServiceDescription{
          iox::RuntimeName_t{iox::TruncateToCapacity_t{}, id.c_str()},
          "Orchestrator",
          "Send"
        }
      ),
      _gpuless_send(
        iox::capro::ServiceDescription{
          iox::RuntimeName_t{iox::TruncateToCapacity_t{}, fmt::format("gpuless-{}", id).c_str()},
          "Orchestrator",
          "Receive"
        }
      ),
      _gpuless_recv(
        iox::capro::ServiceDescription{
          iox::RuntimeName_t{iox::TruncateToCapacity_t{}, fmt::format("gpuless-{}", id).c_str()},
          "Orchestrator",
          "Send"
        }
      ),
      _payload(
        _send.loan().value()
      ),
      _gpuless_payload(
        _gpuless_send.loan().value()
      )
    {}

  };

#ifdef MIGNIFICIENT_WITH_ICEORYX2
  struct CommunicationIceoryxV2
  {
    std::optional<iox2::PortFactoryEvent<iox2::ServiceType::Ipc>> client_event_notify;
    std::optional<iox2::PortFactoryEvent<iox2::ServiceType::Ipc>> client_event_listen;
    std::optional<iox2::Publisher<iox2::ServiceType::Ipc, mignificient::executor::Invocation, void>> client_send;
    std::optional<iox2::Subscriber<iox2::ServiceType::Ipc, mignificient::executor::InvocationResult, void>> client_recv;
    std::optional<iox2::Listener<iox2::ServiceType::Ipc>> client_listener;
    std::optional<iox2::Notifier<iox2::ServiceType::Ipc>> client_notifier;
    std::optional<iox2::WaitSetGuard<iox2::ServiceType::Ipc>> client_listener_guard;

    std::optional<iox2::PortFactoryEvent<iox2::ServiceType::Ipc>> gpuless_event_notify;
    std::optional<iox2::PortFactoryEvent<iox2::ServiceType::Ipc>> gpuless_event_listen;
    std::optional<iox2::Notifier<iox2::ServiceType::Ipc>> gpuless_notifier;
    std::optional<iox2::Listener<iox2::ServiceType::Ipc>> gpuless_listener;
    std::optional<iox2::WaitSetGuard<iox2::ServiceType::Ipc>> gpuless_listener_guard;

    std::optional<iox2::SampleMutUninit<iox2::ServiceType::Ipc, mignificient::executor::Invocation, void>> client_payload;

    CommunicationIceoryxV2(const std::string& id);

  };
#endif

  struct Client
  {

    Client(ipc::IPCBackend backend, const std::string& id, const std::string& fname,
          const ipc::BufferConfig& executor_buf_config = {},
          const ipc::BufferConfig& gpuless_buf_config = {}
    ):
      _id(id),
      _fname(fname),
      _ipc_backend(backend),
      _executor_buffer_config(executor_buf_config),
      _gpuless_buffer_config(gpuless_buf_config),
      _busy(false),
      _event_context{EventSource::CLIENT, this}
    {
      if(backend == ipc::IPCBackend::ICEORYX_V1) {
        _comm_v1.emplace(id);
#ifdef MIGNIFICIENT_WITH_ICEORYX2
      } else if(backend == ipc::IPCBackend::ICEORYX_V2) {
        _comm_v2.emplace(id);
#endif
      } else {
        abort();
      }

    }

    executor::Invocation& request()
    {
      if(_ipc_backend == ipc::IPCBackend::ICEORYX_V1) {
        return *_comm_v1.value()._payload;
#ifdef MIGNIFICIENT_WITH_ICEORYX2
      } else if(_ipc_backend == ipc::IPCBackend::ICEORYX_V2) {
        return _comm_v2.value().client_payload->payload_mut();
#endif
      } else {
        abort();
      }
    }

    const std::string& get_name() const { return _id; }

    iox::popo::Subscriber<mignificient::executor::InvocationResult>& subscriber_v1()
    {
      return _comm_v1.value()._recv;
    }

#ifdef MIGNIFICIENT_WITH_ICEORYX2
    iox2::Subscriber<iox2::ServiceType::Ipc, mignificient::executor::InvocationResult, void>& subscriber_v2()
    {
      return _comm_v2->client_recv.value();
    }
#endif

    iox::popo::Subscriber<int>& gpuless_subscriber()
    {
      return _comm_v1.value()._gpuless_recv;
    }

#ifdef MIGNIFICIENT_WITH_ICEORYX2

    std::array<iox2::WaitSetAttachmentId<iox2::ServiceType::Ipc>, 2> init_v2(iox2::WaitSet<iox2::ServiceType::Ipc>& waitset)
    {
      auto res = waitset.attach_notification(_comm_v2->client_listener.value());
      if(!res.has_value()) {
        spdlog::error("Failed to attach client executor event to waitset: {}", static_cast<uint64_t>(res.error()));
        abort();
      }
      _comm_v2->client_listener_guard = std::move(res.value());

      {
        auto res = waitset.attach_notification(_comm_v2->gpuless_listener.value());
        if(!res.has_value()) {
          spdlog::error("Failed to attach client gpuless event to waitset: {}", static_cast<uint64_t>(res.error()));
          abort();
        }
        _comm_v2->gpuless_listener_guard = std::move(res.value());
      }

      return std::array<iox2::WaitSetAttachmentId<iox2::ServiceType::Ipc>, 2>{
        iox2::WaitSetAttachmentId<iox2::ServiceType::Ipc>::from_guard(*_comm_v2->client_listener_guard),
        iox2::WaitSetAttachmentId<iox2::ServiceType::Ipc>::from_guard(*_comm_v2->gpuless_listener_guard)
      };
    }

    void uninit_v2()
    {
      _comm_v2->client_listener_guard.reset();
      _comm_v2->gpuless_listener_guard.reset();
    }

    bool read_event_client_v2()
    {
      auto res = _comm_v2->client_listener->try_wait_one();
      if(res.has_value()) {
        if(res.value().has_value()) {
          return true;
        } else {
          return false;
        }
      } else {
        spdlog::error("Failed to read executor event for client {}, error {}", _id, res.error());
        return false;
      }
    }

    std::optional<size_t> read_event_gpuless_v2()
    {
      auto res = _comm_v2->gpuless_listener->try_wait_one();
      if(res.has_value()) {
        if(res.value().has_value()) {
          return res.value()->as_value();
        } else {
          return std::nullopt;
        }
      } else {
        spdlog::error("Failed to read gpuless event for client {}, error {}", _id, res.error());
        return std::nullopt;
      }
    }
#endif

    void send_gpuless_msg(GPUlessMessage msg)
    {
      if (_ipc_backend == ipc::IPCBackend::ICEORYX_V1) {

        auto& comm_backend = _comm_v1.value();
        *comm_backend._gpuless_payload.get() = static_cast<int>(msg);
        comm_backend._gpuless_send.publish(std::move(comm_backend._gpuless_payload));
        comm_backend._gpuless_payload = std::move(comm_backend._gpuless_send.loan().value());
      }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
      else if (_ipc_backend == ipc::IPCBackend::ICEORYX_V2) {

        auto& comm_backend = _comm_v2.value();

        comm_backend.gpuless_notifier->notify_with_custom_event_id(iox2::EventId{static_cast<int>(msg)});
      }
#endif
      else {
        abort();
      }

    }

    Context* context()
    {
      return &_event_context;
    }

    const std::string& fname() const
    {
      return _fname;
    }

    const std::string& id() const
    {
      return _id;
    }

    void set_busy(bool busy)
    {
      _busy = busy;
    }

    bool is_busy() const
    {
      return _busy;
    }

    ClientStatus status() const
    {
      return _status;
    }

    bool is_active() const
    {
      return _active;
    }

    GPUInstance* gpu_instance() const
    {
      assert(_gpu_instance);
      return _gpu_instance;
    }

    void set_gpuless_server(GPUlessServer&& server, GPUInstance* gpu_instance)
    {
      _gpuless_server = server;
      _gpu_instance = gpu_instance;
    }

    void set_executor(std::unique_ptr<Executor> executor)
    {
      _executor = std::move(executor);
    }

    void add_invocation(std::unique_ptr<ActiveInvocation> && invoc)
    {
      SPDLOG_DEBUG("[Client] For client {} add a new invocation {}", _id, invoc->uuid());
      _pending_invocations.push(std::move(invoc));
    }

    void finished(std::string_view response);

    void yield();

    void activate_memcpy()
    {
      SPDLOG_DEBUG("[Client] Activate gpuless memcpy for {}", _id);
      send_gpuless_msg(GPUlessMessage::MEMCPY_ONLY);

      _status = ClientStatus::MEMCPY;
    }

    void activate_kernels()
    {
      SPDLOG_DEBUG("[Client] Activate gpuless kernel exec for {}", _id);
      send_gpuless_msg(GPUlessMessage::FULL_EXEC);

      _status = ClientStatus::FULL;
    }

    void send_request()
    {
      if(_active_invocation) {
        _finished_invocation = std::move(_active_invocation);
      }

      _active_invocation = std::move(_pending_invocations.front());
      _pending_invocations.pop();

      strncpy(request().id, std::to_string(_invoc_idx++).c_str(), executor::Invocation::ID_LEN);
      std::copy_n(_active_invocation->input().begin(), _active_invocation->input().size(), request().data);
      request().size = _active_invocation->input().size();

      if (_ipc_backend == ipc::IPCBackend::ICEORYX_V1) {

        SPDLOG_DEBUG("[Client] Activate gpuless executor for {}", _id);
        auto& comm_backend = _comm_v1.value();

        comm_backend._send.publish(std::move(comm_backend._payload));
        comm_backend._payload = comm_backend._send.loan().value();

      }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
      else if (_ipc_backend == ipc::IPCBackend::ICEORYX_V2) {

        auto& comm_backend = _comm_v2.value();

        auto initialized_sample = iox2::assume_init(std::move(comm_backend.client_payload.value()));

        SPDLOG_DEBUG("[Client] Activate gpuless executor for {}", _id);
        {
          auto res = iox2::send(std::move(initialized_sample));
          if(!res.has_value()) {
            spdlog::error("Failed to send executor message: {}", static_cast<uint64_t>(res.error()));
          }
        }

        {
          auto res = comm_backend.client_notifier->notify();
          if(!res.has_value()) {
            spdlog::error("Failed to send executor notification: {}", static_cast<uint64_t>(res.error()));
          }
        }

        comm_backend.client_payload = comm_backend.client_send.value().loan_uninit().value();
      }
#endif
      else {
        abort();
      }

      SPDLOG_DEBUG("[Client] Activate gpuless basic exec for {}", _id);
      send_gpuless_msg(GPUlessMessage::BASIC_EXEC);

      _status = ClientStatus::CPU_ONLY;
    }

    bool executor_active()
    {
      _executor_active = true;

      _active = _gpuless_active && _executor_active;
      return _active;
    }

    bool gpuless_active()
    {
      _gpuless_active = true;

      _active = _gpuless_active && _executor_active;
      return _active;
    }

  private:
    Context _event_context;

    int _invoc_idx = 0;
    bool _busy = false;
    bool _executor_active = false;
    bool _gpuless_active = false;
    std::atomic<bool> _active = false;

    std::string _id;
    std::string _fname;
    ipc::IPCBackend _ipc_backend;  // IPC backend selection (v1 or v2)
    ipc::BufferConfig _executor_buffer_config;  // Buffer sizes for orchestrator-executor channel
    ipc::BufferConfig _gpuless_buffer_config;   // Buffer sizes for orchestrator-gpuless channel

    GPUlessServer _gpuless_server;
    GPUInstance* _gpu_instance;
    std::unique_ptr<Executor> _executor;

    ClientStatus _status = ClientStatus::NOT_ACTIVE;

    // These are only used to wait before registration
    std::unique_ptr<ActiveInvocation> _active_invocation;

    // As soon as previous one is finished, we try to schedule
    // a new one ASAP. If there's a pipeline of requests, then
    // we need to keep the finished one for final HTTP reply.
    std::unique_ptr<ActiveInvocation> _finished_invocation;

    std::queue<std::unique_ptr<ActiveInvocation>> _pending_invocations;

    std::optional<CommunicationIceoryxV1> _comm_v1;
#ifdef MIGNIFICIENT_WITH_ICEORYX2
    std::optional<CommunicationIceoryxV2> _comm_v2;
#endif

  };


}}

#endif
