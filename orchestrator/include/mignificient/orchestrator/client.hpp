#ifndef __MIGNIFICIENT_ORCHESTRATOR_CLIENT_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_CLIENT_HPP__

#include <queue>

#include <iceoryx_posh/popo/subscriber.hpp>
#include <iceoryx_posh/popo/publisher.hpp>
#include <iceoryx_posh/popo/untyped_publisher.hpp>

#include <mignificient/executor/executor.hpp>
#include <mignificient/orchestrator/event.hpp>
#include <mignificient/orchestrator/executor.hpp>
#include <mignificient/orchestrator/invocation.hpp>

namespace mignificient { namespace orchestrator {

  enum class ClientStatus
  {
    NOT_ACTIVE,
    CPU_ONLY,
    MEMCPY,
    FULL
  };

  struct Client
  {

    Client(const std::string& id, const std::string& fname):
      _id(id),
      _fname(fname),
      _busy(false),
      _send(
        iox::capro::ServiceDescription{
          iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, id},
          "Orchestrator",
          "Receive"
        }
      ),
      _recv(
        iox::capro::ServiceDescription{
          iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, id},
          "Orchestrator",
          "Send"
        }
      ),
      _gpuless_send(
        iox::capro::ServiceDescription{
          iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, fmt::format("gpuless-{}", id)},
          "Orchestrator",
          "Receive"
        }
      ),
      _gpuless_recv(
        iox::capro::ServiceDescription{
          iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, fmt::format("gpuless-{}", id)},
          "Orchestrator",
          "Send"
        }
      ),
      _payload(
        _send.loan().value()
      ),
      _gpuless_payload(
        _gpuless_send.loan().value()
      ),
      _event_context{EventSource::CLIENT, this}
    {}

    executor::Invocation& request()
    {
      return *_payload;
    }

    const std::string& get_name() const { return _id; }

    iox::popo::Subscriber<mignificient::executor::InvocationResult>& subscriber()
    {
      return _recv;
    }

    iox::popo::Subscriber<int>& gpuless_subscriber()
    {
      return _gpuless_recv;
    }

    void send_gpuless_msg(GPUlessMessage msg)
    {
      *_gpuless_payload.get() = static_cast<int>(msg);

      _gpuless_send.publish(std::move(_gpuless_payload));
      _gpuless_payload = std::move(_gpuless_send.loan().value());
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
      spdlog::info("[Client] For client {} add a new invocation {}", _id, invoc->uuid());
      _pending_invocations.push(std::move(invoc));
    }

    void finished(std::string_view response);

    void yield();

    void activate_memcpy()
    {
      spdlog::info("[Client] Activate gpuless memcpy for {}", _id);
      send_gpuless_msg(GPUlessMessage::MEMCPY_ONLY);

      _status = ClientStatus::MEMCPY;
    }

    void activate_kernels()
    {
      spdlog::info("[Client] Activate gpuless kernel exec for {}", _id);
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

      request().id = iox::cxx::string<64>{iox::cxx::TruncateToCapacity, std::to_string(_invoc_idx++)};
      request().data.resize(_active_invocation->input().size());
      std::copy_n(_active_invocation->input().begin(), _active_invocation->input().size(), request().data.data());
      request().size = _active_invocation->input().size();

      spdlog::info("[Client] Activate gpuless executor for {}", _id);
      _send.publish(std::move(_payload));
      _payload = _send.loan().value();

      spdlog::info("[Client] Activate gpuless basic exec for {}", _id);
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
    iox::popo::Publisher<mignificient::executor::Invocation> _send;
    iox::popo::Subscriber<mignificient::executor::InvocationResult> _recv;

    iox::popo::Publisher<int> _gpuless_send;
    iox::popo::Subscriber<int> _gpuless_recv;
    iox::popo::Sample<int, iox::mepoo::NoUserHeader> _gpuless_payload;

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


    iox::popo::Sample<mignificient::executor::Invocation, iox::mepoo::NoUserHeader> _payload;

    //std::unordered_multimap<std::string, std::shared_ptr<Executor>> warmContainers_;
    //std::unordered_multimap<std::string, std::shared_ptr<Executor>> lukewarmContainers_;

    //std::vector<std::shared_ptr<SarusContainerExecutor>> getContainers(k
    //    const std::unordered_multimap<std::string, std::shared_ptr<SarusContainerExecutor>>& map,
    //    const std::string& functionName) const {
    //    std::vector<std::shared_ptr<SarusContainerExecutor>> containers;
    //    auto range = map.equal_range(functionName);
    //    for (auto it = range.first; it != range.second; ++it) {
    //        containers.push_back(it->second);
    //    }
    //    return containers;
    //}
  };


}}

#endif
