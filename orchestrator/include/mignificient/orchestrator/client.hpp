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

  struct Client {

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
    {
      //_payload = _send.loan().value();
    }

    executor::Invocation& request()
    {
      return *_payload;
    }

    void send_request()
    {
      _send.publish(std::move(_payload));
      _payload = _send.loan().value();
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

    void set_active()
    {
      _active = true;
    }

    bool is_active() const
    {
      return _active;
    }

    void set_gpuless_server(GPUlessServer&& server)
    {
      _gpulessServer = server;
    }

    void setExecutor(std::shared_ptr<Executor> executor)
    {
      _executor = std::move(executor);
    }

    void add_pending_invocation(ActiveInvocation&& invoc)
    {
      _pending_invocations.push(invoc);
    }

    void finished(std::string_view response)
    {
      auto& invoc = _active_invocations.front();

      invoc.respond(response);

      _active_invocations.pop();
    }

    void send(ActiveInvocation& invoc)
    {
      if(!_active) {
        add_pending_invocation(std::move(invoc));
        return;
      }

      request().id = iox::cxx::string<64>{iox::cxx::TruncateToCapacity, std::to_string(_invoc_idx++)};
      request().data.resize(invoc.input().size());
      std::copy_n(invoc.input().begin(), invoc.input().size(), request().data.data());
      request().size = invoc.input().size();

      send_request();

      _active_invocations.push(std::move(invoc));
    }

    void send_all_pending()
    {
      while(!_pending_invocations.empty()) {

        auto& invoc = _pending_invocations.front();

        request().id = iox::cxx::string<64>{iox::cxx::TruncateToCapacity, std::to_string(_invoc_idx++)};
        request().data.resize(invoc.input().size());
        std::copy_n(invoc.input().begin(), invoc.input().size(), request().data.data());
        request().size = invoc.input().size();

        send_request();

        _active_invocations.push(std::move(invoc));
        _pending_invocations.pop();
      }

      _active = true;
    }

    void executor_active()
    {
      _executor_active = true;

      if(_gpuless_active) {
        send_all_pending();
      }
    }

    void gpuless_active()
    {
      _gpuless_active = true;

      if(_executor_active) {
        send_all_pending();
      }
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

    GPUlessServer _gpulessServer;
    std::shared_ptr<Executor> _executor;

    // FIXME: pointers would likely work better here due to excessive moving
    std::queue<ActiveInvocation> _pending_invocations;
    std::queue<ActiveInvocation> _active_invocations;

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
