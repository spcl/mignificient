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
      _payload(
        _send.loan().value()
      ),
      _event_context{EventSource::CLIENT, this}
    {
      _payload = _send.loan().value();
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

    void setBusy(bool busy)
    {
      _busy = busy;
    }

    bool isBusy() const
    {
      return _busy;
    }

    void setGpulessServer(std::shared_ptr<GPUlessServer> server)
    {
      _gpulessServer = server;
    }

    void setExecutor(std::shared_ptr<Executor> executor)
    {
      _executor = executor;
    }

    void add_pending_invocation(ActiveInvocation&& invoc)
    {
      _pending_invocations.push(invoc);
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
        _pending_invocations.pop();

      }

      _active = true;
    }

  private:
    Context _event_context;

    int _invoc_idx = 0;
    bool _busy = false;
    std::atomic<bool> _active = false;

    std::string _id;
    std::string _fname;
    iox::popo::Publisher<mignificient::executor::Invocation> _send;
    iox::popo::Subscriber<mignificient::executor::InvocationResult> _recv;

    std::shared_ptr<GPUlessServer> _gpulessServer;
    std::shared_ptr<Executor> _executor;

    std::queue<ActiveInvocation> _pending_invocations;

    iox::popo::Sample<mignificient::executor::Invocation, iox::mepoo::NoUserHeader> _payload;

    //std::unordered_multimap<std::string, std::shared_ptr<Executor>> warmContainers_;
    //std::unordered_multimap<std::string, std::shared_ptr<Executor>> lukewarmContainers_;

    //std::vector<std::shared_ptr<SarusContainerExecutor>> getContainers(
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
