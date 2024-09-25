#ifndef __MIGNIFICIENT_ORCHESTRATOR_CLIENT_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_CLIENT_HPP__

#include <iceoryx_posh/popo/subscriber.hpp>
#include <iceoryx_posh/popo/publisher.hpp>

#include <mignificient/executor/executor.hpp>
#include <mignificient/orchestrator/event.hpp>
#include <mignificient/orchestrator/executor.hpp>

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
      //_payload = _send.loan(sizeof(executor::Invocation), alignof(executor::Invocation)).value();
      //_payload = _send.loan().value();
    }

    Client(Client&& obj) = default;

    ~Client()
    {
      //_send.release(_payload);
    }

    executor::Invocation& request()
    {
      //return *static_cast<executor::Invocation*>(_payload);
      return *_payload;
    }

    void send_request()
    {
      _send.publish(std::move(_payload));
      //_payload = _send.loan(
      //  sizeof(executor::Invocation),
      //  alignof(executor::Invocation)
      //).value();
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

  private:
    Context _event_context;

    bool _busy = false;
    std::string _id;
    std::string _fname;
    iox::popo::Publisher<mignificient::executor::Invocation> _send;
    iox::popo::Subscriber<mignificient::executor::InvocationResult> _recv;

    std::shared_ptr<GPUlessServer> _gpulessServer;
    std::shared_ptr<Executor> _executor;

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
