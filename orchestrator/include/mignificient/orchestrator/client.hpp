#ifndef __MIGNIFICIENT_ORCHESTRATOR_CLIENT_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_CLIENT_HPP__

#include <iceoryx_posh/popo/subscriber.hpp>
#include <iceoryx_posh/popo/publisher.hpp>

#include <mignificient/executor/executor.hpp>

namespace mignificient { namespace orchestrator {

  struct Client {

    Client(const std::string& name):
      _id(name),
      _send(
        iox::capro::ServiceDescription{
          iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, name},
          "Orchestrator",
          "Receive"
        }
      ),
      _recv(
        iox::capro::ServiceDescription{
          iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, name},
          "Orchestrator",
          "Send"
        }
      ),
      _payload(
        _send.loan().value()
      )
    {
      //_payload = _send.loan(sizeof(executor::Invocation), alignof(executor::Invocation)).value();
      //_payload = _send.loan().value();
    }

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

  private:
    std::string _id;
    iox::popo::Publisher<mignificient::executor::Invocation> _send;
    iox::popo::Subscriber<mignificient::executor::InvocationResult> _recv;

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
