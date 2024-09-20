#ifndef __MIGNIFICIENT_ORCHESTRATOR_HTTP_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_HTTP_HPP__

#include <memory>

#include <drogon/drogon.h>
#include <iceoryx_posh/popo/user_trigger.hpp>

#include <mignificient/orchestrator/invocation.hpp>

namespace mignificient { namespace orchestrator {

  struct HTTPTrigger {

    void trigger(ActiveInvocation&& invoc)
    {
      {
        std::lock_guard<std::mutex> lock(_mutex);
        _invocations.push(invoc);
      }
      _trigger.trigger();
    }

    iox::popo::UserTrigger& iceoryx_trigger()
    {
      return _trigger;
    }

    int size()
    {
      return _invocations.size();
    }

  private:
    iox::popo::UserTrigger _trigger;

    std::mutex _mutex;
    std::queue<ActiveInvocation> _invocations;
  };

  class HTTPServer : public drogon::HttpController<HTTPServer, false>, public std::enable_shared_from_this<HTTPServer> {
  public:

      METHOD_LIST_BEGIN
      ADD_METHOD_TO(HTTPServer::invoke, "/invoke", drogon::Post);
      METHOD_LIST_END

      void invoke(const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback);

      HTTPServer(Json::Value & config, HTTPTrigger& trigger);
      void run();
      void shutdown();
      void wait();

  private:

      HTTPTrigger& _trigger;

      std::thread _server_thread;
  };

}}

#endif
