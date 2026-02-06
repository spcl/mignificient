#ifndef __MIGNIFICIENT_ORCHESTRATOR_HTTP_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_HTTP_HPP__

#include <memory>
#include <mutex>

#include <drogon/drogon.h>
#include <iceoryx_posh/popo/user_trigger.hpp>

#include <mignificient/orchestrator/invocation.hpp>
#include <mignificient/ipc/config.hpp>

#ifdef MIGNIFICIENT_WITH_ICEORYX2
#include <iox2/iceoryx2.hpp>
#include <iox2/file_descriptor.hpp>
#include <iox2/service_type.hpp>
#include <iox2/waitset.hpp>
#include <unistd.h>
#include <sys/eventfd.h>
#endif

namespace mignificient { namespace orchestrator {

  struct HTTPTrigger {

    HTTPTrigger(ipc::IPCBackend backend):
      _ipc_backend(backend)
    {}

    void trigger(std::unique_ptr<ActiveInvocation> && invoc)
    {
      {
        std::lock_guard<std::mutex> lock(_mutex);
        _invocations.push(std::move(invoc));
      }
      internal_trigger();
    }

    int size()
    {
      return _invocations.size();
    }

    std::vector<std::unique_ptr<ActiveInvocation>> get_invocations()
    {
      std::vector<std::unique_ptr<ActiveInvocation>> invocations;

      {
        std::lock_guard<std::mutex> lock{_mutex};

        while(!_invocations.empty()) {
          invocations.push_back(std::move(_invocations.front()));
          _invocations.pop();
        }
      }

      return invocations;
    }

  private:
    ipc::IPCBackend _ipc_backend;

    virtual void internal_trigger() = 0;

    // TODO: Lock-free queue?
    std::mutex _mutex;
    std::queue<std::unique_ptr<ActiveInvocation>> _invocations;
  };

  struct HTTPTriggerV1 : public HTTPTrigger {
    HTTPTriggerV1() : HTTPTrigger(ipc::IPCBackend::ICEORYX_V1) {}

    template<typename Callback>
    void register_trigger(iox::popo::WaitSet<>& waitset, Callback && callback)
    {
      waitset.attachEvent(
          _trigger,
          callback
      ).or_else(
        [](auto) {
          spdlog::error("Failed to attach subscriber");
          std::exit(EXIT_FAILURE);
        }
      );
    }

    void internal_trigger()
    {
      _trigger.trigger();
    }

  private:
    iox::popo::UserTrigger _trigger;
  };

#ifdef MIGNIFICIENT_WITH_ICEORYX2
  struct HTTPTriggerV2 : public HTTPTrigger {
    HTTPTriggerV2() : HTTPTrigger(ipc::IPCBackend::ICEORYX_V2)
    {
      _event_fd = eventfd(0, EFD_NONBLOCK);
      _iceoryx_fd = iox2::FileDescriptor::create_non_owning(_event_fd).value();
    }

    ~HTTPTriggerV2()
    {
      close(_event_fd);
    }

    void internal_trigger()
    {
      uint64_t val = 1;
      write(_event_fd, &val, sizeof(val));
    }

    void register_trigger(iox2::WaitSet<iox2::ServiceType::Ipc>& waitset)
    {
      _guard = std::move(waitset.attach_notification(_iceoryx_fd->as_view()).value());
    }

    void unregister_trigger()
    {
      _guard.reset();
    }

    bool triggered(iox2::WaitSetAttachmentId<iox2::ServiceType::Ipc>& id)
    {
      if(id.has_event_from(*_guard)) {
        read(_event_fd, nullptr, sizeof(uint64_t)); // Clear the eventfd
        return true;
      }
      return false;
    }

  private:
    int _event_fd;

    std::optional<iox2::WaitSetGuard<iox2::ServiceType::Ipc>> _guard;
    std::optional<iox2::FileDescriptor> _iceoryx_fd;
  };
#endif

  class HTTPServer : public drogon::HttpController<HTTPServer, false>, public std::enable_shared_from_this<HTTPServer> {
  public:

      METHOD_LIST_BEGIN

      /**
       * The JSON should contain the following fields
       * - Function name
       * - Container name OR binary name
       * - Username
       * - Modules to load
       * - MIG size
       * - GPU mem allocation
       */
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
