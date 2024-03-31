
#include "executor.hpp"

#include <iceoryx_hoofs/posix_wrapper/signal_watcher.hpp>
#include <iceoryx_hoofs/posix_wrapper/signal_handler.hpp>
#include <iceoryx_posh/capro/service_description.hpp>

namespace mignificient { namespace executor {

  bool Runtime::_quit = false;
  iox::popo::WaitSet<>* Runtime::_waitset = nullptr;

  void Runtime::_sigHandler(int sig [[maybe_unused]])
  {
    _quit = true;
    if(_waitset)
    {
      _waitset->markForDestruction();
    }
  }

  Runtime::Runtime(const std::string& name):
    last_message(nullptr)
  {
    // FIXME: Name
    constexpr char APP_NAME[] = "gpuless";
    iox::runtime::PoshRuntime::initRuntime(APP_NAME);

    client.emplace(iox::capro::ServiceDescription{iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, name}, "Orchestrator", "Send"});
    orchestrator.emplace(iox::capro::ServiceDescription{iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, name}, "Orchestrator", "Receive"});

    waitset.emplace();
    waitset.value().attachState(orchestrator.value(), iox::popo::SubscriberState::HAS_DATA).or_else([](auto) {
        std::cerr << "failed to attach subscriber" << std::endl;
        std::exit(EXIT_FAILURE);
    });

    _waitset = &waitset.value();
  
    sigint.emplace(iox::posix::registerSignalHandler(iox::posix::Signal::INT, _sigHandler));
    sigterm.emplace(iox::posix::registerSignalHandler(iox::posix::Signal::TERM, _sigHandler));
  }

  void Runtime::gpu_yield()
  {
    client.value().loan()
      .and_then([](auto & msg) {
        *msg = Message::YIELD;
        msg.publish();
      })
      .or_else([](auto & error) {
        std::cout << "yield error " << static_cast<uint64_t>(error) << std::endl;
      });
  }

  InvocationData Runtime::loop_wait()
  {
    if(last_message) {
      orchestrator.value().release(last_message);
      last_message = nullptr;
    }

    bool invocation = false;

    while(!_quit)
    {
      auto notificationVector = waitset->wait();

      // FIXME: handle multiple invocations?
      for (auto& notification : notificationVector)
      {
        auto value = orchestrator.value().take();

        if(value.has_error()) {
          std::cout << "got no data, return code: " << static_cast<uint64_t>(value.get_error()) << std::endl;
        } else {

          last_message = value.value();
          auto* ptr = static_cast<const Invocation*>(last_message);
          return InvocationData{ptr->data.data(), ptr->data.size()};
        }
      }
    }

    return InvocationData{nullptr, 0};
  }

  void Runtime::finish()
  {
    client.value().loan()
      .and_then([](auto & msg) {
        *msg = Message::FINISH;
        msg.publish();
      })
      .or_else([](auto & error) {
        std::cout << "yield error " << static_cast<uint64_t>(error) << std::endl;
      });
  }

}}
