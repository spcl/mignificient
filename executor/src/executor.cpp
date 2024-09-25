
#include <mignificient/executor/executor.hpp>

#include <iceoryx_hoofs/posix_wrapper/signal_watcher.hpp>
#include <iceoryx_hoofs/posix_wrapper/signal_handler.hpp>
#include <iceoryx_posh/capro/service_description.hpp>
#include <spdlog/spdlog.h>

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
    iox::runtime::PoshRuntime::initRuntime(iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, name});

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

    _yield_msg = std::move(client.value().loan().value());
  }

  void Runtime::register_runtime()
  {
    _result = std::move(client.value().loan().value());
    _result.value()->msg = Message::REGISTER;
    _result->publish();
  }

  void Runtime::gpu_yield()
  {
    _yield_msg.value()->msg = Message::YIELD;
    _yield_msg->publish();

    _yield_msg.reset();
    _yield_msg = std::move(client.value().loan().value());
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

          _result.reset();
          _result = std::move(client.value().loan().value());
          _result.value()->size = 0;
          _result.value()->data.resize(InvocationResult::CAPACITY);

          return InvocationData{ptr->data.data(), ptr->size};
        }
      }
    }

    return InvocationData{nullptr, 0};
  }

  void Runtime::finish(int size)
  {
    _result.value()->size = size;
    _result.value()->msg = Message::FINISH;
    _result->publish();

    _result.reset();
    _result = std::move(client.value().loan().value());
  }

  InvocationResultData Runtime::result()
  {
    return InvocationResultData{_result.value()->data.data(), _result.value()->size, InvocationResult::CAPACITY};
  }

}}
