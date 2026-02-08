
#include <mignificient/executor/executor.hpp>

#include <iox/signal_watcher.hpp>
#include <iceoryx_hoofs/posix_wrapper/signal_handler.hpp>
#include <iceoryx_posh/capro/service_description.hpp>
#include <spdlog/spdlog.h>

#include <sys/prctl.h>

namespace mignificient { namespace executor {

  bool Runtime::quit = false;
  iox::popo::WaitSet<>* CommunicationIceoryxV1::_waitset = nullptr;

  void Runtime::sigHandler(int sig [[maybe_unused]])
  {
    quit = true;
    if(CommunicationIceoryxV1::_waitset)
    {
      CommunicationIceoryxV1::_waitset->markForDestruction();
    }
  }

  CommunicationIceoryxV1::CommunicationIceoryxV1(const std::string& name)
  {
    last_message = nullptr;

    iox::runtime::PoshRuntime::initRuntime(iox::RuntimeName_t{iox::TruncateToCapacity_t{}, name.c_str()});

    client.emplace(iox::capro::ServiceDescription{iox::RuntimeName_t{iox::TruncateToCapacity_t{}, name.c_str()}, "Orchestrator", "Send"});
    orchestrator.emplace(iox::capro::ServiceDescription{iox::RuntimeName_t{iox::TruncateToCapacity_t{}, name.c_str()}, "Orchestrator", "Receive"});

    waitset.emplace();
    waitset.value().attachState(orchestrator.value(), iox::popo::SubscriberState::HAS_DATA).or_else([](auto) {
        std::cerr << "failed to attach subscriber" << std::endl;
        std::exit(EXIT_FAILURE);
    });

    _waitset = &waitset.value();

    sigint.emplace(iox::posix::registerSignalHandler(iox::posix::Signal::INT, Runtime::sigHandler).expect("correct signal"));
    sigterm.emplace(iox::posix::registerSignalHandler(iox::posix::Signal::TERM, Runtime::sigHandler).expect("correct signal"));

    _yield_msg = std::move(client.value().loan().value());
  }

  void CommunicationIceoryxV1::gpu_yield()
  {
    _yield_msg.value()->msg = Message::YIELD;
    _yield_msg->publish();

    _yield_msg.reset();
    _yield_msg = std::move(client.value().loan().value());
  }

  void CommunicationIceoryxV1::register_runtime()
  {
    _result = std::move(client.value().loan().value());
    _result.value()->msg = Message::REGISTER;
    _result->publish();
  }

  void CommunicationIceoryxV1::finish(int size)
  {
    _result.value()->size = size;
    _result.value()->msg = Message::FINISH;
    _result->publish();

    _result.reset();
    _result = std::move(client.value().loan().value());
  }

#ifdef MIGNIFICIENT_WITH_ICEORYX2

  void CommunicationIceoryxV2::register_runtime()
  {
    _result = std::move(client_send.value().loan_uninit().value());
    _result.value().payload_mut().msg = Message::REGISTER;

    auto initialized_sample = iox2::assume_init(std::move(_result.value()));

    auto res = iox2::send(std::move(initialized_sample));
    if(!res.has_value()) {
      spdlog::error("Failed to register runtime: {}", static_cast<uint64_t>(res.error()));
    }

    {
      auto res = client_notifier->notify();
      std::cerr << "notify orchestrator " << std::endl;
      if(!res.has_value()) {
        spdlog::error("Failed to send register runtime notification: {}", static_cast<uint64_t>(res.error()));
      }
    }
  }

  void CommunicationIceoryxV2::gpu_yield()
  {
    _yield_msg.value().payload_mut().msg = Message::YIELD;
    auto initialized_sample = iox2::assume_init(std::move(_yield_msg.value()));
    auto res = iox2::send(std::move(initialized_sample));
    if(!res.has_value()) {
      spdlog::error("Failed to send gpu yield: {}", static_cast<uint64_t>(res.error()));
    }

    {
      auto res = client_notifier->notify();
      if(!res.has_value()) {
        spdlog::error("Failed to send gpu yield notification: {}", static_cast<uint64_t>(res.error()));
      }
    }

    _yield_msg.reset();
    _yield_msg = std::move(client_send.value().loan_uninit().value());
  }

  void CommunicationIceoryxV2::finish(int size)
  {
    _result.value().payload_mut().msg = Message::FINISH;
    _result.value().payload_mut().size = size;
    auto initialized_sample = iox2::assume_init(std::move(_result.value()));
    auto res = iox2::send(std::move(initialized_sample));
    if(!res.has_value()) {
      spdlog::error("Failed to send finish invocation message: {}", static_cast<uint64_t>(res.error()));
    }

    {
      auto res = client_notifier->notify();
      if(!res.has_value()) {
        spdlog::error("Failed to send gpu finish invocation notification: {}", static_cast<uint64_t>(res.error()));
      }
    }

    _result.reset();
    _result = std::move(client_send.value().loan_uninit().value());
  }

  CommunicationIceoryxV2::CommunicationIceoryxV2(const std::string& name)
  {
    auto node_result = iox2::NodeBuilder().create<iox2::ServiceType::Ipc>();
    if(!node_result.has_value()) {
      spdlog::error("Failed to create iceoryx2 Node: {}", static_cast<uint64_t>(node_result.error()));
      throw std::runtime_error("Failed to create iceoryx2 Node");
    }
    spdlog::info("Created iceoryx2 Node for executor");
    iox2_node = std::move(node_result.value());

    {
      auto exec_send_service = iox2_node->service_builder(
          iox2::ServiceName::create(fmt::format("{}.Orchestrator.Client.Recv", name).c_str()).value())
      .publish_subscribe<mignificient::executor::InvocationResult>()
      .max_publishers(1)
      .max_subscribers(1)
      .open_or_create();

      if (!exec_send_service.has_value()) {
        spdlog::error("Failed to create iceoryx2 service for orchestrator-client recv: {}", static_cast<uint64_t>(exec_send_service.error()));
        throw std::runtime_error("Failed to create iceoryx2 service");
      }

      auto pub_result = exec_send_service.value().publisher_builder().create();
      if (pub_result.has_value()) {
        client_send = std::move(pub_result.value());
      } else {
        spdlog::error("Failed to create iceoryx2 publisher for orchestrator-client recv: {}", static_cast<uint64_t>(pub_result.error()));
        throw std::runtime_error("Failed to create iceoryx2 publisher");
      }

      auto exec_pub_service = iox2_node->service_builder(
          iox2::ServiceName::create(fmt::format("{}.Orchestrator.Client.Send", name).c_str()).value())
      .publish_subscribe<mignificient::executor::Invocation>()
      .max_publishers(1)
      .max_subscribers(1)
      .open_or_create();

      if (!exec_pub_service.has_value()) {
        spdlog::error("Failed to create iceoryx2 service for orchestrator-client send: {}", static_cast<uint64_t>(exec_pub_service.error()));
        throw std::runtime_error("Failed to create iceoryx2 service");
      }

      auto sub_result = exec_pub_service.value().subscriber_builder().create();
      if (sub_result.has_value()) {
        orchestrator_recv = std::move(sub_result.value());
      } else {
        spdlog::error("Failed to create iceoryx2 subscriber for orchestrator-client send: {}", static_cast<uint64_t>(sub_result.error()));
        throw std::runtime_error("Failed to create iceoryx2 subscriber");
      }

      {
        auto exec_event_service = iox2_node->service_builder(
            iox2::ServiceName::create(fmt::format("{}.Orchestrator.Client.Notify", name).c_str()).value())
        .event().open_or_create();
        if (exec_event_service.has_value()) {
          client_event_listen = std::move(exec_event_service.value());
        } else {
          spdlog::error("Failed to create iceoryx2 service for orchestrator-client notifier: {}", static_cast<uint64_t>(exec_event_service.error()));
          throw std::runtime_error("Failed to create iceoryx2 service");
        }
      }
      {
        auto exec_event_service = iox2_node->service_builder(
            iox2::ServiceName::create(fmt::format("{}.Orchestrator.Client.Listen", name).c_str()).value())
        .event().open_or_create();
        if (exec_event_service.has_value()) {
          client_event_notify = std::move(exec_event_service.value());
        } else {
          spdlog::error("Failed to create iceoryx2 service for orchestrator-client listener: {}", static_cast<uint64_t>(exec_event_service.error()));
          throw std::runtime_error("Failed to create iceoryx2 service");
        }
      }

      auto res = iox2::WaitSetBuilder()
        .signal_handling_mode(iox2::SignalHandlingMode::HandleTerminationRequests)
        .create<iox2::ServiceType::Ipc>();
      if(!res.has_value()) {
        spdlog::error("Failed to create iceoryx2 WaitSet: {}", static_cast<uint64_t>(res.error()));
        throw std::runtime_error("Failed to create iceoryx2 WaitSet");
      }
      waitset = std::move(res.value());

      {
        auto _orchestrator_listener = client_event_listen->listener_builder().create();
        if(!_orchestrator_listener.has_value()) {
          spdlog::error("Failed to create iceoryx2 client listener: {}", static_cast<uint64_t>(_orchestrator_listener.error()));
          throw std::runtime_error("Failed to create iceoryx2 listener");
        }

        orchestrator_listener = std::move(_orchestrator_listener.value());
      }
      {
        auto _client_notifier = client_event_notify->notifier_builder().create();
        if(!_client_notifier.has_value()) {
          spdlog::error("Failed to create iceoryx2 client notifier: {}", static_cast<uint64_t>(_client_notifier.error()));
          throw std::runtime_error("Failed to create iceoryx2 notifier");
        }

        client_notifier = std::move(_client_notifier.value());
      }
      {
        auto yield_msg = client_send.value().loan_uninit();
        if(!yield_msg.has_value()) {
          spdlog::error("Failed to create iceoryx2 client yield msg sample: {}", static_cast<uint64_t>(yield_msg.error()));
          throw std::runtime_error("Failed to create iceoryx2 sample");
        }

        _yield_msg = std::move(yield_msg.value());
      }
    }

  }
#endif

  Runtime::Runtime(mignificient::ipc::IPCBackend backend, const std::string& name):
    _backend(backend)
  {

    if (backend == ipc::IPCBackend::ICEORYX_V1) {
      spdlog::info("Initialize backend iceoryx1");
      _comm_v1.emplace(name);
    }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
    else if (backend == ipc::IPCBackend::ICEORYX_V2) {
      spdlog::info("Initialize backend iceoryx2");
      _comm_v2.emplace(name);
    }
#endif
    else {
      spdlog::error("Unknown backend");
      abort();
    }

    char* cpu_idx = std::getenv("CPU_BIND_IDX");
    if(cpu_idx) {
      int idx = std::atoi(cpu_idx);

      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(idx, &set);
      pid_t pid = getpid();

      spdlog::info("Setting CPU to: {}", idx);
      spdlog::error("Setting CPU to: {}", idx);
      if(sched_setaffinity(pid, sizeof(set), &set) == -1) {
        spdlog::error("Couldn't set the CPU affinity! Error {}", strerror(errno));
        exit(EXIT_FAILURE);
      }
    }
  }

  void Runtime::register_runtime()
  {
    // Get killed on parent's death
    prctl(PR_SET_PDEATHSIG, SIGHUP);

    if (_backend == ipc::IPCBackend::ICEORYX_V1) {
      _comm_v1.value().register_runtime();
    }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
    else if (_backend == ipc::IPCBackend::ICEORYX_V2) {
      _comm_v2.value().register_runtime();
    }
#endif
    else {
      abort();
    }
  }

  void Runtime::gpu_yield()
  {
    if (_backend == ipc::IPCBackend::ICEORYX_V1) {
      _comm_v1.value().gpu_yield();
    }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
    else if (_backend == ipc::IPCBackend::ICEORYX_V2) {
      _comm_v2.value().gpu_yield();
    }
#endif
    else {
      abort();
    }
  }

  InvocationData Runtime::loop_wait()
  {
    if (_backend == ipc::IPCBackend::ICEORYX_V1) {
      return _comm_v1.value().loop_wait();
    }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
    else if (_backend == ipc::IPCBackend::ICEORYX_V2) {
      return _comm_v2.value().loop_wait();
    }
#endif
    else {
      abort();
    }
  }

  InvocationData CommunicationIceoryxV1::loop_wait()
  {
    if(last_message) {
      orchestrator.value().release(last_message);
      last_message = nullptr;
    }

    bool invocation = false;

    while(!Runtime::quit)
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

          return InvocationData{ptr->data, ptr->size};
        }
      }
    }

    return InvocationData{nullptr, 0};
  }

  InvocationData CommunicationIceoryxV2::loop_wait()
  {
    bool invocation = false;

    while(!Runtime::quit)
    {

      auto res = orchestrator_listener.value().blocking_wait_one();
      if(!res.has_value()) {
        spdlog::error("Failed to receive notification from orchestrator!");
        abort();
      }

      if(!res.value().has_value()) {
        continue;
      }

      auto value = orchestrator_recv.value().receive();

      if(!value.has_value()) {
        spdlog::error("got no data, return code: {}", static_cast<uint64_t>(value.error()));
      } else {

        _last_message = std::move(value.value().value());
        const Invocation* ptr = &_last_message->payload();

        _result.reset();
        _result = std::move(client_send.value().loan_uninit().value());
        _result.value().payload_mut().size = 0;

        return InvocationData{ptr->data, ptr->size};
      }
    }

    return InvocationData{nullptr, 0};
  }

  void Runtime::finish(int size)
  {
    if (_backend == ipc::IPCBackend::ICEORYX_V1) {
      _comm_v1.value().finish(size);
    }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
    else if (_backend == ipc::IPCBackend::ICEORYX_V2) {
      _comm_v2.value().finish(size);
    }
#endif
    else {
      abort();
    }
  }

  InvocationResultData Runtime::result()
  {
    //return InvocationResultData{_result.value()->data.data(), _result.value()->size, InvocationResult::CAPACITY};
    InvocationResult* ptr;
    if (_backend == ipc::IPCBackend::ICEORYX_V1) {
      ptr = &(*_comm_v1.value()._result.value());
    }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
    else if (_backend == ipc::IPCBackend::ICEORYX_V2) {
      ptr = &_comm_v2.value()._result->payload_mut();
    }
#endif
    else {
      abort();
    }
    return InvocationResultData{ptr->data, ptr->size, InvocationResult::CAPACITY};
  }

}}
