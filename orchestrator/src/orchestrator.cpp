
#include <iox2/service_type.hpp>
#include <iox2/waitset.hpp>
#include <mignificient/orchestrator/orchestrator.hpp>

#include <stdexcept>

#include <drogon/drogon.h>
#include <iceoryx_posh/internal/popo/base_subscriber.hpp>
#include <iceoryx_posh/popo/subscriber.hpp>
#include <iceoryx_posh/popo/untyped_publisher.hpp>
#include <json/value.h>
#include <spdlog/spdlog.h>

#include <mignificient/executor/executor.hpp>
#include <mignificient/orchestrator/client.hpp>
#include <mignificient/orchestrator/event.hpp>
#include <mignificient/orchestrator/http.hpp>

#ifdef MIGNIFICIENT_WITH_ICEORYX2
#include <iox2/iceoryx2.hpp>
#endif

namespace mignificient { namespace orchestrator {

  bool Orchestrator::_quit;
  iox::popo::WaitSet<>* Orchestrator::_waitset_ptr;
  std::shared_ptr<HTTPServer> Orchestrator::_http_server;
  ipc::IPCConfig Orchestrator::_ipc_config;

#ifdef MIGNIFICIENT_WITH_ICEORYX2
  std::optional<iox2::Node<iox2::ServiceType::Ipc>> Orchestrator::_iox2_node;
#endif

  // iceoryx1 callback: handle gpuless subscriber messages
  void handle_gpuless(iox::popo::Subscriber<int>* sub, Client* client)
  {
    while(client->gpuless_subscriber().hasData()) {

      auto res = client->gpuless_subscriber().take();

      if(*res.value().get() == static_cast<int>(GPUlessMessage::REGISTER)) {
        spdlog::info("Received registration from gpuless server {}", client->id());
        if(client->gpuless_active()) {
          client->gpu_instance()->schedule_next();
        }
      } else if(*res.value().get() == static_cast<int>(GPUlessMessage::SWAP_OFF_CONFIRM)) {
        // FIXME: implement
        throw std::runtime_error("unimplemented");
      } else {
        spdlog::error("Received unknown message from gpuless server, code: {}", *res.value().get());
      }

    }
  }

  // iceoryx1 callback: handle executor subscriber messages
  void handle_client(iox::popo::Subscriber<mignificient::executor::InvocationResult>* sub, Client* client)
  {
    while(client->subscriber_v1().hasData()) {

      auto res = client->subscriber_v1().take();

      if(res.value().get()->msg == executor::Message::FINISH) {

        client->finished(std::string_view{reinterpret_cast<const char*>(res.value().get()->data), res.value().get()->size});

      } else if (res.value().get()->msg == executor::Message::YIELD) {

        client->yield();

      } else {
        spdlog::info("Received registration from gpuless executor {}", client->id());
        if(client->executor_active()) {
          client->gpu_instance()->schedule_next();
        }
      }
    }
  }

#ifdef MIGNIFICIENT_WITH_ICEORYX2
  void handle_gpuless(Client* client)
  {
    while(client->gpuless_subscriber_v2().has_samples()) {

      auto res = client->gpuless_subscriber_v2().receive();
      if(!res.has_value()) {
        spdlog::error("Error receiving from gpuless subscriber for client {}, error {}", client->id(), res.error());
      }

      if(res.value()->payload() == static_cast<int>(GPUlessMessage::REGISTER)) {
        spdlog::info("Received registration from gpuless server {}", client->id());
        if(client->gpuless_active()) {
          client->gpu_instance()->schedule_next();
        }
      } else if(res.value()->payload() == static_cast<int>(GPUlessMessage::SWAP_OFF_CONFIRM)) {
        // FIXME: implement
        throw std::runtime_error("unimplemented");
      } else {
        spdlog::error("Received unknown message from gpuless server, code: {}", res.value()->payload());
      }

    }
  }

  void handle_client(Client* client)
  {
    while(client->subscriber_v2().has_samples()) {

      auto res = client->subscriber_v2().receive();
      if(!res.has_value()) {
        spdlog::error("Error receiving from gpuless subscriber for client {}, error {}", client->id(), res.error());
      }

      auto& payload = res.value()->payload();

      if(payload.msg == executor::Message::FINISH) {

        client->finished(std::string_view{reinterpret_cast<const char*>(payload.data), payload.size});

      } else if (res.value()->payload().msg == executor::Message::YIELD) {

        client->yield();

      } else {
        spdlog::info("Received registration from gpuless executor {}", client->id());
        if(client->executor_active()) {
          client->gpu_instance()->schedule_next();
        }
      }
    }
  }
#endif

  // iceoryx1: WaitSet callback for HTTP trigger
  void Orchestrator::_handle_http_v1(iox::popo::UserTrigger*, Orchestrator* this_ptr)
  {
    auto invocations = this_ptr->_http_trigger_v1->get_invocations();
    SPDLOG_DEBUG("Received new HTTP invocation!");

    for(auto & invoc : invocations) {

      auto [client, success] = this_ptr->_users.process_invocation(std::move(invoc));

      if(client) {

        this_ptr->_waitset->attachEvent(
          client->gpuless_subscriber(),
          iox::popo::SubscriberEvent::DATA_RECEIVED,
          createNotificationCallback(handle_gpuless, *client)
        ).or_else(
          [](auto) {
            spdlog::error("Failed to attach subscriber");
            std::exit(EXIT_FAILURE);
          }
        );

        this_ptr->_waitset->attachEvent(
            client->subscriber_v1(),
            iox::popo::SubscriberEvent::DATA_RECEIVED,
            createNotificationCallback(handle_client, *client)
        ).or_else(
          [](auto) {
            spdlog::error("Failed to attach subscriber");
            std::exit(EXIT_FAILURE);
          }
        );

      }
    }
  }

  void Orchestrator::init(const Json::Value& config)
  {
    // Parse IPC configuration
    _ipc_config = ipc::IPCConfig::from_json(config);

    spdlog::info("IPC Backend: {}", _ipc_config.backend_string());
    spdlog::info("Polling Mode: {}", _ipc_config.polling_mode_string());

    // Log buffer configurations
    for (const auto& [component, buf_config] : _ipc_config.buffer_configs) {
      spdlog::info("Buffer config [{}]: request={} bytes, response={} bytes, capacity={}",
                   component, buf_config.request_size, buf_config.response_size, buf_config.queue_capacity);
    }

    if (_ipc_config.backend == ipc::IPCBackend::ICEORYX_V1) {
      iox::runtime::PoshRuntime::initRuntime(
          iox::RuntimeName_t{iox::TruncateToCapacity_t{}, config["name"].asString().c_str()}
      );
    }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
    else if (_ipc_config.backend == ipc::IPCBackend::ICEORYX_V2) {
      auto node_result = iox2::NodeBuilder().create<iox2::ServiceType::Ipc>();
      if(!node_result.has_value()) {
        spdlog::error("Failed to create iceoryx2 Node: {}", static_cast<uint64_t>(node_result.error()));
        throw std::runtime_error("Failed to create iceoryx2 Node");
      }
      spdlog::info("Created iceoryx2 Node for orchestrator");
      _iox2_node = std::move(node_result.value());
    }
#endif
    else {
      abort();
    }
  }

  Orchestrator::Orchestrator(const Json::Value& config, const std::string& device_db_path):
    _gpu_manager(device_db_path, sharing_model(config["sharing-model"].asString())),
    _users(_gpu_manager, config["executor"], _ipc_config)
  {

    auto http_config = config["http"];

    if (_ipc_config.backend == ipc::IPCBackend::ICEORYX_V1) {

      _waitset.emplace();
      _waitset_ptr = &_waitset.value();

      sigint.emplace(iox::posix::registerSignalHandler(iox::posix::Signal::INT, _sigHandler).expect("correct signal"));
      sigterm.emplace(iox::posix::registerSignalHandler(iox::posix::Signal::TERM, _sigHandler).expect("correct signal"));

      _http_trigger_v1.emplace();
      _http_trigger_v1->register_trigger(*_waitset, iox::popo::createNotificationCallback(Orchestrator::_handle_http_v1, *this));

      _http_server = std::make_shared<HTTPServer>(http_config, _http_trigger_v1.value());
    }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
    else if (_ipc_config.backend == ipc::IPCBackend::ICEORYX_V2) {

      auto res = iox2::WaitSetBuilder()
      .signal_handling_mode(iox2::SignalHandlingMode::HandleTerminationRequests)
      .create<iox2::ServiceType::Ipc>();
      if(!res.has_value()) {
        spdlog::error("Failed to create iceoryx2 WaitSet: {}", static_cast<uint64_t>(res.error()));
        throw std::runtime_error("Failed to create iceoryx2 WaitSet");
      }
      _waitset_v2 = std::move(res.value());

      _http_trigger_v2.emplace();
      _http_trigger_v2->register_trigger(_waitset_v2.value());

      _http_server = std::make_shared<HTTPServer>(http_config, _http_trigger_v2.value());
    }
#endif
    else {
      abort();
    }
  }

  void Orchestrator::run()
  {
    _http_server->run();

    event_loop();

    spdlog::info("Waiting for HTTP server to close down");
    _http_server->wait();
  }

  void Orchestrator::_sigHandler(int sig [[maybe_unused]])
  {
    Orchestrator::_quit = true;

    if(_ipc_config.backend == ipc::IPCBackend::ICEORYX_V1) {
      if(Orchestrator::_waitset_ptr) {
        Orchestrator::_waitset_ptr->markForDestruction();
      }

      if(Orchestrator::_http_server) {
        Orchestrator::_http_server->shutdown();
      }
    }
  }

  Client* Orchestrator::client(int id)
  {
    auto it = clients.find(id);
    if(it != clients.end()) {
      return &(*it).second;
    } else {
      return nullptr;
    }
  }

  void Orchestrator::event_loop()
  {
    if (_ipc_config.backend == ipc::IPCBackend::ICEORYX_V1) {
      _event_loop_v1();
    }
#ifdef MIGNIFICIENT_WITH_ICEORYX2
    else {
      _event_loop_v2();
    }
#endif
  }

  void Orchestrator::_event_loop_v1()
  {
    while(!_quit)
    {
      auto notificationVector = _waitset->wait();

      for (auto& notification : notificationVector)
      {
        (*notification)();
      }
    }
  }

#ifdef MIGNIFICIENT_WITH_ICEORYX2

  void Orchestrator::_event_loop_v2()
  {
    auto res = _waitset_v2->wait_and_process(
      [&](iox2::WaitSetAttachmentId<iox2::ServiceType::Ipc> attachment_id) {

        if (_http_trigger_v2->triggered(attachment_id)) {
          _handle_http_v2();
        } else {

          auto val = _waitset_mappings_v2.find(attachment_id);
          if(val == _waitset_mappings_v2.end()) {
            spdlog::error("Received event for unknown attachment id");
            return iox2::CallbackProgression::Continue;
          }

          Client* client = std::get<0>(val->second);
          if(std::get<1>(val->second)) {

            while(client->read_event_client_v2()) {
              handle_client(client);
            }

          } else {
            while(client->read_event_gpuless_v2()) {
              handle_gpuless(client);
            }
          }

        }

        return iox2::CallbackProgression::Continue;
      }
    );

    if(!res.has_value()) {
      spdlog::error("Error in iceoryx2 event loop: {}", static_cast<uint64_t>(res.error()));
    }
    spdlog::info("Finished iceoryx2 event loop with status {}", res.value());

    //_waitset_mappings_v2.clear();
    //clients.clear();

    //_http_trigger_v2.reset();
    _http_trigger_v2->unregister_trigger();
    _users.apply_clients(
[](Client *client){
        client->uninit_v2();
      }
    );

    if(Orchestrator::_http_server) {
      Orchestrator::_http_server->shutdown();
    }
  }

  void Orchestrator::_handle_http_v2()
  {
    auto invocations = _http_trigger_v2->get_invocations();
    SPDLOG_DEBUG("Received new HTTP invocation!");

    for(auto & invoc : invocations) {

      auto [client, success] = _users.process_invocation(std::move(invoc));

      if(client) {

        auto res = client->init_v2(*_waitset_v2);

        _waitset_mappings_v2.insert(std::make_pair(std::move(res[0]), std::make_tuple(client, true)));
        _waitset_mappings_v2.insert(std::make_pair(std::move(res[1]), std::make_tuple(client, false)));
      }
    }
  }

#endif // MIGNIFICIENT_WITH_ICEORYX2

}}
