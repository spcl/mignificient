
#include <mignificient/orchestrator/orchestrator.hpp>

#include <stdexcept>

#include <drogon/drogon.h>
#include <iceoryx_posh/internal/popo/base_subscriber.hpp>
#include <iceoryx_posh/popo/subscriber.hpp>
#include <iceoryx_posh/popo/untyped_publisher.hpp>
#include <json/value.h>
#include <spdlog/spdlog.h>

#include <mignificient/ipc/config.hpp>
#include <mignificient/executor/executor.hpp>
#include <mignificient/orchestrator/client.hpp>
#include <mignificient/orchestrator/event.hpp>
#include <mignificient/orchestrator/http.hpp>

#ifdef MIGNIFICIENT_WITH_ICEORYX2
#include <iox2/service_type.hpp>
#include <iox2/waitset.hpp>
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

  static void _handle_swap_confirm(int msg, Client* client)
  {
    auto swap_data = client->read_swap_result();
    executor::SwapResult result{};
    if(swap_data.has_value()) {
      result = swap_data.value();
    } else {
      spdlog::warn("No SwapResult data received from gpuless for client {}", client->id());
      result.status = -1;
    }

    if(msg == static_cast<int>(GPUlessMessage::SWAP_OFF_CONFIRM)) {

      spdlog::info("Received SWAP_OFF_CONFIRM from gpuless server {}: {} bytes in {} us",
        client->id(), result.memory_bytes, result.time_us);
      client->set_lukewarm(true);

      if(client->executor_ptr()) {
        auto mb = static_cast<double>(result.memory_bytes) / 1024.0 / 1024.0;
        client->gpu_instance()->release_memory(mb);
      }

      if(client->has_pending_swap_callback()) {
        auto cb = client->take_pending_swap_callback();
        cb(result);
      }

    } else if(msg == static_cast<int>(GPUlessMessage::SWAP_IN_CONFIRM)) {

      spdlog::info("Received SWAP_IN_CONFIRM from gpuless server {}: {} bytes in {} us",
        client->id(), result.memory_bytes, result.time_us);
      client->set_lukewarm(false);

      if(client->executor_ptr()) {
        auto mb = static_cast<double>(result.memory_bytes) / 1024.0 / 1024.0;
        client->gpu_instance()->reclaim_memory(mb);
      }

      if(client->has_pending_swap_callback()) {
        auto cb = client->take_pending_swap_callback();
        cb(result);
      } else if(client->is_swap_in_for_invocation()) {

        // We finished the swap in, and now we just record statistics.
        client->set_swap_in_for_invocation(false);
        client->front_pending_invocation()->set_swap_in_stats(result);

        client->gpu_instance()->schedule_next();

      }
    }
  }

  // iceoryx1 callback: handle gpuless subscriber messages
  void handle_gpuless(iox::popo::Subscriber<int>* sub, Client* client)
  {
    while(client->gpuless_subscriber().hasData()) {

      auto res = client->gpuless_subscriber().take();
      int msg = *res.value().get();

      if(msg == static_cast<int>(GPUlessMessage::REGISTER)) {
        spdlog::info("Received registration from gpuless server {}", client->id());
        if(client->gpuless_active()) {
          client->gpu_instance()->schedule_next();
        }
      } else if(msg == static_cast<int>(GPUlessMessage::SWAP_OFF_CONFIRM) ||
                msg == static_cast<int>(GPUlessMessage::SWAP_IN_CONFIRM)) {
        _handle_swap_confirm(msg, client);
      } else if(msg == static_cast<int>(GPUlessMessage::OUT_OF_MEMORY)) {
        spdlog::error("OOM detected by gpuless server for client {}", client->id());
        // Mark for OOM handling - actual cleanup done in event loop via _check_oom()
        client->set_oom_detected(true);
      } else {
        spdlog::error("Received unknown message from gpuless server, code: {}", msg);
      }

    }
  }

  // iceoryx1 callback: handle executor subscriber messages
  void handle_client(iox::popo::Subscriber<mignificient::executor::InvocationResult>* sub, Client* client)
  {
    while(client->subscriber_v1().hasData()) {

      auto res = client->subscriber_v1().take();

      if(res.value().get()->msg == executor::Message::FINISH) {

        client->send_gpuless_msg(GPUlessMessage::INVOCATION_FINISH);
        client->finished(std::string_view{reinterpret_cast<const char*>(res.value().get()->data), res.value().get()->size});

      } else if (res.value().get()->msg == executor::Message::YIELD) {

        client->yield();

      } else {
        spdlog::info("Received registration from executor {}", client->id());
        if(client->executor_active()) {
          client->gpu_instance()->schedule_next();
        }
      }
    }
  }

#ifdef MIGNIFICIENT_WITH_ICEORYX2
  void handle_gpuless(int msg, Client* client)
  {

    if(msg == static_cast<int>(GPUlessMessage::REGISTER)) {
      spdlog::info("Received registration from gpuless server {}", client->id());
      if(client->gpuless_active()) {
        client->gpu_instance()->schedule_next();
      }
    } else if(msg == static_cast<int>(GPUlessMessage::SWAP_OFF_CONFIRM) ||
              msg == static_cast<int>(GPUlessMessage::SWAP_IN_CONFIRM)) {
      _handle_swap_confirm(msg, client);
    } else if(msg == static_cast<int>(GPUlessMessage::OUT_OF_MEMORY)) {
      spdlog::error("OOM detected by gpuless server for client {}", client->id());
      client->set_oom_detected(true);
    } else {
      spdlog::error("Received unknown message from gpuless server, code: {}", msg);
    }

  }

  void handle_client(Client* client)
  {
    auto res = client->subscriber_v2().receive();
    if(!res.has_value()) {
      spdlog::error("Error receiving from gpuless subscriber for client {}, error {}", client->id(), res.error());
    }

    if(!res.value().has_value()) {
      return;
    }

    auto& payload = res.value()->payload();

    if(payload.msg == executor::Message::FINISH) {

      client->send_gpuless_msg(GPUlessMessage::INVOCATION_FINISH);
      client->finished(std::string_view{reinterpret_cast<const char*>(payload.data), payload.size});

    } else if (res.value()->payload().msg == executor::Message::YIELD) {

      client->yield();

    } else {
      spdlog::info("Received registration from executor {}", client->id());
      if(client->executor_active()) {
        client->gpu_instance()->schedule_next();
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

    auto admin_reqs = this_ptr->_http_trigger_v1->get_admin_requests();
    for (auto& req : admin_reqs) {
      this_ptr->_handle_admin_request(std::move(req));
    }
  }

  void Orchestrator::init(const Json::Value& config)
  {
    // Parse IPC configuration
    _ipc_config = ipc::IPCConfig::from_json(config);

    spdlog::info("IPC Backend: {}", ipc::IPCConfig::backend_string(_ipc_config.backend));
    spdlog::info("Polling Mode: {}", ipc::IPCConfig::polling_mode_string(_ipc_config.polling_mode));

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
    if (config.isMember("timeout-check-interval-ms")) {
      _timeout_check_interval_ms = config["timeout-check-interval-ms"].asInt();
    }

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
    else if (_ipc_config.backend == ipc::IPCBackend::ICEORYX_V2) {
      _event_loop_v2();
    }
#endif
    else {
      abort();
    }
  }

  void Orchestrator::_event_loop_v1()
  {
    while(!_quit)
    {
      auto notificationVector = _waitset->timedWait(iox::units::Duration::fromMilliseconds(_timeout_check_interval_ms));

      for (auto& notification : notificationVector)
      {
        (*notification)();
      }

      _check_timeouts();
      _check_oom();
    }
  }

#ifdef MIGNIFICIENT_WITH_ICEORYX2

  void Orchestrator::_event_loop_v2()
  {
    auto interval_guard = _waitset_v2->attach_interval(
        iox2::bb::Duration::from_millis(_timeout_check_interval_ms));
    if (!interval_guard.has_value()) {
      spdlog::error("Failed to attach timeout interval to waitset");
      return;
    }
    _timeout_interval_guard = std::move(interval_guard.value());
    auto timeout_check_id = iox2::WaitSetAttachmentId<iox2::ServiceType::Ipc>::from_guard(
        *_timeout_interval_guard);

    auto res = _waitset_v2->wait_and_process(
      [&](iox2::WaitSetAttachmentId<iox2::ServiceType::Ipc> attachment_id) {

        if (attachment_id == timeout_check_id) {
          _check_timeouts();
          _check_oom();
          return iox2::CallbackProgression::Continue;
        }

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
            std::optional<size_t> gpuless_msg{};
            while((gpuless_msg = client->read_event_gpuless_v2())) {
              handle_gpuless(gpuless_msg.value(), client);
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

    // We need to deallocate listeners before we destroy the waitset
    _timeout_interval_guard.reset();
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

    auto admin_reqs = _http_trigger_v2->get_admin_requests();
    for (auto& req : admin_reqs) {
      _handle_admin_request(std::move(req));
    }
  }

#endif // MIGNIFICIENT_WITH_ICEORYX2

  void Orchestrator::_handle_admin_request(AdminRequest&& req)
  {
    Json::Value result;

    if (req.type == AdminRequestType::LIST_CONTAINERS) {

      result = _users.list_containers([](Client* c) { return c->status_string(); });

    } else if (req.type == AdminRequestType::KILL_CONTAINER) {

      Client* client = _users.find_client(req.user, req.container);
      if (!client) {
        result["success"] = false;
        result["error"] = "Container not found: " + req.container + " for user: " + req.user;
      } else {
        spdlog::info("Admin kill request for client {}", client->id());

#ifdef MIGNIFICIENT_WITH_ICEORYX2
        if (_ipc_config.backend == ipc::IPCBackend::ICEORYX_V2) {
          // If we don't remove the mappings, iceoryx2 will complain later.
          client->uninit_v2();
          for (auto it = _waitset_mappings_v2.begin(); it != _waitset_mappings_v2.end(); ) {
            if (std::get<0>(it->second) == client) {
              it = _waitset_mappings_v2.erase(it);
            } else {
              ++it;
            }
          }
        }
#endif

        client->timeout_kill();
        _gpu_manager.return_gpu(client->gpu_instance());
        _users.remove_client(req.user, client);
        result["success"] = true;
      }

    } else if (req.type == AdminRequestType::SWAP_OFF) {

      Client* client = _users.find_client(req.user, req.container);
      if (!client) {
        result["success"] = false;
        result["not_found"] = true;
        result["error"] = fmt::format("No container found: {} for user: {}", req.container, req.user);
      } else if (client->is_lukewarm()) {
        result["success"] = false;
        result["error"] = "Container is already swapped off";
      } else if (client->is_busy()) {
        result["success"] = false;
        result["error"] = "Container is busy, cannot swap off";
      } else {
        spdlog::info("Swap-off request for client {}", client->id());

        auto respond_fn = std::move(req.respond);
        client->set_pending_swap_callback(
          [respond_fn](const executor::SwapResult& swap_result) {
            Json::Value r;
            r["success"] = true;
            r["time_us"] = swap_result.time_us;
            r["memory_bytes"] = static_cast<Json::UInt64>(swap_result.memory_bytes);
            r["status"] = swap_result.status;
            respond_fn(r);
          }
        );
        client->swap_off();
        return;  // Response will be sent when SWAP_OFF_CONFIRM arrives
      }

    } else if (req.type == AdminRequestType::SWAP_IN) {

      Client* client = _users.find_client(req.user, req.container);
      if (!client) {
        result["success"] = false;
        result["not_found"] = true;
        result["error"] = fmt::format("No container found: {} for user: {}", req.container, req.user);
      } else if (!client->is_lukewarm()) {
        result["success"] = false;
        result["error"] = "Container is not swapped off";
      } else {
        spdlog::info("Swap-in request for client {}", client->id());

        auto respond_fn = std::move(req.respond);
        client->set_pending_swap_callback(
          [respond_fn](const executor::SwapResult& swap_result) {
            Json::Value r;
            r["success"] = true;
            r["time_us"] = swap_result.time_us;
            r["memory_bytes"] = static_cast<Json::UInt64>(swap_result.memory_bytes);
            r["status"] = swap_result.status;
            respond_fn(r);
          }
        );
        client->swap_in();
        return;  // Response will be sent when SWAP_IN_CONFIRM arrives
      }
    }

    req.respond(result);
  }

  void Orchestrator::_check_timeouts()
  {
    _users.check_timeouts([this](Client* client) {
      spdlog::error("Timeout for client {}", client->id());

#ifdef MIGNIFICIENT_WITH_ICEORYX2
      if (_ipc_config.backend == ipc::IPCBackend::ICEORYX_V2) {
        // Remove waitset mappings for this client before killing
        client->uninit_v2();
        // Erase entries from the mapping by finding those pointing to this client
        for (auto it = _waitset_mappings_v2.begin(); it != _waitset_mappings_v2.end(); ) {
          if (std::get<0>(it->second) == client) {
            it = _waitset_mappings_v2.erase(it);
          } else {
            ++it;
          }
        }
      }
#endif

      client->timeout_kill();
      _gpu_manager.return_gpu(client->gpu_instance());
    });
  }

  void Orchestrator::_check_oom()
  {
    _users.check_oom([this](Client* client) {
      spdlog::error("OOM kill for client {}", client->id());

#ifdef MIGNIFICIENT_WITH_ICEORYX2
      if (_ipc_config.backend == ipc::IPCBackend::ICEORYX_V2) {
        client->uninit_v2();
        for (auto it = _waitset_mappings_v2.begin(); it != _waitset_mappings_v2.end(); ) {
          if (std::get<0>(it->second) == client) {
            it = _waitset_mappings_v2.erase(it);
          } else {
            ++it;
          }
        }
      }
#endif

      client->oom_kill();
      _gpu_manager.return_gpu(client->gpu_instance());
    });
  }

}}
