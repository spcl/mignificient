
#include <mignificient/orchestrator/orchestrator.hpp>

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

namespace mignificient { namespace orchestrator {

  bool Orchestrator::_quit;
  iox::popo::WaitSet<>* Orchestrator::_waitset_ptr;
  std::shared_ptr<HTTPServer> Orchestrator::_http_server;

  void handle_client(iox::popo::Subscriber<mignificient::executor::InvocationResult>* sub, Client* client)
  {
    while(client->subscriber().hasData()) {

      auto res = client->subscriber().take();

      if(res.value().get()->msg == executor::Message::FINISH) {
        spdlog::info("Finish {}", std::string_view{reinterpret_cast<const char*>(res.value().get()->data.data()), res.value().get()->size});
      } else if (res.value().get()->msg == executor::Message::YIELD) {
        spdlog::info("Yield {}");
      } else {
        client->send_all_pending();
      }
    }
  }

  void Orchestrator::_handle_http(iox::popo::UserTrigger*, Orchestrator* this_ptr)
  {
    auto invocations = this_ptr->_http_trigger.get_invocations();
    SPDLOG_DEBUG("Received new invocation!");

    auto clients = this_ptr->_users.process_invocations(invocations);
    for(auto client : clients) {

      this_ptr->_waitset.attachEvent(
          client->subscriber(),
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

  void Orchestrator::init(const Json::Value& config)
  {
    iox::runtime::PoshRuntime::initRuntime(
        iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, config["name"].asString()}
    );
  }

  Orchestrator::Orchestrator(const Json::Value& config, const std::string& device_db_path):
    _gpu_manager(device_db_path, sharing_model(config["sharing-model"].asString())),
    _users(_gpu_manager, config["executor"])
  {
    _waitset_ptr = &_waitset;

    auto http_config = config["http"];
    _http_server = std::make_shared<HTTPServer>(http_config, _http_trigger);

    sigint.emplace(iox::posix::registerSignalHandler(iox::posix::Signal::INT, _sigHandler));
    sigterm.emplace(iox::posix::registerSignalHandler(iox::posix::Signal::TERM, _sigHandler));

    _waitset.attachEvent(
        _http_trigger.iceoryx_trigger(),
        iox::popo::createNotificationCallback(Orchestrator::_handle_http, *this)
    ).or_else(
      [](auto) {
        spdlog::error("Failed to attach subscriber");
        std::exit(EXIT_FAILURE);
      }
    );
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

    if(Orchestrator::_waitset_ptr) {
      Orchestrator::_waitset_ptr->markForDestruction();
    }

    if(Orchestrator::_http_server) {
      Orchestrator::_http_server->shutdown();
    }

  }

  void Orchestrator::add_client()
  {
    //std::string client_name = fmt::format("client_{}", _client_id);

    //const auto & [client, _] = clients.emplace(
    //  std::piecewise_construct,
    //  std::forward_as_tuple(_client_id),
    //  std::forward_as_tuple(client_name)
    //);

    ////_waitset.attachState(client->second.subscriber(), iox::popo::SubscriberEvent::DATA_RECEIVED, _client_id).or_else([](auto) {
    //_waitset.attachEvent(
    //    client->second.subscriber(),
    //    //*client,
    //    iox::popo::SubscriberEvent::DATA_RECEIVED,
    //    //0,
    //    createNotificationCallback(handle_client, client->second)//client->second.context())
    //).or_else(
    //  [](auto) {
    //    spdlog::error("Failed to attach subscriber");
    //    std::exit(EXIT_FAILURE);
    //  }
    //);

    //_client_id++;
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
    while(!_quit)
    {
      auto notificationVector = _waitset.wait();

      for (auto& notification : notificationVector)
      {

        (*notification)();

        //Context* ctx = static_cast<Context*>(notification->getUserDefinedContext());

        //if(ctx->type == EventSource::CLIENT) {


        //} else if (ctx->type == EventSource::HTTP) {
        //  spdlog::info("New http message");
        //}
        //notification->getOrigin<typename T>()
        //auto value = orchestrator.value().take();

        //if(value.has_error()) {
        //  std::cout << "got no data, return code: " << static_cast<uint64_t>(value.get_error()) << std::endl;
        //} else {

        //  last_message = value.value();
        //  auto* ptr = static_cast<const Invocation*>(last_message);
        //  return InvocationData{ptr->data.data(), ptr->data.size()};
        //}
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

  }

    //void loop()
    //{
    //// Create wait set
    //iox::popo::WaitSet<> waitSet;

    //// Create unordered_maps for clients and servers
    //std::unordered_map<iox::popo::Subscriber<ClientData>*, ClientSubscriber> clientSubscribers;
    //std::unordered_map<iox::popo::Subscriber<ServerData>*, ServerSubscriber> serverSubscribers;

    //// Assume we have a way to know how many clients and servers we expect
    //const int NUM_CLIENTS = 5;
    //const int NUM_SERVERS = 3;

    //// Create and attach client subscribers
    //for (int i = 0; i < NUM_CLIENTS; ++i) {
    //    std::string id = "Client" + std::to_string(i);
    //    auto [it, inserted] = clientSubscribers.emplace(std::piecewise_construct,
    //        std::forward_as_tuple(nullptr),
    //        std::forward_as_tuple(id));
    //    it->first = &it->second.getSubscriber();
    //    waitSet.attachState(*it->first, iox::popo::SubscriberState::HAS_DATA);
    //}

    //// Create and attach server subscribers
    //for (int i = 0; i < NUM_SERVERS; ++i) {
    //    std::string id = "Server" + std::to_string(i);
    //    auto [it, inserted] = serverSubscribers.emplace(std::piecewise_construct,
    //        std::forward_as_tuple(nullptr),
    //        std::forward_as_tuple(id));
    //    it->first = &it->second.getSubscriber();
    //    waitSet.attachState(*it->first, iox::popo::SubscriberState::HAS_DATA);
    //}

    //// Main event loop
    //while (true) {
    //    auto notificationVector = waitSet.wait();

    //    for (auto& notification : notificationVector) {
    //        auto& eventOrigin = notification.getOrigin();

    //        // Check if it's a client notification
    //        auto clientIt = clientSubscribers.find(static_cast<iox::popo::Subscriber<ClientData>*>(&eventOrigin));
    //        if (clientIt != clientSubscribers.end()) {
    //            std::cout << "Received notification from client: " << clientIt->second.getId() << std::endl;
    //            // Process client notification
    //            clientIt->second.processData();
    //            continue;
    //        }

    //        // Check if it's a server notification
    //        auto serverIt = serverSubscribers.find(static_cast<iox::popo::Subscriber<ServerData>*>(&eventOrigin));
    //        if (serverIt != serverSubscribers.end()) {
    //            std::cout << "Received notification from server: " << serverIt->second.getId() << std::endl;
    //            // Process server notification
    //            serverIt->second.processData();
    //            continue;
    //        }

    //        // If we get here, it's an unknown notification
    //        std::cerr << "Received notification from unknown source" << std::endl;
    //    }
    //}
    //}

}}

