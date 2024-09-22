#include <mignificient/orchestrator/http.hpp>

#include <drogon/drogon.h>
#include <drogon/HttpTypes.h>

namespace mignificient { namespace orchestrator {

  void HTTPServer::invoke(const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback)
  {

    auto body = req->getBody();
    std::string input(body.data(), body.length());

    _trigger.trigger({callback, input});
  }

  HTTPServer::HTTPServer(Json::Value & config, HTTPTrigger& trigger):
    _trigger(trigger)
  {
    drogon::app().addListener("0.0.0.0", config["port"].asInt());

    drogon::app().setThreadNum(config["threads"].asInt());

    drogon::app().enableServerHeader(false);
    drogon::app().setLogPath("./");
    drogon::app().setLogLevel(trantor::Logger::kWarn);

    drogon::app().disableSigtermHandling();

  }

  void HTTPServer::run()
  {
    drogon::app().registerController<HTTPServer>(shared_from_this());

    _server_thread = std::thread([this]() {
        drogon::app().run();
    });

  }

  void HTTPServer::shutdown()
  { 
		drogon::app().quit();
  }

  void HTTPServer::wait()
  {
    if (_server_thread.joinable()) {
      _server_thread.join();
    }
  }

}}

