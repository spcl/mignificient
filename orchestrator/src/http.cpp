#include <mignificient/orchestrator/http.hpp>

#include <drogon/drogon.h>
#include <drogon/HttpTypes.h>

namespace mignificient { namespace orchestrator {

  void HTTPServer::invoke(const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback)
  {
    auto body = req->getBody();
    std::string input(body.data(), body.length());

    auto invoc = ActiveInvocation::create(callback, input);
    if(invoc) {
      _trigger.trigger(std::move(invoc));
    }
  }

  void HTTPServer::containers(const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback)
  {
    AdminRequest admin_req;
    admin_req.type = AdminRequestType::LIST_CONTAINERS;
    admin_req.respond = [callback](Json::Value result) {
      auto resp = drogon::HttpResponse::newHttpJsonResponse(result);
      callback(resp);
    };

    _trigger.trigger_admin(std::move(admin_req));
  }

  void HTTPServer::kill_container(const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback)
  {
    auto body = req->getJsonObject();
    if (!body || !body->isMember("container") || !body->isMember("user")) {
      Json::Value error;
      error["success"] = false;
      error["error"] = "Missing 'user' or 'container' field in request body";
      auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
      resp->setStatusCode(drogon::k400BadRequest);
      callback(resp);
      return;
    }

    AdminRequest admin_req;
    admin_req.type = AdminRequestType::KILL_CONTAINER;
    admin_req.user = (*body)["user"].asString();
    admin_req.container = (*body)["container"].asString();
    admin_req.respond = [callback](Json::Value result) {
      auto resp = drogon::HttpResponse::newHttpJsonResponse(result);
      if (!result["success"].asBool()) {
        resp->setStatusCode(drogon::k404NotFound);
      }
      callback(resp);
    };

    _trigger.trigger_admin(std::move(admin_req));
  }

  HTTPServer::HTTPServer(Json::Value & config, HTTPTrigger& trigger):
    _trigger(trigger)
  {
    drogon::app().addListener("0.0.0.0", config["port"].asInt());
    spdlog::info("Listening on port {}", config["port"].asInt());

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

