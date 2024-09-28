#ifndef __MIGNIFICIENT_ORCHESTRATOR_INVOCATION_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_INVOCATION_HPP__

#include <stdexcept>
#include <string>

#include <drogon/HttpResponse.h>
#include <drogon/HttpTypes.h>
#include <spdlog/spdlog.h>

namespace mignificient { namespace orchestrator {

  class ActiveInvocation {
  public:

    static std::unique_ptr<ActiveInvocation> create(
      std::function<void(const drogon::HttpResponsePtr&)> http_callback,
      std::string payload
    )
    {
      // Parse input data
      Json::Value input_data;
      Json::Reader reader;
      if (!reader.parse(payload, input_data)) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k400BadRequest);
        resp->setContentTypeCode(drogon::CT_TEXT_PLAIN);
        resp->setBody("Couldn't parse the input data");

        http_callback(resp);
        return nullptr;
      }

      for(const auto& field : {"function", "user", "uuid", "modules", "mig-instance", "gpu-memory"}) {
        if (!input_data.isMember(field)) {

          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k400BadRequest);
          resp->setContentTypeCode(drogon::CT_TEXT_PLAIN);
          resp->setBody(fmt::format("Missing key {}", field));

          http_callback(resp);
          return nullptr;
        }
      }

      try {
        return std::make_unique<ActiveInvocation>(http_callback, input_data);
      } catch (...) {

        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k400BadRequest);
        resp->setContentTypeCode(drogon::CT_TEXT_PLAIN);
        resp->setBody("Parsing error");

        http_callback(resp);

        return nullptr;
      }
    }

    ActiveInvocation(
      std::function<void(const drogon::HttpResponsePtr&)> http_callback,
      Json::Value& input_data
    ):
      _http_callback(std::move(http_callback))
    {

      _input_payload = input_data["input-payload"].asString();
      _function_name = input_data["function"].asString();
      _container = input_data["container"].asString();
      _function_path = input_data["function-path"].asString();
      _user = input_data["user"].asString();
      _uuid = input_data["uuid"].asString();

      int i = 0;
      for(Json::Value& module : input_data["modules"]) {
        _modules[i++] = module.asString();
      }

      _mig_instance = input_data["mig-instance"].asString();
      _gpu_memory = input_data["gpu-memory"].asInt();

    }

    void failure(const std::string& reason)
    {
      auto resp = drogon::HttpResponse::newHttpResponse();
      resp->setStatusCode(drogon::k503ServiceUnavailable);
      resp->setContentTypeCode(drogon::CT_TEXT_PLAIN);

      resp->setBody(reason);

      _http_callback(resp);
    }

    void respond(std::string_view response)
    {
      auto resp = drogon::HttpResponse::newHttpResponse();
      resp->setStatusCode(drogon::k200OK);
      resp->setContentTypeCode(drogon::CT_TEXT_PLAIN);

      // Zero-copy operation
      resp->setViewBody(response.begin(), response.size());
      // This one works with default implementation
      //resp->setBody(std::string{response.begin(), response.size()});

      _http_callback(resp);
    }

    const std::string& function_name() const
    {
      return _function_name;
    }

    const std::string& function_path() const
    {
      return _function_path;
    }

    const std::string& input() const
    {
      return _input_payload;
    }

    const std::string& user() const
    {
      return _user;
    }

    const std::string& uuid() const
    {
      return _uuid;
    }

    float gpu_memory() const
    {
      return _gpu_memory;
    }

  private:
    std::function<void(const drogon::HttpResponsePtr&)> _http_callback;

    std::string _input_payload;
    std::string _function_name;
    std::string _function_path;
    std::string _container;
    std::string _user;
    std::string _mig_instance;
    std::string _uuid;
    float _gpu_memory;
    std::array<std::string, 5> _modules;
  };

}}

#endif
