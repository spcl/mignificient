#ifndef __MIGNIFICIENT_ORCHESTRATOR_INVOCATION_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_INVOCATION_HPP__

#include <chrono>
#include <stdexcept>
#include <string>

#include <drogon/HttpResponse.h>
#include <drogon/HttpTypes.h>
#include <spdlog/spdlog.h>

namespace mignificient { namespace orchestrator {

  enum class Language {
    CPP,
    PYTHON
  };

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

      for(const auto& field : {"function", "user", "uuid", "modules", "mig-instance", "gpu-memory", "timeout"}) {
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
      _begin = std::chrono::high_resolution_clock::now();

      _input_payload = input_data["input-payload"].asString();
      _function_name = input_data["function"].asString();
      _function_handler = input_data["function-handler"].asString();
      _container = input_data["container"].asString();
      _function_path = input_data["function-path"].asString();
      _user = input_data["user"].asString();
      _uuid = input_data["uuid"].asString();

      if(input_data["function-language"].asString() == "cpp") {
        _function_language = Language::CPP;
      } else if(input_data["function-language"].asString() == "python") {
        _function_language = Language::PYTHON;
      }

      _cubin_analysis = input_data["cubin-analysis"].asString();
      //_cuda_binary = input_data["cuda-binary"].asString();
      _ld_preload = !input_data["ld-preload"].isNull() ? input_data["ld-preload"].asString() : std::optional<std::string>{};

      int i = 0;
      for(Json::Value& module : input_data["modules"]) {
        _modules[i++] = module.asString();
      }

      _mig_instance = input_data["mig-instance"].asString();
      _gpu_memory = input_data["gpu-memory"].asInt();

      _timeout_us = static_cast<int64_t>(input_data["timeout"].asDouble() * 1e6);
    }

    void failure(const std::string& reason)
    {
      auto resp = drogon::HttpResponse::newHttpResponse();
      resp->setStatusCode(drogon::k503ServiceUnavailable);
      resp->setContentTypeCode(drogon::CT_TEXT_PLAIN);

      resp->setBody(reason);
      resp->setBody(fmt::format("{{\"result\": null, \"error\": \"{}\"}}", reason));

      _http_callback(resp);
    }

    void respond(std::string_view response)
    {
      auto end = std::chrono::high_resolution_clock::now();
      auto resp = drogon::HttpResponse::newHttpResponse();
      resp->setStatusCode(drogon::k200OK);
      resp->setContentTypeCode(drogon::CT_TEXT_PLAIN);

      // Zero-copy operation
      //resp->setViewBody(response.begin(), response.size());
      resp->setBody(fmt::format("{{\"result\": \"{}\"}}", response));
      // This one works with default implementation
      //resp->setBody(std::string{response.begin(), response.size()});

      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - _begin).count();
      SPDLOG_DEBUG("[Invoc] Responding to the HTTP request for invocation {} from user {} after {} ms", _uuid, _user, duration / 1000.0);
      _http_callback(resp);
    }

    const std::string& function_name() const
    {
      return _function_name;
    }

    const std::string& function_handler() const
    {
      return _function_handler;
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

    Language language() const
    {
      return _function_language;
    }

    const std::optional<std::string>& ld_preload() const
    {
      return _ld_preload;
    }

    // FIXME: remove
    const std::string& cuda_binary() const
    {
      return _cuda_binary;
    }

    const std::string& cubin_analysis() const
    {
      return _cubin_analysis;
    }

    const std::string& uuid() const
    {
      return _uuid;
    }

    float gpu_memory() const
    {
      return _gpu_memory;
    }

    void mark_started()
    {
      _dispatch_time = std::chrono::high_resolution_clock::now();
    }

    bool is_timed_out() const
    {
      auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - _dispatch_time
      ).count();
      return elapsed > _timeout_us;
    }

    int64_t timeout_seconds() const
    {
      return _timeout_us / 1000000;
    }

    void respond_timeout()
    {
      auto resp = drogon::HttpResponse::newHttpResponse();
      resp->setStatusCode(drogon::k200OK);
      resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
      resp->setBody(fmt::format("{{\"result\": null, \"error\": \"timeout after {} seconds\"}}", timeout_seconds()));
      _http_callback(resp);
    }

  private:
    std::function<void(const drogon::HttpResponsePtr&)> _http_callback;

    decltype(std::chrono::high_resolution_clock::now()) _begin;
    Language _function_language;
    std::string _cuda_binary;
    std::string _cubin_analysis;

    std::optional<std::string> _ld_preload;
    std::string _input_payload;
    std::string _function_name;
    std::string _function_handler;
    std::string _function_path;
    std::string _container;
    std::string _user;
    std::string _mig_instance;
    std::string _uuid;
    float _gpu_memory;
    int64_t _timeout_us = 0;
    decltype(std::chrono::high_resolution_clock::now()) _dispatch_time;
    std::array<std::string, 5> _modules;
  };

}}

#endif
