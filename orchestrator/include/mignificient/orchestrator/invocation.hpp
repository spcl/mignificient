#ifndef __MIGNIFICIENT_ORCHESTRATOR_INVOCATION_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_INVOCATION_HPP__

#include <string>

#include <drogon/HttpResponse.h>

namespace mignificient { namespace orchestrator {

  class ActiveInvocation {
  public:
      ActiveInvocation(drogon::HttpResponsePtr response, std::string payload)
          : response_(std::move(response)), payload_(std::move(payload)) {}

      drogon::HttpResponsePtr getResponse() const { return response_; }
      const std::string& getPayload() const { return payload_; }

  private:
      drogon::HttpResponsePtr response_;
      std::string payload_;
  };

}}

#endif
