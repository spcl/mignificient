

#include <memory>

#include <drogon/drogon.h>

namespace mignificient { namespace orchestrator {

  class HTTPServer : public drogon::HttpController<HTTPServer, false>, public std::enable_shared_from_this<HTTPServer> {
  public:

      METHOD_LIST_BEGIN
      ADD_METHOD_TO(HTTPServer::invoke, "/invoke", drogon::Post);
      METHOD_LIST_END

      void invoke(const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback);

      HTTPServer(Json::Value & config);
      void run();
      void shutdown();
      void wait();

  private:

      std::thread _server_thread;
  };

}}

