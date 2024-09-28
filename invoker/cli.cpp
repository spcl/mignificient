
#include <chrono>
#include <drogon/HttpAppFramework.h>
#include <fstream>

#include <drogon/HttpClient.h>
#include <drogon/HttpRequest.h>
#include <drogon/HttpTypes.h>
#include <json/json.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

struct InvocatonResult
{
  std::chrono::high_resolution_clock::time_point start, end;
  double time;
  int iteration;
  int invocation;
};

int main(int argc, char ** argv)
{
  Json::Value config;

  if(argc == 2) {

    spdlog::info("Reading configuration from {}", argv[1]);
    std::ifstream cfg_data{argv[1]};

    if(!cfg_data.is_open()) {
      spdlog::error("Failed to open file with JSON config!");
      return 1;
    }

    Json::CharReaderBuilder builder;
    JSONCPP_STRING errs;
    if (!Json::parseFromStream(builder, cfg_data, &config, &errs)) {
      spdlog::error("Failed to parse JSON config! {}", errs);
      return 1;
    }

  } else {

    spdlog::info("Reading configuration from stdin");

    Json::CharReaderBuilder builder;
    JSONCPP_STRING errs;
    if (!Json::parseFromStream(builder, std::cin, &config, &errs)) {
      spdlog::error("Failed to parse JSON config! {}", errs);
      return 1;
    }
  }

  std::string address = config["address"].asString();
  int iterations = config["iterations"].asInt();
  int parallel_requests = config["parallel-requests"].asInt();
  bool different_users = config["different-users"].asBool();
  Json::Value input_data = config["input"];

  auto client = drogon::HttpClient::newHttpClient(address);

  std::vector<drogon::HttpRequestPtr> requests;
  std::vector<InvocatonResult> results;
  results.resize(iterations * parallel_requests);

  //trantor::Logger::setLogLevel(trantor::Logger::kTrace);
  for(int i = 0; i < iterations; ++i) {

    for (int j = 0; j < parallel_requests; ++j) {

      input_data["uuid"] = fmt::format("invoc-{}-{}", i, j);
      if(different_users) {
        input_data["user"] = fmt::format("user-{}", j);
      } else {
        input_data["user"] = "user";
      }

      auto req = drogon::HttpRequest::newHttpJsonRequest(input_data);
      req->setMethod(drogon::Post);
      req->setPath("/invoke");

      requests.push_back(std::move(req));

    }

  }

  drogon::app().setThreadNum(parallel_requests);

  std::thread drogon_thread([]() {
      drogon::app().run();
  });

  // Would be simpler with std::atoic::wait but requires C++20
  std::mutex mutex;
  std::condition_variable cv;

  for(int i = 0; i < iterations; ++i) {

    int count = 0;

    for (int j = 0; j < parallel_requests; ++j) {

        auto& res = results[i*parallel_requests + j];
        res.start = std::chrono::high_resolution_clock::now();
        client->sendRequest(
          requests[i*parallel_requests + j],
          [&res, &count, &mutex, &cv, parallel_requests](drogon::ReqResult result, const drogon::HttpResponsePtr& response) mutable {

            res.end = std::chrono::high_resolution_clock::now();

            if(result != drogon::ReqResult::Ok) {
              spdlog::error("Failed invocation! Result {}", drogon::to_string_view(result));
            } else {
              spdlog::debug("Finished invocation. Result {}", drogon::to_string_view(result));
            }

            std::lock_guard<std::mutex> lock(mutex);
            if(++count == parallel_requests) {
              cv.notify_one();
            }
          }
        );
    }

    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&count, parallel_requests]() { return count == parallel_requests; });

  }

  for(int i = 0; i < iterations; ++i) {

    for (int j = 0; j < parallel_requests; ++j) {

      auto& res = results[i*parallel_requests + j];
      spdlog::info(
        "Iteration {}, Invocation {}, Time {}",
        i, j,
        std::chrono::duration_cast<std::chrono::microseconds>(res.end-res.start).count()
      );

    }

  }

  drogon::app().quit();
  drogon_thread.join();

  return 0;
}