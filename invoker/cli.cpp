
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

void independent(const std::string& address, int iterations, int parallel_requests, Json::Value& input_data, bool different_users, std::vector<InvocatonResult>& results, const std::string& output_file)
{
  std::vector<drogon::HttpRequestPtr> requests;
  std::vector<drogon::HttpClientPtr> clients;

  //trantor::Logger::setLogLevel(trantor::Logger::kTrace);

  for (int j = 0; j < parallel_requests; ++j) {

    for(int i = 0; i < iterations; ++i) {

      drogon::HttpRequestPtr req;
      if(input_data.isArray()) {

        input_data[j]["uuid"] = fmt::format("invoc-{}-{}", i, j);
        if(different_users) {
          input_data[j]["user"] = fmt::format("user-{}", j);
        } else {
          input_data[j]["user"] = "user";
        }
        req = drogon::HttpRequest::newHttpJsonRequest(input_data[j]);

      } else {

        input_data["uuid"] = fmt::format("invoc-{}-{}", i, j);
        if(different_users) {
          input_data["user"] = fmt::format("user-{}", j);
        } else {
          input_data["user"] = "user";
        }
        req = drogon::HttpRequest::newHttpJsonRequest(input_data);

      }

      req->setMethod(drogon::Post);
      req->setPath("/invoke");

      requests.push_back(std::move(req));
      clients.push_back(drogon::HttpClient::newHttpClient(address));

    }

  }

  {
    std::vector<std::thread> threads;
    for (int i = 0; i < parallel_requests; ++i) {

      threads.emplace_back(
        [iterations, &results, &requests, &clients, parallel_requests, i]() {

          for(int j = 0; j < 1; ++j) {

            auto& res = results[i*parallel_requests + j];
            SPDLOG_DEBUG("Send worker {}, iter {}", i, j);

            res.start = std::chrono::high_resolution_clock::now();
            auto [result, response] = clients[i]->sendRequest(requests[i*iterations + j]);
            res.end = std::chrono::high_resolution_clock::now();

            if(result != drogon::ReqResult::Ok || response->getStatusCode() != drogon::HttpStatusCode::k200OK) {
              spdlog::error("Failed invocation! Result {}");

              if(response) {
                spdlog::error("Status {} Body {}", drogon::to_string_view(result), response->getStatusCode(), response->body());
              }
            } else {
              SPDLOG_DEBUG("Finished invocation. Result {} Body {}", drogon::to_string_view(result), response->body());
            }

          }

        }
      );

    }

    for(auto & t : threads) {
      t.join();
    }
  }

  auto begin = std::chrono::high_resolution_clock::now();
  std::vector<std::thread> threads;
  for (int i = 0; i < parallel_requests; ++i) {

    threads.emplace_back(
      [iterations, &results, &requests, &clients, parallel_requests, i]() {

        for(int j = 0; j < iterations; ++j) {

          auto& res = results[i*iterations + j + 1];
          //SPDLOG_DEBUG("Send worker {}, iter {}, idx {}", i, j, i*iterations + j);

          res.start = std::chrono::high_resolution_clock::now();
          auto [result, response] = clients[i]->sendRequest(requests[i*iterations + j]);
          res.end = std::chrono::high_resolution_clock::now();
          SPDLOG_DEBUG("Finished worker {}, iter {}", i, j);

          if(result != drogon::ReqResult::Ok || response->getStatusCode() != drogon::HttpStatusCode::k200OK) {
            spdlog::error("Failed invocation! Result {} Status {} Body {}", drogon::to_string_view(result), response->getStatusCode(), response->body());
          } else {
            spdlog::info("Finished invocation. Result {} Body {}", drogon::to_string_view(result), response->body());
          }

        }

      }
    );

  }

  for(auto & t : threads) {
    t.join();
  }
  auto end = std::chrono::high_resolution_clock::now();

  spdlog::info("Total time: {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0);
  std::ofstream out{output_file};
  out << "iteration,worker,time\n";

  for (int j = 0; j < parallel_requests; ++j) {

    for(int i = 0; i < iterations + 1; ++i) {

      auto& res = results[j*iterations + i];
      out << fmt::format(
        "{},{},{}\n",
        i, j,
        std::chrono::duration_cast<std::chrono::microseconds>(res.end-res.start).count()
      );

    }

  }

  for(int j = 0; j < parallel_requests; ++j) {

    for (int i = 0; i < iterations; ++i) {

      auto& res = results[j*iterations + i];
      spdlog::info(
        "Client {}, Invocation {}, Time {}",
        j, i,
        std::chrono::duration_cast<std::chrono::microseconds>(res.end-res.start).count()
      );

    }

  }
}

void batches(const std::string& address, int iterations, int parallel_requests, Json::Value& input_data, bool different_users, std::vector<InvocatonResult>& results, const std::string& output_file)
{
  std::vector<drogon::HttpRequestPtr> requests;
  std::vector<drogon::HttpClientPtr> clients;

  //trantor::Logger::setLogLevel(trantor::Logger::kTrace);
  for(int i = 0; i < iterations; ++i) {

    for (int j = 0; j < parallel_requests; ++j) {

      drogon::HttpRequestPtr req;
      if(input_data.isArray()) {

        input_data[j]["uuid"] = fmt::format("invoc-{}-{}", i, j);
        if(different_users) {
          input_data[j]["user"] = fmt::format("user-{}", j);
        } else {
          input_data[j]["user"] = "user";
        }
        req = drogon::HttpRequest::newHttpJsonRequest(input_data[j]);

      } else {

        input_data["uuid"] = fmt::format("invoc-{}-{}", i, j);
        if(different_users) {
          input_data["user"] = fmt::format("user-{}", j);
        } else {
          input_data["user"] = "user";
        }
        req = drogon::HttpRequest::newHttpJsonRequest(input_data);

      }

      req->setMethod(drogon::Post);
      req->setPath("/invoke");

      requests.push_back(std::move(req));
      clients.push_back(drogon::HttpClient::newHttpClient(address));

    }

  }

  // Would be simpler with std::atoic::wait but requires C++20
  std::mutex mutex;
  std::condition_variable cv;

  auto begin = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < iterations; ++i) {

    int count = 0;

    for (int j = 0; j < parallel_requests; ++j) {

        auto& res = results[i*parallel_requests + j];
        res.start = std::chrono::high_resolution_clock::now();
        clients[j]->sendRequest(
          requests[i*parallel_requests + j],
          [&res, &count, &mutex, &cv, parallel_requests](drogon::ReqResult result, const drogon::HttpResponsePtr& response) mutable {

            res.end = std::chrono::high_resolution_clock::now();

            if(result != drogon::ReqResult::Ok || response->getStatusCode() != drogon::HttpStatusCode::k200OK) {
              spdlog::error("Failed invocation! Result {}");

              if(response) {
                spdlog::error("Status {} Body {}", drogon::to_string_view(result), response->getStatusCode(), response->body());
              }
            } else {
              spdlog::info("Finished invocation. Result {} Body {}", drogon::to_string_view(result), response->body());
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
  auto end = std::chrono::high_resolution_clock::now();

  spdlog::info("Total time: {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0);
  std::ofstream out{output_file};
  out << "iteration,worker,time\n";
  for(int i = 0; i < iterations; ++i) {

    for (int j = 0; j < parallel_requests; ++j) {

      auto& res = results[i*parallel_requests + j];
      out << fmt::format(
        "{},{},{}\n",
        i, j,
        std::chrono::duration_cast<std::chrono::microseconds>(res.end-res.start).count()
      );

    }

  }
}

int main(int argc, char ** argv)
{
  Json::Value config;

  if(argc != 3) {
    return 1;
  }

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

  std::string mode = config["mode"].asString();
  std::string address = config["address"].asString();
  int iterations = config["iterations"].asInt();
  int parallel_requests = config["parallel-requests"].asInt();
  bool different_users = config["different-users"].asBool();
  Json::Value input_data = config["inputs"];
  std::string output_file{argv[2]};

  drogon::app().setThreadNum(parallel_requests);

  std::thread drogon_thread([]() {
      drogon::app().run();
  });

  std::vector<InvocatonResult> results;
  results.resize((iterations + 1)* parallel_requests);

  if(mode == "batches") {
    batches(address, iterations, parallel_requests, input_data, different_users, results, output_file);
  } else if(mode == "independent") {
    independent(address, iterations, parallel_requests, input_data, different_users, results, output_file);
  }

  drogon::app().quit();
  drogon_thread.join();

  return 0;
}
