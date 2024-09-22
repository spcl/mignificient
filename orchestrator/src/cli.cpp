
#include <fstream>

#include <iceoryx_hoofs/posix_wrapper/signal_handler.hpp>
#include <json/json.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <mignificient/orchestrator/orchestrator.hpp>

int main(int argc, char ** argv)
{

  if(argc != 3) {
    return 1;
  }

  spdlog::info("Reading configuration from {}, device database from {}", argv[1], argv[2]);

  Json::Value config;
  {
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
  }

  mignificient::orchestrator::Orchestrator::init(config);
  mignificient::orchestrator::Orchestrator orchestrator{config, argv[2]};

  orchestrator.run();

  return 0;
}
