
#include <fstream>

#include <iceoryx_hoofs/posix_wrapper/signal_handler.hpp>
#include <json/json.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <mignificient/orchestrator/orchestrator.hpp>

int main(int argc, char ** argv)
{

  if(argc != 2) {
    return 1;
  }

  spdlog::info("Reading configuration from {}", argv[1]);

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

  std::cerr << config << std::endl;

  mignificient::orchestrator::Orchestrator::init(config);
  mignificient::orchestrator::Orchestrator orchestrator{config};

  orchestrator.run();

  return 0;
}
