
#include <chrono>
#include <cstdint>
#include <iceoryx_hoofs/posix_wrapper/signal_handler.hpp>
#include <iceoryx_posh/popo/subscriber.hpp>
#include <iceoryx_posh/popo/untyped_publisher.hpp>
#include <spdlog/spdlog.h>
#include <tuple>
#include <utility>

#include <spdlog/fmt/fmt.h>

#include "executor.hpp"

namespace mignificient { namespace orchestrator {

  bool quit = false;
  iox::popo::WaitSet<>* waitset_ptr = nullptr;
  std::optional<iox::posix::SignalGuard> sigint;
  std::optional<iox::posix::SignalGuard> sigterm;

  void sigHandler(int sig [[maybe_unused]])
  {
    quit = true;
    if(waitset_ptr)
    {
      waitset_ptr->markForDestruction();
    }
  }

  struct Client {
    std::string id;
    iox::popo::UntypedPublisher send;
    iox::popo::Subscriber<mignificient::executor::InvocationResult> recv;
    void* payload;

    Client(const std::string& name):
      id(name),
      send(
        iox::capro::ServiceDescription{
          iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, name},
          "Orchestrator",
          "Receive"
        }
      ),
      recv(
        iox::capro::ServiceDescription{
          iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, name},
          "Orchestrator",
          "Send"
        }
      )
    {
      payload = send.loan(sizeof(executor::Invocation), alignof(executor::Invocation)).value();
    }

    ~Client()
    {
      send.release(payload);
    }

    executor::Invocation& request()
    {
      return *static_cast<executor::Invocation*>(payload);
    }

    void send_request()
    {
      //std::cerr << "Publish " << name << " " << std::endl;
      send.publish(payload);
      payload = send.loan(sizeof(executor::Invocation), alignof(executor::Invocation)).value();
    }
  };

  struct Orchestrator {

    Orchestrator()
    {
      waitset_ptr = &waitset;
      sigint.emplace(iox::posix::registerSignalHandler(iox::posix::Signal::INT, sigHandler));
      sigterm.emplace(iox::posix::registerSignalHandler(iox::posix::Signal::TERM, sigHandler));
    }

    void add_client()
    {
      std::string client_name = fmt::format("client_{}", _client_id);

      const auto & [client, _] = clients.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(_client_id),
        std::forward_as_tuple(client_name)
      );

      waitset.attachState(client->second.recv, iox::popo::SubscriberState::HAS_DATA, _client_id).or_else([](auto) {
          std::cerr << "failed to attach subscriber" << std::endl;
          std::exit(EXIT_FAILURE);
      });

      _client_id++;
    }

    Client* client(int id)
    {
      auto it = clients.find(id);
      if(it != clients.end()) {
        return &(*it).second;
      } else {
        return nullptr;
      }
    }

    void wait()
    {
      while(!quit)
      {
        auto notificationVector = waitset.wait();

        for (auto& notification : notificationVector)
        {
          auto res = client(notification->getNotificationId())->recv.take();

          if(res.value().get()->msg == executor::Message::FINISH) {
            spdlog::info("Finish {}", std::string_view{reinterpret_cast<const char*>(res.value().get()->data.data()), res.value().get()->size});
          } else {
            spdlog::info("Yield {}");
          }
          //notification->getOrigin<typename T>()
          //auto value = orchestrator.value().take();

          //if(value.has_error()) {
          //  std::cout << "got no data, return code: " << static_cast<uint64_t>(value.get_error()) << std::endl;
          //} else {

          //  last_message = value.value();
          //  auto* ptr = static_cast<const Invocation*>(last_message);
          //  return InvocationData{ptr->data.data(), ptr->data.size()};
          //}
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

    }

  private:

    int _client_id = 0;
    int _invoc_id = 0;

    // FIXME: server activate
    std::unordered_map<int, Client> clients;
    iox::popo::WaitSet<> waitset;

    const void* last_message;

    static bool _quit;
    static iox::popo::WaitSet<>* _waitset;
    static void _sigHandler(int sig);
  };

}}

int main(int argc, char ** argv)
{
  constexpr char APP_NAME[] = "gpuless-orchestrator";
  iox::runtime::PoshRuntime::initRuntime(APP_NAME);

  mignificient::orchestrator::Orchestrator orchestrator;

  orchestrator.add_client();

  auto& client = *orchestrator.client(0);
  client.request().id = "test";
  client.request().data.resize(4);
  int val = 42;
  memcpy(client.request().data.data(), &val, sizeof(int));
  client.request().size = 4;
  client.send_request();

  orchestrator.wait();

}
