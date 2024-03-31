
#include <string>
#include <optional>

#include <iceoryx_posh/popo/subscriber.hpp>
#include <iceoryx_posh/popo/untyped_subscriber.hpp>
#include <iceoryx_posh/popo/publisher.hpp>
#include <iceoryx_hoofs/posix_wrapper/signal_handler.hpp>

namespace mignificient { namespace executor {

  struct Invocation {

    iox::cxx::vector<uint8_t, 5 * 1024 * 1024> data;

    iox::cxx::string<64> id;

  };

  enum class Message {

    INVOKE = 0,
    YIELD = 1,
    FINISH = 2

  };

  struct InvocationData {
    const uint8_t* data;
    size_t size;
  };

  struct Runtime {

    Runtime(const std::string& name);

    InvocationData loop_wait();

    void gpu_yield();

    void finish();

  private:
    std::optional<iox::popo::Publisher<Message>> client;
    std::optional<iox::popo::UntypedSubscriber> orchestrator;
    std::optional<iox::popo::WaitSet<>> waitset;

    std::optional<iox::posix::SignalGuard> sigint;
    std::optional<iox::posix::SignalGuard> sigterm;

    const void* last_message;

    static bool _quit;
    static iox::popo::WaitSet<>* _waitset;
    static void _sigHandler(int sig);
  };

}}
