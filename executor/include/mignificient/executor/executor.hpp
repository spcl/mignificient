
#include <string>
#include <optional>

#include <iceoryx_posh/mepoo/chunk_header.hpp>
#include <iceoryx_posh/popo/subscriber.hpp>
#include <iceoryx_posh/popo/untyped_subscriber.hpp>
#include <iceoryx_posh/popo/publisher.hpp>
#include <iceoryx_hoofs/posix_wrapper/signal_handler.hpp>

#include "function.hpp"

namespace mignificient { namespace executor {

  struct Invocation {
    static constexpr int CAPACITY = 5 * 1024 * 1024;
    iox::cxx::vector<uint8_t, CAPACITY> data;
    size_t size;
    iox::cxx::string<64> id;
  };

  enum class Message {

    YIELD = 0,
    FINISH = 1

  };

  struct InvocationResult {
    static constexpr int CAPACITY = 5 * 1024 * 1024;
    iox::cxx::vector<uint8_t, CAPACITY> data;
    size_t size;
    Message msg;
  };

  struct Runtime {

    Runtime(const std::string& name);

    InvocationData loop_wait();

    void gpu_yield();

    void finish(int size);

    InvocationResultData result();

  private:

    std::optional<iox::popo::Publisher<InvocationResult>> client;
    std::optional<iox::popo::UntypedSubscriber> orchestrator;
    std::optional<iox::popo::WaitSet<>> waitset;

    std::optional<iox::posix::SignalGuard> sigint;
    std::optional<iox::posix::SignalGuard> sigterm;

    std::optional<iox::popo::Sample<InvocationResult, iox::mepoo::NoUserHeader>> _result;

    const void* last_message;

    static bool _quit;
    static iox::popo::WaitSet<>* _waitset;
    static void _sigHandler(int sig);
  };

}

}
