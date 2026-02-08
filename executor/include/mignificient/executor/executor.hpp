#ifndef __MIGNIFICIENT_EXECUTOR_EXECUTOR_HPP__
#define __MIGNIFICIENT_EXECUTOR_EXECUTOR_HPP__

#include <cstdint>
#include <iox2/service_type.hpp>
#include <string>
#include <optional>

#include <iceoryx_posh/mepoo/chunk_header.hpp>
#include <iceoryx_posh/popo/subscriber.hpp>
#include <iceoryx_posh/popo/untyped_subscriber.hpp>
#include <iceoryx_posh/popo/publisher.hpp>
#include <iceoryx_hoofs/posix_wrapper/signal_handler.hpp>

#include "function.hpp"

#include <mignificient/ipc/config.hpp>

#ifdef MIGNIFICIENT_WITH_ICEORYX2
#include <iox2/iceoryx2.hpp>
#endif

namespace mignificient { namespace executor {

  struct SwapResult {
    static constexpr const char* IOX2_TYPE_NAME = "SwapResult";

    double time_us;
    size_t memory_bytes;
    int status;
  };

  struct Invocation {
    // FIXME: replace with static array
    static constexpr int CAPACITY = 1 * 1024 * 1024;
    static constexpr int ID_LEN = 64;
    //static constexpr int CAPACITY = 5 * 1024 * 1024;
    //static constexpr int CAPACITY = 10 * 1024;
    //iox::cxx::vector<uint8_t, CAPACITY> data;
    //std::array<uint8_t, CAPACITY> data;
    uint8_t data[CAPACITY];
    size_t size;
    //iox::string<64> id;
    char id[ID_LEN];
  };

  enum class Message {

    YIELD = 0,
    FINISH = 1,
    REGISTER = 2

  };

  struct InvocationResult {
    static constexpr int CAPACITY = 5 * 1024 * 1024;
    Message msg;
    //iox::vector<uint8_t, CAPACITY> data;
    uint8_t data[CAPACITY];
    size_t size;
  };

  struct CommunicationIceoryxV1
  {

    std::optional<iox::popo::Publisher<InvocationResult>> client;
    std::optional<iox::popo::UntypedSubscriber> orchestrator;
    std::optional<iox::popo::WaitSet<>> waitset;

    static iox::popo::WaitSet<>* _waitset;

    std::optional<iox::posix::SignalGuard> sigint;
    std::optional<iox::posix::SignalGuard> sigterm;

    std::optional<iox::popo::Sample<InvocationResult, iox::mepoo::NoUserHeader>> _result;
    std::optional<iox::popo::Sample<InvocationResult, iox::mepoo::NoUserHeader>> _yield_msg;

    CommunicationIceoryxV1(const std::string& name);

    void gpu_yield();
    void register_runtime();
    void finish(int size);

    InvocationData loop_wait();

    const void* last_message;
  };

#ifdef MIGNIFICIENT_WITH_ICEORYX2
  struct CommunicationIceoryxV2
  {
    std::optional<iox2::Node<iox2::ServiceType::Ipc>> iox2_node;

    std::optional<iox2::PortFactoryEvent<iox2::ServiceType::Ipc>> client_event_listen;
    std::optional<iox2::PortFactoryEvent<iox2::ServiceType::Ipc>> client_event_notify;
    std::optional<iox2::WaitSet<iox2::ServiceType::Ipc>> waitset;

    std::optional<iox2::Publisher<iox2::ServiceType::Ipc, mignificient::executor::InvocationResult, void>> client_send;
    std::optional<iox2::Notifier<iox2::ServiceType::Ipc>> client_notifier;

    std::optional<iox2::Subscriber<iox2::ServiceType::Ipc, mignificient::executor::Invocation, void>> orchestrator_recv;
    std::optional<iox2::Listener<iox2::ServiceType::Ipc>> orchestrator_listener;
    std::optional<iox2::WaitSetGuard<iox2::ServiceType::Ipc>> orchestrator_listener_guard;

    std::optional<iox2::Sample<iox2::ServiceType::Ipc, Invocation, void>> _last_message;
    std::optional<iox2::SampleMutUninit<iox2::ServiceType::Ipc, InvocationResult, void>> _result;
    std::optional<iox2::SampleMutUninit<iox2::ServiceType::Ipc, InvocationResult, void>> _yield_msg;

    CommunicationIceoryxV2(const std::string& id);

    void gpu_yield();
    void register_runtime();
    void finish(int size);

    InvocationData loop_wait();
  };
#endif

  struct Runtime {

    Runtime(mignificient::ipc::IPCBackend config, const std::string& name);

    InvocationData loop_wait();

    void gpu_yield();

    void register_runtime();

    void finish(int size);

    InvocationResultData result();

    static void sigHandler(int sig);

    static bool quit;

  private:

    std::optional<CommunicationIceoryxV1> _comm_v1;
#ifdef MIGNIFICIENT_WITH_ICEORYX2
    std::optional<CommunicationIceoryxV2> _comm_v2;
#endif

    mignificient::ipc::IPCBackend _backend;

  };

}}

#endif
