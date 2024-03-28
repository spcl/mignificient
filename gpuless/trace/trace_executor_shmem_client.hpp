#ifndef GPULESS_TRACE_EXECUTOR_SHMEM_H
#define GPULESS_TRACE_EXECUTOR_SHMEM_H

#include "trace_executor.hpp"

#include <iceoryx_posh/internal/runtime/posh_runtime_impl.hpp>
#include <iceoryx_posh/popo/untyped_client.hpp>

namespace gpuless {

class TraceExecutorShmem : public TraceExecutor {
  private:
    sockaddr_in manager_addr{};
    sockaddr_in exec_addr{};
    int32_t session_id_ = -1;

    uint64_t synchronize_counter_ = 0;
    double synchronize_total_time_ = 0;

    // Not great - internal feature - but we don't have a better solution.
    std::unique_ptr<iox::runtime::PoshRuntimeImpl> _impl;
    std::unique_ptr<iox::popo::UntypedClient> client;

  private:
    bool negotiateSession(manager::instance_profile profile);
    bool getDeviceAttributes();

  public:
    TraceExecutorShmem();
    ~TraceExecutorShmem();

    static void init_runtime();
    static void reset_runtime();

    bool init(const char *ip, short port,
              manager::instance_profile profile) override;
    bool synchronize(gpuless::CudaTrace &cuda_trace) override;
    bool deallocate() override;

    double getSynchronizeTotalTime() const override;

    static iox::runtime::PoshRuntime* runtime_factory_impl(iox::cxx::optional<const iox::RuntimeName_t*> var, TraceExecutorShmem* ptr = nullptr);
};

} // namespace gpuless

#endif // GPULESS_TRACE_EXECUTOR_TCP_H
