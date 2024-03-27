#ifndef GPULESS_TRACEEXECUTOR_H
#define GPULESS_TRACEEXECUTOR_H

#include "cuda_api_calls.hpp"
#include "cuda_trace.hpp"
#include "manager/manager.hpp"
#include "manager/manager_device.hpp"

namespace gpuless {

class CudaTrace;

class TraceExecutor {
  protected:
    size_t device_total_mem = 0;
    std::vector<int32_t> device_attributes;

  public:
    virtual bool init(const char *ip, short port,
                      gpuless::manager::instance_profile profile) = 0;
    virtual bool synchronize(gpuless::CudaTrace &cuda_trace) = 0;
    virtual bool deallocate() = 0;

    size_t totalMem() const { return this->device_total_mem; }
    int32_t deviceAttribute(CUdevice_attribute attribute) {
        if (device_attributes.size() < static_cast<unsigned>(attribute)) {
            SPDLOG_ERROR("Device attribute {} not stored", attribute);
            std::exit(EXIT_FAILURE);
        }
        return device_attributes[attribute];
    }

    virtual double getSynchronizeTotalTime() const { return 0.0; };
};

} // namespace gpuless

#endif // GPULESS_TRACEEXECUTOR_H
