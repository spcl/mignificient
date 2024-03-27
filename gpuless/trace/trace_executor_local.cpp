#include "trace_executor_local.hpp"
#include "dlsym_util.hpp"
#include <dlfcn.h>
#include <iostream>
#include <spdlog/spdlog.h>

namespace gpuless {

TraceExecutorLocal::TraceExecutorLocal() {}

TraceExecutorLocal::~TraceExecutorLocal() = default;

bool TraceExecutorLocal::init(const char *ip, const short port,
                              manager::instance_profile profile) {
    return true;
}

bool TraceExecutorLocal::synchronize(gpuless::CudaTrace &cuda_trace) {
    this->synchronize_counter_++;
    SPDLOG_INFO(
        "TraceExecutorLocal::synchronize() [synchronize_counter={}, size={}]",
        this->synchronize_counter_, cuda_trace.callStack().size());

    auto &vdev = this->cuda_virtual_device_;
    this->cuda_virtual_device_.initRealDevice();

    std::set<uint64_t> required_modules;
    std::set<std::string> required_functions;
    for (auto &apiCall : cuda_trace.callStack()) {
        auto rmod_vec = apiCall->requiredCudaModuleIds();
        required_modules.insert(rmod_vec.begin(), rmod_vec.end());
        auto rfunc_vec = apiCall->requiredFunctionSymbols();
        required_functions.insert(rfunc_vec.begin(), rfunc_vec.end());
    }

    for (const auto &rmod_id : required_modules) {
        auto it = cuda_trace.getModuleIdToFatbinResource().find(rmod_id);
        if (it == cuda_trace.getModuleIdToFatbinResource().end()) {
            SPDLOG_ERROR("Required module {} unknown");
            return false;
        }

        const void *resource_ptr = std::get<0>(it->second);
        bool is_loaded = std::get<2>(it->second);

        if (!is_loaded) {
            CUmodule mod;
            checkCudaErrors(cuModuleLoadData(&mod, resource_ptr));
            vdev.module_registry_.emplace(rmod_id, mod);
            std::get<2>(it->second) = true;
            SPDLOG_DEBUG("Loading module: {}", rmod_id);
        }
    }

    for (const auto &rfunc : required_functions) {
        auto it = cuda_trace.getSymbolToModuleId().find(rfunc);
        if (it == cuda_trace.getSymbolToModuleId().end()) {
            SPDLOG_ERROR("Required function {} unknown");
        }

        auto t = it->second;
        uint64_t module_id = std::get<0>(t);
        bool fn_is_loaded = std::get<1>(t);

        if (!fn_is_loaded) {
            auto mod_it =
                cuda_trace.getModuleIdToFatbinResource().find(module_id);
            if (mod_it == cuda_trace.getModuleIdToFatbinResource().end()) {
                SPDLOG_ERROR("Unknown module {} for function", module_id,
                              rfunc);
            }

            bool module_is_loaded = std::get<2>(mod_it->second);
            if (!module_is_loaded) {
                SPDLOG_ERROR("Module {} not previously loaded", module_id);
            }

            auto mod_reg_it = vdev.module_registry_.find(module_id);
            if (mod_reg_it == vdev.module_registry_.end()) {
                SPDLOG_ERROR("Module {} not in registry", module_id);
            }
            CUmodule module = mod_reg_it->second;

            CUfunction func;
            checkCudaErrors(cuModuleGetFunction(&func, module, rfunc.c_str()));
            vdev.function_registry_.emplace(rfunc, func);
            std::get<1>(t) = true;
            SPDLOG_DEBUG("Function loaded: {}", rfunc);
        }
    }

    for (auto &apiCall : cuda_trace.callStack()) {
        SPDLOG_DEBUG("Executing: {}", apiCall->typeName());
        uint64_t err = apiCall->executeNative(vdev);
        if (err != 0) {
            SPDLOG_ERROR("Failed to execute call trace: {} ({})",
                          apiCall->nativeErrorToString(err), err);
            std::exit(EXIT_FAILURE);
        }
    }

    cuda_trace.markSynchronized();
    return true;
}

bool TraceExecutorLocal::deallocate() { return false; }

} // namespace gpuless
