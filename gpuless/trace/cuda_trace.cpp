#include "cuda_trace.hpp"

#include <utility>

namespace gpuless {

CudaTrace::CudaTrace() = default;

void CudaTrace::record(
    const std::shared_ptr<AbstractCudaApiCall> &cudaApiCall) {
    this->call_stack_.push_back(cudaApiCall);
}

void CudaTrace::markSynchronized() {
    // move current trace to history
    std::move(std::begin(this->call_stack_), std::end(this->call_stack_),
              std::back_inserter(this->synchronized_history_));

    SPDLOG_INFO("Cuda trace history size: {}",
                this->synchronized_history_.size());

    // clear the current trace
    this->call_stack_.clear();
}

const std::shared_ptr<AbstractCudaApiCall> &CudaTrace::historyTop() {
    return this->synchronized_history_.back();
}

std::vector<std::shared_ptr<AbstractCudaApiCall>>& CudaTrace::callStack() {
    return this->call_stack_;
}

void CudaTrace::recordFatbinData(void *data, uint64_t size,
                                 uint64_t module_id) {
    this->module_id_to_fatbin_resource_.emplace(
        module_id, std::make_tuple(data, size, false));
}

void CudaTrace::recordSymbolMapEntry(std::string &symbol, uint64_t module_id) {
    this->symbol_to_module_id_.emplace(symbol,
                                       std::make_pair(module_id, false));
}

void CudaTrace::recordGlobalVarMapEntry(std::string &symbol,
                                        uint64_t module_id) {}

std::map<std::string, std::pair<uint64_t, bool>> &
CudaTrace::getSymbolToModuleId() {
    return symbol_to_module_id_;
}

std::map<uint64_t, std::tuple<void *, uint64_t, bool>> &
CudaTrace::getModuleIdToFatbinResource() {
    return module_id_to_fatbin_resource_;
}

void CudaTrace::setHistoryTop(std::shared_ptr<AbstractCudaApiCall> top) {
    this->synchronized_history_.back() = std::move(top);
}

void CudaTrace::setCallStack(
    const std::vector<std::shared_ptr<AbstractCudaApiCall>> &callStack) {
    call_stack_ = callStack;
}

} // namespace gpuless
