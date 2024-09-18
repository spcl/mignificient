#include <string>
#include <queue>
#include <vector>
#include <memory>
#include <drogon/HttpResponse.h>

// Forward declarations
class GPUDevice;
class ActiveExecutor;

// 1. Class representing the active function invocation

enum class SharingModel {
    SEQUENTIAL,
    OVERLAP_CPU,
    OVERLAP_CPU_MEMCPY,
    FULL_OVERLAP
};

class GPUDevice {
public:

    GPUDevice(std::string name, SharingModel initialModel)
        : name_(std::move(name)), currentModel_(initialModel) {}

    SharingModel getModel() const { return currentModel_; }

    void addPendingInvocation(std::shared_ptr<ActiveInvocation> invocation) {
        pendingInvocations_.push(std::move(invocation));
    }

    void setCurrentInvocation(std::shared_ptr<ActiveInvocation> invocation) {
        currentInvocation_ = std::move(invocation);
    }

    void addExecutor(std::shared_ptr<ActiveExecutor> executor) {
        activeExecutors_.push_back(std::move(executor));
    }

    const std::string& getName() const { return name_; }

private:
    std::string name_;
    SharingModel currentModel_;
    std::queue<std::shared_ptr<ActiveInvocation>> pendingInvocations_;
    std::shared_ptr<ActiveInvocation> currentInvocation_;
    std::vector<std::shared_ptr<ActiveExecutor>> activeExecutors_;
};

