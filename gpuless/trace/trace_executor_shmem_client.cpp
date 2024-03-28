#include "trace_executor_shmem_client.hpp"
#include "../schemas/allocation_protocol_generated.h"
#include "cuda_trace_converter.hpp"

#include <spdlog/spdlog.h>

#include <iceoryx_posh/runtime/posh_runtime.hpp>

namespace gpuless {

iox::runtime::PoshRuntime* TraceExecutorShmem::runtime_factory_impl(iox::cxx::optional<const iox::RuntimeName_t*> var, TraceExecutorShmem* ptr)
{
    static TraceExecutorShmem* obj_ptr = nullptr;
    if(ptr) {
        obj_ptr = ptr;
        return nullptr;
    } else if (var.has_value()) {
        obj_ptr->_impl = std::make_unique<iox::runtime::PoshRuntimeImpl>(var);
        return obj_ptr->_impl.get();
    } else {
        return obj_ptr->_impl.get();
    }
}

iox::runtime::PoshRuntime& runtime_factory(iox::cxx::optional<const iox::RuntimeName_t*> var)
{
    return *TraceExecutorShmem::runtime_factory_impl(var, nullptr);
}

TraceExecutorShmem::TraceExecutorShmem()
{
    iox::runtime::PoshRuntime::setRuntimeFactory(runtime_factory);
    runtime_factory_impl(nullptr, this);

    // FIXME: Parameter
    constexpr char APP_NAME[] = "gpuless-app2";
    iox::runtime::PoshRuntime::initRuntime(APP_NAME);

    // FIXME: Parameter
    client.reset(new iox::popo::UntypedClient({"Example", "Request-Response", "Add"}));
}

TraceExecutorShmem::~TraceExecutorShmem() = default;

bool TraceExecutorShmem::negotiateSession(
    gpuless::manager::instance_profile profile) {
    int socket_fd;
    if ((socket_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        SPDLOG_ERROR("failed to open socket");
        return false;
    }

    if (connect(socket_fd, (sockaddr *)&this->manager_addr,
                sizeof(manager_addr)) < 0) {
        SPDLOG_ERROR("failed to connect");
        return false;
    }

    using namespace gpuless::manager;
    flatbuffers::FlatBufferBuilder builder;

    // make initial request
    auto allocate_request_msg = CreateProtocolMessage(
        builder, Message_AllocateRequest,
        CreateAllocateRequest(builder, profile, -1).Union());
    builder.Finish(allocate_request_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());

    // manager offers some sessions
    std::vector<uint8_t> buffer_offer = recv_buffer(socket_fd);
    auto allocate_offer_msg = GetProtocolMessage(buffer_offer.data());
    auto offered_profiles =
        allocate_offer_msg->message_as_AllocateOffer()->available_profiles();
    int32_t selected_profile = offered_profiles->Get(0);
    this->session_id_ =
        allocate_offer_msg->message_as_AllocateOffer()->session_id();

    // choose a profile and send finalize request
    builder.Reset();
    auto allocate_select_msg = CreateProtocolMessage(
        builder, Message_AllocateSelect,
        CreateAllocateSelect(builder, Status_OK, this->session_id_,
                             selected_profile)
            .Union());
    builder.Finish(allocate_select_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());

    // get server confirmation
    std::vector<uint8_t> buffer_confirm = recv_buffer(socket_fd);
    auto allocate_confirm_msg = GetProtocolMessage(buffer_confirm.data());
    bool ret = false;
    if (allocate_confirm_msg->message_as_AllocateConfirm()->status() ==
        Status_OK) {
        auto port = allocate_confirm_msg->message_as_AllocateConfirm()->port();
        auto ip = allocate_confirm_msg->message_as_AllocateConfirm()->ip();
        this->exec_addr.sin_family = AF_INET;
        this->exec_addr.sin_port = htons(port);
        this->exec_addr.sin_addr = *((struct in_addr *)&ip);
        ret = true;
    }

    close(socket_fd);

    this->getDeviceAttributes();
    return ret;
}

bool TraceExecutorShmem::init(const char *ip, const short port,
                            manager::instance_profile profile) {
    // store and check server address/port
    manager_addr.sin_family = AF_INET;
    manager_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &manager_addr.sin_addr) < 0) {
        SPDLOG_ERROR("Invalid IP address: {}", ip);
        return false;
    }

    bool r = this->negotiateSession(profile);
    if (r) {
        SPDLOG_INFO("Session with {}:{} negotiated", ip, port);
    } else {
        SPDLOG_ERROR("Failed to negotiate session with {}:{}", ip, port);
    }
    return r;
}

bool TraceExecutorShmem::deallocate() {
    int socket_fd;
    if ((socket_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        SPDLOG_ERROR("Failed to open socket");
        return false;
    }
    if (connect(socket_fd, (sockaddr *)&this->manager_addr,
                sizeof(manager_addr)) < 0) {
        SPDLOG_ERROR("Failed to connect");
        return false;
    }

    using namespace gpuless::manager;
    flatbuffers::FlatBufferBuilder builder;
    auto deallocate_request_msg = CreateProtocolMessage(
        builder, Message_DeallocateRequest,
        CreateDeallocateRequest(builder, this->session_id_).Union());
    builder.Finish(deallocate_request_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());

    SPDLOG_DEBUG("Deallocate request sent");

    std::vector<uint8_t> buffer = recv_buffer(socket_fd);
    auto deallocate_confirm_msg = GetProtocolMessage(buffer.data());
    auto status =
        deallocate_confirm_msg->message_as_DeallocateConfirm()->status();
    this->session_id_ = -1;
    return status == Status_OK;
}

bool TraceExecutorShmem::synchronize(CudaTrace &cuda_trace) {
    auto s = std::chrono::high_resolution_clock::now();

    this->synchronize_counter_++;
    SPDLOG_INFO(
        "TraceExecutorTcp::synchronize() [synchronize_counter={}, size={}]",
        this->synchronize_counter_, cuda_trace.callStack().size());

    // collect statistics on synchronizations

    // send trace execution request
    //auto sx = std::chrono::high_resolution_clock::now();
    flatbuffers::FlatBufferBuilder builder;
    CudaTraceConverter::traceToExecRequest(cuda_trace, builder);
    //auto ex = std::chrono::high_resolution_clock::now();
    //auto dx =
    //    std::chrono::duration_cast<std::chrono::microseconds>(ex - sx).count() /
    //    1000000.0;
    //std::cerr << "Request compress " << dx << std::endl;

    int64_t requestSequenceId = 0;
    int64_t expectedResponseSequenceId = requestSequenceId;
    auto s1 = std::chrono::high_resolution_clock::now();
    // FIXME: what should be the alignment here?
    client->loan(builder.GetSize(), 16)
        .and_then([&](auto& requestPayload) {

            auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
            requestHeader->setSequenceId(requestSequenceId);
            expectedResponseSequenceId = requestSequenceId;
            requestSequenceId += 1;

            memcpy(requestPayload, builder.GetBufferPointer(), builder.GetSize());

            client->send(requestPayload).or_else(
                [&](auto& error) { std::cout << "Could not send Request! Error: " << error << std::endl; });

        })
        .or_else([](auto& error) { std::cout << "Could not allocate Request! Error: " << error << std::endl; });
    SPDLOG_INFO("Trace execution request sent");

    //! [take response]
    std::shared_ptr<AbstractCudaApiCall> cuda_api_call = nullptr;

    // FIXME: waitset
    while(true) {

      auto val = client->take();

      if(val.has_error()) {

        if(val.get_error() == iox::popo::ChunkReceiveResult::NO_CHUNK_AVAILABLE) {
          continue;
        } else {
          abort();
        }

      } else {

        auto responsePayload = val.value();
        auto responseHeader = iox::popo::ResponseHeader::fromPayload(responsePayload);
        if (responseHeader->getSequenceId() == expectedResponseSequenceId)
        {

            SPDLOG_INFO("Trace execution response received");
            auto fb_protocol_message_response =
                GetFBProtocolMessage(responsePayload);
            auto fb_trace_exec_response =
                fb_protocol_message_response->message_as_FBTraceExecResponse();
            cuda_api_call =
                CudaTraceConverter::execResponseToTopApiCall(fb_trace_exec_response);
            auto e1 = std::chrono::high_resolution_clock::now();
            auto d1 =
                std::chrono::duration_cast<std::chrono::microseconds>(e1 - s1).count() /
                1000000.0;
            std::cerr << d1 << std::endl;
            //this->synchronize_total_time_2 += d1;
            client->releaseResponse(responsePayload);
        }
        else
        {
            std::cout << "Got Response with outdated sequence ID! Expected = " << expectedResponseSequenceId
                      << "; Actual = " << responseHeader->getSequenceId() << "! -> skip" << std::endl;
        }
        break;

      }

    }

    cuda_trace.markSynchronized();
    cuda_trace.setHistoryTop(cuda_api_call);

    auto e = std::chrono::high_resolution_clock::now();
    auto d =
        std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
        1000000.0;
    this->synchronize_total_time_ += d;

    SPDLOG_INFO(
        "TraceExecutorTcp::synchronize() successful [t={}s, total_time={}s]", d,
        this->synchronize_total_time_);
    return true;
}

bool TraceExecutorShmem::getDeviceAttributes() {
    SPDLOG_INFO("TraceExecutorTcp::getDeviceAttributes()");

    int socket_fd;
    if ((socket_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        SPDLOG_ERROR("failed to open socket");
        return false;
    }
    if (connect(socket_fd, (sockaddr *)&exec_addr, sizeof(exec_addr)) < 0) {
        SPDLOG_ERROR("failed to connect");
        return false;
    }

    flatbuffers::FlatBufferBuilder builder;
    auto attr_request =
        CreateFBProtocolMessage(builder, FBMessage_FBTraceAttributeRequest,
                                CreateFBTraceAttributeRequest(builder).Union());
    builder.Finish(attr_request);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());
    SPDLOG_DEBUG("FBTraceAttributeRequest sent");

    std::vector<uint8_t> response_buffer = recv_buffer(socket_fd);
    SPDLOG_DEBUG("FBTraceAttributeResponse received");

    auto fb_protocol_message_response =
        GetFBProtocolMessage(response_buffer.data());
    auto fb_trace_attribute_response =
        fb_protocol_message_response->message_as_FBTraceAttributeResponse();

    this->device_total_mem = fb_trace_attribute_response->total_mem();
    this->device_attributes.resize(CU_DEVICE_ATTRIBUTE_MAX);
    for (const auto &a : *fb_trace_attribute_response->device_attributes()) {
        int32_t value = a->value();
        auto dev_attr = static_cast<CUdevice_attribute>(a->device_attribute());
        this->device_attributes[dev_attr] = value;
    }

    close(socket_fd);
    return true;
}

double TraceExecutorShmem::getSynchronizeTotalTime() const {
    return synchronize_total_time_;
}

} // namespace gpuless
