#include <iostream>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cuda.h>
#include <spdlog/spdlog.h>

#include "../../schemas/trace_execution_protocol_generated.h"
#include "../../utils.hpp"
#include "../cuda_trace.hpp"
#include "../cuda_trace_converter.hpp"
#include "manager_device.hpp"

extern const int BACKLOG;

static bool g_device_initialized = false;
static int64_t g_sync_counter = 0;

static gpuless::CudaTrace &getCudaTrace() {
    static gpuless::CudaTrace cuda_trace;
    return cuda_trace;
}

static CudaVirtualDevice &getCudaVirtualDevice() {
    static CudaVirtualDevice cuda_virtual_device;
    if (!g_device_initialized) {
        g_device_initialized = true;
        cuda_virtual_device.initRealDevice();
    }
    return cuda_virtual_device;
}

void handle_attributes_request(int socket_fd,
                               const gpuless::FBProtocolMessage *msg) {
    SPDLOG_INFO("Handling device attributes request");

    auto &vdev = getCudaVirtualDevice();

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<CUdeviceAttributeValue>> attrs_vec;
    for (unsigned a = 0; a < vdev.device_attributes.size(); a++) {
        auto fb_attr = CreateCUdeviceAttributeValue(
            builder, static_cast<CUdeviceAttribute>(a),
            vdev.device_attributes[a]);
        attrs_vec.push_back(fb_attr);
    }

    auto attrs = gpuless::CreateFBTraceAttributeResponse(
        builder, gpuless::FBStatus_OK, vdev.device_total_mem,
        builder.CreateVector(attrs_vec));
    auto response = gpuless::CreateFBProtocolMessage(
        builder, gpuless::FBMessage_FBTraceAttributeResponse, attrs.Union());
    builder.Finish(response);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());

    SPDLOG_DEBUG("FBTraceAttributesResponse sent");
}

void handle_execute_request(int socket_fd,
                            const gpuless::FBProtocolMessage *msg) {
    SPDLOG_INFO("Handling trace execution request");
    auto &cuda_trace = getCudaTrace();
    auto &vdev = getCudaVirtualDevice();

    // load new modules
    auto new_modules = msg->message_as_FBTraceExecRequest()->new_modules();
    SPDLOG_INFO("Loading {} new modules", new_modules->size());
    for (const auto &m : *new_modules) {
        CUmodule mod;
        checkCudaErrors(cuModuleLoadData(&mod, m->buffer()->data()));
        vdev.module_registry_.emplace(m->module_id(), mod);
        SPDLOG_DEBUG("Loaded module {}", m->module_id());
    }

    // load new functions
    auto new_functions = msg->message_as_FBTraceExecRequest()->new_functions();
    SPDLOG_INFO("Loading {} new functions", new_functions->size());
    for (const auto &m : *new_functions) {
        auto mod_reg_it = vdev.module_registry_.find(m->module_id());
        if (mod_reg_it == vdev.module_registry_.end()) {
            SPDLOG_ERROR("Module {} not in registry", m->module_id());
        }
        CUmodule mod = mod_reg_it->second;
        CUfunction func;
        checkCudaErrors(cuModuleGetFunction(&func, mod, m->symbol()->c_str()));
        vdev.function_registry_.emplace(m->symbol()->str(), func);
        SPDLOG_DEBUG("Function loaded: {}", m->symbol()->str());
    }

    // execute CUDA api calls
    auto call_stack = gpuless::CudaTraceConverter::execRequestToTrace(
        msg->message_as_FBTraceExecRequest());
    cuda_trace.setCallStack(call_stack);
    SPDLOG_INFO("Execution trace of size {}", call_stack.size());

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
    g_sync_counter++;
    SPDLOG_INFO("Number of synchronizations: {}", g_sync_counter);

    flatbuffers::FlatBufferBuilder builder;
    auto top = cuda_trace.historyTop()->fbSerialize(builder);

    auto fb_trace_exec_response =
        gpuless::CreateFBTraceExecResponse(builder, gpuless::FBStatus_OK, top);
    auto fb_protocol_message = gpuless::CreateFBProtocolMessage(
        builder, gpuless::FBMessage_FBTraceExecResponse,
        fb_trace_exec_response.Union());
    builder.Finish(fb_protocol_message);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());
}

void handle_request(int socket_fd) {
    std::vector<uint8_t> buffer = recv_buffer(socket_fd);
    auto msg = gpuless::GetFBProtocolMessage(buffer.data());

    if (msg->message_type() == gpuless::FBMessage_FBTraceExecRequest) {
        handle_execute_request(socket_fd, msg);
    } else if (msg->message_type() ==
               gpuless::FBMessage_FBTraceAttributeRequest) {
        handle_attributes_request(socket_fd, msg);
    } else {
        SPDLOG_ERROR("Invalid request type");
        return;
    }
}

void manage_device(const std::string& device, uint16_t port) {
    setenv("CUDA_VISIBLE_DEVICES", device.c_str(), 1);

    // start server
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) {
        SPDLOG_ERROR("failed to open socket");
        exit(EXIT_FAILURE);
    }

    int opt = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (void *)&opt, sizeof(opt));

    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = INADDR_ANY;
    sa.sin_port = htons(port);

    if (bind(s, (sockaddr *)&sa, sizeof(sa)) < 0) {
        SPDLOG_ERROR("failed to bind socket");
        close(s);
        exit(EXIT_FAILURE);
    }

    if (listen(s, BACKLOG) < 0) {
        std::cerr << "failed to listen on socket" << std::endl;
        close(s);
        exit(EXIT_FAILURE);
    }

    // initialize cuda device pre-emptively
    getCudaVirtualDevice().initRealDevice();

    int s_new;
    sockaddr remote_addr{};
    socklen_t remote_addrlen = sizeof(remote_addr);
    while ((s_new = accept(s, &remote_addr, &remote_addrlen))) {
        SPDLOG_INFO("manager_device: connection from {}",
                     inet_ntoa(((sockaddr_in *)&remote_addr)->sin_addr));

        // synchronous request handler
        handle_request(s_new);
        close(s_new);
    }

    close(s);
    exit(EXIT_SUCCESS);
}
