#include <iostream>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cuda.h>
#include <spdlog/spdlog.h>

#include <iceoryx_posh/runtime/posh_runtime.hpp>
#include <iceoryx_posh/popo/untyped_server.hpp>
#include <iceoryx_hoofs/posix_wrapper/signal_watcher.hpp>

#include "../../schemas/trace_execution_protocol_generated.h"
#include "../../utils.hpp"
#include "../cuda_trace.hpp"
#include "../cuda_trace_converter.hpp"
#include "iceoryx_posh/popo/wait_set.hpp"
#include "manager_device.hpp"

#include "../trace_executor_shmem_client.hpp"

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

flatbuffers::FlatBufferBuilder handle_attributes_request(const gpuless::FBProtocolMessage *msg, int socket_fd) {
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
    if(socket_fd >= 0) {
        send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());
    }

    SPDLOG_DEBUG("FBTraceAttributesResponse sent");

    return builder;
}

flatbuffers::FlatBufferBuilder handle_execute_request(const gpuless::FBProtocolMessage *msg, int socket_fd) {
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
    //auto s1 = std::chrono::high_resolution_clock::now();
    auto p = msg->message_as_FBTraceExecRequest();
    //auto e2 = std::chrono::high_resolution_clock::now();
    auto call_stack = gpuless::CudaTraceConverter::execRequestToTrace(p);
    //auto e3 = std::chrono::high_resolution_clock::now();
    cuda_trace.setCallStack(call_stack);
    SPDLOG_INFO("Execution trace of size {}", call_stack.size());

    auto s = std::chrono::high_resolution_clock::now();
    for (auto &apiCall : cuda_trace.callStack()) {
        SPDLOG_DEBUG("Executing: {}", apiCall->typeName());
        uint64_t err = apiCall->executeNative(vdev);
        if (err != 0) {
            SPDLOG_ERROR("Failed to execute call trace: {} ({})",
                          apiCall->nativeErrorToString(err), err);
            std::exit(EXIT_FAILURE);
        }
    }
    //auto e = std::chrono::high_resolution_clock::now();
    //auto d =
    //    std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
    //    1000000.0;
    //auto d1 =
    //    std::chrono::duration_cast<std::chrono::microseconds>(e2 - s1).count() /
    //    1000000.0;
    //auto d2 =
    //    std::chrono::duration_cast<std::chrono::microseconds>(e3 - s1).count() /
    //    1000000.0;
    //std::cerr << "replied " << d << " " << d1 << " " << d2 << std::endl;

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
    return builder;
}

void handle_request(int socket_fd) {
    std::vector<uint8_t> buffer = recv_buffer(socket_fd);
    auto msg = gpuless::GetFBProtocolMessage(buffer.data());

    if (msg->message_type() == gpuless::FBMessage_FBTraceExecRequest) {
        handle_execute_request(msg, socket_fd);
    } else if (msg->message_type() ==
               gpuless::FBMessage_FBTraceAttributeRequest) {
        handle_attributes_request(msg, socket_fd);
    } else {
        SPDLOG_ERROR("Invalid request type");
        return;
    }
}

void ShmemServer::setup(const std::string app_name)
{
    iox::runtime::PoshRuntime::initRuntime(iox::RuntimeName_t{iox::cxx::TruncateToCapacity_t{}, app_name});
}

void* ShmemServer::take()
{
    auto ptr = this->server->take();
    if(ptr.has_error()) {
      return nullptr;
    } else {
      return const_cast<void*>(ptr.value());
    }
}

void ShmemServer::release(void* ptr)
{
    this->server->releaseRequest(ptr);
}

void ShmemServer::_process_client(const void* requestPayload)
{
    //auto request = static_cast<const AddRequest*>(requestPayload);
    //std::cout << APP_NAME << " Got Request: " << request->augend << " + " << request->addend << std::endl;

    //auto s = std::chrono::high_resolution_clock::now();
    //handle_request(s_new);
    auto msg = gpuless::GetFBProtocolMessage(requestPayload);
    flatbuffers::FlatBufferBuilder builder;
    //auto e1 = std::chrono::high_resolution_clock::now();

    //std::cerr << "Request" << std::endl;

    if (msg->message_type() == gpuless::FBMessage_FBTraceExecRequest) {
        builder = handle_execute_request(msg, -1);
    } else if (msg->message_type() == gpuless::FBMessage_FBTraceAttributeRequest) {
        builder = handle_attributes_request(msg, -1);
    } else {
        SPDLOG_ERROR("Invalid request type");
        return;
    }
    //auto e = std::chrono::high_resolution_clock::now();
    //auto d =
    //    std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
    //    1000000.0;
    //auto d1 =
    //    std::chrono::duration_cast<std::chrono::microseconds>(e1 - s).count() /
    //    1000000.0;
    //_sum += d;
    //std::cerr << "replied " << d << " , " << d1 << " , total " << _sum << std::endl;

    ////! [send response]
    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    server->loan(requestHeader, sizeof(builder.GetSize()), alignof(1))
        .and_then([&](auto& responsePayload) {

            memcpy(responsePayload, builder.GetBufferPointer(), builder.GetSize());
            //auto response = static_cast<AddResponse*>(responsePayload);
            //response->sum = request->augend + request->addend;
            //std::cout << APP_NAME << " Send Response: " << response->sum << std::endl;
            server->send(responsePayload).or_else(
                [&](auto& error) { std::cout << "Could not send Response! Error: " << error << std::endl; });
        })
        .or_else(
            [&](auto& error) { std::cout << "Could not allocate Response! Error: " << error << std::endl; });
    //! [send response]

    server->releaseRequest(requestPayload);

}

void ShmemServer::loop_wait()
{
    server.reset(new iox::popo::UntypedServer({"Example", "Request-Response", "Add"}));

    iox::popo::WaitSet<> waitset;

    waitset.attachState(*server, iox::popo::ServerState::HAS_REQUEST).or_else([](auto) {
        std::cerr << "failed to attach server" << std::endl;
        std::exit(EXIT_FAILURE);
    });

    // TODO: subscriber for master

    while (!iox::posix::hasTerminationRequested())
    {
        auto notificationVector = waitset.wait();

        for (auto& notification : notificationVector)
        {
            if(notification->doesOriginateFrom(server.get())) {

                //! [take request]
                server->take().and_then([&](auto& requestPayload) {
                    _process_client(requestPayload);
                });

            }
        }

    }
}

void ShmemServer::loop()
{
    server.reset(new iox::popo::UntypedServer({"Example", "Request-Response", "Add"}));

    double sum = 0;
    while (!iox::posix::hasTerminationRequested())
    {
        //! [take request]
        server->take().and_then([&](auto& requestPayload) {
            _process_client(requestPayload);
        });
        //! [take request]

    }
}

void manage_device(const std::string& device, uint16_t port) {
    setenv("CUDA_VISIBLE_DEVICES", device.c_str(), 1);

    const char* manager_type = std::getenv("MANAGER_TYPE");
    const char* poll_type = std::getenv("POLL_TYPE");

    if(std::string_view{manager_type} == "tcp") {

      //// start server
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

    } else {

      ShmemServer shm_server;

      // FIXME: configurable
      shm_server.setup("gpuless-app");

      // initialize cuda device pre-emptively
      getCudaVirtualDevice().initRealDevice();

      if(std::string_view{poll_type} == "WAIT") {
        shm_server.loop_wait();
      } else {
        shm_server.loop();
      }
    }

    exit(EXIT_SUCCESS);
}
