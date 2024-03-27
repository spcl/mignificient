#include <csignal>
#include <iostream>
#include <map>
#include <pthread.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>
#include <sys/wait.h>

#include "../../schemas/allocation_protocol_generated.h"
#include "../../utils.hpp"
#include "manager.hpp"
#include "manager_device.hpp"

using namespace gpuless::manager;

extern const int BACKLOG = 5;
const char *manager_ip = "127.0.0.1";

// nvidia (mig) devices: [device, profile id, session assignment]
static std::mutex lock_devices;
static std::vector<std::tuple<std::string, int, int>> devices = {

};

static std::atomic<int> next_session_id(1);

// device -> [pid, port]
static std::map<std::string, std::pair<pid_t, short>> child_processes;

pid_t fork_device_manager(std::string device, int port) {
    pid_t pid = fork();
    if (pid == 0) {
        manage_device(device, port);
    }
    return pid;
}

static void deallocate_session_devices(int32_t session_id) {
    lock_devices.lock();
    std::string device;
    for (auto &d : devices) {
        if (std::get<2>(d) == session_id) {
            device = std::get<0>(d);
            std::get<2>(d) = NO_SESSION_ASSIGNED;
            SPDLOG_INFO("Session {} deallocated", session_id);
            break;
        }
    }

    // restart the process to properly reset the CUDA device
    auto it = child_processes.find(device);
    if (it == child_processes.end()) {
        SPDLOG_ERROR("no process for device {} found", device);
    }

    SPDLOG_DEBUG("Killing pid={}", it->second.first);
    kill(it->second.first, SIGTERM);
    int wstatus;
    waitpid(it->second.first, &wstatus, 0);

    pid_t new_pid = fork_device_manager(device, it->second.second);
    std::get<0>(it->second) = new_pid;
    SPDLOG_INFO("Device manager restarted for device={}, pid={}, port={}",
                 device, new_pid, it->second.second);

    lock_devices.unlock();
}

static std::vector<int32_t> available_profiles() {
    std::vector<int32_t> r;
    lock_devices.lock();
    for (const auto &d : devices) {
        if (std::get<2>(d) == NO_SESSION_ASSIGNED) {
            r.push_back(std::get<1>(d));
        }
    }
    lock_devices.unlock();
    return r;
}

static std::string assign_device(int32_t profile, int32_t session_id) {
    lock_devices.lock();
    std::string assigned_device;
    for(auto& dev : devices) {
        if (std::get<2>(dev) == NO_SESSION_ASSIGNED &&
            profile == std::get<1>(dev)) {
            std::get<2>(dev) = session_id;
            assigned_device = std::get<0>(dev);
            break;
        }
    }

    lock_devices.unlock();
    return assigned_device;
}

void handle_allocate_request(int socket_fd, const ProtocolMessage *msg) {
    int session_id = next_session_id++;
    SPDLOG_INFO("allocation request for {} (session_id={})\n",
                 msg->message_as_AllocateRequest()->profile(), session_id);

    flatbuffers::FlatBufferBuilder builder;

    // offer a set of instances (all available instances)
    auto profiles = available_profiles();
    auto allocate_offer_msg = CreateProtocolMessage(
        builder, Message_AllocateOffer,
        CreateAllocateOffer(builder, Status_OK, session_id,
                            builder.CreateVector(profiles))
            .Union());
    builder.Finish(allocate_offer_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());

    // client selects a profile
    std::vector<uint8_t> buffer = recv_buffer(socket_fd);
    auto allocate_select_msg = GetProtocolMessage(buffer.data());
    int32_t selected_profile =
        allocate_select_msg->message_as_AllocateSelect()->profile();

    // send confirmation
    std::string assigned_device = assign_device(selected_profile, session_id);
    auto status = assigned_device.empty() ? Status_FAILURE : Status_OK;

    uint32_t selected_ip;
    inet_pton(AF_INET, manager_ip, &selected_ip);
    auto it = child_processes.find(assigned_device);
    if (it == child_processes.end()) {
        SPDLOG_ERROR("assigned_device not in child_processes");
        std::exit(EXIT_FAILURE);
    }
    short selected_port = it->second.second;

    builder.Reset();
    auto allocate_confirm_msg =
        CreateProtocolMessage(builder, Message_AllocateConfirm,
                              CreateAllocateConfirm(builder, status, session_id,
                                                    selected_ip, selected_port)
                                  .Union());
    builder.Finish(allocate_confirm_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());
}

void handle_deallocate_request(int socket_fd, const ProtocolMessage *msg) {
    int32_t session_id = msg->message_as_DeallocateRequest()->session_id();
    SPDLOG_INFO("deallocation request for session_id={}", session_id);

    flatbuffers::FlatBufferBuilder builder;
    auto deallocate_confirm_msg = CreateProtocolMessage(
        builder, Message_DeallocateConfirm,
        CreateDeallocateConfirm(builder, Status_OK, session_id).Union());
    builder.Finish(deallocate_confirm_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());

    deallocate_session_devices(session_id);
}

void *handle_request(void *arg) {
    int socket_fd = *((int *)arg);
    delete ((int *)arg);

    // generic initial request from client
    std::vector<uint8_t> buffer = recv_buffer(socket_fd);
    auto msg = GetProtocolMessage(buffer.data());
    if (msg->message_type() == Message_AllocateRequest) {
        handle_allocate_request(socket_fd, msg);
    } else if (msg->message_type() == Message_DeallocateRequest) {
        handle_deallocate_request(socket_fd, msg);
    } else {
        SPDLOG_ERROR("invalid request");
    }

    close(socket_fd);
    return nullptr;
}

static std::string exec(const char *cmd) {
    std::array<char, 128> buffer{};
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

void setup_devices() {
    std::string cmd = "nvidia-smi -L";
    std::string smi_string = exec(cmd.c_str());
    std::vector<std::string> split_devices;
    string_split(smi_string, '\n', split_devices);
    for(const auto& d : split_devices) {
        std::string::size_type GPU_idx = d.find("GPU-");
        std::string GPU_ID = d.substr(GPU_idx, d.size() - GPU_idx - 1);
        devices.emplace_back(GPU_ID, NO_MIG, NO_SESSION_ASSIGNED);
    }
}

void sigint_handler(int signum) {
    for (const auto &p : child_processes) {
        kill(p.second.first, SIGTERM);
        wait(nullptr);
    }
    exit(signum);
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    signal(SIGINT, sigint_handler);

    // load log level from env variable SPDLOG_LEVEL
    spdlog::cfg::load_env_levels();

    // set manager ip if given as environment variable
    char *manager_ip_env = std::getenv("MANAGER_IP");
    if (manager_ip_env) {
        manager_ip = manager_ip_env;
        SPDLOG_INFO("MANAGER_IP={}", manager_ip);
    }

    // setup devices
    setup_devices();

    // start device management processes
    short next_port = MANAGER_PORT + 1;
    for (const auto &t : devices) {
        std::string device = std::get<0>(t);

        pid_t pid = fork_device_manager(device, next_port);
        child_processes.emplace(device, std::make_pair(pid, next_port));
        SPDLOG_INFO("managing device: {} (profile={},pid={},port={})", device,
                    std::get<1>(t), pid, next_port);
        next_port++;
    }

    // run server
    int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        SPDLOG_ERROR("failed to open socket");
        exit(EXIT_FAILURE);
    }

    int opt = 1;
    setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, (void *)&opt, sizeof(opt));

    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = INADDR_ANY;
    sa.sin_port = htons(MANAGER_PORT);

    if (bind(socket_fd, (sockaddr *)&sa, sizeof(sa)) < 0) {
        SPDLOG_ERROR("failed to bind socket");
        close(socket_fd);
        exit(EXIT_FAILURE);
    }

    if (listen(socket_fd, BACKLOG) < 0) {
        SPDLOG_ERROR("failed to listen on socket");
        close(socket_fd);
        exit(EXIT_FAILURE);
    }

    SPDLOG_INFO("manager running on port {}", MANAGER_PORT);

    int s_new;
    sockaddr remote_addr{};
    socklen_t remote_addrlen = sizeof(remote_addr);
    while ((s_new = accept(socket_fd, &remote_addr, &remote_addrlen))) {
        SPDLOG_INFO("manager: connection from {}",
                     inet_ntoa(((sockaddr_in *)&remote_addr)->sin_addr));

        // handle connection in new thread
        int *s_new_alloc = new int;
        *s_new_alloc = s_new;
        pthread_t t;
        pthread_create(&t, nullptr, &handle_request, s_new_alloc);
    }

    close(socket_fd);
    return EXIT_SUCCESS;
}
