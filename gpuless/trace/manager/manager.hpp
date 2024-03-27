#ifndef __MANAGER_HPP__
#define __MANAGER_HPP__

#include <atomic>
#include <mutex>
#include <sstream>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace gpuless {
namespace manager {

const uint16_t MANAGER_PORT = 8002;
const int NO_SESSION_ASSIGNED = -1;

enum instance_profile : int32_t {
    NO_MIG = 1000,
    MIG_1g5gb = 19,
    MIG_2g10gb = 14,
    MIG_3g20gb = 9,
    MIG_4g20gb = 5,
    MIG_7g40gb = 0,
};

typedef std::tuple<std::string, int, int> device_t;
typedef std::vector<device_t> devices_t;

struct TCPServer {
  int socket_fd;

  void setup(const devices_t & devices);
  void loop();
  void finish();
};

struct ShmemServer {


};

} // namespace manager
} // namespace gpuless

#endif // __MANAGER_HPP__
