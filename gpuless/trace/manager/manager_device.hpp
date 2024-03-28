#ifndef __MANAGER_DEVICE_HPP__
#define __MANAGER_DEVICE_HPP__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <iceoryx_posh/popo/untyped_server.hpp>

void manage_device(const std::string& device, uint16_t port);

struct ShmemServer {

  std::unique_ptr<iox::popo::UntypedServer> server;

  void setup(const std::string app_name);
  void loop();
  void finish();

  void* take();
  void release(void*);

};

#endif // __MANAGER_DEVICE_HPP__
