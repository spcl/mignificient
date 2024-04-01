#ifndef __MANAGER_DEVICE_HPP__
#define __MANAGER_DEVICE_HPP__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <iceoryx_posh/popo/untyped_server.hpp>

void manage_device(const std::string& device, uint16_t port);
void swap_in();
void swap_out()

struct ShmemServer {

  std::unique_ptr<iox::popo::UntypedServer> server;

  void setup(const std::string app_name);
  void loop();
  void loop_wait();
  void finish();

  void* take();
  void release(void*);

  void _process_client(const void* payload);
  double _sum = 0;
};

#endif // __MANAGER_DEVICE_HPP__
