#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <sstream>
#include <iostream>
#include <vector>

#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <cuda.h>

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
void __checkCudaErrors(CUresult r, const char *file, const int line);

std::vector<uint8_t> recv_buffer(int socket_fd);
void recv_buffer(int socket_fd, std::vector<uint8_t> & buf, int msg_len);
void send_buffer(int socket_fd, const uint8_t *buf, size_t len);

void string_split(std::string const &str, const char delim,
                  std::vector<std::string> &out);
std::string string_rstrip(const std::string &str);
std::string cpp_demangle(const std::string &str);

#endif // __UTILS_HPP__
