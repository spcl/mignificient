#ifndef GPULESS_TRACE_EXECUTOR_SHMEM_H
#define GPULESS_TRACE_EXECUTOR_SHMEM_H

#include <cstdint>
#include <queue>

#include "iceoryx_posh/popo/wait_set.hpp"
#include "trace_executor.hpp"

#include <iceoryx_posh/internal/runtime/posh_runtime_impl.hpp>
#include <iceoryx_posh/popo/untyped_client.hpp>

namespace gpuless {


struct MemChunk {

    static constexpr int CHUNK_SIZE = 128 * 1024 * 1024;
    void* ptr;
    std::string name;

    void allocate()
    {
        int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        std::cerr << fd << " " << errno << std::endl;
        int ret  = ftruncate(fd, CHUNK_SIZE);
        std::cerr << fd << " " << ret << " " << errno << std::endl;

        ptr = mmap(NULL, CHUNK_SIZE, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, 0);
        std::cerr << "allocate " << name << " " << CHUNK_SIZE << " " << reinterpret_cast<std::uintptr_t>(ptr) << std::endl;
    }

    void open()
    {
        std::cerr << "open " << name << std::endl;
        int fd = shm_open(name.c_str(), O_RDWR, 0);
        ptr = mmap(NULL, CHUNK_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    }

    void close()
    {
        std::cerr << "Close " << name << std::endl;
        munmap(ptr, CHUNK_SIZE);
        shm_unlink(name.c_str());
    }
};

class MemPoolRead {
    //std::queue<MemChunk> used_chunks;
    std::unordered_map<std::string, void*> used_chunks;
public:

    void* get(const std::string& name)
    {
      auto it = used_chunks.find(name);
      if(it == used_chunks.end()) {

        int fd = shm_open(name.c_str(), O_RDWR, 0);
        auto ptr = mmap(NULL, MemChunk::CHUNK_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        std::cerr << "open " << name << " " << fd << " " << ptr << std::endl;
        used_chunks[name] = ptr;
        return ptr;
      } else {
        std::cerr << "return " << name << " " << (*it).second << std::endl;
        return (*it).second;
      }
    }
};

class MemPool {

    std::queue<MemChunk> chunks;

    //std::queue<MemChunk> used_chunks;
    std::unordered_map<std::string, void*> used_chunks;

    int counter = 0;

public:

    void give(const std::string& name)
    {
      std::cerr << "Return " << name << std::endl;
      chunks.push(MemChunk{used_chunks[name], name});
    }

    MemChunk get()
    {

      if(chunks.empty()) {

        /// FIXME: name
        std::string name = fmt::format("/gpuless_{}", counter++);
        MemChunk chunk{nullptr, name};
        chunk.allocate();
        chunks.push(chunk);

      }

      MemChunk ret = chunks.front(); 
      chunks.pop();
      used_chunks[ret.name] = ret.ptr;

      return ret;
    }

    ~MemPool()
    {
      while(!chunks.empty()) {

        MemChunk ret = chunks.front(); 
        chunks.pop();
        ret.close();

      }
    }
};

class TraceExecutorShmem : public TraceExecutor {
  private:
    sockaddr_in manager_addr{};
    sockaddr_in exec_addr{};
    int32_t session_id_ = -1;
    uint64_t synchronize_counter_ = 0;
    double synchronize_total_time_ = 0;

    // Not great - internal feature - but we don't have a better solution.
    std::unique_ptr<iox::runtime::PoshRuntimeImpl> _impl;
    std::unique_ptr<iox::popo::UntypedClient> client;
    std::optional<iox::popo::WaitSet<>> waitset;

    bool wait_poll;

  private:
    bool negotiateSession(manager::instance_profile profile);
    bool getDeviceAttributes();

  public:

    MemPool _pool;
    TraceExecutorShmem();
    ~TraceExecutorShmem();

    //static void init_runtime();
    //static void reset_runtime();

    bool init(const char *ip, short port,
              manager::instance_profile profile) override;
    bool synchronize(gpuless::CudaTrace &cuda_trace) override;
    bool deallocate() override;

    double getSynchronizeTotalTime() const override;

    static iox::runtime::PoshRuntime* runtime_factory_impl(iox::cxx::optional<const iox::RuntimeName_t*> var, TraceExecutorShmem* ptr = nullptr);
};

} // namespace gpuless

#endif // GPULESS_TRACE_EXECUTOR_TCP_H
