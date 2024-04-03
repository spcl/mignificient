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
    size_t size;

    void allocate(size_t size = CHUNK_SIZE)
    {
        //int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
        std::cerr << fd << " " << errno << std::endl;
        int ret  = ftruncate(fd, CHUNK_SIZE);
        std::cerr << fd << " " << ret << " " << errno << std::endl;

        ptr = mmap(NULL, CHUNK_SIZE, PROT_READ | PROT_WRITE,
                    MAP_SHARED |  MAP_POPULATE, fd, 0);
                    //MAP_SHARED | MAP_LOCKED | MAP_POPULATE, fd, 0);
        std::cerr << "allocate " << name << " " << CHUNK_SIZE << " " << reinterpret_cast<std::uintptr_t>(ptr) << std::endl;

        this->size = size;
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
        //std::cerr << "open " << name << " " << fd << " " << ptr << std::endl;
        used_chunks[name] = ptr;
        return ptr;
      } else {
        //std::cerr << "return " << name << " " << (*it).second << std::endl;
        return (*it).second;
      }
    }
};

class MemPool {

    std::queue<MemChunk> chunks;
    std::vector<MemChunk> user_chunks;

    //std::queue<MemChunk> used_chunks;
    std::unordered_map<std::string, void*> used_chunks;

    std::unordered_map<const void*, MemChunk> borrowed_chunks;

    int counter = 0;

public:

    void give(const std::string& name)
    {
      //std::cerr << "Return " << name << std::endl;
      chunks.push(MemChunk{used_chunks[name], name});
      //std::cerr << "return " << name << " size " << chunks.size() << std::endl;
    }

    void give(void *ptr)
    {
      //std::cerr << "Return " << name << std::endl;
      auto it = borrowed_chunks.find(ptr);
      if(it == borrowed_chunks.end()) {
        //std::cerr << "error!" << std::endl;
        abort();
      }
      bool inserted = false;
      for(int i = 0; i < user_chunks.size(); ++i) {
        if(user_chunks[i].ptr == nullptr) {
          user_chunks[i] = (*it).second;
          inserted = true;
          break;
        }
      }
      if(!inserted)
        user_chunks.push_back((*it).second);
      //std::cerr << "return " << (*it).second.name << " size " << (*it).second.size << " chunks " << user_chunks.size() << std::endl;
    }

    std::optional<std::string> get_name(const void *ptr)
    {
      auto it = borrowed_chunks.find(ptr);
      if(it == borrowed_chunks.end()) {
        return std::optional<std::string>{};
      }
      return (*it).second.name;
    }

    MemChunk get(size_t size)
    {
      //if(chunks.empty()) {

      //  /// FIXME: name
      //  std::string name = fmt::format("/gpuless_user_{}", counter++);
      //  MemChunk chunk{nullptr, name};
      //  chunk.allocate();
      //  user_chunks.push(chunk);

      //}

      // FIXME: turn into list
      //std::cerr << "borrowin: chunks? " << user_chunks.size() << " requested size " << size << std::endl;
      size_t q_size = user_chunks.size();
      int pos = -1;
      int min_size = INT_MAX;
      for(int i = 0; i < q_size; ++i) {

        //std::cerr << "pos " << i << " ptr " << user_chunks[i].ptr << " name " << user_chunks[i].name << " size? " << user_chunks[i].size << std::endl;
        if(user_chunks[i].ptr != nullptr && user_chunks[i].size >= size) {
          if(user_chunks[i].size < min_size) {
            pos = i;
            min_size = user_chunks[i].size;
          }
        }
        //MemChunk ret = user_chunks.front(); 
        //user_chunks.pop();
        //if(ret.size > size) {
        //  //used_chunks[ret.name] = ret.ptr;
        //  borrowed_chunks[ret.ptr] = ret;
        //  std::cerr << "borrow chunk " << ret.name << std::endl;
        //  return ret;
        //}
        //  std::cerr << "cannot borrow chunk " << ret.size << " " << size << std::endl;
        //user_chunks.push(ret);
      }
      if(pos != -1) {
        //std::cerr << "borrow chunk " << user_chunks[pos].name << std::endl;
        MemChunk chunk = user_chunks[pos];
        user_chunks[pos].ptr = nullptr;
        return chunk;
      } else {

        std::string name = fmt::format("/gpuless_user_{}", counter++);
        MemChunk chunk{nullptr, name, 0};
        chunk.allocate(size);
        //used_chunks[chunk.name] = chunk.ptr;
        borrowed_chunks[chunk.ptr] = chunk;
        //std::cerr << "borrow new chunk " << chunk.name << " size " << chunk.size << std::endl;

        return chunk;
      }
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
      for(MemChunk ret : user_chunks)
        ret.close();
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
