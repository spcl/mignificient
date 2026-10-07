#ifndef __MIGNIFICIENT_ORCHESTRATOR_CONTAINER_WORKER_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_CONTAINER_WORKER_HPP__

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace mignificient { namespace orchestrator {

  /**
   * A finished start, or a started container that exited again.
   * container_id set: running. Otherwise `error` says why there is no container.
   */
  struct ContainerStartResult {
    std::string client_id;
    std::optional<std::string> container_id;
    std::string error;
  };

  /**
   * Runs every container CLI call (`<runtime> run/kill/ps`) on one thread, so the
   * orchestrator loop never blocks on Docker. Jobs run in FIFO order.
   *
   * After a successful start the container is watched (`<runtime> ps`, every 500 ms)
   * for `watch_ms`; if it exits in that window, drain() reports it as a failure.
   */
  class ContainerWorker {
  public:
    ContainerWorker(std::string runtime, int watch_ms);
    ~ContainerWorker();

    // Returns immediately; the result comes out of drain().
    void start(std::string client_id, std::vector<std::string> docker_argv);
    // Fire-and-forget kill (--rm cleans up).
    void stop(std::string container_id);
    // Stop a client whose start has not finished: a queued start is dropped, a running one
    // is killed as soon as `run` returns. Its result is not reported.
    void cancel(const std::string& client_id);

    std::vector<ContainerStartResult> drain();

  private:
    struct Job {
      bool start;
      std::string id;  // client id (start) or container id (stop)
      std::vector<std::string> argv;
    };
    struct Watched {
      std::string container_id;
      std::string client_id;
      std::chrono::steady_clock::time_point until;
    };

    void _loop();
    void _run_start(const Job& job);
    void _run_stop(const std::string& container_id);
    void _poll_watched();

    std::string _runtime;
    std::chrono::milliseconds _watch;

    std::mutex _mutex;
    std::condition_variable _cv;
    std::deque<Job> _jobs;
    std::vector<ContainerStartResult> _results;
    std::vector<Watched> _watched;       // worker thread only
    std::string _running_start;          // client id of the start in progress
    std::set<std::string> _cancelled;    // cancelled while their start was running
    bool _quit = false;

    std::thread _thread;
  };

}}

#endif
