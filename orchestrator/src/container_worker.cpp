#include <mignificient/orchestrator/container_worker.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include <spdlog/spdlog.h>

namespace mignificient { namespace orchestrator {

  static std::string _shell_quote(const std::string& arg)
  {
    std::string out = "'";
    for(char c : arg) {
      out += (c == '\'') ? std::string{"'\\''"} : std::string(1, c);
    }
    return out + "'";
  }

  // Runs `command` with stderr merged into stdout; returns the exit status and the output.
  static std::pair<int, std::string> _run(const std::string& command)
  {
    std::string output;
    FILE* pipe = popen((command + " 2>&1").c_str(), "r");
    if(!pipe) {
      return {-1, "popen failed"};
    }
    std::array<char, 256> buffer;
    while(fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
      output += buffer.data();
    }
    int status = pclose(pipe);
    while(!output.empty() && (output.back() == '\n' || output.back() == '\r' || output.back() == ' ')) {
      output.pop_back();
    }
    return {status, output};
  }

  ContainerWorker::ContainerWorker(std::string runtime, int watch_ms):
    _runtime(std::move(runtime)),
    _watch(watch_ms),
    _thread([this] { _loop(); })
  {}

  ContainerWorker::~ContainerWorker()
  {
    {
      std::lock_guard<std::mutex> lock{_mutex};
      _quit = true;
    }
    _cv.notify_one();
    _thread.join();
  }

  void ContainerWorker::start(std::string client_id, std::vector<std::string> docker_argv)
  {
    {
      std::lock_guard<std::mutex> lock{_mutex};
      _jobs.push_back({true, std::move(client_id), std::move(docker_argv)});
    }
    _cv.notify_one();
  }

  void ContainerWorker::stop(std::string container_id)
  {
    {
      std::lock_guard<std::mutex> lock{_mutex};
      _jobs.push_back({false, std::move(container_id), {}});
    }
    _cv.notify_one();
  }

  void ContainerWorker::cancel(const std::string& client_id)
  {
    std::lock_guard<std::mutex> lock{_mutex};
    auto queued = std::find_if(_jobs.begin(), _jobs.end(), [&](const Job& j) { return j.start && j.id == client_id; });
    if(queued != _jobs.end()) {
      spdlog::info("Container start for {} cancelled before it ran", client_id);
      _jobs.erase(queued);
    } else if(_running_start == client_id) {
      _cancelled.insert(client_id);
    }
    // Otherwise it has finished. Drained: the client has its container id and doesn't cancel.
    // Not drained yet: kill the container here.
    for(auto it = _results.begin(); it != _results.end(); ) {
      if(it->client_id != client_id) {
        ++it;
        continue;
      }
      if(it->container_id) {
        _jobs.push_back({false, *it->container_id, {}});
        _cv.notify_one();
      }
      it = _results.erase(it);
    }
  }

  std::vector<ContainerStartResult> ContainerWorker::drain()
  {
    std::lock_guard<std::mutex> lock{_mutex};
    return std::exchange(_results, {});
  }

  void ContainerWorker::_loop()
  {
    std::unique_lock<std::mutex> lock{_mutex};
    while(true) {

      if(_jobs.empty() && !_quit) {
        if(_watched.empty()) {
          _cv.wait(lock, [this] { return !_jobs.empty() || _quit; });
        } else {
          _cv.wait_for(lock, std::chrono::milliseconds(500), [this] { return !_jobs.empty() || _quit; });
        }
      }

      if(!_jobs.empty()) {
        Job job = std::move(_jobs.front());
        _jobs.pop_front();
        if(job.start) {
          // Don't start containers nobody will stop.
          if(_quit) {
            continue;
          }
          _running_start = job.id;
        }
        lock.unlock();
        job.start ? _run_start(job) : _run_stop(job.id);
        lock.lock();
        continue;
      }

      if(_quit) {
        return;
      }

      lock.unlock();
      _poll_watched();
      lock.lock();
    }
  }

  void ContainerWorker::_run_start(const Job& job)
  {
    std::string command;
    for(const auto& arg : job.argv) {
      command += (command.empty() ? "" : " ") + _shell_quote(arg);
    }
    spdlog::info("Starting container executor: {}", command);

    auto begin = std::chrono::steady_clock::now();
    auto [status, output] = _run(command);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();

    // `run -d` prints the id last; anything before it is a warning or pull progress.
    std::string id = output.substr(output.find_last_of('\n') + 1);
    ContainerStartResult result{job.id, std::nullopt, ""};
    if(status != 0 || id.empty()) {
      result.error = output.empty() ? fmt::format("exit status {}", status) : output;
      spdlog::error("[StartupStats] container run for {} failed after {:.1f} ms (status {}): {}", job.id, ms, status, output);
    } else {
      result.container_id = id;
      spdlog::info("[StartupStats] container run for {} took {:.1f} ms, ID: {}", job.id, ms, id);
    }

    std::unique_lock<std::mutex> lock{_mutex};
    _running_start.clear();
    if(_cancelled.erase(job.id)) {
      lock.unlock();
      spdlog::info("Container start for {} was cancelled while running", job.id);
      if(result.container_id) {
        _run_stop(*result.container_id);
      }
      return;
    }
    if(result.container_id) {
      _watched.push_back({id, job.id, std::chrono::steady_clock::now() + _watch});
    }
    _results.push_back(std::move(result));
  }

  void ContainerWorker::_run_stop(const std::string& container_id)
  {
    _watched.erase(
      std::remove_if(_watched.begin(), _watched.end(), [&](const auto& w) { return w.container_id == container_id; }),
      _watched.end()
    );

    std::string command = fmt::format("{} kill {}", _shell_quote(_runtime), _shell_quote(container_id));
    spdlog::info("Stopping container: {}", command);
    auto [status, output] = _run(command);
    if(status != 0) {
      spdlog::warn("Failed to kill container {}: {}", container_id, output);
    }
  }

  void ContainerWorker::_poll_watched()
  {
    auto now = std::chrono::steady_clock::now();
    _watched.erase(
      std::remove_if(_watched.begin(), _watched.end(), [&](const auto& w) { return w.until < now; }),
      _watched.end()
    );
    if(_watched.empty()) {
      return;
    }

    // Running containers only; an exited one (with or without --rm) is missing.
    // one `ps` per 500 ms while any start is in its window; `docker events` if that gets costly.
    auto [status, output] = _run(fmt::format("{} ps -q --no-trunc", _shell_quote(_runtime)));
    if(status != 0) {
      spdlog::warn("Container poll failed: {}", output);
      return;
    }

    std::vector<ContainerStartResult> exited;
    for(auto it = _watched.begin(); it != _watched.end(); ) {
      if(output.find(it->container_id) == std::string::npos) {
        spdlog::error("Container {} of {} exited before the executor registered", it->container_id, it->client_id);
        exited.push_back({it->client_id, std::nullopt, fmt::format("container {} exited before the executor registered", it->container_id.substr(0, 12))});
        it = _watched.erase(it);
      } else {
        ++it;
      }
    }

    std::lock_guard<std::mutex> lock{_mutex};
    for(auto& r : exited) {
      _results.push_back(std::move(r));
    }
  }

}}
