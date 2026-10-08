#ifndef __MIGNIFICIENT_ORCHESTRATOR_CGROUP_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_CGROUP_HPP__

#include <filesystem>
#include <fstream>
#include <string>

#include <sys/types.h>
#include <unistd.h>

#include <spdlog/spdlog.h>

namespace mignificient { namespace orchestrator {

  /**
   * cgroup v2 with a CPU quota for the bare-metal processes of one client, created under executor.cgroup-root
   * (tools/setup-cgroup.sh). Moving a process in needs write access to the common ancestor of its current cgroup
   * and the target: the orchestrator must run inside the delegated tree (the setup script arranges it).
   */
  class CpuCgroup {
  public:

    // Lets the root's children use the cpu controller.
    static bool prepare_root(const std::string& root)
    {
      return _write(root + "/cgroup.subtree_control", "+cpu");
    }

    bool create(const std::string& root, const std::string& name, float cores)
    {
      std::error_code ec;
      std::string path = root + "/" + name;
      std::filesystem::create_directory(path, ec);
      if(ec) {
        spdlog::error("Failed to create cgroup {}: {}", path, ec.message());
        return false;
      }
      _path = path;
      return _write(_path + "/cpu.max", fmt::format("{} {}", static_cast<long>(cores * PERIOD_US), PERIOD_US));
    }

    bool add(pid_t pid)
    {
      return _write(_path + "/cgroup.procs", std::to_string(pid));
    }

    // Once its processes have exited.
    void remove()
    {
      if(!_path.empty() && rmdir(_path.c_str()) != 0) {
        spdlog::error("Failed to remove cgroup {}: {}", _path, strerror(errno));
      }
      _path.clear();
    }

  private:
    static constexpr long PERIOD_US = 100000;

    static bool _write(const std::string& file, const std::string& value)
    {
      std::ofstream f{file};
      // cgroupfs reports a rejected value (EINVAL, EACCES) on the write itself.
      f << value << std::flush;
      if(!f) {
        spdlog::error("Failed to write '{}' to {}", value, file);
        return false;
      }
      return true;
    }

    std::string _path;
  };

}}

#endif
