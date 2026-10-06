#ifndef __MIGNIFICIENT_ORCHESTRATOR_PACKAGE_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_PACKAGE_HPP__

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace mignificient { namespace orchestrator {

  // True if `path` (absolute) resolves, symlinks included, to `root` or below it.
  //
  // The goal is to verify that following a symlink does not escape the root directory.
  // This is a simple security check when dealing with user-supplied paths.
  inline bool path_under(const std::string& root, const std::string& path)
  {
    namespace fs = std::filesystem;
    std::error_code ec;
    if(!fs::path{path}.is_absolute()) {
      return false;
    }
    auto r = fs::weakly_canonical(root, ec);
    if(ec) return false;
    auto p = fs::weakly_canonical(path, ec);
    if(ec) return false;
    // weakly_canonical keeps a trailing separator ("/a/" -> empty last element); drop it.
    if(!r.has_filename()) r = r.parent_path();
    return std::mismatch(r.begin(), r.end(), p.begin(), p.end()).first == r.end();
  }

  // Returns the canonical package path if `path` is absolute, an existing directory
  // with a package.json, and lies under one of `roots`; nullopt otherwise.
  //
  // Package roots are defined during orhcestrator initialization and are used to orhcestrator
  // the set of packages that can be loaded.
  //
  // This is both security check - do not mount arbitrary directories, but also consistency check:
  // if we mount package under container as code, and the symlink goes outside, it will error at runtime.
  inline std::optional<std::string> resolve_package(const std::vector<std::string>& roots, const std::string& path)
  {
    namespace fs = std::filesystem;
    std::error_code ec;
    if(!fs::path{path}.is_absolute()) {
      return std::nullopt;
    }
    auto pkg = fs::canonical(path, ec);
    if(ec || !fs::is_directory(pkg, ec) || !fs::is_regular_file(pkg / "package.json", ec)) {
      return std::nullopt;
    }
    for(const auto& root : roots) {
      if(path_under(root, pkg.string())) {
        return pkg.string();
      }
    }
    return std::nullopt;
  }

}}

#endif
