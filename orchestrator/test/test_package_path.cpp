// Assert-based check of resolve_package / path_under. Run: build/orchestrator/test_package_path
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <mignificient/orchestrator/package.hpp>

using mignificient::orchestrator::path_under;
using mignificient::orchestrator::resolve_package;
namespace fs = std::filesystem;

int main()
{
  std::string tmpl = (fs::temp_directory_path() / "test_package_path_XXXXXX").string();
  fs::path base = mkdtemp(tmpl.data());
  fs::path root = base / "root";
  fs::create_directories(root / "resnet");
  std::ofstream{root / "resnet" / "package.json"} << "{}";
  fs::create_directories(root / "nojson");
  fs::create_directories(base / "outside");
  std::ofstream{base / "outside" / "package.json"} << "{}";
  fs::create_directory_symlink(base / "outside", root / "escape");
  fs::create_directory_symlink(root / "resnet", root / "alias");
  fs::create_directories(base / "rootx" / "pkg");
  std::ofstream{base / "rootx" / "pkg" / "package.json"} << "{}";

  std::vector<std::string> roots{root.string()};

  assert(resolve_package(roots, (root / "resnet").string()) == (root / "resnet").string());
  assert(resolve_package(roots, (root / "resnet/").string()) == (root / "resnet").string());
  assert(resolve_package(roots, (root / "alias").string()) == (root / "resnet").string());   // symlink inside root: ok
  assert(!resolve_package(roots, "root/resnet"));                                             // relative
  assert(!resolve_package(roots, (root / ".." / "outside").string()));                        // <root>/../x
  assert(!resolve_package(roots, "/etc"));
  assert(!resolve_package(roots, (root / "escape").string()));                                // symlink escaping root
  assert(!resolve_package(roots, (root / "nojson").string()));                                // no package.json
  assert(!resolve_package(roots, (root / "missing").string()));                               // missing
  assert(!resolve_package(roots, (base / "rootx" / "pkg").string()));                         // string prefix, not a subdir
  assert(!resolve_package({}, (root / "resnet").string()));                                   // no roots

  assert(path_under((root / "resnet").string(), (root / "resnet" / "handler.py").string()));
  assert(!path_under((root / "resnet").string(), (root / "resnet" / ".." / "x.py").string()));
  assert(!path_under((root / "resnet").string(), (root / "escape" / "x.py").string()));
  assert(!path_under((root / "resnet").string(), "handler.py"));

  fs::remove_all(base);
  std::puts("test_package_path: OK");
  return 0;
}
