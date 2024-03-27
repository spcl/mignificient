#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <array>
#include <spdlog/spdlog.h>

#include "cubin_analysis.hpp"

std::map<std::string, PtxParameterType> &getStrToPtxParameterType() {
    static std::map<std::string, PtxParameterType> map_ = {
        {"s8", s8},     {"s16", s16},     {"s32", s32}, {"s64", s64},
        {"u8", u8},     {"u16", u16},     {"u32", u32}, {"u64", u64},
        {"f16", f16},   {"f16x2", f16x2}, {"f32", f32}, {"f64", f64},
        {"b8", b8},     {"b16", b16},     {"b32", b32}, {"b64", b64},
        {"pred", pred},
    };
    return map_;
}

std::map<PtxParameterType, std::string> &getPtxParameterTypeToStr() {
    static std::map<PtxParameterType, std::string> map_ = {
        {s8, "s8"},     {s16, "s16"},     {s32, "s32"}, {s64, "s64"},
        {u8, "u8"},     {u16, "u16"},     {u32, "u32"}, {u64, "u64"},
        {f16, "f16"},   {f16x2, "f16x2"}, {f32, "f32"}, {f64, "f64"},
        {b8, "b8"},     {b16, "b16"},     {b32, "b32"}, {b64, "b64"},
        {pred, "pred"},
    };
    return map_;
}

std::map<PtxParameterType, int> &getPtxParameterTypeToSize() {
    static std::map<PtxParameterType, int> map_ = {
        {s8, 1},  {s16, 2}, {s32, 4}, {s64, 8},   {u8, 1},   {u16, 2},
        {u32, 4}, {u64, 8}, {f16, 2}, {f16x2, 4}, {f32, 4},  {f64, 8},
        {b8, 1},  {b16, 2}, {b32, 4}, {b64, 8},   {pred, 0},
    };
    return map_;
}

static std::string exec(const char *cmd) {
    std::array<char, 128> buffer{};
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

bool CubinAnalyzer::isInitialized() { return this->initialized_; }

PtxParameterType
CubinAnalyzer::ptxParameterTypeFromString(const std::string &str) {
    auto it = getStrToPtxParameterType().find(str);
    if (it == getStrToPtxParameterType().end()) {
        return PtxParameterType::invalid;
    }
    return it->second;
}

int CubinAnalyzer::byteSizePtxParameterType(PtxParameterType type) {
    auto it = getPtxParameterTypeToSize().find(type);
    if (it == getPtxParameterTypeToSize().end()) {
        return -1;
    }
    return it->second;
}

std::vector<KParamInfo>
CubinAnalyzer::parsePtxParameters(const std::string &params) {
    std::vector<KParamInfo> ps;

    static std::regex r_param(
        "\\.param\\s*(?:\\.align\\s*([0-9]*)\\s*)?\\.([a-zA-Z0-9]*)\\s*([a-zA-"
        "Z0-9_]*)(?:\\[([0-9]*)\\])?",
        std::regex::ECMAScript);

    std::sregex_iterator i =
        std::sregex_iterator(params.begin(), params.end(), r_param);
    for (; i != std::sregex_iterator(); ++i) {
        std::smatch m = *i;
        const std::string &align = m[1];
        const std::string &type = m[2];
        const std::string &name = m[3];
        const std::string &size = m[4];

        int ialign = 0;
        if (!align.empty()) {
            ialign = std::stoi(align);
        }

        int isize = 1;
        if (!size.empty()) {
            isize = std::stoi(size);
        }

        PtxParameterType ptxParameterType = ptxParameterTypeFromString(type);
        int typeSize = byteSizePtxParameterType(ptxParameterType);
        ps.push_back(KParamInfo{
            name,
            ptxParameterType,
            typeSize,
            ialign,
            isize,
        });
    }

    return ps;
}

size_t CubinAnalyzer::pathToHash(const std::filesystem::path &path) {
    auto base = path.filename();
    return std::hash<std::string>{}(base.string());
}

bool CubinAnalyzer::isCached(const std::filesystem::path &fname) {
    std::size_t fname_hash = this->pathToHash(fname);
    std::filesystem::path base_dir(std::getenv("HOME"));
    if (const char* scratch = std::getenv("SCRATCH")) {
        base_dir = scratch;
    }
    std::filesystem::path cache_dir = base_dir / ".cache" / "libgpuless";
    if (!std::filesystem::is_directory(cache_dir)) {
        std::filesystem::create_directories(cache_dir);
    }
    std::filesystem::path cache_file = cache_dir / std::to_string(fname_hash);
    if (std::filesystem::is_regular_file(cache_file)) {
        return true;
    }
    return false;
}

bool CubinAnalyzer::loadAnalysisFromCache(const std::filesystem::path &fname) {
    std::size_t fname_hash = this->pathToHash(fname);
    std::filesystem::path base_dir(std::getenv("HOME"));
    if (const char* scratch = std::getenv("SCRATCH")) {
        base_dir = scratch;
    }
    std::filesystem::path cache_dir = base_dir / ".cache" / "libgpuless";
    std::filesystem::path cache_file = cache_dir / std::to_string(fname_hash);
    if (!std::filesystem::is_regular_file(cache_file)) {
        return false;
    }

    std::map<std::string, std::vector<KParamInfo>> tmp_map;
    std::ifstream in(cache_file);

    while (true) {
        std::string symbol;
        int n_params;
        in >> symbol;
        in >> n_params;

        std::vector<KParamInfo> kparam_infos;
        for (int i = 0; i < n_params; i++) {
            KParamInfo kparam_info;
            uint64_t u64_type;
            in >> kparam_info.paramName;
            in >> u64_type;
            kparam_info.type = PtxParameterType(u64_type);
            in >> kparam_info.typeSize;
            in >> kparam_info.align;
            in >> kparam_info.size;
            kparam_infos.push_back(kparam_info);
        }

        tmp_map.emplace(symbol, kparam_infos);
        if (in.eof()) {
            break;
        }
    }
    in.close();
    this->kernel_to_kparaminfos.insert(tmp_map.begin(), tmp_map.end());
    return true;
}

void CubinAnalyzer::storeAnalysisToCache(
    const std::filesystem::path &fname,
    const std::map<std::string, std::vector<KParamInfo>> &data) {
    SPDLOG_INFO("Storing analysis to cache: {}", fname.string());
    std::size_t fname_hash = this->pathToHash(fname);
    std::filesystem::path base_dir(std::getenv("HOME"));
    if (const char* scratch = std::getenv("SCRATCH")) {
        base_dir = scratch;
    }
    std::filesystem::path cache_dir = base_dir / ".cache" / "libgpuless";
    if (!std::filesystem::is_directory(cache_dir)) {
        std::filesystem::create_directories(cache_dir);
    }
    std::filesystem::path cache_file = cache_dir / std::to_string(fname_hash);

    std::fstream out(cache_file, std::fstream::app);
    for (const auto &d : data) {
        const std::string &symbol = d.first;
        const std::vector<KParamInfo> &kparam_infos = d.second;
        out << symbol << std::endl << kparam_infos.size() << std::endl;
        for (const auto &p : kparam_infos) {
            out << p.paramName << std::endl;
            out << p.type << std::endl;
            out << p.typeSize << std::endl;
            out << p.align << std::endl;
            out << p.size << std::endl;
        }
    }
    out.close();
}

bool CubinAnalyzer::analyzePtx(const std::filesystem::path &fname,
                               int major_version, int minor_version) {
    auto tmp = std::filesystem::temp_directory_path() / "libgpuless";
    SPDLOG_INFO("Using tmp directory: {}", tmp.string());
    if (std::filesystem::is_directory(tmp)) {
        std::filesystem::remove_all(tmp);
    }

    std::filesystem::path bin(fname);
    std::filesystem::create_directory(tmp);
    std::filesystem::copy_file(bin, tmp / bin.filename());
    auto tmp_bin = tmp / bin.filename();

    auto tmp_ptx = tmp / "ptx";
    std::filesystem::create_directory(tmp_ptx);

    std::string arch = std::to_string(major_version * 10 + minor_version);
    std::string cmd =
        "cd " + tmp_ptx.string() + "; cuobjdump -xptx all " + tmp_bin.string();
    exec(cmd.c_str());

    for (const auto &d : std::filesystem::directory_iterator(tmp_ptx)) {
        // analyze single ptx file
        std::string const &f = d.path().string();
        std::ifstream s(f);
        std::stringstream ss;
        ss << s.rdbuf();

        static std::regex r_func_parameters(R"(.entry.*\s(.*)\(([^\)]*)\))",
                                            std::regex::ECMAScript);
        std::string ptx_data = ss.str();
        std::sregex_iterator i = std::sregex_iterator(
            ptx_data.begin(), ptx_data.end(), r_func_parameters);
        std::map<std::string, std::vector<KParamInfo>> tmp_map;
        for (; i != std::sregex_iterator(); ++i) {
            std::smatch m = *i;
            const std::string &entry = m[1];
            const std::string &params = m[2];

            std::vector<KParamInfo> param_infos = parsePtxParameters(params);
            tmp_map.emplace(std::make_pair(entry, param_infos));
        }
        this->storeAnalysisToCache(std::filesystem::canonical(fname), tmp_map);
        this->kernel_to_kparaminfos.insert(tmp_map.begin(), tmp_map.end());
    }

    return true;
}

bool CubinAnalyzer::analyze(const std::vector<std::string>& cuda_binaries,
                            int major_version, int minor_version) {
    bool ret = false;

    for (const auto &cbin : cuda_binaries) {
        std::filesystem::path cuda_binary(cbin);
        SPDLOG_DEBUG("Analyzing: {}", cuda_binary.string());
        if (!std::filesystem::exists(cuda_binary) ||
            !std::filesystem::is_regular_file(cuda_binary)) {
            SPDLOG_ERROR("Invalid file: {}", cbin);
            return false;
        }

        // check if analysis is cached
        if (this->isCached(cbin)) {
            SPDLOG_INFO("Loading analysis from cache for: {}", cbin);
            ret = this->loadAnalysisFromCache(cbin);
        } else {
            ret = this->analyzePtx(cbin, major_version, minor_version);
        }
    }

    this->initialized_ = true;
    return ret;
}

bool CubinAnalyzer::kernel_parameters(std::string &kernel,
                                      std::vector<KParamInfo> &params) const {
    auto it = this->kernel_to_kparaminfos.find(kernel);
    if (it != this->kernel_to_kparaminfos.end()) {
        params = it->second;
        return true;
    }
    return false;
}

bool CubinAnalyzer::kernel_module(std::string &kernel,
                                  std::vector<uint8_t> &module_data) {
    return true;
}
