#ifndef __CUBIN_ANALYSIS_HPP__
#define __CUBIN_ANALYSIS_HPP__

#include <filesystem>
#include <map>
#include <string>
#include <vector>

enum PtxParameterType {
    s8 = 0,
    s16 = 1,
    s32 = 2,
    s64 = 3, // signed integers
    u8 = 4,
    u16 = 5,
    u32 = 6,
    u64 = 7, // unsigned integers
    f16 = 8,
    f16x2 = 9,
    f32 = 10,
    f64 = 11, // floating-point
    b8 = 12,
    b16 = 13,
    b32 = 14,
    b64 = 15,     // untyped bits
    pred = 16,    // predicate
    invalid = 17, // invalid type for signaling errors
};

std::map<std::string, PtxParameterType> &getStrToPtxParameterType();
std::map<PtxParameterType, std::string> &getPtxParameterTypeToStr();
std::map<PtxParameterType, int> &getPtxParameterTypeToSize();

struct KParamInfo {
    std::string paramName;
    PtxParameterType type;
    int typeSize;
    int align;
    int size;
};

class CubinAnalyzer {
  private:
    bool initialized_ = false;
    std::map<std::string, std::vector<KParamInfo>> kernel_to_kparaminfos;

    static PtxParameterType ptxParameterTypeFromString(const std::string &str);
    static int byteSizePtxParameterType(PtxParameterType type);

    bool isCached(const std::filesystem::path &fname);
    bool loadAnalysisFromCache(const std::filesystem::path &fname);
    void storeAnalysisToCache(
        const std::filesystem::path &fname,
        const std::map<std::string, std::vector<KParamInfo>> &data);

    std::vector<KParamInfo> parsePtxParameters(const std::string &params);
    bool analyzePtx(const std::filesystem::path &path, int major_version,
                    int minor_version);
    static size_t pathToHash(const std::filesystem::path &path);

  public:
    CubinAnalyzer() = default;
    bool isInitialized();
    bool analyze(const std::vector<std::string>& cuda_binaries, int major_version,
                 int minor_version);

    bool kernel_parameters(std::string &kernel,
                           std::vector<KParamInfo> &params) const;
    bool kernel_module(std::string &kernel, std::vector<uint8_t> &module_data);
};

#endif // __CUBIN_ANALYSIS_HPP__
