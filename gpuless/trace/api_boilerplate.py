import os
import sys


def create_boilerplate(function="TEST:(int a, int b, int* c, std::vector<double> d)", i=""):
    """usage: function = name:(params)"""

    name, params = function.split(":", maxsplit=1)
    params = params.replace("(", "")
    params = params.replace(")", "")
    names = params.replace(",", "").split(" ")[1::2]
    params = params.split(",")

    nl = "\n"

    class_template = f"""
    class {name} : public CudaRuntimeApiCall {{
    public:
        {nl.join((p + ";") for p in params)}
        
        explicit {name}(size_t size);
        explicit {name}(const FBCudaApiCall *fb_cuda_api_call);
        
        uint64_t executeNative(CudaVirtualDevice &vdev) override;
        
        flatbuffers::Offset<FBCudaApiCall>
        fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
    }};"""

    construct_from_fb = []
    for p, n in zip(params, names):
        if "vector" in p:
            construct_from_fb += [f"""this->{n} = std::vector<uint8_t>(c->size());"""]
        elif "*" in p:
            construct_from_fb += [f"""this->{n} = reinterpret_cast<void *>(c->{n}());"""]
        else:
            construct_from_fb += [f"""this->{n} = c->{n}();"""]

    build_fb = []
    for p, n in zip(params, names):
        if "vector" in p:
            build_fb += [f"""builder.CreateVector(this->{n})"""]
        elif "*" in p:
            build_fb += [f"""reinterpret_cast<uint64_t>(this->{n})"""]
        else:
            build_fb += [f"""this->{n}"""]

    class_impl = f"""
    
    /*
     * {name}
     */
    
    {name}::{name}({", ".join(params)})
        : {", ".join([f"{n}({n})" for n in names])} {{}}
    
    uint64_t {name}::executeNative(CudaVirtualDevice &vdev) {{
        // TODO
        return 0;
    }}
    
    flatbuffers::Offset<FBCudaApiCall>
    {name}::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {{
        auto api_call =
            CreateFB{name}(builder, {", ".join(build_fb)});
        auto api_call_union = CreateFBCudaApiCall(
            builder, FBCudaApiCallUnion_FB{name}, api_call.Union());
        return api_call_union;
    }}
    
    {name}::{name}(const FBCudaApiCall *fb_cuda_api_call) {{
        auto c = fb_cuda_api_call->api_call_as_FB{name}();
        {nl.join(construct_from_fb)}
        // TODO: check!
    }}"""

    fb_getters = []
    for p, n in zip(params, names):
        if "vector" in p:
            fb_getters += [f"""const flatbuffers::Vector<uint8_t> *{n}() const {{
        return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_{n.upper()});
    }}"""]
        else:
            fb_getters += [f"""uint64_t {n}() const {{
        return GetField<uint64_t>(VT_{n.upper()}, 0);
    }}"""]

    fb_verify = []
    for p, n in zip(params, names):
        if "vector" in p:
            fb_verify += [f"""VerifyOffset(verifier, VT_{n.upper()}) &&
               verifier.VerifyVector({n}()) &&"""]
        else:
            fb_verify += [f"""VerifyField<uint64_t>(verifier, VT_{n.upper()}) &&"""]

    fb_adders = []
    for p, n in zip(params, names):
        if "vector" in p:
            fb_adders += [f"""void add_{n}(uint64_t {n}) {{
        fbb_.AddOffset(FB{name}::VT_{n.upper()}, {n});
    }}"""]
        else:
            fb_adders += [f"""void add_{n}(uint64_t {n}) {{
        fbb_.AddElement<uint64_t>(FB{name}::VT_{n.upper()}, {n}, 0);
    }}"""]

    fb_init = []
    for p, n in zip(params, names):
        if "vector" in p:
            fb_init += [f"""flatbuffers::Offset<flatbuffers::Vector<uint8_t>> {n} = 0"""]
        else:
            fb_init += [f"""uint64_t {n} = 0"""]

    fb_call_adders = [f"builder_.add_{n}({n});" for n in names]

    flat_buffer_impl = f"""
    struct FB{name};
    struct FB{name}Builder;
    
    struct FB{name} FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {{
        typedef FB{name}Builder Builder;
        enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {{
            {nl.join([f"VA_{n.upper()} = 0; // TODO" for n in names])}
        }};
        {nl.join(fb_getters)}
        bool Verify(flatbuffers::Verifier &verifier) const {{
            return VerifyTableStart(verifier) &&
                {nl.join(fb_verify)}
                   verifier.EndTable();
        }}
    }};
    
    struct FB{name}Builder {{
        typedef FB{name} Table;
        flatbuffers::FlatBufferBuilder &fbb_;
        flatbuffers::uoffset_t start_;
        {nl.join(fb_adders)}
        explicit FB{name}Builder(flatbuffers::FlatBufferBuilder &_fbb)
                : fbb_(_fbb) {{
            start_ = fbb_.StartTable();
        }}
        flatbuffers::Offset<FB{name}> Finish() {{
            const auto end = fbb_.EndTable(start_);
            auto o = flatbuffers::Offset<FB{name}>(end);
            return o;
        }}
    }};
    
    inline flatbuffers::Offset<FB{name}> CreateFB{name}(
        flatbuffers::FlatBufferBuilder &_fbb,
        {", ".join(fb_init)}
        ) {{
        FB{name}Builder builder_(_fbb);
        {nl.join(fb_call_adders)}
        return builder_.Finish();
    }}
    
    """

    trace_exec_protocol = f"""
    
    // TODO add this with next value to enum FBCudaApiCallUnion
    FBCudaApiCallUnion_FB{name}
    
    
    // TODO add this to EnumValuesFBCudaApiCallUnion
    FBCudaApiCallUnion_FB{name}
    
    
    // TODO add this to EnumNamesFBCudaApiCallUnion
    "FB{name}"
    
    template<> struct FBCudaApiCallUnionTraits<FB{name}> {{
      static const FBCudaApiCallUnion enum_value = FBCudaApiCallUnion_FB{name};
    }};
    
    
    // TODO add this into struct FBCudaApiCall
    const FB{name} *api_call_as_FB{name}() const {{
        return api_call_type() == gpuless::FBCudaApiCallUnion_FB{name} ? static_cast<const FB{name} *>(api_call()) : nullptr;
    }}
    
    // TODO this outside but after
    template<> inline const FB{name} *FBCudaApiCall::api_call_as<FB{name}>() const {{
        return api_call_as_FBCudaMalloc();
    }}
    
    // TODO add this to function VerifyFBCudaApiCallUnion
    case FBCudaApiCallUnion_FB{name}: {{
          auto ptr = reinterpret_cast<const FB{name} *>(obj);
    return verifier.VerifyTable(ptr);
    }}
    """

    files = [f"boilerplate{i}.cpp", f"boilerplate{i}.hpp", f"fb_boilerplate{i}.h", f"trace_ex_boilerplate{i}.h"]

    with open(files[0], "w") as f:
        f.write(class_impl)

    with open(files[1], "w") as f:
        f.write(class_template)

    with open(files[2], "w") as f:
        f.write(flat_buffer_impl)

    with open(files[3], "w") as f:
        f.write(trace_exec_protocol)

    # optional
    for f in files:
        os.system(f"clang-format -i {f}")


if __name__ == "__main__":
    assert len(sys.argv) >= 2
    if len(sys.argv) == 2:
        create_boilerplate(sys.argv[1])
    else:
        for i, f in enumerate(sys.argv[1:]):
            create_boilerplate(f, i)
