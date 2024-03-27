#include "dlsym_util.hpp"
#include <dlfcn.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>

char hostname[HOST_NAME_MAX + 1];




void *real_dlsym(void *handle, const char *symbol) {
#ifdef CSCS
    	static auto internal_dlsym = (decltype(&dlsym))__libc_dlsym(
        		__libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
#else
    	static auto internal_dlsym = (decltype(&dlsym))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
#endif
    return (*internal_dlsym)(handle, symbol);
}
