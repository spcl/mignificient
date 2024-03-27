#ifndef __DLSYM_UTIL_H__
#define __DLSYM_UTIL_H__

#ifdef CSCS 
extern "C" {
void *__libc_dlsym(void *map, const char *name);
void *__libc_dlopen_mode(const char *name, int mode);
}
#endif
void *real_dlsym(void *handle, const char *symbol);

#endif // __DLSYM_UTIL_H__
