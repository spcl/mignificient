#ifndef MIGNIFICIENT_IPC_TYPES_HPP
#define MIGNIFICIENT_IPC_TYPES_HPP

#include <mignificient/ipc/config.hpp>

#ifdef MIGNIFICIENT_WITH_ICEORYX2
  #define IOX_V2_ENABLED 1
#else
  #define IOX_V2_ENABLED 0
#endif

#endif // MIGNIFICIENT_IPC_TYPES_HPP
