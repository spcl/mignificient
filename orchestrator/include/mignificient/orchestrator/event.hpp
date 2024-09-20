#ifndef __MIGNIFICIENT_ORCHESTRATOR_EVENT_HPP__
#define __MIGNIFICIENT_ORCHESTRATOR_EVENT_HPP__

namespace mignificient { namespace orchestrator {

  enum class EventSource {
    HTTP = 0,
    CLIENT = 1,
    GPULESS_SERVER = 2
  };

  struct Context {
    EventSource type;
    void* source;
  };

}}

#endif
