
#include <mignificient/ipc/config.hpp>
#include <mignificient/executor/executor.hpp>

#if defined(WITH_PYTHON)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void define_mignificient_runtime(py::module& m)
{

  m.doc() = "mignificient executor module";

  // FIXME: function
  m.attr("__name__") = "_mignificient.function";

  py::enum_<mignificient::ipc::IPCBackend>(m, "IPCBackend")
    .value("ICEORYX_V1", mignificient::ipc::IPCBackend::ICEORYX_V1)
    .value("ICEORYX_V2", mignificient::ipc::IPCBackend::ICEORYX_V2);

  py::class_<mignificient::executor::Runtime>(m, "Runtime")
      .def(py::init<mignificient::ipc::IPCBackend, const std::string&>())
      .def("loop_wait", &mignificient::executor::Runtime::loop_wait)
      .def("gpu_yield", &mignificient::executor::Runtime::gpu_yield)
      .def("register_runtime", &mignificient::executor::Runtime::register_runtime)
      .def("finish", &mignificient::executor::Runtime::finish)
      .def("result", &mignificient::executor::Runtime::result);

  py::class_<mignificient::InvocationData>(m, "Payload")
      .def(py::init())
      .def_readonly("data", &mignificient::InvocationData::data)
      .def_readonly("size", &mignificient::InvocationData::size)
      .def(
          "view_readable",
          [](mignificient::InvocationData& self) {
            return py::memoryview::from_memory(self.data, sizeof(std::byte) * mignificient::executor::Invocation::CAPACITY);
          }
      );

  py::class_<mignificient::InvocationResultData>(m, "Result")
      .def(py::init())
      .def_readonly("data", &mignificient::InvocationResultData::data)
      .def_readwrite("size", &mignificient::InvocationResultData::size)
      .def("view_readable",
          [](mignificient::InvocationResultData& self) {
            return py::memoryview::from_memory(self.data, sizeof(std::byte) * mignificient::executor::InvocationResult::CAPACITY);
          }
      )
      .def("view_writable", [](mignificient::InvocationResultData& self) {
        return py::memoryview::from_memory(self.data, sizeof(std::byte) * mignificient::executor::InvocationResult::CAPACITY);
      });
}

PYBIND11_MODULE(_mignificient, m)
{
  define_mignificient_runtime(m);
}

#endif
